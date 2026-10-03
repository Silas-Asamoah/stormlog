"""Bounded vLLM profiler windows: lifecycle, ownership, bounds, and import."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    CapabilityEvent,
    load_inference_artifact,
)
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.trace_capture import (
    START_ACKNOWLEDGED,
    START_REJECTED,
    START_UNKNOWN,
    ControlResult,
    HttpProfilerControl,
    TraceCaptureConfig,
    TraceWindows,
    server_root,
    start_outcome,
)


def _kineto(trace_id: str) -> str:
    return json.dumps(
        {
            "baseTimeNanoseconds": 1_000_000_000,
            "trace_id": trace_id,
            "host_name": "server",
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "cudaLaunchKernel",
                    "pid": 1,
                    "tid": 1,
                    "ts": 1.0,
                    "dur": 1.0,
                    "args": {"correlation": 1},
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "gemm",
                    "pid": 0,
                    "tid": 7,
                    "ts": 2.0,
                    "dur": 5.0,
                    "args": {"device": 0, "stream": 7, "correlation": 1},
                },
            ],
        }
    )


class _FakeControl:
    """Records calls; on stop, writes a worker trace and a frontend trace.

    ``start_status`` None answers the start like a connection reset after the
    server received it.
    """

    def __init__(
        self, trace_dir: Path | None, *, start_status: int | None = 200
    ) -> None:
        self.trace_dir = trace_dir
        self.start_status = start_status
        self.calls: list[str] = []

    def post(self, route: str) -> ControlResult:
        self.calls.append(route)
        if route == "/start_profile":
            if self.start_status is None:
                return ControlResult(None, "ConnectionResetError: reset by peer")
            ok = 200 <= self.start_status < 300
            return ControlResult(
                self.start_status, None if ok else f"HTTP {self.start_status}"
            )
        if self.trace_dir is not None:
            stop = self.calls.count("/stop_profile")
            (self.trace_dir / f"rank0.{stop}.pt.trace.json").write_text(
                _kineto(f"T{stop}"), encoding="utf-8"
            )
            (self.trace_dir / f"host_1.async_llm.{stop}.pt.trace.json.gz").write_bytes(
                b"frontend"
            )
        return ControlResult(200)


def _config(trace_dir: Path | None, **overrides: Any) -> TraceCaptureConfig:
    values: dict[str, Any] = {
        "control_url": "http://server:8000",
        "trace_dir": trace_dir,
        "settle_seconds": 0.01,
        "flush_timeout_seconds": 0.5,
    }
    values.update(overrides)
    return TraceCaptureConfig(**values)


def _run_window(
    windows: TraceWindows, phase: str = "measured", body_seconds: float = 0.0
) -> Any:
    async def scenario() -> Any:
        async with windows.window("c1", phase) as window:
            await asyncio.sleep(body_seconds)
        return window

    return asyncio.run(scenario())


def test_config_rejects_unknown_modes_phases_and_bounds() -> None:
    with pytest.raises(ValueError, match="--trace"):
        _config(None, mode="nsys")
    with pytest.raises(ValueError, match="--trace-phase"):
        _config(None, phase="cooldown")
    with pytest.raises(ValueError, match="max_seconds"):
        _config(None, max_seconds=0)
    assert server_root("https://h:8000/v1/chat/completions?x=1") == "https://h:8000"


def test_a_window_starts_stops_and_finds_only_new_worker_traces(
    tmp_path: Path,
) -> None:
    (tmp_path / "rank0.old.pt.trace.json").write_text("{}", encoding="utf-8")
    control = _FakeControl(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    window = _run_window(windows)

    assert control.calls == ["/start_profile", "/stop_profile"]
    assert window.started and window.stop_reason == "phase_end"
    assert [path.name for path in window.files] == ["rank0.1.pt.trace.json"]
    record = window.to_record(session_id="s1")
    assert record["event_type"] == "infer.trace_window"
    assert (record["start_status"], record["stop_status"]) == (200, 200)
    assert record["trace_files"] == ["rank0.1.pt.trace.json"]


def test_other_phases_are_not_profiled(tmp_path: Path) -> None:
    control = _FakeControl(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    assert _run_window(windows, phase="warmup") is None
    assert control.calls == []


def test_a_rejected_start_is_recorded_and_never_stopped(tmp_path: Path) -> None:
    """A 4xx (e.g. a server without the profiler routes) never reached the engine."""
    warnings: list[str] = []
    control = _FakeControl(tmp_path, start_status=404)
    windows = TraceWindows(
        _config(tmp_path), control=control, on_warning=warnings.append
    )

    window = _run_window(windows)

    assert control.calls == ["/start_profile"]
    assert window.start_outcome == START_REJECTED
    assert not window.started and window.stop_reason is None
    assert window.files == []
    assert warnings == ["c1 measured: could not start the profiler (HTTP 404)"]


@pytest.mark.parametrize("start_status", [500, 503, None])
def test_a_start_without_a_clear_answer_is_stopped_before_the_phase(
    tmp_path: Path, start_status: int | None
) -> None:
    """A 5xx or a lost reply can follow a start the engine already ran."""
    warnings: list[str] = []
    control = _FakeControl(tmp_path, start_status=start_status)
    windows = TraceWindows(
        _config(tmp_path, max_seconds=60), control=control, on_warning=warnings.append
    )
    calls_during_phase: list[str] = []

    async def scenario() -> Any:
        async with windows.window("c1", "measured") as window:
            calls_during_phase.extend(control.calls)
        return window

    window = asyncio.run(scenario())

    # Stopped before the phase ran, and only once.
    assert calls_during_phase == ["/start_profile", "/stop_profile"]
    assert control.calls == ["/start_profile", "/stop_profile"]
    assert window.start_outcome == START_UNKNOWN
    assert not window.started and window.started_at_ns is None
    assert window.stop_reason == "start_unknown"
    # A trace the stop wrote is still found, so the record says what exists.
    assert [path.name for path in window.files] == ["rank0.1.pt.trace.json"]
    record = window.to_record(session_id="s1")
    assert (record["start_outcome"], record["stop_reason"]) == (
        START_UNKNOWN,
        "start_unknown",
    )
    assert warnings[0].startswith(
        "c1 measured: the profiler may have started; stopping it"
    )


@pytest.mark.parametrize(
    "failure", [BrokenPipeError("stderr closed"), KeyboardInterrupt()]
)
@pytest.mark.parametrize("cancel_during_start", [False, True])
def test_a_failing_warning_never_skips_the_stop_or_the_record(
    tmp_path: Path, failure: BaseException, cancel_during_start: bool
) -> None:
    """The CLI's warning writes to stderr, which can fail or be interrupted."""
    import time

    class _SlowLostStart(_FakeControl):
        def post(self, route: str) -> ControlResult:
            if route == "/start_profile" and cancel_during_start:
                time.sleep(0.2)
            return super().post(route)

    def warn(_message: str) -> None:
        raise failure

    control = _SlowLostStart(tmp_path, start_status=None)
    windows = TraceWindows(_config(tmp_path), control=control, on_warning=warn)

    async def scenario() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5)

        task = asyncio.create_task(body())
        if cancel_during_start:
            await asyncio.sleep(0.05)
            task.cancel()
        await task

    with pytest.raises((type(failure), asyncio.CancelledError)):
        asyncio.run(scenario())

    assert control.calls == ["/start_profile", "/stop_profile"]
    records = windows.take_records(session_id="s1")
    assert [(r["start_outcome"], r["stop_status"]) for r in records] == [
        (START_UNKNOWN, 200)
    ]


@pytest.mark.parametrize("start_status", [200, None])
def test_a_failed_stop_is_recorded_as_one(
    tmp_path: Path, start_status: int | None
) -> None:
    """A stop that did not reach the engine leaves the profiler running."""

    class _StopFails(_FakeControl):
        def post(self, route: str) -> ControlResult:
            self.calls.append(route)
            if route == "/start_profile":
                if start_status is None:
                    return ControlResult(None, "TimeoutError: timed out")
                return ControlResult(start_status)
            return ControlResult(503, "HTTP 503")

    warnings: list[str] = []
    control = _StopFails(tmp_path)
    windows = TraceWindows(
        _config(tmp_path), control=control, on_warning=warnings.append
    )

    window = _run_window(windows)

    # Sent once, never retried, and said.
    assert control.calls == ["/start_profile", "/stop_profile"]
    assert window.stop is not None and window.stop.status == 503
    assert window.note is not None
    assert "unprofiled" not in window.note
    assert "may still be running" in window.note
    assert warnings[-1] == (
        "c1 measured: the profiler did not confirm the stop (HTTP 503)"
    )


def test_an_unknown_start_stopped_before_the_phase_says_so(tmp_path: Path) -> None:
    control = _FakeControl(tmp_path, start_status=None)
    windows = TraceWindows(_config(tmp_path), control=control)

    window = _run_window(windows)

    assert window.note == (
        "the start's answer does not say whether the profiler started; "
        "it was stopped before the phase, which ran unprofiled"
    )


def test_start_outcome_classifies_every_answer() -> None:
    assert start_outcome(ControlResult(200)) == START_ACKNOWLEDGED
    assert start_outcome(ControlResult(204)) == START_ACKNOWLEDGED
    assert start_outcome(ControlResult(404, "HTTP 404")) == START_REJECTED
    assert start_outcome(ControlResult(401, "HTTP 401")) == START_REJECTED
    assert start_outcome(ControlResult(500, "HTTP 500")) == START_UNKNOWN
    assert start_outcome(ControlResult(302, "HTTP 302")) == START_UNKNOWN
    assert start_outcome(ControlResult(None, "TimeoutError: timed out")) == (
        START_UNKNOWN
    )


def test_the_start_request_is_stamped_around_the_call_itself(tmp_path: Path) -> None:
    import time

    stamps: dict[str, int] = {}

    class _Timed(_FakeControl):
        def post(self, route: str) -> ControlResult:
            if route == "/start_profile":
                stamps["inside"] = time.time_ns()
            return super().post(route)

    control = _Timed(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    window = _run_window(windows)

    assert window.start_outcome == START_ACKNOWLEDGED
    assert window.start_requested_at_ns is not None
    assert window.start_returned_at_ns is not None
    assert window.requested_at_ns <= window.start_requested_at_ns
    assert window.start_requested_at_ns <= stamps["inside"]
    assert stamps["inside"] <= window.start_returned_at_ns
    assert window.started_at_ns == window.start_returned_at_ns
    record = window.to_record(session_id="s1")
    assert record["start_requested_at_ns"] == window.start_requested_at_ns
    assert record["start_returned_at_ns"] == window.start_returned_at_ns
    assert record["start_outcome"] == START_ACKNOWLEDGED


def test_the_time_bound_stops_the_profiler_while_the_phase_runs(
    tmp_path: Path,
) -> None:
    control = _FakeControl(tmp_path)
    windows = TraceWindows(_config(tmp_path, max_seconds=0.05), control=control)

    window = _run_window(windows, body_seconds=0.3)

    assert window.stop_reason == "time_bound"
    assert control.calls == ["/start_profile", "/stop_profile"]


def test_cancellation_still_stops_the_profiler(tmp_path: Path) -> None:
    control = _FakeControl(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    async def scenario() -> None:
        async with windows.window("c1", "measured"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(scenario())
    assert control.calls == ["/start_profile", "/stop_profile"]
    assert windows.windows[0].stop_reason == "cancelled"


def test_without_a_trace_dir_the_trace_is_left_on_the_server() -> None:
    control = _FakeControl(None)
    windows = TraceWindows(_config(None), control=control)

    window = _run_window(windows)

    assert window.files == []
    assert window.note == "trace left on the server; no --trace-dir to read it from"


class _ProfileRoutes(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    status = 200

    def do_POST(self) -> None:  # noqa: N802
        self.server.calls.append(self.path)  # type: ignore[attr-defined]
        body = b"{}"
        self.send_response(self.status if self.path == "/start_profile" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _routes() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProfileRoutes)
    server.calls = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_http_control_reports_status_and_never_raises() -> None:
    with _routes() as server:
        control = HttpProfilerControl(
            f"http://127.0.0.1:{server.server_port}/", api_key="k", timeout=5
        )
        assert control.post("/start_profile") == ControlResult(200)
        assert control.post("/stop_profile") == ControlResult(404, "HTTP 404")
    unreachable = HttpProfilerControl("http://127.0.0.1:1", api_key=None, timeout=1)
    result = unreachable.post("/start_profile")
    assert result.status is None and result.error is not None


def test_a_start_whose_reply_is_lost_is_stopped_over_http(tmp_path: Path) -> None:
    """The engine started profiling, then the connection dropped before the reply."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    routes: list[str] = []
    profiling = threading.Event()

    def serve() -> None:
        for _ in range(2):
            connection, _ = listener.accept()
            request = connection.recv(4096).decode("latin-1")
            route = request.split(" ", 2)[1]
            routes.append(route)
            if route == "/start_profile":
                profiling.set()
                connection.close()  # no reply at all
                continue
            profiling.clear()
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            connection.close()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    windows = TraceWindows(
        _config(None, control_url=f"http://127.0.0.1:{listener.getsockname()[1]}")
    )

    window = _run_window(windows)
    server.join(timeout=5)
    listener.close()

    assert routes == ["/start_profile", "/stop_profile"]
    assert not profiling.is_set()
    assert window.start_outcome == START_UNKNOWN
    assert window.start is not None and window.start.status is None
    assert window.stop is not None and window.stop.ok


def test_cancelling_while_an_unanswered_start_is_in_flight_still_stops(
    tmp_path: Path,
) -> None:
    import time

    class _SlowLostStart(_FakeControl):
        def post(self, route: str) -> ControlResult:
            if route == "/start_profile":
                time.sleep(0.3)
            return super().post(route)

    control = _SlowLostStart(tmp_path, start_status=None)
    windows = TraceWindows(_config(tmp_path), control=control)

    async def scenario() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5)

        task = asyncio.create_task(body())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert control.calls == ["/start_profile", "/stop_profile"]
    records = windows.take_records(session_id="s1")
    assert [(r["start_outcome"], r["stop_reason"]) for r in records] == [
        (START_UNKNOWN, "cancelled")
    ]


class _ChatAndProfile(BaseHTTPRequestHandler):
    """A chat endpoint plus vLLM's profiler routes that write traces on stop."""

    protocol_version = "HTTP/1.1"
    trace_dir: Path

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/stop_profile":
            stops = len(list(self.trace_dir.glob("rank0.*")))
            (self.trace_dir / f"rank0.{stops}.pt.trace.json").write_text(
                _kineto(f"T{stops}"), encoding="utf-8"
            )
        body = b"{}"
        if self.path == "/v1/chat/completions":
            body = json.dumps(
                {
                    "choices": [
                        {"message": {"content": "hi"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


def test_profile_records_trace_windows_and_imports_their_traces(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    handler = type("Handler", (_ChatAndProfile,), {"trace_dir": trace_dir})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "infer.jsonl"
    try:
        InferenceProfiler(
            ProfileConfig(
                endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                model="m",
                concurrency=(1, 2),
                input_tokens=(8,),
                output_tokens=(4,),
                request_count=2,
                output_path=str(output),
                stream=False,
                system_sampler="none",
                tokenizer="none",
                trace=_config(
                    trace_dir,
                    control_url=f"http://127.0.0.1:{server.server_port}",
                    device_uuids={0: "GPU-x"},
                ),
            )
        ).run()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    raw = [json.loads(line) for line in output.read_text().splitlines()]
    windows = [r for r in raw if r.get("event_type") == "infer.trace_window"]
    assert [w["case_id"] for w in windows] == [
        r["case_id"] for r in raw if r.get("event_type") == "infer.phase_window"
    ]
    assert all(w["started"] and w["stop_reason"] == "phase_end" for w in windows)
    records = load_inference_artifact(output)
    activities = [r for r in records if isinstance(r, ActivityReferenceEvent)]
    assert len(activities) == 2
    assert {a.attribution_status for a in activities} == {"unresolved"}
    collector = next(
        r
        for r in records
        if isinstance(r, CapabilityEvent) and r.component == "trace_collector"
    )
    assert len(collector.metadata["summary"]["traces"]) == 2


def test_trace_options_need_trace(tmp_path: Path) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "m",
                "--output",
                str(tmp_path / "out.jsonl"),
                "--trace-dir",
                str(tmp_path),
            ]
        )
    assert code == int(ExitCode.USAGE)
    assert "--trace-dir needs --trace" in stderr.getvalue()


class _SelfStoppingControl(_FakeControl):
    """A server profile that stopped by itself (max_iterations) before our stop."""

    def post(self, route: str) -> ControlResult:
        if route == "/start_profile" and self.trace_dir is not None:
            self.calls.append(route)
            (self.trace_dir / "rank0.auto.pt.trace.json").write_text(
                _kineto("AUTO"), encoding="utf-8"
            )
            return ControlResult(200)
        if route == "/stop_profile":
            self.calls.append(route)
            return ControlResult(200)
        return super().post(route)


def test_a_trace_written_before_the_stop_is_marked_as_stopped_by_the_server(
    tmp_path: Path,
) -> None:
    control = _SelfStoppingControl(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    window = _run_window(windows)

    assert window.stop_reason == "stopped_by_server"
    assert [path.name for path in window.files] == ["rank0.auto.pt.trace.json"]


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _profile_config(port: int, output: Path, trace_dir: Path) -> ProfileConfig:
    return ProfileConfig(
        endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
        model="m",
        concurrency=(1,),
        input_tokens=(8,),
        output_tokens=(4,),
        request_count=2,
        output_path=str(output),
        stream=False,
        system_sampler="none",
        tokenizer="none",
        trace=_config(trace_dir, control_url=f"http://127.0.0.1:{port}"),
    )


def test_a_cancelled_phase_still_records_its_trace_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    server = _serve(type("Handler", (_ChatAndProfile,), {"trace_dir": trace_dir}))
    output = tmp_path / "infer.jsonl"

    async def interrupted(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(0.05)
        raise KeyboardInterrupt

    monkeypatch.setattr(InferenceProfiler, "_run_phase_requests", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            InferenceProfiler(
                _profile_config(server.server_port, output, trace_dir)
            ).run()
    finally:
        server.shutdown()
        server.server_close()

    raw = [json.loads(line) for line in output.read_text().splitlines()]
    types = [r["event_type"] for r in raw]
    assert types[-2:] == ["infer.trace_window", "infer.session"]
    window = raw[-2]
    assert window["stop_reason"] == "cancelled"
    assert window["trace_files"] == ["rank0.0.pt.trace.json"]
    assert window["started_at_ns"] >= window["requested_at_ns"]


class _FailingStart(_ChatAndProfile):
    """Refuses the start with a 4xx, which never reaches the engine."""

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/start_profile":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_POST()


def test_a_requested_trace_that_imported_nothing_is_still_recorded(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    server = _serve(type("Handler", (_FailingStart,), {"trace_dir": trace_dir}))
    output = tmp_path / "infer.jsonl"
    try:
        InferenceProfiler(_profile_config(server.server_port, output, trace_dir)).run()
    finally:
        server.shutdown()
        server.server_close()

    collector = next(
        r
        for r in load_inference_artifact(output)
        if isinstance(r, CapabilityEvent) and r.component == "trace_collector"
    )
    assert collector.available and collector.collected == []
    assert collector.metadata["summary"]["windows"] == [
        {
            "case_id": "c1_in8_out4",
            "started": False,
            "note": "the profiler did not start; see start_status and start_error",
        }
    ]


def test_a_failed_import_warns_instead_of_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stormlog.infer.trace_capture as trace_capture

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(trace_capture, "append_inference_capture", broken)
    warnings: list[str] = []
    control = _FakeControl(tmp_path)
    windows = TraceWindows(
        _config(tmp_path), control=control, on_warning=warnings.append
    )
    _run_window(windows)

    windows.import_into(tmp_path / "infer.jsonl", run_id="r", session=None)  # type: ignore[arg-type]

    assert warnings and "not imported (disk full)" in warnings[0]


def test_a_trace_removed_before_the_import_warns_instead_of_failing(
    tmp_path: Path,
) -> None:
    """A retention job can delete an early window's trace while later cases run."""
    warnings: list[str] = []
    windows = TraceWindows(
        _config(tmp_path), control=_FakeControl(tmp_path), on_warning=warnings.append
    )
    window = _run_window(windows)
    window.files[0].unlink()

    windows.import_into(tmp_path / "infer.jsonl", run_id="r", session=None)  # type: ignore[arg-type]

    assert len(warnings) == 1
    assert "trace file not found" in warnings[0]


def test_a_malformed_reply_is_a_failed_call_not_an_exception() -> None:
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def answer() -> None:
        connection, _ = listener.accept()
        connection.recv(4096)
        connection.sendall(b"NOT-HTTP garbage\r\n\r\n")
        connection.close()

    threading.Thread(target=answer, daemon=True).start()
    control = HttpProfilerControl(
        f"http://127.0.0.1:{listener.getsockname()[1]}", api_key=None, timeout=5
    )

    result = control.post("/start_profile")
    listener.close()

    assert result.status is None
    assert result.error is not None and result.error.startswith("BadStatusLine")


def test_a_window_without_a_new_trace_stops_waiting_after_the_grace(
    tmp_path: Path,
) -> None:
    import time

    class _WritesNothing(_FakeControl):
        def post(self, route: str) -> ControlResult:
            self.calls.append(route)
            return ControlResult(200)

    windows = TraceWindows(
        _config(tmp_path, flush_timeout_seconds=30.0, missing_grace_seconds=0.05),
        control=_WritesNothing(tmp_path),
    )
    started = time.monotonic()

    window = _run_window(windows)

    assert time.monotonic() - started < 5
    assert window.files == []
    assert window.note == f"no new worker trace appeared in {tmp_path}"


def test_an_oversized_trace_is_registered_but_not_parsed(tmp_path: Path) -> None:
    from stormlog.infer.trace_import import TraceFileCollector

    big = tmp_path / "rank0.big.pt.trace.json"
    big.write_text(_kineto("BIG") + " " * 5000, encoding="utf-8")
    small = tmp_path / "rank0.small.pt.trace.json"
    small.write_text(_kineto("SMALL"), encoding="utf-8")

    capture = TraceFileCollector([big, small], max_bytes=2000).collect(
        run_id="r", session_id="s"
    )

    assert capture.summary is not None
    skipped, parsed = capture.summary["traces"]
    assert (skipped["file"], skipped["skipped"]) == (big.name, "max_bytes")
    assert parsed["trace_id"] == "SMALL"
    assert len(capture.attachments) == 2
    assert (
        len([e for e in capture.events if isinstance(e, ActivityReferenceEvent)]) == 1
    )


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--trace-phase", "warmup"), ("--trace-detail", "kernel")],
)
def test_trace_options_with_defaults_also_need_trace(
    tmp_path: Path, flag: str, value: str
) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "m",
                "--output",
                str(tmp_path / "out.jsonl"),
                flag,
                value,
            ]
        )
    assert code == int(ExitCode.USAGE)
    assert f"{flag} needs --trace" in stderr.getvalue()


def test_a_trace_registered_over_the_size_bound_can_be_imported_later(
    tmp_path: Path,
) -> None:
    from stormlog.infer.correlation_capture import append_inference_capture
    from stormlog.infer.correlation_events import (
        ArtifactIdentityEvent,
        CorrelationContext,
    )
    from stormlog.infer.trace_import import (
        TraceFileCollector,
        import_traces_into_artifact,
    )
    from stormlog.session import create_session_summary

    artifact = tmp_path / "infer.jsonl"
    identity = ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id="r",
            session_id="s",
            producer_id="p",
            source="p",
            clock_domain="c",
            clock_kind="wall",
            collection_mode="active",
            provenance="observed",
        ),
        event_id="artifact",
        artifact_kind="inference_profile",
        created_at_ns=1,
    )
    artifact.write_text(json.dumps(identity.to_record()) + "\n", encoding="utf-8")
    big = tmp_path / "rank0.big.pt.trace.json"
    big.write_text(_kineto("BIG") + " " * 5000, encoding="utf-8")
    append_inference_capture(
        artifact,
        run_id="r",
        session=create_session_summary(source="test", session_id="s"),
        trace_collector=TraceFileCollector([big], max_bytes=2000),
    )

    def activities() -> int:
        return sum(
            isinstance(r, ActivityReferenceEvent)
            for r in load_inference_artifact(artifact)
        )

    assert activities() == 0
    imported = import_traces_into_artifact(artifact, [big])
    assert activities() == 1 and imported.summary is not None
    assert imported.summary["already_imported"] == []
    again = import_traces_into_artifact(artifact, [big])
    assert again.summary == {"traces": [], "already_imported": [str(big)]}
    assert activities() == 1


def test_cancelling_while_the_start_is_in_flight_still_stops_and_records(
    tmp_path: Path,
) -> None:
    import time

    class _SlowStart(_FakeControl):
        def post(self, route: str) -> ControlResult:
            if route == "/start_profile":
                time.sleep(0.3)
            return super().post(route)

    control = _SlowStart(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    async def scenario() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5)

        task = asyncio.create_task(body())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert control.calls == ["/start_profile", "/stop_profile"]
    records = windows.take_records(session_id="s1")
    assert [(r["stop_reason"], r["started"]) for r in records] == [("cancelled", True)]
    assert records[0]["trace_files"] == ["rank0.1.pt.trace.json"]
    assert windows.take_records(session_id="s1") == []


def test_ctrl_c_during_the_start_still_stops_the_profiler(tmp_path: Path) -> None:
    """asyncio.run cancels every remaining task on the way out of a Ctrl+C."""
    import time

    class _SlowStart(_FakeControl):
        def post(self, route: str) -> ControlResult:
            if route == "/start_profile":
                time.sleep(0.3)
            return super().post(route)

    control = _SlowStart(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)

    async def main() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5)

        asyncio.create_task(body())
        await asyncio.sleep(0.05)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(main())

    assert control.calls == ["/start_profile", "/stop_profile"]
    assert [w.stop_reason for w in windows.windows] == ["cancelled"]


class _BlockingControl(_FakeControl):
    """Holds one route's call until released; ``entered`` is set inside it."""

    def __init__(self, trace_dir: Path, route: str, **kwargs: Any) -> None:
        super().__init__(trace_dir, **kwargs)
        self.route = route
        self.entered = threading.Event()
        self.release = threading.Event()

    def post(self, route: str) -> ControlResult:
        if route == self.route:
            self.entered.set()
            assert self.release.wait(5)
        return super().post(route)


def _cancel_twice_while_blocked(
    windows: TraceWindows, control: _BlockingControl, *, first_cancel_opens: bool
) -> None:
    """Cancel the window's task, then cancel it again inside the held call."""

    async def scenario() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5)

        task = asyncio.create_task(body())
        if first_cancel_opens:  # the held call starts only after a cancel
            await asyncio.sleep(0.05)
            task.cancel()
        assert await asyncio.to_thread(control.entered.wait, 5)
        if not first_cancel_opens:
            task.cancel()
            await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        control.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("start_status", [200, None])
def test_repeated_cancellation_during_the_start_still_stops_and_records(
    tmp_path: Path, start_status: int | None
) -> None:
    control = _BlockingControl(tmp_path, "/start_profile", start_status=start_status)
    windows = TraceWindows(_config(tmp_path), control=control)

    _cancel_twice_while_blocked(windows, control, first_cancel_opens=False)

    assert control.calls == ["/start_profile", "/stop_profile"]
    (window,) = windows.windows
    assert window.stop is not None and window.stop.ok
    assert "may still be running" not in (window.note or "")


def test_repeated_cancellation_during_the_stop_waits_for_it(tmp_path: Path) -> None:
    control = _BlockingControl(tmp_path, "/stop_profile")
    windows = TraceWindows(_config(tmp_path), control=control)

    _cancel_twice_while_blocked(windows, control, first_cancel_opens=True)

    assert control.calls == ["/start_profile", "/stop_profile"]
    (window,) = windows.windows
    # The record was written after the stop returned, not before.
    assert window.stop is not None and window.stop.ok
    assert "may still be running" not in (window.note or "")


def test_ctrl_c_after_a_cancel_during_the_start_still_stops(tmp_path: Path) -> None:
    """asyncio.run cancels the task again on its way out of the Ctrl+C."""
    control = _BlockingControl(tmp_path, "/start_profile", start_status=None)
    windows = TraceWindows(_config(tmp_path), control=control)

    async def main() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5)

        task = asyncio.create_task(body())
        assert await asyncio.to_thread(control.entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        threading.Timer(0.1, control.release.set).start()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(main())

    assert control.calls == ["/start_profile", "/stop_profile"]
    records = windows.take_records(session_id="s1")
    assert [(r["start_outcome"], r["stop_status"]) for r in records] == [
        (START_UNKNOWN, 200)
    ]


@pytest.mark.parametrize(
    ("bound", "reason"), [(None, "phase_end"), (0.01, "time_bound")]
)
def test_ctrl_c_during_the_stop_still_sends_it_and_records_the_window(
    tmp_path: Path, bound: float | None, reason: str
) -> None:
    """The stop first checks the trace directory; Ctrl+C can land right there."""
    import threading
    import time

    control = _FakeControl(tmp_path)
    windows = TraceWindows(_config(tmp_path, max_seconds=bound), control=control)
    checking = threading.Event()
    real_check = windows._wrote_before_stop

    def slow_check(before: dict[str, int] | None) -> bool:
        checking.set()
        time.sleep(0.2)
        return real_check(before)

    windows._wrote_before_stop = slow_check  # type: ignore[method-assign]

    async def main() -> None:
        async def body() -> None:
            async with windows.window("c1", "measured"):
                await asyncio.sleep(5 if bound else 0)

        asyncio.create_task(body())
        await asyncio.to_thread(checking.wait, 5)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(main())

    assert control.calls == ["/start_profile", "/stop_profile"]
    records = windows.take_records(session_id="s1")
    assert [(r["stop_reason"], r["stop_status"]) for r in records] == [(reason, 200)]


def test_an_unreadable_trace_dir_still_stops_and_records_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = _FakeControl(None)
    windows = TraceWindows(_config(tmp_path), control=control)

    def unreadable(*_args: Any) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(windows, "_wrote_before_stop", unreadable)
    monkeypatch.setattr(windows, "_new_files", unreadable)

    window = _run_window(windows)

    assert control.calls == ["/start_profile", "/stop_profile"]
    assert (window.stop_reason, window.files) == ("phase_end", [])
    assert window.note == f"could not read {tmp_path}: Permission denied"
    assert len(windows.take_records(session_id="s1")) == 1


def test_a_trace_dir_unreadable_at_the_start_is_not_searched_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the files already there, new ones cannot be told apart."""
    from stormlog.infer import trace_capture

    control = _FakeControl(tmp_path)
    windows = TraceWindows(_config(tmp_path), control=control)
    real = trace_capture._worker_traces
    calls = {"n": 0}

    def first_call_fails(directory: Path) -> list[Path]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Permission denied")
        return real(directory)

    monkeypatch.setattr(trace_capture, "_worker_traces", first_call_fails)

    window = _run_window(windows)

    assert control.calls == ["/start_profile", "/stop_profile"]
    assert window.files == []
    assert window.note == (
        f"could not read {tmp_path} before the start: Permission denied"
    )


def test_warmup_tracing_needs_warmup_requests(tmp_path: Path) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "m",
                "--output",
                str(tmp_path / "out.jsonl"),
                "--trace",
                "vllm-torch",
                "--trace-phase",
                "warmup",
            ]
        )
    assert code == int(ExitCode.USAGE)
    assert "--trace-phase warmup needs --warmup-requests >= 1" in stderr.getvalue()


@pytest.mark.parametrize(
    ("name", "worker"),
    [
        ("rank0.1790.pt.trace.json.gz", True),
        ("dp0_pp0_tp0_dcp0_ep0_rank0.1790.pt.trace.json.gz", True),
        ("host_1.async_llm.1790.pt.trace.json.gz", False),
        ("profiler_out_0.txt", False),
    ],
)
def test_worker_trace_names_include_parallel_group_prefixes(
    tmp_path: Path, name: str, worker: bool
) -> None:
    from stormlog.infer.trace_capture import _worker_traces

    (tmp_path / name).write_text("{}", encoding="utf-8")

    assert bool(_worker_traces(tmp_path)) is worker
