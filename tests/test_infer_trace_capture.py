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
    ControlResult,
    HttpProfilerControl,
    TraceCaptureConfig,
    TraceWindows,
    server_root,
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
    """Records calls; on stop, writes a worker trace and a frontend trace."""

    def __init__(self, trace_dir: Path | None, *, start_status: int = 200) -> None:
        self.trace_dir = trace_dir
        self.start_status = start_status
        self.calls: list[str] = []

    def post(self, route: str) -> ControlResult:
        self.calls.append(route)
        if route == "/start_profile":
            ok = 200 <= self.start_status < 300
            return ControlResult(self.start_status, None if ok else "HTTP 500")
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


def test_a_failed_start_is_recorded_and_never_stopped(tmp_path: Path) -> None:
    warnings: list[str] = []
    control = _FakeControl(tmp_path, start_status=500)
    windows = TraceWindows(
        _config(tmp_path), control=control, on_warning=warnings.append
    )

    window = _run_window(windows)

    assert control.calls == ["/start_profile"]
    assert not window.started and window.stop_reason is None
    assert window.files == []
    assert warnings == ["c1 measured: could not start the profiler (HTTP 500)"]


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
