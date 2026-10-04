"""Prometheus export from a real ``infer profile`` run against a fake server."""

import asyncio
import json
import signal
import socket
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from stormlog._export.renders import RenderCache
from stormlog._export.textfile import PRODUCER_LABEL, TextfileWriter
from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.export import ExportPipeline
from stormlog.infer.export_config import ExportConfig
from stormlog.infer.export_metrics import ProfileMetrics
from stormlog.infer.profile import InferenceProfiler, _ctrl_c_held
from tests.export_conformance import check_exposition
from tests.test_infer_profile import _fake_server


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _config(
    endpoint: str, output: Path, export: ExportConfig, **kw: Any
) -> ProfileConfig:
    settings: dict[str, Any] = {
        "endpoint": endpoint,
        "model": "fake-model",
        "concurrency": (1,),
        "input_tokens": (8,),
        "output_tokens": (4,),
        "output_path": str(output),
        "request_count": 3,
        "warmup_requests": 1,
        "stream": True,
        "tokenizer": "none",
        "system_sampler": "none",
        "export": export,
    }
    settings.update(kw)
    return ProfileConfig(**settings)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _capability(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        r
        for r in records
        if r["event_type"] == "infer.capabilities"
        and r["component"] == "export.prometheus"
    )


def test_a_run_writes_final_metrics_to_its_textfile(tmp_path: Path) -> None:
    output = tmp_path / "out" / "infer.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with _fake_server() as endpoint:
        config = _config(
            endpoint, output, ExportConfig(prometheus_textfile_dir=metrics_dir)
        )
        InferenceProfiler(config).run()
    records = _records(output)
    requests = [r for r in records if r["event_type"] == "infer.request"]
    measured = [r for r in requests if r["phase"] == "measured"]
    exposition = check_exposition((metrics_dir / "stormlog-default.prom").read_text())
    name = "stormlog_infer_requests_total"
    assert exposition.value(name, status="ok", phase="measured") == len(measured) == 3
    assert exposition.value(name, status="ok", phase="warmup") == 1
    gaps = sum(len(r["chunk_interarrival_ms"]) for r in measured)
    assert gaps > 0
    assert (
        exposition.value(
            "stormlog_infer_chunk_interarrival_seconds_count", phase="measured"
        )
        == gaps
    )
    assert exposition.value(
        "stormlog_infer_tokens_total",
        direction="output",
        source="server_usage",
        phase="measured",
    ) == sum(r["output_tokens"] for r in measured)
    assert exposition.value("stormlog_run_active") == 0
    # The capability record holds final counts and comes before the last word.
    capability = _capability(records)
    summary = capability["metadata"]["summary"]["records"]
    assert summary["exact"] and summary["applied"] == summary["offered"]
    assert summary["applied"] == len(requests) + 2  # plus two phase windows
    assert records[-1]["event_type"] == "infer.session"
    assert records.index(capability) < len(records) - 1


def test_a_profiler_that_never_runs_holds_no_slot(tmp_path: Path) -> None:
    # In a notebook or a test, a profiler built and dropped must not keep
    # every later one in the process from taking the slot.
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    export = ExportConfig(prometheus_textfile_dir=metrics_dir)
    with _fake_server() as endpoint:
        InferenceProfiler(_config(endpoint, tmp_path / "unrun.jsonl", export))
        assert not (metrics_dir / "stormlog-default.lock").exists()
        output = tmp_path / "infer.jsonl"
        InferenceProfiler(_config(endpoint, output, export)).run()
    exposition = check_exposition((metrics_dir / "stormlog-default.prom").read_text())
    assert exposition.value("stormlog_run_active") == 0


def test_the_endpoint_serves_during_the_run_and_lingers_after(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    port = _free_port()
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(
                prometheus_listen=f"127.0.0.1:{port}", prometheus_linger_seconds=3.0
            ),
        )
        profiler = InferenceProfiler(config)
        runner = threading.Thread(target=profiler.run)
        runner.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if (
                output.exists()
                and '"infer.session"' in output.read_text().splitlines()[-1]
            ):
                if _records(output)[-1]["status"] == "completed":
                    break
            time.sleep(0.05)
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
        exposition = check_exposition(body.read().decode())
        runner.join(20)
    assert (
        exposition.value("stormlog_infer_requests_total", status="ok", phase="measured")
        == 3
    )
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2)
    assert _capability(_records(output))["collected"] == ["endpoint"]


def test_the_cli_says_where_it_lingers_before_it_waits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    port = _free_port()
    with _fake_server() as endpoint:
        code = _cli(
            endpoint,
            tmp_path / "infer.jsonl",
            "--prometheus-listen",
            f"127.0.0.1:{port}",
            "--prometheus-linger",
            "1",
        )
    assert code == 0
    err = capsys.readouterr().err
    assert f"http://127.0.0.1:{port}/metrics" in err and "1 s" in err


def test_a_busy_port_warns_and_the_run_completes(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        warnings: list[str] = []
        with _fake_server() as endpoint:
            config = _config(
                endpoint, output, ExportConfig(prometheus_listen=f"127.0.0.1:{port}")
            )
            report = InferenceProfiler(config, on_warning=warnings.append).run()
    assert report["summary"]["successful_requests"] == 3
    assert any("could not listen" in warning for warning in warnings)
    capability = _capability(_records(output))
    assert capability["available"] is False
    assert capability["metadata"]["summary"]["endpoint_error"]


def test_without_export_flags_nothing_is_exported(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        InferenceProfiler(_config(endpoint, output, ExportConfig())).run()
    components = [
        r["component"]
        for r in _records(output)
        if r["event_type"] == "infer.capabilities"
    ]
    assert "export.prometheus" not in components


def test_ctrl_c_still_freezes_and_records_final_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _press_ctrl_c_after(monkeypatch, 2)
    output = tmp_path / "infer.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(prometheus_textfile_dir=metrics_dir),
            model="slow-model",
            request_count=50,
            warmup_requests=0,
            stream=False,
        )
        profiler = InferenceProfiler(config)
        started = time.perf_counter()
        with pytest.raises(KeyboardInterrupt):
            profiler.run()
        assert time.perf_counter() - started < 8
    records = _records(output)
    assert records[-1]["status"] == "interrupted"
    summary = _capability(records)["metadata"]["summary"]["records"]
    dropped = sum(summary["dropped"].values())
    assert summary["offered"] == summary["applied"] + dropped
    exposition = check_exposition((metrics_dir / "stormlog-default.prom").read_text())
    assert exposition.value("stormlog_run_active") == 0
    sent = [r for r in records if r["event_type"] == "infer.request"]
    assert sum(exposition.matching("stormlog_infer_requests_total")) == len(sent)


def _press_ctrl_c_after(monkeypatch: pytest.MonkeyPatch, records: int) -> None:
    """Ctrl+C once the exporter has applied ``records`` records.

    The run is then under way with its export open, however slowly it
    started; a timer could fire before either on a loaded machine.
    """
    real_apply = ProfileMetrics.apply
    applied = 0

    def apply_then_press(self: ProfileMetrics, envelope: Any) -> None:
        nonlocal applied
        real_apply(self, envelope)
        applied += 1
        if applied == records:
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(ProfileMetrics, "apply", apply_then_press)


def _slow_worker(monkeypatch: pytest.MonkeyPatch, delay: float = 0.05) -> None:
    """An exporter that applies each record ``delay`` late, so it ends behind."""
    real_apply = ProfileMetrics.apply

    def slow_apply(self: ProfileMetrics, envelope: Any) -> None:
        time.sleep(delay)
        real_apply(self, envelope)

    monkeypatch.setattr(ProfileMetrics, "apply", slow_apply)


def test_the_capability_record_has_the_counts_after_the_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With the exporter behind, a record written before the close drained
    # would show fewer applied than offered, and fewer than the textfile.
    _slow_worker(monkeypatch)
    output = tmp_path / "infer.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(prometheus_textfile_dir=metrics_dir),
            concurrency=(4,),
            request_count=12,
            warmup_requests=0,
            stream=False,
        )
        InferenceProfiler(config).run()
    summary = _capability(_records(output))["metadata"]["summary"]["records"]
    assert summary["offered"] == summary["applied"] and summary["exact"]
    exposition = check_exposition((metrics_dir / "stormlog-default.prom").read_text())
    applied = exposition.value("stormlog_metrics_records_applied_total")
    assert applied == summary["applied"]


def test_a_ctrl_c_ends_a_lingering_run_within_the_close_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The interrupted close has 2 s, though the exporter is far behind, and
    # the 30 s linger is skipped.
    _slow_worker(monkeypatch, delay=0.2)
    _press_ctrl_c_after(monkeypatch, 3)
    closes: list[float] = []
    real_close = ExportPipeline.close

    def timed_close(self: ExportPipeline, deadline: float) -> None:
        started = time.perf_counter()
        try:
            real_close(self, deadline)
        finally:
            closes.append(time.perf_counter() - started)

    monkeypatch.setattr(ExportPipeline, "close", timed_close)
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(
                prometheus_listen=f"127.0.0.1:{_free_port()}",
                prometheus_linger_seconds=30,
            ),
            concurrency=(8,),
            request_count=5000,
            warmup_requests=0,
            stream=False,
        )
        profiler = InferenceProfiler(config)
        started = time.perf_counter()
        with pytest.raises(KeyboardInterrupt):
            profiler.run()
        assert time.perf_counter() - started < 20  # no 30 s linger
    # The capture's close, then the run's, which has nothing left to do.
    assert closes and closes[0] < 3.5 and sum(closes[1:]) < 0.5
    summary = _capability(_records(output))["metadata"]["summary"]["records"]
    assert summary["dropped"]["shutdown"] > 0  # it was behind when it stopped


def test_a_ctrl_c_inside_the_close_still_finishes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewers' case: the exporter is behind at the end of the run, the
    # user presses Ctrl+C once while its close waits for it. The close then
    # stops waiting, though the backlog (40 records at 0.2 s) would outlast
    # its 5 s deadline, and the run is recorded as interrupted.
    output = tmp_path / "infer.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    real_apply = ProfileMetrics.apply
    closing = threading.Event()

    def slow_apply(self: ProfileMetrics, envelope: Any) -> None:
        # Nothing is applied before the close starts, so the exporter is
        # seconds behind when the Ctrl+C comes, however fast the run was.
        closing.wait(60)
        time.sleep(0.2)
        real_apply(self, envelope)

    monkeypatch.setattr(ProfileMetrics, "apply", slow_apply)
    real_close = ExportPipeline.close
    closes: list[float] = []

    def noting_close(self: ExportPipeline, deadline: float) -> None:
        closing.set()
        started = time.perf_counter()
        try:
            real_close(self, deadline)
        finally:
            closes.append(time.perf_counter() - started)

    monkeypatch.setattr(ExportPipeline, "close", noting_close)
    main = threading.main_thread().ident
    assert main is not None

    def ctrl_c_in_close() -> None:
        if closing.wait(60):
            time.sleep(0.3)
            # To the main thread, as a terminal's Ctrl+C reaches it.
            signal.pthread_kill(main, signal.SIGINT)

    presser = threading.Thread(target=ctrl_c_in_close, daemon=True)
    presser.start()
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(prometheus_textfile_dir=metrics_dir),
            concurrency=(4,),
            request_count=40,
            warmup_requests=0,
            stream=False,
        )
        profiler = InferenceProfiler(config)
        with pytest.raises(KeyboardInterrupt):
            profiler.run()
    presser.join(5)
    export = profiler.export
    assert export is not None and export.registry.frozen
    assert closes and closes[0] < 2.5
    records = _records(output)
    assert records[-1]["event_type"] == "infer.session"
    assert records[-1]["status"] == "interrupted"
    summary = _capability(records)["metadata"]["summary"]["records"]
    assert summary["offered"] == summary["applied"] + sum(summary["dropped"].values())
    assert summary["dropped"]["shutdown"] > 0
    exposition = check_exposition((metrics_dir / "stormlog-default.prom").read_text())
    assert exposition.value("stormlog_run_active") == 0
    assert not (metrics_dir / "stormlog-default.lock").exists()


def test_a_ctrl_c_while_the_capability_record_is_written_waits_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Ctrl+C lands after the close, as the export's capability record
    # is built: the record is still written, then the run stops.
    real_events = ExportPipeline.capability_events

    def ctrl_c_then_events(self: ExportPipeline, context: Any) -> Any:
        signal.raise_signal(signal.SIGINT)
        return real_events(self, context)

    monkeypatch.setattr(ExportPipeline, "capability_events", ctrl_c_then_events)
    output = tmp_path / "infer.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(prometheus_textfile_dir=metrics_dir),
            request_count=4,
            warmup_requests=0,
            stream=False,
        )
        with pytest.raises(KeyboardInterrupt):
            InferenceProfiler(config).run()
    records = _records(output)
    summary = _capability(records)["metadata"]["summary"]["records"]
    assert summary["offered"] == summary["applied"] + sum(summary["dropped"].values())
    assert records[-1]["event_type"] == "infer.session"
    assert records[-1]["status"] == "interrupted"


def test_a_ctrl_c_while_server_evidence_is_imported_still_ends_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After the capture, the run imports traces and the execution log; a
    # Ctrl+C there must still leave the artifact's terminal record.
    async def interrupted_import(self: InferenceProfiler, output_path: Path) -> None:
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(5)

    monkeypatch.setattr(
        InferenceProfiler, "_import_server_evidence", interrupted_import
    )
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        config = _config(
            endpoint, output, ExportConfig(), request_count=2, warmup_requests=0
        )
        with pytest.raises(KeyboardInterrupt):
            InferenceProfiler(config).run()
    records = _records(output)
    assert records[-1]["event_type"] == "infer.session"
    assert records[-1]["status"] == "interrupted"


def test_a_held_ctrl_c_is_delivered_once_the_block_ends() -> None:
    before = signal.getsignal(signal.SIGINT)
    finished = False
    with pytest.raises(KeyboardInterrupt):
        with _ctrl_c_held():
            signal.raise_signal(signal.SIGINT)
            time.sleep(0.05)  # time for the handler to run, were it not held
            finished = True
    assert finished
    assert signal.getsignal(signal.SIGINT) is before


def test_without_a_python_handler_a_ctrl_c_is_not_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A handler set from C cannot be put back, so the block runs as is.
    monkeypatch.setattr(signal, "getsignal", lambda _signum: None)
    finished = False
    with pytest.raises(KeyboardInterrupt):
        with _ctrl_c_held():
            signal.raise_signal(signal.SIGINT)
            time.sleep(0.05)
            finished = True
    assert not finished


def test_off_the_main_thread_the_block_runs_as_is() -> None:
    before = signal.getsignal(signal.SIGINT)
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            with _ctrl_c_held():
                pass
        except BaseException as exc:  # noqa: B036 - reported below
            errors.append(exc)

    thread = threading.Thread(target=hold)
    thread.start()
    thread.join(5)
    assert not errors
    assert signal.getsignal(signal.SIGINT) is before


# ------------------------------------------------------------------ CLI
def _cli(endpoint: str, output: Path, *extra: str) -> int:
    return infer_main(
        [
            "profile", "--endpoint", endpoint, "--model", "fake-model",
            "--requests", "1", "--input-tokens", "8", "--output-tokens", "4",
            "--tokenizer", "none", "--system-sampler", "none",
            "--output", str(output), *extra,
        ]
    )  # fmt: skip


def test_a_run_over_budget_is_refused_before_it_sends(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        code = _cli(
            endpoint, output, "--prometheus-listen", "127.0.0.1:0",
            "--prometheus-max-series", "10",
        )  # fmt: skip
    assert code == ExitCode.USAGE and not output.exists()


def test_a_held_slot_is_refused_before_it_sends(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    holder = TextfileWriter(
        tmp_path,
        "default",
        RenderCache(lambda: b""),
        const_labels={PRODUCER_LABEL: "default"},
    )
    holder.acquire()
    try:
        with _fake_server() as endpoint:
            code = _cli(endpoint, output, "--prometheus-textfile-dir", str(tmp_path))
    finally:
        holder.close()
    assert code == ExitCode.USAGE and not output.exists()


def test_a_run_that_fails_after_the_export_started_still_ends_it(
    tmp_path: Path,
) -> None:
    # The artifact cannot be opened once the exporter has started, so the
    # run's own close is the first one: it must still write the final file.
    output = tmp_path / "infer.jsonl"
    output.mkdir()
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with _fake_server() as endpoint:
        code = _cli(endpoint, output, "--prometheus-textfile-dir", str(metrics_dir))
    assert code != 0
    text = (metrics_dir / "stormlog-default.prom").read_text()
    assert check_exposition(text).value("stormlog_run_active") == 0
    assert not (metrics_dir / "stormlog-default.lock").exists()
    assert not list(metrics_dir.glob("*.tmp"))


def test_a_textfile_directory_it_cannot_write_is_refused_before_it_sends(
    tmp_path: Path,
) -> None:
    output = tmp_path / "infer.jsonl"
    metrics_dir = tmp_path / "read-only"
    metrics_dir.mkdir()
    metrics_dir.chmod(0o555)
    try:
        with _fake_server() as endpoint:
            code = _cli(endpoint, output, "--prometheus-textfile-dir", str(metrics_dir))
    finally:
        metrics_dir.chmod(0o755)
    assert code == ExitCode.USAGE and not output.exists()


def test_export_flags_without_an_output_are_refused(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        code = _cli(endpoint, output, "--prometheus-slot", "alpha")
    assert code == ExitCode.USAGE and not output.exists()


def test_the_cli_exports_to_a_textfile(tmp_path: Path) -> None:
    output = tmp_path / "out" / "infer.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with _fake_server() as endpoint:
        code = _cli(
            endpoint, output, "--prometheus-textfile-dir", str(metrics_dir),
            "--prometheus-slot", "bench",
        )  # fmt: skip
    assert code == ExitCode.OK
    text = (metrics_dir / "stormlog-bench.prom").read_text()
    assert 'stormlog_producer="bench"' in text
    check_exposition(text)


def test_a_config_for_replace_keeps_its_export(tmp_path: Path) -> None:
    config = _config("http://h/v1/chat/completions", tmp_path / "x", ExportConfig())
    changed = replace(config, request_count=5)
    assert changed.export == config.export
