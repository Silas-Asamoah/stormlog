"""Prometheus export from a real ``infer profile`` run against a fake server."""

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
from stormlog.infer.export_config import ExportConfig
from stormlog.infer.profile import InferenceProfiler
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


def test_ctrl_c_still_freezes_and_records_final_counts(tmp_path: Path) -> None:
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
        timer = threading.Timer(0.8, signal.raise_signal, (signal.SIGINT,))
        timer.start()
        started = time.perf_counter()
        try:
            with pytest.raises(KeyboardInterrupt):
                profiler.run()
        finally:
            timer.cancel()
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
