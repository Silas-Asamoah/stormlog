"""T15: ``infer collect-server``'s health export, through every way it stops."""

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import psutil
import pytest

from stormlog._export.renders import RenderCache
from stormlog._export.textfile import PRODUCER_LABEL, TextfileWriter, slot_paths
from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.export_collector import CollectorExport
from stormlog.infer.export_config import ExportConfig
from stormlog.infer.server_collector import (
    STOP_DURATION_ELAPSED,
    STOP_GPU_IDENTITY_CHANGED,
    STOP_REQUESTED,
    STOP_SERVER_PROCESS_ENDED,
    GpuMemoryReading,
    collect_server_telemetry,
)
from tests.export_conformance import Exposition, check_exposition
from tests.test_infer_telemetry import _FakeGpu


def _export(tmp_path: Path, **config: Any) -> CollectorExport:
    metrics = tmp_path / "metrics"
    metrics.mkdir(exist_ok=True)
    config.setdefault("prometheus_textfile_dir", metrics)
    return CollectorExport(ExportConfig(**config), run_id="run-c", version="9.9")


def _textfile(tmp_path: Path, slot: str = "default") -> Exposition:
    path, _lock = slot_paths(tmp_path / "metrics", slot)
    return check_exposition(path.read_text())


def _series(exposition: Exposition, name: str) -> list[tuple[dict[str, str], float]]:
    return [
        (dict(labels), value)
        for (sample_name, labels), value in exposition.samples.items()
        if sample_name == name
    ]


def _collect(tmp_path: Path, export: CollectorExport, **kw: Any) -> Any:
    kw.setdefault("gpu_source", _FakeGpu())
    kw.setdefault("pid", os.getpid())
    return collect_server_telemetry(
        run_id="run-c",
        output_path=tmp_path / "server.jsonl",
        interval_seconds=0.01,
        observer=export,
        **kw,
    )


def test_the_final_textfile_has_the_identity_polls_and_why_it_stopped(
    tmp_path: Path,
) -> None:
    export = _export(tmp_path)
    result = _collect(tmp_path, export, duration_seconds=0.1, replica_id="r-1")
    assert result.stop_reason == STOP_DURATION_ELAPSED
    exposition = _textfile(tmp_path)
    info = _series(exposition, "stormlog_collector_info")
    assert len(info) == 1
    labels = info[0][0]
    assert labels["pid"] == str(os.getpid())
    assert labels["device_uuid"] == "GPU-live" and labels["replica_id"] == "r-1"
    assert labels["run_id"] == "run-c" and labels["version"] == "9.9"
    assert labels["gpu_instance_id"] == "" and labels["rank"] == ""
    process_start = int(psutil.Process().create_time() * 1e9)
    assert labels["process_start_ns"] == str(process_start)
    assert exposition.value("stormlog_collector_polls_total") == result.polls
    assert (
        exposition.value(
            "stormlog_collector_samples_total",
            metric="device_memory_used_bytes",
            state="valid",
        )
        == result.polls
    )
    assert exposition.value("stormlog_collector_running") == 0
    assert exposition.value("stormlog_run_active") == 0
    stops = {
        labels["reason"]: value
        for labels, value in _series(exposition, "stormlog_collector_stops_total")
    }
    assert stops[STOP_DURATION_ELAPSED] == 1
    assert sum(stops.values()) == 1


def test_a_gpu_identity_change_is_the_recorded_stop(tmp_path: Path) -> None:
    export = _export(tmp_path)
    gpu = _FakeGpu(GpuMemoryReading(None, None, "invalid", "device UUID changed"))
    result = _collect(tmp_path, export, duration_seconds=1.0, gpu_source=gpu)
    assert result.stop_reason == STOP_GPU_IDENTITY_CHANGED
    exposition = _textfile(tmp_path)
    assert (
        exposition.value(
            "stormlog_collector_stops_total", reason=STOP_GPU_IDENTITY_CHANGED
        )
        == 1
    )
    assert (
        exposition.value(
            "stormlog_collector_samples_total",
            metric="device_memory_used_bytes",
            state="invalid",
        )
        == 1
    )


def test_a_server_that_ends_is_the_recorded_stop(tmp_path: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    export = _export(tmp_path)
    timer = threading.Timer(0.2, child.kill)
    timer.start()
    try:
        result = _collect(tmp_path, export, pid=child.pid, gpu_source=None, no_gpu=True)
    finally:
        timer.cancel()
        child.wait(5)
    assert result.stop_reason == STOP_SERVER_PROCESS_ENDED
    exposition = _textfile(tmp_path)
    assert (
        exposition.value(
            "stormlog_collector_stops_total", reason=STOP_SERVER_PROCESS_ENDED
        )
        == 1
    )
    ((labels, _value),) = _series(exposition, "stormlog_collector_info")
    assert labels["device_uuid"] == ""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_the_endpoint_serves_while_collecting_and_stops_with_a_request(
    tmp_path: Path,
) -> None:
    port = _free_port()
    export = _export(tmp_path, prometheus_listen=f"127.0.0.1:{port}")
    stop = threading.Event()
    scraped: list[Exposition] = []

    def scrape_then_stop() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not scraped:
            try:
                body = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/metrics", timeout=2
                ).read()
                exposition = check_exposition(body.decode())
                if (exposition.value("stormlog_collector_polls_total") or 0) >= 3:
                    scraped.append(exposition)
            except OSError:
                time.sleep(0.05)
        stop.set()

    scraper = threading.Thread(target=scrape_then_stop)
    scraper.start()
    result = _collect(tmp_path, export, stop_event=stop)
    scraper.join(5)
    export.stop_serving()
    assert result.stop_reason == STOP_REQUESTED
    assert scraped and scraped[0].value("stormlog_collector_running") == 1
    final = _textfile(tmp_path)
    assert final.value("stormlog_collector_stops_total", reason=STOP_REQUESTED) == 1


def _cli(tmp_path: Path, *flags: str) -> int:
    return infer_main(
        [
            "collect-server",
            "--run-id",
            "run-c",
            "--pid",
            str(os.getpid()),
            "--no-gpu",
            "--interval",
            "0.01",
            "--output",
            str(tmp_path / "server.jsonl"),
            *flags,
        ]
    )


def test_the_cli_exports_and_keeps_its_exit_code(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    code = _cli(
        tmp_path,
        "--duration",
        "0.1",
        "--prometheus-textfile-dir",
        str(metrics),
        "--prometheus-slot",
        "collector-0",
    )
    assert code == ExitCode.OK
    exposition = _textfile(tmp_path, "collector-0")
    assert exposition.value("stormlog_collector_polls_total") > 0
    producers = {
        dict(labels).get("stormlog_producer") for _, labels in exposition.samples
    }
    assert producers == {"collector-0"}


def test_settings_it_cannot_use_exit_2_before_collecting(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    # A live writer (this process) holds the slot.
    holder = TextfileWriter(
        metrics,
        "held",
        RenderCache(lambda: b""),
        const_labels={PRODUCER_LABEL: "held"},
    )
    holder.acquire()
    try:
        for flags in (
            ["--prometheus-textfile-dir", str(metrics), "--prometheus-slot", "held"],
            ["--prometheus-linger", "5"],
            ["--prometheus-listen", "127.0.0.1:9", "--prometheus-max-series", "1"],
            ["--prometheus-listen", "nonsense"],
        ):
            code = _cli(tmp_path, "--duration", "0.1", *flags)
            assert code == ExitCode.USAGE, flags
            assert not (tmp_path / "server.jsonl").exists(), flags
    finally:
        holder.close()


def test_collect_server_has_no_span_or_trace_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        infer_main(["collect-server", "--help"])
    usage = capsys.readouterr().out
    assert "--prometheus-listen" in usage
    for flag in ("--otlp-endpoint", "--trace-context", "--export-content"):
        assert flag not in usage
