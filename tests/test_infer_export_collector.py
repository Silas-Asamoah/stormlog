"""T15: ``infer collect-server``'s health export, through every way it stops."""

import json
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

from stormlog._export.registry import render
from stormlog._export.renders import RenderCache
from stormlog._export.textfile import PRODUCER_LABEL, TextfileWriter, slot_paths
from stormlog.exit_codes import ExitCode
from stormlog.infer import export_collector, server_collector
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
from stormlog.infer.telemetry import ServerIdentity
from tests.export_conformance import Exposition, check_exposition
from tests.test_infer_telemetry import _FakeGpu


def _export(tmp_path: Path, **config: Any) -> CollectorExport:
    metrics = tmp_path / "metrics"
    metrics.mkdir(exist_ok=True)
    config.setdefault("prometheus_textfile_dir", metrics)
    return CollectorExport(ExportConfig(**config), run_id="run-c", version="9.9")


def _identity() -> ServerIdentity:
    return ServerIdentity(
        host=socket.gethostname(),
        pid=os.getpid(),
        process_start_ns=int(psutil.Process().create_time() * 1e9),
    )


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
    assert len(info) == 1 and info[0][1] == 1
    labels = info[0][0]
    assert labels["pid"] == str(os.getpid())
    assert labels["device_uuid"] == "GPU-live" and labels["replica_id"] == "r-1"
    assert labels["run_id"] == "run-c" and labels["version"] == "9.9"
    assert labels["gpu_instance_id"] == "" and labels["rank"] == ""
    process_start = int(psutil.Process().create_time() * 1e9)
    assert labels["process_start_ns"] == str(process_start)
    assert exposition.value("stormlog_collector_polls_total") == result.polls
    last_poll = exposition.value("stormlog_collector_last_poll_timestamp_seconds")
    assert time.time() - 60 < last_poll <= time.time()
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


class _GpuShowing(_FakeGpu):
    """A GPU on which NVML lists the given compute processes."""

    def __init__(self, pids: set[int] | None) -> None:
        super().__init__()
        self.pids = pids

    def compute_pids(self) -> set[int] | None:
        return self.pids


@pytest.mark.parametrize(
    ("gpu", "match"),
    [
        (_GpuShowing({os.getpid()}), "confirmed"),
        (_GpuShowing({1}), "not_seen"),  # chosen by index, the server elsewhere
        (_GpuShowing(None), "unknown"),  # NVML could not list its processes
        (_FakeGpu(), "unknown"),
    ],
)
def test_a_series_says_whether_the_server_was_seen_on_its_gpu(
    tmp_path: Path, gpu: _FakeGpu, match: str
) -> None:
    # device_uuid is the GPU the collector watched; whether the server was
    # on it is a separate fact, never implied.
    export = _export(tmp_path)
    _collect(tmp_path, export, duration_seconds=0.05, gpu_source=gpu)
    exposition = _textfile(tmp_path)
    ((labels, value),) = _series(exposition, "stormlog_collector_info")
    assert labels["device_uuid"] == "GPU-live" and value == 1
    assert "gpu_process_match" not in labels
    assert _match_states(exposition) == {
        state: float(state == match) for state in ("confirmed", "not_seen", "unknown")
    }


def _match_states(exposition: Exposition) -> dict[str, float]:
    return {
        labels["state"]: value
        for labels, value in _series(exposition, "stormlog_collector_gpu_process_match")
    }


class _GpuLater(_GpuShowing):
    """NVML lists the server only from its second listing on, as when the
    collector starts while the engine is still loading its model."""

    def __init__(self) -> None:
        super().__init__(set())
        self.calls = 0

    def compute_pids(self) -> set[int] | None:
        self.calls += 1
        return set() if self.calls == 1 else {os.getpid()}


def test_a_server_that_reaches_its_gpu_later_is_confirmed_then(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_collector, "MATCH_RECHECK_SECONDS", 0.0)
    gpu = _GpuLater()
    export = _export(tmp_path)
    _collect(tmp_path, export, duration_seconds=0.2, gpu_source=gpu)
    assert gpu.calls >= 2
    assert _match_states(_textfile(tmp_path))["confirmed"] == 1


def test_a_child_of_the_server_on_its_gpu_confirms_it(tmp_path: Path) -> None:
    # vLLM's EngineCore is a child of the API server: NVML shows the child.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        export = _export(tmp_path)
        _collect(
            tmp_path, export, duration_seconds=0.05, gpu_source=_GpuShowing({child.pid})
        )
        assert _match_states(_textfile(tmp_path))["confirmed"] == 1
    finally:
        child.kill()
        child.wait()


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

    def end_server() -> None:
        # Reaped at once, as a server's parent would: an unreaped child stays
        # a zombie for the whole collection.
        child.kill()
        child.wait()

    timer = threading.Timer(0.2, end_server)
    timer.start()
    try:
        # The duration only stops a run that missed the end: it fails, not hangs.
        result = _collect(
            tmp_path,
            export,
            pid=child.pid,
            gpu_source=None,
            no_gpu=True,
            duration_seconds=20.0,
        )
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


class _BrokenGpu(_FakeGpu):
    """A GPU source whose second read raises, as a bug in a reader would."""

    def read(self) -> GpuMemoryReading:
        if self.reads >= 1:
            raise RuntimeError("reader bug")
        return super().read()


def test_a_collection_that_raises_is_recorded_as_an_error(tmp_path: Path) -> None:
    export = _export(tmp_path)
    with pytest.raises(RuntimeError, match="reader bug"):
        _collect(tmp_path, export, duration_seconds=5, gpu_source=_BrokenGpu())
    final = _textfile(tmp_path)
    assert final.value("stormlog_collector_running") == 0
    assert final.value("stormlog_collector_stops_total", reason="error") == 1
    assert (
        final.value("stormlog_collector_stops_total", reason=STOP_DURATION_ELAPSED) == 0
    )


def test_an_unknown_stop_reason_is_recorded_as_an_error(tmp_path: Path) -> None:
    export = _export(tmp_path)
    export.identify(_identity())
    export.close("a reason from a newer collector")
    stops = "stormlog_collector_stops_total"
    assert _textfile(tmp_path).value(stops, reason="error") == 1


def test_nothing_changes_after_the_close(tmp_path: Path) -> None:
    export = _export(tmp_path)
    _collect(tmp_path, export, duration_seconds=0.05)
    polls = export.polls
    export.poll([])  # a late poll, after the freeze
    rendered = check_exposition(render(export.registry.snapshot()).decode()).value(
        "stormlog_collector_polls_total"
    )
    assert rendered == polls


def test_a_poll_that_fails_to_count_never_reaches_the_collector(
    tmp_path: Path,
) -> None:
    export = _export(tmp_path)
    export.identify(_identity())
    export.poll([object()])  # type: ignore[list-item]
    export.close(STOP_REQUESTED)


class _Recording:
    """An observer that checks each poll is on disk when it is told."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.on_disk: list[bool] = []
        self.closed: list[str] = []

    def identify(self, identity: Any, gpu_process_match: str = "unknown") -> None:
        pass

    def matched(self, gpu_process_match: str) -> None:
        pass

    def poll(self, samples: Any) -> None:
        lines = self.path.read_text().splitlines()
        self.on_disk.append(
            all(json.dumps(s.to_record(), sort_keys=True) in lines for s in samples)
        )

    def close(self, stop_reason: str) -> None:
        self.closed.append(stop_reason)


def test_the_observer_hears_of_a_poll_once_it_is_written(tmp_path: Path) -> None:
    observer = _Recording(tmp_path / "server.jsonl")
    _collect(tmp_path, observer, duration_seconds=0.05)  # type: ignore[arg-type]
    assert observer.on_disk and all(observer.on_disk)
    assert observer.closed == [STOP_DURATION_ELAPSED]


def test_an_in_flight_render_at_the_close_still_ends_in_the_final_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A periodic render that took its values before the freeze, held until
    # after it: the final file must still say the collector stopped, and why.
    export = _export(tmp_path, prometheus_textfile_interval_seconds=1.0)
    real_render = export_collector.render
    renders: list[int] = []

    def held(snapshot: Any) -> bytes:
        renders.append(1)
        if len(renders) > 1:  # every render after the first, until the freeze
            deadline = time.monotonic() + 10
            while not export.registry.frozen and time.monotonic() < deadline:
                time.sleep(0.01)
        return real_render(snapshot)

    monkeypatch.setattr(export_collector, "render", held)
    _collect(tmp_path, export, duration_seconds=2.5)
    final = _textfile(tmp_path)
    assert len(renders) >= 3
    assert final.value("stormlog_collector_running") == 0
    assert final.value("stormlog_run_active") == 0
    assert (
        final.value("stormlog_collector_stops_total", reason=STOP_DURATION_ELAPSED) == 1
    )


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


def test_the_endpoint_stops_when_the_collection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The collection fails once the export serves: its output cannot be
    # opened. No linger follows a failure, and the endpoint must not outlive
    # the command in a caller's process.
    port = _free_port()
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so nothing can be created under it")
    stops: list[bool] = []
    real_stop = CollectorExport.stop_serving

    def noting_stop(self: CollectorExport) -> None:
        stops.append(self.server is not None)
        real_stop(self)

    monkeypatch.setattr(CollectorExport, "stop_serving", noting_stop)
    code = infer_main(
        [
            "collect-server",
            "--run-id",
            "run-c",
            "--pid",
            str(os.getpid()),
            "--no-gpu",
            "--duration",
            "5",
            "--output",
            str(blocker / "server.jsonl"),
            "--prometheus-listen",
            f"127.0.0.1:{port}",
        ]
    )
    assert code != ExitCode.OK
    assert stops == [True]
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


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


def test_a_collection_that_fails_to_start_frees_its_slot(tmp_path: Path) -> None:
    # The slot is taken before the process is looked up; a --pid that has
    # exited must not leave the lock behind.
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(10)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    code = infer_main(
        [
            "collect-server",
            "--run-id",
            "run-c",
            "--pid",
            str(exited.pid),
            "--no-gpu",
            "--output",
            str(tmp_path / "server.jsonl"),
            "--prometheus-textfile-dir",
            str(metrics),
        ]
    )
    assert code == ExitCode.USAGE
    assert not list(metrics.glob("*.lock"))


def test_the_slot_is_checked_before_the_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    holder = TextfileWriter(
        metrics, "held", RenderCache(lambda: b""), const_labels={PRODUCER_LABEL: "held"}
    )
    holder.acquire()
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(10)
    try:
        code = infer_main(
            [
                "collect-server",
                "--run-id",
                "run-c",
                "--pid",
                str(exited.pid),
                "--no-gpu",
                "--output",
                str(tmp_path / "server.jsonl"),
                "--prometheus-textfile-dir",
                str(metrics),
                "--prometheus-slot",
                "held",
            ]
        )
    finally:
        holder.close()
    assert code == ExitCode.USAGE
    assert "--prometheus-slot" in capsys.readouterr().err


def test_the_endpoint_is_stopped_after_the_linger(tmp_path: Path) -> None:
    port = _free_port()
    code = _cli(
        tmp_path,
        "--duration",
        "0.1",
        "--prometheus-listen",
        f"127.0.0.1:{port}",
        "--prometheus-linger",
        "0.2",
    )
    assert code == 0
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2)


def test_an_output_inside_the_textfile_directory_exits_2(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    code = infer_main(
        [
            "collect-server",
            "--run-id",
            "run-c",
            "--pid",
            str(os.getpid()),
            "--no-gpu",
            "--duration",
            "0.1",
            "--output",
            str(metrics / "server.jsonl"),
            "--prometheus-textfile-dir",
            str(metrics),
        ]
    )
    assert code == ExitCode.USAGE


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
            ["--prometheus-listen", "127.0.0.1:9", "--prometheus-linger", "inf"],
            [
                "--prometheus-textfile-dir",
                str(metrics),
                "--prometheus-textfile-interval",
                "inf",
            ],
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
