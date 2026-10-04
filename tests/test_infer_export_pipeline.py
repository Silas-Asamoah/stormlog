"""The export pipeline: bounded, isolated, and closed in a fixed order."""

import argparse
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from stormlog._export.queue import BoundedQueue
from stormlog._export.registry import BudgetExceeded
from stormlog.infer.correlation_events import CorrelationContext
from stormlog.infer.export import (
    ExportPipeline,
    ExportUsageError,
    HealthMetric,
    HealthValue,
    ReceiverHealth,
)
from stormlog.infer.export_config import (
    ExportConfig,
    add_export_arguments,
    export_config_from_args,
)
from stormlog.infer.export_metrics import ProfileLabels
from tests.export_conformance import Exposition, check_exposition

LABELS = ProfileLabels(
    model="m",
    server="http://127.0.0.1:8000",
    cases=(("c1_in8_out8", "closed"),),
    run_id="run-1",
    session_id="session-1",
    version="0.0.0",
)


def _request(status: str = "ok") -> dict[str, Any]:
    return {
        "event_type": "infer.request",
        "case_id": "c1_in8_out8",
        "phase": "measured",
        "status": status,
        "e2e_latency_ms": 100.0,
        "arrival_mode": "closed",
    }


def _exposition(pipeline: ExportPipeline) -> Exposition:
    pipeline.renders.invalidate()  # a fresh render, not the shared one
    generation = pipeline.renders.acquire()
    try:
        return check_exposition(generation.body.decode())
    finally:
        pipeline.renders.release(generation)


def _wait_for(condition, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def textfile_pipeline(tmp_path: Path) -> Iterator[ExportPipeline]:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    pipeline.start(started_at=1_700_000_000.0)
    yield pipeline
    pipeline.close(2.0)
    pipeline.stop_serving()


def test_observed_records_reach_the_metrics(textfile_pipeline: ExportPipeline) -> None:
    for status in ("ok", "ok", "timeout"):
        textfile_pipeline.observe(_request(status))
    assert _wait_for(lambda: textfile_pipeline.health()["applied"] == 3)
    exposition = _exposition(textfile_pipeline)
    name = "stormlog_infer_requests_total"
    assert exposition.value(name, status="ok", phase="measured") == 2
    assert exposition.value(name, status="timeout", phase="measured") == 1
    assert exposition.value("stormlog_metrics_records_applied_total") >= 0
    assert exposition.value("stormlog_run_info", stormlog_producer="t") == 1


def test_close_applies_what_was_queued_then_freezes(tmp_path: Path) -> None:
    pipeline = ExportPipeline(ExportConfig(prometheus_textfile_dir=tmp_path), LABELS)
    pipeline.start(started_at=0.0)
    for _ in range(50):
        pipeline.observe(_request())
    pipeline.close(5.0)
    summary = pipeline.summary()["records"]
    assert summary["applied"] == 50 and summary["exact"]
    assert summary["dropped"] == {"queue_full": 0, "closed": 0, "shutdown": 0}
    pipeline.observe(_request())  # after close: not taken, and harmless
    assert pipeline.summary()["records"]["offered"] == 50
    text = (tmp_path / "stormlog-default.prom").read_text()
    exposition = check_exposition(text)
    assert (
        exposition.value("stormlog_infer_requests_total", status="ok", phase="measured")
        == 50
    )
    assert exposition.value("stormlog_run_active") == 0


def test_records_a_paused_worker_never_applied_are_dropped_at_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = ExportPipeline(ExportConfig(prometheus_textfile_dir=tmp_path), LABELS)
    release = threading.Event()
    real_apply = pipeline._apply

    def paused(envelope: Any) -> bool:
        release.wait(10)
        return real_apply(envelope)

    monkeypatch.setattr(pipeline, "_apply", paused)
    pipeline.start(started_at=0.0)
    for _ in range(100):
        pipeline.observe(_request())
    started = time.monotonic()
    pipeline.close(0.3)
    assert time.monotonic() - started < 3
    frozen = _exposition(pipeline)
    release.set()  # the worker wakes after the freeze and must change nothing
    time.sleep(0.2)
    summary = pipeline.summary()["records"]
    assert summary["dropped"]["shutdown"] == 100 and not summary["exact"]
    assert _exposition(pipeline).samples == frozen.samples
    assert (
        frozen.value("stormlog_metrics_records_dropped_total", reason="shutdown") == 100
    )


def test_a_full_queue_drops_counts_and_degrades_health(tmp_path: Path) -> None:
    pipeline = ExportPipeline(ExportConfig(prometheus_textfile_dir=tmp_path), LABELS)
    pipeline.queue = BoundedQueue(max_items=2, max_bytes=10**6)  # no worker yet
    for _ in range(5):
        pipeline.observe(_request())
    pipeline._poll_health()
    assert pipeline.health()["dropped"]["queue_full"] == 3
    assert pipeline.health()["status"] == "degraded"
    pipeline.close(1.0)


def test_observe_never_raises(
    textfile_pipeline: ExportPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(record: Any, extras: Any) -> None:
        raise RuntimeError("mapper bug")

    monkeypatch.setattr(textfile_pipeline.metrics, "envelope", broken)
    textfile_pipeline.observe(_request())
    assert textfile_pipeline.summary()["internal_errors"]["observe"] == 1


def test_a_run_over_budget_is_refused_before_it_starts() -> None:
    with pytest.raises(BudgetExceeded):
        ExportPipeline(
            ExportConfig(prometheus_listen="127.0.0.1:0", prometheus_max_series=10),
            LABELS,
        )


def test_a_slot_clash_is_a_usage_error_before_the_run(tmp_path: Path) -> None:
    config = ExportConfig(prometheus_textfile_dir=tmp_path)
    first = ExportPipeline(config, LABELS)
    first.prepare()
    try:
        with pytest.raises(ExportUsageError, match="--prometheus-slot"):
            ExportPipeline(config, LABELS).prepare()
    finally:
        first.close(1.0)


def test_an_endpoint_that_cannot_bind_warns_and_the_run_goes_on() -> None:
    blocker = ExportPipeline(ExportConfig(prometheus_listen="127.0.0.1:0"), LABELS)
    blocker.start(started_at=0.0)
    assert blocker.server is not None
    warnings: list[str] = []
    pipeline = ExportPipeline(
        ExportConfig(prometheus_listen=blocker.server.address),
        LABELS,
        on_warning=warnings.append,
    )
    pipeline.start(started_at=0.0)
    pipeline.observe(_request())
    pipeline.close(1.0)
    assert pipeline.server is None and pipeline.server_error
    assert any("could not listen" in warning for warning in warnings)
    event = pipeline.capability_events(_context())[0]
    assert event.available is False and event.metadata["summary"]["endpoint_error"]
    blocker.close(1.0)
    blocker.stop_serving()


def test_a_non_loopback_endpoint_warns_once() -> None:
    warnings: list[str] = []
    pipeline = ExportPipeline(
        ExportConfig(prometheus_listen="0.0.0.0:0"), LABELS, on_warning=warnings.append
    )
    pipeline.start(started_at=0.0)
    pipeline.close(1.0)
    pipeline.stop_serving()
    assert sum("no authentication" in warning for warning in warnings) == 1


class _Source:
    def __init__(self) -> None:
        self.values: dict[str, HealthValue] = {}

    def health(self) -> Mapping[str, HealthValue]:
        return dict(self.values)

    def health_metrics(self) -> Sequence[HealthMetric]:
        return (
            HealthMetric("depth", "stormlog_watch_depth_bytes", "gauge", "bytes", "d"),
            HealthMetric(
                "incidents_total",
                "stormlog_watch_incidents_total",
                "counter",
                "",
                "i",
                labels=("trigger_kind", "capture_status"),
            ),
            HealthMetric(
                "trigger_state",
                "stormlog_watch_trigger_state",
                "state",
                "",
                "s",
                labels=("trigger_id",),
                states=("inactive", "pending", "firing", "resolving"),
            ),
        )


def test_health_sources_become_metrics_and_none_stays_out(tmp_path: Path) -> None:
    source = _Source()
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_series_headroom=8),
        LABELS,
        health=[("watch", source.health_metrics())],
    )
    pipeline.attach_health("watch", source)
    source.values = {
        "depth": None,
        "incidents_total": {("metric", "captured"): 3},
        "trigger_state": {("p95_ttft",): "firing"},
    }
    pipeline._poll_health()
    exposition = _exposition(pipeline)
    assert not exposition.matching("stormlog_watch_depth_bytes")
    assert (
        exposition.value(
            "stormlog_watch_incidents_total",
            trigger_kind="metric",
            capture_status="captured",
        )
        == 3
    )
    state = "stormlog_watch_trigger_state"
    assert exposition.value(state, trigger_id="p95_ttft", state="firing") == 1
    assert exposition.value(state, trigger_id="p95_ttft", state="pending") == 0
    pipeline.close(1.0)


def test_a_failing_health_source_is_counted_not_raised(tmp_path: Path) -> None:
    class Broken(_Source):
        def health(self) -> Mapping[str, HealthValue]:
            raise RuntimeError("source bug")

    source = Broken()
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path),
        LABELS,
        health=[("broken", source.health_metrics())],
    )
    pipeline.attach_health("broken", source)
    pipeline._poll_health()
    assert pipeline.summary()["internal_errors"]["health"] == 1
    pipeline.close(1.0)


def test_the_span_receiver_reports_through_its_metadata(tmp_path: Path) -> None:
    class FakeReceiver:
        def capability_metadata(self) -> dict[str, Any]:
            return {"spans": 12, "decode_failures": 2, "grpc_attempts": 1}

    health = ReceiverHealth(FakeReceiver())
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path),
        LABELS,
        health=[("receiver", health.health_metrics())],
    )
    pipeline.attach_health("receiver", health)
    pipeline._poll_health()
    exposition = _exposition(pipeline)
    assert exposition.value("stormlog_engine_span_receiver_spans_total") == 12
    requests = "stormlog_engine_span_receiver_requests_total"
    assert exposition.value(requests, outcome="decode_failure") == 2
    assert exposition.value(requests, outcome="grpc_attempt") == 1
    pipeline.close(1.0)


def _context() -> CorrelationContext:
    return CorrelationContext(
        run_id="run-1",
        session_id="session-1",
        producer_id="stormlog.infer.profile",
        source="stormlog.infer.profile",
        clock_domain="host/boot/unix_epoch_ns",
        clock_kind="wall",
        collection_mode="active",
        provenance="observed",
    )


def test_the_capability_record_says_what_was_asked_and_what_worked(
    tmp_path: Path,
) -> None:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_listen="127.0.0.1:0", prometheus_textfile_dir=tmp_path),
        LABELS,
    )
    pipeline.start(started_at=0.0)
    pipeline.observe(_request())
    pipeline.close(2.0)
    event = pipeline.capability_events(_context())[0]
    record = event.to_record()
    assert record["component"] == "export.prometheus" and record["available"]
    assert record["enabled"] == ["endpoint", "textfile"]
    assert record["collected"] == ["endpoint", "textfile"]
    summary = record["metadata"]["summary"]
    assert summary["records"]["applied"] == 1 and summary["slot"] == "default"
    assert summary["textfile"]["path"] == "stormlog-default.prom"
    pipeline.stop_serving()


# ------------------------------------------------------------------ config
def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_export_arguments(parser)
    return parser.parse_args(argv)


def test_flags_build_the_config(tmp_path: Path) -> None:
    config = export_config_from_args(
        _args(
            "--prometheus-listen", "127.0.0.1:9999",
            "--prometheus-linger", "2",
            "--prometheus-textfile-dir", str(tmp_path),
            "--prometheus-slot", "alpha",
            "--prometheus-case-label", "off",
        )
    )  # fmt: skip
    assert config.prometheus_listen == "127.0.0.1:9999"
    assert config.prometheus_linger_seconds == 2.0
    assert config.prometheus_textfile_dir == tmp_path
    assert config.prometheus_slot == "alpha" and not config.prometheus_case_label
    assert config.headroom("profile") == 0 and config.headroom("watch") == 64


def test_no_flags_export_nothing() -> None:
    assert not export_config_from_args(_args()).enabled


@pytest.mark.parametrize(
    "argv",
    [
        ("--prometheus-listen", "9999"),
        ("--prometheus-slot", "a b", "--prometheus-listen", "127.0.0.1:1"),
        ("--prometheus-linger", "2"),
        ("--prometheus-slot", "alpha"),
        ("--prometheus-textfile-remove-on-exit",),
        ("--prometheus-listen", "127.0.0.1:1", "--prometheus-max-bytes", "10"),
        ("--prometheus-listen", "127.0.0.1:1", "--prometheus-max-series", "0"),
        ("--prometheus-listen", "127.0.0.1:1", "--prometheus-series-headroom", "-1"),
        ("--prometheus-listen", "127.0.0.1:1", "--prometheus-linger", "-1"),
    ],
)
def test_unusable_settings_are_refused(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        export_config_from_args(_args(*argv))


def test_the_watch_json_section_uses_the_same_settings(tmp_path: Path) -> None:
    config = ExportConfig.from_mapping(
        {"prometheus_textfile_dir": str(tmp_path), "prometheus_slot": "watch-1"}
    )
    assert config.prometheus_textfile_dir == tmp_path
    with pytest.raises(ValueError, match="unknown export settings"):
        ExportConfig.from_mapping({"prometheus_port": 1})


class _RecordingLock:
    """Wraps the registry's lock and notes every thread that takes it."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.takers: set[int] = set()

    def __enter__(self) -> Any:
        self.takers.add(threading.get_ident())
        return self.inner.__enter__()

    def __exit__(self, *exc: object) -> Any:
        return self.inner.__exit__(*exc)


def test_observe_takes_no_registry_lock_and_does_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import os
    import socket

    pipeline = ExportPipeline(ExportConfig(prometheus_textfile_dir=tmp_path), LABELS)
    lock = _RecordingLock(pipeline.registry._lock)
    pipeline.registry._lock = lock  # type: ignore[assignment]

    def no_io(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("observe did I/O")

    with monkeypatch.context() as patched:
        for target, name in (
            (builtins, "open"),
            (os, "write"),
            (socket.socket, "send"),
            (socket.socket, "sendall"),
            (socket.socket, "connect"),
        ):
            patched.setattr(target, name, no_io)
        for _ in range(10):
            pipeline.observe(_request(), {"chunk_summary": ((0,) * 15, 0.0)})
    assert threading.get_ident() not in lock.takers
    assert pipeline.queue.stats().accepted == 10
    assert pipeline.summary()["internal_errors"]["observe"] == 0
    pipeline.close(1.0)
