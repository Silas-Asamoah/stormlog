"""The export pipeline: bounded, isolated, and closed in a fixed order."""

import argparse
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from stormlog._export.queue import BoundedQueue
from stormlog._export.registry import BudgetExceeded
from stormlog.infer import export as export_module
from stormlog.infer.correlation_events import CorrelationContext
from stormlog.infer.export import (
    RECEIVER_HEALTH,
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
    sampler_warnings,
)
from stormlog.infer.export_metrics import ProfileLabels
from stormlog.infer.export_spans import SpanIdentity
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
    assert summary["dropped"] == {
        "queue_full": 0,
        "closed": 0,
        "shutdown": 0,
        "error": 0,
    }
    pipeline.observe(_request())  # after close: not taken, and harmless
    assert pipeline.summary()["records"]["offered"] == 50
    text = (tmp_path / "stormlog-default.prom").read_text()
    exposition = check_exposition(text)
    assert (
        exposition.value("stormlog_infer_requests_total", status="ok", phase="measured")
        == 50
    )
    assert exposition.value("stormlog_run_active") == 0


def test_a_refused_token_count_makes_the_totals_inexact(tmp_path: Path) -> None:
    # The token total no longer matches the artifact, so exact must say so.
    pipeline = ExportPipeline(ExportConfig(prometheus_textfile_dir=tmp_path), LABELS)
    pipeline.start(started_at=0.0)
    for tokens in (4, -5, 10**400):
        record = _request()
        record.update(output_tokens=tokens, output_token_source="server_usage")
        pipeline.observe(record)
    pipeline.close(5.0)
    records = pipeline.summary()["records"]
    assert records["applied"] == 3 and records["tokens_rejected"] == 2
    assert not records["exact"]
    assert _exposition(pipeline).value("stormlog_metrics_series_rejected_total") == 0


def test_an_interrupted_close_finishes_and_a_later_close_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    pipeline.start(started_at=1_700_000_000.0)
    pipeline.observe(_request())
    worker = pipeline._worker
    assert worker is not None

    def interrupted_join(timeout: float | None = None) -> None:
        raise KeyboardInterrupt  # a second Ctrl+C while the close waits

    monkeypatch.setattr(worker, "join", interrupted_join)
    with pytest.raises(KeyboardInterrupt):
        pipeline.close(2.0)
    monkeypatch.undo()
    # Every step after the interrupted one ran: frozen, final file, lock freed.
    assert pipeline.registry.frozen
    text = (tmp_path / "stormlog-t.prom").read_text()
    assert check_exposition(text).value("stormlog_run_active") == 0
    assert not (tmp_path / "stormlog-t.lock").exists()
    pipeline.close(0.0)  # the run's fallback close
    pipeline.stop_serving()


def test_after_an_interrupt_the_later_steps_do_not_wait_and_none_runs_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    pipeline.start(started_at=1_700_000_000.0)
    deadlines: dict[str, list[float]] = {}

    def recording(step: str) -> None:
        real = getattr(pipeline, step)

        def run(until: float) -> None:
            calls = deadlines.setdefault(step, [])
            calls.append(until)
            if step == "_join_worker" and len(calls) == 1:
                raise KeyboardInterrupt  # a second Ctrl+C while the close waits
            real(until)

        monkeypatch.setattr(pipeline, step, run)

    for step in export_module._CLOSE_STEPS:
        recording(step)
    with pytest.raises(KeyboardInterrupt):
        pipeline.close(30.0)
    pipeline.close(30.0)  # the run's fallback close
    # Each step ran once across both closes, the interrupted one included,
    # and every step after the interrupt was told not to wait.
    assert {step: len(calls) for step, calls in deadlines.items()} == {
        step: 1 for step in export_module._CLOSE_STEPS
    }
    after = export_module._CLOSE_STEPS.index("_join_worker") + 1
    assert all(deadlines[step] == [0.0] for step in export_module._CLOSE_STEPS[after:])
    pipeline.stop_serving()


def test_the_worker_stops_applying_once_the_close_stops_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The worker is held in its first record until the close's wait has
    # ended. Released, it must not go on to the next record: the rest are
    # dropped at shutdown, and the freeze never queues behind them.
    pipeline = ExportPipeline(ExportConfig(prometheus_textfile_dir=tmp_path), LABELS)
    held, release = threading.Event(), threading.Event()
    calls: list[int] = []
    real_apply = pipeline._apply

    def counted(envelope: Any) -> bool:
        calls.append(1)
        if len(calls) == 1:
            held.set()
            release.wait(10)
        return real_apply(envelope)

    monkeypatch.setattr(pipeline, "_apply", counted)
    pipeline.start(started_at=0.0)
    worker = pipeline._worker
    assert worker is not None
    for _ in range(10):
        pipeline.observe(_request())
    assert held.wait(10)
    real_freeze = pipeline._freeze

    def freeze_once_the_worker_has_moved(until: float) -> None:
        release.set()
        # A worker that goes on reaches its next record well within this.
        _wait_for(lambda: not worker.is_alive() or len(calls) > 1)
        real_freeze(until)

    monkeypatch.setattr(pipeline, "_freeze", freeze_once_the_worker_has_moved)
    pipeline.close(0.3)
    assert len(calls) == 1
    records = pipeline.summary()["records"]
    assert records["applied"] == 1
    assert records["dropped"]["shutdown"] == 9
    pipeline.stop_serving()


def test_a_stale_final_file_is_said_so_in_the_capability_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    pipeline.start(started_at=1_700_000_000.0)
    monkeypatch.setattr(pipeline.renders, "is_fresh", lambda _generation: False)
    pipeline.close(5.0)
    assert pipeline.summary()["textfile"]["final_stale"] is True
    pipeline.stop_serving()


def test_a_first_close_with_no_time_left_still_writes_the_final_file(
    tmp_path: Path,
) -> None:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    pipeline.start(started_at=1_700_000_000.0)
    pipeline.close(0.0)
    text = (tmp_path / "stormlog-t.prom").read_text()
    assert check_exposition(text).value("stormlog_run_active") == 0
    assert not (tmp_path / "stormlog-t.lock").exists()
    assert pipeline.summary()["textfile"]["abandoned"] is False
    pipeline.stop_serving()


def test_the_final_file_has_the_frozen_values_when_a_render_was_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer's case, with no scrape: the textfile writer's periodic
    # render takes its values, the run's last records land, and the close
    # freezes, all before that render publishes.
    pipeline = ExportPipeline(
        ExportConfig(
            prometheus_textfile_dir=tmp_path,
            prometheus_slot="t",
            prometheus_textfile_interval_seconds=0.2,
        ),
        LABELS,
    )
    pipeline.start(started_at=1_700_000_000.0)
    for _ in range(3):
        pipeline.observe(_request())
    assert _wait_for(lambda: pipeline.summary()["records"]["applied"] == 3)
    entered, gate = threading.Event(), threading.Event()
    real_render = export_module.render

    def held_render(snapshot: Any) -> bytes:
        if not entered.is_set():
            entered.set()
            gate.wait(10)
        return real_render(snapshot)

    monkeypatch.setattr(export_module, "render", held_render)
    assert entered.wait(10)
    for _ in range(2):
        pipeline.observe(_request())
    assert _wait_for(lambda: pipeline.summary()["records"]["applied"] == 5)
    closer = threading.Thread(target=pipeline.close, args=(5.0,))
    closer.start()
    assert _wait_for(lambda: pipeline.registry.frozen)
    time.sleep(0.1)  # the close has invalidated the renders
    gate.set()
    closer.join(10)
    final = check_exposition((tmp_path / "stormlog-t.prom").read_text())
    assert sum(final.matching("stormlog_infer_requests_total")) == 5
    assert final.value("stormlog_run_active") == 0
    assert pipeline.summary()["textfile"]["final_stale"] is False
    pipeline.stop_serving()


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


def test_a_record_that_fails_to_apply_is_dropped_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    calls: list[int] = []
    real_apply = pipeline.metrics.apply

    def failing_second(envelope: Any) -> None:
        calls.append(1)
        real_apply(envelope)
        if len(calls) == 2:
            raise RuntimeError("a mapping bug")

    monkeypatch.setattr(pipeline.metrics, "apply", failing_second)
    pipeline.start(started_at=1_700_000_000.0)
    for _ in range(3):
        pipeline.observe(_request())
    pipeline.close(2.0)
    records = pipeline.summary()["records"]
    assert records["applied"] == 2
    assert records["dropped"]["error"] == 1 and records["dropped"]["shutdown"] == 0
    assert records["offered"] == records["applied"] + sum(records["dropped"].values())
    assert records["exact"] is False
    # The failed record was undone whole: two requests counted, not three.
    text = (tmp_path / "stormlog-t.prom").read_text()
    assert sum(check_exposition(text).matching("stormlog_infer_requests_total")) == 2
    pipeline.stop_serving()


# Every family a profile can export, with scraping, traces, the span receiver
# and both outputs on. A new family changes this list and the docs together.
FAMILIES = (
    "stormlog_engine_last_scrape_timestamp_seconds",
    "stormlog_engine_metrics_source_changes_total",
    "stormlog_engine_scrape_duration_seconds",
    "stormlog_engine_scrapes_total",
    "stormlog_engine_span_receiver_requests_total",
    "stormlog_engine_span_receiver_spans_total",
    "stormlog_exporter_internal_errors_total",
    "stormlog_health_snapshot_age_seconds",
    "stormlog_infer_abandoned_requests_total",
    "stormlog_infer_chunk_interarrival_seconds",
    "stormlog_infer_dispatch_lag_seconds",
    "stormlog_infer_e2e_from_intended_seconds",
    "stormlog_infer_phases_total",
    "stormlog_infer_request_duration_seconds",
    "stormlog_infer_requests_held_for_slot_total",
    "stormlog_infer_requests_total",
    "stormlog_infer_time_to_first_chunk_seconds",
    "stormlog_infer_time_to_first_token_seconds",
    "stormlog_infer_tokens_total",
    "stormlog_metrics_records_applied_total",
    "stormlog_metrics_records_dropped_total",
    "stormlog_metrics_scrapes_total",
    "stormlog_metrics_series_overflow_total",
    "stormlog_metrics_series_rejected_total",
    "stormlog_metrics_textfile_writes_total",
    "stormlog_run_info",
    "stormlog_run_start_time_seconds",
    "stormlog_trace_windows_total",
)


def test_the_families_are_a_fixed_list_and_each_is_documented(
    tmp_path: Path,
) -> None:
    labels = replace(
        LABELS, metrics_server="http://127.0.0.1:8000/metrics", traces=True
    )
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_listen="127.0.0.1:0"),
        labels,
        health=[(RECEIVER_HEALTH, ReceiverHealth.health_metrics())],
    )
    assert sorted(f.spec.name for f in pipeline.registry.families) == sorted(FAMILIES)
    docs = (Path(__file__).parents[1] / "docs" / "inference_export.md").read_text()
    assert [name for name in FAMILIES if f"`{name}" not in docs] == []


def test_a_record_the_exporter_could_not_read_makes_the_totals_inexact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never offered, so no drop counts it; only the internal error does.
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"), LABELS
    )
    real_envelope = pipeline.metrics.envelope
    calls: list[int] = []

    def failing_first(record: Any, extras: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("a mapping bug")
        return real_envelope(record, extras)

    monkeypatch.setattr(pipeline.metrics, "envelope", failing_first)
    pipeline.start(started_at=1_700_000_000.0)
    for _ in range(3):
        pipeline.observe(_request())
    pipeline.close(2.0)
    summary = pipeline.summary()
    assert summary["records"]["offered"] == 2 and summary["records"]["applied"] == 2
    assert summary["internal_errors"]["observe"] == 1
    assert summary["records"]["exact"] is False
    pipeline.stop_serving()


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


def test_the_close_reads_the_health_sources_once_more(tmp_path: Path) -> None:
    # The frozen values are the sources' values at the end, not at the
    # poller's last tick, up to a second earlier.
    source = _Source()
    pipeline = ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path, prometheus_slot="t"),
        LABELS,
        health=[("watch", source.health_metrics())],
    )
    pipeline.attach_health("watch", source)
    source.values = {"depth": 1}
    pipeline.start(started_at=1_700_000_000.0)
    assert _wait_for(
        lambda: _exposition(pipeline).matching("stormlog_watch_depth_bytes") == [1]
    )
    source.values = {"depth": 2}
    pipeline.close(2.0)
    text = (tmp_path / "stormlog-t.prom").read_text()
    assert check_exposition(text).value("stormlog_watch_depth_bytes") == 2
    pipeline.stop_serving()


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


@pytest.mark.parametrize(
    "argv",
    [
        ("--prometheus-slot", "default"),
        ("--prometheus-case-label", "on"),
        ("--prometheus-max-series", "50000"),
        ("--prometheus-linger", "0"),
        ("--prometheus-textfile-interval", "15"),
    ],
)
def test_an_export_flag_given_its_default_still_needs_a_destination(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match=argv[0]):
        export_config_from_args(_args(*argv))


def test_a_json_setting_given_its_default_still_needs_a_destination() -> None:
    with pytest.raises(ValueError, match="prometheus-slot"):
        ExportConfig.from_mapping({"prometheus_slot": "default"})
    with pytest.raises(ValueError, match="otlp-flush-timeout"):
        ExportConfig.from_mapping({"otlp_flush_timeout_seconds": 5.0})
    with pytest.raises(ValueError, match="otlp-sample-ratio"):
        ExportConfig.from_mapping({"sample_ratio": 1.0})


@pytest.mark.parametrize(
    "argv",
    [
        ("--otlp-flush-timeout", "5"),
        ("--otlp-probe-interval", "8"),
    ],
)
def test_an_otlp_flag_given_its_default_still_needs_a_destination(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match=argv[0]):
        export_config_from_args(_args(*argv))


@pytest.mark.parametrize("value", ["nan", "inf", "1e300", "3601"])
@pytest.mark.parametrize(
    "flag",
    [
        ("--prometheus-linger", "--prometheus-listen", "127.0.0.1:1"),
        ("--prometheus-textfile-interval", "--prometheus-textfile-dir", "."),
    ],
)
def test_a_time_must_be_finite_and_at_most_an_hour(
    flag: tuple[str, str, str], value: str
) -> None:
    # nan passes every comparison; inf and 1e300 overflow the waits that use
    # them, which killed the textfile writer or failed a finished run.
    name, destination, where = flag
    with pytest.raises(ValueError, match="finite"):
        export_config_from_args(_args(name, value, destination, where))


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


@pytest.mark.parametrize(
    "setting",
    [
        {"prometheus_case_label": "off"},  # a truthy string, not False
        {"prometheus_max_series": "100"},
        {"prometheus_max_series": True},
        {"prometheus_listen": 9999},
        {"prometheus_series_headroom": 2.5},
        {"prometheus_linger_seconds": "30"},
        {"prometheus_textfile_dir": 5},
        {"prometheus_textfile_remove_on_exit": 1},
    ],
)
def test_the_watch_json_section_checks_each_type(setting: dict[str, Any]) -> None:
    mapping = {"prometheus_listen": "127.0.0.1:9900", **setting}
    with pytest.raises(ValueError, match=next(iter(setting))):
        ExportConfig.from_mapping(mapping)


# Audit events that are I/O, from the thread under test while it records.
_IO_EVENTS = ("open", "os.", "socket.", "subprocess.", "shutil.", "time.sleep")
_audited: dict[str, Any] = {"thread": None, "events": []}


def _audit(event: str, _args: tuple[Any, ...]) -> None:
    if threading.get_ident() == _audited["thread"] and event.startswith(_IO_EVENTS):
        _audited["events"].append(event)


sys.addaudithook(_audit)  # audit hooks cannot be removed; this one is idle


def test_observe_takes_no_registry_lock_and_does_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    pipeline = ExportPipeline(
        ExportConfig(
            prometheus_textfile_dir=tmp_path, otlp_endpoint="http://127.0.0.1:9"
        ),
        LABELS,
        spans=SpanIdentity(
            run_id="run-1",
            session_id="session-1",
            model="m",
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
        ),
    )
    lock = _RecordingLock(pipeline.registry._lock)
    pipeline.registry._lock = lock  # type: ignore[assignment]
    assert pipeline.otlp is not None
    ledger = pipeline.otlp.exporter.ledger
    ledger_lock = _RecordingLock(ledger._lock)
    ledger._lock = ledger_lock  # type: ignore[assignment]
    unaudited: list[str] = []
    with monkeypatch.context() as patched:
        # Not audited by Python: a write to a descriptor, and a sleep.
        patched.setattr(os, "write", lambda *_a: unaudited.append("os.write"))
        patched.setattr(time, "sleep", lambda *_a: unaudited.append("time.sleep"))
        _audited.update(thread=threading.get_ident(), events=[])
        try:
            for index in range(10):
                record = {**_request(), "request_id": f"r{index}", "x_request_id": "x"}
                pipeline.observe(record, {"chunk_summary": ((0,) * 15, 0.0)})
        finally:
            _audited["thread"] = None
    assert _audited["events"] == [] and unaudited == []
    assert threading.get_ident() not in lock.takers
    assert threading.get_ident() not in ledger_lock.takers
    assert pipeline.queue.stats().accepted == 10
    assert pipeline.otlp.exporter.queue.stats().accepted == 10
    assert pipeline.summary()["internal_errors"]["observe"] == 0
    pipeline.close(1.0)


@pytest.mark.parametrize(
    ("policy", "ratio", "sampler", "says"),
    [
        (
            "preserve-engine",
            1.0,
            "parentbased_traceidratio:0.1",
            "marks every request sampled",
        ),
        (
            "follow-sampling",
            0.5,
            "parentbased_jaeger_remote",
            "marks 50% of requests sampled",
        ),
    ],
)
def test_the_volume_warning_names_the_share_it_marks(
    policy: str, ratio: float, sampler: str, says: str
) -> None:
    config = ExportConfig(
        otlp_file=Path("spans.jsonl"),
        trace_context=policy,
        sample_ratio=ratio,
        server_trace_sampler=sampler,
    )
    (warning,) = sampler_warnings(config)
    assert says in warning


def test_a_span_close_cut_short_is_finished_by_the_next_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Ctrl+C inside the span exporter's own close steps: the pipeline's
    # next close, the run's fallback, finishes them, rather than taking the
    # step as done and leaving the ledger open and the sink unclosed.
    pipeline = ExportPipeline(
        ExportConfig(otlp_file=tmp_path / "spans.jsonl"),
        LABELS,
        spans=SpanIdentity("run", "session", "m", "http://h:8000/v1"),
    )
    pipeline.start(started_at=0.0)
    otlp = pipeline.otlp
    assert otlp is not None
    real_drain = otlp.exporter.queue.drain
    presses = [KeyboardInterrupt()]

    def drain_interrupted_once() -> Any:
        if presses:
            raise presses.pop()
        return real_drain()

    monkeypatch.setattr(otlp.exporter.queue, "drain", drain_interrupted_once)
    with pytest.raises(KeyboardInterrupt):
        pipeline.close(2.0)
    finished = otlp.exporter.closed
    assert not finished
    pipeline.close(2.0)
    assert otlp.exporter.closed and otlp.accounting()["frozen"]
    pipeline.stop_serving()
