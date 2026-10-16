import threading
import time

import pytest

from stormlog.mlx import MemoryTracker
from stormlog.telemetry import load_telemetry_sessions
from stormlog.telemetry_sink import TelemetrySinkConfig
from tests.mlx_fakes import FakeCore, make_runtime


def test_lifecycle_retention_and_sink_rotation(tmp_path):
    core = FakeCore()
    tracker = MemoryTracker(
        runtime=make_runtime(core),
        sampling_interval=10,
        max_history=3,
        telemetry_sink_config=TelemetrySinkConfig(
            tmp_path,
            flush_every_events=1,
            rollover_max_events=3,
            retention_max_files=10,
            write_rollups=False,
        ),
    )
    tracker.start_tracking()
    session = tracker.session_summary.session_id
    thread = tracker._sampler._thread
    tracker.start_tracking()
    assert tracker._sampler._thread is thread
    for value in range(100, 1000, 100):
        core.values["active_memory"] = value
        tracker._sample()
    result = tracker.stop_tracking()
    assert not thread.is_alive()
    assert result.sampled_peak_bytes == 900
    assert result.total_samples == 11 and len(result.samples) == 3
    assert result.history_dropped_samples == 8
    assert result.average_active_bytes == (100 + sum(range(100, 1000, 100)) + 900) / 11
    assert result.session_summary.status == "completed"
    loaded = load_telemetry_sessions(tmp_path)
    assert len(loaded) == 1 and len(loaded[0].events) == 11
    tracker.start_tracking()
    assert tracker.session_summary.session_id != session
    restarted = tracker.stop_tracking()
    assert restarted.total_samples == 2
    assert len(load_telemetry_sessions(tmp_path)) == 2
    assert "eval" not in core.calls and "reset" not in core.calls
    assert not any(isinstance(c, tuple) and c[0] == "sync" for c in core.calls)


def test_gap_health_backoff_recovery_and_zero():
    core = FakeCore()
    tracker = MemoryTracker(runtime=make_runtime(core), sampling_interval=10)
    tracker.start_tracking()
    core.values["active_memory"] = RuntimeError("collector failed")
    assert tracker._sample() == 10
    assert tracker._sample() == 20
    assert tracker._previous is None
    assert tracker.get_results().health["collector_consecutive_failures"] == 2
    core.values["active_memory"] = 0
    tracker._sample()
    events = tracker.get_results().telemetry_events
    samples = [r for r in events if r["event_type"] == "sample"]
    assert samples[-1]["allocator_allocated_bytes"] == 0
    assert samples[-1]["allocator_change_bytes"] is None
    assert "collector_recovered" in {r["event_type"] for r in events}
    assert tracker.get_results().health["collector_health_status"] == "healthy"
    assert tracker.stop_tracking().session_summary.status == "completed"


def test_interruptible_stop_and_phase_context():
    tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=100)
    tracker.start_tracking()
    with tracker.phase("train", {"epoch": 1}):
        with tracker.phase("step"):
            pass
    start = time.monotonic()
    result = tracker.stop_tracking(status="interrupted")
    assert time.monotonic() - start < 1
    assert result.session_summary.status == "interrupted"
    assert [e["event_type"] for e in result.telemetry_events] == [
        "start",
        "phase_enter",
        "phase_enter",
        "phase_exit",
        "phase_exit",
        "stop",
    ]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_interval(value):
    with pytest.raises(ValueError):
        MemoryTracker(runtime=make_runtime(), sampling_interval=value)


def test_sink_failure_evidence_and_shutdown(monkeypatch, tmp_path):
    tracker = MemoryTracker(
        runtime=make_runtime(),
        sampling_interval=10,
        telemetry_sink_config=TelemetrySinkConfig(tmp_path, write_rollups=False),
    )
    tracker.start_tracking()
    monkeypatch.setattr(
        tracker._sink, "append", lambda _: (_ for _ in ()).throw(OSError("disk full"))
    )
    tracker._sample()
    result = tracker.stop_tracking()
    assert "disk full" in result.sink_diagnostics["last_error"]
    assert result.session_summary.status == "incomplete"
    assert not tracker.is_tracking


def test_context_original_exception_and_no_thread_leak():
    tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=0.001)
    error = ValueError("workload")
    with pytest.raises(ValueError) as found:
        with tracker.tracking():
            raise error
    assert found.value is error
    assert not tracker.is_tracking
    assert tracker.session_summary.status == "incomplete"


def test_no_sample_drift_or_sampler_thread_on_stop():
    tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=0.002)
    tracker.start_tracking()
    time.sleep(0.02)
    tracker.stop_tracking()
    assert tracker.get_results().total_samples >= 3
    assert not any(
        t.name == "stormlog-mlx-sampler" and t.is_alive() for t in threading.enumerate()
    )


def test_phase_cannot_reopen_a_closed_sink(tmp_path):
    tracker = MemoryTracker(
        runtime=make_runtime(),
        sampling_interval=10,
        telemetry_sink_config=TelemetrySinkConfig(tmp_path, write_rollups=False),
    )
    tracker.start_tracking()
    phase = tracker.phase("unclosed")
    result = tracker.stop_tracking()
    with pytest.raises(RuntimeError, match="tracking stops"):
        phase.close()
    assert tracker.get_results().total_events == result.total_events
    assert tracker._sink.current_session() is None


def test_workload_exception_preserved_when_final_sampling_fails(monkeypatch):
    tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=10)
    original = KeyboardInterrupt()
    capture = tracker.collector.capture_snapshot

    def final_failure(name="sample"):
        if name == "stop":
            raise RuntimeError("finalization failure")
        return capture(name)

    monkeypatch.setattr(tracker.collector, "capture_snapshot", final_failure)
    with pytest.raises(KeyboardInterrupt) as found:
        with tracker.tracking():
            raise original
    assert found.value is original
    assert not tracker.is_tracking
    assert tracker.session_summary.status == "interrupted"


def test_failed_sink_start_does_not_leave_running_session(monkeypatch, tmp_path):
    from stormlog.telemetry_sink import AppendOnlyTelemetrySink

    monkeypatch.setattr(
        AppendOnlyTelemetrySink,
        "start_session",
        lambda *a: (_ for _ in ()).throw(OSError("start failed")),
    )
    tracker = MemoryTracker(
        runtime=make_runtime(),
        sampling_interval=10,
        telemetry_sink_config=TelemetrySinkConfig(tmp_path, write_rollups=False),
    )
    with pytest.raises(OSError, match="start failed"):
        tracker.start_tracking()
    assert not tracker.is_tracking
    assert tracker.session_summary.status == "incomplete"


def test_alerts_are_bounded_without_allocator_mutations():
    core = FakeCore()
    tracker = MemoryTracker(
        runtime=make_runtime(core),
        sampling_interval=10,
        max_history=2,
        alert_threshold_mb=0.00001,
    )
    tracker.start_tracking()
    for _ in range(5):
        tracker._sample()
    result = tracker.stop_tracking()
    assert len(result.alerts) == 2 and result.total_alerts == 7
    assert result.history_dropped_alerts == 5
    assert all(alert["action"] == "alert_only" for alert in result.alerts)
    assert "reset" not in core.calls and "eval" not in core.calls


def test_identity_from_environment_without_distributed_init(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=10, job_id="job")
    tracker.start_tracking()
    result = tracker.stop_tracking()
    session = result.session_summary
    assert (session.rank, session.local_rank, session.world_size, session.job_id) == (
        2,
        1,
        4,
        "job",
    )
    assert result.telemetry_events[0]["rank"] == 2


def test_sink_disk_failure_with_bounded_buffer(monkeypatch, tmp_path):
    tracker = MemoryTracker(
        runtime=make_runtime(),
        sampling_interval=10,
        telemetry_sink_config=TelemetrySinkConfig(
            tmp_path, write_rollups=False, max_buffer_bytes=8000, flush_every_events=1
        ),
    )
    tracker.start_tracking()

    def fail_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(tracker._sink, "_write_payload_locked", fail_write)
    for _ in range(12):
        tracker._sample()
    result = tracker.stop_tracking()
    assert result.sink_diagnostics["flush_failures"] >= 1
    assert result.sink_diagnostics["dropped_records"] > 0
    assert result.sink_diagnostics["buffered_bytes"] == 0
    assert result.session_summary.status == "incomplete"
    assert not tracker.is_tracking


def test_sampler_failure_is_visible_and_restartable(monkeypatch):
    tracker = MemoryTracker(runtime=make_runtime(), sampling_interval=0.001)
    tracker.start_tracking()
    original = tracker._sampler.callback

    def broken():
        raise RuntimeError("sampler callback failed")

    tracker._sampler.callback = broken
    deadline = time.monotonic() + 1
    while tracker.is_tracking and time.monotonic() < deadline:
        time.sleep(0.001)
    result = tracker.stop_tracking()
    assert result.session_summary.status == "incomplete"
    assert "sampler callback failed" in result.health["sampler_error"]
    tracker._sampler.callback = original
    tracker.start_tracking()
    assert tracker.stop_tracking().session_summary.status == "completed"
