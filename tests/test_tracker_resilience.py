from __future__ import annotations

import threading
from collections import deque
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import pytest

import stormlog.tracker as tracker_mod
from stormlog.cli import _tracker_monitor_summary
from stormlog.collector_health import (
    COLLECTOR_HEALTH_DEGRADED,
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
)
from stormlog.device_collectors import (
    DeviceMemoryCapabilities,
    DeviceMemoryCollector,
    DeviceMemorySample,
    DeviceMemorySampleResult,
)
from stormlog.phases import parse_phase_boundary
from stormlog.telemetry_sink import TelemetrySinkConfig


def _sample(
    *,
    allocated: int,
    reserved: int,
    used: int | None = None,
    total: int | None = 4096,
    free: int | None = None,
    active: int | None = 512,
    inactive: int | None = 256,
) -> DeviceMemorySample:
    resolved_used = max(allocated, reserved) if used is None else used
    resolved_free = (
        total - resolved_used if free is None and total is not None else free
    )
    return DeviceMemorySample(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        used_bytes=resolved_used,
        free_bytes=resolved_free,
        total_bytes=total,
        active_bytes=active,
        inactive_bytes=inactive,
        device_id=0,
    )


class _SequencedCollector:
    def __init__(self, results: list[DeviceMemorySampleResult]) -> None:
        self._results = deque(results)
        self._last = results[-1]

    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> dict[str, object]:
        return {
            "backend": "cuda",
            "supports_device_total": True,
            "supports_device_free": True,
            "sampling_source": "test.collector",
            "telemetry_collector": "stormlog.cuda_tracker",
        }

    def sample(self) -> DeviceMemorySample:
        result = self.sample_with_diagnostics()
        if result.sample is None:
            raise RuntimeError(result.core_error or "collector unavailable")
        return result.sample

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        if self._results:
            self._last = self._results.popleft()
        return self._last


class _DeviceOnlyCollector(DeviceMemoryCollector):
    def __init__(self, sample: DeviceMemorySample) -> None:
        self._sample = sample

    def name(self) -> str:
        return "future"

    def is_available(self) -> bool:
        return True

    def sample(self) -> DeviceMemorySample:
        return self._sample

    def capabilities(self) -> DeviceMemoryCapabilities:
        return DeviceMemoryCapabilities(
            backend="future",
            telemetry_collector="stormlog.future_tracker",
            sampling_source="future.device_memory",
            supports_device_used=True,
            supports_device_free=True,
            supports_device_total=True,
        )


class _NoOpThread:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args
        self.daemon = bool(kwargs.get("daemon", False))

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        _ = timeout


class _JoinRecordingThread:
    def __init__(self, *, stops_within_timeout: bool = True) -> None:
        self.join_timeout: float | None = -1.0
        self.alive = True
        self.stops_within_timeout = stops_within_timeout

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout
        if self.stops_within_timeout:
            self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class _SequencedStopEvent:
    def __init__(self, waits: list[bool]) -> None:
        self._waits = deque(waits)

    def wait(self, timeout: float | None = None) -> bool:
        _ = timeout
        if self._waits:
            return self._waits.popleft()
        return True

    def set(self) -> None:
        return None

    def clear(self) -> None:
        return None


class _PausingDeque(deque):
    def __init__(
        self,
        values: list[tracker_mod.TrackingEvent],
        iter_started: threading.Event,
        writer_finished: threading.Event,
    ) -> None:
        super().__init__(values)
        self.iter_started = iter_started
        self.writer_finished = writer_finished

    def __iter__(self) -> Iterator[tracker_mod.TrackingEvent]:
        for index, item in enumerate(super().__iter__()):
            if index == 0:
                self.iter_started.set()
                self.writer_finished.wait(timeout=0.05)
            yield item


def _assert_reader_snapshots_events_under_lock(
    tracker: tracker_mod.MemoryTracker,
    reader: Callable[[], object],
) -> None:
    iter_started = threading.Event()
    writer_finished = threading.Event()
    writer_errors: list[Exception] = []
    tracker.events = _PausingDeque(list(tracker.events), iter_started, writer_finished)

    def writer() -> None:
        try:
            if not iter_started.wait(timeout=1.0):
                writer_errors.append(AssertionError("reader did not iterate events"))
                return
            tracker._add_event("sample", 0, "concurrent writer")
        except Exception as exc:
            writer_errors.append(exc)
        finally:
            writer_finished.set()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    try:
        reader()
    finally:
        writer_finished.set()
        writer_thread.join(timeout=1.0)

    assert not writer_thread.is_alive(), "writer thread timed out"
    assert writer_errors == []


class _FailingFlushSink:
    def __init__(self) -> None:
        self.flush_calls = 0
        self.close_calls = 0

    def append(self, record: dict[str, object]) -> None:
        _ = record
        return None

    def flush(self, *, force: bool = False) -> None:
        _ = force
        self.flush_calls += 1
        raise OSError("disk full")

    def close(self) -> None:
        self.close_calls += 1


class _FailingAppendSink:
    def __init__(self) -> None:
        self.append_calls = 0
        self.close_calls = 0

    def append(self, record: dict[str, object]) -> None:
        _ = record
        self.append_calls += 1
        raise OSError("disk full")

    def flush(self, *, force: bool = False) -> None:
        _ = force
        return None

    def close(self) -> None:
        self.close_calls += 1


def _build_tracker(
    monkeypatch: pytest.MonkeyPatch,
    collector: _SequencedCollector,
    **kwargs: Any,
) -> tracker_mod.MemoryTracker:
    monkeypatch.setattr(
        tracker_mod.MemoryTracker,
        "_setup_device",
        lambda self, _device: tracker_mod.torch.device("cuda:0"),
    )
    monkeypatch.setattr(
        tracker_mod,
        "build_device_memory_collector",
        lambda _device: collector,
    )
    monkeypatch.setattr(
        tracker_mod,
        "get_gpu_info",
        lambda _device: {"total_memory": 4096},
    )
    return tracker_mod.MemoryTracker(
        sampling_interval=0.01,
        enable_alerts=False,
        **kwargs,
    )


def _device_only_sample(*, used: int = 3500) -> DeviceMemorySample:
    return DeviceMemorySample(
        allocated_bytes=None,
        reserved_bytes=None,
        used_bytes=used,
        free_bytes=4000 - used,
        total_bytes=4000,
        active_bytes=None,
        inactive_bytes=None,
        device_id=0,
    )


def test_memory_tracker_accepts_injected_device_only_collector() -> None:
    tracker = tracker_mod.MemoryTracker(
        collector=_DeviceOnlyCollector(_device_only_sample()),
        sampling_interval=0.01,
    )

    tracker._run_tracking_iteration(0)

    event_types = [event.event_type for event in tracker.get_events()]
    assert "sample" in event_types
    assert "warning" in event_types
    assert "allocation" not in event_types
    assert "deallocation" not in event_types
    assert "peak" not in event_types
    stats = tracker.get_statistics()
    assert stats["current_memory_allocated"] is None
    assert stats["current_memory_reserved"] is None
    assert stats["current_device_used"] == 3500
    assert stats["peak_device_used"] == 3500
    assert stats["peak_memory"] is None
    assert stats["total_allocations"] is None
    assert stats["allocations_per_second"] is None
    assert stats["memory_utilization_percent"] == 87.5
    sample_event = next(
        event for event in tracker.get_events() if event.event_type == "sample"
    )
    record = tracker._telemetry_record_from_event(sample_event)
    assert record["allocator_allocated_bytes"] is None
    assert record["allocator_reserved_bytes"] is None
    assert record["device_used_bytes"] == 3500
    assert (
        record["metadata"]["memory_capabilities"]["supports_fragmentation_analysis"]
        is False
    )


@pytest.mark.parametrize("device_only", [False, True])
def test_monitor_summary_handles_nullable_tracker_lifecycle(
    monkeypatch: pytest.MonkeyPatch, device_only: bool
) -> None:
    monkeypatch.setattr(tracker_mod.threading, "Thread", _NoOpThread)
    if device_only:
        tracker = tracker_mod.MemoryTracker(
            collector=_DeviceOnlyCollector(_device_only_sample()), enable_alerts=False
        )
    else:
        collector = _SequencedCollector(
            [
                DeviceMemorySampleResult(sample=_sample(allocated=64, reserved=256)),
                DeviceMemorySampleResult(sample=_sample(allocated=100, reserved=256)),
                DeviceMemorySampleResult(sample=_sample(allocated=150, reserved=256)),
            ]
        )
        tracker = _build_tracker(monkeypatch, collector)
    tracker.start_tracking()
    try:
        last_allocated = tracker._run_tracking_iteration(0)
        tracker._run_tracking_iteration(last_allocated)
    finally:
        tracker.stop_tracking()

    assert tracker.get_events()[0].memory_allocated is None
    assert _tracker_monitor_summary(tracker)["memory_change_from_baseline"] == (
        None if device_only else 50
    )


def test_memory_tracker_rejects_device_and_collector_together() -> None:
    with pytest.raises(ValueError, match="cannot be provided together"):
        tracker_mod.MemoryTracker(
            device="cuda:0",
            collector=_DeviceOnlyCollector(_device_only_sample()),
        )


@pytest.mark.parametrize("initial_state", ["failure", "partial", "known"])
def test_device_only_tracker_refreshes_capacity_after_initial_sample(
    monkeypatch: pytest.MonkeyPatch, initial_state: str
) -> None:
    recovered = _device_only_sample(used=3960)
    initial = {
        "failure": DeviceMemorySampleResult(
            sample=None, core_error="initial collection failed"
        ),
        "partial": DeviceMemorySampleResult(
            sample=replace(recovered, total_bytes=None, free_bytes=None),
            partial_fields=("device_total_bytes", "device_free_bytes"),
        ),
        "known": DeviceMemorySampleResult(
            sample=replace(recovered, total_bytes=8000, free_bytes=4040)
        ),
    }[initial_state]
    results = iter([initial, DeviceMemorySampleResult(sample=recovered)])
    collector = _DeviceOnlyCollector(recovered)
    monkeypatch.setattr(collector, "sample_with_diagnostics", lambda: next(results))
    tracker = tracker_mod.MemoryTracker(collector=collector)

    tracker._run_tracking_iteration(0)

    stats = tracker.get_statistics()
    assert tracker.total_memory == 4000
    assert stats["device_total"] == 4000
    assert stats["memory_utilization_percent"] == 99.0
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_HEALTHY
    alerts = tracker.get_alerts()
    assert any(
        event.event_type == "critical" and "99.0%" in (event.context or "")
        for event in alerts
    )


def test_device_only_tracker_keeps_capacity_when_sample_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _device_only_sample()
    invalid = replace(sample, allocated_bytes=1, total_bytes=8000, free_bytes=4500)
    results = iter(
        [
            DeviceMemorySampleResult(sample=sample),
            DeviceMemorySampleResult(sample=invalid),
        ]
    )
    collector = _DeviceOnlyCollector(sample)
    monkeypatch.setattr(collector, "sample_with_diagnostics", lambda: next(results))
    tracker = tracker_mod.MemoryTracker(collector=collector)

    tracker._run_tracking_iteration(0)

    assert tracker.total_memory == 4000
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_UNHEALTHY
    )


def test_memory_tracker_degrades_on_capability_sample_mismatch() -> None:
    invalid_sample = DeviceMemorySample(
        allocated_bytes=1,
        reserved_bytes=None,
        used_bytes=2,
        free_bytes=3998,
        total_bytes=4000,
        active_bytes=None,
        inactive_bytes=None,
        device_id=0,
    )
    tracker = tracker_mod.MemoryTracker(
        collector=_DeviceOnlyCollector(invalid_sample),
        sampling_interval=0.01,
    )

    tracker._run_tracking_iteration(0)

    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_UNHEALTHY
    )
    assert all(event.event_type != "sample" for event in tracker.get_events())


def test_memory_tracker_recovers_after_transient_collector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256)),
            DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": "collector unavailable"},
                core_error="collector unavailable",
            ),
            DeviceMemorySampleResult(sample=_sample(allocated=256, reserved=256)),
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    current_time = {"value": 10.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    last_allocated = tracker._run_tracking_iteration(0)
    assert last_allocated == 0
    assert tracker.get_events()[-1].event_type == "collector_degraded"
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_UNHEALTHY
    )
    assert tracker.get_statistics()["collector_next_retry_epoch_s"] == pytest.approx(
        11.0
    )

    current_time["value"] = 10.5
    skipped_allocated = tracker._run_tracking_iteration(last_allocated)
    assert skipped_allocated == last_allocated
    assert [event.event_type for event in tracker.get_events()].count(
        "collector_degraded"
    ) == 1

    current_time["value"] = 11.1
    recovered_allocated = tracker._run_tracking_iteration(last_allocated)
    assert recovered_allocated == 256
    event_types = [event.event_type for event in tracker.get_events()]
    assert event_types.count("collector_degraded") == 1
    assert "collector_recovered" in event_types
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_HEALTHY
    )
    assert tracker.get_statistics()["telemetry_partial"] is False


def test_memory_tracker_keeps_retrying_during_persistent_collector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = DeviceMemorySampleResult(
        sample=None,
        errors={"core_metrics": "collector unavailable"},
        core_error="collector unavailable",
    )
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=64, reserved=64)),
            failure,
            failure,
            failure,
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    tracker._collector_retry_backoff_initial_s = 0.1
    tracker._collector_retry_backoff_cap_s = 0.4
    current_time = {"value": 20.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    last_allocated = tracker._run_tracking_iteration(64)
    assert last_allocated == 64

    current_time["value"] = 20.11
    tracker._run_tracking_iteration(last_allocated)
    current_time["value"] = 20.32
    tracker._run_tracking_iteration(last_allocated)

    stats = tracker.get_statistics()
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_UNHEALTHY
    assert stats["telemetry_partial"] is True
    assert stats["collector_consecutive_failures"] == 3
    assert stats["collector_next_retry_epoch_s"] == pytest.approx(20.72)
    assert [event.event_type for event in tracker.get_events()] == [
        "collector_degraded"
    ]


def test_memory_tracker_emits_partial_sample_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = DeviceMemorySampleResult(
        sample=_sample(
            allocated=128,
            reserved=128,
            total=None,
            free=None,
            active=None,
            inactive=None,
        ),
        partial_fields=(
            "device_total_bytes",
            "device_free_bytes",
            "allocator_active_bytes",
            "allocator_inactive_bytes",
        ),
        errors={
            "device_total_bytes": "total unavailable",
            "allocator_active_bytes": "stats unavailable",
        },
    )
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=128)),
            partial,
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    tracker.stats["peak_memory"] = 128
    current_time = {"value": 30.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    last_allocated = tracker._run_tracking_iteration(128)

    assert last_allocated == 128
    events = tracker.get_events()
    assert [event.event_type for event in events] == ["collector_degraded", "sample"]
    assert events[-1].metadata is not None
    assert events[-1].metadata["collector_health_status"] == COLLECTOR_HEALTH_DEGRADED
    assert events[-1].metadata["telemetry_partial"] is True
    assert events[-1].metadata["collector_partial_fields"] == [
        "device_total_bytes",
        "device_free_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
    ]
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_DEGRADED
    )


def test_memory_tracker_export_preserves_health_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    partial = DeviceMemorySampleResult(
        sample=_sample(
            allocated=256,
            reserved=256,
            total=None,
            free=None,
            active=None,
            inactive=None,
        ),
        partial_fields=("device_total_bytes", "device_free_bytes"),
        errors={"device_total_bytes": "total unavailable"},
    )
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=256, reserved=256)),
            partial,
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    tracker.stats["peak_memory"] = 256
    current_time = {"value": 40.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    tracker._run_tracking_iteration(256)
    output_path = tmp_path / "tracker.json"
    tracker.export_events(str(output_path), format="json")
    payload = output_path.read_text(encoding="utf-8")

    assert "collector_health_status" in payload
    assert "collector_degraded" in payload
    assert "device_total_bytes" in payload


def test_memory_tracker_emits_sample_event_on_healthy_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)

    last_allocated = tracker._run_tracking_iteration(0)

    assert last_allocated == 128
    event_types = [event.event_type for event in tracker.get_events()]
    assert event_types == ["peak", "allocation", "sample"]
    assert tracker.get_events()[-1].context == "Collected telemetry sample."


def test_memory_tracker_streams_events_to_append_only_sink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(
        monkeypatch,
        collector,
        telemetry_sink_config=TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_bytes=1024 * 1024,
            retention_max_total_bytes=1024 * 1024,
        ),
    )

    tracker._run_tracking_iteration(0)
    tracker._close_telemetry_sink()

    segment = tmp_path / "sink" / "segment-000001.jsonl"
    lines = [line for line in segment.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    payload = tracker_mod.json.loads(lines[-1])
    assert payload["collector"] == "stormlog.cuda_tracker"
    assert payload["event_type"] == "sample"
    assert payload["allocator_allocated_bytes"] == 128
    stats = tracker.get_statistics()
    assert stats["final_retained_files"] == 1
    assert stats["rollover_count"] == 0


def test_memory_tracker_records_bounded_history_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector, max_events=3)

    for index in range(5):
        tracker._add_event("allocation", index, f"event-{index}")

    events = tracker.get_events()
    assert [event.context for event in events] == ["event-2", "event-3", "event-4"]
    stats = tracker.get_statistics()
    assert stats["history_window_limit_events"] == 3
    assert stats["history_retained_events"] == 3
    assert stats["history_dropped_events"] == 2


def test_memory_tracker_read_apis_snapshot_events_under_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_names = (
        "get_events",
        "get_memory_timeline",
        "get_statistics",
        "get_alerts",
    )

    for reader_name in reader_names:
        collector = _SequencedCollector(
            [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
        )
        tracker = _build_tracker(monkeypatch, collector, max_events=100)
        tracker._add_event("warning", 0, "seed-warning")
        tracker._add_event("sample", 0, "seed-sample")

        reader: Callable[[], object]
        if reader_name == "get_events":
            reader = partial(tracker.get_events, last_n=1)
        elif reader_name == "get_memory_timeline":
            reader = partial(tracker.get_memory_timeline, interval=1.0)
        elif reader_name == "get_statistics":
            reader = tracker.get_statistics
        else:
            reader = partial(tracker.get_alerts, last_n=1)

        _assert_reader_snapshots_events_under_lock(tracker, reader)


def test_stop_tracking_waits_for_worker_before_recording_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    worker = _JoinRecordingThread()
    tracker.is_tracking = True
    tracker._tracking_thread = cast(Any, worker)

    tracker.stop_tracking()

    assert worker.join_timeout == 5.0
    assert worker.is_alive() is False
    assert tracker.get_events()[-1].event_type == "stop"


def test_stop_tracking_does_not_finalize_while_worker_is_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    worker = _JoinRecordingThread(stops_within_timeout=False)
    tracker.is_tracking = True
    tracker._tracking_thread = cast(Any, worker)

    with pytest.raises(TimeoutError, match="did not stop within 5.0 seconds"):
        tracker.stop_tracking()

    assert worker.join_timeout == 5.0
    assert tracker.is_tracking is True
    assert all(event.event_type != "stop" for event in tracker.get_events())


def test_memory_tracker_disables_failing_sink_and_keeps_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    sink = _FailingFlushSink()
    tracker._telemetry_sink = cast(Any, sink)
    tracker._stop_event = cast(Any, _SequencedStopEvent([False, False, True]))
    iteration_inputs: list[int] = []

    def _run_iteration(last_allocated: int) -> int:
        iteration_inputs.append(last_allocated)
        return last_allocated + 1

    monkeypatch.setattr(tracker, "_run_tracking_iteration", _run_iteration)
    monkeypatch.setattr(tracker_mod.time, "sleep", lambda _: None)

    tracker._tracking_loop()

    assert iteration_inputs == [0, 1]
    assert sink.flush_calls == 1
    assert sink.close_calls == 1
    assert tracker._telemetry_sink is None


def test_memory_tracker_preserves_incomplete_session_after_sink_failure_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    sink = _FailingAppendSink()
    tracker._telemetry_sink = cast(Any, sink)

    tracker._add_event("allocation", 10, "sink_failure")

    assert sink.append_calls == 1
    assert sink.close_calls == 1
    assert tracker._telemetry_sink is None

    tracker.is_tracking = True
    tracker.stop_tracking()

    assert tracker.get_events()[-1].event_type == "stop"
    summary = tracker.get_session_summary()
    assert summary is not None
    assert summary.status == "incomplete"
    assert tracker.get_statistics()["session_status"] == "incomplete"


def test_memory_tracker_start_tracking_resets_collector_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    monkeypatch.setattr(tracker_mod.threading, "Thread", _NoOpThread)
    tracker._set_collector_health(
        status=COLLECTOR_HEALTH_UNHEALTHY,
        telemetry_partial=True,
        last_error="collector unavailable",
        consecutive_failures=3,
        next_retry_epoch_s=42.0,
    )
    tracker._last_observed_sample = _sample(allocated=512, reserved=768)
    tracker.stats["last_memory_check"] = 99.0

    tracker.start_tracking()

    stats = tracker.get_statistics()
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_HEALTHY
    assert stats["collector_last_error"] is None
    assert stats["collector_consecutive_failures"] == 0
    assert stats["collector_next_retry_epoch_s"] is None
    assert stats["current_memory_allocated"] is None
    assert tracker.stats["last_memory_check"] == 0
    assert tracker.get_events()[-1].event_type == "start"
    assert tracker.get_events()[-1].memory_allocated is None


def test_memory_tracker_hides_stale_current_stats_when_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=256, reserved=512)),
            DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": "collector unavailable"},
                core_error="collector unavailable",
            ),
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    current_time = {"value": 10.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    tracker._run_tracking_iteration(0)
    stats = tracker.get_statistics()

    assert tracker._last_observed_sample is not None
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_UNHEALTHY
    assert stats["current_memory_allocated"] is None
    assert stats["current_memory_reserved"] is None
    assert stats["memory_utilization_percent"] is None


def test_memory_tracker_recreates_sink_on_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(
        monkeypatch,
        collector,
        telemetry_sink_config=TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_bytes=1024 * 1024,
            retention_max_total_bytes=1024 * 1024,
        ),
    )
    monkeypatch.setattr(tracker_mod.threading, "Thread", _NoOpThread)

    tracker.start_tracking()
    first_sink = tracker._telemetry_sink
    assert first_sink is not None
    tracker.stop_tracking()
    assert tracker._telemetry_sink is None

    tracker.start_tracking()
    second_sink = cast(Any, tracker._telemetry_sink)

    assert second_sink is not None
    assert second_sink is not first_sink


def test_memory_tracker_emits_structured_phase_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    monkeypatch.setattr(tracker_mod.threading, "Thread", _NoOpThread)

    tracker.start_tracking()
    with tracker.phase("forward", metadata={"microbatch": 4}) as handle:
        assert handle.phase_path == "forward"
    tracker.stop_tracking()

    phase_events = [
        event for event in tracker.get_events() if event.event_type.startswith("phase_")
    ]
    assert [event.event_type for event in phase_events] == [
        "phase_enter",
        "phase_exit",
    ]

    enter_scope = parse_phase_boundary(
        tracker._telemetry_record_from_event(phase_events[0])
    )
    exit_scope = parse_phase_boundary(
        tracker._telemetry_record_from_event(phase_events[1])
    )
    assert enter_scope is not None
    assert exit_scope is not None
    assert enter_scope.path == ("forward",)
    assert enter_scope.attributes == {"microbatch": 4}
    assert exit_scope.scope_id == enter_scope.scope_id
    assert exit_scope.path == enter_scope.path
