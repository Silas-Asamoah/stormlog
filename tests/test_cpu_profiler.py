"""Comprehensive tests for CPU memory profiler and tracker."""

from __future__ import annotations

import contextlib
import csv
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, cast
from unittest.mock import patch

import pytest

from stormlog.cpu_profiler import (
    CPUMemoryProfiler,
    CPUMemorySnapshot,
    CPUMemoryTracker,
    CPUProfileResult,
)
from stormlog.phases import parse_phase_boundary
from stormlog.telemetry_sink import TelemetrySinkConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_process(
    rss: int = 1024 * 1024, vms: int = 2048 * 1024, cpu_pct: float = 5.0
) -> object:
    """Return a mock ``psutil.Process`` that reports fixed memory values."""

    class _MockProcess:
        def oneshot(self) -> object:
            return contextlib.nullcontext()

        def memory_info(self) -> object:
            return SimpleNamespace(rss=rss, vms=vms)

        def cpu_percent(self, interval: object = None) -> float:  # noqa: ARG002
            return cpu_pct

    return _MockProcess()


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.0,
    interval: float = 0.01,
) -> bool:
    """Poll ``predicate`` until true or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _FailingSink:
    def __init__(self, *, fail_on: set[str]) -> None:
        self.fail_on = fail_on
        self.append_calls = 0
        self.flush_calls = 0
        self.close_calls = 0

    def append(self, record: dict[str, object]) -> None:
        _ = record
        self.append_calls += 1
        if "append" in self.fail_on:
            raise OSError("disk full")

    def flush(self, *, force: bool = False) -> None:
        _ = force
        self.flush_calls += 1
        if "flush" in self.fail_on:
            raise OSError("disk full")

    def close(self) -> None:
        self.close_calls += 1
        if "close" in self.fail_on:
            raise OSError("disk full")


class _SequencedStopEvent:
    def __init__(self, waits: list[bool]) -> None:
        self._waits = list(waits)

    def wait(self, timeout: float | None = None) -> bool:
        _ = timeout
        if self._waits:
            return self._waits.pop(0)
        return True

    def set(self) -> None:
        return None

    def clear(self) -> None:
        return None


# ---------------------------------------------------------------------------
# CPUMemorySnapshot
# ---------------------------------------------------------------------------


class TestCPUMemorySnapshot:
    def test_creation(self) -> None:
        snap = CPUMemorySnapshot(timestamp=1.0, rss=100, vms=200, cpu_percent=1.5)
        assert snap.rss == 100
        assert snap.vms == 200
        assert snap.cpu_percent == 1.5

    def test_to_dict(self) -> None:
        snap = CPUMemorySnapshot(timestamp=1.0, rss=100, vms=200, cpu_percent=1.5)
        d = snap.to_dict()
        assert d == {"timestamp": 1.0, "rss": 100, "vms": 200, "cpu_percent": 1.5}


# ---------------------------------------------------------------------------
# CPUProfileResult
# ---------------------------------------------------------------------------


class TestCPUProfileResult:
    def _make_result(self) -> CPUProfileResult:
        before = CPUMemorySnapshot(timestamp=1.0, rss=100, vms=200, cpu_percent=1.0)
        after = CPUMemorySnapshot(timestamp=2.0, rss=300, vms=400, cpu_percent=2.0)
        return CPUProfileResult(
            name="test_fn",
            duration=1.0,
            snapshot_before=before,
            snapshot_after=after,
            peak_rss=350,
        )

    def test_memory_diff(self) -> None:
        result = self._make_result()
        assert result.memory_diff() == 200  # 300 - 100

    def test_to_dict(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert d["name"] == "test_fn"
        assert d["duration"] == 1.0
        assert d["memory_diff"] == 200
        assert d["peak_rss"] == 350
        assert "before" in d
        assert "after" in d


# ---------------------------------------------------------------------------
# CPUMemoryProfiler
# ---------------------------------------------------------------------------


class TestCPUMemoryProfiler:
    """Tests for the lightweight CPU profiler class."""

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_init(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        assert profiler.snapshots == []
        assert profiler.results == []
        assert profiler._monitoring is False
        assert profiler._baseline_snapshot is not None

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_take_snapshot(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=4096, vms=8192, cpu_pct=10.0)
        profiler = CPUMemoryProfiler()
        snap = profiler._take_snapshot()

        assert isinstance(snap, CPUMemorySnapshot)
        assert snap.rss == 4096
        assert snap.vms == 8192
        assert snap.cpu_percent == 10.0
        assert snap.timestamp > 0

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_profile_function(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        result = profiler.profile_function(lambda: 42)

        assert isinstance(result, CPUProfileResult)
        assert result.name == "<lambda>"
        assert result.duration >= 0
        assert len(profiler.results) == 1

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_profile_function_preserves_return_value(self, mock_cls: Any) -> None:
        """profile_function must not swallow the profiled function's return."""
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        # profile_function returns a CPUProfileResult, not the function's return
        result = profiler.profile_function(lambda: 42)
        assert isinstance(result, CPUProfileResult)

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_profile_context(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        with profiler.profile_context("block"):
            _ = sum(range(100))

        assert len(profiler.results) == 1
        assert profiler.results[0].name == "block"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_start_stop_monitoring(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        profiler.start_monitoring(interval=0.05)
        assert _wait_until(lambda: len(profiler.snapshots) > 0)
        profiler.stop_monitoring()

        assert profiler._monitoring is False
        assert len(profiler.snapshots) > 0

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_start_monitoring_idempotent(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()
        profiler.start_monitoring(interval=0.05)
        first_thread = profiler._monitor_thread
        profiler.start_monitoring(interval=0.05)  # should be a no-op
        assert profiler._monitor_thread is first_thread
        assert _wait_until(lambda: len(profiler.snapshots) > 0)
        profiler.stop_monitoring()
        assert profiler._monitoring is False
        assert profiler._monitor_thread is None

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_clear_results(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()
        profiler.profile_function(lambda: None)

        assert len(profiler.results) == 1
        profiler.clear_results()
        assert profiler.results == []
        assert profiler.snapshots == []

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_summary_empty(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=512)
        profiler = CPUMemoryProfiler()
        summary = profiler.get_summary()

        assert summary["mode"] == "cpu"
        assert summary["snapshots_collected"] == 0
        assert summary["baseline_rss"] == 512

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_summary_with_snapshots(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=1000)
        profiler = CPUMemoryProfiler()
        # Manually inject snapshots
        profiler.snapshots.append(
            CPUMemorySnapshot(timestamp=1.0, rss=100, vms=200, cpu_percent=0.0)
        )
        profiler.snapshots.append(
            CPUMemorySnapshot(timestamp=2.0, rss=500, vms=600, cpu_percent=0.0)
        )
        summary = profiler.get_summary()

        assert summary["snapshots_collected"] == 2
        assert summary["peak_memory_usage"] == 500
        assert summary["memory_change_from_baseline"] == 400  # 500 - 100


# ---------------------------------------------------------------------------
# CPUMemoryTracker
# ---------------------------------------------------------------------------


class TestCPUMemoryTracker:
    """Tests for the real-time CPU memory tracker (thread-safe)."""

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_init_defaults(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        assert tracker.is_tracking is False
        assert len(tracker.events) == 0
        assert tracker.stats["peak_memory"] == 0
        assert tracker.stats["total_events"] == 0
        assert isinstance(tracker._events_lock, type(threading.Lock()))

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_start_stop_tracking(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)

        tracker.start_tracking()
        assert tracker.is_tracking is True
        assert _wait_until(lambda: tracker.stats["total_events"] > 0)
        tracker.stop_tracking()
        assert tracker.is_tracking is False

        events = tracker.get_events()  # type: ignore[unreachable, unused-ignore]
        types = [e.event_type for e in events]
        assert "start" in types
        assert "stop" in types

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_start_tracking_idempotent(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        tracker.start_tracking()
        tracker.start_tracking()  # should be a no-op
        tracker.stop_tracking()
        event_types = [event.event_type for event in tracker.get_events()]
        assert event_types.count("start") == 1
        assert event_types.count("stop") == 1

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_stop_tracking_idempotent(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        tracker.stop_tracking()  # not started – should be a no-op

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_start_tracking_does_not_start_thread_when_session_open_fails(
        self, mock_cls: Any
    ) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        thread_start_calls = 0

        class _ThreadSpy:
            def __init__(self, *args: object, **kwargs: object) -> None:
                _ = args, kwargs

            def start(self) -> None:
                nonlocal thread_start_calls
                thread_start_calls += 1

            def join(self, timeout: float | None = None) -> None:
                _ = timeout

        with patch("stormlog.cpu_profiler.threading.Thread", _ThreadSpy):
            with patch.object(
                tracker,
                "_open_session",
                side_effect=RuntimeError("session startup failed"),
            ):
                with pytest.raises(RuntimeError, match="session startup failed"):
                    tracker.start_tracking()

        assert thread_start_calls == 0
        assert tracker.is_tracking is False
        assert tracker._tracking_thread is None

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_add_event_under_lock(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker()

        tracker._add_event("test", 100, "unit test event")

        assert len(tracker.events) == 1
        event = tracker.events[0]
        assert event.event_type == "test"
        assert event.memory_change == 100
        assert event.context == "unit test event"
        assert event.device_id == -1

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_events_no_filter(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("a", 0, "first")
        tracker._add_event("b", 0, "second")

        events = tracker.get_events()
        assert len(events) == 2

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_events_filter_by_type(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("allocation", 10, "alloc")
        tracker._add_event("deallocation", -5, "dealloc")
        tracker._add_event("allocation", 20, "alloc2")

        allocs = tracker.get_events(event_type="allocation")
        assert len(allocs) == 2
        assert all(e.event_type == "allocation" for e in allocs)

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_events_filter_by_since(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        with patch("stormlog.cpu_profiler.time.time", side_effect=[100.0, 200.0]):
            tracker._add_event("a", 0, "old")
            tracker._add_event("b", 0, "new")

        events = tracker.get_events(since=150.0)
        assert len(events) == 1
        assert events[0].context == "new"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_events_filter_last_n(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        for i in range(5):
            tracker._add_event("x", i, f"event_{i}")

        events = tracker.get_events(last_n=2)
        assert len(events) == 2
        assert events[0].context == "event_3"
        assert events[1].context == "event_4"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_clear_events(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("x", 0, "first")
        tracker.stats["peak_memory"] = 9999
        tracker.stats["total_events"] = 42

        tracker.clear_events()

        assert len(tracker.events) == 0
        assert tracker.stats["peak_memory"] == 0
        assert tracker.stats["total_events"] == 0

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_statistics(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=4096)
        tracker = CPUMemoryTracker()
        tracker._add_event("x", 0, "ev")

        stats = tracker.get_statistics()
        assert stats["mode"] == "cpu"
        assert stats["total_events"] == 1
        assert stats["current_memory_allocated"] == 4096
        assert isinstance(stats["tracking_duration_seconds"], float)

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_statistics_with_tracking_duration(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        tracker.start_tracking()
        assert _wait_until(lambda: tracker.stats["total_events"] > 0)
        tracker.stop_tracking()

        stats = tracker.get_statistics()
        assert stats["tracking_duration_seconds"] > 0

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_memory_timeline_empty(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        timeline = tracker.get_memory_timeline()
        assert timeline == {"timestamps": [], "allocated": [], "reserved": []}

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_memory_timeline(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=1024)
        tracker = CPUMemoryTracker()
        tracker._add_event("a", 0, "ev1")
        tracker._add_event("b", 0, "ev2")
        tracker._add_event("c", 0, "ev3")

        tracker.events[0].timestamp = 10.0
        tracker.events[0].memory_allocated = 100
        tracker.events[0].memory_reserved = 110
        tracker.events[1].timestamp = 10.2
        tracker.events[1].memory_allocated = 200
        tracker.events[1].memory_reserved = 220
        tracker.events[2].timestamp = 11.1
        tracker.events[2].memory_allocated = 300
        tracker.events[2].memory_reserved = 330

        timeline = tracker.get_memory_timeline(interval=0.5)
        assert timeline == {
            "timestamps": [10.0, 11.0],
            "allocated": [200.0, 300.0],
            "reserved": [220.0, 330.0],
        }

        coarse_timeline = tracker.get_memory_timeline(interval=2.0)
        assert coarse_timeline == {
            "timestamps": [10.0],
            "allocated": [300.0],
            "reserved": [330.0],
        }

    @pytest.mark.parametrize("interval", [0.0, -1.0])
    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_get_memory_timeline_rejects_nonpositive_interval(
        self, mock_cls: Any, interval: float
    ) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        with pytest.raises(ValueError, match="interval must be > 0"):
            tracker.get_memory_timeline(interval=interval)

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_export_events_csv(self, mock_cls: Any, tmp_path: Path) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("allocation", 10, "csv_test")

        filepath = tmp_path / "events.csv"
        tracker.export_events(str(filepath), format="csv")

        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["schema_version"] == "4"
        assert rows[0]["session_id"]
        assert rows[0]["event_type"] == "allocation"
        assert rows[0]["collector"] == "stormlog.cpu_tracker"
        assert rows[0]["context"] == "csv_test"
        assert rows[0]["job_id"] == ""
        assert rows[0]["rank"] == "0"
        assert rows[0]["local_rank"] == "0"
        assert rows[0]["world_size"] == "1"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_export_events_json(self, mock_cls: Any, tmp_path: Path) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("deallocation", -5, "json_test")

        filepath = tmp_path / "events.json"
        tracker.export_events(str(filepath), format="json")

        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["schema_version"] == 4
        assert data[0]["session_id"]
        assert data[0]["event_type"] == "deallocation"
        assert data[0]["collector"] == "stormlog.cpu_tracker"
        assert isinstance(data[0]["sampling_interval_ms"], int)
        assert isinstance(data[0]["pid"], int)
        assert isinstance(data[0]["host"], str)
        assert data[0]["job_id"] is None
        assert data[0]["rank"] == 0
        assert data[0]["local_rank"] == 0
        assert data[0]["world_size"] == 1
        assert isinstance(data[0]["metadata"], dict)
        capabilities = data[0]["metadata"]["memory_capabilities"]
        assert capabilities["backend"] == "cpu"
        assert capabilities["telemetry_collector"] == "stormlog.cpu_tracker"
        assert capabilities["supports_allocator_allocated"] is True

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_telemetry_export_bypasses_legacy_capability_inference(
        self, mock_cls: Any
    ) -> None:
        mock_cls.return_value = _make_mock_process(rss=4096)
        tracker = CPUMemoryTracker()
        tracker._add_event("sample", 0, "direct_v4", rss=4096)

        with patch(
            "stormlog.telemetry._with_inferred_memory_capabilities",
            side_effect=AssertionError("legacy inference must not run"),
        ):
            record = tracker._telemetry_record_from_event(tracker.get_events()[0])

        assert record["schema_version"] == 4
        assert record["allocator_allocated_bytes"] == 4096
        assert record["device_used_bytes"] == 4096
        assert record["metadata"]["memory_capabilities"] == {
            "backend": "cpu",
            "telemetry_collector": "stormlog.cpu_tracker",
            "sampling_source": "legacy",
            "supports_allocator_allocated": True,
            "supports_allocator_reserved": True,
            "supports_allocator_active": False,
            "supports_allocator_inactive": False,
            "supports_device_used": True,
            "supports_device_free": False,
            "supports_device_total": False,
            "supports_native_allocator_history": False,
            "supports_fragmentation_analysis": True,
            "supports_allocator_attribution": True,
            "supports_bounded_profiling": False,
        }

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_explicit_distributed_identity_is_exported(
        self, mock_cls: Any, tmp_path: Path
    ) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(job_id="job-9", rank=3, local_rank=1, world_size=8)
        tracker._add_event("allocation", 10, "identity_test")

        filepath = tmp_path / "events.json"
        tracker.export_events(str(filepath), format="json")

        data = json.loads(filepath.read_text())
        assert data[0]["job_id"] == "job-9"
        assert data[0]["rank"] == 3
        assert data[0]["local_rank"] == 1
        assert data[0]["world_size"] == 8

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_export_events_unsupported_format(
        self, mock_cls: Any, tmp_path: Path
    ) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("x", 0, "ev")

        with pytest.raises(ValueError, match="Unsupported format"):
            tracker.export_events(str(tmp_path / "out.xml"), format="xml")

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_export_events_empty_is_noop(self, mock_cls: Any, tmp_path: Path) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        filepath = tmp_path / "empty.csv"

        tracker.export_events(str(filepath), format="csv")
        assert not filepath.exists()

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_tracker_streams_events_to_append_only_sink(
        self, mock_cls: Any, tmp_path: Path
    ) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker(
            telemetry_sink_config=TelemetrySinkConfig(
                root_dir=tmp_path / "sink",
                flush_every_events=1,
                flush_every_seconds=1.0,
                rollover_max_bytes=1024,
                retention_max_total_bytes=1024 * 1024,
            )
        )

        tracker._add_event("allocation", 10, "sink_test")
        tracker._close_telemetry_sink()

        segment = tmp_path / "sink" / "segment-000001.jsonl"
        payload = json.loads(segment.read_text(encoding="utf-8").splitlines()[0])
        assert payload["collector"] == "stormlog.cpu_tracker"
        assert payload["event_type"] == "allocation"
        assert payload["context"] == "sink_test"
        stats = tracker.get_statistics()
        assert stats["final_retained_files"] == 1
        assert stats["rollover_count"] == 1

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_tracker_emits_sample_event_during_healthy_iteration(
        self, mock_cls: Any
    ) -> None:
        mock_cls.return_value = _make_mock_process(rss=100)
        tracker = CPUMemoryTracker(sampling_interval=0.01)
        tracker._stop_event = cast(Any, _SequencedStopEvent([False, True]))
        rss_values = iter([100, 128])

        def _current_rss() -> int:
            return next(rss_values)

        tracker._current_rss = _current_rss  # type: ignore[method-assign]

        tracker._tracking_loop()

        event_types = [event.event_type for event in tracker.get_events()]
        assert event_types == ["peak", "allocation", "sample"]
        assert tracker.get_events()[-1].context == "Collected CPU telemetry sample."

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_tracker_recreates_sink_on_restart(
        self, mock_cls: Any, tmp_path: Path
    ) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker(
            telemetry_sink_config=TelemetrySinkConfig(
                root_dir=tmp_path / "sink",
                flush_every_events=1,
                flush_every_seconds=1.0,
                rollover_max_bytes=1024 * 1024,
                retention_max_total_bytes=1024 * 1024,
            )
        )

        class _NoOpThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                _ = args, kwargs

            def start(self) -> None:
                return None

            def is_alive(self) -> bool:
                return False

            def join(self, timeout: float | None = None) -> None:
                _ = timeout

        with patch("stormlog.cpu_profiler.threading.Thread", _NoOpThread):
            tracker.start_tracking()
            first_sink = tracker._telemetry_sink
            assert first_sink is not None
            tracker.stop_tracking()
            assert tracker._telemetry_sink is None

            tracker.start_tracking()
            second_sink = cast(Any, tracker._telemetry_sink)

        assert second_sink is not None
        assert second_sink is not first_sink

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_tracker_emits_structured_phase_boundaries(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker(sampling_interval=0.05)

        class _NoOpThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                _ = args, kwargs

            def start(self) -> None:
                return None

            def is_alive(self) -> bool:
                return False

            def join(self, timeout: float | None = None) -> None:
                _ = timeout

        with patch("stormlog.cpu_profiler.threading.Thread", _NoOpThread):
            tracker.start_tracking()
            with tracker.phase("train_step", metadata={"epoch": 1}) as handle:
                assert handle.phase_path == "train_step"
            tracker.stop_tracking()

        phase_events = [
            event
            for event in tracker.get_events()
            if event.event_type.startswith("phase_")
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
        assert enter_scope.path == ("train_step",)
        assert enter_scope.depth == 1
        assert enter_scope.attributes == {"epoch": 1}
        assert exit_scope.scope_id == enter_scope.scope_id
        assert exit_scope.path == enter_scope.path

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_tracker_disables_sink_after_append_failure(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker()
        sink = _FailingSink(fail_on={"append"})
        tracker._telemetry_sink = cast(Any, sink)

        tracker._add_event("allocation", 10, "sink_failure")

        assert len(tracker.events) == 1
        assert tracker.events[-1].context == "sink_failure"
        assert sink.append_calls == 1
        assert sink.close_calls == 1
        assert tracker._telemetry_sink is None

        tracker.is_tracking = True
        tracker.stop_tracking()

        assert tracker.events[-1].event_type == "stop"
        summary = tracker.get_session_summary()
        assert summary is not None
        assert summary.status == "incomplete"
        assert tracker.get_statistics()["session_status"] == "incomplete"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_tracker_disables_sink_after_flush_failure(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker()
        sink = _FailingSink(fail_on={"flush"})
        tracker._telemetry_sink = cast(Any, sink)

        tracker._flush_telemetry_sink()

        assert sink.flush_calls == 1
        assert sink.close_calls == 1
        assert tracker._telemetry_sink is None

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_stop_tracking_disables_sink_after_close_failure(
        self, mock_cls: Any
    ) -> None:
        mock_cls.return_value = _make_mock_process(rss=2048)
        tracker = CPUMemoryTracker()
        tracker.is_tracking = True
        sink = _FailingSink(fail_on={"close"})
        tracker._telemetry_sink = cast(Any, sink)

        tracker.stop_tracking()

        assert tracker.events[-1].event_type == "stop"
        assert sink.close_calls >= 1
        assert tracker._telemetry_sink is None
        summary = tracker.get_session_summary()
        assert summary is not None
        assert summary.status == "incomplete"
        assert tracker.get_statistics()["session_status"] == "incomplete"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_export_events_with_timestamp(self, mock_cls: Any, tmp_path: Path) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("x", 0, "ts_test")

        result = tracker.export_events_with_timestamp(str(tmp_path), "json")
        assert result.endswith(".json")
        assert "cpu_tracker_" in result

    def test_format_bytes(self) -> None:
        assert CPUMemoryTracker._format_bytes(0) == "0.00 B"
        assert CPUMemoryTracker._format_bytes(1024) == "1.00 KB"
        assert CPUMemoryTracker._format_bytes(1024 * 1024) == "1.00 MB"
        assert CPUMemoryTracker._format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        assert CPUMemoryTracker._format_bytes(1024**4) == "1.00 TB"
        # Stays at TB for very large values
        assert "TB" in CPUMemoryTracker._format_bytes(1024**5)

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_max_events_respected(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(max_events=5)

        for i in range(10):
            tracker._add_event("x", i, f"ev_{i}")

        assert len(tracker.events) == 5
        # Oldest events should have been evicted
        contexts = [e.context for e in tracker.events]
        assert contexts == [f"ev_{i}" for i in range(5, 10)]
        stats = tracker.get_statistics()
        assert stats["history_window_limit_events"] == 5
        assert stats["history_retained_events"] == 5
        assert stats["history_dropped_events"] == 5

    # ------------------------------------------------------------------
    # Thread-safety tests
    # ------------------------------------------------------------------

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_concurrent_add_and_read(self, mock_cls: Any) -> None:
        """Multiple writers and readers should not raise or lose events."""
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(max_events=50_000)

        num_writers = 4
        writes_per_thread = 500
        errors: List[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(writes_per_thread):
                    tracker._add_event("w", i, f"t{thread_id}_{i}")
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(200):
                    _ = tracker.get_events()
                    _ = tracker.get_events(event_type="w")
                    _ = tracker.get_events(last_n=10)
                    _ = tracker.get_statistics()
                    _ = tracker.get_memory_timeline()
            except Exception as exc:
                errors.append(exc)

        threads = []
        for tid in range(num_writers):
            threads.append(threading.Thread(target=writer, args=(tid,)))
        # Two concurrent readers
        threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        for t in threads:
            assert (
                not t.is_alive()
            ), f"Thread {t.name} did not complete (deadlock/timeout)"

        assert errors == [], f"Concurrent access raised errors: {errors}"
        assert len(tracker.events) == num_writers * writes_per_thread

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_concurrent_tracking_and_get_events(self, mock_cls: Any) -> None:
        """get_events() must not raise while _tracking_loop() runs."""
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.01)
        errors: List[Exception] = []

        tracker.start_tracking()

        def reader() -> None:
            try:
                for _ in range(100):
                    _ = tracker.get_events()
                    time.sleep(0.005)
            except Exception as exc:
                errors.append(exc)

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        reader_thread.join(timeout=5)
        assert (
            not reader_thread.is_alive()
        ), "reader_thread timed out (possible deadlock)"

        tracker.stop_tracking()

        assert errors == [], f"get_events() raised during tracking: {errors}"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_concurrent_clear_and_add(self, mock_cls: Any) -> None:
        """clear_events() and _add_event() running concurrently must not raise."""
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        errors: List[Exception] = []

        def adder() -> None:
            try:
                for i in range(500):
                    tracker._add_event("x", i, f"ev_{i}")
            except Exception as exc:
                errors.append(exc)

        def clearer() -> None:
            try:
                for _ in range(50):
                    tracker.clear_events()
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=clearer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive(), "adder thread timed out (possible deadlock)"
        assert not t2.is_alive(), "clearer thread timed out (possible deadlock)"

        assert errors == [], f"Concurrent clear/add raised: {errors}"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_events_lock_exists(self, mock_cls: Any) -> None:
        """Verify the lock attribute is present and is a Lock."""
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        assert hasattr(tracker, "_events_lock")
        # Verify it behaves like a lock: acquire/release should work
        assert tracker._events_lock.acquire(timeout=0.1)
        tracker._events_lock.release()

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_concurrent_export_and_add(self, mock_cls: Any, tmp_path: Path) -> None:
        """export_events() must not raise while _add_event() runs concurrently."""
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        errors: List[Exception] = []

        # Pre-populate some events
        for i in range(20):
            tracker._add_event("x", i, f"ev_{i}")

        def adder() -> None:
            try:
                for i in range(200):
                    tracker._add_event("y", i, f"new_{i}")
            except Exception as exc:
                errors.append(exc)

        def exporter() -> None:
            try:
                for i in range(10):
                    tracker.export_events(str(tmp_path / f"out_{i}.csv"), format="csv")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=exporter)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive(), "adder thread timed out (possible deadlock)"
        assert not t2.is_alive(), "exporter thread timed out (possible deadlock)"

        assert errors == [], f"Concurrent export/add raised: {errors}"

    @patch("stormlog.cpu_profiler.psutil.Process")
    def test_no_deque_mutation_error_under_concurrent_load(
        self, mock_cls: Any, tmp_path: Path
    ) -> None:
        """Regression: concurrent add/read/export must not raise
        ``RuntimeError: deque mutated during iteration`` (the exact error
        that was reproducible on *main* before the thread-safety fix).
        """
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(max_events=50_000)
        errors: List[Exception] = []

        def writer(tid: int) -> None:
            try:
                for i in range(500):
                    tracker._add_event("w", i, f"t{tid}_{i}")
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(300):
                    tracker.get_events()
                    tracker.get_memory_timeline()
                    tracker.get_statistics()
            except Exception as exc:
                errors.append(exc)

        def exporter() -> None:
            try:
                for i in range(20):
                    tracker.export_events(
                        str(tmp_path / f"regression_{i}.csv"), format="csv"
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=exporter),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        for t in threads:
            assert not t.is_alive(), f"Thread {t.name} timed out (possible deadlock)"

        # Explicitly check for the *exact* failure mode fixed by this PR.
        deque_errors = [
            e
            for e in errors
            if isinstance(e, RuntimeError) and "deque mutated" in str(e)
        ]
        assert deque_errors == [], (
            f"Regression: deque-mutation errors still occur under concurrent load: "
            f"{deque_errors}"
        )
        # No other errors should have occurred either.
        assert errors == [], f"Unexpected errors under concurrent load: {errors}"
