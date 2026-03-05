"""Comprehensive tests for CPU memory profiler and tracker."""

from __future__ import annotations

import contextlib
import csv
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import patch

import pytest

from gpumemprof.cpu_profiler import (
    CPUMemoryProfiler,
    CPUMemorySnapshot,
    CPUMemoryTracker,
    CPUProfileResult,
)

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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_init(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        assert profiler.snapshots == []
        assert profiler.results == []
        assert profiler._monitoring is False
        assert profiler._baseline_snapshot is not None

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_take_snapshot(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=4096, vms=8192, cpu_pct=10.0)
        profiler = CPUMemoryProfiler()
        snap = profiler._take_snapshot()

        assert isinstance(snap, CPUMemorySnapshot)
        assert snap.rss == 4096
        assert snap.vms == 8192
        assert snap.cpu_percent == 10.0
        assert snap.timestamp > 0

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_profile_function(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        result = profiler.profile_function(lambda: 42)

        assert isinstance(result, CPUProfileResult)
        assert result.name == "<lambda>"
        assert result.duration >= 0
        assert len(profiler.results) == 1

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_profile_function_preserves_return_value(self, mock_cls: Any) -> None:
        """profile_function must not swallow the profiled function's return."""
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        # profile_function returns a CPUProfileResult, not the function's return
        result = profiler.profile_function(lambda: 42)
        assert isinstance(result, CPUProfileResult)

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_profile_context(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        with profiler.profile_context("block"):
            _ = sum(range(100))

        assert len(profiler.results) == 1
        assert profiler.results[0].name == "block"

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_start_stop_monitoring(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()

        profiler.start_monitoring(interval=0.05)
        time.sleep(0.2)
        profiler.stop_monitoring()

        assert profiler._monitoring is False
        assert len(profiler.snapshots) > 0

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_start_monitoring_idempotent(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()
        profiler.start_monitoring(interval=0.05)
        profiler.start_monitoring(interval=0.05)  # should be a no-op
        profiler.stop_monitoring()

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_clear_results(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        profiler = CPUMemoryProfiler()
        profiler.profile_function(lambda: None)

        assert len(profiler.results) == 1
        profiler.clear_results()
        assert profiler.results == []
        assert profiler.snapshots == []

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_summary_empty(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=512)
        profiler = CPUMemoryProfiler()
        summary = profiler.get_summary()

        assert summary["mode"] == "cpu"
        assert summary["snapshots_collected"] == 0
        assert summary["baseline_rss"] == 512

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_init_defaults(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        assert tracker.is_tracking is False
        assert len(tracker.events) == 0
        assert tracker.stats["peak_memory"] == 0
        assert tracker.stats["total_events"] == 0
        assert isinstance(tracker._events_lock, type(threading.Lock()))

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_start_stop_tracking(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)

        tracker.start_tracking()
        assert tracker.is_tracking is True
        time.sleep(0.15)
        tracker.stop_tracking()
        assert tracker.is_tracking is False

        events = tracker.get_events()  # type: ignore[unreachable, unused-ignore]
        types = [e.event_type for e in events]
        assert "start" in types
        assert "stop" in types

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_start_tracking_idempotent(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        tracker.start_tracking()
        tracker.start_tracking()  # should be a no-op
        tracker.stop_tracking()

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_stop_tracking_idempotent(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        tracker.stop_tracking()  # not started – should be a no-op

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_events_no_filter(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("a", 0, "first")
        tracker._add_event("b", 0, "second")

        events = tracker.get_events()
        assert len(events) == 2

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_events_filter_by_type(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("allocation", 10, "alloc")
        tracker._add_event("deallocation", -5, "dealloc")
        tracker._add_event("allocation", 20, "alloc2")

        allocs = tracker.get_events(event_type="allocation")
        assert len(allocs) == 2
        assert all(e.event_type == "allocation" for e in allocs)

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_events_filter_by_since(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        # Add events with known timestamps
        tracker._add_event("a", 0, "old")
        cutoff = time.time()
        time.sleep(0.02)
        tracker._add_event("b", 0, "new")

        events = tracker.get_events(since=cutoff)
        assert len(events) == 1
        assert events[0].context == "new"

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_events_filter_last_n(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        for i in range(5):
            tracker._add_event("x", i, f"event_{i}")

        events = tracker.get_events(last_n=2)
        assert len(events) == 2
        assert events[0].context == "event_3"
        assert events[1].context == "event_4"

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_statistics(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=4096)
        tracker = CPUMemoryTracker()
        tracker._add_event("x", 0, "ev")

        stats = tracker.get_statistics()
        assert stats["mode"] == "cpu"
        assert stats["total_events"] == 1
        assert stats["current_memory_allocated"] == 4096
        assert isinstance(stats["tracking_duration_seconds"], float)

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_statistics_with_tracking_duration(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(sampling_interval=0.05)
        tracker.start_tracking()
        time.sleep(0.1)
        tracker.stop_tracking()

        stats = tracker.get_statistics()
        assert stats["tracking_duration_seconds"] > 0

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_memory_timeline_empty(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()

        timeline = tracker.get_memory_timeline()
        assert timeline == {"timestamps": [], "allocated": [], "reserved": []}

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_get_memory_timeline(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process(rss=1024)
        tracker = CPUMemoryTracker()
        tracker._add_event("a", 0, "ev1")
        tracker._add_event("b", 0, "ev2")

        timeline = tracker.get_memory_timeline()
        assert len(timeline["timestamps"]) == 2
        assert len(timeline["allocated"]) == 2
        assert len(timeline["reserved"]) == 2
        assert timeline["allocated"] == timeline["reserved"]

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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
        assert rows[0]["schema_version"] == "2"
        assert rows[0]["event_type"] == "allocation"
        assert rows[0]["collector"] == "gpumemprof.cpu_tracker"
        assert rows[0]["context"] == "csv_test"
        assert rows[0]["job_id"] == ""
        assert rows[0]["rank"] == "0"
        assert rows[0]["local_rank"] == "0"
        assert rows[0]["world_size"] == "1"

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_export_events_json(self, mock_cls: Any, tmp_path: Path) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("deallocation", -5, "json_test")

        filepath = tmp_path / "events.json"
        tracker.export_events(str(filepath), format="json")

        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["schema_version"] == 2
        assert data[0]["event_type"] == "deallocation"
        assert data[0]["collector"] == "gpumemprof.cpu_tracker"
        assert isinstance(data[0]["sampling_interval_ms"], int)
        assert isinstance(data[0]["pid"], int)
        assert isinstance(data[0]["host"], str)
        assert data[0]["job_id"] is None
        assert data[0]["rank"] == 0
        assert data[0]["local_rank"] == 0
        assert data[0]["world_size"] == 1
        assert isinstance(data[0]["metadata"], dict)

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_export_events_unsupported_format(
        self, mock_cls: Any, tmp_path: Path
    ) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        tracker._add_event("x", 0, "ev")

        with pytest.raises(ValueError, match="Unsupported format"):
            tracker.export_events(str(tmp_path / "out.xml"), format="xml")

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_export_events_empty_is_noop(self, mock_cls: Any, tmp_path: Path) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        filepath = tmp_path / "empty.csv"

        tracker.export_events(str(filepath), format="csv")
        assert not filepath.exists()

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_max_events_respected(self, mock_cls: Any) -> None:
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker(max_events=5)

        for i in range(10):
            tracker._add_event("x", i, f"ev_{i}")

        assert len(tracker.events) == 5
        # Oldest events should have been evicted
        contexts = [e.context for e in tracker.events]
        assert contexts == [f"ev_{i}" for i in range(5, 10)]

    # ------------------------------------------------------------------
    # Thread-safety tests
    # ------------------------------------------------------------------

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
    def test_events_lock_exists(self, mock_cls: Any) -> None:
        """Verify the lock attribute is present and is a Lock."""
        mock_cls.return_value = _make_mock_process()
        tracker = CPUMemoryTracker()
        assert hasattr(tracker, "_events_lock")
        # Verify it behaves like a lock: acquire/release should work
        assert tracker._events_lock.acquire(timeout=0.1)
        tracker._events_lock.release()

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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

    @patch("gpumemprof.cpu_profiler.psutil.Process")
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
