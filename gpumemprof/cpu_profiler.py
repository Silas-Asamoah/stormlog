"""CPU-only memory profiler and tracker."""

from __future__ import annotations

import csv
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import psutil

from gpumemprof.telemetry import (
    resolve_distributed_identity,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gpumemprof.tracker import TrackingEvent
else:
    try:
        from gpumemprof.tracker import TrackingEvent
    except ImportError:

        @dataclass
        class TrackingEvent:
            """Fallback CPU tracking event used when GPU tracker imports are unavailable."""

            timestamp: float
            event_type: str
            memory_allocated: int
            memory_reserved: int
            memory_change: int
            device_id: int
            context: Optional[str] = None
            job_id: Optional[str] = None
            rank: int = 0
            local_rank: int = 0
            world_size: int = 1
            metadata: Optional[Dict[str, Any]] = None
            active_memory: Optional[int] = None
            inactive_memory: Optional[int] = None
            device_used: Optional[int] = None
            device_free: Optional[int] = None
            device_total: Optional[int] = None
            backend: str = "cpu"


@dataclass
class CPUMemorySnapshot:
    """Point-in-time CPU memory snapshot."""

    timestamp: float
    rss: int
    vms: int
    cpu_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rss": self.rss,
            "vms": self.vms,
            "cpu_percent": self.cpu_percent,
        }


@dataclass
class CPUProfileResult:
    """Results from profiling a CPU function/context."""

    name: str
    duration: float
    snapshot_before: CPUMemorySnapshot
    snapshot_after: CPUMemorySnapshot
    peak_rss: int

    def memory_diff(self) -> int:
        return self.snapshot_after.rss - self.snapshot_before.rss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration": self.duration,
            "memory_diff": self.memory_diff(),
            "peak_rss": self.peak_rss,
            "before": self.snapshot_before.to_dict(),
            "after": self.snapshot_after.to_dict(),
        }


class CPUMemoryProfiler:
    """Lightweight CPU memory profiler mirroring the GPU API."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.snapshots: List[CPUMemorySnapshot] = []
        self.results: List[CPUProfileResult] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 0.1
        self._baseline_snapshot = self._take_snapshot()

    def _take_snapshot(self) -> CPUMemorySnapshot:
        with self.process.oneshot():
            mem = self.process.memory_info()
        cpu_pct = self.process.cpu_percent(interval=None)
        return CPUMemorySnapshot(
            timestamp=time.time(),
            rss=mem.rss,
            vms=mem.vms,
            cpu_percent=cpu_pct,
        )

    def start_monitoring(self, interval: float = 0.1) -> None:
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_interval = interval
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while self._monitoring:
            self.snapshots.append(self._take_snapshot())
            time.sleep(self._monitor_interval)

    def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def profile_function(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> CPUProfileResult:
        before = self._take_snapshot()
        start = time.time()
        _result = func(*args, **kwargs)
        end = time.time()
        after = self._take_snapshot()
        peak_rss = max(before.rss, after.rss)
        profile = CPUProfileResult(
            name=getattr(func, "__name__", "cpu_function"),
            duration=end - start,
            snapshot_before=before,
            snapshot_after=after,
            peak_rss=peak_rss,
        )
        self.results.append(profile)
        return profile

    def profile_context(self, name: str = "context") -> Any:
        class _Context:
            def __init__(self, outer: CPUMemoryProfiler, label: str) -> None:
                self.outer = outer
                self.label = label

            def __enter__(self) -> Any:
                self.before = self.outer._take_snapshot()
                self.start = time.time()
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                after = self.outer._take_snapshot()
                end = time.time()
                peak_rss = max(self.before.rss, after.rss)
                profile = CPUProfileResult(
                    name=self.label,
                    duration=end - self.start,
                    snapshot_before=self.before,
                    snapshot_after=after,
                    peak_rss=peak_rss,
                )
                self.outer.results.append(profile)

        return _Context(self, name)

    def clear_results(self) -> None:
        self.snapshots.clear()
        self.results.clear()
        self._baseline_snapshot = self._take_snapshot()

    def get_summary(self) -> Dict[str, Any]:
        if not self.snapshots:
            snapshots = [self._baseline_snapshot]
        else:
            snapshots = self.snapshots

        rss_values = [snap.rss for snap in snapshots]
        peak = max(rss_values) if rss_values else self._baseline_snapshot.rss
        change = rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0

        return {
            "mode": "cpu",
            "snapshots_collected": len(self.snapshots),
            "peak_memory_usage": peak,
            "memory_change_from_baseline": change,
            "baseline_rss": self._baseline_snapshot.rss,
        }


class CPUMemoryTracker:
    """CPU tracker offering a superset of the GPU tracker interface."""

    def __init__(
        self,
        sampling_interval: float = 0.5,
        max_events: int = 10_000,
        enable_alerts: bool = True,
        job_id: Optional[str] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        self.process = psutil.Process()
        self.sampling_interval = sampling_interval
        self.events: deque[TrackingEvent] = deque(maxlen=max_events)
        self._events_lock = threading.Lock()
        self.is_tracking = False
        self._stop_event = threading.Event()
        self._tracking_thread: Optional[threading.Thread] = None
        self.enable_alerts = enable_alerts
        self.distributed_identity = resolve_distributed_identity(
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            env=os.environ,
        )

        self.stats: Dict[str, Any] = {
            "tracking_start_time": None,
            "peak_memory": 0,
            "total_events": 0,
            "alert_count": 0,
        }

    def _current_rss(self) -> int:
        with self.process.oneshot():
            return int(self.process.memory_info().rss)

    def start_tracking(self) -> None:
        if self.is_tracking:
            return
        self.is_tracking = True
        self._stop_event.clear()
        with self._events_lock:
            self.stats["tracking_start_time"] = time.time()
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True
        )
        self._tracking_thread.start()
        self._add_event("start", 0, "CPU memory tracking started")

    def stop_tracking(self) -> None:
        if not self.is_tracking:
            return
        self.is_tracking = False
        self._stop_event.set()
        if self._tracking_thread:
            self._tracking_thread.join(timeout=1.0)
        self._add_event("stop", 0, "CPU memory tracking stopped")

    def _tracking_loop(self) -> None:
        last_rss = self._current_rss()

        while not self._stop_event.wait(self.sampling_interval):
            try:
                current_rss = self._current_rss()
            except Exception as exc:
                logger.debug("Error sampling RSS in tracking loop: %s", exc)
                continue

            change = current_rss - last_rss
            is_new_peak = False
            with self._events_lock:
                self.stats["total_events"] += 1
                if current_rss > self.stats["peak_memory"]:
                    self.stats["peak_memory"] = current_rss
                    is_new_peak = True

            if is_new_peak:
                self._add_event(
                    "peak",
                    change,
                    f"New CPU peak RSS: {self._format_bytes(current_rss)}",
                )

            if change > 0:
                self._add_event(
                    "allocation",
                    change,
                    f"RSS increased by {self._format_bytes(change)}",
                )
            elif change < 0:
                self._add_event(
                    "deallocation",
                    change,
                    f"RSS decreased by {self._format_bytes(abs(change))}",
                )

            last_rss = current_rss

    def _add_event(self, event_type: str, memory_change: int, context: str) -> None:
        rss = self._current_rss()
        event = TrackingEvent(
            timestamp=time.time(),
            event_type=event_type,
            memory_allocated=rss,
            memory_reserved=rss,
            memory_change=memory_change,
            device_id=-1,
            context=context,
            job_id=self.distributed_identity["job_id"],
            rank=self.distributed_identity["rank"],
            local_rank=self.distributed_identity["local_rank"],
            world_size=self.distributed_identity["world_size"],
        )
        with self._events_lock:
            self.events.append(event)

    def get_events(
        self,
        event_type: Optional[str] = None,
        last_n: Optional[int] = None,
        since: Optional[float] = None,
    ) -> List[TrackingEvent]:
        """
        Get tracking events with optional filtering.

        Args:
            event_type: Filter by event type
            last_n: Get last N events
            since: Get events since timestamp

        Returns:
            List of filtered events
        """
        with self._events_lock:
            events: List[TrackingEvent] = list(self.events)

        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Filter by time
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Limit results
        if last_n:
            events = events[-last_n:]

        return events

    def get_statistics(self) -> Dict[str, Any]:
        rss = self._current_rss()
        with self._events_lock:
            total_events = len(self.events)
            peak_memory = self.stats["peak_memory"]
            tracking_start_time = self.stats.get("tracking_start_time")
        duration = 0.0
        if isinstance(tracking_start_time, (int, float)):
            duration = time.time() - float(tracking_start_time)
        return {
            "mode": "cpu",
            "total_events": total_events,
            "peak_memory": peak_memory,
            "current_memory_allocated": rss,
            "tracking_duration_seconds": duration,
        }

    def get_memory_timeline(self, interval: float = 1.0) -> Dict[str, List[float]]:
        with self._events_lock:
            events_snapshot = list(self.events)

        if not events_snapshot:
            return {"timestamps": [], "allocated": [], "reserved": []}

        timestamps = [event.timestamp for event in events_snapshot]
        allocated = [float(event.memory_allocated) for event in events_snapshot]
        return {
            "timestamps": timestamps,
            "allocated": allocated,
            "reserved": allocated,
        }

    def clear_events(self) -> None:
        with self._events_lock:
            self.events.clear()
            self.stats["peak_memory"] = 0
            self.stats["total_events"] = 0

    def export_events(self, filename: str, format: str = "csv") -> None:
        with self._events_lock:
            events_snapshot = list(self.events)

        host = socket.gethostname()
        pid = os.getpid()
        sampling_interval_ms = int(round(self.sampling_interval * 1000))

        records = [
            telemetry_event_to_dict(
                telemetry_event_from_record(
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "memory_allocated": event.memory_allocated,
                        "memory_reserved": event.memory_reserved,
                        "memory_change": event.memory_change,
                        "device_id": event.device_id,
                        "context": event.context,
                        "job_id": event.job_id,
                        "rank": event.rank,
                        "local_rank": event.local_rank,
                        "world_size": event.world_size,
                        "collector": "gpumemprof.cpu_tracker",
                        "sampling_interval_ms": sampling_interval_ms,
                        "pid": pid,
                        "host": host,
                    },
                    default_collector="gpumemprof.cpu_tracker",
                    default_sampling_interval_ms=sampling_interval_ms,
                )
            )
            for event in events_snapshot
        ]

        if not records:
            return

        if format == "csv":
            with open(filename, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        elif format == "json":
            with open(filename, "w") as jsonfile:
                json.dump(records, jsonfile, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_events_with_timestamp(self, directory: str, format: str) -> str:
        filename = f"{directory}/cpu_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        self.export_events(filename, format=format)
        return filename

    @staticmethod
    def _format_bytes(value: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(value)
        unit_idx = 0
        while size >= 1024 and unit_idx < len(units) - 1:
            size /= 1024
            unit_idx += 1
        return f"{size:.2f} {units[unit_idx]}"
