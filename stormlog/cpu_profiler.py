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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import psutil

from stormlog.phases import PhaseHandle, PhaseRecorder, PhaseToken
from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SESSION_STATUS_RUNNING,
    SessionSummary,
    create_session_summary,
    finalize_session_summary,
    now_ns,
    update_session_summary,
)
from stormlog.telemetry import (
    SCHEMA_VERSION_V4,
    TelemetryEventV4,
    resolve_distributed_identity,
    telemetry_event_to_dict,
)
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig

logger = logging.getLogger(__name__)

_CPU_MEMORY_CAPABILITIES: Dict[str, Any] = {
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

if TYPE_CHECKING:
    from stormlog.tracker import TrackingEvent
else:
    try:
        from stormlog.tracker import TrackingEvent
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
            session_id: Optional[str] = None
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
        if interval <= 0:
            raise ValueError("interval must be > 0")
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
        telemetry_sink_config: Optional[TelemetrySinkConfig] = None,
    ) -> None:
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be > 0")
        if max_events <= 0:
            raise ValueError("max_events must be >= 1")

        self.process = psutil.Process()
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.events: deque[TrackingEvent] = deque(maxlen=max_events)
        self._events_lock = threading.Lock()
        self.is_tracking = False
        self._stop_event = threading.Event()
        self._tracking_thread: Optional[threading.Thread] = None
        self.enable_alerts = enable_alerts
        self._telemetry_sink_config = telemetry_sink_config
        self._telemetry_sink = (
            AppendOnlyTelemetrySink(telemetry_sink_config)
            if telemetry_sink_config is not None
            else None
        )
        self.distributed_identity = resolve_distributed_identity(
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            env=os.environ,
        )
        self.session_source = "stormlog.cpu_profiler"
        self._session_summary: Optional[SessionSummary] = None
        self._history_dropped_events = 0
        self._last_sink_diagnostics: Dict[str, int] = self._empty_sink_diagnostics()
        self._phase_state = PhaseRecorder()

        self.stats: Dict[str, Any] = {
            "tracking_start_time": None,
            "peak_memory": 0,
            "total_events": 0,
            "alert_count": 0,
        }

    @staticmethod
    def _empty_sink_diagnostics() -> Dict[str, int]:
        return {
            "rollover_count": 0,
            "pruned_segment_count": 0,
            "pruned_bytes": 0,
            "final_retained_files": 0,
            "final_retained_bytes": 0,
        }

    def _current_rss(self) -> int:
        with self.process.oneshot():
            return int(self.process.memory_info().rss)

    def _open_session(self) -> SessionSummary:
        if self._session_summary is not None:
            return self._session_summary
        summary = create_session_summary(
            source=self.session_source,
            status=SESSION_STATUS_RUNNING,
            started_at_ns=now_ns(),
            host=socket.gethostname(),
            pid=os.getpid(),
            job_id=self.distributed_identity["job_id"],
            rank=self.distributed_identity["rank"],
            local_rank=self.distributed_identity["local_rank"],
            world_size=self.distributed_identity["world_size"],
        )
        self._session_summary = summary
        if self._telemetry_sink is not None and hasattr(
            self._telemetry_sink, "start_session"
        ):
            self._session_summary = self._telemetry_sink.start_session(summary)
        return self._session_summary

    def get_session_summary(self) -> Optional[SessionSummary]:
        return self._session_summary

    def _ensure_telemetry_sink(self) -> None:
        if self._telemetry_sink is None and self._telemetry_sink_config is not None:
            self._telemetry_sink = AppendOnlyTelemetrySink(self._telemetry_sink_config)

    def start_tracking(self) -> None:
        if self.is_tracking:
            return
        self._session_summary = None
        self._phase_state.reset()
        self._ensure_telemetry_sink()
        self._stop_event.clear()
        with self._events_lock:
            self.events.clear()
            self.stats["peak_memory"] = 0
            self.stats["total_events"] = 0
            self.stats["alert_count"] = 0
            self.stats["tracking_start_time"] = time.time()
            self._history_dropped_events = 0
        self._last_sink_diagnostics = self._empty_sink_diagnostics()
        self._open_session()
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True
        )
        self._tracking_thread.start()
        self.is_tracking = True
        self._add_event("start", 0, "CPU memory tracking started")

    def stop_tracking(self) -> None:
        if not self.is_tracking:
            return
        self.is_tracking = False
        self._stop_event.set()
        if self._tracking_thread:
            self._tracking_thread.join(timeout=1.0)
        self._add_event("stop", 0, "CPU memory tracking stopped")
        self._close_telemetry_sink()
        if self._session_summary is not None:
            self._session_summary = finalize_session_summary(
                self._session_summary,
                ended_at_ns=now_ns(),
            )
        self._phase_state.reset()

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
                    rss=current_rss,
                )

            if change > 0:
                self._add_event(
                    "allocation",
                    change,
                    f"RSS increased by {self._format_bytes(change)}",
                    rss=current_rss,
                )
            elif change < 0:
                self._add_event(
                    "deallocation",
                    change,
                    f"RSS decreased by {self._format_bytes(abs(change))}",
                    rss=current_rss,
                )

            self._add_event(
                "sample",
                0,
                "Collected CPU telemetry sample.",
                rss=current_rss,
            )

            last_rss = current_rss
            self._flush_telemetry_sink()

    def _add_event(
        self,
        event_type: str,
        memory_change: int,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        rss: int | None = None,
    ) -> None:
        if rss is None:
            rss = self._current_rss()
        event = TrackingEvent(
            timestamp=time.time(),
            event_type=event_type,
            memory_allocated=rss,
            memory_reserved=rss,
            memory_change=memory_change,
            device_id=-1,
            session_id=(
                self._open_session().session_id
                if self._session_summary is None
                else self._session_summary.session_id
            ),
            context=context,
            job_id=self.distributed_identity["job_id"],
            rank=self.distributed_identity["rank"],
            local_rank=self.distributed_identity["local_rank"],
            world_size=self.distributed_identity["world_size"],
            metadata=dict(metadata or {}),
        )
        with self._events_lock:
            if len(self.events) == self.max_events:
                self._history_dropped_events += 1
            self.events.append(event)
        self._append_to_telemetry_sink(event)

    def enter_phase(
        self, name: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> PhaseHandle:
        """Enter one structured CPU tracking phase."""
        if not self.is_tracking:
            raise RuntimeError("Tracking must be active before entering a phase.")
        session = self._open_session()
        token, boundary = self._phase_state.enter(
            session_id=session.session_id,
            rank=self.distributed_identity["rank"],
            name=name,
            attrs=metadata,
        )
        self._add_event(
            boundary.event_type,
            0,
            boundary.context,
            metadata=boundary.metadata,
        )
        return PhaseHandle(
            scope_id=boundary.scope_id,
            name=name,
            path=boundary.path,
            close_callback=lambda: self._emit_phase_exit(token),
        )

    @contextmanager
    def phase(self, name: str, *, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Context manager that emits structured CPU phase telemetry."""
        handle = self.enter_phase(name, metadata=metadata)
        try:
            yield handle
        finally:
            handle.close()

    def _emit_phase_exit(self, token: PhaseToken) -> None:
        boundary = self._phase_state.exit(token)
        self._add_event(
            boundary.event_type,
            0,
            boundary.context,
            metadata=boundary.metadata,
        )

    def _telemetry_record_from_event(self, event: TrackingEvent) -> Dict[str, Any]:
        host = socket.gethostname()
        pid = os.getpid()
        sampling_interval_ms = int(round(self.sampling_interval * 1000))
        session_id = event.session_id or (
            self._session_summary.session_id
            if self._session_summary is not None
            else self._open_session().session_id
        )
        metadata = dict(event.metadata or {})
        metadata["memory_capabilities"] = dict(_CPU_MEMORY_CAPABILITIES)
        return telemetry_event_to_dict(
            TelemetryEventV4(
                schema_version=SCHEMA_VERSION_V4,
                session_id=session_id,
                timestamp_ns=int(event.timestamp * 1_000_000_000),
                event_type=event.event_type,
                collector="stormlog.cpu_tracker",
                sampling_interval_ms=sampling_interval_ms,
                pid=pid,
                host=host,
                device_id=event.device_id,
                allocator_allocated_bytes=event.memory_allocated,
                allocator_reserved_bytes=event.memory_reserved,
                allocator_active_bytes=event.active_memory,
                allocator_inactive_bytes=event.inactive_memory,
                allocator_change_bytes=event.memory_change,
                device_used_bytes=event.memory_allocated,
                device_free_bytes=event.device_free,
                device_total_bytes=event.device_total,
                context=event.context,
                job_id=event.job_id,
                rank=event.rank,
                local_rank=event.local_rank,
                world_size=event.world_size,
                metadata=metadata,
            )
        )

    def _append_to_telemetry_sink(self, event: TrackingEvent) -> None:
        if self._telemetry_sink is None:
            return
        try:
            self._telemetry_sink.append(self._telemetry_record_from_event(event))
            self._last_sink_diagnostics = self._telemetry_sink.get_diagnostics()
        except Exception as exc:
            self._disable_telemetry_sink("append", exc)

    def _flush_telemetry_sink(self, *, force: bool = False) -> None:
        if self._telemetry_sink is None:
            return
        try:
            self._telemetry_sink.flush(force=force)
            self._last_sink_diagnostics = self._telemetry_sink.get_diagnostics()
        except Exception as exc:
            self._disable_telemetry_sink("flush", exc)

    def _close_telemetry_sink(self) -> None:
        if self._telemetry_sink is None:
            return
        try:
            self._close_sink_with_status(
                self._telemetry_sink,
                SESSION_STATUS_COMPLETED,
            )
            self._last_sink_diagnostics = self._telemetry_sink.get_diagnostics()
        except Exception as exc:
            self._disable_telemetry_sink("close", exc)
        else:
            self._telemetry_sink = None

    def _disable_telemetry_sink(self, operation: str, exc: Exception) -> None:
        sink = self._telemetry_sink
        if sink is None:
            return
        self._telemetry_sink = None
        logger.warning(
            "Disabling CPU telemetry sink after %s failure: %s",
            operation,
            exc,
        )
        if self._session_summary is not None:
            self._session_summary = update_session_summary(
                self._session_summary,
                status=SESSION_STATUS_INCOMPLETE,
                ended_at_ns=now_ns(),
            )
        try:
            self._close_sink_with_status(sink, SESSION_STATUS_INCOMPLETE)
            if hasattr(sink, "get_diagnostics"):
                self._last_sink_diagnostics = sink.get_diagnostics()
        except Exception as close_exc:
            logger.debug(
                "CPU telemetry sink close failed after %s error: %s",
                operation,
                close_exc,
            )

    @staticmethod
    def _close_sink_with_status(sink: Any, status: str) -> None:
        try:
            sink.close(session_status=status)
        except TypeError:
            sink.close()

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
        with self._events_lock:
            retained_events = len(self.events)
        return {
            "mode": "cpu",
            "total_events": total_events,
            "peak_memory": peak_memory,
            "current_memory_allocated": rss,
            "tracking_duration_seconds": duration,
            "history_window_limit_events": self.max_events,
            "history_retained_events": retained_events,
            "history_dropped_events": self._history_dropped_events,
            **self._last_sink_diagnostics,
            "session_id": (
                self._session_summary.session_id
                if self._session_summary is not None
                else None
            ),
            "session_status": (
                self._session_summary.status
                if self._session_summary is not None
                else None
            ),
        }

    def get_memory_timeline(self, interval: float = 1.0) -> Dict[str, List[float]]:
        if interval <= 0:
            raise ValueError("interval must be > 0")

        with self._events_lock:
            events_snapshot = list(self.events)

        if not events_snapshot:
            return {"timestamps": [], "allocated": [], "reserved": []}

        start_time = events_snapshot[0].timestamp
        timestamps: List[float] = []
        allocated: List[float] = []
        reserved: List[float] = []

        current_bucket = 0
        last_event = events_snapshot[0]
        for event in events_snapshot[1:]:
            bucket = int((event.timestamp - start_time) // interval)
            if bucket == current_bucket:
                last_event = event
                continue

            timestamps.append(start_time + current_bucket * interval)
            allocated.append(float(last_event.memory_allocated or 0))
            reserved.append(float(last_event.memory_reserved or 0))
            current_bucket = bucket
            last_event = event

        timestamps.append(start_time + current_bucket * interval)
        allocated.append(float(last_event.memory_allocated or 0))
        reserved.append(float(last_event.memory_reserved or 0))

        return {
            "timestamps": timestamps,
            "allocated": allocated,
            "reserved": reserved,
        }

    def clear_events(self) -> None:
        with self._events_lock:
            self.events.clear()
            self.stats["peak_memory"] = 0
            self.stats["total_events"] = 0
            self._history_dropped_events = 0

    def export_events(self, filename: str, format: str = "csv") -> None:
        with self._events_lock:
            events_snapshot = list(self.events)

        records = [
            self._telemetry_record_from_event(event) for event in events_snapshot
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
