"""Helpers for bridging MemoryTracker data into the Textual TUI."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from stormlog.session import SessionSummary
from stormlog.telemetry import (
    ProjectedTelemetryRecord,
    TelemetryEvent,
    project_telemetry_events,
    telemetry_event_from_record,
)
from stormlog.telemetry_sink import TelemetrySinkConfig

from ..utils import format_bytes

logger = logging.getLogger(__name__)

try:
    from stormlog.tracker import MemoryTracker as _MemoryTracker
    from stormlog.tracker import MemoryWatchdog as _MemoryWatchdog
    from stormlog.tracker import TrackingEvent as _TrackingEvent

    MemoryTracker: Any = _MemoryTracker
    MemoryWatchdog: Any = _MemoryWatchdog
    TrackingEvent: Any = _TrackingEvent
except ImportError as exc:
    logger.debug("GPU tracker imports unavailable: %s", exc)
    MemoryTracker = None
    MemoryWatchdog = None
    TrackingEvent = None

try:
    from stormlog.cpu_profiler import CPUMemoryTracker as _CPUMemoryTracker

    CPUMemoryTracker: Any = _CPUMemoryTracker
except ImportError as exc:
    logger.debug("CPU tracker imports unavailable: %s", exc)
    CPUMemoryTracker = None

try:
    import torch as _torch

    torch: Any = _torch
except ImportError:
    torch = None


class TrackerUnavailableError(RuntimeError):
    """Raised when the MemoryTracker backend cannot be initialized."""


@dataclass
class TrackerEventView:
    """Lightweight view model for displaying tracking events."""

    timestamp: float
    event_type: str
    message: str
    allocated: str
    reserved: str
    change: str


class TrackerSession:
    """Stateful adapter that exposes MemoryTracker data (GPU or CPU) to the TUI."""

    def __init__(
        self,
        *,
        sampling_interval: float = 0.5,
        auto_cleanup: bool = False,
        max_events_per_poll: int = 50,
        max_events: int = 10_000,
        telemetry_sink_config: Optional[TelemetrySinkConfig] = None,
    ) -> None:
        # Defensive check: ensure at least one tracker is available
        # (In normal operation, imports are required and will raise ImportError if missing.
        # This check handles edge cases like testing scenarios where trackers are monkeypatched.)
        if MemoryTracker is None and CPUMemoryTracker is None:
            raise TrackerUnavailableError(
                "Memory trackers are unavailable. Install torch with CUDA for GPU mode "
                "or ensure the CPU tracker dependencies are installed."
            )
        self.sampling_interval = sampling_interval
        self.auto_cleanup = auto_cleanup
        self.max_events_per_poll = max_events_per_poll
        self.max_events = max_events
        self.telemetry_sink_config = telemetry_sink_config
        self._tracker: Optional[Any] = None
        self._watchdog: Optional[Any] = None
        self._last_seen_ts: Optional[float] = None
        self.backend = "gpu"
        self._cpu_thresholds = {
            "memory_warning_percent": 80.0,
            "memory_critical_percent": 95.0,
        }

    @property
    def is_active(self) -> bool:
        return bool(self._tracker and self._tracker.is_tracking)

    def start(self, **tracker_kwargs: Any) -> None:
        """Start the underlying MemoryTracker."""
        if self.is_active:
            return

        tracker_kwargs.setdefault("sampling_interval", self.sampling_interval)
        tracker_kwargs.setdefault("enable_alerts", True)
        tracker_kwargs.setdefault("telemetry_sink_config", self.telemetry_sink_config)

        backend = "gpu"

        tracker = self._try_gpu_tracker(tracker_kwargs)

        if tracker is None and CPUMemoryTracker is not None:
            backend = "cpu"
            tracker = CPUMemoryTracker(
                sampling_interval=tracker_kwargs["sampling_interval"],
                max_events=self.max_events,
                enable_alerts=tracker_kwargs.get("enable_alerts", True),
                telemetry_sink_config=tracker_kwargs.get("telemetry_sink_config"),
            )

        if tracker is None:
            raise TrackerUnavailableError(
                "Memory trackers are unavailable in this environment."
            )

        tracker.start_tracking()
        self._tracker = tracker
        self._last_seen_ts = None
        self.backend = backend

        if backend == "gpu" and MemoryWatchdog is not None:
            self._watchdog = MemoryWatchdog(tracker, auto_cleanup=self.auto_cleanup)
        else:
            self._watchdog = None

    @staticmethod
    def _try_gpu_tracker(tracker_kwargs: dict[str, Any]) -> Optional[Any]:
        tracker: Optional[Any] = None
        # Try GPU tracker first, fall back to CPU tracker if initialization fails
        if MemoryTracker is not None and torch is not None:
            try:
                tracker = MemoryTracker(**tracker_kwargs)
            except Exception as exc:
                logger.debug(
                    "GPU MemoryTracker init failed, falling back to CPU: %s", exc
                )
        elif MemoryTracker is None:
            logger.debug("GPU MemoryTracker import unavailable, falling back to CPU.")
        else:
            logger.debug("torch is unavailable, falling back to CPU tracking.")

        return tracker

    def stop(self) -> None:
        """Stop tracking and release state."""
        if not self._tracker:
            return

        self._tracker.stop_tracking()
        self._tracker = None
        self._watchdog = None
        self._last_seen_ts = None

    def pull_events(self) -> List[TrackerEventView]:
        """Return newly observed events as lightweight view models."""
        tracker = self._tracker
        if not tracker:
            return []

        since = self._last_seen_ts + 1e-6 if self._last_seen_ts else None
        raw_events = tracker.get_events(since=since)

        if not raw_events:
            return []

        self._last_seen_ts = raw_events[-1].timestamp
        recent_events = (
            raw_events[-self.max_events_per_poll :]
            if self.max_events_per_poll
            else raw_events
        )

        views: List[TrackerEventView] = []
        for event in recent_events:
            views.append(
                TrackerEventView(
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    message=event.context or "",
                    allocated=(
                        format_bytes(event.memory_allocated)
                        if event.memory_allocated is not None
                        else "N/A"
                    ),
                    reserved=(
                        format_bytes(event.memory_reserved)
                        if event.memory_reserved is not None
                        else "N/A"
                    ),
                    change=(
                        format_bytes(event.memory_change)
                        if event.memory_change is not None
                        else "N/A"
                    ),
                )
            )
        return views

    def get_statistics(self) -> Dict[str, Any]:
        """Expose the current tracker statistics."""
        if not self._tracker:
            return {}
        return cast(Dict[str, Any], self._tracker.get_statistics())

    def get_memory_timeline(self, interval: float = 1.0) -> Dict[str, Any]:
        """Return aggregated timeline data from the tracker."""
        if not self._tracker:
            return {}
        return cast(
            Dict[str, Any], self._tracker.get_memory_timeline(interval=interval)
        )

    def get_telemetry_events(self) -> list[TelemetryEvent]:
        """Return normalized telemetry events from the active tracker session."""
        tracker = self._tracker
        if not tracker:
            return []

        sampling_interval_ms = max(0, int(round(self.sampling_interval * 1000)))
        host = socket.gethostname()
        pid = os.getpid()

        backend_name = str(getattr(tracker, "backend", self.backend)).lower()
        collector = self._tracker_collector_name(backend_name)

        raw_events = self._read_tracker_events(tracker)

        normalized: list[TelemetryEvent] = []
        for raw_event in raw_events:
            canonical_event = self._canonical_tracker_event(tracker, raw_event)
            if canonical_event is not None:
                normalized.append(canonical_event)
                continue
            timestamp = getattr(raw_event, "timestamp", None)
            if timestamp is None:
                continue
            event_type = str(getattr(raw_event, "event_type", "sample"))

            allocated = int(getattr(raw_event, "memory_allocated", 0))
            reserved = int(getattr(raw_event, "memory_reserved", allocated))
            memory_change = int(getattr(raw_event, "memory_change", 0))
            device_used = getattr(raw_event, "device_used", None)
            if device_used is None:
                device_used = max(allocated, reserved)

            metadata = dict(getattr(raw_event, "metadata", {}) or {})
            metadata.setdefault("backend", backend_name)
            partial_fields = set(metadata.get("collector_partial_fields", []) or [])
            session_id = self._event_session_id(raw_event)

            device_total = self._event_device_total(raw_event, tracker, partial_fields)

            record = {
                "session_id": session_id,
                "timestamp": float(timestamp),
                "event_type": event_type,
                "collector": collector,
                "sampling_interval_ms": sampling_interval_ms,
                "pid": int(pid),
                "host": host,
                "device_id": int(getattr(raw_event, "device_id", -1)),
                "memory_allocated": allocated,
                "memory_reserved": reserved,
                "memory_change": memory_change,
                "allocator_active_bytes": getattr(raw_event, "active_memory", None),
                "allocator_inactive_bytes": getattr(raw_event, "inactive_memory", None),
                "device_used_bytes": int(device_used),
                "device_free_bytes": getattr(raw_event, "device_free", None),
                "device_total_bytes": device_total,
                "context": getattr(raw_event, "context", None),
                "job_id": getattr(raw_event, "job_id", None),
                "rank": int(getattr(raw_event, "rank", 0)),
                "local_rank": int(getattr(raw_event, "local_rank", 0)),
                "world_size": int(getattr(raw_event, "world_size", 1)),
                "metadata": metadata,
            }
            try:
                normalized.append(
                    telemetry_event_from_record(
                        record,
                        permissive_legacy=True,
                        default_collector=collector,
                        default_sampling_interval_ms=sampling_interval_ms,
                        default_session_id=session_id,
                    )
                )
            except Exception as exc:
                logger.debug(
                    "TrackerSession.get_telemetry_events dropped malformed event: %s",
                    exc,
                )

        return normalized

    @staticmethod
    def _tracker_collector_name(backend_name: str) -> str:
        collector = f"stormlog.{backend_name}_tracker"
        if backend_name == "gpu":
            collector = "stormlog.cuda_tracker"
        elif backend_name == "cpu":
            collector = "stormlog.cpu_tracker"

        return collector

    @staticmethod
    def _canonical_tracker_event(tracker: Any, raw_event: Any) -> TelemetryEvent | None:
        if hasattr(tracker, "_telemetry_record_from_event"):
            try:
                record = tracker._telemetry_record_from_event(raw_event)
                return telemetry_event_from_record(record)
            except Exception as exc:
                logger.debug(
                    "TrackerSession canonical event conversion failed: %s", exc
                )
        return None

    def _event_session_id(self, raw_event: Any) -> Optional[str]:
        session_id = getattr(raw_event, "session_id", None)
        if session_id is None:
            session_summary = self.get_session_summary()
            session_id = (
                session_summary.session_id if session_summary is not None else None
            )

        return cast(Optional[str], session_id)

    @staticmethod
    def _event_device_total(
        raw_event: Any, tracker: Any, partial_fields: set[str]
    ) -> Any:
        device_total = getattr(raw_event, "device_total", None)
        if device_total is None and "device_total_bytes" not in partial_fields:
            tracker_total = getattr(tracker, "total_memory", None)
            device_total = int(tracker_total) if tracker_total is not None else None

        return device_total

    @staticmethod
    def _read_tracker_events(tracker: Any) -> list[Any]:
        raw_events = []
        if hasattr(tracker, "get_events"):
            try:
                raw_events = list(tracker.get_events())
            except Exception as exc:
                logger.debug(
                    "TrackerSession.get_telemetry_events get_events failed: %s", exc
                )
                raw_events = []
        elif hasattr(tracker, "events"):
            raw_events = list(getattr(tracker, "events", []))

        return raw_events

    def telemetry_records(self) -> list[ProjectedTelemetryRecord]:
        """Return backend-neutral projected telemetry records from the live tracker."""

        return project_telemetry_events(self.get_telemetry_events())

    def get_session_summary(self) -> Optional[SessionSummary]:
        """Return the underlying tracker session summary when available."""
        tracker = self._tracker
        if tracker is None or not hasattr(tracker, "get_session_summary"):
            return None
        try:
            summary = tracker.get_session_summary()
        except Exception as exc:
            logger.debug(
                "TrackerSession.get_session_summary failed: %s",
                exc,
            )
            return None
        return cast(Optional[SessionSummary], summary)

    def get_device_label(self) -> Optional[str]:
        """Return the CUDA device label, if tracking."""
        if not self._tracker:
            return None
        if self.backend == "cpu":
            return "cpu"
        return str(getattr(self._tracker, "device", "cuda"))

    def clear_events(self) -> None:
        """Clear buffered events and reset the poll cursor."""
        if not self._tracker:
            return
        self._tracker.clear_events()
        self._last_seen_ts = None

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Return watchdog cleanup stats if available."""
        if not self._watchdog:
            return {}
        return cast(Dict[str, Any], self._watchdog.get_cleanup_stats())

    def set_auto_cleanup(self, enabled: bool) -> None:
        """Toggle automatic watchdog cleanup."""
        self.auto_cleanup = enabled
        if self._watchdog:
            self._watchdog.auto_cleanup = enabled

    def force_cleanup(self, aggressive: bool = False) -> bool:
        """Trigger a watchdog cleanup run."""
        if not self._watchdog:
            return False
        self._watchdog.force_cleanup(aggressive=aggressive)
        return True

    def export_events(self, file_path: str, format: str = "csv") -> bool:
        """Export tracker events to a file."""
        tracker = self._tracker
        if not tracker or not tracker.events:
            return False
        tracker.export_events(file_path, format=format)
        return True

    def get_thresholds(self) -> Dict[str, float]:
        if (
            self.backend == "gpu"
            and self._tracker
            and hasattr(self._tracker, "thresholds")
        ):
            thresholds = self._tracker.thresholds
            return {
                "memory_warning_percent": thresholds.get(
                    "memory_warning_percent", 80.0
                ),
                "memory_critical_percent": thresholds.get(
                    "memory_critical_percent", 95.0
                ),
            }
        return self._cpu_thresholds.copy()

    def set_thresholds(self, warning: float, critical: float) -> None:
        if (
            self.backend == "gpu"
            and self._tracker
            and hasattr(self._tracker, "set_threshold")
        ):
            self._tracker.set_threshold("memory_warning_percent", warning)
            self._tracker.set_threshold("memory_critical_percent", critical)
        else:
            self._cpu_thresholds["memory_warning_percent"] = warning
            self._cpu_thresholds["memory_critical_percent"] = critical
