"""Helpers for bridging MemoryTracker data into the Textual TUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from ..utils import format_bytes

logger = logging.getLogger(__name__)

try:
    from gpumemprof.tracker import MemoryTracker as _MemoryTracker
    from gpumemprof.tracker import MemoryWatchdog as _MemoryWatchdog
    from gpumemprof.tracker import TrackingEvent as _TrackingEvent

    MemoryTracker: Any = _MemoryTracker
    MemoryWatchdog: Any = _MemoryWatchdog
    TrackingEvent: Any = _TrackingEvent
except ImportError as e:
    raise ImportError(
        "gpumemprof.tracker is required for TrackerSession. "
        "Ensure gpumemprof is properly installed."
    ) from e

try:
    from gpumemprof.cpu_profiler import CPUMemoryTracker as _CPUMemoryTracker

    CPUMemoryTracker: Any = _CPUMemoryTracker
except ImportError as e:
    raise ImportError(
        "CPUMemoryTracker is required for TrackerSession. "
        "Ensure gpumemprof is properly installed."
    ) from e

try:
    import torch as _torch

    torch: Any = _torch
except ImportError as e:
    raise ImportError(
        "torch is required for TrackerSession. Install it with: pip install torch"
    ) from e


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

        tracker: Optional[Any] = None
        backend = "gpu"

        # Try GPU tracker first, fall back to CPU tracker if initialization fails
        try:
            tracker = MemoryTracker(**tracker_kwargs)
        except Exception as exc:
            logger.debug("GPU MemoryTracker init failed, falling back to CPU: %s", exc)
            backend = "cpu"
            tracker = CPUMemoryTracker(
                sampling_interval=tracker_kwargs["sampling_interval"],
                max_events=self.max_events,
                enable_alerts=tracker_kwargs.get("enable_alerts", True),
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
                    allocated=format_bytes(event.memory_allocated),
                    reserved=format_bytes(event.memory_reserved),
                    change=format_bytes(event.memory_change),
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
