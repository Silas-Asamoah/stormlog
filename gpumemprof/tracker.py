"""Real-time memory tracking and monitoring."""

import logging
import os
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from .device_collectors import (
    DeviceMemorySample,
    _resolve_device,
    build_device_memory_collector,
    detect_torch_runtime_backend,
)
from .oom_flight_recorder import (
    OOMFlightRecorder,
    OOMFlightRecorderConfig,
    classify_oom_exception,
)
from .telemetry import (
    resolve_distributed_identity,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from .utils import format_bytes, get_gpu_info

logger = logging.getLogger(__name__)


@dataclass
class TrackingEvent:
    """Represents a memory tracking event."""

    timestamp: float
    event_type: str  # 'allocation', 'deallocation', 'peak', 'warning', 'error'
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
    backend: str = "cuda"


class MemoryTracker:
    """Real-time memory tracker with alerts and monitoring."""

    def __init__(
        self,
        device: Optional[Union[str, int, torch.device]] = None,
        sampling_interval: float = 0.1,
        max_events: int = 10000,
        enable_alerts: bool = True,
        enable_oom_flight_recorder: bool = False,
        oom_dump_dir: str = "oom_dumps",
        oom_buffer_size: Optional[int] = None,
        oom_max_dumps: int = 5,
        oom_max_total_mb: int = 256,
        job_id: Optional[str] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """
        Initialize the memory tracker.

        Args:
            device: GPU device to track
            sampling_interval: Sampling interval in seconds
            max_events: Maximum number of events to keep in memory
            enable_alerts: Whether to enable memory alerts
            enable_oom_flight_recorder: Enable automatic OOM dump artifacts
            oom_dump_dir: Directory used for OOM dump bundles
            oom_buffer_size: Event ring-buffer size used for OOM dumps
            oom_max_dumps: Maximum number of retained OOM dump bundles
            oom_max_total_mb: Maximum retained OOM dump storage in MB
        """
        self.device = self._setup_device(device)
        self.collector = build_device_memory_collector(self.device)
        self.backend = self.collector.name()
        self.collector_capabilities = self.collector.capabilities()
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.enable_alerts = enable_alerts
        self.last_oom_dump_path: Optional[str] = None
        self.distributed_identity = resolve_distributed_identity(
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            env=os.environ,
        )

        recorder_buffer_size = (
            oom_buffer_size if oom_buffer_size is not None else max_events
        )
        if recorder_buffer_size <= 0:
            recorder_buffer_size = max_events
        self._oom_flight_recorder = OOMFlightRecorder(
            OOMFlightRecorderConfig(
                enabled=enable_oom_flight_recorder,
                dump_dir=oom_dump_dir,
                buffer_size=recorder_buffer_size,
                max_dumps=oom_max_dumps,
                max_total_mb=oom_max_total_mb,
            )
        )

        # Tracking state
        self.events: deque[TrackingEvent] = deque(maxlen=max_events)
        self.is_tracking = False
        self._tracking_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Memory thresholds for alerts
        self.thresholds: Dict[str, float] = {
            "memory_warning_percent": 80.0,  # Warn at 80% memory usage
            "memory_critical_percent": 95.0,  # Critical at 95% memory usage
            "memory_leak_threshold": float(100 * 1024 * 1024),  # 100MB growth
            "fragmentation_threshold": 0.3,  # 30% fragmentation
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable[[TrackingEvent], None]] = []

        # Statistics
        self.stats: Dict[str, Any] = {
            "peak_memory": 0,
            "total_allocations": 0,
            "total_deallocations": 0,
            "total_allocation_bytes": 0,
            "total_deallocation_bytes": 0,
            "alert_count": 0,
            "tracking_start_time": None,
            "last_memory_check": 0,
        }

        # Get memory limits with backend-aware fallback.
        self.gpu_info = get_gpu_info(self.device) if self.device.type == "cuda" else {}
        initial_sample = self._safe_sample()
        total_memory = initial_sample.total_bytes
        if total_memory is None:
            fallback_total = self.gpu_info.get("total_memory", 0)
            total_memory = (
                int(fallback_total) if isinstance(fallback_total, (int, float)) else 0
            )
        self.total_memory = int(total_memory)

    @property
    def oom_buffer_size(self) -> int:
        """Resolved OOM ring-buffer size."""
        return self._oom_flight_recorder.config.buffer_size

    def _setup_device(
        self, device: Union[str, int, torch.device, None]
    ) -> torch.device:
        """Setup and validate the device for tracking."""
        resolved_device = _resolve_device(device)

        if resolved_device.type not in {"cuda", "mps"}:
            raise ValueError(
                "Only CUDA/ROCm or MPS devices are supported for GPU memory tracking"
            )

        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm backend is not available in this runtime")
            device_index = (
                resolved_device.index
                if resolved_device.index is not None
                else torch.cuda.current_device()
            )
            if device_index >= torch.cuda.device_count():
                raise ValueError(f"Device {resolved_device} is not available")
            return torch.device(f"cuda:{device_index}")
        if detect_torch_runtime_backend() != "mps":
            raise RuntimeError("MPS backend is not available in this runtime")

        return resolved_device

    def _safe_sample(self) -> DeviceMemorySample:
        """Collect one backend sample with defensive fallback values."""
        try:
            return self.collector.sample()
        except Exception as exc:
            logger.debug("Could not sample %s memory: %s", self.backend, exc)
            device_id = 0
            if self.device.type == "cuda":
                try:
                    device_id = (
                        self.device.index
                        if self.device.index is not None
                        else torch.cuda.current_device()
                    )
                except Exception:
                    device_id = 0
            return DeviceMemorySample(
                allocated_bytes=0,
                reserved_bytes=0,
                used_bytes=0,
                free_bytes=None,
                total_bytes=None,
                active_bytes=None,
                inactive_bytes=None,
                device_id=device_id,
            )

    def start_tracking(self) -> None:
        """Start real-time memory tracking."""
        if self.is_tracking:
            return

        self.is_tracking = True
        self._stop_event.clear()
        self.stats["tracking_start_time"] = time.time()

        self._tracking_thread = threading.Thread(target=self._tracking_loop)
        self._tracking_thread.daemon = True
        self._tracking_thread.start()

        # Add initial event
        self._add_event("start", 0, "Memory tracking started")

    def stop_tracking(self) -> None:
        """Stop real-time memory tracking."""
        if not self.is_tracking:
            return

        self.is_tracking = False
        self._stop_event.set()

        if self._tracking_thread:
            self._tracking_thread.join(timeout=1.0)

        # Add final event
        self._add_event("stop", 0, "Memory tracking stopped")

    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread."""
        last_allocated = 0

        while not self._stop_event.wait(self.sampling_interval):
            try:
                # Get current memory usage
                sample = self._safe_sample()
                current_allocated = sample.allocated_bytes
                current_reserved = sample.reserved_bytes

                # Calculate change
                memory_change = current_allocated - last_allocated

                # Update statistics
                self.stats["last_memory_check"] = time.time()
                if current_allocated > self.stats["peak_memory"]:
                    self.stats["peak_memory"] = current_allocated
                    self._add_event(
                        "peak",
                        memory_change,
                        f"New peak memory: {format_bytes(current_allocated)}",
                        sample=sample,
                    )

                # Track allocations/deallocations
                if memory_change > 0:
                    self.stats["total_allocations"] += 1
                    self.stats["total_allocation_bytes"] += memory_change
                    self._add_event(
                        "allocation",
                        memory_change,
                        f"Memory allocated: {format_bytes(memory_change)}",
                        sample=sample,
                    )
                elif memory_change < 0:
                    self.stats["total_deallocations"] += 1
                    self.stats["total_deallocation_bytes"] += abs(memory_change)
                    self._add_event(
                        "deallocation",
                        memory_change,
                        f"Memory freed: {format_bytes(abs(memory_change))}",
                        sample=sample,
                    )

                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts(
                        current_allocated, current_reserved, memory_change
                    )

                last_allocated = current_allocated

            except Exception as e:
                self._add_event("error", 0, f"Tracking error: {str(e)}")
                time.sleep(1.0)  # Back off on errors

    def _add_event(
        self,
        event_type: str,
        memory_change: int,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
        sample: Optional[DeviceMemorySample] = None,
    ) -> None:
        """Add a tracking event."""
        snapshot = sample if sample is not None else self._safe_sample()
        current_allocated = snapshot.allocated_bytes
        current_reserved = snapshot.reserved_bytes

        event = TrackingEvent(
            timestamp=time.time(),
            event_type=event_type,
            memory_allocated=current_allocated,
            memory_reserved=current_reserved,
            memory_change=memory_change,
            device_id=snapshot.device_id,
            context=context,
            job_id=self.distributed_identity["job_id"],
            rank=self.distributed_identity["rank"],
            local_rank=self.distributed_identity["local_rank"],
            world_size=self.distributed_identity["world_size"],
            metadata=metadata,
            active_memory=snapshot.active_bytes,
            inactive_memory=snapshot.inactive_bytes,
            device_used=snapshot.used_bytes,
            device_free=snapshot.free_bytes,
            device_total=snapshot.total_bytes,
            backend=self.backend,
        )

        self.events.append(event)
        self._oom_flight_recorder.record_event(self._tracking_event_payload(event))

        # Trigger callbacks for alerts
        if event_type in ["warning", "critical", "error"]:
            self.stats["alert_count"] += 1
            for callback in self.alert_callbacks:
                try:
                    callback(event)
                except Exception as exc:
                    logger.debug("Alert callback error (suppressed): %s", exc)

    def _check_alerts(self, allocated: int, reserved: int, change: int) -> None:
        """Check for memory alerts and warnings."""
        if self.total_memory == 0:
            return

        # Memory usage percentage
        usage_percent = (allocated / self.total_memory) * 100

        # Critical memory usage
        if usage_percent >= self.thresholds["memory_critical_percent"]:
            self._add_event(
                "critical",
                change,
                f"CRITICAL: Memory usage at {usage_percent:.1f}%",
                {"usage_percent": usage_percent},
            )

        # Warning memory usage
        elif usage_percent >= self.thresholds["memory_warning_percent"]:
            self._add_event(
                "warning",
                change,
                f"WARNING: Memory usage at {usage_percent:.1f}%",
                {"usage_percent": usage_percent},
            )

        # Large allocation warning
        if change > self.thresholds["memory_leak_threshold"]:
            self._add_event(
                "warning",
                change,
                f"Large allocation detected: {format_bytes(change)}",
                {"large_allocation": True},
            )

        # Fragmentation warning
        if reserved > 0:
            fragmentation = (reserved - allocated) / reserved
            if fragmentation > self.thresholds["fragmentation_threshold"]:
                self._add_event(
                    "warning",
                    change,
                    f"High fragmentation: {fragmentation:.1%}",
                    {"fragmentation": fragmentation},
                )

    @staticmethod
    def _tracking_event_payload(event: TrackingEvent) -> Dict[str, Any]:
        """Serialize a TrackingEvent into a stable JSON-safe payload."""
        return {
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
            "metadata": dict(event.metadata or {}),
            "active_memory": event.active_memory,
            "inactive_memory": event.inactive_memory,
            "device_used": event.device_used,
            "device_free": event.device_free,
            "device_total": event.device_total,
            "backend": event.backend,
        }

    def handle_exception(
        self,
        exc: BaseException,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Capture OOM diagnostics for recognized OOM exceptions."""
        classification = classify_oom_exception(exc)
        if not classification.is_oom or classification.reason is None:
            return None
        if not self._oom_flight_recorder.config.enabled:
            return None

        dump_metadata: Dict[str, Any] = {
            "tracker_stats": self.get_statistics(),
            "collector_capabilities": dict(self.collector_capabilities),
            "total_memory_bytes": self.total_memory,
            "sampling_interval_s": self.sampling_interval,
            "job_id": self.distributed_identity["job_id"],
            "rank": self.distributed_identity["rank"],
            "local_rank": self.distributed_identity["local_rank"],
            "world_size": self.distributed_identity["world_size"],
        }
        if metadata:
            dump_metadata.update(metadata)

        sample = self._safe_sample()
        dump_metadata.update(
            {
                "sample_allocated_bytes": sample.allocated_bytes,
                "sample_reserved_bytes": sample.reserved_bytes,
                "sample_used_bytes": sample.used_bytes,
                "sample_free_bytes": sample.free_bytes,
                "sample_total_bytes": sample.total_bytes,
                "sample_device_id": sample.device_id,
            }
        )
        self._add_event(
            "error",
            0,
            f"OOM detected ({classification.reason})",
            metadata={"oom_reason": classification.reason},
            sample=sample,
        )

        try:
            dump_path = self._oom_flight_recorder.dump(
                reason=classification.reason,
                exception=exc,
                context=context,
                backend=self.backend,
                metadata=dump_metadata,
            )
        except Exception as dump_exc:
            logger.debug("OOM flight recorder dump failed: %s", dump_exc)
            return None

        self.last_oom_dump_path = dump_path
        return dump_path

    @contextmanager
    def capture_oom(
        self,
        context: str = "runtime",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Capture OOM diagnostic bundle if a tracked block raises OOM."""
        try:
            yield
        except Exception as exc:
            dump_path = self.handle_exception(exc, context=context, metadata=metadata)
            if dump_path:
                logger.error("OOM flight recorder dump saved to: %s", dump_path)
            raise

    def add_alert_callback(self, callback: Callable[[TrackingEvent], None]) -> None:
        """Add a callback function to be called on alerts."""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[TrackingEvent], None]) -> None:
        """Remove an alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

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
        events = list(self.events)

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

    def get_memory_timeline(self, interval: float = 1.0) -> Dict[str, List]:
        """
        Get memory usage timeline with specified interval.

        Args:
            interval: Time interval in seconds for aggregation

        Returns:
            Dictionary with timeline data
        """
        if not self.events:
            return {"timestamps": [], "allocated": [], "reserved": []}

        # Group events by time intervals
        start_time = self.events[0].timestamp
        end_time = self.events[-1].timestamp

        timestamps = []
        allocated_values = []
        reserved_values = []

        current_time = start_time
        while current_time <= end_time:
            # Find events in this interval
            interval_events = [
                e
                for e in self.events
                if current_time <= e.timestamp < current_time + interval
            ]

            if interval_events:
                # Use the last event in the interval
                last_event = interval_events[-1]
                timestamps.append(current_time)
                allocated_values.append(last_event.memory_allocated)
                reserved_values.append(last_event.memory_reserved)

            current_time += interval

        return {
            "timestamps": timestamps,
            "allocated": allocated_values,
            "reserved": reserved_values,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        current_stats = self.stats.copy()

        if self.events:
            # Calculate additional statistics
            recent_events = [
                e for e in self.events if e.timestamp > time.time() - 3600
            ]  # Last hour
            sample = self._safe_sample()

            current_stats.update(
                {
                    "total_events": len(self.events),
                    "events_last_hour": len(recent_events),
                    "backend": self.backend,
                    "oom_flight_recorder_enabled": self._oom_flight_recorder.config.enabled,
                    "last_oom_dump_path": self.last_oom_dump_path,
                    "current_memory_allocated": sample.allocated_bytes,
                    "current_memory_reserved": sample.reserved_bytes,
                    "memory_utilization_percent": (
                        (sample.used_bytes / self.total_memory * 100)
                        if self.total_memory > 0
                        else 0
                    ),
                    "average_allocation_size": self.stats["total_allocation_bytes"]
                    / max(self.stats["total_allocations"], 1),
                    "average_deallocation_size": self.stats["total_deallocation_bytes"]
                    / max(self.stats["total_deallocations"], 1),
                }
            )

            # Time-based statistics
            if self.stats["tracking_start_time"]:
                tracking_duration = time.time() - self.stats["tracking_start_time"]
                current_stats.update(
                    {
                        "tracking_duration_seconds": tracking_duration,
                        "allocations_per_second": self.stats["total_allocations"]
                        / max(tracking_duration, 1),
                        "bytes_allocated_per_second": self.stats[
                            "total_allocation_bytes"
                        ]
                        / max(tracking_duration, 1),
                    }
                )

        return current_stats

    def export_events(self, filename: str, format: str = "csv") -> None:
        """
        Export tracking events to file.

        Args:
            filename: Output filename
            format: Export format ('csv' or 'json')
        """
        import json

        import pandas as pd

        if not self.events:
            return

        host = socket.gethostname()
        pid = os.getpid()
        sampling_interval_ms = int(round(self.sampling_interval * 1000))
        default_collector = str(
            self.collector_capabilities.get(
                "telemetry_collector", "gpumemprof.cuda_tracker"
            )
        )
        capability_metadata = {
            "backend": self.backend,
            "supports_device_total": bool(
                self.collector_capabilities.get("supports_device_total", False)
            ),
            "supports_device_free": bool(
                self.collector_capabilities.get("supports_device_free", False)
            ),
            "sampling_source": str(
                self.collector_capabilities.get("sampling_source", "unknown")
            ),
        }

        # Convert events to canonical telemetry records.
        records = []
        for event in self.events:
            metadata = dict(event.metadata or {})
            metadata.update(capability_metadata)
            device_used = event.device_used
            if device_used is None:
                device_used = max(event.memory_allocated, event.memory_reserved)
            event_total = (
                event.device_total
                if event.device_total is not None
                else (self.total_memory or None)
            )
            legacy = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "memory_allocated": event.memory_allocated,
                "memory_reserved": event.memory_reserved,
                "memory_change": event.memory_change,
                "allocator_active_bytes": event.active_memory,
                "allocator_inactive_bytes": event.inactive_memory,
                "device_used_bytes": device_used,
                "device_free_bytes": event.device_free,
                "device_total_bytes": event_total,
                "device_id": event.device_id,
                "context": event.context,
                "job_id": event.job_id,
                "rank": event.rank,
                "local_rank": event.local_rank,
                "world_size": event.world_size,
                "metadata": metadata,
                "total_memory": event_total,
                "pid": pid,
                "host": host,
                "collector": default_collector,
                "sampling_interval_ms": sampling_interval_ms,
            }
            telemetry_event = telemetry_event_from_record(
                legacy,
                default_collector=default_collector,
                default_sampling_interval_ms=sampling_interval_ms,
            )
            records.append(telemetry_event_to_dict(telemetry_event))

        if format == "csv":
            df = pd.DataFrame(records)
            df.to_csv(filename, index=False)
        elif format == "json":
            with open(filename, "w") as f:
                json.dump(records, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_events(self) -> None:
        """Clear all tracking events."""
        self.events.clear()

        # Reset statistics
        self.stats.update(
            {
                "peak_memory": 0,
                "total_allocations": 0,
                "total_deallocations": 0,
                "total_allocation_bytes": 0,
                "total_deallocation_bytes": 0,
                "alert_count": 0,
            }
        )

    def set_threshold(self, threshold_name: str, value: Union[int, float]) -> None:
        """
        Set alert threshold.

        Args:
            threshold_name: Name of the threshold
            value: Threshold value
        """
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
        else:
            raise ValueError(f"Unknown threshold: {threshold_name}")

    def get_alerts(self, last_n: Optional[int] = None) -> List[TrackingEvent]:
        """Get all alert events (warnings, critical, errors)."""
        alert_types = ["warning", "critical", "error"]
        alerts = [e for e in self.events if e.event_type in alert_types]

        if last_n:
            alerts = alerts[-last_n:]

        return alerts

    def __enter__(self) -> "MemoryTracker":
        """Context manager entry."""
        self.start_tracking()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_tracking()


class MemoryWatchdog:
    """Memory watchdog for automated memory management."""

    def __init__(
        self,
        tracker: MemoryTracker,
        auto_cleanup: bool = True,
        cleanup_threshold: float = 0.9,
        aggressive_cleanup_threshold: float = 0.95,
    ):
        """
        Initialize memory watchdog.

        Args:
            tracker: MemoryTracker instance to monitor
            auto_cleanup: Whether to automatically clean up memory
            cleanup_threshold: Memory usage threshold to trigger cleanup
            aggressive_cleanup_threshold: Threshold for aggressive cleanup
        """
        self.tracker = tracker
        self.auto_cleanup = auto_cleanup
        self.cleanup_threshold = cleanup_threshold
        self.aggressive_cleanup_threshold = aggressive_cleanup_threshold

        # Register alert callback
        self.tracker.add_alert_callback(self._handle_alert)

        self.cleanup_count = 0
        self.last_cleanup_time = 0.0
        self.min_cleanup_interval = 30.0  # Minimum 30 seconds between cleanups

    def _handle_alert(self, event: TrackingEvent) -> None:
        """Handle memory alerts."""
        if not self.auto_cleanup:
            return

        current_time = time.time()

        # Avoid too frequent cleanups
        if current_time - self.last_cleanup_time < self.min_cleanup_interval:
            return

        # Check if cleanup is needed
        if event.event_type == "critical" or (
            event.event_type == "warning"
            and event.metadata
            and event.metadata.get("usage_percent", 0) >= self.cleanup_threshold * 100
        ):
            self._perform_cleanup(aggressive=event.event_type == "critical")
            self.last_cleanup_time = current_time

    def _perform_cleanup(self, aggressive: bool = False) -> None:
        """Perform memory cleanup."""
        self.cleanup_count += 1

        try:
            backend = self.tracker.backend
            if backend in {"cuda", "rocm"}:
                torch.cuda.empty_cache()
                if aggressive:
                    import gc

                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            elif backend == "mps":
                import gc

                import torch.mps as torch_mps

                if hasattr(torch_mps, "empty_cache"):
                    torch_mps.empty_cache()
                if aggressive:
                    gc.collect()
                    if hasattr(torch_mps, "empty_cache"):
                        torch_mps.empty_cache()
            elif aggressive:
                import gc

                gc.collect()

            # Log cleanup event
            cleanup_type = "aggressive" if aggressive else "standard"
            self.tracker._add_event(
                "cleanup", 0, f"Performed {cleanup_type} memory cleanup"
            )

        except Exception as e:
            self.tracker._add_event("error", 0, f"Cleanup failed: {str(e)}")

    def force_cleanup(self, aggressive: bool = False) -> None:
        """Force immediate memory cleanup."""
        self._perform_cleanup(aggressive)

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        return {
            "cleanup_count": self.cleanup_count,
            "last_cleanup_time": self.last_cleanup_time,
            "auto_cleanup_enabled": self.auto_cleanup,
            "cleanup_threshold": self.cleanup_threshold,
            "aggressive_cleanup_threshold": self.aggressive_cleanup_threshold,
        }
