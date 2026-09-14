"""Real-time memory tracking and monitoring."""

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast

import torch

from .collector_health import (
    COLLECTOR_HEALTH_DEGRADED,
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    CollectorHealthState,
    collector_retry_delay_seconds,
)
from .cuda_native_debug import (
    DEFAULT_TRACE_ALLOC_MAX_ENTRIES,
    capture_cuda_snapshot_artifacts,
    cuda_memory_history_supported,
    start_cuda_memory_history,
    stop_cuda_memory_history,
)
from .device_collectors import (
    DeviceMemoryCapabilities,
    DeviceMemoryCollector,
    DeviceMemorySample,
    DeviceMemorySampleResult,
    _resolve_device,
    build_device_memory_collector,
    detect_torch_runtime_backend,
    validate_device_memory_sample,
)
from .oom_flight_recorder import (
    OOMFlightRecorder,
    OOMFlightRecorderConfig,
    classify_oom_exception,
)
from .phases import PhaseHandle, PhaseRecorder, PhaseToken
from .session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SESSION_STATUS_RUNNING,
    SessionSummary,
    create_session_summary,
    finalize_session_summary,
    now_ns,
    update_session_summary,
)
from .telemetry import (
    SCHEMA_VERSION_V4,
    TelemetryEvent,
    resolve_distributed_identity,
    telemetry_event_to_dict,
)
from .telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig
from .utils import format_bytes, get_gpu_info

logger = logging.getLogger(__name__)


@dataclass
class TrackingEvent:
    """Represents a memory tracking event."""

    timestamp: float
    event_type: str  # 'allocation', 'deallocation', 'peak', 'warning', 'error'
    memory_allocated: Optional[int]
    memory_reserved: Optional[int]
    memory_change: Optional[int]
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
        enable_native_cuda_history: bool = False,
        native_history_max_entries: int = DEFAULT_TRACE_ALLOC_MAX_ENTRIES,
        telemetry_sink_config: Optional[TelemetrySinkConfig] = None,
        collector: Optional[DeviceMemoryCollector] = None,
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
            collector: Optional runtime memory collector. When provided, the
                tracker bypasses torch device resolution and ``device`` must be
                ``None``.
        """
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be > 0")
        if max_events <= 0:
            raise ValueError("max_events must be >= 1")
        if native_history_max_entries <= 0:
            raise ValueError("native_history_max_entries must be >= 1")

        if collector is not None and device is not None:
            raise ValueError("device and collector cannot be provided together")
        self.device = self._setup_device(device) if collector is None else None
        self.collector = (
            build_device_memory_collector(self.device)
            if collector is None
            else collector
        )
        self.backend = self.collector.name()
        raw_capabilities = self.collector.capabilities()
        self.collector_capabilities = (
            raw_capabilities
            if isinstance(raw_capabilities, DeviceMemoryCapabilities)
            else DeviceMemoryCapabilities.from_mapping(
                cast(Mapping[str, Any], raw_capabilities)
            )
        )
        if (
            enable_native_cuda_history
            and not self.collector_capabilities.supports_native_allocator_history
        ):
            raise ValueError(
                "native allocator history is unavailable for backend "
                f"{self.backend!r}"
            )
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.enable_alerts = enable_alerts
        self.enable_native_cuda_history = enable_native_cuda_history
        self.native_history_max_entries = native_history_max_entries
        self._telemetry_sink_config = telemetry_sink_config
        self._telemetry_sink = (
            AppendOnlyTelemetrySink(telemetry_sink_config)
            if telemetry_sink_config is not None
            else None
        )
        self.last_oom_dump_path: Optional[str] = None
        self.distributed_identity = resolve_distributed_identity(
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            env=os.environ,
        )
        self.session_source = "stormlog.tracker"
        self._session_summary: Optional[SessionSummary] = None

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
        self._events_lock = threading.Lock()
        self._history_dropped_events = 0
        self.is_tracking = False
        self._tracking_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._collector_health = CollectorHealthState()
        self._last_observed_sample: Optional[DeviceMemorySample] = None
        self._last_sink_diagnostics: Dict[str, int] = self._empty_sink_diagnostics()
        self._collector_retry_backoff_initial_s = 1.0
        self._collector_retry_backoff_factor = 2.0
        self._collector_retry_backoff_cap_s = 30.0
        self._phase_state = PhaseRecorder()

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
            "peak_device_used": 0,
            "total_allocations": 0,
            "total_deallocations": 0,
            "total_allocation_bytes": 0,
            "total_deallocation_bytes": 0,
            "alert_count": 0,
            "tracking_start_time": None,
            "last_memory_check": 0,
        }

        self._initialize_memory_limits()

    def _initialize_memory_limits(self) -> None:
        # Get memory limits with backend-aware fallback.
        self.gpu_info = (
            get_gpu_info(self.device)
            if self.device is not None and self.device.type == "cuda"
            else {}
        )
        initial_result = self.collector.sample_with_diagnostics()
        initial_sample = initial_result.sample
        if initial_sample is not None:
            try:
                validate_device_memory_sample(
                    initial_sample,
                    self.collector_capabilities,
                    partial_fields=initial_result.partial_fields,
                )
            except ValueError as exc:
                logger.debug("Initial collector sample rejected: %s", exc)
                initial_sample = None
            else:
                self._last_observed_sample = initial_sample
        total_memory = (
            initial_sample.total_bytes if initial_sample is not None else None
        )
        if total_memory is None:
            fallback_total = self.gpu_info.get("total_memory", 0)
            total_memory = (
                int(fallback_total) if isinstance(fallback_total, (int, float)) else 0
            )
        self.total_memory = int(total_memory)

    @staticmethod
    def _empty_sink_diagnostics() -> Dict[str, int]:
        return {
            "rollover_count": 0,
            "pruned_segment_count": 0,
            "pruned_bytes": 0,
            "final_retained_files": 0,
            "final_retained_bytes": 0,
        }

    def _reset_collector_session_state(self) -> None:
        """Reset per-session collector state before a fresh tracking run."""
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )
        self._last_observed_sample = None
        self.stats["last_memory_check"] = 0

    def _reset_tracking_state_for_new_session(self) -> None:
        """Clear per-session in-memory state before starting a new run."""
        with self._events_lock:
            self.events.clear()
            self._history_dropped_events = 0
        self._last_sink_diagnostics = self._empty_sink_diagnostics()
        self.last_oom_dump_path = None
        self.stats.update(
            {
                "peak_memory": 0,
                "peak_device_used": 0,
                "total_allocations": 0,
                "total_deallocations": 0,
                "total_allocation_bytes": 0,
                "total_deallocation_bytes": 0,
                "alert_count": 0,
                "tracking_start_time": None,
                "last_memory_check": 0,
            }
        )
        self._oom_flight_recorder.clear()

    def _open_session(self) -> SessionSummary:
        """Create the active runtime session summary for a tracking run."""
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

    def _ensure_telemetry_sink(self) -> None:
        if self._telemetry_sink is None and self._telemetry_sink_config is not None:
            self._telemetry_sink = AppendOnlyTelemetrySink(self._telemetry_sink_config)

    def get_session_summary(self) -> Optional[SessionSummary]:
        """Return the current or most recent tracking session summary."""
        return self._session_summary

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
        """Collect one backend sample for ad-hoc diagnostics with fallback values."""
        result = self.collector.sample_with_diagnostics()
        if result.sample is not None:
            try:
                validate_device_memory_sample(
                    result.sample,
                    self.collector_capabilities,
                    partial_fields=result.partial_fields,
                )
            except ValueError as exc:
                logger.debug("Rejected invalid %s sample: %s", self.backend, exc)
            else:
                return result.sample

        logger.debug(
            "Could not sample %s memory: %s",
            self.backend,
            result.core_error or "unknown collector error",
        )
        return self._empty_sample()

    def _empty_sample(self) -> DeviceMemorySample:
        """Build a zeroed sample for status-only events without live telemetry."""
        device_id = 0
        if self.device is not None and self.device.type == "cuda":
            try:
                device_id = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            except Exception:
                device_id = 0
        return DeviceMemorySample(
            allocated_bytes=None,
            reserved_bytes=None,
            used_bytes=None,
            free_bytes=None,
            total_bytes=None,
            active_bytes=None,
            inactive_bytes=None,
            device_id=device_id,
        )

    def _event_sample(self, sample: Optional[DeviceMemorySample]) -> DeviceMemorySample:
        if sample is not None:
            return sample
        if self._last_observed_sample is not None:
            return self._last_observed_sample
        return self._empty_sample()

    @staticmethod
    def _collector_error_message(result: DeviceMemorySampleResult) -> Optional[str]:
        if result.core_error:
            return result.core_error
        unique_messages = list(dict.fromkeys(result.errors.values()))
        if not unique_messages:
            return None
        return "; ".join(unique_messages)

    def _set_collector_health(
        self,
        *,
        status: str,
        telemetry_partial: bool,
        partial_fields: tuple[str, ...] = (),
        last_error: Optional[str] = None,
        consecutive_failures: int = 0,
        next_retry_epoch_s: Optional[float] = None,
    ) -> None:
        self._collector_health = CollectorHealthState(
            status=status,
            telemetry_partial=telemetry_partial,
            partial_fields=partial_fields,
            last_error=last_error,
            consecutive_failures=consecutive_failures,
            next_retry_epoch_s=next_retry_epoch_s,
        )

    def _retry_collection_due(self, now: float) -> bool:
        retry_at = self._collector_health.next_retry_epoch_s
        return retry_at is None or now >= retry_at

    def _transition_to_core_failure(
        self,
        result: DeviceMemorySampleResult,
        *,
        event_time: float,
    ) -> None:
        previous_health = self._collector_health
        consecutive_failures = previous_health.consecutive_failures + 1
        retry_delay_s = collector_retry_delay_seconds(
            consecutive_failures,
            initial_delay_s=self._collector_retry_backoff_initial_s,
            factor=self._collector_retry_backoff_factor,
            max_delay_s=self._collector_retry_backoff_cap_s,
        )
        next_retry_epoch_s = event_time + retry_delay_s if retry_delay_s > 0 else None
        error_message = self._collector_error_message(result) or "Collector unavailable"
        self._set_collector_health(
            status=COLLECTOR_HEALTH_UNHEALTHY,
            telemetry_partial=True,
            last_error=error_message,
            consecutive_failures=consecutive_failures,
            next_retry_epoch_s=next_retry_epoch_s,
        )
        if previous_health.status == COLLECTOR_HEALTH_HEALTHY:
            self._add_event(
                "collector_degraded",
                0,
                "Collector unavailable; telemetry paused until recovery.",
                metadata={
                    "collector_transition": "degraded",
                    "collector_degraded_from": previous_health.status,
                    "collector_degradation_reason": error_message,
                    "collector_retry_delay_s": retry_delay_s,
                },
            )

    def _transition_to_sampled_state(
        self,
        result: DeviceMemorySampleResult,
        *,
        sample: DeviceMemorySample,
    ) -> bool:
        previous_health = self._collector_health
        is_partial = result.is_partial
        error_message = self._collector_error_message(result)

        if is_partial:
            self._set_collector_health(
                status=COLLECTOR_HEALTH_DEGRADED,
                telemetry_partial=True,
                partial_fields=result.partial_fields,
                last_error=error_message,
                consecutive_failures=0,
                next_retry_epoch_s=None,
            )
            if previous_health.status == COLLECTOR_HEALTH_HEALTHY:
                self._add_event(
                    "collector_degraded",
                    0,
                    "Collector degraded; telemetry is partial.",
                    metadata={
                        "collector_transition": "degraded",
                        "collector_degraded_from": previous_health.status,
                        "collector_degradation_reason": error_message,
                    },
                    sample=sample,
                )
            return True

        recovered = previous_health.status != COLLECTOR_HEALTH_HEALTHY
        previous_error = previous_health.last_error
        previous_failures = previous_health.consecutive_failures
        previous_status = previous_health.status
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )
        if recovered:
            self._add_event(
                "collector_recovered",
                0,
                "Collector recovered; full telemetry sampling resumed.",
                metadata={
                    "collector_transition": "recovered",
                    "collector_recovered_from": previous_status,
                    "collector_previous_error": previous_error,
                    "collector_previous_failure_count": previous_failures,
                },
                sample=sample,
            )
        return False

    def _run_tracking_iteration(self, last_allocated: int) -> int:
        """Run one collection iteration, preserving health state across failures."""
        now = time.time()
        if not self._retry_collection_due(now):
            return last_allocated

        result = self.collector.sample_with_diagnostics()
        if result.sample is None:
            self._transition_to_core_failure(result, event_time=now)
            return last_allocated

        sample = result.sample
        try:
            validate_device_memory_sample(
                sample,
                self.collector_capabilities,
                partial_fields=result.partial_fields,
            )
        except ValueError as exc:
            invalid_result = DeviceMemorySampleResult(
                sample=None,
                errors={"sample_contract": str(exc)},
                core_error=str(exc),
            )
            self._transition_to_core_failure(invalid_result, event_time=now)
            return last_allocated

        self._last_observed_sample = sample
        current_allocated = sample.allocated_bytes
        has_allocator = self.collector_capabilities.supports_allocator_allocated
        memory_change = (
            current_allocated - last_allocated
            if has_allocator and current_allocated is not None
            else None
        )
        is_partial = self._transition_to_sampled_state(result, sample=sample)

        self.stats["last_memory_check"] = now

        if sample.used_bytes is not None:
            self.stats["peak_device_used"] = max(
                self.stats["peak_device_used"], sample.used_bytes
            )

        if (
            current_allocated is not None
            and current_allocated > self.stats["peak_memory"]
        ):
            self.stats["peak_memory"] = current_allocated
            self._add_event(
                "peak",
                memory_change,
                f"New peak memory: {format_bytes(current_allocated)}",
                sample=sample,
            )

        if memory_change is not None and memory_change > 0:
            self.stats["total_allocations"] += 1
            self.stats["total_allocation_bytes"] += memory_change
            self._add_event(
                "allocation",
                memory_change,
                f"Memory allocated: {format_bytes(memory_change)}",
                sample=sample,
            )
        elif memory_change is not None and memory_change < 0:
            self.stats["total_deallocations"] += 1
            self.stats["total_deallocation_bytes"] += abs(memory_change)
            self._add_event(
                "deallocation",
                memory_change,
                f"Memory freed: {format_bytes(abs(memory_change))}",
                sample=sample,
            )

        if self.enable_alerts:
            self._check_alerts(memory_change, sample=sample)

        partial_fields = ", ".join(result.partial_fields)
        sample_context = (
            f"Collected partial telemetry sample ({partial_fields})."
            if is_partial
            else "Collected telemetry sample."
        )
        self._add_event(
            "sample",
            memory_change,
            sample_context,
            sample=sample,
        )

        return current_allocated if current_allocated is not None else last_allocated

    def start_tracking(self) -> None:
        """Start real-time memory tracking."""
        if self.is_tracking:
            return

        self._reset_collector_session_state()
        self._reset_tracking_state_for_new_session()
        self._session_summary = None
        self._phase_state.reset()
        self._ensure_telemetry_sink()
        self._stop_event.clear()
        self.stats["tracking_start_time"] = time.time()
        self._open_session()

        self._tracking_thread = threading.Thread(target=self._tracking_loop)
        self._tracking_thread.daemon = True
        self._tracking_thread.start()
        self.is_tracking = True

        # Add initial event
        self._add_event("start", 0, "Memory tracking started")

    def stop_tracking(self) -> None:
        """Stop real-time memory tracking."""
        if not self.is_tracking:
            return

        self._stop_event.set()

        if self._tracking_thread:
            self._tracking_thread.join(timeout=5.0)
            if self._tracking_thread.is_alive():
                raise TimeoutError(
                    "Memory tracking thread did not stop within 5.0 seconds"
                )

        self.is_tracking = False

        # Add final event
        self._add_event("stop", 0, "Memory tracking stopped")
        self._close_telemetry_sink()
        if self._session_summary is not None:
            self._session_summary = finalize_session_summary(
                self._session_summary,
                ended_at_ns=now_ns(),
            )
        self._phase_state.reset()

    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread."""
        last_allocated = 0

        while not self._stop_event.wait(self.sampling_interval):
            try:
                last_allocated = self._run_tracking_iteration(last_allocated)
                self._flush_telemetry_sink()
            except Exception as exc:
                self._add_event("error", 0, f"Tracking error: {str(exc)}")
                self._flush_telemetry_sink(force=True)
                time.sleep(1.0)  # Back off on unexpected tracker logic errors

    def _add_event(
        self,
        event_type: str,
        memory_change: Optional[int],
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
        sample: Optional[DeviceMemorySample] = None,
    ) -> None:
        """Add a tracking event."""
        if not self.collector_capabilities.supports_allocator_allocated:
            memory_change = None
        snapshot = self._event_sample(sample)
        current_allocated = snapshot.allocated_bytes
        current_reserved = snapshot.reserved_bytes
        event_metadata = dict(metadata or {})
        event_metadata.update(self._collector_health.to_dict())

        event = TrackingEvent(
            timestamp=time.time(),
            event_type=event_type,
            memory_allocated=current_allocated,
            memory_reserved=current_reserved,
            memory_change=memory_change,
            device_id=snapshot.device_id,
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
            metadata=event_metadata,
            active_memory=snapshot.active_bytes,
            inactive_memory=snapshot.inactive_bytes,
            device_used=snapshot.used_bytes,
            device_free=snapshot.free_bytes,
            device_total=snapshot.total_bytes,
            backend=self.backend,
        )

        with self._events_lock:
            if len(self.events) == self.max_events:
                self._history_dropped_events += 1
            self.events.append(event)
        self._oom_flight_recorder.record_event(self._tracking_event_payload(event))
        self._append_to_telemetry_sink(event)

        # Trigger callbacks for alerts
        if event_type in ["warning", "critical", "error"]:
            self.stats["alert_count"] += 1
            for callback in self.alert_callbacks:
                try:
                    callback(event)
                except Exception as exc:
                    logger.debug("Alert callback error (suppressed): %s", exc)

    def enter_phase(
        self, name: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> PhaseHandle:
        """Enter one structured workload phase while tracking is active."""
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
        """Context manager that emits structured phase enter and exit records."""
        handle = self.enter_phase(name, metadata=metadata)
        try:
            yield handle
        finally:
            self._close_phase_handle(handle)

    def _close_phase_handle(self, handle: PhaseHandle) -> None:
        handle.close()

    def _emit_phase_exit(self, token: PhaseToken) -> None:
        boundary = self._phase_state.exit(token)
        self._add_event(
            boundary.event_type,
            0,
            boundary.context,
            metadata=boundary.metadata,
        )

    def _check_alerts(
        self,
        change: Optional[int],
        *,
        sample: DeviceMemorySample,
    ) -> bool:
        """Check for memory alerts and warnings."""
        if self.total_memory == 0:
            return False

        allocated = sample.allocated_bytes
        reserved = sample.reserved_bytes
        usage_value = (
            allocated
            if self.collector_capabilities.supports_allocator_allocated
            else sample.used_bytes
        )
        if usage_value is None:
            return False

        usage_percent = (usage_value / self.total_memory) * 100
        emitted = False

        # Critical memory usage
        if usage_percent >= self.thresholds["memory_critical_percent"]:
            self._add_event(
                "critical",
                change,
                f"CRITICAL: Memory usage at {usage_percent:.1f}%",
                {"usage_percent": usage_percent},
                sample=sample,
            )
            emitted = True

        # Warning memory usage
        elif usage_percent >= self.thresholds["memory_warning_percent"]:
            self._add_event(
                "warning",
                change,
                f"WARNING: Memory usage at {usage_percent:.1f}%",
                {"usage_percent": usage_percent},
                sample=sample,
            )
            emitted = True

        # Large allocation warning
        if (
            change is not None
            and self.collector_capabilities.supports_allocator_allocated
            and change > self.thresholds["memory_leak_threshold"]
        ):
            self._add_event(
                "warning",
                change,
                f"Large allocation detected: {format_bytes(change)}",
                {"large_allocation": True},
                sample=sample,
            )
            emitted = True

        # Fragmentation warning
        if (
            self.collector_capabilities.supports_fragmentation_analysis
            and reserved is not None
            and allocated is not None
            and reserved > 0
        ):
            fragmentation = (reserved - allocated) / reserved
            if fragmentation > self.thresholds["fragmentation_threshold"]:
                self._add_event(
                    "warning",
                    change,
                    f"High fragmentation: {fragmentation:.1%}",
                    {"fragmentation": fragmentation},
                    sample=sample,
                )
                emitted = True

        return emitted

    @staticmethod
    def _tracking_event_payload(event: TrackingEvent) -> Dict[str, Any]:
        """Serialize a TrackingEvent into a stable JSON-safe payload."""
        return {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "session_id": event.session_id,
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

    def _telemetry_record_from_event(self, event: TrackingEvent) -> Dict[str, Any]:
        host = socket.gethostname()
        pid = os.getpid()
        sampling_interval_ms = int(round(self.sampling_interval * 1000))
        session_id = event.session_id or self._open_session().session_id
        default_collector = self.collector_capabilities.telemetry_collector
        serialized_capabilities = self.collector_capabilities.to_metadata()
        capability_metadata = {
            "backend": self.backend,
            "supports_device_total": self.collector_capabilities.supports_device_total,
            "supports_device_free": self.collector_capabilities.supports_device_free,
            "sampling_source": self.collector_capabilities.sampling_source,
            "memory_capabilities": serialized_capabilities,
        }
        metadata = dict(event.metadata or {})
        metadata.update(capability_metadata)
        partial_fields = set(metadata.get("collector_partial_fields", []) or [])
        event_total = event.device_total
        if (
            event_total is None
            and "device_total_bytes" not in partial_fields
            and self.total_memory
        ):
            event_total = self.total_memory
        telemetry_event = TelemetryEvent(
            schema_version=SCHEMA_VERSION_V4,
            session_id=session_id,
            timestamp_ns=int(event.timestamp * 1_000_000_000),
            event_type=event.event_type,
            collector=default_collector,
            sampling_interval_ms=sampling_interval_ms,
            pid=pid,
            host=host,
            device_id=event.device_id,
            allocator_allocated_bytes=event.memory_allocated,
            allocator_reserved_bytes=event.memory_reserved,
            allocator_active_bytes=event.active_memory,
            allocator_inactive_bytes=event.inactive_memory,
            allocator_change_bytes=event.memory_change,
            device_used_bytes=event.device_used,
            device_free_bytes=event.device_free,
            device_total_bytes=event_total,
            context=event.context,
            job_id=event.job_id,
            rank=event.rank,
            local_rank=event.local_rank,
            world_size=event.world_size,
            metadata=metadata,
        )
        return telemetry_event_to_dict(telemetry_event)

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
            "Disabling telemetry sink after %s failure: %s",
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
                "Telemetry sink close failed after %s error: %s", operation, close_exc
            )

    @staticmethod
    def _close_sink_with_status(sink: Any, status: str) -> None:
        try:
            sink.close(session_status=status)
        except TypeError:
            sink.close()

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

        capabilities: Any = self.collector_capabilities
        if not isinstance(capabilities, DeviceMemoryCapabilities):
            capabilities = DeviceMemoryCapabilities.from_mapping(capabilities)
        dump_metadata: Dict[str, Any] = {
            "tracker_stats": self.get_statistics(),
            "collector_capabilities": capabilities.to_metadata(),
            "total_memory_bytes": self.total_memory,
            "sampling_interval_s": self.sampling_interval,
            "session_id": None,
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
        session_summary = getattr(self, "_session_summary", None)
        dump_metadata["tracker_stats"] = self.get_statistics()
        dump_metadata["session_id"] = (
            session_summary.session_id if session_summary is not None else None
        )

        try:
            dump_path = self._oom_flight_recorder.dump(
                reason=classification.reason,
                exception=exc,
                context=context,
                backend=self.backend,
                metadata=dump_metadata,
                session_summary=session_summary,
            )
        except Exception as dump_exc:
            logger.debug("OOM flight recorder dump failed: %s", dump_exc)
            return None

        self.last_oom_dump_path = dump_path
        return dump_path

    def _capture_native_history_dump(self, bundle_dir: Path) -> None:
        """Add CUDA allocator snapshot artifacts into an OOM dump bundle."""
        if self.device is None:
            return
        try:
            files_written = capture_cuda_snapshot_artifacts(
                bundle_dir,
                device=self.device,
                history_recorded=True,
            )
        except Exception as exc:
            logger.debug("CUDA native history dump failed: %s", exc)
            return

        manifest_path = bundle_dir / "manifest.json"
        metadata_path = bundle_dir / "metadata.json"

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_files = list(manifest.get("files", []))
            for name in files_written:
                if name not in manifest_files:
                    manifest_files.append(name)
            manifest["files"] = manifest_files
            manifest["native_history_enabled"] = True
            manifest["native_history_files"] = files_written
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not update OOM manifest with native history: %s", exc)

        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            custom_metadata = dict(metadata_payload.get("custom_metadata", {}))
            custom_metadata["native_history_enabled"] = True
            custom_metadata["native_history_files"] = files_written
            metadata_payload["custom_metadata"] = custom_metadata
            metadata_path.write_text(
                json.dumps(metadata_payload, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not update OOM metadata with native history: %s", exc)

        try:
            root = Path(self._oom_flight_recorder.config.dump_dir)
            self._oom_flight_recorder._prune_retention(root)
        except Exception as exc:
            logger.debug(
                "Could not reapply OOM retention after native history: %s", exc
            )

    @contextmanager
    def capture_oom(
        self,
        context: str = "runtime",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Capture OOM diagnostic bundle if a tracked block raises OOM."""
        native_history_recorded = MemoryTracker._start_oom_native_history(self)
        try:
            yield
        except Exception as exc:
            dump_path = self.handle_exception(exc, context=context, metadata=metadata)
            if dump_path and native_history_recorded and self.backend == "cuda":
                MemoryTracker._capture_native_history_dump(self, Path(dump_path))
                if not Path(dump_path).exists():
                    self.last_oom_dump_path = None
                    dump_path = None
            if dump_path:
                logger.error("OOM flight recorder dump saved to: %s", dump_path)
            raise
        finally:
            if native_history_recorded:
                try:
                    stop_cuda_memory_history(device=self.device)
                except Exception as exc:
                    logger.debug(
                        "Could not stop CUDA native history recording: %s",
                        exc,
                    )

    def _start_oom_native_history(self) -> bool:
        native_history_recorded = False
        if (
            self.enable_native_cuda_history
            and self._oom_flight_recorder.config.enabled
            and self.backend == "cuda"
            and self.device is not None
            and cuda_memory_history_supported()
        ):
            try:
                start_cuda_memory_history(
                    device=self.device,
                    trace_alloc_max_entries=self.native_history_max_entries,
                )
                native_history_recorded = True
            except Exception as exc:
                logger.debug("Could not start CUDA native history recording: %s", exc)
        return native_history_recorded

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
        with self._events_lock:
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
        if interval <= 0:
            raise ValueError("interval must be > 0")

        with self._events_lock:
            events_snapshot = list(self.events)

        if not events_snapshot:
            return {
                "timestamps": [],
                "allocated": [],
                "reserved": [],
                "device_used": [],
                "device_free": [],
                "device_total": [],
            }

        # Group events by time intervals
        start_time = events_snapshot[0].timestamp

        timestamps = []
        allocated_values: list[Optional[int]] = []
        reserved_values: list[Optional[int]] = []
        device_used_values: list[Optional[int]] = []
        device_free_values: list[Optional[int]] = []
        device_total_values: list[Optional[int]] = []

        current_bucket = 0
        last_event = events_snapshot[0]
        for event in events_snapshot[1:]:
            bucket = int((event.timestamp - start_time) // interval)
            if bucket == current_bucket:
                last_event = event
                continue

            timestamps.append(start_time + current_bucket * interval)
            allocated_values.append(last_event.memory_allocated)
            reserved_values.append(last_event.memory_reserved)
            device_used_values.append(getattr(last_event, "device_used", None))
            device_free_values.append(getattr(last_event, "device_free", None))
            device_total_values.append(getattr(last_event, "device_total", None))
            current_bucket = bucket
            last_event = event

        timestamps.append(start_time + current_bucket * interval)
        allocated_values.append(last_event.memory_allocated)
        reserved_values.append(last_event.memory_reserved)
        device_used_values.append(getattr(last_event, "device_used", None))
        device_free_values.append(getattr(last_event, "device_free", None))
        device_total_values.append(getattr(last_event, "device_total", None))

        return {
            "timestamps": timestamps,
            "allocated": allocated_values,
            "reserved": reserved_values,
            "device_used": device_used_values,
            "device_free": device_free_values,
            "device_total": device_total_values,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        current_stats = self.stats.copy()
        with self._events_lock:
            events_snapshot = list(self.events)
            history_dropped_events = self._history_dropped_events
        recent_events = [e for e in events_snapshot if e.timestamp > time.time() - 3600]
        sample = (
            self._last_observed_sample
            if self._collector_health.status != COLLECTOR_HEALTH_UNHEALTHY
            else None
        )

        current_stats.update(
            {
                "total_events": len(events_snapshot),
                "events_last_hour": len(recent_events),
                "history_window_limit_events": self.max_events,
                "history_retained_events": len(events_snapshot),
                "history_dropped_events": history_dropped_events,
                "backend": self.backend,
                "oom_flight_recorder_enabled": self._oom_flight_recorder.config.enabled,
                "last_oom_dump_path": self.last_oom_dump_path,
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
                **_sample_memory_statistics(sample, self.total_memory),
                "memory_capabilities": self.collector_capabilities.to_metadata(),
                "average_allocation_size": (
                    self.stats["total_allocation_bytes"]
                    / max(self.stats["total_allocations"], 1)
                    if self.collector_capabilities.supports_allocator_allocated
                    else None
                ),
                "average_deallocation_size": (
                    self.stats["total_deallocation_bytes"]
                    / max(self.stats["total_deallocations"], 1)
                    if self.collector_capabilities.supports_allocator_allocated
                    else None
                ),
            }
        )
        current_stats.update(self._collector_health.to_dict())
        current_stats.update(self._last_sink_diagnostics)

        if not self.collector_capabilities.supports_allocator_allocated:
            current_stats.update(
                {
                    "peak_memory": None,
                    "total_allocations": None,
                    "total_deallocations": None,
                    "total_allocation_bytes": None,
                    "total_deallocation_bytes": None,
                    "allocations_per_second": None,
                    "bytes_allocated_per_second": None,
                }
            )

        if self.stats["tracking_start_time"]:
            tracking_duration = time.time() - self.stats["tracking_start_time"]
            current_stats.update(
                {
                    "tracking_duration_seconds": tracking_duration,
                    "allocations_per_second": (
                        self.stats["total_allocations"] / max(tracking_duration, 1)
                        if self.collector_capabilities.supports_allocator_allocated
                        else None
                    ),
                    "bytes_allocated_per_second": (
                        self.stats["total_allocation_bytes"] / max(tracking_duration, 1)
                        if self.collector_capabilities.supports_allocator_allocated
                        else None
                    ),
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

        with self._events_lock:
            events_snapshot = list(self.events)

        if not events_snapshot:
            return

        # Convert events to canonical telemetry records.
        records = [
            self._telemetry_record_from_event(event) for event in events_snapshot
        ]

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
        with self._events_lock:
            self.events.clear()
            self._history_dropped_events = 0

        # Reset statistics
        self.stats.update(
            {
                "peak_memory": 0,
                "peak_device_used": 0,
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
        with self._events_lock:
            events_snapshot = list(self.events)
        alerts = [e for e in events_snapshot if e.event_type in alert_types]

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


def _sample_memory_statistics(
    sample: DeviceMemorySample | None, total_memory: int
) -> Dict[str, Any]:
    return {
        "current_memory_allocated": (
            sample.allocated_bytes if sample is not None else None
        ),
        "current_memory_reserved": (
            sample.reserved_bytes if sample is not None else None
        ),
        "current_device_used": sample.used_bytes if sample is not None else None,
        "current_device_free": sample.free_bytes if sample is not None else None,
        "device_total": sample.total_bytes if sample is not None else None,
        "memory_utilization_percent": (
            (sample.used_bytes / total_memory * 100)
            if sample is not None and sample.used_bytes is not None and total_memory > 0
            else None
        ),
    }
