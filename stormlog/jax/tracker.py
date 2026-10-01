"""
Real-time JAX Memory Tracking.

This module provides real-time monitoring of JAX device memory usage,
integrating with Stormlog's shared telemetry, session, and phase
tracking infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from stormlog.collector_health import (
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    CollectorHealthState,
    collector_retry_delay_seconds,
)
from stormlog.oom_flight_recorder import (
    OOMFlightRecorder,
    OOMFlightRecorderConfig,
    classify_oom_exception,
)
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
    resolve_distributed_identity,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig

from .jax_env import configure_jax_logging
from .utils import _device_zero, get_device_memory_capability, resolve_jax_device

configure_jax_logging()

jax: Any

try:
    import jax as _jax  # noqa: E402

    jax = _jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Results from real-time JAX memory tracking."""

    start_time: float
    end_time: float
    samples_collected: int
    peak_memory_bytes: int
    min_memory_bytes: int
    average_memory_bytes: int
    alert_count: int
    session_summary: Optional[SessionSummary] = None
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)
    memory_usage: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    device_memory_profile_path: Optional[str] = None
    device_memory_available: bool = False
    memory_source: str = "unavailable"
    device_memory_unavailable_reason: Optional[str] = None
    process_memory_bytes: Optional[int] = None

    history_window_limit: int = 0
    history_retained_samples: int = 0
    history_dropped_samples: int = 0
    history_retained_events: int = 0
    history_dropped_events: int = 0
    history_retained_alerts: int = 0
    history_dropped_alerts: int = 0

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory usage in MB."""
        return self.peak_memory_bytes / (1024 * 1024)

    @property
    def average_memory_mb(self) -> float:
        """Average memory usage in MB."""
        return self.average_memory_bytes / (1024 * 1024)

    @property
    def duration(self) -> float:
        """Total tracking duration in seconds."""
        return self.end_time - self.start_time


@dataclass(frozen=True)
class _TrackingResultData:
    retained_memory_usage: List[int]
    retained_timestamps: List[float]
    retained_events: List[Dict[str, Any]]
    retained_alerts: List[Dict[str, Any]]
    total_samples_observed: int
    peak_memory: int
    min_memory: int
    sum_memory: int
    dropped_samples: int
    dropped_events: int
    dropped_alerts: int


class MemoryTracker:
    """Real-time JAX device memory tracker."""

    def __init__(
        self,
        sampling_interval: float = 1.0,
        alert_threshold_mb: Optional[float] = None,
        device_index: Union[int, str] = 0,
        enable_logging: bool = True,
        max_history: int = 10_000,
        job_id: Optional[str] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        telemetry_sink_config: Optional[TelemetrySinkConfig] = None,
        save_device_profile_on_stop: bool = False,
        enable_oom_flight_recorder: bool = False,
        oom_dump_dir: str = "oom_dumps",
        oom_buffer_size: Optional[int] = None,
        oom_max_dumps: int = 5,
        oom_max_total_mb: int = 256,
    ):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX not available. Install with `pip install 'stormlog[jax]'`."
            )
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be > 0")
        if max_history <= 0:
            raise ValueError("max_history must be >= 1")

        self.sampling_interval = sampling_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.device_index = device_index
        self.enable_logging = enable_logging
        self.max_history = max_history
        self.save_device_profile_on_stop = save_device_profile_on_stop

        recorder_buffer_size = (
            oom_buffer_size if oom_buffer_size is not None else max_history
        )
        if recorder_buffer_size <= 0:
            recorder_buffer_size = max_history
        self._oom_flight_recorder = OOMFlightRecorder(
            OOMFlightRecorderConfig(
                enabled=enable_oom_flight_recorder,
                dump_dir=oom_dump_dir,
                buffer_size=recorder_buffer_size,
                max_dumps=oom_max_dumps,
                max_total_mb=oom_max_total_mb,
            )
        )
        self._last_oom_dump_path: Optional[str] = None

        self._initialize_device(device_index)

        # Cache invariant per-process values used in every telemetry record.
        self._cached_pid = os.getpid()
        self._cached_hostname = socket.gethostname()

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

        self.session_source = "stormlog.jax.tracker"
        self._session_summary: Optional[SessionSummary] = None

        # Tracking state
        self.tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        self._memory_usage: deque[int] = deque(maxlen=max_history)
        self._timestamps: deque[float] = deque(maxlen=max_history)
        self._events: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._alerts: deque[Dict[str, Any]] = deque(maxlen=max_history)

        self._history_dropped_samples = 0
        self._history_dropped_events = 0
        self._history_dropped_alerts = 0
        self._collector_failure_event_count = 0

        self._total_samples_observed = 0
        self._peak_memory_bytes = 0
        self._min_memory_bytes: Optional[int] = None
        self._sum_memory_bytes = 0

        self._last_sink_diagnostics: Dict[str, Any] = self._empty_sink_diagnostics()
        self._phase_state = PhaseRecorder()

        # Thread sync
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._collector_health = CollectorHealthState()
        self._last_successful_memory_bytes: Optional[int] = None
        self._session_start_time: Optional[float] = None
        self._session_end_time: Optional[float] = None

        self._collector_retry_backoff_initial_s = 1.0
        self._collector_retry_backoff_factor = 2.0
        self._collector_retry_backoff_cap_s = 30.0

        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        if enable_logging:
            logger.info(
                "JAX Memory Tracker initialized for device selector %r", device_index
            )

    def _initialize_device(self, device_index: Union[int, str]) -> None:
        self._device = None
        self._device_bytes_limit: Optional[int] = None
        self._last_reserved_bytes: Optional[int] = None
        self._device_memory_available = False
        self._device_memory_unavailable_reason: Optional[str] = (
            "No JAX device was selected"
        )
        try:
            if isinstance(device_index, int):
                devices = jax.local_devices()
                if device_index < 0 or device_index >= len(devices):
                    raise ValueError(f"JAX device index {device_index} is out of range")
                self._device = devices[device_index]
            else:
                self._device, self.device_index = resolve_jax_device(device_index)
            capability = get_device_memory_capability(self._device)
            self._device_memory_available = bool(capability["memory_stats_available"])
            self._device_memory_unavailable_reason = capability["memory_stats_error"]
            stats = capability["memory_stats"]
            if self._device_memory_available and "bytes_limit" in stats:
                self._device_bytes_limit = int(stats["bytes_limit"])
        except Exception as exc:
            logger.debug("Could not resolve JAX device %s: %s", device_index, exc)

        # Cache a scalar sentinel for sync barriers — avoids re-allocating
        # a device array on every sample.
        self._sync_sentinel: Any = None
        if self._device is not None:
            try:
                self._sync_sentinel = _device_zero(self._device)
            except Exception:
                pass

    @staticmethod
    def _empty_sink_diagnostics() -> Dict[str, Any]:
        return {
            "rollover_count": 0,
            "pruned_segment_count": 0,
            "pruned_bytes": 0,
            "final_retained_files": 0,
            "final_retained_bytes": 0,
        }

    def _reset_history(self) -> None:
        self._memory_usage.clear()
        self._timestamps.clear()
        self._events.clear()
        self._alerts.clear()
        self._history_dropped_samples = 0
        self._history_dropped_events = 0
        self._history_dropped_alerts = 0
        self._collector_failure_event_count = 0
        self._total_samples_observed = 0
        self._peak_memory_bytes = 0
        self._min_memory_bytes = None
        self._sum_memory_bytes = 0
        self._last_sink_diagnostics = self._empty_sink_diagnostics()
        self._last_oom_dump_path = None
        self._oom_flight_recorder.clear()

    def _tracking_result_data(self) -> _TrackingResultData:
        return _TrackingResultData(
            retained_memory_usage=list(self._memory_usage),
            retained_timestamps=list(self._timestamps),
            retained_events=list(self._events),
            retained_alerts=list(self._alerts),
            total_samples_observed=self._total_samples_observed,
            peak_memory=self._peak_memory_bytes,
            min_memory=(
                self._min_memory_bytes if self._min_memory_bytes is not None else 0
            ),
            sum_memory=self._sum_memory_bytes,
            dropped_samples=self._history_dropped_samples,
            dropped_events=self._history_dropped_events,
            dropped_alerts=self._history_dropped_alerts,
        )

    def _ensure_session_summary(self) -> SessionSummary:
        if self._session_summary is None:
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
            if self._telemetry_sink is not None and hasattr(
                self._telemetry_sink, "start_session"
            ):
                summary = self._telemetry_sink.start_session(summary)
            self._session_summary = summary
        return self._session_summary

    def get_session_summary(self) -> Optional[SessionSummary]:
        return self._session_summary

    @property
    def oom_buffer_size(self) -> int:
        """Resolved OOM ring-buffer size."""
        return self._oom_flight_recorder.config.buffer_size

    def _ensure_telemetry_sink(self) -> None:
        if self._telemetry_sink is None and self._telemetry_sink_config is not None:
            self._telemetry_sink = AppendOnlyTelemetrySink(self._telemetry_sink_config)

    def _get_current_device_memory(self) -> int:
        if self._device is None:
            return 0
        # Flush XLA async dispatch before reading memory stats.
        # JAX dispatches operations asynchronously to XLA. Without this
        # synchronisation barrier, memory_stats() may return stale values
        # that exclude memory from operations still "in flight".  The
        # trade-off is a small allocation (1-element array) and a forced
        # sync that can slightly perturb workload timing at high sample
        # rates.  At the default 1 s interval the overhead should be negligible.
        #
        # Exceptions are intentionally NOT caught here — they propagate to
        # _run_tracking_iteration which routes them through the
        # collector-health degradation path instead of recording a
        # synthetic zero-memory sample.
        if self._sync_sentinel is not None:
            self._sync_sentinel.block_until_ready()
        elif hasattr(jax.numpy, "zeros"):
            _device_zero(self._device).block_until_ready()
        stats = self._device.memory_stats()
        if stats:
            if "bytes_reserved" in stats:
                self._last_reserved_bytes = int(stats["bytes_reserved"])
            else:
                self._last_reserved_bytes = None
            return int(stats.get("bytes_in_use", 0))
        return 0

    def _get_current_cpu_memory(self) -> int:
        if PSUTIL_AVAILABLE and psutil is not None:
            return int(psutil.Process().memory_info().rss)
        return 0

    def _get_current_memory_bytes(self) -> Optional[int]:
        """Return measured device memory, or ``None`` when unsupported."""
        if not self._device_memory_available:
            return None
        return self._get_current_device_memory()

    def _get_current_memory(self) -> float:
        """Return current memory usage in MB."""
        memory_bytes = self._get_current_memory_bytes()
        return memory_bytes / (1024 * 1024) if memory_bytes is not None else 0.0

    def _build_telemetry_event_record(
        self,
        *,
        timestamp: float,
        memory_bytes: int,
        event_type: str = "sample",
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sampling_interval_ms = int(round(self.sampling_interval * 1000))
        # Use the real bytes_reserved value from JAX memory_stats() when
        # available, so downstream consumers can see actual XLA
        # preallocation overhead.  When the value is unavailable we fall
        # back to aliasing allocated_bytes and flag it as approximate.
        reserved_bytes = self._last_reserved_bytes
        if reserved_bytes is None:
            reserved_bytes = memory_bytes
            is_approximate = True
        else:
            is_approximate = False

        session = self._ensure_session_summary()
        health_dict = self._collector_health.to_dict()
        meta = {**metadata, **health_dict} if metadata else health_dict
        meta.update(
            {
                "backend": getattr(self._device, "platform", "unknown"),
                "sampling_source": (
                    "jax_device_stats"
                    if self._device_memory_available
                    else "unavailable"
                ),
                "device_memory_available": self._device_memory_available,
            }
        )
        if not self._device_memory_available:
            meta["collector_partial_fields"] = [
                "allocator_allocated_bytes",
                "allocator_reserved_bytes",
                "device_used_bytes",
            ]
            meta["device_memory_unavailable_reason"] = (
                self._device_memory_unavailable_reason
            )
        if is_approximate:
            meta["allocator_reserved_approximate"] = True
        legacy = {
            "session_id": session.session_id,
            "timestamp": timestamp,
            "type": event_type,
            "memory_mb": memory_bytes / (1024 * 1024),
            "allocator_allocated_bytes": memory_bytes,
            "allocator_reserved_bytes": reserved_bytes,
            "device_id": self.device_index,
            "context": context,
            "metadata": meta,
            "collector": "stormlog.jax.memory_tracker",
            "sampling_interval_ms": sampling_interval_ms,
            "pid": self._cached_pid,
            "host": self._cached_hostname,
            "job_id": self.distributed_identity["job_id"],
            "rank": self.distributed_identity["rank"],
            "local_rank": self.distributed_identity["local_rank"],
            "world_size": self.distributed_identity["world_size"],
        }

        if self._device_bytes_limit is not None:
            legacy["device_total_bytes"] = self._device_bytes_limit

        event = telemetry_event_from_record(
            legacy,
            default_collector="stormlog.jax.memory_tracker",
            default_sampling_interval_ms=sampling_interval_ms,
            default_session_id=session.session_id,
        )
        return telemetry_event_to_dict(event)

    def _append_event(
        self,
        *,
        timestamp: float,
        memory_bytes: int,
        event_type: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._build_telemetry_event_record(
            timestamp=timestamp,
            memory_bytes=memory_bytes,
            event_type=event_type,
            context=context,
            metadata=metadata,
        )
        with self._lock:
            if len(self._events) == self.max_history:
                self._history_dropped_events += 1
            self._events.append(record)
        self._append_to_telemetry_sink(record)
        if self._oom_flight_recorder.config.enabled:
            self._oom_flight_recorder.record_event(record)

    def _set_collector_health(
        self,
        *,
        status: str,
        telemetry_partial: bool,
        last_error: Optional[str] = None,
        consecutive_failures: int = 0,
        next_retry_epoch_s: Optional[float] = None,
    ) -> None:
        self._collector_health = CollectorHealthState(
            status=status,
            telemetry_partial=telemetry_partial,
            last_error=last_error,
            consecutive_failures=consecutive_failures,
            next_retry_epoch_s=next_retry_epoch_s,
        )

    def _retry_collection_due(self, now: float) -> bool:
        retry_at = self._collector_health.next_retry_epoch_s
        return retry_at is None or now >= retry_at

    def _status_memory_value(self) -> int:
        return self._last_successful_memory_bytes or 0

    def _transition_to_failure(self, timestamp: float, exc: BaseException) -> None:
        previous_health = self._collector_health
        consecutive_failures = previous_health.consecutive_failures + 1
        retry_delay_s = collector_retry_delay_seconds(
            consecutive_failures,
            initial_delay_s=self._collector_retry_backoff_initial_s,
            factor=self._collector_retry_backoff_factor,
            max_delay_s=self._collector_retry_backoff_cap_s,
        )
        next_retry_epoch_s = timestamp + retry_delay_s if retry_delay_s > 0 else None
        error_message = str(exc)
        self._set_collector_health(
            status=COLLECTOR_HEALTH_UNHEALTHY,
            telemetry_partial=True,
            last_error=error_message,
            consecutive_failures=consecutive_failures,
            next_retry_epoch_s=next_retry_epoch_s,
        )
        if previous_health.status == COLLECTOR_HEALTH_HEALTHY:
            self._collector_failure_event_count += 1
            self._append_event(
                timestamp=timestamp,
                memory_bytes=self._status_memory_value(),
                event_type="collector_degraded",
                context="Collector unavailable; telemetry paused until recovery.",
                metadata={
                    "collector_transition": "degraded",
                    "collector_degraded_from": previous_health.status,
                    "collector_degradation_reason": error_message,
                    "collector_retry_delay_s": retry_delay_s,
                },
            )
        if self.enable_logging:
            logger.warning("Could not get memory usage: %s", error_message)

    def _transition_to_success(self, timestamp: float) -> None:
        previous_health = self._collector_health
        if previous_health.status != COLLECTOR_HEALTH_HEALTHY:
            previous_error = previous_health.last_error
            previous_failures = previous_health.consecutive_failures
            self._set_collector_health(
                status=COLLECTOR_HEALTH_HEALTHY,
                telemetry_partial=False,
            )
            self._collector_failure_event_count += 1
            self._append_event(
                timestamp=timestamp,
                memory_bytes=self._status_memory_value(),
                event_type="collector_recovered",
                context="Collector recovered; full telemetry sampling resumed.",
                metadata={
                    "collector_transition": "recovered",
                    "collector_recovered_from": previous_health.status,
                    "collector_previous_error": previous_error,
                    "collector_previous_failure_count": previous_failures,
                },
            )
        else:
            self._set_collector_health(
                status=COLLECTOR_HEALTH_HEALTHY,
                telemetry_partial=False,
            )

    def _run_tracking_iteration(self) -> None:
        current_time = time.time()
        if not self._retry_collection_due(current_time):
            return
        if not self._ensure_device_memory_available(current_time):
            return

        try:
            current_memory = self._get_current_memory_bytes()
        except Exception as exc:
            self._transition_to_failure(current_time, exc)
            return
        if current_memory is None:
            return
        self._last_successful_memory_bytes = current_memory
        self._transition_to_success(current_time)

        with self._lock:
            if len(self._memory_usage) == self.max_history:
                self._history_dropped_samples += 1
            self._memory_usage.append(current_memory)
            self._timestamps.append(current_time)

            self._total_samples_observed += 1
            self._peak_memory_bytes = max(self._peak_memory_bytes, current_memory)
            if self._min_memory_bytes is None:
                self._min_memory_bytes = current_memory
            else:
                self._min_memory_bytes = min(self._min_memory_bytes, current_memory)
            self._sum_memory_bytes += current_memory

        self._append_event(
            timestamp=current_time,
            memory_bytes=current_memory,
            event_type="sample",
        )

        if self.alert_threshold_mb is not None:
            current_memory_mb = current_memory / (1024 * 1024)
            if current_memory_mb > self.alert_threshold_mb:
                self._trigger_alert(current_memory_mb, current_time)

    def _ensure_device_memory_available(self, current_time: float) -> bool:
        if not self._device_memory_available:
            capability = get_device_memory_capability(self._device)
            self._device_memory_available = bool(capability["memory_stats_available"])
            self._device_memory_unavailable_reason = capability["memory_stats_error"]
            if not self._device_memory_available:
                self._transition_to_failure(
                    current_time,
                    RuntimeError(self._device_memory_unavailable_reason),
                )
                return False
            stats = capability["memory_stats"]
            if "bytes_limit" in stats:
                self._device_bytes_limit = int(stats["bytes_limit"])

        return True

    def _tracking_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_tracking_iteration()
                self._flush_telemetry_sink()
                self._stop_event.wait(self.sampling_interval)
            except Exception as e:
                if self.enable_logging:
                    logger.error("Error in tracking loop: %s", e)
                self._flush_telemetry_sink(force=True)
                self._stop_event.wait(self.sampling_interval)

    def _trigger_alert(self, memory_mb: float, timestamp: float) -> None:
        alert = {
            "timestamp": timestamp,
            "memory_mb": memory_mb,
            "threshold_mb": self.alert_threshold_mb,
            "message": f"Memory usage {memory_mb:.1f} MB exceeds threshold {self.alert_threshold_mb:.1f} MB",
        }

        with self._lock:
            if len(self._alerts) == self.max_history:
                self._history_dropped_alerts += 1
            self._alerts.append(alert)

        if self.enable_logging:
            logger.warning(alert["message"])

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                if self.enable_logging:
                    logger.error("Error in alert callback: %s", e)

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered alert callback."""
        try:
            self.alert_callbacks.remove(callback)
        except ValueError:
            pass

    def set_alert_threshold(self, threshold_mb: float) -> None:
        self.alert_threshold_mb = threshold_mb
        if self.enable_logging:
            logger.info("Updated alert threshold to %s MB", threshold_mb)

    def check_alerts(self) -> bool:
        with self._lock:
            recent_alerts = [
                alert
                for alert in self._alerts
                if time.time() - alert["timestamp"] < 10.0
            ]
            return len(recent_alerts) > 0

    def start_tracking(self) -> None:
        if self.tracking:
            if self.enable_logging:
                logger.warning("Tracking already started")
            return

        self._session_start_time = time.time()
        self._session_end_time = None
        self._session_summary = None
        self._phase_state.reset()
        self._ensure_telemetry_sink()
        self._stop_event.clear()

        with self._lock:
            self._reset_history()

        self._last_successful_memory_bytes = None
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )

        self._ensure_session_summary()

        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        self.tracking = True

        self._append_event(
            timestamp=self._session_start_time,
            memory_bytes=self._status_memory_value(),
            event_type="start",
            context="Memory tracking started",
        )

        if self.enable_logging:
            logger.info(
                "Started JAX memory tracking with %ss interval", self.sampling_interval
            )

    def stop_tracking(self) -> TrackingResult:
        if not self.tracking:
            if self.enable_logging:
                logger.warning("Tracking not started")
            return self._create_empty_result()

        self.tracking = False
        self._stop_event.set()

        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)

        self._session_end_time = time.time()
        self._append_event(
            timestamp=self._session_end_time,
            memory_bytes=self._status_memory_value(),
            event_type="stop",
            context="Memory tracking stopped",
        )

        profile_path = None
        if self.save_device_profile_on_stop:
            profile_path = self.save_device_memory_profile_to_dir()

        self._close_telemetry_sink()

        if self._session_summary is not None:
            self._session_summary = finalize_session_summary(
                self._session_summary,
                ended_at_ns=now_ns(),
            )

        self._phase_state.reset()

        result = self._create_tracking_result()
        result.device_memory_profile_path = profile_path

        if self.enable_logging:
            logger.info(
                "Stopped memory tracking. Peak usage: %.1f MB",
                result.peak_memory_bytes / (1024 * 1024),
            )

        return result

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self._get_current_memory()
        except Exception:
            return float(self._last_successful_memory_bytes or 0) / (1024 * 1024)

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            retained_events = len(self._events)
            retained_samples = len(self._memory_usage)
            retained_alerts = len(self._alerts)
            peak_memory, average_memory, min_memory = self._memory_statistics_locked()
            collector_failure_event_count = self._collector_failure_event_count
            dropped_samples = self._history_dropped_samples
            dropped_events = self._history_dropped_events
            dropped_alerts = self._history_dropped_alerts
            tracking_start = self._session_start_time
            tracking_end = self._session_end_time

        tracking_duration = (
            (tracking_end or time.time()) - tracking_start
            if isinstance(tracking_start, (int, float))
            else 0.0
        )
        current_memory_mb = (
            self._last_successful_memory_bytes / (1024 * 1024)
            if self._collector_health.status == COLLECTOR_HEALTH_HEALTHY
            and self._last_successful_memory_bytes is not None
            else None
        )
        return {
            "current_memory_mb": current_memory_mb,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "average_memory_mb": average_memory / (1024 * 1024),
            "min_memory_mb": min_memory / (1024 * 1024),
            "collector_failure_event_count": collector_failure_event_count,
            "total_events": retained_events,
            "tracking_duration_seconds": tracking_duration,
            "history_window_limit": self.max_history,
            "history_retained_samples": retained_samples,
            "history_dropped_samples": dropped_samples,
            "history_retained_events": retained_events,
            "history_dropped_events": dropped_events,
            "history_retained_alerts": retained_alerts,
            "history_dropped_alerts": dropped_alerts,
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
            **self._collector_health.to_dict(),
            "oom_flight_recorder_enabled": self._oom_flight_recorder.config.enabled,
            "last_oom_dump_path": self._last_oom_dump_path,
        }

    def _memory_statistics_locked(self) -> tuple[int, float, int]:
        peak_memory = self._peak_memory_bytes if self._total_samples_observed else 0
        average_memory = (
            self._sum_memory_bytes / self._total_samples_observed
            if self._total_samples_observed
            else 0
        )
        min_memory = (
            self._min_memory_bytes
            if self._min_memory_bytes is not None and self._total_samples_observed
            else 0
        )
        return peak_memory, average_memory, min_memory

    def get_tracking_results(self) -> TrackingResult:
        """Get current tracking results without stopping."""
        return self._create_tracking_result()

    def _create_tracking_result(self) -> TrackingResult:
        with self._lock:
            result_data = self._tracking_result_data()
            if (
                not result_data.retained_memory_usage
                and not result_data.retained_events
                and not result_data.retained_alerts
            ):
                return self._create_empty_result()

            session_start, session_end = self._result_time_range(
                result_data.retained_timestamps
            )

            return TrackingResult(
                start_time=session_start,
                end_time=session_end,
                samples_collected=result_data.total_samples_observed,
                memory_usage=result_data.retained_memory_usage,
                timestamps=result_data.retained_timestamps,
                peak_memory_bytes=(
                    result_data.peak_memory if result_data.total_samples_observed else 0
                ),
                average_memory_bytes=(
                    int(result_data.sum_memory / result_data.total_samples_observed)
                    if result_data.total_samples_observed
                    else 0
                ),
                min_memory_bytes=(
                    result_data.min_memory if result_data.total_samples_observed else 0
                ),
                alert_count=len(result_data.retained_alerts)
                + result_data.dropped_alerts,
                telemetry_events=result_data.retained_events,
                session_summary=self._session_summary,
                history_window_limit=self.max_history,
                history_retained_samples=len(result_data.retained_memory_usage),
                history_dropped_samples=result_data.dropped_samples,
                history_retained_events=len(result_data.retained_events),
                history_dropped_events=result_data.dropped_events,
                history_retained_alerts=len(result_data.retained_alerts),
                history_dropped_alerts=result_data.dropped_alerts,
                device_memory_available=self._device_memory_available,
                memory_source=(
                    "jax_device_stats"
                    if self._device_memory_available
                    else "unavailable"
                ),
                device_memory_unavailable_reason=self._device_memory_unavailable_reason,
                process_memory_bytes=self._get_current_cpu_memory(),
            )

    def _result_time_range(self, timestamps: List[float]) -> tuple[float, float]:
        session_start = self._session_start_time
        session_end = self._session_end_time
        if session_start is None:
            session_start = timestamps[0] if timestamps else time.time()
        if session_end is None:
            session_end = timestamps[-1] if timestamps else time.time()

        return session_start, session_end

    def _create_empty_result(self) -> TrackingResult:
        current_time = time.time()
        start_time = self._session_start_time or current_time
        end_time = self._session_end_time or start_time
        return TrackingResult(
            start_time=start_time,
            end_time=end_time,
            samples_collected=0,
            memory_usage=[],
            timestamps=[],
            peak_memory_bytes=0,
            average_memory_bytes=0,
            min_memory_bytes=0,
            alert_count=0,
            session_summary=self._session_summary,
            history_window_limit=self.max_history,
            device_memory_available=self._device_memory_available,
            memory_source=(
                "jax_device_stats" if self._device_memory_available else "unavailable"
            ),
            device_memory_unavailable_reason=self._device_memory_unavailable_reason,
            process_memory_bytes=self._get_current_cpu_memory(),
        )

    # -- Telemetry Sink Management -----------------------------------------

    def _append_to_telemetry_sink(self, record: Dict[str, Any]) -> None:
        if self._telemetry_sink is None:
            return
        try:
            self._telemetry_sink.append(record)
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
            "Disabling JAX telemetry sink after %s failure: %s",
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
                "JAX telemetry sink close failed after %s error: %s",
                operation,
                close_exc,
            )

    @staticmethod
    def _close_sink_with_status(sink: Any, status: str) -> None:
        try:
            sink.close(session_status=status)
        except TypeError:
            sink.close()

    # -- Phases ------------------------------------------------------------

    def enter_phase(
        self, name: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> PhaseHandle:
        if not self.tracking:
            raise RuntimeError("Tracking must be active before entering a phase.")
        session = self._ensure_session_summary()
        token, boundary = self._phase_state.enter(
            session_id=session.session_id,
            rank=self.distributed_identity["rank"],
            name=name,
            attrs=metadata,
        )
        self._append_event(
            timestamp=time.time(),
            memory_bytes=self._status_memory_value(),
            event_type=boundary.event_type,
            context=boundary.context,
            metadata=boundary.metadata,
        )
        return PhaseHandle(
            scope_id=boundary.scope_id,
            name=name,
            path=boundary.path,
            close_callback=lambda: self._emit_phase_exit(token),
        )

    @contextmanager
    def phase(
        self, name: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[PhaseHandle]:
        handle = self.enter_phase(name, metadata=metadata)
        try:
            yield handle
        finally:
            handle.close()

    def _emit_phase_exit(self, token: PhaseToken) -> None:
        boundary = self._phase_state.exit(token)
        self._append_event(
            timestamp=time.time(),
            memory_bytes=self._status_memory_value(),
            event_type=boundary.event_type,
            context=boundary.context,
            metadata=boundary.metadata,
        )

    # -- Context manager protocol ------------------------------------------

    def __enter__(self) -> "MemoryTracker":
        self.start_tracking()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop_tracking()

    # -- OOM Flight Recorder -----------------------------------------------

    @property
    def last_oom_dump_path(self) -> Optional[str]:
        """Path to the most recent OOM dump bundle, or None."""
        return self._last_oom_dump_path

    @last_oom_dump_path.setter
    def last_oom_dump_path(self, value: Optional[str]) -> None:
        self._last_oom_dump_path = value

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
            "sampling_interval_s": self.sampling_interval,
            "session_id": None,
            "job_id": self.distributed_identity["job_id"],
            "rank": self.distributed_identity["rank"],
            "local_rank": self.distributed_identity["local_rank"],
            "world_size": self.distributed_identity["world_size"],
        }
        if metadata:
            dump_metadata.update(metadata)

        current_memory = self._status_memory_value()
        reserved_bytes = self._last_reserved_bytes
        is_approximate = False
        if reserved_bytes is None:
            reserved_bytes = current_memory
            is_approximate = True

        dump_metadata.update(
            {
                "sample_allocated_bytes": current_memory,
                "sample_reserved_bytes": reserved_bytes,
                "sample_device_id": self.device_index,
                "sample_total_bytes": self._device_bytes_limit,
            }
        )
        if is_approximate:
            dump_metadata["allocator_reserved_approximate"] = True
        self._append_event(
            timestamp=time.time(),
            memory_bytes=current_memory,
            event_type="error",
            context=f"OOM detected ({classification.reason})",
            metadata={"oom_reason": classification.reason},
        )

        session_summary = self._session_summary
        dump_metadata["tracker_stats"] = self.get_statistics()
        dump_metadata["session_id"] = (
            session_summary.session_id if session_summary is not None else None
        )

        try:
            dump_path = self._oom_flight_recorder.dump(
                reason=classification.reason,
                exception=exc,
                context=context,
                backend="jax",
                metadata=dump_metadata,
                session_summary=session_summary,
            )
        except Exception as dump_exc:
            logger.debug("OOM flight recorder dump failed: %s", dump_exc)
            return None

        self._enrich_oom_dump(dump_path)

        self._last_oom_dump_path = dump_path
        return dump_path

    def _enrich_oom_dump(self, dump_path: Optional[str]) -> None:
        # Enrich the dump with a JAX device memory profile artifact
        if dump_path is not None:
            profile_path = self.save_device_memory_profile_to_dir(
                output_dir=dump_path,
            )
            if profile_path:
                try:
                    manifest_path = Path(dump_path) / "manifest.json"
                    if manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        files = list(manifest.get("files", []))
                        profile_name = Path(profile_path).name
                        if profile_name not in files:
                            files.append(profile_name)

                        # Automatically generate the HTML visualization
                        try:
                            from .attributed_viz import render_jax_attributed_html
                            from .pprof_parser import parse_jax_memory_profile

                            html_path = str(
                                Path(dump_path) / "jax-device-memory-graph.html"
                            )
                            profile_data = parse_jax_memory_profile(profile_path)
                            render_jax_attributed_html(profile_data, html_path)

                            html_name = Path(html_path).name
                            if html_name not in files:
                                files.append(html_name)
                            manifest["jax_device_profile_html"] = html_name
                        except Exception as viz_exc:
                            logger.debug(
                                "Could not generate HTML memory graph: %s", viz_exc
                            )

                        manifest["files"] = files
                        manifest["jax_device_profile"] = True
                        manifest_path.write_text(
                            json.dumps(manifest, indent=2),
                            encoding="utf-8",
                        )
                except Exception as enrich_exc:
                    logger.debug(
                        "Could not enrich OOM manifest with profile: %s",
                        enrich_exc,
                    )

    @contextmanager
    def capture_oom(
        self,
        context: str = "runtime",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """Capture an OOM diagnostic bundle if the wrapped block raises an OOM."""
        try:
            yield
        except Exception as exc:
            dump_path = self.handle_exception(exc, context=context, metadata=metadata)
            if dump_path:
                logger.error("OOM flight recorder dump saved to: %s", dump_path)
            raise

    def trigger_oom_dump(
        self,
        exception: BaseException,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Manually trigger an OOM diagnostic dump bundle."""
        return self.handle_exception(exception, context=context, metadata=metadata)

    # -- Device Memory Profile Export --------------------------------------

    def save_device_memory_profile(self, output_path: str) -> bool:
        """Save a JAX device memory profile to the given path.

        .. note::

            This method depends on ``jax.profiler.save_device_memory_profile``
            which is only available on GPU/TPU backends with JAX >= 0.4.1.
            On CPU-only installs or older JAX versions the call is a no-op
            and returns ``False``.  The availability is checked at runtime
            via ``hasattr`` guards so no import error is raised.
        """
        if not JAX_AVAILABLE:
            return False

        try:
            if hasattr(jax, "profiler") and hasattr(
                jax.profiler, "save_device_memory_profile"
            ):
                jax.profiler.save_device_memory_profile(output_path)
                if self.enable_logging:
                    logger.info("Saved JAX device memory profile to %s", output_path)
                return True
            else:
                logger.warning(
                    "jax.profiler.save_device_memory_profile is not available."
                )
                return False
        except Exception as exc:
            logger.error("Failed to save JAX device memory profile: %s", exc)
            return False

    def save_device_memory_profile_to_dir(
        self, output_dir: Optional[str] = None
    ) -> Optional[str]:
        """Save a JAX device memory profile to a directory with an auto-generated filename."""
        if output_dir is None:
            output_dir = os.getcwd()

        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            filename = f"jax-device-memory-{timestamp}.prof"
            filepath = os.path.join(output_dir, filename)

            if self.save_device_memory_profile(filepath):
                return filepath
        except Exception as exc:
            logger.error("Failed to setup directory for memory profile: %s", exc)

        return None


class MemoryWatchdog:
    """Automatic memory management and cleanup for JAX workloads."""

    def __init__(
        self,
        max_memory_mb: float = 8000,
        cleanup_threshold_mb: float = 6000,
        check_interval: float = 5.0,
        device_index: int = 0,
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available.")

        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.check_interval = check_interval

        self._device = None
        try:
            devices = jax.local_devices()
            if device_index < len(devices):
                self._device = devices[device_index]
        except Exception:
            pass

        self.active = False
        self.watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.cleanup_callbacks: List[Callable[[], None]] = []

        logger.info("JAX Memory Watchdog initialized with %s MB limit", max_memory_mb)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def _get_memory_usage(self) -> float:
        if self._device is None:
            return 0.0
        try:
            stats = self._device.memory_stats()
            if stats:
                return int(stats.get("bytes_in_use", 0)) / (1024 * 1024)
            return 0.0
        except Exception as exc:
            logger.debug("Watchdog could not get device memory: %s", exc)
            return 0.0

    def _cleanup_memory(self) -> None:
        try:
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()

            import gc

            gc.collect()

            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error("Error in cleanup callback: %s", e)

            logger.info("Performed JAX memory cleanup")
        except Exception as e:
            logger.error("Error during JAX memory cleanup: %s", e)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                current_memory = self._get_memory_usage()

                if current_memory > self.max_memory_mb:
                    logger.warning(
                        "Memory usage %.1f MB exceeds limit %s MB - forcing cleanup",
                        current_memory,
                        self.max_memory_mb,
                    )
                    self._cleanup_memory()
                elif current_memory > self.cleanup_threshold_mb:
                    logger.info(
                        "Memory usage %.1f MB above threshold %s MB - performing cleanup",
                        current_memory,
                        self.cleanup_threshold_mb,
                    )
                    self._cleanup_memory()

                self._stop_event.wait(self.check_interval)
            except Exception as e:
                logger.error("Error in watchdog loop: %s", e)
                break

    def start(self) -> None:
        """Start memory watchdog."""
        if self.active:
            logger.warning("Watchdog already active")
            return
        self.active = True
        self._stop_event.clear()
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        logger.info("Started JAX memory watchdog")

    def stop(self) -> None:
        """Stop memory watchdog."""
        if not self.active:
            return
        self.active = False
        self._stop_event.set()
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5.0)
        logger.info("Stopped JAX memory watchdog")

    def force_cleanup(self, aggressive: bool = False) -> None:
        """Force immediate memory cleanup.

        Args:
            aggressive: When *True*, also delete all live JAX arrays
                reachable via ``jax.live_arrays()`` (if available) before
                running garbage collection.  Use with caution — this can
                invalidate arrays still referenced by user code.
        """
        if aggressive:
            try:
                if hasattr(jax, "live_arrays"):
                    for arr in jax.live_arrays():
                        try:
                            arr.delete()
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("Aggressive cleanup of live arrays failed: %s", e)
        self._cleanup_memory()


# Aliases for backward compatibility
JAXMemoryTracker = MemoryTracker
