"""
Real-time TensorFlow Memory Tracking

This module provides real-time monitoring of GPU memory usage during TensorFlow
model training and inference, with configurable alerts and automatic cleanup.
"""

import logging
import os
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from stormlog.collector_health import (
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    CollectorHealthState,
    collector_retry_delay_seconds,
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


@dataclass
class TrackingResult:
    """Results from real-time memory tracking."""

    start_time: float
    end_time: float
    memory_usage: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    alerts_triggered: List[Dict] = field(default_factory=list)
    peak_memory: float = 0.0
    average_memory: float = 0.0
    min_memory: float = float("inf")
    history_window_limit: int = 0
    history_retained_samples: int = 0
    history_dropped_samples: int = 0
    history_retained_events: int = 0
    history_dropped_events: int = 0
    history_retained_alerts: int = 0
    history_dropped_alerts: int = 0

    @property
    def duration(self) -> float:
        """Total tracking duration."""
        return self.end_time - self.start_time

    @property
    def memory_growth_rate(self) -> float:
        """Memory growth rate in MB/second."""
        if len(self.memory_usage) < 2 or self.duration <= 0:
            return 0.0
        return (self.memory_usage[-1] - self.memory_usage[0]) / self.duration


@dataclass(frozen=True)
class _TrackingResultData:
    retained_memory_usage: List[float]
    retained_timestamps: List[float]
    retained_events: List[Dict[str, Any]]
    retained_alerts: List[Dict[str, Any]]
    total_samples_observed: int
    peak_memory: float
    min_memory: float
    sum_memory: float
    dropped_samples: int
    dropped_events: int
    dropped_alerts: int


class MemoryTracker:
    """Real-time TensorFlow GPU memory tracker."""

    def __init__(
        self,
        sampling_interval: float = 1.0,
        alert_threshold_mb: Optional[float] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        max_history: int = 10_000,
        job_id: Optional[str] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        telemetry_sink_config: Optional[TelemetrySinkConfig] = None,
    ):
        """
        Initialize memory tracker.

        Args:
            sampling_interval: Time between memory samples in seconds
            alert_threshold_mb: Memory threshold for alerts in MB
            device: TensorFlow device to monitor (e.g., '/GPU:0')
            enable_logging: Whether to log events
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Please install TensorFlow.")
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be > 0")
        if max_history <= 0:
            raise ValueError("max_history must be >= 1")

        self.sampling_interval = sampling_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.device = device or self._get_default_device()
        self.enable_logging = enable_logging
        self.max_history = max_history
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
        self.session_source = "stormlog.tensorflow.tracker"
        self._session_summary: Optional[SessionSummary] = None

        # Tracking state
        self.tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        self._memory_usage: deque[float] = deque(maxlen=max_history)
        self._timestamps: deque[float] = deque(maxlen=max_history)
        self._events: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._alerts: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._history_dropped_samples = 0
        self._history_dropped_events = 0
        self._history_dropped_alerts = 0
        self._collector_failure_event_count = 0
        self._total_samples_observed = 0
        self._peak_memory_mb = 0.0
        self._min_memory_mb = float("inf")
        self._sum_memory_mb = 0.0
        self._last_sink_diagnostics: Dict[str, int] = self._empty_sink_diagnostics()
        self._phase_state = PhaseRecorder()

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._collector_health = CollectorHealthState()
        self._last_successful_memory_mb: Optional[float] = None
        self._session_start_time: Optional[float] = None
        self._session_end_time: Optional[float] = None
        self._collector_retry_backoff_initial_s = 1.0
        self._collector_retry_backoff_factor = 2.0
        self._collector_retry_backoff_cap_s = 30.0

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        if enable_logging:
            logging.info(f"TensorFlow Memory Tracker initialized for {self.device}")

    @staticmethod
    def _empty_sink_diagnostics() -> Dict[str, int]:
        return {
            "rollover_count": 0,
            "pruned_segment_count": 0,
            "pruned_bytes": 0,
            "final_retained_files": 0,
            "final_retained_bytes": 0,
        }

    @property
    def memory_usage(self) -> List[float]:
        return list(self._memory_usage)

    @property
    def timestamps(self) -> List[float]:
        return list(self._timestamps)

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    @property
    def alerts(self) -> List[Dict[str, Any]]:
        return list(self._alerts)

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
        self._peak_memory_mb = 0.0
        self._min_memory_mb = float("inf")
        self._sum_memory_mb = 0.0
        self._last_sink_diagnostics = self._empty_sink_diagnostics()

    def _tracking_result_data(self) -> _TrackingResultData:
        return _TrackingResultData(
            retained_memory_usage=list(self._memory_usage),
            retained_timestamps=list(self._timestamps),
            retained_events=list(self._events),
            retained_alerts=list(self._alerts),
            total_samples_observed=self._total_samples_observed,
            peak_memory=self._peak_memory_mb,
            min_memory=self._min_memory_mb,
            sum_memory=self._sum_memory_mb,
            dropped_samples=self._history_dropped_samples,
            dropped_events=self._history_dropped_events,
            dropped_alerts=self._history_dropped_alerts,
        )

    def _append_sample_locked(self, memory_mb: float, timestamp: float) -> None:
        if len(self._memory_usage) == self.max_history:
            self._history_dropped_samples += 1
        self._memory_usage.append(memory_mb)
        self._timestamps.append(timestamp)
        self._total_samples_observed += 1
        self._peak_memory_mb = max(self._peak_memory_mb, memory_mb)
        self._min_memory_mb = min(self._min_memory_mb, memory_mb)
        self._sum_memory_mb += memory_mb

    def _append_event_locked(self, record: Dict[str, Any]) -> None:
        if len(self._events) == self.max_history:
            self._history_dropped_events += 1
        self._events.append(record)

    def _append_alert_locked(self, alert: Dict[str, Any]) -> None:
        if len(self._alerts) == self.max_history:
            self._history_dropped_alerts += 1
        self._alerts.append(alert)

    def _ensure_session_summary(self) -> SessionSummary:
        """Create the active tracking session summary if needed."""
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
        """Return the active or most recent TensorFlow tracking session."""
        return self._session_summary

    def _ensure_telemetry_sink(self) -> None:
        if self._telemetry_sink is None and self._telemetry_sink_config is not None:
            self._telemetry_sink = AppendOnlyTelemetrySink(self._telemetry_sink_config)

    def _device_id(self) -> int:
        """Best-effort device id extraction."""
        if isinstance(self.device, str):
            if "CPU" in self.device.upper():
                return -1
            if ":" in self.device:
                tail = self.device.rsplit(":", 1)[-1]
                if tail.isdigit():
                    return int(tail)
            if "/GPU" in self.device.upper():
                return 0
        return -1

    def _build_telemetry_event_record(
        self,
        *,
        timestamp: float,
        memory_mb: float,
        event_type: str = "sample",
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sampling_interval_ms = int(round(self.sampling_interval * 1000))
        legacy = {
            "session_id": self._ensure_session_summary().session_id,
            "timestamp": timestamp,
            "type": event_type,
            "memory_mb": memory_mb,
            "device_id": self._device_id(),
            "context": context,
            "metadata": {
                **dict(metadata or {}),
                **self._collector_health.to_dict(),
            },
            "collector": "stormlog.tensorflow.memory_tracker",
            "sampling_interval_ms": sampling_interval_ms,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "job_id": self.distributed_identity["job_id"],
            "rank": self.distributed_identity["rank"],
            "local_rank": self.distributed_identity["local_rank"],
            "world_size": self.distributed_identity["world_size"],
        }
        event = telemetry_event_from_record(
            legacy,
            default_collector="stormlog.tensorflow.memory_tracker",
            default_sampling_interval_ms=sampling_interval_ms,
            default_session_id=self._ensure_session_summary().session_id,
        )
        return telemetry_event_to_dict(event)

    def _get_default_device(self) -> str:
        """Get default TensorFlow device."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "/GPU:0"
            else:
                return "/CPU:0"
        except Exception as exc:
            logging.debug("Default device detection failed: %s", exc)
            return "/CPU:0"

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if "/GPU:" in self.device:
            # Extract GPU index from device string
            gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            memory_info = tf.config.experimental.get_memory_info(f"/GPU:{gpu_id}")
            current_bytes = memory_info.get("current", 0)
            if isinstance(current_bytes, (int, float)):
                return float(current_bytes) / (1024 * 1024)
            raise RuntimeError("TensorFlow memory info returned a non-numeric value")

        # CPU memory tracking
        import psutil

        process = psutil.Process()
        return float(process.memory_info().rss) / (1024 * 1024)

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

    def _status_memory_value(self) -> float:
        return float(self._last_successful_memory_mb or 0.0)

    def _append_event(
        self,
        *,
        timestamp: float,
        memory_mb: float,
        event_type: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._build_telemetry_event_record(
            timestamp=timestamp,
            memory_mb=memory_mb,
            event_type=event_type,
            context=context,
            metadata=metadata,
        )
        with self._lock:
            self._append_event_locked(record)
        self._append_to_telemetry_sink(record)

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
                memory_mb=self._status_memory_value(),
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
            logging.warning("Could not get memory usage: %s", error_message)

    def _transition_to_success(self, timestamp: float) -> None:
        previous_health = self._collector_health
        previous_error = previous_health.last_error
        previous_failures = previous_health.consecutive_failures
        if previous_health.status != COLLECTOR_HEALTH_HEALTHY:
            self._set_collector_health(
                status=COLLECTOR_HEALTH_HEALTHY,
                telemetry_partial=False,
            )
            self._collector_failure_event_count += 1
            self._append_event(
                timestamp=timestamp,
                memory_mb=self._status_memory_value(),
                event_type="collector_recovered",
                context="Collector recovered; full telemetry sampling resumed.",
                metadata={
                    "collector_transition": "recovered",
                    "collector_recovered_from": previous_health.status,
                    "collector_previous_error": previous_error,
                    "collector_previous_failure_count": previous_failures,
                },
            )
            return
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )

    def _run_tracking_iteration(self) -> None:
        """Collect one tracking sample or advance degraded-mode state."""
        current_time = time.time()
        if not self._retry_collection_due(current_time):
            return

        try:
            current_memory = self._get_current_memory()
        except Exception as exc:
            self._transition_to_failure(current_time, exc)
            return

        self._last_successful_memory_mb = current_memory
        self._transition_to_success(current_time)

        with self._lock:
            self._append_sample_locked(current_memory, current_time)
        self._append_event(
            timestamp=current_time,
            memory_mb=current_memory,
            event_type="sample",
        )

        if self.alert_threshold_mb and current_memory > self.alert_threshold_mb:
            self._trigger_alert(current_memory, current_time)

    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                self._run_tracking_iteration()
                self._flush_telemetry_sink()
                self._stop_event.wait(self.sampling_interval)
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"Error in tracking loop: {e}")
                self._flush_telemetry_sink(force=True)
                self._stop_event.wait(self.sampling_interval)

    def _trigger_alert(self, memory_mb: float, timestamp: float) -> None:
        """Trigger memory usage alert."""
        alert = {
            "timestamp": timestamp,
            "memory_mb": memory_mb,
            "threshold_mb": self.alert_threshold_mb,
            "message": f"Memory usage {memory_mb:.1f} MB exceeds threshold {self.alert_threshold_mb:.1f} MB",
        }

        with self._lock:
            self._append_alert_locked(alert)

        # Log alert
        if self.enable_logging:
            logging.warning(alert["message"])

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback function for memory alerts."""
        self.alert_callbacks.append(callback)

    def start_tracking(self) -> None:
        """Start real-time memory tracking."""
        if self.tracking:
            if self.enable_logging:
                logging.warning("Tracking already started")
            return

        self._session_start_time = time.time()
        self._session_end_time = None
        self._session_summary = None
        self._phase_state.reset()
        self._ensure_telemetry_sink()
        self._stop_event.clear()

        # Reset tracking data
        with self._lock:
            self._reset_history()
        self._last_successful_memory_mb = None
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )

        self._ensure_session_summary()

        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        self.tracking = True
        self._append_event(
            timestamp=self._session_start_time,
            memory_mb=self._status_memory_value(),
            event_type="start",
            context="Memory tracking started",
        )

        if self.enable_logging:
            logging.info(
                f"Started memory tracking with {self.sampling_interval}s interval"
            )

    def stop_tracking(self) -> TrackingResult:
        """Stop tracking and return results."""
        if not self.tracking:
            if self.enable_logging:
                logging.warning("Tracking not started")
            return self._create_empty_result()

        self.tracking = False
        self._stop_event.set()

        # Wait for tracking thread to finish
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)

        self._session_end_time = time.time()
        self._append_event(
            timestamp=self._session_end_time,
            memory_mb=self._status_memory_value(),
            event_type="stop",
            context="Memory tracking stopped",
        )
        self._close_telemetry_sink()
        if self._session_summary is not None:
            self._session_summary = finalize_session_summary(
                self._session_summary,
                ended_at_ns=now_ns(),
            )
        self._phase_state.reset()
        # Create result
        result = self._create_tracking_result()

        if self.enable_logging:
            logging.info(
                f"Stopped memory tracking. Peak usage: {result.peak_memory:.1f} MB"
            )

        return result

    def _create_tracking_result(self) -> TrackingResult:
        """Create tracking result from collected data."""
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
                memory_usage=result_data.retained_memory_usage,
                timestamps=result_data.retained_timestamps,
                events=result_data.retained_events,
                alerts_triggered=result_data.retained_alerts,
                peak_memory=(
                    result_data.peak_memory
                    if result_data.total_samples_observed
                    else 0.0
                ),
                average_memory=(
                    result_data.sum_memory / result_data.total_samples_observed
                    if result_data.total_samples_observed
                    else 0.0
                ),
                min_memory=(
                    result_data.min_memory
                    if result_data.total_samples_observed
                    else 0.0
                ),
                history_window_limit=self.max_history,
                history_retained_samples=len(result_data.retained_memory_usage),
                history_dropped_samples=result_data.dropped_samples,
                history_retained_events=len(result_data.retained_events),
                history_dropped_events=result_data.dropped_events,
                history_retained_alerts=len(result_data.retained_alerts),
                history_dropped_alerts=result_data.dropped_alerts,
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
        """Create empty tracking result."""
        current_time = time.time()
        start_time = self._session_start_time or current_time
        end_time = self._session_end_time or start_time
        return TrackingResult(
            start_time=start_time,
            end_time=end_time,
            memory_usage=[],
            timestamps=[],
            events=[],
            alerts_triggered=[],
            peak_memory=0.0,
            average_memory=0.0,
            min_memory=0.0,
            history_window_limit=self.max_history,
            history_retained_samples=0,
            history_dropped_samples=0,
            history_retained_events=0,
            history_dropped_events=0,
            history_retained_alerts=0,
            history_dropped_alerts=0,
        )

    def get_current_memory(self) -> float:
        """Get current memory usage."""
        try:
            return self._get_current_memory()
        except Exception:
            return float(self._last_successful_memory_mb or 0.0)

    def get_statistics(self) -> dict[str, Any]:
        """Return current tracker health and latest successful memory sample."""
        with self._lock:
            retained_events = len(self._events)
            retained_samples = len(self._memory_usage)
            retained_alerts = len(self._alerts)
            peak_memory = self._peak_memory_mb if self._total_samples_observed else 0.0
            average_memory = (
                self._sum_memory_mb / self._total_samples_observed
                if self._total_samples_observed
                else 0.0
            )
            min_memory = self._min_memory_mb if self._total_samples_observed else 0.0
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
            self._last_successful_memory_mb
            if self._collector_health.status == COLLECTOR_HEALTH_HEALTHY
            else None
        )
        return {
            "current_memory_mb": current_memory_mb,
            "peak_memory_mb": peak_memory,
            "average_memory_mb": average_memory,
            "min_memory_mb": min_memory,
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
        }

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
        logging.warning(
            "Disabling TensorFlow telemetry sink after %s failure: %s",
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
            logging.debug(
                "TensorFlow telemetry sink close failed after %s error: %s",
                operation,
                close_exc,
            )

    @staticmethod
    def _close_sink_with_status(sink: Any, status: str) -> None:
        try:
            sink.close(session_status=status)
        except TypeError:
            sink.close()

    def enter_phase(
        self, name: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> PhaseHandle:
        """Enter one structured TensorFlow tracking phase."""
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
            memory_mb=self._status_memory_value(),
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
    def phase(self, name: str, *, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Context manager that emits structured TensorFlow phase telemetry."""
        handle = self.enter_phase(name, metadata=metadata)
        try:
            yield handle
        finally:
            handle.close()

    def _emit_phase_exit(self, token: PhaseToken) -> None:
        boundary = self._phase_state.exit(token)
        self._append_event(
            timestamp=time.time(),
            memory_mb=self._status_memory_value(),
            event_type=boundary.event_type,
            context=boundary.context,
            metadata=boundary.metadata,
        )

    def set_alert_threshold(self, threshold_mb: float) -> None:
        """Update alert threshold."""
        self.alert_threshold_mb = threshold_mb
        if self.enable_logging:
            logging.info(f"Updated alert threshold to {threshold_mb} MB")

    def check_alerts(self) -> bool:
        """Check if any alerts have been triggered recently."""
        with self._lock:
            # Check for alerts in the last 10 seconds
            recent_alerts = [
                alert
                for alert in self._alerts
                if time.time() - alert["timestamp"] < 10.0
            ]
            return len(recent_alerts) > 0

    def get_tracking_results(self) -> TrackingResult:
        """Get current tracking results without stopping."""
        return self._create_tracking_result()


class MemoryWatchdog:
    """Automatic memory management and cleanup for TensorFlow."""

    def __init__(
        self,
        max_memory_mb: float = 8000,
        cleanup_threshold_mb: float = 6000,
        check_interval: float = 5.0,
    ):
        """
        Initialize memory watchdog.

        Args:
            max_memory_mb: Maximum memory before forced cleanup
            cleanup_threshold_mb: Memory threshold to trigger cleanup
            check_interval: Time between memory checks in seconds
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available.")

        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.check_interval = check_interval

        self.active = False
        self.watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Cleanup callbacks
        self.cleanup_callbacks: List[Callable[[], None]] = []

        logging.info(f"Memory Watchdog initialized with {max_memory_mb} MB limit")

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                memory_info = tf.config.experimental.get_memory_info("/GPU:0")
                current_bytes = memory_info.get("current", 0)
                if isinstance(current_bytes, (int, float)):
                    return float(current_bytes) / (1024 * 1024)
                return 0.0
            return 0.0
        except Exception as exc:
            logging.debug("Watchdog could not get GPU memory usage: %s", exc)
            return 0.0

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        try:
            # Clear TensorFlow session
            tf.keras.backend.clear_session()

            # Force garbage collection
            import gc

            gc.collect()

            # Call custom cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logging.error(f"Error in cleanup callback: {e}")

            logging.info("Performed memory cleanup")

        except Exception as e:
            logging.error(f"Error during memory cleanup: {e}")

    def _watchdog_loop(self) -> None:
        """Main watchdog loop."""
        while not self._stop_event.is_set():
            try:
                current_memory = self._get_memory_usage()

                if current_memory > self.max_memory_mb:
                    logging.warning(
                        f"Memory usage {current_memory:.1f} MB exceeds limit {self.max_memory_mb} MB - forcing cleanup"
                    )
                    self._cleanup_memory()

                elif current_memory > self.cleanup_threshold_mb:
                    logging.info(
                        f"Memory usage {current_memory:.1f} MB above threshold {self.cleanup_threshold_mb} MB - performing cleanup"
                    )
                    self._cleanup_memory()

                # Wait for next check
                self._stop_event.wait(self.check_interval)

            except Exception as e:
                logging.error(f"Error in watchdog loop: {e}")
                break

    def start(self) -> None:
        """Start memory watchdog."""
        if self.active:
            logging.warning("Watchdog already active")
            return

        self.active = True
        self._stop_event.clear()

        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

        logging.info("Started memory watchdog")

    def stop(self) -> None:
        """Stop memory watchdog."""
        if not self.active:
            return

        self.active = False
        self._stop_event.set()

        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5.0)

        logging.info("Stopped memory watchdog")

    def force_cleanup(self) -> None:
        """Force immediate memory cleanup."""
        self._cleanup_memory()
