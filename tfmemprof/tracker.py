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

from gpumemprof.telemetry import telemetry_event_from_record, telemetry_event_to_dict


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


class MemoryTracker:
    """Real-time TensorFlow GPU memory tracker."""

    def __init__(
        self,
        sampling_interval: float = 1.0,
        alert_threshold_mb: Optional[float] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
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

        self.sampling_interval = sampling_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.device = device or "/GPU:0"
        self.enable_logging = enable_logging

        # Tracking state
        self.tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        self.memory_usage: List[float] = []
        self.timestamps: List[float] = []
        self.events: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        if enable_logging:
            logging.info(f"TensorFlow Memory Tracker initialized for {self.device}")

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
            "timestamp": timestamp,
            "type": event_type,
            "memory_mb": memory_mb,
            "device_id": self._device_id(),
            "context": context,
            "metadata": metadata or {},
            "collector": "tfmemprof.memory_tracker",
            "sampling_interval_ms": sampling_interval_ms,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        event = telemetry_event_from_record(
            legacy,
            default_collector="tfmemprof.memory_tracker",
            default_sampling_interval_ms=sampling_interval_ms,
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
        try:
            if "/GPU:" in self.device:
                # Extract GPU index from device string
                gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                memory_info = tf.config.experimental.get_memory_info(f"/GPU:{gpu_id}")
                current_bytes = memory_info.get("current", 0)
                if isinstance(current_bytes, (int, float)):
                    return float(current_bytes) / (1024 * 1024)
                return 0.0
            else:
                # CPU memory tracking
                import psutil

                process = psutil.Process()
                return float(process.memory_info().rss) / (1024 * 1024)
        except Exception as e:
            if self.enable_logging:
                logging.warning(f"Could not get memory usage: {e}")
            return 0.0

    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                # Sample memory
                current_memory = self._get_current_memory()
                current_time = time.time()

                with self._lock:
                    self.memory_usage.append(current_memory)
                    self.timestamps.append(current_time)

                    self.events.append(
                        self._build_telemetry_event_record(
                            timestamp=current_time,
                            memory_mb=current_memory,
                            event_type="sample",
                        )
                    )

                # Check for alerts
                if self.alert_threshold_mb and current_memory > self.alert_threshold_mb:
                    self._trigger_alert(current_memory, current_time)

                # Wait for next sample
                self._stop_event.wait(self.sampling_interval)

            except Exception as e:
                if self.enable_logging:
                    logging.error(f"Error in tracking loop: {e}")
                break

    def _trigger_alert(self, memory_mb: float, timestamp: float) -> None:
        """Trigger memory usage alert."""
        alert = {
            "timestamp": timestamp,
            "memory_mb": memory_mb,
            "threshold_mb": self.alert_threshold_mb,
            "message": f"Memory usage {memory_mb:.1f} MB exceeds threshold {self.alert_threshold_mb:.1f} MB",
        }

        with self._lock:
            self.alerts.append(alert)

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

        self.tracking = True
        self._stop_event.clear()

        # Reset tracking data
        with self._lock:
            self.memory_usage.clear()
            self.timestamps.clear()
            self.events.clear()
            self.alerts.clear()

        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()

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
            if not self.memory_usage:
                return self._create_empty_result()

            start_time = self.timestamps[0] if self.timestamps else time.time()
            end_time = self.timestamps[-1] if self.timestamps else time.time()

            return TrackingResult(
                start_time=start_time,
                end_time=end_time,
                memory_usage=self.memory_usage.copy(),
                timestamps=self.timestamps.copy(),
                events=self.events.copy(),
                alerts_triggered=self.alerts.copy(),
                peak_memory=max(self.memory_usage) if self.memory_usage else 0.0,
                average_memory=(
                    sum(self.memory_usage) / len(self.memory_usage)
                    if self.memory_usage
                    else 0.0
                ),
                min_memory=min(self.memory_usage) if self.memory_usage else 0.0,
            )

    def _create_empty_result(self) -> TrackingResult:
        """Create empty tracking result."""
        current_time = time.time()
        return TrackingResult(
            start_time=current_time,
            end_time=current_time,
            memory_usage=[],
            timestamps=[],
            events=[],
            alerts_triggered=[],
            peak_memory=0.0,
            average_memory=0.0,
            min_memory=0.0,
        )

    def get_current_memory(self) -> float:
        """Get current memory usage."""
        return self._get_current_memory()

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
                for alert in self.alerts
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
