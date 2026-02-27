"""
Core TensorFlow Stormlog

Main profiling engine for capturing and analyzing GPU memory usage during
TensorFlow model training and inference.
"""

import logging
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional

from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

if TYPE_CHECKING:
    import tensorflow as tf


@dataclass
class MemorySnapshot:
    """Represents a point-in-time memory snapshot."""

    timestamp: float
    name: str
    gpu_memory_mb: float
    cpu_memory_mb: float
    gpu_memory_reserved_mb: float
    gpu_utilization: float
    num_tensors: int
    tensor_sizes: Dict[str, int] = field(default_factory=dict)
    operation_name: Optional[str] = None
    graph_node: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate snapshot data."""
        if self.gpu_memory_mb < 0:
            self.gpu_memory_mb = 0.0
        if self.cpu_memory_mb < 0:
            self.cpu_memory_mb = 0.0


@dataclass
class ProfileResult:
    """Comprehensive profiling results."""

    start_time: float
    end_time: float
    peak_memory_mb: float
    average_memory_mb: float
    min_memory_mb: float
    total_allocations: int
    total_deallocations: int
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    function_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tensor_lifecycle: Dict[str, Any] = field(default_factory=dict)
    memory_fragmentation: float = 0.0
    efficiency_score: float = 0.0

    @property
    def duration(self) -> float:
        """Total profiling duration in seconds."""
        return self.end_time - self.start_time

    @property
    def memory_growth_rate(self) -> float:
        """Memory growth rate in MB/second."""
        if self.duration <= 0:
            return 0.0
        return (self.peak_memory_mb - self.min_memory_mb) / self.duration


class TensorTracker:
    """Tracks TensorFlow tensor lifecycle and memory usage."""

    def __init__(self) -> None:
        self.tensors: weakref.WeakSet[Any] = weakref.WeakSet()
        self.tensor_history: List[Dict[str, Any]] = []
        self.creation_times: Dict[int, float] = {}
        self.tensor_sizes: Dict[int, int] = {}
        self._lock = threading.Lock()

    def track_tensor(
        self, tensor: "tf.Tensor", operation_name: str = "unknown"
    ) -> None:
        """Track a new tensor."""
        if not TF_AVAILABLE or tensor is None:
            return

        with self._lock:
            tensor_id = id(tensor)
            self.tensors.add(tensor)
            self.creation_times[tensor_id] = time.time()

            # Calculate tensor size
            try:
                size_bytes = tensor.numpy().nbytes if hasattr(tensor, "numpy") else 0
                self.tensor_sizes[tensor_id] = size_bytes

                self.tensor_history.append(
                    {
                        "tensor_id": tensor_id,
                        "operation": operation_name,
                        "timestamp": time.time(),
                        "action": "created",
                        "size_bytes": size_bytes,
                        "shape": (
                            tensor.shape.as_list() if hasattr(tensor, "shape") else []
                        ),
                    }
                )
            except Exception as e:
                logging.warning(f"Could not track tensor: {e}")

    def get_active_tensors(self) -> Dict[str, Any]:
        """Get information about currently active tensors."""
        with self._lock:
            active_count = len(self.tensors)
            total_size = sum(self.tensor_sizes.get(id(t), 0) for t in self.tensors)

            return {
                "count": active_count,
                "total_size_mb": total_size / (1024 * 1024),
                "average_size_mb": (
                    (total_size / active_count / (1024 * 1024))
                    if active_count > 0
                    else 0
                ),
            }

    def get_tensor_lifecycle(self) -> List[Dict[str, Any]]:
        """Get complete tensor lifecycle history."""
        with self._lock:
            return self.tensor_history.copy()


class TFMemoryProfiler:
    """Main TensorFlow Stormlog class."""

    def __init__(
        self, device: Optional[str] = None, enable_tensor_tracking: bool = True
    ) -> None:
        """
        Initialize TensorFlow memory profiler.

        Args:
            device: TensorFlow device name (e.g., '/GPU:0', '/CPU:0')
            enable_tensor_tracking: Whether to track individual tensors
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow not available. Please install TensorFlow to use this profiler."
            )

        self.device = device or self._get_default_device()
        self.enable_tensor_tracking = enable_tensor_tracking

        # Initialize components
        self.tensor_tracker: Optional[TensorTracker] = (
            TensorTracker() if enable_tensor_tracking else None
        )
        self.snapshots: List[MemorySnapshot] = []
        self.function_profiles: Dict[str, Dict[str, Any]] = {}
        self.profiling_active = False
        self.profile_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Setup TensorFlow memory growth
        self._setup_tf_memory()

        logging.info(f"TensorFlow Stormlog initialized for device: {self.device}")

    def _get_default_device(self) -> str:
        """Get default TensorFlow device."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "/GPU:0"
            else:
                return "/CPU:0"
        except Exception as exc:
            logging.debug("TF device detection failed: %s", exc)
            return "/CPU:0"

    def _setup_tf_memory(self) -> None:
        """Setup TensorFlow memory growth to avoid OOM errors."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Enabled memory growth for {gpu}")
        except Exception as e:
            logging.warning(f"Could not setup TensorFlow memory growth: {e}")

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            if "/GPU:" in self.device:
                # GPU memory information
                gpu_details = tf.config.experimental.get_memory_info("/GPU:0")
                gpu_memory_mb = gpu_details.get("current", 0) / (1024 * 1024)
                gpu_reserved_mb = gpu_details.get("peak", 0) / (1024 * 1024)
                gpu_utilization = min(
                    100.0,
                    (
                        (gpu_memory_mb / gpu_reserved_mb * 100)
                        if gpu_reserved_mb > 0
                        else 0
                    ),
                )
            else:
                gpu_memory_mb = 0.0
                gpu_reserved_mb = 0.0
                gpu_utilization = 0.0

            # CPU memory (approximate)
            import psutil

            process = psutil.Process()
            cpu_memory_mb = process.memory_info().rss / (1024 * 1024)

            return {
                "gpu_memory_mb": gpu_memory_mb,
                "cpu_memory_mb": cpu_memory_mb,
                "gpu_reserved_mb": gpu_reserved_mb,
                "gpu_utilization": gpu_utilization,
            }

        except Exception as e:
            logging.warning(f"Could not get memory info: {e}")
            return {
                "gpu_memory_mb": 0.0,
                "cpu_memory_mb": 0.0,
                "gpu_reserved_mb": 0.0,
                "gpu_utilization": 0.0,
            }

    def capture_snapshot(self, name: str = "snapshot") -> MemorySnapshot:
        """Capture current memory state."""
        memory_info = self._get_memory_info()

        # Get tensor information
        num_tensors = 0
        tensor_sizes = {}
        if self.tensor_tracker:
            active_tensors = self.tensor_tracker.get_active_tensors()
            num_tensors = active_tensors["count"]
            tensor_sizes = {"total_mb": active_tensors["total_size_mb"]}

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            name=name,
            gpu_memory_mb=memory_info["gpu_memory_mb"],
            cpu_memory_mb=memory_info["cpu_memory_mb"],
            gpu_memory_reserved_mb=memory_info["gpu_reserved_mb"],
            gpu_utilization=memory_info["gpu_utilization"],
            num_tensors=num_tensors,
            tensor_sizes=tensor_sizes,
        )

        with self._lock:
            self.snapshots.append(snapshot)

        return snapshot

    def profile_function(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to profile function memory usage."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__

            # Capture before
            before_snapshot = self.capture_snapshot(f"{func_name}_before")
            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Capture after
                end_time = time.time()
                after_snapshot = self.capture_snapshot(f"{func_name}_after")

                # Calculate metrics
                duration = end_time - start_time
                memory_used = (
                    after_snapshot.gpu_memory_mb - before_snapshot.gpu_memory_mb
                )
                peak_memory = max(
                    before_snapshot.gpu_memory_mb, after_snapshot.gpu_memory_mb
                )

                # Store function profile
                with self._lock:
                    if func_name not in self.function_profiles:
                        self.function_profiles[func_name] = {
                            "calls": 0,
                            "total_duration": 0.0,
                            "total_memory_used": 0.0,
                            "peak_memory": 0.0,
                            "snapshots": [],
                        }

                    profile = self.function_profiles[func_name]
                    profile["calls"] += 1
                    profile["total_duration"] += duration
                    profile["total_memory_used"] += memory_used
                    profile["peak_memory"] = max(profile["peak_memory"], peak_memory)
                    profile["snapshots"].extend([before_snapshot, after_snapshot])

                return result

            except Exception as e:
                # Capture error state
                _error_snapshot = self.capture_snapshot(f"{func_name}_error")
                logging.error(f"Error in profiled function {func_name}: {e}")
                raise

        return wrapper

    @contextmanager
    def profile_context(self, name: str = "context") -> Iterator[None]:
        """Context manager for profiling code blocks."""
        before_snapshot = self.capture_snapshot(f"{name}_start")
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            after_snapshot = self.capture_snapshot(f"{name}_end")

            # Store context profile
            duration = end_time - start_time
            memory_used = after_snapshot.gpu_memory_mb - before_snapshot.gpu_memory_mb

            with self._lock:
                if name not in self.function_profiles:
                    self.function_profiles[name] = {
                        "calls": 0,
                        "total_duration": 0.0,
                        "total_memory_used": 0.0,
                        "peak_memory": 0.0,
                        "snapshots": [],
                    }

                profile = self.function_profiles[name]
                profile["calls"] += 1
                profile["total_duration"] += duration
                profile["total_memory_used"] += memory_used
                profile["peak_memory"] = max(
                    profile["peak_memory"],
                    max(before_snapshot.gpu_memory_mb, after_snapshot.gpu_memory_mb),
                )
                profile["snapshots"].extend([before_snapshot, after_snapshot])

    def start_continuous_profiling(self, interval: float = 1.0) -> None:
        """Start continuous memory profiling."""
        self.profiling_active = True

        def profile_loop() -> None:
            while self.profiling_active:
                self.capture_snapshot("continuous")
                time.sleep(interval)

        self.profile_thread = threading.Thread(target=profile_loop, daemon=True)
        self.profile_thread.start()
        logging.info("Started continuous profiling")

    def stop_continuous_profiling(self) -> None:
        """Stop continuous memory profiling."""
        self.profiling_active = False
        if self.profile_thread:
            self.profile_thread.join(timeout=5.0)
            self.profile_thread = None
        logging.info("Stopped continuous profiling")

    def get_results(self) -> ProfileResult:
        """Get comprehensive profiling results."""
        with self._lock:
            if not self.snapshots:
                # Return empty results
                return ProfileResult(
                    start_time=time.time(),
                    end_time=time.time(),
                    peak_memory_mb=0.0,
                    average_memory_mb=0.0,
                    min_memory_mb=0.0,
                    total_allocations=0,
                    total_deallocations=0,
                    snapshots=[],
                    function_profiles={},
                )

            # Calculate metrics from snapshots
            gpu_memories = [s.gpu_memory_mb for s in self.snapshots]
            peak_memory = max(gpu_memories)
            average_memory = sum(gpu_memories) / len(gpu_memories)
            min_memory = min(gpu_memories)

            # Estimate allocations/deallocations from memory changes
            total_allocations = sum(
                1
                for i in range(1, len(gpu_memories))
                if gpu_memories[i] > gpu_memories[i - 1]
            )
            total_deallocations = sum(
                1
                for i in range(1, len(gpu_memories))
                if gpu_memories[i] < gpu_memories[i - 1]
            )

            # Get tensor lifecycle if available
            tensor_lifecycle = {}
            if self.tensor_tracker:
                tensor_lifecycle = {
                    "history": self.tensor_tracker.get_tensor_lifecycle(),
                    "active": self.tensor_tracker.get_active_tensors(),
                }

            return ProfileResult(
                start_time=self.snapshots[0].timestamp,
                end_time=self.snapshots[-1].timestamp,
                peak_memory_mb=peak_memory,
                average_memory_mb=average_memory,
                min_memory_mb=min_memory,
                total_allocations=total_allocations,
                total_deallocations=total_deallocations,
                snapshots=self.snapshots.copy(),
                function_profiles=self.function_profiles.copy(),
                tensor_lifecycle=tensor_lifecycle,
            )

    def reset(self) -> None:
        """Reset profiler state."""
        with self._lock:
            self.snapshots.clear()
            self.function_profiles.clear()
            if self.tensor_tracker:
                self.tensor_tracker.tensor_history.clear()

        logging.info("Profiler state reset")

    def __enter__(self) -> "TFMemoryProfiler":
        """Context manager entry."""
        self.capture_snapshot("context_start")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.capture_snapshot("context_end")
        if self.profiling_active:
            self.stop_continuous_profiling()
