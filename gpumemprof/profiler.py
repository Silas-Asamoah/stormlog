"""Core GPU Memory Profiler for PyTorch."""

import gc
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot at a specific point in time."""

    timestamp: float
    allocated_memory: int
    reserved_memory: int
    max_memory_allocated: int
    max_memory_reserved: int
    active_memory: int
    inactive_memory: int
    cpu_memory: int
    device_id: int = 0
    operation: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "allocated_memory": self.allocated_memory,
            "reserved_memory": self.reserved_memory,
            "max_memory_allocated": self.max_memory_allocated,
            "max_memory_reserved": self.max_memory_reserved,
            "active_memory": self.active_memory,
            "inactive_memory": self.inactive_memory,
            "cpu_memory": self.cpu_memory,
            "device_id": self.device_id,
            "operation": self.operation,
            "stack_trace": self.stack_trace,
        }


@dataclass
class ProfileResult:
    """Results from profiling a function or operation."""

    function_name: str
    execution_time: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    memory_peak: MemorySnapshot
    memory_allocated: int
    memory_freed: int
    tensors_created: int
    tensors_deleted: int
    call_count: int = 1

    def memory_diff(self) -> int:
        """Calculate memory difference between before and after."""
        return self.memory_after.allocated_memory - self.memory_before.allocated_memory

    def peak_memory_usage(self) -> int:
        """Get peak memory usage during execution."""
        return self.memory_peak.allocated_memory

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "function_name": self.function_name,
            "execution_time": self.execution_time,
            "memory_before": self.memory_before.to_dict(),
            "memory_after": self.memory_after.to_dict(),
            "memory_peak": self.memory_peak.to_dict(),
            "memory_allocated": self.memory_allocated,
            "memory_freed": self.memory_freed,
            "memory_diff": self.memory_diff(),
            "peak_memory_usage": self.peak_memory_usage(),
            "tensors_created": self.tensors_created,
            "tensors_deleted": self.tensors_deleted,
            "call_count": self.call_count,
        }


class GPUMemoryProfiler:
    """Comprehensive GPU memory profiler for PyTorch operations."""

    def __init__(
        self,
        device: Union[str, int, torch.device, None] = None,
        track_tensors: bool = True,
        track_cpu_memory: bool = True,
        collect_stack_traces: bool = False,
    ):
        """
        Initialize the GPU Memory Profiler.

        Args:
            device: GPU device to profile (None for auto-detection)
            track_tensors: Whether to track tensor creation/deletion
            track_cpu_memory: Whether to track CPU memory usage
            collect_stack_traces: Whether to collect stack traces for operations
        """
        self.device = self._setup_device(device)
        self.track_tensors = track_tensors
        self.track_cpu_memory = track_cpu_memory
        self.collect_stack_traces = collect_stack_traces

        self.results: List[ProfileResult] = []
        self.snapshots: List[MemorySnapshot] = []
        self.function_stats: Dict[str, List[ProfileResult]] = defaultdict(list)

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 0.1  # 100ms
        self._tensor_tracker = TensorTracker() if track_tensors else None

        # Initialize baseline measurements
        self._baseline_snapshot = self._take_snapshot("baseline")

    def _setup_device(
        self, device: Union[str, int, torch.device, None]
    ) -> torch.device:
        """Setup and validate the device for profiling."""
        resolved_device: torch.device

        if device is None:
            if torch.cuda.is_available():
                resolved_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                raise RuntimeError("CUDA is not available, cannot profile GPU memory")
        elif isinstance(device, int):
            resolved_device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            resolved_device = torch.device(device)
        else:
            resolved_device = device

        if resolved_device.type != "cuda":
            raise ValueError("Only CUDA devices are supported for GPU memory profiling")

        # Ensure device is available
        device_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
        if device_index >= torch.cuda.device_count():
            raise ValueError(f"Device {resolved_device} is not available")

        return torch.device(f"cuda:{device_index}")

    def _take_snapshot(self, operation: Optional[str] = None) -> MemorySnapshot:
        """Take a memory snapshot at the current moment."""
        torch.cuda.synchronize(self.device)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_memory=torch.cuda.memory_allocated(self.device),
            reserved_memory=torch.cuda.memory_reserved(self.device),
            max_memory_allocated=torch.cuda.max_memory_allocated(self.device),
            max_memory_reserved=torch.cuda.max_memory_reserved(self.device),
            active_memory=torch.cuda.memory_stats(self.device).get(
                "active_bytes.all.current", 0
            ),
            inactive_memory=torch.cuda.memory_stats(self.device).get(
                "inactive_split_bytes.all.current", 0
            ),
            cpu_memory=psutil.virtual_memory().used if self.track_cpu_memory else 0,
            device_id=self.device.index,
            operation=operation,
        )

        if self.collect_stack_traces and operation:
            import traceback

            snapshot.stack_trace = "".join(traceback.format_stack()[-5:])

        return snapshot

    def profile_function(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ProfileResult:
        """
        Profile a single function call.

        Args:
            func: Function to profile
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            ProfileResult with profiling information
        """
        function_name = getattr(func, "__name__", str(func))

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(self.device)

        # Take before snapshot
        memory_before = self._take_snapshot(f"before_{function_name}")

        # Track tensors if enabled
        if self._tensor_tracker:
            tensors_before = self._tensor_tracker.count_tensors()

        # Execute function
        start_time = time.time()

        try:
            _result = func(*args, **kwargs)
            # Ensure all operations complete
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            # Still capture memory state even if function fails
            logger.debug("Profiled function raised, capturing error snapshot: %s", exc)
            memory_after = self._take_snapshot(f"after_{function_name}_error")
            memory_peak = self._take_snapshot(f"peak_{function_name}_error")

            profile_result = ProfileResult(
                function_name=function_name,
                execution_time=time.time() - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                memory_allocated=0,
                memory_freed=0,
                tensors_created=0,
                tensors_deleted=0,
            )

            self.results.append(profile_result)
            self.function_stats[function_name].append(profile_result)
            raise

        end_time = time.time()

        # Take after snapshot
        memory_after = self._take_snapshot(f"after_{function_name}")

        # Get peak memory usage
        memory_stats = torch.cuda.memory_stats(self.device)
        peak_allocated = memory_stats.get(
            "allocated_bytes.all.peak", memory_after.allocated_memory
        )
        memory_peak = MemorySnapshot(
            timestamp=end_time,
            allocated_memory=peak_allocated,
            reserved_memory=memory_stats.get(
                "reserved_bytes.all.peak", memory_after.reserved_memory
            ),
            max_memory_allocated=torch.cuda.max_memory_allocated(self.device),
            max_memory_reserved=torch.cuda.max_memory_reserved(self.device),
            active_memory=memory_stats.get(
                "active_bytes.all.peak", memory_after.active_memory
            ),
            inactive_memory=memory_after.inactive_memory,
            cpu_memory=memory_after.cpu_memory,
            device_id=self.device.index,
            operation=f"peak_{function_name}",
        )

        # Track tensor changes
        tensors_created = 0
        tensors_deleted = 0
        if self._tensor_tracker:
            tensors_after = self._tensor_tracker.count_tensors()
            tensors_created = max(0, tensors_after - tensors_before)
            tensors_deleted = max(0, tensors_before - tensors_after)

        # Create profile result
        profile_result = ProfileResult(
            function_name=function_name,
            execution_time=end_time - start_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=memory_peak,
            memory_allocated=max(
                0, memory_after.allocated_memory - memory_before.allocated_memory
            ),
            memory_freed=max(
                0, memory_before.allocated_memory - memory_after.allocated_memory
            ),
            tensors_created=tensors_created,
            tensors_deleted=tensors_deleted,
        )

        # Store results
        self.results.append(profile_result)
        self.function_stats[function_name].append(profile_result)

        return profile_result

    @contextmanager
    def profile_context(self, name: str = "context") -> Any:
        """
        Context manager for profiling a block of code.

        Args:
            name: Name for the profiled context

        Yields:
            ProfileResult after the context exits
        """
        torch.cuda.reset_peak_memory_stats(self.device)
        memory_before = self._take_snapshot(f"before_{name}")

        if self._tensor_tracker:
            tensors_before = self._tensor_tracker.count_tensors()

        start_time = time.time()

        try:
            yield
            torch.cuda.synchronize(self.device)
        finally:
            end_time = time.time()
            memory_after = self._take_snapshot(f"after_{name}")

            # Get peak memory
            memory_stats = torch.cuda.memory_stats(self.device)
            peak_allocated = memory_stats.get(
                "allocated_bytes.all.peak", memory_after.allocated_memory
            )
            memory_peak = MemorySnapshot(
                timestamp=end_time,
                allocated_memory=peak_allocated,
                reserved_memory=memory_stats.get(
                    "reserved_bytes.all.peak", memory_after.reserved_memory
                ),
                max_memory_allocated=torch.cuda.max_memory_allocated(self.device),
                max_memory_reserved=torch.cuda.max_memory_reserved(self.device),
                active_memory=memory_stats.get(
                    "active_bytes.all.peak", memory_after.active_memory
                ),
                inactive_memory=memory_after.inactive_memory,
                cpu_memory=memory_after.cpu_memory,
                device_id=self.device.index,
                operation=f"peak_{name}",
            )

            # Track tensors
            tensors_created = 0
            tensors_deleted = 0
            if self._tensor_tracker:
                tensors_after = self._tensor_tracker.count_tensors()
                tensors_created = max(0, tensors_after - tensors_before)
                tensors_deleted = max(0, tensors_before - tensors_after)

            profile_result = ProfileResult(
                function_name=name,
                execution_time=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                memory_allocated=max(
                    0, memory_after.allocated_memory - memory_before.allocated_memory
                ),
                memory_freed=max(
                    0, memory_before.allocated_memory - memory_after.allocated_memory
                ),
                tensors_created=tensors_created,
                tensors_deleted=tensors_deleted,
            )

            self.results.append(profile_result)
            self.function_stats[name].append(profile_result)

    def start_monitoring(self, interval: float = 0.1) -> None:
        """
        Start continuous memory monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_interval = interval
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_memory(self) -> None:
        """Background thread for continuous memory monitoring."""
        while self._monitoring:
            snapshot = self._take_snapshot("monitor")
            self.snapshots.append(snapshot)
            time.sleep(self._monitor_interval)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all profiling results."""
        if not self.results:
            return {"message": "No profiling results available"}

        total_functions = len(self.function_stats)
        total_calls = len(self.results)

        # Aggregate statistics
        total_time = sum(r.execution_time for r in self.results)
        total_memory_allocated = sum(r.memory_allocated for r in self.results)
        total_memory_freed = sum(r.memory_freed for r in self.results)
        peak_memory = max(r.peak_memory_usage() for r in self.results)

        # Function statistics
        function_summaries = {}
        for func_name, results in self.function_stats.items():
            function_summaries[func_name] = {
                "call_count": len(results),
                "total_time": sum(r.execution_time for r in results),
                "avg_time": sum(r.execution_time for r in results) / len(results),
                "total_memory_allocated": sum(r.memory_allocated for r in results),
                "avg_memory_allocated": sum(r.memory_allocated for r in results)
                / len(results),
                "peak_memory": max(r.peak_memory_usage() for r in results),
            }

        # Current memory state
        current_snapshot = self._take_snapshot("current")

        return {
            "device": str(self.device),
            "total_functions_profiled": total_functions,
            "total_function_calls": total_calls,
            "total_execution_time": total_time,
            "total_memory_allocated": total_memory_allocated,
            "total_memory_freed": total_memory_freed,
            "net_memory_change": total_memory_allocated - total_memory_freed,
            "peak_memory_usage": peak_memory,
            "current_memory_usage": current_snapshot.allocated_memory,
            "baseline_memory_usage": self._baseline_snapshot.allocated_memory,
            "memory_change_from_baseline": current_snapshot.allocated_memory
            - self._baseline_snapshot.allocated_memory,
            "function_summaries": function_summaries,
            "monitoring_active": self._monitoring,
            "snapshots_collected": len(self.snapshots),
        }

    def clear_results(self) -> None:
        """Clear all profiling results and reset state."""
        self.results.clear()
        self.snapshots.clear()
        self.function_stats.clear()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._baseline_snapshot = self._take_snapshot("new_baseline")

    def __enter__(self) -> "GPUMemoryProfiler":
        """Support for context manager usage."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup when exiting context manager."""
        self.stop_monitoring()


class TensorTracker:
    """Tracks tensor creation and deletion for memory profiling."""

    def __init__(self) -> None:
        self._tensor_count = 0
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Setup hooks to track tensor lifecycle."""
        # Note: This is a simplified version. Full implementation would require
        # more sophisticated tensor tracking using PyTorch's autograd hooks
        pass

    def count_tensors(self) -> int:
        """Count current number of tracked tensors."""
        # Simplified implementation - count all tensors in CUDA memory
        gc.collect()

        tensor_count = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                tensor_count += 1

        return tensor_count
