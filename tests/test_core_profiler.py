"""
Comprehensive tests for the core GPU memory profiler functionality.
"""

import time

import pytest

try:  # Optional dependency: PyTorch
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment w/out torch
    torch = None  # type: ignore[assignment, unused-ignore]

TORCH_AVAILABLE = torch is not None
TORCH_CUDA_AVAILABLE = bool(torch and torch.cuda.is_available())

from gpumemprof import (
    GPUMemoryProfiler,
    MemorySnapshot,
    ProfileResult,
    profile_context,
    profile_function,
)


class TestMemorySnapshot:
    """Test MemorySnapshot functionality."""

    def test_memory_snapshot_creation(self) -> None:
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_memory=1024 * 1024 * 100,  # 100MB
            reserved_memory=1024 * 1024 * 150,  # 150MB
            max_memory_allocated=1024 * 1024 * 120,
            max_memory_reserved=1024 * 1024 * 200,
            active_memory=1024 * 1024 * 90,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 500,  # 500MB
            device_id=0,
        )

        assert snapshot.allocated_memory == 1024 * 1024 * 100
        assert snapshot.reserved_memory == 1024 * 1024 * 150
        assert snapshot.device_id == 0

    def test_memory_snapshot_to_dict(self) -> None:
        """Test converting snapshot to dictionary."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_memory=1024 * 1024 * 100,
            reserved_memory=1024 * 1024 * 150,
            max_memory_allocated=1024 * 1024 * 120,
            max_memory_reserved=1024 * 1024 * 200,
            active_memory=1024 * 1024 * 90,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 500,
            device_id=0,
        )

        snapshot_dict = snapshot.to_dict()
        assert isinstance(snapshot_dict, dict)
        assert "timestamp" in snapshot_dict
        assert "allocated_memory" in snapshot_dict
        assert snapshot_dict["device_id"] == 0


class TestProfileResult:
    """Test ProfileResult functionality."""

    def test_profile_result_creation(self) -> None:
        """Test creating a profile result."""
        before_snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_memory=1024 * 1024 * 100,
            reserved_memory=1024 * 1024 * 150,
            max_memory_allocated=1024 * 1024 * 120,
            max_memory_reserved=1024 * 1024 * 200,
            active_memory=1024 * 1024 * 90,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 500,
            device_id=0,
        )

        after_snapshot = MemorySnapshot(
            timestamp=time.time() + 1,
            allocated_memory=1024 * 1024 * 200,
            reserved_memory=1024 * 1024 * 250,
            max_memory_allocated=1024 * 1024 * 220,
            max_memory_reserved=1024 * 1024 * 300,
            active_memory=1024 * 1024 * 190,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 600,
            device_id=0,
        )

        peak_snapshot = MemorySnapshot(
            timestamp=time.time() + 0.5,
            allocated_memory=1024 * 1024 * 250,
            reserved_memory=1024 * 1024 * 300,
            max_memory_allocated=1024 * 1024 * 250,
            max_memory_reserved=1024 * 1024 * 350,
            active_memory=1024 * 1024 * 240,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 650,
            device_id=0,
        )

        result = ProfileResult(
            function_name="test_function",
            execution_time=1.0,
            memory_before=before_snapshot,
            memory_after=after_snapshot,
            memory_peak=peak_snapshot,
            memory_allocated=1024 * 1024 * 100,
            memory_freed=0,
            tensors_created=5,
            tensors_deleted=0,
        )

        assert result.function_name == "test_function"
        assert result.execution_time == 1.0
        assert result.memory_diff() == 1024 * 1024 * 100  # 200MB - 100MB
        assert result.peak_memory_usage() == 1024 * 1024 * 250

    def test_profile_result_to_dict(self) -> None:
        """Test converting profile result to dictionary."""
        before_snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_memory=1024 * 1024 * 100,
            reserved_memory=1024 * 1024 * 150,
            max_memory_allocated=1024 * 1024 * 120,
            max_memory_reserved=1024 * 1024 * 200,
            active_memory=1024 * 1024 * 90,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 500,
            device_id=0,
        )

        after_snapshot = MemorySnapshot(
            timestamp=time.time() + 1,
            allocated_memory=1024 * 1024 * 200,
            reserved_memory=1024 * 1024 * 250,
            max_memory_allocated=1024 * 1024 * 220,
            max_memory_reserved=1024 * 1024 * 300,
            active_memory=1024 * 1024 * 190,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 600,
            device_id=0,
        )

        peak_snapshot = MemorySnapshot(
            timestamp=time.time() + 0.5,
            allocated_memory=1024 * 1024 * 250,
            reserved_memory=1024 * 1024 * 300,
            max_memory_allocated=1024 * 1024 * 250,
            max_memory_reserved=1024 * 1024 * 350,
            active_memory=1024 * 1024 * 240,
            inactive_memory=1024 * 1024 * 10,
            cpu_memory=1024 * 1024 * 650,
            device_id=0,
        )

        result = ProfileResult(
            function_name="test_function",
            execution_time=1.0,
            memory_before=before_snapshot,
            memory_after=after_snapshot,
            memory_peak=peak_snapshot,
            memory_allocated=1024 * 1024 * 100,
            memory_freed=0,
            tensors_created=5,
            tensors_deleted=0,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "function_name" in result_dict
        assert "execution_time" in result_dict
        assert result_dict["function_name"] == "test_function"


@pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemoryProfiler:
    """Test GPU Memory Profiler functionality."""

    def setup_method(self) -> None:
        """Setup for each test method."""
        self.profiler = GPUMemoryProfiler()
        # Clear any existing allocations
        torch.cuda.empty_cache()

    def teardown_method(self) -> None:
        """Cleanup after each test method."""
        if hasattr(self, "profiler"):
            self.profiler.stop_monitoring()
            self.profiler.clear_results()
        torch.cuda.empty_cache()

    def test_profiler_initialization(self) -> None:
        """Test profiler initialization."""
        assert self.profiler.device.type == "cuda"
        assert self.profiler.results == []
        assert self.profiler.snapshots == []
        assert self.profiler._baseline_snapshot is not None

    def test_take_snapshot(self) -> None:
        """Test taking a memory snapshot."""
        snapshot = self.profiler._take_snapshot("test_operation")

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.operation == "test_operation"
        assert snapshot.device_id == self.profiler.device.index
        assert snapshot.timestamp > 0

    def test_profile_simple_function(self) -> None:
        """Test profiling a simple function."""

        def simple_function() -> object:
            tensor = torch.zeros(1000, 1000, device="cuda")
            return tensor

        result = self.profiler.profile_function(simple_function)

        assert isinstance(result, ProfileResult)
        assert result.function_name == "simple_function"
        assert result.execution_time > 0
        assert result.memory_allocated >= 0
        assert len(self.profiler.results) == 1

    def test_profile_function_with_memory_allocation(self) -> None:
        """Test profiling a function that allocates memory."""

        def allocate_memory() -> object:
            # Allocate approximately 100MB
            tensor = torch.randn(5000, 5000, device="cuda")
            return tensor.sum()

        result = self.profiler.profile_function(allocate_memory)

        assert result.memory_allocated > 50 * 1024 * 1024  # At least 50MB
        assert result.execution_time > 0

    def test_profile_context_manager(self) -> None:
        """Test profiling with context manager."""
        with self.profiler.profile_context("test_context"):
            tensor = torch.randn(2000, 2000, device="cuda")
            result = tensor.sum()

        assert len(self.profiler.results) == 1
        assert self.profiler.results[0].function_name == "test_context"

    def test_monitoring_functionality(self) -> None:
        """Test continuous monitoring."""
        self.profiler.start_monitoring(interval=0.1)

        # Wait for a few monitoring cycles
        time.sleep(0.5)

        # Create some memory activity
        tensor = torch.randn(1000, 1000, device="cuda")

        time.sleep(0.3)

        self.profiler.stop_monitoring()

        # Should have collected some snapshots
        assert len(self.profiler.snapshots) > 0

    def test_get_summary(self) -> None:
        """Test getting profiling summary."""

        # Profile a simple function
        def test_function() -> object:
            return torch.randn(1000, 1000, device="cuda")

        self.profiler.profile_function(test_function)

        summary = self.profiler.get_summary()

        assert isinstance(summary, dict)
        assert "total_functions_profiled" in summary
        assert "total_function_calls" in summary
        assert summary["total_function_calls"] == 1
        assert summary["total_functions_profiled"] == 1

    def test_clear_results(self) -> None:
        """Test clearing profiling results."""

        def test_function() -> object:
            return torch.randn(100, 100, device="cuda")

        self.profiler.profile_function(test_function)
        assert len(self.profiler.results) == 1

        self.profiler.clear_results()
        assert len(self.profiler.results) == 0
        assert len(self.profiler.snapshots) == 0

    def test_context_manager_interface(self) -> None:
        """Test using profiler as context manager."""
        with GPUMemoryProfiler() as profiler:
            tensor = torch.randn(500, 500, device="cuda")

        # Profiler should automatically stop monitoring when exiting context
        assert not profiler._monitoring


class TestProfileDecorators:
    """Test profile decorators and context managers."""

    @pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="CUDA not available")  # type: ignore[misc, unused-ignore]
    def test_profile_function_decorator(self) -> None:
        """Test @profile_function decorator."""

        @profile_function  # type: ignore[misc, untyped-decorator, unused-ignore]
        def decorated_function() -> object:
            return torch.randn(500, 500, device="cuda")

        result = decorated_function()
        assert torch.is_tensor(result)
        # Note: The decorator uses global profiler, so we can't easily check results here

    @pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="CUDA not available")  # type: ignore[misc, unused-ignore]
    def test_profile_context_manager(self) -> None:
        """Test profile_context context manager."""
        with profile_context("test_context"):
            tensor = torch.randn(500, 500, device="cuda")
            result = tensor.sum()

        # Should execute without errors


class TestErrorHandling:
    """Test error handling in profiler."""

    @pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="CUDA not available")  # type: ignore[misc, unused-ignore]
    def test_profile_function_with_exception(self) -> None:
        """Test profiling a function that raises an exception."""
        profiler = GPUMemoryProfiler()

        def failing_function() -> object:
            tensor = torch.randn(100, 100, device="cuda")
            raise ValueError("Test exception")

        with pytest.raises(ValueError):
            profiler.profile_function(failing_function)

        # Should still have recorded the profiling attempt
        assert len(profiler.results) == 1

    def test_invalid_device(self) -> None:
        """Test creating profiler with invalid device."""
        with pytest.raises(ValueError):
            GPUMemoryProfiler(device="cpu")  # Should only accept CUDA devices

    @pytest.mark.skipif(TORCH_CUDA_AVAILABLE, reason="Test requires no CUDA")  # type: ignore[misc, unused-ignore]
    def test_no_cuda_available(self) -> None:
        """Test behavior when CUDA is not available."""
        with pytest.raises(RuntimeError):
            GPUMemoryProfiler()


class TestPerformance:
    """Test performance characteristics of the profiler."""

    @pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="CUDA not available")  # type: ignore[misc, unused-ignore]
    def test_profiler_overhead(self) -> None:
        """Test that profiler doesn't add significant overhead."""
        profiler = GPUMemoryProfiler()

        def compute_intensive_function() -> object:
            # Create a moderately expensive computation
            tensor = torch.randn(2000, 2000, device="cuda")
            for _ in range(10):
                tensor = torch.sin(tensor) + torch.cos(tensor)
            return tensor.sum()

        # Time without profiling
        start_time = time.time()
        result1 = compute_intensive_function()
        unprofiled_time = time.time() - start_time

        # Time with profiling
        start_time = time.time()
        result2 = profiler.profile_function(compute_intensive_function)
        profiled_time = time.time() - start_time

        # Overhead should be minimal (less than 50% increase)
        overhead_ratio = profiled_time / unprofiled_time
        assert (
            overhead_ratio < 1.5
        ), f"Profiler overhead too high: {overhead_ratio:.2f}x"

    @pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="CUDA not available")  # type: ignore[misc, unused-ignore]
    def test_monitoring_performance(self) -> None:
        """Test monitoring performance with high frequency."""
        profiler = GPUMemoryProfiler()

        # Start high-frequency monitoring
        profiler.start_monitoring(interval=0.01)  # 10ms intervals

        # Run for a short time
        start_time = time.time()
        while time.time() - start_time < 1.0:  # 1 second
            tensor = torch.randn(100, 100, device="cuda")

        profiler.stop_monitoring()

        # Should have collected many snapshots without significant performance impact
        assert len(profiler.snapshots) > 50  # Should have ~100 snapshots

        # Verify snapshots are reasonable
        for snapshot in profiler.snapshots[:10]:  # Check first 10
            assert isinstance(snapshot, MemorySnapshot)
            assert snapshot.allocated_memory >= 0
