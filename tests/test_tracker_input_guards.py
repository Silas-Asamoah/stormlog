import pytest

from stormlog.cpu_profiler import CPUMemoryProfiler, CPUMemoryTracker
from stormlog.tracker import MemoryTracker


def test_cpu_profiler_rejects_non_positive_monitor_interval() -> None:
    profiler = CPUMemoryProfiler()
    with pytest.raises(ValueError, match="interval must be > 0"):
        profiler.start_monitoring(interval=0)


def test_cpu_tracker_rejects_non_positive_sampling_interval() -> None:
    with pytest.raises(ValueError, match="sampling_interval must be > 0"):
        CPUMemoryTracker(sampling_interval=0)


def test_cpu_tracker_rejects_non_positive_max_events() -> None:
    with pytest.raises(ValueError, match="max_events must be >= 1"):
        CPUMemoryTracker(max_events=0)


def test_gpu_tracker_rejects_non_positive_sampling_interval() -> None:
    with pytest.raises(ValueError, match="sampling_interval must be > 0"):
        MemoryTracker(sampling_interval=0)


def test_gpu_tracker_rejects_non_positive_max_events() -> None:
    with pytest.raises(ValueError, match="max_events must be >= 1"):
        MemoryTracker(max_events=0)
