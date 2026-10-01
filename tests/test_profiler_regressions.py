import inspect
import subprocess
import sys
import textwrap
from collections import defaultdict, deque
from typing import Any

import pytest

import stormlog.profiler as profiler_module
from stormlog.context_profiler import profile_function
from stormlog.device_collectors import DeviceMemoryCapabilities
from stormlog.profiler import GPUMemoryProfiler, MemorySnapshot, TensorTracker


class _CapabilityCollector:
    def __init__(self, capabilities: DeviceMemoryCapabilities) -> None:
        self._capabilities = capabilities

    def capabilities(self) -> DeviceMemoryCapabilities:
        return self._capabilities


def test_stormlog_import_and_star_import_succeed_when_viz_imports_blocked() -> None:
    code = textwrap.dedent(
        """
        import builtins

        blocked_roots = {"matplotlib", "seaborn", "plotly"}
        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked_roots:
                raise ModuleNotFoundError(f"blocked import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        import stormlog
        from stormlog import *  # noqa: F403,F401

        assert hasattr(stormlog, "GPUMemoryProfiler")
        assert "GPUMemoryProfiler" in globals()
        assert hasattr(stormlog, "MemoryVisualizer")
        assert "MemoryVisualizer" in globals()
        try:
            stormlog.MemoryVisualizer()
        except ImportError as exc:
            assert "optional visualization dependencies" in str(exc)
        else:
            raise AssertionError("Expected ImportError when constructing MemoryVisualizer")

        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


class _DummyProfiler:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_name: str | None = None

    def profile_function(self, func: Any) -> object:
        self.calls += 1
        self.seen_name = getattr(func, "__name__", None)
        func()
        return object()


def test_profile_function_decorator_executes_once_and_returns_result() -> None:
    profiler = _DummyProfiler()
    state = {"calls": 0}

    @profile_function(name="custom_profile_name", profiler=profiler)  # type: ignore[arg-type, unused-ignore]
    def tracked_operation() -> str:
        state["calls"] += 1
        return "ok"

    result: object = tracked_operation()  # type: ignore[misc, unused-ignore]

    assert result == "ok"
    assert state["calls"] == 1
    assert profiler.calls == 1
    assert profiler.seen_name == "custom_profile_name"


def test_gpu_profiler_disables_tensor_scanning_by_default() -> None:
    parameter = inspect.signature(GPUMemoryProfiler.__init__).parameters[
        "track_tensors"
    ]

    assert parameter.default is False


def test_tensor_tracker_count_tensors_does_not_force_memory_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = TensorTracker()

    def fail_if_called() -> None:
        raise AssertionError("count_tensors should not force memory cleanup")

    monkeypatch.setattr(profiler_module.gc, "collect", fail_if_called)
    monkeypatch.setattr(profiler_module.gc, "get_objects", lambda: [])
    monkeypatch.setattr(profiler_module.torch.cuda, "empty_cache", fail_if_called)

    assert tracker.count_tensors() == 0


def test_gpu_profiler_rejects_non_positive_snapshot_limit() -> None:
    with pytest.raises(ValueError, match="max_snapshots must be >= 1"):
        GPUMemoryProfiler(max_snapshots=0)


def test_gpu_profiler_uses_bounded_snapshot_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = MemorySnapshot(
        timestamp=0.0,
        allocated_memory=0,
        reserved_memory=0,
        max_memory_allocated=0,
        max_memory_reserved=0,
        active_memory=0,
        inactive_memory=0,
        cpu_memory=0,
    )
    monkeypatch.setattr(GPUMemoryProfiler, "_setup_device", lambda *_: object())
    monkeypatch.setattr(GPUMemoryProfiler, "_take_snapshot", lambda *_: snapshot)
    monkeypatch.setattr(
        profiler_module,
        "build_device_memory_collector",
        lambda _device: _CapabilityCollector(
            DeviceMemoryCapabilities(
                backend="cuda",
                telemetry_collector="stormlog.cuda_tracker",
                sampling_source="test",
                supports_allocator_allocated=True,
                supports_allocator_reserved=True,
                supports_bounded_profiling=True,
            )
        ),
    )

    profiler = GPUMemoryProfiler(max_snapshots=3)

    assert profiler.snapshots.maxlen == 3


def test_gpu_profiler_rejects_backend_without_bounded_profiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GPUMemoryProfiler, "_setup_device", lambda *_: object())
    monkeypatch.setattr(
        profiler_module,
        "build_device_memory_collector",
        lambda _device: _CapabilityCollector(
            DeviceMemoryCapabilities(
                backend="mps",
                telemetry_collector="stormlog.mps_tracker",
                sampling_source="torch.mps",
                supports_allocator_allocated=True,
                supports_allocator_reserved=True,
                supports_device_used=True,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="bounded profiling requires"):
        GPUMemoryProfiler()


def test_monitoring_retains_only_latest_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = object.__new__(GPUMemoryProfiler)
    profiler.max_snapshots = 3
    profiler.snapshots = deque(maxlen=profiler.max_snapshots)
    profiler._monitor_interval = 0.0
    profiler._monitoring = True
    next_timestamp = 0

    def take_snapshot(operation: str) -> MemorySnapshot:
        nonlocal next_timestamp
        snapshot = MemorySnapshot(
            timestamp=float(next_timestamp),
            allocated_memory=0,
            reserved_memory=0,
            max_memory_allocated=0,
            max_memory_reserved=0,
            active_memory=0,
            inactive_memory=0,
            cpu_memory=0,
            operation=operation,
        )
        next_timestamp += 1
        if next_timestamp == 5:
            profiler._monitoring = False
        return snapshot

    monkeypatch.setattr(profiler, "_take_snapshot", take_snapshot)
    monkeypatch.setattr(profiler_module.time, "sleep", lambda _: None)

    profiler._monitor_memory()

    assert [snapshot.timestamp for snapshot in profiler.snapshots] == [2.0, 3.0, 4.0]


class _ExceptionPathHarness:
    def __init__(self) -> None:
        self._tensor_tracker = None
        self.results: list[object] = []
        self.function_stats: dict[str, list[object]] = defaultdict(list)
        self.device = 0

    def _take_snapshot(self, operation: str | None = None) -> "MemorySnapshot":
        return MemorySnapshot(
            timestamp=0.0,
            allocated_memory=0,
            reserved_memory=0,
            max_memory_allocated=0,
            max_memory_reserved=0,
            active_memory=0,
            inactive_memory=0,
            cpu_memory=0,
            device_id=0,
            operation=operation,
        )


def test_profile_function_reraises_without_duplicating_profiler_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ExceptionPathHarness()
    monkeypatch.setattr(
        profiler_module.torch.cuda, "reset_peak_memory_stats", lambda _device: None
    )

    def failing_operation() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError) as exc_info:
        GPUMemoryProfiler.profile_function(harness, failing_operation)  # type: ignore[arg-type, unused-ignore]

    frame_names = []
    tb = exc_info.value.__traceback__
    while tb:
        frame_names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next

    assert frame_names.count("profile_function") == 1
    assert "failing_operation" in frame_names
    assert len(harness.results) == 1
