import subprocess
import sys
import textwrap
from collections import defaultdict
from typing import Any

import pytest

import gpumemprof.profiler as profiler_module
from gpumemprof.context_profiler import profile_function
from gpumemprof.profiler import GPUMemoryProfiler, MemorySnapshot, TensorTracker


def test_gpumemprof_import_and_star_import_succeed_when_viz_imports_blocked() -> None:
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

        import gpumemprof
        from gpumemprof import *  # noqa: F403,F401

        assert hasattr(gpumemprof, "GPUMemoryProfiler")
        assert "GPUMemoryProfiler" in globals()
        assert hasattr(gpumemprof, "MemoryVisualizer")
        assert "MemoryVisualizer" in globals()
        try:
            gpumemprof.MemoryVisualizer()
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


def test_tensor_tracker_count_tensors_does_not_call_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = TensorTracker()
    state = {"called": False}

    def fail_if_called() -> None:
        state["called"] = True
        raise AssertionError("count_tensors should not call torch.cuda.empty_cache()")

    monkeypatch.setattr(profiler_module.gc, "collect", lambda: 0)
    monkeypatch.setattr(profiler_module.gc, "get_objects", lambda: [])
    monkeypatch.setattr(profiler_module.torch.cuda, "empty_cache", fail_if_called)

    assert tracker.count_tensors() == 0
    assert state["called"] is False


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
