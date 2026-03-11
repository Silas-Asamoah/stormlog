"""Regression tests for TensorFlow runtime fixes."""

from __future__ import annotations

import json
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, cast

import pytest

import tfmemprof.analyzer as tf_analyzer
import tfmemprof.cli as tf_cli
import tfmemprof.context_profiler as tf_context
import tfmemprof.profiler as tf_profiler
import tfmemprof.tracker as tf_tracker


def test_tf_cmd_analyze_rejects_mismatched_timestamps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "tf_results.json"
    input_path.write_text(
        json.dumps(
            {
                "peak_memory": 2.0,
                "average_memory": 1.5,
                "duration": 1.0,
                "memory_usage": [1.0, 2.0],
                "timestamps": [0.0],
            }
        ),
        encoding="utf-8",
    )

    exit_code = tf_cli.cmd_analyze(
        Namespace(
            input=str(input_path),
            detect_leaks=False,
            optimize=False,
            visualize=False,
            report=None,
        )
    )

    assert exit_code == 1
    assert "must have equal length" in capsys.readouterr().out


def test_tf_profile_inference_batches_dataset_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProfiler:
        def __init__(self) -> None:
            self.contexts: list[str] = []

        @contextmanager
        def profile_context(self, name: str) -> Iterator[None]:
            self.contexts.append(name)
            yield

    class _FakeDataset:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def batch(self, batch_size: int) -> list[list[str]]:
            self.batch_sizes.append(batch_size)
            return [["batch-a"], ["batch-b"]]

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], bool]] = []

        def __call__(self, batch: list[str], training: bool = False) -> None:
            self.calls.append((batch, training))

    monkeypatch.setattr(tf_context, "TF_AVAILABLE", True)

    profiler = object.__new__(tf_context.TensorFlowProfiler)
    fake_profiler: Any = _FakeProfiler()
    profiler.profiler = fake_profiler
    dataset = _FakeDataset()
    model = _FakeModel()

    tf_context.TensorFlowProfiler.profile_inference(
        profiler,
        model,
        dataset,
        batch_size=4,
    )

    assert dataset.batch_sizes == [4]
    assert model.calls == [(["batch-a"], False), (["batch-b"], False)]
    assert fake_profiler.contexts == ["inference_batch_0", "inference_batch_1"]


def test_tf_memory_info_uses_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_devices: list[str] = []

    def _get_memory_info(device: str) -> dict[str, int]:
        recorded_devices.append(device)
        return {
            "current": 16 * 1024 * 1024,
            "peak": 32 * 1024 * 1024,
        }

    monkeypatch.setattr(
        tf_profiler,
        "tf",
        SimpleNamespace(
            config=SimpleNamespace(
                experimental=SimpleNamespace(get_memory_info=_get_memory_info)
            )
        ),
    )

    profiler = object.__new__(tf_profiler.TFMemoryProfiler)
    profiler.device = "/GPU:2"

    memory_info = tf_profiler.TFMemoryProfiler._get_memory_info(profiler)

    assert recorded_devices == ["/GPU:2"]
    assert memory_info["gpu_memory_mb"] == 16.0
    assert memory_info["gpu_reserved_mb"] == 32.0


def test_tf_memory_tracker_uses_detected_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _get_default_device(_self: tf_tracker.MemoryTracker) -> str:
        calls["count"] += 1
        return "/CPU:0"

    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)
    monkeypatch.setattr(
        tf_tracker.MemoryTracker, "_get_default_device", _get_default_device
    )

    tracker = tf_tracker.MemoryTracker(enable_logging=False)
    explicit_tracker = tf_tracker.MemoryTracker(device="/GPU:1", enable_logging=False)

    assert tracker.device == "/CPU:0"
    assert explicit_tracker.device == "/GPU:1"
    assert calls["count"] == 1


def test_tf_profile_function_decorator_uses_custom_profile_name() -> None:
    call_count = {"value": 0}
    profiled_names: list[str] = []

    class _FakeProfiler:
        def profile_function(self, func: Any) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                profiled_names.append(func.__name__)
                return func(*args, **kwargs)

            return _wrapped

    profiler: Any = _FakeProfiler()

    def _sample(value: int) -> int:
        call_count["value"] += 1
        return value + 1

    decorated = cast(
        Callable[[int], int],
        tf_context.profile_function(_sample, profiler=profiler, name="custom_step"),
    )

    assert decorated(4) == 5
    assert call_count["value"] == 1
    assert profiled_names == ["custom_step"]


def test_tf_detect_memory_leaks_handles_zero_initial_average() -> None:
    class _TrackingResult:
        memory_usage = [0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 48.0, 64.0, 80.0, 96.0]
        timestamps = list(range(10))
        memory_growth_rate = 9.6

    analyzer = tf_analyzer.MemoryAnalyzer()
    leaks = analyzer.detect_memory_leaks(_TrackingResult())

    assert any(leak["type"] == "insufficient_cleanup" for leak in leaks)


def test_tf_memory_tracker_rejects_non_positive_sampling_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    with pytest.raises(ValueError, match="sampling_interval"):
        tf_tracker.MemoryTracker(sampling_interval=0, enable_logging=False)
