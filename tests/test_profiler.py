from typing import Any

import pytest

try:  # Optional dependency: PyTorch
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment w/out torch
    torch = None  # type: ignore[assignment, unused-ignore]

try:  # Optional dependency: TensorFlow
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - environment w/out tf
    tf = None  # type: ignore[assignment, unused-ignore]

# GPUMemoryProfiler requires torch at import time, so guard the import
GPUMemoryProfiler: Any = None
if torch is not None:
    try:
        from gpumemprof import (  # type: ignore[no-redef, unused-ignore]  # noqa: F811
            GPUMemoryProfiler,
        )
    except ImportError:  # pragma: no cover - torch installed but import still fails
        pass

TORCH_AVAILABLE = torch is not None
TORCH_CUDA_AVAILABLE = bool(torch and torch.cuda.is_available())
TF_AVAILABLE = tf is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")  # type: ignore[misc, unused-ignore]
@pytest.mark.skipif(  # type: ignore[misc, unused-ignore]
    not TORCH_CUDA_AVAILABLE, reason="CUDA-enabled PyTorch not available"
)
def test_profiler() -> None:
    assert GPUMemoryProfiler is not None
    profiler = GPUMemoryProfiler()

    def dummy_func() -> str:
        return "Hello, World!"

    # Test basic profiling
    result = profiler.profile_function(dummy_func)

    # Check that profiling was performed
    assert profiler.results is not None
    assert len(profiler.results) > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")  # type: ignore[misc, unused-ignore]
@pytest.mark.skipif(  # type: ignore[misc, unused-ignore]
    not (TORCH_CUDA_AVAILABLE and TF_AVAILABLE),
    reason="PyTorch (CUDA) and TensorFlow required",
)
def test_profile_tensorflow() -> None:
    assert GPUMemoryProfiler is not None
    profiler = GPUMemoryProfiler()

    def dummy_tensorflow_func() -> object:
        x = tf.constant([1, 2, 3, 4, 5])
        return tf.reduce_sum(x)

    # Test TensorFlow profiling
    result = profiler.profile_function(dummy_tensorflow_func)

    # Check that profiling was performed
    assert profiler.results is not None
    assert len(profiler.results) > 0
