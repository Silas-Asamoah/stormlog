"""Bounded in-process PyTorch profiler capture for import with ``import-trace``.

``capture_torch_trace`` profiles the code inside its block and writes a Kineto
Chrome trace when the block ends, including when it raises. The block is the
bound: there is no step or time limit inside it. Stack, shape, and
memory recording are off unless asked for, since each adds CPU work per
operator. The profiler adds no synchronization per step; stopping it flushes
the CUDA activity buffers once, at the end of the block.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .errors import InferUsageError


class ProfilerBusyError(InferUsageError):
    """Another PyTorch profiler is already running in this process."""


@contextmanager
def capture_torch_trace(
    output: str | Path,
    *,
    cuda: bool | None = None,
    with_stack: bool = False,
    record_shapes: bool = False,
    profile_memory: bool = False,
) -> Iterator[Path]:
    """Profile the block and write its trace to ``output``.

    ``cuda`` defaults to whether CUDA is available. Wrap each iteration in
    ``stormlog.infer.trace_ranges.iteration_range`` so the import can link GPU
    work to iterations.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    if _profiler_running(torch):
        raise ProfilerBusyError(
            "a PyTorch profiler is already running; Stormlog does not take it over"
        )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() if cuda is None else cuda
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)
    profiler = profile(
        activities=activities,
        with_stack=with_stack,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
    )
    profiler.start()
    try:
        yield path
    except BaseException:
        # Keep the block's exception; a failed export must not replace it.
        with suppress(Exception):
            profiler.stop()
            profiler.export_chrome_trace(str(path))
        raise
    profiler.stop()
    profiler.export_chrome_trace(str(path))


def _profiler_running(torch: Any) -> bool:
    """Whether a profiler is active, from the autograd engine's own flag."""
    enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if callable(enabled):
        return bool(enabled())
    return bool(getattr(torch.autograd.profiler, "_is_profiler_enabled", False))


__all__ = ["ProfilerBusyError", "capture_torch_trace"]
