"""Bounded in-process PyTorch profiler capture for import with ``import-trace``.

``capture_torch_trace`` profiles the code inside its block and writes a Kineto
Chrome trace when the block ends, including when it raises. The block is the
bound: there is no step or time limit inside it. Stack, shape, and
memory recording are off unless asked for, since each adds CPU work per
operator. The profiler adds no synchronization per step; stopping it flushes
the CUDA activity buffers once, at the end of the block.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .errors import InferUsageError

# PyTorch runs one Kineto profiler per process; a second one started from
# another thread crashes the process when the first one stops.
_CAPTURE_LOCK = threading.Lock()


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

    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise ProfilerBusyError("another Stormlog capture is running in this process")
    try:
        if _profiler_running(torch):
            raise ProfilerBusyError(
                "a PyTorch profiler is already running; Stormlog does not take it over"
            )
        with _profiled(
            torch,
            Path(output),
            cuda=cuda,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
        ) as path:
            yield path
    finally:
        _CAPTURE_LOCK.release()


@contextmanager
def _profiled(
    torch: Any,
    path: Path,
    *,
    cuda: bool | None,
    with_stack: bool,
    record_shapes: bool,
    profile_memory: bool,
) -> Iterator[Path]:
    from torch.profiler import ProfilerActivity, profile

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
    """Whether a profiler is active in this thread or anywhere in the process.

    ``torch.autograd._profiler_enabled()`` reports only the calling thread, so
    a profiler started on another thread also needs the process-wide flag.
    """
    enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if callable(enabled) and enabled():
        return True
    return bool(getattr(torch.autograd.profiler, "_is_profiler_enabled", False))


__all__ = ["ProfilerBusyError", "capture_torch_trace"]
