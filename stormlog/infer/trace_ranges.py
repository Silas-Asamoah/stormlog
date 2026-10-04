"""Name iteration ranges so a trace importer can link GPU work to iterations.

An engine wraps the CPU code that launches one iteration's GPU work in a
profiler range named ``stormlog.iteration/<producer_id>/<iteration_id>``.
CUPTI ties every kernel, copy, and memset to the CPU call that launched it,
and that call sits inside the range on the same thread. A trace importer can
therefore link GPU work to an iteration without comparing GPU and CPU clocks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from .correlation_events import EntityRef

ITERATION_RANGE_PREFIX = "stormlog.iteration/"


def iteration_range_name(producer_id: str, iteration_id: str) -> str:
    """Return the range name for one iteration of one producer."""
    if not producer_id or "/" in producer_id:
        raise ValueError("producer_id must be non-empty and contain no '/'")
    if not iteration_id or any(char in iteration_id for char in "\r\n"):
        raise ValueError("iteration_id must be non-empty and single-line")
    return f"{ITERATION_RANGE_PREFIX}{producer_id}/{iteration_id}"


def parse_iteration_range(name: str) -> EntityRef | None:
    """Return the iteration a range name refers to, or None for other ranges."""
    if not name.startswith(ITERATION_RANGE_PREFIX):
        return None
    producer_id, separator, iteration_id = name[
        len(ITERATION_RANGE_PREFIX) :
    ].partition("/")
    if not separator or not producer_id or not iteration_id:
        return None
    return EntityRef(producer_id, iteration_id)


@contextmanager
def iteration_range(
    producer_id: str, iteration_id: str, *, nvtx: bool = False
) -> Iterator[None]:
    """Mark one iteration's launches for PyTorch profiler and, optionally, NVTX.

    Without PyTorch the range is a no-op. ``record_function`` costs little when
    no profiler is active. NVTX ranges are opt-in because they are always
    emitted, profiler or not.
    """
    name = iteration_range_name(producer_id, iteration_id)
    with _record_function(name), _nvtx_range(name if nvtx else None):
        yield


def _record_function(name: str) -> Any:
    try:
        from torch.profiler import record_function
    except ImportError:
        return nullcontext()
    return record_function(name)


@contextmanager
def _nvtx_range(name: str | None) -> Iterator[None]:
    nvtx = _torch_nvtx() if name is not None else None
    if nvtx is None:
        yield
        return
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def _torch_nvtx() -> Any:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.nvtx


__all__ = [
    "ITERATION_RANGE_PREFIX",
    "iteration_range",
    "iteration_range_name",
    "parse_iteration_range",
]
