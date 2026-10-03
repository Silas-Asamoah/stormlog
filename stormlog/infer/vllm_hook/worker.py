"""Worker side: wrap each serving step's launches in its iteration range.

The engine core puts ``(producer, iteration)`` on the scheduler output, which
reaches the worker as the same object (uniproc) or pickled (multiproc). The
model runner's ``execute_model`` is wrapped in that iteration's range; when it
returns None, sampling for the same step comes in a later ``sample_tokens``
call, which a FIFO pairs with it. Internal dummy runs (``dummy_run=True``, used
for warm-up and CUDA-graph capture) are never ranged or paired.
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .engine import ITERATION_ATTRIBUTE
from .writer import EpochWriter


@dataclass
class RunnerRecorder:
    """Ranges and counters for one model-runner instance."""

    writer: EpochWriter
    nvtx: bool = False
    range_factory: Callable[[str, str, bool], Any] | None = None
    pending: deque[tuple[str, str]] = field(default_factory=deque)
    range_misses: int = 0

    def status_fields(self) -> dict[str, Any]:
        return {"range_misses": self.range_misses, "pending_samples": len(self.pending)}

    def wrap(self, runner: Any) -> None:
        execute = runner.execute_model
        sample = runner.sample_tokens

        def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("dummy_run"):
                return execute(scheduler_output, *args, **kwargs)
            identity = self._identity(scheduler_output)
            with self._range(identity):
                result = execute(scheduler_output, *args, **kwargs)
            if result is None and identity is not None:
                self.pending.append(identity)
            return result

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            identity = self.pending.popleft() if self.pending else None
            if identity is None:
                self.range_misses += 1
            with self._range(identity):
                return sample(*args, **kwargs)

        runner.execute_model = execute_model
        runner.sample_tokens = sample_tokens

    def _identity(self, scheduler_output: Any) -> tuple[str, str] | None:
        try:
            identity = getattr(scheduler_output, ITERATION_ATTRIBUTE, None)
        except Exception:
            identity = None
        if identity is None:
            self.range_misses += 1
            return None
        return (str(identity[0]), str(identity[1]))

    @contextlib.contextmanager
    def _range(self, identity: tuple[str, str] | None) -> Iterator[None]:
        """The iteration range, or nothing; a failure to open or close is counted."""
        stack = contextlib.ExitStack()
        if identity is not None:
            try:
                stack.enter_context(self._open(identity))
            except Exception:
                self.writer.count_error()
        try:
            yield
        finally:
            try:
                stack.close()
            except Exception:
                self.writer.count_error()

    def _open(self, identity: tuple[str, str]) -> Any:
        if self.range_factory is not None:
            return self.range_factory(identity[0], identity[1], self.nvtx)
        from ..trace_ranges import iteration_range

        return iteration_range(identity[0], identity[1], nvtx=self.nvtx)


def worker_identity(worker: Any) -> dict[str, Any]:
    """Ranks, process and GPU of a worker, each field best effort."""
    return {
        "rank": {
            "global": _call(lambda: int(worker.rank)),
            "tp": _call(_parallel_rank("get_tp_group")),
            "pp": _call(_parallel_rank("get_pp_group")),
            "dp": _call(_parallel_rank("get_dp_group")),
        },
        "local_rank": _call(lambda: int(worker.local_rank)),
        "cuda_ordinal": _call(lambda: int(worker.device.index)),
        "device_uuid": _call(lambda: _device_uuid(worker.device)),
        "trace_rank_suffix": _call(lambda: _trace_rank_suffix(int(worker.rank))),
    }


def _parallel_rank(group: str) -> Callable[[], int]:
    def rank() -> int:
        from vllm.distributed import parallel_state

        return int(getattr(parallel_state, group)().rank_in_group)

    return rank


def _device_uuid(device: Any) -> str:
    import torch

    uuid = str(torch.cuda.get_device_properties(device).uuid)
    return uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"


def _trace_rank_suffix(rank: int) -> str:
    from vllm.distributed.utils import get_worker_rank_suffix

    return str(get_worker_rank_suffix(global_rank=rank))


def _call(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except Exception:
        return None


__all__ = ["RunnerRecorder", "worker_identity"]
