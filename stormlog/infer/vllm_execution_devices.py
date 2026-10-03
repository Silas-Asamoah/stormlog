"""Map a profiler trace to the GPU it ran on, from the vLLM hook's worker hellos.

Each worker epoch's ``hello`` names the worker's host, pid, CUDA ordinal and
GPU UUID (``docs/vllm_execution.md``). A trace names its host and launching
pids and spans a wall-clock window, so a worker epoch that was alive on that
host, with that pid, across that window is the process that wrote it: its
UUID is the trace's device. A pid that no epoch covers, or that several do,
is left for ``--device-uuid``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InferInputError
from .trace_kineto import KinetoTrace
from .vllm_execution_log import SILENCE_NS, EpochRead, LogRead, read_execution_log

STATUS_BOUND = "bound"
STATUS_PARTIAL = "partial"
STATUS_NONE = "none"
STATUS_NO_HOST = "no_host"
STATUS_NO_WINDOW = "no_window"
# A heartbeat is written every second: a trace that ends within this long of
# the last one was written by a process still alive, even if the log was
# copied before the process ended.
SLACK_NS = SILENCE_NS


@dataclass(frozen=True)
class WorkerEpoch:
    """One worker process lifetime, as its hello and liveness records say."""

    epoch: str
    host: str
    boot_id: str | None
    pid: int
    engine_producer: str | None
    cuda_ordinal: int | None
    device_uuid: str | None
    local_rank: int | None
    rank: dict[str, int]
    trace_rank_suffix: str | None
    enabled: bool
    refused: str | None
    started_wall_ns: int | None
    last_seen_wall_ns: int | None
    ended: bool

    def covers(self, start_wall_ns: int, end_wall_ns: int) -> bool:
        """Whether this process was alive across the whole window."""
        if self.started_wall_ns is not None and self.started_wall_ns > start_wall_ns:
            return False
        if self.last_seen_wall_ns is None:
            return not self.ended
        slack = 0 if self.ended else SLACK_NS
        return self.last_seen_wall_ns + slack >= end_wall_ns


@dataclass(frozen=True)
class TraceBinding:
    """The GPU UUIDs a trace's processes map to, and what could not be mapped."""

    status: str
    uuids: dict[int, str] = field(default_factory=dict)
    workers: dict[int, str] = field(default_factory=dict)  # pid -> epoch
    ambiguous: dict[int, list[str]] = field(default_factory=dict)
    unmatched: list[int] = field(default_factory=list)
    conflicts: list[int] = field(default_factory=list)  # ordinals

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "device_uuids": {str(k): v for k, v in sorted(self.uuids.items())},
            "workers": {str(k): v for k, v in sorted(self.workers.items())},
            "ambiguous": {str(k): v for k, v in sorted(self.ambiguous.items())},
            "unmatched": list(self.unmatched),
            "conflicts": list(self.conflicts),
        }


class WorkerIndex:
    """Every worker epoch of a hook directory, ready to bind traces."""

    def __init__(self, workers: Sequence[WorkerEpoch], directory: Path | None = None):
        self.workers = list(workers)
        self.directory = directory

    @classmethod
    def from_log(cls, read: LogRead) -> WorkerIndex:
        return cls([_worker_epoch(epoch) for epoch in read.workers()], read.directory)

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, now_ns: int | None = None
    ) -> WorkerIndex:
        try:
            read = read_execution_log(directory, now_ns=now_ns)
        except (OSError, ValueError) as exc:
            raise InferInputError(f"--vllm-execution-dir {directory}: {exc}") from exc
        return cls.from_log(read)

    def bind_trace(self, trace: KinetoTrace) -> TraceBinding:
        """Bind a loaded trace by its host, launching pids and time window."""
        window = trace_wall_window(trace)
        # Launch calls carry the launching process; a multi-process format
        # (Nsight Systems) also stamps its GPU events.
        pids = sorted(
            {launch.pid for launch in trace.launches.values()}
            | {e.pid for e in trace.gpu_events if e.pid is not None}
        )
        if window is None:
            return TraceBinding(STATUS_NO_WINDOW, unmatched=pids)
        return self.bind(
            host=trace.host, pids=pids, start_wall_ns=window[0], end_wall_ns=window[1]
        )

    def bind(
        self,
        *,
        host: str | None,
        pids: Iterable[int],
        start_wall_ns: int,
        end_wall_ns: int,
    ) -> TraceBinding:
        """The one worker epoch per pid alive on ``host`` across the window."""
        wanted = sorted(set(pids))
        if host is None:
            return TraceBinding(STATUS_NO_HOST, unmatched=wanted)
        matches = _Matches()
        for pid in wanted:
            matches.add(pid, self._alive(host, pid, start_wall_ns, end_wall_ns))
        return matches.binding(wanted)

    def _alive(
        self, host: str, pid: int, start_wall_ns: int, end_wall_ns: int
    ) -> list[WorkerEpoch]:
        return [
            w
            for w in self.workers
            if w.host == host and w.pid == pid and w.covers(start_wall_ns, end_wall_ns)
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory) if self.directory else None,
            "workers": [
                {
                    "epoch": w.epoch,
                    "host": w.host,
                    "pid": w.pid,
                    "cuda_ordinal": w.cuda_ordinal,
                    "device_uuid": w.device_uuid,
                    "enabled": w.enabled,
                    "ended": w.ended,
                }
                for w in self.workers
            ],
        }


def trace_wall_window(trace: KinetoTrace) -> tuple[int, int] | None:
    """The wall-clock span of a trace's launches and GPU events, if any."""
    starts = [event.start_ns for event in trace.gpu_events]
    ends = [event.end_ns for event in trace.gpu_events]
    for launch in trace.launches.values():
        at = trace.base_ns + int(round(launch.ts_us * 1000))
        starts.append(at)
        ends.append(at)
    if not starts:
        return None
    return min(starts), max(ends)


def _worker_epoch(epoch: EpochRead) -> WorkerEpoch:
    hello = epoch.hello or {}
    rank_value = hello.get("rank")
    rank = rank_value if isinstance(rank_value, dict) else {}
    return WorkerEpoch(
        epoch=epoch.epoch,
        host=epoch.host,
        boot_id=epoch.boot_id,
        pid=epoch.pid,
        engine_producer=_text(hello.get("producer")),
        cuda_ordinal=_integer(hello.get("cuda_ordinal")),
        device_uuid=_text(hello.get("device_uuid")),
        local_rank=_integer(hello.get("local_rank")),
        rank={k: v for k, v in rank.items() if _integer(v) is not None},
        trace_rank_suffix=_text(hello.get("trace_rank_suffix")),
        enabled=bool(hello.get("enabled", False)),
        refused=_text(hello.get("refused")),
        started_wall_ns=_integer((hello.get("clock") or {}).get("wall_ns")),
        last_seen_wall_ns=epoch.last_seen_wall_ns,
        ended=epoch.goodbye is not None,
    )


@dataclass
class _Matches:
    """What binding found per pid, folded into one trace's device map."""

    uuids: dict[int, str] = field(default_factory=dict)
    workers: dict[int, str] = field(default_factory=dict)
    ambiguous: dict[int, list[str]] = field(default_factory=dict)
    unmatched: list[int] = field(default_factory=list)
    conflicts: set[int] = field(default_factory=set)

    def add(self, pid: int, found: list[WorkerEpoch]) -> None:
        if len(found) == 1:
            self.workers[pid] = found[0].epoch
            self._add_uuid(found[0])
        elif found:
            self.ambiguous[pid] = [w.epoch for w in found]
        else:
            self.unmatched.append(pid)

    def _add_uuid(self, worker: WorkerEpoch) -> None:
        if worker.cuda_ordinal is None or worker.device_uuid is None:
            return
        known = self.uuids.get(worker.cuda_ordinal)
        if known is not None and known != worker.device_uuid:
            self.conflicts.add(worker.cuda_ordinal)
        self.uuids.setdefault(worker.cuda_ordinal, worker.device_uuid)

    def binding(self, wanted: list[int]) -> TraceBinding:
        uuids = {k: v for k, v in self.uuids.items() if k not in self.conflicts}
        if not wanted or not self.workers:
            status = STATUS_NONE
        elif len(self.workers) == len(wanted):
            status = STATUS_BOUND
        else:
            status = STATUS_PARTIAL
        return TraceBinding(
            status,
            uuids,
            self.workers,
            self.ambiguous,
            self.unmatched,
            sorted(self.conflicts),
        )


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "STATUS_BOUND",
    "STATUS_NONE",
    "STATUS_NO_HOST",
    "STATUS_NO_WINDOW",
    "STATUS_PARTIAL",
    "TraceBinding",
    "WorkerEpoch",
    "WorkerIndex",
    "trace_wall_window",
]
