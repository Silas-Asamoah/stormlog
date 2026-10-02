"""Import Nsight Systems SQLite exports as inference activity references.

``nsys export --type sqlite REPORT.nsys-rep`` writes the CUPTI activity tables
and NVTX ranges of a report into SQLite. This reader turns them into the same
in-memory trace as the Kineto reader, so the import, iteration linking, and
accounting rules are shared:

- GPU work comes from ``CUPTI_ACTIVITY_KIND_KERNEL``, ``_MEMCPY`` and
  ``_MEMSET``; launch calls from ``_RUNTIME``, which also holds driver calls.
  A report can hold several processes, so launches are keyed by process and
  correlation ID.
- Iteration ranges are NVTX ranges named ``stormlog.iteration/...``; emit
  them with ``iteration_range(..., nvtx=True)``.
- A GPU event's ``deviceId`` is the CUDA device ordinal inside its process,
  after ``CUDA_VISIBLE_DEVICES``. ``TARGET_INFO_CUDA_DEVICE`` maps each
  process's ordinals to ``TARGET_INFO_GPU`` rows, which name the UUIDs. Older
  exporters (nsys 2024.4) do not write that table; a report of such an export
  that lists one GPU still names it, since nsys lists every GPU on the host.
  Otherwise the devices stay unnamed: re-export the report with a newer nsys.

Record CUDA graphs with ``--cuda-graph-trace=node``. At the default graph-level
tracing nsys writes one ``CUPTI_ACTIVITY_KIND_GRAPH_TRACE`` row per graph
launch instead of its kernels. This reader does not import those rows; it
counts them in the summary and says how to re-record.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trace_kineto import (
    GpuEvent,
    IterationSpan,
    KinetoTrace,
    LaunchCall,
    index_spans,
)
from .trace_ranges import parse_iteration_range

GPU_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "gpu_kernel",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "gpu_memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "gpu_memset",
}
GRAPH_TABLE = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
# NvtxPushPopRange and NvtxStartEndRange, in any domain. (70 and 71 are ranges
# imported from NVTXT text files, not ones a program emits.)
NVTX_RANGE_TYPES = (59, 60)


def load_nsys_sqlite(path: str | Path) -> KinetoTrace:
    """Read an ``nsys export --type sqlite`` file."""
    path = Path(path)
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            return _load(db, path)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"not an Nsight Systems SQLite export: {exc}") from exc
    except TypeError as exc:
        # A NULL where a column is expected to hold a number.
        raise ValueError(f"malformed Nsight Systems SQLite export: {exc}") from exc


def _load(db: sqlite3.Connection, path: Path) -> KinetoTrace:
    tables = {row[0] for row in db.execute("select name from sqlite_master")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables or "StringIds" not in tables:
        raise ValueError("not an Nsight Systems SQLite export: no CUDA activity")
    strings = dict(db.execute("select id, value from StringIds"))
    trace = KinetoTrace(
        base_ns=_session_start(db, tables),
        host=_meta(db, tables, "META_DATA_CAPTURE", "DEVICE_DISPLAY_NAME"),
        trace_id=path.name,
        rank=None,
        world_size=None,
        engine_version=None,
        cupti_version=None,
        device_names={},
        source="nsys",
    )
    devices = _devices(db, tables)
    if devices.note:
        trace.notes.append(devices.note)
    for table, kind in GPU_TABLES.items():
        if table in tables:
            trace.gpu_events.extend(
                _gpu_events(db, table, kind, strings, devices, trace)
            )
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        trace.launches.update(_launches(db, strings))
    _count_graph_rows(db, tables, trace)
    for thread, span in _nvtx_spans(db, tables, strings):
        trace.spans.setdefault(thread, []).append(span)
    index_spans(trace)
    return trace


def split_global_id(global_id: int) -> tuple[int, int]:
    """(pid, tid) from an nsys globalTid: ``(1 << 48) | pid << 24 | tid``."""
    return (global_id >> 24) & 0xFFFFFF, global_id & 0xFFFFFF


def _session_start(db: sqlite3.Connection, tables: set[str]) -> int:
    if "TARGET_INFO_SESSION_START_TIME" not in tables:
        return 0
    row = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _meta(
    db: sqlite3.Connection, tables: set[str], table: str, name: str
) -> str | None:
    if table not in tables:
        return None
    row = db.execute(f"select value from {table} where name = ?", (name,)).fetchone()
    return str(row[0]) if row else None


@dataclass
class _Devices:
    """The GPU (UUID and name) behind each process's CUDA device ordinal."""

    by_process: dict[tuple[int, int], tuple[str | None, str | None]] = field(
        default_factory=dict
    )
    only_gpu: tuple[str | None, str | None] | None = None
    note: str | None = None

    def lookup(
        self, pid: int | None, device: int | None
    ) -> tuple[str | None, str | None]:
        if pid is not None and device is not None:
            found = self.by_process.get((pid, device))
            if found is not None:
                return found
        return self.only_gpu or (None, None)


def _devices(db: sqlite3.Connection, tables: set[str]) -> _Devices:
    gpus = _gpus(db) if "TARGET_INFO_GPU" in tables else {}
    # nsys lists every GPU on the host, so one listed GPU is the one any
    # process used, whatever its CUDA_VISIBLE_DEVICES.
    devices = _Devices(only_gpu=next(iter(gpus.values())) if len(gpus) == 1 else None)
    if "TARGET_INFO_CUDA_DEVICE" in tables:
        query = "select pid, cudaId, gpuId from TARGET_INFO_CUDA_DEVICE"
        for pid, cuda_id, gpu_id in db.execute(query):
            if None not in (pid, cuda_id, gpu_id) and int(gpu_id) in gpus:
                devices.by_process[(int(pid), int(cuda_id))] = gpus[int(gpu_id)]
    elif len(gpus) > 1:
        devices.note = (
            "this export does not say which GPU each process's CUDA devices "
            "are; re-export the report with nsys 2025.1 or later to name them"
        )
    return devices


def _gpus(db: sqlite3.Connection) -> dict[int, tuple[str | None, str | None]]:
    """UUID and name per nsys GPU id; ids do not follow nvidia-smi order."""
    query = "select id, name, uuid from TARGET_INFO_GPU"
    return {
        int(gpu): (_nvml_uuid(uuid), str(name) if name is not None else None)
        for gpu, name, uuid in db.execute(query)
        if gpu is not None
    }


def _nvml_uuid(uuid: Any) -> str | None:
    if not uuid:
        return None
    text = str(uuid)
    return text if text.startswith(("GPU-", "MIG-")) else f"GPU-{text}"


def _gpu_events(
    db: sqlite3.Connection,
    table: str,
    kind: str,
    strings: dict[int, str],
    devices: _Devices,
    trace: KinetoTrace,
) -> Iterator[GpuEvent]:
    columns = _columns(db, table)
    name = "shortName" if "shortName" in columns else None
    query = (
        f"select start, end, deviceId, streamId, correlationId, globalPid, "
        f"{name or 'NULL'} from {table}"
    )
    for start, end, device, stream, correlation, global_pid, name_id in db.execute(
        query
    ):
        pid = split_global_id(int(global_pid))[0] if global_pid is not None else None
        uuid, device_name = devices.lookup(pid, _int(device))
        yield GpuEvent(
            start_ns=trace.base_ns + int(start),
            end_ns=trace.base_ns + int(end),
            kind=kind,
            name=strings.get(name_id, kind) if name_id is not None else kind,
            device=_int(device),
            stream=_int(stream),
            correlation=_int(correlation) or None,
            graph_id=None,
            pid=pid,
            device_uuid=uuid,
            device_name=device_name,
        )


def _launches(
    db: sqlite3.Connection, strings: dict[int, str]
) -> Iterator[tuple[tuple[int | None, int], LaunchCall]]:
    query = (
        "select start, globalTid, correlationId, nameId "
        "from CUPTI_ACTIVITY_KIND_RUNTIME"
    )
    for start, global_tid, correlation, name_id in db.execute(query):
        if not correlation or global_tid is None:
            continue
        pid, tid = split_global_id(int(global_tid))
        call = LaunchCall(
            pid=pid, tid=tid, ts_us=int(start) / 1000, name=strings.get(name_id, "")
        )
        yield (pid, int(correlation)), call


def _count_graph_rows(
    db: sqlite3.Connection, tables: set[str], trace: KinetoTrace
) -> None:
    if GRAPH_TABLE not in tables:
        return
    rows = int(db.execute(f"select count(*) from {GRAPH_TABLE}").fetchone()[0])
    if rows:
        trace.not_imported["graph_trace_rows"] = rows
        trace.notes.append(
            f"{rows} CUDA graph launches were recorded per graph, not per kernel, "
            "and are not imported; record with --cuda-graph-trace=node"
        )


def _nvtx_spans(
    db: sqlite3.Connection, tables: set[str], strings: dict[int, str]
) -> Iterator[tuple[tuple[int, int], IterationSpan]]:
    if "NVTX_EVENTS" not in tables:
        return
    query = (
        "select start, end, text, textId, globalTid from NVTX_EVENTS "
        f"where eventType in {NVTX_RANGE_TYPES} and end is not null"
    )
    for start, end, text, text_id, global_tid in db.execute(query):
        ref = parse_iteration_range(
            text if text is not None else strings.get(text_id, "")
        )
        if ref is None or global_tid is None:
            continue
        thread = split_global_id(int(global_tid))
        yield thread, IterationSpan(int(start) / 1000, int(end) / 1000, ref)


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"pragma table_info({table})")}


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = ["load_nsys_sqlite", "split_global_id"]
