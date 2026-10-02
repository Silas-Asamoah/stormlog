"""Import Nsight Systems SQLite exports as inference activity references.

``nsys export --type sqlite REPORT.nsys-rep`` writes the CUPTI activity tables
and NVTX ranges of a report into SQLite. This reader turns them into the same
in-memory trace as the Kineto reader, so the import, iteration linking, and
accounting rules are shared:

- GPU work comes from ``CUPTI_ACTIVITY_KIND_KERNEL``, ``_MEMCPY`` and
  ``_MEMSET``; launch calls from ``_RUNTIME`` (and ``_DRIVER`` when present).
  A report can hold several processes, so launches are keyed by process and
  correlation ID.
- Iteration ranges are NVTX ranges named ``stormlog.iteration/...``; emit
  them with ``iteration_range(..., nvtx=True)``.
- GPU UUIDs come from the report itself: ``TARGET_INFO_CUDA_DEVICE`` maps a
  process's CUDA device to a GPU and ``TARGET_INFO_GPU`` names its UUID, so
  ``CUDA_VISIBLE_DEVICES`` renumbering is handled.

Record CUDA graphs with ``--cuda-graph-trace=node``. At the default graph-level
tracing nsys writes one row per graph instead of its kernels, which this reader
does not import.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
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
LAUNCH_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
# NvtxPushPopRange, NvtxStartEndRange, and the domain-scoped push/pop range.
NVTX_RANGE_TYPES = (59, 60, 70)


def load_nsys_sqlite(path: str | Path) -> KinetoTrace:
    """Read an ``nsys export --type sqlite`` file."""
    path = Path(path)
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            return _load(db, path)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"not an Nsight Systems SQLite export: {exc}") from exc


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
        device_names=_device_names(db, tables),
        source="nsys",
    )
    uuids = _device_uuids(db, tables)
    for table, kind in GPU_TABLES.items():
        if table in tables:
            trace.gpu_events.extend(_gpu_events(db, table, kind, strings, uuids, trace))
    for table in LAUNCH_TABLES:
        if table in tables:
            trace.launches.update(_launches(db, table, strings))
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


def _device_names(db: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
    if "TARGET_INFO_GPU" not in tables:
        return {}
    return {
        int(gpu): str(name)
        for gpu, name in db.execute("select id, name from TARGET_INFO_GPU")
    }


def _device_uuids(
    db: sqlite3.Connection, tables: set[str]
) -> dict[tuple[int, int], str]:
    """GPU UUID per (pid, CUDA device), NVML-style with a ``GPU-`` prefix."""
    if not {"TARGET_INFO_GPU", "TARGET_INFO_CUDA_DEVICE"} <= tables:
        return {}
    gpus = {
        int(gpu): str(uuid)
        for gpu, uuid in db.execute("select id, uuid from TARGET_INFO_GPU")
        if uuid
    }
    mapping = {}
    for gpu_id, cuda_id, pid in db.execute(
        "select gpuId, cudaId, pid from TARGET_INFO_CUDA_DEVICE"
    ):
        uuid = gpus.get(int(gpu_id))
        if uuid:
            mapping[(int(pid), int(cuda_id))] = (
                uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
            )
    return mapping


def _gpu_events(
    db: sqlite3.Connection,
    table: str,
    kind: str,
    strings: dict[int, str],
    uuids: dict[tuple[int, int], str],
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
            device_uuid=(
                uuids.get((pid, int(device)))
                if pid is not None and device is not None
                else None
            ),
        )


def _launches(
    db: sqlite3.Connection, table: str, strings: dict[int, str]
) -> Iterator[tuple[tuple[int | None, int], LaunchCall]]:
    query = f"select start, globalTid, correlationId, nameId from {table}"
    for start, global_tid, correlation, name_id in db.execute(query):
        if not correlation or global_tid is None:
            continue
        pid, tid = split_global_id(int(global_tid))
        call = LaunchCall(
            pid=pid, tid=tid, ts_us=int(start) / 1000, name=strings.get(name_id, "")
        )
        yield (pid, int(correlation)), call


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
