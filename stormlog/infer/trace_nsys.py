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
OPTIONAL_COLUMNS = ("shortName", "graphId", "graphNodeId")
GRAPH_TABLE = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"
CUDA_TABLES = {*GPU_TABLES, GRAPH_TABLE, RUNTIME_TABLE}
# The exporter always writes one of these next to StringIds.
EXPORT_MARKERS = {"META_DATA_EXPORT", "TARGET_INFO_SESSION_START_TIME"}
# NvtxPushPopRange and NvtxStartEndRange, in any domain. (70 and 71 are ranges
# imported from NVTXT text files, not ones a program emits.)
NVTX_RANGE_TYPES = (59, 60)


def load_nsys_sqlite(path: str | Path) -> KinetoTrace:
    """Read an ``nsys export --type sqlite`` file."""
    path = Path(path)
    # SQLite reads a URI, so the name is percent-encoded: a raw "%6f" would
    # open a different file, and "#" or "?" would cut the name and drop mode=ro.
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as db:
            return _load(db, path)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"not an Nsight Systems SQLite export: {exc}") from exc
    except TypeError as exc:
        # A NULL where a column is expected to hold a number.
        raise ValueError(f"malformed Nsight Systems SQLite export: {exc}") from exc


def _load(db: sqlite3.Connection, path: Path) -> KinetoTrace:
    tables = _export_tables(db)
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
    _read_gpu_work(db, tables, strings, trace)
    if RUNTIME_TABLE in tables:
        trace.launches.update(_launches(db, strings))
    _count_graph_rows(db, tables, trace)
    if not trace.gpu_events and not trace.not_imported:
        trace.notes.append(
            "the report records CUDA calls but no GPU kernels, copies or memsets"
        )
    for thread, span in _nvtx_spans(db, tables, strings):
        trace.spans.setdefault(thread, []).append(span)
    index_spans(trace)
    return trace


def _export_tables(db: sqlite3.Connection) -> set[str]:
    tables = {row[0] for row in db.execute("select name from sqlite_master")}
    if "StringIds" not in tables or not tables & EXPORT_MARKERS:
        raise ValueError("not an Nsight Systems SQLite export")
    # nsys creates tables lazily: a copy-only report has no kernel table.
    if not tables & CUDA_TABLES:
        raise ValueError("no CUDA activity in this report; record with --trace=cuda")
    return tables


def _read_gpu_work(
    db: sqlite3.Connection,
    tables: set[str],
    strings: dict[int, str],
    trace: KinetoTrace,
) -> None:
    devices = _devices(db, tables)
    if devices.note:
        trace.notes.append(devices.note)
    for table, kind in GPU_TABLES.items():
        if table in tables:
            trace.gpu_events.extend(
                _gpu_events(db, table, kind, strings, devices, trace)
            )
    if not devices.note:
        trace.notes.extend(_unnamed_devices_note(trace.gpu_events))


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
        # A process on a one-GPU host sees that GPU as device 0; another
        # ordinal (a MIG instance, say) breaks the premise, so it stays unnamed.
        if device == 0 and self.only_gpu is not None:
            return self.only_gpu
        return (None, None)


def _devices(db: sqlite3.Connection, tables: set[str]) -> _Devices:
    gpus = _gpus(db) if "TARGET_INFO_GPU" in tables else {}
    # nsys lists every GPU on the host, so one listed GPU is the one any
    # process used, whatever its CUDA_VISIBLE_DEVICES.
    devices = _Devices(only_gpu=next(iter(gpus.values())) if len(gpus) == 1 else None)
    if "TARGET_INFO_CUDA_DEVICE" in tables:
        devices.by_process = _process_devices(db, gpus)
    elif len(gpus) > 1:
        devices.note = (
            "this export does not say which GPU each process's CUDA devices "
            "are; re-export the report with nsys 2025.1 or later to name them"
        )
    return devices


GpuEntry = tuple[str | None, str | None]


def _process_devices(
    db: sqlite3.Connection, gpus: dict[int, GpuEntry]
) -> dict[tuple[int, int], GpuEntry]:
    """Each process's CUDA ordinal mapped to its GPU by TARGET_INFO_CUDA_DEVICE.

    ``gpuId`` points at a ``TARGET_INFO_GPU`` row. NVIDIA's schema reference
    also lists an optional ``uuid`` column that names the GPU directly, even
    when ``gpuId`` is NULL; the nsys 2025.1 and 2025.6 exports seen so far
    carried only ``gpuId``, so which exporters write it is unverified.
    """
    columns = _columns(db, "TARGET_INFO_CUDA_DEVICE")
    uuid_column = "uuid" if "uuid" in columns else "NULL"
    query = f"select pid, cudaId, gpuId, {uuid_column} from TARGET_INFO_CUDA_DEVICE"
    mapping: dict[tuple[int, int], GpuEntry] = {}
    for pid, cuda_id, gpu_id, uuid in db.execute(query):
        entry = _device_row(gpus, gpu_id, uuid, f"process {pid}'s device {cuda_id}")
        if pid is not None and cuda_id is not None and entry is not None:
            mapping[(int(pid), int(cuda_id))] = entry
    return mapping


def _device_row(
    gpus: dict[int, GpuEntry], gpu_id: Any, uuid: Any, what: str
) -> GpuEntry | None:
    """The GPU a device row names: by its own ``uuid``, else by ``gpuId``.

    A row that names two different GPUs is not read either way.
    """
    listed = gpus.get(int(gpu_id)) if gpu_id is not None else None
    direct = _nvml_uuid(uuid)
    if direct is None:
        return listed
    if listed is not None and listed[0] not in (None, direct):
        raise ValueError(
            "malformed Nsight Systems SQLite export: TARGET_INFO_CUDA_DEVICE "
            f"names {what} both {direct} and GPU {gpu_id} ({listed[0]})"
        )
    names = {gpu_uuid: name for gpu_uuid, name in gpus.values()}
    return direct, names.get(direct, listed[1] if listed else None)


def _unnamed_devices_note(events: list[GpuEvent]) -> list[str]:
    unnamed = sorted(
        {(e.pid, e.device) for e in events if e.device_uuid is None},
        key=str,
    )
    if not unnamed:
        return []
    listed = ", ".join(f"process {pid} device {device}" for pid, device in unnamed[:5])
    more = f" and {len(unnamed) - 5} more" if len(unnamed) > 5 else ""
    return [f"the report does not name the GPU of {listed}{more}"]


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
    # Per-node graph captures mark their rows with graphNodeId; newer exporters
    # add graphId. Only the kernel table names its rows.
    optional = ", ".join(
        column if column in columns else "NULL" for column in OPTIONAL_COLUMNS
    )
    query = (
        f"select start, end, deviceId, streamId, correlationId, globalPid, "
        f"{optional} from {table}"
    )
    for row in db.execute(query):
        start, end, device, stream, correlation, global_pid = row[:6]
        name_id, graph_id, node_id = row[6:]
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
            graph_id=_int(graph_id) or None,
            pid=pid,
            device_uuid=uuid,
            device_name=device_name,
            graph_node_id=_int(node_id),
        )


def _launches(
    db: sqlite3.Connection, strings: dict[int, str]
) -> Iterator[tuple[tuple[int | None, int], LaunchCall]]:
    query = "select start, globalTid, correlationId, nameId " f"from {RUNTIME_TABLE}"
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
