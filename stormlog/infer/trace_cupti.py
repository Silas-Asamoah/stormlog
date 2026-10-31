"""Import Stormlog CUPTI traces (``activity.sclz``) as inference activity references.

The experimental CUPTI helper (``native/cupti``) writes one directory per
traced process, holding ``activity.sclz`` and ``cupti_status.json``. The trace
is a magic line followed by frames: a little-endian header of encoded size,
decoded size and record count, then that many zlib-compressed JSON lines. An
all-zero header ends the trace. This reader turns it into the same in-memory
trace as the Kineto reader, so import and accounting rules are shared.

- A trace whose status is not finalized, or names an error, is refused: the
  helper could not vouch for what it wrote.
- Losses the helper counted (CUPTI drops, records past the output limit,
  refused activity buffers, kernels without timestamps) are carried into the
  import summary as ``event_loss``.
- CUPTI timestamps are not converted to wall time, and the trace does not name
  its host or GPU UUIDs: give ``--device-uuid`` to measure devices.
- The helper emits no iteration ranges, so GPU work is not linked to
  iterations.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from .trace_kineto import GpuEvent, KinetoTrace, LaunchCall

TRACE_MAGIC = b"SLCPTZ1\n"
FRAME_HEADER = struct.Struct("<III")
# The helper flushes a frame at 64 KiB of records; this leaves room for one
# oversized record without trusting an arbitrary size from the file.
MAX_FRAME_BYTES = 16 * 1024 * 1024
STATUS_FILENAME = "cupti_status.json"
GPU_KINDS = {"kernel": "gpu_kernel", "memcpy": "gpu_memcpy", "memset": "gpu_memset"}
API_KINDS = frozenset({"runtime", "driver"})
LOSS_COUNTERS = (
    "cupti_dropped_records",
    "local_dropped_records",
    "buffer_refusals",
    "kernels_without_timestamps",
)


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield a finalized trace's records; raise ``ValueError`` on any defect."""
    with Path(path).open("rb") as source:
        if source.read(len(TRACE_MAGIC)) != TRACE_MAGIC:
            raise ValueError("not a Stormlog CUPTI trace")
        while (lines := _read_frame(source)) is not None:
            for line in lines:
                yield _parse_record(line)


def _read_frame(source: BinaryIO) -> list[bytes] | None:
    """One frame's record lines, or None at a clean end frame."""
    header = source.read(FRAME_HEADER.size)
    if len(header) != FRAME_HEADER.size:
        raise ValueError("trace end frame is missing")
    encoded_size, raw_size, count = FRAME_HEADER.unpack(header)
    if (encoded_size, raw_size, count) == (0, 0, 0):
        if source.read(1):
            raise ValueError("trace has bytes after its end frame")
        return None
    if not 0 < encoded_size <= MAX_FRAME_BYTES:
        raise ValueError("frame encoded size is invalid")
    if not 0 < raw_size <= MAX_FRAME_BYTES or count <= 0:
        raise ValueError("frame decoded size or record count is invalid")
    encoded = source.read(encoded_size)
    if len(encoded) != encoded_size:
        raise ValueError("frame is truncated")
    lines = _decode_frame(encoded, raw_size).splitlines()
    if len(lines) != count:
        raise ValueError("frame record count does not match")
    return lines


def _parse_record(line: bytes) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed record: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError("record is not a JSON object")
    return record


def _decode_frame(encoded: bytes, raw_size: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        # Never decode past the declared size, whatever the stream says.
        raw = decoder.decompress(encoded, raw_size + 1)
    except zlib.error as exc:
        raise ValueError(f"frame is corrupt: {exc}") from exc
    if (
        len(raw) != raw_size
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError("frame does not decode to its declared size")
    if not raw.endswith(b"\n"):
        raise ValueError("frame lacks its record terminator")
    return raw


def load_status(trace_path: str | Path) -> dict[str, Any]:
    """The helper's status beside a trace, refused unless it finalized cleanly."""
    path = Path(trace_path).with_name(STATUS_FILENAME)
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{STATUS_FILENAME} is missing beside the trace") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{STATUS_FILENAME} is malformed: {exc}") from exc
    if not isinstance(status, dict):
        raise ValueError(f"{STATUS_FILENAME} is not a JSON object")
    if status.get("initialization_error"):
        raise ValueError(f"the CUPTI helper failed: {status['initialization_error']}")
    if status.get("finalized") is not True:
        raise ValueError("the CUPTI helper did not finalize this trace")
    return status


def load_cupti_trace(path: str | Path) -> KinetoTrace:
    """Read a finalized ``activity.sclz`` and its status."""
    path = Path(path)
    status = load_status(path)
    trace = KinetoTrace(
        base_ns=0,
        host=None,
        trace_id=f"{path.parent.name}/{path.name}",
        rank=None,
        world_size=None,
        engine_version=None,
        cupti_version=_text(status.get("cupti_version")),
        device_names={},
        source="stormlog-cupti",
    )
    for record in iter_records(path):
        _add_record(trace, record)
    trace.event_loss = _event_loss(status)
    lost = {name: count for name, count in trace.event_loss.items() if count}
    if lost:
        trace.notes.append(
            "the CUPTI helper lost records: "
            + ", ".join(f"{name}={count}" for name, count in lost.items())
        )
    if status.get("output_limit_reached"):
        trace.notes.append("the trace reached STORMLOG_CUPTI_MAX_BYTES")
    if not trace.gpu_events:
        trace.notes.append("the trace records no GPU kernels, copies or memsets")
    return trace


def _add_record(trace: KinetoTrace, record: dict[str, Any]) -> None:
    kind = record.get("activity_kind")
    metadata = record.get("metadata") or {}
    correlation = _optional_int(record.get("correlation_id"))
    if kind in API_KINDS:
        _add_launch(trace, kind, metadata, correlation, record.get("cpu_start_ns"))
        return
    start, end = record.get("device_start_ns"), record.get("device_end_ns")
    if kind not in GPU_KINDS or not isinstance(start, int) or not isinstance(end, int):
        reason = f"{kind}_without_timestamps" if kind in GPU_KINDS else str(kind)
        trace.not_imported[reason] = trace.not_imported.get(reason, 0) + 1
        return
    trace.gpu_events.append(
        GpuEvent(
            start_ns=start,
            end_ns=end,
            kind=GPU_KINDS[kind],
            name=str(metadata.get("name") or ""),
            device=_optional_int(metadata.get("device_id")),
            stream=_optional_int(record.get("stream_id")),
            correlation=correlation or None,
            graph_id=_optional_int(record.get("graph_id")) or None,
            graph_node_id=_optional_int(record.get("graph_node_id")),
        )
    )


def _add_launch(
    trace: KinetoTrace,
    kind: str,
    metadata: dict[str, Any],
    correlation: int | None,
    start: Any,
) -> None:
    if not correlation or not isinstance(start, int):
        return
    trace.launches[(None, correlation)] = LaunchCall(
        pid=int(metadata["process_id"]),
        tid=int(metadata["thread_id"]),
        ts_us=start / 1000,
        name=str(metadata.get("name") or f"{kind}:cbid={metadata['cbid']}"),
    )


def _event_loss(status: dict[str, Any]) -> dict[str, int]:
    loss = {}
    for name in LOSS_COUNTERS:
        value = status.get(name, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{STATUS_FILENAME} has an invalid {name}")
        loss[name] = value
    return loss


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected an integer identifier, got {value!r}") from exc


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = ["iter_records", "load_cupti_trace", "load_status"]
