"""Import PyTorch/Kineto profiler traces as inference activity references.

A Kineto trace (PyTorch profiler, or vLLM's ``rank*.pt.trace.json.gz``) lists
GPU kernels, copies, and memsets with device timestamps, and the CPU runtime or
driver calls that launched them. Each GPU event carries the CUDA correlation ID
of its launch call. A CUDA graph launch is one call and many GPU events.

This importer emits one ``infer.activity_ref`` per launch, device, and activity
kind (``detail="launch"``, the default) or per GPU event (``detail="kernel"``). A launch record spans its
first event's start to its last event's end. When it holds several events, its
``metadata.intervals`` lists the exact busy intervals inside that span, so
GPU time accounting does not count the idle gaps inside a CUDA graph replay as
busy. On vLLM 0.30.0 traces a launch record set is about 6x smaller than a
per-event one and gives the same device busy time. A GPU
activity is linked to an iteration only when its launch call sits inside exactly
one ``stormlog.iteration/...`` range on the launching thread. Otherwise it stays
unresolved and says why. Launch calls are CPU work and are never emitted as GPU
activity.
"""

from __future__ import annotations

import gzip
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import __version__
from .correlation_capture import CaptureCapabilities, TraceAttachment, TraceCapture
from .correlation_events import ActivityReferenceEvent, CorrelationContext, EntityRef
from .trace_ranges import parse_iteration_range

PRODUCER_ID = "stormlog.trace.kineto"
GPU_CATEGORIES = {
    "kernel": "gpu_kernel",
    "gpu_memcpy": "gpu_memcpy",
    "gpu_memset": "gpu_memset",
}
LAUNCH_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
RANGE_CATEGORIES = frozenset({"user_annotation"})
SUPPORTED = (
    "gpu_activity",
    "cuda_correlation",
    "iteration_ranges",
    "cuda_graph_ids",
    "streams",
)
Detail = Literal["kernel", "launch"]


@dataclass(frozen=True)
class GpuEvent:
    """One kernel, copy, or memset as the trace recorded it."""

    start_ns: int
    end_ns: int
    kind: str
    name: str
    device: int | None
    stream: int | None
    correlation: int | None
    graph_id: int | None
    # Set by formats that hold several processes (Nsight Systems); a Kineto
    # trace is one process, so its GPU events leave these unset.
    pid: int | None = None
    device_uuid: str | None = None


@dataclass(frozen=True)
class LaunchCall:
    """The CPU runtime or driver call that launched GPU work."""

    pid: int
    tid: int
    ts_us: float
    name: str


@dataclass(frozen=True)
class IterationSpan:
    start_us: float
    end_us: float
    iteration_ref: EntityRef


@dataclass(frozen=True)
class GpuLink:
    """An event's iteration link, or the reason there is none."""

    iteration_ref: EntityRef | None
    reason: str | None
    launch: LaunchCall | None


@dataclass
class KinetoTrace:
    """The parts of a profiler trace the importer uses (Kineto or Nsight)."""

    base_ns: int
    host: str | None
    trace_id: str | None
    rank: int | None
    world_size: int | None
    engine_version: str | None
    cupti_version: str | None
    device_names: dict[int, str]
    source: str = "kineto"
    gpu_events: list[GpuEvent] = field(default_factory=list)
    # Keyed by (pid, correlation); pid is None for single-process formats.
    launches: dict[tuple[int | None, int], LaunchCall] = field(default_factory=dict)
    spans: dict[tuple[int, int], list[IterationSpan]] = field(default_factory=dict)
    span_starts: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    longest_span_us: dict[tuple[int, int], float] = field(default_factory=dict)


def load_kineto_trace(path: str | Path) -> KinetoTrace:
    """Parse a plain or gzipped Kineto Chrome trace."""
    document = _read_json(Path(path))
    if not isinstance(document, dict) or not isinstance(
        document.get("traceEvents"), list
    ):
        raise ValueError("not a Kineto trace: missing traceEvents")
    trace = _trace_header(document)
    for index, event in enumerate(document["traceEvents"]):
        if isinstance(event, dict) and event.get("ph") == "X":
            _add_checked_event(trace, event, index)
    index_spans(trace)
    return trace


def link_gpu_event(trace: KinetoTrace, event: GpuEvent) -> GpuLink:
    """Link a GPU event to the one iteration range around its launch call."""
    launch = (
        trace.launches.get((event.pid, event.correlation))
        if event.correlation
        else None
    )
    if launch is None:
        return GpuLink(None, "no_launch_record", None)
    refs = {span.iteration_ref for span in _enclosing_spans(trace, launch)}
    if not refs:
        return GpuLink(None, "launch_outside_iteration_range", launch)
    if len(refs) > 1:
        return GpuLink(None, "ambiguous_iteration_range", launch)
    return GpuLink(next(iter(refs)), None, launch)


def import_kineto_trace(
    path: str | Path,
    *,
    run_id: str,
    session_id: str,
    attachment: TraceAttachment | None = None,
    device_uuids: Mapping[int, str] | None = None,
    detail: Detail = "launch",
) -> TraceCapture:
    """Return activity references, capabilities, and a summary for one trace."""
    if detail not in ("kernel", "launch"):
        raise ValueError("detail must be 'kernel' or 'launch'")
    return capture_trace(
        load_kineto_trace(path),
        path,
        run_id=run_id,
        session_id=session_id,
        attachment=attachment,
        device_uuids=device_uuids,
        detail=detail,
    )


def capture_trace(
    trace: KinetoTrace,
    path: str | Path,
    *,
    run_id: str,
    session_id: str,
    attachment: TraceAttachment | None = None,
    device_uuids: Mapping[int, str] | None = None,
    detail: Detail = "launch",
) -> TraceCapture:
    """Build the capture for a loaded trace of any supported format."""
    trace_key = attachment.attachment_id if attachment else _trace_key(trace, path)
    builder = _EventBuilder(
        trace=trace,
        run_id=run_id,
        session_id=session_id,
        trace_key=trace_key,
        attachment_id=attachment.attachment_id if attachment else None,
        device_uuids=dict(device_uuids or {}),
    )
    groups = _group_events(trace, detail)
    events = tuple(
        event
        for key, members in groups.items()
        for event in builder.build(key, members)
    )
    return TraceCapture(
        capabilities=_capabilities(trace),
        events=events,
        attachments=(attachment,) if attachment else (),
        summary=_summary(trace, groups, builder, detail, path),
    )


# ---------------------------------------------------------------- parsing


def _read_json(path: Path) -> Any:
    """Load the trace; a truncated or corrupt file is a ``ValueError``."""
    with path.open("rb") as handle:
        magic = handle.read(2)
    try:
        if magic == b"\x1f\x8b":
            with gzip.open(path, "rt", encoding="utf-8") as compressed:
                return json.load(compressed)
        with path.open(encoding="utf-8") as plain:
            return json.load(plain)
    except (EOFError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
        raise ValueError(f"truncated or corrupt trace file: {exc}") from exc


def _add_checked_event(trace: KinetoTrace, event: dict[str, Any], index: int) -> None:
    try:
        _add_event(trace, event)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"traceEvents[{index}] is malformed: {exc!r}") from exc


def _trace_header(document: dict[str, Any]) -> KinetoTrace:
    distributed = document.get("distributedInfo") or {}
    devices = {
        int(item["id"]): str(item.get("name", ""))
        for item in document.get("deviceProperties") or []
        if isinstance(item, dict) and "id" in item
    }
    cupti = document.get("cupti_version")
    return KinetoTrace(
        base_ns=int(document.get("baseTimeNanoseconds") or 0),
        host=document.get("host_name"),
        trace_id=document.get("trace_id"),
        rank=_optional_int(distributed.get("rank")),
        world_size=_optional_int(distributed.get("world_size")),
        engine_version=document.get("vllm_version"),
        cupti_version=str(cupti) if cupti is not None else None,
        device_names=devices,
    )


def _add_event(trace: KinetoTrace, event: dict[str, Any]) -> None:
    category = event.get("cat")
    args = event.get("args") or {}
    if category in GPU_CATEGORIES:
        trace.gpu_events.append(_gpu_event(trace, event, args))
    elif category in LAUNCH_CATEGORIES and args.get("correlation"):
        trace.launches[(None, int(args["correlation"]))] = LaunchCall(
            pid=int(event["pid"]),
            tid=int(event["tid"]),
            ts_us=_event_times(event)[0],
            name=str(event.get("name", "")),
        )
    elif category in RANGE_CATEGORIES:
        _add_span(trace, event)


def _gpu_event(
    trace: KinetoTrace, event: dict[str, Any], args: dict[str, Any]
) -> GpuEvent:
    start_us, duration_us = _event_times(event)
    end_us = start_us + duration_us
    graph_id = _optional_int(args.get("graph id"))
    return GpuEvent(
        start_ns=_to_ns(trace, start_us),
        end_ns=_to_ns(trace, end_us),
        kind=GPU_CATEGORIES[event["cat"]],
        name=str(event.get("name", "")),
        device=_optional_int(args.get("device")),
        stream=_optional_int(args.get("stream")),
        correlation=_optional_int(args.get("correlation")) or None,
        graph_id=graph_id or None,
    )


def _add_span(trace: KinetoTrace, event: dict[str, Any]) -> None:
    ref = parse_iteration_range(str(event.get("name", "")))
    if ref is None:
        return
    start_us, duration_us = _event_times(event)
    span = IterationSpan(start_us, start_us + duration_us, ref)
    trace.spans.setdefault((int(event["pid"]), int(event["tid"])), []).append(span)


def _event_times(event: dict[str, Any]) -> tuple[float, float]:
    """An event's start and duration in microseconds, checked before any use.

    A negative or non-finite value would otherwise hide inside the merged span
    of a multi-event launch, where the record's own span check cannot see it.
    """
    start_us = float(event["ts"])
    duration_us = float(event.get("dur") or 0.0)
    if not math.isfinite(start_us) or not math.isfinite(duration_us):
        raise ValueError("ts and dur must be finite")
    if duration_us < 0:
        raise ValueError(f"negative duration {duration_us}")
    return start_us, duration_us


def index_spans(trace: KinetoTrace) -> None:
    """Sort each thread's iteration spans and index them for launch lookups."""
    for thread, spans in trace.spans.items():
        spans.sort(key=lambda span: (span.start_us, -span.end_us))
        trace.span_starts[thread] = [span.start_us for span in spans]
        trace.longest_span_us[thread] = max(
            span.end_us - span.start_us for span in spans
        )


def _enclosing_spans(trace: KinetoTrace, launch: LaunchCall) -> list[IterationSpan]:
    """Spans on the launch thread whose interval contains the launch call.

    Only spans that start no earlier than the longest span's length before the
    call can contain it, so the scan stops there.
    """
    thread = (launch.pid, launch.tid)
    spans = trace.spans.get(thread)
    if not spans:
        return []
    earliest = launch.ts_us - trace.longest_span_us[thread]
    found = []
    for index in range(
        bisect_right(trace.span_starts[thread], launch.ts_us) - 1, -1, -1
    ):
        span = spans[index]
        if span.start_us < earliest:
            break
        # Chrome trace "X" events cover [ts, ts + dur).
        if launch.ts_us < span.end_us:
            found.append(span)
    return found


def _to_ns(trace: KinetoTrace, value_us: float) -> int:
    return trace.base_ns + int(round(value_us * 1000))


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_key(trace: KinetoTrace, path: str | Path) -> str:
    return trace.trace_id or Path(path).name


# ---------------------------------------------------------------- grouping


GroupKey = tuple[Any, ...]


def _group_events(trace: KinetoTrace, detail: Detail) -> dict[GroupKey, list[GpuEvent]]:
    """Group GPU events into the records to emit, with each record's link."""
    groups: dict[GroupKey, list[GpuEvent]] = defaultdict(list)
    links: dict[tuple[int | None, int | None], GpuLink] = {}
    for index, event in enumerate(trace.gpu_events):
        launch = (event.pid, event.correlation)
        if launch not in links:
            links[launch] = link_gpu_event(trace, event)
        link = links[launch]
        launch_key: Any = launch if detail == "launch" else index
        if event.correlation is None:
            launch_key = ("uncorrelated", index)
        groups[(event.device, event.kind, launch_key, link)].append(event)
    return groups


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge intervals into disjoint, sorted ones; touching intervals join."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


# ---------------------------------------------------------------- events


@dataclass
class _EventBuilder:
    trace: KinetoTrace
    run_id: str
    session_id: str
    trace_key: str
    attachment_id: str | None
    device_uuids: dict[int, str]
    count: int = 0
    record_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _contexts: dict[tuple[int | None, int | None, str | None], CorrelationContext] = (
        field(default_factory=dict)
    )

    def build(
        self, key: GroupKey, members: list[GpuEvent]
    ) -> list[ActivityReferenceEvent]:
        device, kind, _, link = key
        uuid = members[0].device_uuid or self.device_uuids.get(device)
        pid = link.launch.pid if link.launch else members[0].pid
        context = self._context(device, pid, uuid)
        metadata = _group_metadata(members, link)
        start = min(event.start_ns for event in members)
        end = max(event.end_ns for event in members)
        self.record_spans.setdefault(_device_key(members[0]), []).append((start, end))
        return [self._event(context, kind, link, members, start, end, metadata)]

    def _event(
        self,
        context: CorrelationContext,
        kind: str,
        link: GpuLink,
        members: list[GpuEvent],
        start_ns: int,
        end_ns: int,
        metadata: dict[str, Any],
    ) -> ActivityReferenceEvent:
        self.count += 1
        identity = f"{self.trace_key}/{self.count}"
        streams = {event.stream for event in members}
        graphs = {event.graph_id for event in members}
        return ActivityReferenceEvent(
            context=context,
            event_id=identity,
            activity_ref=EntityRef(PRODUCER_ID, identity),
            activity_kind=kind,
            activity_domain="gpu",
            attribution_status="linked" if link.iteration_ref else "unresolved",
            iteration_ref=link.iteration_ref,
            trace_attachment_id=self.attachment_id,
            cuda_correlation_id=members[0].correlation,
            stream_id=streams.pop() if len(streams) == 1 else None,
            graph_id=graphs.pop() if len(graphs) == 1 else None,
            start_ns=start_ns,
            end_ns=end_ns,
            metadata=metadata,
        )

    def _context(
        self, device: int | None, pid: int | None, uuid: str | None
    ) -> CorrelationContext:
        key = (device, pid, uuid)
        if key not in self._contexts:
            self._contexts[key] = _context(self, pid, uuid)
        return self._contexts[key]


def _device_key(event: GpuEvent) -> str:
    """How the summary names a device: per process when the format has several."""
    return str(event.device) if event.pid is None else f"{event.pid}/{event.device}"


def _context(
    builder: _EventBuilder, pid: int | None, uuid: str | None
) -> CorrelationContext:
    trace = builder.trace
    return CorrelationContext(
        run_id=builder.run_id,
        session_id=builder.session_id,
        producer_id=PRODUCER_ID,
        source=trace.source,
        source_version=__version__,
        engine="vllm" if trace.engine_version else None,
        engine_version=trace.engine_version,
        backend="cupti",
        backend_version=trace.cupti_version,
        host=trace.host,
        pid=pid,
        device_uuid=uuid,
        rank=trace.rank,
        world_size=trace.world_size,
        clock_domain=(
            f"{trace.source}:{trace.host or 'unknown-host'}:{builder.trace_key}"
        ),
        clock_kind="device",
        collection_mode="imported",
        provenance="observed",
    )


def _group_metadata(members: list[GpuEvent], link: GpuLink) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "event_count": len(members),
        "summed_duration_ns": sum(event.end_ns - event.start_ns for event in members),
        "first_name": members[0].name,
        "device_index": members[0].device,
    }
    if link.launch is not None:
        metadata["launch_api"] = link.launch.name
    if link.reason is not None:
        metadata["unresolved_reason"] = link.reason
    streams = sorted({event.stream for event in members if event.stream is not None})
    if len(streams) > 1:
        metadata["streams"] = streams
    if len(members) > 1:
        metadata.update(_launch_intervals(members))
    return metadata


def _launch_intervals(members: list[GpuEvent]) -> dict[str, Any]:
    """Busy intervals inside a launch, as offsets from its first event."""
    busy = merge_intervals((event.start_ns, event.end_ns) for event in members)
    origin = busy[0][0]
    return {
        "intervals": [[start - origin, end - start] for start, end in busy],
        "busy_ns": sum(end - start for start, end in busy),
    }


# ---------------------------------------------------------------- summary


def _capabilities(trace: KinetoTrace) -> CaptureCapabilities:
    events = trace.gpu_events
    found = {
        "gpu_activity": bool(events),
        "cuda_correlation": any(event.correlation for event in events),
        "iteration_ranges": bool(trace.spans),
        "cuda_graph_ids": any(event.graph_id for event in events),
        "streams": any(event.stream is not None for event in events),
    }
    return CaptureCapabilities(
        supported=SUPPORTED,
        enabled=SUPPORTED,
        collected=tuple(name for name in SUPPORTED if found[name]),
    )


def _summary(
    trace: KinetoTrace,
    groups: dict[GroupKey, list[GpuEvent]],
    builder: _EventBuilder,
    detail: Detail,
    path: str | Path,
) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    linked = 0
    for key, members in groups.items():
        link: GpuLink = key[3]
        if link.reason is None:
            linked += len(members)
        else:
            reasons[link.reason] += len(members)
    return {
        "format": trace.source,
        "detail": detail,
        "path": str(Path(path).resolve()),
        "trace_id": trace.trace_id,
        "host": trace.host,
        "rank": trace.rank,
        "processes": sorted({launch.pid for launch in trace.launches.values()}),
        "engine_version": trace.engine_version,
        "cupti_version": trace.cupti_version,
        "gpu_events": len(trace.gpu_events),
        "activity_records": builder.count,
        "linked_gpu_events": linked,
        "unresolved_gpu_events": dict(sorted(reasons.items())),
        "graph_gpu_events": sum(
            1 for event in trace.gpu_events if _from_graph(trace, event)
        ),
        "devices": _device_summary(trace, builder),
        "event_loss": None,
        "event_loss_note": "the trace does not report dropped CUPTI records",
    }


def _from_graph(trace: KinetoTrace, event: GpuEvent) -> bool:
    """A CUDA graph's work: a graph ID, or launched by cudaGraphLaunch."""
    if event.graph_id:
        return True
    launch = trace.launches.get((event.pid, event.correlation or 0))
    return launch is not None and launch.name.startswith("cudaGraphLaunch")


def _device_summary(
    trace: KinetoTrace, builder: _EventBuilder
) -> dict[str, dict[str, Any]]:
    """Per device: exact busy time, and the time the record spans cover.

    Spans also cover idle gaps inside multi-event launches; the records'
    ``metadata.intervals`` exclude them, so accounting matches ``busy_ns``.
    """
    by_device: dict[str, list[GpuEvent]] = defaultdict(list)
    for event in trace.gpu_events:
        by_device[_device_key(event)].append(event)
    summary = {}
    for key, events in sorted(by_device.items()):
        device = events[0].device
        busy = merge_intervals((event.start_ns, event.end_ns) for event in events)
        records = merge_intervals(builder.record_spans.get(key, []))
        summary[key] = {
            "name": trace.device_names.get(device) if device is not None else None,
            "device_uuid": events[0].device_uuid
            or (builder.device_uuids.get(device) if device is not None else None),
            "busy_ns": sum(end - start for start, end in busy),
            "launch_span_ns": sum(end - start for start, end in records),
            "summed_ns": sum(event.end_ns - event.start_ns for event in events),
        }
    return summary


__all__ = [
    "Detail",
    "GpuEvent",
    "IterationSpan",
    "LaunchCall",
    "capture_trace",
    "index_spans",
    "GpuLink",
    "KinetoTrace",
    "import_kineto_trace",
    "link_gpu_event",
    "load_kineto_trace",
    "merge_intervals",
]
