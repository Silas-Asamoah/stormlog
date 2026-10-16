"""Versioned, engine-neutral inference execution records.

The v1 endpoint records in :mod:`stormlog.infer.events` remain client
observations. These v2 records describe server execution without assigning
shared work to individual requests merely because they participated in it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

CORRELATION_SCHEMA_VERSION = 2
# Artifact transport only; standalone model records remain embedded v2/v3.
CONTEXT_REFERENCE_SCHEMA_VERSION = 4
# An activity reference whose metadata lists busy intervals: the device is busy
# inside them, not from start_ns to end_ns, so a v2-only reader must refuse it.
ACTIVITY_INTERVALS_SCHEMA_VERSION = 3


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _optional_nonnegative(value: object, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or null")


def _required_nonnegative(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _span(start_ns: int | None, end_ns: int | None) -> None:
    _optional_nonnegative(start_ns, "start_ns")
    _optional_nonnegative(end_ns, "end_ns")
    if start_ns is not None and end_ns is not None and end_ns < start_ns:
        raise ValueError("end_ns must be >= start_ns")


def _capability_names(values: object, name: str) -> None:
    if not isinstance(values, list) or any(
        not isinstance(item, str) or not item for item in values
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class EntityRef:
    """An ID scoped to its producer within a run."""

    producer_id: str
    id: str

    def __post_init__(self) -> None:
        _nonempty(self.producer_id, "producer_id")
        _nonempty(self.id, "id")


@dataclass(frozen=True)
class CorrelationContext:
    """Capture identity and evidence quality shared by one emitted record."""

    run_id: str
    session_id: str
    producer_id: str
    source: str
    clock_domain: str
    clock_kind: str
    collection_mode: str
    provenance: str
    source_version: str | None = None
    engine: str | None = None
    engine_version: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    host: str | None = None
    pid: int | None = None
    device_uuid: str | None = None
    rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "session_id", "producer_id", "source", "clock_domain"):
            _nonempty(getattr(self, name), name)
        for name in (
            "source_version",
            "engine",
            "engine_version",
            "backend",
            "backend_version",
            "host",
            "device_uuid",
        ):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        for name in ("pid", "rank", "local_rank"):
            _optional_nonnegative(getattr(self, name), name)
        _optional_nonnegative(self.world_size, "world_size")
        if self.world_size == 0:
            raise ValueError("world_size must be >= 1")
        if self.clock_kind not in {"monotonic", "wall", "device"}:
            raise ValueError("clock_kind must be monotonic, wall, or device")
        if self.collection_mode not in {"active", "passive", "imported"}:
            raise ValueError("collection_mode must be active, passive, or imported")
        if self.provenance not in {"observed", "reported", "estimated"}:
            raise ValueError("provenance must be observed, reported, or estimated")


@dataclass(frozen=True, kw_only=True)
class CorrelationEvent:
    """Common event identity; subclasses own their event-specific evidence."""

    context: CorrelationContext
    event_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    EVENT_TYPE: ClassVar[str]

    def __post_init__(self) -> None:
        if not isinstance(self.context, CorrelationContext):
            raise ValueError("context must be a CorrelationContext")
        _nonempty(self.event_id, "event_id")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    @property
    def schema_version(self) -> int:
        return CORRELATION_SCHEMA_VERSION

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.EVENT_TYPE,
            **asdict(self),
        }

    @property
    def elapsed_ns(self) -> int | None:
        """Only local monotonic spans have an unqualified duration."""
        start_ns = cast(int | None, getattr(self, "start_ns", None))
        end_ns = cast(int | None, getattr(self, "end_ns", None))
        if self.context.clock_kind != "monotonic" or start_ns is None or end_ns is None:
            return None
        return end_ns - start_ns


@dataclass(frozen=True, kw_only=True)
class ArtifactIdentityEvent(CorrelationEvent):
    """Versioned run identity for an inference JSONL artifact."""

    EVENT_TYPE: ClassVar[str] = "infer.artifact"
    artifact_kind: str
    created_at_ns: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.artifact_kind, "artifact_kind")
        _required_nonnegative(self.created_at_ns, "created_at_ns")


@dataclass(frozen=True, kw_only=True)
class RequestEvent(CorrelationEvent):
    """One logical request observation; attempts are separate identities."""

    EVENT_TYPE: ClassVar[str] = "infer.request"
    request_ref: EntityRef
    attempt_ref: EntityRef | None = None
    backend_request_ref: EntityRef | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    status: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    kv_bytes: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_ref(self.request_ref, "request_ref")
        _optional_ref(self.attempt_ref, "attempt_ref")
        _optional_ref(self.backend_request_ref, "backend_request_ref")
        _span(self.start_ns, self.end_ns)
        for name in ("input_tokens", "output_tokens", "kv_bytes"):
            _optional_nonnegative(getattr(self, name), name)
        if self.status is not None:
            _nonempty(self.status, "status")


@dataclass(frozen=True, kw_only=True)
class IterationEvent(CorrelationEvent):
    """One server execution iteration, recorded once for all participants."""

    EVENT_TYPE: ClassVar[str] = "infer.iteration"
    iteration_ref: EntityRef
    batch_ref: EntityRef | None = None
    start_ns: int | None = None
    end_ns: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_ref(self.iteration_ref, "iteration_ref")
        _optional_ref(self.batch_ref, "batch_ref")
        _span(self.start_ns, self.end_ns)


@dataclass(frozen=True, kw_only=True)
class StageEvent(CorrelationEvent):
    """A generic named stage span, for LLM and non-LLM workloads."""

    EVENT_TYPE: ClassVar[str] = "infer.stage"
    stage_ref: EntityRef
    name: str
    request_ref: EntityRef | None = None
    iteration_ref: EntityRef | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    kv_bytes: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_ref(self.stage_ref, "stage_ref")
        _nonempty(self.name, "name")
        _optional_ref(self.request_ref, "request_ref")
        _optional_ref(self.iteration_ref, "iteration_ref")
        if self.request_ref is None and self.iteration_ref is None:
            raise ValueError("stage must reference a request or iteration")
        _span(self.start_ns, self.end_ns)
        for name in ("input_tokens", "output_tokens", "kv_bytes"):
            _optional_nonnegative(getattr(self, name), name)


@dataclass(frozen=True, kw_only=True)
class MembershipEvent(CorrelationEvent):
    """One request's role in one shared execution iteration."""

    EVENT_TYPE: ClassVar[str] = "infer.membership"
    request_ref: EntityRef
    iteration_ref: EntityRef
    role: str
    attempt_ref: EntityRef | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    kv_bytes: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_ref(self.request_ref, "request_ref")
        _require_ref(self.iteration_ref, "iteration_ref")
        _nonempty(self.role, "role")
        _optional_ref(self.attempt_ref, "attempt_ref")
        for name in ("input_tokens", "output_tokens", "kv_bytes"):
            _optional_nonnegative(getattr(self, name), name)


@dataclass(frozen=True, kw_only=True)
class ActivityReferenceEvent(CorrelationEvent):
    """A scoped trace activity and its optional link to an iteration."""

    EVENT_TYPE: ClassVar[str] = "infer.activity_ref"
    activity_ref: EntityRef
    activity_kind: str
    attribution_status: str
    activity_domain: str = "unknown"
    iteration_ref: EntityRef | None = None
    correlation_scope: EntityRef | None = None
    trace_attachment_id: str | None = None
    runtime_correlation_id: int | None = None
    cuda_correlation_id: int | None = None
    stream_id: int | None = None
    graph_id: int | None = None
    start_ns: int | None = None
    end_ns: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_ref(self.activity_ref, "activity_ref")
        _nonempty(self.activity_kind, "activity_kind")
        if self.activity_domain not in {"gpu", "runtime", "cpu", "unknown"}:
            raise ValueError("activity_domain must be gpu, runtime, cpu, or unknown")
        _optional_ref(self.iteration_ref, "iteration_ref")
        _optional_ref(self.correlation_scope, "correlation_scope")
        if self.attribution_status not in {"linked", "unresolved"}:
            raise ValueError("attribution_status must be linked or unresolved")
        if (self.iteration_ref is None) != (self.attribution_status == "unresolved"):
            raise ValueError("attribution_status must match the iteration link")
        if self.trace_attachment_id is not None:
            _nonempty(self.trace_attachment_id, "trace_attachment_id")
        for name in (
            "runtime_correlation_id",
            "cuda_correlation_id",
            "stream_id",
            "graph_id",
        ):
            _optional_nonnegative(getattr(self, name), name)
        _span(self.start_ns, self.end_ns)
        if "intervals" in self.metadata:
            activity_busy_intervals(self)

    @property
    def schema_version(self) -> int:
        """Busy intervals change what the span means, so they take version 3."""
        if "intervals" in self.metadata:
            return ACTIVITY_INTERVALS_SCHEMA_VERSION
        return CORRELATION_SCHEMA_VERSION


def activity_busy_intervals(
    activity: ActivityReferenceEvent,
) -> list[tuple[int, int]] | None:
    """The intervals in which a GPU activity kept its device busy, or None.

    Without ``metadata.intervals`` the activity is busy for ``[start_ns,
    end_ns)``, and None means it has no complete span. With them (schema
    version 3) it is busy only inside those ``[offset_ns, duration_ns]`` pairs
    from ``start_ns``, which must be sorted, disjoint, and inside the span;
    anything else is a ``ValueError``.
    """
    start, end = activity.start_ns, activity.end_ns
    raw = activity.metadata.get("intervals")
    if raw is not None:
        return _busy_intervals(raw, start, end)
    if start is None or end is None:
        return None
    return [(start, end)]


def _busy_intervals(
    raw: object, start: int | None, end: int | None
) -> list[tuple[int, int]]:
    if start is None or end is None:
        raise ValueError("metadata.intervals needs start_ns and end_ns")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            "metadata.intervals must be a non-empty list of "
            "[offset_ns, duration_ns] pairs"
        )
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for item in raw:
        offset, duration = _interval_pair(item)
        if offset < cursor:
            raise ValueError("metadata.intervals must be sorted and disjoint")
        cursor = offset + duration
        if start + cursor > end:
            raise ValueError("metadata.intervals must lie inside start_ns..end_ns")
        intervals.append((start + offset, start + cursor))
    return intervals


def _interval_pair(item: object) -> tuple[int, int]:
    if (
        isinstance(item, (list, tuple))
        and len(item) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in item)
        and all(v >= 0 for v in item)
    ):
        return item[0], item[1]
    raise ValueError(
        "metadata.intervals entries must be [offset_ns, duration_ns] pairs of "
        "non-negative integers"
    )


@dataclass(frozen=True, kw_only=True)
class CapabilityEvent(CorrelationEvent):
    """Availability and outcome for one optional capture component."""

    EVENT_TYPE: ClassVar[str] = "infer.capabilities"
    component: str
    available: bool
    supported: list[str] = field(default_factory=list)
    enabled: list[str] = field(default_factory=list)
    collected: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.component, "component")
        if not isinstance(self.available, bool):
            raise ValueError("available must be a boolean")
        for name in ("supported", "enabled", "collected"):
            _capability_names(getattr(self, name), name)
        if not set(self.collected) <= set(self.enabled) <= set(self.supported):
            raise ValueError("collected must be enabled and enabled must be supported")
        if not self.available and (self.supported or self.enabled or self.collected):
            raise ValueError("unavailable components cannot report capabilities")


@dataclass(frozen=True, kw_only=True)
class ClockAlignmentEvent(CorrelationEvent):
    """Offset and uncertainty for comparing two distinct clock domains."""

    EVENT_TYPE: ClassVar[str] = "infer.clock_alignment"
    from_clock_domain: str
    to_clock_domain: str
    offset_ns: int
    uncertainty_ns: int
    valid_from_ns: int | None = None
    valid_to_ns: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.from_clock_domain, "from_clock_domain")
        _nonempty(self.to_clock_domain, "to_clock_domain")
        if not isinstance(self.offset_ns, int) or isinstance(self.offset_ns, bool):
            raise ValueError("offset_ns must be an integer")
        _required_nonnegative(self.uncertainty_ns, "uncertainty_ns")
        _span(self.valid_from_ns, self.valid_to_ns)


@dataclass(frozen=True)
class LegacyInferenceRecord:
    """An unchanged v1 record; server correlation remains unknown."""

    raw: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return dict(self.raw)


InferenceRecord = CorrelationEvent | LegacyInferenceRecord

_EVENT_TYPES: dict[str, type[CorrelationEvent]] = {
    ArtifactIdentityEvent.EVENT_TYPE: ArtifactIdentityEvent,
    RequestEvent.EVENT_TYPE: RequestEvent,
    IterationEvent.EVENT_TYPE: IterationEvent,
    StageEvent.EVENT_TYPE: StageEvent,
    MembershipEvent.EVENT_TYPE: MembershipEvent,
    ActivityReferenceEvent.EVENT_TYPE: ActivityReferenceEvent,
    CapabilityEvent.EVENT_TYPE: CapabilityEvent,
    ClockAlignmentEvent.EVENT_TYPE: ClockAlignmentEvent,
}
# Schema versions a reader accepts per event type; every other type is v2 only.
_SCHEMA_VERSIONS: dict[str, tuple[int, ...]] = {
    ActivityReferenceEvent.EVENT_TYPE: (
        CORRELATION_SCHEMA_VERSION,
        ACTIVITY_INTERVALS_SCHEMA_VERSION,
    ),
}


def is_correlation_event_type(event_type: object) -> bool:
    """Whether ``event_type`` names a correlation event that carries a context."""
    return isinstance(event_type, str) and event_type in _EVENT_TYPES


_REF_FIELDS = (
    "request_ref",
    "attempt_ref",
    "backend_request_ref",
    "iteration_ref",
    "batch_ref",
    "stage_ref",
    "activity_ref",
    "correlation_scope",
)


def _require_ref(value: object, name: str) -> None:
    if not isinstance(value, EntityRef):
        raise ValueError(f"{name} must be an EntityRef")


def _optional_ref(value: object, name: str) -> None:
    if value is not None:
        _require_ref(value, name)


def parse_inference_record(record: Mapping[str, Any]) -> InferenceRecord:
    """Parse one embedded v1/v2/v3 JSONL record without fabricating missing evidence."""
    version = record.get("schema_version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("schema_version must be an integer")
    if version == CONTEXT_REFERENCE_SCHEMA_VERSION:
        raise ValueError("v4 requires load_inference_artifact or a sequence decoder")
    event_type = record.get("event_type")
    if not isinstance(event_type, str) or not event_type.startswith("infer."):
        raise ValueError("event_type must name an inference event")
    if version == 1:
        return LegacyInferenceRecord(dict(record))
    if version not in _SCHEMA_VERSIONS.get(event_type, (CORRELATION_SCHEMA_VERSION,)):
        raise ValueError(f"unsupported inference schema_version: {version}")
    event = _parse_correlation_record(event_type, record)
    _check_declared_version(event, version)
    return event


def _check_declared_version(event: CorrelationEvent, version: int) -> None:
    """A record's version must say what its content means."""
    if event.schema_version == version:
        return
    if version == ACTIVITY_INTERVALS_SCHEMA_VERSION:
        raise ValueError(
            f"{event.EVENT_TYPE} schema_version {version} requires metadata.intervals"
        )
    raise ValueError(
        f"{event.EVENT_TYPE} with metadata.intervals requires schema_version "
        f"{event.schema_version}"
    )


def _parse_correlation_record(
    event_type: str, record: Mapping[str, Any]
) -> CorrelationEvent:
    event_class = _EVENT_TYPES.get(event_type)
    if event_class is None:
        raise ValueError(f"unsupported inference event_type: {event_type}")
    values = dict(record)
    values.pop("schema_version")
    values.pop("event_type")
    context = values.get("context")
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    try:
        values["context"] = CorrelationContext(**context)
        _parse_ref_fields(values)
        return event_class(**values)
    except TypeError as exc:
        raise ValueError(f"invalid {event_type} record: {exc}") from exc


def _parse_ref_fields(values: dict[str, Any]) -> None:
    for name in _REF_FIELDS:
        value = values.get(name)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        values[name] = EntityRef(**value)


def load_inference_artifact(path: str | Path) -> list[InferenceRecord]:
    """Read mixed legacy and correlation records in their original order."""
    from .correlation_codec import iter_inference_records

    events: list[InferenceRecord] = []
    for line_number, record in iter_inference_records(path):
        try:
            events.append(parse_inference_record(record))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
    return events


__all__ = [
    "ACTIVITY_INTERVALS_SCHEMA_VERSION",
    "CORRELATION_SCHEMA_VERSION",
    "CONTEXT_REFERENCE_SCHEMA_VERSION",
    "ActivityReferenceEvent",
    "ArtifactIdentityEvent",
    "CapabilityEvent",
    "ClockAlignmentEvent",
    "CorrelationContext",
    "CorrelationEvent",
    "EntityRef",
    "InferenceRecord",
    "IterationEvent",
    "LegacyInferenceRecord",
    "MembershipEvent",
    "RequestEvent",
    "StageEvent",
    "activity_busy_intervals",
    "is_correlation_event_type",
    "load_inference_artifact",
    "parse_inference_record",
]
