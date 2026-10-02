"""Resolve inference evidence before computing overlap-aware GPU time."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Iterable

from .correlation_events import (
    ActivityReferenceEvent,
    ClockAlignmentEvent,
    CorrelationEvent,
    EntityRef,
    InferenceRecord,
    IterationEvent,
    MembershipEvent,
    RequestEvent,
    StageEvent,
)


@dataclass(frozen=True)
class UnresolvedReference:
    event_id: str
    kind: str
    ref: EntityRef


@dataclass(frozen=True)
class CorrelationGraph:
    run_id: str | None
    requests: dict[tuple[EntityRef, EntityRef | None], RequestEvent]
    iterations: dict[EntityRef, IterationEvent]
    stages: dict[EntityRef, StageEvent]
    memberships: tuple[MembershipEvent, ...]
    activities: dict[EntityRef, ActivityReferenceEvent]
    alignments: tuple[ClockAlignmentEvent, ...]
    unresolved: tuple[UnresolvedReference, ...]


@dataclass(frozen=True)
class DeviceClock:
    device_uuid: str
    clock_domain: str
    clock_kind: str = "monotonic"


@dataclass(frozen=True)
class GpuTime:
    summed_activity_ns: int
    busy_ns: int
    activity_count: int


@dataclass(frozen=True)
class IterationTiming:
    elapsed_ns: int | None
    gpu: dict[DeviceClock, GpuTime]


@dataclass(frozen=True)
class RunAccounting:
    run_id: str | None
    iterations: dict[EntityRef, IterationTiming]
    device_totals: dict[DeviceClock, GpuTime]
    unattributed_activity_refs: tuple[EntityRef, ...]
    unmeasured_gpu_activity_refs: tuple[EntityRef, ...]


@dataclass(frozen=True)
class RequestShareEstimate:
    request_ref: EntityRef
    duration_ns: int
    model: str
    attempt_ref: EntityRef | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.duration_ns, int)
            or isinstance(self.duration_ns, bool)
            or self.duration_ns < 0
        ):
            raise ValueError("estimated duration_ns must be non-negative")
        if not self.model:
            raise ValueError("request share needs a model name")


@dataclass(frozen=True)
class ShareBudget:
    iteration_ref: EntityRef
    device_clock: DeviceClock
    budget_ns: int
    shares: tuple[RequestShareEstimate, ...]
    unattributed_ns: int


@dataclass(frozen=True)
class AlignedTimestamp:
    value_ns: int
    uncertainty_ns: int
    clock_domain: str


def resolve_inference_events(records: Iterable[InferenceRecord]) -> CorrelationGraph:
    """Deduplicate events, then resolve references independent of delivery order."""
    run_id, events = _deduplicate_events(records)
    requests: dict[tuple[EntityRef, EntityRef | None], RequestEvent] = {}
    iterations: dict[EntityRef, IterationEvent] = {}
    stages: dict[EntityRef, StageEvent] = {}
    memberships: dict[tuple[object, ...], MembershipEvent] = {}
    activities: dict[EntityRef, ActivityReferenceEvent] = {}
    alignments: dict[tuple[object, ...], ClockAlignmentEvent] = {}
    for event in events:
        if isinstance(event, RequestEvent):
            _put_entity(requests, (event.request_ref, event.attempt_ref), event)
        elif isinstance(event, IterationEvent):
            _put_entity(iterations, event.iteration_ref, event)
        elif isinstance(event, StageEvent):
            _put_entity(stages, event.stage_ref, event)
        elif isinstance(event, MembershipEvent):
            _put_entity(memberships, _membership_key(event), event)
        elif isinstance(event, ActivityReferenceEvent):
            _put_entity(activities, event.activity_ref, event)
        elif isinstance(event, ClockAlignmentEvent):
            _put_entity(alignments, _alignment_key(event), event)
    ordered_memberships = tuple(sorted(memberships.values(), key=_event_sort_key))
    ordered_alignments = tuple(sorted(alignments.values(), key=_event_sort_key))
    graph = CorrelationGraph(
        run_id=run_id,
        requests=requests,
        iterations=iterations,
        stages=stages,
        memberships=ordered_memberships,
        activities=activities,
        alignments=ordered_alignments,
        unresolved=(),
    )
    return replace(graph, unresolved=_find_unresolved(graph))


def _deduplicate_events(
    records: Iterable[InferenceRecord],
) -> tuple[str | None, tuple[CorrelationEvent, ...]]:
    run_id: str | None = None
    seen: dict[tuple[str, str, str], CorrelationEvent] = {}
    for record in records:
        if not isinstance(record, CorrelationEvent):
            continue
        if run_id is None:
            run_id = record.context.run_id
        if record.context.run_id != run_id:
            raise ValueError("correlation artifact contains multiple run_id values")
        key = (record.context.session_id, record.context.producer_id, record.event_id)
        previous = seen.get(key)
        if previous is not None and previous != record:
            raise ValueError("conflicting event identity")
        seen[key] = record
    return run_id, tuple(seen.values())


def _put_entity(table: dict, key: object, event: CorrelationEvent) -> None:
    previous = table.get(key)
    if previous is not None and not _same_entity(previous, event):
        raise ValueError("conflicting entity identity")
    if previous is None or event.event_id < previous.event_id:
        table[key] = event


def _same_entity(left: CorrelationEvent, right: CorrelationEvent) -> bool:
    left_record = left.to_record()
    right_record = right.to_record()
    left_record.pop("event_id")
    right_record.pop("event_id")
    return left_record == right_record


def _membership_key(event: MembershipEvent) -> tuple[object, ...]:
    return (event.request_ref, event.attempt_ref, event.iteration_ref, event.role)


def _alignment_key(event: ClockAlignmentEvent) -> tuple[object, ...]:
    return (
        event.context.session_id,
        event.context.producer_id,
        event.from_clock_domain,
        event.to_clock_domain,
        event.offset_ns,
        event.uncertainty_ns,
        event.valid_from_ns,
        event.valid_to_ns,
    )


def _event_sort_key(event: CorrelationEvent) -> tuple[str, str, str]:
    return event.context.session_id, event.context.producer_id, event.event_id


def _find_unresolved(graph: CorrelationGraph) -> tuple[UnresolvedReference, ...]:
    missing: list[UnresolvedReference] = []
    known_requests = {request for request, _attempt in graph.requests}
    for membership in graph.memberships:
        if membership.request_ref not in known_requests:
            missing.append(
                UnresolvedReference(
                    membership.event_id, "request", membership.request_ref
                )
            )
        elif (
            membership.attempt_ref is not None
            and (membership.request_ref, membership.attempt_ref) not in graph.requests
        ):
            missing.append(
                UnresolvedReference(
                    membership.event_id, "attempt", membership.attempt_ref
                )
            )
        if membership.iteration_ref not in graph.iterations:
            missing.append(
                UnresolvedReference(
                    membership.event_id, "iteration", membership.iteration_ref
                )
            )
    for stage in graph.stages.values():
        _missing_optional_ref(
            missing, stage.event_id, "request", stage.request_ref, known_requests
        )
        _missing_optional_ref(
            missing, stage.event_id, "iteration", stage.iteration_ref, graph.iterations
        )
    for activity in graph.activities.values():
        _missing_optional_ref(
            missing,
            activity.event_id,
            "iteration",
            activity.iteration_ref,
            graph.iterations,
        )
    return tuple(
        sorted(missing, key=lambda item: (item.event_id, item.kind, item.ref.id))
    )


def _missing_optional_ref(
    missing: list[UnresolvedReference],
    event_id: str,
    kind: str,
    ref: EntityRef | None,
    known: Collection[EntityRef],
) -> None:
    if ref is not None and ref not in known:
        missing.append(UnresolvedReference(event_id, kind, ref))


def account_gpu_time(graph: CorrelationGraph) -> RunAccounting:
    """Keep summed activity, interval union, and iteration elapsed separate."""
    device_groups: dict[DeviceClock, list[list[tuple[int, int]]]] = {}
    iteration_groups: dict[
        EntityRef, dict[DeviceClock, list[list[tuple[int, int]]]]
    ] = {}
    unmeasured: list[EntityRef] = []
    unattributed: list[EntityRef] = []
    for activity in graph.activities.values():
        if activity.attribution_status == "unresolved":
            unattributed.append(activity.activity_ref)
        if activity.activity_domain != "gpu":
            continue
        measured = _measured_busy(activity)
        if measured is None:
            unmeasured.append(activity.activity_ref)
            continue
        key, busy = measured
        device_groups.setdefault(key, []).append(busy)
        if activity.iteration_ref in graph.iterations:
            by_device = iteration_groups.setdefault(activity.iteration_ref, {})
            by_device.setdefault(key, []).append(busy)
    iteration_timings = {
        ref: IterationTiming(
            elapsed_ns=iteration.elapsed_ns,
            gpu={
                key: _gpu_time(items)
                for key, items in iteration_groups.get(ref, {}).items()
            },
        )
        for ref, iteration in graph.iterations.items()
    }
    return RunAccounting(
        run_id=graph.run_id,
        iterations=iteration_timings,
        device_totals={key: _gpu_time(items) for key, items in device_groups.items()},
        unattributed_activity_refs=tuple(sorted(unattributed, key=_ref_sort_key)),
        unmeasured_gpu_activity_refs=tuple(sorted(unmeasured, key=_ref_sort_key)),
    )


def _measured_busy(
    activity: ActivityReferenceEvent,
) -> tuple[DeviceClock, list[tuple[int, int]]] | None:
    """The device clock and busy intervals of a measurable GPU activity."""
    key = _device_clock(activity)
    busy = activity_busy_intervals(activity) if key is not None else None
    if key is None or busy is None:
        return None
    return key, busy


def _device_clock(activity: ActivityReferenceEvent) -> DeviceClock | None:
    context = activity.context
    if (
        context.device_uuid is None
        or context.clock_kind not in {"monotonic", "device"}
        or activity.start_ns is None
        or activity.end_ns is None
    ):
        return None
    return DeviceClock(context.device_uuid, context.clock_domain, context.clock_kind)


def activity_busy_intervals(
    activity: ActivityReferenceEvent,
) -> list[tuple[int, int]] | None:
    """The intervals in which a GPU activity kept its device busy.

    An activity covers ``[start_ns, end_ns)`` unless its metadata lists
    ``intervals``: ``[offset_ns, duration_ns]`` pairs from ``start_ns``, sorted
    and disjoint, inside the span. A launch record uses them so that idle gaps
    inside one launch are not counted as busy. Malformed intervals return None
    and the activity stays unmeasured rather than falling back to its span.
    """
    start, end = activity.start_ns, activity.end_ns
    if start is None or end is None:
        return None
    raw = activity.metadata.get("intervals")
    if raw is None:
        return [(start, end)]
    return _offset_intervals(raw, start, end)


def _offset_intervals(
    raw: object, start: int, end: int
) -> list[tuple[int, int]] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for item in raw:
        pair = _int_pair(item)
        if pair is None or pair[0] < cursor or pair[1] < 0:
            return None
        offset, duration = pair
        cursor = offset + duration
        if start + cursor > end:
            return None
        intervals.append((start + offset, start + cursor))
    return intervals


def _int_pair(item: object) -> tuple[int, int] | None:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return None
    first, second = item
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (first, second)):
        return None
    return first, second


def _gpu_time(activities: list[list[tuple[int, int]]]) -> GpuTime:
    """Summed and merged busy time of activities given as their busy intervals."""
    intervals = [interval for busy in activities for interval in busy]
    return GpuTime(
        summed_activity_ns=sum(end - start for start, end in intervals),
        busy_ns=_merged_duration(intervals),
        activity_count=len(activities),
    )


def _merged_duration(intervals: list[tuple[int, int]]) -> int:
    busy = 0
    current_start: int | None = None
    current_end = 0
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            busy += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        busy += current_end - current_start
    return busy


def _ref_sort_key(ref: EntityRef) -> tuple[str, str]:
    return ref.producer_id, ref.id


def validate_request_shares(
    graph: CorrelationGraph,
    accounting: RunAccounting,
    *,
    iteration_ref: EntityRef,
    device_clock: DeviceClock,
    shares: tuple[RequestShareEstimate, ...],
    unattributed_ns: int,
) -> ShareBudget:
    """Validate an explicit model against one measured iteration GPU budget."""
    iteration = accounting.iterations.get(iteration_ref)
    if iteration is None or device_clock not in iteration.gpu:
        raise ValueError("no measured GPU budget for iteration and device clock")
    if (
        not isinstance(unattributed_ns, int)
        or isinstance(unattributed_ns, bool)
        or unattributed_ns < 0
    ):
        raise ValueError("unattributed_ns must be non-negative")
    members = {
        (event.request_ref, event.attempt_ref)
        for event in graph.memberships
        if event.iteration_ref == iteration_ref
    }
    _validate_share_members(shares, members)
    budget_ns = iteration.gpu[device_clock].busy_ns
    if sum(share.duration_ns for share in shares) + unattributed_ns != budget_ns:
        raise ValueError(
            "estimated shares and remainder must equal the measured budget"
        )
    return ShareBudget(iteration_ref, device_clock, budget_ns, shares, unattributed_ns)


def _validate_share_members(
    shares: tuple[RequestShareEstimate, ...],
    members: set[tuple[EntityRef, EntityRef | None]],
) -> None:
    seen: set[tuple[EntityRef, EntityRef | None]] = set()
    models: set[str] = set()
    for share in shares:
        models.add(share.model)
        identity = (share.request_ref, share.attempt_ref)
        if not _is_member(identity, members):
            raise ValueError("estimated request is not a member of the iteration")
        if identity in seen:
            raise ValueError("duplicate request share")
        if any(
            member_request == share.request_ref
            and (member_attempt is None or share.attempt_ref is None)
            for member_request, member_attempt in seen
        ):
            raise ValueError("generic and attempt-specific request shares overlap")
        seen.add(identity)
    if len(models) > 1:
        raise ValueError("request shares must use one estimation model")


def _is_member(
    identity: tuple[EntityRef, EntityRef | None],
    members: set[tuple[EntityRef, EntityRef | None]],
) -> bool:
    request_ref, attempt_ref = identity
    if attempt_ref is None:
        return any(member_request == request_ref for member_request, _ in members)
    return identity in members


def align_timestamp(
    timestamp_ns: int,
    *,
    from_clock_domain: str,
    to_clock_domain: str,
    alignments: Iterable[ClockAlignmentEvent],
) -> AlignedTimestamp:
    """Translate one timestamp while preserving calibration uncertainty."""
    if (
        not isinstance(timestamp_ns, int)
        or isinstance(timestamp_ns, bool)
        or timestamp_ns < 0
    ):
        raise ValueError("timestamp_ns must be non-negative")
    if from_clock_domain == to_clock_domain:
        return AlignedTimestamp(timestamp_ns, 0, to_clock_domain)
    matches = covering_alignments(
        timestamp_ns,
        from_clock_domain=from_clock_domain,
        to_clock_domain=to_clock_domain,
        alignments=alignments,
    )
    if len(matches) != 1:
        raise ValueError("no valid clock alignment or multiple ambiguous alignments")
    alignment = matches[0]
    return AlignedTimestamp(
        timestamp_ns + alignment.offset_ns,
        alignment.uncertainty_ns,
        to_clock_domain,
    )


def covering_alignments(
    timestamp_ns: int,
    *,
    from_clock_domain: str,
    to_clock_domain: str,
    alignments: Iterable[ClockAlignmentEvent],
) -> list[ClockAlignmentEvent]:
    """Return the alignments whose domains and validity window cover a timestamp.

    ``align_timestamp`` needs exactly one; callers can use the count to tell a
    missing alignment from an ambiguous one.
    """
    return [
        item
        for item in alignments
        if item.from_clock_domain == from_clock_domain
        and item.to_clock_domain == to_clock_domain
        and _alignment_covers(item, timestamp_ns)
    ]


def _alignment_covers(alignment: ClockAlignmentEvent, timestamp_ns: int) -> bool:
    return (
        alignment.valid_from_ns is None or timestamp_ns >= alignment.valid_from_ns
    ) and (alignment.valid_to_ns is None or timestamp_ns < alignment.valid_to_ns)


__all__ = [
    "AlignedTimestamp",
    "CorrelationGraph",
    "DeviceClock",
    "GpuTime",
    "IterationTiming",
    "RequestShareEstimate",
    "RunAccounting",
    "ShareBudget",
    "UnresolvedReference",
    "account_gpu_time",
    "align_timestamp",
    "covering_alignments",
    "resolve_inference_events",
    "validate_request_shares",
]
