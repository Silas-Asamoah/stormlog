"""Who a case's numbers are about, and the interval its rates divide by.

A case's measured requests are its cohort. Their statuses split the cohort
into explicit populations (offered, dropped, sent, unreachable,
delivery_unknown, rejected, accepted, successful, failed, timed out,
cancelled), each counted once, and the cohort is checked for the records a
run should have: unique identities, every scheduled arrival exactly once,
and times inside the case's window. A duplicated record cannot stand in for
a missing one.

Rates divide by an interval named by what it is. An open loop's rate counts
the requests scheduled in its window, however late they finished, per second
of the schedule's own window, which ends where the schedule does and not
where sending happened to end. A closed loop's rate counts every measured
request per second from the start of the phase to the end of its drain.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal

from .arrivals import (
    BURST,
    CLOSED,
    RATE_MODES,
    REPLAY,
    ArrivalSpec,
    arrival_offsets,
    scheduled_endpoint,
)
from .report_stats import int_value, is_number
from .slo import (
    CriterionCounts,
    RequestSloOutcome,
    SloEvaluation,
    SloSpec,
    evaluate_request,
)

if TYPE_CHECKING:
    from .vllm_analysis import JoinedSpans

OK: Final = "ok"
DROPPED: Final = "dropped"
UNREACHABLE: Final = "unreachable"
DELIVERY_UNKNOWN: Final = "delivery_unknown"
REJECTED: Final = "rejected"
_COUNTED = {
    "ok": "successful",
    "error": "failed",
    "timeout": "timed_out",
    "cancelled": "cancelled",
    DROPPED: DROPPED,
    UNREACHABLE: UNREACHABLE,
    DELIVERY_UNKNOWN: DELIVERY_UNKNOWN,
    REJECTED: REJECTED,
}
_NOT_ACCEPTED = frozenset({DROPPED, UNREACHABLE, DELIVERY_UNKNOWN, REJECTED})
# A request may end up to this long after the drain was recorded: the window
# record is written after the drain returns.
_END_SLACK_NS = 1_000_000_000
_SEGMENT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")

IntervalKind = Literal[
    "scheduled_window",
    "measured_span",
    "dispatch_window",
    "drain",
    "request_span",
    "segment",
]
NumeratorCohort = Literal["arrival_cohort", "all_measured", "overlapping"]
Membership = Literal["arrival", "overlap"]


@dataclass(frozen=True)
class Population:
    """A cohort's requests by what happened to them.

    ``offered = dropped + sent``; ``sent = unreachable + delivery_unknown +
    rejected + accepted``; ``accepted = successful + failed + timed_out +
    cancelled + sum(other)``. ``accepted`` is the client's view: nothing
    refused it. ``server_admitted`` counts requests the server confirmed it
    saw, through a joined span or execution record; it is None when the run
    has no such evidence at all.
    """

    offered: int
    scheduled: int | None
    dropped: int
    sent: int
    unreachable: int
    delivery_unknown: int
    rejected: int
    accepted: int
    successful: int
    failed: int
    timed_out: int
    cancelled: int
    other: Mapping[str, int] = field(default_factory=dict)
    server_admitted: int | None = None
    server_evidence_coverage: float | None = None
    cohort_valid: bool = True
    issues: tuple[str, ...] = ()

    @property
    def censored(self) -> int:
        """Requests whose latency is known only to exceed what was observed."""
        return self.timed_out + self.cancelled

    def to_record(self) -> dict[str, Any]:
        return {
            "offered": self.offered,
            "scheduled": self.scheduled,
            "dropped": self.dropped,
            "sent": self.sent,
            "unreachable": self.unreachable,
            "delivery_unknown": self.delivery_unknown,
            "rejected": self.rejected,
            "accepted": self.accepted,
            "successful": self.successful,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "censored": self.censored,
            "other": dict(sorted(self.other.items())),
            "server_admitted": self.server_admitted,
            "server_evidence_coverage": self.server_evidence_coverage,
            "cohort_valid": self.cohort_valid,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class MeasuredInterval:
    """A span of client wall time, named by kind and by what it counts."""

    kind: IntervalKind
    started_at_ns: int
    ended_at_ns: int
    numerator_cohort: NumeratorCohort

    @property
    def seconds(self) -> float | None:
        """Its length; None when it is empty, so nothing divides by zero."""
        span = self.ended_at_ns - self.started_at_ns
        return span / 1e9 if span > 0 else None

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "seconds": self.seconds,
            "numerator_cohort": self.numerator_cohort,
        }


@dataclass(frozen=True)
class CaseIntervals:
    """The interval rates divide by, and the others a reader may want.

    ``rate`` is the scheduled window for an open loop, the measured span for
    a closed loop, and the span of the recorded requests for an artifact
    older than phase windows. It is None for an open loop whose schedule has
    no known end, and for a phase that never recorded its window, or
    recorded one without bounds. ``rate_reason`` says why it is not the
    first choice when it is not.
    """

    rate: MeasuredInterval | None
    scheduled_window: MeasuredInterval | None = None
    dispatch_window: MeasuredInterval | None = None
    drain: MeasuredInterval | None = None
    measured_span: MeasuredInterval | None = None
    configured_rate_per_second: float | None = None
    realized_offered_rate_per_second: float | None = None
    rate_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "rate": _interval_record(self.rate),
            "scheduled_window": _interval_record(self.scheduled_window),
            "dispatch_window": _interval_record(self.dispatch_window),
            "drain": _interval_record(self.drain),
            "measured_span": _interval_record(self.measured_span),
            "configured_rate_per_second": self.configured_rate_per_second,
            "realized_offered_rate_per_second": self.realized_offered_rate_per_second,
            "rate_reason": self.rate_reason,
        }


@dataclass(frozen=True)
class Segment:
    """A slice of a case, by offsets from its measured phase's start."""

    name: str
    start_offset_ns: int
    end_offset_ns: int

    def __post_init__(self) -> None:
        if _SEGMENT_NAME.fullmatch(self.name) is None:
            raise ValueError(f"segment name {self.name!r} is not a short identifier")
        if not 0 <= self.start_offset_ns < self.end_offset_ns:
            raise ValueError(
                f"segment {self.name}: offsets must satisfy 0 <= start < end"
            )


@dataclass(frozen=True)
class SegmentPopulation:
    """The requests that belong to one segment, and the segment's interval."""

    population: Population
    interval: MeasuredInterval
    membership: Membership
    request_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CasePopulation:
    case_id: str
    population: Population
    intervals: CaseIntervals
    segments: Mapping[str, SegmentPopulation] = field(default_factory=dict)


def case_populations(
    records: Sequence[Mapping[str, Any]],
    *,
    segments: Sequence[Segment] = (),
    membership: Membership = "arrival",
    server_admitted_ids: Collection[str] | None = None,
) -> dict[str, CasePopulation]:
    """Each measured case's cohort, populations and intervals.

    ``server_admitted_ids`` are the ``x_request_id`` values the server
    confirmed it saw. Segment ``membership`` is ``arrival`` (by intended
    arrival, falling back to the send time) or ``overlap`` (any request whose
    span meets the segment).
    """
    _require_unique_names(segments)
    requests = _measured(records, "infer.request")
    windows = {
        str(w.get("case_id")): w for w in _measured(records, "infer.phase_window")
    }
    workload = _workload(records)
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for record in requests:
        by_case.setdefault(str(record.get("case_id", "unknown")), []).append(record)
    return {
        case_id: _case(
            case_id,
            case_requests,
            windows.get(case_id),
            workload,
            segments=segments,
            membership=membership,
            server_ids=server_admitted_ids,
        )
        for case_id, case_requests in sorted(by_case.items())
    }


def count_population(
    requests: Iterable[Mapping[str, Any]],
    *,
    scheduled: int | None = None,
    server_admitted_ids: Collection[str] | None = None,
    issues: Sequence[str] = (),
    cohort_valid: bool = True,
) -> Population:
    """Count requests by status; the identities in ``Population`` always hold."""
    requests = list(requests)
    statuses = Counter(str(record.get("status")) for record in requests)
    counted = Counter({name: 0 for name in set(_COUNTED.values())})
    other: dict[str, int] = {}
    for status, count in statuses.items():
        if status in _COUNTED:
            counted[_COUNTED[status]] += count
        else:
            other[status] = count
    offered = sum(statuses.values())
    sent = offered - counted[DROPPED]
    accepted = (
        sent - counted[UNREACHABLE] - counted[DELIVERY_UNKNOWN] - counted[REJECTED]
    )
    admitted, coverage = _server_admission(requests, accepted, server_admitted_ids)
    return Population(
        offered=offered,
        scheduled=scheduled,
        dropped=counted[DROPPED],
        sent=sent,
        unreachable=counted[UNREACHABLE],
        delivery_unknown=counted[DELIVERY_UNKNOWN],
        rejected=counted[REJECTED],
        accepted=accepted,
        successful=counted["successful"],
        failed=counted["failed"],
        timed_out=counted["timed_out"],
        cancelled=counted["cancelled"],
        other=other,
        server_admitted=admitted,
        server_evidence_coverage=coverage,
        cohort_valid=cohort_valid,
        issues=tuple(issues),
    )


def _case(
    case_id: str,
    requests: list[Mapping[str, Any]],
    window: Mapping[str, Any] | None,
    workload: Mapping[str, Any] | None,
    *,
    segments: Sequence[Segment],
    membership: Membership,
    server_ids: Collection[str] | None,
) -> CasePopulation:
    scheduled = _scheduled(window)
    if window is None:
        scheduled = _scheduled_from_workload(case_id, workload)
    hard, soft = _cohort_issues(requests, window, scheduled)
    hard.extend(_window_issues(window, workload))
    population = count_population(
        requests,
        scheduled=scheduled,
        server_admitted_ids=server_ids,
        issues=hard + soft,
        cohort_valid=not hard,
    )
    intervals = _intervals(case_id, requests, window, workload)
    return CasePopulation(
        case_id=case_id,
        population=population,
        intervals=intervals,
        segments={
            segment.name: _segment(segment, requests, intervals, membership, server_ids)
            for segment in segments
        },
    )


# ----------------------------------------------------------------- cohort


def _cohort_issues(
    requests: list[Mapping[str, Any]],
    window: Mapping[str, Any] | None,
    scheduled: int | None,
) -> tuple[list[str], list[str]]:
    """Problems that make the cohort untrustworthy, and ones worth a note."""
    hard = [
        *_duplicate_issues(requests, "request_id"),
        *_duplicate_issues(requests, "x_request_id"),
        *_index_issues(requests, scheduled),
        *_session_issues(requests, window),
        *_bound_issues(requests, window),
    ]
    soft = []
    if any(not isinstance(r.get("request_index"), int) for r in requests):
        soft.append("request_index_unrecorded")
    still_running = int_value(
        ((window or {}).get("abandoned_requests") or {}).get("still_running")
    )
    if still_running:
        soft.append(f"abandoned_requests_at_start: {still_running}")
    return hard, soft


def _window_issues(
    window: Mapping[str, Any] | None, workload: Mapping[str, Any] | None
) -> list[str]:
    """A phase that never recorded its end, or recorded it without bounds.

    A run that records its workload also records each measured phase's
    window once the phase drains, so a case without one was interrupted.
    Only an artifact older than both has neither.
    """
    if window is None:
        return [] if workload is None else ["phase_window_missing"]
    return [] if _window_bounds(window) is not None else ["phase_window_incomplete"]


def _duplicate_issues(requests: list[Mapping[str, Any]], key: str) -> list[str]:
    values = [r.get(key) for r in requests if r.get(key) is not None]
    repeated = sum(count - 1 for count in Counter(values).values() if count > 1)
    return [f"duplicate_{key}: {repeated}"] if repeated else []


def _index_issues(
    requests: list[Mapping[str, Any]], scheduled: int | None
) -> list[str]:
    """Every arrival exactly once: no index repeated, none missing."""
    indexes = [
        index
        for index in (r.get("request_index") for r in requests)
        if isinstance(index, int) and not isinstance(index, bool)
    ]
    if len(indexes) != len(requests):
        # Artifacts before request indexes, noted as a soft issue; a known
        # schedule still needs one record per arrival.
        if scheduled is None or scheduled == len(requests):
            return []
        return [f"offered_differs_from_scheduled: {len(requests)} of {scheduled}"]
    return _coverage_issues(
        Counter(indexes), scheduled if scheduled is not None else len(requests)
    )


def _coverage_issues(counts: Counter[int], expected: int) -> list[str]:
    issues = []
    if any(count > 1 for count in counts.values()):
        issues.append("request_index_repeated")
    missing = expected - len(set(counts) & set(range(expected)))
    if missing:
        issues.append(f"records_missing: {missing} of {expected}")
    if any(index < 0 or index >= expected for index in counts):
        issues.append("request_index_out_of_range")
    return issues


def _session_issues(
    requests: list[Mapping[str, Any]], window: Mapping[str, Any] | None
) -> list[str]:
    sessions = {r.get("session_id") for r in requests}
    if window is not None:
        sessions.add(window.get("session_id"))
    return ["session_mismatch"] if len(sessions) > 1 else []


def _bound_issues(
    requests: list[Mapping[str, Any]], window: Mapping[str, Any] | None
) -> list[str]:
    start = window.get("started_at_ns") if window is not None else None
    drained = window.get("drained_at_ns") if window is not None else None
    outside = sum(1 for record in requests if _outside(record, start, drained))
    return [f"outside_window: {outside}"] if outside else []


def _outside(record: Mapping[str, Any], start: Any, drained: Any) -> bool:
    """A request whose times are missing, reversed, or outside its phase."""
    began, ended = record.get("started_at_ns"), record.get("ended_at_ns")
    if not is_number(began) or not is_number(ended) or began > ended:
        return True
    if is_number(start) and began < start:
        return True
    return is_number(drained) and ended > drained + _END_SLACK_NS


# ----------------------------------------------------------------- intervals


def _intervals(
    case_id: str,
    requests: list[Mapping[str, Any]],
    window: Mapping[str, Any] | None,
    workload: Mapping[str, Any] | None,
) -> CaseIntervals:
    if window is None and workload is not None:
        return CaseIntervals(rate=None, rate_reason="phase_window_missing")
    if window is None:
        span = _request_span(requests)
        return CaseIntervals(rate=span, rate_reason="no_phase_window")
    if _window_bounds(window) is None:
        return CaseIntervals(rate=None, rate_reason="phase_window_incomplete")
    base = _phase_intervals(requests, window)
    if str(window.get("arrival_mode", CLOSED)) == CLOSED:
        return base
    endpoint = _endpoint_ns(case_id, window, workload)
    configured = _configured_rate(case_id, workload)
    if endpoint is None:
        # Dividing by the span to the drain's end would make an open loop's
        # rate depend on when its requests finished.
        return replace(
            base,
            rate=None,
            configured_rate_per_second=configured,
            rate_reason="endpoint_undeclared",
        )
    started = int_value(window.get("started_at_ns"))
    scheduled = MeasuredInterval(
        "scheduled_window", started, started + endpoint, "arrival_cohort"
    )
    seconds = scheduled.seconds
    offered = _scheduled(window) or len(requests)
    return replace(
        base,
        rate=scheduled,
        scheduled_window=scheduled,
        configured_rate_per_second=configured,
        realized_offered_rate_per_second=offered / seconds if seconds else None,
    )


def _phase_intervals(
    requests: list[Mapping[str, Any]], window: Mapping[str, Any]
) -> CaseIntervals:
    """A phase's span, dispatch and drain, with the span as the rate interval."""
    started = int_value(window.get("started_at_ns"))
    drained = int_value(window.get("drained_at_ns"))
    measured = MeasuredInterval("measured_span", started, drained, "all_measured")
    return CaseIntervals(
        rate=measured,
        dispatch_window=_dispatch_window(requests),
        drain=MeasuredInterval(
            "drain",
            int_value(window.get("window_ended_at_ns")),
            drained,
            "all_measured",
        ),
        measured_span=measured,
    )


def _window_bounds(window: Mapping[str, Any]) -> tuple[int, int] | None:
    started, drained = window.get("started_at_ns"), window.get("drained_at_ns")
    if not is_number(started) or not is_number(drained):
        return None
    return int_value(started), int_value(drained)


def _scheduled_from_workload(
    case_id: str, workload: Mapping[str, Any] | None
) -> int | None:
    """An open loop's scheduled arrivals, recomputed from its seeded spec."""
    case, measurement = _workload_case(case_id, workload)
    spec = _arrival_spec((case or {}).get("arrival"))
    if spec is None or not spec.open_loop or workload is None or measurement is None:
        return None
    try:
        offsets = arrival_offsets(
            spec,
            count=measurement.get("request_count"),
            duration_seconds=measurement.get("duration_seconds"),
            seed=int_value(workload.get("seed")),
        )
    except (TypeError, ValueError):
        return None
    return len(offsets)


def _endpoint_ns(
    case_id: str, window: Mapping[str, Any], workload: Mapping[str, Any] | None
) -> int | None:
    """The recorded endpoint, or one recomputed from the seeded workload spec."""
    recorded = window.get("scheduled_endpoint_offset_ns")
    if is_number(recorded):
        return int_value(recorded)
    case, measurement = _workload_case(case_id, workload)
    spec = _arrival_spec((case or {}).get("arrival"))
    if spec is None or workload is None or measurement is None:
        return None
    endpoint = scheduled_endpoint(
        spec,
        count=measurement.get("request_count"),
        duration_seconds=measurement.get("duration_seconds"),
        seed=int_value(workload.get("seed")),
    )
    return None if endpoint is None else round(endpoint * 1e9)


def _configured_rate(case_id: str, workload: Mapping[str, Any] | None) -> float | None:
    arrival = (_workload_case(case_id, workload)[0] or {}).get("arrival") or {}
    if arrival.get("mode") in RATE_MODES and is_number(arrival.get("rate_per_second")):
        return float(arrival["rate_per_second"])
    if arrival.get("mode") == BURST:
        size, interval = arrival.get("burst_size"), arrival.get(
            "burst_interval_seconds"
        )
        if is_number(size) and is_number(interval) and interval > 0:
            return float(size) / float(interval)
    return None


def _arrival_spec(arrival: Any) -> ArrivalSpec | None:
    """The arrival shape of a recorded case; a replay's offsets are not recorded."""
    if not isinstance(arrival, Mapping) or arrival.get("mode") in (None, REPLAY):
        return None
    try:
        return ArrivalSpec(
            mode=str(arrival.get("mode")),
            rate_per_second=arrival.get("rate_per_second"),
            burst_size=arrival.get("burst_size"),
            burst_interval_seconds=arrival.get("burst_interval_seconds"),
        )
    except (TypeError, ValueError):
        return None


def _workload_case(
    case_id: str, workload: Mapping[str, Any] | None
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if workload is None:
        return None, None
    case = next(
        (c for c in workload.get("cases") or [] if c.get("case_id") == case_id), None
    )
    measurement = workload.get("measurement")
    return case, measurement if isinstance(measurement, Mapping) else None


def _dispatch_window(requests: list[Mapping[str, Any]]) -> MeasuredInterval | None:
    sends = [
        int_value(r.get("started_at_ns"))
        for r in requests
        if r.get("status") != DROPPED and is_number(r.get("started_at_ns"))
    ]
    if not sends:
        return None
    return MeasuredInterval("dispatch_window", min(sends), max(sends), "all_measured")


def _request_span(requests: list[Mapping[str, Any]]) -> MeasuredInterval | None:
    """First start to last end over every measured request, successful or not.

    A request missing either bound is left out rather than paired with
    another request's.
    """
    bounds = [
        (int_value(r["started_at_ns"]), int_value(r["ended_at_ns"]))
        for r in requests
        if is_number(r.get("started_at_ns")) and is_number(r.get("ended_at_ns"))
    ]
    if not bounds:
        return None
    start = min(began for began, _ended in bounds)
    end = max(ended for _began, ended in bounds)
    return MeasuredInterval("request_span", start, end, "all_measured")


# ----------------------------------------------------------------- segments


def _segment(
    segment: Segment,
    requests: list[Mapping[str, Any]],
    intervals: CaseIntervals,
    membership: Membership,
    server_ids: Collection[str] | None,
) -> SegmentPopulation:
    case_span = intervals.measured_span or intervals.rate
    origin = case_span.started_at_ns if case_span is not None else 0
    begin, end = origin + segment.start_offset_ns, origin + segment.end_offset_ns
    if case_span is not None:
        begin = max(begin, case_span.started_at_ns)
        end = min(end, max(case_span.ended_at_ns, begin))
    members = [r for r in requests if _belongs(r, begin, end, membership)]
    cohort: NumeratorCohort = (
        "arrival_cohort" if membership == "arrival" else "overlapping"
    )
    return SegmentPopulation(
        population=count_population(members, server_admitted_ids=server_ids),
        interval=MeasuredInterval("segment", begin, end, cohort),
        membership=membership,
        request_ids=tuple(str(r.get("request_id")) for r in members),
    )


def _belongs(
    record: Mapping[str, Any], begin: int, end: int, membership: Membership
) -> bool:
    if membership == "arrival":
        arrived = record.get("intended_at_ns")
        if not is_number(arrived):
            arrived = record.get("started_at_ns")
        return is_number(arrived) and begin <= arrived < end
    started, ended = record.get("started_at_ns"), record.get("ended_at_ns")
    return is_number(started) and is_number(ended) and started < end and ended >= begin


# ----------------------------------------------------------------- helpers


def _server_admission(
    requests: Iterable[Mapping[str, Any]],
    accepted: int,
    server_ids: Collection[str] | None,
) -> tuple[int | None, float | None]:
    """Requests the server confirmed, and the share of accepted ones it did.

    A request whose delivery the client could not confirm counts as
    admitted when the server saw it, but it is not among the accepted, so
    the coverage counts only accepted requests.
    """
    if server_ids is None:
        return None, None
    seen = [
        r
        for r in requests
        if r.get("status") != DROPPED and r.get("x_request_id") in server_ids
    ]
    seen_accepted = sum(1 for r in seen if r.get("status") not in _NOT_ACCEPTED)
    return len(seen), (seen_accepted / accepted if accepted else None)


def _scheduled(window: Mapping[str, Any] | None) -> int | None:
    value = (window or {}).get("scheduled_arrivals")
    return int_value(value) if is_number(value) else None


def _measured(
    records: Sequence[Mapping[str, Any]], event_type: str
) -> list[Mapping[str, Any]]:
    return [
        r
        for r in records
        if r.get("event_type") == event_type and r.get("phase") == "measured"
    ]


def _workload(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((r for r in records if r.get("event_type") == "infer.workload"), None)


def _interval_record(interval: MeasuredInterval | None) -> dict[str, Any] | None:
    return None if interval is None else interval.to_record()


def _require_unique_names(segments: Sequence[Segment]) -> None:
    names = [segment.name for segment in segments]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"segment {', '.join(repeated)} is defined more than once")


def goodput(
    requests: Sequence[Mapping[str, Any]],
    spec: SloSpec,
    interval: MeasuredInterval | None,
    *,
    spans: JoinedSpans | None = None,
    slo_source: str = "flags",
    cohort: Population | None = None,
) -> SloEvaluation:
    """Judge a case's offered requests against ``spec``; rates per ``interval``.

    ``requests`` are the case's measured requests, dropped ones included;
    ``spans`` are their joined vLLM spans, for server criteria; ``cohort``
    is their population, whose validity the evaluation carries.
    """
    outcomes = [_judged(record, spec, spans) for record in requests]
    met = sum(1 for outcome in outcomes if outcome.outcome == "met")
    unknown = sum(1 for outcome in outcomes if outcome.outcome == "unknown")
    good_tokens = sum(
        int_value(record.get("output_tokens"))
        for record, outcome in zip(requests, outcomes)
        if outcome.outcome == "met"
    )
    reason = _unmeasurable(spec, outcomes)
    figures = _figures(met, unknown, len(outcomes), good_tokens, interval)
    valid, issues = _cohort_fields(cohort)
    return SloEvaluation(
        slo_name=spec.name,
        slo_digest=spec.digest(),
        slo_source=slo_source,
        status="evaluated" if reason is None else "unmeasurable",
        reason=reason,
        population_declared=spec.population,
        population_evaluated="offered",
        offered=len(outcomes),
        met=met,
        missed=len(outcomes) - met - unknown,
        unknown=unknown,
        evidence_coverage=_evidence_coverage(outcomes),
        per_criterion=_per_criterion(spec, outcomes, len(outcomes)),
        interval=interval,
        **(figures if reason is None else dict.fromkeys(figures)),
        cohort_valid=valid,
        cohort_issues=issues,
    )


def _cohort_fields(cohort: Population | None) -> tuple[bool | None, tuple[str, ...]]:
    return (None, ()) if cohort is None else (cohort.cohort_valid, cohort.issues)


def _figures(
    met: int,
    unknown: int,
    offered: int,
    good_tokens: int,
    interval: MeasuredInterval | None,
) -> dict[str, float | None]:
    """Attainment and goodput bounds: unknown outcomes missed, then met."""
    return {
        "attainment_lower": _share(met, offered),
        "attainment_upper": _share(met + unknown, offered),
        "goodput_lower_rps": rate(met, interval),
        "goodput_upper_rps": rate(met + unknown, interval),
        "goodput_lower_output_tps": rate(good_tokens, interval),
    }


def _judged(
    record: Mapping[str, Any], spec: SloSpec, spans: JoinedSpans | None
) -> RequestSloOutcome:
    request_id = str(record.get("x_request_id"))
    if spans is None:
        return evaluate_request(record, spec)
    return evaluate_request(
        record,
        spec,
        span=spans.by_request.get(request_id),
        missing_span_reason=spans.quarantined.get(request_id, "no_joined_span"),
    )


def _unmeasurable(spec: SloSpec, outcomes: Sequence[RequestSloOutcome]) -> str | None:
    """Why the policy cannot be judged per request here, or None.

    An aggregate-only criterion never can. Nor can a criterion that no
    successful request could be judged on, such as client TTFT without
    streaming or a server criterion without spans. Requests that did not
    succeed are missed whatever their criteria say, so a case where none
    succeeded is still measurable: its goodput is zero.
    """
    successful = [outcome for outcome in outcomes if outcome.status == OK]
    for criterion in spec.criteria:
        if criterion.definition.per_request is None:
            return f"{criterion.key}: aggregate_only"
        judged = [outcome.criteria[criterion.key] for outcome in successful]
        if judged and all(item.outcome == "unknown" for item in judged):
            reasons = Counter(item.reason for item in judged)
            return f"{criterion.key}: {reasons.most_common(1)[0][0]}"
    return None


def _evidence_coverage(outcomes: Sequence[RequestSloOutcome]) -> float | None:
    successful = [outcome for outcome in outcomes if outcome.status == OK]
    complete = sum(
        1
        for outcome in successful
        if all(item.outcome != "unknown" for item in outcome.criteria.values())
    )
    return _share(complete, len(successful))


def _per_criterion(
    spec: SloSpec, outcomes: Sequence[RequestSloOutcome], offered: int
) -> dict[str, CriterionCounts]:
    """Each criterion's outcomes among successful requests, and its bounds.

    A criterion no successful request could be judged on has no bounds,
    as the evaluation has none: ``[0, 1]`` would read as a figure.
    """
    counts = {}
    for criterion in spec.criteria:
        judged = Counter(
            outcome.criteria[criterion.key].outcome
            for outcome in outcomes
            if outcome.status == OK
        )
        good = judged["pass"] + judged["not_applicable"]
        unjudgeable = judged["unknown"] > 0 and judged["unknown"] == judged.total()
        bounds = (
            (None, None)
            if criterion.definition.per_request is None or unjudgeable
            else (_share(good, offered), _share(good + judged["unknown"], offered))
        )
        counts[criterion.key] = CriterionCounts(
            passed=judged["pass"],
            failed=judged["fail"],
            not_applicable=judged["not_applicable"],
            unknown=judged["unknown"],
            attainment_lower=bounds[0],
            attainment_upper=bounds[1],
        )
    return counts


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def rate(count: float, interval: MeasuredInterval | None) -> float | None:
    """``count`` per second of ``interval``; None when it has no length."""
    seconds = interval.seconds if interval is not None else None
    if seconds is None or not math.isfinite(seconds):
        return None
    return count / seconds


__all__ = [
    "CaseIntervals",
    "CasePopulation",
    "MeasuredInterval",
    "Population",
    "Segment",
    "SegmentPopulation",
    "case_populations",
    "count_population",
    "goodput",
    "rate",
]
