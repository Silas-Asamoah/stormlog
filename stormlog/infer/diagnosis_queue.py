"""Queue saturation: requests waited to be scheduled because the engine was
full.

The class compares the subject's ``scheduler_wait`` with its reference's
(the difference of medians, with a bootstrap interval), and asks whether
most of the subject's requests waited through an engine at capacity: most of
the steps scheduled while each one waited running ``max_num_seqs`` (counting
the slots freed in the step before, which async scheduling refills one step
late) or scheduling ``max_num_batched_tokens``. Four competitors
must be ruled out before the waits may be called a fault: a stalled engine,
a paused scheduler, requests blocked by their own constraints, and time
spent before the queue rather than in it. Others are reported: preemptions
upstream, the client holding requests back, the API server.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Callable

from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import Line
from .diagnosis_join import Execution
from .diagnosis_kv import held_admissions
from .diagnosis_loop import LoopGapConfig, stalls_over_limit
from .diagnosis_metrics import aggregate_assessment, subject_signal
from .diagnosis_model import (
    CONTRIBUTING,
    NOT_RULED_OUT,
    RULED_OUT,
    UNTESTABLE,
    UPSTREAM,
    Alternative,
    Criteria,
    Finding,
    Observation,
    met,
)
from .diagnosis_segments import MERGED_INGRESS
from .diagnosis_selection import Subject
from .diagnosis_stats import INSUFFICIENT_SAMPLES, Difference, median_difference
from .diagnosis_steps import Step, Steps, loop_steps, merge_intervals
from .diagnosis_thresholds import (
    QUEUE_COMPETITOR_FLOOR,
    QUEUE_CONTRIBUTION,
    QUEUE_FRONT_SHARE,
    QUEUE_KV_SHARE,
    QUEUE_STALL_SHARE,
    QUEUE_WITNESS_SHARE,
    resolve_threshold,
)
from .diagnosis_vocabulary import COMPONENT_SCHEDULER, QUEUE_SATURATION
from .scrape_window import gauge_median, gauge_window

NO_SERVER_QUEUE_SIGNAL = "no_server_queue_signal"
NO_CAPACITY_WITNESS = "no_capacity_witness"
NO_CLIENT_LATENCY = "no_client_latency"
SEVERAL_ENGINES = "several_engines"
WAIT = "scheduler_wait"
WAITING_BY_REASON = "vllm:num_requests_waiting_by_reason"


@dataclass(frozen=True)
class _Waits:
    """Each execution's wait to be scheduled, as one segment family."""

    segment: str  # scheduler_wait, or the legacy merged segment
    subject: dict[str, float]
    reference: dict[str, float]
    executions: dict[str, Execution]  # the subject's, by key

    @property
    def producer(self) -> str | None:
        """The one engine that ran the subject; None for several."""
        producers = {execution.producer for execution in self.executions.values()}
        return next(iter(producers)) if len(producers) == 1 else None


def assess_queue(context: Context, subject: Subject) -> Assessment:
    """The queue class on one subject."""
    waits = _waits(context, subject)
    producer = waits.producer if waits else None
    if waits is None or not waits.subject:
        # Without the hook, vLLM's metrics can only describe the window.
        aggregate = aggregate_assessment(
            context,
            subject,
            QUEUE_SATURATION,
            "A median of {value:.1f} requests waited in vLLM's queue over the window.",
            NO_SERVER_QUEUE_SIGNAL,
        )
        return aggregate or Assessment(
            QUEUE_SATURATION, subject.key, UNSUPPORTED, [NO_SERVER_QUEUE_SIGNAL]
        )
    if producer is None:
        return Assessment(QUEUE_SATURATION, subject.key, UNSUPPORTED, [SEVERAL_ENGINES])
    excess = median_difference(
        list(waits.subject.values()), list(waits.reference.values())
    )
    if excess is None:
        return Assessment(
            QUEUE_SATURATION, subject.key, PARTIAL, [INSUFFICIENT_SAMPLES]
        )
    if excess.low <= 0:
        return Assessment(QUEUE_SATURATION, subject.key, ASSESSED, ["not_observed"])
    return _finding(context, subject, waits, excess, producer)


def _waits(context: Context, subject: Subject) -> _Waits | None:
    """The waits of the subject's and the reference's executions, on the
    engine's clock; on a log without enqueued records, the merged ingress
    segment, labelled."""
    found: dict[str, dict[str, dict[str, float]]] = {}
    executions: dict[str, Execution] = {}
    for arm, reference in (("subject", False), ("reference", True)):
        for execution in context.subject_executions(subject, reference=reference):
            segments = context.engine_segments(execution)
            key = _key(execution)
            for name in (WAIT, MERGED_INGRESS):
                if name in segments:
                    found.setdefault(name, {}).setdefault(arm, {})[key] = float(
                        segments[name]
                    )
            if not reference:
                executions[key] = execution
    if not found:
        return None
    segment = WAIT if WAIT in found else MERGED_INGRESS
    arms = found[segment]
    return _Waits(
        segment, arms.get("subject", {}), arms.get("reference", {}), executions
    )


def _key(execution: Execution) -> str:
    return (execution.attempt or execution.event.request_ref).id


def _finding(
    context: Context, subject: Subject, waits: _Waits, excess: Difference, producer: str
) -> Assessment:
    steps = context.steps(producer)
    waiting = list(waits.executions.values())
    span = _wait_span(context, waiting)
    # Steps run while a subject's request waited; the span from the first
    # wait to the last can hold stretches in which nobody did.
    intervals = _wait_intervals(context, waiting)
    spanning = steps.within(intervals)
    witness = _witness(context, producer, steps, intervals)
    if witness.requests_share is None:
        # Only where the hook could not measure: engine-global metrics
        # never overrule the steps the requests waited through.
        witness = _exporter_witness(context, subject, witness)
    ttft = _ttft_excess(context, subject)
    kv_hold, held = _kv_hold(context, subject, producer, waiting, excess)
    alternatives = [
        _engine_stall(context, producer, waiting, excess),
        _scheduler_paused(
            context,
            producer,
            steps,
            merge_intervals(_wait_intervals(context, waiting)),
            excess,
        ),
        _blocked_waiting(waiting),
        _engine_ingress(context, subject, waits, excess),
        kv_hold,
        _client_admission(context, subject),
        _api_server(context, subject, excess),
    ]
    finding = Finding(
        kind=QUEUE_SATURATION,
        component=COMPONENT_SCHEDULER,
        subject=subject.as_dict(),
        title="Requests waited to be scheduled while the engine was full",
        message=_message(waits, excess, witness),
        gates={
            "capacity_witness": witness.held,
            "usable_timing": waits.segment == WAIT,
        },
        alternatives=alternatives,
        observations=_observations(waits, excess, ttft, witness),
        location={
            "component": COMPONENT_SCHEDULER,
            "engine_producer": producer,
            **_iterations(spanning),
        },
        window=context.window(subject),
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        contribution_lower=excess.low / 1e6,
        metrics={
            "scheduler_wait_excess_ms": round(excess.estimate / 1e6, 3),
            "scheduler_wait_excess_low_ms": round(excess.low / 1e6, 3),
            "ttft_excess_ms": None if ttft is None else round(ttft.estimate / 1e6, 3),
            **witness.metrics(),
        },
        experiment=_experiment(excess),
        explains="explains_ttft_excess",
        detail={
            "capacity_witness_source": witness.source,
            "kv_hold": _kv_claim(context, held, ttft),
        },
    )
    finding.condition, finding.contribution = _criteria(
        context, producer, span, waits, excess, ttft, witness, alternatives
    )
    finding.segments = {
        "decomposition": "ttft" if subject.basis == "client" else "engine_ttft",
        "wait_segment": waits.segment,
    }
    finding.support, finding.display = _support(context, subject, spanning)
    reasons = [] if witness.held else [NO_CAPACITY_WITNESS]
    if subject.basis == "engine":
        reasons.append(NO_CLIENT_LATENCY)
    status = PARTIAL if reasons else ASSESSED
    finding.status = status
    finding.partial_reasons = reasons
    return Assessment(QUEUE_SATURATION, subject.key, status, reasons, [finding])


# ----------------------------------------------------------------- witness
@dataclass(frozen=True)
class _Witness:
    """Whether the engine was full while the subject's requests waited, and
    what said so."""

    held: bool
    source: str | None  # "hook" or "exporter": the evidence that held
    requests_share: float | None  # requests whose wait ran mostly at capacity
    steps_share: float | None  # steps run while requests waited, at capacity
    max_num_seqs_share: float | None  # the same, by their members alone
    steps: int  # busy steps run while requests waited

    def metrics(self) -> dict[str, float | None]:
        return {
            "requests_waiting_at_capacity_share": self.requests_share,
            "steps_at_capacity_share": self.steps_share,
            "steps_at_max_num_seqs_share": self.max_num_seqs_share,
        }


def _witness(
    context: Context, producer: str, steps: Steps, waits: list[tuple[int, int]]
) -> _Witness:
    """Whether most of the subject's requests waited mostly through steps at
    capacity. The excess is a median over the requests, so the witness must
    speak for the median request: a few overflow requests waiting through
    full steps do not, while most waited only for the next step to begin."""
    seqs, tokens = _capacity(context, producer)
    busy = [step for step in steps.within(waits) if step.members]
    if not waits or (seqs is None and tokens is None):
        return _Witness(False, None, None, None, None, len(busy))
    # A request that waited through no step waited only for the next one to
    # begin: measured, and not held by capacity.
    mostly = [_mostly_full(steps.within([wait]), seqs, tokens) for wait in waits]
    share = sum(mostly) / len(mostly)
    needed = resolve_threshold(QUEUE_WITNESS_SHARE, context.thresholds)[0]
    held = share >= needed
    return _Witness(
        held,
        "hook" if held else None,
        round(share, 4),
        _share(busy, lambda step: _at_capacity(step, seqs, tokens)),
        None if seqs is None else _share(busy, lambda step: step.members >= seqs),
        len(busy),
    )


def _share(steps: list[Step], holds: Callable[[Step], bool]) -> float | None:
    if not steps:
        return None
    return round(sum(holds(step) for step in steps) / len(steps), 4)


def _mostly_full(waited: list[Step], seqs: int | None, tokens: int | None) -> bool:
    """Whether at least half the busy steps run while one request waited
    were at capacity; a request that waited through none waited only for
    the next step to begin."""
    busy = [step for step in waited if step.members]
    full = sum(_at_capacity(step, seqs, tokens) for step in busy)
    return bool(busy) and 2 * full >= len(busy)


def _exporter_witness(context: Context, subject: Subject, hook: _Witness) -> _Witness:
    """Requests waiting for scheduling capacity, as vLLM's own metric
    counts them (its help text: "waiting for scheduling capacity", not only
    KV): a witness only from an exporter asserted to be this engine."""
    if not context.metrics_from_engine:
        return hook
    signal = subject_signal(context, subject, QUEUE_SATURATION)
    if signal is None or not signal.sufficient:
        # A window the signal cannot decide on (across a restart, out of
        # order, too few scrapes) witnesses nothing either.
        return hook
    capacity = _capacity_waiting(context, subject)
    if capacity is not None and capacity > 0:
        return replace(hook, held=True, source="exporter")
    return hook


def _capacity_waiting(context: Context, subject: Subject) -> float | None:
    """The median of vLLM's capacity-waiting gauge over the subject's
    window; None when the window is not one series of one exporter."""
    scrapes = context.scrapes(subject.start_ns, subject.end_ns)
    window = gauge_window(scrapes, WAITING_BY_REASON, labels={"reason": "capacity"})
    return None if window.reasons else gauge_median(window)


def _capacity(context: Context, producer: str) -> tuple[int | None, int | None]:
    """The engine's max_num_seqs and max_num_batched_tokens, from its hello."""
    epoch = context.epoch_of(producer)
    config = epoch.config if epoch is not None else {}
    seqs, tokens = config.get("max_num_seqs"), config.get("max_num_batched_tokens")
    return (
        seqs if isinstance(seqs, int) else None,
        tokens if isinstance(tokens, int) else None,
    )


def _at_capacity(step: Step, seqs: int | None, tokens: int | None) -> bool:
    """Running max_num_seqs, counting the slots the step before freed and
    this one could not refill yet, or scheduling the token budget."""
    return (seqs is not None and step.members + step.refill >= seqs) or (
        tokens is not None and step.total_tokens >= tokens
    )


# ------------------------------------------------------------- competitors
def _engine_stall(
    context: Context, producer: str, waiting: list[Execution], excess: Difference
) -> Alternative:
    """Engine-loop stalls, by the online trigger's own rules, against the
    wait excess, request by request. While the queue stays busy a stall
    postpones every later admission by its length, so a request is held by
    the stalls since its queue was last empty, up to its own admission, and
    by one that ended at most its own length before then (a request reaching
    the engine during a stall enters the queue only when the loop resumes),
    by no more than its own wait. The excess is a median over the requests,
    so it is set against the median request's held time: stalls spread over
    a long saturation hold each request by those before it, not by all."""
    intervals = _wait_intervals(context, waiting)
    if not intervals:
        return Alternative("engine_stall", UNTESTABLE, "no wait was placed", True)
    config = LoopGapConfig(thresholds=dict(context.thresholds or {}))
    stalls = [
        (stall.start_mono_ns, stall.start_mono_ns + stall.duration_ns)
        for stall, _ in stalls_over_limit(
            loop_steps(context.view, producer), (), config
        )
    ]
    busy = _busy_periods(context, producer)
    held = [
        _held_by_stalls(stalls, wait, _busy_since(busy, wait)) for wait in intervals
    ]
    typical = median(held)
    status = _by_share(context, typical / excess.estimate, QUEUE_STALL_SHARE)
    reason = (
        f"{len(stalls)} engine stalls; the median waiting request was held "
        f"{typical / 1e6:.1f} ms by those since its queue was last empty, "
        f"against a wait excess of {excess.estimate / 1e6:.1f} ms"
    )
    return Alternative("engine_stall", status, reason, True)


def _busy_periods(context: Context, producer: str) -> list[tuple[int, int]]:
    """When anyone at all waited in the engine's queue, the subject's
    requests or not."""
    executions = [e for e in context.view.executions.values() if e.producer == producer]
    return merge_intervals(_wait_intervals(context, executions))


def _busy_since(busy: list[tuple[int, int]], wait: tuple[int, int]) -> int:
    """When the queue was last empty before a wait began."""
    index = bisect_right([start for start, _ in busy], wait[0]) - 1
    return min(busy[index][0], wait[0]) if index >= 0 else wait[0]


def _held_by_stalls(
    stalls: list[tuple[int, int]], wait: tuple[int, int], since: int
) -> int:
    """How long stalls held one wait back: their part from when its queue
    was last empty to its admission, and the whole of one that ended at most
    its own length before then; never more than the wait itself."""
    held = 0
    for start, end in stalls:
        inside = _overlap((start, end), (since, wait[1]))
        before = end <= since and _holds_back((start, end), (since, wait[1]))
        held += inside or (end - start if before else 0)
    return min(held, wait[1] - wait[0])


def _holds_back(stall: tuple[int, int], wait: tuple[int, int]) -> bool:
    """Whether the stall overlaps the wait, or ended at most its own length
    before the wait began."""
    start, end = stall
    return start < wait[1] and wait[0] - (end - start) <= end


def _wait_intervals(
    context: Context, executions: list[Execution]
) -> list[tuple[int, int]]:
    """Each execution's wait on the engine clock: queue entry (or admission)
    to its first schedule() call."""
    intervals = []
    for execution in executions:
        enqueued = execution.metadata.get("enqueued_mono_ns")
        start = enqueued if isinstance(enqueued, int) else execution.event.start_ns
        first = execution.memberships[0][1] if execution.memberships else None
        step = context.view.iterations.get(first.iteration_ref) if first else None
        end = step[1].start_ns if step is not None else None
        if start is not None and end is not None and end > start:
            intervals.append((start, end))
    return intervals


def _scheduler_paused(
    context: Context,
    producer: str,
    steps: Steps,
    waits: list[tuple[int, int]],
    excess: Difference,
) -> Alternative:
    """Ruled out by the hook's pause records where nothing was lost, else by
    admissions: a paused scheduler admits nobody, so the longest stretch
    without an admission while requests waited is the longest pause that
    could hide there, and it must be a minor share of the wait excess. Each
    is judged over the stretches in which a subject's request waited, not
    between them."""
    kind = "scheduler_paused"
    if not waits:
        return Alternative(kind, UNTESTABLE, "no wait was placed", True)
    paused = _pause_over(steps, waits)
    if paused is not None:
        return Alternative(kind, NOT_RULED_OUT, f"{paused} overlaps the waits", True)
    if _pauses_recorded_whole(context.epoch_of(producer), waits):
        return Alternative(
            kind, RULED_OUT, "no pause transition, and nothing lost", True
        )
    longest = max(_admission_gap(steps, wait) for wait in waits)
    floor = resolve_threshold(QUEUE_COMPETITOR_FLOOR, context.thresholds)[0]
    records = (
        "records may have been lost over the waits"
        if context.observes(producer, "pause")
        else "no pause records"
    )
    reason = (
        f"{records}; admissions stopped for up to {longest / 1e6:.1f} ms "
        f"while requests waited, against a wait excess of {excess.estimate / 1e6:.1f} ms"
    )
    if longest < floor * excess.estimate:
        return Alternative(kind, RULED_OUT, reason, True)
    # Without pause records, a full engine admits nobody either.
    return Alternative(kind, UNTESTABLE, reason, True)


def _pause_over(steps: Steps, waits: list[tuple[int, int]]) -> str | None:
    """The first pause that overlaps a wait, as its state."""
    for start, end, state in steps.paused_intervals(waits[-1][1]):
        if any(_overlap((start, end), wait) for wait in waits):
            return str(state)
    return None


def _pauses_recorded_whole(epoch: Any, waits: list[tuple[int, int]]) -> bool:
    """The hook records pauses, and lost nothing over any wait."""
    observes = epoch.observes if epoch is not None else None
    if observes is None or "pause" not in observes:
        return False
    return all(_covered(epoch, wait) for wait in waits)


def _admission_gap(steps: Steps, span: tuple[int, int]) -> int:
    """The longest stretch of a wait without an admission. The wait ends
    at the call that admitted its request, so a pause must begin and end
    inside it."""
    admissions = [step.start_ns for step in steps.between(*span) if step.admitted]
    edges = [span[0], *admissions, span[1]]
    return max(b - a for a, b in zip(edges, edges[1:]))


def _blocked_waiting(executions: list[Execution]) -> Alternative:
    """Requests held by their own constraints: grammar or streaming input."""
    kind = "blocked_waiting"
    flags = [
        (e.metadata.get("structured_output"), e.metadata.get("resumable"))
        for e in executions
    ]
    if flags and all(s is False and r is False for s, r in flags):
        reason = "no request used structured output or streaming input"
        return Alternative(kind, RULED_OUT, reason, True)
    if any(s is True or r is True for s, r in flags):
        reason = "some waiting requests were constrained"
        return Alternative(kind, NOT_RULED_OUT, reason, True)
    reason = "the hook did not record the requests' constraints"
    return Alternative(kind, UNTESTABLE, reason, True)


def _engine_ingress(
    context: Context, subject: Subject, waits: _Waits, excess: Difference
) -> Alternative:
    """The excess lies before the queue, not in it."""
    if waits.segment != WAIT:
        return Alternative(
            "engine_ingress",
            UNTESTABLE,
            "ingress and the queue wait are one segment on this log",
            True,
        )
    ingress = _engine_excess(context, subject, "engine_ingress")
    if ingress is None:
        return Alternative(
            "engine_ingress", UNTESTABLE, "too few ingress samples", True
        )
    status = _by_share(context, ingress.estimate / excess.estimate, QUEUE_FRONT_SHARE)
    reason = f"engine_ingress excess {ingress.estimate / 1e6:.1f} ms"
    return Alternative("engine_ingress", status, reason, True)


def _by_share(context: Context, share: float, cut: str) -> str:
    """A competitor by the share of the wait excess it explains itself:
    ruled out below the floor, contributing up to its cut, else not ruled
    out."""
    floor = resolve_threshold(QUEUE_COMPETITOR_FLOOR, context.thresholds)[0]
    if share < floor:
        return RULED_OUT
    if share < resolve_threshold(cut, context.thresholds)[0]:
        return CONTRIBUTING
    return NOT_RULED_OUT


def _kv_hold(
    context: Context,
    subject: Subject,
    producer: str,
    waiting: list[Execution],
    excess: Difference,
) -> tuple[Alternative, float]:
    """KV pressure upstream of the queue: the time each request waited
    behind the subject's own preempted requests, which hold the head of the
    queue until they resume, as the median over the requests against the
    median wait excess. Only the subject's allocation preemptions count:
    another client's, or a reset's, are no evidence of its KV pressure. At
    the upstream share it is ``upstream`` here, and stays so only if the
    subject's KV finding is eligible (``diagnosis_roles``). Also returns
    the median request's held time, in ns."""
    kind = "kv_preemption_pressure"
    holds = held_admissions(context, subject, producer)
    intervals = _wait_intervals(context, waiting)
    if not holds or not intervals:
        reason = "no allocation preemption of the subject's requests held the queue"
        return Alternative(kind, RULED_OUT, reason), 0.0
    typical = median(sum(_overlap(h, w) for h in holds) for w in intervals)
    status = _by_share(context, typical / excess.estimate, QUEUE_KV_SHARE)
    reason = (
        f"the median request waited {typical / 1e6:.1f} ms behind the subject's "
        "preempted requests awaiting their resume, against a wait excess of "
        f"{excess.estimate / 1e6:.1f} ms"
    )
    status = UPSTREAM if status == NOT_RULED_OUT else status
    return Alternative(kind, status, reason), float(typical)


def _kv_claim(context: Context, held: float, ttft: Difference | None) -> dict[str, Any]:
    """What KV pressure explains through this queue, if it is upstream: the
    median request's held time against the TTFT excess, by the queue's own
    contribution share."""
    share = resolve_threshold(QUEUE_CONTRIBUTION, context.thresholds)[0]
    explains = ttft is not None and ttft.estimate > 0 and held >= share * ttft.estimate
    return {"held_p50_ms": round(held / 1e6, 3), "explains_ttft_excess": explains}


def _client_admission(context: Context, subject: Subject) -> Alternative:
    if subject.basis == "engine":
        return Alternative("client_admission", UNTESTABLE, "no client requests")
    held = sum(
        1
        for r in subject.requests
        if context.view.client[r].terminal_raw.get("held_for_slot")
    )
    if held:
        return Alternative(
            "client_admission",
            NOT_RULED_OUT,
            f"{held} requests were held at the client",
        )
    return Alternative(
        "client_admission", RULED_OUT, "no request was held at the client"
    )


def _api_server(context: Context, subject: Subject, excess: Difference) -> Alternative:
    if subject.basis == "engine":
        return Alternative("host_stall@api_server", UNTESTABLE, "no client requests")
    front = context.segment_excess(subject, "send_to_ingress")
    if front is None:
        return Alternative(
            "host_stall@api_server", UNTESTABLE, "send_to_ingress not placed"
        )
    status = _by_share(context, front.estimate / excess.estimate, QUEUE_FRONT_SHARE)
    reason = f"send_to_ingress excess {front.estimate / 1e6:.1f} ms"
    return Alternative("host_stall@api_server", status, reason)


# ---------------------------------------------------------------- helpers
def _engine_excess(context: Context, subject: Subject, name: str) -> Difference | None:
    """An engine segment's excess over the subject's executions."""
    arms = [
        [
            float(value)
            for e in context.subject_executions(subject, reference=reference)
            if (value := context.engine_segments(e).get(name)) is not None
        ]
        for reference in (False, True)
    ]
    return median_difference(arms[0], arms[1])


def _ttft_excess(context: Context, subject: Subject) -> Difference | None:
    """The client's TTFT excess, or without client requests the engine's
    own, from admission to the first step that kept a token."""
    if subject.basis == "engine":
        return _engine_excess(context, subject, "engine_ttft")
    return context.total_excess(subject, "ttft")


def _wait_span(context: Context, executions: list[Execution]) -> tuple[int, int] | None:
    """From the earliest entry into the queue to the latest first schedule."""
    intervals = _wait_intervals(context, executions)
    if not intervals:
        return None
    return min(s for s, _ in intervals), max(e for _, e in intervals)


def _covered(epoch: Any, span: tuple[int, int]) -> bool:
    """Whether the hook's loss coverage spans the whole interval."""
    spans = (epoch.coverage or {}).get("spans") or [] if epoch is not None else []
    return any(
        s.get("start_mono_ns", 0) <= span[0] and span[1] <= s.get("end_mono_ns", -1)
        for s in spans
    )


def _overlap(interval: tuple[int, int], span: tuple[int, int]) -> int:
    return max(0, min(interval[1], span[1]) - max(interval[0], span[0]))


def _criteria(
    context: Context,
    producer: str,
    span: tuple[int, int] | None,
    waits: _Waits,
    excess: Difference,
    ttft: Difference | None,
    witness: _Witness,
    alternatives: list[Alternative],
) -> tuple[Criteria, Criteria]:
    epoch = context.epoch_of(producer)
    covered = span is not None and _covered(epoch, span)
    condition = met(
        direct_evidence=True,
        sufficient_samples=excess.n >= 20 and excess.n_ref >= 20,
        robust_to_clock=True,  # engine-clock segments are exact
    )
    contribution_share = (
        ttft is not None
        and ttft.estimate > 0
        and excess.estimate / ttft.estimate
        >= resolve_threshold(QUEUE_CONTRIBUTION, context.thresholds)[0]
    )
    contribution = met(
        excess_ci_excludes_zero=excess.excludes_zero,
        explains_ttft_excess=contribution_share,
        # An upstream cause is not excluded: the queue may be its consequence.
        competitors_excluded=all(a.status == RULED_OUT for a in alternatives),
        witness=witness.held,
    )
    return (
        Criteria(condition.met, condition.unmet, coverage_unknown=not covered),
        contribution,
    )


def _observations(
    waits: _Waits,
    excess: Difference,
    ttft: Difference | None,
    witness: _Witness,
) -> list[Observation]:
    label = (
        "scheduler_wait"
        if waits.segment == WAIT
        else "engine_ingress_to_schedule (admission to first schedule; this log has no queue entry stamps)"
    )
    out = [
        Observation(
            "o1",
            f"{label} rose by {excess.estimate / 1e6:.1f} ms (95% CI {excess.low / 1e6:.1f} to {excess.high / 1e6:.1f}) against the reference.",
            f"{waits.segment}_excess_ms",
            round(excess.estimate / 1e6, 3),
            (round(excess.low / 1e6, 3), round(excess.high / 1e6, 3)),
            excess.n,
            excess.n_ref,
        )
    ]
    if ttft is not None:
        out.append(
            Observation(
                "o2",
                f"TTFT rose by {ttft.estimate / 1e6:.1f} ms (95% CI {ttft.low / 1e6:.1f} to {ttft.high / 1e6:.1f}).",
                "ttft_excess_ms",
                round(ttft.estimate / 1e6, 3),
                (round(ttft.low / 1e6, 3), round(ttft.high / 1e6, 3)),
                ttft.n,
                ttft.n_ref,
            )
        )
    if witness.requests_share is not None and witness.steps_share is not None:
        out.append(
            Observation(
                "o3",
                f"{witness.requests_share:.0%} of the subject's requests waited mostly through steps at capacity; {witness.steps_share:.0%} of the {witness.steps} steps scheduled while requests waited ran at capacity, counting the slots freed in the step before.",
                "requests_waiting_at_capacity_share",
                witness.requests_share,
                n=witness.steps,
            )
        )
    return out


def _message(waits: _Waits, excess: Difference, witness: _Witness) -> str:
    share = (
        ""
        if witness.requests_share is None
        else f"; {witness.requests_share:.0%} of the subject's requests waited mostly through steps at capacity"
    )
    return f"Median {waits.segment} rose by {excess.estimate / 1e6:.1f} ms against the reference{share}."


def _experiment(excess: Difference) -> dict[str, Any]:
    return {
        "change": "rerun at a lower offered load, or with a larger max_num_seqs, same seed",
        "prediction": f"median scheduler_wait falls by at least {excess.low / 1e6:.1f} ms",
    }


def _iterations(spanning: list[Step]) -> dict[str, Any]:
    if not spanning:
        return {"iterations": None}
    return {
        "iterations": {
            "first": spanning[0].iteration,
            "last": spanning[-1].iteration,
            "count": len(spanning),
        }
    }


def _support(
    context: Context, subject: Subject, spanning: list[Step]
) -> tuple[list[Line], list[Line]]:
    lines: list[Line] = []
    for request_id in subject.requests:
        lines.extend(context.view.client[request_id].lines())
    executions = context.subject_executions(subject)
    for execution in executions:
        lines.append(execution.line)
        lines.extend(line for line, _ in execution.memberships)
    lines.extend(step.line for step in spanning)
    longest = sorted(executions, key=lambda e: -_wait_of(context, e))[:4]
    display = [_display_line(context, e) for e in longest]
    display += [step.line for step in spanning[:4]]
    return lines, display[:8]


def _display_line(context: Context, execution: Execution) -> Line:
    """A waiting request's own record: the client's, or the engine's."""
    request = context.view.client.get(execution.event.request_ref.id)
    lines = request.lines() if request is not None else []
    return lines[-1] if lines else execution.line


def _wait_of(context: Context, execution: Execution) -> float:
    segments = context.engine_segments(execution)
    return float(segments.get(WAIT, segments.get(MERGED_INGRESS, 0)))


__all__ = ["assess_queue"]
