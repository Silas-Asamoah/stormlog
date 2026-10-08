"""Queue saturation: requests waited to be scheduled because the engine was
full.

The class compares the subject's ``scheduler_wait`` with its reference's
(the difference of medians, with a bootstrap interval), and asks whether the
engine was at capacity over the steps spanning the waits: running at
``max_num_seqs``, or scheduling ``max_num_batched_tokens``. Four competitors
must be ruled out before the waits may be called a fault: a stalled engine,
a paused scheduler, requests blocked by their own constraints, and time
spent before the queue rather than in it. Others are reported: preemptions
upstream, the client holding requests back, the API server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import Line
from .diagnosis_join import Execution
from .diagnosis_loop import LoopGapConfig, stalls_over_limit
from .diagnosis_metrics import aggregate_assessment, subject_signal
from .diagnosis_model import (
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
from .diagnosis_steps import Step, Steps, loop_steps
from .diagnosis_thresholds import (
    LOOP_NO_BASELINE_FLOOR_NS,
    QUEUE_CONTRIBUTION,
    QUEUE_STALL_COVERAGE,
    QUEUE_WITNESS_SHARE,
    resolve_threshold,
)
from .diagnosis_vocabulary import COMPONENT_SCHEDULER, QUEUE_SATURATION

NO_SERVER_QUEUE_SIGNAL = "no_server_queue_signal"
NO_CAPACITY_WITNESS = "no_capacity_witness"
NO_CLIENT_LATENCY = "no_client_latency"
SEVERAL_ENGINES = "several_engines"
WAIT = "scheduler_wait"


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
    spanning = steps.between(*span) if span else []
    witness = _witness(context, producer, spanning)
    if not witness[0]:
        witness = _exporter_witness(context, subject, witness)
    ttft = _ttft_excess(context, subject)
    alternatives = [
        _engine_stall(context, producer, waiting),
        _scheduler_paused(context, producer, steps, span),
        _blocked_waiting(waiting),
        _engine_ingress(context, subject, waits, excess),
        _preemption(spanning),
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
            "capacity_witness": witness[0],
            "usable_timing": waits.segment == WAIT,
        },
        alternatives=alternatives,
        observations=_observations(waits, excess, ttft, witness, spanning),
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
            "steps_at_capacity_share": witness[1],
        },
        experiment=_experiment(excess),
    )
    finding.condition, finding.contribution = _criteria(
        context, producer, span, waits, excess, ttft, witness, alternatives
    )
    finding.segments = {
        "decomposition": "ttft" if subject.basis == "client" else "engine_ttft",
        "wait_segment": waits.segment,
    }
    finding.support, finding.display = _support(context, subject, spanning)
    reasons = [] if witness[0] else [NO_CAPACITY_WITNESS]
    if subject.basis == "engine":
        reasons.append(NO_CLIENT_LATENCY)
    status = PARTIAL if reasons else ASSESSED
    finding.status = status
    finding.partial_reasons = reasons
    return Assessment(QUEUE_SATURATION, subject.key, status, reasons, [finding])


# ----------------------------------------------------------------- witness
def _witness(
    context: Context, producer: str, spanning: list[Step]
) -> tuple[bool, float | None]:
    """Whether most steps spanning the waits ran at capacity: at
    max_num_seqs running, or at the max_num_batched_tokens budget."""
    seqs, tokens = _capacity(context, producer)
    busy = [step for step in spanning if step.members]
    if not busy or (seqs is None and tokens is None):
        return False, None
    full = sum(1 for step in busy if _at_capacity(step, seqs, tokens))
    share = full / len(busy)
    needed = resolve_threshold(QUEUE_WITNESS_SHARE, context.thresholds)[0]
    return share >= needed, round(share, 4)


def _exporter_witness(
    context: Context, subject: Subject, hook: tuple[bool, float | None]
) -> tuple[bool, float | None]:
    """Requests waiting for scheduling capacity, as vLLM's own metric
    counts them (its help text: "waiting for scheduling capacity", not only
    KV): a witness only from an exporter asserted to be this engine."""
    if not context.metrics_from_engine:
        return hook
    signal = subject_signal(context, subject, QUEUE_SATURATION)
    reasons = signal.detail.get("waiting_by_reason") if signal is not None else None
    capacity = reasons.get("capacity") if isinstance(reasons, dict) else None
    if isinstance(capacity, (int, float)) and capacity > 0:
        return True, hook[1]
    return hook


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
    return (seqs is not None and step.members >= seqs) or (
        tokens is not None and step.total_tokens >= tokens
    )


# ------------------------------------------------------------- competitors
def _engine_stall(
    context: Context, producer: str, waiting: list[Execution]
) -> Alternative:
    """Engine-loop stalls, by the online trigger's own rules, covering most
    of the subject's waiting time: then the engine, not its capacity, kept
    the requests waiting."""
    intervals = _wait_intervals(context, waiting)
    if not intervals:
        return Alternative("engine_stall", UNTESTABLE, "no wait was placed", True)
    config = LoopGapConfig(thresholds=dict(context.thresholds or {}))
    stalls = stalls_over_limit(loop_steps(context.view, producer), (), config)
    stalled = [(s.start_mono_ns, s.start_mono_ns + s.duration_ns) for s, _ in stalls]
    total = sum(end - start for start, end in intervals)
    covered = sum(_covered_by(interval, stalled) for interval in intervals)
    share = covered / total if total else 0.0
    needed = resolve_threshold(QUEUE_STALL_COVERAGE, context.thresholds)[0]
    status = NOT_RULED_OUT if share >= needed else RULED_OUT
    reason = f"{len(stalls)} engine stalls cover {share:.0%} of the waiting time"
    return Alternative("engine_stall", status, reason, True)


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


def _covered_by(interval: tuple[int, int], stalls: list[tuple[int, int]]) -> int:
    """How much of ``interval`` the union of ``stalls`` covers."""
    pieces = sorted(
        (max(start, interval[0]), min(end, interval[1]))
        for start, end in stalls
        if start < interval[1] and interval[0] < end
    )
    covered, reach = 0, interval[0]
    for start, end in pieces:
        start = max(start, reach)
        if end > start:
            covered += end - start
            reach = end
    return covered


def _scheduler_paused(
    context: Context, producer: str, steps: Steps, span: tuple[int, int] | None
) -> Alternative:
    """Ruled out by the hook's pause records where nothing was lost, else by
    admissions continuing throughout the waits."""
    kind = "scheduler_paused"
    if span is None:
        return Alternative(kind, UNTESTABLE, "no step spans the waits", True)
    paused = [s for s in steps.paused_intervals(span[1]) if _overlap(s[:2], span)]
    if paused:
        return Alternative(
            kind, NOT_RULED_OUT, f"{paused[0][2]} overlaps the waits", True
        )
    epoch = context.epoch_of(producer)
    observes = epoch.observes if epoch is not None else None
    if observes is not None and "pause" in observes and _covered(epoch, span):
        return Alternative(
            kind, RULED_OUT, "no pause transition, and nothing lost", True
        )
    if _admissions_continue(context, steps, span):
        return Alternative(
            kind, RULED_OUT, "admissions continued throughout the waits", True
        )
    return Alternative(
        kind, UNTESTABLE, "no pause records, and admissions stopped", True
    )


def _admissions_continue(context: Context, steps: Steps, span: tuple[int, int]) -> bool:
    """Whether some request was admitted at least every stall floor
    throughout the span: a paused scheduler admits nobody."""
    admissions = [step.start_ns for step in steps.between(*span) if step.admitted]
    if not admissions:
        return False
    floor = resolve_threshold(LOOP_NO_BASELINE_FLOOR_NS, context.thresholds)[0]
    edges = [span[0], *admissions, span[1]]
    return max(b - a for a, b in zip(edges, edges[1:])) < floor


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
    if ingress.estimate < 0.25 * excess.estimate:
        return Alternative(
            "engine_ingress",
            RULED_OUT,
            f"engine_ingress excess {ingress.estimate / 1e6:.1f} ms",
            True,
        )
    return Alternative(
        "engine_ingress",
        NOT_RULED_OUT,
        f"engine_ingress excess {ingress.estimate / 1e6:.1f} ms",
        True,
    )


def _preemption(spanning: list[Step]) -> Alternative:
    preempted = sum(step.preempted for step in spanning)
    if preempted:
        return Alternative(
            "kv_preemption_pressure",
            UPSTREAM,
            f"{preempted} preemptions in the steps spanning the waits",
        )
    return Alternative(
        "kv_preemption_pressure",
        RULED_OUT,
        "no preemption in the steps spanning the waits",
    )


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
    if front.estimate < 0.25 * excess.estimate:
        return Alternative(
            "host_stall@api_server",
            RULED_OUT,
            f"send_to_ingress excess {front.estimate / 1e6:.1f} ms",
        )
    return Alternative(
        "host_stall@api_server",
        NOT_RULED_OUT,
        f"send_to_ingress excess {front.estimate / 1e6:.1f} ms",
    )


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
    witness: tuple[bool, float | None],
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
        competitors_excluded=all(
            a.status in (RULED_OUT, UPSTREAM) for a in alternatives
        ),
        witness=witness[0],
    )
    return (
        Criteria(condition.met, condition.unmet, coverage_unknown=not covered),
        contribution,
    )


def _observations(
    waits: _Waits,
    excess: Difference,
    ttft: Difference | None,
    witness: tuple[bool, float | None],
    spanning: list[Step],
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
    if witness[1] is not None:
        busy = len([s for s in spanning if s.members])
        out.append(
            Observation(
                "o3",
                f"{witness[1]:.0%} of the {busy} steps spanning the waits ran at capacity.",
                "steps_at_capacity_share",
                witness[1],
                n=busy,
            )
        )
    return out


def _message(
    waits: _Waits, excess: Difference, witness: tuple[bool, float | None]
) -> str:
    share = (
        ""
        if witness[1] is None
        else f"; {witness[1]:.0%} of the steps spanning the waits ran at capacity"
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
