"""KV preemption pressure: running requests were preempted because KV
blocks ran out, waited to resume, and recomputed what they had.

A preemption is the engine's to blame on memory only when it came from the
allocation loop. vLLM also preempts every running request when the prefix
cache is reset, and lists those preemptions in the next step with its own;
the import attributes them to the reset. "No reset" is established only
where the hook records resets and lost nothing: otherwise the cause of the
step's preemptions is unknown, and the finding is only an observation that
preemption and recomputation happened.

The cost per affected request is observed, not estimated: the wait from the
preempting ``schedule()`` call to the call that resumed it, and the
positions it computed again (below the highest context it had reached).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .correlation_events import StageEvent
from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import Line
from .diagnosis_join import Execution
from .diagnosis_metrics import aggregate_assessment
from .diagnosis_model import (
    RULED_OUT,
    Alternative,
    Finding,
    Observation,
    met,
)
from .diagnosis_selection import Subject
from .diagnosis_stats import Difference
from .diagnosis_vocabulary import COMPONENT_KV_CACHE, KV_PREEMPTION_PRESSURE

NOT_OBSERVED = "not_observed"
NO_HOOK = "no_hook_preemption_data"
CAUSE_UNKNOWN = "preemption_cause_unknown"
PREEMPTED = "engine.preempted"


@dataclass(frozen=True)
class _Cost:
    """What preemption cost one affected request."""

    request_id: str
    preemptions: int
    resume_waits_ns: tuple[int, ...]  # preempting schedule() to resume entry
    recomputed: int  # positions computed again


def assess_kv(context: Context, subject: Subject) -> Assessment:
    """The KV class on one subject."""
    producers = {e.producer for e in context.subject_executions(subject)}
    producer = next(iter(producers)) if len(producers) == 1 else None
    if producer is None:
        aggregate = aggregate_assessment(
            context,
            subject,
            KV_PREEMPTION_PRESSURE,
            "vLLM counted {value:.0f} preemptions over the window.",
            NO_HOOK,
        )
        return aggregate or Assessment(
            KV_PREEMPTION_PRESSURE, subject.key, UNSUPPORTED, [NO_HOOK]
        )
    span = _span(context, subject)
    stages = _stages(context, producer, span)
    preemptions = [(line, s) for line, s in stages if s.name == PREEMPTED]
    costs = _costs(context, subject, preemptions)
    if not costs:
        return Assessment(KV_PREEMPTION_PRESSURE, subject.key, ASSESSED, [NOT_OBSERVED])
    reset = context.reset_absent(producer, span)
    finding = _finding(context, subject, producer, costs, reset, preemptions)
    reasons = [] if reset.status == RULED_OUT else [CAUSE_UNKNOWN]
    status = PARTIAL if reasons else ASSESSED
    finding.status = status
    finding.partial_reasons = reasons
    return Assessment(KV_PREEMPTION_PRESSURE, subject.key, status, reasons, [finding])


def _span(context: Context, subject: Subject) -> tuple[int, int] | None:
    """The subject's requests' engine lifetimes: first admission to last
    step, on the engine's monotonic clock."""
    starts, ends = [], []
    for execution in context.subject_executions(subject):
        if execution.event.start_ns is not None:
            starts.append(execution.event.start_ns)
        for _, membership in execution.memberships[-1:]:
            step = context.view.iterations.get(membership.iteration_ref)
            if step is not None and step[1].end_ns is not None:
                ends.append(step[1].end_ns)
    return (min(starts), max(ends)) if starts and ends else None


def _stages(
    context: Context, producer: str, span: tuple[int, int] | None
) -> list[tuple[Line, StageEvent]]:
    if span is None:
        return []
    return [
        (line, stage)
        for line, stage in context.view.stages
        if stage.stage_ref.producer_id == producer
        and stage.start_ns is not None
        and span[0] <= stage.start_ns <= span[1]
    ]


def _costs(
    context: Context, subject: Subject, preemptions: list[tuple[Line, StageEvent]]
) -> list[_Cost]:
    by_attempt: dict[str, list[int]] = {}
    for _, stage in preemptions:
        attempt = stage.metadata.get("attempt")
        if isinstance(attempt, str) and stage.start_ns is not None:
            by_attempt.setdefault(attempt, []).append(stage.start_ns)
    costs = []
    for execution in context.subject_executions(subject):
        attempt = execution.attempt.id if execution.attempt else None
        if attempt in by_attempt:
            costs.append(_cost(context, execution, by_attempt[attempt]))
    return costs


def _cost(context: Context, execution: Execution, preempted_at: list[int]) -> _Cost:
    starts = [
        step[1].start_ns
        for _, membership in execution.memberships
        if (step := context.view.iterations.get(membership.iteration_ref)) is not None
        and step[1].start_ns is not None
    ]
    waits = []
    for at in sorted(preempted_at):
        resume = next((start for start in starts if start > at), None)
        if resume is not None:
            waits.append(resume - at)
    request_id = execution.event.request_ref.id
    return _Cost(request_id, len(preempted_at), tuple(waits), _recomputed(execution))


def _recomputed(execution: Execution) -> int:
    """Positions scheduled again below the highest context the request had
    reached: chunked recovery, cache hits on resume and repeated preemption
    are all counted right, since the high-water mark only rises."""
    high, again = 0, 0
    for _, membership in execution.memberships:
        data = membership.metadata
        before, scheduled = data.get("computed_before"), membership.input_tokens
        if isinstance(before, int) and isinstance(scheduled, int):
            again += max(0, min(before + scheduled, high) - before)
        after = data.get("computed_after")
        if data.get("outcome") == "kept" and isinstance(after, int):
            high = max(high, after)
    return again


def _finding(
    context: Context,
    subject: Subject,
    producer: str,
    costs: list[_Cost],
    reset: Alternative,
    preemptions: list[tuple[Line, StageEvent]],
) -> Finding:
    e2e = context.total_excess(subject, "e2e")
    waits = [w for cost in costs for w in cost.resume_waits_ns]
    recomputed = sum(cost.recomputed for cost in costs)
    count = sum(cost.preemptions for cost in costs)
    finding = Finding(
        kind=KV_PREEMPTION_PRESSURE,
        component=COMPONENT_KV_CACHE,
        subject=subject.as_dict(),
        title="Running requests were preempted for KV space and recomputed",
        message=f"{count} allocation preemptions hit {len(costs)} of the subject's requests.",
        gates={"allocation_cause_established": reset.status == RULED_OUT},
        alternatives=[reset],
        condition=met(
            direct_evidence=True,
            sufficient_samples=len(costs) >= 3,
            robust_to_clock=True,
        ),
        contribution=met(
            excess_ci_excludes_zero=e2e is not None and e2e.low > 0,
            explains_e2e_excess=_explains(waits, e2e, len(subject.requests)),
        ),
        contribution_lower=None if e2e is None else e2e.low / 1e6,
        observations=_observations(count, costs, waits, recomputed, e2e),
        location={"component": COMPONENT_KV_CACHE, "engine_producer": producer},
        window=context.window(subject),
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        metrics=_metrics(count, costs, waits, recomputed),
        experiment={
            "change": "rerun with a larger gpu_memory_utilization, or a smaller max_num_seqs, same seed",
            "prediction": "no allocation preemption, and end-to-end latency falls by the resume waits",
        },
        explains="explains_e2e_excess",
    )
    finding.support = [line for line, _ in preemptions] + _client_lines(
        context, [cost.request_id for cost in costs]
    )
    finding.display = finding.support[:8]
    return finding


def _client_lines(context: Context, request_ids: list[str]) -> list[Line]:
    """The affected requests' own records, when the client wrote them."""
    return [
        line
        for request_id in request_ids
        if request_id in context.view.client
        for line in context.view.client[request_id].lines()
    ]


def _metrics(
    count: int, costs: list[_Cost], waits: list[int], recomputed: int
) -> dict[str, float | int | None]:
    return {
        "allocation_preemptions": count,
        "affected_requests": len(costs),
        "recomputed_positions": recomputed,
        "resume_wait_p50_ms": round(median(waits) / 1e6, 3) if waits else None,
    }


def _explains(waits: list[int], e2e: Difference | None, requests: int) -> bool:
    """The resume waits add up to at least half the subject's excess."""
    if e2e is None or e2e.estimate <= 0 or not requests:
        return False
    return sum(waits) >= 0.5 * e2e.estimate * requests


def _observations(
    count: int,
    costs: list[_Cost],
    waits: list[int],
    recomputed: int,
    e2e: Difference | None,
) -> list[Observation]:
    out = [
        Observation(
            "o1",
            f"{count} allocation preemptions hit {len(costs)} of the subject's requests.",
            "allocation_preemptions",
            count,
            n=len(costs),
        ),
        Observation(
            "o2",
            f"Those requests computed {recomputed} positions again after resuming.",
            "recomputed_positions",
            recomputed,
        ),
    ]
    if waits:
        out.append(
            Observation(
                "o3",
                f"From the preempting schedule() call to the resume, the median wait was {median(waits) / 1e6:.1f} ms.",
                "preemption_to_resume_entry_p50_ms",
                round(median(waits) / 1e6, 3),
                n=len(waits),
            )
        )
    if e2e is not None:
        out.append(
            Observation(
                "o4",
                f"End-to-end latency rose by {e2e.estimate / 1e6:.1f} ms (95% CI {e2e.low / 1e6:.1f} to {e2e.high / 1e6:.1f}).",
                "e2e_excess_ms",
                round(e2e.estimate / 1e6, 3),
                (round(e2e.low / 1e6, 3), round(e2e.high / 1e6, 3)),
                e2e.n,
                e2e.n_ref,
            )
        )
    return out


__all__ = ["assess_kv"]
