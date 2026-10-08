"""Prefix-cache loss: requests that should have found their shared prefix
cached found less of it.

A request is *warm* when an earlier request of its prefix group had finished
prefilling the shared span (its ``computed_after`` reached it) before this
request entered the scheduler: the prefix had been computed, so it could be
cached. Concurrent first use is not warm by construction. What a warm
request should find is its group's own experience: the cached tokens of the
reference's warm requests of the same group, not a length derived from a
tokenizer. The finding is warm requests finding less than that, as a paired
excess with an interval.

Two competitors are indispensable: a reset of the prefix cache (ruled out
only where the hook records resets and lost nothing), and the workload
sharing less (the declared prefix groups and shared lengths changed).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import Line
from .diagnosis_model import (
    NOT_RULED_OUT,
    RULED_OUT,
    UNTESTABLE,
    Alternative,
    Finding,
    Observation,
    met,
)
from .diagnosis_selection import Subject
from .diagnosis_stats import INSUFFICIENT_SAMPLES, Difference, median_difference
from .diagnosis_thresholds import PREFIX_WORKING_SET_RATIO, resolve_threshold
from .diagnosis_vocabulary import COMPONENT_PREFIX_CACHE, PREFIX_CACHE_LOSS

NOT_OBSERVED = "not_observed"
NO_CACHE_EVIDENCE = "no_per_request_cache_evidence"
NO_DECLARED_SHARING = "no_declared_sharing"
TOO_FEW_WARM = "too_few_warm_requests"


@dataclass
class _Warmth:
    """When each prefix group's shared span was first prefilled, per request."""

    done: dict[Any, list[tuple[int, str]]] = field(default_factory=dict)

    def warmed_at(self, group: Any, request_id: str, entered: int) -> int | None:
        """When another request of the group last finished prefilling the
        shared span before ``entered``; None when none had: not warm."""
        times = self.done.get(group, [])
        index = bisect_left(times, (entered, ""))
        earlier = [at for at, other in times[:index] if other != request_id]
        return earlier[-1] if earlier else None


def assess_prefix(context: Context, subject: Subject) -> Assessment:
    """The prefix-cache class on one subject."""
    if not subject.requests:
        return _verdict(subject, UNSUPPORTED, NO_DECLARED_SHARING)
    producer = context.producer_of(subject.requests)
    if producer is None:
        return _verdict(subject, UNSUPPORTED, NO_CACHE_EVIDENCE)
    groups = {r: g for r in context.view.client if (g := _group(context, r))}
    if not any(r in groups for r in subject.requests):
        return _verdict(subject, UNSUPPORTED, NO_DECLARED_SHARING)
    warmth = _warmth(context, groups)
    subject_warm = _warm_cached(context, subject.requests, groups, warmth)
    reference_warm = _warm_cached(context, subject.reference, groups, warmth)
    if not subject_warm:
        # Nothing was cached for these requests to find, as at first use.
        return _verdict(subject, ASSESSED, TOO_FEW_WARM)
    expected = _expected(reference_warm, groups)
    deficits = _deficits(subject_warm, groups, expected)
    reference_deficits = _deficits(reference_warm, groups, expected)
    excess = median_difference(deficits, reference_deficits)
    if excess is None or excess.low <= 0:
        return _no_loss_measured(subject, excess)
    finding = _finding(context, subject, producer, groups, subject_warm, excess)
    return Assessment(PREFIX_CACHE_LOSS, subject.key, ASSESSED, [], [finding])


def _no_loss_measured(subject: Subject, excess: Difference | None) -> Assessment:
    """Too few warm requests on a side to tell a loss from none, or a
    shortfall whose interval does not exclude zero."""
    if excess is None:
        return _verdict(subject, PARTIAL, INSUFFICIENT_SAMPLES)
    return _verdict(subject, ASSESSED, NOT_OBSERVED)


def _verdict(subject: Subject, status: str, reason: str) -> Assessment:
    return Assessment(PREFIX_CACHE_LOSS, subject.key, status, [reason])


def _group(context: Context, request_id: str) -> tuple[Any, int] | None:
    """A request's declared prefix group and shared length, if it shares."""
    raw = context.view.client[request_id].terminal_raw
    group, shared = raw.get("prefix_group"), raw.get("shared_prefix_tokens")
    if group is None or not isinstance(shared, int) or shared <= 0:
        return None
    return group, shared


def _warmth(context: Context, groups: dict[str, tuple[Any, int]]) -> _Warmth:
    """For every grouped request, when its first step reaching the shared
    span completed."""
    warmth = _Warmth()
    for request_id, (group, shared) in groups.items():
        for execution in context.view.executions_of(request_id):
            for _, membership in execution.memberships:
                after = membership.metadata.get("computed_after")
                step = context.view.iterations.get(membership.iteration_ref)
                if (
                    isinstance(after, int)
                    and after >= shared
                    and step
                    and step[1].end_ns
                ):
                    warmth.done.setdefault(group, []).append(
                        (step[1].end_ns, request_id)
                    )
                    break
    for times in warmth.done.values():
        times.sort()
    return warmth


@dataclass(frozen=True)
class _Warm:
    """A warm request: what it found cached, when it entered the engine,
    and when its group had last finished prefilling the shared span."""

    cached: int
    entered: int
    warmed: int


def _warm_cached(
    context: Context,
    request_ids: list[str],
    groups: dict[str, tuple[Any, int]],
    warmth: _Warmth,
) -> dict[str, _Warm]:
    """Warm requests and the tokens they found cached when admitted."""
    found = {}
    for request_id in request_ids:
        if request_id not in groups:
            continue
        for execution in context.view.executions_of(request_id)[:1]:
            data = execution.metadata
            entered = data.get("enqueued_mono_ns") or execution.event.start_ns
            cached = data.get("cached_at_admission")
            if not isinstance(entered, int) or not isinstance(cached, int):
                continue
            warmed = warmth.warmed_at(groups[request_id][0], request_id, entered)
            if warmed is not None:
                found[request_id] = _Warm(cached, entered, warmed)
    return found


def _expected(
    reference: dict[str, _Warm], groups: dict[str, tuple[Any, int]]
) -> dict[Any, float]:
    """Each group's warm reference: what its warm requests found cached."""
    by_group: dict[Any, list[int]] = {}
    for request_id, warm in reference.items():
        by_group.setdefault(groups[request_id][0], []).append(warm.cached)
    return {group: float(median(values)) for group, values in by_group.items()}


def _deficits(
    warm: dict[str, _Warm],
    groups: dict[str, tuple[Any, int]],
    expected: dict[Any, float],
) -> list[float]:
    """How many tokens short of its group's warm reference each request was."""
    return [
        expected[groups[r][0]] - found.cached
        for r, found in warm.items()
        if groups[r][0] in expected
    ]


def _finding(
    context: Context,
    subject: Subject,
    producer: str,
    groups: dict[str, tuple[Any, int]],
    warm: dict[str, _Warm],
    excess: Difference,
) -> Finding:
    sharing = _sharing(context, subject, groups)
    working_set = _working_set(context, subject, groups)
    reset = context.reset_absent(producer, _exposure(warm))
    prefill = context.total_excess(subject, "ttft")
    return Finding(
        kind=PREFIX_CACHE_LOSS,
        component=COMPONENT_PREFIX_CACHE,
        subject=subject.as_dict(),
        title="Warm requests found less of their shared prefix cached",
        message=f"Warm requests were a median {excess.estimate:.0f} tokens short of their group's cached prefix.",
        gates={"warm_requests": True},
        alternatives=[reset, sharing, working_set],
        condition=met(direct_evidence=True, sufficient_samples=excess.n >= 20),
        contribution=met(
            excess_ci_excludes_zero=excess.excludes_zero,
            ttft_rose=prefill is not None and prefill.low > 0,
        ),
        # In ms of latency, as every kind's: the most the loss can explain
        # is the TTFT excess.
        contribution_lower=(
            prefill.low / 1e6 if prefill is not None and prefill.low > 0 else None
        ),
        observations=[
            Observation(
                "o1",
                f"{len(warm)} warm requests were a median {excess.estimate:.0f} tokens (95% CI {excess.low:.0f} to {excess.high:.0f}) short of their group's warm reference.",
                "uncached_prefix_tokens_excess",
                round(excess.estimate, 3),
                (round(excess.low, 3), round(excess.high, 3)),
                excess.n,
                excess.n_ref,
                provenance="derived",
            )
        ],
        location={"component": COMPONENT_PREFIX_CACHE, "engine_producer": producer},
        window=context.window(subject),
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        metrics={
            "warm_requests": len(warm),
            "uncached_tokens_excess": round(excess.estimate, 3),
        },
        detail={"token_count_provenance": "engine_cached_at_admission"},
        experiment={
            "change": "rerun without the competing load or cache reset, same seed and prompts",
            "prediction": "warm requests find their group's prefix cached again",
        },
        support=_lines(context, list(warm)),
        display=_lines(context, list(warm))[:8],
        explains="ttft_rose",
    )


def _exposure(warm: dict[str, _Warm]) -> tuple[int, int]:
    """Where a reset would explain the misses: from the earliest time a
    group was warm for one of these requests to the last one's entry."""
    return (
        min(found.warmed for found in warm.values()),
        max(found.entered for found in warm.values()),
    )


def _sharing(
    context: Context, subject: Subject, groups: dict[str, tuple[Any, int]]
) -> Alternative:
    """The workload shared less: fewer requests in groups, or shorter spans."""

    def profile(request_ids: list[str]) -> tuple[float, float]:
        shared = [groups[r][1] for r in request_ids if r in groups]
        share = len(shared) / len(request_ids) if request_ids else 0.0
        return share, float(median(shared)) if shared else 0.0

    now, before = profile(subject.requests), profile(subject.reference)
    if not subject.reference:
        return Alternative(
            "prefix_sharing_drop", UNTESTABLE, "no reference requests", True
        )
    unchanged = now[0] >= 0.9 * before[0] and now[1] >= 0.9 * before[1]
    status = RULED_OUT if unchanged else NOT_RULED_OUT
    reason = f"{now[0]:.0%} of requests share a median {now[1]:.0f} tokens, against {before[0]:.0%} sharing {before[1]:.0f}"
    return Alternative("prefix_sharing_drop", status, reason, True)


def _working_set(
    context: Context, subject: Subject, groups: dict[str, tuple[Any, int]]
) -> Alternative:
    """The workload's working set grew: its requests declared more distinct
    shared prefixes (in tokens) than as many of the latest reference
    requests did, and a cache that evicts the least recently used loses
    the old prefixes without any fault."""
    kind = "prefix_working_set_growth"
    recent = subject.reference[-len(subject.requests) :]
    if not recent:
        return Alternative(kind, UNTESTABLE, "no reference requests", True)

    def size(request_ids: list[str]) -> int:
        return sum(
            tokens for _, tokens in {groups[r] for r in request_ids if r in groups}
        )

    now, before = size(subject.requests), size(recent)
    ratio = resolve_threshold(PREFIX_WORKING_SET_RATIO, context.thresholds)[0]
    status = RULED_OUT if now <= ratio * before else NOT_RULED_OUT
    reason = (
        f"the requests declared {now} distinct shared prefix tokens, against "
        f"{before} in as many reference requests"
    )
    return Alternative(kind, status, reason, True)


def _lines(context: Context, request_ids: list[str]) -> list[Line]:
    lines: list[Line] = []
    for request_id in request_ids:
        lines.extend(context.view.client[request_id].lines())
        for execution in context.view.executions_of(request_id):
            lines.append(execution.line)
    return lines


__all__ = ["assess_prefix"]
