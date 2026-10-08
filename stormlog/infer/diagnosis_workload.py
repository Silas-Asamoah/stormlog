"""What the workload asked for: more requests, longer prompts or outputs, or
less shared prefix. These are never faults; they say what changed in the
demand, at ``info``, so a finding about the engine is read beside them.

- ``load_increase``: the subject's requests arrived faster than the
  reference's (a rate ratio with an exact conditional interval).
- ``longer_inputs`` and ``longer_outputs``: its prompts or outputs were
  longer (a difference of medians with a bootstrap interval).
- ``prefix_sharing_drop``: fewer of its requests declared a shared prefix
  (an exact interval on the share).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy import stats

from .diagnosis_context import ASSESSED, UNSUPPORTED, Assessment, Context
from .diagnosis_model import NOT_DETERMINED, Finding, Observation, met
from .diagnosis_selection import Subject, assignment_ns
from .diagnosis_stats import Difference, median_difference
from .diagnosis_thresholds import (
    WORKLOAD_LENGTH_RATIO,
    WORKLOAD_RATE_RATIO,
    WORKLOAD_SHARE_DROP,
    resolve_threshold,
)
from .diagnosis_vocabulary import (
    COMPONENT_WORKLOAD,
    LOAD_INCREASE,
    LONGER_INPUTS,
    LONGER_OUTPUTS,
    PREFIX_SHARING_DROP,
)

NOT_OBSERVED = "not_observed"
NO_CLIENT_REQUESTS = "no_client_requests"
NO_REFERENCE = "no_reference"


@dataclass(frozen=True)
class RateRatio:
    """The subject's arrival rate over the reference's, with its exact 95%
    interval, and whether the interval's lower end passes the threshold."""

    n: int
    seconds: float
    n_ref: int
    seconds_ref: float
    low: float
    high: float
    passes: bool

    @property
    def ratio(self) -> float:
        return (self.n / self.seconds) / (self.n_ref / self.seconds_ref)


def arrival_ratio(context: Context, subject: Subject) -> RateRatio | None:
    """None without client requests on either side."""
    rates = _rates(context, subject) if subject.requests else None
    if rates is None:
        return None
    (n, seconds), (n_ref, seconds_ref) = rates
    low, high = rate_ratio_interval(n, seconds, n_ref, seconds_ref)
    needed = resolve_threshold(WORKLOAD_RATE_RATIO, context.thresholds)[0]
    return RateRatio(n, seconds, n_ref, seconds_ref, low, high, low >= needed)


def assess_load(context: Context, subject: Subject) -> Assessment:
    """More arrivals per second than the reference."""
    if not subject.requests:
        return _verdict(LOAD_INCREASE, subject, UNSUPPORTED, NO_CLIENT_REQUESTS)
    rates = arrival_ratio(context, subject)
    if rates is None:
        return _verdict(LOAD_INCREASE, subject, UNSUPPORTED, NO_REFERENCE)
    if not rates.passes:
        return _verdict(LOAD_INCREASE, subject, ASSESSED, NOT_OBSERVED)
    n, seconds, n_ref, seconds_ref = (
        rates.n,
        rates.seconds,
        rates.n_ref,
        rates.seconds_ref,
    )
    statement = (
        f"Requests arrived at {n / seconds:.2f}/s against {n_ref / seconds_ref:.2f}/s "
        f"in the reference: {rates.ratio:.2f}x (95% CI {rates.low:.2f} to {rates.high:.2f})."
    )
    return _workload(
        context,
        subject,
        LOAD_INCREASE,
        "More requests arrived than in the reference",
        Observation(
            "o1",
            statement,
            "arrival_rate_ratio",
            round(rates.ratio, 4),
            (round(rates.low, 4), round(rates.high, 4)),
            n,
            n_ref,
        ),
    )


def assess_inputs(context: Context, subject: Subject) -> Assessment:
    return _lengths(context, subject, LONGER_INPUTS, "prompt_tokens", "prompts")


def assess_outputs(context: Context, subject: Subject) -> Assessment:
    return _lengths(context, subject, LONGER_OUTPUTS, "output_tokens", "outputs")


def assess_sharing(context: Context, subject: Subject) -> Assessment:
    """Fewer requests declaring a shared prefix than in the reference."""
    if not subject.requests:
        return _verdict(PREFIX_SHARING_DROP, subject, UNSUPPORTED, NO_CLIENT_REQUESTS)
    shares = [_sharing(context, ids) for ids in (subject.requests, subject.reference)]
    if not subject.requests or not subject.reference or None in shares:
        return _verdict(PREFIX_SHARING_DROP, subject, UNSUPPORTED, NO_REFERENCE)
    (shared, n), (shared_ref, n_ref) = shares  # type: ignore[misc]
    upper = _clopper_pearson(shared, n)[1]
    drop = resolve_threshold(WORKLOAD_SHARE_DROP, context.thresholds)[0]
    if upper >= shared_ref / n_ref - drop:
        return _verdict(PREFIX_SHARING_DROP, subject, ASSESSED, NOT_OBSERVED)
    statement = (
        f"{shared / n:.0%} of the subject's requests declared a shared prefix "
        f"(95% CI up to {upper:.0%}), against {shared_ref / n_ref:.0%} in the reference."
    )
    return _workload(
        context,
        subject,
        PREFIX_SHARING_DROP,
        "Fewer requests shared a prefix than in the reference",
        Observation(
            "o1",
            statement,
            "shared_prefix_share",
            round(shared / n, 4),
            n=n,
            n_ref=n_ref,
        ),
    )


# ----------------------------------------------------------------- helpers
def rate_ratio_interval(
    n: int, seconds: float, n_ref: int, seconds_ref: float
) -> tuple[float, float]:
    """An exact 95% interval on the ratio of two Poisson rates: given the
    total count, the subject's share is binomial, and its Clopper-Pearson
    interval maps onto the ratio."""
    low, high = _clopper_pearson(n, n + n_ref)
    scale = seconds_ref / seconds

    def ratio(p: float) -> float:
        return float("inf") if p >= 1 else p / (1 - p) * scale

    return ratio(low), ratio(high)


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    low = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(0.025, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(0.975, successes + 1, trials - successes))
    )
    return low, high


def _rates(
    context: Context, subject: Subject
) -> tuple[tuple[int, float], tuple[int, float]] | None:
    """(count, seconds) for the subject and its reference, by arrival."""
    spans = [_span(context, ids) for ids in (subject.requests, subject.reference)]
    if None in spans:
        return None
    if subject.start_ns is not None and subject.end_ns is not None:
        spans[0] = (len(subject.requests), (subject.end_ns - subject.start_ns) / 1e9)
    (n, seconds), (n_ref, seconds_ref) = spans  # type: ignore[misc]
    if seconds <= 0 or seconds_ref <= 0 or n_ref == 0:
        return None
    return (n, seconds), (n_ref, seconds_ref)


def _span(context: Context, request_ids: list[str]) -> tuple[int, float] | None:
    times = sorted(
        at
        for at in (assignment_ns(context.view.client[r]) for r in request_ids)
        if at is not None
    )
    if len(times) < 2:
        return None
    return len(times), (times[-1] - times[0]) / 1e9


@dataclass(frozen=True)
class LengthChange:
    """The median prompt or output length's change against the reference,
    and whether it is longer by the threshold's ratio."""

    excess: Difference
    reference: float
    passes: bool


def length_change(
    context: Context, subject: Subject, field: str
) -> LengthChange | None:
    """``field`` is ``prompt_tokens`` or ``output_tokens``; None without
    enough of either side."""
    arms = [
        [
            float(v)
            for v in (_raw(context, r).get(field) for r in ids)
            if isinstance(v, int)
        ]
        for ids in (subject.requests, subject.reference)
    ]
    excess = median_difference(arms[0], arms[1])
    if excess is None:
        return None
    reference = sorted(arms[1])[len(arms[1]) // 2]
    needed = resolve_threshold(WORKLOAD_LENGTH_RATIO, context.thresholds)[0]
    passes = excess.low > 0 and reference + excess.estimate >= needed * reference
    return LengthChange(excess, reference, passes)


def _lengths(
    context: Context, subject: Subject, kind: str, field: str, noun: str
) -> Assessment:
    if not subject.requests:
        return _verdict(kind, subject, UNSUPPORTED, NO_CLIENT_REQUESTS)
    change = length_change(context, subject, field)
    if change is None:
        return _verdict(kind, subject, UNSUPPORTED, NO_REFERENCE)
    if not change.passes:
        return _verdict(kind, subject, ASSESSED, NOT_OBSERVED)
    excess = change.excess
    statement = (
        f"The median of the subject's {noun} grew by {excess.estimate:.0f} tokens "
        f"(95% CI {excess.low:.0f} to {excess.high:.0f})."
    )
    return _workload(
        context,
        subject,
        kind,
        f"The workload's {noun} were longer than the reference's",
        Observation(
            "o1",
            statement,
            f"{field}_excess",
            round(excess.estimate, 3),
            (round(excess.low, 3), round(excess.high, 3)),
            excess.n,
            excess.n_ref,
        ),
    )


def _sharing(context: Context, request_ids: list[str]) -> tuple[int, int] | None:
    raws = [_raw(context, r) for r in request_ids]
    known = [raw for raw in raws if "shared_prefix_tokens" in raw]
    if not known:
        return None
    shared = sum(1 for raw in known if (raw.get("shared_prefix_tokens") or 0) > 0)
    return shared, len(known)


def _workload(
    context: Context, subject: Subject, kind: str, title: str, observation: Observation
) -> Assessment:
    finding = Finding(
        kind=kind,
        component=COMPONENT_WORKLOAD,
        subject=subject.as_dict(),
        title=title,
        message=observation.statement,
        condition=met(direct_evidence=True, sufficient_samples=True),
        contribution=NOT_DETERMINED,
        observations=[observation],
        location={"component": COMPONENT_WORKLOAD},
        window=context.window(subject),
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        metrics={observation.metric: observation.value},
        experiment={
            "change": "rerun the reference's workload against the same server",
            "prediction": "the latency excess falls by what the demand explains",
        },
    )
    finding.support = [
        line for r in subject.requests for line in context.view.client[r].lines()[-1:]
    ]
    finding.display = finding.support[:8]
    return Assessment(kind, subject.key, ASSESSED, [], [finding])


def _verdict(kind: str, subject: Subject, status: str, reason: str) -> Assessment:
    return Assessment(kind, subject.key, status, [reason])


def _raw(context: Context, request_id: str) -> dict[str, Any]:
    return context.view.client[request_id].terminal_raw


__all__ = [
    "LengthChange",
    "RateRatio",
    "arrival_ratio",
    "assess_inputs",
    "assess_load",
    "assess_outputs",
    "assess_sharing",
    "length_change",
    "rate_ratio_interval",
]
