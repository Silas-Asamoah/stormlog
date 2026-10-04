"""Latency quantiles, how much data they need, and what failures do to them.

A quantile from n i.i.d. observations has a distribution-free confidence
interval made of two order statistics (Le Boudec, *Performance Evaluation of
Computer and Communication Systems*, Theorem 2.1): ``[X(j), X(k)]`` covers the
p-quantile with probability ``B(k-1) - B(j-1)``, where B is the Binomial(n, p)
distribution function. The ranks come from n and p alone, never from the data.

``sufficient`` is the project-wide rule: the equal-tailed interval (each tail
at most alpha/2) exists with at least ``margin`` order statistics above its
upper rank, 5 by default. It certifies that statement and nothing more: not a
precision in milliseconds, and not coverage under the dependence queueing
creates between requests. ``n_min_exists`` is the smallest n for which any
interval exists at all (Le Boudec's own tables: 6 for the median, 59 for the
95th percentile).

Two estimands describe a case's latency:

- ``successful``: the latency of requests that succeeded.
- ``failure_penalized``: every offered request, with each one that did not
  succeed ranked worst. That is a policy penalty, not an observed latency.
  The p-quantile is the successful values' quantile at level p / (1 - f),
  where f is the share that did not succeed; above level 1 it falls in the
  failure mass and has no value. That is the estimand's definition, not an
  identity with interpolating over a sample padded with infinite values: it
  stays finite at level 1, and otherwise differs from that by less than one
  gap between order statistics. When a successful request has no value the
  level cannot be computed, and the estimate has none.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from scipy import stats

Tails = Literal["symmetric", "narrowest"]
Estimand = Literal["successful", "failure_penalized"]

DEFAULT_CONFIDENCE = 0.95
DEFAULT_MARGIN = 5
ASSUMPTIONS = (
    "requests are independent and identically distributed, with continuous "
    "latencies; the dependence queueing creates between requests is not "
    "covered, and sufficiency is no precision in milliseconds"
)
# Beyond this the minimum is searched in steps, then refined.
_SEARCH_LIMIT = 10_000_000


@dataclass(frozen=True)
class SufficiencyRule:
    """Which order-statistic interval must exist for a quantile to count."""

    confidence: float = DEFAULT_CONFIDENCE
    margin: int = DEFAULT_MARGIN
    tails: Tails = "symmetric"

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")

    def to_record(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "margin": self.margin,
            "tails": self.tails,
            "assumptions": ASSUMPTIONS,
        }


@dataclass(frozen=True)
class OrderStatisticInterval:
    """``[X(lower_rank), X(upper_rank)]`` with 1-based ranks.

    ``coverage`` is what the ranks achieve for i.i.d. continuous data.
    ``upper_ms`` is None when the upper rank falls among penalized failures.
    """

    lower_rank: int
    upper_rank: int
    lower_ms: float
    upper_ms: float | None
    coverage: float

    @property
    def width_ms(self) -> float | None:
        return None if self.upper_ms is None else self.upper_ms - self.lower_ms


@dataclass(frozen=True)
class QuantileEstimate:
    """One quantile of one latency metric, with how far it can be trusted."""

    estimand: Estimand
    p: float
    value_ms: float | None
    n: int
    sufficient: bool
    n_min: int
    n_min_exists: int
    interval: OrderStatisticInterval | None
    penalized: bool = False
    observed_lower_bound_ms: float | None = None
    reason: str | None = None


def quantile(values: Sequence[float], p: float) -> float | None:
    """The p-quantile by linear interpolation between order statistics.

    The same rule as ``report_stats.percentile``, for any level in [0, 1].
    """
    if not values or not 0 <= p <= 1:
        return None
    ordered = sorted(values)
    rank = p * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def quantile_minimum_n(
    p: float,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    margin: int = DEFAULT_MARGIN,
    tails: Tails = "symmetric",
) -> int:
    """The smallest n for which the rule's interval exists.

    Symmetric, margin 5 (the default): 20, 230, 1,164 and 11,665 for p50,
    p95, p99 and p99.9. Narrowest, margin 0: Le Boudec's 6, 59, 299, 2,995.
    """
    _check_level(p)
    rule = SufficiencyRule(confidence=confidence, margin=margin, tails=tails)
    return _minimum_n(p, rule)


def quantile_interval(
    values: Sequence[float],
    p: float,
    rule: SufficiencyRule = SufficiencyRule(),
    *,
    penalized: int = 0,
) -> OrderStatisticInterval | None:
    """The rule's order-statistic interval, or None when it does not exist.

    ``penalized`` failures rank above every value, so an upper rank among
    them leaves the upper bound unknown.
    """
    _check_level(p)
    n = len(values) + penalized
    ranks = _ranks(n, p, rule)
    if ranks is None:
        return None
    lower, upper = ranks
    ordered = sorted(values)
    if lower > len(ordered):
        return None
    return OrderStatisticInterval(
        lower_rank=lower,
        upper_rank=upper,
        lower_ms=ordered[lower - 1],
        upper_ms=ordered[upper - 1] if upper <= len(ordered) else None,
        coverage=_coverage(n, p, lower, upper),
    )


def successful_quantile(
    values: Sequence[float], p: float, rule: SufficiencyRule = SufficiencyRule()
) -> QuantileEstimate:
    """A quantile of the successful requests' latencies."""
    return QuantileEstimate(
        estimand="successful",
        p=p,
        value_ms=quantile(values, p),
        n=len(values),
        sufficient=len(values) >= _minimum_n(p, rule),
        n_min=_minimum_n(p, rule),
        n_min_exists=_minimum_n(p, _EXISTS),
        interval=quantile_interval(values, p, rule),
    )


def penalized_quantile(
    successful: Sequence[float],
    failures: int,
    p: float,
    rule: SufficiencyRule = SufficiencyRule(),
    *,
    timeout_elapsed_ms: Sequence[float] | None = None,
    successful_missing: int = 0,
) -> QuantileEstimate:
    """A quantile over every offered request, failures ranked worst.

    ``successful_missing`` counts successful requests with no value. They
    belong to the offered count but cannot be ranked, so the estimate is
    then undefined (reason ``successful_values_missing``) rather than taken
    over a smaller cohort that inflates the share that failed.

    ``timeout_elapsed_ms`` gives the elapsed time of each failure when every
    failure was a timeout; only then does a quantile in the failure mass get
    an observed lower bound: the quantile had each timeout ended when it was
    abandoned. A request cancelled after 1 ms is no evidence of a 60 s
    latency.
    """
    offered = len(successful) + successful_missing + failures
    reason = _undefined_reason(len(successful), successful_missing, failures)
    if reason is not None:
        return QuantileEstimate(
            estimand="failure_penalized",
            p=p,
            value_ms=None,
            n=offered,
            sufficient=offered >= _minimum_n(p, rule),
            n_min=_minimum_n(p, rule),
            n_min_exists=_minimum_n(p, _EXISTS),
            interval=None,
            reason=reason,
        )
    level = p * offered / len(successful) if successful else math.inf
    in_failures = level > 1
    return QuantileEstimate(
        estimand="failure_penalized",
        p=p,
        value_ms=None if in_failures else quantile(successful, level),
        n=offered,
        sufficient=offered >= _minimum_n(p, rule),
        n_min=_minimum_n(p, rule),
        n_min_exists=_minimum_n(p, _EXISTS),
        interval=quantile_interval(successful, p, rule, penalized=failures),
        penalized=in_failures,
        observed_lower_bound_ms=(
            _observed_bound(successful, failures, p, timeout_elapsed_ms)
            if in_failures
            else None
        ),
    )


def _undefined_reason(successful: int, missing: int, failures: int) -> str | None:
    if missing:
        return "successful_values_missing"
    if not successful and not failures:
        return "no_values"
    return None


def _observed_bound(
    successful: Sequence[float],
    failures: int,
    p: float,
    timeout_elapsed_ms: Sequence[float] | None,
) -> float | None:
    """The quantile had every timeout ended when it was abandoned.

    A timed-out request took at least its elapsed time, and an order
    statistic never falls when a value rises, so this bounds the quantile
    from below under the report's own interpolation.
    """
    if not timeout_elapsed_ms or len(timeout_elapsed_ms) != failures:
        return None
    return quantile([*successful, *timeout_elapsed_ms], p)


# Le Boudec's own tables: the narrowest interval, nothing required above it.
_EXISTS = SufficiencyRule(margin=0, tails="narrowest")


def _check_level(p: float) -> None:
    if not 0 < p < 1:
        raise ValueError("the quantile level must be between 0 and 1")


@lru_cache(maxsize=256)
def _minimum_n(p: float, rule: SufficiencyRule) -> int:
    """Smallest n with an interval; found by doubling, then bisection.

    Existence is monotone in n for both constructions.
    """
    high = 2
    while _ranks(high, p, rule) is None:
        high *= 2
        if high > _SEARCH_LIMIT:
            raise ValueError(f"no n up to {_SEARCH_LIMIT:,} supports p={p}")
    low = high // 2
    while low + 1 < high:
        middle = (low + high) // 2
        if _ranks(middle, p, rule) is None:
            low = middle
        else:
            high = middle
    return high


def _ranks(n: int, p: float, rule: SufficiencyRule) -> tuple[int, int] | None:
    """1-based ranks ``(j, k)`` of the rule's interval for n, or None."""
    if n < 1:
        return None
    if rule.tails == "symmetric":
        return _symmetric_ranks(n, p, rule)
    return _narrowest_ranks(n, p, rule)


def _symmetric_ranks(n: int, p: float, rule: SufficiencyRule) -> tuple[int, int] | None:
    """Each tail at most alpha/2, with N ~ Binomial(n, p) below the quantile.

    j is the largest rank with P(N <= j - 1) <= alpha/2; k the smallest with
    P(N >= k) <= alpha/2.
    """
    alpha = (1 - rule.confidence) / 2
    below = int(stats.binom.ppf(alpha, n, p))
    while below >= 0 and stats.binom.cdf(below, n, p) > alpha:
        below -= 1
    while below + 1 <= n and stats.binom.cdf(below + 1, n, p) <= alpha:
        below += 1
    upper = int(stats.binom.ppf(1 - alpha, n, p)) + 1
    while upper > 1 and stats.binom.sf(upper - 2, n, p) <= alpha:
        upper -= 1
    while stats.binom.sf(upper - 1, n, p) > alpha:
        upper += 1
    lower = below + 1
    if lower < 1 or upper > n - rule.margin:
        return None
    return lower, upper


def _narrowest_ranks(n: int, p: float, rule: SufficiencyRule) -> tuple[int, int] | None:
    """The narrowest ranks reaching the confidence; ties go to the more central."""
    top = n - rule.margin
    if top < 1 or _coverage(n, p, 1, top) < rule.confidence:
        return None
    best: tuple[int, int] | None = None
    for upper in range(top, 0, -1):
        lower = _widest_lower(n, p, upper, rule.confidence)
        if lower is None:
            break
        if best is None or upper - lower < best[1] - best[0]:
            best = (lower, upper)
    return best


def _widest_lower(n: int, p: float, upper: int, confidence: float) -> int | None:
    """The largest j with coverage(j, upper) >= confidence, or None."""
    if _coverage(n, p, 1, upper) < confidence:
        return None
    low, high = 1, upper - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _coverage(n, p, middle, upper) >= confidence:
            low = middle
        else:
            high = middle - 1
    return low


def _coverage(n: int, p: float, lower: int, upper: int) -> float:
    return float(stats.binom.cdf(upper - 1, n, p) - stats.binom.cdf(lower - 1, n, p))


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MARGIN",
    "OrderStatisticInterval",
    "QuantileEstimate",
    "SufficiencyRule",
    "penalized_quantile",
    "quantile",
    "quantile_interval",
    "quantile_minimum_n",
    "successful_quantile",
]
