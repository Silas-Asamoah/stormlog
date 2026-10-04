"""Which per-run values a comparison compares, and how to read them.

Each metric reads one value from a case of a run's report, or says why it
cannot: a latency quantile that falls among failed requests is
``penalized``, one with too few requests is ``insufficient_tail_samples``.
SLO attainment and goodput are read as ``(lower, upper)`` bounds when some
outcomes are unknown.

Latency quantiles are the ``failure_penalized`` estimand: every offered
request counts, failures ranked worst, so a candidate cannot improve its
p99 by failing its slow requests.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .comparison_stats import (
    DIFFERENCE,
    HIGHER_IS_BETTER,
    LOG_RATIO,
    LOWER_IS_BETTER,
    RELATIVE,
    RunValue,
)
from .report_stats import is_number

PENALIZED = "penalized"
INSUFFICIENT_TAIL = "insufficient_tail_samples"
UNMEASURABLE = "unmeasurable"
NO_SLO = "no_slo_policy"
RATE_UNAVAILABLE = "rate_unavailable"
NO_VALUE = "no_value"
LATENCY_KEYS = (
    "client.ttft",
    "client.e2e",
    "client.tpot",
    "client.e2e_from_intended",
    "server.ttft",
    "server.e2e",
)
LEVELS = ("p50", "p90", "p95", "p99")

Reading = tuple[RunValue, str | None]


@dataclass(frozen=True)
class MetricSpec:
    """One compared metric: its direction, scale, units and reader."""

    name: str
    direction: str
    scale: str
    unit: str
    value_unit: str | None
    read: Callable[[Mapping[str, Any]], Reading]
    slo: bool = False
    # A fraction's requests per run: its gate needs them.
    trials: Callable[[Mapping[str, Any]], int | None] | None = None


def default_metrics(case: Mapping[str, Any]) -> list[MetricSpec]:
    """The metrics a case's report can give, in a stable order."""
    metrics = [
        MetricSpec(
            "goodput_rps",
            HIGHER_IS_BETTER,
            LOG_RATIO,
            RELATIVE,
            "requests_per_second",
            _goodput,
            slo=True,
        ),
        MetricSpec(
            "attainment",
            HIGHER_IS_BETTER,
            DIFFERENCE,
            "fraction",
            "fraction",
            _attainment,
            slo=True,
            trials=_slo_offered,
        ),
        MetricSpec(
            "throughput_rps",
            HIGHER_IS_BETTER,
            LOG_RATIO,
            RELATIVE,
            "requests_per_second",
            _throughput("requests_per_second"),
        ),
        MetricSpec(
            "output_tps",
            HIGHER_IS_BETTER,
            LOG_RATIO,
            RELATIVE,
            "tokens_per_second",
            _throughput("output_tokens_per_second"),
        ),
        MetricSpec(
            "failure_fraction",
            LOWER_IS_BETTER,
            DIFFERENCE,
            "fraction",
            "fraction",
            _failure_fraction,
            trials=_offered,
        ),
    ]
    present = ((case.get("latency") or {}).get("metrics") or {}).keys()
    for key in LATENCY_KEYS:
        if key not in present:
            continue
        for level in LEVELS:
            metrics.append(
                MetricSpec(
                    f"{key}.{level}",
                    LOWER_IS_BETTER,
                    LOG_RATIO,
                    RELATIVE,
                    "milliseconds",
                    _latency(key, level),
                )
            )
    return metrics


# The effect unit of each named metric, for gates given by name or pattern.
_FRACTION_METRICS = ("attainment", "failure_fraction")


def vacuous_budget(pattern: str, budget: float) -> str | None:
    """Why a budget on the metrics a pattern matches could never fail, if so.

    A fraction above 1 is beyond every fraction, and a rate cannot fall by
    100% or more, so a relative budget of 1 on one cannot be exceeded.
    """
    matched = _matching(pattern)
    if budget > 1 and any(name in _FRACTION_METRICS for name in matched):
        return f"a fraction budget of {budget:g} is above 1: it can never fail"
    rates = sorted(name for name in matched if name in _RATE_METRICS)
    if budget >= 1 and rates:
        return (
            f"{', '.join(rates)} cannot fall by {budget:.0%}, so the budget can "
            "never fail"
        )
    return None


_RATE_METRICS = frozenset({"goodput_rps", "throughput_rps", "output_tps"})


def _matching(pattern: str) -> list[str]:
    names = list(_FRACTION_METRICS) + [
        *sorted(_RATE_METRICS),
        *(f"{key}.{level}" for key in LATENCY_KEYS for level in LEVELS),
    ]
    return [name for name in names if fnmatch.fnmatchcase(name, pattern)]


def metric_names(pattern: str) -> list[str]:
    """Every known metric name a name or pattern matches."""
    return _matching(pattern)


RATE_METRICS = _RATE_METRICS


def metric_unit(pattern: str) -> str | None:
    """The effect unit of the metrics a name or pattern matches, if it is one."""
    matched = _matching(pattern)
    units = {"fraction" if name in _FRACTION_METRICS else RELATIVE for name in matched}
    return units.pop() if len(units) == 1 else None


def _goodput(case: Mapping[str, Any]) -> Reading:
    slo = case.get("slo")
    if not isinstance(slo, Mapping):
        return None, NO_SLO
    if slo.get("status") != "evaluated":
        return None, UNMEASURABLE
    value = _bounds(slo.get("goodput_lower_rps"), slo.get("goodput_upper_rps"))
    return value, None if value is not None else _rate_reason(case)


def _attainment(case: Mapping[str, Any]) -> Reading:
    slo = case.get("slo")
    if not isinstance(slo, Mapping):
        return None, NO_SLO
    if slo.get("status") != "evaluated":
        return None, UNMEASURABLE
    return _bounds(slo.get("attainment_lower"), slo.get("attainment_upper")), None


def _throughput(key: str) -> Callable[[Mapping[str, Any]], Reading]:
    def read(case: Mapping[str, Any]) -> Reading:
        value = (case.get("throughput") or {}).get(key)
        if is_number(value):
            return float(value), None
        return None, _rate_reason(case)

    return read


def _rate_reason(case: Mapping[str, Any]) -> str:
    """Why a case has no rate: its interval's reason, or that it has none."""
    reason = (case.get("intervals") or {}).get("rate_reason")
    return str(reason) if reason else RATE_UNAVAILABLE


def _count(value: Any) -> int | None:
    return int(value) if is_number(value) and value > 0 else None


def _offered(case: Mapping[str, Any]) -> int | None:
    return _count((case.get("population") or {}).get("offered"))


def _slo_offered(case: Mapping[str, Any]) -> int | None:
    slo = case.get("slo")
    return _count(slo.get("offered")) if isinstance(slo, Mapping) else None


def _failure_fraction(case: Mapping[str, Any]) -> Reading:
    population = case.get("population") or {}
    offered, successful = population.get("offered"), population.get("successful")
    if not (is_number(offered) and is_number(successful)) or offered <= 0:
        return None, "population_unrecorded"
    return (offered - successful) / offered, None


def _latency(key: str, level: str) -> Callable[[Mapping[str, Any]], Reading]:
    def read(case: Mapping[str, Any]) -> Reading:
        metric = ((case.get("latency") or {}).get("metrics") or {}).get(key) or {}
        return _estimate((metric.get("failure_penalized") or {}).get(level) or {})

    return read


def _estimate(estimate: Mapping[str, Any]) -> Reading:
    """A quantile estimate's value, or why it cannot be gated."""
    if estimate.get("penalized"):
        return None, PENALIZED
    value = estimate.get("value_ms")
    if value == math.inf:
        # Worse than any value: the comparison fails it, never drops it.
        return math.inf, None
    number = float(value) if is_number(value) else None
    if estimate.get("sufficient") is False:
        return number, INSUFFICIENT_TAIL
    if number is not None:
        return number, None
    # Such as successful_values_missing: no value is never attrition.
    return None, str(estimate.get("reason") or NO_VALUE)


def _bounds(lower: Any, upper: Any) -> RunValue:
    if not (is_number(lower) and is_number(upper)):
        return None
    if lower == upper:
        return float(lower)
    return float(lower), float(upper)


def evidence_coverage(case: Mapping[str, Any]) -> float | None:
    """The share of a case's successful requests every SLO criterion judged."""
    slo = case.get("slo")
    value = slo.get("evidence_coverage") if isinstance(slo, Mapping) else None
    return float(value) if is_number(value) else None


__all__ = [
    "INSUFFICIENT_TAIL",
    "LATENCY_KEYS",
    "LEVELS",
    "NO_SLO",
    "PENALIZED",
    "UNMEASURABLE",
    "MetricSpec",
    "default_metrics",
    "evidence_coverage",
    "metric_names",
    "metric_unit",
]
