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


def metric_unit(pattern: str) -> str | None:
    """The effect unit of the metrics a name or pattern matches, if it is one."""
    names = list(_FRACTION_METRICS) + [
        "goodput_rps",
        "throughput_rps",
        "output_tps",
        *(f"{key}.{level}" for key in LATENCY_KEYS for level in LEVELS),
    ]
    matched = [name for name in names if fnmatch.fnmatchcase(name, pattern)]
    units = {"fraction" if name in _FRACTION_METRICS else RELATIVE for name in matched}
    return units.pop() if len(units) == 1 else None


def _goodput(case: Mapping[str, Any]) -> Reading:
    slo = case.get("slo")
    if not isinstance(slo, Mapping):
        return None, NO_SLO
    if slo.get("status") != "evaluated":
        return None, UNMEASURABLE
    return _bounds(slo.get("goodput_lower_rps"), slo.get("goodput_upper_rps")), None


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
        return (float(value) if is_number(value) else None), None

    return read


def _failure_fraction(case: Mapping[str, Any]) -> Reading:
    population = case.get("population") or {}
    offered, successful = population.get("offered"), population.get("successful")
    if not (is_number(offered) and is_number(successful)) or offered <= 0:
        return None, None
    return (offered - successful) / offered, None


def _latency(key: str, level: str) -> Callable[[Mapping[str, Any]], Reading]:
    def read(case: Mapping[str, Any]) -> Reading:
        metric = ((case.get("latency") or {}).get("metrics") or {}).get(key) or {}
        estimate = (metric.get("failure_penalized") or {}).get(level) or {}
        if estimate.get("penalized"):
            return None, PENALIZED
        if estimate.get("sufficient") is False:
            value = estimate.get("value_ms")
            return (float(value) if is_number(value) else None), INSUFFICIENT_TAIL
        value = estimate.get("value_ms")
        return (float(value) if is_number(value) else None), None

    return read


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
    "metric_unit",
]
