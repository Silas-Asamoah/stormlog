"""Denominator-safe analysis of native probe trial measurements."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .models import ResultStatus


def analyze_trials(
    trials: Iterable[Mapping[str, Any]], *, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    """Aggregate individual trials while retaining failures and unknown metrics."""
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    materialized = list(trials)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for trial in materialized:
        groups[_group_key(trial)].append(trial)
    summaries = {
        f"{configuration_id}:{mode}": _summarize_group(
            group, bootstrap_samples=bootstrap_samples
        )
        for (configuration_id, mode), group in sorted(groups.items())
    }
    return {
        "schema_version": 1,
        "artifact_kind": "native_probe_analysis",
        "trial_count": len(materialized),
        "groups": summaries,
    }


def _group_key(trial: Mapping[str, Any]) -> tuple[str, str]:
    configuration_id = trial.get("configuration_id")
    mode = trial.get("mode")
    if not isinstance(configuration_id, str) or not configuration_id:
        raise ValueError("trial configuration_id must be a nonempty string")
    if not isinstance(mode, str) or not mode:
        raise ValueError("trial mode must be a nonempty string")
    return configuration_id, mode


def _summarize_group(
    trials: Sequence[Mapping[str, Any]], *, bootstrap_samples: int
) -> dict[str, Any]:
    statuses = {status.value: 0 for status in ResultStatus}
    metrics: dict[str, list[float]] = defaultdict(list)
    unknown_metrics: dict[str, int] = defaultdict(int)
    trial_ids: list[str] = []
    for trial in trials:
        status = _status(trial)
        statuses[status.value] += 1
        trial_ids.append(_trial_id(trial))
        raw_metrics = trial.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("trial metrics must be an object")
        _collect_metrics(raw_metrics, metrics, unknown_metrics)
    return {
        "trial_ids": trial_ids,
        "trial_count": len(trials),
        "status_counts": statuses,
        "failure_count": sum(
            statuses[value]
            for value in ("fail", "partial", "unsupported", "untested", "timeout")
        ),
        "metrics": {
            name: _summarize_metric(
                values,
                seed=_stable_seed(name, trial_ids),
                bootstrap_samples=bootstrap_samples,
            )
            for name, values in sorted(metrics.items())
        },
        "unknown_metric_counts": dict(sorted(unknown_metrics.items())),
    }


def _collect_metrics(
    raw_metrics: Mapping[str, Any],
    metrics: dict[str, list[float]],
    unknown_metrics: dict[str, int],
) -> None:
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be nonempty strings")
        if value is None:
            unknown_metrics[name] += 1
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {name!r} must be numeric or null")
        elif not math.isfinite(float(value)):
            raise ValueError(f"metric {name!r} must be finite")
        else:
            metrics[name].append(float(value))


def _summarize_metric(
    values: Sequence[float], *, seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    ordered = sorted(values)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    lower, upper = _bootstrap_median_interval(
        ordered, seed=seed, sample_count=bootstrap_samples
    )
    return {
        "count": len(ordered),
        "individual_values": ordered,
        "median": median,
        "median_absolute_deviation": statistics.median(deviations),
        "bootstrap_median_95_ci": [lower, upper],
    }


def _bootstrap_median_interval(
    values: Sequence[float], *, seed: int, sample_count: int
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(sample_count)
    )
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _status(trial: Mapping[str, Any]) -> ResultStatus:
    raw_status = trial.get("status")
    try:
        return ResultStatus(raw_status)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid trial status: {raw_status!r}") from error


def _trial_id(trial: Mapping[str, Any]) -> str:
    value = trial.get("trial_id")
    if not isinstance(value, str) or not value:
        raise ValueError("trial_id must be a nonempty string")
    return value


def _stable_seed(metric: str, trial_ids: Sequence[str]) -> int:
    value = "\x00".join((metric, *trial_ids))
    return sum((index + 1) * ord(character) for index, character in enumerate(value))
