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
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for trial in materialized:
        groups[_group_key(trial)].append(trial)
    summaries = {
        f"{configuration_id}:{workload_id}:{mode}": _summarize_group(
            group, bootstrap_samples=bootstrap_samples
        )
        for (configuration_id, workload_id, mode), group in sorted(groups.items())
    }
    return {
        "schema_version": 2,
        "artifact_kind": "native_probe_analysis",
        "trial_count": len(materialized),
        "groups": summaries,
    }


def paired_perturbations(
    trials: Iterable[Mapping[str, Any]], metric: str
) -> list[dict[str, Any]]:
    """Compare each measured mode with its matched profiler-off trial."""
    materialized = list(trials)
    seen: set[tuple[str, str, int, str]] = set()
    for trial in materialized:
        identity = (*_pair_key(trial), _mode(trial))
        if identity in seen:
            raise ValueError(f"duplicate paired trial identity: {identity}")
        seen.add(identity)
        _trial_metric(trial, metric)
    baselines: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for trial in materialized:
        if trial.get("mode") == "off":
            baseline_key = _pair_key(trial)
            if baseline_key in baselines:
                raise ValueError(f"duplicate baseline trial identity: {baseline_key}")
            baselines[baseline_key] = trial
    comparisons: list[dict[str, Any]] = []
    for trial in materialized:
        if trial.get("mode") == "off":
            continue
        baseline = baselines.get(_pair_key(trial))
        if baseline is None:
            comparisons.append(
                _unavailable_comparison(trial, "matched off trial missing")
            )
            continue
        comparisons.append(_paired_delta(baseline, trial, metric))
    return comparisons


def _pair_key(trial: Mapping[str, Any]) -> tuple[str, str, int]:
    configuration = trial.get("configuration_id")
    workload = trial.get("workload_id")
    repetition = trial.get("repetition")
    if not isinstance(configuration, str) or not isinstance(workload, str):
        raise ValueError("paired trials require configuration_id and workload_id")
    if isinstance(repetition, bool) or not isinstance(repetition, int):
        raise ValueError("paired trials require an integer repetition")
    return configuration, workload, repetition


def _paired_delta(
    baseline: Mapping[str, Any], trial: Mapping[str, Any], metric: str
) -> dict[str, Any]:
    baseline_value = _trial_metric(baseline, metric)
    profiled_value = _trial_metric(trial, metric)
    if baseline_value is None or profiled_value is None:
        return _unavailable_comparison(
            trial, f"{metric} denominator or value unavailable"
        )
    if baseline_value == 0:
        return _unavailable_comparison(trial, f"{metric} baseline denominator is zero")
    delta = 100 * (profiled_value - baseline_value) / baseline_value
    return {
        "trial_id": _trial_id(trial),
        "baseline_trial_id": _trial_id(baseline),
        "metric": metric,
        "baseline": baseline_value,
        "profiled": profiled_value,
        "percent_delta": delta,
        "status": "measured",
    }


def _trial_metric(trial: Mapping[str, Any], metric: str) -> float | None:
    metrics = trial.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("trial metrics must be an object")
    value = metrics.get(metric)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {metric!r} must be numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"metric {metric!r} must be finite")
    return numeric


def _unavailable_comparison(trial: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "trial_id": _trial_id(trial),
        "baseline_trial_id": None,
        "status": "unknown",
        "reason": reason,
    }


def _group_key(trial: Mapping[str, Any]) -> tuple[str, str, str]:
    configuration_id = trial.get("configuration_id")
    mode = trial.get("mode")
    workload_id = trial.get("workload_id")
    if not isinstance(configuration_id, str) or not configuration_id:
        raise ValueError("trial configuration_id must be a nonempty string")
    if not isinstance(mode, str) or not mode:
        raise ValueError("trial mode must be a nonempty string")
    if not isinstance(workload_id, str) or not workload_id:
        raise ValueError("trial workload_id must be a nonempty string")
    return configuration_id, workload_id, mode


def _mode(trial: Mapping[str, Any]) -> str:
    mode = trial.get("mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError("paired trials require a nonempty mode")
    return mode


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
