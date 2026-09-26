"""Conservative normalization and fidelity gates for raw trial evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

LOSS_DOMAINS = (
    "vendor_activity",
    "bpf_transport",
    "profiler_export",
    "artifact_storage",
)


def classify_overlap(
    trusted: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare overlap only after the trusted trace demonstrates concurrency."""
    observed = trusted.get("overlap_observed_in_trusted_trace")
    if observed is not True:
        return {
            "status": "partial",
            "eligible": False,
            "reason": "trusted trace did not demonstrate overlap",
            "overlap_preserved_by_candidate": None,
        }
    interval = candidate.get("concurrent_interval_ns")
    preserved = (
        isinstance(interval, int) and not isinstance(interval, bool) and interval > 0
    )
    return {
        "status": "pass" if preserved else "fail",
        "eligible": True,
        "reason": "candidate compared with demonstrated trusted overlap",
        "overlap_preserved_by_candidate": preserved,
    }


def validate_measurement_window(window: Mapping[str, Any] | None) -> list[str]:
    """Return violations of the shared warmup/capture/flush boundary contract."""
    if window is None:
        return ["measurement window is missing"]
    required = (
        "range_id",
        "marker",
        "warmup_iterations",
        "measured_iterations",
        "host_started_ns",
        "host_finished_ns",
        "clock",
        "flush_completed",
    )
    errors = [
        f"measurement window missing {key}" for key in required if key not in window
    ]
    for field in ("range_id", "marker", "clock"):
        if not isinstance(window.get(field), str) or not window[field].strip():
            errors.append(f"measurement window {field} must be a nonempty string")
    for field, minimum in (("warmup_iterations", 0), ("measured_iterations", 1)):
        value = window.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f"measurement window {field} must be an integer >= {minimum}")
    started = window.get("host_started_ns")
    finished = window.get("host_finished_ns")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
    ):
        errors.append("measurement window timestamps must be integers")
    elif started >= finished:
        errors.append("measurement window timestamps are not ordered")
    if window.get("flush_completed") is not True:
        errors.append("collector flush did not complete")
    return errors


def normalize_loss(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep independent loss domains and never convert unknown into zero."""
    source = raw or {}
    domains: dict[str, Any] = {}
    complete = True
    for name in LOSS_DOMAINS:
        value = source.get(name)
        if not isinstance(value, Mapping):
            domains[name] = {
                "status": "unknown",
                "lost_records": None,
                "expected_records": None,
                "reason": "collector did not report this loss domain",
            }
            complete = False
            continue
        lost = value.get("lost_records")
        expected = value.get("expected_records")
        status = value.get("status", "unknown")
        if status != "reported" or lost is None or expected is None:
            complete = False
        if isinstance(lost, int) and lost > 0:
            complete = False
        domains[name] = dict(value)
    return {"complete": complete, "domains": domains}


def normalize_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Produce one analysis input while preserving unsupported semantics."""
    required = (
        "trial_id",
        "configuration_id",
        "workload_id",
        "mode",
        "repetition",
        "status",
        "metrics",
    )
    if any(key not in trial for key in required):
        raise ValueError("trial is missing required identity or result fields")
    if trial.get("status") == "pass" and trial.get("measurement_window") is None:
        # A process exit alone cannot establish that a measurement was usable.
        trial = {**trial, "status": "partial"}
    metrics = _mapping(trial.get("metrics"))
    if metrics is None:
        raise ValueError("trial metrics must be an object")
    for name, value in metrics.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"metric {name!r} must be finite or null")
    errors = validate_measurement_window(_mapping(trial.get("measurement_window")))
    artifacts = _artifact_rows(trial.get("artifacts"))
    validate_unique_artifacts(artifacts)
    missing = [
        row.get("artifact_id")
        for row in artifacts
        if row.get("status") in {"missing", "malformed"}
    ]
    loss = normalize_loss(_mapping(trial.get("loss")))
    status = trial.get("status", "fail")
    if status == "pass" and (errors or missing):
        status = "partial"
    return {
        "schema_version": 1,
        "artifact_kind": "native_probe_normalized_trial",
        "trial_id": trial["trial_id"],
        "configuration_id": trial["configuration_id"],
        "workload_id": trial["workload_id"],
        "mode": trial["mode"],
        "repetition": trial["repetition"],
        "status": status,
        "metrics": dict(metrics),
        "measurement_window": trial.get("measurement_window"),
        "loss": loss,
        "pressure_controls": dict(_mapping(trial.get("pressure_controls")) or {}),
        "raw_artifact_ids": _artifact_ids(artifacts, "present"),
        "limitations": [*trial.get("limitations", []), *errors],
    }


def validate_unique_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate artifact identities or paths before evidence promotion."""
    for field in ("artifact_id", "path"):
        values = [row.get(field) for row in artifacts if row.get(field) is not None]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate artifact {field}")


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _artifact_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _artifact_ids(artifacts: Sequence[Mapping[str, Any]], status: str) -> list[Any]:
    return [row.get("artifact_id") for row in artifacts if row.get("status") == status]
