"""Build the ``stormlog.report`` envelope for diagnose bundles.

The PyTorch, TensorFlow and JAX diagnose commands produce the same
``diagnostic_summary.json`` shape (``risk_flags``, ``suggestions`` and a few
memory counters), so one builder turns that summary into findings with
evidence pointers back into the bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .report import (
    REPORT_FILENAME,
    REPORT_FORMAT,
    REPORT_SCHEMA_VERSION,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    Artifact,
    Evidence,
    Finding,
    build_report,
)

DIAGNOSE_REPORT_KIND = "diagnose"
DIAGNOSE_MANIFEST_SCHEMA_VERSION = 2
SUMMARY_FILENAME = "diagnostic_summary.json"

# risk flag -> (finding kind, severity, title, summary metric for the flag)
_RISK_FLAG_FINDINGS: Mapping[str, tuple[str, str, str, str]] = {
    "oom_occurred": (
        "oom",
        SEVERITY_CRITICAL,
        "The allocator recorded out-of-memory events",
        "num_ooms",
    ),
    "high_utilization": (
        "high_utilization",
        SEVERITY_WARNING,
        "Device memory utilization is above the risk threshold",
        "utilization_ratio",
    ),
    "fragmentation_warning": (
        "fragmentation",
        SEVERITY_WARNING,
        "Allocator fragmentation is above the warning threshold",
        "fragmentation_ratio",
    ),
}

_METRIC_KEYS = (
    "allocated_bytes",
    "reserved_bytes",
    "peak_bytes",
    "total_bytes",
    "allocator_gap_bytes",
    "utilization_ratio",
    "fragmentation_ratio",
    "num_ooms",
)

_ARTIFACT_KINDS: Mapping[str, str] = {
    "manifest.json": "diagnose_manifest",
    SUMMARY_FILENAME: "diagnose_summary",
    "environment.json": "environment",
    "telemetry_timeline.json": "telemetry_timeline",
}


def build_diagnose_report(
    *,
    tool_name: str,
    summary: Mapping[str, Any],
    exit_code: int,
    session_id: str,
    files: Sequence[str],
    thresholds: Mapping[str, float] | None = None,
    tool_version: str | None = None,
) -> dict[str, Any]:
    """Return the report dict for one diagnose bundle.

    ``thresholds`` maps a risk flag to the threshold the command applied, so
    the finding can show the observed value next to it.
    """
    risk_flags = _risk_flags(summary)
    findings = [
        _finding(flag, summary, session_id, (thresholds or {}).get(flag))
        for flag, raised in risk_flags.items()
        if raised
    ]
    raised_flags = [finding.kind for finding in findings]
    if raised_flags:
        verdict_summary = "Memory risk detected: " + ", ".join(raised_flags)
    else:
        verdict_summary = "No memory risk detected"
    return build_report(
        report_kind=DIAGNOSE_REPORT_KIND,
        tool_name=tool_name,
        command="diagnose",
        exit_code=exit_code,
        summary=verdict_summary,
        findings=findings,
        metrics=_metrics(summary),
        artifacts=[_artifact(name) for name in files],
        recommendations=[str(item) for item in summary.get("suggestions", [])],
        session_id=session_id,
        payload={"backend": summary.get("backend"), "risk_flags": dict(risk_flags)},
        tool_version=tool_version,
    )


def _risk_flags(summary: Mapping[str, Any]) -> dict[str, bool]:
    raw = summary.get("risk_flags")
    if not isinstance(raw, Mapping):
        return {}
    return {str(flag): bool(raised) for flag, raised in raw.items()}


def _finding(
    flag: str,
    summary: Mapping[str, Any],
    session_id: str,
    threshold: float | None,
) -> Finding:
    kind, severity, title, metric_key = _RISK_FLAG_FINDINGS.get(
        flag, (flag, SEVERITY_WARNING, f"Risk flag {flag} is set", flag)
    )
    metrics: dict[str, float | int | None] = {}
    observed = summary.get(metric_key)
    if isinstance(observed, (int, float)) and not isinstance(observed, bool):
        metrics[metric_key] = observed
    if threshold is not None:
        metrics["threshold"] = threshold
    return Finding(
        id=f"{DIAGNOSE_REPORT_KIND}.{flag}",
        kind=kind,
        severity=severity,
        title=title,
        metrics=metrics,
        evidence=[
            Evidence(
                kind="diagnose_summary",
                path=SUMMARY_FILENAME,
                pointer=f"/risk_flags/{flag}",
            ),
            Evidence(kind="session", session_id=session_id),
        ],
    )


def _metrics(summary: Mapping[str, Any]) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for key in _METRIC_KEYS:
        if key not in summary:
            continue
        value = summary[key]
        if value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            metrics[key] = value
    return metrics


def _artifact(name: str) -> Artifact:
    if name == "manifest.json":
        return Artifact(
            kind=_ARTIFACT_KINDS[name],
            path=name,
            schema_version=DIAGNOSE_MANIFEST_SCHEMA_VERSION,
        )
    if name == REPORT_FILENAME:
        return Artifact(
            kind="report",
            path=name,
            format=REPORT_FORMAT,
            schema_version=REPORT_SCHEMA_VERSION,
        )
    return Artifact(kind=_ARTIFACT_KINDS.get(name, "file"), path=name)


__all__ = ["DIAGNOSE_REPORT_KIND", "build_diagnose_report"]
