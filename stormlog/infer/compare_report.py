"""``infer compare``'s report: the ``stormlog.report`` v1 envelope and text.

The envelope's ``report_kind`` is ``inference_comparison`` and its payload
is ``stormlog.infer.comparison`` v1. Each finding points at the metric it
is about with a JSON pointer into the payload. On stdout that pointer has
no ``path`` and resolves inside the report itself; a report written to a
file names the file in ``path``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..report import Evidence, Finding, build_report
from .compare import Comparison
from .comparison_stats import GateOutcome

REPORT_KIND = "inference_comparison"
_ID_UNSAFE = re.compile(r"[^a-z0-9_.]+")


def comparison_report(
    comparison: Comparison,
    *,
    exit_code: int,
    argv: Sequence[str] | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """The envelope for a finished comparison."""
    path = report_path.name if report_path is not None else None
    findings = [
        *_gate_findings(comparison, path),
        *_attainment_findings(comparison, path),
        *_exclusion_findings(comparison),
    ]
    if comparison.comparability.status == "unverified":
        findings.append(_unverified(comparison, path))
    return build_report(
        report_kind=REPORT_KIND,
        tool_name="stormlog",
        command="infer compare",
        exit_code=exit_code,
        summary=_summary(comparison, exit_code),
        findings=findings,
        metrics={
            "metrics_compared": sum(
                len(c["metrics"]) for c in comparison.cases.values()
            ),
            "gates": len(comparison.gates),
            "gates_failed": len(comparison.failed),
            "gates_not_evaluable": len(comparison.not_evaluable),
            "runs_excluded": len(comparison.excluded),
        },
        payload=comparison.to_payload(),
        argv=argv,
    )


def error_report(
    message: str, *, exit_code: int, argv: Sequence[str] | None = None
) -> dict[str, Any]:
    """The envelope when the runs could not be compared at all (exit 5)."""
    finding = Finding(
        id="inference.comparison.invalid_input",
        kind="invalid_input",
        severity="critical",
        title="The runs could not be compared",
        message=message,
    )
    return build_report(
        report_kind=REPORT_KIND,
        tool_name="stormlog",
        command="infer compare",
        exit_code=exit_code,
        summary=message,
        findings=[finding],
        argv=argv,
    )


def _summary(comparison: Comparison, exit_code: int) -> str:
    gates = len(comparison.gates)
    if not gates and comparison.spec.min_attainment is None:
        return f"{_metric_count(comparison)} metrics compared; no gate configured"
    failed, unknown = len(comparison.failed), len(comparison.not_evaluable)
    if exit_code == 0 and unknown:
        return (
            f"{unknown} of {gates} gates could not be evaluated (allowed); none failed"
        )
    if exit_code == 0:
        return f"every gate passed ({gates} gates)"
    return f"{failed} gate(s) failed and {unknown} could not be evaluated, of {gates}"


def _metric_count(comparison: Comparison) -> int:
    return sum(len(case["metrics"]) for case in comparison.cases.values())


def _pointer_evidence(pointer: str, path: str | None, description: str) -> Evidence:
    return Evidence(
        kind="comparison_metric", path=path, pointer=pointer, description=description
    )


def _gate_findings(comparison: Comparison, path: str | None) -> list[Finding]:
    findings = []
    for case_id, name, gate in comparison.gates:
        if gate.status == "pass":
            continue
        failed = gate.status == "fail"
        where = "absent_gates" if gate.reason == "metric_absent" else "metrics"
        pointer = f"/payload/cases/{_escape(case_id)}/{where}/{_escape(name)}"
        findings.append(
            Finding(
                id=_finding_id(
                    "regression" if failed else "not_evaluable", case_id, name
                ),
                kind="regression" if failed else "not_evaluable",
                severity="critical" if failed else "warning",
                title=(
                    f"{name} in {case_id} failed its {gate.rule.rule} gate"
                    if failed
                    else f"{name} in {case_id} could not be gated"
                ),
                message=_gate_message(gate),
                evidence=[_pointer_evidence(pointer, path, "the metric's comparison")],
            )
        )
    return findings


def _gate_message(gate: GateOutcome) -> str | None:
    """A gate's reason, and for a fraction the claim about runs it rests on."""
    statement = (gate.claim or {}).get("statement")
    if statement is None:
        return gate.reason
    return f"{gate.reason}: {statement}" if gate.reason else statement


def _attainment_findings(comparison: Comparison, path: str | None) -> list[Finding]:
    findings = []
    for case_id, case in comparison.cases.items():
        gate = case.get("attainment_gate") or {}
        if gate.get("status") != "fail":
            continue
        pointer = f"/payload/cases/{_escape(case_id)}/attainment_gate"
        findings.append(
            Finding(
                id=_finding_id("attainment", case_id),
                kind="attainment",
                severity="critical",
                title=f"too few candidate runs in {case_id} met the attainment target",
                metrics={
                    "runs_meeting": gate.get("runs_meeting"),
                    "runs": gate.get("runs"),
                },
                evidence=[_pointer_evidence(pointer, path, "the run-level gate")],
            )
        )
    return findings


def _exclusion_findings(comparison: Comparison) -> list[Finding]:
    return [
        Finding(
            id=_finding_id("excluded_run", item["case"], str(index)),
            kind="excluded_run",
            severity="warning",
            title=f"a {item['arm']} run was set aside for {item['case']}",
            message=f"{item['run']}: {_set_aside_reasons(item)}",
        )
        for index, item in enumerate(comparison.excluded)
    ]


def _set_aside_reasons(item: Mapping[str, Any]) -> str:
    """Why a run was set aside, the evidence for an external cause, and the
    attempt kept in its place."""
    text = ", ".join(item["reasons"])
    evidence = item.get("evidence") or {}
    if evidence:
        text += "; evidence: " + "; ".join(evidence.values())
    kept = item.get("attempt_kept")
    return text if kept is None else f"{text}; kept {kept}"


def _unverified(comparison: Comparison, path: str | None) -> Finding:
    names = sorted(item.name for item in comparison.comparability.unverified)
    return Finding(
        id="inference.comparison.unverified",
        kind="unverified",
        severity="warning",
        title="the runs could not be shown to measure the same server",
        message="unknown on a side: " + ", ".join(names),
        evidence=[
            _pointer_evidence("/payload/compatibility", path, "the comparability check")
        ],
    )


def _finding_id(kind: str, *parts: str) -> str:
    cleaned = [_ID_UNSAFE.sub("_", part.lower()).strip("_.") or "x" for part in parts]
    return ".".join(["inference.comparison", kind, *cleaned])


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def comparison_lines(comparison: Comparison) -> list[str]:
    """The text report: design, comparability, then each case's gated metrics."""
    lines = [
        "Inference Comparison",
        "--------------------",
        f"Design: {comparison.design}; mode {comparison.spec.mode}; "
        f"comparability {comparison.comparability.status}",
    ]
    lines += [
        f"Set aside: {item['run']} for {item['case']} ({_set_aside_reasons(item)})"
        for item in comparison.excluded
    ]
    lines += [f"Observer: {issue}" for issue in comparison.observer_issues]
    lines += [
        f"Warning: {warning}" for warning in comparison.diagnostics.get("warnings", [])
    ]
    for case_id, case in comparison.cases.items():
        ungated = (
            "" if case.get("gated", True) else " (overlap: diagnostics only, not gated)"
        )
        lines.append(f"- {case_id}{ungated}:")
        for name, metric in case["metrics"].items():
            lines.append("  " + _metric_line(name, metric))
        gate = case.get("attainment_gate")
        if gate:
            lines.append(
                f"  min attainment: {gate.get('status')} ({gate.get('runs_meeting')} of {gate.get('runs')} runs)"
            )
    return lines


def _metric_line(name: str, metric: Any) -> str:
    estimate = metric.worst
    if estimate is None or estimate.effect is None:
        shown = metric.reason or "no estimate"
    elif metric.scale == "log_ratio":
        shown = f"{estimate.effect:+.1%} [{estimate.lower:+.1%}, {estimate.upper:+.1%}]"
    else:
        shown = f"{estimate.effect:+.4g} [{estimate.lower:+.4g}, {estimate.upper:+.4g}] {metric.unit}"
        if metric.unit == "fraction":
            # Its gate is a claim about runs; the interval only describes.
            shown += " (descriptive)"
    gate = metric.gate
    judged = (
        ""
        if gate is None
        else f"; gate {gate.status}" + (f" ({gate.reason})" if gate.reason else "")
    )
    if gate is not None and gate.claim:
        judged += f": {gate.claim['statement']}"
    return f"{name}: {shown}; {metric.verdict.get('direction', '')}{judged}"


__all__ = ["REPORT_KIND", "comparison_lines", "comparison_report", "error_report"]
