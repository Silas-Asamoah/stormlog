"""Read a diagnosis report: its text view, and the records behind a finding.

The text view gives the verdict, the coverage of every kind, and each
finding with its observations, confidence, competitors and experiment, and
where its records are. ``inspect`` never diagnoses again: it reads a saved
report and prints the records a finding rests on, by physical line. When the
artifact changed since the report (its SHA-256 differs), each record is
found again by its ID and its line's hash is checked; a finding whose
support was kept only as line ranges cannot be found again, and says so.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .diagnosis_inputs import InputFile, Line, read_input

UNRESOLVABLE = "support_unresolvable_after_modification"


class InspectError(ValueError):
    """The report or its source cannot answer the question."""


# -------------------------------------------------------------- text view
def render_text(report: dict[str, Any]) -> str:
    payload = report.get("payload") or {}
    lines = [
        f"Diagnosis: {report['verdict']['summary']}",
        f"Outcome: {payload.get('outcome')}",
        "",
        "Coverage:",
    ]
    for kind, entry in sorted((payload.get("coverage") or {}).items()):
        reasons = ", ".join(entry.get("reasons") or [])
        lines.append(f"  {kind:<28} {entry['status']:<12} {reasons}".rstrip())
    details = payload.get("findings_detail") or {}
    for finding in report.get("findings") or []:
        lines.extend(["", *_finding_text(details.get(finding["id"]) or {}, finding)])
    return "\n".join(lines) + "\n"


def _finding_text(detail: dict[str, Any], finding: dict[str, Any]) -> list[str]:
    return [
        *_heading(detail, finding),
        *(f"    - {o['statement']}" for o in detail.get("observations") or []),
        *(_competitor(a) for a in detail.get("alternatives") or []),
        *_experiment(detail.get("experiment") or {}),
        *_pointers(detail),
    ]


def _heading(detail: dict[str, Any], finding: dict[str, Any]) -> list[str]:
    confidence = detail.get("confidence") or {}
    condition = (confidence.get("condition") or {}).get("level")
    contribution = (confidence.get("contribution") or {}).get("level") or (
        "not assessed"
    )
    component = (detail.get("location") or {}).get("component")
    lines = [
        f"[{detail.get('rank')}] {finding['severity'].upper()} {finding['kind']} "
        f"at {component}: {finding['title']}",
        f"    id {finding['id']}; claim {detail.get('claim')}, cause "
        f"{detail.get('cause')}, role {detail.get('role')}, driver {detail.get('driver')}",
        f"    confidence {confidence.get('level')} (condition {condition}, "
        f"contribution {contribution})",
    ]
    failed = (detail.get("eligibility") or {}).get("failed") or []
    if failed:
        lines.append(f"    not eligible for a fault claim: {', '.join(failed)}")
    return lines


def _competitor(alternative: dict[str, Any]) -> str:
    mark = " (indispensable)" if alternative.get("indispensable") else ""
    return (
        f"    competitor {alternative['kind']}{mark}: {alternative['status']}, "
        f"{alternative['reason']}"
    )


def _experiment(experiment: dict[str, Any]) -> list[str]:
    if not experiment:
        return []
    return [
        f"    experiment: {experiment.get('change')}; "
        f"expect {experiment.get('prediction')}"
    ]


def _pointers(detail: dict[str, Any]) -> list[str]:
    lines = [
        f"    at {e['path']}:{e['line']} {e.get('record_id') or ''}".rstrip()
        for e in detail.get("evidence") or []
    ]
    lines.append(
        f"    {detail.get('evidence_shown')} of {detail.get('evidence_total')} "
        "supporting records shown; --inspect --all lists them"
    )
    return lines


# ---------------------------------------------------------------- inspect
def inspect(
    report: dict[str, Any], finding_id: str, report_path: Path, *, everything: bool
) -> Iterator[str]:
    """The records behind one finding, line by line, as text.

    Raises:
        InspectError: for an unknown finding, a missing source, or support
            that cannot be found again in a changed source.
    """
    payload = report.get("payload") or {}
    detail = (payload.get("findings_detail") or {}).get(finding_id)
    if detail is None:
        raise InspectError(f"no finding {finding_id} in the report")
    for described in payload.get("inputs") or []:
        source = _locate(Path(described["path"]), report_path)
        current = read_input(source)
        changed = current.sha256 != described["sha256"]
        if changed:
            yield f"warning: {source} changed since the diagnosis; records are found by their IDs"
        support = (detail.get("support") or {}).get(described["path"]) or {}
        wanted = _wanted(detail, support, everything)
        yield from _resolved(current, wanted, support, changed)


def _locate(path: Path, report_path: Path) -> Path:
    """The source as recorded, else relative to the report, else beside it:
    a report and its artifact moved together still resolve."""
    for candidate in (path, report_path.parent / path, report_path.parent / path.name):
        if candidate.is_file():
            return candidate
    raise InspectError(f"source {path} not found")


def _wanted(
    detail: dict[str, Any], support: dict[str, Any], everything: bool
) -> list[tuple[int, str | None, str]]:
    """The (line, record_id, sha256) triples to show."""
    if not everything:
        return [
            (e["line"], e.get("record_id"), e["sha256"])
            for e in detail.get("evidence") or []
        ]
    if support.get("support_identity") == "triples":
        return [
            (line, record_id, digest) for line, record_id, digest in support["lines"]
        ]
    return [
        (line, None, "")
        for start, end in support.get("ranges") or []
        for line in range(start, end + 1)
    ]


def _resolved(
    source: InputFile,
    wanted: list[tuple[int, str | None, str]],
    support: dict[str, Any],
    changed: bool,
) -> Iterator[str]:
    by_id = {line.record_id: line for line in source.lines if line.record_id}
    for number, record_id, digest in wanted:
        line = _find(source, by_id, number, record_id, digest, changed, support)
        yield f"{source.path}:{line.number} {line.record_id or ''} " + json.dumps(
            line.raw, sort_keys=True
        )


def _find(
    source: InputFile,
    by_id: dict[str, Line],
    number: int,
    record_id: str | None,
    digest: str,
    changed: bool,
    support: dict[str, Any],
) -> Line:
    if not changed:
        return source.lines[number]
    if not digest or support.get("support_identity") == "ranges_only":
        raise InspectError(UNRESOLVABLE)
    line = by_id.get(record_id or "")
    if line is None or line.sha256 != digest:
        raise InspectError(
            f"record {record_id} (line {number}) is not in the changed source"
        )
    return line


__all__ = ["UNRESOLVABLE", "InspectError", "inspect", "render_text"]
