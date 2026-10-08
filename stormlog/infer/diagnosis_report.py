"""How a diagnosis finding is written: its payload detail, its envelope
finding with display pointers, and the input it cites."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..report import Evidence
from ..report import Finding as EnvelopeFinding
from .diagnosis_context import Context
from .diagnosis_inputs import INPUT_KIND, InputFile, Line
from .diagnosis_model import (
    DISPLAY_LIMIT,
    DRIVER_UNDETERMINED,
    Finding,
    support_block,
)
from .diagnosis_thresholds import THRESHOLDS_VERSION


def inputs_block(source: InputFile) -> dict[str, Any]:
    """The artifact's identity, so a reader can tell whether it changed."""
    return source.describe()


def finding_detail(
    finding_id: str, finding: Finding, rank: int, context: Context
) -> dict[str, Any]:
    """``payload.findings_detail[<id>]``: everything the finding claims and
    every record it rests on."""
    display = finding.display[:DISPLAY_LIMIT]
    subject = next(
        (s for s in context.subjects() if s.key == finding.subject.get("key")), None
    )
    return {
        "id": finding_id,
        "kind": finding.kind,
        "status": finding.status,
        "partial_reasons": list(finding.partial_reasons),
        "role": finding.role,
        "secondary_to": list(finding.secondary_to),
        "rank": rank,
        "severity": finding.severity,
        "cause": finding.cause,
        "claim": finding.claim,
        "driver": DRIVER_UNDETERMINED,
        "subject": finding.subject,
        "location": finding.location,
        "window": finding.window,
        "first_detectable_ns": finding.first_detectable_ns,
        "detection_evidence": (
            subject.detection_evidence() if subject is not None else None
        ),
        "segments": finding.segments,
        "observations": [o.as_dict() for o in finding.observations],
        "alternatives": [a.as_dict() for a in finding.alternatives],
        "eligibility": {
            "eligible": finding.eligible,
            "gates": dict(sorted(finding.gates.items())),
            "failed": finding.failed_gates,
            "contested": finding.contested,
        },
        "confidence": {
            "level": finding.confidence_level,
            "condition": finding.condition.as_dict(),
            "contribution": {
                **finding.contribution.as_dict(),
                **({"explains": finding.explains} if finding.explains else {}),
            },
            "driver": {"level": None, "met": [], "unmet": ["not_determined"]},
        },
        "evidence": [_pointer(line, context) for line in display],
        "evidence_shown": len(display),
        "evidence_total": len({line.number for line in finding.support}),
        "support": {str(context.view.source.path): support_block(finding.support)},
        "detail": finding.detail,
        "experiment": finding.experiment,
        "thresholds": {"version": THRESHOLDS_VERSION},
    }


def envelope_finding(
    finding_id: str, finding: Finding, context: Context, report_dir: Path | None
) -> EnvelopeFinding:
    """The envelope's finding: its verdict and up to eight pointers."""
    path = _relative(context.view.source.path, report_dir)
    return EnvelopeFinding(
        id=finding_id,
        kind=finding.kind,
        severity=finding.severity,
        title=finding.title,
        message=finding.message,
        metrics=dict(finding.metrics),
        evidence=[
            Evidence(
                kind=INPUT_KIND,
                path=path,
                pointer=f"/{line.number}",
                session_id=context.view.session_id,
                record_id=line.record_id,
                description=line.event_type,
            )
            for line in finding.display[:DISPLAY_LIMIT]
        ],
    )


def _pointer(line: Line, context: Context) -> dict[str, Any]:
    return {
        "kind": line.event_type,
        "path": str(context.view.source.path),
        "line": line.number,
        "record_id": line.record_id,
        "sha256": line.sha256,
    }


def _relative(path: Path, report_dir: Path | None) -> str:
    """A path relative to the report's directory, as the report contract
    asks, or as given when there is no report file."""
    if report_dir is None:
        return str(path)
    return os.path.relpath(path.resolve(), report_dir.resolve())


__all__ = ["envelope_finding", "finding_detail", "inputs_block"]
