"""Diagnose an inference artifact: ``diagnose_artifact`` and its report.

The diagnosis reads one artifact, joins its evidence, chooses its subjects,
runs every class on every subject, grades and ranks the findings, and
returns a ``stormlog.report`` v1 report whose payload is
``stormlog.inference_diagnosis`` v1. It reads only the file it is given and
is deterministic: the same artifact and options give the same report, with
the generation time injectable for tests. See ``docs/inference_diagnosis.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exit_codes import ExitCode
from ..report import build_report, validate_report
from .diagnosis_client import (
    assess_api_server,
    assess_capture_pause,
    assess_client_admission,
)
from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import read_input
from .diagnosis_join import RunView, join
from .diagnosis_kv import assess_kv
from .diagnosis_memory import memory_ledger
from .diagnosis_model import SECONDARY, UPSTREAM, Finding, rank_findings
from .diagnosis_prefix import assess_prefix
from .diagnosis_queue import assess_queue
from .diagnosis_report import envelope_finding, finding_detail, inputs_block
from .diagnosis_selection import SelectionOptions, Subject, select
from .diagnosis_thresholds import DEFAULT_THRESHOLDS, THRESHOLDS_VERSION
from .diagnosis_vocabulary import (
    CAPTURE_PAUSE,
    CLIENT_ADMISSION,
    HOST_STALL,
    KINDS,
    KV_PREEMPTION_PRESSURE,
    LOAD_INCREASE,
    LONGER_INPUTS,
    LONGER_OUTPUTS,
    PREFIX_CACHE_LOSS,
    PREFIX_SHARING_DROP,
    QUEUE_SATURATION,
    WORKLOAD_KINDS,
)
from .diagnosis_workload import (
    assess_inputs,
    assess_load,
    assess_outputs,
    assess_sharing,
)
from .telemetry import TelemetrySample, load_telemetry

REPORT_KIND = "inference_diagnosis"
PAYLOAD_FORMAT = "stormlog.inference_diagnosis"
PAYLOAD_VERSION = 1
DIAGNOSER_VERSION = "1"
NOT_IMPLEMENTED = "not_assessed_by_this_version"

Assess = Callable[[Context, Subject], Assessment]

# The classes this version assesses, per kind.
CLASSES: dict[str, tuple[Assess, ...]] = {
    QUEUE_SATURATION: (assess_queue,),
    KV_PREEMPTION_PRESSURE: (assess_kv,),
    PREFIX_CACHE_LOSS: (assess_prefix,),
    CLIENT_ADMISSION: (assess_client_admission,),
    HOST_STALL: (assess_api_server,),
    CAPTURE_PAUSE: (assess_capture_pause,),
    LOAD_INCREASE: (assess_load,),
    LONGER_INPUTS: (assess_inputs,),
    LONGER_OUTPUTS: (assess_outputs,),
    PREFIX_SHARING_DROP: (assess_sharing,),
}
# Parts of a kind this version does not assess yet.
NOT_YET: dict[str, str] = {
    HOST_STALL: "engine_core_and_worker_not_assessed_by_this_version",
}


@dataclass(frozen=True)
class DiagnoseOptions:
    """What the caller asked for, and what it asserts."""

    request_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    thresholds: dict[str, float] = field(default_factory=dict)
    # The operator's statement that the scraped exporter is the engine whose
    # hook log was imported; without it metrics never witness hook findings.
    metrics_from_engine: bool = False
    generated_at_ns: int | None = None
    report_dir: Path | None = None  # evidence paths are relative to it
    # On-host collector artifacts (``infer collect-server``) for the memory
    # ledger: the only files read besides the artifact.
    server_telemetry: tuple[str, ...] = ()
    argv: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_ids": list(self.request_ids),
            "case_ids": list(self.case_ids),
            "thresholds": dict(sorted(self.thresholds.items())),
            "metrics_from_engine": self.metrics_from_engine,
            "server_telemetry": list(self.server_telemetry),
        }


def diagnose_artifact(
    artifact: str | Path,
    *,
    windows: Sequence[tuple[int, int]] | None = None,
    options: DiagnoseOptions | None = None,
) -> dict[str, Any]:
    """Diagnose ``artifact``; return a validated ``stormlog.report`` dict.

    ``windows`` are declared subjects, as (start_ns, end_ns) on the
    artifact's clock; without any declared subject the diagnosis selects
    incidents itself.

    Raises:
        OSError: when the artifact cannot be read.
        ValueError: for unknown or non-finite threshold overrides, or empty
            windows.
    """
    options = options or DiagnoseOptions()
    _check(options, windows)
    source = read_input(artifact)
    telemetry = [s for path in options.server_telemetry for s in load_telemetry(path)]
    view = join(source)
    selection = select(
        view,
        SelectionOptions(
            thresholds=dict(options.thresholds),
            windows=tuple((int(s), int(e)) for s, e in windows or ()),
            request_ids=options.request_ids,
            case_ids=options.case_ids,
        ),
    )
    context = Context(
        view,
        selection,
        thresholds=dict(options.thresholds),
        metrics_from_engine=options.metrics_from_engine,
    )
    assessments = [
        assess(context, subject)
        for subject in selection.subjects
        for assessors in CLASSES.values()
        for assess in assessors
    ]
    return _report(context, assessments, options, windows or (), telemetry)


def _check(options: DiagnoseOptions, windows: Sequence[tuple[int, int]] | None) -> None:
    unknown = sorted(set(options.thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise ValueError(f"unknown threshold keys: {', '.join(unknown)}")
    if not all(math.isfinite(value) for value in options.thresholds.values()):
        # A NaN is never exceeded: every finding it gates would vanish silently.
        raise ValueError("threshold overrides must be finite numbers")
    for start, end in windows or ():
        if not end > start:
            raise ValueError(f"window {start},{end} is empty")


# ------------------------------------------------------------------ report
def _report(
    context: Context,
    assessments: list[Assessment],
    options: DiagnoseOptions,
    windows: Sequence[tuple[int, int]],
    telemetry: Sequence[TelemetrySample],
) -> dict[str, Any]:
    view = context.view
    findings: list[Finding] = [f for a in assessments for f in a.findings]
    link_edges(findings, view.run_id)
    ranked = rank_findings(findings, view.run_id)
    details = {}
    for rank, (finding_id, finding) in enumerate(ranked, start=1):
        details[finding_id] = finding_detail(finding_id, finding, rank, context)
    warnings = sum(1 for _, f in ranked if f.severity == "warning")
    exit_code = ExitCode.FINDINGS if warnings else ExitCode.OK
    coverage = _coverage(context, assessments)
    outcome = _outcome(context, ranked)
    diagnoser = _diagnoser(options, windows)
    payload = {
        "format": PAYLOAD_FORMAT,
        "schema_version": PAYLOAD_VERSION,
        "diagnoser": diagnoser,
        "inputs": [inputs_block(view.source)],
        "outcome": outcome,
        "join": _join(context),
        "selection": context.selection.as_dict(),
        "coverage": coverage,
        "findings_detail": details,
        "memory": memory_ledger(context, telemetry),
        "thresholds": _thresholds(options),
    }
    report = build_report(
        report_kind=REPORT_KIND,
        tool_name="stormlog",
        command="infer diagnose",
        exit_code=int(exit_code),
        summary=_summary(ranked, coverage, outcome, context),
        findings=[
            envelope_finding(fid, f, context, options.report_dir) for fid, f in ranked
        ],
        metrics={
            "findings": len(ranked),
            "warning_findings": warnings,
            "subjects": len(context.selection.subjects),
        },
        session_id=view.session_id,
        run_id=view.run_id,
        payload=payload,
        argv=list(options.argv) if options.argv is not None else None,
        generated_at_utc=_utc(diagnoser["generated_at_ns"]),
    )
    validate_report(report)
    return report


# Causes that come before another kind's mechanism: (upstream, downstream).
# KV preemption leaves requests waiting to resume and keeps new ones out, so
# a queue on the same subject is its consequence.
EDGES: tuple[tuple[str, str], ...] = ((KV_PREEMPTION_PRESSURE, QUEUE_SATURATION),)


def link_edges(findings: Sequence[Finding], run_id: str | None) -> None:
    """Make a finding secondary to the fault upstream of it on the same
    subject: the downstream finding names the upstream kind as `upstream`
    among its competitors, and the upstream one is a warning. A secondary
    claims a condition, never the fault."""
    for upstream_kind, downstream_kind in EDGES:
        faults = {
            str(f.subject.get("key")): f.identity(run_id)
            for f in findings
            if f.kind == upstream_kind and f.severity == "warning"
        }
        for finding in findings:
            key = str(finding.subject.get("key"))
            if (
                finding.kind == downstream_kind
                and key in faults
                and _names_upstream(finding, upstream_kind)
            ):
                finding.role = SECONDARY
                finding.secondary_to = [faults[key]]


def _names_upstream(finding: Finding, kind: str) -> bool:
    return any(a.kind == kind and a.status == UPSTREAM for a in finding.alternatives)


def _diagnoser(
    options: DiagnoseOptions, windows: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """The version, a digest of everything that decides the result, and when."""
    config = {
        "thresholds_version": THRESHOLDS_VERSION,
        "thresholds": {**DEFAULT_THRESHOLDS, **options.thresholds},
        "options": options.as_dict(),
        "windows": [list(window) for window in windows],
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    generated = options.generated_at_ns
    return {
        "version": DIAGNOSER_VERSION,
        "config_digest": hashlib.sha256(canonical.encode()).hexdigest(),
        "generated_at_ns": generated if generated is not None else time.time_ns(),
        "metrics_from_engine": options.metrics_from_engine,
    }


def _thresholds(options: DiagnoseOptions) -> dict[str, Any]:
    return {
        "version": THRESHOLDS_VERSION,
        "overridden": sorted(options.thresholds),
        "values": {**DEFAULT_THRESHOLDS, **options.thresholds},
    }


def _join(context: Context) -> dict[str, Any]:
    view: RunView = context.view
    producers = sorted({e.producer for e in view.executions.values()})
    return {
        "client_requests": len(view.client),
        "dispatch_records": view.has_dispatch_records(),
        "first_content_records": view.has_first_content_records(),
        "engine_executions": len(view.executions),
        "engines": [
            {
                "epoch": epoch.epoch,
                "producer": epoch.producer,
                "state": epoch.state,
                "observes": epoch.observes,
            }
            for epoch in sorted(view.engines.values(), key=lambda e: e.epoch)
        ],
        "clocks": [context.clock(producer).describe() for producer in producers],
        "problems": list(view.source.problems[:20]),
    }


def _coverage(context: Context, assessments: list[Assessment]) -> dict[str, Any]:
    """Per kind: its status over the subjects, with reasons and the detail
    by subject; kinds this version does not assess say so."""
    coverage: dict[str, Any] = {}
    for kind in sorted(KINDS):
        if kind not in CLASSES:
            coverage[kind] = {
                "status": UNSUPPORTED,
                "reasons": [NOT_IMPLEMENTED],
                "by_subject": {},
            }
            continue
        coverage[kind] = _kind_coverage(
            kind, [a for a in assessments if a.kind == kind]
        )
    return coverage


def _kind_coverage(kind: str, mine: list[Assessment]) -> dict[str, Any]:
    statuses = {a.status for a in mine}
    reasons = {r for a in mine for r in a.reasons}
    if kind in NOT_YET:  # a part of the kind this version does not assess
        statuses.add(PARTIAL)
        reasons.add(NOT_YET[kind])
    return {
        "status": _overall(statuses),
        "reasons": sorted(reasons),
        "by_subject": {a.subject_key: a.as_dict() for a in mine},
    }


def _overall(statuses: set[str]) -> str:
    """Unsupported only when every subject was; partial when any was not
    fully assessed; assessed otherwise, including with no subjects."""
    if statuses == {UNSUPPORTED}:
        return UNSUPPORTED
    return PARTIAL if statuses - {ASSESSED} else ASSESSED


def _unexplained(context: Context, ranked: list[tuple[str, Finding]]) -> list[Subject]:
    """Incident subjects with no eligible mechanism or instrumentation
    finding: a change in demand alone says what drove it, not what slowed."""
    explained = {
        str(f.subject.get("key"))
        for _, f in ranked
        if f.eligible and f.kind not in WORKLOAD_KINDS
    }
    return [
        s for s in context.selection.subjects if s.incident and s.key not in explained
    ]


def _untested(context: Context) -> int:
    """Windows automatic selection abstained on: too few requests of their
    own, or too few in their reference. Declared subjects test no window."""
    selection = context.selection
    if not selection.tested:
        return 0
    return sum(1 for window in selection.windows if window.status is not None)


def _outcome(context: Context, ranked: list[tuple[str, Finding]]) -> str:
    """``inconclusive`` when an incident subject has no eligible
    explanation, or when automatic selection could test no window, so it
    ruled no incident out either; ``findings`` when there are any; else
    ``no_findings``."""
    selection = context.selection
    tested_none = selection.tested and _untested(context) == len(selection.windows)
    if _unexplained(context, ranked) or tested_none:
        return "inconclusive"
    return "findings" if ranked else "no_findings"


def _summary(
    ranked: list[tuple[str, Finding]],
    coverage: dict[str, Any],
    outcome: str,
    context: Context,
) -> str:
    warnings = sum(1 for _, f in ranked if f.severity == "warning")
    counts = (
        (warnings, "warning finding"),
        (len(ranked) - warnings, "info finding"),
        (len(_unexplained(context, ranked)), "incident unexplained"),
        (
            sum(1 for e in coverage.values() if e["status"] == UNSUPPORTED),
            "kind unsupported",
        ),
        (_untested(context), "window untested"),
    )
    return f"{outcome}: " + "; ".join(_plural(n, noun) for n, noun in counts)


def _plural(count: int, noun: str) -> str:
    """'1 warning finding', '2 warning findings', '2 incidents unexplained'."""
    if count == 1:
        return f"1 {noun}"
    head, _, tail = noun.partition(" ")
    if head in ("incident", "kind", "window"):
        return f"{count} {head}s {tail}"
    return f"{count} {noun}s"


def _utc(generated_at_ns: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated_at_ns / 1e9))


__all__ = [
    "CLASSES",
    "DIAGNOSER_VERSION",
    "PAYLOAD_FORMAT",
    "PAYLOAD_VERSION",
    "REPORT_KIND",
    "EDGES",
    "DiagnoseOptions",
    "diagnose_artifact",
    "link_edges",
]
