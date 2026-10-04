"""One run, summarized for comparison with other runs.

A comparison never trusts the summary a run wrote about itself: it reads
the artifact's raw records again, through the same analysis ``infer
analyze`` does, and adds what a comparison needs on top:

- the run's labels (experiment, arm, block), which pair it with others;
- its comparable fields and their provenance (``compatibility``);
- its observers' states;
- the protocol failures that exclude it from a comparison, each with its
  reason.

A protocol failure is a fault of the measurement, not of the server under
test: the run did not finish, a case's cohort is invalid, the server's
identity changed during the run, a required cache reset was not
acknowledged, or the server probe did not complete. Failed requests, or a
server that served nothing, are outcomes, never protocol failures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis import analyze_inference_events
from .compatibility import RunField, run_fields
from .errors import InferInputError
from .slo import SloSpec

COMPLETED = "completed"


@dataclass(frozen=True)
class RunSummary:
    """A run's report, labels, fields and protocol failures."""

    path: Path | None
    sha256: str | None
    run_id: str | None
    session_id: str | None
    session_status: str | None
    labels: Mapping[str, Any]
    fields: Mapping[str, RunField]
    report: Mapping[str, Any]
    protocol_failures: tuple[str, ...] = ()
    case_failures: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def cases(self) -> Mapping[str, Mapping[str, Any]]:
        cases = self.report.get("cases")
        return cases if isinstance(cases, Mapping) else {}

    @property
    def observers(self) -> Mapping[str, Any]:
        block = self.report.get("observers")
        return (
            (block or {}).get("observers") or {} if isinstance(block, Mapping) else {}
        )

    @property
    def name(self) -> str:
        return str(self.path) if self.path is not None else str(self.run_id)

    def label(self, key: str) -> Any:
        return self.labels.get(key)

    def failures_for(self, case_id: str) -> tuple[str, ...]:
        """Why this run cannot stand for ``case_id``: the run's and the case's."""
        return self.protocol_failures + tuple(self.case_failures.get(case_id, ()))


def summarize_run(
    path: str | Path,
    *,
    slo: SloSpec | None = None,
    slo_source: str = "flags",
    span_paths: Iterable[str | Path] = (),
) -> RunSummary:
    """Summarize one artifact; InferInputError when it cannot be read."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise InferInputError(f"{source}: {exc}") from exc
    report = analyze_inference_events(
        source, vllm_span_paths=span_paths, slo=slo, slo_source=slo_source
    )
    records = _records(raw, source)
    return summary_from_records(
        records, report, path=source, sha256=hashlib.sha256(raw).hexdigest()
    )


def summary_from_records(
    records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    path: Path | None = None,
    sha256: str | None = None,
) -> RunSummary:
    """A run summary from records already read and analyzed."""
    identity = next(
        (r for r in records if r.get("event_type") == "infer.artifact"), None
    )
    context = (identity or {}).get("context") or {}
    status = _session_status(records)
    return RunSummary(
        path=path,
        sha256=sha256,
        run_id=context.get("run_id"),
        session_id=context.get("session_id"),
        session_status=status,
        labels=_labels(records),
        fields=run_fields(records),
        report=report,
        protocol_failures=tuple(_run_failures(records, report, status)),
        case_failures=_case_failures(report),
    )


def _records(raw: bytes, source: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise InferInputError(f"{source}: {exc}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def _session_status(records: Sequence[Mapping[str, Any]]) -> str | None:
    """The last session record's status: completed, interrupted, incomplete."""
    statuses = [
        r.get("status") for r in records if r.get("event_type") == "infer.session"
    ]
    return str(statuses[-1]) if statuses else None


def _labels(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for record in records:
        config = record.get("config")
        if record.get("event_type") == "infer.session" and isinstance(config, Mapping):
            labels = config.get("labels")
            return dict(labels) if isinstance(labels, Mapping) else {}
    return {}


def _run_failures(
    records: Sequence[Mapping[str, Any]], report: Mapping[str, Any], status: str | None
) -> list[str]:
    failures = []
    if status != COMPLETED:
        failures.append(f"session_{status or 'unknown'}")
    manifest = report.get("manifest") or {}
    if manifest.get("protocol_failure"):
        failures.append(str(manifest["protocol_failure"]))
    if any(
        r.get("event_type") == "infer.server_probe" and r.get("incomplete")
        for r in records
    ):
        failures.append("probe_incomplete")
    return failures


def _case_failures(report: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    failures: dict[str, tuple[str, ...]] = {}
    for case_id, case in (report.get("cases") or {}).items():
        reasons = []
        if not (case.get("population") or {}).get("cohort_valid", True):
            reasons.append("cohort_invalid")
        cache = case.get("cache") or {}
        if cache.get("requested") == "cold" and cache.get("acknowledged") is False:
            reasons.append("cache_reset_not_acknowledged")
        if reasons:
            failures[str(case_id)] = tuple(reasons)
    return failures


__all__ = ["COMPLETED", "RunSummary", "summarize_run", "summary_from_records"]
