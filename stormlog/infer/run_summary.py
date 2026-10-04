"""One run, summarized for comparison with other runs.

A comparison never trusts the summary a run wrote about itself: it reads
the artifact's raw records again, through the same analysis ``infer
analyze`` does, and adds what a comparison needs on top:

- the run's labels (experiment, arm, block), which pair it with others;
- its comparable fields and their provenance (``compatibility``);
- its observers' states;
- the protocol failures that exclude it from a comparison, each with its
  reason, and the outcome failures that never do.

A protocol failure is a fault of the measurement, not of the server under
test: a case's cohort is invalid, the server's identity changed during the
run, the before description is of another server, a required cache reset
was not acknowledged, the server probe did not complete, or an experiment
runner recorded an external cause (``infer.run_state`` with state
``protocol_failure``: a server that never became healthy, a failed
prelude, a preemption). A run that did not finish is an outcome, like
failed requests, a server that served nothing, or an outcome the runner
recorded (state ``outcome_failure``: a server that exited, a step that
failed): the treatment may have caused it, so it is compared, never set
aside, unless an external cause is recorded. Outcome beats protocol:
without an external cause, such a run's other run-level faults, and a
cohort an unfinished run cut short, are outcomes too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from .analysis import analyze_inference_events
from .compatibility import RunField, run_fields
from .errors import InferInputError
from .populations import Membership, Segment
from .slo import SloSpec

COMPLETED = "completed"
SEGMENT_SEPARATOR = "/"
# Written by an experiment runner: how the run ended, and why.
RUN_STATE_EVENT = "infer.run_state"
PROTOCOL_FAILURE = "protocol_failure"
OUTCOME_FAILURE = "outcome_failure"
# Cohort issues an unfinished run leaves behind: outcomes, not faults.
_CUT_SHORT = ("phase_window_missing", "records_missing")


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
    started_at_ns: int | None = None
    # Kept as data: what went wrong with the run that the treatment may cause.
    outcome_failures: tuple[str, ...] = ()

    @property
    def cases(self) -> Mapping[str, Mapping[str, Any]]:
        cases = self.report.get("cases")
        return cases if isinstance(cases, Mapping) else {}

    @cached_property
    def comparable_cases(self) -> dict[str, Mapping[str, Any]]:
        """Every case, and each case's segments as ``case/segment``."""
        found: dict[str, Mapping[str, Any]] = {}
        for case_id, case in self.cases.items():
            found[str(case_id)] = case
            for name, segment in (case.get("segments") or {}).items():
                found[f"{case_id}{SEGMENT_SEPARATOR}{name}"] = segment
        return found

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
        """Why this run cannot stand for ``case_id``: the run's and the case's.

        A segment fails with its case.
        """
        case = case_id.split(SEGMENT_SEPARATOR, 1)[0]
        return self.protocol_failures + tuple(self.case_failures.get(case, ()))


def summarize_run(
    path: str | Path,
    *,
    slo: SloSpec | None = None,
    slo_source: str = "flags",
    span_paths: Iterable[str | Path] = (),
    segments: Sequence[Segment] = (),
    membership: Membership = "arrival",
) -> RunSummary:
    """Summarize one artifact; InferInputError when it cannot be read."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise InferInputError(f"{source}: {exc}") from exc
    report = analyze_inference_events(
        source,
        vllm_span_paths=span_paths,
        slo=slo,
        slo_source=slo_source,
        segments=segments,
        segment_membership=membership,
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
    protocol, outcomes = _run_failures(records, report, status)
    return RunSummary(
        path=path,
        sha256=sha256,
        run_id=context.get("run_id"),
        session_id=context.get("session_id"),
        session_status=status,
        labels=_labels(records),
        fields=run_fields(records),
        report=report,
        protocol_failures=tuple(protocol),
        case_failures=_case_failures(report, finished=status == COMPLETED),
        started_at_ns=_started_at(records),
        outcome_failures=tuple(outcomes),
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


def _started_at(records: Sequence[Mapping[str, Any]]) -> int | None:
    """When the run started: its first session record's wall time."""
    for record in records:
        value = record.get("timestamp_ns")
        if record.get("event_type") == "infer.session" and isinstance(value, int):
            return value
    return None


def _labels(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for record in records:
        config = record.get("config")
        if record.get("event_type") == "infer.session" and isinstance(config, Mapping):
            labels = config.get("labels")
            return dict(labels) if isinstance(labels, Mapping) else {}
    return {}


def _run_failures(
    records: Sequence[Mapping[str, Any]], report: Mapping[str, Any], status: str | None
) -> tuple[list[str], list[str]]:
    """The run's protocol failures, which set it aside, and its outcomes.

    Outcome beats protocol: in a run that did not finish, or that a runner
    recorded as an outcome failure, with no external cause recorded, the
    run's faults are outcomes too, since the treatment that stopped the
    server may also have cut its probe short or restarted it with a new
    identity.
    """
    protocol: list[str] = []
    manifest = report.get("manifest") or {}
    if manifest.get("protocol_failure"):
        protocol.append(str(manifest["protocol_failure"]))
    if any(
        r.get("event_type") == "infer.server_probe" and r.get("incomplete")
        for r in records
    ):
        protocol.append("probe_incomplete")
    external, runner_outcomes = _runner_state(records)
    outcomes = [] if status == COMPLETED else [f"session_{status or 'unknown'}"]
    outcomes += runner_outcomes
    if not outcomes or external:
        return protocol + external, outcomes
    return [], outcomes + protocol


def _runner_state(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """How a runner says the run ended: an external cause, which a set-aside
    needs, or an outcome it recorded, such as a server that exited."""
    state = next(
        (r for r in reversed(records) if r.get("event_type") == RUN_STATE_EVENT),
        None,
    )
    if state is None:
        return [], []
    reasons = [str(reason) for reason in state.get("reasons") or ["unstated"]]
    if state.get("state") == PROTOCOL_FAILURE:
        return [f"external:{reason}" for reason in reasons], []
    if state.get("state") == OUTCOME_FAILURE:
        return [], [f"runner:{reason}" for reason in reasons]
    return [], []


def _case_failures(
    report: Mapping[str, Any], *, finished: bool
) -> dict[str, tuple[str, ...]]:
    failures: dict[str, tuple[str, ...]] = {}
    for case_id, case in (report.get("cases") or {}).items():
        reasons = []
        if _cohort_invalid(case.get("population") or {}, finished=finished):
            reasons.append("cohort_invalid")
        cache = case.get("cache") or {}
        if cache.get("requested") == "cold" and cache.get("acknowledged") is False:
            reasons.append("cache_reset_not_acknowledged")
        if reasons:
            failures[str(case_id)] = tuple(reasons)
    return failures


def _cohort_invalid(population: Mapping[str, Any], *, finished: bool) -> bool:
    """An invalid cohort, unless an unfinished run cut it short: an outcome."""
    if population.get("cohort_valid", True):
        return False
    return finished or not _cut_short(population)


def _cut_short(population: Mapping[str, Any]) -> bool:
    """Every hard cohort issue is what an unfinished run leaves behind."""
    issues = [
        str(issue)
        for issue in population.get("issues") or []
        if not str(issue).startswith(("request_index_unrecorded", "abandoned"))
    ]
    return bool(issues) and all(issue.startswith(_CUT_SHORT) for issue in issues)


__all__ = [
    "COMPLETED",
    "RUN_STATE_EVENT",
    "SEGMENT_SEPARATOR",
    "RunSummary",
    "summarize_run",
    "summary_from_records",
]
