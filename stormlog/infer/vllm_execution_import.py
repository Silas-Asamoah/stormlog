"""Import a vLLM execution-hook log into an inference artifact.

The artifact's ``infer.artifact`` record supplies the run and session; its
``infer.request`` records supply the ``X-Request-Id`` values the reducer binds
to; its phase and trace windows let foreign-only iterations be placed; and the
entities it already holds, with the high-water marks of earlier imports, keep
a re-import from writing anything twice. The reduced records are appended
through ``append_inference_capture`` as an engine adapter's capture. The raw
log itself is never registered as an attachment: it names other clients'
requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..session import create_session_summary
from .correlation_capture import (
    CaptureCapabilities,
    EngineCapture,
    append_inference_capture,
)
from .correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CapabilityEvent,
    ClockAlignmentEvent,
    EntityRef,
    InferenceRecord,
    IterationEvent,
    LegacyInferenceRecord,
    MembershipEvent,
    RequestEvent,
    load_inference_artifact,
)
from .errors import InferInputError, InferUsageError
from .trace_import import artifact_run_identity
from .vllm_execution import (
    ReduceOptions,
    RunFacts,
    RunRequest,
    Window,
    reduce_execution_log,
)
from .vllm_execution_log import (
    FORMAT,
    STATE_ALIVE,
    EpochRead,
    LogRead,
    read_execution_log,
)

SOURCE = "stormlog.infer.import_execution"
SUPPORTED = ("iterations", "memberships", "requests", "clock_alignment")
_COLLECTED_BY = {
    "iterations": IterationEvent,
    "memberships": MembershipEvent,
    "requests": RequestEvent,
    "clock_alignment": ClockAlignmentEvent,
}


def import_execution_into_artifact(
    artifact: str | Path,
    directory: str | Path,
    *,
    raw_foreign_ids: bool = False,
    envelope_path: str | Path | None = None,
    now_ns: int | None = None,
) -> EngineCapture:
    """Append the log's new, final iterations to ``artifact``; return the capture.

    The capture's summary, recorded on the engine adapter's capability event,
    carries each epoch's high-water mark, which the next import starts from.
    """
    run_id, session_id = artifact_run_identity(artifact)
    records = _load_records(artifact)
    facts = run_facts_from_records(records, run_id, session_id)
    read = _read_log(directory, execution_high_water(records), now_ns)
    capture = reduce_to_capture(read, facts, ReduceOptions(raw_foreign_ids))
    _append_capture(artifact, run_id, session_id, capture, envelope_path)
    return capture


def record_failed_execution_import(
    artifact: str | Path,
    directory: str | Path,
    error: str,
    *,
    envelope_path: str | Path | None = None,
) -> EngineCapture:
    """Record that the log was configured but could not be imported.

    The engine adapter's capability event then says what was supported and
    that nothing was collected, with the error in its summary, so the
    analysis reports partial coverage rather than an absent component.
    """
    run_id, session_id = artifact_run_identity(artifact)
    capture = EngineCapture(
        capabilities=CaptureCapabilities(SUPPORTED, SUPPORTED, ()),
        summary={"execution": {"directory": str(directory), "failed": error}},
    )
    _append_capture(artifact, run_id, session_id, capture, envelope_path)
    return capture


def flush_execution_log(
    directory: str | Path,
    *,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.25,
    now_ns: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Ask every live epoch's writer to seal its open segment, and wait for
    proof that it did: the ``flush`` file gone, or a record written after the
    request (a heartbeat comes every second). Ended epochs need no flush.
    """
    read = read_execution_log(directory, now_ns=now_ns)
    waiting, errors = _request_flushes(read)
    requested = sorted(waiting)
    deadline = clock() + timeout_seconds
    while waiting and clock() < deadline:
        sleep(poll_seconds)
        for epoch in read_execution_log(directory, now_ns=now_ns).epochs:
            if epoch.epoch in waiting and _sealed(epoch, waiting[epoch.epoch]):
                del waiting[epoch.epoch]
    return {
        "requested": requested,
        "flushed": [name for name in requested if name not in waiting],
        "timed_out": sorted(waiting),
        "errors": errors,
    }


def _request_flushes(read: LogRead) -> tuple[dict[str, int], list[str]]:
    """Touch ``flush`` in every live epoch; return each one's last sequence."""
    waiting: dict[str, int] = {}
    errors: list[str] = []
    for epoch in read.epochs:
        if epoch.state != STATE_ALIVE:
            continue
        try:
            (epoch.directory / "flush").touch()
        except OSError as exc:
            errors.append(f"{epoch.epoch}: {exc}")
            continue
        waiting[epoch.epoch] = epoch.last_seq if epoch.last_seq is not None else -1
    return waiting, errors


def _sealed(epoch: EpochRead, last_seq_before: int) -> bool:
    if not (epoch.directory / "flush").exists():
        return True
    return epoch.last_seq is not None and epoch.last_seq > last_seq_before


class _Fixed:
    """An engine adapter that hands ``append_inference_capture`` a capture."""

    def __init__(self, capture: EngineCapture) -> None:
        self.capture = capture

    def collect(self, *, run_id: str, session_id: str) -> EngineCapture:
        return self.capture


def _append_capture(
    artifact: str | Path,
    run_id: str,
    session_id: str,
    capture: EngineCapture,
    envelope_path: str | Path | None,
) -> None:
    try:
        append_inference_capture(
            artifact,
            run_id=run_id,
            session=create_session_summary(source=SOURCE, session_id=session_id),
            engine_adapter=_Fixed(capture),
            envelope_path=envelope_path,
        )
    except (InferInputError, InferUsageError):
        raise
    except ValueError as exc:
        # Every remaining rejection is about the artifact or its envelope.
        raise InferInputError(f"cannot import into {artifact}: {exc}") from exc


def reduce_to_capture(
    read: LogRead, facts: RunFacts, options: ReduceOptions | None = None
) -> EngineCapture:
    """Reduce a read log into an engine capture for ``append_inference_capture``."""
    result = reduce_execution_log(read, facts, options)
    counts = {
        name: sum(isinstance(event, cls) for event in result.events)
        for name, cls in _COLLECTED_BY.items()
    }
    summary = {
        "execution": {
            "directory": str(read.directory.resolve()),
            "format": FORMAT,
            "records": counts,
            "high_water": dict(result.high_water),
            "epochs": result.summary["epochs"],
            "pseudonyms": result.summary["pseudonyms"],
            "notes": list(read.notes),
        }
    }
    return EngineCapture(
        capabilities=CaptureCapabilities(
            supported=SUPPORTED,
            enabled=SUPPORTED,
            collected=tuple(name for name in SUPPORTED if counts[name]),
        ),
        events=tuple(result.events),
        summary=summary,
    )


def run_facts_from_records(
    records: Iterable[InferenceRecord], run_id: str, session_id: str
) -> RunFacts:
    """What an artifact's records tell the reducer about the run."""
    requests: dict[str, RunRequest] = {}
    windows: list[Window] = []
    referenced: set[EntityRef] = set()
    iterations: set[EntityRef] = set()
    attempts: set[EntityRef] = set()
    admissions: dict[tuple[str, int], EntityRef] = {}
    alignments: set[str] = set()
    client_clock_domain: str | None = None
    for record in records:
        if isinstance(record, ArtifactIdentityEvent):
            client_clock_domain = record.context.clock_domain
        elif isinstance(record, ActivityReferenceEvent):
            if record.iteration_ref is not None:
                referenced.add(record.iteration_ref)
        elif isinstance(record, IterationEvent):
            iterations.add(record.iteration_ref)
        elif isinstance(record, RequestEvent):
            if record.attempt_ref is not None:
                attempts.add(record.attempt_ref)
                _note_admission(admissions, record)
        elif isinstance(record, ClockAlignmentEvent):
            alignments.add(record.event_id)
        elif isinstance(record, LegacyInferenceRecord):
            _legacy_facts(record.raw, requests, windows)
    return RunFacts(
        run_id=run_id,
        session_id=session_id,
        client_clock_domain=client_clock_domain,
        requests=requests,
        windows=tuple(windows),
        referenced_iterations=frozenset(referenced),
        existing_iterations=frozenset(iterations),
        existing_attempts=frozenset(attempts),
        existing_alignments=frozenset(alignments),
        existing_admissions=admissions,
    )


def _note_admission(
    admissions: dict[tuple[str, int], EntityRef], record: RequestEvent
) -> None:
    """Index an execution import's request by the alias that admitted it."""
    epoch, seq = record.metadata.get("epoch"), record.metadata.get("admission_seq")
    if record.context.source != SOURCE or record.attempt_ref is None:
        return
    if isinstance(epoch, str) and isinstance(seq, int) and not isinstance(seq, bool):
        admissions[(epoch, seq)] = record.attempt_ref


def _legacy_facts(
    raw: dict[str, Any], requests: dict[str, RunRequest], windows: list[Window]
) -> None:
    kind = raw.get("event_type")
    if kind == "infer.request":
        x_request_id = raw.get("x_request_id")
        if isinstance(x_request_id, str) and x_request_id:
            requests[x_request_id] = RunRequest(
                str(raw.get("request_id")),
                x_request_id,
                _text(raw.get("case_id")),
                _text(raw.get("phase")),
            )
    elif kind == "infer.phase_window":
        _add_window(windows, "phase", raw, "started_at_ns", "drained_at_ns")
    elif kind == "infer.trace_window":
        _add_window(windows, "trace", raw, "started_at_ns", "stopped_at_ns")


def _add_window(
    windows: list[Window], kind: str, raw: dict[str, Any], start: str, end: str
) -> None:
    start_ns, end_ns = _integer(raw.get(start)), _integer(raw.get(end))
    if start_ns is not None and end_ns is not None and end_ns >= start_ns:
        windows.append(
            Window(
                kind,
                start_ns,
                end_ns,
                _text(raw.get("case_id")),
                _text(raw.get("phase")),
            )
        )


def execution_high_water(records: Iterable[InferenceRecord]) -> dict[str, int]:
    """The highest sequence each epoch was imported to, from earlier imports."""
    marks: dict[str, int] = {}
    for record in records:
        if (
            not isinstance(record, CapabilityEvent)
            or record.component != "engine_adapter"
        ):
            continue
        summary = record.metadata.get("summary") or {}
        execution = summary.get("execution") if isinstance(summary, dict) else None
        if not isinstance(execution, dict):
            continue
        for epoch, seq in (execution.get("high_water") or {}).items():
            mark = _integer(seq)
            if mark is not None:
                marks[str(epoch)] = max(mark, marks.get(str(epoch), mark))
    return marks


def _load_records(artifact: str | Path) -> list[InferenceRecord]:
    try:
        return load_inference_artifact(artifact)
    except (OSError, ValueError) as exc:
        raise InferInputError(f"{artifact}: {exc}") from exc


def _read_log(
    directory: str | Path, high_water: dict[str, int], now_ns: int | None
) -> LogRead:
    try:
        read = read_execution_log(directory, high_water=high_water, now_ns=now_ns)
    except (OSError, ValueError) as exc:
        raise InferInputError(f"{directory}: {exc}") from exc
    if not read.epochs:
        raise InferInputError(
            f"{directory}: no vLLM execution-hook epoch under it; expected "
            "<host>-<boot id>/<role>-<pid>-<start ns>/ directories"
        )
    return read


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "SOURCE",
    "SUPPORTED",
    "execution_high_water",
    "flush_execution_log",
    "import_execution_into_artifact",
    "record_failed_execution_import",
    "reduce_to_capture",
    "run_facts_from_records",
]
