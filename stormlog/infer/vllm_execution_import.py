"""Import a vLLM execution-hook log into an inference artifact.

The artifact's ``infer.artifact`` record supplies the run and session; its
``infer.dispatch`` and ``infer.request`` records supply the ``X-Request-Id``
values the reducer binds to; its phase and trace windows, and phases begun but
not yet measured, let foreign-only iterations be placed; and the
entities it already holds, with the high-water marks of earlier imports, keep
a re-import from writing anything twice. The reduced records are appended
through ``append_inference_capture`` as an engine adapter's capture. The raw
log itself is never registered as an attachment: it names other clients'
requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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
    SCHEME_RAW,
    SCHEME_WITHHELD,
    ReduceOptions,
    RunFacts,
    RunRequest,
    Window,
    foreign_scheme,
    reduce_execution_log,
)
from .vllm_execution_log import (
    FORMAT,
    STATE_ALIVE,
    STATE_UNKNOWN,
    EpochRead,
    Importer,
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
    importer: Importer | None = None,
    server_stopped: bool = False,
) -> EngineCapture:
    """Append the log's new, final iterations to ``artifact``; return the capture.

    The capture's summary, recorded on the engine adapter's capability event,
    carries each epoch's high-water mark, which the next import starts from.
    ``server_stopped`` says the server that wrote the log is no longer
    running, so an epoch without ``goodbye`` is gone and its pending steps
    are final; without it, an epoch whose liveness cannot be judged from
    here (another host or boot) keeps them for a later import.
    """
    run_id, session_id = artifact_run_identity(artifact)
    records = _load_records(artifact)
    facts = run_facts_from_records(records, run_id, session_id)
    read = _read_log(directory, execution_high_water(records), importer, server_stopped)
    _check_foreign_schemes(read, execution_foreign_schemes(records), raw_foreign_ids)
    capture = reduce_to_capture(read, facts, ReduceOptions(raw_foreign_ids))
    _append_capture(artifact, run_id, session_id, capture, envelope_path)
    return capture


def _execution_summaries(records: Iterable[InferenceRecord]) -> list[dict[str, Any]]:
    """Every execution import's summary, in artifact order."""
    summaries = []
    for record in records:
        if (
            not isinstance(record, CapabilityEvent)
            or record.component != "engine_adapter"
        ):
            continue
        summary = record.metadata.get("summary") or {}
        execution = summary.get("execution") if isinstance(summary, dict) else None
        if isinstance(execution, dict):
            summaries.append(execution)
    return summaries


def execution_foreign_schemes(records: Iterable[InferenceRecord]) -> dict[str, str]:
    """How each epoch's other clients were written by earlier imports."""
    schemes: dict[str, str] = {}
    for execution in _execution_summaries(records):
        for epoch, summary in (execution.get("epochs") or {}).items():
            scheme = summary.get("foreign_ids") if isinstance(summary, dict) else None
            if isinstance(scheme, str) and scheme:
                schemes[str(epoch)] = scheme
    return schemes


def _check_foreign_schemes(
    read: LogRead, recorded: dict[str, str], raw_foreign_ids: bool
) -> None:
    """Refuse to mix raw IDs and pseudonyms within one epoch.

    An attempt written under one scheme cannot be found under the other, so
    the same backend execution would get a second request and split
    memberships. A withheld epoch wrote no other client's identity, so any
    later scheme may follow it, and withholding may follow either.
    """
    for epoch in read.engines():
        before = recorded.get(epoch.epoch)
        now = foreign_scheme(epoch, raw_foreign_ids)
        if before in (None, now) or SCHEME_WITHHELD in (before, now):
            continue
        hint = "--raw-foreign-ids" if before == SCHEME_RAW else "no --raw-foreign-ids"
        raise InferInputError(
            f"{epoch.epoch}: other clients' IDs were imported as {before}; this "
            f"import would write them as {now}, which would duplicate their "
            f"requests. Import this epoch with {hint}, as before."
        )


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
    importer: Importer | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Ask every live epoch's writer to seal its open segment, and wait for
    proof that it did: the ``flush`` file gone, or a record written after the
    request (a heartbeat comes every second). Ended epochs need no flush; an
    epoch whose liveness cannot be judged from here is asked like a live one.
    """
    read = read_execution_log(directory, importer=importer)
    waiting, errors = _request_flushes(read)
    requested = sorted(waiting)
    deadline = clock() + timeout_seconds
    while waiting and clock() < deadline:
        sleep(poll_seconds)
        for epoch in read_execution_log(directory, importer=importer).epochs:
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
        if epoch.state not in (STATE_ALIVE, STATE_UNKNOWN):
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
    client = _ClientFacts()
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
            client.add(record.raw)
    return RunFacts(
        run_id=run_id,
        session_id=session_id,
        client_clock_domain=client_clock_domain,
        requests=client.requests,
        windows=client.all_windows(),
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


# The end of a phase that has begun but whose window is not yet recorded,
# while the run is still going: every later step may lie inside it.
_OPEN_END_NS = 2**63 - 1


@dataclass
class _ClientFacts:
    """What the client's v1 records say: its requests and its windows."""

    requests: dict[str, RunRequest] = field(default_factory=dict)
    windows: list[Window] = field(default_factory=list)
    # Phases begun and not yet measured, by (case, phase), at their start.
    begun: dict[tuple[str | None, str | None], int] = field(default_factory=dict)
    session_end_ns: int | None = None

    def add(self, raw: dict[str, Any]) -> None:
        kind = raw.get("event_type")
        if kind in ("infer.request", "infer.dispatch"):
            self._add_request(raw)
        elif kind == "infer.phase_start":
            started_at_ns = _integer(raw.get("started_at_ns"))
            if started_at_ns is not None:
                self.begun[_phase_key(raw)] = started_at_ns
        elif kind == "infer.phase_window":
            self.begun.pop(_phase_key(raw), None)
            _add_window(self.windows, "phase", raw, "started_at_ns", "drained_at_ns")
        elif kind == "infer.trace_window":
            _add_window(self.windows, "trace", raw, "started_at_ns", "stopped_at_ns")
        elif kind == "infer.session" and raw.get("status") != "running":
            self.session_end_ns = _integer(raw.get("timestamp_ns"))

    def _add_request(self, raw: dict[str, Any]) -> None:
        # A request is bound from its send, so a prefix of a growing
        # artifact binds the requests still in flight.
        x_request_id = raw.get("x_request_id")
        if isinstance(x_request_id, str) and x_request_id:
            self.requests[x_request_id] = RunRequest(
                str(raw.get("request_id")),
                x_request_id,
                _text(raw.get("case_id")),
                _text(raw.get("phase")),
            )

    def all_windows(self) -> tuple[Window, ...]:
        """The recorded windows, plus each begun phase up to the session's
        end, or open-ended while the session is still running."""
        end_ns = _OPEN_END_NS if self.session_end_ns is None else self.session_end_ns
        begun = [
            Window("phase", start_ns, end_ns, case_id, phase)
            for (case_id, phase), start_ns in self.begun.items()
            if start_ns <= end_ns
        ]
        return (*self.windows, *begun)


def _phase_key(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    return _text(raw.get("case_id")), _text(raw.get("phase"))


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
    directory: str | Path,
    high_water: dict[str, int],
    importer: Importer | None,
    server_stopped: bool,
) -> LogRead:
    try:
        read = read_execution_log(
            directory,
            high_water=high_water,
            importer=importer,
            server_stopped=server_stopped,
        )
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
    "execution_foreign_schemes",
    "execution_high_water",
    "flush_execution_log",
    "import_execution_into_artifact",
    "record_failed_execution_import",
    "reduce_to_capture",
    "run_facts_from_records",
]
