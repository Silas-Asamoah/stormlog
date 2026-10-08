"""Join an artifact's evidence per request, per step and per engine.

The client's records (``infer.request`` at the end, ``infer.dispatch`` at the
send, ``infer.first_content``) are grouped by the client's request ID. The
execution import's records are grouped by the engine: each run-owned
``infer.request`` is one backend execution (an attempt) of a client request,
with its memberships in steps. Engine epochs come from the import's
capability summaries, where each epoch's hello configuration and loss
coverage are kept. Nothing here is a judgement; the classes judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .correlation_events import (
    ArtifactIdentityEvent,
    CapabilityEvent,
    ClockAlignmentEvent,
    EntityRef,
    IterationEvent,
    LegacyInferenceRecord,
    MembershipEvent,
    RequestEvent,
    StageEvent,
)
from .diagnosis_inputs import InputFile, Line

EXECUTION_SOURCE = "stormlog.infer.import_execution"
OWN = "run"


@dataclass
class ClientRequest:
    """One request as the client recorded it, from its send to its end."""

    request_id: str
    terminal: Line | None = None
    dispatch: Line | None = None
    first_content: Line | None = None

    def _field(self, name: str) -> Any:
        for line in (self.terminal, self.dispatch, self.first_content):
            value = (line.raw or {}).get(name) if line is not None else None
            if value is not None:
                return value
        return None

    @property
    def case_id(self) -> str | None:
        return _text(self._field("case_id"))

    @property
    def phase(self) -> str | None:
        return _text(self._field("phase"))

    @property
    def x_request_id(self) -> str | None:
        return _text(self._field("x_request_id"))

    @property
    def intended_at_ns(self) -> int | None:
        return _integer(self._field("intended_at_ns"))

    @property
    def sent_at_ns(self) -> int | None:
        """The client's send stamp: the dispatch record's, else the end's."""
        for line in (self.dispatch, self.terminal):
            value = _integer((line.raw or {}).get("started_at_ns")) if line else None
            if value is not None:
                return value
        return None

    @property
    def first_content_at_ns(self) -> int | None:
        if self.first_content is not None:
            return _integer((self.first_content.raw or {}).get("first_content_at_ns"))
        raw = (self.terminal.raw or {}) if self.terminal is not None else {}
        started, ttft = _integer(raw.get("started_at_ns")), raw.get("ttft_ms")
        if started is None or not isinstance(ttft, (int, float)):
            return None
        return started + round(float(ttft) * 1e6)

    @property
    def ended_at_ns(self) -> int | None:
        raw = (self.terminal.raw or {}) if self.terminal is not None else {}
        return _integer(raw.get("ended_at_ns"))

    @property
    def status(self) -> str | None:
        """The outcome; None while the request is still in flight."""
        raw = (self.terminal.raw or {}) if self.terminal is not None else {}
        return _text(raw.get("status"))

    @property
    def terminal_raw(self) -> dict[str, Any]:
        return (self.terminal.raw or {}) if self.terminal is not None else {}

    def lines(self) -> list[Line]:
        return [
            line
            for line in (self.dispatch, self.first_content, self.terminal)
            if line is not None
        ]


@dataclass
class Execution:
    """One backend execution (attempt) of a request, with its steps."""

    line: Line
    event: RequestEvent
    memberships: list[tuple[Line, MembershipEvent]] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.event.metadata

    @property
    def ownership(self) -> str:
        return str(self.metadata.get("ownership") or "unknown")

    @property
    def attempt(self) -> EntityRef | None:
        return self.event.attempt_ref

    @property
    def producer(self) -> str:
        return self.event.context.producer_id


@dataclass
class EngineEpoch:
    """One engine process lifetime, as the latest import summarized it."""

    epoch: str
    producer: str | None
    config: dict[str, Any]
    coverage: dict[str, Any] | None
    state: str | None
    unanchored: list[dict[str, Any]] = field(default_factory=list)

    @property
    def observes(self) -> list[str] | None:
        observes = (self.coverage or {}).get("observes")
        return list(observes) if isinstance(observes, list) else None


@dataclass
class RunView:
    """Everything the classes read, grouped once."""

    source: InputFile
    identity: tuple[Line, ArtifactIdentityEvent] | None = None
    client: dict[str, ClientRequest] = field(default_factory=dict)
    executions: dict[EntityRef, Execution] = field(default_factory=dict)
    iterations: dict[EntityRef, tuple[Line, IterationEvent]] = field(
        default_factory=dict
    )
    stages: list[tuple[Line, StageEvent]] = field(default_factory=list)
    alignments: list[tuple[Line, ClockAlignmentEvent]] = field(default_factory=list)
    scrapes: list[Line] = field(default_factory=list)
    spans: list[Line] = field(default_factory=list)
    phase_starts: list[Line] = field(default_factory=list)
    phase_windows: list[Line] = field(default_factory=list)
    trace_windows: list[Line] = field(default_factory=list)
    workload: Line | None = None
    sessions: list[Line] = field(default_factory=list)
    engines: dict[str, EngineEpoch] = field(default_factory=dict)

    @property
    def run_id(self) -> str | None:
        return None if self.identity is None else self.identity[1].context.run_id

    @property
    def session_id(self) -> str | None:
        return None if self.identity is None else self.identity[1].context.session_id

    @property
    def clock_domain(self) -> str | None:
        """The artifact's own clock, R: every placed time is on it."""
        if self.identity is None:
            return None
        return self.identity[1].context.clock_domain

    def executions_of(self, request_id: str) -> list[Execution]:
        """The run's executions of one client request, in admission order."""
        found = [
            execution
            for execution in self.executions.values()
            if execution.ownership == OWN
            and execution.event.request_ref.id == request_id
        ]
        return sorted(found, key=lambda e: e.event.start_ns or 0)

    def has_dispatch_records(self) -> bool:
        return any(request.dispatch is not None for request in self.client.values())

    def has_first_content_records(self) -> bool:
        return any(r.first_content is not None for r in self.client.values())


def join(source: InputFile) -> RunView:
    """Group the artifact's records for the classes."""
    view = RunView(source)
    memberships: list[tuple[Line, MembershipEvent]] = []
    for line in source.records():
        record = line.record
        if isinstance(record, LegacyInferenceRecord):
            _add_legacy(view, line)
        elif isinstance(record, MembershipEvent):
            memberships.append((line, record))
        elif record is not None:
            _add_event(view, line, record)
    for line, membership in memberships:
        execution = view.executions.get(
            membership.attempt_ref or membership.request_ref
        )
        if execution is not None:
            execution.memberships.append((line, membership))
    for execution in view.executions.values():
        execution.memberships.sort(key=lambda pair: _step_order(view, pair[1]))
    return view


def _add_event(view: RunView, line: Line, record: Any) -> None:
    if isinstance(record, ArtifactIdentityEvent):
        view.identity = view.identity or (line, record)
    elif isinstance(record, RequestEvent):
        if record.context.source == EXECUTION_SOURCE:
            key = record.attempt_ref or record.request_ref
            view.executions.setdefault(key, Execution(line, record))
    elif isinstance(record, IterationEvent):
        view.iterations.setdefault(record.iteration_ref, (line, record))
    elif isinstance(record, StageEvent):
        view.stages.append((line, record))
    elif isinstance(record, ClockAlignmentEvent):
        view.alignments.append((line, record))
    elif isinstance(record, CapabilityEvent):
        _add_engine_summary(view, record)


_CLIENT_KINDS = {
    "infer.request": "terminal",
    "infer.dispatch": "dispatch",
    "infer.first_content": "first_content",
}
_LISTS = {
    "infer.vllm_scrape": "scrapes",
    "infer.vllm_span": "spans",
    "infer.phase_start": "phase_starts",
    "infer.phase_window": "phase_windows",
    "infer.trace_window": "trace_windows",
    "infer.session": "sessions",
}


def _add_legacy(view: RunView, line: Line) -> None:
    raw = line.raw or {}
    kind = line.event_type or ""
    if kind in _CLIENT_KINDS:
        request_id = _text(raw.get("request_id"))
        if request_id is not None:
            request = view.client.setdefault(request_id, ClientRequest(request_id))
            if getattr(request, _CLIENT_KINDS[kind]) is None:
                setattr(request, _CLIENT_KINDS[kind], line)
    elif kind in _LISTS:
        getattr(view, _LISTS[kind]).append(line)
    elif kind == "infer.workload":
        view.workload = view.workload or line


def _add_engine_summary(view: RunView, record: CapabilityEvent) -> None:
    """Each import restates every epoch it read; the latest statement wins,
    and dated facts without a step are kept from every import."""
    if record.component != "engine_adapter":
        return
    for name, epoch in _epoch_summaries(record).items():
        earlier = view.engines.get(name)
        facts = list(earlier.unanchored) if earlier is not None else []
        facts += [
            fact
            for fact in epoch.get("unanchored") or ()
            if isinstance(fact, dict) and fact not in facts
        ]
        coverage = epoch.get("coverage")
        view.engines[name] = EngineEpoch(
            epoch=name,
            producer=_text(epoch.get("producer")),
            config=dict(epoch.get("config") or {}),
            coverage=coverage if isinstance(coverage, dict) else None,
            state=_text(epoch.get("state")),
            unanchored=facts,
        )


def _epoch_summaries(record: CapabilityEvent) -> dict[str, dict[str, Any]]:
    """The engine epochs an import's capability summary describes."""
    summary = record.metadata.get("summary")
    execution = summary.get("execution") if isinstance(summary, dict) else None
    epochs = execution.get("epochs") if isinstance(execution, dict) else None
    if not isinstance(epochs, dict):
        return {}
    return {
        str(name): epoch
        for name, epoch in epochs.items()
        if isinstance(epoch, dict) and epoch.get("role", "engine") == "engine"
    }


def _step_order(view: RunView, membership: MembershipEvent) -> tuple[int, str]:
    found = view.iterations.get(membership.iteration_ref)
    start = found[1].start_ns if found is not None else None
    return (start if start is not None else 0, membership.iteration_ref.id)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "ClientRequest",
    "EngineEpoch",
    "Execution",
    "RunView",
    "join",
]
