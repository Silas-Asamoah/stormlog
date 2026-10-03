"""Reduce the vLLM execution hook's raw log into canonical correlation records.

The hook (``docs/vllm_execution.md``) writes immutable snapshots: an ``alias``
when a request is admitted, a ``scheduled`` snapshot when a step is scheduled,
a ``completed`` one when its output has been processed, and a ``terminal`` when
the request is freed. This module turns them into ``infer.iteration``,
``infer.membership``, ``infer.request`` and ``infer.clock_alignment`` records,
each written exactly once:

- an iteration is reduced only once it is final: its ``completed`` snapshot
  arrived, or it never will because the epoch ended (a ``goodbye``, or
  silence) or a later iteration completed, which makes it ``incomplete``; a
  pending iteration waits for a later import, with no late corrections;
- a request record holds admission facts only, which never change; how the
  request finished goes on the membership of the iteration it was freed in;
- the per-epoch high-water mark advances only past records that were fully
  reduced, and entities the artifact already holds are never emitted again.

Requests are bound to the run exactly, by the alias's external ID against the
``X-Request-Id`` values the client recorded; the internal ID's shape only
proposes a candidate for that same match. Other clients' requests are kept as
members under keyed pseudonyms, never by their IDs.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Iterator

from .. import __version__
from .correlation_events import (
    ClockAlignmentEvent,
    CorrelationContext,
    CorrelationEvent,
    EntityRef,
    IterationEvent,
    MembershipEvent,
    RequestEvent,
)
from .vllm_execution_log import STATE_ALIVE, EpochRead, LogRead, RawRecord

SOURCE = "stormlog.infer.import_execution"
OWN = "run"
FOREIGN = "foreign"
UNRESOLVED = "unresolved"
STATE_COMPLETE = "complete"
STATE_INCOMPLETE = "incomplete"
REASON_EPOCH_ENDED = "epoch_ended"
REASON_COMPLETED_MISSING = "completed_missing"
ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"
ROLE_SPEC_DECODE = "spec_decode"
ROLE_UNKNOWN = "unknown"
_ROLE_BY_PHASE = {"context": ROLE_PREFILL, "generation": ROLE_DECODE}

# vLLM's request IDs: an optional n>1 child index, the API's prefix, the
# external ID, and the 8 random characters vLLM adds unless told not to.
_INTERNAL_ID = re.compile(r"^(?:(\d+)_)?(chatcmpl-|cmpl-)(.*?)(?:-([0-9a-f]{8}))?$")
_COMPLETION_INDEX = re.compile(r"^(.*)-(\d+)$")


@dataclass(frozen=True)
class Window:
    """A client-clock window: a run phase or a trace window."""

    kind: str
    start_ns: int
    end_ns: int
    case_id: str | None = None
    phase: str | None = None


@dataclass(frozen=True)
class RunRequest:
    """One request the client sent, as the artifact recorded it."""

    request_id: str
    x_request_id: str
    case_id: str | None
    phase: str | None


@dataclass(frozen=True)
class RunFacts:
    """What the artifact tells the reducer about the run it binds to."""

    run_id: str
    session_id: str
    client_clock_domain: str | None
    requests: dict[str, RunRequest]  # by x_request_id
    windows: tuple[Window, ...] = ()
    referenced_iterations: frozenset[EntityRef] = frozenset()
    existing_iterations: frozenset[EntityRef] = frozenset()
    existing_attempts: frozenset[EntityRef] = frozenset()
    existing_alignments: frozenset[str] = frozenset()  # event ids


@dataclass(frozen=True)
class ReduceOptions:
    raw_foreign_ids: bool = False


@dataclass(frozen=True)
class Binding:
    """How an internal ID relates to the run."""

    ownership: str
    request: RunRequest | None = None
    child_index: int | None = None
    completion_index: int | None = None
    external: str | None = None
    via: str = "none"  # alias, internal, or none


@dataclass
class Execution:
    """One backend execution of an internal ID; a reused ID has several."""

    key: str
    internal: str
    binding: Binding
    alias: dict[str, Any] | None = None
    alias_seq: int | None = None
    admitted_mono_ns: int | None = None  # None: admitted before these records
    prompt_tokens: int | None = None  # at the first sighting; may grow if resumable
    cached_at_admission: int | None = None
    resumable: bool | None = None
    terminal: dict[str, Any] | None = None
    terminal_seq: int | None = None
    reused: bool = False
    seen_final: bool = False
    memberships: list[Member] = field(default_factory=list)


@dataclass
class Member:
    """One execution's part in one kept iteration."""

    iteration: Iteration
    execution: Execution
    data: dict[str, Any]
    outcome: dict[str, Any] | None
    finish: dict[str, Any] | None = None


@dataclass
class Iteration:
    iteration: str
    scheduled: RawRecord
    completed: RawRecord | None = None
    state: str = STATE_COMPLETE
    reason: str | None = None
    members: list[Member] = field(default_factory=list)

    @property
    def number(self) -> int | None:
        return int(self.iteration) if self.iteration.isdigit() else None

    @property
    def last_seq(self) -> int | None:
        """The last sequence a record of this iteration's step can have."""
        return self.completed.seq if self.completed is not None else None


@dataclass
class ReduceResult:
    events: list[CorrelationEvent]
    summary: dict[str, Any]
    high_water: dict[str, int]


def reduce_execution_log(
    read: LogRead, facts: RunFacts, options: ReduceOptions | None = None
) -> ReduceResult:
    """Reduce every engine epoch of ``read`` for the run in ``facts``."""
    options = options or ReduceOptions()
    events: list[CorrelationEvent] = []
    epochs: dict[str, dict[str, Any]] = {}
    high_water: dict[str, int] = {}
    for epoch in read.engines():
        epoch_events, summary = _EpochReducer(epoch, facts, options).reduce()
        events.extend(epoch_events)
        epochs[epoch.epoch] = summary
        if summary["high_water_seq"] is not None:
            high_water[epoch.epoch] = summary["high_water_seq"]
    for epoch in read.workers():
        epochs[epoch.epoch] = {**epoch.summary(), "reduced": False}
        if epoch.last_seq is not None:
            high_water[epoch.epoch] = epoch.last_seq
    return ReduceResult(
        events, {"epochs": epochs, "pseudonyms": "hmac-sha256-keyed"}, high_water
    )


class _EpochReducer:
    """Reduce one engine epoch; see the module docstring for the rules."""

    def __init__(self, epoch: EpochRead, facts: RunFacts, options: ReduceOptions):
        self.epoch = epoch
        self.facts = facts
        self.options = options
        self.hello = epoch.hello or {}
        self.producer = str(self.hello.get("producer") or f"vllm:{epoch.epoch}")
        self.host = epoch.host
        self.boot = epoch.boot_id or "unknown-boot"
        self.mono_domain = f"{self.host}/{self.boot}/monotonic_ns"
        self.wall_domain = f"{self.host}/{self.boot}/unix_epoch_ns"
        self.binder = _Binder(facts, _randomization(self.hello))
        self.executions: dict[str, Execution] = {}
        self.iterations: dict[str, Iteration] = {}
        self.counts: dict[str, int] = {}

    def reduce(self) -> tuple[list[CorrelationEvent], dict[str, Any]]:
        self._gather()
        final, pending = self._split_iterations()
        kept = [item for item in final if self._keep(item)]
        for item in kept:
            self._collect_members(item)
        self._attach_finishes(pending)
        events: list[CorrelationEvent] = []
        alignment = self._alignment()
        if alignment is not None:
            events.append(alignment)
        for item in kept:
            events.extend(self._iteration_events(item))
        events.extend(self._request_events())
        return events, self._summary(kept, pending)

    # ----------------------------------------------------------- gathering
    def _gather(self) -> None:
        aliases: dict[str, list[RawRecord]] = {}
        first_use: dict[str, int] = {}
        for record in self.epoch.records:
            if record.kind == "alias":
                aliases.setdefault(str(record.data.get("internal")), []).append(record)
            elif record.kind == "scheduled":
                item = Iteration(str(record.data.get("iteration")), record)
                self.iterations[item.iteration] = item
                for member in _members(record):
                    _note_use(first_use, member, record.data.get("start_mono_ns"))
            elif record.kind == "terminal":
                _note_use(first_use, record.data, record.data.get("mono_ns"))
        for record in self.epoch.records:
            if record.kind == "completed":
                self._attach_completed(record)
        self._build_executions(aliases, first_use)
        for record in self.epoch.records:
            if record.kind == "terminal":
                self._attach_terminal(record)

    def _attach_completed(self, record: RawRecord) -> None:
        item = self.iterations.get(str(record.data.get("iteration")))
        if item is None:
            self._count("completed_without_scheduled")
        elif item.completed is None:
            item.completed = record

    def _build_executions(
        self, aliases: dict[str, list[RawRecord]], first_use: dict[str, int]
    ) -> None:
        """One execution per admission; a use before any admission in these
        records continues an execution admitted earlier."""
        for internal in sorted(set(aliases) | set(first_use)):
            admissions = sorted(aliases.get(internal, ()), key=_admission_time)
            used = first_use.get(internal)
            if used is not None and (
                not admissions or used < _admission_time(admissions[0])
            ):
                self._continue_execution(internal)
            for record in admissions:
                self._admit_execution(internal, record)
            executions = [e for e in self.executions.values() if e.internal == internal]
            for execution in executions:
                execution.reused = len(executions) > 1 or execution.key != internal

    def _continue_execution(self, internal: str) -> Execution:
        binding = self.binder.bind(internal, None)
        key = self._latest_known_key(internal, binding)
        execution = Execution(key, internal, binding)
        self.executions[key] = execution
        return execution

    def _admit_execution(self, internal: str, record: RawRecord) -> None:
        binding = self.binder.bind(internal, _text(record.data.get("external")))
        key = self._free_key(internal, binding)
        self.executions[key] = Execution(
            key,
            internal,
            binding,
            alias=record.data,
            alias_seq=record.seq,
            admitted_mono_ns=_integer(record.data.get("mono_ns")),
        )

    def _latest_known_key(self, internal: str, binding: Binding) -> str:
        """The key of the last execution of this ID the artifact holds."""
        latest = internal
        for key in _execution_keys(internal):
            if self._attempt_ref(key, binding) not in self.facts.existing_attempts:
                break
            latest = key
        return latest

    def _free_key(self, internal: str, binding: Binding) -> str:
        """The first key neither this import nor the artifact has used."""
        for key in _execution_keys(internal):
            if key in self.executions:
                continue
            if self._attempt_ref(key, binding) not in self.facts.existing_attempts:
                return key
        raise AssertionError("unreachable")  # pragma: no cover

    def _execution_for(self, internal: str, mono_ns: int | None) -> Execution:
        """The execution of an internal ID at a monotonic time: the one
        admitted last before it, or the continued one for a time before any."""
        candidates = sorted(
            (e for e in self.executions.values() if e.internal == internal),
            key=lambda e: -1 if e.admitted_mono_ns is None else e.admitted_mono_ns,
        )
        if mono_ns is not None:
            admitted = [
                e
                for e in candidates
                if e.admitted_mono_ns is None or e.admitted_mono_ns <= mono_ns
            ]
            if admitted:
                return admitted[-1]
        if candidates:
            return candidates[0]
        return self._continue_execution(internal)

    def _attach_terminal(self, record: RawRecord) -> None:
        execution = self._execution_for(
            str(record.data.get("internal")), _integer(record.data.get("mono_ns"))
        )
        if execution.terminal is None:
            execution.terminal = record.data
            execution.terminal_seq = record.seq

    # ------------------------------------------------------------ selection
    def _split_iterations(self) -> tuple[list[Iteration], list[Iteration]]:
        """Final iterations in sequence order, and the pending ones."""
        ended = self.epoch.state != STATE_ALIVE
        latest_completed = self._latest_completed_number()
        final: list[Iteration] = []
        pending: list[Iteration] = []
        for item in sorted(self.iterations.values(), key=lambda i: i.scheduled.seq):
            if item.completed is not None:
                final.append(item)
            elif ended:
                _mark_incomplete(item, REASON_EPOCH_ENDED)
                final.append(item)
            elif _superseded(item, latest_completed):
                _mark_incomplete(item, REASON_COMPLETED_MISSING)
                final.append(item)
            else:
                pending.append(item)
        return final, pending

    def _latest_completed_number(self) -> int | None:
        numbers = [
            _digits(record.data.get("iteration"))
            for record in self.epoch.records
            if record.kind == "completed"
        ]
        known = [number for number in numbers if number is not None]
        return max(known) if known else None

    def _keep(self, item: Iteration) -> bool:
        ref = EntityRef(self.producer, item.iteration)
        if ref in self.facts.existing_iterations:
            self._count("already_imported")
            return False
        executions = [self._member_execution(m, item) for m in _members(item.scheduled)]
        for execution in executions:
            execution.seen_final = True
        if ref in self.facts.referenced_iterations:
            return True
        if any(e.binding.ownership == OWN for e in executions):
            return True
        if not executions:
            # An idle scheduler step: nothing to attribute, nothing lost.
            self._count("empty_counted")
            return False
        if self._placed_in_a_window(item):
            self._count("foreign_only_placed")
            return True
        self._count("foreign_only_counted")
        return False

    def _member_execution(self, member: dict[str, Any], item: Iteration) -> Execution:
        return self._execution_for(
            str(member.get("internal")),
            _integer(item.scheduled.data.get("start_mono_ns")),
        )

    def _placed_in_a_window(self, item: Iteration) -> bool:
        """Only a direct alignment places an iteration on the client's clock:
        the engine's wall clock and the client's must be one domain."""
        if self.facts.client_clock_domain != self.wall_domain:
            return False
        start = _integer(item.scheduled.data.get("start_wall_ns"))
        done = item.completed.data if item.completed is not None else {}
        end = _integer(done.get("wall_ns"))
        if end is None:
            end = _integer(item.scheduled.data.get("end_wall_ns"))
        if start is None or end is None:
            return False
        return any(w.start_ns <= start and end <= w.end_ns for w in self.facts.windows)

    def _collect_members(self, item: Iteration) -> None:
        outcomes = {str(m.get("internal")): m for m in _members(item.completed)}
        for data in _members(item.scheduled):
            execution = self._member_execution(data, item)
            if execution.prompt_tokens is None and data.get("sighting") == "first":
                execution.prompt_tokens = _integer(data.get("prompt_tokens"))
                execution.resumable = bool(data.get("resumable", False))
                execution.cached_at_admission = _integer(
                    data.get("cached_at_admission")
                )
            member = Member(
                item, execution, data, outcomes.get(str(data.get("internal")))
            )
            item.members.append(member)
            execution.memberships.append(member)

    def _attach_finishes(self, pending: list[Iteration]) -> None:
        """Put each terminal on the membership of the step it was freed in,
        else on the execution's last kept membership; a terminal inside a
        pending step waits with it."""
        waiting = {
            self._member_execution(m, item).key
            for item in pending
            for m in _members(item.scheduled)
        }
        for execution in self.executions.values():
            seq = execution.terminal_seq
            if execution.terminal is None or seq is None:
                continue
            target = self._freed_in(execution, seq)
            if target is not None:
                target.finish = _finish(execution.terminal, in_step=True)
            elif execution.key in waiting:
                continue
            elif execution.memberships:
                last = execution.memberships[-1]
                last.finish = _finish(execution.terminal, in_step=False)
            else:
                self._count("finish_unattached")

    def _freed_in(self, execution: Execution, seq: int) -> Member | None:
        for member in execution.memberships:
            item = member.iteration
            upper = item.last_seq if item.last_seq is not None else self.epoch.last_seq
            if upper is not None and item.scheduled.seq < seq <= upper:
                return member
        return None

    # --------------------------------------------------------------- records
    def _context(self) -> CorrelationContext:
        return CorrelationContext(
            run_id=self.facts.run_id,
            session_id=self.facts.session_id,
            producer_id=self.producer,
            source=SOURCE,
            source_version=__version__,
            engine="vllm",
            engine_version=_text(self.hello.get("vllm_version")),
            host=self.host,
            pid=self.epoch.pid,
            clock_domain=self.mono_domain,
            clock_kind="monotonic",
            collection_mode="imported",
            provenance="observed",
        )

    def _alignment(self) -> ClockAlignmentEvent | None:
        clock = self.hello.get("clock") or {}
        wall, mono = _integer(clock.get("wall_ns")), _integer(clock.get("mono_ns"))
        event_id = f"alignment:{self.epoch.epoch}"
        if wall is None or mono is None or event_id in self.facts.existing_alignments:
            return None
        gap = _integer(clock.get("gap_ns")) or 0
        goodbye = self.epoch.goodbye or {}
        return ClockAlignmentEvent(
            context=self._context(),
            event_id=event_id,
            metadata={"epoch": self.epoch.epoch, "gap_ns": gap, "ended": bool(goodbye)},
            from_clock_domain=self.mono_domain,
            to_clock_domain=self.wall_domain,
            offset_ns=wall - mono,
            uncertainty_ns=(gap + 1) // 2,
            valid_from_ns=mono,
            valid_to_ns=_integer(goodbye.get("mono_ns")),
        )

    def _iteration_events(self, item: Iteration) -> list[CorrelationEvent]:
        ref = EntityRef(self.producer, item.iteration)
        data = item.scheduled.data
        done = item.completed.data if item.completed is not None else {}
        start, end = _integer(data.get("start_mono_ns")), _integer(done.get("mono_ns"))
        if start is None or end is None or end < start:
            end = None
        events: list[CorrelationEvent] = [
            IterationEvent(
                context=self._context(),
                event_id=f"iteration:{self.producer}:{item.iteration}",
                metadata=_iteration_metadata(item, data, done, start, end),
                iteration_ref=ref,
                start_ns=start,
                end_ns=end,
            )
        ]
        events.extend(self._membership_event(ref, member) for member in item.members)
        self._count(item.state)
        if _update_failed(item):
            self._count("update_failed")
        return events

    def _membership_event(self, ref: EntityRef, member: Member) -> MembershipEvent:
        role = _role(member.data)
        attempt = self._attempt_ref(member.execution.key, member.execution.binding)
        return MembershipEvent(
            context=self._context(),
            event_id=f"membership:{self.producer}:{ref.id}:{attempt.id}:{role}",
            metadata=_membership_metadata(member),
            request_ref=self._request_ref(member.execution),
            iteration_ref=ref,
            role=role,
            attempt_ref=attempt,
            input_tokens=_integer(member.data.get("scheduled")),
            output_tokens=_integer((member.outcome or {}).get("retained")),
        )

    def _request_events(self) -> list[CorrelationEvent]:
        events: list[CorrelationEvent] = []
        for execution in self.executions.values():
            if not execution.memberships:
                continue
            attempt = self._attempt_ref(execution.key, execution.binding)
            if attempt in self.facts.existing_attempts:
                continue
            events.append(self._request(execution, attempt))
        return events

    def _request(self, execution: Execution, attempt: EntityRef) -> RequestEvent:
        binding, alias = execution.binding, execution.alias or {}
        own = binding.ownership == OWN
        metadata: dict[str, Any] = {
            "ownership": binding.ownership,
            "bound_via": binding.via,
            "child_index": binding.child_index,
            "completion_index": binding.completion_index,
            "reused_internal_id": execution.reused,
            "admission_seen": execution.alias is not None,
            "admitted_wall_ns": _integer(alias.get("wall_ns")),
            "cached_at_admission": execution.cached_at_admission,
            "resumable": execution.resumable,
        }
        if own and binding.request is not None:
            metadata.update(
                x_request_id=binding.request.x_request_id,
                request_id=binding.request.request_id,
                case_id=binding.request.case_id,
                phase=binding.request.phase,
                external=binding.external,
            )
        return RequestEvent(
            context=self._context(),
            event_id=f"request:{self.producer}:{attempt.id}",
            metadata=metadata,
            request_ref=self._request_ref(execution),
            attempt_ref=attempt,
            backend_request_ref=EntityRef("vllm", execution.internal) if own else None,
            start_ns=_integer(alias.get("mono_ns")),
            input_tokens=execution.prompt_tokens,
        )

    def _request_ref(self, execution: Execution) -> EntityRef:
        binding = execution.binding
        if binding.ownership == OWN and binding.request is not None:
            return EntityRef("stormlog", binding.request.request_id)
        attempt = self._attempt_ref(execution.key, binding)
        return EntityRef(binding.ownership, attempt.id)

    def _attempt_ref(self, key: str, binding: Binding) -> EntityRef:
        if binding.ownership == OWN or self.options.raw_foreign_ids:
            return EntityRef(self.producer, key)
        return EntityRef(self.producer, self._pseudonym(key))

    def _pseudonym(self, key: str) -> str:
        """``HMAC(HMAC(epoch key, run_id), key)``: stable within a run, and
        not derivable from the ID without the key file."""
        inner = hmac.new(
            self.epoch.key or b"", self.facts.run_id.encode(), hashlib.sha256
        )
        return hmac.new(inner.digest(), key.encode(), hashlib.sha256).hexdigest()[:16]

    # --------------------------------------------------------------- summary
    def _count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def _high_water(self, pending: list[Iteration]) -> int | None:
        """Just below the first record a later import still needs: a pending
        step, or the admission of a request no final step has shown yet."""
        last = self.epoch.last_seq
        before = self.epoch.high_water_before
        if last is None:
            return before
        if self.epoch.state != STATE_ALIVE:
            return last
        waiting = [item.scheduled.seq for item in pending]
        waiting.extend(
            e.alias_seq
            for e in self.executions.values()
            if e.alias_seq is not None and not e.seen_final
        )
        mark = min(waiting) - 1 if waiting else last
        return mark if before is None else max(mark, before)

    def _summary(
        self, kept: list[Iteration], pending: list[Iteration]
    ) -> dict[str, Any]:
        summary = self.epoch.summary()
        summary.update(
            producer=self.producer,
            reduced=True,
            config=dict(self.hello.get("config") or {}),
            iterations_kept=len(kept),
            iterations_complete=self.counts.get(STATE_COMPLETE, 0),
            iterations_incomplete=self.counts.get(STATE_INCOMPLETE, 0),
            iterations_pending=len(pending),
            iterations_already_imported=self.counts.get("already_imported", 0),
            foreign_only_placed=self.counts.get("foreign_only_placed", 0),
            foreign_only_counted=self.counts.get("foreign_only_counted", 0),
            empty_counted=self.counts.get("empty_counted", 0),
            iterations_update_failed=self.counts.get("update_failed", 0),
            completed_without_scheduled=self.counts.get(
                "completed_without_scheduled", 0
            ),
            finish_unattached=self.counts.get("finish_unattached", 0),
            executions=_ownership_counts(self.executions),
            pseudonym_key="epoch" if self.epoch.key else "run_id_only",
            high_water_seq=self._high_water(pending),
        )
        return summary


class _Binder:
    """Exact binding of vLLM request IDs to the run's requests."""

    def __init__(self, facts: RunFacts, randomization: bool | None = None) -> None:
        self.requests = facts.requests
        # Whether vLLM added its random suffix to internal IDs; None: unknown.
        self.randomization = randomization

    def bind(self, internal: str, external: str | None) -> Binding:
        match = _INTERNAL_ID.match(internal)
        child = int(match.group(1)) if match is not None and match.group(1) else None
        if external is not None:
            bound = self._bind_external(external, child, "alias")
            return bound or Binding(FOREIGN, child_index=child, via="alias")
        if match is None:
            return Binding(UNRESOLVED, child_index=child)
        for candidate in self._candidates(match):
            bound = self._bind_external(candidate, child, "internal")
            if bound is not None:
                return bound
        return Binding(FOREIGN, child_index=child, via="internal")

    def _candidates(self, match: re.Match[str]) -> list[str]:
        """Without the alias, the ID's shape proposes the external ID: with
        vLLM's random suffix stripped when the hello says one was added, as it
        is when none was, and both ways when that is unknown."""
        prefix, body, suffix = match.group(2), match.group(3), match.group(4)
        stripped = prefix + body
        whole = f"{stripped}-{suffix}" if suffix else stripped
        if self.randomization is True:
            return [stripped]
        if self.randomization is False:
            return [whole]
        return [stripped, whole]

    def _bind_external(
        self, external: str, child: int | None, via: str
    ) -> Binding | None:
        request, index = self._lookup(external)
        if request is None:
            return None
        return Binding(OWN, request, child, index, external, via)

    def _lookup(self, external: str) -> tuple[RunRequest | None, int | None]:
        if external.startswith("chatcmpl-"):
            return self.requests.get(external[len("chatcmpl-") :]), None
        if external.startswith("cmpl-"):
            body = external[len("cmpl-") :]
            match = _COMPLETION_INDEX.match(body)
            if match is not None and match.group(1) in self.requests:
                return self.requests[match.group(1)], int(match.group(2))
            return self.requests.get(body), None
        return None, None


def _execution_keys(internal: str) -> Iterator[str]:
    yield internal
    for index in count(2):
        yield f"{internal}#{index}"


def _admission_time(record: RawRecord) -> int:
    return _integer(record.data.get("mono_ns")) or 0


def _note_use(first_use: dict[str, int], data: dict[str, Any], mono: Any) -> None:
    internal, at = str(data.get("internal")), _integer(mono)
    if at is not None and at < first_use.get(internal, at + 1):
        first_use[internal] = at


def _mark_incomplete(item: Iteration, reason: str) -> None:
    item.state = STATE_INCOMPLETE
    item.reason = reason


def _superseded(item: Iteration, latest_completed: int | None) -> bool:
    """vLLM processes outputs in schedule order, so a step whose output is
    missing while a later step's arrived will never see its own."""
    number = item.number
    return (
        number is not None
        and latest_completed is not None
        and number < latest_completed
    )


def _randomization(hello: dict[str, Any]) -> bool | None:
    config = hello.get("config")
    value = config.get("request_id_randomization") if isinstance(config, dict) else None
    return value if isinstance(value, bool) else None


def _update_failed(item: Iteration) -> bool:
    """Whether vLLM's update_from_output raised for this step: the outcome of
    every member is unknown, and nothing is read from its computed_after."""
    return item.completed is not None and bool(item.completed.data.get("update_failed"))


def _members(record: RawRecord | None) -> list[dict[str, Any]]:
    if record is None:
        return []
    members = record.data.get("members")
    return (
        [m for m in members if isinstance(m, dict)] if isinstance(members, list) else []
    )


def _role(member: dict[str, Any]) -> str:
    """vLLM's own phase decides the role, never the token counts: a request
    resumed after preemption recomputes past its prompt and is still context."""
    phase = _text(member.get("phase"))
    drafts = _integer(member.get("drafts_scheduled")) or 0
    if phase == "generation" and drafts > 0:
        return ROLE_SPEC_DECODE
    return _ROLE_BY_PHASE.get(phase, ROLE_UNKNOWN) if phase else ROLE_UNKNOWN


def _finish(terminal: dict[str, Any], *, in_step: bool) -> dict[str, Any]:
    return {
        "status": terminal.get("status"),
        "finish_reason": terminal.get("finish_reason"),
        "output_tokens": terminal.get("output_tokens"),
        "mono_ns": terminal.get("mono_ns"),
        "wall_ns": terminal.get("wall_ns"),
        "in_step": in_step,
    }


def _iteration_metadata(
    item: Iteration,
    data: dict[str, Any],
    done: dict[str, Any],
    start: int | None,
    end: int | None,
) -> dict[str, Any]:
    ownerships = [m.execution.binding.ownership for m in item.members]
    return {
        "state": item.state,
        "incomplete_reason": item.reason,
        "epoch": item.scheduled.epoch,
        "start_wall_ns": data.get("start_wall_ns"),
        "schedule_end_wall_ns": data.get("end_wall_ns"),
        "schedule_end_mono_ns": data.get("end_mono_ns"),
        "completed_wall_ns": done.get("wall_ns"),
        "update_failed": _update_failed(item),
        "scheduler_residence_ns": (
            end - start if start is not None and end is not None else None
        ),
        "total_tokens": data.get("total_tokens"),
        "zero_token": data.get("zero_token"),
        "preempted": len(data.get("preempted") or []),
        "members": len(ownerships),
        "run_members": ownerships.count(OWN),
        "foreign_members": ownerships.count(FOREIGN),
        "unresolved_members": ownerships.count(UNRESOLVED),
    }


def _membership_metadata(member: Member) -> dict[str, Any]:
    data, outcome = member.data, member.outcome or {}
    kept = outcome.get("outcome") == "kept"
    metadata: dict[str, Any] = {
        "ownership": member.execution.binding.ownership,
        "state": member.iteration.state,
        "sighting": data.get("sighting"),
        "phase": data.get("phase"),
        "computed_before": data.get("computed_before"),
        "prompt_tokens": data.get("prompt_tokens"),
        "prefill_scheduled": data.get("prefill_scheduled"),
        "past_prompt_scheduled": data.get("past_prompt_scheduled"),
        "drafts_scheduled": data.get("drafts_scheduled"),
        "cached_at_admission": data.get("cached_at_admission"),
        "recompute": data.get("recompute"),
        "output_before": data.get("output_before"),
        "resumable": data.get("resumable"),
        "processed_prefill": data.get("prefill_scheduled") if kept else 0,
        "outcome": outcome.get("outcome", "unknown"),
        "stale": outcome.get("stale"),
        "sampled": outcome.get("sampled"),
        "accepted_drafts": outcome.get("accepted_drafts"),
        "finish_reason": outcome.get("finish_reason"),
        "computed_after": outcome.get("computed_after"),
        "update_failed": _update_failed(member.iteration),
        "cache_actions": "unknown",
    }
    if member.finish is not None:
        metadata["finish"] = member.finish
    return metadata


def _ownership_counts(executions: dict[str, Execution]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for execution in executions.values():
        counts[execution.binding.ownership] = (
            counts.get(execution.binding.ownership, 0) + 1
        )
    return counts


def _digits(value: Any) -> int | None:
    return int(value) if isinstance(value, str) and value.isdigit() else _integer(value)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "FOREIGN",
    "OWN",
    "REASON_COMPLETED_MISSING",
    "REASON_EPOCH_ENDED",
    "SOURCE",
    "STATE_COMPLETE",
    "STATE_INCOMPLETE",
    "UNRESOLVED",
    "Binding",
    "ReduceOptions",
    "ReduceResult",
    "RunFacts",
    "RunRequest",
    "Window",
    "reduce_execution_log",
]
