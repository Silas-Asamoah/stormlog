"""Incidents: opened when a trigger fires, sealed into a bundle after them.

A firing trigger opens an incident, or joins one still collecting its
post-window. The incident's pre-window reaches back from the firing to the
start of the trigger's first violating window, capped at ``pre_seconds``,
and a trigger that joins widens it to its own; an event or a test trigger,
which has no window, reaches back the whole ``pre_seconds``. The
post-window runs ``post_seconds`` past the firing. A test trigger never
joins an incident and is never joined: it qualifies the capture path on
its own. When the post-window ends
the incident is sealed: its scrapes, its windows and its ``infer.incident``
record are written as generation 0 of a bundle, off the watcher's loop, and
the record goes to the ledger once the bundle is published. The scrapes are
written from the history's compressed copies, expanded one at a time, so a
seal never holds them parsed. Membership is frozen at the seal: a later
firing never changes a sealed bundle.

Incidents are bounded three ways, and every firing they turn away is counted
by reason: at most ``max_open_incidents`` collect at once, at most
``max_incidents_per_hour`` open in any trailing hour, and at most 16
triggers join one incident. Health and test incidents have open and hourly
budgets of their own, the same sizes, so a flapping exporter or a test run
never crowds out an incident that counts toward the exit code.
"""

from __future__ import annotations

import contextlib
import json
import socket
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ... import __version__
from ..correlation_events import ArtifactIdentityEvent, CorrelationContext
from ..host_clock import host_boot_id, wall_clock_domain
from ..vllm_telemetry import SCRAPE_OK
from .config import IncidentLimits
from .disk import BudgetExceeded
from .evaluate import KIND_HEALTH, KIND_TEST, TickResult
from .history import Held, ScrapeHistory, expand
from .records import (
    INCIDENT,
    INCIDENT_EVENT,
    MAX_JOINED_TRIGGERS,
    capture_fields,
    empty_loss,
    envelope,
    trigger_fields,
)
from .stats import NO_DETAIL, WatchStats
from .store import STATUS_COMPLETED, STATUS_INTERRUPTED, IncidentStore, PrunedBundle

_NS = 1_000_000_000
_HOUR_NS = 3600 * _NS
# Reserved beyond the bundle's own bytes, for the store's bookkeeping slack.
_RESERVE_BASE = 256 * 1024
WINDOW_EVENT = "infer.incident_window"
# A window's first scrape may land up to a tick, plus loop lag, after it opens.
_FIRST_TICK_SLACK = 1.5

Emit = Callable[[dict[str, Any]], object]
Task = Callable[[], None]
OnPruned = Callable[[list[PrunedBundle]], None]


@dataclass(frozen=True)
class WatchClock:
    """The watcher's clocks: monotonic for decisions, wall for records."""

    mono_ns: Callable[[], int] = time.monotonic_ns
    wall_ns: Callable[[], int] = time.time_ns
    mono_origin_ns: int = field(default_factory=time.monotonic_ns)
    wall_origin_ns: int = field(default_factory=time.time_ns)

    def to_wall(self, mono_ns: int) -> int:
        """A monotonic instant on the wall clock of the watcher's start."""
        return self.wall_origin_ns + (mono_ns - self.mono_origin_ns)


@dataclass
class OpenIncident:
    """An incident still collecting its post-window."""

    incident_id: str
    trigger: dict[str, Any]
    counts_toward_exit: bool
    detected_at_mono: int
    pre_start_mono: int
    post_end_mono: int
    joined: list[dict[str, Any]] = field(default_factory=list)
    suppressed: Counter[str] = field(default_factory=Counter)
    loss_at_open: dict[str, int] = field(default_factory=dict)


@dataclass
class _Lane:
    """The incidents one kind of firing has open, and when it opened each in
    the trailing hour."""

    open: set[str] = field(default_factory=set)
    opened: deque[int] = field(default_factory=deque)

    def refusal(self, at_mono: int, limits: IncidentLimits) -> str | None:
        """Why a new incident may not open now, or None."""
        while self.opened and self.opened[0] <= at_mono - _HOUR_NS:
            self.opened.popleft()
        if len(self.opened) >= limits.max_incidents_per_hour:
            return "rate_limit"
        if len(self.open) >= limits.max_open_incidents:
            return "open_limit"
        return None


# Health and test incidents are budgeted apart from the triggers' own.
_SIDE_KINDS = frozenset({KIND_HEALTH, KIND_TEST})


@dataclass(frozen=True)
class _HeldScrapes:
    """The scrapes a seal writes, as the history holds them: compressed.

    Each is expanded once here, for its status and size, and once more as
    the bundle is written, so the seal holds only references to the
    history's own copies.
    """

    items: list[Held]
    ok: tuple[bool, ...]
    expanded_bytes: int

    @classmethod
    def of(cls, items: list[Held]) -> _HeldScrapes:
        ok = []
        size = 0
        for _stamp, blob in items:
            text = expand(blob)
            size += len(text) + 1
            ok.append(json.loads(text).get("status") == SCRAPE_OK)
        return cls(items, tuple(ok), size)


@dataclass(frozen=True)
class Identity:
    """Who wrote an incident: the watch session, its run and its owner."""

    session_id: str
    run_id: str
    owner: str
    host: str = field(default_factory=socket.gethostname)
    boot_id: str | None = field(default_factory=host_boot_id)

    @property
    def clock_domain(self) -> str:
        return wall_clock_domain(self.host, self.boot_id)


class IncidentManager:
    """Open, join and seal incidents within their limits."""

    def __init__(
        self,
        *,
        store: IncidentStore,
        history: ScrapeHistory,
        limits: IncidentLimits,
        identity: Identity,
        clock: WatchClock,
        stats: WatchStats,
        emit: Emit,
        submit: Callable[[Task], bool],
        post: Callable[[Task], None],
        loss: Callable[[], dict[str, int]],
        tick_seconds: float = 1.0,
        on_pruned: OnPruned | None = None,
    ) -> None:
        self.store = store
        self.history = history
        self.limits = limits
        self.identity = identity
        self.clock = clock
        self.stats = stats
        self._emit = emit
        self._submit = submit
        self._post = post
        self._loss = loss
        # Bundles the store removed to make room for a seal, for the ledger.
        self._on_pruned = on_pruned
        self._tick_seconds = tick_seconds
        self._slack_seconds = _FIRST_TICK_SLACK * tick_seconds
        self.open: dict[str, OpenIncident] = {}
        self._lanes = {False: _Lane(), True: _Lane()}  # keyed by "is a side kind"
        # Every firing, joins and refusals included.
        self.firings = 0
        # Incidents recorded, and those of them that count toward exit 3.
        self.recorded = 0
        self.recorded_counting = 0
        self.persisted = 0
        self.persist_failures = 0
        self.sealed: list[dict[str, Any]] = []

    # ------------------------------------------------------------ firings

    def on_fired(self, result: TickResult) -> str | None:
        """A trigger fired: open or join an incident; the id, or None."""
        assert result.transition is not None
        state = result.transition
        trigger = self._trigger_record(result)
        window_ns = int(result.spec.sustain.window * _NS)
        pending_since = state.pending_since_ns or result.at_ns
        return self._admit(
            trigger,
            counts=bool(result.spec.counts_toward_exit),
            at_mono=result.at_ns,
            reach_mono=pending_since - window_ns,
        )

    def on_test(self, at_mono: int, *, requested_wall_ns: int | None) -> str | None:
        """A qualification test trigger: an incident with no sustain."""
        trigger = trigger_fields(
            trigger_id="test",
            kind=KIND_TEST,
            reason="test trigger",
            fired_at_ns=self.clock.to_wall(at_mono),
            counts_toward_exit=False,
            requested_at_ns=requested_wall_ns,
        )
        return self._admit(
            trigger, counts=False, at_mono=at_mono, reach_mono=self._earliest(at_mono)
        )

    def on_event(self, trigger_id: str, reason: str, at_mono: int) -> str | None:
        """A health event that needs no sustain, such as an exporter restart."""
        trigger = trigger_fields(
            trigger_id=trigger_id,
            kind=KIND_HEALTH,
            reason=reason,
            fired_at_ns=self.clock.to_wall(at_mono),
            counts_toward_exit=False,
        )
        return self._admit(
            trigger, counts=False, at_mono=at_mono, reach_mono=self._earliest(at_mono)
        )

    def _earliest(self, at_mono: int) -> int:
        """How far back a pre-window from ``at_mono`` may reach."""
        return at_mono - int(self.limits.pre_seconds * _NS)

    def _admit(
        self, trigger: dict[str, Any], *, counts: bool, at_mono: int, reach_mono: int
    ) -> str | None:
        self.firings += 1
        joinable = [
            incident
            for incident in self.open.values()
            if at_mono <= incident.post_end_mono
            and KIND_TEST not in (incident.trigger["kind"], trigger["kind"])
        ]
        if joinable:
            incident = joinable[0]
            if len(incident.joined) >= MAX_JOINED_TRIGGERS:
                incident.suppressed["join_limit"] += 1
                self.stats.add("suppressed_total", labels=("join_limit",))
                return None
            incident.joined.append(trigger)
            incident.counts_toward_exit |= counts
            incident.pre_start_mono = min(
                incident.pre_start_mono, max(reach_mono, self._earliest(at_mono))
            )
            return incident.incident_id
        reason = self._lane(trigger).refusal(at_mono, self.limits)
        if reason is not None:
            self.stats.add("suppressed_total", labels=(reason,))
            return None
        return self._open(
            trigger, counts=counts, at_mono=at_mono, reach_mono=reach_mono
        )

    def _lane(self, trigger: Mapping[str, Any]) -> _Lane:
        return self._lanes[trigger["kind"] in _SIDE_KINDS]

    def _open(
        self, trigger: dict[str, Any], *, counts: bool, at_mono: int, reach_mono: int
    ) -> str:
        incident_id = self.store.new_incident_id(self.clock.to_wall(at_mono))
        incident = OpenIncident(
            incident_id=incident_id,
            trigger=trigger,
            counts_toward_exit=counts,
            detected_at_mono=at_mono,
            pre_start_mono=max(reach_mono, self._earliest(at_mono)),
            post_end_mono=at_mono + int(self.limits.post_seconds * _NS),
            loss_at_open=self._loss(),
        )
        self.open[incident_id] = incident
        lane = self._lane(trigger)
        lane.open.add(incident_id)
        lane.opened.append(at_mono)
        record = envelope(
            INCIDENT_EVENT,
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            timestamp_ns=self.clock.to_wall(at_mono),
        )
        record.update(
            incident_id=incident_id,
            event="opened",
            trigger_id=trigger["trigger_id"],
            kind=trigger["kind"],
            rearm_basis=None,
        )
        self._emit(record)
        return incident_id

    # -------------------------------------------------------------- seals

    def on_tick(self, at_mono: int) -> list[str]:
        """Seal every incident whose post-window has ended."""
        due = [i for i in self.open.values() if i.post_end_mono <= at_mono]
        for incident in due:
            self._seal(incident, at_mono, interrupted=False)
        return [incident.incident_id for incident in due]

    def close(self, at_mono: int) -> None:
        """Seal what is still open, as interrupted: the watch is stopping."""
        for incident in list(self.open.values()):
            self._seal(incident, at_mono, interrupted=True)

    def _seal(self, incident: OpenIncident, at_mono: int, *, interrupted: bool) -> None:
        del self.open[incident.incident_id]
        self._lane(incident.trigger).open.discard(incident.incident_id)
        end_mono = min(at_mono, incident.post_end_mono)
        held = _HeldScrapes.of(
            self.history.ring.compressed(incident.pre_start_mono, end_mono)
        )
        record = self._incident_record(incident, held, end_mono)
        record["status"] = STATUS_INTERRUPTED if interrupted else STATUS_COMPLETED
        lines = _Lines(self._bundle_head(incident, end_mono), held, _line(record))
        status = STATUS_INTERRUPTED if interrupted else STATUS_COMPLETED
        sealed_at = self.clock.to_wall(at_mono)
        protected = frozenset(self.open)

        def write() -> None:
            # On the store's worker; the bookkeeping goes back to the loop,
            # whatever the write raised: the incident is always recorded.
            try:
                record["bundle"], record["bundle_error"] = self._write_bundle(
                    incident.incident_id, lines, status, sealed_at, protected
                )
            except Exception as exc:
                record["bundle"], record["bundle_error"] = None, _error_text(exc)
            finally:
                pruned = self.store.take_pruned()
                self._post(lambda: self._finish(incident, record, pruned))

        if not self._submit(write):
            record["bundle"] = None
            record["bundle_error"] = "the store's writer refused it: busy or stalled"
            self._finish(incident, record, [])

    def _write_bundle(
        self,
        incident_id: str,
        lines: _Lines,
        status: str,
        sealed_at: int,
        protected: frozenset[str],
    ) -> tuple[str | None, str | None]:
        """The bundle's path, or None and why it could not be written."""
        reserve = _RESERVE_BASE + lines.nbytes
        try:
            writer = self.store.new_bundle(incident_id, reserve, protected=protected)
        except OSError as exc:  # creating the bundle: ENOSPC, EACCES
            return None, _error_text(exc)
        if writer is None:
            return None, f"the store cannot hold {reserve} bytes within its limits"
        try:
            with writer.file("incident.jsonl") as out:
                for line in lines:
                    out.write(line)
            writer.publish(
                status=status,
                complete=status == STATUS_COMPLETED,
                sealed_at_ns=sealed_at,
            )
        except (BudgetExceeded, OSError) as exc:
            with contextlib.suppress(OSError):
                writer.abandon()
            return None, _error_text(exc)
        return f"incidents/{incident_id}", None

    def _finish(
        self,
        incident: OpenIncident,
        record: dict[str, Any],
        pruned: list[PrunedBundle],
    ) -> None:
        if pruned and self._on_pruned is not None:
            self._on_pruned(pruned)
        persisted = record.get("bundle") is not None
        self.persisted += int(persisted)
        self.persist_failures += int(not persisted)
        self.recorded += 1
        self.recorded_counting += int(incident.counts_toward_exit)
        capture = record["capture"]["status"]
        self.stats.add("incidents_total", labels=(record["trigger"]["kind"], capture))
        for name in ("pre_window", "post_window"):
            window = record[name]
            self.stats.add(
                "incident_windows_total",
                labels=(
                    name.removesuffix("_window"),
                    window["fidelity"],
                    window["detail_collected"] or NO_DETAIL,
                ),
            )
        self.sealed.append(
            {
                "incident_id": incident.incident_id,
                "kind": record["trigger"]["kind"],
                "trigger_id": record["trigger"]["trigger_id"],
                "counts_toward_exit": incident.counts_toward_exit,
                "persisted": persisted,
                "bundle": record.get("bundle"),
                "bundle_error": record.get("bundle_error"),
            }
        )
        self._emit(record)

    # ------------------------------------------------------------ records

    def _trigger_record(self, result: TickResult) -> dict[str, Any]:
        state = result.transition
        assert state is not None
        evaluation = result.evaluation
        spec = result.spec
        return trigger_fields(
            trigger_id=spec.trigger_id,
            kind=spec.kind,
            reason=_reason(result),
            fired_at_ns=self.clock.to_wall(result.at_ns),
            counts_toward_exit=bool(spec.counts_toward_exit),
            threshold=evaluation.threshold,
            observed=evaluation.observed,
            observed_bounds=evaluation.observed_bounds,
            samples=evaluation.samples,
            pending_since_ns=(
                self.clock.to_wall(state.pending_since_ns)
                if state.pending_since_ns is not None
                else None
            ),
            sustained_ns=state.accumulated_ns,
            window_seconds=spec.sustain.window,
            hold_seconds=spec.sustain.hold,
            clear_seconds=spec.sustain.clear,
            detail=evaluation.detail,
        )

    def _incident_record(
        self, incident: OpenIncident, held: _HeldScrapes, end_mono: int
    ) -> dict[str, Any]:
        record = envelope(
            INCIDENT,
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            timestamp_ns=self.clock.to_wall(end_mono),
        )
        detected = incident.detected_at_mono
        record.update(
            incident_id=incident.incident_id,
            detected_at_ns=self.clock.to_wall(detected),
            owner=self.identity.owner,
            trigger=incident.trigger,
            joined_triggers=incident.joined,
            counts_toward_exit=incident.counts_toward_exit,
            last_informative_before_capture=None,
            first_informative_after_mask=None,
            pre_window=self._window(held, incident.pre_start_mono, detected),
            # Requested in full: one cut short by the watch's end is partial.
            post_window=self._window(
                held, detected, end_mono, requested_end=incident.post_end_mono
            ),
            deep_window=None,
            # Deep capture arrives in a later step; health never captures.
            capture=capture_fields(
                (
                    "health_only"
                    if incident.trigger["kind"] == KIND_HEALTH
                    else "disabled"
                ),
                owner=self.identity.owner,
            ),
            possibly_self_induced=False,
            self_induced_reason=None,
            outstanding_requests=None,
            perturbation_id=None,
            rearm_basis=None,
            suppressed=dict(incident.suppressed),
            bundle=None,
            bundle_error=None,
            traces=[],
            attachment_ids=[f"incident:{incident.incident_id}"],
            request_refs=[],
            request_ref_total=0,
            engine_span_links=[],
            loss=self._loss_since(incident),
            trace_loss=None,
        )
        return record

    def _window(
        self,
        scrapes: _HeldScrapes,
        start_mono: int,
        end_mono: int,
        *,
        requested_end: int | None = None,
    ) -> dict[str, Any]:
        """A window's bounds and how much of it the scrapes cover.

        Partial when a scrape failed, when the history began late or the
        window was cut short of ``requested_end``, or when ticks were missed:
        fewer scrapes were attempted than one a tick, give or take one.
        """
        inside = [
            index
            for index, (stamp, _blob) in enumerate(scrapes.items)
            if start_mono <= stamp.mono_ns <= end_mono
        ]
        ok = sum(scrapes.ok[index] for index in inside)
        requested = ((requested_end or end_mono) - start_mono) / _NS
        first = scrapes.items[inside[0]][0].mono_ns if inside else end_mono
        held = (end_mono - first) / _NS
        failed = len(inside) - ok
        expected = int(requested / self._tick_seconds)
        covered = held + self._slack_seconds >= requested
        fidelity = _fidelity(ok, failed, covered, len(inside), expected)
        return {
            "start_ns": self.clock.to_wall(start_mono),
            "end_ns": self.clock.to_wall(end_mono),
            "clock_domain": self.identity.clock_domain,
            "fidelity": fidelity,
            "detail_requested": "metrics",
            "detail_collected": "metrics" if ok else None,
            "fidelity_detail": {
                "scrapes": {
                    "attempted": len(inside),
                    "expected": expected,
                    "ok": ok,
                    "failed": failed,
                    "requested_seconds": requested,
                    "held_seconds": held,
                }
            },
        }

    def _loss_since(self, incident: OpenIncident) -> dict[str, int | None]:
        loss = empty_loss()
        now = self._loss()
        for key, value in now.items():
            loss[key] = max(0, value - incident.loss_at_open.get(key, 0))
        return loss

    def _bundle_head(self, incident: OpenIncident, end_mono: int) -> list[bytes]:
        """The bundle's lines before its scrapes: session, artifact, windows."""
        identity = self.identity
        artifact = ArtifactIdentityEvent(
            context=CorrelationContext(
                run_id=identity.run_id,
                session_id=identity.session_id,
                producer_id="stormlog.infer.watch",
                source="stormlog.infer.watch",
                source_version=__version__,
                host=identity.host,
                clock_domain=identity.clock_domain,
                clock_kind="wall",
                collection_mode="passive",
                provenance="observed",
            ),
            event_id="artifact",
            metadata={"boot_id": identity.boot_id, "incident_id": incident.incident_id},
            artifact_kind="inference_jsonl",
            created_at_ns=self.clock.to_wall(incident.detected_at_mono),
        )
        records: list[Mapping[str, Any]] = [
            {
                "schema_version": 1,
                "event_type": "infer.session",
                "session_id": identity.session_id,
                "timestamp_ns": self.clock.to_wall(incident.pre_start_mono),
                "status": "completed",
                "config": {"source": "stormlog.infer.watch", "collection": "passive"},
            },
            artifact.to_record(),
        ]
        for name, start, end in (
            ("pre", incident.pre_start_mono, incident.detected_at_mono),
            ("post", incident.detected_at_mono, end_mono),
        ):
            records.append(
                {
                    "schema_version": 1,
                    "event_type": WINDOW_EVENT,
                    "session_id": identity.session_id,
                    "run_id": identity.run_id,
                    "incident_id": incident.incident_id,
                    "window": name,
                    "start_ns": self.clock.to_wall(start),
                    "end_ns": self.clock.to_wall(end),
                }
            )
        return [_line(r) for r in records]


@dataclass(frozen=True)
class _Lines:
    """A bundle's ``incident.jsonl``: its head, the held scrapes expanded one
    at a time as they are written, and the incident record."""

    head: list[bytes]
    scrapes: _HeldScrapes
    tail: bytes

    @property
    def nbytes(self) -> int:
        head = sum(len(line) for line in self.head)
        return head + self.scrapes.expanded_bytes + len(self.tail)

    def __iter__(self) -> Iterator[bytes]:
        yield from self.head
        for _stamp, blob in self.scrapes.items:
            yield expand(blob) + b"\n"
        yield self.tail


def _line(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fidelity(
    ok: int, failed: int, covered: bool, attempted: int, expected: int
) -> str:
    """Missing with no good scrape; partial with a failed one, with time not
    covered, or with fewer scrapes attempted than one a tick, give or take one."""
    if not ok:
        return "missing"
    if failed or not covered or attempted < expected - 1:
        return "partial"
    return "complete"


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _reason(result: TickResult) -> str:
    evaluation = result.evaluation
    observed = evaluation.observed
    threshold = evaluation.threshold
    if observed is None or threshold is None:
        return f"{result.spec.trigger_id} sustained"
    return (
        f"{result.spec.trigger_id}: {observed:g} against {threshold:g} for "
        f"{result.spec.sustain.hold:g} s"
    )


__all__ = [
    "Identity",
    "IncidentManager",
    "OpenIncident",
    "WINDOW_EVENT",
    "WatchClock",
]
