"""Dated stages for what happens to requests between steps.

The hook's ``scheduled`` records list the requests each ``schedule()`` call
preempted, and its ``cache_reset`` and ``pause`` records say when the prefix
cache was emptied and the scheduler paused. The import writes them as
``infer.stage`` records, an existing generic type, so older readers keep
reading the artifact:

- ``engine.preempted``: one per attempt a step preempted to free memory, on
  that step, dated by its ``schedule()`` call;
- ``engine.cache_reset``: one per reset, and ``engine.preempted_by_reset``
  for each request it preempted, dated by the reset's call;
- ``engine.pause_transition``: one per pause-state change;
- ``engine.profile_call``: one per ``EngineCore.profile`` call, dated by
  its own bracket, the time it held the engine loop.

vLLM lists a reset's preemptions in the next step's ``preempted`` too, so a
step's preemptions that a reset since the previous step made are the reset's,
not the step's. A point stage refers to the last iteration written before it
completed; one with none is a dated fact in the epoch's summary instead. A
stage is written only once the records it refers to are, and once: its ID is
fixed by the epoch and the raw record's ``seq``.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from .correlation_events import EntityRef, StageEvent
from .vllm_execution_log import STATE_ENDED, STATE_GONE, RawRecord

if TYPE_CHECKING:
    from .vllm_execution import Iteration, _EpochReducer

PREEMPTED = "engine.preempted"
PREEMPTED_BY_RESET = "engine.preempted_by_reset"
CACHE_RESET = "engine.cache_reset"
PAUSE_TRANSITION = "engine.pause_transition"
PROFILE_CALL = "engine.profile_call"
_STAMP = ("wall_ns", "mono_ns", "wall_after_ns")


class StageBuilder:
    """The stages one import of one epoch writes, and what it could not."""

    def __init__(
        self,
        reducer: _EpochReducer,
        kept: list[Iteration],
        pending: list[Iteration] | None = None,
    ) -> None:
        self.reducer = reducer
        self.kept = kept
        self.pending = pending or []
        producer = reducer.producer
        self.written_iterations = {
            ref.id
            for ref in reducer.facts.existing_iterations
            if ref.producer_id == producer
        } | {item.iteration for item in kept}
        self.written_attempts = set(reducer.facts.existing_attempts) | {
            reducer._attempt_for(execution)
            for execution in reducer.executions.values()
            if execution.memberships and not execution.withheld
        }
        observes = (reducer.hello or {}).get("observes")
        self.reset_observed = isinstance(observes, list) and "cache_reset" in observes
        self.live = reducer.epoch.state not in (STATE_ENDED, STATE_GONE)
        self.events: list[StageEvent] = []
        self.counts: Counter[str] = Counter()
        self.unanchored: list[dict[str, Any]] = []
        # Raw records a later import must read again: their stages wait on a
        # request record or on the step that tells their preemptions apart.
        self.holds: list[int] = []
        self._completions = _completions(reducer.epoch.records)
        self._earlier = _earlier_written(reducer)

    def build(self) -> list[StageEvent]:
        by_reset = self.reducer.reset_preemptions
        if self.live:
            # A reset is read again until the step that lists its preemptions
            # is final: unread, or read but pending, that step would be read
            # next time without the reset, and its preemptions taken for the
            # step's own.
            if by_reset.unlisted_from is not None:
                self.holds.append(by_reset.unlisted_from)
            for item in self.pending:
                first = by_reset.first_reset.get(item.scheduled.seq)
                if first is not None:
                    self.holds.append(first)
        for item in self.kept:
            self._step_preemptions(item, by_reset.of(item))
        for record in self.reducer.epoch.records:
            if record.kind == "cache_reset":
                self._cache_reset(record)
            elif record.kind == "pause":
                self._pause(record)
            elif record.kind == "engine_profile":
                self._profile_call(record)
        return self.events

    def summary(self) -> dict[str, Any]:
        return {
            "stages": dict(sorted(self.counts.items())),
            "unanchored": self.unanchored,
        }

    # ------------------------------------------------------------ preemptions
    def _step_preemptions(self, item: Iteration, by_reset: set[str]) -> None:
        data = item.scheduled.data
        start, end = _integer(data.get("start_mono_ns")), _integer(
            data.get("end_mono_ns")
        )
        for internal in sorted(set(_texts(data.get("preempted"))) - by_reset):
            self._attempt_stage(
                PREEMPTED,
                f"preempted/{item.iteration}",
                internal,
                at_mono_ns=start,
                iteration=item.iteration,
                span=(start, end),
                record=item.scheduled,
                source_seq=item.final_seq,
                details={
                    "by": "schedule",
                    "reset_observed": self.reset_observed,
                    **_prefixed(data, "start_"),
                    **_prefixed(data, "end_"),
                },
                can_wait=False,
            )

    def _attempt_stage(
        self,
        name: str,
        key: str,
        internal: str,
        *,
        at_mono_ns: int | None,
        iteration: str | None,
        span: tuple[int | None, int | None],
        record: RawRecord,
        source_seq: int | None,
        details: dict[str, Any],
        can_wait: bool,
    ) -> None:
        reducer = self.reducer
        execution = reducer._attempt_execution(internal, at_mono_ns)
        if reducer.withholds(execution):
            self.counts["withheld"] += 1
            return
        attempt = reducer._attempt_for(execution)
        if attempt not in self.written_attempts:
            if can_wait and self.live and execution.key in reducer.waiting_keys:
                self.holds.append(record.seq)
            else:
                self.counts["unreferenced"] += 1
            return
        # Written only with its request record: an import must reach both.
        needed = [source_seq, reducer.request_source(execution)]
        self._emit(
            name,
            f"{key}/{attempt.id}",
            record,
            request_ref=reducer._request_ref(execution),
            iteration=iteration,
            span=span,
            metadata={
                "source_seq_max": max(
                    (s for s in needed if s is not None), default=None
                ),
                "attempt": attempt.id,
                "ownership": execution.binding.ownership,
                **details,
            },
        )

    # ------------------------------------------------------------ point stages
    def _cache_reset(self, record: RawRecord) -> None:
        data = record.data
        start, end = _integer(data.get("start_mono_ns")), _integer(
            data.get("end_mono_ns")
        )
        anchor = self._anchor(record.seq)
        details = {
            "reset_running_requests": bool(data.get("reset_running_requests")),
            "reset_connector": bool(data.get("reset_connector")),
            "succeeded": data.get("succeeded"),
            "raised": bool(data.get("raised")),
            "running": len(_texts(data.get("running"))),
            **_prefixed(data, "start_"),
            **_prefixed(data, "end_"),
        }
        self._point(CACHE_RESET, "cache_reset", record, anchor, (start, end), details)
        if not data.get("reset_running_requests"):
            return
        # vLLM preempts every running request before it checks the reset
        # succeeded, so the preemptions stand even when it failed.
        for internal in sorted(set(_texts(data.get("running")))):
            self._attempt_stage(
                PREEMPTED_BY_RESET,
                f"preempted_by_reset/{record.seq}",
                internal,
                at_mono_ns=start,
                iteration=anchor,
                span=(start, end),
                record=record,
                source_seq=record.seq,
                details={"by": "cache_reset", "reset_seq": record.seq},
                can_wait=True,
            )

    def _pause(self, record: RawRecord) -> None:
        data = record.data
        at = _integer(data.get("mono_ns"))
        details = {
            "from": data.get("from"),
            "to": data.get("to"),
            **_prefixed(data, ""),
        }
        self._point(
            PAUSE_TRANSITION,
            "pause",
            record,
            self._anchor(record.seq),
            (at, at),
            details,
        )

    def _profile_call(self, record: RawRecord) -> None:
        data = record.data
        start, end = _integer(data.get("start_mono_ns")), _integer(
            data.get("end_mono_ns")
        )
        details = {
            "is_start": bool(data.get("is_start")),
            "raised": bool(data.get("raised")),
            **_prefixed(data, "start_"),
            **_prefixed(data, "end_"),
        }
        self._point(
            PROFILE_CALL,
            "profile_call",
            record,
            self._anchor(record.seq),
            (start, end),
            details,
        )

    def _point(
        self,
        name: str,
        key: str,
        record: RawRecord,
        anchor: str | None,
        span: tuple[int | None, int | None],
        details: dict[str, Any],
    ) -> None:
        if anchor is None:
            start, end = span
            self.unanchored.append(
                {
                    "name": name,
                    "seq": record.seq,
                    "start_mono_ns": start,
                    "end_mono_ns": end,
                    **details,
                }
            )
            self.counts["unanchored"] += 1
            return
        self._emit(
            name,
            f"{key}/{record.seq}",
            record,
            request_ref=None,
            iteration=anchor,
            span=span,
            metadata={"source_seq_max": record.seq, **details},
        )

    def _anchor(self, seq: int) -> str | None:
        """The last iteration written whose step completed before ``seq``:
        one whose completion this read holds, or one an earlier import wrote
        from records that all lie before this read. vLLM completes steps in
        the order it scheduled them, so the highest number is the last."""
        numbers = [
            number
            for completed, number in self._completions
            if completed < seq and str(number) in self.written_iterations
        ]
        if self._earlier is not None:
            numbers.append(self._earlier)
        return str(max(numbers)) if numbers else None

    def _emit(
        self,
        name: str,
        key: str,
        record: RawRecord,
        *,
        request_ref: EntityRef | None,
        iteration: str | None,
        span: tuple[int | None, int | None],
        metadata: dict[str, Any],
    ) -> None:
        reducer = self.reducer
        event_id = f"stage:{reducer.producer}:{key}"
        if event_id in reducer.facts.existing_stages:
            self.counts["already_imported"] += 1
            return
        start, end = span
        if start is not None and end is not None and end < start:
            end = None
        self.events.append(
            StageEvent(
                context=reducer._context(),
                event_id=event_id,
                metadata={"epoch": reducer.epoch.epoch, "seq": record.seq, **metadata},
                stage_ref=EntityRef(reducer.producer, key),
                name=name,
                request_ref=request_ref,
                iteration_ref=(
                    None
                    if iteration is None
                    else EntityRef(reducer.producer, iteration)
                ),
                start_ns=start,
                end_ns=end,
            )
        )
        self.counts[name] += 1


class ResetPreemptions:
    """Per step, the preemptions a cache reset since the previous step made:
    vLLM lists them in that step's ``preempted`` with its own."""

    def __init__(self, records: list[RawRecord]) -> None:
        self.by_step: dict[int, set[str]] = {}
        # Per listing step (by its scheduled seq), the first reset it lists.
        self.first_reset: dict[int, int] = {}
        since: set[str] = set()
        first: int | None = None
        for record in records:
            if record.kind == "cache_reset" and record.data.get(
                "reset_running_requests"
            ):
                running = set(_texts(record.data.get("running")))
                if running and first is None:
                    # A reset with nothing running (vLLM's default pause
                    # resets after aborting all) has nothing to list.
                    first = record.seq
                since |= running
            elif record.kind == "scheduled":
                if since:
                    self.by_step[record.seq] = since
                if first is not None:
                    self.first_reset[record.seq] = first
                since, first = set(), None
        # The first reset whose preemptions no step read so far lists.
        self.unlisted_from: int | None = first

    def of(self, item: Iteration) -> set[str]:
        return self.by_step.get(item.scheduled.seq, set())


def _completions(records: list[RawRecord]) -> list[tuple[int, int]]:
    """(seq, iteration number) of every completion read."""
    found = []
    for record in records:
        number = record.data.get("iteration")
        if record.kind == "completed" and isinstance(number, str) and number.isdigit():
            found.append((record.seq, int(number)))
    return found


def _earlier_written(reducer: _EpochReducer) -> int | None:
    """The last iteration an earlier import wrote from records that all lie
    before this read: none of its records is read again."""
    in_read = {str(r.data.get("iteration")) for r in reducer.epoch.records}
    numbers = [
        int(ref.id)
        for ref in reducer.facts.existing_iterations
        if ref.producer_id == reducer.producer
        and ref.id.isdigit()
        and ref.id not in in_read
    ]
    return max(numbers, default=None)


def _prefixed(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    """A bracketed stamp's wall reads, which the stage's own span (on the
    monotonic clock) leaves out."""
    return {
        f"{prefix}{name}": data.get(f"{prefix}{name}")
        for name in _STAMP
        if name != "mono_ns"
    }


def _texts(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "CACHE_RESET",
    "PAUSE_TRANSITION",
    "PREEMPTED",
    "PREEMPTED_BY_RESET",
    "PROFILE_CALL",
    "ResetPreemptions",
    "StageBuilder",
]
