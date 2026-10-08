"""An engine's imported steps in order, and the facts classes read off them.

A step is one ``schedule()`` call and the output processing that completed
it: when it started and completed on the engine's monotonic clock, how many
requests it ran, how many tokens it scheduled, how many it admitted for the
first time, how many it preempted, and how many slots the step before freed
that it did not refill. Pauses and resets come from the import's stages and
from the dated facts it kept in the epoch's summary.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Sequence

from .correlation_events import IterationEvent
from .diagnosis_inputs import Line
from .diagnosis_join import RunView
from .diagnosis_loop import Step as LoopStep

PAUSED_STATES = frozenset({"PAUSED_NEW", "PAUSED_ALL"})


@dataclass(frozen=True)
class Step:
    iteration: str
    start_ns: int
    end_ns: int | None  # completion; None when it never completed
    completed_wall_ns: int | None
    members: int
    total_tokens: int
    admitted: int  # requests first scheduled in it
    preempted: int
    line: Line
    # Requests that finished in the step before and do not run in this one:
    # under async scheduling a slot freed at a step's end is refilled one
    # step late, so a full engine's step can run that many short.
    refill: int = 0


@dataclass(frozen=True)
class Pause:
    """A change of the scheduler's pause state."""

    mono_ns: int
    wall_ns: int | None
    before: str | None
    after: str | None
    line: Line | None  # None for a dated fact from an import summary


@dataclass
class Steps:
    """One engine's steps, in schedule order."""

    producer: str
    steps: list[Step] = field(default_factory=list)
    pauses: list[Pause] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.steps.sort(key=lambda step: step.start_ns)
        self._starts = [step.start_ns for step in self.steps]

    def between(self, start_ns: int, end_ns: int) -> list[Step]:
        """Steps whose schedule call started in [start, end]."""
        low = bisect_left(self._starts, start_ns)
        high = bisect_right(self._starts, end_ns)
        return self.steps[low:high]

    def within(self, intervals: Sequence[tuple[int, int]]) -> list[Step]:
        """Steps whose schedule call started inside the union of the
        half-open ``intervals`` [start, end), each once, in order: for a
        wait ending at the call that ran the request, the steps run while it
        still waited."""
        return [
            step
            for start, end in merge_intervals(intervals)
            for step in self.steps[
                bisect_left(self._starts, start) : bisect_left(self._starts, end)
            ]
        ]

    def cadence_before(self, at_ns: int, window_ns: int) -> float | None:
        """Median interval between completions in the ``window_ns`` before
        ``at_ns``: the engine's causal step cadence."""
        done = [
            step.end_ns
            for step in self.between(at_ns - window_ns, at_ns)
            if step.end_ns is not None and step.end_ns <= at_ns and step.members
        ]
        gaps = [b - a for a, b in zip(done, done[1:]) if b > a]
        return float(median(gaps)) if gaps else None

    def paused_intervals(
        self, end_of_time: int, *, wall: bool = False
    ) -> list[tuple[int, int, str]]:
        """Spans spent in a pause state, from the pause transitions, on the
        engine's monotonic clock or, with ``wall``, its wall clock."""
        spans: list[tuple[int, int, str]] = []
        current: tuple[int, str] | None = None
        for pause in sorted(self.pauses, key=lambda p: p.mono_ns):
            at = pause.wall_ns if wall else pause.mono_ns
            if at is None:
                continue
            if current is not None:
                spans.append((current[0], at, current[1]))
                current = None
            if pause.after in PAUSED_STATES:
                current = (at, pause.after)
        if current is not None:
            spans.append((current[0], end_of_time, current[1]))
        return spans


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The union of ``intervals`` as disjoint intervals, in order; touching
    intervals join."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def loop_steps(view: RunView, producer: str) -> list[LoopStep]:
    """The engine's imported steps in the engine-loop rules' own form, so
    the offline diagnosis finds the stalls an online trigger would. A step's
    members are the attempts its written memberships name; withheld
    members are missing."""
    sets, prompts = _attempt_sets(view, producer), _prompts(view, producer)
    found = (
        _loop_step(iteration, sets, prompts)
        for ref, (_, iteration) in view.iterations.items()
        if ref.producer_id == producer
    )
    return sorted(
        (step for step in found if step is not None),
        key=lambda step: step.start_mono_ns,
    )


AttemptSets = dict[str, dict[str, set[str]]]


def _attempt_sets(view: RunView, producer: str) -> AttemptSets:
    """Per step, the attempts it ran ("members"), those that finished in it
    ("finished") and its streaming-input members ("streaming")."""
    sets: AttemptSets = {"members": {}, "finished": {}, "streaming": {}}
    for execution in view.executions.values():
        if execution.producer != producer or execution.attempt is None:
            continue
        for _, membership in execution.memberships:
            step, attempt = membership.iteration_ref.id, execution.attempt.id
            sets["members"].setdefault(step, set()).add(attempt)
            data = membership.metadata
            finish = data.get("finish")
            if isinstance(finish, dict) and finish.get("in_step"):
                sets["finished"].setdefault(step, set()).add(attempt)
            if data.get("resumable") is True:
                sets["streaming"].setdefault(step, set()).add(attempt)
    return sets


def _prompts(view: RunView, producer: str) -> dict[str, dict[str, int]]:
    """Per step, each member's prompt length then: a streaming-input
    request's grows when its client's next turn arrives."""
    prompts: dict[str, dict[str, int]] = {}
    for execution in view.executions.values():
        if execution.producer != producer or execution.attempt is None:
            continue
        for _, membership in execution.memberships:
            tokens = _optional(membership.metadata.get("prompt_tokens"))
            if tokens is not None:
                step = prompts.setdefault(membership.iteration_ref.id, {})
                step[execution.attempt.id] = tokens
    return prompts


def _loop_step(
    iteration: IterationEvent, sets: AttemptSets, prompts: dict[str, dict[str, int]]
) -> LoopStep | None:
    data = iteration.metadata
    start_wall = _optional(data.get("start_wall_ns"))
    end_wall = _optional(data.get("schedule_end_wall_ns"))
    end_mono = _optional(data.get("schedule_end_mono_ns"))
    if iteration.start_ns is None or None in (start_wall, end_wall, end_mono):
        return None
    step = iteration.iteration_ref.id
    return LoopStep(
        iteration=step,
        start_wall_ns=start_wall or 0,
        start_mono_ns=iteration.start_ns,
        end_wall_ns=end_wall or 0,
        end_mono_ns=end_mono or 0,
        completed_wall_ns=_optional(data.get("completed_wall_ns")),
        completed_mono_ns=iteration.end_ns,
        members=frozenset(sets["members"].get(step, ())),
        total_tokens=_count(data.get("total_tokens")),
        finished=frozenset(sets["finished"].get(step, ())),
        prompts=tuple(sorted(prompts.get(step, {}).items())),
        streaming=frozenset(sets["streaming"].get(step, ())),
    )


def steps_of(view: RunView, producer: str) -> Steps:
    """The steps, pause transitions and admissions of one engine."""
    admitted = _first_sightings(view, producer)
    steps = sorted(
        (
            _step(line, iteration, admitted.get(iteration.iteration_ref.id, 0))
            for ref, (line, iteration) in view.iterations.items()
            if ref.producer_id == producer and iteration.start_ns is not None
        ),
        key=lambda step: step.start_ns,
    )
    sets = _attempt_sets(view, producer)
    owed = refills(
        [frozenset(sets["members"].get(step.iteration, ())) for step in steps],
        [frozenset(sets["finished"].get(step.iteration, ())) for step in steps],
    )
    steps = [replace(step, refill=n) for step, n in zip(steps, owed)]
    return Steps(producer, steps, _pauses(view, producer))


def refills(
    members: Sequence[frozenset[str]], finished: Sequence[frozenset[str]]
) -> list[int]:
    """Per step, in schedule order, the requests that finished in the step
    before and are not among its members. A request vLLM could not foresee
    finishing (an end of sequence) was already planned into the next step,
    where it is discarded, so it is a member there and not counted twice."""
    owed = [0] * len(members)
    for index in range(1, len(members)):
        owed[index] = len(finished[index - 1] - members[index])
    return owed


def _step(line: Line, iteration: IterationEvent, admitted: int) -> Step:
    metadata = iteration.metadata
    return Step(
        iteration=iteration.iteration_ref.id,
        start_ns=iteration.start_ns or 0,
        end_ns=iteration.end_ns,
        completed_wall_ns=_optional(metadata.get("completed_wall_ns")),
        members=_count(metadata.get("members")),
        total_tokens=_count(metadata.get("total_tokens")),
        admitted=admitted,
        preempted=_count(metadata.get("preempted")),
        line=line,
    )


def _first_sightings(view: RunView, producer: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for execution in view.executions.values():
        if execution.producer != producer:
            continue
        for _, membership in execution.memberships:
            if membership.metadata.get("sighting") == "first":
                step = membership.iteration_ref.id
                counts[step] = counts.get(step, 0) + 1
    return counts


def _pauses(view: RunView, producer: str) -> list[Pause]:
    """Pause transitions: the import's stages, and the dated facts it kept
    for transitions before any step was written."""
    found: dict[tuple[str, int], Pause] = {}
    for line, stage in view.stages:
        if stage.name != "engine.pause_transition" or stage.start_ns is None:
            continue
        if stage.stage_ref.producer_id != producer:
            continue
        data = stage.metadata
        key = (str(data.get("epoch")), _count(data.get("seq")))
        found[key] = Pause(
            stage.start_ns,
            _optional(data.get("wall_ns")),
            _text(data.get("from")),
            _text(data.get("to")),
            line,
        )
    for epoch in view.engines.values():
        if epoch.producer != producer:
            continue
        for fact in epoch.unanchored:
            mono = fact.get("start_mono_ns")
            if fact.get("name") == "engine.pause_transition" and isinstance(mono, int):
                found.setdefault(
                    (epoch.epoch, _count(fact.get("seq"))),
                    Pause(
                        mono,
                        _optional(fact.get("wall_ns")),
                        _text(fact.get("from")),
                        _text(fact.get("to")),
                        None,
                    ),
                )
    return list(found.values())


def _optional(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "PAUSED_STATES",
    "Pause",
    "Step",
    "Steps",
    "loop_steps",
    "merge_intervals",
    "refills",
    "steps_of",
]
