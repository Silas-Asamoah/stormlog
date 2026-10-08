"""An engine's imported steps in order, and the facts classes read off them.

A step is one ``schedule()`` call and the output processing that completed
it: when it started and completed on the engine's monotonic clock, how many
requests it ran, how many tokens it scheduled, how many it admitted for the
first time, and how many it preempted. Pauses and resets come from the
import's stages and from the dated facts it kept in the epoch's summary.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from statistics import median
from typing import Any

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
    members: int
    total_tokens: int
    admitted: int  # requests first scheduled in it
    preempted: int
    line: Line


@dataclass
class Steps:
    """One engine's steps, in schedule order."""

    producer: str
    steps: list[Step] = field(default_factory=list)
    pauses: list[tuple[int, str | None, str | None, Line | None]] = field(
        default_factory=list
    )  # (mono, from, to, line)

    def __post_init__(self) -> None:
        self.steps.sort(key=lambda step: step.start_ns)
        self._starts = [step.start_ns for step in self.steps]

    def between(self, start_ns: int, end_ns: int) -> list[Step]:
        """Steps whose schedule call started in [start, end]."""
        low = bisect_left(self._starts, start_ns)
        high = bisect_right(self._starts, end_ns)
        return self.steps[low:high]

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

    def paused_intervals(self, end_of_time: int) -> list[tuple[int, int, str]]:
        """Spans spent in a pause state, from the pause transitions."""
        spans: list[tuple[int, int, str]] = []
        current: tuple[int, str] | None = None
        for at, _before, after, _line in sorted(self.pauses, key=lambda p: p[0]):
            if current is not None:
                spans.append((current[0], at, current[1]))
                current = None
            if after in PAUSED_STATES:
                current = (at, after)
        if current is not None:
            spans.append((current[0], end_of_time, current[1]))
        return spans


def loop_steps(view: RunView, producer: str) -> list[LoopStep]:
    """The engine's imported steps in the engine-loop rules' own form, so
    the offline diagnosis finds the stalls an online trigger would. A step's
    members are the attempts its written memberships name; withheld
    members are missing."""
    members, finished = _attempt_sets(view, producer)
    found = (
        _loop_step(iteration, members, finished)
        for ref, (_, iteration) in view.iterations.items()
        if ref.producer_id == producer
    )
    return sorted(
        (step for step in found if step is not None),
        key=lambda step: step.start_mono_ns,
    )


def _attempt_sets(
    view: RunView, producer: str
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Per step, the attempts it ran and those that finished in it."""
    members: dict[str, set[str]] = {}
    finished: dict[str, set[str]] = {}
    for execution in view.executions.values():
        if execution.producer != producer or execution.attempt is None:
            continue
        for _, membership in execution.memberships:
            step = membership.iteration_ref.id
            members.setdefault(step, set()).add(execution.attempt.id)
            finish = membership.metadata.get("finish")
            if isinstance(finish, dict) and finish.get("in_step"):
                finished.setdefault(step, set()).add(execution.attempt.id)
    return members, finished


def _loop_step(
    iteration: IterationEvent,
    members: dict[str, set[str]],
    finished: dict[str, set[str]],
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
        members=frozenset(members.get(step, ())),
        total_tokens=_count(data.get("total_tokens")),
        finished=frozenset(finished.get(step, ())),
    )


def steps_of(view: RunView, producer: str) -> Steps:
    """The steps, pause transitions and admissions of one engine."""
    admitted = _first_sightings(view, producer)
    steps = [
        _step(line, iteration, admitted.get(iteration.iteration_ref.id, 0))
        for ref, (line, iteration) in view.iterations.items()
        if ref.producer_id == producer and iteration.start_ns is not None
    ]
    return Steps(producer, steps, _pauses(view, producer))


def _step(line: Line, iteration: IterationEvent, admitted: int) -> Step:
    metadata = iteration.metadata
    return Step(
        iteration=iteration.iteration_ref.id,
        start_ns=iteration.start_ns or 0,
        end_ns=iteration.end_ns,
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


def _pauses(
    view: RunView, producer: str
) -> list[tuple[int, str | None, str | None, Line | None]]:
    """Pause transitions: the import's stages, and the dated facts it kept
    for transitions before any step was written."""
    found: dict[tuple[str, int], tuple[int, str | None, str | None, Line | None]] = {}
    for line, stage in view.stages:
        if stage.name != "engine.pause_transition" or stage.start_ns is None:
            continue
        if stage.stage_ref.producer_id != producer:
            continue
        key = (str(stage.metadata.get("epoch")), _count(stage.metadata.get("seq")))
        found[key] = (
            stage.start_ns,
            _text(stage.metadata.get("from")),
            _text(stage.metadata.get("to")),
            line,
        )
    for epoch in view.engines.values():
        if epoch.producer != producer:
            continue
        for fact in epoch.unanchored:
            if fact.get("name") == "engine.pause_transition" and isinstance(
                fact.get("start_mono_ns"), int
            ):
                key = (epoch.epoch, _count(fact.get("seq")))
                found.setdefault(
                    key,
                    (
                        fact["start_mono_ns"],
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


__all__ = ["PAUSED_STATES", "Step", "Steps", "loop_steps", "steps_of"]
