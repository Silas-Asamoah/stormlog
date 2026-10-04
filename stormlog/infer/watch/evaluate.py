"""Evaluate every configured trigger once per tick.

Each trigger pairs a predicate with a :class:`~.triggers.Sustain`. A window
predicate is asked about the window :func:`~.predicates.select_window` cuts;
a health predicate about the history's tail. An evaluation whose window
overlaps one of the watcher's own perturbation intervals is classified as
masked, whatever its value, so a capture's pause can neither advance nor
reset a trigger. A trigger on a family vLLM records at a request's
completion (e2e, TPOT, the ``request_*`` histograms) widens its window by
the completion horizon first, since requests the pause delayed finish up to
one request lifetime later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from .predicates import (
    Entry,
    Evaluation,
    WindowPredicate,
    overlaps,
    select_window,
)
from .triggers import DATA_GAP, MASKED, Sustain, Transition, TriggerState

KIND_METRIC = "metric"
KIND_SIGNAL = "signal"
KIND_HEALTH = "health"
KIND_SLO = "slo"
KIND_TEST = "test"
KINDS = (KIND_METRIC, KIND_SIGNAL, KIND_HEALTH, KIND_SLO, KIND_TEST)
ACTION_RECORD = "record"
ACTION_DEEP_CAPTURE = "deep_capture"
WHEN_ALWAYS = "always"
WHEN_UNEXPLAINED = "unexplained"
_NS = 1_000_000_000


@runtime_checkable
class HistoryPredicate(Protocol):
    """A question about the history's last ``tail_scrapes`` scrapes rather
    than one window."""

    @property
    def tail_scrapes(self) -> int: ...

    def evaluate_history(self, history: Sequence[Entry]) -> Evaluation: ...


@dataclass(frozen=True)
class TriggerSpec:
    """One configured trigger."""

    trigger_id: str
    kind: str
    sustain: Sustain
    predicate: WindowPredicate | HistoryPredicate
    action: str = ACTION_RECORD
    deep_capture_when: str = WHEN_ALWAYS
    counts_toward_exit: bool = True
    completion_recorded: bool = False

    def __post_init__(self) -> None:
        if not self.trigger_id:
            raise ValueError("trigger_id must be non-empty")
        if self.kind not in KINDS or self.kind == KIND_TEST:
            raise ValueError(f"trigger kind {self.kind!r} is not evaluated here")
        if self.action not in (ACTION_RECORD, ACTION_DEEP_CAPTURE):
            raise ValueError(f"unknown action {self.action!r}")
        if self.deep_capture_when not in (WHEN_ALWAYS, WHEN_UNEXPLAINED):
            raise ValueError(f"unknown deep_capture_when {self.deep_capture_when!r}")
        if self.kind == KIND_HEALTH and self.action != ACTION_RECORD:
            raise ValueError("health triggers only record incidents")


@dataclass(frozen=True)
class TickResult:
    """What one trigger's evaluation found at one tick."""

    spec: TriggerSpec
    at_ns: int
    evaluation: Evaluation
    transition: Transition | None
    state: str


@dataclass
class TriggerEngine:
    """The configured triggers and their states."""

    specs: Sequence[TriggerSpec]
    tick_seconds: float
    states: dict[str, TriggerState] = field(init=False)

    def __post_init__(self) -> None:
        ids = [spec.trigger_id for spec in self.specs]
        if len(ids) != len(set(ids)):
            raise ValueError("trigger ids must be unique")
        if self.tick_seconds <= 0:
            raise ValueError("tick_seconds must be > 0")
        self.states = {
            spec.trigger_id: TriggerState(spec.sustain) for spec in self.specs
        }

    def tick(
        self,
        at_ns: int,
        history: Sequence[Entry],
        *,
        perturbations: Sequence[tuple[int, int]] = (),
        completion_horizon_ns: int = 0,
    ) -> list[TickResult]:
        """Evaluate every trigger at ``at_ns`` (monotonic) and advance it."""
        results = []
        for spec in self.specs:
            evaluation, since_ns = self._evaluate(spec, at_ns, history)
            # Masked over all the evaluation read: from its earliest scrape,
            # which can lie before t - W, or from t - W, whichever is first.
            reach = min(since_ns, at_ns - int(spec.sustain.window * _NS))
            if spec.completion_recorded:
                reach -= completion_horizon_ns
            if overlaps(reach, at_ns, perturbations):
                evaluation = replace(
                    evaluation,
                    classification=MASKED,
                    reasons=(*evaluation.reasons, "perturbation"),
                )
            state = self.states[spec.trigger_id]
            transition = state.observe(at_ns, evaluation.classification)
            results.append(TickResult(spec, at_ns, evaluation, transition, state.state))
        return results

    def _evaluate(
        self, spec: TriggerSpec, at_ns: int, history: Sequence[Entry]
    ) -> tuple[Evaluation, int]:
        """The evaluation, and the earliest monotonic instant it read."""
        predicate = spec.predicate
        if isinstance(predicate, HistoryPredicate):
            tail = history[-predicate.tail_scrapes :]
            since = tail[0][0].mono_ns if tail else at_ns
            return predicate.evaluate_history(history), since
        selection = select_window(
            history,
            at_ns=at_ns,
            window_ns=int(spec.sustain.window * _NS),
            tick_ns=int(self.tick_seconds * _NS),
        )
        if selection.reason is not None:
            return Evaluation(DATA_GAP, reasons=(selection.reason,)), at_ns
        assert selection.sample_start_ns is not None  # set with every window
        return predicate.evaluate(selection.scrapes), selection.sample_start_ns


__all__ = [
    "ACTION_DEEP_CAPTURE",
    "ACTION_RECORD",
    "KINDS",
    "KIND_HEALTH",
    "KIND_METRIC",
    "KIND_SIGNAL",
    "KIND_SLO",
    "KIND_TEST",
    "WHEN_ALWAYS",
    "WHEN_UNEXPLAINED",
    "HistoryPredicate",
    "TickResult",
    "TriggerEngine",
    "TriggerSpec",
]
