"""When a condition counts as sustained: the trigger state machine.

Each tick classifies a trigger's window as violating, clear, a data gap, or
masked (it overlaps the watcher's own capture perturbation). An evaluation at
tick ``t_k`` classifies the interval since the previous tick, and accumulated
times are sums of such intervals on the monotonic clock.

- ``inactive`` becomes ``pending`` on a violating evaluation, with nothing
  accumulated yet: the first violating evaluation starts the clock at zero.
- ``pending`` adds each later violating interval and fires once the total
  reaches the hold time ``F``. Masked time, data-gap time up to ``G``, and
  clear time up to ``clear_tolerance`` in all since it went pending pause
  the clock. More data-gap time than ``G`` since the last informative
  evaluation, or more clear time in all, resets to ``inactive``. Masked
  time never counts toward ``G``.
- ``firing`` holds through violating, masked and data-gap evaluations; a
  clear one starts ``resolving``.
- ``resolving`` returns to ``firing`` on a violating evaluation (the same
  episode, counted as a re-entry), pauses on masked and data-gap time, and
  resolves to ``inactive`` after ``C`` of clear time. The clear evaluation
  that starts ``resolving`` is not counted, so resolution takes ``C`` plus
  up to one tick of clear, as firing takes ``F`` after the first violating
  evaluation. Only then can the trigger fire again.

So a predicate whose observations stay violating for less than ``F`` never
fires, and one that turns violating at ``a`` and stays so fires by
``a + Δ + ceil((F + j) / Δ)·Δ + j`` plus any paused time, for ticks
scheduled every ``Δ`` and each at most ``j`` late: ``a + Δ + F`` only when
``F`` is a multiple of ``Δ`` and the ticks run on time. For a predicate over
a window of ``W``, whose first scrape may have returned up to one tick
before the window opens, an observable violation of length ``d`` keeps it
true for at most ``d + W + Δ``, so ``d < F - W - Δ`` never fires. These
statements are about what the watcher
observes, not about the fault that caused it; ``docs/incident_capture.md``
has the derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

VIOLATING = "violating"
CLEAR = "clear"
DATA_GAP = "data_gap"
MASKED = "masked"
CLASSIFICATIONS = (VIOLATING, CLEAR, DATA_GAP, MASKED)

INACTIVE = "inactive"
PENDING = "pending"
FIRING = "firing"
RESOLVING = "resolving"
STATES = (INACTIVE, PENDING, FIRING, RESOLVING)

# What a transition record says happened.
EVENT_PENDING = "pending"
EVENT_FIRED = "fired"
EVENT_RESOLVING = "resolving"
EVENT_REENTERED = "reentered"
EVENT_RESOLVED = "resolved"
EVENT_RESET = "reset"

RESET_DATA_GAP = "data_gap"
RESET_CLEAR = "clear"

_NS = 1_000_000_000


@dataclass(frozen=True)
class Sustain:
    """How long a condition must hold, and what interrupts it, in seconds.

    ``window`` is the predicate's evaluation window ``W``, ``hold`` the time
    ``F`` it must be violating, ``clear`` the time ``C`` it must be clear to
    resolve, ``gap`` the data-gap time ``G`` a pending trigger survives, and
    ``clear_tolerance`` the clear time, in all, it survives.
    """

    window: float
    hold: float
    clear: float
    gap: float
    clear_tolerance: float

    def __post_init__(self) -> None:
        for name in ("window", "hold", "clear", "gap"):
            if getattr(self, name) <= 0:
                raise ValueError(f"sustain {name} must be > 0")
        if self.clear_tolerance < 0:
            raise ValueError("sustain clear_tolerance must be >= 0")
        if self.hold < self.window:
            raise ValueError("sustain hold must be >= window (F >= W)")

    @classmethod
    def with_defaults(
        cls, *, window: float, hold: float, clear: float | None, tick: float
    ) -> Sustain:
        """``C = F``, ``G = F / 2`` and ``clear_tolerance = min(2Δ, F / 10)``."""
        return cls(
            window=window,
            hold=hold,
            clear=hold if clear is None else clear,
            gap=hold / 2,
            clear_tolerance=min(2 * tick, hold / 10),
        )

    def shortest_firing_violation(self, tick: float = 0.0) -> float:
        """Observable violations shorter than this never fire: ``F - W - Δ``,
        for ticks every ``Δ`` (0 for evaluation without a schedule), and
        never below 0. A window's first scrape may have returned up to a
        tick before the window opens, so a violation stays in view up to
        ``W + Δ`` after it ends."""
        return max(0.0, self.hold - self.window - tick)

    def fire_bound(self, tick: float, late: float = 0.0) -> float:
        """How long after a predicate turns violating, and stays so, the
        trigger has fired, plus any paused time: ``Δ + ceil((F + j)/Δ)·Δ + j``
        for ticks every ``Δ``, each at most ``j`` late.

        The first violating evaluation comes within ``Δ + j``; accumulation
        is measured between evaluations, so a late first tick shortens it by
        up to ``j``, and the firing tick can itself run ``j`` late.
        """
        ticks = math.ceil((self.hold + late) / tick - 1e-9)
        return tick + ticks * tick + late

    def detection_bound(self, tick: float, late: float = 0.0) -> float:
        """A persistent change fires within ``W`` plus :meth:`fire_bound` of
        its onset, plus any paused time, for a window predicate that needs a
        full window: ``F + W + Δ`` when the ticks run on time and ``F`` is a
        multiple of ``Δ``."""
        return self.window + self.fire_bound(tick, late)


@dataclass(frozen=True)
class Transition:
    """One state change, as the ledger records it."""

    event: str
    at_ns: int
    state: str
    pending_since_ns: int | None
    fired_at_ns: int | None
    accumulated_ns: int
    reason: str | None = None


@dataclass
class TriggerState:
    """One trigger's progress through the state machine."""

    sustain: Sustain
    state: str = INACTIVE
    pending_since_ns: int | None = None
    fired_at_ns: int | None = None
    resolved_at_ns: int | None = None
    accumulated_ns: int = 0
    reentries: int = 0
    _last_tick_ns: int | None = field(default=None, repr=False)
    _gap_ns: int = field(default=0, repr=False)
    _clear_total_ns: int = field(default=0, repr=False)
    _clear_ns: int = field(default=0, repr=False)

    def observe(self, at_ns: int, classification: str) -> Transition | None:
        """Classify the interval ending at ``at_ns``; return any transition."""
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification {classification!r}")
        if self._last_tick_ns is not None and at_ns < self._last_tick_ns:
            raise ValueError("evaluations must be in monotonic order")
        interval = 0 if self._last_tick_ns is None else at_ns - self._last_tick_ns
        self._last_tick_ns = at_ns
        if self.state == INACTIVE:
            return self._from_inactive(at_ns, classification)
        if self.state == PENDING:
            return self._from_pending(at_ns, classification, interval)
        if self.state == FIRING:
            return self._from_firing(at_ns, classification)
        return self._from_resolving(at_ns, classification, interval)

    def _from_inactive(self, at_ns: int, classification: str) -> Transition | None:
        if classification != VIOLATING:
            return None
        self.state = PENDING
        self.pending_since_ns = at_ns
        self.fired_at_ns = None
        self.resolved_at_ns = None
        self.accumulated_ns = 0
        self.reentries = 0
        self._gap_ns = 0
        self._clear_total_ns = 0
        return self._transition(EVENT_PENDING, at_ns)

    def _from_pending(
        self, at_ns: int, classification: str, interval: int
    ) -> Transition | None:
        if classification == VIOLATING:
            self.accumulated_ns += interval
            self._gap_ns = 0
            if self.accumulated_ns >= int(self.sustain.hold * _NS):
                return self._fire(at_ns)
            return None
        if classification == MASKED:
            return None
        if classification == DATA_GAP:
            self._gap_ns += interval
            if self._gap_ns > int(self.sustain.gap * _NS):
                return self._reset(at_ns, RESET_DATA_GAP)
            return None
        self._gap_ns = 0
        # All clear time since pending counts, not only the latest run: a
        # predicate that keeps flapping clear must not accumulate to F.
        self._clear_total_ns += interval
        if self._clear_total_ns > int(self.sustain.clear_tolerance * _NS):
            return self._reset(at_ns, RESET_CLEAR)
        return None

    def _from_firing(self, at_ns: int, classification: str) -> Transition | None:
        if classification != CLEAR:
            return None
        self.state = RESOLVING
        self._clear_ns = 0
        return self._transition(EVENT_RESOLVING, at_ns)

    def _from_resolving(
        self, at_ns: int, classification: str, interval: int
    ) -> Transition | None:
        if classification == VIOLATING:
            self.state = FIRING
            self.reentries += 1
            self._clear_ns = 0
            return self._transition(EVENT_REENTERED, at_ns)
        if classification != CLEAR:
            return None
        self._clear_ns += interval
        if self._clear_ns >= int(self.sustain.clear * _NS):
            self.state = INACTIVE
            self.resolved_at_ns = at_ns
            return self._transition(EVENT_RESOLVED, at_ns)
        return None

    def _fire(self, at_ns: int) -> Transition:
        self.state = FIRING
        self.fired_at_ns = at_ns
        return self._transition(EVENT_FIRED, at_ns)

    def _reset(self, at_ns: int, reason: str) -> Transition:
        transition = Transition(
            EVENT_RESET,
            at_ns,
            INACTIVE,
            self.pending_since_ns,
            None,
            self.accumulated_ns,
            reason,
        )
        self.state = INACTIVE
        self.pending_since_ns = None
        self.accumulated_ns = 0
        return transition

    def _transition(self, event: str, at_ns: int) -> Transition:
        return Transition(
            event,
            at_ns,
            self.state,
            self.pending_since_ns,
            self.fired_at_ns,
            self.accumulated_ns,
        )


__all__ = [
    "CLASSIFICATIONS",
    "CLEAR",
    "DATA_GAP",
    "EVENT_FIRED",
    "EVENT_PENDING",
    "EVENT_REENTERED",
    "EVENT_RESET",
    "EVENT_RESOLVED",
    "EVENT_RESOLVING",
    "FIRING",
    "INACTIVE",
    "MASKED",
    "PENDING",
    "RESET_CLEAR",
    "RESET_DATA_GAP",
    "RESOLVING",
    "STATES",
    "Sustain",
    "Transition",
    "TriggerState",
    "VIOLATING",
]
