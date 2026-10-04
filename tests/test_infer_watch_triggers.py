"""The trigger state machine: accumulation, guarantees, resets and re-entry."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from stormlog.infer.watch.triggers import (
    CLEAR,
    DATA_GAP,
    EVENT_FIRED,
    EVENT_PENDING,
    EVENT_REENTERED,
    EVENT_RESET,
    EVENT_RESOLVED,
    EVENT_RESOLVING,
    FIRING,
    INACTIVE,
    MASKED,
    PENDING,
    RESET_CLEAR,
    RESET_DATA_GAP,
    VIOLATING,
    Sustain,
    Transition,
    TriggerState,
)

S = 1_000_000_000
DEFAULT = Sustain.with_defaults(window=30, hold=60, clear=None, tick=1)


def _run(
    sustain: Sustain,
    classify: Callable[[float], str],
    *,
    until: float,
    step: float = 1.0,
    start: float = 0.0,
) -> tuple[TriggerState, list[tuple[float, Transition]]]:
    """Evaluate every ``step`` seconds from ``start`` to ``until``."""
    state = TriggerState(sustain)
    transitions: list[tuple[float, Transition]] = []
    ticks = round((until - start) / step)
    for index in range(ticks + 1):
        at = start + index * step
        found = state.observe(round(at * S), classify(at))
        if found is not None:
            transitions.append((at, found))
    return state, transitions


def _events(transitions: list[tuple[float, Transition]]) -> list[tuple[float, str]]:
    return [(round(at, 3), t.event) for at, t in transitions]


def test_defaults_and_validation() -> None:
    assert (DEFAULT.window, DEFAULT.hold, DEFAULT.clear) == (30, 60, 60)
    assert DEFAULT.gap == 30 and DEFAULT.clear_tolerance == 2
    assert DEFAULT.shortest_firing_violation() == 30
    assert DEFAULT.detection_bound(1) == 91
    with pytest.raises(ValueError, match="F >= W"):
        Sustain(window=30, hold=20, clear=10, gap=10, clear_tolerance=0)
    with pytest.raises(ValueError, match="> 0"):
        Sustain(window=30, hold=60, clear=0, gap=10, clear_tolerance=0)


def test_accumulation_starts_at_zero_on_the_first_violation() -> None:
    _state, transitions = _run(
        DEFAULT, lambda t: VIOLATING if t >= 10 else CLEAR, until=80
    )
    # Pending at 10 with nothing accumulated; F = 60 more seconds to fire.
    assert _events(transitions) == [(10, EVENT_PENDING), (70, EVENT_FIRED)]
    assert transitions[1][1].accumulated_ns == 60 * S
    assert transitions[1][1].pending_since_ns == 10 * S


def test_astras_short_spike_never_fires() -> None:
    """A window predicate true from -0.1 s to 59.4 s (d = 29.5 s, W = 30)."""
    state, transitions = _run(
        DEFAULT, lambda t: VIOLATING if t <= 59.4 else CLEAR, until=120
    )
    # Violating ticks 0..59 accumulate 59 s, short of F.
    assert all(t.event != EVENT_FIRED for _at, t in transitions)
    assert state.state == INACTIVE


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("duration", [10.0, 29.0, 29.9, 59.9])
def test_violations_shorter_than_f_never_fire_at_any_tick_phase(
    offset: float, duration: float
) -> None:
    onset = 5.0 + offset
    _state, transitions = _run(
        DEFAULT,
        lambda t: VIOLATING if onset <= t < onset + duration else CLEAR,
        until=200,
    )
    assert all(t.event != EVENT_FIRED for _at, t in transitions)


@pytest.mark.parametrize("offset", [0.0, 0.3, 0.9])
def test_a_persistent_violation_fires_within_f_plus_one_tick(offset: float) -> None:
    onset = 5.0 + offset
    _state, transitions = _run(
        DEFAULT, lambda t: VIOLATING if t >= onset else CLEAR, until=200
    )
    fired = [at for at, t in transitions if t.event == EVENT_FIRED]
    assert len(fired) == 1
    assert onset + 60 <= fired[0] <= onset + 60 + 1


def test_the_worked_example_resets_on_the_outage_and_fires_after_it() -> None:
    """W=30, F=60, G=30: violating from 30, scrapes failing over 90-121, and
    windows data gaps until their start scrape follows the outage (152)."""

    def classify(t: float) -> str:
        if t < 30:
            return CLEAR
        if t < 90:
            return VIOLATING
        if t < 152:
            return DATA_GAP
        return VIOLATING

    _state, transitions = _run(DEFAULT, classify, until=260)
    assert _events(transitions) == [
        (30, EVENT_PENDING),
        (120, EVENT_RESET),
        (152, EVENT_PENDING),
        (212, EVENT_FIRED),
    ]
    reset = transitions[1][1]
    assert reset.reason == RESET_DATA_GAP
    assert reset.accumulated_ns == 59 * S
    # Inside the stated bound: reset + F + W + tick + data-gap time after it.
    assert 212 <= 120 + 60 + 30 + 1 + 32


def test_masked_time_pauses_the_clock_and_never_counts_toward_the_gap() -> None:
    def classify(t: float) -> str:
        if t < 20:
            return VIOLATING  # 19 s accumulated
        if t < 120:
            return MASKED  # a 100 s capture perturbation
        if t < 140:
            return DATA_GAP  # 20 s, under G = 30
        return VIOLATING

    state, transitions = _run(DEFAULT, classify, until=200)
    # 19 s before the mask, then 41 violating intervals from (139, 140] on.
    assert _events(transitions) == [(0, EVENT_PENDING), (180, EVENT_FIRED)]
    assert state.state == FIRING


def test_a_short_clear_run_is_tolerated_and_a_long_one_resets() -> None:
    tolerated, transitions = _run(
        DEFAULT, lambda t: CLEAR if 20 <= t < 22 else VIOLATING, until=70
    )
    assert _events(transitions) == [(0, EVENT_PENDING), (62, EVENT_FIRED)]
    assert tolerated.state == FIRING
    reset_state, transitions = _run(
        DEFAULT, lambda t: CLEAR if 20 <= t < 24 else VIOLATING, until=70
    )
    assert (22, EVENT_RESET) in _events(transitions)
    assert [t.reason for _at, t in transitions if t.event == EVENT_RESET] == [
        RESET_CLEAR
    ]
    assert reset_state.state == PENDING  # started again at 24, fires at 84


def test_firing_resolves_after_c_and_reenters_without_a_new_episode() -> None:
    def classify(t: float) -> str:
        if t <= 60:
            return VIOLATING  # fires at 60
        if t <= 70:
            return CLEAR  # resolving from 61
        if t <= 75:
            return VIOLATING  # back to firing: the same episode
        if t <= 80:
            return MASKED  # firing holds
        return CLEAR  # resolving from 81, resolved after C = 60 s of clear

    state, transitions = _run(DEFAULT, classify, until=200)
    assert _events(transitions) == [
        (0, EVENT_PENDING),
        (60, EVENT_FIRED),
        (61, EVENT_RESOLVING),
        (71, EVENT_REENTERED),
        (81, EVENT_RESOLVING),
        (141, EVENT_RESOLVED),
    ]
    assert state.state == INACTIVE and state.reentries == 1
    # Re-armed: the next violation starts a new pending run.
    assert state.observe(201 * S, VIOLATING) is not None
    assert state.state == PENDING


def test_resolving_pauses_on_masked_and_data_gap_time() -> None:
    def classify(t: float) -> str:
        if t <= 60:
            return VIOLATING
        if t <= 90:
            return CLEAR  # 29 s of clear from 61
        if t <= 200:
            return DATA_GAP  # never resolves without clear evidence
        return CLEAR

    state, transitions = _run(DEFAULT, classify, until=240)
    resolved = [at for at, t in transitions if t.event == EVENT_RESOLVED]
    assert resolved == [231]  # 29 s before the gap plus 31 s after it
    assert state.state == INACTIVE


def test_evaluations_must_be_in_order() -> None:
    state = TriggerState(DEFAULT)
    state.observe(10 * S, VIOLATING)
    with pytest.raises(ValueError, match="monotonic"):
        state.observe(5 * S, VIOLATING)
    with pytest.raises(ValueError, match="classification"):
        state.observe(11 * S, "maybe")
    assert state.state == PENDING
