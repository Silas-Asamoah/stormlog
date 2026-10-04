"""Effect timing, realization and recovery from the reference channel
(#221 design A.4), on synthetic series."""

from __future__ import annotations

from typing import Callable

import pytest

from stormlog.infer.qualify.recovery import (
    START,
    TIMEOUT,
    WAIT,
    Actions,
    AllWithin,
    Baseline,
    Context,
    Criterion,
    NoEvents,
    Signals,
    effect_timing,
    held_from,
    next_episode,
    priming_check,
    realization,
)

S = 1_000_000_000
MS = 1_000_000


def every_second(
    start: int, end: int, value: Callable[[int], float]
) -> list[tuple[int, float]]:
    return [(second * S, value(second)) for second in range(start, end)]


def context(
    signals: Signals, actions: Actions = Actions(), until: int = 200
) -> Context:
    baseline = Baseline.measure(signals, 0, 45 * S)
    return Context(signals, baseline, actions, start_ns=60 * S, until_ns=until * S)


def kv_signals(preemptions: list[int]) -> Signals:
    usage = every_second(0, 200, lambda s: 0.9 if 60 <= s <= 80 else 0.45)
    return Signals(victim_preemptions=preemptions, kv_usage=usage)


def test_kv_pressure_runs_from_the_first_victim_preemption_to_its_recovery() -> None:
    signals = kv_signals([63 * S, 70 * S])
    timing = effect_timing("F2", context(signals))
    assert (timing.onset_ns, timing.basis) == (
        63 * S,
        "reference_hook_preempted_victim",
    )
    # The last violation is the 80 s scrape; recovery holds for 10 s after it.
    assert timing.end_ns == 80 * S + 1
    assert timing.recovery_held_at_ns == 90 * S + 1
    realized, checks = realization("F2", context(signals), timing)
    assert realized and checks[0].value == 2


def test_kv_pressure_without_a_victim_preemption_is_not_realized() -> None:
    signals = kv_signals([])
    timing = effect_timing("F2", context(signals))
    assert timing.onset_ns is None
    assert realization("F2", context(signals), timing)[0] is False
    # Its workload twin is realized exactly when nothing was preempted.
    twin = effect_timing("T2", context(signals, Actions(first_admission_ns=61 * S)))
    assert twin.onset_ns == 61 * S
    assert realization("T2", context(signals), twin)[0] is True


def queue_signals() -> Signals:
    waits = [
        (
            second * S + 500 * MS,
            0.5 if 60 <= second < 90 else 0.02 + 0.001 * (second % 5),
        )
        for second in range(0, 200)
    ]
    waiting = every_second(0, 200, lambda s: 10.0 if 60 <= s <= 91 else float(s % 3))
    return Signals(waits=waits, waiting=waiting)


def test_queue_saturation_starts_when_the_median_wait_passes_the_baseline_p95() -> None:
    signals = queue_signals()
    timing = effect_timing("F1", context(signals))
    assert timing.onset_ns == 60 * S
    assert timing.end_ns == 91 * S + 1
    assert timing.recovery_held_at_ns == 101 * S + 1
    realized, checks = realization("F1", context(signals), timing)
    assert realized
    assert [check.name for check in checks] == ["onset_reached", "no_victim_preemption"]
    # The same waits are no workload twin: they left the baseline.
    assert realization("T1", context(signals), timing)[0] is False


def test_prefix_loss_follows_the_victims_cached_fraction() -> None:
    cached = every_second(0, 200, lambda s: 0.2 if 60 <= s < 85 else 0.95)
    signals = Signals(cached_fraction=cached)
    timing = effect_timing("F3", context(signals))
    assert timing.onset_ns == 60 * S
    # The 10 s from 80 s to 90 s already hold more recovered samples than
    # dipped ones, so their median is back.
    assert timing.end_ns == 80 * S
    assert realization("F3", context(signals), timing)[0]
    twin = Signals(cached_fraction=every_second(0, 200, lambda s: 0.95))
    twin_timing = effect_timing("T3", context(twin, Actions(first_send_ns=60 * S)))
    assert twin_timing.onset_ns == 60 * S
    assert realization("T3", context(twin), twin_timing)[0]
    assert not realization("T3", context(signals), twin_timing)[0]


def stalled_steps(pulses: list[tuple[int, int]], keep_stepping: bool) -> list[int]:
    steps = []
    for tick in range(0, 200_000, 20):
        at = tick * MS
        paused = any(stop < at < cont for stop, cont in pulses)
        if keep_stepping or not paused:
            steps.append(at)
    return steps


PULSES = [(60 * S, 60 * S + 100 * MS), (62 * S, 62 * S + 100 * MS)]


def test_an_engine_stall_is_timed_from_the_first_stop_and_recovers_after_the_last() -> (
    None
):
    signals = Signals(step_starts=stalled_steps(PULSES, keep_stepping=False))
    actions = Actions(
        first_stop_confirmed_ns=60 * S,
        last_continue_ns=62 * S + 100 * MS,
        pulses=PULSES,
    )
    timing = effect_timing("F4a", context(signals, actions))
    assert timing.onset_ns == 60 * S
    # From the last SIGCONT the median step gap is the baseline's again; the
    # one 100 ms gap the pulse left doesn't move it.
    assert timing.end_ns == 62 * S + 100 * MS
    assert timing.recovery_held_at_ns == timing.end_ns + 5 * S
    assert realization("F4a", context(signals, actions), timing)[0]
    # An engine that kept stepping was not stalled, though the API server was.
    busy = Signals(step_starts=stalled_steps(PULSES, keep_stepping=True))
    assert not realization("F4a", context(busy, actions), timing)[0]
    assert realization("F4b", context(busy, actions), timing)[0]


def test_a_capture_pause_runs_from_the_stop_request_through_the_drain() -> None:
    actions = Actions(stop_requested_ns=60 * S, stop_returned_ns=61 * S, drain_ns=2 * S)
    timing = effect_timing("I1", context(Signals(), actions))
    assert (timing.onset_ns, timing.end_ns) == (60 * S, 63 * S)
    assert realization("I1", context(Signals(), actions), timing)[0]
    open_window = Actions(stop_requested_ns=60 * S)
    assert not realization(
        "I1",
        context(Signals(), open_window),
        effect_timing("I1", context(Signals(), open_window)),
    )[0]


def test_recovery_must_be_seen_whole_before_it_counts() -> None:
    criteria: list[Criterion] = [
        NoEvents([65 * S]),
        AllWithin(every_second(0, 100, lambda s: 1.0)),
    ]
    assert held_from(criteria, 60 * S, 80 * S, 10 * S) == 65 * S + 1
    # Not yet observed for the full 10 s.
    assert held_from(criteria, 60 * S, 74 * S, 10 * S) is None
    # Without samples, a criterion that needs them does not hold.
    empty = [AllWithin([])]
    assert held_from(empty, 60 * S, 80 * S, 10 * S) is None
    assert (
        held_from([AllWithin([], require_samples=False)], 60 * S, 80 * S, 10 * S)
        == 60 * S
    )


def test_a_hold_can_begin_a_hold_before_its_first_sample() -> None:
    # A violation at 0 s, the next sample at 12 s: [2 s, 12 s] holds, though
    # no sample or violation marks 2 s. A window's samples change where one
    # enters it (a hold before the sample) as well as where one leaves it.
    sparse = AllWithin([(0, 9.0), (12 * S, 1.0), (40 * S, 1.0)], high=5.0)
    assert sparse.holds(2 * S, 12 * S)
    assert held_from([sparse], 0, 40 * S, 10 * S) == 2 * S


def test_the_priming_check_needs_a_warm_cache() -> None:
    warm = Signals(cached_fraction=every_second(0, 30, lambda s: 0.93))
    cold = Signals(cached_fraction=every_second(0, 30, lambda s: 0.8))
    assert priming_check(warm, 30 * S) == (True, 0.93)
    assert priming_check(cold, 30 * S) == (False, 0.8)
    assert priming_check(Signals(), 30 * S) == (False, None)


@pytest.mark.parametrize(
    ("held", "now", "decision"),
    [
        (120, 150, WAIT),  # recovered, but not 60 s after the action
        (120, 160, START),
        (170, 165, WAIT),  # 60 s passed, but recovery holds only at 170 s
        (None, 249, WAIT),
        (None, 250, TIMEOUT),  # 150 s after the action
    ],
)
def test_the_next_episode_waits_for_recovery_within_its_limits(
    held: int | None, now: int, decision: str
) -> None:
    held_ns = None if held is None else held * S
    assert next_episode(100 * S, held_ns, now * S) == decision


def test_an_unknown_episode_type_has_no_rule() -> None:
    with pytest.raises(KeyError):
        effect_timing("F9", context(Signals()))


def test_cadence_recovers_when_the_step_gaps_look_like_the_baseline_again() -> None:
    # A served engine idles between requests, so its step gaps have a long
    # tail: some gap in any 5 s is above the baseline's p95. Recovery is the
    # median gap back within the baseline's p95, not every gap.
    import random

    rng = random.Random(221)
    steps, at = [], 0
    while at < 120 * S:
        steps.append(at)
        at += int(rng.expovariate(1 / 0.03) * S)
        if PULSES[0][0] <= at <= PULSES[-1][1]:
            at = max(at, PULSES[-1][1])
    signals = Signals(step_starts=steps)
    actions = Actions(
        first_stop_confirmed_ns=60 * S,
        last_continue_ns=62 * S + 100 * MS,
        pulses=PULSES,
    )
    timing = effect_timing("F4a", context(signals, actions))
    assert timing.end_ns is not None
    assert timing.end_ns - (62 * S + 100 * MS) < 1 * S
