"""Effect timing, realization and recovery from the reference channel
(#221 design A.4), on synthetic series."""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Callable

import pytest

from stormlog.infer.qualify.recovery import (
    START,
    TIMEOUT,
    WAIT,
    Actions,
    AllWithin,
    Baseline,
    CadenceWithin,
    Context,
    Criterion,
    GapStats,
    MostlyWithin,
    NoEvents,
    Point,
    Signals,
    Thresholds,
    Timing,
    added_mechanisms,
    effect_timing,
    held_from,
    next_episode,
    observation_of,
    priming_check,
    realization,
)

# Every episode type A.4's catalog injects, a short twin among them; X1–X3
# are judged by C.6's outage criteria instead.
CATALOG_TYPES = (
    "F1", "T1", "F2", "T2", "F3", "T3", "T3b", "F4a", "F4b", "H0", "W1",
    "I1", "P", "N", "F5", "R0", "S-F1",
)  # fmt: skip

S = 1_000_000_000
MS = 1_000_000

# A victim request always in flight: every step gap is a busy one.
ALWAYS = ((0, 10**18),)


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
    return Signals(in_flight=None, victim_preemptions=preemptions, kv_usage=usage)


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
    # Three victim admissions a second, as at the design's 3 req/s.
    waits = [
        (
            second * S + third * 333 * MS,
            0.5 if 60 <= second < 90 else 0.02 + 0.001 * ((3 * second + third) % 5),
        )
        for second in range(0, 200)
        for third in range(3)
    ]
    waiting = every_second(0, 200, lambda s: 10.0 if 60 <= s <= 91 else float(s % 3))
    return Signals(in_flight=None, waits=waits, waiting=waiting)


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
    signals = Signals(in_flight=None, cached_fraction=cached)
    timing = effect_timing("F3", context(signals))
    assert timing.onset_ns == 60 * S
    # The effect ends with the dip, just after its last sample at 84 s: the
    # 10 s from 80 s already hold more recovered samples than dipped ones,
    # but a hold that begins with a dipped sample would end the effect
    # while it was still under way.
    assert timing.end_ns == 84 * S + 1
    assert realization("F3", context(signals), timing)[0]
    twin = Signals(in_flight=None, cached_fraction=every_second(0, 200, lambda s: 0.95))
    twin_timing = effect_timing("T3", context(twin, Actions(first_send_ns=60 * S)))
    assert twin_timing.onset_ns == 60 * S
    assert realization("T3", context(twin), twin_timing)[0]
    assert not realization("T3", context(signals), twin_timing)[0]


def test_a_cache_dip_just_after_the_holds_first_sample_is_seen() -> None:
    # Found rerunning the full-catalog e2e on #276: F3's victim kept one
    # cached request at the onset, then missed for 2.8 s; its 6 s hold had
    # a recovered first sample and a recovered median, so the effect ended
    # at its onset. Here: 10 samples a second, one cached at 60 s, a dip to
    # 64 s, then cached again. The onset's 5 s window must look recovered
    # too, so the effect ends where the dip does, just after its last
    # dipped sample at 63.9 s.
    def cached_at(tenth: int) -> float:
        return 0.0 if 601 <= tenth < 640 else 0.95

    cached = [(tenth * S // 10, cached_at(tenth)) for tenth in range(2000)]
    signals = Signals(in_flight=None, cached_fraction=cached)
    timing = effect_timing("F3", context(signals))
    assert timing.onset_ns == 60 * S
    assert timing.end_ns == 639 * S // 10 + 1


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
    signals = Signals(
        in_flight=ALWAYS, step_starts=stalled_steps(PULSES, keep_stepping=False)
    )
    actions = Actions(
        first_stop_confirmed_ns=60 * S,
        last_continue_ns=62 * S + 100 * MS,
        pulses=PULSES,
    )
    timing = effect_timing("F4a", context(signals, actions))
    assert timing.onset_ns == 60 * S
    # The pulse's own 100 ms gap ends at the last SIGCONT; from just after
    # it the cadence is the baseline's again.
    assert timing.end_ns == 62 * S + 100 * MS + 1
    assert timing.recovery_held_at_ns == timing.end_ns + 10 * S
    assert realization("F4a", context(signals, actions), timing)[0]
    # An engine that kept stepping was not stalled, though the API server was.
    busy = Signals(
        in_flight=ALWAYS, step_starts=stalled_steps(PULSES, keep_stepping=True)
    )
    assert not realization("F4a", context(busy, actions), timing)[0]
    assert realization("F4b", context(busy, actions), timing)[0]


def test_a_capture_pause_runs_from_the_stop_request_through_the_drain() -> None:
    actions = Actions(
        capture_started_ns=50 * S,
        stop_requested_ns=60 * S,
        stop_returned_ns=61 * S,
        drain_ns=2 * S,
    )
    timing = effect_timing("I1", context(Signals(in_flight=None), actions))
    assert (timing.onset_ns, timing.end_ns) == (60 * S, 63 * S)
    assert realization("I1", context(Signals(in_flight=None), actions), timing)[0]
    open_window = Actions(stop_requested_ns=60 * S)
    assert not realization(
        "I1",
        context(Signals(in_flight=None), open_window),
        effect_timing("I1", context(Signals(in_flight=None), open_window)),
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
    warm = Signals(in_flight=None, cached_fraction=every_second(0, 30, lambda s: 0.93))
    cold = Signals(in_flight=None, cached_fraction=every_second(0, 30, lambda s: 0.8))
    assert priming_check(warm, 30 * S) == (True, 0.93)
    assert priming_check(cold, 30 * S) == (False, 0.8)
    assert priming_check(Signals(in_flight=None), 30 * S) == (False, None)


@pytest.mark.parametrize(
    ("held", "now", "decision"),
    [
        (120, 150, WAIT),  # recovered, but not 60 s after the action
        (120, 160, START),
        (170, 165, WAIT),  # 60 s passed, but recovery holds only at 170 s
        (None, 249, WAIT),
        (None, 250, TIMEOUT),  # 150 s after the action
        # Recovery that held only after the timeout is a timeout, whenever
        # the harness asks.
        (255, 250, TIMEOUT),
        (255, 256, TIMEOUT),
    ],
)
def test_the_next_episode_waits_for_recovery_within_its_limits(
    held: int | None, now: int, decision: str
) -> None:
    held_ns = None if held is None else held * S
    assert next_episode(100 * S, held_ns, now * S) == decision


def test_an_unknown_episode_type_has_no_rule() -> None:
    with pytest.raises(KeyError):
        effect_timing("F9", context(Signals(in_flight=None)))


def test_cadence_recovers_when_the_step_gaps_look_like_the_baseline_again() -> None:
    # Jittered step gaps have a tail: some gap in any 5 s is above the
    # baseline's p95. Recovery allows as many as chance does, not none.
    import random

    rng = random.Random(221)
    steps, at = [], 0
    while at < 120 * S:
        steps.append(at)
        at += int(rng.expovariate(1 / 0.03) * S)
        if PULSES[0][0] <= at <= PULSES[-1][1]:
            at = max(at, PULSES[-1][1])
    signals = Signals(in_flight=ALWAYS, step_starts=steps)
    actions = Actions(
        first_stop_confirmed_ns=60 * S,
        last_continue_ns=62 * S + 100 * MS,
        pulses=PULSES,
    )
    timing = effect_timing("F4a", context(signals, actions))
    assert timing.end_ns is not None
    assert timing.end_ns - (62 * S + 100 * MS) < 1 * S


def queue_run(seed: int, elevated_after: float) -> tuple[int | None, int]:
    """F1 at 3 req/s: waits 20x the baseline from 45 s to 75 s; then a
    share ``elevated_after`` of them stays 4x for 60 s more."""
    import random

    rng = random.Random(seed)
    waits, at = [], 0
    while at < 300 * S:
        scale = 1.0
        if 45 * S <= at < 75 * S:
            scale = 20.0
        elif 75 * S <= at < 135 * S and rng.random() < elevated_after:
            scale = 4.0
        waits.append((at, rng.lognormvariate(-4.0, 0.5) * scale))
        at += int(rng.expovariate(3.0) * S)
    waiting = every_second(0, 300, lambda s: float(s % 3))
    signals = Signals(in_flight=None, waits=waits, waiting=waiting)
    timing = effect_timing(
        "F1",
        Context(
            signals,
            Baseline.measure(signals, 0, 45 * S),
            Actions(),
            start_ns=45 * S,
            until_ns=300 * S,
        ),
    )
    return timing.end_ns, 75 * S


def test_queue_saturation_ends_when_the_waits_are_back() -> None:
    # Every wait at or below the baseline's p95 almost never holds for 10 s
    # (5% of normal waits are above it), so the effect ended 6 s late on
    # median and up to 90 s late. As many as chance allows may be above it.
    # A hold's mean wait must also be near the baseline's, which a chance
    # run of slow waits delays by a few seconds: over 200 seeds, 3 end more
    # than 3 s late (2 without the mean bound), none more than 5.1 s.
    lags = []
    for seed in range(40):
        end, back = queue_run(seed, elevated_after=0.0)
        assert end is not None, seed
        lags.append((end - back) / S)
    assert all(-3.0 <= lag <= 6.0 for lag in lags), lags
    assert sum(lag > 3.0 for lag in lags) <= 1, lags


def test_queue_saturation_lasts_while_most_waits_stay_long() -> None:
    # 4 in 5 waits stay 4x the baseline's for 60 s after the overload ends.
    for seed in range(20):
        end, back = queue_run(seed, elevated_after=0.8)
        assert end is not None and end >= back + 55 * S, seed


@pytest.mark.parametrize("episode_type", ["F4A", "F9", "X1"])
def test_a_type_without_a_rule_is_refused_not_realized(episode_type: str) -> None:
    ctx = context(Signals(in_flight=None))
    with pytest.raises(KeyError):
        realization(episode_type, ctx, Timing(None, "none"))


def test_every_catalog_type_has_a_timing_and_a_realization_rule() -> None:
    actions = Actions(
        first_send_ns=61 * S,
        first_admission_ns=61 * S,
        first_stop_confirmed_ns=61 * S,
        last_continue_ns=62 * S,
        pulses=[(61 * S, 62 * S)],
        stop_requested_ns=61 * S,
        stop_returned_ns=62 * S,
        capture_started_ns=55 * S,
        slot_ns=(61 * S, 101 * S),
    )
    ctx = context(Signals(in_flight=None), actions)
    for episode_type in CATALOG_TYPES:
        timing = effect_timing(episode_type, ctx)
        realization(episode_type, ctx, timing)


def test_a_null_run_is_scored_over_its_scheduled_slot() -> None:
    ctx = context(Signals(in_flight=None), Actions(slot_ns=(61 * S, 101 * S)))
    timing = effect_timing("N", ctx)
    assert (timing.onset_ns, timing.end_ns) == (61 * S, 101 * S)
    assert timing.recovery_held_at_ns == 101 * S
    assert realization("N", ctx, timing) == (True, [])
    with pytest.raises(ValueError, match="slot"):
        effect_timing("N", context(Signals(in_flight=None)))


def test_a_short_twin_follows_its_faults_rules() -> None:
    signals = kv_signals([63 * S])
    assert effect_timing("S-F2", context(signals)) == effect_timing(
        "F2", context(signals)
    )


def test_a_rank_pulse_needs_the_peers_wait_to_lengthen() -> None:
    pulses = [(61 * S, 61 * S + 100 * MS)]
    stalled = Actions(
        first_stop_confirmed_ns=61 * S, last_continue_ns=pulses[0][1], pulses=pulses
    )
    signals = Signals(
        in_flight=ALWAYS, step_starts=stalled_steps(pulses, keep_stepping=False)
    )
    timing = effect_timing("F5", context(signals, stalled))
    assert not realization("F5", context(signals, stalled), timing)[0]
    waited = replace(stalled, peer_wait_extended=True)
    assert realization("F5", context(signals, waited), timing)[0]
    assert realization("R0", context(signals, stalled), timing)[0]


def test_a_cache_twin_needs_the_engine_wide_hit_ratio_to_fall() -> None:
    cached = every_second(0, 200, lambda s: 0.95)
    falling = every_second(0, 200, lambda s: 0.4 if 60 <= s < 90 else 0.7)
    steady = every_second(0, 200, lambda s: 0.7)
    actions = Actions(first_send_ns=60 * S)
    fell = context(
        Signals(in_flight=None, cached_fraction=cached, engine_hit_ratio=falling),
        actions,
    )
    held = context(
        Signals(in_flight=None, cached_fraction=cached, engine_hit_ratio=steady),
        actions,
    )
    assert realization("T3b", fell, effect_timing("T3b", fell))[0]
    realized, checks = realization("T3b", held, effect_timing("T3b", held))
    assert not realized
    assert [check.name for check in checks] == [
        "cached_fraction_unchanged",
        "engine_hit_ratio_fell",
    ]


def test_a_capture_must_have_started_as_well_as_stopped() -> None:
    stopped_only = Actions(stop_requested_ns=60 * S, stop_returned_ns=61 * S)
    ctx = context(Signals(in_flight=None), stopped_only)
    assert not realization("I1", ctx, effect_timing("I1", ctx))[0]
    both = context(
        Signals(in_flight=None), replace(stopped_only, capture_started_ns=50 * S)
    )
    assert realization("I1", both, effect_timing("I1", both))[0]


def test_an_api_server_pulse_that_also_stalled_the_engine_adds_that_mechanism() -> None:
    # A.4: F4b stays realized, and its realized set adds host_stall at the
    # engine core, when the engine stopped stepping with requests in flight.
    actions = Actions(
        first_stop_confirmed_ns=60 * S,
        last_continue_ns=62 * S + 100 * MS,
        pulses=PULSES,
    )
    stalled = context(
        Signals(in_flight=ALWAYS, step_starts=stalled_steps(PULSES, False)), actions
    )
    realized, checks = realization("F4b", stalled, effect_timing("F4b", stalled))
    assert realized
    assert added_mechanisms("F4b", checks) == ("host_stall@engine_core",)
    busy = context(
        Signals(in_flight=ALWAYS, step_starts=stalled_steps(PULSES, True)), actions
    )
    realized, checks = realization("F4b", busy, effect_timing("F4b", busy))
    assert realized and added_mechanisms("F4b", checks) == ()


def test_a_twin_that_leaves_the_signals_alone_is_realized() -> None:
    # A twin's recovery already holds at its onset; its effect, the benign
    # change, lasts as long as its action, as N's slot does (the neighbor
    # ran from 60 s to 90 s), and its realization is judged over all of it.
    cached = every_second(0, 200, lambda s: 0.95)
    falling = every_second(0, 200, lambda s: 0.4 if 60 <= s < 90 else 0.7)
    actions = Actions(
        first_send_ns=60 * S, first_admission_ns=60 * S, action_end_ns=90 * S
    )
    t3b = context(
        Signals(in_flight=None, cached_fraction=cached, engine_hit_ratio=falling),
        actions,
    )
    timing = effect_timing("T3b", t3b)
    assert (timing.onset_ns, timing.end_ns) == (60 * S, 90 * S)
    assert realization("T3b", t3b, timing)[0]
    # T2's no-preemption check covers the whole neighbor run: a preemption
    # at 85 s, late in the action, fails it.
    flat = every_second(0, 200, lambda s: 0.45)
    late = context(
        Signals(in_flight=None, victim_preemptions=[85 * S], kv_usage=flat), actions
    )
    assert not realization("T2", late, effect_timing("T2", late))[0]


def test_a_baseline_too_thin_to_compare_with_never_recovers() -> None:
    # No victim request in flight through the 0-45 s baseline, so it has no
    # busy gap; then an engine 3x slow forever. An all-infinite baseline let
    # any 20 gaps "recover" at the SIGCONT; now nothing does.
    steps = [tick * 60 * MS for tick in range(0, 200_000 // 60)]
    signals = Signals(in_flight=[(50 * S, 200 * S)], step_starts=steps)
    actions = Actions(
        first_stop_confirmed_ns=60 * S,
        last_continue_ns=62 * S + 100 * MS,
        pulses=PULSES,
    )
    ctx = context(signals, actions)
    assert ctx.baseline.steps.count == 0
    assert effect_timing("F4a", ctx).end_ns is None


def test_a_queue_twin_without_baseline_waits_never_recovers() -> None:
    # T1 recovers on the victim's waits; a baseline with none gave an
    # infinite p95, and 5 s waits forever recovered at once.
    waits = [(tenth * S // 10, 5.0) for tenth in range(550, 2000)]
    waiting = every_second(0, 200, lambda s: 2.0)
    signals = Signals(in_flight=None, waits=waits, waiting=waiting)
    ctx = context(signals, Actions(first_send_ns=60 * S))
    assert (ctx.baseline.wait_count, ctx.baseline.waiting_count) == (0, 46)
    timing = effect_timing("T1", ctx)
    assert timing.onset_ns == 60 * S
    assert timing.end_ns is None


def test_an_engine_hung_now_is_seen_whatever_recovery_found_earlier() -> None:
    # rev-220-b's delta-3 closure, G4: with an open-loop victim the live
    # loop could say START on a hold found in an idle stretch while a
    # request sent since was stuck. engine_stalled reads the present: the
    # busy part of the gap still open at now.
    from stormlog.infer.qualify.recovery import engine_stalled

    steps = [tick * 20 * MS for tick in range(100 * 50)]  # to 100 s
    sent = [(0, 45 * S), (99 * S, 300 * S)]  # a request in flight since 99 s
    ctx = context(Signals(in_flight=sent, step_starts=steps))
    assert not engine_stalled(ctx, 100 * S)
    assert engine_stalled(ctx, 101 * S)
    # Nothing in flight: idle time is no stall.
    idle = context(Signals(in_flight=[(0, 45 * S)], step_starts=steps))
    assert not engine_stalled(idle, 101 * S)


def test_a_thin_baseline_says_why_recovery_can_never_hold() -> None:
    # fable-design's A2 delta 2, N0: a run whose baseline was too thin
    # timed out with nothing in its truth but recovery_timeout. The rule
    # now names each series that is too thin, and by how much, so the
    # harness can say so; a baseline that is thick enough names none.
    from stormlog.infer.qualify.recovery import recovery_blocked

    no_waits = Signals(
        in_flight=None,
        waits=[(tenth * S // 10, 5.0) for tenth in range(550, 2000)],
        waiting=every_second(50, 200, lambda s: 2.0),
    )
    assert recovery_blocked("T1", context(no_waits)) == (
        "baseline_too_thin: 0 waits of the 20 a hold needs",
        "baseline_too_thin: 0 waiting counts of the 5 a hold needs",
    )
    idle = Signals(
        in_flight=[(50 * S, 200 * S)], step_starts=list(range(0, 200 * S, S))
    )
    assert recovery_blocked("F4b", context(idle)) == (
        "baseline_too_thin: 0 busy step gaps of the 20 a hold needs",
        "baseline_too_thin: 0 chunk gaps of the 20 a hold needs",
    )
    healthy = Signals(
        in_flight=None,
        waits=[(tenth * S // 10, 0.08) for tenth in range(2000)],
        waiting=every_second(0, 200, lambda s: 2.0),
    )
    assert recovery_blocked("F1", context(healthy)) == ()
    assert recovery_blocked("N", context(healthy)) == ()


def test_a_missing_reference_signal_leaves_its_check_incomplete() -> None:
    # T3b's engine-wide hit ratio comes from scrapes. With none, the check
    # can't be judged: it neither fails the twin nor passes it, and the
    # observation is incomplete. The victim's own check still decides.
    cached = every_second(0, 200, lambda s: 0.95)
    ctx = context(
        Signals(in_flight=None, cached_fraction=cached), Actions(first_send_ns=60 * S)
    )
    realized, checks = realization("T3b", ctx, effect_timing("T3b", ctx))
    assert realized
    assert [(c.name, c.incomplete) for c in checks] == [
        ("cached_fraction_unchanged", False),
        ("engine_hit_ratio_fell", True),
    ]
    assert checks[1].to_record()["incomplete"] is True
    assert observation_of(checks) == "incomplete"
    # With every gating check incomplete, nothing was judged: not realized.
    bare = context(Signals(in_flight=None), Actions(first_send_ns=60 * S))
    realized, checks = realization("T3", bare, effect_timing("T3", bare))
    assert not realized and observation_of(checks) == "incomplete"


def test_a_gap_that_begins_in_idle_time_keeps_its_busy_part() -> None:
    # Busy 0-1 s at 20 ms; idle from 1.0 s; a request in flight from 1.2 s
    # waits until the next step at 3.2 s. The gap from the last idle step
    # used to be dropped whole; its busy 2 s is a stall the victim felt.
    steps = [tick * 20 * MS for tick in range(50)] + [3200 * MS]
    steps += [3200 * MS + tick * 20 * MS for tick in range(1, 200)]
    signals = Signals(
        in_flight=[(0, 1000 * MS), (1200 * MS, 10 * S)], step_starts=steps
    )
    gaps = dict(signals.busy_step_gaps())
    assert gaps[3200 * MS] == pytest.approx(2.0)
    # And a gap whose later step lies in idle time is no busy gap at all.
    idle = Signals(in_flight=[(0, 500 * MS)], step_starts=[400 * MS, 900 * MS])
    assert idle.busy_step_gaps() == []


def test_a_stall_still_open_at_a_holds_end_is_a_long_gap() -> None:
    # rev-220-b's D1: steps every 20 ms for 0.5 s, then none until 30.5 s.
    # The 25 gaps in [0, 5 s] all look normal, but the engine hasn't stepped
    # for the hold's last 4.5 s: that open gap is counted, so it can't hold.
    baseline = GapStats(count=250, mean=0.020, p95=0.026, p99=0.030)
    steps = [tick * 20 * MS for tick in range(26)] + [30_500 * MS]
    steps += [30_500 * MS + tick * 20 * MS for tick in range(1, 400)]
    gaps = Signals(in_flight=ALWAYS, step_starts=steps).busy_step_gaps()
    cadence = CadenceWithin(gaps, baseline, Thresholds(), ALWAYS)
    assert not cadence.holds(0, 5 * S)
    # Recovery is found only once the engine steps again.
    start = held_from([cadence], 0, 40 * S, 5 * S)
    assert start is not None and start >= 30_500 * MS
    # A live poll during the hang, at 5.5 s, finds no recovery at all.
    assert held_from([cadence], 0, 5_500 * MS, 5 * S) is None
    # With no victim request in flight at the hold's end, the quiet tail is
    # idle time, not a stall.
    idle = CadenceWithin(gaps, baseline, Thresholds(), [(0, 520 * MS)])
    assert idle.holds(0, 5 * S)


def test_queue_recovery_doesnt_hold_through_recurring_bursts() -> None:
    # rev-220-b's D9: a chance allowance counts samples out of band, not
    # how far out. 10 of 100 waits at 30 s (375x the p95) passed as chance,
    # and so did a saturated waiting gauge 1 sample in 10. The bursts start
    # half a second into the hold, so the first-sample rule doesn't decide.
    # The mean bound refuses these 30 s waits too; the F1 test above is the
    # one where only the ceiling sees its bursts.
    waits = every_second(0, 45, lambda s: 0.08 + 0.001 * (s % 7))
    burst_waits = [
        (S * 50 + tenth * S // 10, 30.0 if tenth % 10 == 5 else 0.08)
        for tenth in range(100)
    ]
    waiting = every_second(0, 45, lambda s: float(s % 7))
    burst_gauge = every_second(50, 60, lambda s: 30.0 if s == 55 else 3.0)
    signals = Signals(
        in_flight=None, waits=waits + burst_waits, waiting=waiting + burst_gauge
    )
    ctx = context(signals, Actions(first_send_ns=48 * S))
    waits_rule, gauge_rule = _queue_criteria_of(ctx)
    assert not waits_rule.holds(50 * S, 60 * S)
    assert not gauge_rule.holds(50 * S, 60 * S)
    # Without the far-out samples, the same holds pass.
    calm = Signals(
        in_flight=None,
        waits=waits + [(t, 0.08) for t, _v in burst_waits],
        waiting=waiting + every_second(50, 60, lambda s: 3.0),
    )
    calm_waits, calm_gauge = _queue_criteria_of(context(calm, ctx.actions))
    assert calm_waits.holds(50 * S, 60 * S) and calm_gauge.holds(50 * S, 60 * S)


def test_f1_recovery_waits_out_wait_bursts_beyond_the_ceiling() -> None:
    # rev-220-b's delta-2 mutation run and Fable's P2-2: with the waits'
    # 2x p99 ceiling set to infinity every test passed, because each burst
    # fixture also failed the gauge, the first-sample rule or the mean.
    # Here only the ceiling can see the bursts. After F1's overload
    # (45-75 s), one wait in 30 is 0.5 s (about 3x the 0.17 s ceiling)
    # until 135 s: few enough for the chance allowance, a hold's mean
    # stays under 1.25x the baseline's, and the waiting gauge is in band
    # throughout. The effect can't end before the last burst.
    def wait_at(tenth: int) -> float:
        if 450 <= tenth < 750:
            return 2.0
        if 750 <= tenth < 1350 and tenth % 30 == 15:
            return 0.5
        return 0.08 + 0.001 * (tenth % 7)

    waits = [(tenth * S // 10, wait_at(tenth)) for tenth in range(3000)]
    waiting = every_second(0, 300, lambda s: 20.0 if 45 <= s < 75 else float(s % 7))
    signals = Signals(in_flight=None, waits=waits, waiting=waiting)
    context = Context(
        signals,
        Baseline.measure(signals, 0, 45 * S),
        Actions(),
        start_ns=45 * S,
        until_ns=300 * S,
    )
    timing = effect_timing("F1", context)
    last_burst = 1335 * S // 10
    assert timing.end_ns is not None and timing.end_ns > last_burst


def test_queue_recovery_doesnt_hold_through_bursts_under_the_ceilings() -> None:
    # rev-220-b's delta 2, E6: the ceilings bound how far out a sample may
    # be, not how often. One wait in ten at 0.5 s (under the 0.6 s ceiling,
    # twice a baseline p99 that two slow waits set), and a waiting count of
    # 11 one scrape in five (under twice the highest, 6), passed as chance.
    # The hold's means now count them.
    waits = every_second(0, 45, lambda s: 0.3 if s in (10, 30) else 0.05)
    burst_waits = [
        (S * 50 + tenth * S // 10, 0.5 if tenth % 10 == 5 else 0.05)
        for tenth in range(100)
    ]
    waiting = every_second(0, 45, lambda s: float(s % 7))
    burst_gauge = every_second(50, 60, lambda s: 11.0 if s % 5 == 2 else 3.0)
    signals = Signals(
        in_flight=None, waits=waits + burst_waits, waiting=waiting + burst_gauge
    )
    waits_rule, gauge_rule = _queue_criteria_of(
        context(signals, Actions(first_send_ns=48 * S))
    )
    ceiling = 2 * signals_baseline(signals).wait_p99
    assert all(value < ceiling for _t, value in burst_waits)
    assert not waits_rule.holds(50 * S, 60 * S)
    assert not gauge_rule.holds(50 * S, 60 * S)


def signals_baseline(signals: Signals) -> Baseline:
    return Baseline.measure(signals, 0, 45 * S)


def test_the_gauges_mean_slack_is_relative_on_a_wide_band() -> None:
    # rev-220-b's delta-3 closure, G3: a "+1" slack on the mean waiting
    # count swamped the 1.25x on a 0-6 band (ceiling 4 for a mean of 3), so
    # one saturated scrape of 11 in ten (mean 3.8) passed. The ceiling is
    # now 1.25x, with a floor of 1 for a near-empty queue.
    waits = [(tenth * S // 10, 0.08) for tenth in range(600)]
    waiting = every_second(0, 45, lambda s: float(s % 7))
    burst = every_second(50, 60, lambda s: 11.0 if s == 55 else 3.0)
    signals = Signals(in_flight=None, waits=waits, waiting=waiting + burst)
    _waits, gauge = _queue_criteria_of(context(signals, Actions(first_send_ns=48 * S)))
    assert not gauge.holds(50 * S, 60 * S)
    calm = Signals(
        in_flight=None,
        waits=waits,
        waiting=waiting + every_second(50, 60, lambda s: 3.0),
    )
    _waits, calm_gauge = _queue_criteria_of(
        context(calm, Actions(first_send_ns=48 * S))
    )
    assert calm_gauge.holds(50 * S, 60 * S)
    # A near-empty queue keeps a floor of 1 waiting request on the mean.
    empty = Signals(
        in_flight=None,
        waits=waits,
        waiting=every_second(0, 45, lambda s: 0.0)
        + every_second(50, 60, lambda s: 1.0 if s % 2 else 0.0),
    )
    _waits, empty_gauge = _queue_criteria_of(
        context(empty, Actions(first_send_ns=48 * S))
    )
    assert isinstance(empty_gauge, MostlyWithin) and empty_gauge.mean_ceiling == 1.0


def _half_second(start: int, end: int, value: Callable[[int], float]) -> list[Point]:
    return [(half * S // 2, value(half)) for half in range(2 * start, 2 * end)]


def test_the_gauge_ceiling_alone_refuses_a_spike_beyond_it() -> None:
    # rev-220-b's delta-3 closure, G6: with the gauge's 2x ceiling removed
    # no test failed, since every spike fixture also moved the mean. Here
    # one scrape of 13 (over the ceiling of 12) among 20 at 3 keeps the
    # mean at 3.5, under 1.25x the baseline's 3, and one outside sample is
    # within the chance allowance: only the ceiling refuses it.
    waits = [(tenth * S // 10, 0.08) for tenth in range(600)]
    baseline = _half_second(0, 45, lambda half: float(half % 7))
    spike = _half_second(50, 60, lambda half: 13.0 if half == 111 else 3.0)
    calm = _half_second(50, 60, lambda half: 6.0 if half == 111 else 3.0)
    for hold, holds in ((spike, False), (calm, True)):
        signals = Signals(in_flight=None, waits=waits, waiting=baseline + hold)
        _waits, gauge = _queue_criteria_of(
            context(signals, Actions(first_send_ns=48 * S))
        )
        assert gauge.holds(50 * S, 60 * S) is holds


def test_the_chance_allowance_alone_refuses_too_many_mild_waits() -> None:
    # G6: MostlyWithin's allowance, set unlimited, survived. 15 of 100
    # waits just over the baseline's p95 (0.09 s against 0.086 s) are too
    # many for chance (the 99% point of Binomial(100, 0.05) is 11), yet
    # under the ceiling (0.172 s) and the mean bound; 8 are within chance.
    baseline = [(tenth * S // 10, 0.08 + 0.001 * (tenth % 7)) for tenth in range(450)]
    waiting = every_second(0, 60, lambda s: float(s % 7))
    for mild, holds in ((15, False), (8, True)):
        hold = [
            (S * 50 + tenth * S // 10, 0.09 if 0 < tenth <= mild else 0.083)
            for tenth in range(100)
        ]
        signals = Signals(in_flight=None, waits=baseline + hold, waiting=waiting)
        waits_rule, _gauge = _queue_criteria_of(
            context(signals, Actions(first_send_ns=48 * S))
        )
        assert waits_rule.holds(50 * S, 60 * S) is holds


def _queue_criteria_of(ctx: Context) -> list[Criterion]:
    from stormlog.infer.qualify.recovery import _queue_criteria

    return _queue_criteria(ctx)


def test_the_long_gap_allowance_alone_refuses_too_many_long_gaps() -> None:
    # G6: with the allowance unlimited no test failed. A baseline with 10
    # long gaps in 2,000 (twice its p99 is 42 ms; the cap 60 ms) allows
    # about 7 in a 500-gap hold. Nine 50 ms gaps among 20 ms ones are under
    # the cap, keep the mean under 1.25x and the p95 count within chance:
    # only the allowance refuses them. Five are fine.
    baseline = GapStats(
        count=2000, mean=0.020, p95=0.021, p99=0.021, long_count=10, p999=0.2
    )
    for long, holds in ((9, False), (5, True)):
        gaps = [
            0.05 if index % 50 == 25 and index < 50 * long else 0.02
            for index in range(500)
        ]
        steps = list(itertools.accumulate((round(g * S) for g in gaps), initial=0))
        cadence = CadenceWithin(
            Signals(in_flight=ALWAYS, step_starts=steps).busy_step_gaps(),
            baseline,
            Thresholds(),
            ALWAYS,
        )
        assert cadence.holds(0, steps[-1]) is holds


def test_a_hold_needs_its_minimum_samples() -> None:
    # 19 healthy gaps are one short of the cadence minimum; 19 waits and 4
    # waiting counts are one short of theirs.
    baseline = GapStats(count=250, mean=0.020, p95=0.026, p99=0.030)
    steps = [tick * 20 * MS for tick in range(21)]
    cadence = CadenceWithin(
        Signals(in_flight=ALWAYS, step_starts=steps).busy_step_gaps(),
        baseline,
        Thresholds(),
        ALWAYS,
    )
    assert not cadence.holds(0, 380 * MS)  # 19 gaps
    assert cadence.holds(0, 400 * MS)  # 20 gaps
    waits = [(tick * 100 * MS, 0.05) for tick in range(20)]
    rule = MostlyWithin(waits, Thresholds(), min_samples=20, high=0.1)
    assert not rule.holds(0, 1800 * MS) and rule.holds(0, 1900 * MS)
    gauge = [(second * S, 1.0) for second in range(5)]
    rule = MostlyWithin(gauge, Thresholds(), min_samples=5, high=2.0)
    assert not rule.holds(0, 3 * S) and rule.holds(0, 4 * S)


def test_an_idle_gap_in_a_hold_is_not_the_engines() -> None:
    # The victim idles 500 ms just after the last SIGCONT, with no step; the
    # engine is healthy, so recovery holds at once. Judged on every gap,
    # the idle 500 ms would be a long gap and hold recovery off.
    last = PULSES[-1][1]
    idle = (last + 2 * S, last + 2_500 * MS)
    steps = [
        t
        for t in stalled_steps(PULSES, keep_stepping=False)
        if not idle[0] < t < idle[1]
    ]
    signals = Signals(in_flight=[(0, idle[0]), (idle[1], 200 * S)], step_starts=steps)
    actions = Actions(
        first_stop_confirmed_ns=60 * S, last_continue_ns=last, pulses=PULSES
    )
    timing = effect_timing("F4a", context(signals, actions))
    assert timing.end_ns is not None and timing.end_ns - last < S


def test_a_cadence_holds_within_its_rate_tolerance() -> None:
    # Baseline gaps of 10 and 30 ms (mean 20 ms, p95 and p99 30 ms). Gaps of
    # 22 ms are 10% slower on average, inside the 20% tolerance, with none
    # long or above the p95: the cadence is back.
    baseline = GapStats(count=250, mean=0.020, p95=0.030, p99=0.030)
    steps = [tick * 22 * MS for tick in range(40)]
    gaps = Signals(in_flight=ALWAYS, step_starts=steps).busy_step_gaps()
    cadence = CadenceWithin(gaps, baseline, Thresholds(), ALWAYS)
    assert cadence.holds(0, 39 * 22 * MS)
    slow = [tick * 26 * MS for tick in range(40)]  # 30% slower: outside it
    gaps = Signals(in_flight=ALWAYS, step_starts=slow).busy_step_gaps()
    assert not CadenceWithin(gaps, baseline, Thresholds(), ALWAYS).holds(
        0, 39 * 26 * MS
    )


def test_a_queue_hold_cannot_begin_with_an_outside_sample() -> None:
    # One wait above the band (but under the ceiling) then 20 in band: a
    # hold may start after it, not with it, or the effect would end before
    # its last sample.
    waits = [(0, 0.3)] + [(tick * 100 * MS, 0.05) for tick in range(1, 21)]
    rule = MostlyWithin(waits, Thresholds(), min_samples=20, high=0.1, ceiling=1.0)
    assert not rule.holds(0, 2 * S)
    assert rule.holds(1, 2 * S)


def test_the_open_gap_counts_by_itself_and_in_the_mean() -> None:
    # Baseline mean 20 ms (ceiling 25 ms), p95 26 ms, p99 30 ms (longest
    # 60 ms). 10 s of 20 ms gaps then a 100 ms open gap: the mean stays
    # low, but the open gap is long. 20 gaps of 24 ms then a 50 ms open
    # gap: no gap is long, but the open one tips the mean over.
    baseline = GapStats(count=250, mean=0.020, p95=0.026, p99=0.030)

    def cadence(spacing_ms: int, count: int) -> CadenceWithin:
        steps = [tick * spacing_ms * MS for tick in range(count + 1)]
        gaps = Signals(in_flight=ALWAYS, step_starts=steps).busy_step_gaps()
        return CadenceWithin(gaps, baseline, Thresholds(), ALWAYS)

    steady = cadence(20, 500)
    assert steady.holds(0, 10 * S)
    assert not steady.holds(0, 10 * S + 100 * MS)
    tight = cadence(24, 20)
    assert tight.holds(0, 480 * MS)
    assert not tight.holds(0, 530 * MS)


def test_a_queue_twin_without_a_baseline_gauge_never_recovers() -> None:
    # Plenty of baseline waits, but no waiting count was scraped in the
    # baseline: the gauge's band would be [0, 0] by default, and a quiet
    # gauge afterwards would "recover" against it.
    waits = [(tenth * S // 10, 0.08) for tenth in range(0, 2000)]
    waiting = every_second(60, 200, lambda s: 0.0)
    ctx = context(
        Signals(in_flight=None, waits=waits, waiting=waiting),
        Actions(first_send_ns=60 * S),
    )
    assert ctx.baseline.waiting_count == 0
    assert effect_timing("T1", ctx).end_ns is None


def test_a_hit_ratio_without_a_baseline_is_incomplete_not_failed() -> None:
    # The episode has ratios but the baseline has none to compare them with.
    cached = every_second(0, 200, lambda s: 0.95)
    ratios = every_second(60, 200, lambda s: 0.4)
    ctx = context(
        Signals(in_flight=None, cached_fraction=cached, engine_hit_ratio=ratios),
        Actions(first_send_ns=60 * S),
    )
    realized, checks = realization("T3b", ctx, effect_timing("T3b", ctx))
    assert realized and checks[1].incomplete
