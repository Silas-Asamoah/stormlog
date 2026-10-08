"""Pulse recovery on jittered step cadences (#221 design A.4).

An engine back at its baseline must recover soon after the last SIGCONT;
an engine still degraded must not. The degraded engines are the reviewers'
attacks on the median rule: a slow minority of steps, bimodal stalls,
uniform slowness hidden among idle gaps, pulses the injector never recorded,
and a slow resume.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import pytest

from stormlog.infer.qualify.recovery import (
    Actions,
    Baseline,
    Context,
    Signals,
    Thresholds,
    Timing,
    effect_timing,
    recovery_blocked,
)

S = 1_000_000_000
MS = 1_000_000

# A victim request always in flight: every step gap is a busy one.
ALWAYS = ((0, 10**18),)
BASELINE_END = 45 * S
FIRST_PULSE = 55 * S
DEGRADED_FOR = 60 * S

Gap = Callable[[random.Random, float], float]


def busy(rng: random.Random) -> float:
    """A busy engine's step gap: about 20 ms, jittered."""
    return rng.lognormvariate(0, 0.25) * 0.020


def slow_share(share: float, factor: float = 10.0) -> Gap:
    return lambda rng, _t: busy(rng) * (factor if rng.random() < share else 1.0)


def periodic(normal_s: float, stalled_s: float, gap_s: float) -> Gap:
    """``normal_s`` of busy steps, then ``stalled_s`` of ``gap_s`` gaps."""
    period = normal_s + stalled_s
    return lambda rng, t: busy(rng) if t % period < normal_s else gap_s


@dataclass
class Engine:
    """Step starts, and when a victim request was in flight."""

    steps: list[int] = field(default_factory=list)
    idle: list[tuple[int, int]] = field(default_factory=list)

    def in_flight(self) -> list[tuple[int, int]]:
        """Everything between idle gaps, as A2's reference channel gives it."""
        intervals, start = [], 0
        for before, after in self.idle:
            intervals.append((start, before))
            start = after
        intervals.append((start, self.steps[-1]))
        return intervals


PULSES = [
    (FIRST_PULSE + i * 2 * S, FIRST_PULSE + i * 2 * S + 100 * MS) for i in range(10)
]
LAST = PULSES[-1][1]


def run(
    seed: int,
    degraded: Gap | None,
    *,
    idle_share: float = 0.0,
    unrecorded: list[tuple[int, int]] | None = None,
    stall_after_last_s: float = 0.0,
    idle_when_degraded: Callable[[float], bool] | None = None,
    known_in_flight: bool = True,
) -> Timing:
    """F4a: 45 s of baseline, ten recorded 100 ms pulses, 60 s of the
    degraded engine after the last SIGCONT, then the baseline engine."""
    rng = random.Random(seed)
    engine = Engine()
    stops = PULSES + (unrecorded or [])
    at = 0
    while at < LAST + 200 * S:
        engine.steps.append(at)
        idle = False
        if LAST <= at < LAST + DEGRADED_FOR and degraded is not None:
            gap = degraded(rng, (at - LAST) / S)
            idle = idle_when_degraded is not None and idle_when_degraded(gap)
        elif at < LAST and rng.random() < idle_share:
            gap, idle = rng.uniform(0.15, 0.40), True
        else:
            gap = busy(rng)
        if at == engine.steps[-1] and LAST <= at < LAST + 50 * MS:
            gap = max(gap, stall_after_last_s)
        following = at + int(gap * S)
        for stop, cont in stops:
            if stop < following < cont:
                following = cont
        if idle:
            engine.idle.append((at, following))
        at = following
    in_flight = engine.in_flight() if known_in_flight else None
    signals = Signals(step_starts=engine.steps, in_flight=in_flight)
    actions = Actions(
        first_stop_confirmed_ns=FIRST_PULSE, last_continue_ns=LAST, pulses=PULSES
    )
    baseline = Baseline.measure(signals, 0, BASELINE_END)
    context = Context(
        signals,
        baseline,
        actions,
        start_ns=FIRST_PULSE,
        until_ns=LAST + Thresholds().recovery_timeout_ns,
    )
    return effect_timing("F4a", context)


def recovered_by(timing: Timing, seconds_after_last: float) -> bool:
    return timing.end_ns is not None and timing.end_ns - LAST <= seconds_after_last * S


SEEDS = range(4)


@pytest.mark.parametrize("idle_share", [0.0, 0.06])
def test_an_engine_back_at_its_baseline_recovers_at_once(idle_share: float) -> None:
    timings = [run(seed, None, idle_share=idle_share) for seed in SEEDS]
    assert all(recovered_by(timing, 1.0) for timing in timings)


DEGRADED: dict[str, Gap] = {
    "40% of steps 10x slow": slow_share(0.40),
    "45% of steps 10x slow": slow_share(0.45),
    "49% of steps 10x slow": slow_share(0.49),
    "51% of steps 10x slow": slow_share(0.51),
    "bimodal: 2 s normal, 2 s of 500 ms gaps": periodic(2.0, 2.0, 0.5),
    "bimodal: stopped 400 ms of every 1 s": periodic(0.6, 0.4, 0.4),
    "every step 3x slow": lambda rng, _t: busy(rng) * 3,
    "every step 2x slow": lambda rng, _t: busy(rng) * 2,
    "40% of steps 3x slow": slow_share(0.40, 3.0),
    # Within the rate tolerance and no long gap: only the share above the
    # baseline's p95 shows it.
    "30% of steps 1.6x slow": slow_share(0.30, 1.6),
}


@pytest.mark.parametrize("idle_share", [0.0, 0.06])
@pytest.mark.parametrize("name", sorted(DEGRADED))
def test_an_engine_still_degraded_does_not_recover(
    name: str, idle_share: float
) -> None:
    timings = [run(seed, DEGRADED[name], idle_share=idle_share) for seed in SEEDS]
    # The degradation lasts 60 s; recovery can't begin much before it ends.
    assert not any(recovered_by(timing, 55.0) for timing in timings)


@pytest.mark.parametrize(
    "name", ["every step 2x slow", "bimodal: 2 s normal, 2 s of 500 ms gaps"]
)
def test_without_in_flight_intervals_cadence_never_recovers(name: str) -> None:
    # A caller that doesn't know when the victim was in flight says None.
    # No gap can then be shown busy, so recovery never holds: not for a
    # degraded engine, which every gap would let recover at once with 6%
    # idle gaps in the baseline, and not for a healthy one either.
    for degraded in (DEGRADED[name], None):
        timings = [
            run(seed, degraded, idle_share=0.06, known_in_flight=False)
            for seed in SEEDS
        ]
        assert all(timing.end_ns is None for timing in timings)


@pytest.mark.parametrize("idle_share", [0.0, 0.06])
def test_slow_steps_between_idle_gaps_do_not_recover(idle_share: float) -> None:
    # Half the gaps are 100 ms with nothing in flight; the busy ones are 7x
    # slow. Idle time measures the traffic, so only the slow steps count.
    def diluted(rng: random.Random, _t: float) -> float:
        return rng.uniform(0.08, 0.12) if rng.random() < 0.5 else busy(rng) * 7

    timings = [
        run(
            seed,
            diluted,
            idle_share=idle_share,
            idle_when_degraded=lambda gap: 0.08 <= gap <= 0.12,
        )
        for seed in SEEDS
    ]
    assert not any(recovered_by(timing, 55.0) for timing in timings)


@pytest.mark.parametrize("idle_share", [0.0, 0.06])
def test_pulses_the_injector_never_recorded_hold_recovery_off(
    idle_share: float,
) -> None:
    unrecorded = [
        (LAST + (1 + i) * 2 * S, LAST + (1 + i) * 2 * S + 100 * MS) for i in range(29)
    ]
    timings = [
        run(seed, None, idle_share=idle_share, unrecorded=unrecorded) for seed in SEEDS
    ]
    assert not any(recovered_by(timing, 55.0) for timing in timings)


@pytest.mark.parametrize("idle_share", [0.0, 0.06])
def test_a_slow_resume_ends_the_effect_after_the_stall(idle_share: float) -> None:
    timings = [
        run(seed, None, idle_share=idle_share, stall_after_last_s=2.0) for seed in SEEDS
    ]
    for timing in timings:
        assert timing.end_ns is not None
        assert 2 * S <= timing.end_ns - LAST <= 3 * S


INTERMITTENT: dict[str, Gap] = {
    # The mean stays within tolerance (0.89x): only the long gaps show.
    "1% of gaps 20x, the rest 0.7x": lambda rng, _t: busy(rng)
    * (20 if rng.random() < 0.01 else 0.7),
    "5% of gaps 3x": slow_share(0.05, 3.0),
    "every step 1.15x slow": lambda rng, _t: busy(rng) * 1.15,
}


@pytest.mark.parametrize("name", sorted(INTERMITTENT))
def test_the_rules_resolution_for_mild_or_intermittent_degradation(name: str) -> None:
    # The search tries every start, so a stretch of a degraded engine that
    # happens to look normal for a whole hold is found. Over 10 s that is
    # rare: of 20 seeds, at most a quarter recover more than 10 s before the
    # 60 s of degradation end (1 and 4 for the intermittent engines, none
    # for a uniform 1.15x). A 5 s hold let 11, 13 and 12 of 20 through.
    early = 0
    for seed in range(20):
        timing = run(seed, INTERMITTENT[name])
        assert timing.end_ns is not None, seed
        early += timing.end_ns - LAST < DEGRADED_FOR - 10 * S
    assert early <= 5


def lognormal_steps(rng: random.Random, start: int, end: int) -> list[int]:
    steps, at = [], start
    while at < end:
        steps.append(at)
        at += int(rng.lognormvariate(0, 0.25) * 0.020 * S)
    return steps


@pytest.mark.parametrize("episode_type", ["F4a", "F4b"])
def test_jittered_engines_recover_in_every_seed(episode_type: str) -> None:
    # The reviewers' noise probe: after ten pulses every series returns to
    # the baseline's distribution, so recovery must hold in all 40 seeds.
    # Over a 10 s hold, 39 recover at the SIGCONT; one waits 13.5 s for a
    # window without a chance run of exceedances.
    for seed in range(40):
        rng = random.Random(seed)
        pulses = [
            (55 * S + i * 2 * S, 55 * S + i * 2 * S + 100 * MS) for i in range(10)
        ]
        steps = lognormal_steps(rng, 0, 55 * S)
        for (_stop, cont), (stop, _cont) in zip(pulses, pulses[1:]):
            steps += lognormal_steps(rng, cont, stop)
        steps += lognormal_steps(rng, pulses[-1][1], pulses[-1][1] + 200 * S)
        chunks = [
            (at * 50 * MS, rng.lognormvariate(0, 0.25) * 0.05)
            for at in range(1, (pulses[-1][1] + 200 * S) // (50 * MS))
        ]
        signals = Signals(in_flight=ALWAYS, step_starts=steps, chunk_gaps=chunks)
        actions = Actions(
            first_stop_confirmed_ns=pulses[0][0],
            last_continue_ns=pulses[-1][1],
            pulses=pulses,
        )
        context = Context(
            signals,
            Baseline.measure(signals, 0, 45 * S),
            actions,
            start_ns=55 * S,
            until_ns=pulses[-1][1] + 150 * S,
        )
        timing = effect_timing(episode_type, context)
        assert timing.end_ns is not None, seed
        assert timing.end_ns - pulses[-1][1] <= 15 * S, seed


def test_the_front_end_waits_for_its_chunk_cadence_too() -> None:
    # Steps are back at once, but the victim's chunks keep arriving 3x apart
    # for 60 s: the API server stall (F4b) has not recovered.
    rng = random.Random(7)
    steps = lognormal_steps(rng, 0, LAST + 200 * S)
    chunks = [
        (
            at * 50 * MS,
            rng.lognormvariate(0, 0.25)
            * 0.05
            * (3 if LAST <= at * 50 * MS < LAST + DEGRADED_FOR else 1),
        )
        for at in range(1, (LAST + 200 * S) // (50 * MS))
    ]
    signals = Signals(in_flight=ALWAYS, step_starts=steps, chunk_gaps=chunks)
    actions = Actions(
        first_stop_confirmed_ns=FIRST_PULSE, last_continue_ns=LAST, pulses=PULSES
    )
    context = Context(
        signals,
        Baseline.measure(signals, 0, BASELINE_END),
        actions,
        start_ns=FIRST_PULSE,
        until_ns=LAST + 150 * S,
    )
    stall = effect_timing("F4a", context)
    frontend = effect_timing("F4b", context)
    assert recovered_by(stall, 1.0)
    assert frontend.end_ns is not None
    assert frontend.end_ns - LAST >= 55 * S


@pytest.mark.parametrize(
    ("label", "factor", "share"),
    [
        ("every gap 2x", 2.0, 1.0),
        ("40% of gaps 3x", 3.0, 0.40),
        ("55% of gaps 3x", 3.0, 0.55),
    ],
)
def test_an_idling_engines_slowdown_does_not_recover(
    label: str, factor: float, share: float
) -> None:
    # Lens-a's case: exponential 30 ms gaps (an engine that idles between
    # requests, without in-flight intervals), slowed for 150 s after the last
    # SIGCONT. Every pulse recovery must stay open.
    for seed in range(4):
        rng = random.Random(seed)
        steps, at = [], 0
        while at < LAST + 150 * S:
            steps.append(at)
            gap = rng.expovariate(1 / 0.030)
            if at >= LAST and rng.random() < share:
                gap *= factor
            at += int(gap * S)
            for stop, cont in PULSES:
                if stop < at < cont:
                    at = cont
        signals = Signals(in_flight=ALWAYS, step_starts=steps)
        actions = Actions(
            first_stop_confirmed_ns=FIRST_PULSE, last_continue_ns=LAST, pulses=PULSES
        )
        context = Context(
            signals,
            Baseline.measure(signals, 0, BASELINE_END),
            actions,
            start_ns=FIRST_PULSE,
            until_ns=LAST + 150 * S,
        )
        assert effect_timing("F4a", context).end_ns is None, (label, seed)


@pytest.mark.parametrize("runs_s", [0.5, 1.0, 4.0])
def test_an_engine_that_stalls_after_resuming_has_not_recovered(runs_s: float) -> None:
    # rev-220-b's D1: back at its baseline for a moment after the last
    # SIGCONT, then hung for 30 s with requests in flight. The gaps before
    # the hang look normal; the hang is a gap still open at the hold's end,
    # so the effect can't end until the engine steps again.
    def resume_then_hang() -> Gap:
        hung = [False]

        def gap(rng: random.Random, t: float) -> float:
            if t >= runs_s and not hung[0]:
                hung[0] = True
                return 30.0
            return busy(rng)

        return gap

    for seed in SEEDS:
        timing = run(seed, resume_then_hang())
        assert timing.end_ns is not None
        assert timing.end_ns - LAST >= (runs_s + 29) * S, seed


def with_prefill(share: float, decode: float = 0.005, prefill: float = 0.025) -> Gap:
    """5 ms decode steps, with ``share`` of them 25 ms prefill steps: each
    prefill step longer than twice the p99, and shorter than the smallest
    F4a/F4b dose (60 ms), which G0 checks the real engines are."""

    def gap(rng: random.Random, _t: float) -> float:
        length = prefill if rng.random() < share else decode
        return length * rng.lognormvariate(0, 0.2)

    return gap


def prefill_run(
    seed: int,
    share: float,
    stalls: list[tuple[int, int]],
    *,
    decode: float = 0.005,
    prefill: float = 0.025,
) -> Timing:
    """F4a on an engine with prefill steps all along: baseline, ten pulses,
    and ``stalls`` that nobody recorded (in the baseline too, if there)."""
    context = prefill_context(seed, share, stalls, decode=decode, prefill=prefill)
    return effect_timing("F4a", context)


def prefill_context(
    seed: int,
    share: float,
    stalls: list[tuple[int, int]],
    *,
    decode: float = 0.005,
    prefill: float = 0.025,
) -> Context:
    rng, steps, at = random.Random(seed), [], 0
    gap = with_prefill(share, decode, prefill)
    while at < LAST + 200 * S:
        steps.append(at)
        at += int(gap(rng, 0.0) * S)
        for stop, cont in PULSES + stalls:
            if stop < at < cont:
                at = cont
    signals = Signals(step_starts=steps, in_flight=ALWAYS)
    actions = Actions(
        first_stop_confirmed_ns=FIRST_PULSE, last_continue_ns=LAST, pulses=PULSES
    )
    return Context(
        signals,
        Baseline.measure(signals, 0, BASELINE_END),
        actions,
        start_ns=FIRST_PULSE,
        until_ns=LAST + Thresholds().recovery_timeout_ns,
    )


@pytest.mark.parametrize("share", [0.006, 0.008, 0.010])
def test_an_engine_with_rare_prefill_steps_recovers(share: float) -> None:
    # rev-220-b's delta 2, E5: with under 1% of steps prefill, the p99 is a
    # decode step's, so every prefill step was a long gap. A hold needed
    # 10 s without one, and a healthy engine took a median of 19-35 s to
    # recover, or timed out (12 of 20 recovered at 0.8%). Long gaps up to
    # the baseline's own share of them are now normal. A prefill step in
    # the tail past the 60 ms cap still ends a hold, so a seed may take a
    # few seconds (one of 20 at 1%: 6.2 s).
    timings = [prefill_run(seed, share, []) for seed in range(20)]
    assert all(recovered_by(timing, 10.0) for timing in timings)
    assert sum(recovered_by(timing, 1.0) for timing in timings) >= 19


@pytest.mark.parametrize("share", [0.004, 0.008])
def test_an_engine_whose_prefill_steps_outlast_a_dose_fails_the_dose_check(
    share: float,
) -> None:
    # Astra's closure of delta 3, H4: with 250 ms prefill steps (past the
    # 60 ms cap) a healthy engine fell back to the strict rule, and
    # recovered at the last SIGCONT in 3 of 20 at 0.8%, timing out in 8.
    # G0's dose check was only in the plan. Now a baseline whose gaps too
    # long for a hold recur at one or more per hold says so, and the rule
    # never holds; a seed whose tail the rule tolerates recovers at once.
    blocked = 0
    for seed in range(20):
        context = prefill_context(seed, share, [], decode=0.020, prefill=0.250)
        reasons = recovery_blocked("F4a", context)
        timing = effect_timing("F4a", context)
        if reasons:
            blocked += 1
            assert reasons[0].startswith("dose_check_failed: "), reasons
            assert "busy step gaps" in reasons[0]
            assert timing.end_ns is None, seed
        else:
            assert recovered_by(timing, 1.0), seed
    assert blocked >= 15


def test_the_dose_check_passes_short_prefill_steps_and_a_paused_baseline() -> None:
    # The engines the cap was built for: 25 ms prefill steps at 1%, and G2's
    # baseline with three 1 s pauses (0.67 such gaps per hold), which the
    # cap keeps from widening the tolerance, not from recovering.
    pauses = [(t * S, t * S + S) for t in (10, 20, 30)]
    for seed in range(5):
        short = prefill_context(seed, 0.010, [])
        paused = prefill_context(seed, 0.0, pauses, decode=0.020)
        assert recovery_blocked("F4a", short) == ()
        assert recovery_blocked("F4a", paused) == ()


def test_stalls_as_long_as_a_dose_still_hold_recovery_off() -> None:
    # The allowance is for gaps like the baseline's, and never for one as
    # long as the smallest F4a/F4b dose: a 100 ms stall every 2 s for 40 s
    # after the last SIGCONT keeps any hold from starting before the last.
    stalls = [(LAST + k * 2 * S, LAST + k * 2 * S + 100 * MS) for k in range(1, 20)]
    for seed in range(20):
        timing = prefill_run(seed, 0.008, stalls)
        assert timing.end_ns is not None and timing.end_ns >= stalls[-1][1], seed


def test_a_baseline_with_long_pauses_never_widens_the_tolerance() -> None:
    # rev-220-b's delta-3 closure, G2: twice the baseline's p99.9 rests on
    # its few largest gaps, so three 1 s pauses in a 20 ms engine's
    # baseline let 1 s stalls every 10 s recover early in 16 of 20 runs.
    # The tolerance is capped at the smallest dose, 60 ms.
    pauses = [(t * S, t * S + S) for t in (10, 20, 30)]
    stalls = [(LAST + k * 10 * S + 2 * S, LAST + k * 10 * S + 3 * S) for k in range(6)]
    for seed in range(20):
        timing = prefill_run(seed, 0.0, pauses + stalls, decode=0.020)
        assert timing.end_ns is not None and timing.end_ns >= stalls[-1][1], seed


def test_stalls_between_the_cap_and_ten_times_it_hold_recovery_off() -> None:
    # Astra's closure of delta 3, H7: no test told the 60 ms cap from 600 ms.
    # On G2's baseline with three 1 s pauses (twice its p99.9 is 2 s), a cap
    # of 600 ms tolerates 300 ms stalls every 4 s by the baseline's share
    # of long gaps, and the effect ends among them; at 60 ms none is.
    pauses = [(t * S, t * S + S) for t in (10, 20, 30)]
    stalls = [
        (LAST + k * 4 * S + S, LAST + k * 4 * S + S + 300 * MS) for k in range(10)
    ]
    for seed in range(4):
        timing = prefill_run(seed, 0.0, pauses + stalls, decode=0.020)
        assert timing.end_ns is not None and timing.end_ns >= stalls[-1][1], seed
