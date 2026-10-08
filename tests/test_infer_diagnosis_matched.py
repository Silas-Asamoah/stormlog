"""The matched design: units, matching, and its uncertainty."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_matched import (
    Band,
    Columns,
    Design,
    Epoch,
    Side,
    Span,
    Window,
    circular_resample,
    interference,
    matched_effects,
    nearest_medians,
    victim_matrix,
)
from stormlog.infer.diagnosis_selection import select
from stormlog.infer.diagnosis_units import Prefill, Unit, epoch_units
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run

SECOND = 1_000_000_000


def unit(
    at_ms: float,
    cadence_ms: float,
    *,
    running: int = 8,
    prefill: int = 0,
    decoders: int | None = None,
    context: float = 100.0,
    after_refill: bool = False,
) -> Unit:
    prefills = (Prefill("p", prefill, 0, 0),) if prefill else ()
    count = decoders if decoders is not None else running - len(prefills)
    return Unit(
        iteration=f"i{at_ms}",
        completed_ns=int(at_ms * MS),
        cadence_ns=int(cadence_ms * MS),
        running=running,
        drafts=0,
        refill=False,
        after_refill=after_refill,
        context=context,
        decoders=tuple(f"d{k}" for k in range(count)),
        prefills=prefills,
    )


def effects(units: list[Unit], subject: tuple[float, float]) -> list[float]:
    window = (int(subject[0] * MS), int(subject[1] * MS))
    found = matched_effects(Span.of(units, window, Design().window_ns), Design())
    return [round(e / MS, 3) for e in found.effects]


def test_a_treated_step_is_compared_with_the_same_batch_spent_on_decode() -> None:
    """Run 1's shape: 8 running at saturation, 3 ms a step; a step that
    spends one slot on prefill takes 0.5 ms more; a step of 7 decoders is
    the async bubble after a finish, 12 ms. The treated step's controls are
    the 8-decoder steps, never the bubble, which shares its decoder count."""
    units = [unit(k * 3.0, 3.0) for k in range(100)]
    units += [unit(300 + k * 12.0, 12.0, running=7) for k in range(20)]
    units += [unit(600.0, 3.5, prefill=100)]

    assert effects(units, (590, 610)) == [0.5]


def test_a_step_after_a_refill_short_one_is_compared_with_its_like() -> None:
    """The step after one that ran short completes late either way: it is
    matched with others that followed one."""
    units = [unit(k * 3.0, 3.0) for k in range(60)]
    units += [unit(200 + k * 12.0, 12.0, after_refill=True) for k in range(10)]
    units += [unit(400.0, 12.5, prefill=100, after_refill=True)]

    assert effects(units, (390, 410)) == [0.5]


def test_the_context_band_is_on_the_mean_per_decode_member() -> None:
    """Six decoders and two prefills against eight decoders, each decoder at
    100 tokens of context: their sums differ by a third, their means not."""
    units = [unit(k * 3.0, 3.0, context=100.0) for k in range(20)]
    units += [unit(100.0, 4.0, prefill=50, decoders=6, context=100.0)]
    units += [unit(103.0, 4.0, prefill=50, decoders=6, context=200.0)]

    assert effects(units, (90, 110)) == [1.0]


def test_controls_are_only_the_past_within_the_window() -> None:
    later = [unit(100 + k * 3.0, 3.0) for k in range(20)]
    stale = [unit(k * 3.0, 3.0) for k in range(20)]  # over 30 s before
    treated = unit(31_000.0 + 60, 4.0, prefill=100)

    assert effects([unit(50.0, 4.0, prefill=100), *later], (40, 60)) == []
    assert effects([*stale, treated], (31_000, 31_100)) == []


def test_the_nearest_controls_in_time_are_used() -> None:
    """68 older controls at 10 ms, then the 32 nearest at 3 ms."""
    units = [unit(k * 10.0, 10.0) for k in range(68)]
    units += [unit(700 + k * 3.0, 3.0) for k in range(32)]
    units += [unit(1000.0, 3.5, prefill=100)]

    assert effects(units, (990, 1010)) == [0.5]


def test_a_treated_step_needs_five_controls() -> None:
    four = [unit(k * 3.0, 3.0) for k in range(4)]
    treated = unit(50.0, 3.5, prefill=100)

    assert effects([*four, treated], (40, 60)) == []
    assert effects([*four, unit(13.0, 3.0), treated], (40, 60)) == [0.5]


def test_support_counts_every_treated_step() -> None:
    units = [unit(k * 3.0, 3.0) for k in range(20)]
    units += [unit(100.0, 3.5, prefill=100), unit(110.0, 3.5, prefill=100, running=5)]
    span = Span.of(units, (90 * MS, 120 * MS), Design().window_ns)

    found = matched_effects(span, Design())

    assert (found.treated, found.matched, found.support) == (2, 1, 0.5)


def _brute(
    target: Side, pool: Side, bands: list[Band], window: Window, nearest: int
) -> list[float]:
    """The rule written out row by row."""
    out = []
    for i, at in enumerate(target.time):
        rows = []
        for j, when in enumerate(pool.time):
            if pool.key[j] != target.key[i]:
                continue
            if window.causal and when >= at:
                continue
            if window.before_ns is not None and when < at - window.before_ns:
                continue
            mine = target.bands[i]
            if all(
                abs(pool.bands[j, c] - mine[c])
                <= max(b.tolerance * abs(mine[c]), b.slack)
                for c, b in enumerate(bands)
            ):
                rows.append((abs(int(when) - int(at)), j))
        rows.sort()
        chosen = [pool.value[j] for _, j in rows[:nearest]]
        out.append(float(np.median(chosen)) if len(chosen) >= 3 else float("nan"))
    return out


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seed", range(6))
def test_the_nearest_rows_are_found_exactly(causal: bool, seed: int) -> None:
    """The vectorised look widens until no closer qualifying row is left:
    it agrees with the rule written out row by row."""
    rng = np.random.default_rng(seed)

    def side(n: int) -> Side:
        return Side(
            time=np.sort(rng.choice(10_000, size=n, replace=False)).astype(np.int64),
            key=rng.integers(0, 3, size=n).astype(np.int64),
            bands=rng.uniform(50, 150, size=(n, 1)),
            value=rng.normal(size=n),
        )

    target, pool = side(40), side(900)
    bands = [Band("context", 0.05, 2.0)]
    window = Window(None if seed % 2 else 4_000, causal)

    found, _ = nearest_medians(target, pool, bands, window, nearest=7, minimum=3)

    np.testing.assert_allclose(found, _brute(target, pool, bands, window, 7))


def test_units_are_steps_whose_decode_set_continues(tmp_path: Path) -> None:
    """A request's first step prefills it beside the running decoders. The
    engine's first step only prefilled, and its second continued no decode
    set: neither is a unit."""
    requests = [
        SimRequest(f"r{k}", k * 25 * MS, prompt=40, output=30) for k in range(4)
    ]
    view = join(read_input(build_run(tmp_path, requests, Engine())))
    context = Context(view, select(view))
    (producer,) = {e.producer for e in view.executions.values()}

    units = epoch_units(context, producer).units

    treated = [u for u in units if u.treated]
    assert [u.dose for u in treated] == [40, 40, 40]
    assert all(u.decoders and u.running == len(u.decoders) + 1 for u in treated)
    assert len(units) == len(view.iterations) - 2
    assert all(u.cadence_ns == pytest.approx(10.1 * MS, abs=1) for u in units)


def test_columns_bin_the_dose_and_keep_times_exact() -> None:
    units = [unit(1.0, 3.0, prefill=d) for d in (0, 256, 257, 1025)]
    late = Unit("i", 1_790_000_000_000_000_001, 3, 1, 0, False, False, 1.0, ("d",))

    columns = Columns.of([*units, late])

    assert list(columns.exact["bin"]) == [-1, 0, 1, 2, -1]
    assert int(columns.time[-1]) == 1_790_000_000_000_000_001


# ------------------------------------------------------------- bootstrap
def synthetic(
    rng: np.random.Generator,
    *,
    seconds: float = 40.0,
    cadence_ms: float = 3.3,
    subject_s: tuple[float, float] = (30.0, 40.0),
    effect_ms: float = 0.0,
    treated_share: float = 0.1,
    requests: int = 40,
) -> Epoch:
    """An engine at saturation: cadence with serial noise, the async bubble
    after a finish (one step late, the next after it marked), prefill on a
    share of steps with ``effect_ms`` more per step, and the subject's
    requests decoding throughout."""
    units: list[Unit] = []
    at, noise, previous_refill = 0.0, 0.0, False
    while at < seconds * 1000:
        noise = 0.8 * noise + rng.normal(0, 0.2)
        refill = bool(rng.random() < 0.05)
        cadence = cadence_ms + noise + (9.0 if refill else 0.0)
        dose = int(rng.integers(20, 2000)) if rng.random() < treated_share else 0
        cadence += effect_ms if dose else 0.0
        at += max(cadence, 0.5)
        units.append(
            Unit(
                iteration=f"i{len(units)}",
                completed_ns=int(at * MS),
                cadence_ns=int(max(cadence, 0.5) * MS),
                running=8,
                drafts=0,
                refill=refill,
                after_refill=previous_refill,
                context=100.0 + at / 1000,
                decoders=tuple(f"a{int(rng.integers(requests))}" for _ in range(7)),
                prefills=(Prefill("p", dose, 0, 0),) if dose else (),
            )
        )
        previous_refill = refill
    subject = (int(subject_s[0] * SECOND), int(subject_s[1] * SECOND))
    span = Span.of(units, subject, Design().window_ns)
    index = {f"a{k}": k for k in range(requests)}
    return Epoch(span, victim_matrix(span.columns, index, requests))


def test_the_circular_bootstrap_draws_the_span_s_end_as_often_as_its_middle() -> None:
    """A moving block from origins inside the span draws its last second
    about half as often as its middle; wrapping draws every unit alike."""
    times = np.arange(0, 20 * SECOND, 10 * MS)
    rng = np.random.default_rng(1)
    drawn = np.zeros(len(times))
    for _ in range(2000):
        rows, resampled = circular_resample(times, SECOND, rng)
        np.add.at(drawn, rows, 1)
        assert len(rows) == len(times) or abs(len(rows) - len(times)) <= 100
        assert np.all(np.diff(resampled) > 0)

    last = drawn[times >= 19 * SECOND].mean()
    middle = drawn[(times >= 9 * SECOND) & (times < 10 * SECOND)].mean()
    assert last / middle == pytest.approx(1.0, abs=0.08)


def test_shared_controls_widen_the_interval() -> None:
    """R7: twenty treated steps all compared with the same twenty controls,
    half fast and half slow. Resampling the treated steps alone gives the
    one effect, 5 ms, twenty times; resampling the controls with them gives
    an interval that reaches zero."""
    controls = [unit(k * 1000.0, 5.0 if k % 2 else 15.0) for k in range(20)]
    treated = [unit(20_000 + k * 1000.0, 15.0, prefill=100) for k in range(20)]
    span = Span.of([*controls, *treated], (20 * SECOND, 40 * SECOND), 30 * SECOND)
    epoch = Epoch(span, victim_matrix(span.columns, {}, 0))

    found = interference([epoch], Design(), min_support=0.5)

    assert found.pooled is not None and found.pooled.estimate == pytest.approx(5 * MS)
    assert found.pooled.low is not None and found.pooled.high is not None
    assert found.pooled.low <= 0 < found.pooled.high


def test_the_bootstrap_is_seeded() -> None:
    first = interference(
        [synthetic(np.random.default_rng(3), effect_ms=1.0)],
        Design(),
        min_support=0.5,
        replicates=50,
    )
    again = interference(
        [synthetic(np.random.default_rng(3), effect_ms=1.0)],
        Design(),
        min_support=0.5,
        replicates=50,
    )

    assert first.pooled == again.pooled and first.victims == again.victims
    assert first.pooled is not None and first.pooled.mc_error is not None


def test_a_known_effect_is_recovered_with_its_victims() -> None:
    found = interference(
        [synthetic(np.random.default_rng(4), effect_ms=1.0)],
        Design(),
        min_support=0.5,
        replicates=99,
    )

    assert found.pooled is not None and found.pooled.above(0)
    assert found.pooled.estimate == pytest.approx(1.0 * MS, abs=0.2 * MS)
    assert found.victims is not None and found.victims.estimate > 0
    assert found.support is not None and found.support > 0.9


def _bin_epoch(
    effect_ms: float, dose: int, count: int, *, extra: list[Unit] | None = None
) -> Epoch:
    units = [unit(k * 3.0, 3.0) for k in range(40)]
    units += [unit(200 + k * 3.0, 3.0 + effect_ms, prefill=dose) for k in range(count)]
    span = Span.of([*units, *(extra or [])], (190 * MS, 400 * MS), 30 * SECOND)
    return Epoch(span, victim_matrix(span.columns, {}, 0))


def test_the_pooled_statistic_weighs_matched_bins_and_epochs_by_tokens() -> None:
    """A bin whose steps found no controls is left out, not counted as no
    effect; a ten-step epoch weighs by its tokens, not as one epoch."""
    unmatched = [unit(300 + k * 3.0, 30.0, prefill=2000, running=3) for k in range(5)]
    small = _bin_epoch(1.0, 100, 10, extra=unmatched)  # 1,000 tokens at 1 ms
    large = _bin_epoch(3.0, 1000, 10)  # 10,000 tokens at 3 ms

    alone = interference([small], Design(), min_support=0.0, replicates=0)
    both = interference([small, large], Design(), min_support=0.0, replicates=0)

    assert alone.pooled is not None and alone.pooled.estimate == pytest.approx(MS)
    assert both.pooled is not None
    assert both.pooled.estimate == pytest.approx((1_000 * 1 + 10_000 * 3) / 11_000 * MS)
    assert both.by_dose_n == {"1-256": 10, "257-1024": 10}


def test_a_request_s_contribution_sums_the_treated_steps_it_decoded_in() -> None:
    units = [unit(k * 3.0, 3.0) for k in range(40)]
    units += [unit(200.0, 4.0, prefill=100), unit(203.0, 5.0, prefill=100)]
    span = Span.of(units, (190 * MS, 210 * MS), 30 * SECOND)
    epoch = Epoch(span, victim_matrix(span.columns, {"d0": 0, "x": 1}, 2))

    found = interference([epoch], Design(), min_support=0.0, replicates=0)

    assert list(found.contributions / MS) == pytest.approx([3.0, 0.0])


def test_no_interval_is_drawn_where_none_could_pass_the_gate() -> None:
    alone = [unit(200 + k * 3.0, 4.0, prefill=100) for k in range(5)]
    unmatched = Span.of(alone, (190 * MS, 400 * MS), 30 * SECOND)
    thin = Epoch(unmatched, victim_matrix(unmatched.columns, {}, 0))

    short = interference([thin], Design(), min_support=0.5)
    faster = interference([_bin_epoch(-1.0, 100, 10)], Design(), min_support=0.5)

    assert (short.screened, short.pooled) == ("insufficient_common_support", None)
    assert faster.screened == "no_positive_effect"
    assert faster.pooled is not None and faster.pooled.low is None


def test_null_epochs_rarely_pass_the_gate() -> None:
    """Prefill that costs nothing, with the bubble on: over 200 seeded
    epochs, the pooled interval's lower end is above zero at most 5% of the
    time (one-sided, 2.5% nominal)."""
    passes = 0
    for seed in range(200):
        epoch = synthetic(
            np.random.default_rng(1000 + seed),
            seconds=20.0,
            cadence_ms=30.0,
            subject_s=(12.0, 20.0),
            treated_share=0.3,
        )
        found = interference(
            [epoch], Design(), min_support=0.0, replicates=99, seed=seed
        )
        passes += bool(found.pooled is not None and found.pooled.above(0))

    bound = stats.beta.ppf(0.975, passes + 1, 200 - passes)
    assert passes <= 10, f"{passes} of 200 passed (95% upper bound {bound:.3f})"


def test_a_subject_s_interference_fits_its_budget() -> None:
    """Run 1's size: 18k steps over a minute, a 15 s subject, B = 499. The
    budget is 10 s per subject on the GPU box's CPU; this bound is loose for
    shared CI runners."""
    epoch = synthetic(
        np.random.default_rng(6), seconds=60.0, subject_s=(45.0, 60.0), effect_ms=0.5
    )
    started = time.perf_counter()

    found = interference([epoch], Design(), min_support=0.5)

    elapsed = time.perf_counter() - started
    assert found.pooled is not None and found.pooled.replicates == 499
    assert elapsed < 30.0, f"{elapsed:.1f} s"
