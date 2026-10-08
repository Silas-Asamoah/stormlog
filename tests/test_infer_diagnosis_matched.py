"""The matched design: units, matching, and its uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_matched import (
    Band,
    Columns,
    Design,
    Side,
    Span,
    Window,
    matched_effects,
    nearest_medians,
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
        for j, time in enumerate(pool.time):
            if pool.key[j] != target.key[i]:
                continue
            if window.causal and time >= at:
                continue
            if window.before_ns is not None and time < at - window.before_ns:
                continue
            mine = target.bands[i]
            if all(
                abs(pool.bands[j, c] - mine[c])
                <= max(b.tolerance * abs(mine[c]), b.slack)
                for c, b in enumerate(bands)
            ):
                rows.append((abs(int(time) - int(at)), j))
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
