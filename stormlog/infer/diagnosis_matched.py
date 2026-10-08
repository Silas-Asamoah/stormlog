"""The matched design: did steps that also prefilled complete later than
decode-only steps like them?

Its units are ``diagnosis_units``'. A treated unit scheduled prefill; its
dose is the prefill tokens, binned 1-256, 257-1024 and over 1024. Its
controls are decode-only units of the same epoch that share its match key
exactly: its running requests (decode and prefill members, so a step that
spent a slot on prefill is compared with steps of the same batch that spent
it on decode), its drafts, and whether it and the step before it ran short
of a refill. Under async scheduling a step after a finish runs before the
freed slot is known and completes late; matching on it, rather than
excluding such steps, keeps every treated unit in the support. A control's
decode context, the mean ``computed_before`` per decode member, lies within
15% of the treated unit's. Of the controls completed in the 30 s before it,
the 32 nearest in time are used: only the past, so the design is causal.
With fewer than 5 the treated unit is unmatched, and the share matched is
the common support. A treated unit's effect is its cadence minus its
controls' median.

The interference gate is one pooled statistic: within each epoch, the
median effect of each dose bin, weighted by the bin's share of the matched
treated units' prefill tokens, renormalized over the bins that matched;
across epochs, by the same token share. Per-bin medians are descriptive. A
decode member's ``estimated_contribution`` is the sum of the effects of the
treated units it decoded in, never a measured delay.

Uncertainty comes from a circular block bootstrap over each epoch's span,
from the match window before the subject to its end: 1 s blocks from
uniformly drawn origins, wrapping past the span's end to its start, laid
end to end, so every unit, the subject's last included, is drawn equally
often. Each replicate re-runs the matching on its own resampled units, so
controls shared by many treated units and the serial dependence of
neighbouring steps both widen the interval. B = 499 seeded replicates,
with the Monte Carlo error of the interval's 2.5% quantile reported; the
bootstrap runs only where an interval could decide the gate.

A subject's units are also compared with its reference's, unit for unit:
each subject unit with the 32 reference units nearest in decode context
that share its key and bands, its ratio the cadence over their median, the
statistic the median ratio, its interval from both spans resampled
together. Nearest in context, not time: the reference's last steps abut
the subject, and may be the incident's own beginning. Decode-only units
are matched as controls are; treated units also on their dose bin and
prefill member count, dose, cached prefix and longest prefill, so they
compare the engine's cost of mixing the same prefill into the same batch.
The driver's capacity compares both, each with its like.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from .diagnosis_stats import SEED
from .diagnosis_thresholds import (
    MIXED_BLOCK_NS,
    MIXED_CACHED_SLACK,
    MIXED_CONTEXT_TOLERANCE,
    MIXED_DOSE_TOLERANCE,
    MIXED_MATCH_WINDOW_NS,
    MIXED_MIN_CONTROLS,
    MIXED_NEAREST_CONTROLS,
    resolve_threshold,
)
from .diagnosis_units import Unit

# (low, high) prefill tokens of each dose bin; None is unbounded.
DOSE_BINS: tuple[tuple[int, int | None], ...] = ((1, 256), (257, 1024), (1025, None))
# What a control shares with a treated unit exactly.
CONTROL_KEY = ("running", "drafts", "refill", "after_refill")
EXACT = (*CONTROL_KEY, "bin", "members")
BANDED = ("context", "dose", "cached", "longest")
# Cells one look at the pool may hold: rows times width.
LOOK_CELLS = 2_000_000
REPLICATES = 499
# Block lengths reported beside the chosen one, for findings only.
SENSITIVITY_BLOCKS_NS = (500_000_000, 2_000_000_000)
SENSITIVITY_REPLICATES = 199
INSUFFICIENT_SUPPORT = "insufficient_common_support"
NO_POSITIVE_EFFECT = "no_positive_effect"
NOT_ABOVE_FLOOR = "not_above_floor"


def dose_bin(tokens: int) -> int:
    """The index of the bin a dose falls in; -1 for none."""
    for index, (low, high) in enumerate(DOSE_BINS):
        if tokens >= low and (high is None or tokens <= high):
            return index
    return -1


def bin_name(index: int) -> str:
    low, high = DOSE_BINS[index]
    return f"{low}-{high}" if high is not None else f"{low}+"


@dataclass(frozen=True)
class Columns:
    """Units as columns, one row per unit, in completion order."""

    units: tuple[Unit, ...]
    time: np.ndarray
    cadence: np.ndarray
    exact: Mapping[str, np.ndarray]
    banded: Mapping[str, np.ndarray]

    @classmethod
    def of(cls, units: Sequence[Unit]) -> Columns:
        return cls(
            units=tuple(units),
            time=_column(units, "completed_ns", np.int64),
            cadence=_column(units, "cadence_ns", float),
            exact={name: _column(units, name, np.int64) for name in EXACT},
            banded={name: _column(units, name, float) for name in BANDED},
        )

    def __len__(self) -> int:
        return len(self.units)

    @property
    def treated(self) -> np.ndarray:
        return np.asarray(self.banded["dose"] > 0)


def _column(units: Sequence[Unit], name: str, dtype: type) -> np.ndarray:
    return np.asarray([_value(unit, name) for unit in units], dtype=dtype)


def _value(unit: Unit, name: str) -> int | float:
    """A unit's column value, exact: a time in ns is past a float's
    integer precision."""
    if name == "bin":
        return dose_bin(unit.dose)
    if name == "members":
        return len(unit.prefills)
    value: int | float = getattr(unit, name)
    return value


@dataclass(frozen=True)
class Band:
    """A covariate a match must agree on: within ``tolerance`` of the
    target's value, or within ``slack`` of it when that is wider."""

    column: str
    tolerance: float
    slack: float = 0.0


@dataclass(frozen=True)
class Match:
    """What a target unit shares with the units it is matched with."""

    exact: tuple[str, ...]
    bands: tuple[Band, ...]


@dataclass(frozen=True)
class Design:
    """The matching rule's parameters, from the threshold table."""

    window_ns: int = 30_000_000_000
    tolerance: float = 0.15
    min_controls: int = 5
    nearest: int = 32
    block_ns: int = 1_000_000_000
    # A treated unit's dose, cached prefix and longest prefill against its
    # reference's: within this fraction, the prefix also within the slack.
    dose_tolerance: float = 0.15
    cached_slack: float = 64.0

    @classmethod
    def from_thresholds(cls, overrides: Mapping[str, float] | None) -> Design:
        def value(key: str) -> float:
            return resolve_threshold(key, overrides)[0]

        return cls(
            window_ns=int(value(MIXED_MATCH_WINDOW_NS)),
            tolerance=value(MIXED_CONTEXT_TOLERANCE),
            min_controls=int(value(MIXED_MIN_CONTROLS)),
            nearest=int(value(MIXED_NEAREST_CONTROLS)),
            block_ns=int(value(MIXED_BLOCK_NS)),
            dose_tolerance=value(MIXED_DOSE_TOLERANCE),
            cached_slack=value(MIXED_CACHED_SLACK),
        )

    @property
    def controls(self) -> Match:
        return Match(CONTROL_KEY, (Band("context", self.tolerance),))

    @property
    def treated(self) -> Match:
        """A treated unit's match in another span: the same prefill, split
        the same way, into the same batch. One long prompt costs more
        attention than several short ones of its total."""
        tolerance = self.dose_tolerance
        return Match(
            (*CONTROL_KEY, "bin", "members"),
            (
                Band("context", self.tolerance),
                Band("dose", tolerance),
                Band("cached", tolerance, self.cached_slack),
                Band("longest", tolerance),
            ),
        )


def key_ids(sides: Sequence[Columns], names: Sequence[str]) -> list[np.ndarray]:
    """Each side's exact-match key as one integer per row, in one id space
    across the sides."""
    stacked = (
        np.concatenate(
            [np.stack([side.exact[n] for n in names], axis=1) for side in sides]
        )
        if names
        else np.zeros((sum(len(s) for s in sides), 1), dtype=np.int64)
    )
    if not len(stacked):
        return [np.zeros(0, dtype=np.int64) for _ in sides]
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    bounds = np.cumsum([0, *(len(side) for side in sides)])
    return [ids[a:b] for a, b in zip(bounds, bounds[1:])]


@dataclass(frozen=True)
class Side:
    """Rows to match or match with: their times, key ids, banded values (one
    column per band) and the value whose median is taken."""

    time: np.ndarray
    key: np.ndarray
    bands: np.ndarray
    value: np.ndarray

    @classmethod
    def of(
        cls,
        columns: Columns,
        keys: np.ndarray,
        match: Match,
        rows: np.ndarray,
        times: np.ndarray,
    ) -> Side:
        bands = (
            np.stack([columns.banded[b.column][rows] for b in match.bands], axis=1)
            if match.bands
            else np.zeros((len(rows), 0))
        )
        return cls(times, keys[rows], bands, columns.cadence[rows])


@dataclass(frozen=True)
class Window:
    """Where a target's matches may lie: completed in the ``before_ns``
    before it (None: any time), and only before it (``causal``) or on
    either side."""

    before_ns: int | None
    causal: bool


def nearest_medians(
    target: Side,
    pool: Side,
    bands: Sequence[Band],
    window: Window,
    *,
    nearest: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each target row, the median value of the ``nearest`` pool rows in
    time that share its key, lie in its window and agree with it on every
    band, NaN with fewer than ``minimum``; and how many were used."""
    medians = np.full(len(target.time), np.nan)
    counts = np.zeros(len(target.time), dtype=np.int64)
    if not len(target.time) or not len(pool.time):
        return medians, counts
    index = _Index.of(target, pool, window)
    pending = np.arange(len(target.time))
    width = 2 * nearest
    while pending.size:
        unsure = []
        size = max(1, LOOK_CELLS // (2 * width))
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            look = _Look(index, target, pool, bands, chunk, width, nearest, window)
            done = ~look.unsure()
            counts[chunk[done]] = look.count[done]
            found = _row_medians(look.values[done], look.count[done])
            medians[chunk[done]] = np.where(look.count[done] >= minimum, found, np.nan)
            unsure.append(chunk[~done])
        pending = np.concatenate(unsure)
        width *= 4
    return medians, counts


@dataclass(frozen=True)
class _Index:
    """The pool sorted by (key, time), and each target's place in it: its
    group's slice inside the window, and where its own time falls."""

    order: np.ndarray
    time: np.ndarray  # pool times in that order
    low: np.ndarray
    here: np.ndarray
    high: np.ndarray

    @classmethod
    def of(cls, target: Side, pool: Side, window: Window) -> _Index:
        stamps = np.unique(np.concatenate([target.time, pool.time]))
        span = len(stamps) + 1

        def place(key: np.ndarray, times: np.ndarray) -> np.ndarray:
            return key * span + np.searchsorted(stamps, times, side="left")

        composite = place(pool.key, pool.time)
        order = np.argsort(composite, kind="stable")
        sorted_composite = composite[order]

        def find(values: np.ndarray) -> np.ndarray:
            return np.searchsorted(sorted_composite, values, side="left")

        base = target.key * span
        here = find(place(target.key, target.time))
        if window.before_ns is None:
            low = find(base)
        else:
            low = find(place(target.key, target.time - window.before_ns))
        high = here if window.causal else find(base + span)
        return cls(order, pool.time[order], low, here, high)


class _Look:
    """One look at up to ``width`` pool rows on either side of each pending
    target: the ``nearest`` that qualify, and whether a wider look could
    change them."""

    def __init__(
        self,
        index: _Index,
        target: Side,
        pool: Side,
        bands: Sequence[Band],
        pending: np.ndarray,
        width: int,
        nearest: int,
        window: Window,
    ) -> None:
        self.index, self.pending, self.width = index, pending, width
        self.causal = window.causal
        here = index.here[pending]
        after = 0 if window.causal else width
        self.after = after
        candidates = here[:, None] + np.arange(-width, after)[None, :]
        valid = (candidates >= index.low[pending, None]) & (
            candidates < index.high[pending, None]
        )
        rows = index.order[np.clip(candidates, 0, len(index.order) - 1)]
        for column, band in enumerate(bands):
            mine = target.bands[pending, column][:, None]
            allowed = np.maximum(band.tolerance * np.abs(mine), band.slack)
            valid &= np.abs(pool.bands[rows, column] - mine) <= allowed
        self.at = target.time[pending]
        gap = np.abs(pool.time[rows] - self.at[:, None]).astype(float)
        distance = np.where(valid, gap, np.inf)
        take = min(nearest, distance.shape[1])
        picked = np.argpartition(distance, take - 1, axis=1)[:, :take]
        chosen = np.take_along_axis(distance, picked, axis=1)
        usable = np.isfinite(chosen)
        self.count = usable.sum(axis=1)
        self.full = self.count >= nearest
        self.farthest = np.where(usable, chosen, -np.inf).max(axis=1)
        values = pool.value[np.take_along_axis(rows, picked, axis=1)]
        self.values = np.where(usable, values, np.nan)

    def unsure(self) -> np.ndarray:
        """Rows a wider look could change: short of ``nearest`` with rows
        left to look at, or with a closer row just outside the look."""
        index, pending = self.index, self.pending
        here = index.here[pending]
        before_edge = here - self.width - 1
        open_before = before_edge >= index.low[pending]
        after_edge = here + self.after
        open_after = (after_edge < index.high[pending]) & (not self.causal)
        times = index.time
        last = len(times) - 1
        gap_before = np.where(
            open_before, self.at - times[np.clip(before_edge, 0, last)], np.inf
        )
        gap_after = np.where(
            open_after, times[np.clip(after_edge, 0, last)] - self.at, np.inf
        )
        short = ~self.full & (open_before | open_after)
        closer = self.full & (self.farthest > np.minimum(gap_before, gap_after))
        return np.asarray(short | closer)


def _row_medians(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Each row's median over its first ``count`` values once sorted (NaN
    sorts last)."""
    if not len(values):
        return np.zeros(0)
    ordered = np.sort(values, axis=1)
    last = np.maximum(counts, 1)
    low = np.take_along_axis(ordered, ((last - 1) // 2)[:, None], axis=1)[:, 0]
    high = np.take_along_axis(ordered, (last // 2)[:, None], axis=1)[:, 0]
    return np.where(counts > 0, (low + high) / 2, np.nan)


# ------------------------------------------------------- within a subject
@dataclass(frozen=True)
class Span:
    """One epoch's units from the match window before a subject to its
    end: those completed inside the subject are its own."""

    columns: Columns
    own: np.ndarray  # completed inside the subject
    keys: np.ndarray  # the control key's ids

    @classmethod
    def of(
        cls, units: Sequence[Unit], subject: tuple[int, int], window_ns: int
    ) -> Span:
        kept = [
            u for u in units if subject[0] - window_ns <= u.completed_ns <= subject[1]
        ]
        columns = Columns.of(kept)
        own = (columns.time >= subject[0]) & (columns.time <= subject[1])
        return cls(columns, own, key_ids([columns], CONTROL_KEY)[0])


@dataclass
class Effects:
    """The treated units of a subject and the effects of those matched."""

    treated: int = 0
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    effects: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def matched(self) -> int:
        return len(self.rows)

    @property
    def support(self) -> float | None:
        return self.matched / self.treated if self.treated else None


def matched_effects(
    span: Span,
    design: Design,
    picked: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> Effects:
    """Match the subject's treated units in ``picked`` (rows, possibly
    repeated, at ``times``; all rows at their own times by default) with
    the decode-only units among them."""
    columns = span.columns
    if picked is None or times is None:
        picked, times = np.arange(len(columns)), columns.time
    treated = columns.treated[picked]
    targets = np.flatnonzero(treated & span.own[picked])
    controls = np.flatnonzero(~treated)
    match = design.controls
    medians, _ = nearest_medians(
        Side.of(columns, span.keys, match, picked[targets], times[targets]),
        Side.of(columns, span.keys, match, picked[controls], times[controls]),
        match.bands,
        Window(design.window_ns, causal=True),
        nearest=design.nearest,
        minimum=design.min_controls,
    )
    found = np.isfinite(medians)
    rows = picked[targets][found]
    return Effects(
        treated=len(targets),
        rows=rows,
        effects=columns.cadence[rows] - medians[found],
    )


# ------------------------------------------------------------- bootstrap
@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile 95% interval from the replicates
    in which it could be estimated, and the Monte Carlo error of the
    interval's lower end: half the width of the band its order statistic
    falls in, 95% of the time, over reruns with other seeds."""

    estimate: float
    low: float | None = None
    high: float | None = None
    replicates: int = 0
    mc_error: float | None = None

    def above(self, floor: float) -> bool:
        return self.low is not None and self.low > floor

    def as_dict(self, scale: float = 1.0, digits: int = 3) -> dict[str, Any]:
        def scaled(value: float | None) -> float | None:
            return None if value is None else round(value / scale, digits)

        return {
            "estimate": scaled(self.estimate),
            "ci": [scaled(self.low), scaled(self.high)],
            "replicates": self.replicates,
            "mc_error": scaled(self.mc_error),
        }


def interval(estimate: float, draws: Sequence[float]) -> Interval:
    values = np.sort(np.asarray([d for d in draws if np.isfinite(d)], dtype=float))
    count = len(values)
    if not count:
        return Interval(estimate)
    low = float(values[int(0.025 * (count - 1))])
    high = float(values[int(0.975 * (count - 1))])
    rank, spread = 0.025 * count, 1.96 * np.sqrt(count * 0.025 * 0.975)
    band = (
        values[max(0, int(np.floor(rank - spread)))],
        values[min(count - 1, int(np.ceil(rank + spread)))],
    )
    return Interval(estimate, low, high, count, float(band[1] - band[0]) / 2)


def circular_resample(
    times: np.ndarray, block_ns: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Blocks of ``block_ns`` from origins drawn uniformly over the span,
    each wrapping past the span's end to its start, laid end to end and cut
    at the span's length: the rows they hold, and each row's time in the
    resampled sequence. ``times`` are sorted."""
    if not len(times):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    start = int(times[0])
    length = int(times[-1]) - start + 1
    offset = times - start
    blocks = -(-length // block_ns)
    origins = rng.integers(0, length, size=blocks)
    ends = origins + block_ns
    lows = np.stack([origins, np.zeros(blocks, dtype=np.int64)], axis=1).ravel()
    highs = np.stack([np.minimum(ends, length), np.maximum(ends - length, 0)], axis=1)
    first = np.searchsorted(offset, lows, side="left")
    last = np.searchsorted(offset, highs.ravel(), side="left")
    counts = np.maximum(last - first, 0)
    placed = np.arange(blocks) * block_ns - origins
    shifts = np.stack([placed, placed + length], axis=1).ravel()
    rows = _ranges(first, counts)
    resampled = times[rows] + np.repeat(shifts, counts)
    kept = resampled < start + length
    return rows[kept], resampled[kept]


def _ranges(first: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The concatenated ranges [first, first + count)."""
    total = int(counts.sum())
    if not total:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(first - np.cumsum(counts) + counts, counts)
    return np.asarray(starts + np.arange(total), dtype=np.int64)


# ----------------------------------------------------------- interference
@dataclass(frozen=True)
class Epoch:
    """One epoch's part of a subject: its span, and which of the subject's
    requests each unit decoded for (units by requests)."""

    span: Span
    victims: sparse.csr_matrix


@dataclass
class Interference:
    """The subject's matched interference: the pooled gate statistic, the
    dose bins, and what it cost the subject's requests, in ns."""

    treated: int
    matched: int
    pooled: Interval | None
    by_dose: dict[str, Interval]
    by_dose_n: dict[str, int]
    victims: Interval | None  # the median request's summed effect
    contributions: np.ndarray  # each request's summed effect
    screened: str | None = None  # why no interval was drawn
    sensitivity: dict[str, Interval] = field(default_factory=dict)

    @property
    def support(self) -> float | None:
        return self.matched / self.treated if self.treated else None


@dataclass(frozen=True)
class _Stats:
    """One replicate's (or the data's) statistics."""

    pooled: float
    by_dose: dict[int, float]
    by_dose_n: dict[int, int]
    contributions: np.ndarray

    @classmethod
    def of(cls, epochs: Sequence[Epoch], found: Sequence[Effects]) -> _Stats:
        weighted, weights = 0.0, 0.0
        pooled_bins: dict[int, list[np.ndarray]] = {}
        contributions = np.zeros(_requests(epochs))
        for epoch, effects in zip(epochs, found):
            columns = epoch.span.columns
            bins = columns.exact["bin"][effects.rows]
            doses = columns.banded["dose"][effects.rows]
            for index in np.unique(bins):
                mine = bins == index
                weight = float(doses[mine].sum())
                weighted += weight * float(np.median(effects.effects[mine]))
                weights += weight
                pooled_bins.setdefault(int(index), []).append(effects.effects[mine])
            per_unit = np.bincount(
                effects.rows, weights=effects.effects, minlength=len(columns)
            )
            contributions += epoch.victims.T @ per_unit
        return cls(
            weighted / weights if weights else float("nan"),
            {b: float(np.median(np.concatenate(v))) for b, v in pooled_bins.items()},
            {b: sum(len(x) for x in v) for b, v in pooled_bins.items()},
            contributions,
        )

    @property
    def victims(self) -> float:
        return float(np.median(self.contributions)) if len(self.contributions) else 0.0


def _requests(epochs: Sequence[Epoch]) -> int:
    return int(epochs[0].victims.shape[1]) if epochs else 0


def interference(
    epochs: Sequence[Epoch],
    design: Design,
    *,
    min_support: float,
    replicates: int = REPLICATES,
    seed: int = SEED,
) -> Interference:
    """The point estimates, and their intervals where they can decide the
    gate: with common support and a positive pooled effect."""
    found = [matched_effects(epoch.span, design) for epoch in epochs]
    point = _Stats.of(epochs, found)
    requests = _requests(epochs)
    treated = sum(f.treated for f in found)
    matched = sum(f.matched for f in found)
    result = Interference(
        treated=treated,
        matched=matched,
        pooled=None if np.isnan(point.pooled) else Interval(point.pooled),
        by_dose={bin_name(b): Interval(v) for b, v in sorted(point.by_dose.items())},
        by_dose_n={bin_name(b): n for b, n in sorted(point.by_dose_n.items())},
        victims=Interval(point.victims) if requests else None,
        contributions=point.contributions,
    )
    result.screened = _screen(result, min_support)
    if result.screened is None:
        _draw(result, epochs, design, replicates, seed, point)
    return result


def _screen(result: Interference, min_support: float) -> str | None:
    support = result.support
    if support is None or support < min_support:
        return INSUFFICIENT_SUPPORT
    if result.pooled is None or result.pooled.estimate <= 0:
        return NO_POSITIVE_EFFECT
    return None


def _draw(
    result: Interference,
    epochs: Sequence[Epoch],
    design: Design,
    replicates: int,
    seed: int,
    point: _Stats,
) -> None:
    draws = [
        _Stats.of(epochs, found)
        for found in _replicates(epochs, design, design.block_ns, replicates, seed)
    ]
    result.pooled = interval(point.pooled, [d.pooled for d in draws])
    result.by_dose = {
        bin_name(b): interval(v, [d.by_dose.get(b, np.nan) for d in draws])
        for b, v in sorted(point.by_dose.items())
    }
    if _requests(epochs):
        result.victims = interval(point.victims, [d.victims for d in draws])


def _replicates(
    epochs: Sequence[Epoch], design: Design, block_ns: int, replicates: int, seed: int
) -> list[list[Effects]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(replicates):
        found = []
        for epoch in epochs:
            picked, times = circular_resample(epoch.span.columns.time, block_ns, rng)
            found.append(matched_effects(epoch.span, design, picked, times))
        out.append(found)
    return out


def sensitivity(
    epochs: Sequence[Epoch],
    design: Design,
    pooled: float,
    *,
    replicates: int = SENSITIVITY_REPLICATES,
    seed: int = SEED,
) -> dict[str, Interval]:
    """The pooled statistic's interval at the other block lengths."""
    found = {}
    for block_ns in SENSITIVITY_BLOCKS_NS:
        draws = [
            _Stats.of(epochs, effects).pooled
            for effects in _replicates(epochs, design, block_ns, replicates, seed)
        ]
        found[f"{block_ns / 1e9:g}s"] = interval(pooled, draws)
    return found


def victim_matrix(
    columns: Columns, index: Mapping[str, int], requests: int
) -> sparse.csr_matrix:
    """Units by requests: 1 where the unit decoded for the request, through
    any of its attempts in ``index``."""
    rows, cols = [], []
    for row, unit in enumerate(columns.units):
        for attempt in unit.decoders:
            if attempt in index:
                rows.append(row)
                cols.append(index[attempt])
    data = np.ones(len(rows))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(columns), requests))


# ------------------------------------------------ subject against reference
@dataclass(frozen=True)
class Arms:
    """A subject's units and its reference's, on one epoch."""

    subject: Columns
    reference: Columns


@dataclass
class Ratio:
    """Subject units against matched reference units: the median ratio of
    their cadences, with its interval where it could pass the floor."""

    targets: int
    matched: int
    ratio: Interval | None
    screened: str | None = None

    @property
    def support(self) -> float | None:
        return self.matched / self.targets if self.targets else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ratio": None if self.ratio is None else self.ratio.as_dict(digits=4),
            "common_support": _rounded(self.support),
            "matched": self.matched,
            "units": self.targets,
            "screened": self.screened,
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


@dataclass(frozen=True)
class Comparison:
    """Which units are compared, each with its like: (treated, match) pairs,
    decode-only units by one match and treated units by another; and what
    an interval must clear."""

    matches: tuple[tuple[bool, Match], ...]
    floor: float
    min_support: float


def compare(
    arms: Sequence[Arms],
    comparison: Comparison,
    design: Design,
    *,
    replicates: int = REPLICATES,
    seed: int = SEED,
) -> Ratio:
    """The subject's units against its reference's; the interval only where
    support is met and the point estimate is above the floor."""
    keyed = [_Keyed.of(arm, comparison) for arm in arms]
    point = [k.ratios(design, None) for k in keyed]
    ratios = np.concatenate([r for r, _ in point]) if point else np.zeros(0)
    result = Ratio(sum(n for _, n in point), len(ratios), None)
    result.screened = _ratio_screen(result, ratios, comparison)
    if result.screened is None:
        rng = np.random.default_rng(seed)
        draws = [_replicate_ratio(keyed, design, rng) for _ in range(replicates)]
        result.ratio = interval(float(np.median(ratios)), draws)
    return result


def _ratio_screen(
    result: Ratio, ratios: np.ndarray, comparison: Comparison
) -> str | None:
    if not len(ratios):
        return INSUFFICIENT_SUPPORT
    result.ratio = Interval(float(np.median(ratios)))
    if (result.support or 0.0) < comparison.min_support:
        return INSUFFICIENT_SUPPORT
    if result.ratio.estimate <= comparison.floor:
        return NOT_ABOVE_FLOOR
    return None


def _replicate_ratio(
    keyed: Sequence[_Keyed], design: Design, rng: np.random.Generator
) -> float:
    found = [k.ratios(design, rng)[0] for k in keyed]
    ratios = np.concatenate(found) if found else np.zeros(0)
    return float(np.median(ratios)) if len(ratios) else float("nan")


@dataclass(frozen=True)
class _Keyed:
    """One epoch's arms, with each match's key ids for both."""

    arm: Arms
    comparison: Comparison
    keys: tuple[list[np.ndarray], ...]

    @classmethod
    def of(cls, arm: Arms, comparison: Comparison) -> _Keyed:
        sides = [arm.subject, arm.reference]
        keys = tuple(key_ids(sides, match.exact) for _, match in comparison.matches)
        return cls(arm, comparison, keys)

    def ratios(
        self, design: Design, rng: np.random.Generator | None
    ) -> tuple[np.ndarray, int]:
        """The matched subject units' cadence ratios, and how many subject
        units there were; resampled when ``rng`` is given."""
        sides = (self.arm.subject, self.arm.reference)
        picks = [_pick(columns, design, rng) for columns in sides]
        found, targets = [], 0
        for (treated, match), keys in zip(self.comparison.matches, self.keys):
            pair = [
                _side(columns, key, match, rows[columns.treated[rows] == treated])
                for columns, key, rows in zip(sides, keys, picks)
            ]
            medians, _ = nearest_medians(
                pair[0],
                pair[1],
                match.bands,
                Window(None, causal=False),
                nearest=design.nearest,
                minimum=design.min_controls,
            )
            matched = np.isfinite(medians)
            found.append(pair[0].value[matched] / medians[matched])
            targets += len(medians)
        return np.concatenate(found) if found else np.zeros(0), targets


def _pick(
    columns: Columns, design: Design, rng: np.random.Generator | None
) -> np.ndarray:
    if rng is None:
        return np.arange(len(columns))
    return circular_resample(columns.time, design.block_ns, rng)[0]


def _side(columns: Columns, keys: np.ndarray, match: Match, rows: np.ndarray) -> Side:
    """Rows ordered by decode context, not time: the reference's last steps
    abut the subject, and may be the incident's own beginning."""
    order = np.round(columns.banded["context"][rows] * 1000).astype(np.int64)
    return Side.of(columns, keys, match, rows, order)


__all__ = [
    "BANDED",
    "INSUFFICIENT_SUPPORT",
    "NOT_ABOVE_FLOOR",
    "NO_POSITIVE_EFFECT",
    "REPLICATES",
    "CONTROL_KEY",
    "Arms",
    "Comparison",
    "DOSE_BINS",
    "EXACT",
    "Band",
    "Columns",
    "Design",
    "Effects",
    "Epoch",
    "Interference",
    "Interval",
    "Match",
    "Ratio",
    "Side",
    "Span",
    "Window",
    "bin_name",
    "circular_resample",
    "compare",
    "dose_bin",
    "interference",
    "interval",
    "key_ids",
    "matched_effects",
    "nearest_medians",
    "sensitivity",
    "victim_matrix",
]
