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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .diagnosis_thresholds import (
    MIXED_CONTEXT_TOLERANCE,
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

    @classmethod
    def from_thresholds(cls, overrides: Mapping[str, float] | None) -> Design:
        def value(key: str) -> float:
            return resolve_threshold(key, overrides)[0]

        return cls(
            window_ns=int(value(MIXED_MATCH_WINDOW_NS)),
            tolerance=value(MIXED_CONTEXT_TOLERANCE),
            min_controls=int(value(MIXED_MIN_CONTROLS)),
            nearest=int(value(MIXED_NEAREST_CONTROLS)),
        )

    @property
    def controls(self) -> Match:
        return Match(CONTROL_KEY, (Band("context", self.tolerance),))


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


__all__ = [
    "BANDED",
    "CONTROL_KEY",
    "DOSE_BINS",
    "EXACT",
    "Band",
    "Columns",
    "Design",
    "Effects",
    "Match",
    "Side",
    "Span",
    "Window",
    "bin_name",
    "dose_bin",
    "key_ids",
    "matched_effects",
    "nearest_medians",
]
