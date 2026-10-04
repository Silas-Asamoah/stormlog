"""Aggregate one window of vLLM ``/metrics`` scrapes for one engine.

The caller decides which scrapes form a window; this module aggregates
exactly the scrapes it is given, in the order given. Every figure belongs to
one ``engine`` label and is never summed across engines.

Counters and histograms are differenced between consecutive scrapes, so a
counter that went backwards between two interior scrapes is caught even when
the window's first and last values look consistent. Nothing is differenced
across a reset, a recreated series, a missing series, a non-finite sample or a
change of exporter: the result holds ``None`` and says why, never a zero.

A scrape samples the server at some instant between the moment it was stamped
(``observed_at_ns``) and the moment its response came back. Window durations
are measured between those intervals' midpoints and carry the bounds the
intervals allow. Records written before scrapes carried ``completed_at_ns``
are bounded by ``observed_at_ns + duration_ms``, which leaves out the fetch
thread's start delay, so such windows say ``placement: "approximate"``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from .vllm_metrics import (
    ENGINE_LABELS,
    CompactScrape,
    HistogramValue,
    bucket_boundary,
    created_family_for,
    encode_number,
)
from .vllm_telemetry import SCRAPE_OK, VllmScrapeRecord

STATE_RESOLVED = "resolved"
STATE_UNRESOLVED = "unresolved"
REASON_SCRAPE_MISSING = "scrape_missing"
REASON_SCRAPE_FAILED = "scrape_failed"
REASON_ENGINE_RESTART = "engine_restart"
REASON_ENGINES_CHANGED = "engine_set_changed"
REASON_IDENTITY_UNKNOWN = "exporter_identity_unknown"
REASON_COUNTER_RESET = "counter_reset"
REASON_COUNTER_RECREATED = "counter_recreated"
REASON_SERIES_MISSING = "series_missing"
REASON_BOUNDARIES_CHANGED = "bucket_boundaries_changed"
REASON_NON_FINITE = "non_finite_sample"
REASON_TOO_FEW_SCRAPES = "too_few_scrapes"
REASON_OUT_OF_ORDER = "scrapes_out_of_order"
REASON_ENGINE_REQUIRED = "engine_required"
REASON_AMBIGUOUS_SERIES = "ambiguous_series"
REASON_NO_OBSERVATIONS = "no_observations"
REASON_OVERFLOW_BUCKET = "quantile_in_overflow_bucket"
REASON_SERIES_LABELS_CHANGED = "series_labels_changed"
REASON_DUPLICATE_TIME = "duplicate_scrape_time"
REASON_HISTOGRAM_INCONSISTENT = "histogram_inconsistent"

PLACEMENT_COMPLETED = "completed_at"
PLACEMENT_APPROXIMATE = "approximate"

Labels = Mapping[str, str]
SeriesKey = tuple[str, tuple[tuple[str, str], ...]]
Identity = tuple[int, tuple[str, ...]]
Value = float | HistogramValue


# ------------------------------------------------------------------ results
@dataclass(frozen=True)
class WindowCheck:
    """Whether a window's scrapes can be aggregated, and how long it lasted."""

    sufficient: bool
    reasons: tuple[str, ...]
    scrapes: int
    engine: str | None
    seconds: float | None
    seconds_bounds: tuple[float, float] | None
    placement: str
    # Failed scrapes inside the window; they leave fewer samples, and a
    # counter is differenced across them, so only a failed first or last
    # scrape (which shortens the window itself) makes it insufficient.
    failed: int = 0


@dataclass(frozen=True)
class GaugeWindow:
    """A gauge's finite samples over the window and their statistics."""

    n: int
    non_finite: int
    min: float | None
    mean: float | None
    max: float | None
    last: float | None
    values: tuple[float, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CounterWindow:
    """A counter's increase over the window, from consecutive deltas."""

    delta: float | None
    rate_per_s: float | None
    rate_bounds: tuple[float, float | None] | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HistogramShare:
    """The share of a histogram's window observations above a value.

    ``lo`` and ``hi`` bound the share between bucket boundaries; they are
    equal when the value is itself a boundary.
    """

    count_delta: float | None
    lo: float | None
    hi: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class QuantileBounds:
    """The bucket boundaries that contain a quantile of the window.

    ``lo`` is None when the quantile is in the first bucket, whose lower
    bound the exposition does not give; ``hi`` is None when it is in the
    ``+Inf`` bucket, with the reason ``quantile_in_overflow_bucket``.
    """

    count_delta: float | None
    lo: float | None
    hi: float | None
    reasons: tuple[str, ...]


# ------------------------------------------------------------- the window
def check_window(
    scrapes: Sequence[VllmScrapeRecord],
    *,
    engine: str | None = None,
    min_scrapes: int = 2,
) -> WindowCheck:
    """Check that a window is one exporter's, long enough, and in order."""
    ok = _ok_scrapes(scrapes)
    reasons = _window_reasons(scrapes, ok, engine, min_scrapes)
    seconds, bounds = _window_seconds(ok)
    return WindowCheck(
        sufficient=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        scrapes=len(ok),
        engine=engine,
        seconds=seconds,
        seconds_bounds=bounds,
        placement=_placement(ok),
        failed=len(scrapes) - len(ok),
    )


def _window_reasons(
    scrapes: Sequence[VllmScrapeRecord],
    ok: Sequence[VllmScrapeRecord],
    engine: str | None,
    min_scrapes: int,
) -> list[str]:
    reasons: list[str] = []
    if (
        scrapes
        and SCRAPE_OK != scrapes[0].status
        or scrapes[-1:]
        and (SCRAPE_OK != scrapes[-1].status)
    ):
        # A failed boundary scrape shortens the window the caller chose.
        reasons.append(REASON_SCRAPE_FAILED)
    if len(ok) < min_scrapes:
        reasons.append(REASON_TOO_FEW_SCRAPES)
    reasons.extend(order_reasons(scrapes))
    reasons.extend(window_identity_reasons(ok))
    reasons.extend(_engine_reasons(ok, engine))
    return reasons


def _engine_reasons(ok: Sequence[VllmScrapeRecord], engine: str | None) -> list[str]:
    """A window's figures are one engine's. A named engine must be in every
    scrape; with none named, no scrape may hold several, or two families of
    one signal could each be read from a different engine."""
    if engine is not None:
        missing = any(engine not in _engines(scrape) for scrape in ok)
        return [REASON_SERIES_MISSING] if missing else []
    several = any(len(_engines(scrape)) > 1 for scrape in ok)
    return [REASON_ENGINE_REQUIRED] if several else []


def order_reasons(scrapes: Sequence[VllmScrapeRecord]) -> list[str]:
    """Scrapes must be in strictly increasing stamp order: an earlier stamp
    after a later one is out of order, and two at one instant give a window
    of no length. Durations run between sample midpoints, so a later stamp
    whose midpoint is no later (a quick scrape inside a slow one's interval)
    is out of order too: which sampled first is unknown."""
    pairs = list(zip(scrapes, scrapes[1:]))
    reasons = []
    if any(_inverted(a, b) for a, b in pairs):
        reasons.append(REASON_OUT_OF_ORDER)
    if any(b.observed_at_ns == a.observed_at_ns for a, b in pairs):
        reasons.append(REASON_DUPLICATE_TIME)
    return reasons


def _inverted(a: VllmScrapeRecord, b: VllmScrapeRecord) -> bool:
    """``b`` is stamped before ``a``, or after it with no later midpoint."""
    if b.observed_at_ns == a.observed_at_ns:
        return False  # a duplicate, not an inversion
    stamped_before = b.observed_at_ns < a.observed_at_ns
    return stamped_before or sum(sample_interval(b)) <= sum(sample_interval(a))


def window_identity_reasons(ok: Sequence[VllmScrapeRecord]) -> list[str]:
    """Why the successful scrapes are not one exporter's, checked on every
    scrape against the first. An exporter with no process start cannot be
    told apart from a restarted one, so it is unknown rather than assumed."""
    if not ok:
        return []
    identity = exporter_identity(ok[0])
    if identity is None:
        return [REASON_IDENTITY_UNKNOWN]
    found: set[str] = set()
    for other in ok[1:]:
        found |= identity_differences(identity, exporter_identity(other))
    order = (REASON_ENGINE_RESTART, REASON_ENGINES_CHANGED, REASON_IDENTITY_UNKNOWN)
    return [reason for reason in order if reason in found]


def exporter_identity(scrape: VllmScrapeRecord) -> Identity | None:
    """Which exporter answered: its process start and its engine set."""
    found = scrape.discovery
    if found is None or found.process_start_ns is None:
        return None
    return found.process_start_ns, tuple(sorted(found.engines))


def identity_differences(identity: Identity, other: Identity | None) -> set[str]:
    """How another scrape's exporter differs from the first scrape's."""
    if other is None:
        return {REASON_IDENTITY_UNKNOWN}
    reasons = set()
    if other[0] != identity[0]:
        reasons.add(REASON_ENGINE_RESTART)
    if other[1] != identity[1]:
        reasons.add(REASON_ENGINES_CHANGED)
    return reasons


def _engines(scrape: VllmScrapeRecord) -> tuple[str, ...]:
    return scrape.discovery.engines if scrape.discovery is not None else ()


def sample_interval(scrape: VllmScrapeRecord) -> tuple[int, int]:
    """When the server was sampled: between the stamp and the response."""
    completed = getattr(scrape, "completed_at_ns", None)
    if isinstance(completed, int) and completed >= scrape.observed_at_ns:
        return scrape.observed_at_ns, completed
    duration_ns = round((scrape.duration_ms or 0.0) * 1e6)
    return scrape.observed_at_ns, scrape.observed_at_ns + duration_ns


def _placement(ok: Sequence[VllmScrapeRecord]) -> str:
    exact = all(
        isinstance(getattr(scrape, "completed_at_ns", None), int) for scrape in ok
    )
    return PLACEMENT_COMPLETED if exact and ok else PLACEMENT_APPROXIMATE


def _window_seconds(
    ok: Sequence[VllmScrapeRecord],
) -> tuple[float | None, tuple[float, float] | None]:
    """Midpoint-to-midpoint seconds between the first and last scrape, with
    the shortest and longest durations the two sample intervals allow."""
    if len(ok) < 2:
        return None, None
    first_lo, first_hi = sample_interval(ok[0])
    last_lo, last_hi = sample_interval(ok[-1])
    seconds = ((last_lo + last_hi) - (first_lo + first_hi)) / 2e9
    shortest = max(0.0, (last_lo - first_hi) / 1e9)
    longest = (last_hi - first_lo) / 1e9
    return seconds, (shortest, longest)


# ---------------------------------------------------------------- gauges
def gauge_window(
    scrapes: Sequence[VllmScrapeRecord],
    family: str,
    *,
    labels: Labels | None = None,
    engine: str | None = None,
) -> GaugeWindow:
    """A gauge's samples over the window's successful scrapes, one
    exporter's: samples from both sides of a restart are not one gauge.

    "Every sample at least ``t``" is ``gauge.min >= t`` with ``n`` checked
    against the caller's floor; :func:`share_at_least` gives the fraction.
    """
    samples: list[float] = []
    ok = list(_ok_scrapes(scrapes))
    reasons: list[str] = window_identity_reasons(ok)
    seen: list[Mapping[str, str]] = []
    for scrape in ok:
        value, found, reason = series_match(scrape.scrape, family, labels, engine)
        if reason is not None:
            reasons.append(reason)
        elif isinstance(value, float):
            samples.append(value)
            seen.append(found or {})
    if any(other != seen[0] for other in seen[1:]):
        reasons.append(REASON_SERIES_LABELS_CHANGED)
    stats = sample_stats(samples)
    if stats["non_finite"]:
        reasons.append(REASON_NON_FINITE)
    finite = tuple(value for value in samples if math.isfinite(value))
    return GaugeWindow(
        n=len(finite),
        non_finite=stats["non_finite"],
        min=stats["min"],
        mean=stats["mean"],
        max=stats["max"],
        last=stats["last"],
        values=finite,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def share_at_least(gauge: GaugeWindow, threshold: float) -> float | None:
    """The fraction of the gauge's finite samples at or above ``threshold``."""
    if not gauge.values:
        return None
    return sum(1 for value in gauge.values if value >= threshold) / len(gauge.values)


def gauge_median(gauge: GaugeWindow) -> float | None:
    return median(gauge.values) if gauge.values else None


def sample_stats(values: Sequence[float]) -> dict[str, Any]:
    """Over the finite samples; a NaN or an infinity is counted, not averaged."""
    finite = [value for value in values if math.isfinite(value)]
    return {
        "min": min(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
        "max": max(finite) if finite else None,
        "last": finite[-1] if finite else None,
        "samples": len(values),
        "non_finite": len(values) - len(finite),
    }


# -------------------------------------------------------------- counters
def counter_window(
    scrapes: Sequence[VllmScrapeRecord],
    family: str,
    *,
    labels: Labels | None = None,
    engine: str | None = None,
) -> CounterWindow:
    """A counter's increase: the sum of its consecutive-scrape deltas."""
    ok = _ok_scrapes(scrapes)
    reasons = list(_pair_preconditions(ok))
    if reasons:
        # Pairs out of order, at one instant or from two exporters are not
        # differenced at all, so their deltas add no misleading reasons.
        return CounterWindow(None, None, None, tuple(dict.fromkeys(reasons)))
    total = 0.0
    for before, after in zip(ok, ok[1:]):
        delta = _counter_step(before, after, family, labels, engine)
        if delta["state"] != STATE_RESOLVED:
            reasons.append(delta["state"])
        else:
            total += delta["delta"]
    if reasons:
        return CounterWindow(None, None, None, tuple(dict.fromkeys(reasons)))
    seconds, bounds = _window_seconds(ok)
    rate, rate_bounds = _rates(total, seconds, bounds)
    return CounterWindow(total, rate, rate_bounds, ())


def _pair_preconditions(ok: Sequence[VllmScrapeRecord]) -> list[str]:
    reasons = [*order_reasons(ok), *window_identity_reasons(ok)]
    if len(ok) < 2:
        reasons.append(REASON_TOO_FEW_SCRAPES)
    return reasons


def _counter_step(
    before: VllmScrapeRecord,
    after: VllmScrapeRecord,
    family: str,
    labels: Labels | None,
    engine: str | None,
) -> dict[str, Any]:
    a, a_labels, a_reason = series_match(before.scrape, family, labels, engine)
    b, b_labels, b_reason = series_match(after.scrape, family, labels, engine)
    if a_reason or b_reason:
        return {"state": a_reason or b_reason, "delta": None}
    if a_labels != b_labels:
        return {"state": REASON_SERIES_LABELS_CHANGED, "delta": None}
    recreated = created_changed(before.scrape, after.scrape, family, labels, engine)
    return counter_pair_delta(a, b, None, recreated)


def _rates(
    total: float, seconds: float | None, bounds: tuple[float, float] | None
) -> tuple[float | None, tuple[float, float | None] | None]:
    if seconds is None or bounds is None or seconds <= 0:
        return None, None
    shortest, longest = bounds
    fastest = total / shortest if shortest > 0 else None
    return total / seconds, (total / longest, fastest)


def counter_pair_delta(
    a: Value | None,
    b: Value | None,
    epoch_reason: str | None,
    recreated: bool,
) -> dict[str, Any]:
    """One counter's change between two scrapes, or why there is none."""
    if not isinstance(a, float) or not isinstance(b, float):
        return {"state": REASON_SERIES_MISSING, "delta": None}
    if not (math.isfinite(a) and math.isfinite(b)):
        # Kept as the strict-JSON strings the record uses, never subtracted.
        start, end = encode_number(a), encode_number(b)
        return {"state": REASON_NON_FINITE, "delta": None, "start": start, "end": end}
    if epoch_reason is not None:
        return {"state": epoch_reason, "delta": None, "start": a, "end": b}
    if recreated:
        return {"state": REASON_COUNTER_RECREATED, "delta": None, "start": a, "end": b}
    if b < a:
        return {"state": REASON_COUNTER_RESET, "delta": None, "start": a, "end": b}
    return {"state": STATE_RESOLVED, "delta": b - a, "start": a, "end": b}


# ------------------------------------------------------------ histograms
def histogram_share_above(
    scrapes: Sequence[VllmScrapeRecord],
    family: str,
    value: float,
    *,
    labels: Labels | None = None,
    engine: str | None = None,
) -> HistogramShare:
    """The share of the window's observations above ``value``, bounded by the
    bucket boundaries on either side of it."""
    count, buckets, reasons = _histogram_window(scrapes, family, labels, engine)
    if reasons or count is None or buckets is None:
        return HistogramShare(count, None, None, reasons)
    if count <= 0:
        return HistogramShare(count, None, None, (REASON_NO_OBSERVATIONS,))
    at_or_below_ceiling = _cumulative_at(buckets, value, ceiling=True, total=count)
    at_or_below_floor = _cumulative_at(buckets, value, ceiling=False, total=count)
    lo = (count - at_or_below_ceiling) / count
    hi = (count - at_or_below_floor) / count
    return HistogramShare(count, lo, hi, ())


def histogram_quantile_bounds(
    scrapes: Sequence[VllmScrapeRecord],
    family: str,
    q: float,
    *,
    labels: Labels | None = None,
    engine: str | None = None,
) -> QuantileBounds:
    """The bucket containing the window's ``q`` quantile: the first bucket
    whose cumulative count reaches ``q`` of the observations."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    count, buckets, reasons = _histogram_window(scrapes, family, labels, engine)
    if reasons or count is None or buckets is None:
        return QuantileBounds(count, None, None, reasons)
    if count <= 0:
        return QuantileBounds(count, None, None, (REASON_NO_OBSERVATIONS,))
    rank = q * count
    lower: float | None = None
    for boundary, cumulative in buckets:
        if cumulative >= rank:
            if math.isinf(boundary):
                return QuantileBounds(count, lower, None, (REASON_OVERFLOW_BUCKET,))
            return QuantileBounds(count, lower, boundary, ())
        lower = boundary
    return QuantileBounds(count, lower, None, (REASON_OVERFLOW_BUCKET,))


def _cumulative_at(
    buckets: Sequence[tuple[float, float]],
    value: float,
    *,
    ceiling: bool,
    total: float,
) -> float:
    """Cumulative count at the smallest boundary at or above ``value``
    (``ceiling``), or at the largest boundary at or below it. Every
    observation is at or below ``+Inf``, so a missing ceiling is the total."""
    if ceiling:
        return next((c for boundary, c in buckets if boundary >= value), total)
    below = [c for boundary, c in buckets if boundary <= value]
    return below[-1] if below else 0.0


def _histogram_window(
    scrapes: Sequence[VllmScrapeRecord],
    family: str,
    labels: Labels | None,
    engine: str | None,
) -> tuple[float | None, list[tuple[float, float]] | None, tuple[str, ...]]:
    """The window's observation count and cumulative buckets, summed from
    consecutive deltas, or the reasons they cannot be."""
    ok = _ok_scrapes(scrapes)
    reasons = _pair_preconditions(ok)
    if reasons:
        return None, None, tuple(dict.fromkeys(reasons))
    deltas: list[dict[str, Any]] = []
    for before, after in zip(ok, ok[1:]):
        delta = _histogram_step(before, after, family, labels, engine)
        if delta["state"] != STATE_RESOLVED:
            reasons.append(delta["state"])
        elif not _consistent_step(delta):
            reasons.append(REASON_HISTOGRAM_INCONSISTENT)
        deltas.append(delta)
    if reasons:
        return None, None, tuple(dict.fromkeys(reasons))
    return _summed_window(deltas)


def _summed_window(
    deltas: Sequence[dict[str, Any]],
) -> tuple[float | None, list[tuple[float, float]] | None, tuple[str, ...]]:
    shapes = {tuple(le for le, _ in delta["buckets"]) for delta in deltas}
    if len(shapes) != 1:
        return None, None, (REASON_BOUNDARIES_CHANGED,)
    les = next(iter(shapes))
    count = sum(float(delta["count"]) for delta in deltas)
    buckets = sorted(
        (bucket_boundary(le), sum(float(delta["buckets"][i][1]) for delta in deltas))
        for i, le in enumerate(les)
    )
    return count, buckets, ()


def _consistent_step(delta: Mapping[str, Any]) -> bool:
    """One step's deltas are themselves a histogram: cumulative counts that
    never fall as the boundary rises, none above the step's count, and the
    ``+Inf`` bucket equal to it. Two scrapes that are each valid can differ
    by a step that is not one, and its shares would fall outside 0 to 1."""
    count = float(delta["count"])
    buckets = sorted((bucket_boundary(le), float(c)) for le, c in delta["buckets"])
    counts = [c for _, c in buckets]
    if any(after < before for before, after in zip(counts, counts[1:])):
        return False
    if any(c > count for c in counts):
        return False
    # Every observation is at or below +Inf: a _count that disagrees with the
    # +Inf bucket makes every share computed from either one unreliable.
    return not buckets or not math.isinf(buckets[-1][0]) or buckets[-1][1] == count


def _histogram_step(
    before: VllmScrapeRecord,
    after: VllmScrapeRecord,
    family: str,
    labels: Labels | None,
    engine: str | None,
) -> dict[str, Any]:
    a, a_labels, a_reason = series_match(before.scrape, family, labels, engine)
    b, b_labels, b_reason = series_match(after.scrape, family, labels, engine)
    if a_reason or b_reason:
        return {"state": a_reason or b_reason}
    if a_labels != b_labels:
        return {"state": REASON_SERIES_LABELS_CHANGED}
    recreated = created_changed(
        before.scrape, after.scrape, family, labels, engine, kind="histogram"
    )
    return histogram_pair_delta(
        a if isinstance(a, HistogramValue) else None,
        b if isinstance(b, HistogramValue) else None,
        None,
        recreated,
    )


def histogram_pair_delta(
    a: HistogramValue | None,
    b: HistogramValue | None,
    epoch_reason: str | None,
    recreated: bool,
) -> dict[str, Any]:
    """One histogram's change between two scrapes, or why there is none."""
    if a is None or b is None:
        return {"state": REASON_SERIES_MISSING}
    parts = _complete(a, b)
    if parts is None:
        # A _sum or _count the exposition lacked is not a zero to subtract.
        return {"state": REASON_SERIES_MISSING, "missing": _missing_components(a, b)}
    if not _finite_histograms(a, b, parts):
        return {"state": REASON_NON_FINITE}
    reason = _histogram_reason(a, b, epoch_reason, recreated)
    if reason is not None:
        return {"state": reason}
    a_count, a_sum, b_count, b_sum = parts
    count = b_count - a_count
    total = b_sum - a_sum
    return {
        "state": STATE_RESOLVED,
        "count": count,
        "sum": total,
        "mean": total / count if count else None,
        "buckets": [
            [le, after - before]
            for (le, before), (_le, after) in zip(a.buckets, b.buckets)
        ],
    }


def _histogram_reason(
    a: HistogramValue, b: HistogramValue, epoch_reason: str | None, recreated: bool
) -> str | None:
    """Why a histogram window cannot be differenced, in the order checked for
    counters: the epoch, the family's ``*_created`` stamp, then its shape and
    monotonicity. Every cumulative part must be non-decreasing: the count,
    the sum and each bucket."""
    if epoch_reason is not None:
        return epoch_reason
    if recreated:
        return REASON_COUNTER_RECREATED
    if [le for le, _ in a.buckets] != [le for le, _ in b.buckets]:
        return REASON_BOUNDARIES_CHANGED
    backwards = _less(b.count, a.count) or _less(b.sum, a.sum)
    if backwards or any(
        after < before for (_, before), (_, after) in zip(a.buckets, b.buckets)
    ):
        return REASON_COUNTER_RESET
    return None


def _less(after: float | None, before: float | None) -> bool:
    return after is not None and before is not None and after < before


def _complete(
    a: HistogramValue, b: HistogramValue
) -> tuple[float, float, float, float] | None:
    """Both histograms' count and sum, or None when any is missing."""
    if a.count is None or a.sum is None or b.count is None or b.sum is None:
        return None
    return a.count, a.sum, b.count, b.sum


def _finite_histograms(
    a: HistogramValue, b: HistogramValue, parts: tuple[float, ...]
) -> bool:
    """Every count, sum and bucket of both is a finite number."""
    counts = [count for _, count in (*a.buckets, *b.buckets)]
    return all(math.isfinite(value) for value in (*parts, *counts))


def _missing_components(a: HistogramValue, b: HistogramValue) -> list[str]:
    parts = (
        ("start_sum", a.sum),
        ("start_count", a.count),
        ("end_sum", b.sum),
        ("end_count", b.count),
    )
    return [name for name, value in parts if value is None]


# ---------------------------------------------------------------- series
def series_value(
    scrape: CompactScrape | None,
    family: str,
    labels: Labels | None,
    engine: str | None,
) -> tuple[Value | None, str | None]:
    """The one series of ``family`` matching ``labels`` and ``engine``.

    Without ``engine`` the scrape must hold a single engine. Two matching
    series are ambiguous rather than summed: label sets such as
    ``reason`` or ``finished_reason`` are kept apart unless the caller names
    one.
    """
    value, _labels, reason = series_match(scrape, family, labels, engine)
    return value, reason


def series_match(
    scrape: CompactScrape | None,
    family: str,
    labels: Labels | None,
    engine: str | None,
) -> tuple[Value | None, Mapping[str, str] | None, str | None]:
    """As :func:`series_value`, with the matched series' full label set, so
    two scrapes can be checked to hold the same series."""
    if scrape is None:
        return None, None, REASON_SERIES_MISSING
    matches = [
        (scrape.labels(set_id), value)
        for set_id, value in scrape.series(family).items()
        if _labels_match(scrape.labels(set_id), labels, engine)
    ]
    if engine is None and len({found.get("engine") for found, _ in matches}) > 1:
        return None, None, REASON_ENGINE_REQUIRED
    if not matches:
        return None, None, REASON_SERIES_MISSING
    if len(matches) > 1:
        return None, None, REASON_AMBIGUOUS_SERIES
    return matches[0][1], matches[0][0], None


def _labels_match(
    found: Mapping[str, str], wanted: Labels | None, engine: str | None
) -> bool:
    if engine is not None and found.get("engine") != engine:
        return False
    return all(found.get(key) == value for key, value in (wanted or {}).items())


def created_changed(
    before: CompactScrape | None,
    after: CompactScrape | None,
    family: str,
    labels: Labels | None,
    engine: str | None,
    *,
    kind: str = "counter",
) -> bool:
    """The family's ``*_created`` stamp differs between two scrapes."""
    created = created_family_for(family, kind)
    if created is None:
        return False
    a, _a_reason = series_value(before, created, labels, engine)
    b, _b_reason = series_value(after, created, labels, engine)
    return a != b


def engine_index(scrape: CompactScrape, engine: str) -> dict[SeriesKey, Value]:
    """Series of one engine keyed by (family, extra labels beyond the engine's)."""
    indexed: dict[SeriesKey, Value] = {}
    for name, by_set in scrape.values.items():
        for set_id, value in by_set.items():
            labels = scrape.labels(set_id)
            if labels.get("engine") != engine:
                continue
            extra = tuple(
                sorted((k, v) for k, v in labels.items() if k not in ENGINE_LABELS)
            )
            indexed[(name, extra)] = value
    return indexed


def series_by_extra(
    name: str, indexed: Mapping[SeriesKey, Value]
) -> dict[tuple[tuple[str, str], ...], Value]:
    return {
        extra: value for (family, extra), value in indexed.items() if family == name
    }


def _ok_scrapes(scrapes: Sequence[VllmScrapeRecord]) -> list[VllmScrapeRecord]:
    return [scrape for scrape in scrapes if scrape.status == SCRAPE_OK]


__all__ = [
    "CounterWindow",
    "GaugeWindow",
    "HistogramShare",
    "QuantileBounds",
    "WindowCheck",
    "check_window",
    "counter_pair_delta",
    "counter_window",
    "created_changed",
    "engine_index",
    "exporter_identity",
    "gauge_median",
    "gauge_window",
    "histogram_pair_delta",
    "histogram_quantile_bounds",
    "histogram_share_above",
    "identity_differences",
    "sample_interval",
    "sample_stats",
    "order_reasons",
    "series_by_extra",
    "series_match",
    "series_value",
    "share_at_least",
    "window_identity_reasons",
]
