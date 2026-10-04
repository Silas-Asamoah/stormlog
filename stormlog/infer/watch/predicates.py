"""What a trigger asks of a window of scrapes, and how the window is chosen.

:func:`select_window` cuts a window ``[t - W, t]`` from the scrape history on
the watcher's monotonic clock: the end scrape is the latest that finished at
or before ``t``, and must have succeeded and finished within one tick plus
the scrape timeout of it, since a tick can land while a slow scrape is
still in flight; the start scrape is the latest successful one that finished within one tick
of ``t - W``. When either is missing the evaluation is a data gap. A scrape
that failed between them only leaves the window fewer samples; the
evaluation counts it in its ``failed_scrapes`` detail.

The predicates aggregate the chosen scrapes with
:mod:`stormlog.infer.scrape_window`, which differences consecutive scrapes, so
a counter reset or an exporter restart anywhere inside the window makes the
evaluation a data gap with the reason, never a value. Every figure is
engine-wide: it describes all of the server's traffic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..diagnosis_signals import SignalConfig, evaluate_signal
from ..scrape_window import (
    Labels,
    check_window,
    counter_window,
    exporter_identity,
    gauge_window,
    histogram_share_above,
    share_at_least,
)
from ..vllm_telemetry import SCRAPE_OK, VllmScrapeRecord
from .history import Stamped
from .triggers import CLEAR, DATA_GAP, VIOLATING

REASON_END_STALE = "end_scrape_stale"
REASON_END_FAILED = "end_scrape_failed"
REASON_START_MISSING = "start_scrape_missing"
REASON_TOO_FEW_SAMPLES = "too_few_samples"
REASON_NO_RECENT_SCRAPE = "no_recent_scrape"
# Counters whose standstill, while requests run or wait, means no progress.
PROGRESS_COUNTERS = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total")
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"

Entry = tuple[Stamped, VllmScrapeRecord]
_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class Selection:
    """The scrapes of one window, or why there is no window.

    The scrapes are stamped on the watcher's monotonic clock (see
    :func:`on_monotonic_clock`), so every duration and rate measured on them
    is too. ``start_ns`` and ``end_ns`` are when the window's first and last
    fetch returned.
    """

    scrapes: tuple[VllmScrapeRecord, ...] = ()
    start_ns: int | None = None
    end_ns: int | None = None
    reason: str | None = None
    # When the window's first fetch began: the earliest instant it reads.
    sample_start_ns: int | None = None


@dataclass(frozen=True)
class Evaluation:
    """One evaluation of a predicate: its classification and the evidence."""

    classification: str
    observed: float | None = None
    observed_bounds: tuple[float, float | None] | None = None
    threshold: float | None = None
    samples: float | None = None
    reasons: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)


def select_window(
    history: Sequence[Entry],
    *,
    at_ns: int,
    window_ns: int,
    tick_ns: int,
    scrape_timeout_ns: int = 0,
) -> Selection:
    """The scrapes of the window ending at ``at_ns``, or a data-gap reason."""
    done = [entry for entry in history if entry[0].done_mono_ns <= at_ns]
    if not done or done[-1][0].done_mono_ns < at_ns - tick_ns - scrape_timeout_ns:
        return Selection(reason=REASON_END_STALE)
    end = done[-1]
    if end[1].status != SCRAPE_OK:
        return Selection(reason=REASON_END_FAILED)
    start = _start_entry(done, at_ns - window_ns, tick_ns)
    if start is None:
        return Selection(reason=REASON_START_MISSING)
    first, last = start[0].done_mono_ns, end[0].done_mono_ns
    scrapes = tuple(
        on_monotonic_clock(stamp, record)
        for stamp, record in done
        if first <= stamp.done_mono_ns <= last
    )
    return Selection(scrapes, first, last, sample_start_ns=start[0].mono_ns)


def on_monotonic_clock(stamp: Stamped, record: VllmScrapeRecord) -> VllmScrapeRecord:
    """The record as sampled on the watcher's monotonic clock.

    Window aggregation times counters and histograms from a record's
    ``observed_at_ns`` and ``duration_ms``, which are the client's wall
    clock: an NTP step inside a window would bend every rate. Here they
    become the fetch's monotonic start and its duration to the response,
    which bound the server's sample instant just as well. A record that
    carries ``completed_at_ns`` (the response's wall-clock time, which
    window aggregation prefers to the duration) has it moved too: a
    monotonic start and a wall-clock end would make every window decades
    long.
    """
    # A record's stamp must be positive; the offset changes no duration.
    changes: dict[str, Any] = {
        "observed_at_ns": stamp.mono_ns + 1,
        "duration_ms": max(0, stamp.done_mono_ns - stamp.mono_ns) / 1e6,
    }
    if getattr(record, "completed_at_ns", None) is not None:
        changes["completed_at_ns"] = max(stamp.done_mono_ns, stamp.mono_ns) + 1
    return replace(record, **changes)


def _start_entry(done: Sequence[Entry], target_ns: int, tick_ns: int) -> Entry | None:
    """The latest successful scrape finished within one tick of ``target_ns``."""
    near = [
        entry
        for entry in done
        if entry[1].status == SCRAPE_OK
        and abs(entry[0].done_mono_ns - target_ns) <= tick_ns
    ]
    return near[-1] if near else None


@runtime_checkable
class WindowPredicate(Protocol):
    """A question about one window of scrapes."""

    def evaluate(self, scrapes: Sequence[VllmScrapeRecord]) -> Evaluation: ...


@dataclass(frozen=True)
class GaugeAtLeast:
    """A gauge at or above ``threshold``: in every sample (``share=1``), or in
    at least ``share`` of them."""

    family: str
    threshold: float
    share: float = 1.0
    min_samples: int = 2
    labels: Labels | None = None
    engine: str | None = None

    def evaluate(self, scrapes: Sequence[VllmScrapeRecord]) -> Evaluation:
        # A gauge is read sample by sample, never differenced, so the window
        # is checked here for what differencing would catch: scrapes out of
        # order or at one instant, and two exporters' samples in one window.
        window = check_window(scrapes, engine=self.engine, min_scrapes=1)
        gauge = gauge_window(
            scrapes, self.family, labels=self.labels, engine=self.engine
        )
        found = tuple(dict.fromkeys((*window.reasons, *gauge.reasons)))
        if found or gauge.n < self.min_samples:
            reasons = found or (REASON_TOO_FEW_SAMPLES,)
            return Evaluation(DATA_GAP, samples=gauge.n, reasons=reasons)
        fraction = share_at_least(gauge, self.threshold)
        assert fraction is not None  # n >= min_samples >= 1
        observed = gauge.min if self.share >= 1.0 else fraction
        violating = fraction >= self.share
        return Evaluation(
            VIOLATING if violating else CLEAR,
            observed=observed,
            threshold=self.threshold if self.share >= 1.0 else self.share,
            samples=gauge.n,
            detail={"min": gauge.min, "max": gauge.max, "share_at_least": fraction},
        )


@dataclass(frozen=True)
class CounterRateAtLeast:
    """A counter rising at ``rate_per_s`` or faster, judged on the lower bound
    of its rate so that clock uncertainty cannot make it fire."""

    family: str
    rate_per_s: float
    labels: Labels | None = None
    engine: str | None = None

    def evaluate(self, scrapes: Sequence[VllmScrapeRecord]) -> Evaluation:
        counter = counter_window(
            scrapes, self.family, labels=self.labels, engine=self.engine
        )
        if counter.reasons or counter.rate_bounds is None:
            return Evaluation(DATA_GAP, reasons=counter.reasons or ("no_rate",))
        lower, upper = counter.rate_bounds
        return Evaluation(
            VIOLATING if lower >= self.rate_per_s else CLEAR,
            observed=lower,
            observed_bounds=(lower, upper),
            threshold=self.rate_per_s,
            samples=counter.delta,
            detail={"delta": counter.delta, "rate_per_s": counter.rate_per_s},
        )


@dataclass(frozen=True)
class HistogramShareAbove:
    """More than ``share`` of the window's observations above ``value``.

    The threshold usually falls between bucket bounds, so the share is an
    interval ``[lo, hi]``; the predicate fires only on ``lo``, so bucket
    resolution can never make it fire.
    """

    family: str
    value: float
    share: float
    min_samples: int = 20
    labels: Labels | None = None
    engine: str | None = None

    def evaluate(self, scrapes: Sequence[VllmScrapeRecord]) -> Evaluation:
        found = histogram_share_above(
            scrapes, self.family, self.value, labels=self.labels, engine=self.engine
        )
        if found.reasons or found.lo is None or found.count_delta is None:
            return Evaluation(DATA_GAP, reasons=found.reasons or ("no_share",))
        if found.count_delta < self.min_samples:
            return Evaluation(
                DATA_GAP, samples=found.count_delta, reasons=(REASON_TOO_FEW_SAMPLES,)
            )
        return Evaluation(
            VIOLATING if found.lo > self.share else CLEAR,
            observed=found.lo,
            observed_bounds=(found.lo, found.hi),
            threshold=self.share,
            samples=found.count_delta,
            detail={"value": self.value},
        )


@dataclass(frozen=True)
class SignalExceeds:
    """One of #218's ``/metrics`` signals, judged by its shared threshold.

    A signal is engine-wide, so its incident only *suspects* the mechanism;
    the record says so.
    """

    kind: str
    config: SignalConfig = field(default_factory=SignalConfig)

    def evaluate(self, scrapes: Sequence[VllmScrapeRecord]) -> Evaluation:
        signal = evaluate_signal(self.kind, scrapes, self.config)
        detail = {
            "signal": self.kind,
            "status": "suspected",
            "thresholds_version": signal.thresholds_version,
            "threshold_overridden": signal.threshold_overridden,
            "signal_detail": dict(signal.detail),
        }
        if not signal.sufficient or signal.exceeds is None:
            reasons = (signal.reason,) if signal.reason else ("insufficient",)
            return Evaluation(DATA_GAP, reasons=reasons, detail=detail)
        return Evaluation(
            VIOLATING if signal.exceeds else CLEAR,
            observed=signal.value,
            threshold=signal.threshold,
            detail=detail,
        )


# ------------------------------------------------------------------ health


@dataclass(frozen=True)
class ScrapeFailures:
    """The last ``consecutive`` scrapes all failed. A failed scrape is
    evidence here, not a gap in it."""

    consecutive: int = 3

    @property
    def tail_scrapes(self) -> int:
        return self.consecutive

    @property
    def when_stale(self) -> str:
        return VIOLATING  # no scrape finishing is the failure itself

    def evaluate_history(self, history: Sequence[Entry]) -> Evaluation:
        tail = history[-self.consecutive :]
        if len(tail) < self.consecutive:
            return Evaluation(DATA_GAP, reasons=(REASON_TOO_FEW_SAMPLES,))
        failed = sum(record.status != SCRAPE_OK for _stamp, record in tail)
        return Evaluation(
            VIOLATING if failed == self.consecutive else CLEAR,
            observed=float(failed),
            threshold=float(self.consecutive),
            samples=float(len(tail)),
        )


@dataclass(frozen=True)
class ScrapeFailureShare:
    """At least ``share`` of the last ``scrapes`` scrapes failed.

    Isolated failures never make :class:`ScrapeFailures` fire, and every
    window trigger is judged on fewer samples across them, so a scraper
    failing now and then would otherwise degrade with nothing said. Here a
    failed scrape is evidence, as it is for ScrapeFailures.
    """

    share: float = 0.05
    scrapes: int = 60

    def __post_init__(self) -> None:
        if not 0.0 < self.share <= 1.0:
            raise ValueError("failed-scrape share must be in (0, 1]")
        if self.scrapes < 1:
            raise ValueError("failed-scrape share needs scrapes >= 1")

    @property
    def tail_scrapes(self) -> int:
        return self.scrapes

    @property
    def when_stale(self) -> str:
        return VIOLATING

    def evaluate_history(self, history: Sequence[Entry]) -> Evaluation:
        tail = history[-self.scrapes :]
        if len(tail) < self.scrapes:
            return Evaluation(DATA_GAP, reasons=(REASON_TOO_FEW_SAMPLES,))
        failed = sum(record.status != SCRAPE_OK for _stamp, record in tail)
        fraction = failed / len(tail)
        return Evaluation(
            VIOLATING if fraction >= self.share else CLEAR,
            observed=fraction,
            threshold=self.share,
            samples=float(len(tail)),
            detail={"failed": failed},
        )


@dataclass(frozen=True)
class FrozenExporter:
    """The server answers, but nothing progresses while requests run or wait.

    Over the last ``ticks + 1`` scrapes, all successful: running plus waiting
    stayed above zero, and no progress counter moved. This is how a stalled
    engine looks from ``/metrics``: the frontend keeps answering with the last
    values its engine reported.
    """

    ticks: int = 5
    engine: str | None = None

    @property
    def tail_scrapes(self) -> int:
        return self.ticks + 1

    @property
    def when_stale(self) -> str:
        return DATA_GAP  # old scrapes say nothing about progress now

    def evaluate_history(self, history: Sequence[Entry]) -> Evaluation:
        tail = self._tail(history)
        if tail is None:
            return Evaluation(DATA_GAP, reasons=(REASON_TOO_FEW_SAMPLES,))
        busy = self._least_busy(tail)
        progress, reasons = self._progress(tail)
        if busy is None or progress is None:
            return Evaluation(DATA_GAP, reasons=reasons or ("series_missing",))
        frozen = busy > 0 and progress == 0
        return Evaluation(
            VIOLATING if frozen else CLEAR,
            observed=progress,
            samples=float(len(tail)),
            detail={"least_busy": busy},
        )

    def _tail(self, history: Sequence[Entry]) -> list[VllmScrapeRecord] | None:
        """The last ``ticks + 1`` scrapes, when all of them succeeded."""
        tail = [record for _stamp, record in history[-(self.ticks + 1) :]]
        complete = len(tail) == self.ticks + 1
        return tail if complete and all(r.status == SCRAPE_OK for r in tail) else None

    def _least_busy(self, tail: Sequence[VllmScrapeRecord]) -> float | None:
        """The smallest running-plus-waiting count, or None if one is missing."""
        counts = [self._busy(record) for record in tail]
        return None if None in counts else min(c for c in counts if c is not None)

    def _progress(
        self, tail: Sequence[VllmScrapeRecord]
    ) -> tuple[float | None, tuple[str, ...]]:
        """How far the progress counters moved, or why that is unknown."""
        moved = 0.0
        for family in PROGRESS_COUNTERS:
            counter = counter_window(tail, family, engine=self.engine)
            if counter.reasons or counter.delta is None:
                return None, counter.reasons or ("no_delta",)
            moved += counter.delta
        return moved, ()

    def _busy(self, record: VllmScrapeRecord) -> float | None:
        total = 0.0
        for family in (RUNNING, WAITING):
            gauge = gauge_window([record], family, engine=self.engine)
            if gauge.reasons or gauge.last is None:
                return None
            total += gauge.last
        return total


def exporter_restarted(previous: VllmScrapeRecord, latest: VllmScrapeRecord) -> bool:
    """The exporter's identity (process start and engine set) changed.

    A point event, so the watcher records it directly instead of sustaining it.
    """
    before, after = exporter_identity(previous), exporter_identity(latest)
    return before is not None and after is not None and before != after


def overlaps(start_ns: int, end_ns: int, intervals: Sequence[tuple[int, int]]) -> bool:
    """``[start, end]`` meets any of the closed ``intervals``."""
    return any(lo <= end_ns and start_ns <= hi for lo, hi in intervals)


__all__ = [
    "PROGRESS_COUNTERS",
    "REASON_END_FAILED",
    "REASON_END_STALE",
    "REASON_START_MISSING",
    "REASON_NO_RECENT_SCRAPE",
    "REASON_TOO_FEW_SAMPLES",
    "CounterRateAtLeast",
    "Evaluation",
    "FrozenExporter",
    "GaugeAtLeast",
    "HistogramShareAbove",
    "ScrapeFailureShare",
    "ScrapeFailures",
    "Selection",
    "SignalExceeds",
    "WindowPredicate",
    "exporter_restarted",
    "on_monotonic_clock",
    "overlaps",
    "select_window",
]
