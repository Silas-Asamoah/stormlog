"""Effect timing, realization and recovery, judged from the reference channel.

The harness keeps a reference channel beside every diagnosed configuration:
the execution hook, a tailer, and scrapes every second. From it this module
says, per mechanism and at event time on the victim's clock (#221 design
A.4):

- when an episode's effect began and ended;
- whether the mechanism was realized, on the victim's own requests where it
  can be attributed to them;
- when recovery held, so the next episode may start.

None of it depends on the diagnosed configuration's capture or its latency.
The thresholds are frozen from ``dev_v1``; the defaults are the design's.
"""

from __future__ import annotations

import bisect
import functools
import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

Point = tuple[int, float]  # (event time in ns, value)

SECOND = 1_000_000_000

START = "start"
WAIT = "wait"
TIMEOUT = "timeout"


@dataclass(frozen=True)
class Thresholds:
    """The frozen rules; the defaults are #221 design A.4's."""

    window_ns: int = 5 * SECOND
    hold_ns: int = 10 * SECOND
    cadence_hold_ns: int = 10 * SECOND
    cached_loss_below: float = 0.5
    cached_recovered_at: float = 0.9
    kv_margin: float = 0.05
    priming_window_ns: int = 10 * SECOND
    priming_cached_at_least: float = 0.9
    min_recovery_ns: int = 60 * SECOND
    recovery_timeout_ns: int = 150 * SECOND
    # Cadence recovery (F4a, F4b, H0, P): over the hold, the busy gaps'
    # mean may exceed the baseline's by at most this share of the rate...
    rate_tolerance: float = 0.2
    # ...no more gaps may be longer than this many times the baseline's p99
    # than the baseline's own share of them allows by chance, and none this
    # many times its p99.9...
    long_gap_factor: float = 2.0
    # ...and no more gaps may lie above the baseline's p95 than chance
    # allows: this quantile of Binomial(n, exceedance_share).
    exceedance_share: float = 0.05
    exceedance_quantile: float = 0.99
    min_cadence_samples: int = 20
    # Queue recovery (F1, T1, W1): as many waits above the baseline's p95,
    # and waiting counts outside its range, as the same chance allows.
    min_wait_samples: int = 20
    min_gauge_samples: int = 5
    # T3b: the engine-wide prefix hit ratio falls at least this far.
    hit_ratio_drop: float = 0.05


@dataclass(frozen=True)
class Signals:
    """Reference-channel series on the victim's clock, each sorted by time.

    ``waits`` and ``cached_fraction`` are the victim's own requests: the
    admission-to-schedule wait in seconds at the request's first schedule,
    and its cached share of the prompt. ``victim_preemptions`` are the times
    a victim request was preempted, from the hook's ``scheduled.preempted``.
    ``waiting`` and ``kv_usage`` are scraped gauges. ``step_starts`` are hook
    step starts, and ``chunk_gaps`` the victim's gaps between streamed
    chunks, in seconds, at the later chunk. ``engine_hit_ratio`` is the
    engine-wide prefix-cache hit ratio between consecutive scrapes.
    ``in_flight`` holds the sorted, disjoint intervals during which at least
    one victim request was in flight. It has no default: a caller that
    doesn't know says None, and then no step gap counts as busy, so cadence
    recovery never holds rather than judging idle gaps as the engine's.
    """

    waits: Sequence[Point] = ()
    cached_fraction: Sequence[Point] = ()
    victim_preemptions: Sequence[int] = ()
    waiting: Sequence[Point] = ()
    kv_usage: Sequence[Point] = ()
    step_starts: Sequence[int] = ()
    chunk_gaps: Sequence[Point] = ()
    engine_hit_ratio: Sequence[Point] = ()
    in_flight: Sequence[tuple[int, int]] | None = field(kw_only=True)

    def step_gaps(self) -> list[Point]:
        """Each step's gap from the one before, in seconds, at its start."""
        starts = self.step_starts
        return [
            (later, (later - earlier) / SECOND)
            for earlier, later in zip(starts, starts[1:])
        ]

    def busy_step_gaps(self) -> list[Point]:
        """Each step gap's busy part, in seconds, at its later step: from the
        later of the earlier step and the opening of the in-flight interval
        the later step falls in. Idle time measures the traffic, not the
        engine, so it is left out; but a request that arrived in idle time
        and waited for a step felt that wait, so it counts. A gap whose later
        step falls in idle time is dropped. With the intervals unknown, no
        gap can be shown busy."""
        intervals = self.in_flight
        if intervals is None:
            return []
        opens = [start for start, _end in intervals]
        gaps = []
        for earlier, later in zip(self.step_starts, self.step_starts[1:]):
            opened = busy_since(intervals, opens, later)
            if opened is not None and later > max(earlier, opened):
                gaps.append((later, (later - max(earlier, opened)) / SECOND))
        return gaps


def busy_since(
    intervals: Sequence[tuple[int, int]], opens: Sequence[int], at_ns: int
) -> int | None:
    """When the in-flight interval holding ``at_ns`` opened, or None if no
    victim request was in flight then."""
    index = bisect.bisect_right(opens, at_ns) - 1
    if index >= 0 and intervals[index][1] >= at_ns:
        return intervals[index][0]
    return None


@dataclass(frozen=True)
class Actions:
    """The injector's own times, on the victim's clock. ``slot_ns`` is a
    null run's scheduled slot (N). ``peer_wait_extended`` says whether the
    other rank's NCCL kernels lengthened during an F5 pulse, from Nsight
    (C.6); None means it wasn't measured. ``action_end_ns`` is when the
    injector's action ended, the neighbor's last request included."""

    first_send_ns: int | None = None
    first_admission_ns: int | None = None
    action_end_ns: int | None = None
    first_stop_confirmed_ns: int | None = None
    last_continue_ns: int | None = None
    capture_started_ns: int | None = None
    stop_requested_ns: int | None = None
    stop_returned_ns: int | None = None
    drain_ns: int = 0
    pulses: Sequence[tuple[int, int]] = ()  # (stopped confirmed, continued)
    slot_ns: tuple[int, int] | None = None
    peer_wait_extended: bool | None = None


# ------------------------------------------------------------------ series


def between(points: Sequence[Point], start_ns: int, end_ns: int) -> list[float]:
    """The values of ``points`` at times in [start_ns, end_ns]."""
    times = [time for time, _value in points]
    first = bisect.bisect_left(times, start_ns)
    last = bisect.bisect_right(times, end_ns)
    return [value for _time, value in points[first:last]]


def events_between(times: Sequence[int], start_ns: int, end_ns: int) -> int:
    return bisect.bisect_right(times, end_ns) - bisect.bisect_left(times, start_ns)


def quantile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile by linear interpolation, or None for no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def first_window(
    points: Sequence[Point],
    start_ns: int,
    end_ns: int,
    width_ns: int,
    test: Callable[[float], bool],
) -> int | None:
    """The start of the first ``width_ns`` window from ``start_ns`` whose
    median value passes ``test``."""
    window = start_ns
    while window < end_ns:
        values = between(points, window, min(window + width_ns, end_ns) - 1)
        if values and test(statistics.median(values)):
            return window
        window += width_ns
    return None


# ------------------------------------------------------------------ criteria


class Criterion(Protocol):
    """Something that holds, or not, over an interval of event time."""

    def holds(self, start_ns: int, end_ns: int) -> bool: ...

    def change_points(self) -> Sequence[int]: ...


class AllWithin:
    """Every sample in the interval lies in [low, high]; with
    ``require_samples``, an interval without a sample does not hold."""

    def __init__(
        self,
        points: Sequence[Point],
        low: float = float("-inf"),
        high: float = float("inf"),
        *,
        require_samples: bool = True,
    ) -> None:
        self.samples = [time for time, _value in points]
        self.violations = [time for time, value in points if not low <= value <= high]
        self.require_samples = require_samples

    def holds(self, start_ns: int, end_ns: int) -> bool:
        if events_between(self.violations, start_ns, end_ns):
            return False
        return not self.require_samples or bool(
            events_between(self.samples, start_ns, end_ns)
        )

    def change_points(self) -> Sequence[int]:
        # A hold can begin just after a violation, or once a sample is seen.
        return self.violations + (self.samples if self.require_samples else [])


class Never:
    """Never holds: a criterion whose baseline is too thin to compare with."""

    def holds(self, start_ns: int, end_ns: int) -> bool:
        return False

    def change_points(self) -> Sequence[int]:
        return []


class NoEvents:
    """No event in the interval."""

    def __init__(self, times: Sequence[int]) -> None:
        self.times = list(times)

    def holds(self, start_ns: int, end_ns: int) -> bool:
        return events_between(self.times, start_ns, end_ns) == 0

    def change_points(self) -> Sequence[int]:
        return self.times


class MedianWithin:
    """The interval's first sample and its median lie in [low, high]. A
    hold that began with an outside sample would end the effect while it
    was still under way: a short dip is a minority of a long hold."""

    def __init__(
        self,
        points: Sequence[Point],
        low: float = float("-inf"),
        high: float = float("inf"),
    ) -> None:
        self.times = [time for time, _value in points]
        self.values = [value for _time, value in points]
        self.low = low
        self.high = high

    def holds(self, start_ns: int, end_ns: int) -> bool:
        first = bisect.bisect_left(self.times, start_ns)
        last = bisect.bisect_right(self.times, end_ns)
        values = self.values[first:last]
        if not values or not self.low <= values[0] <= self.high:
            return False
        return self.low <= statistics.median(values) <= self.high

    def change_points(self) -> Sequence[int]:
        return self.times


class MostlyWithin:
    """At least ``min_samples`` samples in the interval, the first of them
    inside [low, high], none above ``ceiling``, their mean at most
    ``mean_ceiling``, and no more outside the band than chance allows. The
    band is a baseline's tail, which ``exceedance_share`` of normal samples
    leave; the ceiling bounds how far out an allowed sample may be, and the
    mean how often samples may come near it, so recurring bursts beyond the
    band don't pass as chance. A hold that began with an outside sample
    would end the effect before its last sample."""

    def __init__(
        self,
        points: Sequence[Point],
        thresholds: Thresholds,
        *,
        min_samples: int,
        low: float = float("-inf"),
        high: float = float("inf"),
        ceiling: float = float("inf"),
        mean_ceiling: float = float("inf"),
    ) -> None:
        self.times = [time for time, _value in points]
        self.inside = [low <= value <= high for _time, value in points]
        self.outside = _prefix(float(not inside) for inside in self.inside)
        self.over = _prefix(float(value > ceiling) for _time, value in points)
        self.sums = _prefix(value for _time, value in points)
        self.mean_ceiling = mean_ceiling
        self.min_samples = min_samples
        self.thresholds = thresholds

    def holds(self, start_ns: int, end_ns: int) -> bool:
        first = bisect.bisect_left(self.times, start_ns)
        last = bisect.bisect_right(self.times, end_ns)
        count = last - first
        if count < self.min_samples or not self.inside[first]:
            return False
        if self.over[last] > self.over[first]:
            return False
        if (self.sums[last] - self.sums[first]) / count > self.mean_ceiling:
            return False
        return self.outside[last] - self.outside[first] <= allowed_exceedances(
            count,
            self.thresholds.exceedance_share,
            self.thresholds.exceedance_quantile,
        )

    def change_points(self) -> Sequence[int]:
        return self.times


class CadenceWithin:
    """A gap series is back to its baseline over the interval: at least
    ``min_cadence_samples`` gaps; their mean within ``rate_tolerance`` of
    the baseline's rate; no more longer than ``long_gap_factor`` times the
    baseline's p99 than the baseline's own share of such gaps allows by
    chance (none, when it had none: a healthy engine's prefill steps, under
    1% of its gaps, are normal, not a stall), and none longer than
    ``long_gap_factor`` times its p99.9; and no more above the baseline's
    p95 than chance allows.
    The mean weighs a long gap by its length, so a slow minority shows. A
    baseline of fewer than ``min_cadence_samples`` gaps can't be compared
    with, so it never holds.

    A gap is known when its later event arrives, so the time from the
    hold's last event to its end is a gap still open: its busy part (the
    time since the last event or since ``busy`` intervals last opened,
    whichever is later; all of it when ``busy`` is None) counts as one
    more gap. A stall still under way at the hold's end, an engine hung
    since its last step, is then a long gap, not an unseen one."""

    def __init__(
        self,
        points: Sequence[Point],
        baseline: GapStats,
        thresholds: Thresholds,
        busy: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        self.times = [time for time, _value in points]
        values = [value for _time, value in points]
        self.sums = _prefix(values)
        self.above = _prefix(float(value > baseline.p95) for value in values)
        self.longest = thresholds.long_gap_factor * baseline.p99
        self.long = _prefix(float(value > self.longest) for value in values)
        self.long_share = baseline.long_count / baseline.count if baseline.count else 0
        never = thresholds.long_gap_factor * baseline.p999
        self.too_long = _prefix(float(value > never) for value in values)
        self.mean_ceiling = baseline.mean / (1 - thresholds.rate_tolerance)
        self.thresholds = thresholds
        self.comparable = baseline.count >= thresholds.min_cadence_samples
        self.busy = busy
        self.opens = None if busy is None else [start for start, _end in busy]

    def open_gap(self, last_event_ns: int, end_ns: int) -> float:
        """The busy part, in seconds, of the gap still open at ``end_ns``."""
        since: int | None = last_event_ns
        if self.busy is not None and self.opens is not None:
            opened = busy_since(self.busy, self.opens, end_ns)
            since = None if opened is None else max(last_event_ns, opened)
        return 0.0 if since is None else max(0, end_ns - since) / SECOND

    def holds(self, start_ns: int, end_ns: int) -> bool:
        first = bisect.bisect_left(self.times, start_ns)
        last = bisect.bisect_right(self.times, end_ns)
        count = last - first
        if not self.comparable or count < self.thresholds.min_cadence_samples:
            return False
        tail = self.open_gap(self.times[last - 1], end_ns)
        if tail > self.longest or not self._long_gaps_fit(first, last, count):
            return False
        total = self.sums[last] - self.sums[first] + tail
        if total / (count + (tail > 0)) > self.mean_ceiling:
            return False
        above = self.above[last] - self.above[first]
        return above <= allowed_exceedances(
            count,
            self.thresholds.exceedance_share,
            self.thresholds.exceedance_quantile,
        )

    def change_points(self) -> Sequence[int]:
        return self.times

    def _long_gaps_fit(self, first: int, last: int, count: int) -> bool:
        """None far longer than any the baseline had, and no more long ones
        than its share of them allows (the open gap is never one of those:
        a stall may be under way)."""
        if self.too_long[last] > self.too_long[first]:
            return False
        long = self.long[last] - self.long[first]
        if long == 0 or self.long_share <= 0:
            return long == 0
        quantile = self.thresholds.exceedance_quantile
        return long <= allowed_exceedances(count, self.long_share, quantile)


def _prefix(values: Iterable[float]) -> list[float]:
    return list(itertools.accumulate(values, initial=0.0))


@functools.lru_cache(maxsize=4096)
def allowed_exceedances(count: int, share: float, quantile: float) -> int:
    """How many of ``count`` samples may lie beyond a baseline quantile by
    chance: the ``quantile`` point of Binomial(count, share)."""
    log_share, log_rest = math.log(share), math.log1p(-share)
    total = 0.0
    for k in range(count + 1):
        log_pmf = (
            math.lgamma(count + 1)
            - math.lgamma(k + 1)
            - math.lgamma(count - k + 1)
            + k * log_share
            + (count - k) * log_rest
        )
        total += math.exp(log_pmf)
        if total >= quantile:
            return k
    return count


def held_from(
    criteria: Sequence[Criterion],
    from_ns: int,
    until_ns: int,
    hold_ns: int,
) -> int | None:
    """The earliest time from ``from_ns`` after which every criterion holds
    for ``hold_ns``, with the whole hold observed by ``until_ns``. What a
    hold sees changes only where a change point leaves it (a start just
    after the point) or enters it (a start one hold before the point), so
    those are the candidates."""
    candidates = {from_ns}
    for criterion in criteria:
        for point in criterion.change_points():
            candidates.update(
                start
                for start in (point + 1, point - hold_ns)
                if from_ns <= start < until_ns
            )
    for start in sorted(candidates):
        end = start + hold_ns
        if end > until_ns:
            return None
        if all(criterion.holds(start, end) for criterion in criteria):
            return start
    return None


# ------------------------------------------------------------------ baseline


@dataclass(frozen=True)
class GapStats:
    """A gap series in the baseline: how many gaps, their mean, p95, p99
    and p99.9, in seconds, and how many were long (over ``long_gap_factor``
    times the p99). With no gaps nothing can be compared with it."""

    count: int = 0
    mean: float = math.inf
    p95: float = math.inf
    p99: float = math.inf
    long_count: int = 0
    p999: float = math.inf

    @classmethod
    def of(cls, values: Sequence[float], long_gap_factor: float = 2.0) -> GapStats:
        if not values:
            return cls()
        p99 = quantile(values, 0.99) or math.inf
        return cls(
            count=len(values),
            mean=statistics.fmean(values),
            p95=_q95(values),
            p99=p99,
            long_count=sum(1 for value in values if value > long_gap_factor * p99),
            p999=quantile(values, 0.999) or math.inf,
        )


@dataclass(frozen=True)
class Baseline:
    """What normal looked like in the baseline segment, with how many
    samples each figure rests on: a rule whose baseline has too few never
    holds (or, for a realization check, is incomplete)."""

    wait_p95: float
    waiting_low: float
    waiting_high: float
    kv_max: float
    steps: GapStats
    chunks: GapStats
    cached_median: float
    hit_ratio_median: float = 0.0
    wait_p99: float = math.inf
    wait_count: int = 0
    waiting_count: int = 0
    hit_ratio_count: int = 0
    wait_mean: float = math.inf
    waiting_mean: float = math.inf

    @classmethod
    def measure(
        cls,
        signals: Signals,
        start_ns: int,
        end_ns: int,
        thresholds: Thresholds | None = None,
    ) -> Baseline:
        factor = (thresholds or Thresholds()).long_gap_factor
        waits = between(signals.waits, start_ns, end_ns)
        gauge = between(signals.waiting, start_ns, end_ns)
        ratios = between(signals.engine_hit_ratio, start_ns, end_ns)
        waiting = gauge or [0.0]
        return cls(
            wait_p95=_q95(waits),
            wait_p99=_q(waits, 0.99),
            waiting_low=min(waiting),
            waiting_high=max(waiting),
            kv_max=max(between(signals.kv_usage, start_ns, end_ns) or [0.0]),
            steps=GapStats.of(
                between(signals.busy_step_gaps(), start_ns, end_ns), factor
            ),
            chunks=GapStats.of(between(signals.chunk_gaps, start_ns, end_ns), factor),
            cached_median=_median(between(signals.cached_fraction, start_ns, end_ns)),
            hit_ratio_median=_median(ratios),
            wait_count=len(waits),
            waiting_count=len(gauge),
            hit_ratio_count=len(ratios),
            wait_mean=statistics.fmean(waits) if waits else math.inf,
            waiting_mean=statistics.fmean(gauge) if gauge else math.inf,
        )


def _q95(values: Sequence[float]) -> float:
    return _q(values, 0.95)


def _q(values: Sequence[float], q: float) -> float:
    found = quantile(values, q)
    return float("inf") if found is None else found


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


# ------------------------------------------------------------------ per mechanism


@dataclass(frozen=True)
class Timing:
    """An episode's effect, on the victim's clock."""

    onset_ns: int | None
    basis: str
    end_ns: int | None = None
    recovery_held_at_ns: int | None = None


@dataclass(frozen=True)
class Context:
    """Everything a mechanism's rule reads."""

    signals: Signals
    baseline: Baseline
    actions: Actions
    start_ns: int
    until_ns: int
    thresholds: Thresholds = field(default_factory=Thresholds)


def _queue_criteria(context: Context) -> list[Criterion]:
    """Waits back below the baseline's p95, and the waiting count within its
    range, but for as many exceptions as chance allows: 5% of normal waits
    are above a p95, so "every wait" would almost never hold. But none may
    be far out: no wait above ``long_gap_factor`` times the baseline's p99,
    and no waiting count above that factor times the baseline's highest
    (or 1), as cadence bounds its gaps. Nor may they be near those bounds
    too often: the hold's mean wait is at most the baseline's over
    ``1 - rate_tolerance`` (1.25 times), as cadence bounds its mean gap, and
    its mean waiting count too, or at most one more. A
    baseline with fewer samples than a criterion needs in its hold never
    holds."""
    baseline, thresholds = context.baseline, context.thresholds
    factor = thresholds.long_gap_factor
    slack = 1 / (1 - thresholds.rate_tolerance)
    waits: Criterion = Never()
    if baseline.wait_count >= thresholds.min_wait_samples:
        waits = MostlyWithin(
            context.signals.waits,
            thresholds,
            min_samples=thresholds.min_wait_samples,
            high=baseline.wait_p95,
            ceiling=factor * baseline.wait_p99,
            mean_ceiling=slack * baseline.wait_mean,
        )
    waiting: Criterion = Never()
    if baseline.waiting_count >= thresholds.min_gauge_samples:
        waiting = MostlyWithin(
            context.signals.waiting,
            thresholds,
            min_samples=thresholds.min_gauge_samples,
            low=baseline.waiting_low,
            high=baseline.waiting_high,
            ceiling=factor * max(baseline.waiting_high, 1.0),
            mean_ceiling=max(slack * baseline.waiting_mean, baseline.waiting_mean + 1),
        )
    return [waits, waiting]


def _kv_criteria(context: Context) -> list[Criterion]:
    ceiling = context.baseline.kv_max + context.thresholds.kv_margin
    return [
        NoEvents(context.signals.victim_preemptions),
        AllWithin(context.signals.kv_usage, high=ceiling),
    ]


def _cache_criteria(context: Context) -> list[Criterion]:
    threshold = context.thresholds.cached_recovered_at
    return [MedianWithin(context.signals.cached_fraction, low=threshold)]


def _cadence_criteria(context: Context, *, chunks: bool) -> list[Criterion]:
    """Cadence is back when the busy step gaps (and, for the front end, the
    victim's chunk gaps) look like the baseline's again: like with like,
    since idle gaps measure the traffic, not the engine."""
    signals, baseline = context.signals, context.baseline
    thresholds = context.thresholds
    busy = signals.in_flight
    criteria: list[Criterion] = [
        CadenceWithin(signals.busy_step_gaps(), baseline.steps, thresholds, busy)
    ]
    if chunks:
        criteria.append(
            CadenceWithin(signals.chunk_gaps, baseline.chunks, thresholds, busy)
        )
    return criteria


def _queue_onset(context: Context) -> tuple[int | None, str]:
    p95 = context.baseline.wait_p95
    onset = first_window(
        context.signals.waits,
        context.start_ns,
        context.until_ns,
        context.thresholds.window_ns,
        lambda median: median > p95,
    )
    return onset, "victim_wait_median_above_baseline_p95"


def _preemption_onset(context: Context) -> tuple[int | None, str]:
    times = context.signals.victim_preemptions
    index = bisect.bisect_left(times, context.start_ns)
    onset = times[index] if index < len(times) else None
    return onset, "reference_hook_preempted_victim"


def _cache_onset(context: Context) -> tuple[int | None, str]:
    threshold = context.thresholds.cached_loss_below
    onset = first_window(
        context.signals.cached_fraction,
        context.start_ns,
        context.until_ns,
        context.thresholds.window_ns,
        lambda median: median < threshold,
    )
    return onset, "victim_cached_fraction_median_below_0.5"


def _action_onset(name: str, basis: str) -> Callable[[Context], tuple[int | None, str]]:
    def onset(context: Context) -> tuple[int | None, str]:
        return getattr(context.actions, name), basis

    return onset


@dataclass(frozen=True)
class Mechanism:
    """One row of A.4's table: how the effect begins, and what recovery is."""

    onset: Callable[[Context], tuple[int | None, str]]
    recovery: Callable[[Context], list[Criterion]]
    end_from: str = "onset"  # recovery is sought from the onset or the action's end
    hold: str = "hold_ns"
    # A workload twin leaves the signals alone, so its recovery holds at
    # once; its effect, the benign change, lasts as long as its action.
    spans_action: bool = False


def _stall_recovery(context: Context) -> list[Criterion]:
    return _cadence_criteria(context, chunks=False)


def _frontend_recovery(context: Context) -> list[Criterion]:
    return _cadence_criteria(context, chunks=True)


def _load_recovery(context: Context) -> list[Criterion]:
    return _queue_criteria(context) + _kv_criteria(context)


_FIRST_SEND = _action_onset("first_send_ns", "neighbor_first_send")
_FIRST_ADMISSION = _action_onset("first_admission_ns", "neighbor_first_admission")
_FIRST_STOP = _action_onset("first_stop_confirmed_ns", "first_sigstop_confirmed")

MECHANISMS: dict[str, Mechanism] = {
    "F1": Mechanism(_queue_onset, _queue_criteria),
    "T1": Mechanism(_FIRST_SEND, _queue_criteria, spans_action=True),
    "F2": Mechanism(_preemption_onset, _kv_criteria),
    "T2": Mechanism(_FIRST_ADMISSION, _kv_criteria, spans_action=True),
    "F3": Mechanism(_cache_onset, _cache_criteria),
    "T3": Mechanism(_FIRST_SEND, _cache_criteria, spans_action=True),
    "T3b": Mechanism(_FIRST_SEND, _cache_criteria, spans_action=True),
    "F4a": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "F4b": Mechanism(_FIRST_STOP, _frontend_recovery, "action_end", "cadence_hold_ns"),
    "H0": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "P": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "W1": Mechanism(_FIRST_SEND, _load_recovery),
    "F5": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "R0": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
}


def base_type(episode_type: str) -> str:
    """The fault whose rules a short twin (``S-<x>``) follows."""
    return episode_type[2:] if episode_type.startswith("S-") else episode_type


def effect_timing(episode_type: str, context: Context) -> Timing:
    """When the episode's effect began and ended, and when recovery held.

    Effect end is the start of the first interval over which the recovery
    criteria hold; recovery holds at its end. For pulses, recovery is sought
    from the last ``SIGCONT``. I1's effect runs from the stop request to the
    stop's return plus #219's drain, and N's is its scheduled slot. A short
    twin follows its fault's rules.

    Raises:
        KeyError: for an episode type with no rule here.
        ValueError: for N without a scheduled slot.
    """
    episode_type = base_type(episode_type)
    if episode_type in _TIMED_BY_ACTIONS:
        return _TIMED_BY_ACTIONS[episode_type](context)
    mechanism = MECHANISMS[episode_type]
    onset, basis = mechanism.onset(context)
    if onset is None:
        return Timing(None, basis)
    since = onset
    if mechanism.end_from == "action_end":
        since = context.actions.last_continue_ns or onset
    hold = getattr(context.thresholds, mechanism.hold)
    end = held_from(mechanism.recovery(context), since, context.until_ns, hold)
    action_end = context.actions.action_end_ns
    if mechanism.spans_action and end is not None and action_end is not None:
        end = max(end, action_end)
    return Timing(onset, basis, end, None if end is None else end + hold)


def _capture_timing(context: Context) -> Timing:
    actions = context.actions
    if actions.stop_requested_ns is None or actions.stop_returned_ns is None:
        return Timing(actions.stop_requested_ns, "capture_stop_requested")
    end = actions.stop_returned_ns + actions.drain_ns
    return Timing(actions.stop_requested_ns, "capture_stop_requested", end, end)


def _slot_timing(context: Context) -> Timing:
    """N injects nothing: its scored window is the slot it was scheduled."""
    slot = context.actions.slot_ns
    if slot is None:
        raise ValueError("a null run (N) needs its scheduled slot")
    return Timing(slot[0], "scheduled_slot", slot[1], slot[1])


_TIMED_BY_ACTIONS: dict[str, Callable[[Context], Timing]] = {
    "I1": _capture_timing,
    "N": _slot_timing,
}


# ------------------------------------------------------------------ realization


@dataclass(frozen=True)
class Check:
    """One realization check, as recorded in the ground truth. A check that
    doesn't gate is recorded only, as when it adds a realized mechanism. An
    incomplete check had no signal to judge (the reference channel lacked
    it): it neither passes nor fails, and the episode's observation is
    incomplete."""

    name: str
    passed: bool
    value: Any = None
    gating: bool = True
    incomplete: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "layer": "realization",
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "gating": self.gating,
            "incomplete": self.incomplete,
        }


def _absent(name: str) -> Check:
    return Check(name, False, None, incomplete=True)


def realization(
    episode_type: str, context: Context, timing: Timing
) -> tuple[bool, list[Check]]:
    """Whether the mechanism occurred (A.4's realization column), with the
    checks behind the verdict. A short twin follows its fault's rule; a type
    whose column is empty (W1, P, N) is realized when its action took place.

    Raises:
        KeyError: for a type with no rule here, a typo or X1–X3 (judged by
            C.6's outage criteria) among them.
    """
    checks = _REALIZATION[base_type(episode_type)](context, timing)
    gating = [check for check in checks if check.gating]
    judged = [check for check in gating if not check.incomplete]
    if gating and not judged:
        return False, checks  # nothing could be judged: never realized vacuously
    return all(check.passed for check in judged), checks


def observation_of(checks: Sequence[Check]) -> str | None:
    """``incomplete`` when a check had no signal to judge, else None."""
    return "incomplete" if any(check.incomplete for check in checks) else None


def added_mechanisms(episode_type: str, checks: Sequence[Check]) -> tuple[str, ...]:
    """Mechanisms an episode realized beyond its label (A.4): an API-server
    pulse (F4b) whose engine stopped stepping also stalled the engine core."""
    if base_type(episode_type) != "F4b":
        return ()
    stalled = any(
        check.name == "engine_progress_during_pulse" and not check.passed
        for check in checks
    )
    return ("host_stall@engine_core",) if stalled else ()


def _window_of(context: Context, timing: Timing) -> tuple[int, int]:
    """Realization is judged over the effect and the whole action: a twin
    leaves the signals alone, so its effect window is empty, but its action
    runs on."""
    end = timing.end_ns if timing.end_ns is not None else context.until_ns
    action_end = context.actions.action_end_ns
    return context.start_ns, max(end, action_end if action_end is not None else end)


def _victim_preemptions(context: Context, timing: Timing) -> int:
    start, end = _window_of(context, timing)
    return events_between(context.signals.victim_preemptions, start, end)


def _realized_queue(context: Context, timing: Timing) -> list[Check]:
    return [
        Check("onset_reached", timing.onset_ns is not None),
        Check("no_victim_preemption", _victim_preemptions(context, timing) == 0),
    ]


def _waits_in_baseline(context: Context, timing: Timing) -> list[Check]:
    onset, _basis = _queue_onset(context)
    return [Check("waits_within_baseline", onset is None)]


def _realized_kv(context: Context, timing: Timing) -> list[Check]:
    count = _victim_preemptions(context, timing)
    return [Check("victim_preempted", count > 0, count)]


def _no_preemption(context: Context, timing: Timing) -> list[Check]:
    count = _victim_preemptions(context, timing)
    return [Check("no_victim_preemption", count == 0, count)]


def _realized_cache_loss(context: Context, timing: Timing) -> list[Check]:
    onset, _basis = _cache_onset(context)
    return [Check("cached_fraction_below_0.5", onset is not None)]


def _hit_ratio_fell(context: Context, timing: Timing) -> list[Check]:
    """T3b: the engine-wide hit ratio falls, though the victim's doesn't.
    Without the ratio in the baseline or the episode, it is incomplete."""
    start, end = _window_of(context, timing)
    values = between(context.signals.engine_hit_ratio, start, end)
    if not values or not context.baseline.hit_ratio_count:
        return [_absent("engine_hit_ratio_fell")]
    median = statistics.median(values)
    ceiling = context.baseline.hit_ratio_median - context.thresholds.hit_ratio_drop
    return [Check("engine_hit_ratio_fell", median <= ceiling, median)]


def _cache_unchanged(context: Context, timing: Timing) -> list[Check]:
    start, end = _window_of(context, timing)
    values = between(context.signals.cached_fraction, start, end)
    if not values:
        return [_absent("cached_fraction_unchanged")]
    median = statistics.median(values)
    threshold = context.thresholds.cached_recovered_at
    return [Check("cached_fraction_unchanged", median >= threshold, median)]


def _stopped(context: Context, timing: Timing) -> list[Check]:
    pulses = context.actions.pulses
    return [Check("stopped_state_confirmed", bool(pulses), len(pulses))]


def _engine_stalled(context: Context, timing: Timing) -> list[Check]:
    """F4a: no hook step started while the engine was stopped."""
    steps = context.signals.step_starts
    during = sum(
        events_between(steps, stop + 1, cont - 1)
        for stop, cont in context.actions.pulses
    )
    return _stopped(context, timing) + [
        Check("no_step_during_pulse", during == 0, during)
    ]


def _engine_progressed(context: Context, timing: Timing) -> list[Check]:
    """F4b: whether the engine kept stepping in every pulse while a victim
    request was in flight. It doesn't gate: an engine that stalled too adds
    that mechanism instead (``added_mechanisms``)."""
    signals = context.signals
    judged = [
        (stop, cont)
        for stop, cont in context.actions.pulses
        if signals.in_flight is None or _overlaps(signals.in_flight, stop, cont)
    ]
    progressed = sum(
        1
        for stop, cont in judged
        if events_between(signals.step_starts, stop + 1, cont - 1)
    )
    check = Check(
        "engine_progress_during_pulse",
        progressed == len(judged),
        [progressed, len(judged)],
        gating=False,
    )
    return _stopped(context, timing) + [check]


def _overlaps(intervals: Sequence[tuple[int, int]], start: int, end: int) -> bool:
    return any(left <= end and start <= right for left, right in intervals)


def _captured(context: Context, timing: Timing) -> list[Check]:
    actions = context.actions
    both = (
        actions.capture_started_ns is not None and actions.stop_returned_ns is not None
    )
    return [Check("capture_started_and_stopped", both)]


def _peer_waited(context: Context, timing: Timing) -> list[Check]:
    """F5: the stopped rank's peer waits longer in NCCL (C.6, from Nsight)."""
    extended = context.actions.peer_wait_extended
    return _stopped(context, timing) + [
        Check("nccl_wait_asymmetry", extended is True, extended)
    ]


def _declared_empty(context: Context, timing: Timing) -> list[Check]:
    """A.4's column is empty: realized when the action took place."""
    return []


_REALIZATION: dict[str, Callable[[Context, Timing], list[Check]]] = {
    "F1": _realized_queue,
    "T1": _waits_in_baseline,
    "F2": _realized_kv,
    "T2": _no_preemption,
    "F3": _realized_cache_loss,
    "T3": _cache_unchanged,
    "T3b": lambda context, timing: _cache_unchanged(context, timing)
    + _hit_ratio_fell(context, timing),
    "F4a": _engine_stalled,
    "F4b": _engine_progressed,
    "H0": _stopped,
    "I1": _captured,
    "W1": _declared_empty,
    "P": _declared_empty,
    "N": _declared_empty,
    "F5": _peer_waited,
    "R0": _stopped,
}


# ------------------------------------------------------------------ the run


def priming_check(
    signals: Signals, priming_end_ns: int, thresholds: Thresholds = Thresholds()
) -> tuple[bool, float | None]:
    """A.4 step 2: the victim's median cached fraction over the last 10 s of
    priming is at least 0.9, or the run is a protocol failure."""
    start = priming_end_ns - thresholds.priming_window_ns
    values = between(signals.cached_fraction, start, priming_end_ns)
    if not values:
        return False, None
    median = statistics.median(values)
    return median >= thresholds.priming_cached_at_least, median


def next_episode(
    action_end_ns: int,
    recovery_held_at_ns: int | None,
    now_ns: int,
    thresholds: Thresholds = Thresholds(),
) -> str:
    """Whether the next episode may start: once the previous one's recovery
    has held, and no sooner than 60 s after its action ended. Recovery that
    holds only after 150 s, or not by then, is a timeout, which skips the
    run's remaining episodes, whenever the harness asks."""
    deadline = action_end_ns + thresholds.recovery_timeout_ns
    if recovery_held_at_ns is not None and recovery_held_at_ns > deadline:
        return TIMEOUT
    earliest = action_end_ns + thresholds.min_recovery_ns
    if recovery_held_at_ns is not None and now_ns >= max(earliest, recovery_held_at_ns):
        return START
    return TIMEOUT if now_ns >= deadline else WAIT


__all__ = [
    "MECHANISMS",
    "SECOND",
    "START",
    "TIMEOUT",
    "WAIT",
    "Actions",
    "AllWithin",
    "Baseline",
    "CadenceWithin",
    "Check",
    "Context",
    "Criterion",
    "GapStats",
    "Mechanism",
    "MedianWithin",
    "MostlyWithin",
    "NoEvents",
    "Signals",
    "Thresholds",
    "Timing",
    "added_mechanisms",
    "allowed_exceedances",
    "base_type",
    "between",
    "effect_timing",
    "first_window",
    "held_from",
    "next_episode",
    "observation_of",
    "priming_check",
    "quantile",
    "realization",
]
