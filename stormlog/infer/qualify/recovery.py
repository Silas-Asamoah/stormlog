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
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

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
    cadence_hold_ns: int = 5 * SECOND
    cached_loss_below: float = 0.5
    cached_recovered_at: float = 0.9
    kv_margin: float = 0.05
    priming_window_ns: int = 10 * SECOND
    priming_cached_at_least: float = 0.9
    min_recovery_ns: int = 60 * SECOND
    recovery_timeout_ns: int = 150 * SECOND


@dataclass(frozen=True)
class Signals:
    """Reference-channel series on the victim's clock, each sorted by time.

    ``waits`` and ``cached_fraction`` are the victim's own requests: the
    admission-to-schedule wait in seconds at the request's first schedule,
    and its cached share of the prompt. ``victim_preemptions`` are the times
    a victim request was preempted, from the hook's ``scheduled.preempted``.
    ``waiting`` and ``kv_usage`` are scraped gauges. ``step_starts`` are hook
    step starts, and ``chunk_gaps`` the victim's gaps between streamed
    chunks, in seconds, at the later chunk.
    """

    waits: Sequence[Point] = ()
    cached_fraction: Sequence[Point] = ()
    victim_preemptions: Sequence[int] = ()
    waiting: Sequence[Point] = ()
    kv_usage: Sequence[Point] = ()
    step_starts: Sequence[int] = ()
    chunk_gaps: Sequence[Point] = ()

    def step_gaps(self) -> list[Point]:
        """Each step's gap from the one before, in seconds, at its start."""
        starts = self.step_starts
        return [
            (later, (later - earlier) / SECOND)
            for earlier, later in zip(starts, starts[1:])
        ]


@dataclass(frozen=True)
class Actions:
    """The injector's own times, on the victim's clock."""

    first_send_ns: int | None = None
    first_admission_ns: int | None = None
    first_stop_confirmed_ns: int | None = None
    last_continue_ns: int | None = None
    stop_requested_ns: int | None = None
    stop_returned_ns: int | None = None
    drain_ns: int = 0
    pulses: Sequence[tuple[int, int]] = ()  # (stopped confirmed, continued)


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


class NoEvents:
    """No event in the interval."""

    def __init__(self, times: Sequence[int]) -> None:
        self.times = list(times)

    def holds(self, start_ns: int, end_ns: int) -> bool:
        return events_between(self.times, start_ns, end_ns) == 0

    def change_points(self) -> Sequence[int]:
        return self.times


class MedianWithin:
    """The interval's median sample lies in [low, high]."""

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
        return bool(values) and self.low <= statistics.median(values) <= self.high

    def change_points(self) -> Sequence[int]:
        return self.times


def held_from(
    criteria: Sequence[Criterion],
    from_ns: int,
    until_ns: int,
    hold_ns: int,
) -> int | None:
    """The earliest time from ``from_ns`` after which every criterion holds
    for ``hold_ns``, with the whole hold observed by ``until_ns``. A
    criterion can only start to hold just after one of its change points,
    so those are the candidates."""
    candidates = {from_ns}
    for criterion in criteria:
        candidates.update(
            point + 1
            for point in criterion.change_points()
            if from_ns <= point < until_ns
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
class Baseline:
    """What normal looked like in the baseline segment."""

    wait_p95: float
    waiting_low: float
    waiting_high: float
    kv_max: float
    step_gap_p95: float
    chunk_gap_p95: float
    cached_median: float

    @classmethod
    def measure(cls, signals: Signals, start_ns: int, end_ns: int) -> Baseline:
        waiting = between(signals.waiting, start_ns, end_ns) or [0.0]
        return cls(
            wait_p95=_q95(between(signals.waits, start_ns, end_ns)),
            waiting_low=min(waiting),
            waiting_high=max(waiting),
            kv_max=max(between(signals.kv_usage, start_ns, end_ns) or [0.0]),
            step_gap_p95=_q95(between(signals.step_gaps(), start_ns, end_ns)),
            chunk_gap_p95=_q95(between(signals.chunk_gaps, start_ns, end_ns)),
            cached_median=_median(between(signals.cached_fraction, start_ns, end_ns)),
        )


def _q95(values: Sequence[float]) -> float:
    found = quantile(values, 0.95)
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
    baseline = context.baseline
    return [
        AllWithin(context.signals.waits, high=baseline.wait_p95),
        AllWithin(context.signals.waiting, baseline.waiting_low, baseline.waiting_high),
    ]


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
    """Cadence is back when the median step gap (and, for the front end, the
    median chunk gap) is within the baseline's p95: a served engine idles
    between requests, so some single gap in any interval is longer."""
    signals, baseline = context.signals, context.baseline
    criteria: list[Criterion] = [
        MedianWithin(signals.step_gaps(), high=baseline.step_gap_p95)
    ]
    if chunks:
        criteria.append(MedianWithin(signals.chunk_gaps, high=baseline.chunk_gap_p95))
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
    "T1": Mechanism(_FIRST_SEND, _queue_criteria),
    "F2": Mechanism(_preemption_onset, _kv_criteria),
    "T2": Mechanism(_FIRST_ADMISSION, _kv_criteria),
    "F3": Mechanism(_cache_onset, _cache_criteria),
    "T3": Mechanism(_FIRST_SEND, _cache_criteria),
    "T3b": Mechanism(_FIRST_SEND, _cache_criteria),
    "F4a": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "F4b": Mechanism(_FIRST_STOP, _frontend_recovery, "action_end", "cadence_hold_ns"),
    "H0": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "P": Mechanism(_FIRST_STOP, _stall_recovery, "action_end", "cadence_hold_ns"),
    "W1": Mechanism(_FIRST_SEND, _load_recovery),
}


def effect_timing(episode_type: str, context: Context) -> Timing:
    """When the episode's effect began and ended, and when recovery held.

    Effect end is the start of the first interval over which the recovery
    criteria hold; recovery holds at its end. For pulses, recovery is sought
    from the last ``SIGCONT``. I1's effect runs from the stop request to the
    stop's return plus #219's drain.

    Raises:
        KeyError: for an episode type with no rule here.
    """
    if episode_type == "I1":
        return _capture_timing(context)
    mechanism = MECHANISMS[episode_type]
    onset, basis = mechanism.onset(context)
    if onset is None:
        return Timing(None, basis)
    since = onset
    if mechanism.end_from == "action_end":
        since = context.actions.last_continue_ns or onset
    hold = getattr(context.thresholds, mechanism.hold)
    end = held_from(mechanism.recovery(context), since, context.until_ns, hold)
    return Timing(onset, basis, end, None if end is None else end + hold)


def _capture_timing(context: Context) -> Timing:
    actions = context.actions
    if actions.stop_requested_ns is None or actions.stop_returned_ns is None:
        return Timing(actions.stop_requested_ns, "capture_stop_requested")
    end = actions.stop_returned_ns + actions.drain_ns
    return Timing(actions.stop_requested_ns, "capture_stop_requested", end, end)


# ------------------------------------------------------------------ realization


@dataclass(frozen=True)
class Check:
    """One realization check, as recorded in the ground truth."""

    name: str
    passed: bool
    value: Any = None

    def to_record(self) -> dict[str, Any]:
        return {
            "layer": "realization",
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
        }


def realization(
    episode_type: str, context: Context, timing: Timing
) -> tuple[bool, list[Check]]:
    """Whether the mechanism occurred (A.4's realization column), with the
    checks behind the verdict. An episode type the column leaves empty is
    realized when its action took place."""
    rule = _REALIZATION.get(episode_type)
    checks = [] if rule is None else rule(context, timing)
    return all(check.passed for check in checks), checks


def _window_of(context: Context, timing: Timing) -> tuple[int, int]:
    end = timing.end_ns if timing.end_ns is not None else context.until_ns
    return context.start_ns, end


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


def _cache_unchanged(context: Context, timing: Timing) -> list[Check]:
    start, end = _window_of(context, timing)
    values = between(context.signals.cached_fraction, start, end)
    median = statistics.median(values) if values else None
    threshold = context.thresholds.cached_recovered_at
    return [
        Check(
            "cached_fraction_unchanged",
            median is not None and median >= threshold,
            median,
        )
    ]


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
    """F4b: the engine kept stepping while the API server was stopped."""
    steps = context.signals.step_starts
    during = sum(
        events_between(steps, stop, cont) for stop, cont in context.actions.pulses
    )
    return _stopped(context, timing) + [
        Check("engine_progress_during_pulse", during > 0, during)
    ]


def _captured(context: Context, timing: Timing) -> list[Check]:
    actions = context.actions
    return [Check("capture_started_and_stopped", actions.stop_returned_ns is not None)]


_REALIZATION: dict[str, Callable[[Context, Timing], list[Check]]] = {
    "F1": _realized_queue,
    "T1": _waits_in_baseline,
    "F2": _realized_kv,
    "T2": _no_preemption,
    "F3": _realized_cache_loss,
    "T3": _cache_unchanged,
    "T3b": _cache_unchanged,
    "F4a": _engine_stalled,
    "F4b": _engine_progressed,
    "H0": _stopped,
    "I1": _captured,
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
    has held, and no sooner than 60 s after its action ended; a timeout 150 s
    after it skips the run's remaining episodes."""
    earliest = action_end_ns + thresholds.min_recovery_ns
    if recovery_held_at_ns is not None and now_ns >= max(earliest, recovery_held_at_ns):
        return START
    if now_ns >= action_end_ns + thresholds.recovery_timeout_ns:
        return TIMEOUT
    return WAIT


__all__ = [
    "MECHANISMS",
    "SECOND",
    "START",
    "TIMEOUT",
    "WAIT",
    "Actions",
    "AllWithin",
    "Baseline",
    "Check",
    "Context",
    "Criterion",
    "Mechanism",
    "MedianWithin",
    "NoEvents",
    "Signals",
    "Thresholds",
    "Timing",
    "between",
    "effect_timing",
    "first_window",
    "held_from",
    "next_episode",
    "priming_check",
    "quantile",
    "realization",
]
