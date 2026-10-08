"""Place engine times on the artifact's clock, as intervals or not at all.

The artifact's own clock, R, is the client's wall clock. A client stamp is a
read of R. An engine stamp on the same host and boot is a read of the same
wall clock, bracketed by the hook: the wall time at its monotonic read lies
between ``wall_ns`` and ``wall_after_ns``. A stamp from a hook without the
second read has no stated bound, and one from another host has no shared
clock whose continuity anyone watched; both are ``unknown``, never a guess.

A pair of reads, one by the client and one by the engine, is only comparable
if the wall clock did not jump between them. The engine's ``wall - mono``
offset, sampled at every step it records, splits each epoch into continuity
segments where that offset stayed within slew; a pair whose client read falls
outside the engine read's segment is withheld. A bracketed sample's offset
is an interval, ``[wall - mono, wall_after - mono]``: a thread descheduled
between its reads widens it, which says nothing about a jump, so a segment
ends only where a sample's interval leaves every offset the segment's
earlier samples allow, widened by slew. Two jumps that cancel between two
samples cannot be seen, so a segment is ``monitored`` to within its largest
sample gap, never verified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .diagnosis_join import RunView

Interval = tuple[int, int]

BASIS_CLIENT = "client_clock"
BASIS_SAME_HOST = "same_host_wall"
UNKNOWN_LEGACY = "legacy_unbracketed_stamp"
UNKNOWN_OTHER_HOST = "cross_host_continuity_unknown"
UNKNOWN_DISCONTINUITY = "wall_clock_discontinuity"
UNKNOWN_NO_STAMP = "no_stamp"
# Offset movement allowed between samples: NTP slews at up to 500 ppm; 100
# ppm covers normal slewing, and anything above 10 us is never ignored.
SLEW_PPM = 100
SLEW_FLOOR_NS = 10_000


@dataclass(frozen=True)
class Placed:
    """A time on R: an interval, or why it has none."""

    interval: Interval | None
    basis: str | None = None
    unknown: str | None = None
    mono_ns: int | None = None  # an engine read's own monotonic time

    @classmethod
    def unknown_because(cls, reason: str) -> Placed:
        return cls(None, None, reason)


@dataclass(frozen=True)
class Segment:
    """Engine samples whose ``wall - mono`` offset stayed within slew."""

    mono_start: int
    mono_end: int
    wall_start: int
    wall_end: int
    samples: int
    max_gap_ns: int


class EngineClock:
    """How one engine's wall stamps relate to R."""

    def __init__(self, view: RunView, producer: str) -> None:
        self.producer = producer
        self.same_host = _same_host(view, producer)
        self.segments = continuity_segments(_samples(view, producer))

    def describe(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "same_host": self.same_host,
            "continuity": "monitored" if self.segments else "unknown",
            "segments": [
                {
                    "mono_start_ns": s.mono_start,
                    "mono_end_ns": s.mono_end,
                    "wall_start_ns": s.wall_start,
                    "wall_end_ns": s.wall_end,
                    "samples": s.samples,
                    "max_sample_gap_ns": s.max_gap_ns,
                }
                for s in self.segments
            ],
        }

    def place(self, wall_ns: Any, wall_after_ns: Any, mono_ns: Any) -> Placed:
        """An engine wall stamp, read at ``mono_ns``, on R."""
        if not self.same_host:
            return Placed.unknown_because(UNKNOWN_OTHER_HOST)
        wall, after = _integer(wall_ns), _integer(wall_after_ns)
        if wall is None:
            return Placed.unknown_because(UNKNOWN_NO_STAMP)
        if after is None or after < wall:
            return Placed.unknown_because(UNKNOWN_LEGACY)
        return Placed((wall, after), BASIS_SAME_HOST, mono_ns=_integer(mono_ns))

    def comparable(self, client_ns: int, engine: Placed) -> str | None:
        """Why a client read and an engine read cannot be subtracted, if
        they cannot: the engine's own placement, or a possible wall jump
        between them. The engine read's segment is found by its monotonic
        time; the client read must lie within that segment's wall span,
        or, beyond the run's first or last sample, within one sample gap."""
        if engine.interval is None:
            return engine.unknown or UNKNOWN_NO_STAMP
        index = self._segment_index(engine.mono_ns)
        if index is None:
            return UNKNOWN_DISCONTINUITY
        segment = self.segments[index]
        before = segment.max_gap_ns if index == 0 else 0
        after = segment.max_gap_ns if index == len(self.segments) - 1 else 0
        if not segment.wall_start - before <= client_ns <= segment.wall_end + after:
            return UNKNOWN_DISCONTINUITY
        return None

    def _segment_index(self, mono_ns: int | None) -> int | None:
        """The segment an engine read at ``mono_ns`` falls in; reads before
        the first sample or after the last belong to the outer segments,
        and a read between two segments to none."""
        if mono_ns is None or not self.segments:
            return None
        last = len(self.segments) - 1
        for index, segment in enumerate(self.segments):
            low = segment.mono_start if index else mono_ns
            high = segment.mono_end if index < last else mono_ns
            if low <= mono_ns <= high:
                return index
        return None


Sample = tuple[int, ...]  # (mono, wall) or (mono, wall, wall_after)


def continuity_segments(samples: Sequence[Sample]) -> list[Segment]:
    """Split (mono, wall[, wall_after]) samples where the offset moved
    beyond slew: where a sample's offset interval misses the offsets the
    segment's samples so far allow, each widened by slew since."""
    ordered = sorted(set(samples))
    segments: list[Segment] = []
    start = 0
    allowed = _offsets(ordered[0]) if ordered else (0, 0)
    for index in range(1, len(ordered) + 1):
        if index < len(ordered):
            narrowed = _narrowed(allowed, ordered[index - 1], ordered[index])
            if narrowed is not None:
                allowed = narrowed
                continue
            allowed = _offsets(ordered[index])
        part = ordered[start:index]
        gaps = [b[0] - a[0] for a, b in zip(part, part[1:])]
        segments.append(
            Segment(
                mono_start=part[0][0],
                mono_end=part[-1][0],
                wall_start=min(sample[1] for sample in part),
                wall_end=max(sample[1] for sample in part),
                samples=len(part),
                max_gap_ns=max(gaps, default=0),
            )
        )
        start = index
    return segments


def _offsets(sample: Sample) -> tuple[int, int]:
    """The ``wall - mono`` offsets a sample allows: one, or with its second
    wall read the bracket's span."""
    mono, wall = sample[0], sample[1]
    after = sample[2] if len(sample) > 2 and sample[2] >= wall else wall
    return wall - mono, after - mono


def _narrowed(
    allowed: tuple[int, int], before: Sample, after: Sample
) -> tuple[int, int] | None:
    """The offsets still allowed after ``after``: those the segment allowed,
    widened by slew over the time since ``before``, that ``after``'s own
    interval also allows; None when there are none."""
    slew = max(SLEW_FLOOR_NS, (after[0] - before[0]) * SLEW_PPM // 1_000_000)
    low, high = _offsets(after)
    low, high = max(low, allowed[0] - slew), min(high, allowed[1] + slew)
    return (low, high) if low <= high else None


def _samples(view: RunView, producer: str) -> list[Sample]:
    """Every (mono, wall, wall_after) read the import kept for one engine:
    each step's schedule entry and completion; a read without its second
    wall read gives (mono, wall)."""
    samples: list[Sample] = []
    for ref, (_, iteration) in view.iterations.items():
        if ref.producer_id != producer:
            continue
        data = iteration.metadata
        for mono, wall, after in (
            (
                iteration.start_ns,
                data.get("start_wall_ns"),
                data.get("start_wall_after_ns"),
            ),
            (
                iteration.end_ns,
                data.get("completed_wall_ns"),
                data.get("completed_wall_after_ns"),
            ),
        ):
            if isinstance(mono, int) and isinstance(wall, int):
                samples.append(
                    (mono, wall, after) if isinstance(after, int) else (mono, wall)
                )
    return samples


def _same_host(view: RunView, producer: str) -> bool:
    """Whether the engine's wall clock is R: its host and boot are R's."""
    for _, iteration in view.iterations.values():
        if iteration.iteration_ref.producer_id == producer:
            domain = iteration.context.clock_domain
            wall = domain.removesuffix("/monotonic_ns") + "/unix_epoch_ns"
            return view.clock_domain is not None and wall == view.clock_domain
    return False


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "BASIS_CLIENT",
    "BASIS_SAME_HOST",
    "EngineClock",
    "Interval",
    "Placed",
    "Segment",
    "UNKNOWN_DISCONTINUITY",
    "UNKNOWN_LEGACY",
    "UNKNOWN_NO_STAMP",
    "UNKNOWN_OTHER_HOST",
    "continuity_segments",
]
