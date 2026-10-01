"""Arrival schedules for inference profiling workloads.

A closed loop sends the next request when a worker's previous request
finishes, so the server's speed sets the offered load. The open-loop modes
here fix when each request is meant to arrive before the run starts, so a
slow server builds a queue instead of slowing the traffic down.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLOSED = "closed"
FIXED_RATE = "fixed-rate"
POISSON = "poisson"
BURST = "burst"
REPLAY = "replay"
ARRIVAL_MODES = (CLOSED, FIXED_RATE, POISSON, BURST, REPLAY)
RATE_MODES = frozenset({FIXED_RATE, POISSON})
# A larger schedule is almost always a typo in --rate or --duration.
MAX_ARRIVALS = 1_000_000


@dataclass(frozen=True)
class ArrivalTrace:
    """Recorded arrival offsets to replay, with where they came from."""

    offsets_seconds: tuple[float, ...]
    source: str
    case_id: str | None = None

    def digest(self) -> str:
        payload = json.dumps([round(o, 9) for o in self.offsets_seconds])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArrivalSpec:
    """How the requests of one workload case arrive."""

    mode: str = CLOSED
    rate_per_second: float | None = None
    burst_size: int | None = None
    burst_interval_seconds: float | None = None
    trace: ArrivalTrace | None = None

    def __post_init__(self) -> None:
        if self.mode not in ARRIVAL_MODES:
            raise ValueError(f"arrival mode must be one of {', '.join(ARRIVAL_MODES)}")
        _require(self.mode, RATE_MODES, self.rate_per_second, "a rate")
        _require(self.mode, {BURST}, self.burst_size, "a burst size")
        _require(self.mode, {BURST}, self.burst_interval_seconds, "a burst interval")
        _require(self.mode, {REPLAY}, self.trace, "an arrival trace")
        if self.rate_per_second is not None and not _positive(self.rate_per_second):
            raise ValueError("arrival rate must be a positive number")
        if self.burst_size is not None and self.burst_size < 1:
            raise ValueError("burst size must be >= 1")
        interval = self.burst_interval_seconds
        if interval is not None and not _positive(interval):
            raise ValueError("burst interval must be a positive number")

    @property
    def open_loop(self) -> bool:
        return self.mode != CLOSED

    def case_label(self) -> str:
        """Short case-id prefix naming the arrival shape."""
        if self.mode in RATE_MODES:
            name = "fixed" if self.mode == FIXED_RATE else "poisson"
            return f"{name}{self.rate_per_second:g}"
        if self.mode == BURST:
            return f"burst{self.burst_size}x{self.burst_interval_seconds:g}s"
        return self.mode

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"mode": self.mode}
        if self.mode in RATE_MODES:
            record["rate_per_second"] = self.rate_per_second
        if self.mode == BURST:
            record["burst_size"] = self.burst_size
            record["burst_interval_seconds"] = self.burst_interval_seconds
        if self.trace is not None:
            record["trace"] = {
                "source": self.trace.source,
                "case_id": self.trace.case_id,
                "arrivals": len(self.trace.offsets_seconds),
                "digest": self.trace.digest(),
            }
        return record


def arrival_offsets(
    spec: ArrivalSpec,
    *,
    count: int | None,
    duration_seconds: float | None,
    seed: int,
) -> list[float]:
    """Return when each request should arrive, in seconds from phase start.

    The first request arrives at 0. ``count`` caps the number of arrivals and
    ``duration_seconds`` keeps only arrivals before the window closes; a
    replay without either sends the whole trace.
    """
    if spec.mode == CLOSED:
        raise ValueError("a closed loop has no arrival schedule")
    if count is None and duration_seconds is None and spec.mode != REPLAY:
        raise ValueError("an open-loop schedule needs a request count or duration")
    offsets = _unbounded_offsets(spec, seed)
    return _bounded(offsets, count=count, duration_seconds=duration_seconds)


def _unbounded_offsets(spec: ArrivalSpec, seed: int) -> Iterator[float]:
    if spec.mode == FIXED_RATE:
        return _fixed_offsets(spec)
    if spec.mode == POISSON:
        return _poisson_offsets(spec, seed)
    if spec.mode == BURST:
        return _burst_offsets(spec)
    assert spec.trace is not None
    return iter(spec.trace.offsets_seconds)


def _fixed_offsets(spec: ArrivalSpec) -> Iterator[float]:
    assert spec.rate_per_second is not None
    index = 0
    while True:
        yield index / spec.rate_per_second
        index += 1


def _poisson_offsets(spec: ArrivalSpec, seed: int) -> Iterator[float]:
    assert spec.rate_per_second is not None
    generator = random.Random(seed)
    offset = 0.0
    while True:
        yield offset
        offset += generator.expovariate(spec.rate_per_second)


def _burst_offsets(spec: ArrivalSpec) -> Iterator[float]:
    assert spec.burst_size is not None and spec.burst_interval_seconds is not None
    index = 0
    while True:
        yield index // spec.burst_size * spec.burst_interval_seconds
        index += 1


def _bounded(
    offsets: Iterator[float], *, count: int | None, duration_seconds: float | None
) -> list[float]:
    bounded: list[float] = []
    for offset in offsets:
        if count is not None and len(bounded) >= count:
            break
        if duration_seconds is not None and offset >= duration_seconds:
            break
        if len(bounded) == MAX_ARRIVALS:
            raise ValueError(
                f"the arrival schedule has more than {MAX_ARRIVALS:,} requests; "
                "check --rate, --duration, --requests and --warmup-requests"
            )
        bounded.append(offset)
    return bounded


def load_arrival_trace(path: str | Path, *, case_id: str | None = None) -> ArrivalTrace:
    """Read arrival offsets from a trace or from a Stormlog inference artifact.

    A trace has one JSON object per line with ``offset_ms``. An inference
    artifact contributes the measured requests of one case: their intended
    arrival times when recorded, otherwise the times they were sent.
    """
    records = _json_lines(Path(path))
    if any("offset_ms" in record for record in records):
        offsets = [_offset_ms(record) / 1000.0 for record in records]
        return _trace(offsets, source="offset_ms trace", case_id=None)
    arrivals = _artifact_arrivals(records)
    selected = _select_case(arrivals, case_id)
    starts = arrivals[selected]
    return _trace(
        [(start - min(starts)) / 1e9 for start in starts],
        source="stormlog artifact",
        case_id=selected,
    )


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"arrival trace line {number} is not a JSON object")
        records.append(record)
    return records


def _offset_ms(record: dict[str, Any]) -> float:
    value = record.get("offset_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("every arrival trace line needs a numeric offset_ms")
    if not math.isfinite(value) or value < 0:
        raise ValueError("offset_ms must be a finite number >= 0")
    return float(value)


def _artifact_arrivals(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    arrivals: dict[str, list[int]] = {}
    for record in records:
        if record.get("event_type") != "infer.request":
            continue
        if record.get("phase") != "measured" or record.get("status") == "dropped":
            continue
        start = record.get("intended_at_ns", record.get("started_at_ns"))
        if isinstance(start, int) and not isinstance(start, bool):
            arrivals.setdefault(str(record.get("case_id")), []).append(start)
    if not arrivals:
        raise ValueError("arrival trace has no offset_ms lines or measured requests")
    return arrivals


def _select_case(arrivals: dict[str, list[int]], case_id: str | None) -> str:
    if case_id is not None:
        if case_id not in arrivals:
            raise ValueError(f"arrival trace has no measured case {case_id!r}")
        return case_id
    if len(arrivals) > 1:
        cases = ", ".join(sorted(arrivals))
        raise ValueError(f"choose one case with --arrival-trace-case: {cases}")
    return next(iter(arrivals))


def _trace(offsets: list[float], *, source: str, case_id: str | None) -> ArrivalTrace:
    ordered = sorted(offsets)
    first = ordered[0]
    return ArrivalTrace(
        offsets_seconds=tuple(offset - first for offset in ordered),
        source=source,
        case_id=case_id,
    )


def _require(mode: str, modes: Any, value: object, what: str) -> None:
    if mode in modes and value is None:
        raise ValueError(f"{mode} arrivals need {what}")
    if mode not in modes and value is not None:
        raise ValueError(f"{mode} arrivals do not take {what}")


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0
