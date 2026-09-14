"""Regression tests for GPU tracker timeline aggregation."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any, cast

import pytest

from stormlog.tracker import MemoryTracker, TrackingEvent


def _make_event(timestamp: float, allocated: int) -> TrackingEvent:
    return TrackingEvent(
        timestamp=timestamp,
        event_type="sample",
        memory_allocated=allocated,
        memory_reserved=allocated * 2,
        memory_change=0,
        device_id=0,
    )


def _make_tracker(events: Sequence[object]) -> MemoryTracker:
    tracker = object.__new__(MemoryTracker)
    tracker.events = cast(Any, deque(events))
    tracker._events_lock = threading.Lock()
    return tracker


class _CountingEvent:
    def __init__(self, timestamp: float, timestamp_reads: list[int]) -> None:
        self._timestamp = timestamp
        self._timestamp_reads = timestamp_reads
        self.memory_allocated = int(timestamp)
        self.memory_reserved = int(timestamp) * 2

    @property
    def timestamp(self) -> float:
        self._timestamp_reads[0] += 1
        return self._timestamp


def test_memory_timeline_uses_last_event_from_each_nonempty_bucket() -> None:
    tracker = _make_tracker(
        [
            _make_event(0.0, 10),
            _make_event(0.4, 20),
            _make_event(1.1, 30),
            _make_event(3.2, 40),
        ]
    )

    timeline = tracker.get_memory_timeline(interval=1.0)

    assert timeline == {
        "timestamps": [0.0, 1.0, 3.0],
        "allocated": [20, 30, 40],
        "reserved": [40, 60, 80],
        "device_used": [None, None, None],
        "device_free": [None, None, None],
        "device_total": [None, None, None],
    }


def test_memory_timeline_reads_each_timestamp_once() -> None:
    timestamp_reads = [0]
    events = [_CountingEvent(float(index), timestamp_reads) for index in range(25)]
    tracker = _make_tracker(events)

    tracker.get_memory_timeline(interval=1.0)

    assert timestamp_reads[0] <= len(events) + 1


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_memory_timeline_rejects_non_positive_interval(interval: float) -> None:
    tracker = _make_tracker([_make_event(0.0, 10)])

    with pytest.raises(ValueError, match="interval must be > 0"):
        tracker.get_memory_timeline(interval=interval)
