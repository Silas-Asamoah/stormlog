"""The versioned threshold table shared by online triggers and the diagnoser.

A trigger and a diagnosis read the same value under the same key, so they can
never disagree about what "saturated" means. Every value here is provisional
until it has been read from real runs; a change to any value is a new table
version, and every result records the version it used and whether a caller
overrode a value.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

THRESHOLDS_VERSION = "diagnosis_thresholds_v1"

QUEUE_MEDIAN_WAITING = "queue_saturation.median_waiting_requests"
KV_PREEMPTIONS = "kv_preemption_pressure.preemptions"
PREFIX_HIT_RATIO_DROP = "prefix_cache_loss.hit_ratio_drop"
LOOP_STALL_FACTOR = "host_stall.stall_factor"
LOOP_STALL_FLOOR_NS = "host_stall.stall_floor_ns"
LOOP_NO_BASELINE_FLOOR_NS = "host_stall.no_baseline_floor_ns"
LOOP_BASELINE_WINDOW_NS = "host_stall.baseline_window_ns"
LOOP_MIN_BUSY_STEPS = "host_stall.min_busy_steps"
LOOP_MATCHED_BIN_MIN_STEPS = "host_stall.matched_bin_min_steps"
LOOP_HEARTBEAT_GRACE_NS = "host_stall.heartbeat_grace_ns"

DEFAULT_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        # Requests waiting in the median scrape of the window.
        QUEUE_MEDIAN_WAITING: 1.0,
        # Preemptions counted in the window: any preemption is evidence.
        KV_PREEMPTIONS: 1.0,
        # Fall of the prefix-cache hit ratio below the caller's reference.
        PREFIX_HIT_RATIO_DROP: 0.2,
        # An engine-loop stall is at least this many times the median
        # completion cadence of the busy steps before it...
        LOOP_STALL_FACTOR: 10.0,
        # ...and at least this long,
        LOOP_STALL_FLOOR_NS: 50_000_000.0,
        # or, with no such steps to compare with, at least this long.
        LOOP_NO_BASELINE_FLOOR_NS: 500_000_000.0,
        # The busy steps compared with lie in this window before the stall,
        LOOP_BASELINE_WINDOW_NS: 30_000_000_000.0,
        # and there must be this many of them,
        LOOP_MIN_BUSY_STEPS: 20.0,
        # or this many of the stall's own work bucket (scheduled tokens within
        # a factor of two) to compare it with steps of its size.
        LOOP_MATCHED_BIN_MIN_STEPS: 20.0,
        # A stall still going on is judged only while the hook's writer was
        # heard from this recently: three of its one-second heartbeats, since
        # under load they slip (2.3 s apart on a real vLLM 0.30.0 run).
        LOOP_HEARTBEAT_GRACE_NS: 3_000_000_000.0,
    }
)


def resolve_threshold(
    key: str, overrides: Mapping[str, float] | None = None
) -> tuple[float, bool]:
    """The threshold for ``key`` and whether a caller overrode the table.

    Raises:
        KeyError: for a key the table does not have.
    """
    if overrides and key in overrides:
        return float(overrides[key]), True
    return DEFAULT_THRESHOLDS[key], False


__all__ = [
    "DEFAULT_THRESHOLDS",
    "KV_PREEMPTIONS",
    "LOOP_BASELINE_WINDOW_NS",
    "LOOP_HEARTBEAT_GRACE_NS",
    "LOOP_MATCHED_BIN_MIN_STEPS",
    "LOOP_MIN_BUSY_STEPS",
    "LOOP_NO_BASELINE_FLOOR_NS",
    "LOOP_STALL_FACTOR",
    "LOOP_STALL_FLOOR_NS",
    "PREFIX_HIT_RATIO_DROP",
    "QUEUE_MEDIAN_WAITING",
    "THRESHOLDS_VERSION",
    "resolve_threshold",
]
