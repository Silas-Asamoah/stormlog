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

DEFAULT_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        # Requests waiting in the median scrape of the window.
        QUEUE_MEDIAN_WAITING: 1.0,
        # Preemptions counted in the window: any preemption is evidence.
        KV_PREEMPTIONS: 1.0,
        # Fall of the prefix-cache hit ratio below the caller's reference.
        PREFIX_HIT_RATIO_DROP: 0.2,
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
    "PREFIX_HIT_RATIO_DROP",
    "QUEUE_MEDIAN_WAITING",
    "THRESHOLDS_VERSION",
    "resolve_threshold",
]
