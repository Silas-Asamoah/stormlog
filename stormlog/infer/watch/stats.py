"""The watcher's own health, as gauges and counters for an exporter.

:class:`WatchStats` holds every figure in memory under one lock, so
:meth:`WatchStats.health` is a pure, cheap read: it copies, resets nothing,
and never touches the disk. :meth:`WatchStats.health_metrics` describes each
figure the way #220's health sources declare theirs: a family with labels is
a mapping from a tuple of label values to a number. Label values come only
from the watch config or from closed vocabularies, so series stay bounded.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .records import (
    CAPTURE_STATUSES,
    DETAIL_LEVELS,
    FIDELITY,
    PRUNE_REASONS,
    SUPPRESSION_REASONS,
    TRIGGER_KINDS,
    TRIGGER_STATES,
)

GAUGE = "gauge"
COUNTER = "counter"
STATE = "state"
_PREFIX = "stormlog_watch_"


@dataclass(frozen=True)
class MetricDescriptor:
    """One figure of :meth:`WatchStats.health`, field for field as #220's
    ``HealthMetric``: its snapshot key, metric name, kind, unit, help, the
    label names of a keyed family, a state metric's closed states, and the
    closed values of each label that has them. A label left out of
    ``enums``, such as ``trigger_id``, is bounded by the watch config."""

    key: str
    name: str
    kind: str
    unit: str
    help: str
    labels: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    enums: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _metric(
    key: str,
    kind: str,
    unit: str,
    help_text: str,
    labels: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    enums: dict[str, tuple[str, ...]] | None = None,
) -> MetricDescriptor:
    return MetricDescriptor(
        key, _PREFIX + key, kind, unit, help_text, labels, states, enums or {}
    )


# Closed label values not already named in the record vocabularies.
EVICTION_CAUSES = ("age", "bytes")
SCRAPE_OUTCOMES = ("ok", "failed", "oversized")
WINDOWS = ("pre", "post", "deep")
NO_DETAIL = "none"
SINKS = ("ledger", "export")


DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    _metric("history_bytes", GAUGE, "bytes", "Compressed scrape bytes held."),
    _metric("history_capacity_bytes", GAUGE, "bytes", "Scrape history byte bound."),
    _metric("history_seconds", GAUGE, "seconds", "Span of scrapes held."),
    _metric("history_capacity_seconds", GAUGE, "seconds", "History age bound."),
    _metric("captures_active", GAUGE, "", "Profiler windows open now (0 or 1)."),
    _metric("cooldown_remaining_seconds", GAUGE, "seconds", "Until deep capture."),
    _metric("retention_bytes", GAUGE, "bytes", "Bytes the incident store holds."),
    _metric("retention_incidents", GAUGE, "", "Bundles the incident store holds."),
    _metric("loop_lag_seconds_max", GAUGE, "seconds", "Worst tick lateness."),
    _metric(
        "trigger_state",
        STATE,
        "",
        "Each trigger's state.",
        labels=("trigger_id",),
        states=TRIGGER_STATES,
    ),
    _metric(
        "history_evictions_total",
        COUNTER,
        "",
        "Scrapes evicted from history, by cause; one too large for the "
        "whole history is counted in scrapes_total as oversized.",
        labels=("cause",),
        enums={"cause": EVICTION_CAUSES},
    ),
    _metric(
        "scrapes_total",
        COUNTER,
        "",
        "Scrapes by outcome.",
        labels=("outcome",),
        enums={"outcome": SCRAPE_OUTCOMES},
    ),
    _metric("ticks_missed_total", COUNTER, "", "Ticks skipped behind a slow scrape."),
    _metric("frozen_ticks_total", COUNTER, "", "Ticks with a frozen exporter."),
    _metric(
        "incidents_total",
        COUNTER,
        "",
        "Incidents by trigger kind and capture status.",
        labels=("trigger_kind", "capture_status"),
        enums={"trigger_kind": TRIGGER_KINDS, "capture_status": CAPTURE_STATUSES},
    ),
    _metric(
        "incident_windows_total",
        COUNTER,
        "",
        "Incident windows by fidelity and detail collected.",
        labels=("window", "fidelity", "detail_collected"),
        enums={
            "window": WINDOWS,
            "fidelity": FIDELITY,
            "detail_collected": (*DETAIL_LEVELS, NO_DETAIL),
        },
    ),
    _metric(
        "suppressed_total",
        COUNTER,
        "",
        "Firings that opened no incident or capture, by reason.",
        labels=("reason",),
        enums={"reason": SUPPRESSION_REASONS},
    ),
    _metric(
        "pruned_total",
        COUNTER,
        "",
        "Bundles removed, by reason: retention, or room for a new incident.",
        labels=("reason",),
        enums={"reason": PRUNE_REASONS},
    ),
    _metric(
        "pruned_bytes_total",
        COUNTER,
        "bytes",
        "Bytes of the bundles removed, by reason.",
        labels=("reason",),
        enums={"reason": PRUNE_REASONS},
    ),
    _metric(
        "sink_dropped_total",
        COUNTER,
        "",
        "Records a sink could not take, by sink.",
        labels=("sink",),
        enums={"sink": SINKS},
    ),
)

_COUNTERS = {d.key for d in DESCRIPTORS if d.kind == COUNTER and not d.labels}
_LABELLED = {d.key: len(d.labels) for d in DESCRIPTORS if d.labels}
# Each labelled family's closed values by label position; None where open.
_ENUMS = {
    d.key: tuple(d.enums.get(label) for label in d.labels)
    for d in DESCRIPTORS
    if d.labels
}
_GAUGES = {d.key for d in DESCRIPTORS if d.kind == GAUGE}


class WatchStats:
    """Counters and gauges the watcher updates and an exporter reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scalars: dict[str, float] = {}
        self._families: dict[str, Counter[tuple[str, ...]]] = {
            key: Counter() for key in _LABELLED
        }
        self._states: dict[tuple[str, ...], str] = {}

    def add(self, key: str, amount: float = 1, labels: tuple[str, ...] = ()) -> None:
        """Increase a counter, or one series of a labelled counter."""
        with self._lock:
            if key in _LABELLED:
                self._check_labels(key, labels)
                self._families[key][labels] += int(amount)
            elif key in _COUNTERS:
                self._scalars[key] = self._scalars.get(key, 0) + amount
            else:
                raise KeyError(f"{key} is not a counter")

    def set(self, key: str, value: float | None) -> None:
        """Set a gauge; None omits it from the snapshot."""
        if key not in _GAUGES:
            raise KeyError(f"{key} is not a gauge")
        with self._lock:
            if value is None:
                self._scalars.pop(key, None)
            else:
                self._scalars[key] = value

    def set_trigger_state(self, trigger_id: str, state: str) -> None:
        if state not in TRIGGER_STATES:
            raise ValueError(f"unknown trigger state {state!r}")
        with self._lock:
            self._states[(trigger_id,)] = state

    def health(self) -> dict[str, Any]:
        """A copy of every figure; resets nothing."""
        with self._lock:
            snapshot: dict[str, Any] = dict(self._scalars)
            for key, family in self._families.items():
                snapshot[key] = dict(family)
            snapshot["trigger_state"] = dict(self._states)
            return snapshot

    def health_metrics(self) -> tuple[MetricDescriptor, ...]:
        return DESCRIPTORS

    @staticmethod
    def _check_labels(key: str, labels: tuple[str, ...]) -> None:
        if len(labels) != _LABELLED[key] or not all(labels):
            raise ValueError(f"{key} takes {_LABELLED[key]} non-empty label values")
        for value, allowed in zip(labels, _ENUMS[key]):
            if allowed is not None and value not in allowed:
                raise ValueError(f"{key}: {value!r} is not one of {allowed}")


def counter_value(
    snapshot: Mapping[str, Any], key: str, labels: tuple[str, ...] = ()
) -> float:
    """Read one counter or labelled series from a snapshot; 0 when absent."""
    value = snapshot.get(key, 0)
    if isinstance(value, Mapping):
        return float(value.get(labels, 0))
    return float(value)


__all__ = [
    "DESCRIPTORS",
    "EVICTION_CAUSES",
    "NO_DETAIL",
    "SCRAPE_OUTCOMES",
    "SINKS",
    "WINDOWS",
    "MetricDescriptor",
    "WatchStats",
    "counter_value",
]
