"""Per-case arrival accounting for inference reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .report_stats import int_value, is_number, number_values, percentile

# Statuses of requests that were never sent.
UNSENT_STATUSES = frozenset({"dropped"})


def arrival_summary(
    requests: list[dict[str, Any]], window: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Count what was offered, sent and completed, and how late requests left.

    ``requests`` are all the measured requests of one case, whatever their
    outcome, and ``window`` is the case's ``infer.case_window`` record when
    the artifact has one. A closed loop has no offered rate: the server's
    speed sets it.
    """
    statuses = Counter(str(record.get("status")) for record in requests)
    unsent = sum(statuses[status] for status in UNSENT_STATUSES)
    mode = _arrival_mode(requests)
    return {
        "mode": mode,
        "offered": len(requests),
        "sent": len(requests) - unsent,
        "completed": statuses["ok"],
        "dropped": statuses["dropped"],
        "failed": {
            status: count
            for status, count in sorted(statuses.items())
            if status != "ok" and status not in UNSENT_STATUSES
        },
        "held_for_slot": sum(
            1
            for record in requests
            if record.get("held_for_slot")
            and record.get("status") not in UNSENT_STATUSES
        ),
        "peak_in_flight": _peak(requests, "in_flight_at_dispatch"),
        "offered_rate_per_second": (
            None if mode == "closed" else _offered_rate(requests)
        ),
        "dispatch_lag_ms": _spread(number_values(requests, "dispatch_lag_ms")),
        **_window_seconds(window),
    }


def latency_from_intended_ms(requests: list[dict[str, Any]]) -> list[float]:
    """End-to-end latency measured from when each request was due.

    It includes any time a request waited before it was sent, which a
    latency measured from the send leaves out.
    """
    values: list[float] = []
    for record in requests:
        intended, ended = record.get("intended_at_ns"), record.get("ended_at_ns")
        if is_number(intended) and is_number(ended):
            values.append((int_value(ended) - int_value(intended)) / 1_000_000.0)
    return values


def arrival_lines(arrivals: Any) -> list[str]:
    """One text-report line for a case whose arrivals are worth showing."""
    if not isinstance(arrivals, dict):
        return []
    eventful = arrivals.get("dropped") or arrivals.get("failed")
    if arrivals.get("mode") == "closed" and not eventful:
        return []
    parts = [
        str(arrivals.get("mode")),
        f"offered {arrivals.get('offered')}",
        f"sent {arrivals.get('sent')}",
        *_outcome_counts(arrivals),
        f"peak in flight {arrivals.get('peak_in_flight')}",
    ]
    lag = (arrivals.get("dispatch_lag_ms") or {}).get("p95")
    if is_number(lag):
        parts.append(f"dispatch lag p95 {lag:.2f} ms")
    return ["  arrivals: " + ", ".join(parts)]


def _outcome_counts(arrivals: dict[str, Any]) -> list[str]:
    counts = {
        "dropped": arrivals.get("dropped"),
        "held for a slot": arrivals.get("held_for_slot"),
        **(arrivals.get("failed") or {}),
    }
    return [f"{label} {count}" for label, count in counts.items() if count]


def _window_seconds(window: dict[str, Any] | None) -> dict[str, float | None]:
    """How long arrivals ran, and how long the requests after them took."""
    bounds = [
        (window or {}).get(field)
        for field in ("started_at_ns", "window_ended_at_ns", "drained_at_ns")
    ]
    if not all(is_number(bound) for bound in bounds):
        return {"window_seconds": None, "drain_seconds": None}
    started, ended, drained = (int_value(bound) for bound in bounds)
    return {
        "window_seconds": max(ended - started, 0) / 1e9,
        "drain_seconds": max(drained - ended, 0) / 1e9,
    }


def _arrival_mode(requests: list[dict[str, Any]]) -> str:
    # Artifacts written before arrival modes existed were closed loops.
    modes = {str(record.get("arrival_mode", "closed")) for record in requests}
    return modes.pop() if len(modes) == 1 else "mixed"


def _peak(requests: list[dict[str, Any]], field: str) -> int | None:
    return max(
        (
            int_value(record[field])
            for record in requests
            if is_number(record.get(field))
        ),
        default=None,
    )


def _offered_rate(requests: list[dict[str, Any]]) -> float | None:
    intended = sorted(
        int_value(record["intended_at_ns"])
        for record in requests
        if is_number(record.get("intended_at_ns"))
    )
    if len(intended) < 2 or intended[-1] == intended[0]:
        return None
    return (len(intended) - 1) / ((intended[-1] - intended[0]) / 1e9)


def _spread(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values, default=None),
    }
