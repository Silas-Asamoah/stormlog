"""The victim's SLO outcomes, for the impact layer (#221 design A.6, layer 4).

Each victim request is counted in the interval it arrived in (arrival
membership: its intended arrival, else its send). This is a stand-in for
#213's ``evaluate_request`` on the client criteria, which PR C uses once #213
lands, and the rule is the same:

- a request that did not succeed (timed out, rejected, errored, never sent,
  cancelled) is a violation, whatever its latencies;
- a successful one is a violation if any criterion fails (above its limit),
  else ``unknown`` if any criterion's value is missing, not finite or
  negative (no latency can be), else ``met``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from stormlog.infer.qualify.ground_truth import OutcomeCounts

MET = "met"
VIOLATION = "violation"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Slo:
    """The client-boundary criteria (#221's ``SLO_DX``): TTFT and e2e."""

    ttft_ms: float | None = None
    e2e_ms: float | None = None

    @property
    def defined(self) -> bool:
        return self.ttft_ms is not None or self.e2e_ms is not None


def outcome(record: dict[str, Any], slo: Slo) -> str:
    if record.get("status") != "ok":
        return VIOLATION
    verdicts = {
        _judge(value, limit)
        for limit, value in (
            (slo.ttft_ms, record.get("ttft_ms")),
            (slo.e2e_ms, record.get("e2e_latency_ms")),
        )
        if limit is not None
    }
    if VIOLATION in verdicts:
        return VIOLATION
    return UNKNOWN if UNKNOWN in verdicts else MET


def _judge(value: Any, limit: float) -> str:
    """One criterion: unknown without a value a latency can have."""
    real = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not real or not math.isfinite(value) or value < 0:
        return UNKNOWN
    return MET if value <= limit else VIOLATION


def arrived_at(record: dict[str, Any]) -> int:
    return int(record.get("intended_at_ns") or record["started_at_ns"])


def count_outcomes(
    records: Iterable[dict[str, Any]], start_ns: int, end_ns: int, slo: Slo
) -> OutcomeCounts:
    """The victim's outcomes for requests that arrived in [start_ns, end_ns]."""
    tally = {MET: 0, VIOLATION: 0, UNKNOWN: 0}
    for record in records:
        if record.get("event_type") != "infer.request":
            continue
        if start_ns <= arrived_at(record) <= end_ns:
            tally[outcome(record, slo)] += 1
    return OutcomeCounts(
        violations=tally[VIOLATION], met=tally[MET], unknown=tally[UNKNOWN]
    )


__all__ = [
    "MET",
    "UNKNOWN",
    "VIOLATION",
    "Slo",
    "arrived_at",
    "count_outcomes",
    "outcome",
]
