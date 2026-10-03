"""The victim's SLO outcomes, for the impact layer (#221 design A.6, layer 4).

Each victim request is counted in the interval it arrived in (arrival
membership: its intended arrival, else its send). A request that met the SLO
is ``met``; one that missed it, or failed (timed out, was rejected, errored,
or was never sent), is a violation; a cancelled one is ``unknown``. This is a
stand-in for #213's ``evaluate_request``, which PR C uses once #213 lands;
the rule is the same: missed, unreachable included, is a violation, and
unknown is not.
"""

from __future__ import annotations

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
    status = record.get("status")
    if status == "cancelled":
        return UNKNOWN
    if status != "ok":
        return VIOLATION
    return VIOLATION if _missed(record, slo) else MET


def _missed(record: dict[str, Any], slo: Slo) -> bool:
    for limit, value in (
        (slo.ttft_ms, record.get("ttft_ms")),
        (slo.e2e_ms, record.get("e2e_latency_ms")),
    ):
        if limit is not None and (value is None or value > limit):
            return True
    return False


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
