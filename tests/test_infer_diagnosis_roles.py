"""Roles: an upstream competitor settled against the subject's findings."""

from __future__ import annotations

from typing import Any

from stormlog.infer.diagnosis_model import (
    NOT_RULED_OUT,
    PRIMARY,
    RULED_OUT,
    UNTESTABLE,
    UPSTREAM,
    Alternative,
    Criteria,
    Finding,
)
from stormlog.infer.diagnosis_roles import link_roles

HELD = (
    "the median request waited 900.0 ms behind the subject's preempted "
    "requests awaiting their resume, against a wait excess of 1000.0 ms"
)


def _queue(hold: Alternative) -> Finding:
    """A queue that is otherwise the fault: at capacity, explaining the
    TTFT rise, every other competitor ruled out."""
    return Finding(
        kind="queue_saturation",
        component="scheduler",
        subject={"key": "window:c1:0"},
        title="t",
        message="m",
        gates={"capacity_witness": True, "usable_timing": True},
        alternatives=[Alternative("engine_stall", RULED_OUT, "r", True), hold],
        condition=Criteria(met=("direct_evidence", "sufficient_samples")),
        contribution=Criteria(
            met=("excess_ci_excludes_zero", "explains_ttft_excess", "witness"),
            unmet=("competitors_excluded",),
        ),
        incident=True,
        explains="explains_ttft_excess",
        detail={"kv_hold": {"held_p50_ms": 900.0, "explains_ttft_excess": True}},
    )


def _kv(**changes: Any) -> Finding:
    values: dict[str, Any] = {
        "kind": "kv_preemption_pressure",
        "component": "kv_cache",
        "subject": {"key": "window:c1:0"},
        "title": "t",
        "message": "m",
        "gates": {"allocation_cause_established": True},
        "alternatives": [Alternative("prefix_cache_reset", RULED_OUT, "r", True)],
        "condition": Criteria(met=("direct_evidence",)),
        "contribution": Criteria(
            met=("excess_ci_excludes_zero",), unmet=("explains_e2e_excess",)
        ),
        "incident": True,
        "explains": "explains_e2e_excess",
    }
    values.update(changes)
    return Finding(**values)


def test_waits_behind_preempted_requests_of_unknown_cause_are_no_queue_fault() -> None:
    """A log without reset records leaves the preemptions' cause unknown, so
    the KV finding is an observation and nothing is upstream. Yet the waits
    were spent behind preempted requests, not behind a full engine: the
    queue cannot claim them as its fault."""
    hold = Alternative("kv_preemption_pressure", UPSTREAM, HELD, True)
    queue = _queue(hold)
    kv = _kv(
        gates={"allocation_cause_established": False},
        alternatives=[Alternative("prefix_cache_reset", UNTESTABLE, "r", True)],
    )

    link_roles([queue, kv], "run-1")

    (settled,) = [a for a in queue.alternatives if a.kind == hold.kind]
    assert (settled.status, settled.indispensable) == (NOT_RULED_OUT, True)
    assert queue.role == PRIMARY and not queue.eligible
    assert (queue.claim, queue.severity) == ("observation", "info")
