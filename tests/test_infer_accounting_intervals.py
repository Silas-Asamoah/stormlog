"""GPU time accounting with busy intervals inside one launch record."""

from __future__ import annotations

import pytest

from stormlog.infer.correlation_accounting import (
    account_gpu_time,
    resolve_inference_events,
)
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    CorrelationContext,
    EntityRef,
)


def _launch(intervals: object) -> list[ActivityReferenceEvent]:
    context = CorrelationContext(
        run_id="run-1",
        session_id="session-1",
        producer_id="trace",
        source="trace",
        device_uuid="GPU-1",
        clock_domain="trace-clock",
        clock_kind="device",
        collection_mode="imported",
        provenance="observed",
    )
    return [
        ActivityReferenceEvent(
            context=context,
            event_id="launch-1",
            activity_ref=EntityRef("trace", "launch-1"),
            activity_kind="gpu_kernel",
            activity_domain="gpu",
            attribution_status="unresolved",
            start_ns=100,
            end_ns=200,
            metadata={} if intervals is None else {"intervals": intervals},
        )
    ]


def test_a_launch_without_intervals_is_busy_for_its_whole_span() -> None:
    accounting = account_gpu_time(resolve_inference_events(_launch(None)))
    total = next(iter(accounting.device_totals.values()))

    assert total.busy_ns == 100


def test_launch_intervals_replace_the_span_in_gpu_time() -> None:
    accounting = account_gpu_time(
        resolve_inference_events(_launch([[0, 30], [50, 50]]))
    )
    total = next(iter(accounting.device_totals.values()))

    assert (total.busy_ns, total.summed_activity_ns, total.activity_count) == (
        80,
        80,
        1,
    )


@pytest.mark.parametrize(
    "intervals",
    [
        "0,30",
        [],
        [[0, 30], [20, 10]],
        [[0, 150]],
        [[0, -1]],
        [[0, True]],
        [[0, 1.5]],
        [[0]],
    ],
)
def test_malformed_launch_intervals_leave_the_activity_unmeasured(
    intervals: object,
) -> None:
    accounting = account_gpu_time(resolve_inference_events(_launch(intervals)))

    assert accounting.device_totals == {}
    assert accounting.unmeasured_gpu_activity_refs == (EntityRef("trace", "launch-1"),)
