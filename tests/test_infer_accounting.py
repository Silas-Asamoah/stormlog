"""Shared execution accounting and incomplete evidence behavior."""

from dataclasses import replace

import pytest

from stormlog.infer.correlation_accounting import (
    DeviceClock,
    RequestShareEstimate,
    account_gpu_time,
    align_timestamp,
    resolve_inference_events,
    validate_request_shares,
)
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ClockAlignmentEvent,
    CorrelationContext,
    EntityRef,
    IterationEvent,
    LegacyInferenceRecord,
    MembershipEvent,
    RequestEvent,
    StageEvent,
)


def _context(
    producer_id: str,
    *,
    host: str = "worker-a",
    device_uuid: str | None = "GPU-123",
) -> CorrelationContext:
    return CorrelationContext(
        run_id="run-1",
        session_id=f"session-{host}",
        producer_id=producer_id,
        source=producer_id,
        host=host,
        pid=42,
        device_uuid=device_uuid,
        clock_domain=f"{host}/monotonic",
        clock_kind="monotonic",
        collection_mode="passive",
        provenance="observed",
    )


def _shared_events():
    engine = _context("engine-a")
    trace = _context("trace-a")
    request_a = EntityRef("client", "A")
    request_b = EntityRef("client", "B")
    iteration_1 = EntityRef("engine-a", "I1")
    iteration_2 = EntityRef("engine-a", "I2")
    events = [
        RequestEvent(context=engine, event_id="rA", request_ref=request_a),
        RequestEvent(context=engine, event_id="rB", request_ref=request_b),
        IterationEvent(
            context=engine,
            event_id="i1",
            iteration_ref=iteration_1,
            start_ns=0,
            end_ns=10_000_000,
        ),
        IterationEvent(
            context=engine,
            event_id="i2",
            iteration_ref=iteration_2,
            start_ns=20_000_000,
            end_ns=25_000_000,
        ),
        MembershipEvent(
            context=engine,
            event_id="mA1",
            request_ref=request_a,
            iteration_ref=iteration_1,
            role="prefill",
        ),
        MembershipEvent(
            context=engine,
            event_id="mB1",
            request_ref=request_b,
            iteration_ref=iteration_1,
            role="decode",
        ),
        MembershipEvent(
            context=engine,
            event_id="mA2",
            request_ref=request_a,
            iteration_ref=iteration_2,
            role="decode",
        ),
        ActivityReferenceEvent(
            context=trace,
            event_id="k1",
            activity_ref=EntityRef("trace-a", "K1"),
            activity_kind="kernel",
            activity_domain="gpu",
            attribution_status="linked",
            iteration_ref=iteration_1,
            start_ns=0,
            end_ns=8_000_000,
        ),
        ActivityReferenceEvent(
            context=trace,
            event_id="k2",
            activity_ref=EntityRef("trace-a", "K2"),
            activity_kind="kernel",
            activity_domain="gpu",
            attribution_status="linked",
            iteration_ref=iteration_1,
            start_ns=2_000_000,
            end_ns=10_000_000,
        ),
        ActivityReferenceEvent(
            context=trace,
            event_id="k3",
            activity_ref=EntityRef("trace-a", "K3"),
            activity_kind="kernel",
            activity_domain="gpu",
            attribution_status="linked",
            iteration_ref=iteration_2,
            start_ns=20_000_000,
            end_ns=25_000_000,
        ),
        StageEvent(
            context=engine,
            event_id="stage-A",
            stage_ref=EntityRef("engine-a", "preprocess-A"),
            name="preprocessing",
            request_ref=request_a,
        ),
    ]
    return events, request_a, request_b, iteration_1, iteration_2


def test_shared_mixed_iteration_counts_gpu_time_once_and_merges_overlap() -> None:
    events, request_a, request_b, iteration_1, iteration_2 = _shared_events()
    graph = resolve_inference_events(
        [
            *reversed(events),
            events[7],
            events[2],
            replace(events[7], event_id="k1-redelivered"),
        ]
    )
    report = account_gpu_time(graph)
    key = DeviceClock("GPU-123", "worker-a/monotonic")

    assert not graph.unresolved
    assert len(graph.memberships) == 3
    assert {
        (m.request_ref, m.role)
        for m in graph.memberships
        if m.iteration_ref == iteration_1
    } == {
        (request_a, "prefill"),
        (request_b, "decode"),
    }
    assert report.iterations[iteration_1].elapsed_ns == 10_000_000
    assert report.iterations[iteration_1].gpu[key].summed_activity_ns == 16_000_000
    assert report.iterations[iteration_1].gpu[key].busy_ns == 10_000_000
    assert report.iterations[iteration_2].gpu[key].busy_ns == 5_000_000
    assert report.device_totals[key].busy_ns == 15_000_000
    assert not hasattr(report, "request_gpu_ns")
    assert report == account_gpu_time(resolve_inference_events(events))


def test_estimated_shares_require_an_explicit_model_and_remainder() -> None:
    events, request_a, request_b, iteration_1, _ = _shared_events()
    graph = resolve_inference_events(events)
    report = account_gpu_time(graph)
    key = DeviceClock("GPU-123", "worker-a/monotonic")
    shares = (
        RequestShareEstimate(request_a, 6_000_000, "token-weighted-v1"),
        RequestShareEstimate(request_b, 3_000_000, "token-weighted-v1"),
    )

    budget = validate_request_shares(
        graph,
        report,
        iteration_ref=iteration_1,
        device_clock=key,
        shares=shares,
        unattributed_ns=1_000_000,
    )
    assert budget.budget_ns == 10_000_000
    assert budget.unattributed_ns == 1_000_000
    with pytest.raises(ValueError, match="equal the measured budget"):
        validate_request_shares(
            graph,
            report,
            iteration_ref=iteration_1,
            device_clock=key,
            shares=shares,
            unattributed_ns=0,
        )
    with pytest.raises(ValueError, match="not a member"):
        validate_request_shares(
            graph,
            report,
            iteration_ref=iteration_1,
            device_clock=key,
            shares=(RequestShareEstimate(EntityRef("client", "C"), 10_000_000, "m"),),
            unattributed_ns=0,
        )
    with pytest.raises(ValueError, match="unattributed_ns must be non-negative"):
        validate_request_shares(
            graph,
            report,
            iteration_ref=iteration_1,
            device_clock=key,
            shares=shares,
            unattributed_ns=1_000_000.0,
        )
    with pytest.raises(ValueError, match="one estimation model"):
        validate_request_shares(
            graph,
            report,
            iteration_ref=iteration_1,
            device_clock=key,
            shares=(shares[0], replace(shares[1], model="equal-split-v1")),
            unattributed_ns=1_000_000,
        )


def test_generic_and_attempt_specific_shares_cannot_overlap() -> None:
    events, request_a, _request_b, iteration_1, _ = _shared_events()
    context = next(event.context for event in events if isinstance(event, RequestEvent))
    attempt = EntityRef("client", "attempt-A")
    events.extend(
        [
            RequestEvent(
                context=context,
                event_id="rA-attempt",
                request_ref=request_a,
                attempt_ref=attempt,
            ),
            MembershipEvent(
                context=context,
                event_id="mA-attempt",
                request_ref=request_a,
                attempt_ref=attempt,
                iteration_ref=iteration_1,
                role="decode",
            ),
        ]
    )
    graph = resolve_inference_events(events)
    report = account_gpu_time(graph)
    key = DeviceClock("GPU-123", "worker-a/monotonic")

    with pytest.raises(ValueError, match="overlap"):
        validate_request_shares(
            graph,
            report,
            iteration_ref=iteration_1,
            device_clock=key,
            shares=(
                RequestShareEstimate(request_a, 5_000_000, "model"),
                RequestShareEstimate(
                    request_a, 5_000_000, "model", attempt_ref=attempt
                ),
            ),
            unattributed_ns=0,
        )


def test_distinct_attempt_specific_shares_remain_valid() -> None:
    events, request_a, _request_b, iteration_1, _ = _shared_events()
    context = next(event.context for event in events if isinstance(event, RequestEvent))
    attempt_a = EntityRef("client", "attempt-A")
    attempt_b = EntityRef("client", "attempt-B")
    events.extend(
        [
            RequestEvent(
                context=context,
                event_id="rA-attempt-a",
                request_ref=request_a,
                attempt_ref=attempt_a,
            ),
            RequestEvent(
                context=context,
                event_id="rA-attempt-b",
                request_ref=request_a,
                attempt_ref=attempt_b,
            ),
            MembershipEvent(
                context=context,
                event_id="mA-attempt-a",
                request_ref=request_a,
                attempt_ref=attempt_a,
                iteration_ref=iteration_1,
                role="decode",
            ),
            MembershipEvent(
                context=context,
                event_id="mA-attempt-b",
                request_ref=request_a,
                attempt_ref=attempt_b,
                iteration_ref=iteration_1,
                role="decode",
            ),
        ]
    )
    graph = resolve_inference_events(events)
    report = account_gpu_time(graph)
    key = DeviceClock("GPU-123", "worker-a/monotonic")

    budget = validate_request_shares(
        graph,
        report,
        iteration_ref=iteration_1,
        device_clock=key,
        shares=(
            RequestShareEstimate(request_a, 4_000_000, "model", attempt_ref=attempt_a),
            RequestShareEstimate(request_a, 6_000_000, "model", attempt_ref=attempt_b),
        ),
        unattributed_ns=0,
    )
    assert budget.budget_ns == 10_000_000


def test_clock_alignment_carries_cross_host_uncertainty() -> None:
    alignment = ClockAlignmentEvent(
        context=_context("sync", host="worker-a", device_uuid=None),
        event_id="alignment",
        from_clock_domain="worker-b/monotonic",
        to_clock_domain="worker-a/monotonic",
        offset_ns=-50,
        uncertainty_ns=4,
        valid_from_ns=100,
        valid_to_ns=200,
    )
    graph = resolve_inference_events([alignment])

    result = align_timestamp(
        150,
        from_clock_domain="worker-b/monotonic",
        to_clock_domain="worker-a/monotonic",
        alignments=graph.alignments,
    )
    assert (result.value_ns, result.uncertainty_ns) == (100, 4)
    with pytest.raises(ValueError, match="no valid clock alignment"):
        align_timestamp(
            150,
            from_clock_domain="worker-b/monotonic",
            to_clock_domain="worker-a/monotonic",
            alignments=(),
        )
    with pytest.raises(ValueError, match="no valid clock alignment"):
        align_timestamp(
            250,
            from_clock_domain="worker-b/monotonic",
            to_clock_domain="worker-a/monotonic",
            alignments=graph.alignments,
        )


def test_partial_evidence_and_legacy_records_do_not_create_fake_gpu_time() -> None:
    events, *_ = _shared_events()
    unknown = ActivityReferenceEvent(
        context=_context("trace-b", host="worker-b", device_uuid=None),
        event_id="unknown",
        activity_ref=EntityRef("trace-b", "K9"),
        activity_kind="kernel",
        activity_domain="gpu",
        attribution_status="unresolved",
        start_ns=0,
        end_ns=10,
    )
    missing = MembershipEvent(
        context=_context("engine-a"),
        event_id="missing",
        request_ref=EntityRef("client", "unknown"),
        iteration_ref=EntityRef("engine-a", "unknown"),
        role="decode",
    )
    graph = resolve_inference_events(
        [
            LegacyInferenceRecord({"schema_version": 1, "event_type": "infer.request"}),
            unknown,
            missing,
        ]
    )
    report = account_gpu_time(graph)

    assert graph.unresolved
    assert report.device_totals == {}
    assert report.unattributed_activity_refs == (unknown.activity_ref,)
    assert report.unmeasured_gpu_activity_refs == (unknown.activity_ref,)
    assert (
        resolve_inference_events(
            [
                LegacyInferenceRecord(
                    {"schema_version": 1, "event_type": "infer.request"}
                )
            ]
        ).run_id
        is None
    )
    assert account_gpu_time(resolve_inference_events(events)).device_totals


def test_conflicting_duplicate_identity_is_rejected() -> None:
    events, *_ = _shared_events()
    changed = replace(events[2], end_ns=11_000_000)
    with pytest.raises(ValueError, match="conflicting event identity"):
        resolve_inference_events([events[2], changed])


def test_device_totals_keep_hosts_and_clock_domains_separate() -> None:
    events, *_ = _shared_events()
    other_host_activity = ActivityReferenceEvent(
        context=_context("trace-b", host="worker-b", device_uuid="GPU-456"),
        event_id="worker-b-kernel",
        activity_ref=EntityRef("trace-b", "K1"),
        activity_kind="kernel",
        activity_domain="gpu",
        attribution_status="unresolved",
        start_ns=0,
        end_ns=10_000_000,
    )

    report = account_gpu_time(resolve_inference_events([*events, other_host_activity]))

    assert (
        report.device_totals[DeviceClock("GPU-123", "worker-a/monotonic")].busy_ns
        == 15_000_000
    )
    assert (
        report.device_totals[DeviceClock("GPU-456", "worker-b/monotonic")].busy_ns
        == 10_000_000
    )


def test_device_and_monotonic_clock_kinds_do_not_merge() -> None:
    events, *_ = _shared_events()
    device_activity = replace(
        events[7],
        context=replace(events[7].context, clock_kind="device"),
        activity_ref=EntityRef("trace-a", "K-device"),
        event_id="device-clock-kernel",
    )
    report = account_gpu_time(resolve_inference_events([*events, device_activity]))

    assert (
        report.device_totals[DeviceClock("GPU-123", "worker-a/monotonic")].busy_ns
        == 15_000_000
    )
    assert (
        report.device_totals[
            DeviceClock("GPU-123", "worker-a/monotonic", "device")
        ].busy_ns
        == 8_000_000
    )
