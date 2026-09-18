"""Versioned inference correlation records and compatibility tests."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CapabilityEvent,
    ClockAlignmentEvent,
    CorrelationContext,
    EntityRef,
    IterationEvent,
    LegacyInferenceRecord,
    MembershipEvent,
    RequestEvent,
    StageEvent,
    load_inference_artifact,
    parse_inference_record,
)
from stormlog.infer.events import JsonlEventWriter


def _context(**changes: object) -> CorrelationContext:
    values = {
        "run_id": "run-1",
        "session_id": "session-1",
        "producer_id": "engine-0",
        "source": "fake-engine",
        "source_version": "2.1",
        "engine": "fake-engine",
        "engine_version": "2.1",
        "backend": "cuda",
        "backend_version": "13.0",
        "host": "worker-a",
        "pid": 42,
        "device_uuid": "GPU-123",
        "rank": 0,
        "local_rank": 0,
        "world_size": 2,
        "clock_domain": "worker-a/monotonic",
        "clock_kind": "monotonic",
        "collection_mode": "passive",
        "provenance": "reported",
    }
    values.update(changes)
    return CorrelationContext(**values)


def test_v2_records_round_trip_through_existing_jsonl_writer(tmp_path) -> None:
    context = _context()
    request = EntityRef("client", "logical-request")
    attempt = EntityRef("client", "attempt-2")
    iteration = EntityRef("engine-0", "iteration-7")
    records = [
        RequestEvent(
            context=context,
            event_id="request-event",
            request_ref=request,
            attempt_ref=attempt,
            backend_request_ref=EntityRef("engine-0", "backend-99"),
            start_ns=100,
            end_ns=300,
            status="completed",
            input_tokens=12,
            output_tokens=3,
        ),
        IterationEvent(
            context=context,
            event_id="iteration-event",
            iteration_ref=iteration,
            batch_ref=EntityRef("engine-0", "batch-4"),
            start_ns=120,
            end_ns=220,
        ),
        StageEvent(
            context=context,
            event_id="stage-event",
            stage_ref=EntityRef("engine-0", "stage-1"),
            name="preprocessing",
            request_ref=request,
            start_ns=100,
            end_ns=120,
        ),
        MembershipEvent(
            context=context,
            event_id="membership-event",
            request_ref=request,
            attempt_ref=attempt,
            iteration_ref=iteration,
            role="prefill",
        ),
        ActivityReferenceEvent(
            context=context,
            event_id="activity-event",
            activity_ref=EntityRef("trace-0", "kernel-1"),
            iteration_ref=iteration,
            activity_kind="gpu_kernel",
            activity_domain="gpu",
            attribution_status="linked",
            trace_attachment_id="trace-attachment",
            runtime_correlation_id=7,
            cuda_correlation_id=8,
            stream_id=9,
            graph_id=10,
            start_ns=130,
            end_ns=160,
        ),
        ClockAlignmentEvent(
            context=context,
            event_id="alignment-event",
            from_clock_domain="worker-b/monotonic",
            to_clock_domain="worker-a/monotonic",
            offset_ns=-50,
            uncertainty_ns=4,
        ),
    ]

    path = tmp_path / "inference.jsonl"
    with JsonlEventWriter(path) as writer:
        for event in records:
            writer.append(event.to_record())

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/inference_correlation_v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    for event in records:
        validator.validate(event.to_record())

    assert load_inference_artifact(path) == records
    assert all(event.to_record()["schema_version"] == 2 for event in records)
    assert records[0].elapsed_ns == 200
    assert records[3].to_record()["request_ref"] == {
        "producer_id": "client",
        "id": "logical-request",
    }


def test_legacy_records_remain_readable_without_server_evidence(tmp_path) -> None:
    legacy = {
        "schema_version": 1,
        "event_type": "infer.request",
        "session_id": "session-legacy",
        "request_id": "r1",
        "started_at_ns": 100,
        "ended_at_ns": 200,
        "e2e_latency_ms": 0.1,
    }
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    [loaded] = load_inference_artifact(path)

    assert isinstance(loaded, LegacyInferenceRecord)
    assert loaded.to_record() == legacy
    assert not hasattr(loaded, "iteration_ref")


def test_reader_rejects_unsupported_versions_and_invalid_spans() -> None:
    with pytest.raises(ValueError, match="unsupported inference schema_version"):
        parse_inference_record({"schema_version": 3, "event_type": "infer.request"})
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        parse_inference_record({"schema_version": True, "event_type": "infer.request"})
    with pytest.raises(ValueError, match="end_ns must be >= start_ns"):
        IterationEvent(
            context=_context(),
            event_id="bad",
            iteration_ref=EntityRef("engine-0", "i1"),
            start_ns=20,
            end_ns=10,
        )
    with pytest.raises(ValueError, match="attribution_status"):
        ActivityReferenceEvent(
            context=_context(),
            event_id="bad-link",
            activity_ref=EntityRef("trace-0", "kernel-1"),
            activity_kind="gpu_kernel",
            attribution_status="linked",
        )


def test_non_llm_stage_and_unattributed_activity_need_no_token_fields() -> None:
    stage = StageEvent(
        context=_context(backend="metal", device_uuid=None),
        event_id="image-stage",
        stage_ref=EntityRef("engine-0", "resize"),
        name="image_resize",
        request_ref=EntityRef("client", "image-request"),
    )
    activity = ActivityReferenceEvent(
        context=_context(device_uuid=None),
        event_id="unknown-link",
        activity_ref=EntityRef("trace-0", "unresolved-kernel"),
        activity_kind="gpu_kernel",
        attribution_status="unresolved",
    )

    assert parse_inference_record(stage.to_record()) == stage
    assert parse_inference_record(activity.to_record()) == activity
    assert stage.to_record()["input_tokens"] is None
    assert activity.iteration_ref is None


def test_public_v2_schema_matches_serialized_records() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/inference_correlation_v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    artifact = ArtifactIdentityEvent(
        context=_context(),
        event_id="artifact-1",
        artifact_kind="inference_jsonl",
        created_at_ns=100,
    )
    validator.validate(artifact.to_record())
    assert parse_inference_record(artifact.to_record()) == artifact
    event = MembershipEvent(
        context=_context(),
        event_id="membership-1",
        request_ref=EntityRef("client", "r1"),
        iteration_ref=EntityRef("engine-0", "i1"),
        role="decode",
    )

    validator.validate(event.to_record())
    capabilities = CapabilityEvent(
        context=_context(),
        event_id="capabilities-1",
        component="trace_collector",
        available=True,
        supported=["gpu_activity"],
        enabled=["gpu_activity"],
        collected=[],
    )
    validator.validate(capabilities.to_record())
    assert parse_inference_record(capabilities.to_record()) == capabilities
    with pytest.raises(ValidationError):
        validator.validate({**event.to_record(), "schema_version": True})
