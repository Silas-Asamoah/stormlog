"""Optional engine and trace capture integration with the run catalog."""

import json
from dataclasses import replace

import pytest

from stormlog import query as query_api
from stormlog.infer.correlation_capture import (
    CaptureCapabilities,
    EngineCapture,
    TraceAttachment,
    TraceCapture,
    append_inference_capture,
    link_trace_activities,
)
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    CapabilityEvent,
    CorrelationContext,
    EntityRef,
    IterationEvent,
    LegacyInferenceRecord,
    MembershipEvent,
    load_inference_artifact,
)
from stormlog.session import create_session_summary


def _context(producer_id: str, *, run_id: str = "run-1") -> CorrelationContext:
    return CorrelationContext(
        run_id=run_id,
        session_id="session-1",
        producer_id=producer_id,
        source=producer_id,
        host="worker-a",
        pid=42,
        device_uuid="GPU-123",
        clock_domain="worker-a/monotonic",
        clock_kind="monotonic",
        collection_mode="passive",
        provenance="observed",
    )


class _Engine:
    def collect(self, *, run_id: str, session_id: str) -> EngineCapture:
        assert (run_id, session_id) == ("run-1", "session-1")
        context = _context("engine-0")
        iteration = EntityRef("engine-0", "iteration-7")
        scope = EntityRef("runtime-0", "cuda-context-1")
        return EngineCapture(
            capabilities=CaptureCapabilities(
                supported=("iterations", "membership", "runtime_correlation"),
                enabled=("iterations", "membership", "runtime_correlation"),
                collected=("iterations", "membership", "runtime_correlation"),
            ),
            events=(
                IterationEvent(
                    context=context,
                    event_id="iteration-7",
                    iteration_ref=iteration,
                    start_ns=100,
                    end_ns=200,
                ),
                MembershipEvent(
                    context=context,
                    event_id="membership-7",
                    request_ref=EntityRef("client", "r1"),
                    iteration_ref=iteration,
                    role="prefill",
                ),
                ActivityReferenceEvent(
                    context=context,
                    event_id="launch-7",
                    activity_ref=EntityRef("engine-0", "launch-7"),
                    iteration_ref=iteration,
                    activity_kind="runtime_launch",
                    attribution_status="linked",
                    correlation_scope=scope,
                    runtime_correlation_id=17,
                ),
            ),
        )


class _Trace:
    def __init__(self, trace_path) -> None:
        self.trace_path = trace_path

    def collect(self, *, run_id: str, session_id: str) -> TraceCapture:
        assert (run_id, session_id) == ("run-1", "session-1")
        return TraceCapture(
            capabilities=CaptureCapabilities(
                supported=("gpu_activity", "runtime_correlation"),
                enabled=("gpu_activity", "runtime_correlation"),
                collected=("gpu_activity", "runtime_correlation"),
            ),
            events=(
                ActivityReferenceEvent(
                    context=_context("trace-0"),
                    event_id="kernel-1",
                    activity_ref=EntityRef("trace-0", "kernel-1"),
                    activity_kind="gpu_kernel",
                    attribution_status="unresolved",
                    correlation_scope=EntityRef("runtime-0", "cuda-context-1"),
                    runtime_correlation_id=17,
                    trace_attachment_id="trace-1",
                    start_ns=130,
                    end_ns=160,
                ),
            ),
            attachments=(
                TraceAttachment(
                    attachment_id="trace-1",
                    title="Worker CUDA trace",
                    path=self.trace_path,
                    storage="copy",
                ),
            ),
        )


def _legacy_artifact(path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "infer.request",
                "session_id": "session-1",
                "request_id": "r1",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_optional_capture_joins_scoped_ids_and_indexes_trace(tmp_path) -> None:
    artifact = tmp_path / "infer.jsonl"
    _legacy_artifact(artifact)
    trace_path = tmp_path / "worker.trace"
    trace_path.write_text("raw trace", encoding="utf-8")
    session = create_session_summary(
        source="stormlog.infer.profile",
        session_id="session-1",
        host="worker-a",
        pid=42,
    )

    append_inference_capture(
        artifact,
        run_id="run-1",
        session=session,
        engine_adapter=_Engine(),
        trace_collector=_Trace(trace_path),
        envelope_path=tmp_path / "stormlog_run.json",
    )

    records = load_inference_artifact(artifact)
    assert isinstance(records[0], LegacyInferenceRecord)
    kernel = next(
        record
        for record in records
        if isinstance(record, ActivityReferenceEvent)
        and record.activity_kind == "gpu_kernel"
    )
    assert kernel.attribution_status == "linked"
    assert kernel.iteration_ref == EntityRef("engine-0", "iteration-7")
    assert [
        record.component for record in records if isinstance(record, CapabilityEvent)
    ] == [
        "engine_adapter",
        "trace_collector",
    ]
    store = query_api.open([tmp_path])
    attachments = store.list_run_attachments(
        query_api.RunAttachmentFilter(run_id="run-1")
    )
    assert {(row.attachment_id, row.kind) for row in attachments} >= {
        ("trace-1", "profiler_trace"),
        ("infer-jsonl:session-1:infer.jsonl", "inference_jsonl"),
    }
    append_inference_capture(
        artifact,
        run_id="run-1",
        session=session,
        envelope_path=tmp_path / "stormlog_run.json",
    )
    envelope = json.loads((tmp_path / "stormlog_run.json").read_text())
    assert len(envelope["attachments"]) == 2


def test_missing_adapters_are_recorded_without_fake_server_events(tmp_path) -> None:
    artifact = tmp_path / "infer.jsonl"
    _legacy_artifact(artifact)
    session = create_session_summary(source="test", session_id="session-1")

    append_inference_capture(
        artifact,
        run_id="run-1",
        session=session,
        envelope_path=tmp_path / "stormlog_run.json",
    )

    records = load_inference_artifact(artifact)
    capabilities = [r for r in records if isinstance(r, CapabilityEvent)]
    assert len(capabilities) == 2
    assert all(not r.available and not r.collected for r in capabilities)
    assert not any(isinstance(r, IterationEvent) for r in records)


def test_rejects_wrong_run_before_writing_artifact(tmp_path) -> None:
    artifact = tmp_path / "infer.jsonl"
    _legacy_artifact(artifact)
    before = artifact.read_bytes()
    session = create_session_summary(source="test", session_id="session-1")

    class WrongRun(_Engine):
        def collect(self, *, run_id: str, session_id: str) -> EngineCapture:
            capture = super().collect(run_id=run_id, session_id=session_id)
            events = tuple(
                replace(event, context=_context("engine-0", run_id="wrong"))
                for event in capture.events
            )
            return replace(capture, events=events)

    with pytest.raises(ValueError, match="run_id"):
        append_inference_capture(
            artifact,
            run_id="run-1",
            session=session,
            engine_adapter=WrongRun(),
            envelope_path=tmp_path / "stormlog_run.json",
        )

    assert artifact.read_bytes() == before
    assert not (tmp_path / "stormlog_run.json").exists()


def test_ambiguous_or_unscoped_correlation_stays_unresolved() -> None:
    engine = _Engine().collect(run_id="run-1", session_id="session-1")
    seed = next(
        event for event in engine.events if isinstance(event, ActivityReferenceEvent)
    )
    other_seed = replace(
        seed,
        event_id="launch-8",
        iteration_ref=EntityRef("engine-0", "iteration-8"),
    )
    trace = ActivityReferenceEvent(
        context=_context("trace-0"),
        event_id="kernel-1",
        activity_ref=EntityRef("trace-0", "kernel-1"),
        activity_kind="gpu_kernel",
        attribution_status="unresolved",
        correlation_scope=seed.correlation_scope,
        runtime_correlation_id=17,
    )

    assert link_trace_activities((seed, other_seed), (trace,)) == (trace,)
    assert (
        link_trace_activities(
            (seed,),
            (replace(trace, correlation_scope=EntityRef("runtime-0", "other")),),
        )[0].attribution_status
        == "unresolved"
    )


def test_capability_outcomes_must_be_subsets() -> None:
    with pytest.raises(ValueError, match="collected must be enabled"):
        CaptureCapabilities(supported=("gpu_activity",), collected=("gpu_activity",))
