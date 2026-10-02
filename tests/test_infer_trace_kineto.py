"""Kineto trace import: GPU activity, launch correlation, and iteration ranges."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.correlation_accounting import (
    DeviceClock,
    account_gpu_time,
    resolve_inference_events,
)
from stormlog.infer.correlation_capture import (
    TraceAttachment,
    TraceCapture,
    append_inference_capture,
)
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CapabilityEvent,
    CorrelationContext,
    CorrelationEvent,
    EntityRef,
    IterationEvent,
    load_inference_artifact,
)
from stormlog.infer.trace_kineto import (
    import_kineto_trace,
    link_gpu_event,
    load_kineto_trace,
    merge_intervals,
)
from stormlog.session import create_session_summary

BASE_NS = 1_790_000_000_000_000_000
ENGINE = "engine-0"
PID, TID = 10, 11


def _span(name: str, ts: float, dur: float, tid: int = TID) -> dict[str, Any]:
    return {
        "ph": "X",
        "cat": "user_annotation",
        "name": name,
        "pid": PID,
        "tid": tid,
        "ts": ts,
        "dur": dur,
    }


def _launch(name: str, correlation: int, ts: float, tid: int = TID) -> dict[str, Any]:
    return {
        "ph": "X",
        "cat": "cuda_runtime",
        "name": name,
        "pid": PID,
        "tid": tid,
        "ts": ts,
        "dur": 2.0,
        "args": {"correlation": correlation},
    }


def _gpu(
    category: str,
    correlation: int,
    ts: float,
    dur: float,
    *,
    stream: int = 7,
    graph_id: int = 0,
) -> dict[str, Any]:
    return {
        "ph": "X",
        "cat": category,
        "name": f"{category}-{correlation}",
        "pid": 0,
        "tid": stream,
        "ts": ts,
        "dur": dur,
        "args": {
            "device": 0,
            "stream": stream,
            "correlation": correlation,
            "graph id": graph_id,
        },
    }


def _trace_document() -> dict[str, Any]:
    """Two iterations, a graph replay, overlap, and two unlinkable events.

    - correlation 1: launched in iteration 1; two kernels on two streams that
      overlap ([20, 40) and [30, 50) us: union 30, sum 40).
    - correlation 2: a CUDA graph launched in iteration 2; three kernels at
      [130, 140), [141, 150), [150, 160).
    - correlation 3: a copy launched outside any iteration range.
    - correlation 4: a kernel whose launch call is not in the trace.
    """
    it1 = f"stormlog.iteration/{ENGINE}/it-1"
    it2 = f"stormlog.iteration/{ENGINE}/it-2"
    return {
        "schemaVersion": 1,
        "baseTimeNanoseconds": BASE_NS,
        "trace_id": "TRACE1",
        "host_name": "worker-a",
        "vllm_version": "0.30.0",
        "cupti_version": 130001,
        "distributedInfo": {"rank": 0, "world_size": 1},
        "deviceProperties": [{"id": 0, "name": "NVIDIA A30"}],
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": PID, "args": {"name": "x"}},
            _span(it1, 0.0, 100.0),
            _span(it2, 100.0, 100.0),
            _span("unrelated range", 0.0, 300.0),
            _launch("cudaLaunchKernel", 1, 10.0),
            _launch("cudaGraphLaunch", 2, 120.0),
            _launch("cudaMemcpyAsync", 3, 250.0),
            _gpu("kernel", 1, 20.0, 20.0, stream=7),
            _gpu("kernel", 1, 30.0, 20.0, stream=8),
            _gpu("kernel", 2, 130.0, 10.0, graph_id=5),
            _gpu("kernel", 2, 141.0, 9.0, graph_id=5),
            _gpu("kernel", 2, 150.0, 10.0, graph_id=5),
            _gpu("gpu_memcpy", 3, 260.0, 10.0),
            _gpu("kernel", 4, 300.0, 10.0),
        ],
    }


def _write(tmp_path: Path, document: dict[str, Any], *, compress: bool = False) -> Path:
    payload = json.dumps(document).encode("utf-8")
    if compress:
        path = tmp_path / "rank0.pt.trace.json.gz"
        path.write_bytes(gzip.compress(payload))
    else:
        path = tmp_path / "rank0.pt.trace.json"
        path.write_bytes(payload)
    return path


def _activities(capture: TraceCapture) -> list[ActivityReferenceEvent]:
    return [e for e in capture.events if isinstance(e, ActivityReferenceEvent)]


def _ns(us: float) -> int:
    return BASE_NS + int(round(us * 1000))


def test_load_reads_header_events_and_ranges(tmp_path: Path) -> None:
    trace = load_kineto_trace(_write(tmp_path, _trace_document(), compress=True))

    assert trace.base_ns == BASE_NS
    assert (trace.host, trace.trace_id, trace.rank) == ("worker-a", "TRACE1", 0)
    assert trace.engine_version == "0.30.0"
    assert trace.cupti_version == "130001"
    assert trace.device_names == {0: "NVIDIA A30"}
    assert len(trace.gpu_events) == 7
    assert set(trace.launches) == {1, 2, 3}
    assert [span.iteration_ref.id for span in trace.spans[(PID, TID)]] == [
        "it-1",
        "it-2",
    ]


def test_links_only_through_the_launch_call_and_its_range(tmp_path: Path) -> None:
    trace = load_kineto_trace(_write(tmp_path, _trace_document()))
    links = {
        event.correlation: link_gpu_event(trace, event) for event in trace.gpu_events
    }

    assert links[1].iteration_ref == EntityRef(ENGINE, "it-1")
    assert links[2].iteration_ref == EntityRef(ENGINE, "it-2")
    assert links[3].reason == "launch_outside_iteration_range"
    assert links[4].reason == "no_launch_record"


def test_overlapping_iteration_ranges_on_one_thread_are_ambiguous(
    tmp_path: Path,
) -> None:
    document = _trace_document()
    document["traceEvents"].append(
        _span(f"stormlog.iteration/{ENGINE}/other", 5.0, 10.0)
    )
    trace = load_kineto_trace(_write(tmp_path, document))
    event = next(e for e in trace.gpu_events if e.correlation == 1)

    assert link_gpu_event(trace, event).reason == "ambiguous_iteration_range"


def test_a_range_on_another_thread_does_not_link(tmp_path: Path) -> None:
    document = _trace_document()
    document["traceEvents"].append(_launch("cudaLaunchKernel", 9, 50.0, tid=99))
    document["traceEvents"].append(_gpu("kernel", 9, 60.0, 5.0))
    trace = load_kineto_trace(_write(tmp_path, document))
    event = next(e for e in trace.gpu_events if e.correlation == 9)

    assert link_gpu_event(trace, event).reason == "launch_outside_iteration_range"


def test_kernel_detail_emits_one_record_per_gpu_event(tmp_path: Path) -> None:
    capture = import_kineto_trace(
        _write(tmp_path, _trace_document()),
        run_id="run-1",
        session_id="session-1",
        device_uuids={0: "GPU-abc"},
        detail="kernel",
    )
    activities = _activities(capture)

    assert len(activities) == 7
    assert {a.activity_domain for a in activities} == {"gpu"}
    assert {a.activity_kind for a in activities} == {"gpu_kernel", "gpu_memcpy"}
    assert {a.context.device_uuid for a in activities} == {"GPU-abc"}
    assert {a.context.clock_kind for a in activities} == {"device"}
    graph = [a for a in activities if a.cuda_correlation_id == 2]
    assert {a.graph_id for a in graph} == {5}
    assert {a.iteration_ref for a in graph} == {EntityRef(ENGINE, "it-2")}
    unresolved = {
        a.cuda_correlation_id: a.metadata["unresolved_reason"]
        for a in activities
        if a.attribution_status == "unresolved"
    }
    assert unresolved == {
        3: "launch_outside_iteration_range",
        4: "no_launch_record",
    }


def test_launch_detail_spans_each_launch_and_keeps_its_exact_busy_time(
    tmp_path: Path,
) -> None:
    capture = import_kineto_trace(
        _write(tmp_path, _trace_document()),
        run_id="run-1",
        session_id="session-1",
    )
    activities = _activities(capture)
    spans = {
        a.cuda_correlation_id: [
            (b.start_ns, b.end_ns)
            for b in activities
            if b.cuda_correlation_id == a.cuda_correlation_id
        ]
        for a in activities
    }

    assert spans[1] == [(_ns(20), _ns(50))]
    assert spans[2] == [(_ns(130), _ns(160))]
    overlapped = next(a for a in activities if a.cuda_correlation_id == 1)
    assert overlapped.stream_id is None
    assert overlapped.metadata["streams"] == [7, 8]
    assert overlapped.metadata["event_count"] == 2
    assert overlapped.metadata["summed_duration_ns"] == 40_000
    assert overlapped.metadata["busy_ns"] == 30_000
    graph = next(a for a in activities if a.cuda_correlation_id == 2)
    assert (graph.metadata["busy_ns"], graph.metadata["idle_inside_ns"]) == (
        29_000,
        1_000,
    )


@pytest.mark.parametrize(
    ("detail", "busy_us"),
    # Exact union: [20,50) [130,140) [141,160) [260,270) [300,310) = 79 us.
    # Launch records also count the 1 us gap inside the graph replay.
    [("kernel", 79), ("launch", 80)],
)
def test_accounting_busy_time_by_detail_mode(
    tmp_path: Path, detail: str, busy_us: int
) -> None:
    capture = import_kineto_trace(
        _write(tmp_path, _trace_document()),
        run_id="run-1",
        session_id="session-1",
        device_uuids={0: "GPU-abc"},
        detail=detail,  # type: ignore[arg-type]
    )
    accounting = account_gpu_time(resolve_inference_events(_with_iterations(capture)))
    device = next(iter(accounting.device_totals.values()))

    assert device.busy_ns == busy_us * 1000
    assert capture.summary is not None
    assert capture.summary["devices"]["0"]["busy_ns"] == 79_000
    assert capture.summary["devices"]["0"]["record_busy_ns"] == busy_us * 1000
    it1 = accounting.iterations[EntityRef(ENGINE, "it-1")]
    assert it1.gpu[DeviceClock("GPU-abc", "kineto:worker-a:TRACE1", "device")].busy_ns
    assert len(accounting.unattributed_activity_refs) == 2


def test_unknown_device_uuid_leaves_gpu_activity_unmeasured(tmp_path: Path) -> None:
    capture = import_kineto_trace(
        _write(tmp_path, _trace_document()), run_id="run-1", session_id="session-1"
    )
    accounting = account_gpu_time(resolve_inference_events(_with_iterations(capture)))

    assert accounting.device_totals == {}
    assert len(accounting.unmeasured_gpu_activity_refs) == 4
    assert capture.summary is not None
    assert capture.summary["devices"]["0"]["device_uuid"] is None


def test_summary_reports_coverage_and_does_not_claim_zero_loss(
    tmp_path: Path,
) -> None:
    capture = import_kineto_trace(
        _write(tmp_path, _trace_document()), run_id="run-1", session_id="session-1"
    )
    summary = capture.summary
    assert summary is not None

    assert summary["gpu_events"] == 7
    assert summary["linked_gpu_events"] == 5
    assert summary["unresolved_gpu_events"] == {
        "launch_outside_iteration_range": 1,
        "no_launch_record": 1,
    }
    assert summary["graph_gpu_events"] == 3
    assert summary["devices"]["0"]["busy_ns"] == 79_000
    assert summary["devices"]["0"]["summed_ns"] == 89_000
    assert summary["event_loss"] is None
    assert set(capture.capabilities.collected) == {
        "gpu_activity",
        "cuda_correlation",
        "iteration_ranges",
        "cuda_graph_ids",
        "streams",
    }


def test_a_trace_without_iteration_ranges_is_collected_but_unlinked(
    tmp_path: Path,
) -> None:
    document = _trace_document()
    document["traceEvents"] = [
        event
        for event in document["traceEvents"]
        if not str(event.get("name", "")).startswith("stormlog.iteration/")
    ]
    capture = import_kineto_trace(
        _write(tmp_path, document), run_id="run-1", session_id="session-1"
    )

    assert "iteration_ranges" not in capture.capabilities.collected
    assert {a.attribution_status for a in _activities(capture)} == {"unresolved"}


def test_rejects_files_that_are_not_kineto_traces(tmp_path: Path) -> None:
    path = tmp_path / "not-a-trace.json"
    path.write_text(json.dumps({"events": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="traceEvents"):
        load_kineto_trace(path)
    with pytest.raises(ValueError, match="detail"):
        import_kineto_trace(
            _write(tmp_path, _trace_document()),
            run_id="run-1",
            session_id="session-1",
            detail="everything",  # type: ignore[arg-type]
        )


def test_merge_intervals_joins_overlapping_and_touching_spans() -> None:
    assert merge_intervals([(5, 9), (0, 3), (3, 4), (8, 12)]) == [(0, 4), (5, 12)]
    assert merge_intervals([]) == []


def test_capture_records_the_import_summary_and_registers_the_trace(
    tmp_path: Path,
) -> None:
    trace_path = _write(tmp_path, _trace_document(), compress=True)
    artifact = tmp_path / "infer.jsonl"
    artifact.write_text("", encoding="utf-8")
    attachment = TraceAttachment(
        attachment_id="vllm-rank0",
        title="vLLM worker trace, rank 0",
        path=trace_path,
        storage="reference",
    )

    class _Collector:
        def collect(self, *, run_id: str, session_id: str) -> TraceCapture:
            return import_kineto_trace(
                trace_path,
                run_id=run_id,
                session_id=session_id,
                attachment=attachment,
                device_uuids={0: "GPU-abc"},
            )

    append_inference_capture(
        artifact,
        run_id="run-1",
        session=create_session_summary(source="test", session_id="session-1"),
        trace_collector=_Collector(),
        envelope_path=tmp_path / "stormlog_run.json",
    )

    records = load_inference_artifact(artifact)
    capability = next(
        r
        for r in records
        if isinstance(r, CapabilityEvent) and r.component == "trace_collector"
    )
    assert capability.metadata["summary"]["linked_gpu_events"] == 5
    activities = [r for r in records if isinstance(r, ActivityReferenceEvent)]
    assert {a.trace_attachment_id for a in activities} == {"vllm-rank0"}
    envelope = json.loads((tmp_path / "stormlog_run.json").read_text())
    assert "vllm-rank0" in {row["attachment_id"] for row in envelope["attachments"]}


def test_record_function_ranges_survive_a_real_profiler_export(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from torch.profiler import ProfilerActivity, profile

    from stormlog.infer.trace_ranges import iteration_range

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        for step in range(2):
            with iteration_range(ENGINE, f"step-{step}"):
                torch.ones(8) + 1
    path = tmp_path / "cpu.json"
    profiler.export_chrome_trace(str(path))

    trace = load_kineto_trace(path)
    refs = [span.iteration_ref for spans in trace.spans.values() for span in spans]
    assert refs == [EntityRef(ENGINE, "step-0"), EntityRef(ENGINE, "step-1")]


def _with_iterations(capture: TraceCapture) -> list[CorrelationEvent]:
    context = CorrelationContext(
        run_id="run-1",
        session_id="session-1",
        producer_id=ENGINE,
        source="test",
        clock_domain="engine/monotonic",
        clock_kind="monotonic",
        collection_mode="active",
        provenance="observed",
    )
    events: list[CorrelationEvent] = [
        ArtifactIdentityEvent(
            context=context, event_id="artifact", artifact_kind="test", created_at_ns=0
        )
    ]
    for name in ("it-1", "it-2"):
        events.append(
            IterationEvent(
                context=context,
                event_id=name,
                iteration_ref=EntityRef(ENGINE, name),
            )
        )
    events.extend(capture.events)
    return events
