"""Importing profiler traces into an existing inference artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CapabilityEvent,
    CorrelationContext,
    EntityRef,
    load_inference_artifact,
)
from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.trace_import import (
    artifact_run_identity,
    import_traces_into_artifact,
    parse_device_uuids,
)


def _artifact(path: Path) -> Path:
    identity = ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id="run-9",
            session_id="session-9",
            producer_id="stormlog.infer.profile",
            source="stormlog.infer.profile",
            clock_domain="client/wall",
            clock_kind="wall",
            collection_mode="active",
            provenance="observed",
        ),
        event_id="artifact",
        artifact_kind="inference_profile",
        created_at_ns=1,
    )
    path.write_text(json.dumps(identity.to_record()) + "\n", encoding="utf-8")
    return path


def _trace(path: Path, trace_id: str) -> Path:
    events: list[dict[str, Any]] = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "stormlog.iteration/engine/step-1",
            "pid": 1,
            "tid": 1,
            "ts": 0.0,
            "dur": 50.0,
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "pid": 1,
            "tid": 1,
            "ts": 5.0,
            "dur": 1.0,
            "args": {"correlation": 1},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm",
            "pid": 0,
            "tid": 7,
            "ts": 10.0,
            "dur": 20.0,
            "args": {"device": 0, "stream": 7, "correlation": 1},
        },
    ]
    document = {
        "baseTimeNanoseconds": 1_000_000_000,
        "trace_id": trace_id,
        "host_name": "worker-a",
        "traceEvents": events,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_run_identity_comes_from_the_artifact_record(tmp_path: Path) -> None:
    assert artifact_run_identity(_artifact(tmp_path / "a.jsonl")) == (
        "run-9",
        "session-9",
    )
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps({"schema_version": 1, "event_type": "infer.request"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InferInputError, match="infer.artifact"):
        artifact_run_identity(legacy)


def test_device_uuid_pairs() -> None:
    assert parse_device_uuids(["1=GPU-b", "GPU-a"]) == {0: "GPU-a", 1: "GPU-b"}
    assert parse_device_uuids([]) == {}
    with pytest.raises(InferUsageError, match="INDEX=UUID"):
        parse_device_uuids(["x=GPU-a"])
    with pytest.raises(InferUsageError, match="given twice"):
        parse_device_uuids(["0=GPU-a", "0=GPU-b"])


def test_import_appends_activity_from_every_trace(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    traces = [
        _trace(tmp_path / "rank0.pt.trace.json", "T0"),
        _trace(tmp_path / "rank1.pt.trace.json", "T1"),
    ]

    capture = import_traces_into_artifact(artifact, traces, device_uuids={0: "GPU-a"})

    assert capture.summary is not None
    assert [s["trace_id"] for s in capture.summary["traces"]] == ["T0", "T1"]
    records = load_inference_artifact(artifact)
    activities = [r for r in records if isinstance(r, ActivityReferenceEvent)]
    assert {a.trace_attachment_id for a in activities} == {
        "kineto:rank0.pt.trace.json",
        "kineto:rank1.pt.trace.json",
    }
    assert {a.iteration_ref for a in activities} == {EntityRef("engine", "step-1")}
    assert {a.context.run_id for a in activities} == {"run-9"}
    collector = next(
        r
        for r in records
        if isinstance(r, CapabilityEvent) and r.component == "trace_collector"
    )
    assert len(collector.metadata["summary"]["traces"]) == 2
    envelope = json.loads((tmp_path / "stormlog_run.json").read_text())
    assert {row["attachment_id"] for row in envelope["attachments"]} >= {
        "kineto:rank0.pt.trace.json",
        "kineto:rank1.pt.trace.json",
    }


def test_cli_imports_and_reports_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(tmp_path / "rank0.pt.trace.json", "T0")

    code = main(["import-trace", str(artifact), str(trace), "--device-uuid", "0=GPU-a"])

    assert code == int(ExitCode.OK)
    out = capsys.readouterr().out
    assert "1 GPU events as 1 records; 1 linked to iterations, 0 unresolved" in out
    assert "device 0 (GPU-a): busy 0.020 ms" in out


def test_cli_rejects_a_missing_trace(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")

    code = main(["import-trace", str(artifact), str(tmp_path / "missing.json")])

    assert code == int(ExitCode.INVALID_INPUT)
    assert artifact.read_text(encoding="utf-8").count("\n") == 1


def test_cli_exits_invalid_input_for_a_file_that_is_not_a_trace(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{not json", encoding="utf-8")

    code = main(["import-trace", str(artifact), str(bogus)])

    assert code == int(ExitCode.INVALID_INPUT)


def test_cli_exits_usage_for_a_malformed_device_uuid(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(tmp_path / "rank0.pt.trace.json", "T0")

    code = main(["import-trace", str(artifact), str(trace), "--device-uuid", "a=b"])

    assert code == int(ExitCode.USAGE)


def test_cli_exits_invalid_input_for_an_artifact_without_run_identity(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps({"schema_version": 1, "event_type": "infer.request"}) + "\n",
        encoding="utf-8",
    )
    trace = _trace(tmp_path / "rank0.pt.trace.json", "T0")

    assert main(["import-trace", str(legacy), str(trace)]) == int(
        ExitCode.INVALID_INPUT
    )


def test_cli_exits_invalid_input_for_a_truncated_gzip_trace(tmp_path: Path) -> None:
    import gzip

    artifact = _artifact(tmp_path / "infer.jsonl")
    whole = gzip.compress(_trace(tmp_path / "t.json", "T0").read_bytes())
    truncated = tmp_path / "rank0.pt.trace.json.gz"
    truncated.write_bytes(whole[: len(whole) // 2])

    assert main(["import-trace", str(artifact), str(truncated)]) == int(
        ExitCode.INVALID_INPUT
    )
