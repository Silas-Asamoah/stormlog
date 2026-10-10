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
    trace_attachment_id,
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


def _trace(path: Path, trace_id: str, pid: int = 1) -> Path:
    events: list[dict[str, Any]] = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "stormlog.iteration/engine/step-1",
            "pid": pid,
            "tid": 1,
            "ts": 0.0,
            "dur": 50.0,
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "pid": pid,
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
    parsed = parse_device_uuids(["1=GPU-b", "GPU-a", "rank1.pt.trace.json:0=GPU-c"])
    assert parsed.shared == {0: "GPU-a", 1: "GPU-b"}
    traces = [Path("run/rank0.pt.trace.json"), Path("run/rank1.pt.trace.json")]
    bound = parsed.bind(traces)
    assert bound.for_trace(traces[1]) == {0: "GPU-c", 1: "GPU-b"}
    assert bound.for_trace(traces[0]) == {0: "GPU-a", 1: "GPU-b"}
    assert bound.scoped(traces[0]) == {}
    with pytest.raises(ValueError, match="bind"):
        parsed.for_trace(traces[1])
    assert parse_device_uuids([]).shared == {}
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
        trace_attachment_id(path) for path in traces
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
        trace_attachment_id(path) for path in traces
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


@pytest.mark.parametrize("compact_seed", [False, True])
def test_importing_the_same_trace_twice_skips_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], compact_seed
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    if compact_seed:
        from stormlog.infer.correlation_codec import CorrelationRecordEncoder

        encoder = CorrelationRecordEncoder()
        rows = [json.loads(line) for line in artifact.read_text().splitlines()]
        compact = [row for record in rows for row in encoder.encode(record)]
        artifact.write_text("".join(json.dumps(row) + "\n" for row in compact))
    trace = _trace(tmp_path / "rank0.pt.trace.json", "T0")
    assert main(["import-trace", str(artifact), str(trace)]) == int(ExitCode.OK)
    lines = artifact.read_text(encoding="utf-8").count("\n")
    capsys.readouterr()

    assert main(["import-trace", str(artifact), str(trace)]) == int(ExitCode.OK)

    assert artifact.read_text(encoding="utf-8").count("\n") == lines
    assert f"Skipped {trace}: already imported into this run" in capsys.readouterr().out


def test_combined_capture_unions_what_each_trace_collected(tmp_path: Path) -> None:
    from stormlog.infer.trace_import import TraceFileCollector

    with_ranges = _trace(tmp_path / "rank0.pt.trace.json", "T0")
    document = json.loads(with_ranges.read_text(encoding="utf-8"))
    document["traceEvents"] = document["traceEvents"][1:]
    without = tmp_path / "rank1.pt.trace.json"
    without.write_text(json.dumps(document), encoding="utf-8")

    capture = TraceFileCollector([with_ranges, without]).collect(
        run_id="run-9", session_id="session-9"
    )

    assert "iteration_ranges" in capture.capabilities.collected
    assert len(capture.attachments) == 2
    assert capture.summary is not None
    assert [t["linked_gpu_events"] for t in capture.summary["traces"]] == [1, 0]


def test_imports_through_a_symlinked_directory_store_paths_that_open(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # /scratch -> /mnt/nvme/scratch, as on many GPU clusters.
    real = tmp_path / "mnt" / "nvme" / "scratch"
    (real / "run").mkdir(parents=True)
    linked = tmp_path / "scratch"
    linked.symlink_to(real, target_is_directory=True)
    artifact = _artifact(linked / "run" / "infer.jsonl")
    trace = _trace(linked / "run" / "rank0.pt.trace.json", "T0")

    assert main(["import-trace", str(artifact), str(trace)]) == int(ExitCode.OK)
    capsys.readouterr()
    assert main(["import-trace", str(artifact), str(trace)]) == int(ExitCode.OK)

    assert "already imported into this run" in capsys.readouterr().out
    envelope = linked / "run" / "stormlog_run.json"
    row = next(
        r
        for r in json.loads(envelope.read_text())["attachments"]
        if r["attachment_id"] == trace_attachment_id(trace)
    )
    assert (envelope.parent / row["path"]).is_file()
    records = load_inference_artifact(artifact)
    assert len([r for r in records if isinstance(r, ActivityReferenceEvent)]) == 1


def test_traces_with_the_same_name_in_different_directories_are_distinct(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    first = _trace(_dir(tmp_path / "run-a") / "rank0.pt.trace.json", "A")
    second = _trace(_dir(tmp_path / "run-b") / "rank0.pt.trace.json", "B")

    assert main(["import-trace", str(artifact), str(first), str(second)]) == int(
        ExitCode.OK
    )

    activities = [
        r
        for r in load_inference_artifact(artifact)
        if isinstance(r, ActivityReferenceEvent)
    ]
    assert {a.trace_attachment_id for a in activities} == {
        trace_attachment_id(first),
        trace_attachment_id(second),
    }
    assert trace_attachment_id(first) != trace_attachment_id(second)


def test_a_shared_ordinal_used_by_two_processes_needs_per_trace_uuids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    one = _trace(tmp_path / "worker-a.pt.trace.json", "A", pid=101)
    two = _trace(tmp_path / "worker-b.pt.trace.json", "B", pid=202)

    shared = main(
        ["import-trace", str(artifact), str(one), str(two), "--device-uuid", "0=GPU-a"]
    )

    assert shared == int(ExitCode.USAGE)
    assert "give a UUID per trace as TRACE_FILE:0=UUID" in capsys.readouterr().err
    scoped = main(
        [
            "import-trace",
            str(artifact),
            str(one),
            str(two),
            "--device-uuid",
            "worker-a.pt.trace.json:0=GPU-a",
            "--device-uuid",
            "worker-b.pt.trace.json:0=GPU-b",
        ]
    )
    assert scoped == int(ExitCode.OK)
    uuids = {
        a.trace_attachment_id: a.context.device_uuid
        for a in load_inference_artifact(artifact)
        if isinstance(a, ActivityReferenceEvent)
    }
    assert uuids == {
        trace_attachment_id(one): "GPU-a",
        trace_attachment_id(two): "GPU-b",
    }


def test_a_shared_ordinal_is_fine_for_several_traces_of_one_process(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    windows = [
        _trace(tmp_path / f"rank0.{n}.pt.trace.json", f"W{n}", pid=7) for n in (1, 2)
    ]

    code = main(
        ["import-trace", str(artifact), *map(str, windows), "--device-uuid", "0=GPU-a"]
    )

    assert code == int(ExitCode.OK)


def _dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_the_shared_ordinal_check_tells_ranks_apart(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    ranks = []
    for rank in (0, 1):
        path = _trace(tmp_path / f"node{rank}.pt.trace.json", f"R{rank}", pid=7)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["distributedInfo"] = {"rank": rank, "world_size": 2}
        path.write_text(json.dumps(document), encoding="utf-8")
        ranks.append(path)

    code = main(
        ["import-trace", str(artifact), *map(str, ranks), "--device-uuid", "GPU-a"]
    )

    assert code == int(ExitCode.USAGE)


@pytest.mark.parametrize("detail", ["launch", "kernel"])
def test_cli_rejects_a_negative_gpu_duration_at_both_details(
    tmp_path: Path, detail: str, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(tmp_path / "rank0.pt.trace.json", "T0")
    document = json.loads(trace.read_text(encoding="utf-8"))
    document["traceEvents"].append(
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm",
            "pid": 0,
            "tid": 7,
            "ts": 20.0,
            "dur": -100.0,
            "args": {"device": 0, "stream": 7, "correlation": 1},
        }
    )
    trace.write_text(json.dumps(document), encoding="utf-8")

    code = main(
        [
            "import-trace",
            str(artifact),
            str(trace),
            "--detail",
            detail,
            "--device-uuid",
            "0=GPU-a",
        ]
    )

    assert code == int(ExitCode.INVALID_INPUT)
    assert "negative duration" in capsys.readouterr().err
    assert artifact.read_text(encoding="utf-8").count("\n") == 1


def test_a_file_name_shared_by_two_traces_cannot_select_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two workers' rank0 traces from different GPUs need different UUIDs."""
    artifact = _artifact(tmp_path / "infer.jsonl")
    first = _trace(_dir(tmp_path / "worker-a") / "rank0.pt.trace.json", "A", pid=101)
    second = _trace(_dir(tmp_path / "worker-b") / "rank0.pt.trace.json", "B", pid=202)

    code = main(
        [
            "import-trace",
            str(artifact),
            str(first),
            str(second),
            "--device-uuid",
            "rank0.pt.trace.json:0=GPU-a",
        ]
    )

    assert code == int(ExitCode.USAGE)
    assert "2 traces in this import are named rank0.pt.trace.json" in (
        capsys.readouterr().err
    )
    assert artifact.read_text(encoding="utf-8").count("\n") == 1


def test_a_trace_path_selects_one_trace_and_a_unique_file_name_another(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    first = _trace(_dir(tmp_path / "worker-a") / "rank0.pt.trace.json", "A", pid=101)
    second = _trace(_dir(tmp_path / "worker-b") / "rank1.pt.trace.json", "B", pid=202)

    code = main(
        [
            "import-trace",
            str(artifact),
            str(first),
            str(second),
            "--device-uuid",
            f"{first}:0=GPU-a",
            "--device-uuid",
            "rank1.pt.trace.json:0=GPU-b",
        ]
    )

    assert code == int(ExitCode.OK)
    uuids = {
        a.trace_attachment_id: a.context.device_uuid
        for a in load_inference_artifact(artifact)
        if isinstance(a, ActivityReferenceEvent)
    }
    assert uuids == {
        trace_attachment_id(first): "GPU-a",
        trace_attachment_id(second): "GPU-b",
    }


def test_a_selector_that_names_no_trace_is_refused_before_importing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(tmp_path / "rank0.pt.trace.json", "T0")

    code = main(
        [
            "import-trace",
            str(artifact),
            str(trace),
            "--device-uuid",
            "rank9.pt.trace.json:0=GPU-a",
        ]
    )

    assert code == int(ExitCode.USAGE)
    assert "no trace in this import is rank9.pt.trace.json" in capsys.readouterr().err
    assert artifact.read_text(encoding="utf-8").count("\n") == 1


def test_a_selector_for_an_already_imported_trace_is_still_accepted(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    first = _trace(tmp_path / "rank0.pt.trace.json", "T0", pid=101)
    second = _trace(tmp_path / "rank1.pt.trace.json", "T1", pid=202)
    first_only = ["--device-uuid", "rank0.pt.trace.json:0=GPU-a"]
    assert main(["import-trace", str(artifact), str(first), *first_only]) == int(
        ExitCode.OK
    )

    # rank0 is skipped as already imported; its selector must still match it.
    code = main(
        [
            "import-trace",
            str(artifact),
            str(first),
            str(second),
            *first_only,
            "--device-uuid",
            "rank1.pt.trace.json:0=GPU-b",
        ]
    )

    assert code == int(ExitCode.OK)
    uuids = {
        a.trace_attachment_id: a.context.device_uuid
        for a in load_inference_artifact(artifact)
        if isinstance(a, ActivityReferenceEvent)
    }
    assert uuids == {
        trace_attachment_id(first): "GPU-a",
        trace_attachment_id(second): "GPU-b",
    }
