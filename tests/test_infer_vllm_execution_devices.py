"""Binding profiler traces to GPUs through the vLLM hook's worker hellos."""

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
    load_inference_artifact,
)
from stormlog.infer.errors import InferInputError
from stormlog.infer.trace_kineto import load_kineto_trace
from stormlog.infer.vllm_execution_devices import (
    STATUS_BOUND,
    STATUS_NO_HOST,
    STATUS_NONE,
    STATUS_PARTIAL,
    WorkerIndex,
    trace_wall_window,
)
from tests.vllm_execution_helpers import (
    HOST,
    SECOND,
    WALL_OFFSET,
    engine_log,
    goodbye,
    heartbeat,
    hello,
    write_epoch,
)

ENGINE_PID = 2600
T0 = 1_000 * SECOND  # the workers' monotonic clock at their hello
W0 = T0 + WALL_OFFSET  # the same instant on the wall clock
NOW = W0 + 100 * SECOND


def _worker(
    root: Path,
    pid: int,
    *,
    start_ns: int = 7,
    hello_mono_ns: int = T0,
    ordinal: int = 0,
    uuid: str = "GPU-a",
    records: list[dict[str, Any]] | None = None,
) -> Path:
    first = hello(
        "worker",
        pid,
        start_ns,
        mono_ns=hello_mono_ns,
        engine_pid=ENGINE_PID,
        rank={"tp": ordinal, "pp": 0, "dp": 0},
        local_rank=ordinal,
        cuda_ordinal=ordinal,
        device_uuid=uuid,
        trace_rank_suffix=f"rank{ordinal}",
    )
    return write_epoch(root, "worker", pid, start_ns, [first, *(records or [])])


def _trace(path: Path, *, pid: int, base_ns: int, host: str = HOST) -> Path:
    events = [
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
        "baseTimeNanoseconds": base_ns,
        "host_name": host,
        "traceEvents": events,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _artifact(path: Path) -> Path:
    identity = ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id="run-1",
            session_id="session-1",
            producer_id="stormlog.infer.profile",
            source="stormlog.infer.profile",
            clock_domain="client/wall",
            clock_kind="wall",
            collection_mode="active",
            provenance="observed",
        ),
        event_id="artifact",
        artifact_kind="inference_jsonl",
        created_at_ns=1,
    )
    path.write_text(json.dumps(identity.to_record()) + "\n", encoding="utf-8")
    return path


def test_the_index_lists_every_worker_hello(tmp_path: Path) -> None:
    engine_log(tmp_path, [])
    _worker(tmp_path, 2601, records=[heartbeat(T0 + 10 * SECOND, 1)])
    _worker(
        tmp_path, 2602, ordinal=1, uuid="GPU-b", records=[goodbye(T0 + 5 * SECOND, 1)]
    )

    index = WorkerIndex.from_directory(tmp_path, now_ns=NOW)

    assert [(w.pid, w.cuda_ordinal, w.device_uuid) for w in index.workers] == [
        (2601, 0, "GPU-a"),
        (2602, 1, "GPU-b"),
    ]
    first, second = index.workers
    assert (first.host, first.started_wall_ns, first.ended) == (HOST, W0, False)
    assert first.last_seen_wall_ns == W0 + 10 * SECOND
    assert (second.ended, second.last_seen_wall_ns) == (True, W0 + 5 * SECOND)
    assert first.engine_producer is not None and first.engine_producer.endswith(
        ":2600:7"
    )
    assert index.summary()["workers"][0]["device_uuid"] == "GPU-a"


def test_a_trace_binds_to_the_worker_alive_on_its_host_and_pid(tmp_path: Path) -> None:
    _worker(tmp_path, 2601, records=[heartbeat(T0 + 10 * SECOND, 1)])
    index = WorkerIndex.from_directory(tmp_path, now_ns=NOW)

    inside = index.bind(
        host=HOST, pids=[2601], start_wall_ns=W0 + SECOND, end_wall_ns=W0 + 2 * SECOND
    )
    assert inside.status == STATUS_BOUND
    assert inside.uuids == {0: "GPU-a"} and inside.workers == {2601: "worker-2601-7"}
    # Before the hello, or long after the last heartbeat: not this process.
    before = index.bind(
        host=HOST, pids=[2601], start_wall_ns=W0 - SECOND, end_wall_ns=W0 + SECOND
    )
    assert before.status == STATUS_NONE and before.unmatched == [2601]
    after = index.bind(
        host=HOST,
        pids=[2601],
        start_wall_ns=W0 + 50 * SECOND,
        end_wall_ns=W0 + 60 * SECOND,
    )
    assert after.status == STATUS_NONE
    # Within the heartbeat slack after the last one: still alive.
    close = index.bind(
        host=HOST,
        pids=[2601],
        start_wall_ns=W0 + 10 * SECOND,
        end_wall_ns=W0 + 30 * SECOND,
    )
    assert close.status == STATUS_BOUND
    elsewhere = index.bind(
        host="other-host",
        pids=[2601],
        start_wall_ns=W0 + SECOND,
        end_wall_ns=W0 + 2 * SECOND,
    )
    assert elsewhere.status == STATUS_NONE
    unknown = index.bind(
        host=None, pids=[2601], start_wall_ns=W0 + SECOND, end_wall_ns=W0 + 2 * SECOND
    )
    assert unknown.status == STATUS_NO_HOST


def test_a_reused_pid_is_told_apart_by_lifetime(tmp_path: Path) -> None:
    _worker(tmp_path, 2601, start_ns=7, records=[goodbye(T0 + 5 * SECOND, 1)])
    _worker(
        tmp_path,
        2601,
        start_ns=8,
        hello_mono_ns=T0 + 20 * SECOND,
        uuid="GPU-b",
        records=[heartbeat(T0 + 40 * SECOND, 1)],
    )
    index = WorkerIndex.from_directory(tmp_path, now_ns=NOW)

    first = index.bind(
        host=HOST, pids=[2601], start_wall_ns=W0 + SECOND, end_wall_ns=W0 + 4 * SECOND
    )
    assert first.uuids == {0: "GPU-a"} and first.workers == {2601: "worker-2601-7"}
    second = index.bind(
        host=HOST,
        pids=[2601],
        start_wall_ns=W0 + 25 * SECOND,
        end_wall_ns=W0 + 30 * SECOND,
    )
    assert second.uuids == {0: "GPU-b"} and second.workers == {2601: "worker-2601-8"}
    # A window after the first process ended and before the second began.
    between = index.bind(
        host=HOST,
        pids=[2601],
        start_wall_ns=W0 + 6 * SECOND,
        end_wall_ns=W0 + 10 * SECOND,
    )
    assert between.status == STATUS_NONE


def test_two_live_epochs_for_one_pid_are_ambiguous(tmp_path: Path) -> None:
    _worker(tmp_path, 2601, start_ns=7, records=[heartbeat(T0 + 10 * SECOND, 1)])
    _worker(
        tmp_path,
        2601,
        start_ns=8,
        uuid="GPU-b",
        records=[heartbeat(T0 + 10 * SECOND, 1)],
    )
    _worker(
        tmp_path,
        2602,
        start_ns=9,
        ordinal=1,
        uuid="GPU-c",
        records=[heartbeat(T0 + 10 * SECOND, 1)],
    )
    index = WorkerIndex.from_directory(tmp_path, now_ns=NOW)

    binding = index.bind(
        host=HOST,
        pids=[2601, 2602],
        start_wall_ns=W0 + SECOND,
        end_wall_ns=W0 + 2 * SECOND,
    )

    assert binding.status == STATUS_PARTIAL
    assert binding.uuids == {1: "GPU-c"}
    assert binding.ambiguous == {2601: ["worker-2601-7", "worker-2601-8"]}
    assert binding.summary()["ambiguous"] == {
        "2601": ["worker-2601-7", "worker-2601-8"]
    }


def test_a_loaded_trace_binds_by_its_window_and_launching_pid(tmp_path: Path) -> None:
    _worker(tmp_path / "hook", 2601, records=[heartbeat(T0 + 10 * SECOND, 1)])
    trace = load_kineto_trace(
        _trace(tmp_path / "rank0.pt.trace.json", pid=2601, base_ns=W0 + SECOND)
    )
    assert trace_wall_window(trace) == (W0 + SECOND + 5_000, W0 + SECOND + 30_000)

    binding = WorkerIndex.from_directory(tmp_path / "hook", now_ns=NOW).bind_trace(
        trace
    )

    assert binding.status == STATUS_BOUND and binding.uuids == {0: "GPU-a"}


def test_import_trace_takes_the_device_from_the_execution_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = _worker(
        tmp_path / "hook", 2601, records=[heartbeat(T0 + 10 * SECOND, 1)]
    ).parent.parent
    trace = _trace(tmp_path / "rank0.pt.trace.json", pid=2601, base_ns=W0 + SECOND)

    code = main(
        ["import-trace", str(artifact), str(trace), "--vllm-execution-dir", str(hook)]
    )

    assert code == int(ExitCode.OK)
    out = capsys.readouterr().out
    assert "device 0 (GPU-a, from the vLLM execution log): busy 0.020 ms" in out
    records = load_inference_artifact(artifact)
    activity = next(r for r in records if isinstance(r, ActivityReferenceEvent))
    assert activity.context.device_uuid == "GPU-a"
    collector = next(
        r
        for r in records
        if isinstance(r, CapabilityEvent) and r.component == "trace_collector"
    )
    (summary,) = collector.metadata["summary"]["traces"]
    assert summary["devices"]["0"]["device_uuid_source"] == "execution_log"
    assert summary["execution_log"]["status"] == "bound"
    assert summary["execution_log"]["workers"] == {"2601": "worker-2601-7"}


def test_device_uuid_wins_over_the_execution_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = _worker(
        tmp_path / "hook", 2601, records=[heartbeat(T0 + 10 * SECOND, 1)]
    ).parent.parent
    trace = _trace(tmp_path / "rank0.pt.trace.json", pid=2601, base_ns=W0 + SECOND)

    code = main(
        [
            "import-trace",
            str(artifact),
            str(trace),
            "--vllm-execution-dir",
            str(hook),
            "--device-uuid",
            "0=GPU-given",
        ]
    )

    assert code == int(ExitCode.OK)
    assert "device 0 (GPU-given): busy" in capsys.readouterr().out
    activity = next(
        r
        for r in load_inference_artifact(artifact)
        if isinstance(r, ActivityReferenceEvent)
    )
    assert activity.context.device_uuid == "GPU-given"


def test_an_unmatched_process_is_reported_and_left_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = _worker(
        tmp_path / "hook", 2601, records=[goodbye(T0 + 5 * SECOND, 1)]
    ).parent.parent
    trace = _trace(tmp_path / "rank0.pt.trace.json", pid=2601, base_ns=W0 + 50 * SECOND)

    code = main(
        ["import-trace", str(artifact), str(trace), "--vllm-execution-dir", str(hook)]
    )

    assert code == int(ExitCode.OK)
    out = capsys.readouterr().out
    assert "device 0 (unknown UUID, not measured)" in out
    assert (
        "execution log: no worker epoch covers process 2601; give --device-uuid" in out
    )


def test_a_missing_execution_dir_is_invalid_input(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(tmp_path / "rank0.pt.trace.json", pid=2601, base_ns=W0)
    code = main(
        [
            "import-trace",
            str(artifact),
            str(trace),
            "--vllm-execution-dir",
            str(tmp_path / "nope"),
        ]
    )
    assert code == int(ExitCode.INVALID_INPUT)
    with pytest.raises(InferInputError, match="vllm-execution-dir"):
        WorkerIndex.from_directory(tmp_path / "nope")
    assert artifact.read_text(encoding="utf-8").count("\n") == 1
