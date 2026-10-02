"""Nsight Systems SQLite exports: GPU work, launches, NVTX ranges, and GPU UUIDs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CorrelationContext,
    EntityRef,
    load_inference_artifact,
)
from stormlog.infer.trace_import import TraceFileCollector, trace_attachment_id
from stormlog.infer.trace_kineto import link_gpu_event
from stormlog.infer.trace_nsys import load_nsys_sqlite, split_global_id

SESSION_NS = 1_790_000_000_000_000_000


def _gid(pid: int, tid: int) -> int:
    return (1 << 48) | (pid << 24) | tid


def _export(path: Path) -> Path:
    """Two processes, both using correlation 1, on different physical GPUs.

    Process 100 sees physical GPU 1 as CUDA device 0; process 200 sees GPU 0.
    Each launches a kernel inside its own NVTX iteration range. Process 100
    also replays a CUDA graph (two kernels) and copies outside any range.
    """
    with closing(sqlite3.connect(path)) as db:
        db.executescript(
            """
            create table StringIds (id integer, value text);
            create table TARGET_INFO_SESSION_START_TIME (utcEpochNs integer, utcTime text, localTime text);
            create table META_DATA_CAPTURE (name text, value text);
            create table TARGET_INFO_GPU (id integer, name text, uuid text);
            create table TARGET_INFO_CUDA_DEVICE (gpuId integer, cudaId integer, pid integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer,
                streamId integer, correlationId integer, globalPid integer, shortName integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, deviceId integer,
                streamId integer, correlationId integer, globalPid integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, globalTid integer,
                correlationId integer, nameId integer);
            create table NVTX_EVENTS (start integer, end integer, eventType integer, text text,
                textId integer, globalTid integer);
            """
        )
        db.executemany(
            "insert into StringIds values (?, ?)",
            [
                (1, "gemm"),
                (2, "cudaLaunchKernel"),
                (3, "cudaGraphLaunch"),
                (4, "stormlog.iteration/engine/step-1"),
                (5, "cudaMemcpyAsync"),
            ],
        )
        db.execute(
            "insert into TARGET_INFO_SESSION_START_TIME values (?, '', '')",
            (SESSION_NS,),
        )
        db.execute(
            "insert into META_DATA_CAPTURE values ('DEVICE_DISPLAY_NAME', 'node-7')"
        )
        db.executemany(
            "insert into TARGET_INFO_GPU values (?, ?, ?)",
            [(0, "NVIDIA A30", "aaaa-0000"), (1, "NVIDIA A30", "bbbb-1111")],
        )
        db.executemany(
            "insert into TARGET_INFO_CUDA_DEVICE values (?, ?, ?)",
            [(1, 0, 100), (0, 0, 200)],
        )
        db.executemany(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?, ?)",
            [
                (2_000, 3_000, 0, 7, 1, _gid(100, 0), 1),
                (2_500, 3_500, 0, 7, 1, _gid(200, 0), 1),
                (6_000, 7_000, 0, 7, 2, _gid(100, 0), 1),
                (7_500, 8_000, 0, 7, 2, _gid(100, 0), 1),
            ],
        )
        db.execute(
            "insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?, ?)",
            (12_000, 13_000, 0, 8, 3, _gid(100, 0)),
        )
        db.executemany(
            "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, ?, ?)",
            [
                (1_000, 1_100, _gid(100, 101), 1, 2),
                (1_200, 1_300, _gid(200, 201), 1, 2),
                (5_000, 5_100, _gid(100, 101), 2, 3),
                (11_000, 11_100, _gid(100, 101), 3, 5),
            ],
        )
        db.executemany(
            "insert into NVTX_EVENTS values (?, ?, ?, ?, ?, ?)",
            [
                (500, 9_000, 59, None, 4, _gid(100, 101)),
                (
                    500,
                    9_000,
                    59,
                    "stormlog.iteration/engine/other-1",
                    None,
                    _gid(200, 201),
                ),
                (400, None, 34, "stormlog.iteration/engine/mark", None, _gid(100, 101)),
            ],
        )
        db.commit()
    return path


def test_global_ids_split_into_pid_and_tid() -> None:
    assert split_global_id(_gid(1825, 1830)) == (1825, 1830)


def test_reads_gpu_work_launches_ranges_and_per_process_gpu_uuids(
    tmp_path: Path,
) -> None:
    trace = load_nsys_sqlite(_export(tmp_path / "run.sqlite"))

    assert (trace.source, trace.host, trace.base_ns) == ("nsys", "node-7", SESSION_NS)
    assert len(trace.gpu_events) == 5
    kernels = {(e.pid, e.start_ns - SESSION_NS): e for e in trace.gpu_events}
    assert kernels[(100, 2_000)].device_uuid == "GPU-bbbb-1111"
    assert kernels[(200, 2_500)].device_uuid == "GPU-aaaa-0000"
    assert set(trace.launches) == {(100, 1), (200, 1), (100, 2), (100, 3)}
    assert [s.iteration_ref for s in trace.spans[(100, 101)]] == [
        EntityRef("engine", "step-1")
    ]


def test_colliding_correlation_ids_link_within_their_own_process(
    tmp_path: Path,
) -> None:
    trace = load_nsys_sqlite(_export(tmp_path / "run.sqlite"))
    links = {(e.pid, e.correlation): link_gpu_event(trace, e) for e in trace.gpu_events}

    assert links[(100, 1)].iteration_ref == EntityRef("engine", "step-1")
    assert links[(200, 1)].iteration_ref == EntityRef("engine", "other-1")
    assert links[(100, 2)].iteration_ref == EntityRef("engine", "step-1")
    assert links[(100, 3)].reason == "launch_outside_iteration_range"


def test_import_uses_the_reports_gpu_uuids_and_counts_graph_work(
    tmp_path: Path,
) -> None:
    export = _export(tmp_path / "run.sqlite")
    capture = TraceFileCollector([export]).collect(run_id="r", session_id="s")
    activities = [e for e in capture.events if isinstance(e, ActivityReferenceEvent)]

    assert {a.context.device_uuid for a in activities} == {
        "GPU-aaaa-0000",
        "GPU-bbbb-1111",
    }
    assert {a.context.source for a in activities} == {"nsys"}
    assert {a.trace_attachment_id for a in activities} == {
        trace_attachment_id(export, "nsys")
    }
    assert capture.summary is not None
    trace = capture.summary["traces"][0]
    assert trace["format"] == "nsys"
    assert trace["graph_gpu_events"] == 2
    assert set(trace["devices"]) == {"100/0", "200/0"}
    assert trace["devices"]["100/0"]["device_uuid"] == "GPU-bbbb-1111"


def _artifact(path: Path) -> Path:
    identity = ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id="r",
            session_id="s",
            producer_id="p",
            source="p",
            clock_domain="c",
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


def test_cli_imports_an_export_and_registers_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    report = tmp_path / "run.nsys-rep"
    report.write_bytes(b"binary report")

    code = main(
        [
            "import-trace",
            str(artifact),
            str(_export(tmp_path / "run.sqlite")),
            str(report),
        ]
    )

    assert code == int(ExitCode.OK)
    out = capsys.readouterr().out
    assert "5 GPU events as 4 records; 4 linked to iterations, 1 unresolved" in out
    assert "Registered run.nsys-rep without importing it" in out
    envelope = json.loads((tmp_path / "stormlog_run.json").read_text())
    rows = {row["attachment_id"]: row for row in envelope["attachments"]}
    assert rows[trace_attachment_id(report, "nsys")]["metadata"] == {
        "format": "nsys-rep"
    }
    records = load_inference_artifact(artifact)
    assert len([r for r in records if isinstance(r, ActivityReferenceEvent)]) == 4


def test_cli_rejects_a_sqlite_file_that_is_not_an_export(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    other = tmp_path / "other.sqlite"
    with closing(sqlite3.connect(other)) as db:
        db.execute("create table t (x integer)")

    assert main(["import-trace", str(artifact), str(other)]) == int(
        ExitCode.INVALID_INPUT
    )
