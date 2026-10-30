"""Stormlog CUPTI traces: framing, helper status, loss and launch correlation."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main
from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CorrelationContext,
    load_inference_artifact,
)
from stormlog.infer.trace_cupti import iter_records, load_cupti_trace
from stormlog.infer.trace_import import TraceFileCollector

MAGIC = b"SLCPTZ1\n"
END = struct.pack("<III", 0, 0, 0)


def _frame(records: list[dict[str, Any]]) -> bytes:
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in records)
    encoded = zlib.compress(raw)
    return struct.pack("<III", len(encoded), len(raw), len(records)) + encoded


def _api(correlation: int, start: int, name: str | None = "cudaLaunchKernel") -> dict:
    return {
        "activity_kind": "runtime",
        "correlation_id": str(correlation),
        "cpu_start_ns": start,
        "cpu_end_ns": start + 500,
        "device_start_ns": None,
        "device_end_ns": None,
        "stream_id": None,
        "graph_id": None,
        "graph_node_id": None,
        "metadata": {"cbid": 211, "name": name, "process_id": 42, "thread_id": 7},
    }


def _kernel(correlation: int, start: int | None, graph: str | None = None) -> dict:
    return {
        "activity_kind": "kernel",
        "correlation_id": str(correlation),
        "cpu_start_ns": None,
        "cpu_end_ns": None,
        "device_start_ns": start,
        "device_end_ns": None if start is None else start + 2_000,
        "stream_id": "7",
        "graph_id": graph,
        "graph_node_id": "3" if graph else None,
        "metadata": {"name": "gemm", "device_id": 0, "context_id": 1},
    }


def _status(**overrides: Any) -> dict[str, Any]:
    status = {
        "schema_version": 1,
        "pid": 42,
        "cupti_version": 130001,
        "finalized": True,
        "initialization_error": None,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "buffer_refusals": 0,
        "kernels_without_timestamps": 0,
        "output_limit_reached": False,
    }
    status.update(overrides)
    return status


def _trace(
    directory: Path,
    frames: list[bytes] | None = None,
    *,
    tail: bytes = END,
    status: dict[str, Any] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if frames is None:
        frames = [
            _frame([_api(1, 1_000), _kernel(1, 5_000)]),
            _frame(
                [
                    _api(2, 2_000, name="cudaGraphLaunch"),
                    _kernel(2, 9_000, graph="12"),
                    _kernel(3, 11_000),
                ]
            ),
        ]
    path = directory / "activity.sclz"
    path.write_bytes(MAGIC + b"".join(frames) + tail)
    (directory / "cupti_status.json").write_text(json.dumps(status or _status()))
    return path


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


def test_reads_launches_kernels_and_graph_ids(tmp_path: Path) -> None:
    trace = load_cupti_trace(_trace(tmp_path / "pid-42-abc"))

    assert trace.source == "stormlog-cupti"
    assert trace.cupti_version == "130001"
    assert trace.launches[(None, 1)].name == "cudaLaunchKernel"
    assert trace.launches[(None, 1)].pid == 42
    assert trace.launches[(None, 1)].ts_us == 1.0
    kernels = sorted(trace.gpu_events, key=lambda e: e.start_ns)
    assert [(k.correlation, k.graph_id, k.stream) for k in kernels] == [
        (1, None, 7),
        (2, 12, 7),
        (3, None, 7),
    ]
    assert kernels[1].graph_node_id == 3
    assert trace.event_loss == {
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "buffer_refusals": 0,
        "kernels_without_timestamps": 0,
    }


def test_an_api_record_without_a_name_falls_back_to_its_callback_id(
    tmp_path: Path,
) -> None:
    path = _trace(tmp_path, [_frame([_api(1, 1_000, name=None), _kernel(1, 5_000)])])

    assert load_cupti_trace(path).launches[(None, 1)].name == "runtime:cbid=211"


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (b"", "end frame is missing"),
        (END + b"x", "bytes after its end frame"),
        (struct.pack("<III", 50, 10, 1) + b"short", "truncated"),
        (struct.pack("<III", 0, 10, 1), "encoded size is invalid"),
    ],
)
def test_framing_defects_are_refused(tmp_path: Path, tail: bytes, message: str) -> None:
    path = _trace(tmp_path, [_frame([_kernel(1, 5_000)])], tail=tail)

    with pytest.raises(ValueError, match=message):
        list(iter_records(path))


def test_a_frame_whose_record_count_disagrees_is_refused(tmp_path: Path) -> None:
    frame = bytearray(_frame([_kernel(1, 5_000)]))
    frame[8:12] = struct.pack("<I", 2)

    with pytest.raises(ValueError, match="record count does not match"):
        list(iter_records(_trace(tmp_path, [bytes(frame)])))


def test_a_frame_that_decodes_past_its_declared_size_is_refused(
    tmp_path: Path,
) -> None:
    frame = bytearray(_frame([_kernel(1, 5_000)]))
    raw_size = struct.unpack("<I", frame[4:8])[0]
    frame[4:8] = struct.pack("<I", raw_size - 1)

    with pytest.raises(ValueError, match="declared size"):
        list(iter_records(_trace(tmp_path, [bytes(frame)])))


def test_a_file_without_the_magic_is_not_a_trace(tmp_path: Path) -> None:
    path = tmp_path / "activity.sclz"
    path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="not a Stormlog CUPTI trace"):
        list(iter_records(path))


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (_status(finalized=False), "did not finalize"),
        (
            _status(
                finalized=False,
                initialization_error="another CUPTI client holds this process",
            ),
            "another CUPTI client",
        ),
        (_status(buffer_refusals=-1), "invalid buffer_refusals"),
    ],
)
def test_an_unvouched_status_is_refused(
    tmp_path: Path, status: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_cupti_trace(_trace(tmp_path, status=status))


def test_a_missing_status_is_refused(tmp_path: Path) -> None:
    path = _trace(tmp_path)
    (tmp_path / "cupti_status.json").unlink()

    with pytest.raises(ValueError, match="cupti_status.json is missing"):
        load_cupti_trace(path)


def test_counted_losses_reach_the_import_summary(tmp_path: Path) -> None:
    path = _trace(
        tmp_path,
        [_frame([_api(1, 1_000), _kernel(1, 5_000), _kernel(2, None)])],
        status=_status(
            buffer_refusals=3, kernels_without_timestamps=1, output_limit_reached=True
        ),
    )

    capture = TraceFileCollector([path]).collect(run_id="r", session_id="s")

    (summary,) = (capture.summary or {})["traces"]
    assert summary["format"] == "stormlog-cupti"
    assert summary["event_loss"]["buffer_refusals"] == 3
    assert summary["event_loss_note"] is None
    assert summary["not_imported"] == {"kernel_without_timestamps": 1}
    assert any("buffer_refusals=3" in note for note in summary["notes"])
    assert any("STORMLOG_CUPTI_MAX_BYTES" in note for note in summary["notes"])


def test_cli_imports_a_trace_into_an_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(tmp_path / "cupti" / "pid-42-abc")

    code = main(
        [
            "import-trace",
            str(artifact),
            str(trace),
            "--device-uuid",
            "0=GPU-00000000-0000-0000-0000-000000000000",
        ]
    )

    assert code == int(ExitCode.OK)
    assert "3 GPU events" in capsys.readouterr().out
    refs = [
        r
        for r in load_inference_artifact(artifact)
        if isinstance(r, ActivityReferenceEvent)
    ]
    assert refs
    assert {r.context.source for r in refs} == {"stormlog-cupti"}


def test_cli_refuses_a_trace_from_a_failed_helper(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    trace = _trace(
        tmp_path,
        status=_status(finalized=False, initialization_error="helper built against"),
    )

    assert main(["import-trace", str(artifact), str(trace)]) == int(
        ExitCode.INVALID_INPUT
    )
