"""Protocol v2 window-scaling runner: capture isolation, phases and windows."""

from __future__ import annotations

import json
import socket
import sys
import textwrap
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from scripts.native_probes.workloads import vllm_window as v2

ROOT = Path(__file__).resolve().parents[1]

STUB_SERVER = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    STOP_SECONDS = float(sys.argv[2])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            if self.path == "/stop_profile":
                time.sleep(STOP_SECONDS)
            if self.path in ("/start_profile", "/stop_profile"):
                self.send_response(200)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"choices": [{"delta": {"content": "x"}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\\n\\n")
            self.wfile.write(b'data: {"usage": {"completion_tokens": 1}}\\n\\n')
            self.wfile.write(b"data: [DONE]\\n\\n")

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_capture_environment_is_never_inherited() -> None:
    inherited = {
        "PATH": "/usr/bin",
        "CUDA_INJECTION64_PATH": "/stale/libstormlog_cupti.so",
        "STORMLOG_CUPTI_ACTIVITIES": "driver",
    }
    for mode in ("off", "kineto", "nsys"):
        environment = v2.mode_environment(
            mode, inherited, capture_dir=Path("/c"), helper=None
        )
        assert "CUDA_INJECTION64_PATH" not in environment
        assert not any(key.startswith("STORMLOG_CUPTI_") for key in environment)
        assert environment["PATH"] == "/usr/bin"


def test_cupti_modes_defer_capture_until_started() -> None:
    environment = v2.mode_environment(
        "cupti-min", {}, capture_dir=Path("/c"), helper=Path("/h.so")
    )
    assert environment["CUDA_INJECTION64_PATH"] == "/h.so"
    assert environment["STORMLOG_CUPTI_DEFER_START"] == "1"
    assert environment["STORMLOG_CUPTI_ACTIVITIES"] == "runtime,driver,kernel"
    full = v2.mode_environment(
        "cupti-full", {}, capture_dir=Path("/c"), helper=Path("/h.so")
    )
    assert full["STORMLOG_CUPTI_ACTIVITIES"].split(",")[:2] == ["driver", "runtime"]
    with pytest.raises(ValueError):
        v2.mode_environment("cupti-min", {}, capture_dir=Path("/c"), helper=None)


def test_server_argv_modes(tmp_path: Path) -> None:
    eager = v2.server_argv("off", tmp_path, 8000)
    assert "--enforce-eager" in eager and "--profiler-config" not in eager
    assert "--enforce-eager" not in v2.server_argv(
        "off", tmp_path, 8000, cuda_graphs=True
    )
    kineto = v2.server_argv("kineto", tmp_path, 8000)
    config = json.loads(kineto[kineto.index("--profiler-config") + 1])
    assert config["profiler"] == "torch"
    assert config["torch_profiler_with_stack"] is False
    assert config["torch_profiler_dir"] == str(tmp_path / "kineto")
    nsys = v2.server_argv("nsys", tmp_path, 8000)
    assert nsys[0] == "nsys" and "--capture-range=cudaProfilerApi" in nsys


def test_window_indices_use_request_index_not_position() -> None:
    rows = [{"request_id": f"stormlog-118-measured-{i:04d}"} for i in (3, 0, 2, 1)]
    assert v2.window_indices(rows, 2) == [1, 3]


def test_memory_summary_assigns_phases_and_growth() -> None:
    phases = {
        "capture_start_requested_ns": 10,
        "stop_requested_ns": 20,
        "stop_acknowledged_ns": 30,
    }
    samples: list[dict[str, Any]] = [
        {"at_ns": 0, "target_rss_bytes": 100},
        {"at_ns": 5, "target_rss_bytes": 100},
        {"at_ns": 15, "target_rss_bytes": 150},
        {"at_ns": 25, "target_rss_bytes": 400},
        {"at_ns": 35, "target_rss_bytes": 120},
        {"at_ns": 36, "target_rss_bytes": None},
    ]
    summary = v2.memory_summary(samples, phases)
    assert summary["target_rss_baseline_median_bytes"] == 100
    assert summary["target_rss_stopping_peak_bytes"] == 400
    assert summary["target_rss_added_peak_bytes"] == 300
    assert summary["incomplete_samples"] == 1


def test_protocol_v2_matches_runner() -> None:
    protocol = json.loads(
        (ROOT / "benchmarks/native_probes/protocol_v2.json").read_text()
    )
    assert set(protocol["window_scaling"]["modes"]) <= set(v2.MODES)
    assert protocol["window_scaling"]["windows"] == sorted(
        protocol["window_scaling"]["windows"]
    )
    assert protocol["supersedes"] == "benchmarks/native_probes/final_run_proposal.json"


def test_kineto_trial_samples_through_a_slow_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB_SERVER)
    port = _free_port()
    monkeypatch.setattr(
        v2,
        "server_argv",
        lambda *args, **kwargs: [sys.executable, str(stub), str(port), "1.0"],
    )
    monkeypatch.setattr(v2, "SETTLE_SECONDS", 0.5)
    args = v2.parse_args(
        [
            "--mode",
            "kineto",
            "--window",
            "5",
            "--warmup",
            "2",
            "--measured",
            "15",
            "--output",
            str(tmp_path / "trial"),
            "--port",
            str(port),
        ]
    )
    assert v2.run(args) == 0
    result = json.loads((tmp_path / "trial/result.json").read_text())
    assert result["ok"] is True
    assert result["stop_seconds"] >= 1.0
    assert result["latency_window"]["offered"] == 5
    assert result["latency_after_window"]["offered"] == 10
    assert result["memory"]["samples_stopping"] > 0
    assert result["memory"]["samples_baseline"] > 0
    assert len((tmp_path / "trial/requests.jsonl").read_text().splitlines()) == 15


def _write_trace(path: Path, records: list[dict[str, Any]]) -> None:
    import zlib

    from scripts.native_probes.workloads.vllm_open_loop import (
        CUPTI_FRAME_HEADER,
        CUPTI_TRACE_MAGIC,
    )

    raw = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    encoded = zlib.compress(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        CUPTI_TRACE_MAGIC
        + CUPTI_FRAME_HEADER.pack(len(encoded), len(raw), len(records))
        + encoded
        + CUPTI_FRAME_HEADER.pack(0, 0, 0)
    )


def _kernel(
    cid: str, start: int, end: int, stream: str, graph: str | None = None
) -> dict[str, Any]:
    return {
        "activity_kind": "kernel",
        "correlation_id": cid,
        "device_start_ns": start,
        "device_end_ns": end,
        "stream_id": stream,
        "graph_id": graph,
        "metadata": {"name": "void at::native::fill_kernel<float>(int)"},
    }


def test_cupti_summary_correlates_windows_and_overlap(tmp_path: Path) -> None:
    from scripts.native_probes import cupti_trace

    records = [
        {"activity_kind": "runtime", "correlation_id": "1", "cpu_start_ns": 5},
        {"activity_kind": "runtime", "correlation_id": "2", "cpu_start_ns": 15},
        {"activity_kind": "driver", "correlation_id": "3", "cpu_start_ns": 16},
        _kernel("1", 10, 20, "7"),
        _kernel("2", 30, 50, "7", graph="9"),
        _kernel("3", 40, 60, "8"),
        _kernel("4", 70, 80, "8"),
    ]
    trace = tmp_path / "pid-1-a" / "activity.sclz"
    _write_trace(trace, records)
    whole = cupti_trace.cupti_summary([trace])
    assert whole["kernels"] == 4
    assert whole["correlation_coverage"] == 0.75
    assert whole["graph_node_kernels"] == 1
    assert whole["stream_overlap_ns"] == 10
    assert whole["kernel_counts_by_name"] == {"fill_kernel": 4}
    windowed = cupti_trace.cupti_summary([trace], window=(10, 20))
    assert windowed["kernels"] == 2 and windowed["correlation_coverage"] == 1.0


def test_compare_requires_names_coverage_and_exact_counts() -> None:
    from scripts.native_probes import cupti_trace

    base = {
        "kernels": 2,
        "correlation_coverage": 1.0,
        "graph_node_kernels": 0,
        "kernel_counts_by_name": {"a": 1, "b": 1},
    }
    assert cupti_trace.compare(base, dict(base), exact=True)["pass"] is True
    other = {**base, "kernel_counts_by_name": {"a": 2, "b": 1}, "kernels": 3}
    assert cupti_trace.compare(base, other, exact=True)["pass"] is False
    assert cupti_trace.compare(base, other, exact=False)["pass"] is True
    missing = {**base, "kernel_counts_by_name": {"a": 2}}
    assert cupti_trace.compare(base, missing, exact=False)["pass"] is False


def test_rss_cap_kills_server_and_records_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB_SERVER)
    port = _free_port()
    monkeypatch.setattr(
        v2,
        "server_argv",
        lambda *args, **kwargs: [sys.executable, str(stub), str(port), "0"],
    )
    monkeypatch.setattr(v2, "SETTLE_SECONDS", 0.1)
    args = v2.parse_args(
        [
            "--mode",
            "off",
            "--window",
            "2",
            "--warmup",
            "1",
            "--measured",
            "5",
            "--rss-cap-bytes",
            "1",
            "--output",
            str(tmp_path / "trial"),
            "--port",
            str(port),
        ]
    )
    assert v2.run(args) == 1
    result = json.loads((tmp_path / "trial/result.json").read_text())
    assert "rss_cap_exceeded" in result["errors"]
    assert "rss_cap_exceeded_ns" in result["phases_ns"]


@pytest.mark.parametrize("name", ["stormlog_validated", "source_backed"])
def test_matrices_match_schema(name: str) -> None:
    schema = json.loads(
        (
            ROOT / "benchmarks/native_probes/schemas/capability_matrix.schema.json"
        ).read_text()
    )
    matrix = json.loads(
        (ROOT / f"benchmarks/native_probes/matrices/{name}.json").read_text()
    )
    jsonschema.validate(matrix, schema)


def test_validated_cells_link_existing_evidence_by_checksum() -> None:
    import hashlib

    matrix = json.loads(
        (ROOT / "benchmarks/native_probes/matrices/stormlog_validated.json").read_text()
    )
    for candidate in matrix["candidates"]:
        for field, claim in candidate["claims"].items():
            if claim["status"] != "STORMLOG_VALIDATED":
                continue
            roles = {row["role"]: row for row in claim["evidence_roles"]}
            assert set(roles) == {
                "environment",
                "command",
                "raw_artifact",
                "trial",
                "analysis",
            }, (candidate["id"], field)
            for row in roles.values():
                digest = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
                assert digest == row["sha256"], (candidate["id"], field, row["path"])
