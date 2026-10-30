"""Qualification checks A, C and D for `stormlog infer profile --trace cupti`.

Declared in benchmarks/native_probes/qualification_cupti_cli.json. Each check
serves vLLM with a helper build, drives it through the stormlog CLI, and keeps
what the CLI, the helper and the artifact reported; failures are recorded, never
dropped.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scripts.native_probes.remote import micro_fidelity
from scripts.native_probes.workloads import vllm_open_loop as v1
from scripts.native_probes.workloads import vllm_window

ROOT = Path("/home/v2")
PORT = 8000
ENDPOINT = f"http://127.0.0.1:{PORT}"


def _gpu_uuid() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]


def _helper_env(capture: Path, helper: Path) -> dict[str, str]:
    capture.mkdir(parents=True, mode=0o700)
    return vllm_window.mode_environment(
        "cupti-min", os.environ, capture_dir=capture, helper=helper
    )


@contextmanager
def _server(argv: list[str], env: dict[str, str], out: Path) -> Iterator[None]:
    with (
        (out / "server-stdout.log").open("xb") as stdout,
        (out / "server-stderr.log").open("xb") as stderr,
    ):
        server = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr, env=env, start_new_session=True
        )
        try:
            v1._wait_ready(ENDPOINT, server)
            yield
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def _profile(out: Path, *trace: str) -> dict[str, Any]:
    artifact = out / "infer.jsonl"
    command = [
        str(ROOT / "venv/bin/stormlog"),
        "infer",
        "profile",
        "--base-url",
        f"{ENDPOINT}/v1",
        "--model",
        v1.MODEL,
        "--concurrency",
        "8",
        "--warmup-requests",
        "20",
        "--requests",
        "400",
        "--output",
        str(artifact),
        "--trace-device-uuid",
        f"0={_gpu_uuid()}",
        *trace,
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=1800, check=False
    )
    (out / "cli-stdout.log").write_text(completed.stdout)
    (out / "cli-stderr.log").write_text(completed.stderr)
    records = []
    if artifact.exists():
        records = [json.loads(line) for line in artifact.read_text().splitlines()]
    windows = [r for r in records if r.get("event_type") == "infer.trace_window"]
    refs = [r for r in records if r.get("event_type") == "infer.activity_ref"]
    capabilities = [r for r in records if r.get("event_type") == "infer.capabilities"]
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
        "trace_windows": windows,
        "activity_refs": len(refs),
        "activity_ref_sample": refs[:1],
        "capabilities": capabilities,
    }


def _statuses(capture: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text()) for p in sorted(capture.rglob("cupti_status.json"))
    ]


def check_a(
    out: Path, helper: Path = ROOT / "libstormlog_cupti_cu13.so"
) -> dict[str, Any]:
    """End to end in CUDA-graph mode, the deployment configuration."""
    out.mkdir(parents=True)
    capture = out / "cupti"
    env = _helper_env(capture, helper)
    argv = vllm_window.server_argv("off", out, PORT, cuda_graphs=True)
    with _server(argv, env, out):
        sockets = [str(p.relative_to(capture)) for p in capture.rglob("stop.sock")]
        result = _profile(
            out,
            "--trace",
            "cupti",
            "--trace-dir",
            str(capture),
            "--trace-max-seconds",
            "20",
        )
    result["sockets_before_profile"] = sockets
    result["helper_status"] = _statuses(capture)
    return result


def check_c(
    out: Path, helper: Path = ROOT / "libstormlog_cupti_system.so"
) -> dict[str, Any]:
    """A helper built for another CUDA major refuses to capture."""
    out.mkdir(parents=True)
    capture = out / "cupti"
    env = _helper_env(capture, helper)
    argv = vllm_window.server_argv("off", out, PORT)
    with _server(argv, env, out):
        statuses = _statuses(capture)
        sockets = [str(p.relative_to(capture)) for p in capture.rglob("stop.sock")]
        result = _profile(out, "--trace", "cupti", "--trace-dir", str(capture))
    result["helper_status_at_ready"] = statuses
    result["sockets_at_ready"] = sockets
    return result


def check_d_kineto(out: Path) -> dict[str, Any]:
    """The helper and vLLM's torch profiler in one server."""
    out.mkdir(parents=True)
    (out / "kineto").mkdir(mode=0o700)
    capture = out / "cupti"
    env = _helper_env(capture, ROOT / "libstormlog_cupti_cu13.so")
    argv = vllm_window.server_argv("kineto", out, PORT)
    with _server(argv, env, out):
        statuses = _statuses(capture)
        sockets = [str(p.relative_to(capture)) for p in capture.rglob("stop.sock")]
        result = _profile(
            out,
            "--trace",
            "vllm-torch",
            "--trace-dir",
            str(out / "kineto"),
            "--trace-max-seconds",
            "10",
        )
    result["helper_status_at_ready"] = statuses
    result["sockets_at_ready"] = sockets
    result["helper_status_at_exit"] = _statuses(capture)
    return result


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"started_ns": time.time_ns()}
    checks = {
        "A_cli_end_to_end": lambda: check_a(out / "a"),
        "C_runtime_mismatch": lambda: check_c(out / "c"),
        "D_coexistence_kineto": lambda: check_d_kineto(out / "d-kineto"),
        "D_coexistence_nsys": lambda: micro_fidelity.nsys_trial(
            "w1-eager", out / "d-nsys", with_cupti=True
        ),
    }
    for name, check in checks.items():
        try:
            report[name] = check()
        except Exception as exc:  # recorded, so one failure does not hide others
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
        (out / "qualify.json").write_text(
            json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
        )
    print(json.dumps({"qualify": str(out / "qualify.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
