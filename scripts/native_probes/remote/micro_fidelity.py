"""Phase 3 on one GPU host: microbenchmark fidelity, induced loss, coexistence.

Each microbenchmark runs twice in separate processes: once under the CUPTI
helper (runtime and kernel activity, whole process) and once under Nsight
Systems. Kernels are attributed to the measured NVTX range through their host
launch in both tools, then compared with exact per-name counts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.native_probes import cupti_trace
from scripts.native_probes.nsys_reference import measured_kernels

ROOT = Path("/home/v2")
PYTHON = str(ROOT / "venv/bin/python")
HELPER = str(ROOT / "libstormlog_cupti.so")
WORKLOADS = ["w1-eager", "w2-overlap", "w2-serialized", "w3-graph"]


def _bench(workload: str, *extra: str) -> list[str]:
    return [
        PYTHON,
        "-m",
        "scripts.native_probes.workloads.cuda_microbench",
        "--workload",
        workload,
        *extra,
    ]


def _cupti_env(capture: Path, **overrides: str) -> dict[str, str]:
    capture.mkdir(parents=True, mode=0o700)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("STORMLOG_CUPTI_") and key != "CUDA_INJECTION64_PATH"
    }
    environment.update(
        {
            "CUDA_INJECTION64_PATH": HELPER,
            "STORMLOG_CUPTI_OUTPUT_DIR": str(capture),
            "STORMLOG_CUPTI_ACTIVITIES": "runtime,driver,kernel",
            "STORMLOG_CUPTI_MAX_BYTES": str(16 * 1024**3),
        }
    )
    environment.update(overrides)
    return environment


def _run(
    command: list[str], out: Path, environment: dict[str, str] | None = None
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT / "src",
        env=environment,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    (out / "stdout.log").write_text(completed.stdout)
    (out / "stderr.log").write_text(completed.stderr)
    result_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("{")
    ]
    return {
        "exit_code": completed.returncode,
        "result": json.loads(result_lines[-1]) if result_lines else None,
    }


def _statuses(capture: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text()) for p in sorted(capture.rglob("cupti_status.json"))
    ]


def cupti_trial(workload: str, out: Path, *extra: str, **env: str) -> dict[str, Any]:
    out.mkdir(parents=True)
    capture = out / "cupti"
    run = _run(_bench(workload, *extra), out, _cupti_env(capture, **env))
    statuses = _statuses(capture)
    traces = sorted(capture.rglob("activity.sclz"))
    summary: dict[str, Any] = {"run": run, "status": statuses}
    if run["result"] and traces:
        window = run["result"]["measurement_window"]
        summary["kernels"] = cupti_trace.cupti_summary(
            traces, window=(window["host_started_ns"], window["host_finished_ns"])
        )
    return summary


def nsys_trial(workload: str, out: Path, *, with_cupti: bool = False) -> dict[str, Any]:
    out.mkdir(parents=True)
    report = out / "trace"
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        f"--output={report}",
        *_bench(workload),
    ]
    environment = _cupti_env(out / "cupti") if with_cupti else None
    run = _run(command, out, environment)
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            f"--output={report}.sqlite",
            f"{report}.nsys-rep",
        ],
        capture_output=True,
        check=False,
    )
    summary: dict[str, Any] = {
        "run": run,
        "reference": measured_kernels(Path(f"{report}.sqlite")),
    }
    if with_cupti:
        summary["cupti_status"] = _statuses(out / "cupti")
    return summary


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"fidelity": {}, "loss": {}, "coexistence": {}}
    for workload in WORKLOADS:
        cupti = cupti_trial(workload, out / f"{workload}-cupti")
        nsys = nsys_trial(workload, out / f"{workload}-nsys")
        entry: dict[str, Any] = {"cupti": cupti, "nsys": nsys}
        reference = nsys["reference"]
        if "kernels" in cupti and reference.get("kernel_counts_by_name"):
            reference_summary = {
                "kernels": reference["measured_launches"],
                "correlation_coverage": 1.0
                - (
                    reference["missing_kernel_correlations"]
                    / max(1, reference["measured_host_launch_calls"])
                ),
                "graph_node_kernels": None,
                "kernel_counts_by_name": {
                    cupti_trace.short_kernel_name(name): count
                    for name, count in reference["kernel_counts_by_name"].items()
                },
            }
            entry["comparison"] = cupti_trace.compare(
                cupti["kernels"], reference_summary, exact=True
            )
        report["fidelity"][workload] = entry
    loss_cases: tuple[tuple[str, dict[str, str]], ...] = (
        ("nominal", {}),
        (
            "small-buffer-slow-consumer",
            {
                "STORMLOG_CUPTI_BUFFER_BYTES": "65536",
                "STORMLOG_CUPTI_CONSUMER_DELAY_MS": "200",
            },
        ),
        ("output-limit", {"STORMLOG_CUPTI_MAX_BYTES": "1048576"}),
    )
    for label, env in loss_cases:
        report["loss"][label] = cupti_trial("w4-stress", out / f"w4-{label}", **env)
    report["coexistence"]["nsys-plus-cupti"] = nsys_trial(
        "w1-eager", out / "coexist", with_cupti=True
    )
    (out / "phase3.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str)
    )
    print(json.dumps({"phase3": str(out / "phase3.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
