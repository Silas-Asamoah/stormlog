"""Write protocol v2 field results into the Stormlog-validated matrix.

Only fields that v2 measured change. Each STORMLOG_VALIDATED cell links five
tracked evidence files by SHA-256; measured negative results are UNSUPPORTED or
VERSION_DEPENDENT with the measurement in the detail. Everything else keeps its
previous status.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "benchmarks/native_probes"
MATRIX = ROOT / BASE / "matrices/stormlog_validated.json"
SOURCE_MATRIX = ROOT / BASE / "matrices/source_backed.json"
V1_COMMIT = "dcb58a4"
DECISION = f"{BASE}/decision_2026-10-05.md"
SCOPE = (
    "Scope: one NVIDIA L4, driver 580.126.20, Linux 6.8 container as root, "
    "vLLM 0.30.0 V1 with Qwen2.5-0.5B, torch 2.13.0+cu130, helper linked to "
    "CUPTI 13.0.85."
)


def _role(role: str, path: str) -> dict[str, str]:
    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    return {"role": role, "path": path, "sha256": digest}


def _roles(analysis: str) -> list[dict[str, str]]:
    return [
        _role("environment", f"{BASE}/results_v2/environment.txt"),
        _role("command", f"{BASE}/protocol_v2.json"),
        _role("raw_artifact", f"{BASE}/results_v2/raw_manifest.txt"),
        _role("trial", f"{BASE}/results_v2/trials.jsonl"),
        _role("analysis", f"{BASE}/results_v2/{analysis}"),
    ]


def _claim(status: str, detail: str, analysis: str | None = None) -> dict:
    claim: dict = {
        "status": status,
        "detail": f"{detail} {SCOPE}",
        "evidence": [DECISION],
    }
    if status == "STORMLOG_VALIDATED":
        assert analysis is not None
        claim["evidence_roles"] = _roles(analysis)
    return claim


DIRECT_CUPTI = {
    "signal": _claim(
        "STORMLOG_VALIDATED",
        "Runtime, driver and concurrent-kernel activity records were captured "
        "in every v2 trial, with deferred start and stop controls.",
        "window_scaling.md",
    ),
    "linux": _claim(
        "STORMLOG_VALIDATED",
        "All v2 trials ran on Linux x86_64.",
        "window_scaling.md",
    ),
    "nvidia": _claim(
        "STORMLOG_VALIDATED",
        "Validated on an L4 (compute capability 8.9) only; other "
        "architectures are untested.",
        "window_scaling.md",
    ),
    "host_api_timing": _claim(
        "STORMLOG_VALIDATED",
        "Runtime and driver API records carry CPU start and end timestamps; "
        "kernels launched by cuBLASLt and Triton correlate only through "
        "driver records, so the driver activity kind is required.",
        "fidelity.json",
    ),
    "device_execution_timing": _claim(
        "STORMLOG_VALIDATED",
        "0 kernels without device timestamps across vLLM eager, vLLM CUDA "
        "graph and W1-W4 runs with CUPTI 13.0. With CUPTI 12.6 under the same "
        "CUDA 13 runtime, about 39% of kernels had none while 0 drops were "
        "reported.",
        "fidelity.json",
    ),
    "cpu_gpu_correlation": _claim(
        "STORMLOG_VALIDATED",
        "100% of kernels correlated to a runtime or driver launch in vLLM "
        "eager and graph runs and W1-W3, matching Nsight Systems.",
        "fidelity.json",
    ),
    "kernel_identity": _claim(
        "STORMLOG_VALIDATED",
        "Exact per-name kernel counts matched Nsight on W1 (3000), W2 "
        "(32000 each) and W3 (3000); vLLM kernel name sets matched in eager "
        "and graph mode.",
        "fidelity.json",
    ),
    "graph_replay": _claim(
        "STORMLOG_VALIDATED",
        "W3 graph replay matched Nsight exactly (3000 graph-node kernels); in "
        "vLLM CUDA-graph mode 93.5% of kernels carried graph IDs in both "
        "CUPTI and Nsight.",
        "fidelity.json",
    ),
    "streams": _claim(
        "UNKNOWN",
        "Stream IDs matched Nsight, but the W2 overlap workload produced no "
        "meaningful concurrent execution on the L4 in either tool, so overlap "
        "fidelity is untested.",
    ),
    "startup_injection": _claim(
        "STORMLOG_VALIDATED",
        "CUDA_INJECTION64_PATH injection reached the vLLM V1 EngineCore "
        "process in every CUPTI trial without engine changes.",
        "window_scaling.md",
    ),
    "engine_coupling": _claim(
        "STORMLOG_VALIDATED",
        "No vLLM code or configuration change was needed; capture is " "process-level.",
        "window_scaling.md",
    ),
    "runtime_coupling": _claim(
        "VERSION_DEPENDENT",
        "Measured: the helper must link the CUPTI matching the target "
        "process's CUDA runtime. CUPTI 12.6 under the torch cu130 runtime "
        "silently lost about 39% of kernel timestamps; CUPTI 13.0 lost none.",
    ),
    "buffer_loss_visibility": _claim(
        "UNSUPPORTED",
        "Measured gaps: output-limit drops were reported (1,277,656 records), "
        "but the helper has no buffer cap, so a slow consumer grew the backlog "
        "(72 MiB, 224 s flush) instead of dropping visibly, and under Nsight "
        "the helper captured 0-1 records while reporting a clean finalize. "
        "Timestamp loss is now counted by the helper.",
    ),
    "profiler_coexistence": _claim(
        "UNSUPPORTED",
        "With Nsight Systems on the same process, Nsight captured all 3000 "
        "W1 kernels and the helper captured 0-1 records per process with no "
        "error reported.",
    ),
}

PYTORCH_KINETO = {
    "linux": _claim(
        "STORMLOG_VALIDATED",
        "vLLM torch profiler trials ran on Linux x86_64.",
        "window_scaling.md",
    ),
    "nvidia": _claim(
        "STORMLOG_VALIDATED",
        "Validated on an L4 only.",
        "window_scaling.md",
    ),
    "late_attach": _claim(
        "STORMLOG_VALIDATED",
        "vLLM /start_profile and /stop_profile started and stopped capture on "
        "a running server in every Kineto trial.",
        "window_scaling.md",
    ),
}

CONTEXT = {
    "direct-cupti": {
        "status": "MEASURED_V2_DEFER",
        "detail": (
            "Protocol v2 measured direct CUPTI against vLLM's Kineto profiler "
            "across 50-2200 request windows (3 repetitions each), plus "
            "Nsight fidelity, induced loss, coexistence and ablations. "
            "Fidelity matched Nsight; p95 perturbation was +25-27% in eager "
            "mode and memory grew with the window because the helper's "
            "consumer could not keep up. Decision: defer. Post-hoc, in "
            "CUDA-graph mode at W=200, p95 perturbation was +0.8%."
        ),
        "evidence": [DECISION, f"{BASE}/protocol_v2.json"],
    },
    "pytorch-kineto": {
        "status": "MEASURED_V2",
        "detail": (
            "vLLM stack-free Kineto added 1.7, 7.3, 29.3 and 74.5 GiB RSS at "
            "50, 200, 800 and 2200 request windows; stop/export took 33, 138, "
            "551 s and timed out at 600 s, blocking serving."
        ),
        "evidence": [DECISION, f"{BASE}/results_v2/window_scaling.md"],
    },
}

NOT_RUN = "Not measured in protocol v1 or v2."
OTHER_CONTEXT = {
    "vllm-proton": (
        "Not measured in protocol v2. v1 bounded smokes are in git history at "
        f"commit {V1_COMMIT} and qualify no field."
    ),
    "ebpf-semantic": NOT_RUN,
    "cupti-ebpf-hybrid": NOT_RUN,
    "neutrino": NOT_RUN,
    "rocprofiler": f"{NOT_RUN} Requires AMD hardware (issue #235).",
    "no-native-collector": (
        "Protocol v2 ran profiler-off only as the baseline for the other modes. "
        "That does not show existing telemetry and imported traces answer the "
        "attribution question, so no field is qualified."
    ),
}

ASSUMPTIONS = {
    "direct-cupti": [
        "Request and phase attribution, pinned host memory, stream overlap, "
        "and non-root or other container deployments remain unqualified."
    ],
    "pytorch-kineto": [
        "Kineto trace completeness against Nsight and its cost in CUDA-graph "
        "mode remain unqualified."
    ],
}


def _unmeasured(source_status: str) -> dict:
    return {
        "status": "UNKNOWN",
        "detail": (
            "NO QUALIFYING STORMLOG RESULT FOR THIS FIELD. Not measured in "
            f"protocol v2. Source-backed status: {source_status}."
        ),
        "evidence": [DECISION],
    }


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    source = {
        candidate["id"]: candidate["claims"]
        for candidate in json.loads(SOURCE_MATRIX.read_text())["candidates"]
    }
    updates = {"direct-cupti": DIRECT_CUPTI, "pytorch-kineto": PYTORCH_KINETO}
    for candidate in matrix["candidates"]:
        measured = updates.get(candidate["id"], {})
        for field in candidate["claims"]:
            candidate["claims"][field] = measured.get(
                field, _unmeasured(source[candidate["id"]][field]["status"])
            )
        if candidate["id"] in CONTEXT:
            candidate["validation_context"] = CONTEXT[candidate["id"]]
        else:
            candidate["validation_context"] = {
                "status": "NOT_RUN_IN_V2",
                "detail": OTHER_CONTEXT[candidate["id"]],
                "evidence": [DECISION],
            }
        if candidate["id"] in ASSUMPTIONS:
            candidate["unverified_assumptions"] = ASSUMPTIONS[candidate["id"]]
    matrix["reviewed_on"] = "2026-10-05"
    matrix["qualification_rule"] = (
        "UNKNOWN means no qualifying Stormlog result for that field; it does "
        "not mean unsupported. STORMLOG_VALIDATED cells link five tracked v2 "
        "evidence files by SHA-256 and hold only within the stated scope. "
        "UNSUPPORTED and VERSION_DEPENDENT cells from v2 state the measurement "
        "that produced them."
    )
    MATRIX.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
