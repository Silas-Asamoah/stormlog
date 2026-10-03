"""Argv-only command construction for independently measured profiler modes."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from .models import (
    ArtifactExpectation,
    CommandSpec,
    ExperimentMode,
    ProcessRole,
    ProcessRoleSpec,
    WorkloadId,
)


@dataclass(frozen=True)
class Workload:
    """Parameters shared by profiler-off and profiler-on microbenchmarks."""

    workload_id: WorkloadId
    warmup: int = 100
    iterations: int = 1_000
    matrix_size: int = 512
    seed: int = 118
    launches_per_iteration: int = 100
    overlap_elements: int = 1_048_576
    overlap_operations: int = 8
    measurement_range_id: str = "stormlog-native-probe-measured"
    producer_buffer_bytes: int = 8 * 1024 * 1024
    transport_buffer_bytes: int = 8 * 1024 * 1024
    output_byte_bound: int = 256 * 1024 * 1024
    consumer_delay_ms: float = 0.0
    flush_interval_ms: float = 100.0
    postprocessor_delay_ms: float = 0.0


def microbenchmark_command(
    mode: ExperimentMode,
    workload: Workload,
    artifact_directory: Path,
    *,
    cupti_library: Path | None = None,
    vendor: str = "nvidia",
) -> CommandSpec:
    """Build a command for modes that can launch a bounded microbenchmark."""
    artifact_directory = artifact_directory.resolve()
    base = _workload_argv(workload)
    if mode is ExperimentMode.OFF:
        return CommandSpec(base, {}, 600.0)
    if mode is ExperimentMode.PUBLIC_PYTORCH:
        trace = artifact_directory / "pytorch-trace.json"
        return CommandSpec((*base, "--torch-trace", str(trace)), {}, 600.0)
    if mode is ExperimentMode.TRUSTED:
        return _trusted_command(base, artifact_directory, vendor)
    if mode is ExperimentMode.DIRECT_CUPTI:
        if cupti_library is None:
            raise ValueError("direct-cupti requires a pinned injection library")
        return _direct_cupti(
            base,
            artifact_directory,
            cupti_library,
            workload.output_byte_bound,
            workload.producer_buffer_bytes,
            workload.consumer_delay_ms,
        )
    if mode is ExperimentMode.AMD_ROCPROFILER:
        return _rocprofiler_command(base, artifact_directory)
    if mode is ExperimentMode.DETAILED_COUNTER:
        return _counter_command(base, artifact_directory, vendor)
    raise ValueError(
        f"{mode.value} requires a session-managed engine, eBPF, or programmable "
        "probe adapter; use the exact commands in benchmarks/native_probes/experiment.json"
    )


def vllm_command(
    mode: ExperimentMode,
    artifact_directory: Path,
    *,
    cupti_library: Path | None = None,
) -> CommandSpec:
    """Build a fresh-server, fixed-arrival vLLM trial command."""
    supported = {
        ExperimentMode.OFF,
        ExperimentMode.PUBLIC_ENGINE,
        ExperimentMode.PROTON,
        ExperimentMode.DIRECT_CUPTI,
        ExperimentMode.TRUSTED,
    }
    if mode not in supported:
        raise ValueError(f"{mode.value} has no vLLM session adapter")
    directory = artifact_directory.resolve()
    base = (
        sys.executable,
        "-m",
        "scripts.native_probes.workloads.vllm_open_loop",
        "--mode",
        mode.value,
        "--output",
        str(directory / "vllm"),
    )
    if mode is ExperimentMode.DIRECT_CUPTI:
        if cupti_library is None:
            raise ValueError("direct-cupti requires a pinned injection library")
        return _direct_cupti(
            base,
            directory / "vllm",
            cupti_library,
            256 * 1024 * 1024,
            8 * 1024 * 1024,
            0,
        )
    return CommandSpec(base, {}, 1800.0)


def vllm_expected_artifacts(mode: ExperimentMode) -> tuple[ArtifactExpectation, ...]:
    """Require request accounting and profiler output from the vLLM adapter."""
    common = (
        ArtifactExpectation(
            "vllm-request", "input", "vllm/request.json", "helper_agent", "json"
        ),
        ArtifactExpectation(
            "vllm-schedule", "input", "vllm/schedule.json", "helper_agent", "json"
        ),
        ArtifactExpectation(
            "vllm-requests", "raw_log", "vllm/requests.jsonl", "helper_agent", "jsonl"
        ),
        ArtifactExpectation(
            "vllm-warmup", "raw_log", "vllm/warmup.jsonl", "helper_agent", "jsonl"
        ),
        ArtifactExpectation(
            "vllm-server-stdout", "log", "vllm/server-stdout.log", "target", "text"
        ),
        ArtifactExpectation(
            "vllm-server-stderr", "log", "vllm/server-stderr.log", "target", "text"
        ),
        ArtifactExpectation(
            "vllm-server-command",
            "command",
            "vllm/server-command.json",
            "helper_agent",
            "json",
        ),
        ArtifactExpectation(
            "vllm-server-exit",
            "status",
            "vllm/server-exit.json",
            "helper_agent",
            "json",
        ),
        ArtifactExpectation(
            "vllm-versions", "status", "vllm/versions.json", "helper_agent", "json"
        ),
        ArtifactExpectation(
            "vllm-environment",
            "status",
            "vllm/server-environment.json",
            "helper_agent",
            "json",
        ),
        ArtifactExpectation(
            "vllm-resource-samples",
            "raw_log",
            "vllm/resource-samples.jsonl",
            "helper_agent",
            "jsonl",
        ),
    )
    profiler = {
        ExperimentMode.PUBLIC_ENGINE: ("public-engine", "target", "torch-profiler"),
        ExperimentMode.PROTON: ("proton", "target", "proton-trace"),
        ExperimentMode.DIRECT_CUPTI: ("cupti", "target", "stormlog-zlib-frames-v1"),
        ExperimentMode.TRUSTED: ("nsys", "profiler_wrapper", "vendor-native"),
    }
    if mode not in profiler:
        return common
    path, producer, format_name = profiler[mode]
    artifacts = (
        *common,
        ArtifactExpectation(
            "vllm-profiler",
            "raw_trace",
            f"vllm/{path}",
            producer,
            format_name,
            True,
            True,
            mode is ExperimentMode.DIRECT_CUPTI,
        ),
    )
    if mode is ExperimentMode.DIRECT_CUPTI:
        return (
            *artifacts,
            ArtifactExpectation(
                "vllm-cupti-stop",
                "status",
                "vllm/cupti-stop.json",
                "helper_agent",
                "json",
            ),
        )
    return artifacts


def vllm_process_roles(mode: ExperimentMode) -> tuple[ProcessRoleSpec, ...]:
    """Account for client, server, and optional profiler wrapper separately."""
    roles = [
        ProcessRoleSpec(ProcessRole.HELPER_AGENT, "root"),
        ProcessRoleSpec(ProcessRole.TARGET, "descendant_argv_contains", "vllm"),
    ]
    if mode is ExperimentMode.TRUSTED:
        roles.append(
            ProcessRoleSpec(
                ProcessRole.PROFILER_WRAPPER, "descendant_argv_contains", "nsys"
            )
        )
    return tuple(roles)


def _workload_argv(workload: Workload) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "scripts.native_probes.workloads.cuda_microbench",
        "--workload",
        workload.workload_id.value,
        "--warmup",
        str(workload.warmup),
        "--iterations",
        str(workload.iterations),
        "--matrix-size",
        str(workload.matrix_size),
        "--seed",
        str(workload.seed),
        "--launches-per-iteration",
        str(workload.launches_per_iteration),
        "--overlap-elements",
        str(workload.overlap_elements),
        "--overlap-operations",
        str(workload.overlap_operations),
        "--measurement-range-id",
        workload.measurement_range_id,
        "--producer-buffer-bytes",
        str(workload.producer_buffer_bytes),
        "--transport-buffer-bytes",
        str(workload.transport_buffer_bytes),
        "--output-byte-bound",
        str(workload.output_byte_bound),
        "--consumer-delay-ms",
        str(workload.consumer_delay_ms),
        "--flush-interval-ms",
        str(workload.flush_interval_ms),
        "--postprocessor-delay-ms",
        str(workload.postprocessor_delay_ms),
    )


def expected_artifacts(
    mode: ExperimentMode, vendor: str = "nvidia"
) -> tuple[ArtifactExpectation, ...]:
    """Declare raw outputs before a trial, including ownership and loss needs."""
    stdout = ArtifactExpectation(
        "workload-stdout", "log", "logs/stdout.log", "target", "jsonl", True, False
    )
    stderr = ArtifactExpectation(
        "workload-stderr", "log", "logs/stderr.log", "target", "text", True, False
    )
    trusted_path = "vendor-trace.nsys-rep" if vendor == "nvidia" else "rocprofiler"
    counter_path = "counter-report.ncu-rep" if vendor == "nvidia" else "rocprofiler"
    raw: dict[ExperimentMode, tuple[ArtifactExpectation, ...]] = {
        ExperimentMode.PUBLIC_PYTORCH: (
            ArtifactExpectation(
                "pytorch-trace",
                "raw_trace",
                "pytorch-trace.json",
                "target",
                "chrome-trace-json",
                True,
                True,
                False,
            ),
        ),
        ExperimentMode.TRUSTED: (
            ArtifactExpectation(
                "vendor-trace",
                "raw_trace",
                trusted_path,
                "profiler_wrapper",
                "vendor-native",
                True,
                True,
                True,
            ),
        ),
        ExperimentMode.DIRECT_CUPTI: (
            ArtifactExpectation(
                "cupti-trace",
                "raw_trace",
                "cupti",
                "target",
                "stormlog-zlib-frames-v1",
                True,
                True,
                True,
            ),
        ),
        ExperimentMode.AMD_ROCPROFILER: (
            ArtifactExpectation(
                "rocprofiler-trace",
                "raw_trace",
                "rocprofiler",
                "profiler_wrapper",
                "rocprofiler-csv",
                True,
                True,
                True,
            ),
        ),
        ExperimentMode.DETAILED_COUNTER: (
            ArtifactExpectation(
                "counter-report",
                "raw_report",
                counter_path,
                "profiler_wrapper",
                "vendor-native",
                True,
                True,
                False,
            ),
        ),
    }
    return (stdout, stderr, *raw.get(mode, ()))


def process_roles(mode: ExperimentMode) -> tuple[ProcessRoleSpec, ...]:
    """Return deterministic role discovery for one wrapper topology."""
    wrapped = {
        ExperimentMode.TRUSTED,
        ExperimentMode.AMD_ROCPROFILER,
        ExperimentMode.DETAILED_COUNTER,
    }
    if mode in wrapped:
        return (
            ProcessRoleSpec(ProcessRole.PROFILER_WRAPPER, "root"),
            ProcessRoleSpec(
                ProcessRole.TARGET, "descendant_argv_contains", "cuda_microbench"
            ),
        )
    return (ProcessRoleSpec(ProcessRole.TARGET, "root"),)


def _trusted_command(
    base: tuple[str, ...], artifact_directory: Path, vendor: str
) -> CommandSpec:
    if vendor == "nvidia":
        return CommandSpec(
            (
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--cuda-graph-trace=node",
                "--force-overwrite=false",
                f"--output={artifact_directory / 'vendor-trace'}",
                *base,
            ),
            {},
            900.0,
        )
    if vendor == "amd":
        return _rocprofiler_command(base, artifact_directory)
    raise ValueError("vendor must be 'nvidia' or 'amd'")


def _direct_cupti(
    base: tuple[str, ...],
    artifact_directory: Path,
    library: Path,
    output_byte_bound: int,
    activity_buffer_bytes: int,
    consumer_delay_ms: float,
) -> CommandSpec:
    if (
        activity_buffer_bytes < 65536
        or activity_buffer_bytes > 64 * 1024 * 1024
        or output_byte_bound <= 0
        or not math.isfinite(consumer_delay_ms)
        or consumer_delay_ms < 0
        or consumer_delay_ms > 1000
        or not float(consumer_delay_ms).is_integer()
    ):
        raise ValueError("invalid direct-CUPTI pressure control")
    resolved_library = library.resolve(strict=True)
    return CommandSpec(
        base,
        {
            "CUDA_INJECTION64_PATH": str(resolved_library),
            "STORMLOG_CUPTI_OUTPUT_DIR": str(artifact_directory / "cupti"),
            "STORMLOG_CUPTI_MAX_BYTES": str(output_byte_bound),
            "STORMLOG_CUPTI_BUFFER_BYTES": str(activity_buffer_bytes),
            "STORMLOG_CUPTI_CONSUMER_DELAY_MS": str(int(consumer_delay_ms)),
            "STORMLOG_CUPTI_ACTIVITIES": (
                "driver,runtime,kernel,memcpy,memset,synchronization"
            ),
        },
        900.0,
    )


def _rocprofiler_command(
    base: tuple[str, ...], artifact_directory: Path
) -> CommandSpec:
    return CommandSpec(
        (
            "rocprofv3",
            "--hip-trace",
            "--kernel-trace",
            "--memory-copy-trace",
            "--output-format",
            "csv",
            "--output-directory",
            str(artifact_directory / "rocprofiler"),
            "--",
            *base,
        ),
        {},
        900.0,
    )


def _counter_command(
    base: tuple[str, ...], artifact_directory: Path, vendor: str
) -> CommandSpec:
    if vendor == "nvidia":
        return CommandSpec(
            (
                "ncu",
                "--set",
                "basic",
                "--target-processes",
                "all",
                "--export",
                str(artifact_directory / "counter-report"),
                *base,
            ),
            {},
            1_800.0,
        )
    if vendor == "amd":
        return _rocprofiler_command(base, artifact_directory)
    raise ValueError("vendor must be 'nvidia' or 'amd'")
