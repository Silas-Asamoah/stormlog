"""Argv-only command construction for independently measured profiler modes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .models import CommandSpec, ExperimentMode


@dataclass(frozen=True)
class Workload:
    """Parameters shared by profiler-off and profiler-on microbenchmarks."""

    workload_id: str
    warmup: int = 100
    iterations: int = 1_000
    matrix_size: int = 512
    seed: int = 118


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
        return _direct_cupti(base, artifact_directory, cupti_library)
    if mode is ExperimentMode.AMD_ROCPROFILER:
        return _rocprofiler_command(base, artifact_directory)
    if mode is ExperimentMode.DETAILED_COUNTER:
        return _counter_command(base, artifact_directory, vendor)
    raise ValueError(
        f"{mode.value} requires a session-managed engine, eBPF, or programmable "
        "probe adapter; use the exact commands in experiment_specification.md"
    )


def _workload_argv(workload: Workload) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "research.native_probes.workloads.cuda_microbench",
        "--workload",
        workload.workload_id,
        "--warmup",
        str(workload.warmup),
        "--iterations",
        str(workload.iterations),
        "--matrix-size",
        str(workload.matrix_size),
        "--seed",
        str(workload.seed),
    )


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
                f"--output={artifact_directory / 'nsight-system'}",
                *base,
            ),
            {},
            900.0,
        )
    if vendor == "amd":
        return _rocprofiler_command(base, artifact_directory)
    raise ValueError("vendor must be 'nvidia' or 'amd'")


def _direct_cupti(
    base: tuple[str, ...], artifact_directory: Path, library: Path
) -> CommandSpec:
    resolved_library = library.resolve(strict=True)
    return CommandSpec(
        base,
        {
            "CUDA_INJECTION64_PATH": str(resolved_library),
            "STORMLOG_CUPTI_OUTPUT_DIR": str(artifact_directory),
            "STORMLOG_CUPTI_MAX_BYTES": str(256 * 1024 * 1024),
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
            str(artifact_directory),
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
                str(artifact_directory / "nsight-compute"),
                *base,
            ),
            {},
            1_800.0,
        )
    if vendor == "amd":
        return _rocprofiler_command(base, artifact_directory)
    raise ValueError("vendor must be 'nvidia' or 'amd'")
