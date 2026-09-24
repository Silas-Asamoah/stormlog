"""Read-only capability and environment discovery for experiment hosts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import ExperimentMode, ResultStatus

_TOOLS_BY_MODE: Mapping[ExperimentMode, tuple[str, ...]] = {
    ExperimentMode.OFF: (),
    ExperimentMode.PUBLIC_PYTORCH: (),
    ExperimentMode.PUBLIC_ENGINE: ("vllm",),
    ExperimentMode.PROTON: ("proton-viewer",),
    ExperimentMode.TRUSTED: ("nsys", "rocprofv3"),
    ExperimentMode.EBPF_SEMANTIC: ("bpftool", "bpftrace"),
    ExperimentMode.DIRECT_CUPTI: ("nvcc",),
    ExperimentMode.HYBRID_CUPTI_EBPF: ("nvcc", "bpftool"),
    ExperimentMode.PROGRAMMABLE: ("neutrino",),
    ExperimentMode.AMD_ROCPROFILER: ("rocprofv3",),
    ExperimentMode.DETAILED_COUNTER: ("ncu", "rocprofv3"),
}


def collect_environment(host_id: str, repository: Path) -> dict[str, Any]:
    """Return a deterministic, secret-minimized environment manifest."""
    if not host_id or not host_id.strip():
        raise ValueError("host_id must not be empty")
    resolved_repository = repository.resolve(strict=True)
    tools = _tool_inventory(_all_tools())
    accelerator = _accelerator_inventory(tools)
    modes = {
        mode.value: _mode_capability(mode, tools, accelerator)
        for mode in ExperimentMode
    }
    return {
        "schema_version": 1,
        "artifact_kind": "native_probe_environment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_id": host_id,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": _repository_relative_executable(
                Path(sys.executable), resolved_repository
            ),
        },
        "source": _source_identity(resolved_repository),
        "accelerator": accelerator,
        "tools": tools,
        "modes": modes,
        "limitations": _limitations(accelerator, tools),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Create a new manifest without overwriting prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _all_tools() -> tuple[str, ...]:
    return tuple(sorted({tool for tools in _TOOLS_BY_MODE.values() for tool in tools}))


def _tool_inventory(tools: Iterable[str]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for tool in tools:
        path = shutil.which(tool)
        inventory[tool] = {
            "available": path is not None,
            "path": path,
        }
    return inventory


def _accelerator_inventory(tools: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    nvidia = (
        bool(tools.get("nvcc", {}).get("available")) and platform.system() != "Darwin"
    )
    amd = bool(tools.get("rocprofv3", {}).get("available"))
    return {
        "nvidia_toolchain_detected": nvidia,
        "amd_toolchain_detected": amd,
        "linux": platform.system() == "Linux",
    }


def _mode_capability(
    mode: ExperimentMode,
    tools: Mapping[str, Mapping[str, Any]],
    accelerator: Mapping[str, Any],
) -> dict[str, Any]:
    required = _TOOLS_BY_MODE[mode]
    present = [tool for tool in required if tools[tool]["available"]]
    status, reason = _mode_status(mode, required, present, accelerator)
    return {
        "status": status.value,
        "reason": reason,
        "required_tools": list(required),
        "detected_tools": present,
    }


def _mode_status(
    mode: ExperimentMode,
    required: tuple[str, ...],
    present: list[str],
    accelerator: Mapping[str, Any],
) -> tuple[ResultStatus, str]:
    if mode is ExperimentMode.OFF:
        return ResultStatus.PASS, "profiler-off control is available"
    if mode is ExperimentMode.PUBLIC_PYTORCH:
        return ResultStatus.UNTESTED, "framework import is checked by the workload"
    if mode in {ExperimentMode.EBPF_SEMANTIC, ExperimentMode.HYBRID_CUPTI_EBPF}:
        if not accelerator["linux"]:
            return ResultStatus.UNSUPPORTED, "Linux eBPF is unavailable on this host"
    if mode in {ExperimentMode.DIRECT_CUPTI, ExperimentMode.PROTON}:
        if not accelerator["nvidia_toolchain_detected"]:
            return ResultStatus.UNSUPPORTED, "NVIDIA CUDA/CUPTI is unavailable"
    if mode is ExperimentMode.AMD_ROCPROFILER:
        if not accelerator["amd_toolchain_detected"]:
            return ResultStatus.UNSUPPORTED, "AMD ROCProfiler is unavailable"
    if required and not present:
        return ResultStatus.UNSUPPORTED, "none of the required tools were detected"
    if len(present) != len(required):
        return ResultStatus.PARTIAL, "only part of the required toolchain was detected"
    return ResultStatus.UNTESTED, "tooling detected; runtime experiment has not run"


def _source_identity(repository: Path) -> dict[str, Any]:
    return {
        "repository": _git(repository, "remote", "get-url", "origin"),
        "working_tree": repository.name,
        "revision": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "dirty": bool(_git(repository, "status", "--porcelain")),
    }


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository_relative_executable(executable: Path, repository: Path) -> str:
    try:
        return str(executable.resolve().relative_to(repository))
    except ValueError:
        return executable.name


def _limitations(
    accelerator: Mapping[str, Any], tools: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    limitations: list[str] = []
    if not accelerator["nvidia_toolchain_detected"]:
        limitations.append("UNTESTED - NVIDIA HARDWARE/TOOLCHAIN UNAVAILABLE")
    if not accelerator["amd_toolchain_detected"]:
        limitations.append("UNTESTED - AMD HARDWARE/TOOLCHAIN UNAVAILABLE")
    if not accelerator["linux"]:
        limitations.append("UNTESTED - LINUX EBPF HOST UNAVAILABLE")
    if not tools.get("vllm", {}).get("available"):
        limitations.append("UNTESTED - VLLM UNAVAILABLE")
    return limitations
