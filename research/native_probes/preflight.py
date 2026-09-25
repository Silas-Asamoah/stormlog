"""Read-only, vendor-aware capability discovery for experiment hosts."""

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
from typing import Any, Callable, Mapping, Sequence

from .models import ResultStatus

Probe = Callable[[Sequence[str]], tuple[bool, str]]
_TOOLS = (
    "nvcc",
    "nvidia-smi",
    "nsys",
    "ncu",
    "rocminfo",
    "rocm-smi",
    "hipconfig",
    "rocprofv3",
    "bpftool",
    "bpftrace",
    "vllm",
    "proton-viewer",
)


def collect_environment(
    host_id: str,
    repository: Path,
    *,
    command_probe: Probe | None = None,
    system: str | None = None,
) -> dict[str, Any]:
    """Return a manifest without equating an installed tool with usable hardware."""
    if not host_id or not host_id.strip():
        raise ValueError("host_id must not be empty")
    repo = repository.resolve(strict=True)
    probe = command_probe or _run_probe
    tools = _tool_inventory(probe)
    system_name = system or platform.system()
    nvidia = _nvidia_inventory(tools, probe)
    amd = _amd_inventory(tools, probe)
    ebpf = _ebpf_inventory(system_name)
    return {
        "schema_version": 2,
        "artifact_kind": "native_probe_environment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_id": host_id,
        "platform": {
            "system": system_name,
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": _repository_relative_executable(
                Path(sys.executable), repo
            ),
        },
        "source": _source_identity(repo),
        "accelerators": {"nvidia": nvidia, "amd": amd},
        "linux_ebpf": ebpf,
        "tools": tools,
        "mode_qualifications": _qualifications(nvidia, amd, ebpf, tools),
        "limitations": _limitations(nvidia, amd, ebpf, tools),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Create a new owner-only manifest without overwriting evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
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


def _tool_inventory(probe: Probe) -> dict[str, dict[str, Any]]:
    inventory = {}
    for name in _TOOLS:
        path = shutil.which(name)
        ok, version = probe((path or name, "--version")) if path else (False, "")
        inventory[name] = {
            "available": path is not None,
            "path": path,
            "version": version.splitlines()[0][:500] if ok and version else None,
        }
    return inventory


def _nvidia_inventory(
    tools: Mapping[str, Mapping[str, Any]], probe: Probe
) -> dict[str, Any]:
    if tools["nvidia-smi"]["available"]:
        driver_ok, output = probe(
            (
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            )
        )
    else:
        driver_ok, output = False, ""
    rows = (
        [line.strip() for line in output.splitlines() if line.strip()]
        if driver_ok
        else []
    )
    cupti_roots = (Path("/usr/local/cuda/extras/CUPTI"), Path("/opt/cuda/extras/CUPTI"))
    return {
        "cuda_toolkit_detected": tools["nvcc"]["available"],
        "cuda_toolkit_version": tools["nvcc"]["version"],
        "cupti_detected": any(path.exists() for path in cupti_roots),
        "cupti_version": None,
        "driver_detected": driver_ok,
        "driver_version": _csv_column(rows, 2),
        "gpu_detected": bool(rows),
        "device_count": len(rows),
        "device_models": _csv_column(rows, 0),
        "compute_capabilities": _csv_column(rows, 1),
        "runtime_initialization_usable": bool(rows),
        "nsight_systems_detected": tools["nsys"]["available"],
        "nsight_compute_detected": tools["ncu"]["available"],
        "vllm_detected": tools["vllm"]["available"],
        "proton_detected": tools["proton-viewer"]["available"],
    }


def _amd_inventory(
    tools: Mapping[str, Mapping[str, Any]], probe: Probe
) -> dict[str, Any]:
    runtime_ok, output = (
        probe(("rocminfo",)) if tools["rocminfo"]["available"] else (False, "")
    )
    models = (
        [
            line.split(":", 1)[1].strip()
            for line in output.splitlines()
            if line.strip().startswith("Marketing Name:")
        ]
        if runtime_ok
        else []
    )
    return {
        "rocm_detected": tools["hipconfig"]["available"]
        or tools["rocminfo"]["available"],
        "rocm_version": tools["hipconfig"]["version"],
        "rocprofiler_sdk_detected": tools["rocprofv3"]["available"],
        "rocprofv3_detected": tools["rocprofv3"]["available"],
        "gpu_detected": bool(models),
        "device_count": len(models),
        "device_models": models,
        "runtime_initialization_usable": runtime_ok and bool(models),
    }


def _ebpf_inventory(system_name: str) -> dict[str, Any]:
    linux = system_name == "Linux"
    return {
        "linux": linux,
        "kernel_version": platform.release() if linux else None,
        "btf_available": linux and Path("/sys/kernel/btf/vmlinux").is_file(),
        "tracefs_available": linux
        and any(
            Path(path).is_dir()
            for path in ("/sys/kernel/tracing", "/sys/kernel/debug/tracing")
        ),
        "lockdown_state": (
            _read_optional(Path("/sys/kernel/security/lockdown")) if linux else None
        ),
        "unprivileged_bpf_disabled": (
            _read_optional(Path("/proc/sys/kernel/unprivileged_bpf_disabled"))
            if linux
            else None
        ),
        "perf_event_paranoid": (
            _read_optional(Path("/proc/sys/kernel/perf_event_paranoid"))
            if linux
            else None
        ),
        "effective_capabilities": _effective_capabilities() if linux else {},
    }


def _qualifications(
    nvidia: Mapping[str, Any],
    amd: Mapping[str, Any],
    ebpf: Mapping[str, Any],
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "off": _qualification(
            True, (), "profiler-off control is available", ResultStatus.PASS, tools
        ),
        "public-pytorch-nvidia": _vendor_qualification(nvidia, (), "NVIDIA", tools),
        "public-pytorch-amd": _vendor_qualification(amd, (), "AMD", tools),
        "trusted-nvidia": _vendor_qualification(nvidia, ("nsys",), "NVIDIA", tools),
        "trusted-amd": _vendor_qualification(amd, ("rocprofv3",), "AMD", tools),
        "detailed-counter-nvidia": _vendor_qualification(
            nvidia, ("ncu",), "NVIDIA", tools
        ),
        "detailed-counter-amd": _vendor_qualification(
            amd, ("rocprofv3",), "AMD", tools
        ),
        "direct-cupti-nvidia": _vendor_qualification(
            nvidia, (), "NVIDIA CUPTI", tools, require="cupti_detected"
        ),
        "amd-rocprofiler": _vendor_qualification(amd, ("rocprofv3",), "AMD", tools),
        "ebpf-semantic-linux": _qualification(
            bool(ebpf["linux"]),
            ("bpftool",),
            "Linux eBPF host required",
            ResultStatus.UNTESTED,
            tools,
        ),
        "hybrid-cupti-ebpf-nvidia": _qualification(
            bool(nvidia["runtime_initialization_usable"] and ebpf["linux"]),
            ("bpftool",),
            "NVIDIA runtime and Linux eBPF required",
            ResultStatus.UNTESTED,
            tools,
        ),
        "proton-nvidia": _vendor_qualification(
            nvidia, ("vllm", "proton-viewer"), "NVIDIA", tools
        ),
    }


def _vendor_qualification(
    vendor: Mapping[str, Any],
    required: tuple[str, ...],
    label: str,
    tools: Mapping[str, Mapping[str, Any]],
    *,
    require: str | None = None,
) -> dict[str, Any]:
    usable = bool(vendor.get("runtime_initialization_usable"))
    if require:
        usable = usable and bool(vendor.get(require))
    return _qualification(
        usable,
        required,
        f"{label} hardware/runtime required",
        ResultStatus.UNTESTED,
        tools,
    )


def _qualification(
    usable: bool,
    required: tuple[str, ...],
    reason: str,
    status: ResultStatus,
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    present = [name for name in required if tools[name]["available"]]
    if not usable:
        result = ResultStatus.UNSUPPORTED
    elif len(present) != len(required):
        result, reason = (
            ResultStatus.UNSUPPORTED,
            "required vendor-specific tooling is absent",
        )
    else:
        result = status
        if status is ResultStatus.UNTESTED:
            reason = "requirements detected; runtime experiment has not run"
    return {
        "status": result.value,
        "reason": reason,
        "required_tools": list(required),
        "detected_tools": present,
    }


def _run_probe(argv: Sequence[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def _csv_column(rows: Sequence[str], index: int) -> list[str]:
    return [
        parts[index].strip() for row in rows if len(parts := row.split(",")) > index
    ]


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _effective_capabilities() -> dict[str, bool | None]:
    status = _read_optional(Path("/proc/self/status")) or ""
    value = next(
        (line.split()[1] for line in status.splitlines() if line.startswith("CapEff:")),
        None,
    )
    if value is None:
        return {
            name: None
            for name in ("CAP_BPF", "CAP_PERFMON", "CAP_SYS_PTRACE", "CAP_SYS_ADMIN")
        }
    mask = int(value, 16)
    return {
        "CAP_SYS_PTRACE": bool(mask & (1 << 19)),
        "CAP_SYS_ADMIN": bool(mask & (1 << 21)),
        "CAP_PERFMON": bool(mask & (1 << 38)),
        "CAP_BPF": bool(mask & (1 << 39)),
    }


def _source_identity(repository: Path) -> dict[str, Any]:
    return {
        "repository": _git(repository, "remote", "get-url", "origin"),
        "working_tree": repository.name,
        "revision": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "dirty": bool(_git(repository, "status", "--porcelain")),
    }


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _repository_relative_executable(executable: Path, repository: Path) -> str:
    try:
        return str(executable.resolve().relative_to(repository))
    except ValueError:
        return executable.name


def _limitations(
    nvidia: Mapping[str, Any],
    amd: Mapping[str, Any],
    ebpf: Mapping[str, Any],
    tools: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    limitations = []
    if not nvidia["runtime_initialization_usable"]:
        limitations.append("UNTESTED - NVIDIA HARDWARE/RUNTIME UNAVAILABLE")
    if not amd["runtime_initialization_usable"]:
        limitations.append("UNTESTED - AMD HARDWARE/RUNTIME UNAVAILABLE")
    if not ebpf["linux"]:
        limitations.append("UNTESTED - LINUX EBPF HOST UNAVAILABLE")
    if not tools["vllm"]["available"]:
        limitations.append("UNTESTED - VLLM UNAVAILABLE")
    return limitations
