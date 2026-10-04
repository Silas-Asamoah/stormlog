"""``stormlog infer describe-server``: one description of a running vLLM server.

It runs on the host that serves vLLM and puts together what that host can
observe: the server's processes from ``/proc``, its GPUs from NVML, its
model files, its start-up log and its Python packages. What vLLM reports
about itself over HTTP is the profile's probe, not this.

The description is one JSON document, ``stormlog.infer.server_description``
v1, with no credentials in it (see ``server_privacy``) and its own SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InferInputError, InferUsageError
from .host_clock import host_boot_id
from .server_gpu import GpuReader, NvmlGpuReader, describe_gpus
from .server_log import read_server_log
from .server_model import describe_model, hub_cache_dir, launch_arguments
from .server_privacy import (
    CREDENTIAL_PATHS_VERSION,
    RUNTIME_PACKAGES,
    redact_environ,
)
from .server_process import (
    API_SERVER,
    PROC,
    ProcessInfo,
    boot_time_s,
    process_tree,
    read_environ,
    read_process,
)

DESCRIPTION_FORMAT = "stormlog.infer.server_description"
DESCRIPTION_VERSION = 1

_PYTHON_TIMEOUT_SECONDS = 60
_NVIDIA_SMI_TIMEOUT_SECONDS = 20
# Run in the server's own interpreter: its Python and package versions.
_PYTHON_PROBE = (
    "import json, platform, importlib.metadata as m\n"
    "names = {names!r}\n"
    "found = {{}}\n"
    "for name in names:\n"
    "    try:\n"
    "        found[name] = m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        found[name] = None\n"
    "print(json.dumps({{'python': platform.python_version(), 'packages': found}}))\n"
)
PYTHON_PACKAGES = (*RUNTIME_PACKAGES, "vllm")


@dataclass(frozen=True)
class DescribeOptions:
    """What to read beyond the processes, and where."""

    pid: int
    server_log: Path | None = None
    python: str | None = "auto"
    hash_weights: bool = False
    verify_model_files: bool = False
    digest_cache: Path | None = None
    no_gpu: bool = False
    proc: Path = PROC


def describe_server(
    options: DescribeOptions,
    *,
    gpu_reader: Callable[[], GpuReader] = NvmlGpuReader,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """The description of the server rooted at ``options.pid``.

    Raises InferUsageError when the process cannot be read here, and
    InferInputError when a named log file cannot be read.
    """
    root = read_process(options.pid, options.proc)
    if root is None:
        raise InferUsageError(
            f"--pid {options.pid}: no such process in {options.proc}; run "
            "describe-server on the Linux host that serves vLLM"
        )
    tree = process_tree(options.pid, options.proc)
    environ = read_environ(options.pid, options.proc) or {}
    issues = _process_issues(root, environ)
    document: dict[str, Any] = {
        "format": DESCRIPTION_FORMAT,
        "version": DESCRIPTION_VERSION,
        "observed_at_ns": time.time_ns(),
        "redaction": CREDENTIAL_PATHS_VERSION,
        "host": _host(options.proc),
        "server": _server(root, tree, environ, options.proc),
        "gpus": _gpus(options, gpu_reader, tree),
        "model": _model(root, environ, options),
        "log": _log(options.server_log),
        "runtime": _runtime(options.python, root, run),
        "nvidia_smi": _nvidia_smi(run, options.no_gpu),
        "issues": issues,
    }
    document["sha256"] = description_digest(document)
    return document


def description_digest(document: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a description, its own digest left out."""
    content = {key: value for key, value in document.items() if key != "sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_description(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def load_description(path: Path) -> dict[str, Any]:
    """A description file, checked for its format and its own digest."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise InferInputError(f"server description {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("format") != DESCRIPTION_FORMAT:
        raise InferInputError(f"server description {path}: not {DESCRIPTION_FORMAT}")
    if document.get("version") != DESCRIPTION_VERSION:
        raise InferInputError(
            f"server description {path}: version {document.get('version')!r} "
            f"is not {DESCRIPTION_VERSION}"
        )
    if document.get("sha256") != description_digest(document):
        raise InferInputError(f"server description {path}: its sha256 does not match")
    return document


def _host(proc: Path) -> dict[str, Any]:
    describer = read_process(os.getpid(), proc)
    return {
        "hostname": socket.gethostname(),
        "boot_id": host_boot_id(),
        "boot_time_s": boot_time_s(proc),
        "nproc": os.cpu_count(),
        "describer_cpus_allowed_list": (
            describer.cpus_allowed_list if describer is not None else None
        ),
    }


def _server(
    root: ProcessInfo, tree: list[ProcessInfo], environ: Mapping[str, str], proc: Path
) -> dict[str, Any]:
    boot = boot_time_s(proc)
    return {
        "pid": root.pid,
        "start_ticks": root.start_ticks,
        "launch": launch_arguments(root.cmdline).to_record(),
        "processes": [info.to_record(boot_time_s=boot) for info in tree],
        "start_method": _start_method(tree, environ),
        "environ": redact_environ(environ),
    }


def _process_issues(root: ProcessInfo, environ: Mapping[str, str]) -> list[str]:
    issues = []
    if root.role != API_SERVER:
        issues.append(
            f"PID {root.pid} is a {root.role} process, not vLLM's API server; "
            "the description covers only it and its descendants"
        )
    if not environ:
        issues.append("the server's environment could not be read")
    return issues


def _start_method(
    tree: list[ProcessInfo], environ: Mapping[str, str]
) -> dict[str, Any]:
    """vLLM's worker start method, configured and as the processes show it."""
    spawned = any("multiprocessing.spawn" in " ".join(info.cmdline) for info in tree)
    return {
        "configured": environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
        "spawn_marker": spawned,
    }


def _gpus(
    options: DescribeOptions,
    gpu_reader: Callable[[], GpuReader],
    tree: list[ProcessInfo],
) -> dict[str, Any] | None:
    if options.no_gpu:
        return None
    reader = gpu_reader()
    try:
        return describe_gpus(reader, {info.pid for info in tree}).to_record()
    finally:
        reader.close()


def _model(
    root: ProcessInfo, environ: Mapping[str, str], options: DescribeOptions
) -> dict[str, Any]:
    launch = launch_arguments(root.cmdline)
    cwd = _cwd(root.pid, options.proc)
    return describe_model(
        launch,
        hub_cache=hub_cache_dir(environ, launch.download_dir),
        cwd=cwd,
        hash_weights=options.hash_weights,
        verify_blobs=options.verify_model_files,
        digest_cache=options.digest_cache,
    )


def _cwd(pid: int, proc: Path) -> Path | None:
    try:
        return Path(os.readlink(proc / str(pid) / "cwd"))
    except OSError:
        return None


def _log(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return {"path": str(path), **read_server_log(path)}
    except OSError as exc:
        raise InferInputError(f"--server-log {path}: {exc}") from exc


def _runtime(
    python: str | None,
    root: ProcessInfo,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any] | None:
    """Python and package versions, as the server's interpreter reports them."""
    interpreter = _interpreter(python, root)
    if interpreter is None:
        return None
    script = _PYTHON_PROBE.format(names=list(PYTHON_PACKAGES))
    try:
        result = run(
            [interpreter, "-c", script],
            capture_output=True,
            text=True,
            timeout=_PYTHON_TIMEOUT_SECONDS,
            check=False,
        )
        reported = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"interpreter": interpreter, "unavailable": str(exc)}
    if not isinstance(reported, dict):
        return {"interpreter": interpreter, "unavailable": result.stderr[-500:]}
    return {"interpreter": interpreter, **reported}


def _interpreter(python: str | None, root: ProcessInfo) -> str | None:
    """``auto`` is the interpreter on the server's command line, if it shows one."""
    if python in (None, "none"):
        return None
    if python != "auto":
        return python
    first = root.cmdline[0] if root.cmdline else ""
    return first if Path(first).name.startswith("python") else None


def _nvidia_smi(
    run: Callable[..., subprocess.CompletedProcess[str]], no_gpu: bool
) -> dict[str, Any] | None:
    """The digest and size of ``nvidia-smi -q -x``, never its text."""
    if no_gpu:
        return None
    try:
        result = run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}
    if result.returncode != 0:
        return {"unavailable": f"exit {result.returncode}"}
    raw = result.stdout.encode()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


__all__ = [
    "DESCRIPTION_FORMAT",
    "DESCRIPTION_VERSION",
    "PYTHON_PACKAGES",
    "DescribeOptions",
    "describe_server",
    "description_digest",
    "load_description",
    "write_description",
]
