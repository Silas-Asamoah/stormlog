"""Failure-retaining runner for one native probe experiment trial."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import psutil

from .models import ResultStatus, TrialSpec
from .preflight import write_manifest

_POLL_SECONDS = 0.05
_SECRET_MARKERS = ("KEY", "PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")


@dataclass(frozen=True)
class ProcessMetrics:
    """Peak and cumulative resources observed for a process tree."""

    wall_time_ms: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    peak_rss_bytes: int
    peak_threads: int
    read_bytes: int | None
    write_bytes: int | None


def run_trial(spec: TrialSpec, output_root: Path) -> dict[str, Any]:
    """Execute one argv-only trial and preserve all outputs and failures."""
    _validate_environment(spec.command.environment)
    trial_directory = output_root / spec.configuration_id / "trials" / spec.trial_id
    trial_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs_directory = trial_directory / "logs"
    logs_directory.mkdir(mode=0o700)
    stdout_path = logs_directory / "stdout.log"
    stderr_path = logs_directory / "stderr.log"
    started_at_ns = time.time_ns()
    status, return_code, resources = _execute(spec, stdout_path, stderr_path)
    artifacts = [_artifact(stdout_path), _artifact(stderr_path)]
    metrics = _workload_metrics(stdout_path)
    metrics.update(
        {
            "wall_time_ms": resources.wall_time_ms,
            "target_cpu_user_seconds": resources.cpu_user_seconds,
            "target_cpu_system_seconds": resources.cpu_system_seconds,
            "target_peak_rss_bytes": float(resources.peak_rss_bytes),
            "target_peak_threads": float(resources.peak_threads),
            "target_read_bytes": _optional_float(resources.read_bytes),
            "target_write_bytes": _optional_float(resources.write_bytes),
        }
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "native_probe_trial",
        "trial_id": spec.trial_id,
        "configuration_id": spec.configuration_id,
        "workload_id": spec.workload_id,
        "mode": spec.mode.value,
        "repetition": spec.repetition,
        "status": status.value,
        "started_at_ns": started_at_ns,
        "finished_at_ns": time.time_ns(),
        "return_code": return_code,
        "command": {
            "argv": list(spec.command.argv),
            "environment": dict(sorted(spec.command.environment.items())),
            "timeout_seconds": spec.command.timeout_seconds,
        },
        "metrics": metrics,
        "artifacts": artifacts,
        "limitations": _limitations(status, return_code),
    }
    write_manifest(trial_directory / "manifest.json", manifest)
    return manifest


def _execute(
    spec: TrialSpec, stdout_path: Path, stderr_path: Path
) -> tuple[ResultStatus, int | None, ProcessMetrics]:
    environment = os.environ.copy()
    environment.update(spec.command.environment)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            spec.command.argv,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        metrics = _observe(process, started, spec.command.timeout_seconds)
    if process.returncode is None:
        raise RuntimeError("trial process did not reach a terminal state")
    status = ResultStatus.PASS if process.returncode == 0 else ResultStatus.FAIL
    if metrics[0]:
        status = ResultStatus.TIMEOUT
    return status, process.returncode, metrics[1]


def _observe(
    process: subprocess.Popen[bytes], started: float, timeout_seconds: float
) -> tuple[bool, ProcessMetrics]:
    root = psutil.Process(process.pid)
    peak_rss = 0
    peak_threads = 0
    cpu_user = 0.0
    cpu_system = 0.0
    read_bytes: int | None = 0
    write_bytes: int | None = 0
    timed_out = False
    while process.poll() is None:
        process_rows = _process_tree(root)
        sample = _sample_processes(process_rows)
        peak_rss = max(peak_rss, sample[0])
        peak_threads = max(peak_threads, sample[1])
        cpu_user = max(cpu_user, sample[2])
        cpu_system = max(cpu_system, sample[3])
        read_bytes = _maximum_optional(read_bytes, sample[4])
        write_bytes = _maximum_optional(write_bytes, sample[5])
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            _terminate(process)
            break
        time.sleep(_POLL_SECONDS)
    process.wait()
    elapsed_ms = (time.monotonic() - started) * 1_000
    return timed_out, ProcessMetrics(
        wall_time_ms=elapsed_ms,
        cpu_user_seconds=cpu_user,
        cpu_system_seconds=cpu_system,
        peak_rss_bytes=peak_rss,
        peak_threads=peak_threads,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
    )


def _process_tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return [root]


def _sample_processes(
    processes: list[psutil.Process],
) -> tuple[int, int, float, float, int | None, int | None]:
    rss = 0
    threads = 0
    user = 0.0
    system = 0.0
    read_bytes: int | None = 0
    write_bytes: int | None = 0
    for process in processes:
        try:
            memory = process.memory_info()
            cpu = process.cpu_times()
            io = process.io_counters()
            rss += memory.rss
            threads += process.num_threads()
            user += cpu.user
            system += cpu.system
            read_bytes = _add_optional(read_bytes, io.read_bytes)
            write_bytes = _add_optional(write_bytes, io.write_bytes)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        except AttributeError:
            read_bytes = None
            write_bytes = None
    return rss, threads, user, system, read_bytes, write_bytes


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _workload_metrics(stdout_path: Path) -> dict[str, float | None]:
    last_object: Mapping[str, Any] | None = None
    with stdout_path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(value, Mapping)
                and value.get("artifact_kind") == "workload_result"
            ):
                last_object = value
    if last_object is None:
        return {}
    raw_metrics = last_object.get("metrics", {})
    if not isinstance(raw_metrics, Mapping):
        return {}
    return {
        name: _metric_value(name, value)
        for name, value in raw_metrics.items()
        if isinstance(name, str)
    }


def _metric_value(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"workload metric {name!r} must be numeric or null")
    return float(value)


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _validate_environment(environment: Mapping[str, str]) -> None:
    for key in environment:
        if any(marker in key.upper() for marker in _SECRET_MARKERS):
            raise ValueError(f"refusing to persist secret-like environment key: {key}")


def _limitations(status: ResultStatus, return_code: int | None) -> list[str]:
    if status is ResultStatus.PASS:
        return []
    if status is ResultStatus.TIMEOUT:
        return ["trial exceeded its declared timeout; partial output was retained"]
    return [f"trial process exited with code {return_code}; output was retained"]


def _add_optional(left: int | None, right: int) -> int | None:
    return None if left is None else left + right


def _maximum_optional(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return max(left, right)


def _optional_float(value: int | None) -> float | None:
    return None if value is None else float(value)
