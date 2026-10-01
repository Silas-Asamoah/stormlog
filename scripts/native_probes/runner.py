"""Failure-retaining runner for one native probe experiment trial."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import psutil

from .models import ProcessRole, ProcessRoleSpec, ResultStatus, TrialSpec
from .normalization import validate_measurement_window, validate_unique_artifacts
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


def run_trial(
    spec: TrialSpec, output_root: Path, *, revision: str | None = None
) -> dict[str, Any]:
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
    artifacts = _collect_artifacts(spec, trial_directory)
    validate_unique_artifacts(artifacts)
    workload_result = _workload_result(stdout_path)
    metrics = dict(workload_result.get("metrics", {})) if workload_result else {}
    malformed_result = workload_result is None
    if status is ResultStatus.PASS and malformed_result:
        status = ResultStatus.PARTIAL
    missing_required = any(
        row["status"] != "present" and row["required"] for row in artifacts
    )
    if status is ResultStatus.PASS and missing_required:
        status = ResultStatus.PARTIAL
    manifest = {
        "schema_version": 2,
        "artifact_kind": "native_probe_trial",
        "trial_id": spec.trial_id,
        "revision": revision,
        "configuration_id": spec.configuration_id,
        "workload_id": spec.workload_id.value,
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
        "resources": resources,
        "measurement_window": (
            workload_result.get("measurement_window") if workload_result else None
        ),
        "ground_truth": (
            workload_result.get("ground_truth") if workload_result else None
        ),
        "loss": {},
        "pressure_controls": dict(spec.pressure_controls),
        "artifacts": artifacts,
        "limitations": _limitations(
            status, return_code, missing_required, malformed_result
        ),
    }
    write_manifest(trial_directory / "manifest.json", manifest)
    return manifest


def _execute(
    spec: TrialSpec, stdout_path: Path, stderr_path: Path
) -> tuple[ResultStatus, int | None, dict[str, Any]]:
    environment = os.environ.copy()
    environment.update(spec.command.environment)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                spec.command.argv,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except OSError as error:
            stderr.write(f"unable to start command: {error}\n".encode())
            unknown = {
                "status": "unknown",
                "wall_time_ms": None,
                "cpu_user_seconds": None,
                "cpu_system_seconds": None,
                "peak_rss_bytes": None,
                "peak_threads": None,
                "read_bytes": None,
                "write_bytes": None,
            }
            resources = {role.role.value: dict(unknown) for role in spec.process_roles}
            resources[ProcessRole.SYSTEM.value] = dict(unknown)
            return ResultStatus.FAIL, None, resources
        metrics = _observe(
            process, started, spec.command.timeout_seconds, spec.process_roles
        )
    if process.returncode is None:
        raise RuntimeError("trial process did not reach a terminal state")
    status = ResultStatus.PASS if process.returncode == 0 else ResultStatus.FAIL
    if metrics[0]:
        status = ResultStatus.TIMEOUT
    return status, process.returncode, metrics[1]


def _observe(
    process: subprocess.Popen[bytes],
    started: float,
    timeout_seconds: float,
    role_specs: tuple[ProcessRoleSpec, ...],
) -> tuple[bool, dict[str, Any]]:
    root = psutil.Process(process.pid)
    roles = role_specs or (ProcessRoleSpec(ProcessRole.TARGET, "root"),)
    observed: dict[ProcessRole, ProcessMetrics] = {}
    system_start = psutil.cpu_times()
    timed_out = False
    while process.poll() is None:
        process_rows = _process_tree(root)
        for role in roles:
            selected = _select_role(process_rows, process.pid, role)
            if selected:
                observed[role.role] = _merge_metrics(
                    observed.get(role.role), _sample_processes(selected)
                )
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            _terminate(process)
            break
        time.sleep(_POLL_SECONDS)
    process.wait()
    elapsed_ms = (time.monotonic() - started) * 1_000
    result = {
        role.role.value: _role_manifest(observed.get(role.role), elapsed_ms)
        for role in roles
    }
    system_end = psutil.cpu_times()
    result[ProcessRole.SYSTEM.value] = {
        "status": "observed",
        "wall_time_ms": elapsed_ms,
        "cpu_user_seconds": max(0.0, system_end.user - system_start.user),
        "cpu_system_seconds": max(0.0, system_end.system - system_start.system),
        "peak_rss_bytes": None,
        "peak_threads": None,
        "read_bytes": None,
        "write_bytes": None,
    }
    return timed_out, result


def _select_role(
    processes: list[psutil.Process], root_pid: int, spec: ProcessRoleSpec
) -> list[psutil.Process]:
    if spec.discovery == "root":
        return [row for row in processes if row.pid == root_pid]
    if spec.discovery == "descendant_argv_contains" and spec.argv_contains:
        return _matching_descendants(processes, root_pid, spec.argv_contains)
    return []


def _matching_descendants(
    processes: list[psutil.Process], root_pid: int, marker: str
) -> list[psutil.Process]:
    selected = []
    for row in processes:
        if row.pid == root_pid:
            continue
        try:
            command = row.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if any(marker in item for item in command):
            selected.append(row)
    return selected


def _merge_metrics(
    previous: ProcessMetrics | None,
    sample: tuple[int, int, float, float, int | None, int | None],
) -> ProcessMetrics:
    current = ProcessMetrics(
        0.0, sample[2], sample[3], sample[0], sample[1], sample[4], sample[5]
    )
    if previous is None:
        return current
    return ProcessMetrics(
        0.0,
        max(previous.cpu_user_seconds, current.cpu_user_seconds),
        max(previous.cpu_system_seconds, current.cpu_system_seconds),
        max(previous.peak_rss_bytes, current.peak_rss_bytes),
        max(previous.peak_threads, current.peak_threads),
        _maximum_optional(previous.read_bytes, current.read_bytes),
        _maximum_optional(previous.write_bytes, current.write_bytes),
    )


def _role_manifest(
    metrics: ProcessMetrics | None, wall_time_ms: float
) -> dict[str, Any]:
    if metrics is None:
        return {
            "status": "unknown",
            "wall_time_ms": None,
            "cpu_user_seconds": None,
            "cpu_system_seconds": None,
            "peak_rss_bytes": None,
            "peak_threads": None,
            "read_bytes": None,
            "write_bytes": None,
        }
    return {
        "status": "observed",
        "wall_time_ms": wall_time_ms,
        "cpu_user_seconds": metrics.cpu_user_seconds,
        "cpu_system_seconds": metrics.cpu_system_seconds,
        "peak_rss_bytes": metrics.peak_rss_bytes,
        "peak_threads": metrics.peak_threads,
        "read_bytes": metrics.read_bytes,
        "write_bytes": metrics.write_bytes,
    }


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


def _workload_result(stdout_path: Path) -> Mapping[str, Any] | None:
    last_object: Mapping[str, Any] | None = None
    with stdout_path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # Wrappers such as ncu may interleave human-readable progress
                # with the workload's structured result on stdout.
                continue
            if (
                isinstance(value, Mapping)
                and value.get("artifact_kind") == "workload_result"
            ):
                if not _valid_workload_result(value):
                    return None
                last_object = value
    return last_object


def _valid_workload_result(value: Mapping[str, Any]) -> bool:
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping) or not _valid_workload_metrics(metrics):
        return False
    window = value.get("measurement_window")
    return isinstance(window, Mapping) and not validate_measurement_window(window)


def _valid_workload_metrics(metrics: Mapping[Any, Any]) -> bool:
    try:
        for name, metric in metrics.items():
            if not isinstance(name, str):
                return False
            _metric_value(name, metric)
    except ValueError:
        return False
    return True


def _metric_value(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"workload metric {name!r} must be numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"workload metric {name!r} must be finite")
    return numeric


def _artifact(
    path: Path,
    *,
    artifact_id: str,
    kind: str,
    producer: str,
    format_name: str,
    required: bool,
    sensitive: bool,
    loss_metadata_expected: bool,
) -> dict[str, Any]:
    checksum, size = _artifact_digest(path)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "path": str(path),
        "sha256": checksum,
        "bytes": size,
        "producer": producer,
        "format": format_name,
        "required": required,
        "sensitive": sensitive,
        "loss_metadata_expected": loss_metadata_expected,
        "status": "present",
        "storage": "local",
        "durable_location": None,
    }


def _artifact_digest(path: Path) -> tuple[str, int]:
    """Hash files directly and directories through a canonical member manifest."""
    if path.is_file():
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    manifest: list[dict[str, Any]] = []
    total_size = 0
    for artifact_path in sorted(row for row in path.rglob("*") if row.is_file()):
        file_digest = hashlib.sha256()
        file_size = 0
        with artifact_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                file_digest.update(chunk)
                file_size += len(chunk)
        manifest.append(
            {
                "path": artifact_path.relative_to(path).as_posix(),
                "bytes": file_size,
                "sha256": file_digest.hexdigest(),
            }
        )
        total_size += file_size
    encoded = json.dumps(
        manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), total_size


def _collect_artifacts(spec: TrialSpec, trial_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected in spec.expected_artifacts:
        path = trial_directory / expected.relative_path
        if path.is_file() or path.is_dir():
            artifact = _artifact(
                path,
                artifact_id=expected.artifact_id,
                kind=expected.kind,
                producer=expected.producer,
                format_name=expected.format,
                required=expected.required,
                sensitive=expected.sensitive,
                loss_metadata_expected=expected.loss_metadata_expected,
            )
            if expected.format == "chrome-trace-json" and not _usable_chrome_trace(
                path
            ):
                artifact["status"] = "malformed"
            rows.append(artifact)
        else:
            rows.append(
                {
                    "artifact_id": expected.artifact_id,
                    "kind": expected.kind,
                    "path": str(path),
                    "sha256": None,
                    "bytes": None,
                    "producer": expected.producer,
                    "format": expected.format,
                    "required": expected.required,
                    "sensitive": expected.sensitive,
                    "loss_metadata_expected": expected.loss_metadata_expected,
                    "status": "missing",
                    "storage": "local",
                    "durable_location": None,
                }
            )
    return rows


def _usable_chrome_trace(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    events = value.get("traceEvents") if isinstance(value, Mapping) else None
    return isinstance(events, list) and any(
        _is_device_activity_event(event) for event in events
    )


def _is_device_activity_event(event: object) -> bool:
    """Accept complete GPU kernel events with timing and device identity.

    Chrome trace exporters commonly encode these as ``cat: kernel``,
    ``ph: X`` events with ``ts``/``dur`` and ``args.device``. The generic
    category covers CUDA and ROCProfiler without relying on vendor-specific
    kernel names.
    """
    if not isinstance(event, Mapping):
        return False
    category = event.get("cat")
    args = event.get("args")
    device = args.get("device") if isinstance(args, Mapping) else None
    category_text = category.lower() if isinstance(category, str) else ""
    return (
        event.get("ph") == "X"
        and "kernel" in category_text
        and _finite_number(event.get("ts"))
        and _positive_finite_number(event.get("dur"))
        and _has_device_identity(device)
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_finite_number(value: object) -> bool:
    return _finite_number(value) and isinstance(value, (int, float)) and value > 0


def _has_device_identity(device: object) -> bool:
    return (
        isinstance(device, int) and not isinstance(device, bool) and device >= 0
    ) or (isinstance(device, str) and bool(device.strip()))


def _validate_environment(environment: Mapping[str, str]) -> None:
    for key in environment:
        if any(marker in key.upper() for marker in _SECRET_MARKERS):
            raise ValueError(f"refusing to persist secret-like environment key: {key}")


def _limitations(
    status: ResultStatus,
    return_code: int | None,
    missing_required: bool,
    malformed_result: bool,
) -> list[str]:
    if status is ResultStatus.PASS:
        return []
    if missing_required and return_code == 0:
        return ["one or more required profiler artifacts were not usable"]
    if malformed_result and return_code == 0:
        return ["workload result is missing or malformed; output was retained"]
    if status is ResultStatus.TIMEOUT:
        return ["trial exceeded its declared timeout; partial output was retained"]
    if return_code is None:
        return ["trial process could not be started; failure evidence was retained"]
    return [f"trial process exited with code {return_code}; output was retained"]


def _add_optional(left: int | None, right: int) -> int | None:
    return None if left is None else left + right


def _maximum_optional(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return max(left, right)


def _optional_float(value: int | None) -> float | None:
    return None if value is None else float(value)
