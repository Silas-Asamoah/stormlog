"""Failure-retaining runner for one native probe experiment trial."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import psutil

from .models import (
    ExperimentMode,
    ProcessRole,
    ProcessRoleSpec,
    ResultStatus,
    TrialSpec,
    WorkloadId,
)
from .normalization import validate_measurement_window, validate_unique_artifacts
from .preflight import write_manifest
from .workloads.vllm_open_loop import (
    CUPTI_FRAME_HEADER,
    CUPTI_TRACE_MAGIC,
    _inspect_cupti_trace,
)

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
    watchdog = _measurement_watchdog(spec, trial_directory)
    trial_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    if (
        spec.mode is ExperimentMode.DIRECT_CUPTI
        and spec.workload_id is not WorkloadId.VLLM
    ):
        configured_output = spec.command.environment.get("STORMLOG_CUPTI_OUTPUT_DIR")
        if not configured_output:
            raise ValueError("direct-CUPTI trial requires an output directory")
        output_directory = Path(configured_output)
        if not output_directory.is_relative_to(trial_directory.resolve()):
            raise ValueError("CUPTI output directory must be inside the trial")
        output_directory.mkdir(mode=0o700)
    logs_directory = trial_directory / "logs"
    logs_directory.mkdir(mode=0o700)
    stdout_path = logs_directory / "stdout.log"
    stderr_path = logs_directory / "stderr.log"
    started_at_ns = time.time_ns()
    status, return_code, resources, watchdog_observation = _execute(
        spec, stdout_path, stderr_path, watchdog
    )
    artifacts = _collect_artifacts(spec, trial_directory)
    validate_unique_artifacts(artifacts)
    unusable_cupti = False
    if (
        status is ResultStatus.PASS
        and spec.mode is ExperimentMode.DIRECT_CUPTI
        and spec.workload_id is not WorkloadId.VLLM
    ):
        for artifact in artifacts:
            if (
                artifact["artifact_id"] == "cupti-trace"
                and artifact["status"] == "present"
            ):
                if not _usable_cupti_microbench(Path(artifact["path"])):
                    artifact["status"] = "malformed"
                    unusable_cupti = True
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
    shutdown_failures = (
        _vllm_shutdown_failures(trial_directory / "vllm")
        if spec.workload_id is WorkloadId.VLLM
        else []
    )
    if status is ResultStatus.PASS and shutdown_failures:
        status = ResultStatus.PARTIAL
    pressure_unsupported = (
        spec.workload_id.value == "w4-stress"
        and spec.mode.value == "direct-cupti"
        and spec.pressure_controls.get("variant_status") == "unsupported"
    )
    if status is ResultStatus.PASS and pressure_unsupported:
        status = ResultStatus.UNSUPPORTED
    limitations = _limitations(status, return_code, missing_required, malformed_result)
    if unusable_cupti:
        limitations.append(
            "direct CUPTI trace lacked a valid timed device kernel capture"
        )
    limitations.extend(shutdown_failures)
    if pressure_unsupported:
        limitations.append(
            str(spec.pressure_controls.get("variant_reason", "W4 pressure unsupported"))
        )
    pressure_controls = dict(spec.pressure_controls)
    if watchdog_observation is not None:
        pressure_controls["watchdog_observation"] = watchdog_observation
        if watchdog_observation.get("error") and status in {
            ResultStatus.PASS,
            ResultStatus.UNSUPPORTED,
        }:
            status = ResultStatus.FAIL
            limitations.append(str(watchdog_observation["error"]))
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
        "loss": (workload_result.get("loss", {}) if workload_result else {}),
        "pressure_controls": pressure_controls,
        "artifacts": artifacts,
        "limitations": limitations,
    }
    write_manifest(trial_directory / "manifest.json", manifest)
    return manifest


def _execute(
    spec: TrialSpec,
    stdout_path: Path,
    stderr_path: Path,
    watchdog: tuple[Path, int] | None,
) -> tuple[ResultStatus, int | None, dict[str, Any], dict[str, Any] | None]:
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
            return ResultStatus.FAIL, None, resources, None
        timed_out, resources, watchdog_observation = _observe(
            process,
            started,
            spec.command.timeout_seconds,
            spec.process_roles,
            watchdog,
        )
    if process.returncode is None:
        raise RuntimeError("trial process did not reach a terminal state")
    status = ResultStatus.PASS if process.returncode == 0 else ResultStatus.FAIL
    if timed_out:
        status = ResultStatus.TIMEOUT
    return status, process.returncode, resources, watchdog_observation


def _measurement_watchdog(
    spec: TrialSpec, trial_directory: Path
) -> tuple[Path, int] | None:
    value = spec.pressure_controls.get("target_timeout_after_measurement_start_ms")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("measurement watchdog must be a positive integer")
    marker = spec.command.environment.get("STORMLOG_MEASUREMENT_START_FILE")
    expected = trial_directory.resolve() / "measurement-start.ns"
    if marker != str(expected):
        raise ValueError("measurement watchdog marker must be inside its trial")
    return expected, value


def _observe(
    process: subprocess.Popen[bytes],
    started: float,
    timeout_seconds: float,
    role_specs: tuple[ProcessRoleSpec, ...],
    watchdog: tuple[Path, int] | None,
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    root = psutil.Process(process.pid)
    roles = role_specs or (ProcessRoleSpec(ProcessRole.TARGET, "root"),)
    observed: dict[ProcessRole, ProcessMetrics] = {}
    system_start = psutil.cpu_times()
    timed_out = False
    watchdog_observation: dict[str, Any] | None = (
        {
            "configured_timeout_ms": watchdog[1],
            "marker_monotonic_ns": None,
            "fired": False,
            "error": None,
        }
        if watchdog is not None
        else None
    )
    while process.poll() is None:
        process_rows = _process_tree(root)
        for role in roles:
            selected = _select_role(process_rows, process.pid, role)
            if selected:
                observed[role.role] = _merge_metrics(
                    observed.get(role.role), _sample_processes(selected)
                )
        if watchdog is not None and watchdog_observation is not None:
            marker_path, delay_ms = watchdog
            if (
                watchdog_observation["marker_monotonic_ns"] is None
                and marker_path.exists()
            ):
                try:
                    marker_ns = int(marker_path.read_text(encoding="ascii").strip())
                    if marker_ns <= 0 or marker_ns > time.monotonic_ns():
                        raise ValueError("invalid measurement start time")
                    watchdog_observation["marker_monotonic_ns"] = marker_ns
                except (OSError, ValueError) as error:
                    watchdog_observation["error"] = (
                        f"invalid measurement marker: {error}"
                    )
                    _terminate(process)
                    break
            marker_ns = watchdog_observation["marker_monotonic_ns"]
            if (
                marker_ns is not None
                and time.monotonic_ns() >= marker_ns + delay_ms * 1_000_000
            ):
                timed_out = True
                watchdog_observation["fired"] = True
                _terminate(process)
                break
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            if watchdog_observation is not None:
                watchdog_observation["error"] = (
                    "overall trial timeout before measured watchdog"
                )
            _terminate(process)
            break
        time.sleep(_POLL_SECONDS)
    process.wait()
    if (
        watchdog_observation is not None
        and watchdog_observation["marker_monotonic_ns"] is None
        and watchdog_observation["error"] is None
    ):
        watchdog_observation["error"] = "measurement start marker missing"
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
    return timed_out, result, watchdog_observation


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


def _vllm_shutdown_failures(directory: Path) -> list[str]:
    """Retain known vLLM teardown failures even when the wrapper exits zero."""
    failures: list[str] = []
    exit_path = directory / "server-exit.json"
    try:
        exit_status = json.loads(exit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        failures.append("vLLM server exit status is missing or malformed")
    else:
        if not isinstance(exit_status, Mapping):
            failures.append("vLLM server exit status is not an object")
        else:
            if exit_status.get("return_code") != 0:
                failures.append("vLLM server did not exit successfully")
            if exit_status.get("profile_stop_completed") is not True:
                failures.append("vLLM profiler stop did not complete")
    markers = {
        "force killing remaining process": "vLLM force killed a process at shutdown",
        "EngineDeadError": "vLLM engine died at shutdown",
        "leaked semaphore objects": "vLLM leaked a semaphore at shutdown",
        "External init callback must run in same thread as registerClient": (
            "vLLM profiler reported a CUPTI initialization error"
        ),
    }
    for name in ("server-stdout.log", "server-stderr.log"):
        try:
            with (directory / name).open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    for marker, message in markers.items():
                        if marker in line and message not in failures:
                            failures.append(message)
        except OSError:
            failures.append(f"vLLM {name} is missing or unreadable")
    return failures


def _usable_chrome_trace(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    events = value.get("traceEvents") if isinstance(value, Mapping) else None
    return isinstance(events, list) and any(
        _is_device_activity_event(event) for event in events
    )


def _usable_cupti_microbench(directory: Path) -> bool:
    """Reject a finalized but empty injection trace from a successful GPU trial."""
    statuses = list(directory.glob("pid-*/cupti_status.json"))
    if len(statuses) != 1:
        return False
    trace = statuses[0].parent / "activity.sclz"
    try:
        report = json.loads(statuses[0].read_text(encoding="utf-8"))
        inspected = _inspect_cupti_trace(trace)
        if (
            not isinstance(report, Mapping)
            or report.get("finalized") is not True
            or report.get("initialization_error") is not None
            or report.get("delivered_records") != inspected["records"]
            or report.get("bytes_written") != inspected["encoded_bytes"]
        ):
            return False
        with trace.open("rb") as source:
            if source.read(len(CUPTI_TRACE_MAGIC)) != CUPTI_TRACE_MAGIC:
                return False
            while True:
                encoded_size, raw_size, records = CUPTI_FRAME_HEADER.unpack(
                    source.read(CUPTI_FRAME_HEADER.size)
                )
                if (encoded_size, raw_size, records) == (0, 0, 0):
                    return False
                for line in zlib.decompress(source.read(encoded_size)).splitlines():
                    row = json.loads(line)
                    if _is_cupti_kernel(row):
                        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError, zlib.error):
        return False


def _is_cupti_kernel(row: object) -> bool:
    if not isinstance(row, Mapping) or row.get("activity_kind") != "kernel":
        return False
    start = row.get("device_start_ns")
    end = row.get("device_end_ns")
    metadata = row.get("metadata")
    device = metadata.get("device_id") if isinstance(metadata, Mapping) else None
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and start < end
        and isinstance(device, int)
        and not isinstance(device, bool)
        and device >= 0
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
    if status in {ResultStatus.PASS, ResultStatus.UNSUPPORTED}:
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
