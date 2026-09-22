"""Opt-in orchestration for startup-injected CUPTI Activity capture."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import stat

# Targets are launched with an explicit argument vector and no shell.
import subprocess  # nosec B404
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .collector_health import CollectorHealthState
from .native_trace_store import (
    NativeTraceArtifact,
    NativeTraceIdentity,
    NativeTraceLoss,
    NativeTraceManifest,
    native_trace_artifact_from_file,
    register_native_trace_attachment,
    write_native_trace_manifest,
)

CUPTI_INJECTION_ENV = "CUDA_INJECTION64_PATH"
CUPTI_STATUS_FILENAME = "cupti_status.json"
CUPTI_TRACE_FILENAME = "activity.ndjson"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MAX_STATUS_BYTES = 1024 * 1024
DEFAULT_ACTIVITIES = ("driver", "runtime", "kernel", "memcpy", "memset")
SUPPORTED_ACTIVITIES = (*DEFAULT_ACTIVITIES, "synchronization")
_NATIVE_ENV_PREFIX = "STORMLOG_CUPTI_"
WaitOutcome = Literal["completed", "timed_out", "cancelled"]


def _validate_command_and_identity(
    command: Sequence[str], session_id: str, capture_id: str
) -> None:
    if not command or any(not value or "\0" in value for value in command):
        raise ValueError("command must contain non-empty arguments")
    if not session_id:
        raise ValueError("session_id is required")
    if capture_id and not _is_safe_identifier(capture_id):
        raise ValueError("capture_id contains unsupported characters")


def _validate_capture_limits(
    max_bytes: int, timeout_seconds: float | None, rank: int | None
) -> None:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if rank is not None and rank < 0:
        raise ValueError("rank must be >= 0")


def _validate_activities(activities: Sequence[str]) -> None:
    if not activities or any(not value for value in activities):
        raise ValueError("activities must contain non-empty values")
    if any(value not in SUPPORTED_ACTIVITIES for value in activities):
        raise ValueError("activities contain an unsupported CUPTI activity")
    if len(activities) != len(set(activities)):
        raise ValueError("activities must not contain duplicates")


@dataclass(frozen=True)
class CuptiCaptureConfig:
    """Validated launch configuration for one bounded CUPTI capture."""

    command: tuple[str, ...]
    injection_library: Path
    output_root: Path
    session_id: str
    capture_id: str = ""
    max_bytes: int = DEFAULT_MAX_BYTES
    activities: tuple[str, ...] = DEFAULT_ACTIVITIES
    timeout_seconds: float | None = None
    cwd: Path | None = None
    run_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    device_id: str | None = None

    def __post_init__(self) -> None:
        _validate_command_and_identity(self.command, self.session_id, self.capture_id)
        _validate_capture_limits(self.max_bytes, self.timeout_seconds, self.rank)
        _validate_activities(self.activities)


@dataclass(frozen=True)
class CuptiCaptureResult:
    """Completed launch result and its locally registered evidence."""

    capture_id: str
    capture_directory: Path
    manifest_path: Path
    attachment_path: Path
    target_returncode: int
    timed_out: bool
    cancelled: bool
    health: CollectorHealthState


def capture_cupti_activity(
    config: CuptiCaptureConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> CuptiCaptureResult:
    """Launch a command with the separately built CUPTI injection library."""
    _validate_supported_host()
    library = _validate_injection_library(config.injection_library)
    base_environment = _validated_base_environment(environment)
    capture_id = config.capture_id or f"cupti-{uuid.uuid4().hex}"
    capture_directory = _create_capture_directory(config.output_root, capture_id)
    launch_environment = _capture_environment(
        config,
        capture_directory,
        library,
        base_environment=base_environment,
    )
    started_epoch_ns = time.time_ns()
    try:
        process = subprocess.Popen(  # nosec B603
            config.command,
            cwd=config.cwd,
            env=launch_environment,
            shell=False,
            start_new_session=True,
        )
    except OSError:
        capture_directory.rmdir()
        raise
    wait_outcome = _wait_for_target(process, config.timeout_seconds)
    timed_out = wait_outcome == "timed_out"
    ended_epoch_ns = time.time_ns()
    status, status_error = _read_native_status(
        capture_directory,
        expected_pid=process.pid,
        expected_activities=config.activities,
    )
    artifacts, artifact_error = _capture_artifacts(capture_directory)
    health = _validated_capture_health(
        status,
        status_error,
        wait_outcome,
        artifacts,
        artifact_error,
    )
    loss = _capture_loss(status, artifacts, health)
    enabled = _status_string_tuple(status, "enabled_activities")
    manifest = NativeTraceManifest(
        capture_id=capture_id,
        backend="cupti_activity",
        identity=NativeTraceIdentity(
            run_id=config.run_id,
            session_id=config.session_id,
            job_id=config.job_id,
            rank=config.rank,
            pid=process.pid,
            device_id=config.device_id,
        ),
        helper_executable=str(library),
        helper_version=str(status.get("helper_version", "unknown")),
        started_ns=started_epoch_ns,
        ended_ns=ended_epoch_ns,
        clock_domains=("unix_epoch_ns", "cupti_timestamp_ns"),
        requested_activities=config.activities,
        enabled_activities=tuple(
            value for value in enabled if value in config.activities
        ),
        max_bytes=config.max_bytes,
        privilege="same-process-startup-injection",
        target_selector=f"launched-pid:{process.pid}",
        health=health,
        loss=loss,
        artifacts=artifacts,
        metadata={
            "target_executable": config.command[0],
            "target_argument_count": len(config.command) - 1,
            "target_returncode": process.returncode,
            "timed_out": timed_out,
            "cancelled": wait_outcome == "cancelled",
            "cupti_version": status.get("cupti_version"),
            "compiled_cupti_api_version": status.get("compiled_cupti_api_version"),
            "compiled_cuda_version": status.get("compiled_cuda_version"),
            "driver_version": status.get("driver_version"),
            "runtime_version": status.get("runtime_version"),
            "native_status_error": status_error,
            "native_artifact_error": artifact_error,
        },
    )
    manifest_path = write_native_trace_manifest(capture_directory, manifest)
    attachment_path = register_native_trace_attachment(
        capture_directory, manifest_path, manifest
    )
    return CuptiCaptureResult(
        capture_id=capture_id,
        capture_directory=capture_directory,
        manifest_path=manifest_path,
        attachment_path=attachment_path,
        target_returncode=int(process.returncode or 0),
        timed_out=timed_out,
        cancelled=wait_outcome == "cancelled",
        health=health,
    )


def _validate_supported_host() -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            "CUPTI startup injection is currently supported on Linux only"
        )


def _validate_injection_library(value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise ValueError("injection_library must be a regular, non-symlink file")
    path = expanded.resolve(strict=True)
    if not path.is_file():
        raise ValueError("injection_library must be a regular, non-symlink file")
    if path.stat().st_mode & 0o022:
        raise ValueError("injection_library must not be group- or world-writable")
    return path


def _is_safe_identifier(value: str) -> bool:
    return all(
        character.isalnum() or character in {"-", "_", "."} for character in value
    )


def _create_capture_directory(root: Path, capture_id: str) -> Path:
    root_path = root.expanduser().resolve()
    root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not root_path.is_dir():
        raise ValueError("output_root must be a directory")
    capture_directory = root_path / capture_id
    capture_directory.mkdir(mode=0o700)
    capture_directory.chmod(0o700)
    return capture_directory


def _validated_base_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    values = dict(os.environ if environment is None else environment)
    if CUPTI_INJECTION_ENV in values:
        raise ValueError(f"{CUPTI_INJECTION_ENV} is already set")
    for name in values:
        if name.startswith(_NATIVE_ENV_PREFIX):
            raise ValueError(
                f"reserved native capture environment is already set: {name}"
            )
    return values


def _capture_environment(
    config: CuptiCaptureConfig,
    capture_directory: Path,
    library: Path,
    *,
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    values = dict(base_environment)
    values.update(
        {
            CUPTI_INJECTION_ENV: str(library),
            f"{_NATIVE_ENV_PREFIX}OUTPUT_DIR": str(capture_directory),
            f"{_NATIVE_ENV_PREFIX}MAX_BYTES": str(config.max_bytes),
            f"{_NATIVE_ENV_PREFIX}ACTIVITIES": ",".join(config.activities),
        }
    )
    return values


def _wait_for_target(
    process: subprocess.Popen[Any], timeout_seconds: float | None
) -> WaitOutcome:
    try:
        process.wait(timeout=timeout_seconds)
        return "completed"
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return "timed_out"
    except KeyboardInterrupt:
        _terminate_process_group(process)
        return "cancelled"


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _read_native_status(
    capture_directory: Path,
    *,
    expected_pid: int,
    expected_activities: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
    path = capture_directory / CUPTI_STATUS_FILENAME
    try:
        payload = _load_native_status(path)
        _validate_native_status(payload)
        if payload.get("pid") != expected_pid:
            raise ValueError("status pid does not match the launched target")
        requested = _status_string_tuple(payload, "requested_activities")
        if requested != tuple(expected_activities):
            raise ValueError("status activities do not match the capture request")
        return payload, None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {}, f"native status unavailable: {exc}"


def _load_native_status(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        stat_result = os.fstat(handle.fileno())
        _validate_status_file(stat_result)
        content = handle.read(MAX_STATUS_BYTES + 1)
    if len(content.encode("utf-8")) > MAX_STATUS_BYTES:
        raise ValueError("status exceeds the maximum supported size")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("status must contain an object")
    return payload


def _validate_status_file(stat_result: os.stat_result) -> None:
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError("status must be a regular file")
    if stat_result.st_mode & 0o077:
        raise ValueError("status must be owner-only")
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        raise ValueError("status must be owned by the current user")
    if stat_result.st_nlink != 1:
        raise ValueError("status must have exactly one hard link")
    if stat_result.st_size > MAX_STATUS_BYTES:
        raise ValueError("status exceeds the maximum supported size")


def _validate_native_status(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "helper_version",
        "pid",
        "cupti_version",
        "compiled_cupti_api_version",
        "compiled_cuda_version",
        "driver_version",
        "runtime_version",
        "started_timestamp_ns",
        "ended_timestamp_ns",
        "requested_activities",
        "enabled_activities",
        "delivered_records",
        "cupti_dropped_records",
        "local_dropped_records",
        "bytes_written",
        "bytes_dropped",
        "finalized",
        "initialization_error",
    }
    _validate_status_shape(payload, expected)
    _validate_status_scalars(payload, expected)
    _validate_status_activities(payload)


def _validate_status_shape(payload: Mapping[str, Any], expected: set[str]) -> None:
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("status does not match schema version 1")


def _validate_status_scalars(payload: Mapping[str, Any], expected: set[str]) -> None:
    for name in expected - {
        "schema_version",
        "helper_version",
        "requested_activities",
        "enabled_activities",
        "finalized",
        "initialization_error",
        "driver_version",
        "runtime_version",
    }:
        _require_status_integer(payload, name)
    if payload["pid"] == 0:
        raise ValueError("status pid must be a positive integer")
    if not isinstance(payload["finalized"], bool):
        raise ValueError("status finalized must be a boolean")
    _validate_status_text_fields(payload)


def _validate_status_text_fields(payload: Mapping[str, Any]) -> None:
    if (
        not isinstance(payload.get("helper_version"), str)
        or not payload["helper_version"]
    ):
        raise ValueError("status helper_version must be a non-empty string")
    error = payload["initialization_error"]
    if error is not None and (not isinstance(error, str) or not error):
        raise ValueError("status initialization_error must be null or non-empty")
    for name in ("driver_version", "runtime_version"):
        version = payload[name]
        if version is not None and (not isinstance(version, str) or not version):
            raise ValueError(f"status {name} must be null or non-empty")


def _validate_status_activities(payload: Mapping[str, Any]) -> None:
    for name in ("requested_activities", "enabled_activities"):
        activities = _status_string_tuple(payload, name)
        if len(activities) != len(payload[name]) or any(
            value not in SUPPORTED_ACTIVITIES for value in activities
        ):
            raise ValueError(f"status {name} contains invalid activities")
    if payload["ended_timestamp_ns"] < payload["started_timestamp_ns"]:
        raise ValueError("status timestamp bounds are invalid")


def _require_status_integer(payload: Mapping[str, Any], name: str) -> None:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"status {name} must be a non-negative integer")


def _status_string_tuple(status: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = status.get(name, ())
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return ()
    return tuple(dict.fromkeys(value))


def _capture_health(
    status: Mapping[str, Any], status_error: str | None, wait_outcome: WaitOutcome
) -> CollectorHealthState:
    if wait_outcome == "timed_out":
        return _failed_health("target timed out before native capture completed")
    if wait_outcome == "cancelled":
        return CollectorHealthState(
            status="degraded",
            telemetry_partial=True,
            partial_fields=("native_trace",),
            last_error="native capture was cancelled",
            consecutive_failures=1,
        )
    if status_error is not None:
        return _failed_health(status_error)
    initialization_error = status.get("initialization_error")
    if initialization_error:
        return _failed_health(f"CUPTI initialization failed: {initialization_error}")
    if status.get("finalized") is not True:
        return _failed_health("native capture did not finalize")
    requested = set(_status_string_tuple(status, "requested_activities"))
    enabled = set(_status_string_tuple(status, "enabled_activities"))
    if enabled != requested:
        return CollectorHealthState(
            status="degraded",
            telemetry_partial=True,
            partial_fields=("native_trace",),
            last_error="some requested CUPTI activities could not be enabled",
            consecutive_failures=1,
        )
    dropped = _status_non_negative_int(status, "cupti_dropped_records")
    local_dropped = _status_non_negative_int(status, "local_dropped_records")
    if dropped + local_dropped:
        return CollectorHealthState(
            status="degraded",
            telemetry_partial=True,
            partial_fields=("native_trace",),
            last_error="native capture dropped activity records",
            consecutive_failures=1,
        )
    return CollectorHealthState()


def _validated_capture_health(
    status: Mapping[str, Any],
    status_error: str | None,
    wait_outcome: WaitOutcome,
    artifacts: Sequence[NativeTraceArtifact],
    artifact_error: str | None,
) -> CollectorHealthState:
    health = _capture_health(status, status_error, wait_outcome)
    if artifact_error is not None:
        return _failed_health(artifact_error)
    if health.status == "healthy" and any(
        artifact.kind == "native_trace_partial" for artifact in artifacts
    ):
        health = _partial_health("native capture retained a partial trace artifact")
    size_error = _validate_artifact_size(status, artifacts)
    return _failed_health(size_error) if size_error is not None else health


def _failed_health(message: str) -> CollectorHealthState:
    return CollectorHealthState(
        status="unhealthy",
        telemetry_partial=True,
        partial_fields=("native_trace",),
        last_error=message,
        consecutive_failures=1,
    )


def _partial_health(message: str) -> CollectorHealthState:
    return CollectorHealthState(
        status="degraded",
        telemetry_partial=True,
        partial_fields=("native_trace",),
        last_error=message,
        consecutive_failures=1,
    )


def _status_non_negative_int(status: Mapping[str, Any], name: str) -> int:
    value = status.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return int(value)


def _capture_artifacts(
    capture_directory: Path,
) -> tuple[tuple[NativeTraceArtifact, ...], str | None]:
    trace = capture_directory / CUPTI_TRACE_FILENAME
    if not trace.exists():
        partial = trace.with_name(f"{trace.name}.partial")
        if not partial.exists():
            return (), "native capture produced no trace artifact"
        relative = partial.name
        kind = "native_trace_partial"
    else:
        relative = trace.name
        kind = "native_trace"
    try:
        artifact = native_trace_artifact_from_file(
            capture_directory,
            relative,
            kind=kind,
            content_type="application/x-ndjson",
            sensitive_fields=("symbols", "addresses"),
        )
    except (OSError, ValueError) as exc:
        return (), f"native trace artifact rejected: {exc}"
    return (artifact,), None


def _capture_loss(
    status: Mapping[str, Any],
    artifacts: Sequence[NativeTraceArtifact],
    health: CollectorHealthState,
) -> NativeTraceLoss:
    delivered = _status_non_negative_int(status, "delivered_records")
    cupti_dropped = _status_non_negative_int(status, "cupti_dropped_records")
    local_dropped = _status_non_negative_int(status, "local_dropped_records")
    dropped = cupti_dropped + local_dropped
    bytes_written = sum(artifact.size_bytes for artifact in artifacts)
    bytes_dropped = _status_non_negative_int(status, "bytes_dropped")
    truncated = bool(dropped or bytes_dropped or health.telemetry_partial)
    flush_outcome = "complete" if not truncated else "partial"
    if not artifacts:
        flush_outcome = "failed"
    return NativeTraceLoss(
        delivered_records=delivered,
        dropped_records=dropped,
        bytes_written=bytes_written,
        bytes_dropped=bytes_dropped,
        truncated=truncated,
        flush_outcome=flush_outcome,
    )


def _validate_artifact_size(
    status: Mapping[str, Any], artifacts: Sequence[NativeTraceArtifact]
) -> str | None:
    if not status or len(artifacts) != 1:
        return None
    expected = _status_non_negative_int(status, "bytes_written")
    actual = artifacts[0].size_bytes
    if expected != actual:
        return (
            "native trace size does not match helper status: "
            f"expected {expected} bytes, found {actual}"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run an opt-in CUPTI capture around one target command."""
    parser = argparse.ArgumentParser(
        prog="stormlog native-trace",
        description="Launch a command with bounded CUPTI Activity capture.",
    )
    parser.add_argument("--injection-library", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--device-id", default=None)
    parser.add_argument(
        "--activity",
        action="append",
        choices=SUPPORTED_ACTIVITIES,
        help="Activity to collect. Repeat to replace the default activity set.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if not command:
        parser.error("a target command is required after --")
    try:
        result = capture_cupti_activity(
            CuptiCaptureConfig(
                command=command,
                injection_library=args.injection_library,
                output_root=args.output_dir,
                session_id=args.session_id,
                capture_id=args.capture_id,
                max_bytes=args.max_bytes,
                activities=tuple(args.activity or DEFAULT_ACTIVITIES),
                timeout_seconds=args.timeout,
                cwd=args.cwd,
                run_id=args.run_id,
                job_id=args.job_id,
                rank=args.rank,
                device_id=args.device_id,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"stormlog native-trace: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "capture_id": result.capture_id,
                "manifest": str(result.manifest_path),
                "health": result.health.status,
                "target_returncode": result.target_returncode,
                "timed_out": result.timed_out,
                "cancelled": result.cancelled,
            },
            sort_keys=True,
        )
    )
    if result.timed_out:
        return 124
    if result.cancelled:
        return 130
    if result.target_returncode < 0:
        return 128 - result.target_returncode
    return result.target_returncode


__all__ = [
    "CUPTI_INJECTION_ENV",
    "CUPTI_STATUS_FILENAME",
    "CUPTI_TRACE_FILENAME",
    "DEFAULT_ACTIVITIES",
    "MAX_STATUS_BYTES",
    "SUPPORTED_ACTIVITIES",
    "CuptiCaptureConfig",
    "CuptiCaptureResult",
    "capture_cupti_activity",
    "main",
]
