from __future__ import annotations

import json
import os
import signal
import stat
import sys
from pathlib import Path
from unittest import mock

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from stormlog.native_trace_capture import (
    CUPTI_INJECTION_ENV,
    MAX_STATUS_BYTES,
    CuptiCaptureConfig,
    _read_native_status,
    _terminate_process_group,
    _wait_for_target,
    capture_cupti_activity,
)
from stormlog.native_trace_store import native_trace_artifact_from_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _injection_library(tmp_path: Path) -> Path:
    library = tmp_path / "libstormlog_cupti_injection.so"
    library.write_bytes(b"synthetic-test-library")
    library.chmod(0o555)
    return library


def _synthetic_target(
    *,
    status: bool = True,
    sleep: float = 0.0,
    partial: bool = False,
    reported_bytes: int = 21,
) -> tuple[str, ...]:
    source = (
        """
import json
import os
import pathlib
import time

root = pathlib.Path(os.environ["STORMLOG_CUPTI_OUTPUT_DIR"])
time.sleep(SLEEP)
trace = root / TRACE_NAME
trace.write_text('{"schema_version":1}\\n', encoding="utf-8")
trace.chmod(0o600)
if WRITE_STATUS:
    payload = {
        "schema_version": 1,
        "helper_version": "test",
        "pid": os.getpid(),
        "cupti_version": 12000,
        "compiled_cupti_api_version": 28,
        "compiled_cuda_version": 12090,
        "driver_version": None,
        "runtime_version": None,
        "started_timestamp_ns": 1,
        "ended_timestamp_ns": 2,
        "requested_activities": ["runtime", "kernel", "memcpy"],
        "enabled_activities": ["runtime", "kernel", "memcpy"],
        "delivered_records": 1,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "bytes_written": REPORTED_BYTES,
        "bytes_dropped": 0,
        "finalized": True,
        "initialization_error": None,
    }
    target = root / "cupti_status.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    target.chmod(0o600)
""".replace(
            "SLEEP", repr(sleep)
        )
        .replace("WRITE_STATUS", repr(status))
        .replace(
            "TRACE_NAME",
            repr("activity.ndjson.partial" if partial else "activity.ndjson"),
        )
        .replace("REPORTED_BYTES", repr(reported_bytes))
    )
    return (sys.executable, "-c", source)


def test_capture_launches_without_shell_and_registers_manifest(tmp_path: Path) -> None:
    config = CuptiCaptureConfig(
        command=_synthetic_target(),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        capture_id="capture-1",
        session_id="session-1",
        activities=("runtime", "kernel", "memcpy"),
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        result = capture_cupti_activity(
            config, environment={"PATH": os.environ["PATH"]}
        )

    assert result.target_returncode == 0
    assert result.health.status == "healthy"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture"]["enabled_activities"] == [
        "runtime",
        "kernel",
        "memcpy",
    ]
    assert manifest["capture"]["target_selector"].startswith("launched-pid:")
    assert manifest["metadata"]["target_executable"] == config.command[0]
    assert manifest["metadata"]["target_argument_count"] == len(config.command) - 1
    assert manifest["artifacts"][0]["path"] == "activity.ndjson"
    assert result.attachment_path.name == "stormlog_attachments.json"


def test_capture_reports_missing_native_status_without_failing_target(
    tmp_path: Path,
) -> None:
    config = CuptiCaptureConfig(
        command=_synthetic_target(status=False),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        session_id="session-1",
        activities=("runtime", "kernel", "memcpy"),
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        result = capture_cupti_activity(
            config, environment={"PATH": os.environ["PATH"]}
        )

    assert result.target_returncode == 0
    assert result.health.status == "unhealthy"
    assert result.health.telemetry_partial is True
    assert "status unavailable" in (result.health.last_error or "")


def test_capture_never_reports_partial_artifact_as_healthy(tmp_path: Path) -> None:
    config = CuptiCaptureConfig(
        command=_synthetic_target(partial=True),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        session_id="session-1",
        activities=("runtime", "kernel", "memcpy"),
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        result = capture_cupti_activity(
            config, environment={"PATH": os.environ["PATH"]}
        )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.health.status == "degraded"
    assert manifest["artifacts"][0]["kind"] == "native_trace_partial"
    assert manifest["loss"]["truncated"] is True
    assert manifest["loss"]["flush_outcome"] == "partial"


def test_capture_rejects_trace_size_that_disagrees_with_status(tmp_path: Path) -> None:
    config = CuptiCaptureConfig(
        command=_synthetic_target(reported_bytes=20),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        session_id="session-1",
        activities=("runtime", "kernel", "memcpy"),
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        result = capture_cupti_activity(
            config, environment={"PATH": os.environ["PATH"]}
        )

    assert result.health.status == "unhealthy"
    assert "size does not match" in (result.health.last_error or "")


def test_capture_terminates_target_process_group_on_timeout(tmp_path: Path) -> None:
    config = CuptiCaptureConfig(
        command=_synthetic_target(sleep=30.0),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        session_id="session-1",
        timeout_seconds=0.05,
        activities=("runtime", "kernel", "memcpy"),
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        result = capture_cupti_activity(
            config, environment={"PATH": os.environ["PATH"]}
        )

    assert result.timed_out is True
    assert result.health.status == "unhealthy"
    assert result.target_returncode < 0


def test_wait_terminates_process_group_on_keyboard_interrupt() -> None:
    process = mock.Mock()
    process.pid = 123
    process.wait.side_effect = [KeyboardInterrupt(), None]

    with mock.patch("stormlog.native_trace_capture.os.killpg") as killpg:
        outcome = _wait_for_target(process, None)

    assert outcome == "cancelled"
    killpg.assert_called_once_with(123, signal.SIGTERM)


def test_process_exit_race_during_termination_is_not_an_error() -> None:
    process = mock.Mock()
    process.pid = 123

    with mock.patch(
        "stormlog.native_trace_capture.os.killpg", side_effect=ProcessLookupError
    ):
        _terminate_process_group(process)

    process.wait.assert_called_once_with()


def test_capture_rejects_existing_cuda_injection_environment(tmp_path: Path) -> None:
    config = CuptiCaptureConfig(
        command=(sys.executable, "--version"),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        session_id="session-1",
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        with pytest.raises(ValueError, match=CUPTI_INJECTION_ENV):
            capture_cupti_activity(
                config,
                environment={CUPTI_INJECTION_ENV: "/untrusted/injection.so"},
            )


def test_failed_process_launch_removes_empty_capture_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "captures"
    config = CuptiCaptureConfig(
        command=(str(tmp_path / "missing-executable"),),
        injection_library=_injection_library(tmp_path),
        output_root=output_root,
        capture_id="failed-launch",
        session_id="session-1",
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        with pytest.raises(OSError):
            capture_cupti_activity(config, environment={})

    assert list(output_root.iterdir()) == []


def test_capture_does_not_change_existing_output_root_mode(tmp_path: Path) -> None:
    output_root = tmp_path / "captures"
    output_root.mkdir(mode=0o750)
    output_root.chmod(0o750)
    config = CuptiCaptureConfig(
        command=_synthetic_target(),
        injection_library=_injection_library(tmp_path),
        output_root=output_root,
        session_id="session-1",
        activities=("runtime", "kernel", "memcpy"),
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Linux"
    ):
        capture_cupti_activity(config, environment={"PATH": os.environ["PATH"]})

    assert stat.S_IMODE(output_root.stat().st_mode) == 0o750


def test_native_status_read_is_bounded(tmp_path: Path) -> None:
    status = tmp_path / "cupti_status.json"
    status.write_bytes(b" " * (MAX_STATUS_BYTES + 1))
    status.chmod(0o600)

    payload, error = _read_native_status(
        tmp_path,
        expected_pid=123,
        expected_activities=("runtime",),
    )

    assert payload == {}
    assert "maximum supported size" in (error or "")


def test_capture_is_explicitly_unsupported_on_macos(tmp_path: Path) -> None:
    config = CuptiCaptureConfig(
        command=(sys.executable, "--version"),
        injection_library=_injection_library(tmp_path),
        output_root=tmp_path / "captures",
        session_id="session-1",
    )

    with mock.patch(
        "stormlog.native_trace_capture.platform.system", return_value="Darwin"
    ):
        with pytest.raises(RuntimeError, match="Linux only"):
            capture_cupti_activity(config)


def test_existing_native_artifact_must_be_owner_only(tmp_path: Path) -> None:
    trace = tmp_path / "activity.ndjson"
    trace.write_text("{}\n", encoding="utf-8")
    trace.chmod(0o644)

    with pytest.raises(ValueError, match="owner-only"):
        native_trace_artifact_from_file(tmp_path, trace.name)

    trace.chmod(0o600)
    artifact = native_trace_artifact_from_file(tmp_path, trace.name)
    assert artifact.size_bytes == 3
    assert stat.S_IMODE(trace.stat().st_mode) == 0o600


def test_existing_native_artifact_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.ndjson"
    target.write_text("{}\n", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "activity.ndjson"
    link.symlink_to(target.name)

    with pytest.raises(ValueError, match="symlink"):
        native_trace_artifact_from_file(tmp_path, link.name)


def test_existing_native_artifact_rejects_hard_link(tmp_path: Path) -> None:
    trace = tmp_path / "activity.ndjson"
    trace.write_text("{}\n", encoding="utf-8")
    trace.chmod(0o600)
    os.link(trace, tmp_path / "duplicate.ndjson")

    with pytest.raises(ValueError, match="hard link"):
        native_trace_artifact_from_file(tmp_path, trace.name)


def test_cupti_status_schema_accepts_native_contract() -> None:
    schema = json.loads(
        (REPO_ROOT / "docs/schemas/native_cupti_status_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": 1,
        "helper_version": "0.1.0",
        "pid": 123,
        "cupti_version": 120900,
        "compiled_cupti_api_version": 28,
        "compiled_cuda_version": 12090,
        "driver_version": None,
        "runtime_version": None,
        "started_timestamp_ns": 1,
        "ended_timestamp_ns": 2,
        "requested_activities": ["runtime", "kernel"],
        "enabled_activities": ["runtime", "kernel"],
        "delivered_records": 3,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "bytes_written": 120,
        "bytes_dropped": 0,
        "finalized": True,
        "initialization_error": None,
    }

    jsonschema.Draft202012Validator(schema).validate(payload)
