"""Tests for JAX diagnostic bundle builder."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.jax.diagnose import (
    build_diagnostic_summary,
    collect_environment,
    run_diagnose,
    run_timeline_capture,
)
from stormlog.report import load_report
from tests.jax_test_helpers import fake_jax_runtime, jax_mark  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


@jax_mark
def test_collect_environment() -> None:
    """Verify collect_environment returns expected structure."""
    env = collect_environment(device_index=0)
    assert "system" in env
    assert "backend" in env
    assert "device" in env
    assert "fragmentation" in env
    assert "JAX does not expose fragmentation" in env["fragmentation"]["note"]


@jax_mark
def test_run_timeline_capture_zero_duration() -> None:
    """Verify run_timeline_capture with zero duration returns empty."""
    timeline = run_timeline_capture(0, 0.0, 0.5)
    assert timeline["timestamps"] == []
    assert timeline["allocated"] == []
    assert timeline["reserved"] == []


@jax_mark
def test_run_timeline_capture_positive_duration() -> None:
    """Verify a real capture distinguishes usable and unavailable metrics."""
    # Use a small interval to ensure at least one sample
    timeline = run_timeline_capture(0, 0.2, 0.05)
    if timeline["device_memory_available"]:
        assert len(timeline["timestamps"]) > 0
        assert len(timeline["allocated"]) > 0
        assert len(timeline["reserved"]) > 0
        assert len(timeline["allocated"]) == len(timeline["reserved"])
    else:
        assert timeline["timestamps"] == []
        assert timeline["allocated"] == []


@jax_mark
@mock.patch("stormlog.jax.diagnose.time.sleep")
@mock.patch("stormlog.jax.diagnose.MemoryTracker")
def test_run_timeline_capture_preserves_reserved_telemetry(
    mock_tracker_cls: Any, mock_sleep: Any
) -> None:
    """Verify timeline capture preserves JAX reserved bytes when available."""
    tracker = mock_tracker_cls.return_value
    tracker.stop_tracking.return_value = SimpleNamespace(
        timestamps=[1.0],
        memory_usage=[1000],
        telemetry_events=[
            {
                "event_type": "sample",
                "timestamp_ns": 2_000_000_000,
                "allocator_allocated_bytes": 1000,
                "allocator_reserved_bytes": 1500,
            }
        ],
    )

    timeline = run_timeline_capture(0, 0.2, 0.05)

    tracker.start_tracking.assert_called_once()
    tracker.stop_tracking.assert_called_once()
    mock_sleep.assert_called_once_with(0.2)
    assert timeline["timestamps"] == [2.0]
    assert timeline["allocated"] == [1000.0]
    assert timeline["reserved"] == [1500.0]


@jax_mark
@mock.patch("stormlog.jax.diagnose.get_device_info")
@mock.patch("stormlog.jax.diagnose.get_backend_info")
def test_build_diagnostic_summary(mock_backend: Any, mock_device: Any) -> None:
    """Verify build_diagnostic_summary returns a valid payload."""
    mock_backend.return_value = {"runtime_backend": "gpu"}
    mock_device.return_value = {
        "memory_stats": {
            "bytes_in_use": 1000,
            "peak_bytes_in_use": 2000,
            "bytes_limit": 10000,
        }
    }

    summary, risk_detected = build_diagnostic_summary(0)

    assert summary["backend"] == "gpu"
    assert summary["allocated_bytes"] == 1000
    assert summary["reserved_bytes"] == 1000
    assert summary["peak_bytes"] == 2000
    assert summary["total_bytes"] == 10000
    assert summary["allocator_gap_bytes"] == 0
    assert summary["allocator_reserved_approximate"] is True
    assert summary["utilization_ratio"] == 0.1
    assert summary["num_ooms"] == 0
    assert not risk_detected


@jax_mark
@mock.patch("stormlog.jax.diagnose.get_device_info")
@mock.patch("stormlog.jax.diagnose.get_backend_info")
def test_build_diagnostic_summary_uses_reserved_bytes(
    mock_backend: Any, mock_device: Any
) -> None:
    """Verify diagnostics keep reserved bytes distinct from allocated bytes."""
    mock_backend.return_value = {"runtime_backend": "gpu"}
    mock_device.return_value = {
        "memory_stats": {
            "bytes_in_use": 1000,
            "bytes_reserved": 1500,
            "peak_bytes_in_use": 2000,
            "bytes_limit": 10000,
        }
    }

    summary, risk_detected = build_diagnostic_summary(0)

    assert summary["allocated_bytes"] == 1000
    assert summary["reserved_bytes"] == 1500
    assert summary["allocator_gap_bytes"] == 500
    assert "allocator_reserved_approximate" not in summary
    assert not risk_detected


@jax_mark
@mock.patch("stormlog.jax.diagnose.get_device_info")
@mock.patch("stormlog.jax.diagnose.get_backend_info")
def test_build_diagnostic_summary_high_utilization(
    mock_backend: Any, mock_device: Any
) -> None:
    """Verify high utilization triggers memory risk."""
    mock_backend.return_value = {"runtime_backend": "gpu"}
    mock_device.return_value = {
        "memory_stats": {
            "bytes_in_use": 9000,
            "peak_bytes_in_use": 9000,
            "bytes_limit": 10000,
        }
    }

    summary, risk_detected = build_diagnostic_summary(0)
    assert risk_detected is True
    assert summary["risk_flags"]["high_utilization"] is True


@jax_mark
def test_run_diagnose(tmp_path: Path) -> None:
    """Verify full diagnose orchestrator produces valid artifacts."""
    artifact_dir, exit_code = run_diagnose(
        output=str(tmp_path),
        device_index=0,
        duration=0.1,
        interval=0.05,
        command_line="test_cmd",
    )

    assert artifact_dir.exists()
    assert artifact_dir.is_dir()
    assert exit_code in (ExitCode.OK, ExitCode.FINDINGS)

    # Check files
    env_file = artifact_dir / "environment.json"
    timeline_file = artifact_dir / "telemetry_timeline.json"
    summary_file = artifact_dir / "diagnostic_summary.json"
    manifest_file = artifact_dir / "manifest.json"

    assert env_file.exists()
    assert timeline_file.exists()
    assert summary_file.exists()
    assert manifest_file.exists()
    report = load_report(artifact_dir / "report.json")
    assert report["verdict"]["exit_code"] == exit_code
    assert report["tool"]["name"] == "jaxmemprof"
    assert "report.json" in {item["path"] for item in report["artifacts"]}

    with open(manifest_file) as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == 2
    assert "session_id" in manifest
    assert "test_cmd" in manifest["command_line"]
