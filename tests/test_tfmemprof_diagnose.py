"""Tests for tfmemprof diagnose command."""

import json
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import tfmemprof.cli as tfmemprof_cli
import tfmemprof.diagnose as diagnose_module


def _patch_tfmemprof_diagnose_env(
    monkeypatch: pytest.MonkeyPatch,
    gpu_available: bool = False,
    risk_detected: bool = False,
) -> None:
    """Patch diagnose module's env/summary dependencies for predictable output."""
    monkeypatch.setattr(
        diagnose_module,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "python_version": "3.10",
            "tensorflow_version": "2.15.0",
            "gpu": {"available": gpu_available, "count": 1 if gpu_available else 0},
            "backend": {"runtime_backend": "cuda" if gpu_available else "cpu"},
        },
    )
    if gpu_available and risk_detected:
        gpu_info = {
            "available": True,
            "count": 1,
            "devices": [
                {
                    "id": 0,
                    "name": "GPU 0",
                    "current_memory_mb": 9000,
                    "peak_memory_mb": 10000,
                }
            ],
            "total_memory": 10000,
        }
    elif gpu_available:
        gpu_info = {
            "available": True,
            "count": 1,
            "devices": [
                {
                    "id": 0,
                    "name": "GPU 0",
                    "current_memory_mb": 500,
                    "peak_memory_mb": 1000,
                }
            ],
            "total_memory": 1000,
        }
    else:
        gpu_info = {"available": False, "error": "No GPU devices found", "devices": []}

    monkeypatch.setattr(diagnose_module, "get_gpu_info", lambda: gpu_info)
    monkeypatch.setattr(
        diagnose_module,
        "get_backend_info",
        lambda: {"runtime_backend": "cuda" if gpu_available else "cpu"},
    )


def _patch_tfmemprof_timeline_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch run_timeline_capture to return fixed data without starting a tracker."""

    def _fake_capture(
        device: object, duration_seconds: float, interval: float
    ) -> dict[str, list[object]]:
        if duration_seconds <= 0:
            return {"timestamps": [], "allocated": [], "reserved": []}
        return {
            "timestamps": [0.0, 0.5, 1.0],
            "allocated": [0, 1000000, 2000000],
            "reserved": [0, 1000000, 2000000],
        }

    monkeypatch.setattr(diagnose_module, "run_timeline_capture", _fake_capture)


def _patch_tfmemprof_build_summary(
    monkeypatch: pytest.MonkeyPatch, risk_detected: bool = False
) -> None:
    """Patch build_diagnostic_summary to return controlled (summary, risk_detected)."""

    def _fake_build(device: object) -> tuple[dict[str, object], bool]:
        summary = {
            "backend": "cuda",
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_bytes": 0,
            "total_bytes": 1,
            "utilization_ratio": 0.9 if risk_detected else 0.2,
            "fragmentation_ratio": 0,
            "num_ooms": 0,
            "risk_flags": {
                "oom_occurred": False,
                "high_utilization": risk_detected,
                "fragmentation_warning": False,
            },
            "suggestions": ["Suggestion 1"] if risk_detected else [],
        }
        return summary, risk_detected

    monkeypatch.setattr(diagnose_module, "build_diagnostic_summary", _fake_build)


def test_tfmemprof_diagnose_produces_artifact_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """tfmemprof diagnose with duration=0 produces directory with required files."""
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)

    args = SimpleNamespace(
        output=str(tmp_path),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    exit_code = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    assert exit_code in (0, 2)
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 1
    artifact_dir = dirs[0]
    assert artifact_dir.is_dir()
    assert "tfmemprof-diagnose-" in artifact_dir.name

    assert (artifact_dir / "environment.json").exists()
    assert (artifact_dir / "diagnostic_summary.json").exists()
    assert (artifact_dir / "telemetry_timeline.json").exists()
    assert (artifact_dir / "manifest.json").exists()

    with open(artifact_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert "files" in manifest
    assert "exit_code" in manifest
    assert "risk_detected" in manifest
    for name in manifest["files"]:
        assert (artifact_dir / name).exists()

    with open(artifact_dir / "diagnostic_summary.json") as f:
        summary = json.load(f)
    assert "backend" in summary
    assert "risk_flags" in summary
    assert "suggestions" in summary


def test_tfmemprof_diagnose_invalid_duration_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid --duration < 0 returns 1."""
    args = SimpleNamespace(
        output=None,
        device="/GPU:0",
        duration=-1,
        interval=0.5,
    )
    exit_code = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "duration" in err.lower()


def test_tfmemprof_diagnose_invalid_interval_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid --interval <= 0 returns 1."""
    args = SimpleNamespace(
        output=None,
        device="/GPU:0",
        duration=0,
        interval=0,
    )
    exit_code = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "interval" in err.lower()


def test_tfmemprof_diagnose_exit_code_zero_when_no_risk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When risk is false, cmd_diagnose returns 0."""
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=True, risk_detected=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)

    args = SimpleNamespace(
        output=str(tmp_path),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    exit_code = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 0


def test_tfmemprof_diagnose_exit_code_two_when_risk_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When risk is detected, returns 2."""
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=True, risk_detected=True)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=True)

    args = SimpleNamespace(
        output=str(tmp_path),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    exit_code = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 2


def test_tfmemprof_diagnose_stdout_contains_artifact_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout summary contains artifact path and status."""
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)

    args = SimpleNamespace(
        output=str(tmp_path),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    out = capsys.readouterr().out

    assert "Artifact:" in out
    assert "Status:" in out
    assert "tfmemprof-diagnose-" in out
    assert "OK" in out or "MEMORY_RISK" in out or "FAILED" in out


def test_tfmemprof_diagnose_default_output_creates_timestamped_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no --output, artifact is created in cwd with timestamped name."""
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        output=None,
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    assert dirs[0].name.startswith("tfmemprof-diagnose-")


def test_tfmemprof_diagnose_output_existing_dir_creates_timestamped_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --output is an existing directory, create timestamped subdir inside."""
    out_dir = tmp_path / "myout"
    out_dir.mkdir()
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)

    args = SimpleNamespace(
        output=str(out_dir),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1
    assert subdirs[0].name.startswith("tfmemprof-diagnose-")
    assert (subdirs[0] / "manifest.json").exists()


def test_tfmemprof_diagnose_invalid_output_returns_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --output is an existing file (cannot create dir), returns 1."""
    existing_file = tmp_path / "existing_file"
    existing_file.write_text("x")
    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)

    args = SimpleNamespace(
        output=str(existing_file),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    exit_code = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 1


def test_tfmemprof_build_summary_without_capacity_does_not_flag_high_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When device capacity is unknown, high-utilization risk should remain false."""
    monkeypatch.setattr(
        diagnose_module,
        "get_system_info",
        lambda: {"backend": {"runtime_backend": "cuda"}},
    )
    monkeypatch.setattr(
        diagnose_module,
        "get_backend_info",
        lambda: {"runtime_backend": "cuda"},
    )
    monkeypatch.setattr(
        diagnose_module,
        "get_gpu_info",
        lambda: {
            "devices": [
                {
                    "id": 0,
                    "current_memory_mb": 9000,
                    "peak_memory_mb": 10000,
                }
            ]
        },
    )

    summary, risk_detected = diagnose_module.build_diagnostic_summary("/GPU:0")

    assert summary["total_bytes"] == 0
    assert summary["risk_flags"]["high_utilization"] is False
    assert risk_detected is False


def test_tfmemprof_diagnose_same_second_creates_unique_artifact_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two runs in same second should not overwrite the same artifact directory."""

    class _FixedDateTime:
        @staticmethod
        def utcnow() -> "real_datetime":
            return real_datetime(2026, 2, 15, 12, 0, 0)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _patch_tfmemprof_diagnose_env(monkeypatch, gpu_available=False)
    _patch_tfmemprof_timeline_capture(monkeypatch)
    _patch_tfmemprof_build_summary(monkeypatch, risk_detected=False)
    monkeypatch.setattr(diagnose_module, "datetime", _FixedDateTime)

    args = SimpleNamespace(
        output=str(out_dir),
        device="/GPU:0",
        duration=0,
        interval=0.5,
    )
    code_one = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    code_two = tfmemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    subdirs = sorted([path.name for path in out_dir.iterdir() if path.is_dir()])
    assert code_one in (0, 2)
    assert code_two in (0, 2)
    assert len(subdirs) == 2
    assert subdirs[0].startswith("tfmemprof-diagnose-20260215-120000")
    assert subdirs[1].startswith("tfmemprof-diagnose-20260215-120000")
    assert subdirs[0] != subdirs[1]
