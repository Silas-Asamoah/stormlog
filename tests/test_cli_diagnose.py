"""Tests for the Stormlog diagnose command."""

import json
from contextlib import contextmanager
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast

import pytest

import stormlog.cli as gpumemprof_cli
import stormlog.diagnose as diagnose_module
import stormlog.tracker as tracker_module
from stormlog.exit_codes import ExitCode


def _patch_diagnose_env(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool = False,
    risk_detected: bool = False,
) -> None:
    """Patch diagnose module's env/summary dependencies for predictable output."""
    monkeypatch.setattr(
        diagnose_module,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "architecture": "x86_64",
            "python_version": "3.10",
            "cuda_available": cuda_available,
            "detected_backend": "cuda" if cuda_available else "cpu",
        },
    )
    gpu_info: dict[str, object]
    frag_info: dict[str, object]
    if cuda_available and risk_detected:
        gpu_info = {
            "device_id": 0,
            "device_name": "Test GPU",
            "total_memory": 1024**3,
            "allocated_memory": int(0.9 * 1024**3),
            "reserved_memory": int(0.95 * 1024**3),
            "max_memory_allocated": int(0.9 * 1024**3),
            "memory_stats": {"num_ooms": 1},
        }
        frag_info = {
            "fragmentation_ratio": 0.35,
            "utilization_ratio": 0.9,
        }
    elif cuda_available:
        gpu_info = {
            "device_id": 0,
            "device_name": "Test GPU",
            "total_memory": 1024**3,
            "allocated_memory": 100 * 1024**2,
            "reserved_memory": 150 * 1024**2,
            "max_memory_allocated": 100 * 1024**2,
            "memory_stats": {"num_ooms": 0},
        }
        frag_info = {
            "fragmentation_ratio": 0.1,
            "utilization_ratio": 0.2,
        }
    else:
        gpu_info = {"error": "CUDA is not available"}
        frag_info = {"error": "CUDA is not available"}

    monkeypatch.setattr(diagnose_module, "get_gpu_info", lambda device: gpu_info)
    monkeypatch.setattr(
        diagnose_module,
        "check_memory_fragmentation",
        lambda device: frag_info,
    )
    monkeypatch.setattr(
        diagnose_module,
        "suggest_memory_optimization",
        lambda frag: ["Suggestion 1"] if risk_detected else [],
    )


def _patch_timeline_capture(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_diagnose_produces_artifact_bundle_with_duration_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invocation with duration=0 produces directory with required files."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    assert exit_code in (ExitCode.OK, ExitCode.FINDINGS)
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 1
    artifact_dir = dirs[0]
    assert artifact_dir.is_dir()
    assert "stormlog-diagnose-" in artifact_dir.name

    assert (artifact_dir / "environment.json").exists()
    assert (artifact_dir / "diagnostic_summary.json").exists()
    assert (artifact_dir / "telemetry_timeline.json").exists()
    assert (artifact_dir / "manifest.json").exists()

    with open(artifact_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert "files" in manifest
    assert "exit_code" in manifest
    assert "risk_detected" in manifest
    assert "environment.json" in manifest["files"]
    assert "manifest.json" in manifest["files"]
    for name in manifest["files"]:
        assert (artifact_dir / name).exists()

    with open(artifact_dir / "diagnostic_summary.json") as f:
        summary = json.load(f)
    assert "backend" in summary
    assert "risk_flags" in summary
    assert "suggestions" in summary


def test_diagnose_artifact_completeness_with_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With duration > 0, telemetry_timeline.json has data (when capture is mocked)."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=5.0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    dirs = list(tmp_path.iterdir())
    artifact_dir = dirs[0]
    with open(artifact_dir / "telemetry_timeline.json") as f:
        timeline = json.load(f)
    assert "timestamps" in timeline
    assert "allocated" in timeline
    assert "reserved" in timeline
    assert len(timeline["timestamps"]) == 3
    assert len(timeline["allocated"]) == 3


def test_diagnose_invalid_duration_returns_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid --duration < 0 returns USAGE (2) and prints error."""
    args = SimpleNamespace(
        output=None,
        device=None,
        duration=-1,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == ExitCode.USAGE
    err = capsys.readouterr().err
    assert "duration" in err.lower()


def test_diagnose_invalid_interval_returns_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid --interval <= 0 returns USAGE (2) and prints error."""
    args = SimpleNamespace(
        output=None,
        device=None,
        duration=0,
        interval=0,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == ExitCode.USAGE
    err = capsys.readouterr().err
    assert "interval" in err.lower()


def test_diagnose_exit_code_zero_when_no_risk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When risk flags are false, cmd_diagnose returns 0."""
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 0


def test_diagnose_exports_bundle_to_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exported: dict[str, Any] = {}
    artifact_dir = tmp_path / "stormlog-diagnose-bundle"
    artifact_dir.mkdir()
    wandb_config = SimpleNamespace(enabled=True)

    monkeypatch.setattr(
        gpumemprof_cli,
        "wandb_config_from_namespace",
        lambda args: wandb_config,
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "ensure_wandb_available",
        lambda config: exported.setdefault("ensured", config),
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "export_diagnose_bundle_to_wandb",
        lambda config, **kwargs: exported.update(config=config, kwargs=kwargs),
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "_import_runtime_symbols",
        lambda module_name, symbols, feature: (lambda **kwargs: (artifact_dir, 0),),
    )

    exit_code = gpumemprof_cli.cmd_diagnose(
        cast(
            Any,
            SimpleNamespace(
                output=str(tmp_path),
                device=None,
                duration=0,
                interval=0.5,
                native_history=False,
                native_history_max_entries=100000,
                wandb=True,
            ),
        )
    )

    assert exit_code == 0
    assert exported["ensured"] is wandb_config
    assert exported["config"] is wandb_config
    assert exported["kwargs"]["command_name"] == "gpumemprof-diagnose"
    assert exported["kwargs"]["artifact_dir"] == artifact_dir


def test_diagnose_exit_code_findings_when_risk_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When risk is detected (e.g. high utilization, OOM), returns FINDINGS (3)."""
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=True)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == ExitCode.FINDINGS
    assert int(exit_code) == 3
    assert "Status: MEMORY_RISK (exit_code=3)" in capsys.readouterr().out
    manifest = json.loads(
        (next(tmp_path.iterdir()) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["exit_code"] == 3
    assert manifest["risk_detected"] is True


def test_diagnose_fallback_manifest_preserves_risk_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=True)
    _patch_timeline_capture(monkeypatch)
    recorded_risks: list[bool] = []
    call_count = {"value": 0}

    def _fake_write_manifest(
        artifact_dir: Path,
        *,
        command_line: str,
        files_written: list[str],
        exit_code: int,
        risk_detected: bool,
        session_summary: object,
        native_history: bool,
        error: str | None = None,
    ) -> None:
        _ = (
            command_line,
            files_written,
            exit_code,
            session_summary,
            native_history,
            error,
        )
        recorded_risks.append(risk_detected)
        manifest_path = artifact_dir / "manifest.json"
        if call_count["value"] == 0:
            call_count["value"] += 1
            raise OSError("disk full")
        manifest_path.write_text(
            json.dumps({"risk_detected": risk_detected}),
            encoding="utf-8",
        )

    monkeypatch.setattr(diagnose_module, "_write_manifest", _fake_write_manifest)

    artifact_dir, exit_code = diagnose_module.run_diagnose(
        str(tmp_path),
        None,
        0,
        0.5,
        "gpumemprof diagnose",
    )

    assert exit_code == 1
    assert recorded_risks == [True, True]
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["risk_detected"] is True


def test_diagnose_stdout_contains_artifact_path_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout summary contains artifact path and status line."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    out = capsys.readouterr().out

    assert "Artifact:" in out
    assert "Status:" in out
    assert "stormlog-diagnose-" in out
    assert "OK" in out or "MEMORY_RISK" in out or "FAILED" in out


def test_diagnose_stdout_findings_when_risk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When risk detected, stdout includes findings line."""
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=True)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    out = capsys.readouterr().out

    assert "Findings:" in out
    assert "MEMORY_RISK" in out


def test_diagnose_default_output_creates_timestamped_dir_in_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no --output, artifact is created in cwd with timestamped name."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        output=None,
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    assert dirs[0].name.startswith("stormlog-diagnose-")


def test_diagnose_output_existing_dir_creates_timestamped_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --output is an existing directory, create timestamped subdir inside."""
    out_dir = tmp_path / "myout"
    out_dir.mkdir()
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(out_dir),
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1
    assert subdirs[0].name.startswith("stormlog-diagnose-")
    assert (subdirs[0] / "manifest.json").exists()


def test_diagnose_invalid_output_returns_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --output is an existing file (cannot create dir), returns 1."""
    existing_file = tmp_path / "existing_file"
    existing_file.write_text("x")
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(existing_file),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    assert exit_code == 1


def test_run_timeline_capture_uses_memory_tracker_for_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS runtime should use MemoryTracker, not CPUMemoryTracker."""
    created = {}

    class _FakeTracker:
        def __init__(
            self,
            device: object = None,
            sampling_interval: float = 0.5,
            enable_alerts: bool = False,
        ) -> None:
            created["device"] = device
            created["sampling_interval"] = sampling_interval
            created["enable_alerts"] = enable_alerts

        def start_tracking(self) -> None:
            created["started"] = True

        def stop_tracking(self) -> None:
            created["stopped"] = True

        def get_memory_timeline(self, interval: float = 0.5) -> dict[str, list[object]]:
            created["interval"] = interval
            return {"timestamps": [0.0], "allocated": [1], "reserved": [2]}

    monkeypatch.setattr(diagnose_module, "detect_torch_runtime_backend", lambda: "mps")
    monkeypatch.setattr(tracker_module, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(diagnose_module.time, "sleep", lambda _: None)

    timeline = diagnose_module.run_timeline_capture(
        device=None, duration_seconds=0.1, interval=0.05
    )

    assert created["device"] == "mps"
    assert created["started"] is True
    assert created["stopped"] is True
    assert timeline["allocated"] == [1]
    assert timeline["reserved"] == [2]


def test_collect_environment_uses_mps_backend_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS environment collection should include backend sample instead of CUDA error."""

    class _Sample:
        device_id = 0
        allocated_bytes = 123
        reserved_bytes = 456
        total_bytes = 789

    monkeypatch.setattr(
        diagnose_module,
        "get_system_info",
        lambda: {"detected_backend": "mps"},
    )
    monkeypatch.setattr(
        diagnose_module, "get_gpu_info", lambda _: {"error": "CUDA is not available"}
    )
    monkeypatch.setattr(
        diagnose_module, "_collect_backend_sample", lambda _: ("mps", _Sample())
    )

    env = diagnose_module.collect_environment(device=None)

    assert env["gpu"]["backend"] == "mps"
    assert env["gpu"]["allocated_memory"] == 123
    assert env["fragmentation"]["note"]


def test_diagnose_same_second_creates_unique_artifact_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two runs in same second should not overwrite the same artifact directory."""

    class _FixedDateTime:
        @staticmethod
        def utcnow() -> "real_datetime":
            return real_datetime(2026, 2, 15, 12, 0, 0)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)
    monkeypatch.setattr(diagnose_module, "datetime", _FixedDateTime)

    args = SimpleNamespace(
        output=str(out_dir),
        device=None,
        duration=0,
        interval=0.5,
    )
    code_one = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]
    code_two = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    subdirs = sorted([path.name for path in out_dir.iterdir() if path.is_dir()])
    assert code_one in (ExitCode.OK, ExitCode.FINDINGS)
    assert code_two in (ExitCode.OK, ExitCode.FINDINGS)
    assert len(subdirs) == 2
    assert subdirs[0].startswith("stormlog-diagnose-20260215-120000")
    assert subdirs[1].startswith("stormlog-diagnose-20260215-120000")
    assert subdirs[0] != subdirs[1]


def test_diagnose_native_history_writes_snapshot_artifacts_for_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=False)
    _patch_timeline_capture(monkeypatch)

    history_calls: dict[str, object] = {}

    @contextmanager
    def _fake_native_history(
        device: object = None,
        trace_alloc_max_entries: int = 0,
    ) -> Iterator[None]:
        history_calls["device"] = device
        history_calls["trace_alloc_max_entries"] = trace_alloc_max_entries
        yield

    def _fake_capture(
        artifact_dir: Path,
        *,
        device: object = None,
        history_recorded: bool,
    ) -> list[str]:
        history_calls["capture_device"] = device
        history_calls["history_recorded"] = history_recorded
        snapshot_path = artifact_dir / "cuda_allocator_snapshot.pickle"
        snapshot_path.write_text("snapshot", encoding="utf-8")
        metadata_path = artifact_dir / "cuda_native_debug_metadata.json"
        metadata_path.write_text("{}", encoding="utf-8")
        return [snapshot_path.name, metadata_path.name]

    monkeypatch.setattr(diagnose_module, "detect_torch_runtime_backend", lambda: "cuda")
    monkeypatch.setattr(diagnose_module, "cuda_memory_history_supported", lambda: True)
    monkeypatch.setattr(diagnose_module, "cuda_memory_history", _fake_native_history)
    monkeypatch.setattr(
        diagnose_module,
        "capture_cuda_snapshot_artifacts",
        _fake_capture,
    )

    args = SimpleNamespace(
        output=str(tmp_path),
        device=0,
        duration=0,
        interval=0.5,
        native_history=True,
        native_history_max_entries=77,
    )

    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    assert exit_code == 0
    artifact_dir = next(tmp_path.iterdir())
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["native_history_enabled"] is True
    assert "cuda_allocator_snapshot.pickle" in manifest["files"]
    assert "cuda_native_debug_metadata.json" in manifest["files"]
    assert history_calls["device"] == 0
    assert history_calls["trace_alloc_max_entries"] == 77
    assert history_calls["capture_device"] == 0
    assert history_calls["history_recorded"] is True


def test_diagnose_native_history_rejects_non_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    monkeypatch.setattr(diagnose_module, "detect_torch_runtime_backend", lambda: "mps")

    args = SimpleNamespace(
        output=None,
        device=None,
        duration=0,
        interval=0.5,
        native_history=True,
        native_history_max_entries=1000,
    )

    exit_code = gpumemprof_cli.cmd_diagnose(args)  # type: ignore[arg-type, unused-ignore]

    assert exit_code == ExitCode.USAGE
    err = capsys.readouterr().err
    assert "only for CUDA runtimes" in err


def test_run_diagnose_rejects_non_positive_native_history_max_entries() -> None:
    with pytest.raises(ValueError, match="native_history_max_entries must be >= 1"):
        diagnose_module.run_diagnose(
            output=None,
            device=0,
            duration=0,
            interval=0.5,
            command_line="gpumemprof diagnose --native-history",
            native_history=True,
            native_history_max_entries=0,
        )
