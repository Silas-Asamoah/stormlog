from __future__ import annotations

from pathlib import Path

import pytest


def test_cpu_telemetry_scenario_writes_exports(tmp_path: Path) -> None:
    from examples.scenarios.cpu_telemetry_scenario import (
        run_scenario as run_cpu_telemetry,
    )

    output_dir = tmp_path / "cpu"
    summary = run_cpu_telemetry(
        output_dir=output_dir,
        tracker_interval=0.05,
        workload_duration_s=0.5,
    )

    assert summary["exported_event_count"] > 0  # type: ignore[operator, unused-ignore]
    assert (output_dir / "cpu_tracker_events.json").exists()
    assert (output_dir / "cpu_tracker_events.csv").exists()
    assert (output_dir / "cpu_profiler_summary.json").exists()


def test_mps_telemetry_scenario_passes_or_skips(tmp_path: Path) -> None:
    from examples.scenarios.mps_telemetry_scenario import (
        run_scenario as run_mps_telemetry,
    )

    output_dir = tmp_path / "mps"
    summary = run_mps_telemetry(
        output_dir=output_dir,
        duration_s=1.0,
        interval_s=0.5,
    )

    assert summary["status"] in {"PASS", "SKIP"}
    if summary["status"] == "PASS":
        assert summary["collector"] == "gpumemprof.mps_tracker"


def test_oom_flight_recorder_scenario_simulated_passes_or_skips(tmp_path: Path) -> None:
    from examples.scenarios.oom_flight_recorder_scenario import (
        run_scenario as run_oom_scenario,
    )

    output_dir = tmp_path / "oom"
    summary = run_oom_scenario(
        output_dir=output_dir,
        mode="simulated",
        sampling_interval=0.1,
        stress_max_mb=256,
        stress_step_mb=64,
    )

    assert summary["status"] in {"PASS", "SKIP"}
    if summary["status"] == "PASS":
        assert summary["oom_dump_exists"] is True


def test_tf_end_to_end_scenario_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    from examples.scenarios.tf_end_to_end_scenario import run_scenario as run_tf_e2e

    output_dir = tmp_path / "tf"
    summary = run_tf_e2e(
        output_dir=output_dir,
        monitor_duration_s=1.0,
        monitor_interval_s=0.5,
        track_interrupt_after_s=8.0,
    )

    assert summary["status"] == "PASS"
    assert summary["track_event_count"] > 0  # type: ignore[operator, unused-ignore]
    assert (output_dir / "tf_monitor.json").exists()
    assert (output_dir / "tf_track.json").exists()
    assert (output_dir / "tf_diagnose" / "manifest.json").exists()
    assert (output_dir / "tf_e2e_summary.json").exists()
