"""Tests for the analyze CLI command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pytest

import gpumemprof.cli as gpumemprof_cli
from gpumemprof.cli import cmd_analyze
from gpumemprof.telemetry import telemetry_event_to_dict
from tests.gap_test_helpers import BASE_NS, INTERVAL_NS, build_gap_event

matplotlib.use("Agg")

_GB = 1024**3


def _build_cross_rank_events() -> list:
    events = []
    for rank, offset_ns, values in (
        (0, 0, [1 * _GB, 1 * _GB, 1 * _GB, int(1.34 * _GB)]),
        (1, 20_000_000, [1 * _GB, 1 * _GB, 1 * _GB, int(1.30 * _GB)]),
        (2, 40_000_000, [1 * _GB, 1 * _GB, int(1.38 * _GB), int(1.40 * _GB)]),
    ):
        for index, device_used in enumerate(values):
            allocator_reserved = max(device_used - 128 * 1024**2, 0)
            allocator_allocated = max(allocator_reserved - 64 * 1024**2, 0)
            events.append(
                build_gap_event(
                    index=index,
                    allocator_allocated=allocator_allocated,
                    allocator_reserved=allocator_reserved,
                    device_used=device_used,
                    collector="gpumemprof.cuda_tracker",
                    rank=rank,
                    local_rank=rank,
                    world_size=3,
                    job_id="cli-job",
                    host=f"host-{rank}",
                    timestamp_ns=BASE_NS + offset_ns + index * INTERVAL_NS,
                )
            )
    return events


def test_cmd_analyze_reports_cross_rank_findings_and_writes_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "telemetry.json"
    report_path = tmp_path / "report.json"
    plot_dir = tmp_path / "plots"
    input_path.write_text(
        json.dumps(
            [telemetry_event_to_dict(event) for event in _build_cross_rank_events()]
        ),
        encoding="utf-8",
    )

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(input_path),
            output=str(report_path),
            format="json",
            visualization=True,
            plot_dir=str(plot_dir),
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Distributed Analysis:" in stdout
    assert "Top first-cause suspect: rank 2" in stdout
    assert report_path.exists()
    assert (plot_dir / "cross_rank_timeline.png").exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cross_rank_analysis"]["first_cause_suspects"][0]["rank"] == 2


def test_cmd_analyze_non_telemetry_falls_back_gracefully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "results.json"
    input_path.write_text(json.dumps({"results": []}), encoding="utf-8")

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(input_path),
            output=None,
            format="json",
            visualization=False,
            plot_dir=str(tmp_path / "plots"),
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Analyzing profiling results from:" in stdout
    assert "Notes: JSON payload does not contain telemetry events" in stdout


def test_cmd_analyze_non_telemetry_array_falls_back_gracefully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "results.json"
    input_path.write_text(
        json.dumps(
            [
                {"function_name": "train_step", "duration_ms": 12.5},
                {"function_name": "eval_step", "duration_ms": 8.0},
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(input_path),
            output=None,
            format="json",
            visualization=False,
            plot_dir=str(tmp_path / "plots"),
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Analyzing profiling results from:" in stdout
    assert "Notes: JSON payload does not contain telemetry events" in stdout
    assert "Error parsing telemetry events" not in stdout


def test_cmd_analyze_missing_input_returns_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_path = tmp_path / "missing.json"

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(missing_path),
            output=None,
            format="json",
            visualization=False,
            plot_dir=str(tmp_path / "plots"),
        )
    )

    assert exit_code == 1
    assert "Error: Input file" in capsys.readouterr().out


def test_cmd_analyze_malformed_telemetry_returns_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "broken.json"
    input_path.write_text(json.dumps([{"timestamp": "oops"}]), encoding="utf-8")

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(input_path),
            output=None,
            format="json",
            visualization=False,
            plot_dir=str(tmp_path / "plots"),
        )
    )

    assert exit_code == 1
    assert "Error parsing telemetry events:" in capsys.readouterr().out


def test_main_exits_nonzero_for_analyze_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.json"
    monkeypatch.setattr(
        gpumemprof_cli.sys,
        "argv",
        ["gpumemprof", "analyze", str(missing_path)],
    )

    with pytest.raises(SystemExit) as excinfo:
        gpumemprof_cli.main()

    assert excinfo.value.code == 1
    assert "Error: Input file" in capsys.readouterr().out


def test_cmd_analyze_visualization_dependency_error_reports_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "telemetry.json"
    input_path.write_text(
        json.dumps(
            [telemetry_event_to_dict(event) for event in _build_cross_rank_events()]
        ),
        encoding="utf-8",
    )

    original_import_runtime_symbols = gpumemprof_cli._import_runtime_symbols

    def _patched_import_runtime_symbols(
        module_name: str,
        symbols: tuple[str, ...],
        feature: str,
    ) -> tuple[Any, ...]:
        if module_name == ".visualizer":
            raise ImportError("dlopen(PIL/_imaging): incompatible architecture")
        return original_import_runtime_symbols(module_name, symbols, feature)

    monkeypatch.setattr(
        gpumemprof_cli,
        "_import_runtime_symbols",
        _patched_import_runtime_symbols,
    )

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(input_path),
            output=None,
            format="json",
            visualization=True,
            plot_dir=str(tmp_path / "plots"),
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert (
        "Visualization skipped: Visualization dependencies are unavailable." in stdout
    )
    assert "stormlog[viz]" in stdout
