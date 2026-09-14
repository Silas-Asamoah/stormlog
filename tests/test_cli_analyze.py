"""Tests for the analyze CLI command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pytest

import stormlog.cli as gpumemprof_cli
from stormlog.cli import cmd_analyze
from stormlog.telemetry import telemetry_event_from_record, telemetry_event_to_dict
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig
from tests.gap_test_helpers import BASE_NS, INTERVAL_NS, build_gap_event

_stormlog_phases: Any
try:
    import stormlog.phases as _stormlog_phases
except ImportError:  # pragma: no cover - phase package may land in another slice
    _stormlog_phases = None

matplotlib.use("Agg")

_GB = 1024**3

_DEVICE_ONLY_CAPABILITIES = {
    "backend": "future",
    "telemetry_collector": "stormlog.future_tracker",
    "sampling_source": "future.device_memory",
    "supports_allocator_allocated": False,
    "supports_allocator_reserved": False,
    "supports_allocator_active": False,
    "supports_allocator_inactive": False,
    "supports_device_used": True,
    "supports_device_free": True,
    "supports_device_total": True,
    "supports_native_allocator_history": False,
    "supports_fragmentation_analysis": False,
    "supports_allocator_attribution": False,
    "supports_bounded_profiling": False,
}


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
                    collector="stormlog.cuda_tracker",
                    rank=rank,
                    local_rank=rank,
                    world_size=3,
                    job_id="cli-job",
                    host=f"host-{rank}",
                    timestamp_ns=BASE_NS + offset_ns + index * INTERVAL_NS,
                )
            )
    return events


def _build_cross_rank_events_with_phases() -> list:
    session_id = "session-cli-phase"
    events = []
    for rank in range(3):
        events.append(
            telemetry_event_from_record(
                {
                    "schema_version": 3,
                    "session_id": session_id,
                    "timestamp_ns": BASE_NS - 1_000_000 + rank * 100_000,
                    "event_type": "phase_enter",
                    "collector": "stormlog.cuda_tracker",
                    "sampling_interval_ms": 100,
                    "pid": 1,
                    "host": f"host-{rank}",
                    "job_id": "cli-job",
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 3,
                    "device_id": 0,
                    "allocator_allocated_bytes": 1 * _GB,
                    "allocator_reserved_bytes": 1 * _GB,
                    "allocator_active_bytes": None,
                    "allocator_inactive_bytes": None,
                    "allocator_change_bytes": 0,
                    "device_used_bytes": 1 * _GB,
                    "device_free_bytes": 15 * _GB,
                    "device_total_bytes": 16 * _GB,
                    "context": "Phase entered: train / forward",
                    "metadata": {
                        "phase_scope": {
                            "action": "enter",
                            "name": "forward",
                            "path": ["train", "forward"],
                            "depth": 2,
                            "scope_id": f"phase-{rank}",
                            "parent_scope_id": "phase-train",
                            "thread_id": 1,
                            "thread_name": "MainThread",
                            "sequence": rank + 1,
                        }
                    },
                }
            )
        )
    for event in _build_cross_rank_events():
        record = telemetry_event_to_dict(event)
        record["schema_version"] = 3
        record["session_id"] = session_id
        events.append(telemetry_event_from_record(record))
    return events


def _build_single_rank_gap_events_with_phase() -> list:
    session_id = "session-cli-gap-phase"
    events = [
        telemetry_event_from_record(
            {
                "schema_version": 3,
                "session_id": session_id,
                "timestamp_ns": BASE_NS - 1_000_000,
                "event_type": "phase_enter",
                "collector": "stormlog.cuda_tracker",
                "sampling_interval_ms": 100,
                "pid": 1,
                "host": "host-0",
                "job_id": "cli-gap-job",
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "device_id": 0,
                "allocator_allocated_bytes": 2_000_000_000,
                "allocator_reserved_bytes": 2_500_000_000,
                "allocator_active_bytes": None,
                "allocator_inactive_bytes": None,
                "allocator_change_bytes": 0,
                "device_used_bytes": 2_500_000_000,
                "device_free_bytes": 13 * _GB,
                "device_total_bytes": 16 * _GB,
                "context": "Phase entered: train / forward",
                "metadata": {
                    "phase_scope": {
                        "action": "enter",
                        "name": "forward",
                        "path": ["train", "forward"],
                        "depth": 2,
                        "scope_id": "phase-gap",
                        "parent_scope_id": "phase-train",
                        "thread_id": 1,
                        "thread_name": "MainThread",
                        "sequence": 1,
                    }
                },
            }
        )
    ]
    for index in range(12):
        events.append(
            telemetry_event_from_record(
                {
                    "schema_version": 3,
                    "session_id": session_id,
                    "timestamp_ns": BASE_NS + index * INTERVAL_NS,
                    "event_type": "sample",
                    "collector": "stormlog.cuda_tracker",
                    "sampling_interval_ms": 100,
                    "pid": 1,
                    "host": "host-0",
                    "job_id": "cli-gap-job",
                    "rank": 0,
                    "local_rank": 0,
                    "world_size": 1,
                    "device_id": 0,
                    "allocator_allocated_bytes": 2_000_000_000,
                    "allocator_reserved_bytes": 2_500_000_000,
                    "allocator_active_bytes": None,
                    "allocator_inactive_bytes": None,
                    "allocator_change_bytes": 0,
                    "device_used_bytes": 3_000_000_000 + index * 150_000_000,
                    "device_free_bytes": None,
                    "device_total_bytes": 16 * _GB,
                    "context": "sample",
                    "metadata": {},
                }
            )
        )
    return events


def _build_single_rank_gap_events_with_ambiguous_phase() -> list:
    session_id = "session-cli-gap-ambiguous"
    phase_events = []
    for thread_id, sequence, scope_id in (
        (1, 1, "phase-gap-a"),
        (2, 2, "phase-gap-b"),
    ):
        phase_events.append(
            telemetry_event_from_record(
                {
                    "schema_version": 3,
                    "session_id": session_id,
                    "timestamp_ns": BASE_NS - 1_000_000 + sequence,
                    "event_type": "phase_enter",
                    "collector": "stormlog.cuda_tracker",
                    "sampling_interval_ms": 100,
                    "pid": 1,
                    "host": "host-0",
                    "job_id": "cli-gap-job",
                    "rank": 0,
                    "local_rank": 0,
                    "world_size": 1,
                    "device_id": 0,
                    "allocator_allocated_bytes": 2_000_000_000,
                    "allocator_reserved_bytes": 2_500_000_000,
                    "allocator_active_bytes": None,
                    "allocator_inactive_bytes": None,
                    "allocator_change_bytes": 0,
                    "device_used_bytes": 2_500_000_000,
                    "device_free_bytes": 13 * _GB,
                    "device_total_bytes": 16 * _GB,
                    "context": "Phase entered: train / forward",
                    "metadata": {
                        "phase_scope": {
                            "action": "enter",
                            "name": "forward",
                            "path": ["train", "forward"],
                            "depth": 2,
                            "scope_id": scope_id,
                            "parent_scope_id": "phase-train",
                            "thread_id": thread_id,
                            "thread_name": f"thread-{thread_id}",
                            "sequence": sequence,
                        }
                    },
                }
            )
        )
    sample_events = []
    for event in _build_single_rank_gap_events_with_phase()[1:]:
        record = telemetry_event_to_dict(event)
        record["schema_version"] = 3
        record["session_id"] = session_id
        sample_events.append(telemetry_event_from_record(record))
    return [*phase_events, *sample_events]


def _build_single_rank_gap_events_with_ambiguous_phase_labels() -> list:
    session_id = "session-cli-gap-ambiguous-labels"
    phase_events = []
    for sequence, scope_id, name, path in (
        (1, "phase-gap-forward", "forward", ["train", "forward"]),
        (2, "phase-gap-communication", "communication", ["train", "communication"]),
    ):
        phase_events.append(
            telemetry_event_from_record(
                {
                    "schema_version": 3,
                    "session_id": session_id,
                    "timestamp_ns": BASE_NS - 1_000_000 + sequence,
                    "event_type": "phase_enter",
                    "collector": "stormlog.cuda_tracker",
                    "sampling_interval_ms": 100,
                    "pid": 1,
                    "host": "host-0",
                    "job_id": "cli-gap-job",
                    "rank": 0,
                    "local_rank": 0,
                    "world_size": 1,
                    "device_id": 0,
                    "allocator_allocated_bytes": 2_000_000_000,
                    "allocator_reserved_bytes": 2_500_000_000,
                    "allocator_active_bytes": None,
                    "allocator_inactive_bytes": None,
                    "allocator_change_bytes": 0,
                    "device_used_bytes": 2_500_000_000,
                    "device_free_bytes": 13 * _GB,
                    "device_total_bytes": 16 * _GB,
                    "context": f"Phase entered: {' / '.join(path)}",
                    "metadata": {
                        "phase_scope": {
                            "action": "enter",
                            "name": name,
                            "path": path,
                            "depth": len(path),
                            "scope_id": scope_id,
                            "parent_scope_id": "phase-train",
                            "thread_id": sequence,
                            "thread_name": f"thread-{sequence}",
                            "sequence": sequence,
                        }
                    },
                }
            )
        )
    sample_events = []
    for event in _build_single_rank_gap_events_with_phase()[1:]:
        record = telemetry_event_to_dict(event)
        record["schema_version"] = 3
        record["session_id"] = session_id
        sample_events.append(telemetry_event_from_record(record))
    return [*phase_events, *sample_events]


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


def test_cmd_analyze_explains_device_only_capability_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = []
    for index in range(3):
        record = telemetry_event_to_dict(
            build_gap_event(
                index=index,
                allocator_allocated=1,
                allocator_reserved=1,
                device_used=40 + index,
                device_total=100,
                collector="stormlog.future_tracker",
            )
        )
        record.update(
            {
                "schema_version": 4,
                "session_id": "device-only-cli",
                "allocator_allocated_bytes": None,
                "allocator_reserved_bytes": None,
                "allocator_change_bytes": None,
            }
        )
        record["metadata"] = {"memory_capabilities": dict(_DEVICE_ONLY_CAPABILITIES)}
        records.append(record)
    input_path = tmp_path / "device-only.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(input_path),
            output=None,
            format="json",
            visualization=False,
            plot_dir=None,
            session_id=None,
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Capability warning:" in output
    assert "does not expose allocator counters" in output


def test_cmd_analyze_surfaces_phase_summaries_when_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if _stormlog_phases is None:
        pytest.skip("stormlog.phases is not available in this slice")

    input_path = tmp_path / "telemetry.json"
    input_path.write_text(
        json.dumps(
            [
                telemetry_event_to_dict(event)
                for event in _build_cross_rank_events_with_phases()
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
            session_id=None,
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Suspect phase: train / forward" in stdout


def test_cmd_analyze_surfaces_ambiguous_gap_phase_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if _stormlog_phases is None:
        pytest.skip("stormlog.phases is not available in this slice")

    input_path = tmp_path / "telemetry-gap-ambiguous.json"
    input_path.write_text(
        json.dumps(
            [
                telemetry_event_to_dict(event)
                for event in _build_single_rank_gap_events_with_ambiguous_phase()
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
            session_id=None,
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Top gap phase: (ambiguous) train / forward" in stdout


def test_cmd_analyze_surfaces_likely_gap_phase_summary_for_multi_label_ambiguity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if _stormlog_phases is None:
        pytest.skip("stormlog.phases is not available in this slice")

    input_path = tmp_path / "telemetry-gap-likely.json"
    input_path.write_text(
        json.dumps(
            [
                telemetry_event_to_dict(event)
                for event in _build_single_rank_gap_events_with_ambiguous_phase_labels()
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
            session_id=None,
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Top gap phase: (likely) train / communication" in stdout


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


def test_cmd_analyze_reads_append_only_sink_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    for event in _build_cross_rank_events():
        sink.append(telemetry_event_to_dict(event))
    sink.close()

    exit_code = cmd_analyze(
        argparse.Namespace(
            input_file=str(tmp_path / "sink"),
            output=None,
            format="json",
            visualization=False,
            plot_dir=str(tmp_path / "plots"),
        )
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Distributed Analysis:" in stdout
    assert "Top first-cause suspect: rank 2" in stdout


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


@pytest.mark.parametrize(
    "load_error", [ValueError("bad events"), RuntimeError("bad events")]
)
@pytest.mark.parametrize(
    ("filename", "payload", "expected_code", "expected_message"),
    [
        (
            "results.json",
            '{"results": []}',
            0,
            "Notes: JSON payload does not contain telemetry events",
        ),
        (
            "events.json",
            '{"events": []}',
            1,
            "Error parsing telemetry events: bad events",
        ),
        (
            "events.jsonl",
            '{"results": []}',
            1,
            "Error parsing telemetry events: bad events",
        ),
        ("broken.json", "{", 1, "Error loading input file:"),
    ],
)
def test_cmd_analyze_loader_failure_preserves_json_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    load_error: Exception,
    filename: str,
    payload: str,
    expected_code: int,
    expected_message: str,
) -> None:
    input_path = tmp_path / filename
    input_path.write_text(payload, encoding="utf-8")

    def fail_to_load(*args: Any, **kwargs: Any) -> Any:
        raise load_error

    monkeypatch.setattr(
        gpumemprof_cli, "_import_runtime_symbols", lambda *args: (fail_to_load,)
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

    assert exit_code == expected_code
    assert expected_message in capsys.readouterr().out


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
