from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from gpumemprof.telemetry import (
    TelemetryEventV2,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from gpumemprof.tui.distributed_diagnostics import (
    build_distributed_model,
    load_distributed_artifacts,
    parse_rank_filter,
)


def _make_event(
    *,
    timestamp: float,
    rank: int,
    world_size: int,
    event_type: str = "sample",
    allocated: int = 0,
    reserved: int = 0,
    used: int = 0,
    total: int | None = None,
    context: str = "",
) -> TelemetryEventV2:
    return telemetry_event_from_record(
        {
            "timestamp": timestamp,
            "event_type": event_type,
            "memory_allocated": allocated,
            "memory_reserved": reserved,
            "memory_change": 0,
            "device_used_bytes": used or reserved,
            "device_total_bytes": total,
            "device_id": 0,
            "collector": "gpumemprof.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "test-host",
            "context": context,
            "job_id": "job-1",
            "rank": rank,
            "local_rank": rank,
            "world_size": world_size,
            "metadata": {},
        },
        permissive_legacy=True,
        default_collector="gpumemprof.cuda_tracker",
        default_sampling_interval_ms=100,
    )


def _write_csv_events(path: Path, events: list[TelemetryEventV2]) -> None:
    records = [telemetry_event_to_dict(event) for event in events]
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            row: dict[str, str] = {}
            for key, value in record.items():
                if isinstance(value, dict):
                    row[key] = json.dumps(value)
                elif value is None:
                    row[key] = ""
                else:
                    row[key] = str(value)
            writer.writerow(row)


def test_parse_rank_filter_supports_all_lists_and_ranges() -> None:
    available = list(range(8))
    assert parse_rank_filter("all", available) == set(available)
    assert parse_rank_filter("0,2,4-6", available) == {0, 2, 4, 5, 6}


def test_parse_rank_filter_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="start>end"):
        parse_rank_filter("5-2", [0, 1, 2, 3, 4, 5])


def test_build_distributed_model_computes_rank_metrics_and_missing_ranks() -> None:
    events = [
        _make_event(
            timestamp=1.0,
            rank=0,
            world_size=4,
            allocated=10,
            reserved=12,
            used=14,
            total=100,
        ),
        _make_event(
            timestamp=2.0,
            rank=0,
            world_size=4,
            allocated=20,
            reserved=24,
            used=25,
            total=100,
        ),
        _make_event(
            timestamp=1.5,
            rank=2,
            world_size=4,
            allocated=8,
            reserved=10,
            used=11,
            total=100,
        ),
    ]

    model = build_distributed_model(events)
    assert model.expected_ranks == [0, 1, 2, 3]
    assert model.present_ranks == [0, 2]
    assert model.missing_ranks == [1, 3]

    row_rank_0 = next(row for row in model.rows if row.rank == 0)
    assert row_rank_0.availability == "present"
    assert row_rank_0.samples == 2
    assert row_rank_0.allocated_delta_bytes == 10
    assert row_rank_0.reserved_delta_bytes == 12
    assert row_rank_0.hidden_gap_latest_bytes == 1
    assert row_rank_0.hidden_gap_peak_abs_bytes == 2

    row_rank_1 = next(row for row in model.rows if row.rank == 1)
    assert row_rank_1.availability == "missing"
    assert row_rank_1.samples == 0


def test_build_distributed_model_includes_earliest_and_most_severe_indicators() -> None:
    events = [
        _make_event(
            timestamp=1.0,
            rank=0,
            world_size=3,
            allocated=10,
            reserved=15,
            used=30,
            total=100,
            context="gap breach",
        ),
        _make_event(
            timestamp=2.0,
            rank=1,
            world_size=3,
            event_type="warning",
            allocated=10,
            reserved=12,
            used=12,
            total=100,
            context="warning alert",
        ),
        _make_event(
            timestamp=3.0,
            rank=2,
            world_size=3,
            event_type="critical",
            allocated=10,
            reserved=12,
            used=12,
            total=100,
            context="critical alert",
        ),
    ]

    model = build_distributed_model(events)
    indicators = {indicator.kind: indicator for indicator in model.indicators}

    assert {"earliest", "most_severe"} <= set(indicators)
    assert indicators["earliest"].rank == 0
    assert indicators["earliest"].signal == "gap_ratio_breach"
    assert indicators["most_severe"].rank == 2
    assert indicators["most_severe"].severity == "critical"
    assert indicators["most_severe"].signal == "alert:critical"


def test_load_distributed_artifacts_merges_json_and_csv_inputs(
    tmp_path: Path,
) -> None:
    json_event = _make_event(
        timestamp=1.0,
        rank=0,
        world_size=2,
        allocated=10,
        reserved=12,
        used=12,
        total=100,
    )
    csv_event = _make_event(
        timestamp=2.0,
        rank=1,
        world_size=2,
        allocated=20,
        reserved=24,
        used=24,
        total=100,
    )

    json_path = tmp_path / "events.json"
    json_path.write_text(
        json.dumps([telemetry_event_to_dict(json_event)]), encoding="utf-8"
    )

    csv_path = tmp_path / "events.csv"
    _write_csv_events(csv_path, [csv_event])

    result = load_distributed_artifacts([json_path, csv_path])
    assert len(result.events) == 2
    assert result.warnings == []
    assert str(json_path) in result.sources_loaded
    assert str(csv_path) in result.sources_loaded


def test_load_distributed_artifacts_reads_directory_event_payloads(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = [
        {
            "timestamp": 1.0,
            "event_type": "allocation",
            "memory_allocated": 10,
            "memory_reserved": 10,
            "memory_change": 10,
            "device_id": 0,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "collector": "legacy",
            "sampling_interval_ms": 100,
            "pid": 1,
            "host": "host",
            "metadata": {},
        }
    ]
    events_path = artifact_dir / "events.json"
    events_path.write_text(json.dumps(payload), encoding="utf-8")

    result = load_distributed_artifacts([artifact_dir])
    assert len(result.events) == 1
    assert str(events_path) in result.sources_loaded


def test_load_distributed_artifacts_synthesizes_events_from_diagnose_timeline(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "diag"
    artifact_dir.mkdir()
    timeline_path = artifact_dir / "telemetry_timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "timestamps": [1.0, 2.0, 3.0],
                "allocated": [100, 120, 140],
                "reserved": [100, 130, 150],
            }
        ),
        encoding="utf-8",
    )

    result = load_distributed_artifacts([artifact_dir])
    assert len(result.events) == 3
    assert any("Synthesized telemetry events" in warning for warning in result.warnings)
    assert all(event.rank == 0 for event in result.events)
    assert str(timeline_path) in result.sources_loaded


def test_load_distributed_artifacts_merges_mixed_directory_event_and_timeline_ranks(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    rank0_dir = artifact_dir / "rank0"
    rank1_dir = artifact_dir / "rank1"
    rank0_dir.mkdir(parents=True)
    rank1_dir.mkdir(parents=True)

    rank0_event = _make_event(
        timestamp=1.0,
        rank=0,
        world_size=2,
        allocated=100,
        reserved=120,
        used=120,
        total=1000,
    )
    events_path = rank0_dir / "events.json"
    events_path.write_text(
        json.dumps([telemetry_event_to_dict(rank0_event)]),
        encoding="utf-8",
    )

    timeline_path = rank1_dir / "telemetry_timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "timestamps": [1.0, 2.0],
                "allocated": [200, 240],
                "reserved": [220, 260],
            }
        ),
        encoding="utf-8",
    )

    result = load_distributed_artifacts([artifact_dir])

    assert sorted({event.rank for event in result.events}) == [0, 1]
    assert str(events_path) in result.sources_loaded
    assert str(timeline_path) in result.sources_loaded


def test_load_distributed_artifacts_preserves_rank_identity_across_timeline_bundles(
    tmp_path: Path,
) -> None:
    rank0_bundle = tmp_path / "rank0-bundle"
    rank1_bundle = tmp_path / "rank1-bundle"
    rank0_bundle.mkdir()
    rank1_bundle.mkdir()
    rank0_timeline = rank0_bundle / "telemetry_timeline.json"
    rank1_timeline = rank1_bundle / "telemetry_timeline.json"
    rank0_timeline.write_text(
        json.dumps(
            {
                "timestamps": [1.0, 2.0],
                "allocated": [100, 120],
                "reserved": [100, 120],
            }
        ),
        encoding="utf-8",
    )
    rank1_timeline.write_text(
        json.dumps(
            {
                "timestamps": [1.0, 2.0],
                "allocated": [200, 220],
                "reserved": [200, 220],
            }
        ),
        encoding="utf-8",
    )

    result = load_distributed_artifacts([rank0_bundle, rank1_bundle])
    model = build_distributed_model(result.events)

    rank_to_allocated: dict[int, list[int]] = {}
    for event in result.events:
        rank_to_allocated.setdefault(event.rank, []).append(
            event.allocator_allocated_bytes
        )

    assert sorted(rank_to_allocated.keys()) == [0, 1]
    assert sorted(rank_to_allocated[0]) == [100, 120]
    assert sorted(rank_to_allocated[1]) == [200, 220]
    assert model.present_ranks == [0, 1]
    assert str(rank0_timeline) in result.sources_loaded
    assert str(rank1_timeline) in result.sources_loaded


@dataclass(slots=True)
class _SyntheticEvent:
    timestamp_ns: int
    event_type: str
    rank: int
    world_size: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    device_used_bytes: int
    device_total_bytes: int | None
    context: str | None = None


def test_build_distributed_model_scale_bound_64_ranks_x_2000_samples() -> None:
    synthetic_events: list[_SyntheticEvent] = []
    for rank in range(64):
        base_ns = rank * 10_000_000_000
        for sample_index in range(2000):
            allocated = 1_000_000 + sample_index
            reserved = allocated + 100
            synthetic_events.append(
                _SyntheticEvent(
                    timestamp_ns=base_ns + sample_index * 1_000_000,
                    event_type="sample",
                    rank=rank,
                    world_size=64,
                    allocator_allocated_bytes=allocated,
                    allocator_reserved_bytes=reserved,
                    device_used_bytes=reserved + 5,
                    device_total_bytes=None,
                    context=None,
                )
            )

    events = cast(list[TelemetryEventV2], synthetic_events)
    model = build_distributed_model(events)

    assert len(model.present_ranks) == 64
    assert len(model.expected_ranks) == 64
    assert len(model.rows) == 64
    assert len(model.per_rank_timelines) == 64
    assert all(
        row.samples == 2000 for row in model.rows if row.availability == "present"
    )
