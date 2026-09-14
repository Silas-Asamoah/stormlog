from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

_stormlog_phases: Any
try:
    import stormlog.phases as _stormlog_phases
except ImportError:  # pragma: no cover - phase package may land in another slice
    _stormlog_phases = None

from stormlog.device_collectors import DeviceMemoryCapabilities
from stormlog.session import create_session_summary, stable_legacy_session_id
from stormlog.telemetry import (
    LoadedTelemetrySession,
    TelemetryEvent,
    TelemetryEventV2,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from stormlog.timeline_markers import MARKER_KIND_ALERT, MARKER_KIND_PHASE
from stormlog.tui import distributed_diagnostics as diagnostics
from stormlog.tui.distributed_diagnostics import (
    build_distributed_model,
    load_distributed_artifacts,
    parse_rank_filter,
)


def _make_event(
    *,
    timestamp: float,
    rank: int,
    world_size: int,
    session_id: str | None = None,
    event_type: str = "sample",
    allocated: int = 0,
    reserved: int = 0,
    used: int = 0,
    total: int | None = None,
    context: str = "",
    metadata: dict[str, object] | None = None,
) -> TelemetryEvent:
    if session_id is not None:
        used_bytes = used or reserved
        record: dict[str, object] = {
            "schema_version": 3,
            "session_id": session_id,
            "timestamp_ns": int(timestamp * 1_000_000_000),
            "event_type": event_type,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "test-host",
            "device_id": 0,
            "allocator_allocated_bytes": allocated,
            "allocator_reserved_bytes": reserved,
            "allocator_active_bytes": None,
            "allocator_inactive_bytes": None,
            "allocator_change_bytes": 0,
            "device_used_bytes": used_bytes,
            "device_free_bytes": None if total is None else max(0, total - used_bytes),
            "device_total_bytes": total,
            "context": context,
            "job_id": "job-1",
            "rank": rank,
            "local_rank": rank,
            "world_size": world_size,
            "metadata": metadata or {},
        }
    else:
        record = {
            "timestamp": timestamp,
            "event_type": event_type,
            "memory_allocated": allocated,
            "memory_reserved": reserved,
            "memory_change": 0,
            "device_used_bytes": used or reserved,
            "device_total_bytes": total,
            "device_id": 0,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "test-host",
            "context": context,
            "job_id": "job-1",
            "rank": rank,
            "local_rank": rank,
            "world_size": world_size,
            "metadata": metadata or {},
        }
    return telemetry_event_from_record(
        record,
        permissive_legacy=True,
        default_collector="stormlog.cuda_tracker",
        default_sampling_interval_ms=100,
    )


def _write_csv_events(path: Path, events: list[TelemetryEvent]) -> None:
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


def _make_device_only_event(*, timestamp: float, used: int) -> TelemetryEvent:
    return telemetry_event_from_record(
        {
            "schema_version": 4,
            "session_id": "device-only-session",
            "timestamp_ns": int(timestamp * 1_000_000_000),
            "event_type": "sample",
            "collector": "stormlog.future_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "test-host",
            "device_id": 0,
            "allocator_allocated_bytes": None,
            "allocator_reserved_bytes": None,
            "allocator_active_bytes": None,
            "allocator_inactive_bytes": None,
            "allocator_change_bytes": None,
            "device_used_bytes": used,
            "device_free_bytes": 100 - used,
            "device_total_bytes": 100,
            "context": "device sample",
            "job_id": "job-device",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "metadata": {
                "memory_capabilities": {
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
            },
        }
    )


def _flatten_result_events(result: object) -> list[TelemetryEvent]:
    sessions = getattr(result, "sessions")
    return [event for session in sessions for event in session.events]


def test_parse_rank_filter_supports_all_lists_and_ranges() -> None:
    available = list(range(8))
    assert parse_rank_filter("all", available) == set(available)
    assert parse_rank_filter("0,2,4-6", available) == {0, 2, 4, 5, 6}


def test_parse_rank_filter_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="start>end"):
        parse_rank_filter("5-2", [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    ("expression", "available", "expected"),
    [
        (" , 0, 0, 2-4, 99, ", [0, 2, 4], {0, 2, 4}),
        (" * ", [1, 3], {1, 3}),
        ("", [1], {1}),
        ("bad-range", [], set()),
    ],
)
def test_rank_filter_preserves_empty_and_unavailable_rank_semantics(
    expression: str, available: list[int], expected: set[int]
) -> None:
    assert parse_rank_filter(expression, available) == expected


@pytest.mark.parametrize("expression", ["-2", "2-", "4-2", "2-x", "x"])
def test_rank_filter_rejects_malformed_tokens(expression: str) -> None:
    with pytest.raises(ValueError):
        parse_rank_filter(expression, [0, 1, 2])


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


def test_build_distributed_model_degrades_to_device_only_diagnostics() -> None:
    model = build_distributed_model(
        [
            _make_device_only_event(timestamp=1.0, used=40),
            _make_device_only_event(timestamp=2.0, used=65),
        ]
    )

    assert model.rows[0].allocated_delta_bytes is None
    assert model.rows[0].reserved_delta_bytes is None
    assert model.rows[0].device_used_delta_bytes == 25
    assert model.rows[0].hidden_gap_latest_bytes is None
    assert model.per_rank_timelines[0] == {
        "timestamps_ns": [1_000_000_000, 2_000_000_000],
        "device_used": [40, 65],
    }
    assert model.indicators == []
    assert model.warnings == [
        "Allocator-native diagnostics are unavailable; showing device memory "
        "usage only."
    ]


def test_build_distributed_model_explains_disabled_fragmentation() -> None:
    events = [
        _make_event(
            timestamp=float(index),
            rank=0,
            world_size=1,
            allocated=100_000_000,
            reserved=1_000_000_000,
            used=2_500_000_000,
            total=16 * 1024**3,
        )
        for index in range(12)
    ]
    for event in events:
        event.metadata["memory_capabilities"]["supports_fragmentation_analysis"] = False

    model = build_distributed_model(events)

    assert all(
        "fragmentation" not in indicator.signal for indicator in model.indicators
    )
    assert model.per_rank_timelines[0]["allocated"] == [100_000_000] * 12
    assert model.warnings == ["Collector capabilities disable fragmentation analysis."]


@pytest.mark.parametrize(
    ("used_values", "reserved_values", "expected_gaps"),
    [
        ([None, None, None], [120, 140, 170], [None, None, None]),
        ([140, None, 190], [120, 140, 170], [20, None, 20]),
        ([140, 165, None], [120, 140, 170], [20, 25, None]),
        ([None, None, None], [None, None, None], [None, None, None]),
    ],
)
def test_allocator_timeline_preserves_partial_counters(
    used_values: list[int | None],
    reserved_values: list[int | None],
    expected_gaps: list[int | None],
) -> None:
    capabilities = DeviceMemoryCapabilities(
        backend="future",
        telemetry_collector="stormlog.future_tracker",
        sampling_source="future.memory",
        supports_allocator_allocated=True,
        supports_allocator_reserved=any(value is not None for value in reserved_values),
        supports_device_used=any(value is not None for value in used_values),
        supports_device_total=True,
    )
    events = [
        replace(
            _make_event(
                timestamp=float(index),
                rank=0,
                world_size=1,
                allocated=allocated,
                total=1000,
            ),
            collector="stormlog.future_tracker",
            allocator_reserved_bytes=reserved,
            device_used_bytes=used,
            device_free_bytes=None,
            metadata={"memory_capabilities": capabilities.to_metadata()},
        )
        for index, (allocated, reserved, used) in enumerate(
            zip([100, 125, 150], reserved_values, used_values), start=1
        )
    ]

    model = build_distributed_model(events)

    assert model.per_rank_timelines[0] == {
        "timestamps_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000],
        "allocated": [100, 125, 150],
        "reserved": reserved_values,
        "gap": expected_gaps,
    }
    assert model.rows[0].allocated_delta_bytes == 50
    assert all(
        "showing device memory usage only" not in warning for warning in model.warnings
    )


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


def test_build_distributed_model_surfaces_collective_attribution_signals() -> None:
    events: list[TelemetryEvent] = []
    for rank in (0, 1):
        for index in range(12):
            timestamp = 1.0 + index * 0.1 + rank * 0.001
            reserved = 2_000_000_000
            used = reserved + 40_000_000
            if index == 6:
                used = reserved + 1_600_000_000
            events.append(
                _make_event(
                    timestamp=timestamp,
                    rank=rank,
                    world_size=2,
                    allocated=reserved - 100_000_000,
                    reserved=reserved,
                    used=used,
                    total=16 * 1024**3,
                    context="sample",
                )
            )
            if index == 6:
                events.append(
                    _make_event(
                        timestamp=timestamp + 0.01,
                        rank=rank,
                        world_size=2,
                        event_type="collective",
                        allocated=reserved - 100_000_000,
                        reserved=reserved,
                        used=reserved,
                        total=16 * 1024**3,
                        context="NCCL all_reduce phase",
                        metadata={"phase": "communication.collective"},
                    )
                )

    model = build_distributed_model(events)
    collective_indicators = [
        indicator
        for indicator in model.indicators
        if indicator.signal.startswith("collective:")
    ]

    assert collective_indicators
    for indicator in collective_indicators:
        assert indicator.confidence is not None
        assert indicator.reason_codes
        assert "marker_collective_token" in indicator.reason_codes


def test_build_distributed_model_surfaces_phase_paths_in_rows_and_indicators() -> None:
    if _stormlog_phases is None:
        pytest.skip("stormlog.phases is not available in this slice")

    session_id = "session-diagnostics-phase"
    events = [
        _make_event(
            timestamp=0.9,
            session_id=session_id,
            rank=0,
            world_size=2,
            event_type="phase_enter",
            allocated=10,
            reserved=10,
            used=10,
            total=100,
            context="Phase entered: train / forward",
            metadata={
                "phase_scope": {
                    "action": "enter",
                    "name": "forward",
                    "path": ["train", "forward"],
                    "depth": 2,
                    "scope_id": "phase-0",
                    "parent_scope_id": "phase-train",
                    "thread_id": 1,
                    "thread_name": "MainThread",
                    "sequence": 1,
                }
            },
        ),
        _make_event(
            timestamp=1.0,
            session_id=session_id,
            rank=0,
            world_size=2,
            allocated=10,
            reserved=15,
            used=30,
            total=100,
            context="gap breach",
        ),
        _make_event(
            timestamp=1.1,
            session_id=session_id,
            rank=1,
            world_size=2,
            allocated=12,
            reserved=12,
            used=12,
            total=100,
        ),
        _make_event(
            timestamp=1.2,
            session_id=session_id,
            rank=0,
            world_size=2,
            event_type="phase_exit",
            allocated=10,
            reserved=10,
            used=10,
            total=100,
            context="Phase exited: train / forward",
            metadata={
                "phase_scope": {
                    "action": "exit",
                    "name": "forward",
                    "path": ["train", "forward"],
                    "depth": 2,
                    "scope_id": "phase-0",
                    "parent_scope_id": "phase-train",
                    "thread_id": 1,
                    "thread_name": "MainThread",
                    "sequence": 2,
                }
            },
        ),
    ]

    model = build_distributed_model(events)
    row_rank_0 = next(row for row in model.rows if row.rank == 0)
    earliest_indicator = next(
        indicator for indicator in model.indicators if indicator.kind == "earliest"
    )

    assert row_rank_0.first_anomaly_phase_path == "train / forward"
    assert earliest_indicator.phase_path == "train / forward"
    assert "Phase: train / forward." in earliest_indicator.details
    assert [marker.kind for marker in model.markers_by_rank[0]] == [MARKER_KIND_PHASE]
    assert model.markers_by_rank[0][0].label == "Phase: train / forward"


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
    json_record = telemetry_event_to_dict(json_event)
    json_record.pop("session_id")
    json_path.write_text(json.dumps([json_record]), encoding="utf-8")

    csv_path = tmp_path / "events.csv"
    csv_record = telemetry_event_to_dict(csv_event)
    csv_record.pop("session_id")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_record.keys()))
        writer.writeheader()
        writer.writerow(
            {
                key: (
                    json.dumps(value)
                    if isinstance(value, dict)
                    else ("" if value is None else str(value))
                )
                for key, value in csv_record.items()
            }
        )

    result = load_distributed_artifacts([json_path, csv_path])
    all_events = _flatten_result_events(result)
    assert len(result.events) == 2
    assert len(result.sessions) == 1
    assert len(all_events) == 2
    assert result.selected_session_id == result.sessions[0].summary.session_id
    assert result.warnings == []
    assert str(json_path) in result.sources_loaded
    assert str(csv_path) in result.sources_loaded


def test_load_distributed_artifacts_derives_selected_session_markers(
    tmp_path: Path,
) -> None:
    warning_event = telemetry_event_to_dict(
        _make_event(
            timestamp=1.0,
            session_id="session-markers",
            rank=0,
            world_size=1,
            event_type="warning",
            allocated=10,
            reserved=20,
            used=20,
            total=100,
            context="fragmentation warning",
        )
    )
    sample_event = telemetry_event_to_dict(
        _make_event(
            timestamp=2.0,
            session_id="session-markers",
            rank=0,
            world_size=1,
            allocated=12,
            reserved=22,
            used=22,
            total=100,
        )
    )
    json_path = tmp_path / "events.json"
    json_path.write_text(json.dumps([warning_event, sample_event]), encoding="utf-8")

    result = load_distributed_artifacts([json_path])

    assert [marker.kind for marker in result.markers] == [MARKER_KIND_ALERT]
    assert result.markers[0].label == "fragmentation warning"
    assert result.markers[0].session_id == "session-markers"


def test_load_distributed_artifacts_adds_merged_session_for_multi_rank_job_inputs(
    tmp_path: Path,
) -> None:
    first_event = telemetry_event_to_dict(
        _make_event(
            timestamp=1.0,
            rank=0,
            world_size=2,
            allocated=10,
            reserved=12,
            used=12,
            total=100,
        )
    )
    first_event["session_id"] = "session-a"
    second_event = telemetry_event_to_dict(
        _make_event(
            timestamp=2.0,
            rank=1,
            world_size=2,
            allocated=20,
            reserved=24,
            used=24,
            total=100,
        )
    )
    second_event["session_id"] = "session-b"

    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps([first_event]), encoding="utf-8")
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps([second_event]), encoding="utf-8")

    result = load_distributed_artifacts([first_path, second_path])
    assert len(result.events) == 2
    assert len(result.sessions) == 3
    assert {session.summary.session_id for session in result.sessions} >= {
        "session-a",
        "session-b",
    }
    assert result.selected_session_id not in {"session-a", "session-b"}

    selected = next(
        session
        for session in result.sessions
        if session.summary.session_id == result.selected_session_id
    )
    assert sorted({event.rank for event in selected.events}) == [0, 1]
    assert selected.summary.job_id == "job-1"
    assert selected.summary.world_size == 2

    explicit = load_distributed_artifacts(
        [first_path, second_path],
        session_id="session-a",
    )
    assert explicit.selected_session_id == "session-a"
    assert len(explicit.events) == 1
    assert explicit.events[0].rank == 0


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("second_status", ["completed", "incomplete"])
def test_session_merge_preserves_summary_identity_order_and_deduplication(
    distributed: bool,
    second_status: str,
) -> None:
    early = _make_event(timestamp=1, rank=0, world_size=2, session_id="session-a")
    late = _make_event(timestamp=2, rank=1, world_size=2, session_id="session-b")
    first = LoadedTelemetrySession(
        summary=create_session_summary(
            source="rank-a",
            status="completed",
            session_id="session-a",
            started_at_ns=10,
            ended_at_ns=30,
            host="host-a",
            pid=1,
            job_id="job-1",
            rank=0,
            local_rank=1,
            world_size=2,
        ),
        events=[late, early],
        sources_loaded=["b.json", "a.json"],
        warnings=["first warning", "shared warning"],
    )
    second = LoadedTelemetrySession(
        summary=create_session_summary(
            source="rank-b",
            status=second_status,
            session_id="session-b",
            started_at_ns=20,
            ended_at_ns=40,
            host="host-b",
            pid=2,
            job_id="job-1",
            rank=1,
            local_rank=0,
            world_size=2,
        ),
        events=[replace(early, session_id="session-b"), late],
        sources_loaded=["b.json"],
        warnings=["shared warning", "last warning"],
    )

    if distributed:
        merged = diagnostics._merge_distributed_path_sessions([first, second])
        expected_id = stable_legacy_session_id(
            "distributed.rank_group", "job-1", 2, "session-a", "session-b"
        )
        expected_source = "stormlog.diagnostics.distributed"
    else:
        merged = diagnostics._merge_legacy_flat_sessions([first, second])
        expected_id = stable_legacy_session_id("distributed.legacy", "a.json", "b.json")
        expected_source = "stormlog.diagnostics.legacy"

    assert merged.summary == create_session_summary(
        source=expected_source,
        status=second_status,
        session_id=expected_id,
        started_at_ns=10,
        ended_at_ns=40 if second_status == "completed" else None,
        host="multiple",
        pid=-1,
        job_id="job-1",
        rank=0,
        local_rank=0,
        world_size=2,
    )
    assert merged.events == [
        replace(early, session_id=expected_id),
        replace(late, session_id=expected_id),
    ]
    assert merged.sources_loaded == ["a.json", "b.json"]
    assert merged.warnings == ["first warning", "shared warning", "last warning"]


def test_artifact_loader_empty_inputs_and_missing_selection(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    result = load_distributed_artifacts([missing])

    assert result.events == []
    assert result.sessions == []
    assert result.markers == []
    assert result.sources_loaded == []
    assert result.selected_session_id is None
    assert result.warnings == [f"Path does not exist: {missing}"]
    with pytest.raises(ValueError, match="Requested session_id not found: absent"):
        load_distributed_artifacts([], session_id="absent")


def test_distributed_model_filters_missing_ranks_and_orders_samples() -> None:
    events = [
        _make_event(timestamp=3, rank=0, world_size=3, allocated=30),
        _make_event(timestamp=2, rank=1, world_size=2, event_type="warning"),
        _make_event(timestamp=1, rank=0, world_size=3, allocated=10),
    ]

    model = build_distributed_model(events, selected_ranks={0, 2, 8})

    assert model.expected_ranks == [0, 2]
    assert model.present_ranks == [0]
    assert model.missing_ranks == [2]
    assert [row.rank for row in model.rows] == [0, 2]
    assert model.rows[0].allocated_delta_bytes == 20
    assert model.per_rank_timelines[0]["allocated"] == [10, 30]
    assert model.indicators == []
    assert model.warnings == [
        "Inconsistent world_size values detected; using max observed world_size."
    ]


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
    all_events = _flatten_result_events(result)

    assert sorted({event.rank for event in all_events}) == [0, 1]
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
    all_events = _flatten_result_events(result)
    model = build_distributed_model(all_events)

    rank_to_allocated: dict[int, list[int]] = {}
    for event in all_events:
        allocated = event.allocator_allocated_bytes
        assert allocated is not None
        rank_to_allocated.setdefault(event.rank, []).append(allocated)

    assert sorted(rank_to_allocated.keys()) == [0, 1]
    assert sorted(rank_to_allocated[0]) == [100, 120]
    assert sorted(rank_to_allocated[1]) == [200, 220]
    assert model.present_ranks == [0, 1]
    assert str(rank0_timeline) in result.sources_loaded
    assert str(rank1_timeline) in result.sources_loaded


def test_load_distributed_artifacts_ignores_local_rank_path_segments_for_inference(
    tmp_path: Path,
) -> None:
    local_dash_bundle = tmp_path / "node-local-rank3"
    local_underscore_bundle = tmp_path / "node_local_rank4"
    local_dash_bundle.mkdir()
    local_underscore_bundle.mkdir()

    local_dash_timeline = local_dash_bundle / "telemetry_timeline.json"
    local_underscore_timeline = local_underscore_bundle / "telemetry_timeline.json"
    local_dash_timeline.write_text(
        json.dumps(
            {
                "timestamps": [1.0, 2.0],
                "allocated": [100, 120],
                "reserved": [120, 140],
            }
        ),
        encoding="utf-8",
    )
    local_underscore_timeline.write_text(
        json.dumps(
            {
                "timestamps": [1.0, 2.0],
                "allocated": [200, 220],
                "reserved": [220, 240],
            }
        ),
        encoding="utf-8",
    )

    result = load_distributed_artifacts([local_dash_bundle, local_underscore_bundle])
    all_events = _flatten_result_events(result)
    model = build_distributed_model(all_events)

    assert sorted({event.rank for event in all_events}) == [0, 1]
    assert model.present_ranks == [0, 1]
    assert model.missing_ranks == []
    assert str(local_dash_timeline) in result.sources_loaded
    assert str(local_underscore_timeline) in result.sources_loaded


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
