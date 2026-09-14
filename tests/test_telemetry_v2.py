"""Tests for TelemetryEvent v2 schema and legacy conversions."""

from __future__ import annotations

import csv
import json
from collections import UserDict
from pathlib import Path

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from stormlog.phases import parse_phase_boundary
from stormlog.telemetry import (
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
    UNKNOWN_HOST,
    UNKNOWN_PID,
    TelemetryEventV2,
    TelemetryEventV4,
    load_telemetry_events,
    load_telemetry_sessions,
    resolve_distributed_identity,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig
from stormlog.tui.distributed_diagnostics import load_distributed_artifacts


def _schema(version: int) -> dict[str, object]:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / f"telemetry_event_v{version}.schema.json"
    )
    result: dict[str, object] = json.loads(schema_path.read_text(encoding="utf-8"))
    return result


def _make_valid_event() -> TelemetryEventV2:
    return TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
        timestamp_ns=1_700_000_000_000_000_000,
        event_type="sample",
        collector="stormlog.cuda_tracker",
        sampling_interval_ms=100,
        pid=1234,
        host="host-a",
        job_id="job-123",
        rank=1,
        local_rank=1,
        world_size=8,
        device_id=0,
        allocator_allocated_bytes=1024,
        allocator_reserved_bytes=2048,
        allocator_active_bytes=512,
        allocator_inactive_bytes=1536,
        allocator_change_bytes=256,
        device_used_bytes=2048,
        device_free_bytes=4096,
        device_total_bytes=6144,
        context="unit",
        metadata={"origin": "test"},
    )


def test_telemetry_event_v2_serialization_validates_against_schema() -> None:
    event = _make_valid_event()
    record = telemetry_event_to_dict(event)

    validate_telemetry_record(record)
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V2))


def test_validate_telemetry_record_rejects_missing_fields() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record.pop("collector")

    with pytest.raises(ValueError, match="Missing required telemetry fields"):
        validate_telemetry_record(record)


def test_validate_telemetry_record_rejects_negative_allocator_counter() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["allocator_allocated_bytes"] = -1

    with pytest.raises(ValueError, match="allocator_allocated_bytes must be >= 0"):
        validate_telemetry_record(record)


def test_validate_telemetry_record_rejects_unknown_fields() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["unknown_counter"] = 42

    with pytest.raises(ValueError, match=r"Unknown telemetry fields: unknown_counter"):
        validate_telemetry_record(record)


def test_validate_telemetry_record_rejects_non_dict_metadata() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["metadata"] = UserDict({"origin": "wrapped"})

    with pytest.raises(ValueError, match="metadata must be an object"):
        validate_telemetry_record(record)


def test_v4_accepts_device_only_memory_counters() -> None:
    capabilities = {
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
    event = TelemetryEventV4(
        schema_version=SCHEMA_VERSION_V4,
        session_id="device-only",
        timestamp_ns=1,
        event_type="sample",
        collector="stormlog.future_tracker",
        sampling_interval_ms=100,
        pid=1,
        host="host",
        device_id=0,
        allocator_allocated_bytes=None,
        allocator_reserved_bytes=None,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=None,
        device_used_bytes=3,
        device_free_bytes=1,
        device_total_bytes=4,
        context=None,
        metadata={"memory_capabilities": capabilities},
    )

    record = telemetry_event_to_dict(event)

    validate_telemetry_record(record)
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V4))


def test_v4_rejects_malformed_capability_metadata() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["schema_version"] = SCHEMA_VERSION_V4
    record["session_id"] = "session"
    record["metadata"] = {"memory_capabilities": {"backend": "cuda"}}

    with pytest.raises(ValueError, match="Missing memory capability fields"):
        validate_telemetry_record(record)


def test_v4_rejects_unknown_capability_fields() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["schema_version"] = SCHEMA_VERSION_V4
    record["session_id"] = "session"
    record = telemetry_event_to_dict(telemetry_event_from_record(record))
    record["metadata"]["memory_capabilities"]["future_flag"] = True

    with pytest.raises(ValueError, match="Unknown memory capability fields"):
        validate_telemetry_record(record)


def test_v4_rejects_counter_declared_unsupported() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["schema_version"] = SCHEMA_VERSION_V4
    record["session_id"] = "session"
    record = telemetry_event_to_dict(telemetry_event_from_record(record))
    capabilities = record["metadata"]["memory_capabilities"]
    capabilities["supports_device_used"] = False

    with pytest.raises(ValueError, match="device_used_bytes must be null"):
        validate_telemetry_record(record)


@pytest.mark.parametrize("schema_version", [SCHEMA_VERSION_V2, SCHEMA_VERSION_V3])
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"timestamp_ns": -1}, "timestamp_ns must be >= 0"),
        ({"sampling_interval_ms": -1}, "sampling_interval_ms must be >= 0"),
        ({"pid": -2}, "pid must be >= -1"),
        ({"allocator_reserved_bytes": -1}, "allocator_reserved_bytes must be >= 0"),
        (
            {"allocator_active_bytes": -1},
            "allocator_active_bytes must be >= 0 when provided",
        ),
        (
            {"allocator_inactive_bytes": -1},
            "allocator_inactive_bytes must be >= 0 when provided",
        ),
        ({"device_used_bytes": -1}, "device_used_bytes must be >= 0"),
        ({"device_free_bytes": -1}, "device_free_bytes must be >= 0 when provided"),
        ({"device_total_bytes": -1}, "device_total_bytes must be >= 0 when provided"),
        (
            {"device_used_bytes": 6145},
            "device_used_bytes cannot exceed device_total_bytes",
        ),
        (
            {"device_free_bytes": 6145},
            "device_free_bytes cannot exceed device_total_bytes",
        ),
        # Coercion precedes range checks within each group of memory counters.
        (
            {"allocator_allocated_bytes": -1, "allocator_reserved_bytes": "bad"},
            "allocator_reserved_bytes must be an integer",
        ),
        (
            {"device_used_bytes": -1, "device_total_bytes": "bad"},
            "device_total_bytes must be an integer",
        ),
        (
            {"rank": -1, "device_id": "bad"},
            "rank must be >= 0",
        ),
        (
            {"rank": 8, "metadata": []},
            "metadata must be an object",
        ),
    ],
)
def test_telemetry_validation_preserves_error_order(
    schema_version: int, updates: dict[str, object], message: str
) -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record.update(schema_version=schema_version, **updates)
    if schema_version == SCHEMA_VERSION_V3:
        record["session_id"] = "validation-session"

    with pytest.raises(ValueError) as exc_info:
        validate_telemetry_record(record)

    assert str(exc_info.value) == message


def test_telemetry_validation_accepts_nullable_counters_and_unknown_pid() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record.update(
        pid=-1,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=-1024,
        device_free_bytes=None,
        device_total_bytes=None,
    )
    original = dict(record)

    validate_telemetry_record(record)

    assert record == original


def test_legacy_gpumemprof_record_converts_to_v4() -> None:
    legacy = {
        "timestamp": 1700000000.25,
        "event_type": "allocation",
        "memory_allocated": 10_000,
        "memory_reserved": 15_000,
        "memory_change": 512,
        "device_id": 0,
        "context": "alloc",
        "metadata_usage_percent": 75.5,
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="stormlog.cuda_tracker",
        default_sampling_interval_ms=100,
    )
    record = telemetry_event_to_dict(event)

    assert record["schema_version"] == SCHEMA_VERSION_V4
    assert isinstance(record["session_id"], str)
    assert record["session_id"]
    assert record["collector"] == "stormlog.cuda_tracker"
    assert record["allocator_allocated_bytes"] == 10_000
    assert record["allocator_reserved_bytes"] == 15_000
    assert record["allocator_change_bytes"] == 512
    assert record["metadata"]["usage_percent"] == 75.5
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V4))


def test_legacy_cpu_record_converts_with_defaults() -> None:
    legacy = {
        "timestamp": 1700000001.0,
        "event_type": "allocation",
        "memory_allocated": 8_192,
        "memory_change": 1_024,
        "context": "cpu",
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="stormlog.cpu_tracker",
        default_sampling_interval_ms=200,
    )
    record = telemetry_event_to_dict(event)

    assert record["collector"] == "stormlog.cpu_tracker"
    assert record["device_id"] == -1
    assert record["allocator_reserved_bytes"] == record["allocator_allocated_bytes"]
    assert record["device_total_bytes"] is None
    assert record["pid"] == UNKNOWN_PID
    assert record["host"] == UNKNOWN_HOST
    assert record["job_id"] is None
    assert record["rank"] == 0
    assert record["local_rank"] == 0
    assert record["world_size"] == 1
    assert isinstance(record["session_id"], str)
    assert record["session_id"]
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V4))


def test_legacy_record_uses_backend_metadata_for_collector() -> None:
    legacy = {
        "timestamp": 1700000001.0,
        "event_type": "sample",
        "memory_allocated": 4096,
        "memory_reserved": 8192,
        "memory_change": 0,
        "metadata": {"backend": "mps"},
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="legacy.unknown",
        default_sampling_interval_ms=100,
    )
    record = telemetry_event_to_dict(event)

    assert record["collector"] == "stormlog.mps_tracker"


def test_legacy_tf_record_converts_with_defaults() -> None:
    legacy = {
        "timestamp": 1700000002.5,
        "type": "sample",
        "memory_mb": 2.0,
        "device": "/GPU:0",
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="legacy.unknown",
        default_sampling_interval_ms=500,
    )
    record = telemetry_event_to_dict(event)

    assert record["collector"] == "stormlog.tensorflow.memory_tracker"
    assert record["device_id"] == 0
    assert record["allocator_allocated_bytes"] == 2 * 1024 * 1024
    assert record["device_used_bytes"] == 2 * 1024 * 1024
    assert isinstance(record["session_id"], str)
    assert record["session_id"]
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V4))


def test_phase_boundary_record_round_trips_through_v3_schema() -> None:
    event = telemetry_event_from_record(
        {
            "schema_version": SCHEMA_VERSION_V3,
            "session_id": "session-phase",
            "timestamp_ns": 1_700_000_000_000_000_100,
            "event_type": "phase_enter",
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "host-a",
            "job_id": "job-123",
            "rank": 1,
            "local_rank": 1,
            "world_size": 8,
            "device_id": 0,
            "allocator_allocated_bytes": 1024,
            "allocator_reserved_bytes": 2048,
            "allocator_active_bytes": 512,
            "allocator_inactive_bytes": 1536,
            "allocator_change_bytes": 0,
            "device_used_bytes": 2048,
            "device_free_bytes": 4096,
            "device_total_bytes": 6144,
            "context": "Phase entered: train / step",
            "metadata": {
                "phase_scope": {
                    "action": "enter",
                    "name": "step",
                    "path": ["train", "step"],
                    "depth": 2,
                    "scope_id": "session-phase:2",
                    "parent_scope_id": "session-phase:1",
                    "thread_id": 88,
                    "thread_name": "MainThread",
                    "sequence": 2,
                    "attributes": {"epoch": 3},
                }
            },
        }
    )

    record = telemetry_event_to_dict(event)

    validate_telemetry_record(record)
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V4))
    scope = parse_phase_boundary(record)
    assert scope is not None
    assert scope.path == ("train", "step")
    assert scope.attributes == {"epoch": 3}


def test_load_telemetry_sessions_preserves_phase_scope_metadata_across_formats(
    tmp_path: Path,
) -> None:
    sample_record = telemetry_event_to_dict(
        telemetry_event_from_record(
            {
                "schema_version": SCHEMA_VERSION_V3,
                "session_id": "session-phase",
                "timestamp_ns": 1_700_000_000_000_000_000,
                "event_type": "sample",
                "collector": "stormlog.cuda_tracker",
                "sampling_interval_ms": 100,
                "pid": 1234,
                "host": "host-a",
                "job_id": "job-123",
                "rank": 1,
                "local_rank": 1,
                "world_size": 8,
                "device_id": 0,
                "allocator_allocated_bytes": 1024,
                "allocator_reserved_bytes": 2048,
                "allocator_active_bytes": 512,
                "allocator_inactive_bytes": 1536,
                "allocator_change_bytes": 0,
                "device_used_bytes": 2048,
                "device_free_bytes": 4096,
                "device_total_bytes": 6144,
                "context": "sample",
                "metadata": {},
            }
        )
    )
    enter_record = dict(sample_record)
    enter_record.update(
        {
            "timestamp_ns": sample_record["timestamp_ns"] + 10,
            "event_type": "phase_enter",
            "context": "Phase entered: train / step",
            "metadata": {
                "phase_scope": {
                    "action": "enter",
                    "name": "step",
                    "path": ["train", "step"],
                    "depth": 2,
                    "scope_id": "session-phase:2",
                    "parent_scope_id": "session-phase:1",
                    "thread_id": 88,
                    "thread_name": "MainThread",
                    "sequence": 2,
                    "attributes": {"epoch": 3},
                }
            },
        }
    )
    exit_record = dict(enter_record)
    exit_record.update(
        {
            "timestamp_ns": enter_record["timestamp_ns"] + 10,
            "event_type": "phase_exit",
            "context": "Phase exited: train / step",
            "metadata": {
                "phase_scope": {
                    "action": "exit",
                    "name": "step",
                    "path": ["train", "step"],
                    "depth": 2,
                    "scope_id": "session-phase:2",
                    "parent_scope_id": "session-phase:1",
                    "thread_id": 88,
                    "thread_name": "MainThread",
                    "sequence": 3,
                    "attributes": {"epoch": 3},
                }
            },
        }
    )
    records = [sample_record, enter_record, exit_record]

    json_path = tmp_path / "events.json"
    json_path.write_text(json.dumps(records), encoding="utf-8")

    csv_path = tmp_path / "events.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_record.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value)
                        if isinstance(value, dict)
                        else ("" if value is None else str(value))
                    )
                    for key, value in record.items()
                }
            )

    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    for record in records:
        sink.append(record)
    sink.close()

    for source in (json_path, tmp_path / "sink"):
        loaded = load_telemetry_sessions(source, permissive_legacy=True)
        assert len(loaded) == 1
        phase_events = [
            event for event in loaded[0].events if event.event_type.startswith("phase_")
        ]
        assert [event.event_type for event in phase_events] == [
            "phase_enter",
            "phase_exit",
        ]
        enter_scope = parse_phase_boundary(phase_events[0])
        exit_scope = parse_phase_boundary(phase_events[1])
        assert enter_scope is not None
        assert exit_scope is not None
        assert enter_scope.path == ("train", "step")
        assert enter_scope.attributes == {"epoch": 3}
        assert exit_scope.scope_id == enter_scope.scope_id

    artifact_result = load_distributed_artifacts([csv_path])
    csv_phase_events = [
        event
        for event in artifact_result.events
        if event.event_type.startswith("phase_")
    ]
    assert [event.event_type for event in csv_phase_events] == [
        "phase_enter",
        "phase_exit",
    ]
    csv_enter_scope = parse_phase_boundary(csv_phase_events[0])
    csv_exit_scope = parse_phase_boundary(csv_phase_events[1])
    assert csv_enter_scope is not None
    assert csv_exit_scope is not None
    assert csv_enter_scope.path == ("train", "step")
    assert csv_enter_scope.attributes == {"epoch": 3}
    assert csv_exit_scope.scope_id == csv_enter_scope.scope_id


def test_resolve_distributed_identity_uses_torchrun_env() -> None:
    identity = resolve_distributed_identity(
        env={
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "TORCHELASTIC_RUN_ID": "train-42",
        }
    )

    assert identity == {
        "job_id": "train-42",
        "rank": 3,
        "local_rank": 1,
        "world_size": 8,
    }


def test_resolve_distributed_identity_prefers_explicit_overrides() -> None:
    identity = resolve_distributed_identity(
        job_id="manual-job",
        rank=5,
        local_rank=2,
        world_size=16,
        env={"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"},
    )

    assert identity == {
        "job_id": "manual-job",
        "rank": 5,
        "local_rank": 2,
        "world_size": 16,
    }


def test_resolve_distributed_identity_explicit_overrides_bypass_partial_env() -> None:
    identity = resolve_distributed_identity(
        rank=5,
        local_rank=2,
        world_size=16,
        env={"WORLD_SIZE": "8"},
    )

    assert identity == {
        "job_id": None,
        "rank": 5,
        "local_rank": 2,
        "world_size": 16,
    }


def test_resolve_distributed_identity_reads_job_id_without_rank_env_inference() -> None:
    identity = resolve_distributed_identity(
        rank=5,
        local_rank=2,
        world_size=16,
        env={
            "RANK": "oops",
            "WORLD_SIZE": "8",
            "TORCHELASTIC_RUN_ID": "train-42",
        },
    )

    assert identity == {
        "job_id": "train-42",
        "rank": 5,
        "local_rank": 2,
        "world_size": 16,
    }


def test_resolve_distributed_identity_skips_partial_env() -> None:
    identity = resolve_distributed_identity(
        env={"WORLD_SIZE": "8", "TORCHELASTIC_RUN_ID": "train-42"}
    )

    assert identity == {
        "job_id": "train-42",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
    }


def test_resolve_distributed_identity_keeps_inferred_local_rank() -> None:
    identity = resolve_distributed_identity(
        rank=7,
        world_size=16,
        env={"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"},
    )

    assert identity == {
        "job_id": None,
        "rank": 7,
        "local_rank": 1,
        "world_size": 16,
    }


def test_validate_telemetry_record_rejects_invalid_rank_metadata() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["rank"] = 8

    with pytest.raises(ValueError, match="rank must be < world_size"):
        telemetry_event_from_record(record)


def test_v2_record_keeps_metadata_identity_keys_opaque() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record.pop("job_id")
    record.pop("rank")
    record.pop("local_rank")
    record.pop("world_size")
    record["metadata"] = {
        "job_id": "meta-job",
        "rank": "metadata-rank",
        "local_rank": "metadata-local-rank",
        "world_size": "metadata-world-size",
        "origin": "test",
    }

    validate_telemetry_record(record)
    round_tripped = telemetry_event_to_dict(telemetry_event_from_record(record))

    assert round_tripped["job_id"] is None
    assert round_tripped["rank"] == 0
    assert round_tripped["local_rank"] == 0
    assert round_tripped["world_size"] == 1
    for key, value in record["metadata"].items():
        assert round_tripped["metadata"][key] == value
    assert (
        round_tripped["metadata"]["memory_capabilities"]["supports_allocator_allocated"]
        is True
    )


def test_load_telemetry_events_reads_dict_events_payload(tmp_path: Path) -> None:
    payload = {
        "peak_memory": 123.4,
        "events": [
            {
                "timestamp": 1700000003.0,
                "type": "sample",
                "memory_mb": 1.0,
                "device": "/GPU:0",
            }
        ],
    }
    path = tmp_path / "tf_track.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    events = load_telemetry_events(path, events_key="events")

    assert len(events) == 1
    assert events[0].schema_version == SCHEMA_VERSION_V4


def test_load_telemetry_events_reads_append_only_sink_directory(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    first = telemetry_event_to_dict(_make_valid_event())
    second = telemetry_event_to_dict(_make_valid_event())
    second["timestamp_ns"] = first["timestamp_ns"] + 1
    sink.append(first)
    sink.append(second)
    sink.close()

    events = load_telemetry_events(tmp_path / "sink")

    assert len(events) == 2
    assert events[0].timestamp_ns == first["timestamp_ns"]
    assert events[1].timestamp_ns == second["timestamp_ns"]


def test_load_telemetry_events_reads_unlisted_sink_segments(tmp_path: Path) -> None:
    sink_dir = tmp_path / "sink"
    sink_dir.mkdir()

    first = telemetry_event_to_dict(_make_valid_event())
    second = telemetry_event_to_dict(_make_valid_event())
    second["timestamp_ns"] = first["timestamp_ns"] + 1

    first_payload = json.dumps(first) + "\n"
    second_payload = json.dumps(second) + "\n"
    (sink_dir / "segment-000001.jsonl").write_text(first_payload, encoding="utf-8")
    (sink_dir / "segment-000002.jsonl").write_text(second_payload, encoding="utf-8")
    (sink_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.append_only_telemetry_sink",
                "segments": [
                    {
                        "filename": "segment-000001.jsonl",
                        "event_count": 1,
                        "size_bytes": len(first_payload.encode("utf-8")),
                        "closed": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = load_telemetry_events(sink_dir)
    assert len(events) == 2
    assert events[0].timestamp_ns == first["timestamp_ns"]
    assert events[1].timestamp_ns == second["timestamp_ns"]


def test_load_telemetry_events_prefers_non_empty_session_over_empty_completed_summary(
    tmp_path: Path,
) -> None:
    sink_dir = tmp_path / "sink"
    sink_dir.mkdir()

    record = telemetry_event_to_dict(_make_valid_event())
    payload = json.dumps(record) + "\n"
    segment_path = sink_dir / "segment-000001.jsonl"
    segment_path.write_text(payload, encoding="utf-8")
    (sink_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "stormlog.append_only_telemetry_sink",
                "sessions": [
                    {
                        "session_id": "completed-empty",
                        "status": "completed",
                        "started_at_ns": record["timestamp_ns"] - 100,
                        "ended_at_ns": record["timestamp_ns"] - 50,
                        "host": record["host"],
                        "pid": record["pid"],
                        "job_id": record["job_id"],
                        "rank": record["rank"],
                        "local_rank": record["local_rank"],
                        "world_size": record["world_size"],
                        "source": "stormlog.telemetry_sink",
                    },
                    {
                        "session_id": "running-live",
                        "status": "running",
                        "started_at_ns": record["timestamp_ns"],
                        "ended_at_ns": None,
                        "host": record["host"],
                        "pid": record["pid"],
                        "job_id": record["job_id"],
                        "rank": record["rank"],
                        "local_rank": record["local_rank"],
                        "world_size": record["world_size"],
                        "source": "stormlog.telemetry_sink",
                    },
                ],
                "segments": [
                    {
                        "filename": segment_path.name,
                        "event_count": 1,
                        "size_bytes": len(payload.encode("utf-8")),
                        "closed": False,
                        "session_id": "running-live",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sessions = load_telemetry_sessions(sink_dir)
    assert [session.summary.session_id for session in sessions] == [
        "completed-empty",
        "running-live",
    ]
    assert sessions[0].events == []
    assert len(sessions[1].events) == 1

    events = load_telemetry_events(sink_dir)
    assert len(events) == 1
    assert events[0].session_id == "running-live"

    assert load_telemetry_events(sink_dir, session_id="completed-empty") == []
    targeted = load_telemetry_events(sink_dir, session_id="running-live")
    assert len(targeted) == 1
    assert targeted[0].session_id == "running-live"


def test_append_only_sink_truncates_partial_segment_tail_on_resume(
    tmp_path: Path,
) -> None:
    sink_dir = tmp_path / "sink"
    sink_dir.mkdir()

    first = telemetry_event_to_dict(_make_valid_event())
    second = telemetry_event_to_dict(_make_valid_event())
    second["timestamp_ns"] = first["timestamp_ns"] + 1

    first_payload = json.dumps(first) + "\n"
    partial_payload = '{"schema_version": 2, "timestamp_ns":'
    segment_path = sink_dir / "segment-000001.jsonl"
    segment_path.write_text(
        first_payload + partial_payload,
        encoding="utf-8",
    )
    (sink_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.append_only_telemetry_sink",
                "segments": [
                    {
                        "filename": segment_path.name,
                        "event_count": 1,
                        "size_bytes": len(
                            (first_payload + partial_payload).encode("utf-8")
                        ),
                        "closed": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=sink_dir,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(second)
    sink.close()

    sessions = load_telemetry_sessions(sink_dir)
    assert len(sessions) == 2
    assert [session.summary.status for session in sessions] == [
        "completed",
        "incomplete",
    ]
    assert sessions[0].events[0].timestamp_ns == second["timestamp_ns"]
    assert sessions[1].events[0].timestamp_ns == first["timestamp_ns"]

    events = load_telemetry_events(sink_dir)
    assert len(events) == 1
    assert events[0].timestamp_ns == second["timestamp_ns"]


def test_load_telemetry_events_ignores_truncated_jsonl_tail(tmp_path: Path) -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    segment = tmp_path / "segment-000001.jsonl"
    segment.write_text(
        json.dumps(record) + "\n" + '{"schema_version": 2, "timestamp_ns":',
        encoding="utf-8",
    )

    events = load_telemetry_events(segment)

    assert len(events) == 1
    assert events[0].timestamp_ns == record["timestamp_ns"]


def test_legacy_conversion_can_be_disabled() -> None:
    with pytest.raises(ValueError, match="Legacy record conversion is disabled"):
        telemetry_event_from_record(
            {"timestamp": 1.0, "memory_allocated": 1},
            permissive_legacy=False,
        )


def test_v2_record_missing_required_nullable_field_is_rejected() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record.pop("context")

    with pytest.raises(ValueError, match="Missing required telemetry fields"):
        telemetry_event_from_record(record)


def test_v2_record_with_unknown_field_is_rejected() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["unknown_counter"] = 42

    with pytest.raises(ValueError, match=r"Unknown telemetry fields: unknown_counter"):
        telemetry_event_from_record(record)


def test_schema_version_must_be_integer_when_present() -> None:
    legacy = {
        "schema_version": "2",
        "timestamp": 1700000001.0,
        "event_type": "allocation",
        "memory_allocated": 8_192,
    }

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        telemetry_event_from_record(legacy)


def test_schema_version_bool_is_rejected_when_present() -> None:
    legacy = {
        "schema_version": True,
        "timestamp": 1700000001.0,
        "event_type": "allocation",
        "memory_allocated": 8_192,
    }

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        telemetry_event_from_record(legacy)


def test_unsupported_schema_version_is_rejected_without_legacy_fallback() -> None:
    legacy = {
        "schema_version": 5,
        "timestamp": 1700000001.0,
        "event_type": "allocation",
        "memory_allocated": 8_192,
    }

    with pytest.raises(ValueError, match="Unsupported schema_version: 5"):
        telemetry_event_from_record(legacy, permissive_legacy=True)


def test_load_telemetry_events_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "schema_version": 5,
            "timestamp": 1700000005.0,
            "event_type": "allocation",
            "memory_allocated": 1024,
        }
    ]
    path = tmp_path / "unsupported_schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported schema_version: 5"):
        load_telemetry_events(path, permissive_legacy=True)


def test_legacy_total_memory_null_is_accepted() -> None:
    legacy = {
        "timestamp": 1700000004.0,
        "event_type": "allocation",
        "memory_allocated": 1024,
        "memory_reserved": 2048,
        "memory_change": 256,
        "device_id": 0,
        "total_memory": None,
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="stormlog.cuda_tracker",
        default_sampling_interval_ms=100,
    )
    record = telemetry_event_to_dict(event)

    assert record["schema_version"] == SCHEMA_VERSION_V4
    assert isinstance(record["session_id"], str)
    assert record["session_id"]
    assert record["device_total_bytes"] is None
    assert record["device_free_bytes"] is None
    jsonschema.validate(instance=record, schema=_schema(SCHEMA_VERSION_V4))
