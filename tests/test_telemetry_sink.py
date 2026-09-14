from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from stormlog.telemetry import load_telemetry_sessions
from stormlog.telemetry_rollups import ROLLUP_FILENAME, read_telemetry_rollups
from stormlog.telemetry_sink import (
    AppendOnlyTelemetrySink,
    TelemetrySinkConfig,
    read_telemetry_sink_manifest,
)


def _segment_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _event_record(
    *,
    session_id: str,
    timestamp_ns: int,
    event_type: str = "sample",
    rank: int = 0,
    allocated: int = 100,
    reserved: int = 150,
    used: int = 175,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "session_id": session_id,
        "timestamp_ns": timestamp_ns,
        "event_type": event_type,
        "collector": "stormlog.cuda_tracker",
        "sampling_interval_ms": 100,
        "pid": 123,
        "host": "host-a",
        "job_id": "job-a",
        "rank": rank,
        "local_rank": rank,
        "world_size": 2,
        "device_id": 0,
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": reserved,
        "allocator_active_bytes": None,
        "allocator_inactive_bytes": None,
        "allocator_change_bytes": reserved - allocated,
        "device_used_bytes": used,
        "device_free_bytes": None,
        "device_total_bytes": 1000,
        "context": event_type,
        "metadata": metadata or {"backend": "cuda"},
    }


def test_append_only_sink_writes_jsonl_segment_and_manifest(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )

    sink.append({"schema_version": 2, "event_type": "start", "seq": 1})
    sink.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert len(manifest["sessions"]) == 1
    session = manifest["sessions"][0]
    assert session["status"] == "completed"
    assert len(manifest["segments"]) == 1
    assert manifest["segments"][0]["event_count"] == 1
    assert manifest["segments"][0]["closed"] is True
    assert manifest["segments"][0]["session_id"] == session["session_id"]

    records = _segment_records(tmp_path / manifest["segments"][0]["filename"])
    assert records == [{"event_type": "start", "schema_version": 2, "seq": 1}]


def test_append_only_sink_close_writes_rollup_sidecar(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )

    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.append(
        _event_record(
            session_id="session-a",
            timestamp_ns=2,
            allocated=250,
            reserved=300,
            used=400,
        )
    )
    sink.close()

    rollups = read_telemetry_rollups(tmp_path)

    assert rollups is not None
    assert (tmp_path / ROLLUP_FILENAME).exists()
    assert rollups.coverage.retained_event_count == 2
    assert rollups.sessions[0].session.session_id == "session-a"
    assert rollups.sessions[0].counters.device_used_bytes.value == 400


def test_append_only_sink_rolls_over_by_event_count(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_events=2,
            retention_max_files=4,
        )
    )

    sink.append({"seq": 1})
    sink.append({"seq": 2})
    sink.append({"seq": 3})
    sink.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    segments = manifest["segments"]
    assert [segment["event_count"] for segment in segments] == [2, 1]
    assert [segment["closed"] for segment in segments] == [True, True]

    first = _segment_records(tmp_path / segments[0]["filename"])
    second = _segment_records(tmp_path / segments[1]["filename"])
    assert [record["seq"] for record in first] == [1, 2]
    assert [record["seq"] for record in second] == [3]


def test_append_only_sink_prunes_oldest_closed_segments(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_events=1,
            rollover_max_bytes=1024,
            retention_max_files=2,
            retention_max_total_bytes=1024 * 1024,
        )
    )

    sink.append({"seq": 1})
    sink.append({"seq": 2})
    sink.append({"seq": 3})
    sink.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    filenames = [segment["filename"] for segment in manifest["segments"]]
    assert filenames == ["segment-000002.jsonl", "segment-000003.jsonl"]
    assert not (tmp_path / "segment-000001.jsonl").exists()


def test_append_only_sink_exposes_rollover_and_prune_diagnostics(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_events=1,
            rollover_max_bytes=1024,
            retention_max_files=2,
            retention_max_total_bytes=1024 * 1024,
        )
    )

    sink.append({"seq": 1})
    sink.append({"seq": 2})
    sink.append({"seq": 3})
    sink.close()

    diagnostics = sink.get_diagnostics()
    assert diagnostics["rollover_count"] == 3
    assert diagnostics["pruned_segment_count"] == 1
    assert diagnostics["pruned_bytes"] > 0
    assert diagnostics["final_retained_files"] == 2
    assert diagnostics["final_retained_bytes"] > 0


def test_append_only_sink_rollup_coverage_reflects_retention_pruning(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_events=1,
            rollover_max_bytes=1024,
            retention_max_files=2,
            retention_max_total_bytes=1024 * 1024,
        )
    )

    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.append(_event_record(session_id="session-a", timestamp_ns=2))
    sink.append(_event_record(session_id="session-a", timestamp_ns=3))
    sink.close()

    rollups = read_telemetry_rollups(tmp_path)

    assert rollups is not None
    assert rollups.coverage.pruned_segment_count == 1
    assert rollups.coverage.pruned_bytes is not None
    assert rollups.coverage.pruned_bytes > 0
    assert rollups.coverage.retained_segment_filenames == [
        "segment-000002.jsonl",
        "segment-000003.jsonl",
    ]
    assert rollups.coverage.retained_event_count == 2


def test_append_only_sink_flushes_without_new_events(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=10,
            flush_every_seconds=0.05,
        )
    )

    try:
        sink.append({"seq": 1})

        segment_path = tmp_path / "segment-000001.jsonl"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if segment_path.exists() and _segment_records(segment_path) == [{"seq": 1}]:
                break
            time.sleep(0.01)
        else:
            pytest.fail("append-only sink did not flush buffered records in time")
    finally:
        sink.close()


def test_append_only_sink_resume_skips_stale_manifest_segment_reuse(
    tmp_path: Path,
) -> None:
    first = tmp_path / "segment-000001.jsonl"
    second = tmp_path / "segment-000002.jsonl"
    third = tmp_path / "segment-000003.jsonl"
    first.write_text(json.dumps({"seq": 1}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"seq": 2}) + "\n", encoding="utf-8")
    third.write_text(json.dumps({"seq": 3}) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.append_only_telemetry_sink",
                "segments": [
                    {
                        "filename": "segment-000001.jsonl",
                        "event_count": 1,
                        "size_bytes": first.stat().st_size,
                        "closed": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append({"seq": 4})
    sink.close()

    assert _segment_records(second) == [{"seq": 2}]

    resumed_records = _segment_records(third)
    next_segment = tmp_path / "segment-000004.jsonl"
    if next_segment.exists():
        resumed_records.extend(_segment_records(next_segment))

    assert [record["seq"] for record in resumed_records] == [3, 4]


def test_append_only_sink_recovery_marks_prior_session_interrupted(
    tmp_path: Path,
) -> None:
    config = TelemetrySinkConfig(
        root_dir=tmp_path,
        flush_every_events=1,
        flush_every_seconds=1.0,
    )

    first_sink = AppendOnlyTelemetrySink(config)
    first_sink.append(
        {
            "schema_version": 3,
            "session_id": "session-a",
            "event_type": "start",
            "timestamp_ns": 1,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1,
            "host": "host",
            "device_id": 0,
            "allocator_allocated_bytes": 1,
            "allocator_reserved_bytes": 1,
            "allocator_active_bytes": None,
            "allocator_inactive_bytes": None,
            "allocator_change_bytes": 0,
            "device_used_bytes": 1,
            "device_free_bytes": None,
            "device_total_bytes": None,
            "context": "first",
            "metadata": {},
        }
    )
    if first_sink._handle is not None:
        first_sink._handle.close()
        first_sink._handle = None
    first_sink._stop_flush_thread()

    recovered_sink = AppendOnlyTelemetrySink(config)
    recovered_sink.append(
        {
            "schema_version": 3,
            "session_id": "session-b",
            "event_type": "start",
            "timestamp_ns": 2,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1,
            "host": "host",
            "device_id": 0,
            "allocator_allocated_bytes": 1,
            "allocator_reserved_bytes": 1,
            "allocator_active_bytes": None,
            "allocator_inactive_bytes": None,
            "allocator_change_bytes": 0,
            "device_used_bytes": 1,
            "device_free_bytes": None,
            "device_total_bytes": None,
            "context": "second",
            "metadata": {},
        }
    )
    recovered_sink.close()

    sessions = load_telemetry_sessions(tmp_path)
    assert [session.summary.session_id for session in sessions] == [
        "session-b",
        "session-a",
    ]
    assert [session.summary.status for session in sessions] == [
        "completed",
        "interrupted",
    ]
    assert [event.context for event in sessions[0].events] == ["second"]
    assert [event.context for event in sessions[1].events] == ["first"]


def test_append_only_sink_recovery_rebuilds_interrupted_rollup(
    tmp_path: Path,
) -> None:
    config = TelemetrySinkConfig(
        root_dir=tmp_path,
        flush_every_events=1,
        flush_every_seconds=1.0,
    )
    first_sink = AppendOnlyTelemetrySink(config)
    first_sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    if first_sink._handle is not None:
        first_sink._handle.close()
        first_sink._handle = None
    first_sink._stop_flush_thread()

    recovered_sink = AppendOnlyTelemetrySink(config)
    assert read_telemetry_rollups(tmp_path) is None

    recovered_sink.close()

    rollups = read_telemetry_rollups(tmp_path)

    assert rollups is not None
    assert rollups.sessions[0].session.session_id == "session-a"
    assert rollups.sessions[0].session.status == "interrupted"


def test_append_only_sink_config_can_disable_rollups(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            write_rollups=False,
        )
    )

    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.close()

    assert not (tmp_path / ROLLUP_FILENAME).exists()


def test_append_only_sink_malformed_rollup_source_does_not_fail_close(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )

    sink.append({"seq": 1})
    sink.close()

    assert _segment_records(tmp_path / "segment-000001.jsonl") == [{"seq": 1}]
    assert not (tmp_path / ROLLUP_FILENAME).exists()
    assert "telemetry rollup write failed" in caplog.text


def test_telemetry_sink_config_rejects_total_retention_below_rollover() -> None:
    with pytest.raises(
        ValueError, match="retention_max_total_bytes must be >= rollover_max_bytes"
    ):
        TelemetrySinkConfig(
            root_dir=Path("/tmp/telemetry"),
            rollover_max_bytes=2048,
            retention_max_total_bytes=1024,
        )


def test_read_telemetry_sink_manifest_tolerates_malformed_root_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "oops",
                "format": 7,
                "sessions": {"bad": "shape"},
                "segments": "segment-000001.jsonl",
            }
        ),
        encoding="utf-8",
    )

    manifest = read_telemetry_sink_manifest(tmp_path)

    assert manifest is not None
    assert manifest.schema_version == 1
    assert manifest.format == "stormlog.append_only_telemetry_sink"
    assert manifest.sessions == []
    assert manifest.segments == []


def test_manifest_parsing_skips_bad_entries_and_coerces_segment_counters(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sessions": [None, {"session_id": "missing-fields"}],
                "segments": [
                    None,
                    {"filename": 12},
                    {
                        "filename": "segment-000001.jsonl",
                        "event_count": "3",
                        "size_bytes": -1,
                        "closed": "yes",
                        "session_id": 12,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = read_telemetry_sink_manifest(tmp_path)
    assert manifest is not None
    assert manifest.sessions == []
    assert len(manifest.segments) == 1
    segment = manifest.segments[0]
    assert (segment.filename, segment.event_count, segment.size_bytes) == (
        "segment-000001.jsonl",
        3,
        0,
    )
    assert segment.closed is True
    assert segment.session_id is None


def test_append_only_sink_close_stops_flush_thread_after_manifest_failure(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append({"seq": 1})
    assert sink._flush_thread is not None
    assert sink._flush_thread.is_alive()

    def _fail_manifest_write() -> None:
        raise OSError("manifest write failed")

    sink._write_manifest_locked = _fail_manifest_write  # type: ignore[method-assign]

    with pytest.raises(OSError, match="manifest write failed"):
        sink.close()

    assert sink._flush_thread is None
