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
    first_sink._close_fd_locked()  # a crash leaves the segment open
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
    first_sink._close_fd_locked()  # a crash leaves the segment open
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


class _FailingDisk:
    """Makes ``os.write`` in the sink fail with ENOSPC until healed.

    ``partial`` first writes half the payload, then fails, as a disk that
    fills up in the middle of a write does.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, partial: bool) -> None:
        import errno
        import os

        from stormlog import telemetry_sink

        self.failing = True
        self.calls = 0
        real_write = os.write

        def write(fd: int, data: bytes | memoryview) -> int:
            self.calls += 1
            if not self.failing:
                return real_write(fd, data)
            if partial and len(data) > 1:
                real_write(fd, bytes(data[: len(data) // 2]))
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(telemetry_sink.os, "write", write)


def _bounded_sink(tmp_path: Path, **overrides: object) -> AppendOnlyTelemetrySink:
    values: dict[str, object] = {
        "root_dir": tmp_path,
        "flush_every_events": 1,
        "flush_every_seconds": 60.0,
        "max_buffer_bytes": 4096,
        "failure_backoff_seconds": 0.01,
        "failure_backoff_max_seconds": 0.05,
        "write_rollups": False,
    }
    values.update(overrides)
    return AppendOnlyTelemetrySink(TelemetrySinkConfig(**values))  # type: ignore[arg-type]


def test_a_failed_write_is_cut_back_so_no_partial_line_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _bounded_sink(tmp_path)
    sink.append({"seq": 1})
    disk = _FailingDisk(monkeypatch, partial=True)
    sink.append({"seq": 2, "pad": "x" * 200})
    segment = next(tmp_path.glob("segment-*.jsonl"))
    # Only the first, whole record is on disk; the half-written one was cut.
    assert [json.loads(line) for line in segment.read_text().splitlines()] == [
        {"seq": 1}
    ]
    disk.failing = False
    time.sleep(0.02)  # past the backoff
    sink.append({"seq": 3})
    sink.close()
    assert [r["seq"] for r in _segment_records(segment)] == [1, 2, 3]


def test_a_default_sink_still_raises_but_cuts_the_partial_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path, flush_every_events=1, write_rollups=False
        )
    )
    sink.append({"seq": 1})
    _FailingDisk(monkeypatch, partial=True)
    with pytest.raises(OSError):
        sink.append({"seq": 2, "pad": "x" * 200})
    segment = next(tmp_path.glob("segment-*.jsonl"))
    assert segment.read_text() == '{"seq": 1}\n'
    sink._stop_flush_thread()


def test_a_bounded_sink_holds_at_most_its_buffer_under_sustained_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tracemalloc

    sink = _bounded_sink(tmp_path, max_buffer_bytes=8192)
    disk = _FailingDisk(monkeypatch, partial=False)
    record = {"seq": 0, "pad": "y" * 500}
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for seq in range(5000):  # about 2.6 MB offered to a dead disk
        sink.append({**record, "seq": seq})
    grown = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()

    health = sink.failure_diagnostics()
    assert isinstance(health["buffered_bytes"], int)
    assert health["buffered_bytes"] <= 8192
    assert health["buffered_records"] == 15  # 15 lines of 524 bytes fit
    assert health["dropped_records"] == 5000 - 15
    assert isinstance(health["flush_failures"], int)
    assert health["flush_failures"] >= 1
    assert health["last_flush_error"] is not None
    assert "No space left" in str(health["last_flush_error"])
    # The backoff spaces the retries out: not one write per append.
    assert disk.calls < 100
    assert grown < 256 * 1024
    sink.close()
    assert sink.failure_diagnostics()["dropped_records"] == 5000
    assert sink.failure_diagnostics()["buffered_records"] == 0


def test_a_bounded_sink_recovers_when_the_disk_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _bounded_sink(tmp_path)
    disk = _FailingDisk(monkeypatch, partial=False)
    sink.append({"seq": 1})
    sink.append({"seq": 2})
    assert sink.failure_diagnostics()["buffered_records"] == 2
    disk.failing = False
    time.sleep(0.06)  # past the longest backoff
    sink.append({"seq": 3})
    health = sink.failure_diagnostics()
    assert health["buffered_records"] == 0
    assert health["consecutive_flush_failures"] == 0
    sink.close()
    segment = next(tmp_path.glob("segment-*.jsonl"))
    assert [r["seq"] for r in _segment_records(segment)] == [1, 2, 3]


def test_a_bounded_sink_counts_a_manifest_failure_instead_of_raising(
    tmp_path: Path,
) -> None:
    sink = _bounded_sink(tmp_path)

    def fail() -> None:
        raise OSError("manifest write failed")

    sink._write_manifest_locked = fail  # type: ignore[method-assign]
    # The session's first manifest fails: counted, and the write backs off.
    sink.append({"seq": 1})
    assert sink.failure_diagnostics()["flush_failures"] == 1
    assert sink.failure_diagnostics()["buffered_records"] == 1
    time.sleep(0.02)  # past the backoff
    # The records are written; the manifest after them fails again, counted.
    sink.append({"seq": 2})
    assert sink.failure_diagnostics()["flush_failures"] == 2
    assert sink.failure_diagnostics()["buffered_records"] == 0
    segment = next(tmp_path.glob("segment-*.jsonl"))
    assert _segment_records(segment) == [{"seq": 1}, {"seq": 2}]
    sink._stop_flush_thread()


def _failing_manifest(sink: AppendOnlyTelemetrySink) -> None:
    """Every manifest write fails, as ENOSPC on manifest.tmp would."""

    def fail() -> None:
        raise OSError(28, "No space left on device")

    sink._write_manifest_locked = fail  # type: ignore[method-assign]


def test_a_bounded_sink_closes_without_raising_when_the_manifest_fails(
    tmp_path: Path,
) -> None:
    sink = _bounded_sink(tmp_path)
    sink.append({"seq": 1})
    _failing_manifest(sink)
    sink.close()
    health = sink.failure_diagnostics()
    assert health["flush_failures"] == 1
    assert "No space left" in str(health["last_flush_error"])
    assert sink._flush_thread is None


def test_a_bounded_sink_starts_a_session_without_raising_when_the_manifest_fails(
    tmp_path: Path,
) -> None:
    sink = _bounded_sink(tmp_path)
    _failing_manifest(sink)
    session = sink.start_session()
    assert sink.current_session() == session
    assert sink.failure_diagnostics()["flush_failures"] == 1
    sink._stop_flush_thread()


def test_a_bounded_sink_retries_a_segment_it_could_not_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _bounded_sink(
        tmp_path,
        rollover_max_events=1,
        retention_max_files=1,
        rollover_max_bytes=1 << 20,
        retention_max_total_bytes=1 << 21,
    )
    real_unlink = Path.unlink
    stuck = {"on": True}

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if stuck["on"] and self.name.startswith("segment-"):
            raise PermissionError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    sink.append({"seq": 1})  # segment 1, closed by rollover
    sink.append({"seq": 2})  # segment 2: segment 1 cannot be pruned
    health = sink.failure_diagnostics()
    assert health["flush_failures"] == 1
    assert "Permission denied" in str(health["last_flush_error"])
    assert (tmp_path / "segment-000001.jsonl").exists()
    stuck["on"] = False
    time.sleep(0.06)  # past the backoff
    sink.append({"seq": 3})  # pruned now, on the next flush
    assert not (tmp_path / "segment-000001.jsonl").exists()
    sink.close()


def test_a_failed_cut_back_is_repaired_from_the_file_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial line the cut-back could not remove, and the sink's size that
    no longer matched the file, used to truncate fsynced records later."""
    import errno
    import os

    from stormlog import telemetry_sink

    sink = _bounded_sink(tmp_path, max_buffer_bytes=1 << 20)
    sink.append({"seq": 1})
    real_write, real_ftruncate = os.write, os.ftruncate
    state = {"half": True, "truncate_fails": True}

    def write(fd: int, data: bytes | memoryview) -> int:
        if state["half"]:
            state["half"] = False
            real_write(fd, bytes(data[: len(data) // 2]))
            raise OSError(errno.EIO, "Input/output error")
        return real_write(fd, data)

    def ftruncate(fd: int, length: int) -> None:
        if state["truncate_fails"]:
            state["truncate_fails"] = False
            raise OSError(errno.EIO, "Input/output error")
        real_ftruncate(fd, length)

    monkeypatch.setattr(telemetry_sink.os, "write", write)
    monkeypatch.setattr(telemetry_sink.os, "ftruncate", ftruncate)
    sink.append({"seq": 2, "pad": "x" * 100})  # half written, not cut back
    time.sleep(0.02)
    sink.append({"seq": 3})
    time.sleep(0.02)
    state["half"] = True  # a later failure, whose cut-back works
    sink.append({"seq": 4, "pad": "z" * 100})
    time.sleep(0.06)
    sink.append({"seq": 5})
    sink.close()
    segment = next(tmp_path.glob("segment-*.jsonl"))
    lines = segment.read_bytes().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2, 3, 4, 5]


def test_an_interrupted_write_is_cut_back_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signal handler can raise between two chunks of a short write."""
    import os

    from stormlog import telemetry_sink

    sink = _bounded_sink(tmp_path, max_buffer_bytes=1 << 20)
    sink.append({"seq": 1})
    real_write = os.write
    calls = {"n": 0}

    def interrupted(fd: int, data: bytes | memoryview) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data[: len(data) // 2]))  # a short write
        raise KeyboardInterrupt

    monkeypatch.setattr(telemetry_sink.os, "write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sink.append({"seq": 2, "pad": "x" * 100})
    monkeypatch.undo()
    sink.close()  # in a finally block, say
    segment = next(tmp_path.glob("segment-*.jsonl"))
    lines = segment.read_bytes().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2]


def test_a_bounded_sink_on_a_healthy_disk_flushes_instead_of_dropping(
    tmp_path: Path,
) -> None:
    """A burst that fills the buffer before the next scheduled flush."""
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path, max_buffer_bytes=64 * 1024, write_rollups=False
        )
    )
    for seq in range(200):  # 2 KB each: 400 KB against a 64 KiB bound
        sink.append({"seq": seq, "pad": "p" * 2000})
    assert sink.failure_diagnostics()["dropped_records"] == 0
    sink.close()
    segment = next(tmp_path.glob("segment-*.jsonl"))
    assert [r["seq"] for r in _segment_records(segment)] == list(range(200))


def test_a_bounded_buffer_holds_what_it_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tiny records kept as a list of bytes objects held 5.7 times the bound,
    and the flush joined them into one more copy."""
    import tracemalloc

    bound = 256 * 1024
    sink = _bounded_sink(
        tmp_path,
        max_buffer_bytes=bound,
        flush_every_events=1_000_000,
        failure_backoff_seconds=3600.0,
        failure_backoff_max_seconds=3600.0,
    )
    sink.append({"s": 0})
    disk = _FailingDisk(monkeypatch, partial=False)
    sink.flush(force=True)  # fails: the next retry is an hour away
    tracemalloc.start()
    try:
        held_before = tracemalloc.get_traced_memory()[0]
        while not sink.failure_diagnostics()["dropped_records"]:
            sink.append({"s": 1})
        held = tracemalloc.get_traced_memory()[0] - held_before
        disk.failing = False
        sink._flush_retry_at = 0.0
        tracemalloc.reset_peak()
        before_flush = tracemalloc.get_traced_memory()[0]
        sink.flush(force=True)
        flush_peak = tracemalloc.get_traced_memory()[1] - before_flush
    finally:
        tracemalloc.stop()
    assert held < 1.25 * bound
    assert flush_peak < 64 * 1024
    assert sink.failure_diagnostics()["buffered_records"] == 0
    sink.close()


def test_bounded_mode_settings_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_buffer_bytes"):
        TelemetrySinkConfig(root_dir=tmp_path, max_buffer_bytes=0)
    with pytest.raises(ValueError, match="failure_backoff_seconds"):
        TelemetrySinkConfig(root_dir=tmp_path, failure_backoff_seconds=0)
    with pytest.raises(ValueError, match="failure_backoff_max_seconds"):
        TelemetrySinkConfig(
            root_dir=tmp_path,
            failure_backoff_seconds=2.0,
            failure_backoff_max_seconds=1.0,
        )
