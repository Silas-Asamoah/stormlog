from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import stormlog.query as query_api
import stormlog.run_catalog as run_catalog_api
from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SESSION_STATUS_INTERRUPTED,
    create_session_summary,
    session_summary_to_dict,
)
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig


def test_query_is_canonical_export_surface_for_public_run_contracts() -> None:
    public_types = {
        "CatalogRunEnvelope",
        "RunAttachmentFilter",
        "RunAttachmentRow",
        "RunFilter",
        "RunRow",
    }

    assert public_types <= set(query_api.__all__)
    assert public_types.isdisjoint(run_catalog_api.__all__)
    for name in public_types:
        assert getattr(run_catalog_api, name) is getattr(query_api, name)


def _event_record(
    *,
    session_id: str,
    timestamp_ns: int,
    event_type: str = "sample",
    rank: int = 0,
    world_size: int = 2,
    allocated: int = 100,
    reserved: int = 150,
    used: int = 175,
    metadata: dict[str, Any] | None = None,
    context: str | None = None,
) -> dict[str, Any]:
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
        "world_size": world_size,
        "device_id": 0,
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": reserved,
        "allocator_active_bytes": None,
        "allocator_inactive_bytes": None,
        "allocator_change_bytes": reserved - allocated,
        "device_used_bytes": used,
        "device_free_bytes": None,
        "device_total_bytes": 1000,
        "context": context or event_type,
        "metadata": metadata or {"backend": "cuda"},
    }


def _write_json_events(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_oom_bundle(
    root: Path,
    *,
    session_id: str,
    session_status: str | None = SESSION_STATUS_INTERRUPTED,
    created_at_utc: str = "2026-05-12T00:00:00Z",
) -> Path:
    bundle = root / "oom_dump_20260512T000000Z_123_cuda_1"
    bundle.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "bundle_name": bundle.name,
        "created_at_utc": created_at_utc,
        "reason": "message_pattern:out of memory",
        "backend": "cuda",
        "event_count": 2,
        "session_id": session_id,
        "files": ["manifest.json", "metadata.json"],
    }
    if session_status is not None:
        manifest["session_status"] = session_status
    metadata = {
        "exception_type": "RuntimeError",
        "exception_module": "builtins",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return bundle


def _iso_from_ns(timestamp_ns: int) -> str:
    timestamp_s = timestamp_ns / 1_000_000_000
    return (
        datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_diagnose_bundle(
    root: Path,
    *,
    session_id: str,
    started_at_ns: int,
    ended_at_ns: int,
) -> Path:
    summary = create_session_summary(
        source="stormlog.test.diagnose",
        status=SESSION_STATUS_COMPLETED,
        session_id=session_id,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        host="host-a",
        pid=123,
        job_id="job-a",
        rank=0,
        local_rank=0,
        world_size=2,
    )
    diagnose = root / "diagnose_bundle"
    diagnose.mkdir()
    (diagnose / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_iso": _iso_from_ns(started_at_ns),
                "command_line": "gpumemprof diagnose",
                "files": ["manifest.json"],
                "exit_code": 0,
                "risk_detected": False,
                "session_id": session_id,
                "session_status": SESSION_STATUS_COMPLETED,
                "session": session_summary_to_dict(summary),
            }
        ),
        encoding="utf-8",
    )
    return diagnose


def _write_attachment_sidecar(
    root: Path,
    *,
    session_id: str,
    start_ns: int,
    end_ns: int | None = None,
) -> Path:
    trace_path = root / "profiler.trace"
    trace_path.write_text("trace", encoding="utf-8")
    sidecar = root / "stormlog_attachments.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.attachments",
                "attachments": [
                    {
                        "attachment_id": "profiler-trace-1",
                        "title": "Profiler trace",
                        "kind": "profiler",
                        "path": trace_path.name,
                        "session_id": session_id,
                        "job_id": "job-a",
                        "rank": 0,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "created_at_utc": _iso_from_ns(start_ns),
                        "updated_at_utc": _iso_from_ns(start_ns),
                        "metadata": {"tool": "profiler"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _write_run_envelope(
    root: Path,
    *,
    run_id: str,
    session_id: str,
    job_id: str | None = "job-a",
    rank: int = 0,
    start_ns: int = 100,
    end_ns: int = 200,
) -> Path:
    trace_path = root / "rank0.trace"
    trace_path.write_text("trace", encoding="utf-8")
    envelope = root / "stormlog_run.json"
    envelope.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.run_envelope",
                "run_id": run_id,
                "title": "Training run",
                "description": "Run envelope fixture",
                "job_id": job_id,
                "started_at_ns": start_ns,
                "ended_at_ns": end_ns,
                "source_namespace": "wandb",
                "source_ref": "entity/project/run-a",
                "tags": ["training"],
                "sessions": [
                    {
                        "session_id": session_id,
                        "job_id": job_id,
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": 2,
                        "role": "rank",
                        "source_namespace": "stormlog",
                        "source_ref": f"rank-{rank}",
                        "metadata": {},
                    }
                ],
                "attachments": [
                    {
                        "attachment_id": "rank0-trace",
                        "title": "Rank 0 trace",
                        "kind": "profiler_trace",
                        "storage": "reference",
                        "path": trace_path.name,
                        "session_id": session_id,
                        "job_id": job_id,
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": 2,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "source_namespace": "nsys",
                        "source_ref": "rank0",
                        "metadata": {"tool": "nsys"},
                    }
                ],
                "metadata": {"owner": "training"},
            }
        ),
        encoding="utf-8",
    )
    return envelope


def test_list_sessions_uses_sink_manifest_without_loading_events(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.close()

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("list_sessions should not materialize sink events")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    store = query_api.open([tmp_path])
    rows = store.list_sessions()

    assert len(rows) == 1
    assert rows[0].session_id == "session-a"
    assert rows[0].status == SESSION_STATUS_COMPLETED
    assert rows[0].source_kind == "sink"
    assert rows[0].event_count == 1


def test_correlate_collects_same_session_evidence_across_artifacts(
    tmp_path: Path,
) -> None:
    base_ns = 1_800_000_000_000_000_000
    session_id = "session-correlate"
    sink_dir = tmp_path / "sink"
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=sink_dir,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.start_session(
        create_session_summary(
            source="stormlog.test",
            status=SESSION_STATUS_COMPLETED,
            session_id=session_id,
            started_at_ns=base_ns,
            host="host-a",
            pid=123,
            job_id="job-a",
            rank=0,
            local_rank=0,
            world_size=2,
        )
    )
    sink.append(
        _event_record(
            session_id=session_id,
            timestamp_ns=base_ns,
            event_type="phase_enter",
            metadata={
                "backend": "cuda",
                "phase_scope": {
                    "action": "enter",
                    "name": "forward",
                    "path": ["train", "forward"],
                    "depth": 2,
                    "scope_id": "scope-1",
                    "parent_scope_id": None,
                    "thread_id": 1,
                    "thread_name": "MainThread",
                    "sequence": 1,
                },
            },
        )
    )
    sink.append(
        _event_record(
            session_id=session_id,
            timestamp_ns=base_ns + 10,
            event_type="warning",
            context="High fragmentation: 40.0%",
        )
    )
    sink.append(
        _event_record(
            session_id=session_id,
            timestamp_ns=base_ns + 20,
            event_type="phase_exit",
            metadata={
                "backend": "cuda",
                "phase_scope": {
                    "action": "exit",
                    "name": "forward",
                    "path": ["train", "forward"],
                    "depth": 2,
                    "scope_id": "scope-1",
                    "parent_scope_id": None,
                    "thread_id": 1,
                    "thread_name": "MainThread",
                    "sequence": 2,
                },
            },
        )
    )
    sink.close()
    _write_oom_bundle(
        tmp_path,
        session_id=session_id,
        session_status=SESSION_STATUS_COMPLETED,
        created_at_utc=_iso_from_ns(base_ns + 10),
    )
    _write_diagnose_bundle(
        tmp_path,
        session_id=session_id,
        started_at_ns=base_ns,
        ended_at_ns=base_ns + 30,
    )
    _write_attachment_sidecar(
        tmp_path,
        session_id=session_id,
        start_ns=base_ns,
        end_ns=base_ns + 30,
    )

    result = query_api.open([tmp_path]).correlate(
        query_api.CorrelationFilter(
            session_id=session_id,
            at_ns=base_ns + 10,
            window_ns=1_000,
        )
    )

    kinds = {row.kind for row in result.evidence}
    assert {
        "telemetry_event",
        "timeline_marker",
        "alert",
        "rollup_window",
        "oom_bundle",
        "diagnose_bundle",
        "attachment",
    }.issubset(kinds)
    assert all(row.confidence in {"high", "medium"} for row in result.evidence)
    attachment = next(row for row in result.evidence if row.kind == "attachment")
    assert attachment.source_path.endswith("profiler.trace")
    assert result.anchor["clock_domain"] == "unix_epoch_ns"
    assert result.anchor["clock_normalization"] == "producer_emitted_epoch_ns"


def test_correlate_distributed_scope_uses_job_id_across_ranks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(session_id="session-r0", timestamp_ns=100, rank=0),
            _event_record(session_id="session-r1", timestamp_ns=105, rank=1),
        ],
    )

    result = query_api.open([path]).correlate(
        query_api.CorrelationFilter(
            job_id="job-a",
            scope="distributed",
            at_ns=100,
            window_ns=10,
        )
    )

    event_rows = [row for row in result.evidence if row.kind == "telemetry_event"]
    assert {row.rank for row in event_rows} == {0, 1}
    assert {row.confidence for row in event_rows} <= {"high", "medium"}
    assert any("same_job_distributed" in row.reasons for row in event_rows)


def test_correlate_allows_low_confidence_time_only_matches(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [_event_record(session_id="session-time-only", timestamp_ns=100)],
    )

    result = query_api.open([path]).correlate(
        query_api.CorrelationFilter(at_ns=100, window_ns=0)
    )

    assert result.evidence
    assert {row.confidence for row in result.evidence} == {"low"}
    assert all(
        "time_only_missing_shared_identifier" in row.reasons for row in result.evidence
    )


def test_correlate_sorts_interval_containing_anchor_as_nearest(
    tmp_path: Path,
) -> None:
    session_id = "session-interval"
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [_event_record(session_id=session_id, timestamp_ns=110)],
    )
    _write_attachment_sidecar(
        tmp_path,
        session_id=session_id,
        start_ns=50,
        end_ns=150,
    )

    result = query_api.open([tmp_path]).correlate(
        query_api.CorrelationFilter(
            session_id=session_id,
            at_ns=100,
            window_ns=100,
        )
    )

    assert [row.kind for row in result.evidence[:2]] == [
        "attachment",
        "telemetry_event",
    ]


def test_attachment_sidecar_discovery_resolves_paths_and_reports_warnings(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sidecar = _write_attachment_sidecar(
        tmp_path,
        session_id="session-attachment",
        start_ns=100,
    )
    bad_sidecar_dir = tmp_path / "bad"
    bad_sidecar_dir.mkdir()
    (bad_sidecar_dir / "stormlog_attachments.json").write_text(
        json.dumps({"schema_version": 1, "format": "wrong"}),
        encoding="utf-8",
    )

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("attachment listing should not materialize telemetry")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    store = query_api.open([tmp_path])
    rows = store.list_attachments(
        query_api.AttachmentFilter(session_id="session-attachment")
    )

    assert len(rows) == 1
    assert rows[0].sidecar_path == str(sidecar)
    assert rows[0].attachment_id == "profiler-trace-1"
    assert rows[0].updated_at_utc == rows[0].created_at_utc
    assert rows[0].path is not None
    assert rows[0].path.endswith("profiler.trace")
    assert any(
        "unrecognized attachment sidecar" in item.message
        for item in store.catalog.warnings
    )


def test_list_runs_uses_explicit_run_envelope_and_indexes_attachments(
    tmp_path: Path,
) -> None:
    session_id = "session-explicit-run"
    track_path = tmp_path / "track.json"
    _write_json_events(
        track_path,
        [_event_record(session_id=session_id, timestamp_ns=100)],
    )
    envelope = _write_run_envelope(
        tmp_path,
        run_id="run-explicit",
        session_id=session_id,
        start_ns=100,
        end_ns=200,
    )

    store = query_api.open([tmp_path])
    runs = store.list_runs(query_api.RunFilter(run_id="run-explicit"))
    attachments = store.list_run_attachments(
        query_api.RunAttachmentFilter(
            run_id="run-explicit",
            source_namespace="nsys",
            source_ref="rank0",
        )
    )
    local_artifacts = store.list_run_attachments(
        query_api.RunAttachmentFilter(run_id="run-explicit", kind="telemetry_file")
    )

    assert len(runs) == 1
    assert runs[0].explicit is True
    assert runs[0].source_path == str(envelope)
    assert runs[0].sessions == (session_id,)
    assert runs[0].source_namespace == "wandb"
    assert len(attachments) == 1
    assert attachments[0].path is not None
    assert attachments[0].path.endswith("rank0.trace")
    assert attachments[0].storage == "reference"
    assert len(local_artifacts) == 1
    assert local_artifacts[0].path == str(track_path)


def test_list_runs_includes_implicit_contexts_for_uncovered_mixed_roots(
    tmp_path: Path,
) -> None:
    _write_json_events(
        tmp_path / "covered_track.json",
        [_event_record(session_id="session-covered", timestamp_ns=100)],
    )
    _write_json_events(
        tmp_path / "uncovered_track.json",
        [_event_record(session_id="session-uncovered", timestamp_ns=200)],
    )
    _write_run_envelope(
        tmp_path,
        run_id="run-covered",
        session_id="session-covered",
        start_ns=100,
        end_ns=150,
    )

    rows = query_api.open([tmp_path]).list_runs()

    assert {row.run_id for row in rows} == {"run-covered", "job:job-a"}
    implicit = next(row for row in rows if row.run_id == "job:job-a")
    assert implicit.explicit is False
    assert implicit.sessions == ("session-uncovered",)


def test_duplicate_run_envelope_ids_reject_every_conflicting_envelope(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    _write_run_envelope(first, run_id="run-duplicate", session_id="session-a")
    _write_run_envelope(second, run_id="run-duplicate", session_id="session-b")
    _write_json_events(
        tmp_path / "session_a_track.json",
        [_event_record(session_id="session-a", timestamp_ns=100)],
    )
    _write_json_events(
        tmp_path / "session_b_track.json",
        [_event_record(session_id="session-b", timestamp_ns=200)],
    )

    store = query_api.open([tmp_path])

    assert store.catalog.run_envelopes == []
    assert all(row.explicit is False for row in store.list_runs())
    assert not any(
        row.source_kind == "run_envelope_attachment"
        for row in store.list_run_attachments()
    )
    duplicate_warnings = [
        warning
        for warning in store.catalog.warnings
        if "duplicate run envelope run_id 'run-duplicate'" in warning.message
    ]
    assert {warning.path for warning in duplicate_warnings} == {
        str(first / "stormlog_run.json"),
        str(second / "stormlog_run.json"),
    }


def test_list_runs_groups_implicit_distributed_sessions_by_job_id(
    tmp_path: Path,
) -> None:
    _write_json_events(
        tmp_path / "rank0_track.json",
        [_event_record(session_id="session-r0", timestamp_ns=100, rank=0)],
    )
    _write_json_events(
        tmp_path / "rank1_track.json",
        [_event_record(session_id="session-r1", timestamp_ns=105, rank=1)],
    )

    rows = query_api.open([tmp_path]).list_runs(query_api.RunFilter(job_id="job-a"))

    assert len(rows) == 1
    assert rows[0].run_id == "job:job-a"
    assert rows[0].explicit is False
    assert rows[0].session_count == 2
    assert set(rows[0].sessions) == {"session-r0", "session-r1"}
    assert rows[0].ranks == (0, 1)


def test_list_runs_keeps_reused_sink_sessions_separate_without_job_id(
    tmp_path: Path,
) -> None:
    first_summary = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        session_id="session-first",
        started_at_ns=100,
        host="host-a",
        pid=123,
    )
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.start_session(first_summary)
    sink.append(_event_record(session_id="session-first", timestamp_ns=100))
    sink.close()

    second_summary = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        session_id="session-second",
        started_at_ns=200,
        host="host-a",
        pid=123,
    )
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.start_session(second_summary)
    sink.append(_event_record(session_id="session-second", timestamp_ns=200))
    sink.close()

    rows = query_api.open([tmp_path]).list_runs()

    assert {row.run_id for row in rows} == {
        "session:session-first",
        "session:session-second",
    }
    assert all(row.session_count == 1 for row in rows)


def test_list_run_attachments_filters_sidecars_by_run_and_namespace(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "stormlog_attachments.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.attachments",
                "attachments": [
                    {
                        "attachment_id": "wandb-run",
                        "title": "W&B run",
                        "kind": "experiment",
                        "url": "https://wandb.ai/example/project/runs/run-a",
                        "run_id": "run-explicit",
                        "session_id": "session-sidecar",
                        "job_id": "job-a",
                        "rank": 0,
                        "storage": "reference",
                        "source_namespace": "wandb",
                        "source_ref": "example/project/run-a",
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = query_api.open([tmp_path]).list_run_attachments(
        query_api.RunAttachmentFilter(
            run_id="run-explicit",
            kind="experiment",
            source_namespace="wandb",
            source_ref="example/project/run-a",
        )
    )

    assert len(rows) == 1
    assert rows[0].source_path == "https://wandb.ai/example/project/runs/run-a"
    assert rows[0].storage == "reference"


def test_conflicting_envelope_session_identity_warns_and_omits_ambiguous_sidecar(
    tmp_path: Path,
) -> None:
    session_id = "session-conflict"
    _write_json_events(
        tmp_path / "track.json",
        [_event_record(session_id=session_id, timestamp_ns=100)],
    )
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    _write_run_envelope(first, run_id="run-a", session_id=session_id)
    _write_run_envelope(second, run_id="run-b", session_id=session_id)
    sidecar = tmp_path / "stormlog_attachments.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.attachments",
                "attachments": [
                    {
                        "attachment_id": "ambiguous",
                        "title": "Ambiguous attachment",
                        "kind": "experiment",
                        "url": "https://example.invalid/run",
                        "session_id": session_id,
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = query_api.open([tmp_path])
    rows = store.list_run_attachments(query_api.RunAttachmentFilter(kind="experiment"))

    assert rows == []
    assert any(
        "ambiguous run session_id 'session-conflict'" in warning.message
        for warning in store.catalog.warnings
    )


def test_oom_run_attachment_keeps_session_rank_identity(tmp_path: Path) -> None:
    session = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        session_id="session-rank-oom",
        started_at_ns=100,
        host="host-a",
        pid=123,
        job_id="job-a",
        rank=1,
        local_rank=1,
        world_size=2,
    )
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.start_session(session)
    sink.append(_event_record(session_id="session-rank-oom", timestamp_ns=100, rank=1))
    sink.close()
    _write_oom_bundle(
        tmp_path,
        session_id="session-rank-oom",
        session_status=SESSION_STATUS_COMPLETED,
    )

    rows = query_api.open([tmp_path]).list_run_attachments(
        query_api.RunAttachmentFilter(
            job_id="job-a",
            rank=1,
            kind="oom_bundle",
        )
    )

    assert len(rows) == 1
    assert rows[0].job_id == "job-a"
    assert rows[0].rank == 1
    assert rows[0].local_rank == 1
    assert rows[0].world_size == 2


def test_malformed_run_envelope_warns_without_blocking_discovery(
    tmp_path: Path,
) -> None:
    (tmp_path / "stormlog_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.run_envelope",
                "run_id": "run-bad",
            }
        ),
        encoding="utf-8",
    )
    _write_json_events(
        tmp_path / "track.json",
        [_event_record(session_id="session-still-loads", timestamp_ns=100)],
    )

    store = query_api.open([tmp_path])

    assert store.list_sessions()[0].session_id == "session-still-loads"
    assert any(
        "unrecognized run envelope shape" in warning.message
        for warning in store.catalog.warnings
    )


def test_run_envelope_rejects_null_arrays_and_invalid_entry_metadata(
    tmp_path: Path,
) -> None:
    null_arrays = tmp_path / "null_arrays"
    null_arrays.mkdir()
    (null_arrays / "stormlog_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.run_envelope",
                "run_id": "run-null-arrays",
                "sessions": None,
                "attachments": None,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    bad_entry = tmp_path / "bad_entry"
    bad_entry.mkdir()
    (bad_entry / "stormlog_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.run_envelope",
                "run_id": "run-bad-entry",
                "sessions": [{"session_id": "session-without-metadata"}],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    store = query_api.open([tmp_path])

    assert store.catalog.run_envelopes == []
    messages = [warning.message for warning in store.catalog.warnings]
    assert messages.count("unrecognized run envelope shape") == 2


def test_run_envelope_runtime_parser_enforces_complete_v1_schema(
    tmp_path: Path,
) -> None:
    invalid_payloads = [
        {
            "schema_version": 1,
            "format": "stormlog.run_envelope",
            "run_id": "run-invalid-tag",
            "tags": ["valid", 1],
            "metadata": {},
        },
        {
            "schema_version": 1,
            "format": "stormlog.run_envelope",
            "run_id": "run-duplicate-tag",
            "tags": ["duplicate", "duplicate"],
            "metadata": {},
        },
        {
            "schema_version": 1,
            "format": "stormlog.run_envelope",
            "run_id": "run-float-time",
            "started_at_ns": 1.5,
            "metadata": {},
        },
        {
            "schema_version": 1,
            "format": "stormlog.run_envelope",
            "run_id": "run-negative-rank",
            "sessions": [{"session_id": "session-a", "rank": -1, "metadata": {}}],
            "metadata": {},
        },
        {
            "schema_version": 1,
            "format": "stormlog.run_envelope",
            "run_id": "run-zero-world-size",
            "attachments": [
                {
                    "title": "Trace",
                    "kind": "profiler_trace",
                    "storage": "copy",
                    "path": "trace.json",
                    "world_size": 0,
                    "metadata": {},
                }
            ],
            "metadata": {},
        },
        {
            "schema_version": 1,
            "format": "stormlog.run_envelope",
            "run_id": "run-extra-property",
            "unexpected": True,
            "metadata": {},
        },
    ]
    for index, payload in enumerate(invalid_payloads):
        directory = tmp_path / f"invalid_{index}"
        directory.mkdir()
        (directory / "stormlog_run.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    store = query_api.open([tmp_path])

    assert store.catalog.run_envelopes == []
    messages = [warning.message for warning in store.catalog.warnings]
    assert messages.count("unrecognized run envelope shape") == len(invalid_payloads)


def test_list_sessions_discovers_sink_manifest_file(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.close()

    rows = query_api.open([tmp_path / "manifest.json"]).list_sessions()

    assert len(rows) == 1
    assert rows[0].session_id == "session-a"
    assert rows[0].source_kind == "sink"


def test_query_events_filters_and_adds_provenance(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(session_id="session-a", timestamp_ns=1, rank=0),
            _event_record(
                session_id="session-a",
                timestamp_ns=2,
                event_type="collector_degraded",
                rank=1,
                metadata={
                    "backend": "cuda",
                    "collector_health_status": "degraded",
                },
            ),
        ],
    )

    rows = query_api.open([path]).query_events(
        query_api.EventFilter(
            rank=1,
            event_type="collector_degraded",
            collector_health_status="degraded",
            backend="cuda",
        )
    )

    assert len(rows) == 1
    payload = rows[0].as_dict()
    assert payload["session_id"] == "session-a"
    assert payload["rank"] == 1
    assert payload["source_kind"] == "telemetry_json"
    assert payload["source_path"].endswith("track.json")
    assert payload["session_status"] == SESSION_STATUS_INCOMPLETE


def test_query_events_limit_applies_after_global_sort(tmp_path: Path) -> None:
    late_path = tmp_path / "late_track.json"
    early_path = tmp_path / "early_track.json"
    _write_json_events(
        late_path,
        [_event_record(session_id="session-late", timestamp_ns=200)],
    )
    _write_json_events(
        early_path,
        [_event_record(session_id="session-early", timestamp_ns=100)],
    )

    rows = query_api.open([late_path, early_path]).query_events(
        query_api.EventFilter(limit=1)
    )

    assert len(rows) == 1
    assert rows[0].event.session_id == "session-early"
    assert rows[0].event.timestamp_ns == 100


@pytest.mark.parametrize(
    ("filters", "timestamps"),
    [
        (query_api.EventFilter(time_start_ns=0, time_end_ns=0), [0]),
        (query_api.EventFilter(time_start_ns=1, time_end_ns=2), [1, 2]),
        (query_api.EventFilter(time_start_ns=2, time_end_ns=1), []),
        (query_api.EventFilter(rank=0, has_alert=False), [0, 1, 2]),
        (query_api.EventFilter(rank=1), []),
        (query_api.EventFilter(session_id="missing"), []),
        (query_api.EventFilter(status="completed"), []),
    ],
)
def test_query_event_filter_boundaries(
    tmp_path: Path, filters: query_api.EventFilter, timestamps: list[int]
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(session_id="session-a", timestamp_ns=value)
            for value in range(3)
        ],
    )

    rows = query_api.open([path]).query_events(filters)

    assert [row.event.timestamp_ns for row in rows] == timestamps


def test_event_filters_short_circuit_before_alert_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record(session_id="session-a", timestamp_ns=1)])
    store = query_api.open([path])

    def unexpected_alert_classification(event: Any) -> bool:
        raise AssertionError("earlier filters must reject before classifying alerts")

    monkeypatch.setattr(query_api, "is_alert_event", unexpected_alert_classification)

    assert store.query_events(query_api.EventFilter(rank=1, has_alert=True)) == []
    assert (
        store.query_events(query_api.EventFilter(time_start_ns=2, has_alert=True)) == []
    )


@pytest.mark.parametrize(
    ("session_id", "job_id", "scope", "conflict"),
    [
        ("anchor-session", "other-job", "local", False),
        ("other-session", "anchor-job", "local", True),
        ("other-session", "anchor-job", "distributed", False),
        ("other-session", "other-job", "distributed", True),
        (None, None, "local", False),
        (None, "other-job", "distributed", True),
    ],
)
def test_correlation_identity_conflicts(
    session_id: str | None, job_id: str | None, scope: str, conflict: bool
) -> None:
    evidence = query_api.CorrelationEvidence(
        evidence_id="test",
        kind="telemetry_event",
        title="test",
        session_id=session_id,
        job_id=job_id,
        rank=None,
        world_size=None,
        start_ns=None,
        end_ns=None,
        source_path="test",
        source_kind="telemetry_json",
    )

    assert (
        query_api._has_identity_conflict(
            evidence,
            {"session_id": "anchor-session", "job_id": "anchor-job"},
            scope,
        )
        is conflict
    )


def test_discovery_preserves_root_and_nested_manifest_priority(tmp_path: Path) -> None:
    hybrid = {
        "command_line": "stormlog diagnose",
        "files": [],
        "risk_detected": False,
        "session_id": "session-a",
        "segments": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(hybrid), encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "manifest.json").write_text(json.dumps(hybrid), encoding="utf-8")

    catalog = query_api.ArtifactCatalog([tmp_path])

    assert [(source.path, source.source_kind) for source in catalog.sources] == [
        (tmp_path, "diagnose_bundle"),
        (tmp_path, "sink"),
        (nested, "diagnose_bundle"),
    ]


@pytest.mark.parametrize("metadata", ['{"backend": "cuda"}', "{bad", "[]"])
def test_csv_normalization_preserves_metadata_and_numeric_fallback(
    metadata: str,
) -> None:
    normalized = query_api._normalize_csv_record(
        {"rank": " 1e0 ", "timestamp_ns": "1700000000000000123", "metadata": metadata}
    )

    assert normalized["rank"] == 1
    assert normalized["timestamp_ns"] == 1_700_000_000_000_000_123
    assert normalized["metadata"] == (
        {"backend": "cuda"} if metadata.startswith('{"') else {}
    )


def test_query_all_exports_run_contracts() -> None:
    namespace: dict[str, Any] = {}

    exec("from stormlog.query import *", {}, namespace)

    assert namespace["RunFilter"] is query_api.RunFilter
    assert namespace["RunAttachmentFilter"] is query_api.RunAttachmentFilter
    assert namespace["RunRow"] is query_api.RunRow
    assert namespace["RunAttachmentRow"] is query_api.RunAttachmentRow


def test_list_oom_bundles_links_to_sessions(tmp_path: Path) -> None:
    session = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_INTERRUPTED,
        session_id="session-oom",
        started_at_ns=10,
        host="host-a",
        pid=123,
    )
    diagnose = tmp_path / "diag"
    diagnose.mkdir()
    (diagnose / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_iso": "2026-05-12T00:00:00Z",
                "command_line": "gpumemprof diagnose",
                "files": ["manifest.json"],
                "exit_code": 2,
                "risk_detected": True,
                "session_id": "session-oom",
                "session_status": SESSION_STATUS_INTERRUPTED,
                "session": session_summary_to_dict(session),
            }
        ),
        encoding="utf-8",
    )
    _write_oom_bundle(tmp_path, session_id="session-oom")

    store = query_api.open([tmp_path])
    session_rows = store.list_sessions(query_api.SessionFilter(has_oom_bundle=True))
    oom_rows = store.list_oom_bundles(query_api.OOMBundleFilter(backend="cuda"))

    assert [row.session_id for row in session_rows] == ["session-oom"]
    assert session_rows[0].oom_bundle_count == 1
    assert len(oom_rows) == 1
    assert oom_rows[0].session_id == "session-oom"
    assert oom_rows[0].session_status == SESSION_STATUS_INTERRUPTED
    assert oom_rows[0].exception_type == "RuntimeError"


def test_list_oom_bundles_uses_only_manifest_backed_session_status(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    session = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        session_id="session-manifest-only",
        started_at_ns=10,
        host="host-a",
        pid=123,
    )
    diagnose = tmp_path / "diag"
    diagnose.mkdir()
    (diagnose / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_iso": "2026-05-12T00:00:00Z",
                "command_line": "gpumemprof diagnose",
                "files": ["manifest.json"],
                "exit_code": 0,
                "risk_detected": False,
                "session_id": "session-manifest-only",
                "session_status": SESSION_STATUS_COMPLETED,
                "session": session_summary_to_dict(session),
            }
        ),
        encoding="utf-8",
    )
    _write_oom_bundle(
        tmp_path,
        session_id="session-manifest-only",
        session_status=None,
    )
    _write_json_events(
        tmp_path / "track.json",
        [_event_record(session_id="flat-session", timestamp_ns=1)],
    )

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("OOM listing should not materialize flat telemetry")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    rows = query_api.open([tmp_path]).list_oom_bundles()

    assert len(rows) == 1
    assert rows[0].session_id == "session-manifest-only"
    assert rows[0].session_status == SESSION_STATUS_COMPLETED


def test_query_summaries_cover_sessions_peaks_alerts_and_gap_growth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-a",
                timestamp_ns=1,
                rank=0,
                allocated=100,
                reserved=150,
                used=200,
            ),
            _event_record(
                session_id="session-a",
                timestamp_ns=2,
                rank=0,
                allocated=180,
                reserved=220,
                used=330,
            ),
            _event_record(
                session_id="session-a",
                timestamp_ns=3,
                event_type="warning",
                rank=0,
            ),
            _event_record(
                session_id="session-a",
                timestamp_ns=4,
                event_type="collector_degraded",
                rank=1,
            ),
        ],
    )
    store = query_api.open([path])

    status_rows = store.summarize("session_count_by_status")
    peak_rows = store.summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )
    alert_rows = store.summarize("alert_count", group_by="session-rank")
    collector_rows = store.summarize(
        "collector_degradation_transitions",
        group_by="rank",
    )
    gap_rows = store.summarize("hidden_memory_gap_growth", group_by="session")

    assert status_rows[0].status == "incomplete"
    assert status_rows[0].value == 1
    assert peak_rows[0].value == 220
    assert alert_rows[0].value == 1
    assert collector_rows[0].rank == 1
    assert collector_rows[0].value == 1
    assert gap_rows[0].value == 60
    assert gap_rows[0].details["peak_gap_bytes"] == 110


def test_query_summary_uses_fresh_sink_rollup_without_loading_events(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-rollup",
            timestamp_ns=1,
            allocated=100,
            reserved=150,
        )
    )
    sink.append(
        _event_record(
            session_id="session-rollup",
            timestamp_ns=2,
            allocated=200,
            reserved=275,
        )
    )
    sink.close()

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("fresh rollup summary should not materialize events")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    rows = query_api.open([tmp_path]).summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-rollup"
    assert rows[0].value == 275
    assert rows[0].details["timestamp_ns"] == 2


def test_query_summary_falls_back_when_sink_rollup_is_stale(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    original_load = query_api.load_telemetry_sessions
    load_calls = 0

    def _counted_load(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _counted_load)

    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-stale",
            timestamp_ns=1,
            reserved=150,
        )
    )
    sink.close()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"][0]["event_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = query_api.open([tmp_path]).summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-stale"
    assert rows[0].value == 150
    assert load_calls > 0


def test_query_summary_falls_back_when_sink_rollup_is_malformed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    original_load = query_api.load_telemetry_sessions
    load_calls = 0

    def _counted_load(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _counted_load)

    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-malformed",
            timestamp_ns=1,
            reserved=160,
        )
    )
    sink.close()
    (tmp_path / "rollups.json").write_text("{bad", encoding="utf-8")

    rows = query_api.open([tmp_path]).summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-malformed"
    assert rows[0].value == 160
    assert load_calls > 0


def test_query_hidden_gap_rollup_order_matches_raw_fallback(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    for rank, used_values in ((10, (200, 260)), (2, (180, 230))):
        for offset, used in enumerate(used_values):
            sink.append(
                _event_record(
                    session_id="session-order",
                    timestamp_ns=rank * 10 + offset,
                    rank=rank,
                    world_size=16,
                    reserved=100,
                    used=used,
                )
            )
    sink.close()

    store = query_api.open([tmp_path])
    rollup_rows = store.summarize("hidden_memory_gap_growth", group_by="rank")
    (tmp_path / "rollups.json").write_text("{bad", encoding="utf-8")
    raw_rows = query_api.open([tmp_path]).summarize(
        "hidden_memory_gap_growth",
        group_by="rank",
    )

    assert [(row.session_id, row.rank, row.status) for row in rollup_rows] == [
        (row.session_id, row.rank, row.status) for row in raw_rows
    ]
    assert [row.value for row in rollup_rows] == [row.value for row in raw_rows]


def test_query_alert_count_rollup_matches_raw_fallback_for_metadata_severity(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-alert",
            timestamp_ns=1,
            event_type="sample",
            metadata={"backend": "cuda", "severity": " Warning "},
        )
    )
    sink.close()

    rollup_rows = query_api.open([tmp_path]).summarize(
        "alert_count",
        group_by="session",
    )
    (tmp_path / "rollups.json").write_text("{bad", encoding="utf-8")
    raw_rows = query_api.open([tmp_path]).summarize(
        "alert_count",
        group_by="session",
    )

    assert [(row.session_id, row.value) for row in rollup_rows] == [
        (row.session_id, row.value) for row in raw_rows
    ]
    assert rollup_rows[0].value == 1


def test_catalog_discovers_csv_telemetry(tmp_path: Path) -> None:
    path = tmp_path / "track.csv"
    record = _event_record(session_id="session-csv", timestamp_ns=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value) if key == "metadata" else value
                for key, value in record.items()
            }
        )

    rows = query_api.open([tmp_path]).list_sessions(
        query_api.SessionFilter(source_kind="telemetry_csv")
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-csv"
    assert rows[0].source_kind == "telemetry_csv"


def test_csv_telemetry_preserves_large_integer_fields(tmp_path: Path) -> None:
    path = tmp_path / "track.csv"
    record = _event_record(
        session_id="session-csv",
        timestamp_ns=1_700_000_000_000_000_123,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value) if key == "metadata" else value
                for key, value in record.items()
            }
        )

    rows = query_api.open([path]).query_events()

    assert len(rows) == 1
    assert rows[0].event.timestamp_ns == record["timestamp_ns"]
    assert rows[0].source_kind == "telemetry_csv"


def test_list_issues_groups_alerts_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-alert-a",
                timestamp_ns=10,
                event_type="warning",
                context="High fragmentation: 40.0%",
            ),
            _event_record(
                session_id="session-alert-b",
                timestamp_ns=20,
                event_type="warning",
                context="High fragmentation: 51.5%",
            ),
        ],
    )

    rows = query_api.open([path]).list_issues(
        query_api.IssueFilter(kind="alert", session_id="session-alert-a")
    )

    assert len(rows) == 1
    issue = rows[0]
    assert issue.kind == "alert"
    assert issue.state == "open"
    assert issue.hit_count == 2
    assert issue.first_seen_ns == 10
    assert issue.last_seen_ns == 20
    assert issue.affected_sessions == ("session-alert-a", "session-alert-b")
    assert issue.fingerprint.dimensions["category"] == "high_fragmentation"
    assert issue.representative_evidence.event_type == "warning"


def test_list_issues_supports_state_overrides(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-collector",
                timestamp_ns=10,
                event_type="collector_degraded",
                metadata={
                    "backend": "cuda",
                    "collector_health_status": "degraded",
                    "collector_partial_fields": ["device_free_bytes"],
                    "collector_last_error": "RuntimeError: failed at sample 42",
                },
            )
        ],
    )
    store = query_api.open([path])
    original = store.list_issues(query_api.IssueFilter(kind="collector_degradation"))[0]

    overridden = store.list_issues(
        query_api.IssueFilter(state="ignored"),
        state_overrides={original.fingerprint_id: "ignored"},
    )

    assert len(overridden) == 1
    assert overridden[0].state == "ignored"
    assert overridden[0].details["error_stem"] == "runtimeerror"


def test_list_issues_includes_oom_bundles_and_telemetry_ooms(tmp_path: Path) -> None:
    _write_oom_bundle(tmp_path, session_id="session-oom-bundle")
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-oom-event",
                timestamp_ns=50,
                event_type="error",
                metadata={
                    "backend": "cuda",
                    "oom_reason": "message_pattern:out of memory",
                    "oom_dump_path": str(tmp_path / "oom"),
                },
            )
        ],
    )

    rows = query_api.open([tmp_path]).list_issues(query_api.IssueFilter(kind="oom"))

    assert len(rows) == 1
    assert rows[0].severity == "critical"
    assert rows[0].hit_count == 2
    assert rows[0].fingerprint.dimensions == {
        "backend": "cuda",
        "reason": "message_pattern:out of memory",
    }
    assert {"session-oom-bundle", "session-oom-event"} == {
        session for row in rows for session in row.affected_sessions
    }


def test_list_issues_includes_hidden_memory_anomalies(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-gap",
                timestamp_ns=timestamp_ns,
                allocated=90,
                reserved=100,
                used=used,
            )
            for timestamp_ns, used in enumerate([160, 220, 280, 340, 400], start=1)
        ],
    )

    rows = query_api.open([path]).list_issues(
        query_api.IssueFilter(kind="hidden_memory_anomaly")
    )

    assert rows
    assert rows[0].kind == "hidden_memory_anomaly"
    assert rows[0].details["classification"] == "persistent_drift"
    assert rows[0].affected_sessions == ("session-gap",)
