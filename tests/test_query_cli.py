from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from stormlog.query_cli import main as query_main
from stormlog.tui import run_app


def _event_record(
    *,
    session_id: str = "session-cli",
    timestamp_ns: int = 1,
    event_type: str = "sample",
    rank: int = 0,
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
        "world_size": 2,
        "device_id": 0,
        "allocator_allocated_bytes": 100,
        "allocator_reserved_bytes": 150,
        "allocator_active_bytes": None,
        "allocator_inactive_bytes": None,
        "allocator_change_bytes": 50,
        "device_used_bytes": 200,
        "device_free_bytes": None,
        "device_total_bytes": 1000,
        "context": context or event_type,
        "metadata": {"backend": "cuda"},
    }


def _write_json_events(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


@pytest.mark.parametrize("error", [BrokenPipeError(), RuntimeError("output failed")])
def test_query_output_failure_returns_status_without_leaking_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    import stormlog.query_cli as query_cli

    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    def fail_to_emit(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(query_cli, "_emit_rows", fail_to_emit)
    assert query_main(["events", str(path), "--json"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        "" if isinstance(error, BrokenPipeError) else "Error: output failed\n"
    )


def _write_attachment_sidecar(path: Path) -> None:
    path.write_text(
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
                        "run_id": "run-sidecar",
                        "session_id": "session-cli",
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


def test_query_sessions_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    assert query_main(["sessions", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["session_id"] == "session-cli"
    assert payload[0]["source_kind"] == "telemetry_json"


def test_query_runs_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_json_events(
        tmp_path / "rank0_track.json",
        [_event_record(session_id="session-r0", timestamp_ns=1, rank=0)],
    )
    _write_json_events(
        tmp_path / "rank1_track.json",
        [_event_record(session_id="session-r1", timestamp_ns=2, rank=1)],
    )

    assert query_main(["runs", str(tmp_path), "--job-id", "job-a", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["run_id"] == "job:job-a"
    assert payload[0]["session_count"] == 2
    assert payload[0]["ranks"] == [0, 1]


def test_query_attachments_csv_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    assert (
        query_main(["attachments", str(path), "--kind", "telemetry_file", "--csv"]) == 0
    )

    output = capsys.readouterr().out
    assert "run_id,kind,title,storage" in output
    assert "job:job-a" in output
    assert "telemetry_file" in output


def test_query_attachments_table_filters_source_namespace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_attachment_sidecar(tmp_path / "stormlog_attachments.json")

    assert (
        query_main(
            [
                "attachments",
                str(tmp_path),
                "--run-id",
                "run-sidecar",
                "--source-namespace",
                "wandb",
                "--source-ref",
                "example/project/run-a",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Run Id" in output
    assert "W&B run" in output
    assert "wandb" in output


def test_query_events_table_and_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(timestamp_ns=1, rank=0),
            _event_record(timestamp_ns=2, rank=1),
        ],
    )

    assert query_main(["events", str(path), "--rank", "1", "--limit", "1"]) == 0

    output = capsys.readouterr().out
    assert "Session Id" in output
    assert "session-cli" in output
    assert "  1  " in output or output.rstrip().endswith("  1")


def test_query_ooms_csv_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "oom_dump_20260512T000000Z_123_cuda_1"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "bundle_name": bundle.name,
                "created_at_utc": "2026-05-12T00:00:00Z",
                "reason": "message_pattern:out of memory",
                "backend": "cuda",
                "event_count": 1,
                "session_id": "session-cli",
                "session_status": "interrupted",
                "files": ["manifest.json"],
            }
        ),
        encoding="utf-8",
    )

    assert query_main(["ooms", str(tmp_path), "--csv"]) == 0

    output = capsys.readouterr().out
    assert "bundle_path,created_at_utc,backend" in output
    assert "session-cli" in output


def test_query_summary_rejects_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    with pytest.raises(SystemExit) as excinfo:
        query_main(
            [
                "summary",
                str(path),
                "--metric",
                "session_count_by_status",
                "--csv",
            ]
        )

    assert excinfo.value.code == 2
    assert "unrecognized arguments: --csv" in capsys.readouterr().err


def test_query_issues_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                timestamp_ns=1,
                event_type="warning",
                context="High fragmentation: 40.0%",
            )
        ],
    )

    assert query_main(["issues", str(path), "--kind", "alert", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["kind"] == "alert"
    assert payload[0]["hit_count"] == 1
    assert payload[0]["fingerprint"]["dimensions"]["category"] == ("high_fragmentation")


def test_query_issues_table_output_filters_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-visible",
                timestamp_ns=1,
                event_type="warning",
            ),
            _event_record(
                session_id="session-hidden",
                timestamp_ns=2,
                event_type="critical",
            ),
        ],
    )

    assert query_main(["issues", str(path), "--session-id", "session-visible"]) == 0

    output = capsys.readouterr().out
    assert "Fingerprint Id" in output
    assert "session-visible" in output
    assert "session-hidden" not in output


def test_query_issues_rejects_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record(event_type="warning")])

    with pytest.raises(SystemExit) as excinfo:
        query_main(["issues", str(path), "--csv"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments: --csv" in capsys.readouterr().err


def test_query_correlate_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record(timestamp_ns=100)])

    assert query_main(["correlate", str(path), "--at-ns", "100", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["anchor"]["at_ns"] == 100
    assert payload["anchor"]["clock_domain"] == "unix_epoch_ns"
    assert payload["evidence"][0]["kind"] == "telemetry_event"
    assert payload["evidence"][0]["confidence"] == "low"


def test_query_correlate_table_filters_kind_rank_and_scope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(session_id="session-r0", timestamp_ns=100, rank=0),
            _event_record(
                session_id="session-r1",
                timestamp_ns=101,
                rank=1,
                event_type="warning",
            ),
        ],
    )

    assert (
        query_main(
            [
                "correlate",
                str(path),
                "--at-ns",
                "100",
                "--job-id",
                "job-a",
                "--scope",
                "distributed",
                "--rank",
                "1",
                "--kind",
                "alert",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Confidence" in output
    assert "Alert: warning" in output
    assert "Telemetry Event" not in output


def test_query_correlate_requires_anchor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record(timestamp_ns=100)])

    with pytest.raises(SystemExit) as excinfo:
        query_main(["correlate", str(path)])

    assert excinfo.value.code == 2
    assert "one of the arguments --at-ns --record-id is required" in (
        capsys.readouterr().err
    )


def test_stormlog_dispatcher_preserves_no_arg_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_app = types.ModuleType("stormlog.tui.app")

    def _fake_run_app() -> None:
        calls.append("tui")

    fake_app.run_app = _fake_run_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stormlog.tui.app", fake_app)

    run_app([])

    assert calls == ["tui"]


def test_stormlog_dispatcher_routes_query(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    with pytest.raises(SystemExit) as excinfo:
        run_app(["query", "sessions", str(path), "--json"])

    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)[0]["session_id"] == "session-cli"
