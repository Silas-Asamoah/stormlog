from __future__ import annotations

import builtins
import json
import math
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import stormlog._wandb.tracking as tracking_export
import stormlog.cuda_native_debug as native_debug
from stormlog._wandb.attribution import build_attribution_preview_html
from stormlog._wandb.core import read_json_if_exists
from stormlog._wandb.dashboard import tracking_dashboard_html
from stormlog._wandb.tracking import _TIMELINE_MAX_POINTS, sample_timeline_rows
from stormlog.session import create_session_summary
from stormlog.wandb_integration import (
    ensure_wandb_available,
    export_diagnose_bundle_to_wandb,
    export_tracking_run_to_wandb,
    wandb_config_from_namespace,
)


class _FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[tuple[str, str | None]] = []
        self.directories: list[tuple[str, str | None]] = []

    def add_file(self, local_path: str, name: str | None = None) -> None:
        self.files.append((local_path, name))

    def add_dir(self, local_path: str, name: str | None = None) -> None:
        self.directories.append((local_path, name))


class _FakeTable:
    def __init__(self, *, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class _FakeHtml:
    def __init__(self, html: str) -> None:
        self.html = html


class _FakePlotModule:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def line_series(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"kind": "line_series", "kwargs": kwargs}


class _FakeRun:
    def __init__(self, owner: "_FakeWandbModule") -> None:
        self.owner = owner
        self.summary: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.logged_steps: list[int | None] = []
        self.artifacts: list[_FakeArtifact] = []
        self.finished = False

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.logged.append(payload)
        self.logged_steps.append(step)

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True
        self.owner.run = None


class _FakeWandbModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run: _FakeRun | None = None
        self.init_calls: list[dict[str, Any]] = []
        self.created_runs: list[_FakeRun] = []
        self.Artifact = _FakeArtifact
        self.Table = _FakeTable
        self.Html = _FakeHtml
        self.plot = _FakePlotModule()

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        run = _FakeRun(self)
        self.created_runs.append(run)
        self.run = run
        return run


@pytest.mark.parametrize("managed", [False, True])
def test_tracking_artifact_failure_respects_run_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, managed: bool
) -> None:
    fake = _FakeWandbModule()
    run = fake.init()
    monkeypatch.setattr(
        tracking_export, "resolve_run", lambda *a, **kw: (fake, run, managed)
    )
    output = tmp_path / "events.json"
    output.write_text("[]", encoding="utf-8")

    def fail_upload(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(tracking_export, "log_file_artifact", fail_upload)
    with pytest.raises(RuntimeError, match="upload failed"):
        export_tracking_run_to_wandb(
            wandb_config_from_namespace(
                Namespace(wandb=True, wandb_log_artifacts=True)
            ),
            command_name="track",
            session_summary=None,
            stats={},
            events=[],
            output_path=output,
        )
    assert run.finished is managed


def test_wandb_config_from_namespace_collects_explicit_values() -> None:
    config = wandb_config_from_namespace(
        Namespace(
            wandb=True,
            wandb_project="stormlog-tests",
            wandb_entity="team",
            wandb_mode="offline",
            wandb_run_id="run-123",
            wandb_name="track smoke",
            wandb_group="job-42",
            wandb_job_type="stormlog-track",
            wandb_log_artifacts=True,
            wandb_log_attribution=True,
        )
    )

    assert config.enabled is True
    assert config.project == "stormlog-tests"
    assert config.entity == "team"
    assert config.mode == "offline"
    assert config.run_id == "run-123"
    assert config.run_name == "track smoke"
    assert config.group == "job-42"
    assert config.job_type == "stormlog-track"
    assert config.log_tables is True
    assert config.log_artifacts is True
    assert config.log_attribution is True


def test_read_json_if_exists_ignores_non_object_payloads(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('["not", "an", "object"]', encoding="utf-8")

    assert read_json_if_exists(payload_path) is None


def test_ensure_wandb_available_reports_optional_dependency_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "wandb" or name.startswith("wandb."):
            raise ModuleNotFoundError("No module named 'wandb'", name="wandb")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(ImportError, match="stormlog\\[wandb\\]"):
        ensure_wandb_available(wandb_config_from_namespace(Namespace(wandb=True)))


def test_export_tracking_run_logs_metrics_tables_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    output_path = tmp_path / "track.json"
    output_path.write_text("{}", encoding="utf-8")

    sink_dir = tmp_path / "sink"
    sink_dir.mkdir()
    (sink_dir / "manifest.json").write_text("{}", encoding="utf-8")

    oom_dir = tmp_path / "oom"
    oom_dir.mkdir()
    (oom_dir / "cuda_allocator_state_history_annotated.html").write_text(
        "<html><body>stormlog attribution</body></html>",
        encoding="utf-8",
    )
    with (oom_dir / native_debug.SNAPSHOT_PICKLE_FILENAME).open("wb") as handle:
        pickle.dump(
            {
                "segments": [
                    {
                        "address": 4096,
                        "segment_type": "large",
                        "total_size": 128,
                        "allocated_size": 64,
                        "active_size": 64,
                        "blocks": [
                            {
                                "address": 8192,
                                "size": 64,
                                "state": "active_allocated",
                                "frames": [],
                            }
                        ],
                    }
                ],
                "device_traces": [
                    [
                        {
                            "action": "alloc",
                            "addr": 8192,
                            "size": 64,
                            "time_us": 100,
                            "frames": [],
                        }
                    ]
                ],
            },
            handle,
        )
    (oom_dir / "cuda_tensor_attribution.json").write_text(
        json.dumps(
            {
                "attributed_storage_pointers": [
                    {
                        "storage_ptr": "0x1",
                        "storage_ptr_int": 8192,
                        "names": ["model.layer.weight"],
                        "tensor_count": 1,
                        "tensors": [
                            {
                                "shape": [4, 4],
                                "dtype": "torch.float32",
                                "size_bytes": 64,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    session_summary = create_session_summary(
        source="stormlog.tracker",
        session_id="session-12345678",
        job_id="train-42",
        rank=2,
        local_rank=0,
        world_size=8,
    )
    config = wandb_config_from_namespace(
        Namespace(
            wandb=True,
            wandb_project="stormlog-tests",
            wandb_group=None,
            wandb_job_type=None,
            wandb_log_artifacts=True,
            wandb_log_attribution=True,
        )
    )

    export_tracking_run_to_wandb(
        config,
        command_name="stormlog-track",
        session_summary=session_summary,
        stats={
            "backend": "cuda",
            "peak_memory": 4096,
            "total_events": 3,
            "alert_count": 1,
            "tracking_duration_seconds": 5.0,
            "collector_health_status": "healthy",
            "history_dropped_events": 0,
        },
        events=[
            {
                "timestamp": 9.5,
                "event_type": "allocation",
                "context": "warmup",
                "memory_allocated": 1024,
                "memory_reserved": 2048,
                "memory_change": 1024,
                "device_used": 2048,
                "device_total": 8192,
                "job_id": "train-42",
                "rank": 2,
            },
            {
                "timestamp": 10.0,
                "event_type": "warning",
                "context": "memory high",
                "memory_allocated": 2048,
                "memory_reserved": 4096,
                "memory_change": 512,
                "device_used": 4096,
                "device_total": 8192,
                "job_id": "train-42",
                "rank": 2,
            },
            {
                "timestamp": 10.5,
                "event_type": "peak",
                "context": "peak memory",
                "memory_allocated": 3072,
                "memory_reserved": 4096,
                "memory_change": 1024,
                "device_used": 5120,
                "device_total": 8192,
                "job_id": "train-42",
                "rank": 2,
            },
        ],
        output_path=output_path,
        telemetry_sink_dir=sink_dir,
        oom_dump_path=oom_dir,
    )

    assert fake_wandb.init_calls
    assert fake_wandb.init_calls[0]["project"] == "stormlog-tests"
    assert fake_wandb.init_calls[0]["group"] == "train-42"
    assert fake_wandb.init_calls[0]["job_type"] == "stormlog-track"

    run = fake_wandb.created_runs[0]
    assert run.finished is True
    assert run.summary["stormlog_session_id"] == "session-12345678"
    assert run.summary["stormlog_backend"] == "cuda"
    assert run.summary["stormlog_peak_memory_bytes"] == 4096
    assert run.summary["stormlog_total_events"] == 3
    assert run.summary["stormlog_chart_point_count"] == 3
    assert (
        run.summary["stormlog_tracking_dashboard_file"]
        == "stormlog_tracking_dashboard.html"
    )
    assert (
        run.summary["stormlog_attribution_html_file"]
        == "cuda_allocator_state_history_annotated.html"
    )
    assert run.logged_steps.count(None) >= 3
    assert any("stormlog_alerts" in payload for payload in run.logged)
    assert any("stormlog_memory_timeline_table" in payload for payload in run.logged)
    assert any("stormlog_memory_timeline_plot" in payload for payload in run.logged)
    assert any("stormlog_tracking_dashboard" in payload for payload in run.logged)
    assert any("stormlog_attribution_html" in payload for payload in run.logged)
    assert any("stormlog_tensor_attribution" in payload for payload in run.logged)
    attribution_html = next(
        payload["stormlog_attribution_html"].html
        for payload in run.logged
        if "stormlog_attribution_html" in payload
    )
    assert "Stormlog GPU Attribution Preview" in attribution_html
    assert "<script>" not in attribution_html
    assert {artifact.type for artifact in run.artifacts} == {
        "stormlog-attribution",
        "stormlog-oom-dump",
        "stormlog-telemetry-sink",
        "stormlog-track-output",
        "stormlog-tracking-dashboard",
    }


def test_export_diagnose_bundle_logs_summary_and_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    artifact_dir = tmp_path / "stormlog-diagnose"
    artifact_dir.mkdir()
    session_summary = create_session_summary(
        source="gpumemprof diagnose",
        session_id="diag-12345678",
        job_id="job-7",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "session": {
                    "session_id": session_summary.session_id,
                    "status": session_summary.status,
                    "started_at_ns": session_summary.started_at_ns,
                    "ended_at_ns": session_summary.ended_at_ns,
                    "host": session_summary.host,
                    "pid": session_summary.pid,
                    "job_id": session_summary.job_id,
                    "rank": session_summary.rank,
                    "local_rank": session_summary.local_rank,
                    "world_size": session_summary.world_size,
                    "source": session_summary.source,
                },
                "risk_detected": True,
                "exit_code": 2,
                "native_history_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "diagnostic_summary.json").write_text(
        json.dumps(
            {
                "allocated_bytes": 1024,
                "reserved_bytes": 2048,
                "peak_bytes": 4096,
                "total_bytes": 8192,
                "utilization_ratio": 0.8,
                "fragmentation_ratio": 0.2,
                "num_ooms": 1,
                "risk_flags": {
                    "oom_occurred": True,
                    "high_utilization": False,
                    "fragmentation_warning": False,
                },
                "suggestions": ["Reduce batch size"],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "cuda_allocator_state_history_annotated.html").write_text(
        "<html><body>attribution</body></html>",
        encoding="utf-8",
    )
    (artifact_dir / "cuda_tensor_attribution.json").write_text(
        json.dumps({"attributed_storage_pointers": []}),
        encoding="utf-8",
    )

    config = wandb_config_from_namespace(
        Namespace(
            wandb=True,
            wandb_project="stormlog-diag",
            wandb_log_artifacts=True,
            wandb_log_attribution=True,
        )
    )

    export_diagnose_bundle_to_wandb(
        config,
        command_name="stormlog-diagnose",
        artifact_dir=artifact_dir,
    )

    run = fake_wandb.created_runs[0]
    assert run.finished is True
    assert run.summary["stormlog_artifact_dir"] == "stormlog-diagnose"
    assert run.summary["stormlog_session_id"] == "diag-12345678"
    assert run.summary["stormlog_allocated_bytes"] == 1024
    assert (
        run.summary["stormlog_attribution_html_file"]
        == "cuda_allocator_state_history_annotated.html"
    )
    assert any("stormlog_diagnostic_suggestions" in payload for payload in run.logged)
    assert any("stormlog_attribution_html" in payload for payload in run.logged)
    attribution_html = next(
        payload["stormlog_attribution_html"].html
        for payload in run.logged
        if "stormlog_attribution_html" in payload
    )
    assert "Stormlog GPU Attribution Preview" in attribution_html
    assert "<script>" not in attribution_html
    assert {artifact.type for artifact in run.artifacts} == {
        "stormlog-attribution",
        "stormlog-diagnose",
    }


def test_tracking_visual_artifacts_respect_log_artifacts_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    export_tracking_run_to_wandb(
        wandb_config_from_namespace(Namespace(wandb=True)),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[
            {
                "timestamp": 1.0,
                "event_type": "sample",
                "memory_allocated": 128,
                "memory_reserved": 256,
                "device_used": 256,
                "device_total": 1024,
            }
        ],
    )

    run = fake_wandb.created_runs[0]
    assert run.artifacts == []
    assert "stormlog_tracking_dashboard_file" not in run.summary
    assert any("stormlog_tracking_dashboard" in payload for payload in run.logged)


def test_attribution_inline_logging_respects_log_artifacts_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    attribution_dir = tmp_path / "attribution"
    attribution_dir.mkdir()
    (attribution_dir / native_debug.TRACE_HTML_ANNOTATED_FILENAME).write_text(
        "<html><body>stormlog attribution</body></html>",
        encoding="utf-8",
    )
    (attribution_dir / native_debug.TENSOR_ATTRIBUTION_FILENAME).write_text(
        json.dumps(
            {
                "storage_pointer_count": 1,
                "attributed_storage_pointers": [
                    {
                        "storage_ptr": "0x1",
                        "storage_ptr_int": 8192,
                        "names": ["model.layer.weight"],
                        "tensor_count": 1,
                        "tensors": [
                            {
                                "shape": [4, 4],
                                "dtype": "torch.float32",
                                "size_bytes": 64,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    export_tracking_run_to_wandb(
        wandb_config_from_namespace(
            Namespace(
                wandb=True,
                wandb_log_artifacts=False,
                wandb_log_attribution=True,
            )
        ),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[],
        attribution_bundle_dir=attribution_dir,
    )

    run = fake_wandb.created_runs[0]
    assert run.artifacts == []
    assert any("stormlog_attribution_html" in payload for payload in run.logged)
    assert any("stormlog_tensor_attribution" in payload for payload in run.logged)


def test_attribution_preview_falls_back_when_snapshot_pickle_is_corrupt(
    tmp_path: Path,
) -> None:
    attribution_dir = tmp_path / "attribution"
    attribution_dir.mkdir()
    (attribution_dir / native_debug.SNAPSHOT_PICKLE_FILENAME).write_bytes(
        b"not a pickle"
    )
    (attribution_dir / native_debug.TENSOR_ATTRIBUTION_FILENAME).write_text(
        json.dumps(
            {
                "storage_pointer_count": 1,
                "attributed_storage_pointers": [
                    {
                        "storage_ptr": "0x1",
                        "names": ["model.layer.weight"],
                        "tensor_count": 1,
                        "tensors": [
                            {
                                "shape": [4, 4],
                                "dtype": "torch.float32",
                                "size_bytes": 64,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = build_attribution_preview_html(attribution_dir)

    assert "Snapshot data was unavailable" in html
    assert "model.layer.weight" in html


def test_tracking_timeline_sampling_pins_alert_rows_and_last_row() -> None:
    rows = [
        {
            "sample_index": index,
            "event_type": "sample",
            "allocated_bytes": index,
        }
        for index in range(_TIMELINE_MAX_POINTS * 3)
    ]
    rows[251]["event_type"] = "warning"
    rows[399]["event_type"] = "peak"
    rows[501]["event_type"] = "error"

    sampled = sample_timeline_rows(rows)

    sampled_indices = [row["sample_index"] for row in sampled]
    assert len(sampled) <= _TIMELINE_MAX_POINTS
    assert sampled_indices == sorted(sampled_indices)
    assert 251 in sampled_indices
    assert 399 in sampled_indices
    assert 501 in sampled_indices
    assert rows[-1]["sample_index"] in sampled_indices


def test_tracking_timeline_sampling_keeps_latest_pinned_rows_when_over_budget() -> None:
    rows = [
        {
            "sample_index": index,
            "event_type": "warning",
            "allocated_bytes": index,
        }
        for index in range(_TIMELINE_MAX_POINTS + 20)
    ]

    sampled = sample_timeline_rows(rows)

    assert [row["sample_index"] for row in sampled] == list(
        range(20, _TIMELINE_MAX_POINTS + 20)
    )


def test_tracking_dashboard_escapes_alert_cells_and_handles_invalid_elapsed() -> None:
    html = tracking_dashboard_html(
        [
            {
                "sample_index": "<script>alert(1)</script>",
                "event_type": "warning",
                "elapsed_seconds": "oops",
                "context": "<b>unsafe</b>",
                "allocated_bytes": 128,
                "reserved_bytes": 256,
            }
        ],
        alert_event_types=frozenset({"warning"}),
    )

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;b&gt;unsafe&lt;/b&gt;" in html
    assert ">n/a<" in html


def test_tracking_plots_preserve_missing_metric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    export_tracking_run_to_wandb(
        wandb_config_from_namespace(Namespace(wandb=True)),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[
            {
                "timestamp": 1.0,
                "event_type": "sample",
                "memory_reserved": 256,
                "device_used": 256,
            },
            {
                "timestamp": 2.0,
                "event_type": "sample",
                "memory_reserved": 512,
                "device_used": 512,
            },
        ],
    )

    memory_plot_call = next(
        call
        for call in fake_wandb.plot.calls
        if call.get("title") == "Stormlog Memory Timeline"
    )
    assert memory_plot_call["keys"] == ["reserved_bytes", "device_used_bytes"]
    assert all(not math.isnan(value) for value in memory_plot_call["ys"][0])


def test_export_uses_active_wandb_run_without_creating_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandbModule()
    active_run = _FakeRun(fake_wandb)
    fake_wandb.run = active_run
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    export_tracking_run_to_wandb(
        wandb_config_from_namespace(Namespace(wandb=True)),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[],
    )

    assert fake_wandb.init_calls == []
    assert active_run.finished is False
    assert active_run.summary["stormlog_peak_memory_bytes"] == 128
