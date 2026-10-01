from __future__ import annotations

import builtins
import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import stormlog._mlflow.tracking as tracking_export
import stormlog.cuda_native_debug as native_debug
from stormlog.mlflow_integration import (
    ensure_mlflow_available,
    export_diagnose_bundle_to_mlflow,
    export_tracking_run_to_mlflow,
    mlflow_config_from_namespace,
)
from stormlog.session import create_session_summary


class _FakeRun:
    """Opaque run handle; the fluent API operates on the active run."""


class _FakeMlflowModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.active: _FakeRun | None = None
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.start_kwargs: dict[str, Any] | None = None
        self.tags: dict[str, str] = {}
        self.metrics: dict[str, list[tuple[float, int | None]]] = {}
        self.texts: list[tuple[str, str]] = []
        self.tables: list[tuple[dict[str, list[Any]], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.directories: list[tuple[str, str | None]] = []
        self.figures: list[tuple[Any, str]] = []
        self.end_calls = 0

    def active_run(self) -> _FakeRun | None:
        return self.active

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, **kwargs: Any) -> _FakeRun:
        self.start_kwargs = kwargs
        run = _FakeRun()
        self.active = run
        return run

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.metrics.setdefault(key, []).append((value, None))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for key, value in metrics.items():
            self.metrics.setdefault(key, []).append((value, step))

    def log_text(self, text: str, artifact_file: str) -> None:
        self.texts.append((text, artifact_file))

    def log_table(self, data: dict[str, list[Any]], artifact_file: str) -> None:
        self.tables.append((data, artifact_file))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        self.directories.append((local_dir, artifact_path))

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        self.figures.append((figure, artifact_file))

    def end_run(self) -> None:
        self.end_calls += 1
        self.active = None


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> _FakeMlflowModule:
    fake = _FakeMlflowModule()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _block_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)


@pytest.mark.parametrize("managed", [False, True])
def test_tracking_artifact_failure_respects_run_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, managed: bool
) -> None:
    fake = _FakeMlflowModule()
    run = fake.start_run()
    monkeypatch.setattr(
        tracking_export, "resolve_run", lambda *a, **kw: (fake, run, managed)
    )
    output = tmp_path / "events.json"
    output.write_text("[]", encoding="utf-8")

    def fail_upload(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(tracking_export, "log_file_artifact", fail_upload)
    with pytest.raises(RuntimeError, match="upload failed"):
        export_tracking_run_to_mlflow(
            mlflow_config_from_namespace(
                Namespace(mlflow=True, mlflow_log_artifacts=True)
            ),
            command_name="track",
            session_summary=None,
            stats={},
            events=[],
            output_path=output,
        )
    assert fake.end_calls == int(managed)


def test_mlflow_config_from_namespace_collects_explicit_values() -> None:
    config = mlflow_config_from_namespace(
        Namespace(
            mlflow=True,
            mlflow_tracking_uri="file:./mlruns",
            mlflow_experiment="stormlog-tests",
            mlflow_run_id="run-123",
            mlflow_run_name="track smoke",
            mlflow_group="job-42",
            mlflow_job_type="stormlog-track",
            mlflow_log_artifacts=True,
            mlflow_log_attribution=True,
        )
    )

    assert config.enabled is True
    assert config.tracking_uri == "file:./mlruns"
    assert config.experiment == "stormlog-tests"
    assert config.run_id == "run-123"
    assert config.run_name == "track smoke"
    assert config.group == "job-42"
    assert config.job_type == "stormlog-track"
    assert config.log_tables is True
    assert config.log_artifacts is True
    assert config.log_attribution is True


def test_ensure_mlflow_available_reports_optional_dependency_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals: dict[str, object] | None = None,  # noqa: A002
        locals: dict[str, object] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(ImportError, match=r"stormlog\[mlflow\]"):
        ensure_mlflow_available(mlflow_config_from_namespace(Namespace(mlflow=True)))


def test_export_tracking_run_logs_metrics_tables_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)

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
    config = mlflow_config_from_namespace(
        Namespace(
            mlflow=True,
            mlflow_tracking_uri="file:./mlruns",
            mlflow_experiment="stormlog-tests",
            mlflow_group=None,
            mlflow_job_type=None,
            mlflow_log_artifacts=True,
            mlflow_log_attribution=True,
        )
    )

    export_tracking_run_to_mlflow(
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

    assert fake_mlflow.tracking_uri == "file:./mlruns"
    assert fake_mlflow.experiment == "stormlog-tests"
    assert fake_mlflow.start_kwargs is not None
    assert fake_mlflow.start_kwargs["run_name"].startswith("stormlog-track-")
    assert fake_mlflow.tags["stormlog_session_id"] == "session-12345678"
    assert fake_mlflow.tags["stormlog_group"] == "train-42"
    assert fake_mlflow.tags["stormlog_job_type"] == "stormlog-track"
    assert fake_mlflow.end_calls == 1

    assert fake_mlflow.metrics["stormlog_peak_memory_bytes"] == [(4096.0, None)]
    assert fake_mlflow.metrics["stormlog_total_events"] == [(3.0, None)]
    assert fake_mlflow.metrics["stormlog_chart_point_count"] == [(3.0, None)]
    assert "stormlog_backend" not in fake_mlflow.metrics  # strings go to tags
    assert fake_mlflow.tags["stormlog_backend"] == "cuda"
    assert fake_mlflow.tags["stormlog_rank"] == "2"
    assert fake_mlflow.tags["stormlog_local_rank"] == "0"
    assert fake_mlflow.tags["stormlog_world_size"] == "8"
    assert "stormlog_rank" not in fake_mlflow.metrics
    assert "stormlog_local_rank" not in fake_mlflow.metrics
    assert "stormlog_world_size" not in fake_mlflow.metrics
    assert fake_mlflow.tags["stormlog_attribution_html_file"] == (
        "cuda_allocator_state_history_annotated.html"
    )

    timeline_steps = {
        step
        for values in fake_mlflow.metrics.values()
        for _, step in values
        if step is not None
    }
    assert timeline_steps == {0, 1, 2}
    assert len(fake_mlflow.metrics["stormlog_timeline_allocated_bytes"]) == 3

    table_files = {artifact_file for _, artifact_file in fake_mlflow.tables}
    table_prefixes = {artifact_file.rsplit("/", 1)[0] for artifact_file in table_files}
    assert len(table_prefixes) == 1
    table_prefix = table_prefixes.pop()
    assert table_prefix.startswith("stormlog-exports/session-12345678/")
    assert {artifact_file.rsplit("/", 1)[1] for artifact_file in table_files} == {
        "stormlog_alerts.json",
        "stormlog_memory_timeline.json",
        "stormlog_tensor_attribution.json",
    }

    text_files = {artifact_file for _, artifact_file in fake_mlflow.texts}
    assert "stormlog_tracking_dashboard.html" in text_files
    assert "stormlog_attribution_preview.html" in text_files
    attribution_text = next(
        text
        for text, artifact_file in fake_mlflow.texts
        if artifact_file == "stormlog_attribution_preview.html"
    )
    assert "Stormlog GPU Attribution Preview" in attribution_text
    assert "<script>" not in attribution_text

    artifact_paths = {artifact_path for _, artifact_path in fake_mlflow.artifacts}
    assert "stormlog-track-output-session-12345678" in artifact_paths
    assert "stormlog-tracking-dashboard-session-12345678" in artifact_paths
    assert "stormlog-attribution-session-12345678" in artifact_paths
    assert {directory_path for directory_path, _ in fake_mlflow.directories} == {
        str(sink_dir),
        str(oom_dir),
    }
    assert fake_mlflow.figures == []


def test_export_diagnose_bundle_logs_summary_and_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)

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

    config = mlflow_config_from_namespace(
        Namespace(
            mlflow=True,
            mlflow_experiment="stormlog-diag",
            mlflow_log_artifacts=True,
            mlflow_log_attribution=True,
        )
    )

    export_diagnose_bundle_to_mlflow(
        config,
        command_name="stormlog-diagnose",
        artifact_dir=artifact_dir,
    )

    assert fake_mlflow.experiment == "stormlog-diag"
    assert fake_mlflow.end_calls == 1
    assert fake_mlflow.metrics["stormlog_allocated_bytes"] == [(1024.0, None)]
    assert fake_mlflow.tags["stormlog_artifact_dir"] == "stormlog-diagnose"
    assert fake_mlflow.tags["stormlog_session_id"] == "diag-12345678"
    assert fake_mlflow.tags["stormlog_attribution_html_file"] == (
        "cuda_allocator_state_history_annotated.html"
    )
    assert len(fake_mlflow.tables) == 1
    assert fake_mlflow.tables[0][1].startswith("stormlog-exports/diag-12345678/")
    assert fake_mlflow.tables[0][1].endswith("/stormlog_diagnostic_suggestions.json")
    assert {artifact_file for _, artifact_file in fake_mlflow.texts} == {
        "stormlog_attribution_preview.html"
    }
    assert fake_mlflow.directories == [
        (str(artifact_dir), "stormlog-diagnose-diag-12345678")
    ]


def test_tracking_visual_artifacts_respect_log_artifacts_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(Namespace(mlflow=True)),
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

    assert fake_mlflow.artifacts == []
    assert fake_mlflow.directories == []
    assert "stormlog_tracking_dashboard_file" not in fake_mlflow.tags
    assert {artifact_file for _, artifact_file in fake_mlflow.texts} == {
        "stormlog_tracking_dashboard.html"
    }


def test_attribution_inline_logging_respects_log_artifacts_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)

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

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(
            Namespace(
                mlflow=True,
                mlflow_log_artifacts=False,
                mlflow_log_attribution=True,
            )
        ),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[],
        attribution_bundle_dir=attribution_dir,
    )

    assert fake_mlflow.artifacts == []
    assert {artifact_file for _, artifact_file in fake_mlflow.texts} == {
        "stormlog_attribution_preview.html"
    }
    assert len(fake_mlflow.tables) == 1
    assert fake_mlflow.tables[0][1].startswith("stormlog-exports/")
    assert fake_mlflow.tables[0][1].endswith("/stormlog_tensor_attribution.json")


def test_export_uses_active_mlflow_run_without_creating_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)
    fake_mlflow.active = _FakeRun()

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(
            Namespace(
                mlflow=True,
                mlflow_group="active-group",
                mlflow_job_type="active-job-type",
            )
        ),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[],
    )

    assert fake_mlflow.start_kwargs is None
    assert fake_mlflow.end_calls == 0
    assert fake_mlflow.tags["stormlog_command"] == "stormlog-track"
    assert fake_mlflow.tags["stormlog_group"] == "active-group"
    assert fake_mlflow.tags["stormlog_job_type"] == "active-job-type"
    assert "stormlog_session_id" in fake_mlflow.tags
    assert fake_mlflow.metrics["stormlog_peak_memory_bytes"] == [(128.0, None)]


def test_tracking_resumes_existing_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(
            Namespace(mlflow=True, mlflow_run_id="existing-run-1")
        ),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[],
    )

    assert fake_mlflow.start_kwargs is not None
    assert fake_mlflow.start_kwargs["run_id"] == "existing-run-1"
    assert "run_name" not in fake_mlflow.start_kwargs
    assert fake_mlflow.experiment is None
    assert fake_mlflow.end_calls == 1


def test_tracking_resume_forwards_explicit_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(
            Namespace(
                mlflow=True,
                mlflow_run_id="existing-run-1",
                mlflow_run_name="explicit rename",
            )
        ),
        command_name="stormlog-track",
        session_summary=create_session_summary(source="stormlog.tracker"),
        stats={"peak_memory": 128},
        events=[],
    )

    assert fake_mlflow.start_kwargs is not None
    assert fake_mlflow.start_kwargs["run_id"] == "existing-run-1"
    assert fake_mlflow.start_kwargs["run_name"] == "explicit rename"
    assert fake_mlflow.end_calls == 1


def test_repeated_exports_use_distinct_table_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    _block_matplotlib(monkeypatch)
    config = mlflow_config_from_namespace(Namespace(mlflow=True))
    summary = create_session_summary(
        source="stormlog.tracker",
        session_id="retry-session",
    )
    events = [
        {
            "timestamp": 1.0,
            "event_type": "sample",
            "memory_allocated": 128,
            "memory_reserved": 256,
            "device_used": 256,
            "device_total": 1024,
        }
    ]

    for _ in range(2):
        export_tracking_run_to_mlflow(
            config,
            command_name="stormlog-track",
            session_summary=summary,
            stats={"peak_memory": 128},
            events=events,
        )

    table_paths = [artifact_file for _, artifact_file in fake_mlflow.tables]
    assert len(table_paths) == 2
    assert len(set(table_paths)) == 2
    assert all(
        path.startswith("stormlog-exports/retry-session/")
        and path.endswith("/stormlog_memory_timeline.json")
        for path in table_paths
    )


def test_memory_plots_are_skipped_when_matplotlib_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(Namespace(mlflow=True)),
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
            },
            {
                "timestamp": 2.0,
                "event_type": "sample",
                "memory_allocated": 256,
                "memory_reserved": 512,
                "device_used": 512,
                "device_total": 1024,
            },
        ],
    )

    assert fake_mlflow.figures == []
    assert {artifact_file for _, artifact_file in fake_mlflow.texts} == {
        "stormlog_tracking_dashboard.html"
    }


class _FakeFigure:
    pass


class _FakePyplot:
    def __init__(self) -> None:
        self.figures: list[_FakeFigure] = []

    def figure(self, figsize: tuple[float, float] | None = None) -> _FakeFigure:
        figure = _FakeFigure()
        self.figures.append(figure)
        return figure

    def plot(self, *args: Any, **kwargs: Any) -> None:
        return None

    def xlabel(self, *args: Any, **kwargs: Any) -> None:
        return None

    def ylabel(self, *args: Any, **kwargs: Any) -> None:
        return None

    def title(self, *args: Any, **kwargs: Any) -> None:
        return None

    def legend(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tight_layout(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self, figure: _FakeFigure) -> None:
        return None


class _FakeMatplotlibModule(ModuleType):
    def __init__(self, pyplot: _FakePyplot) -> None:
        super().__init__("matplotlib")
        self.pyplot = pyplot
        self.backend: str | None = None

    def use(self, backend: str) -> None:
        self.backend = backend


def test_memory_plots_are_logged_when_matplotlib_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    pyplot = _FakePyplot()
    fake_matplotlib = _FakeMatplotlibModule(pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    export_tracking_run_to_mlflow(
        mlflow_config_from_namespace(Namespace(mlflow=True)),
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
            },
            {
                "timestamp": 2.0,
                "event_type": "sample",
                "memory_allocated": 256,
                "memory_reserved": 512,
                "device_used": 512,
                "device_total": 1024,
            },
        ],
    )

    assert fake_matplotlib.backend == "Agg"
    figure_files = {artifact_file for _, artifact_file in fake_mlflow.figures}
    assert figure_files == {
        "stormlog_memory_timeline.png",
        "stormlog_memory_utilization.png",
    }
