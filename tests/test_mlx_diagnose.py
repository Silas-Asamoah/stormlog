import json

import pytest

from stormlog.mlx.artifacts import CaptureOptions
from stormlog.mlx.diagnose import capture_process, run_diagnose
from stormlog.report import load_report
from tests.mlx_fakes import make_runtime


def test_bundle_environment_capabilities_and_evidence(tmp_path):
    from stormlog import query

    directory, outcome = run_diagnose(
        output=tmp_path / "bundle", runtime=make_runtime()
    )
    assert outcome.exit_code == 0
    report = load_report(directory / "report.json")
    assert report["report_kind"] == "diagnose"
    environment = json.loads((directory / "environment.json").read_text())
    assert environment["memory_capabilities"]["supports_allocator_allocated"]
    assert not environment["memory_capabilities"]["supports_device_total"]
    summary = json.loads((directory / "diagnostic_summary.json").read_text())
    assert summary["total_bytes"] is None
    assert summary["utilization_ratio"] is None
    assert summary["fragmentation_ratio"] is None
    for artifact in report["artifacts"]:
        assert (directory / artifact["path"]).exists()
    assert query.open([directory]).list_sessions()


def test_runtime_unavailable_bundle(tmp_path, monkeypatch):
    from stormlog.mlx.runtime import MLXMetalUnavailableError

    def fail():
        raise MLXMetalUnavailableError("no device")

    monkeypatch.setattr("stormlog.mlx.artifacts.MLXRuntime", fail)
    directory, outcome = run_diagnose(output=tmp_path / "bundle")
    assert outcome.exit_code == 2
    report = load_report(directory / "report.json")
    assert report["verdict"]["status"] == "usage"
    assert report["findings"] == []
    assert "no device" in report["verdict"]["summary"]


def test_interrupt_and_core_sampling_failure_are_incomplete(tmp_path):
    def interrupt(seconds):
        raise KeyboardInterrupt()

    directory, outcome = capture_process(
        output=tmp_path / "interrupted", runtime=make_runtime(), wait=interrupt
    )
    assert outcome.exit_code == 130
    assert load_report(directory / "report.json")["verdict"]["status"] == "interrupted"
    runtime = make_runtime()
    runtime.core.values["active_memory"] = RuntimeError("getter failed")
    directory, outcome = run_diagnose(output=tmp_path / "failed", runtime=runtime)
    assert outcome.exit_code == 1
    assert load_report(directory / "report.json")["verdict"]["status"] == "error"
    summary = json.loads((directory / "diagnostic_summary.json").read_text())
    assert summary["allocated_bytes"] is None


def test_collision_safe_output_and_argument_validation(tmp_path):
    for _ in range(2):
        directory, _ = run_diagnose(output=tmp_path, runtime=make_runtime())
        assert directory.parent == tmp_path
    assert len(list(tmp_path.iterdir())) == 2
    for duration in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            run_diagnose(output=tmp_path, duration=duration)
    with pytest.raises(ValueError):
        CaptureOptions(max_history=0)


def test_summary_write_failure_never_leaves_passing_report(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.write_text

    def fail(self, *args, **kwargs):
        if self.name == "diagnostic_summary.json":
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    directory, outcome = run_diagnose(
        output=tmp_path / "bundle", runtime=make_runtime()
    )
    assert outcome.exit_code == 1
    report = load_report(directory / "report.json")
    assert report["verdict"]["status"] == "error"
    assert all((directory / a["path"]).exists() for a in report["artifacts"])
