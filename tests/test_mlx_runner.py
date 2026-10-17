import json
import os
import sys
import threading

import pytest

from stormlog.mlx.analyzer import analyze
from stormlog.mlx.artifacts import CaptureOptions
from stormlog.mlx.runner import run_script
from stormlog.report import load_report
from tests.mlx_fakes import make_runtime


def execute(tmp_path, code, **kwargs):
    script = tmp_path / "script.py"
    script.write_text(code)
    return run_script(
        script,
        output=tmp_path / "capture",
        options=CaptureOptions(interval=10),
        runtime=make_runtime(),
        **kwargs,
    )


def test_same_process_argv_import_path_and_cwd_restored(tmp_path):
    original_argv, original_path = sys.argv, sys.path
    (tmp_path / "sibling.py").write_text("VALUE = 7")
    code = f"""
import json, os, sys
from pathlib import Path
import sibling
assert __name__ == '__main__'
Path({str(tmp_path / 'observed.json')!r}).write_text(json.dumps({{'pid': os.getpid(), 'cwd': os.getcwd(), 'argv': sys.argv, 'sibling': sibling.VALUE}}))
sys.argv = ['changed']
sys.path = []
"""
    directory, outcome = execute(tmp_path, code, args=["--output", "user-value"])
    assert outcome.exit_code == 0
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert observed["pid"] == os.getpid()
    assert observed["cwd"] == os.getcwd()
    assert observed["argv"][1:] == ["--output", "user-value"]
    assert observed["sibling"] == 7
    assert sys.argv is original_argv and sys.path is original_path
    report = load_report(directory / "report.json")
    assert report["payload"]["workload"]["status"] == "completed"
    assert report["payload"]["final_lazy_work"] == "application_responsibility"
    assert analyze(directory)["session"]["status"] == "completed"


@pytest.mark.parametrize(
    "statement,expected,script_exit",
    [
        ("raise SystemExit()", 0, None),
        ("raise SystemExit(0)", 0, 0),
        ("raise SystemExit(7)", 1, 7),
        ("raise SystemExit('failed')", 1, "failed"),
        ("raise ValueError('user failure')", 1, None),
        ("raise KeyboardInterrupt()", 130, None),
        ("raise RuntimeError('[malloc] Unable to allocate 4096 bytes.')", 3, None),
        ("raise RuntimeError('Metal compiler failed')", 1, None),
    ],
)
def test_exit_codes_and_finalized_artifacts(tmp_path, statement, expected, script_exit):
    directory, outcome = execute(tmp_path, statement)
    report = load_report(directory / "report.json")
    assert outcome.exit_code == expected == report["verdict"]["exit_code"]
    if "SystemExit" in statement:
        assert report["payload"]["workload"]["system_exit"] == script_exit
    assert not any(t.name.startswith("stormlog-mlx") for t in threading.enumerate())
    session = analyze(directory)["session"]
    assert session["status"] == (
        "completed"
        if expected == 0
        else "interrupted" if expected == 130 else "incomplete"
    )
    if expected == 3:
        assert outcome.oom_dump
        assert report["findings"][0]["kind"] == "oom_occurred"
        from stormlog import query

        assert len(query.open([directory]).list_oom_bundles()) == 1


def test_missing_script_does_not_initialize_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stormlog.mlx.artifacts.MLXRuntime", lambda: pytest.fail("runtime discovery")
    )
    directory, outcome = run_script(
        tmp_path / "missing.py", output=tmp_path / "capture"
    )
    assert outcome.exit_code == 5
    assert load_report(directory / "report.json")["verdict"]["exit_code"] == 5


def test_runtime_failure_does_not_run_script(tmp_path, monkeypatch):
    from stormlog.mlx.runtime import MLXNotInstalledError

    def fail():
        raise MLXNotInstalledError("install stormlog[mlx]")

    monkeypatch.setattr("stormlog.mlx.artifacts.MLXRuntime", fail)
    script = tmp_path / "script.py"
    script.write_text("raise AssertionError('workload must not run')")
    directory, outcome = run_script(script, output=tmp_path / "capture")
    assert outcome.exit_code == 2
    assert (
        load_report(directory / "report.json")["payload"]["workload"]["status"]
        == "not_started"
    )


def test_runner_never_evaluates_synchronizes_or_resets(tmp_path):
    runtime = make_runtime()
    script = tmp_path / "script.py"
    script.write_text("pending = object()")
    run_script(script, output=tmp_path / "capture", runtime=runtime)
    assert "eval" not in runtime.core.calls
    assert "reset" not in runtime.core.calls
    assert not any(isinstance(c, tuple) and c[0] == "sync" for c in runtime.core.calls)


def test_cleanup_failure_preserves_workload_status(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("stormlog.mlx.artifacts.write_report", fail)
    directory, outcome = execute(tmp_path, "raise KeyboardInterrupt()")
    assert outcome.exit_code == 130
    assert outcome.artifact_errors
    assert json.loads((directory / "manifest.json").read_text())["exit_code"] == 130
    assert not (directory / "report.json").exists()


def test_threshold_and_bounded_retention_roundtrip(tmp_path):
    from stormlog import query

    script = tmp_path / "script.py"
    script.write_text("import time\ntime.sleep(.06)")
    options = CaptureOptions(
        interval=0.002,
        max_history=2,
        rollover_max_events=5,
        retention_max_files=20,
        alert_threshold_mb=0.00001,
    )
    directory, outcome = run_script(
        script, output=tmp_path / "capture", options=options, runtime=make_runtime()
    )
    assert outcome.exit_code == 3
    tracking = json.loads((directory / "tracking.json").read_text())
    assert tracking["total_samples"] > 2
    assert len(tracking["samples"]) == 2
    assert (
        len(query.open([directory / "telemetry"]).query_events())
        == tracking["total_events"]
    )
    assert analyze(directory)["session"]["status"] == "completed"
