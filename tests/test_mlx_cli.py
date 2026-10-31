import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from stormlog.mlx.cli import main
from stormlog.report import load_report
from tests.mlx_fakes import make_runtime


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stormlog.mlx.artifacts.MLXRuntime", make_runtime)
    monkeypatch.setattr("stormlog.mlx.runtime.MLXRuntime", make_runtime)


@pytest.mark.parametrize(
    "args", [[], ["--help"], ["run", "--help"], ["analyze", "--help"]]
)
def test_help_is_lazy(args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def no_discovery() -> None:
        pytest.fail("native discovery")

    monkeypatch.setattr("stormlog.mlx.runtime.MLXRuntime", no_discovery)
    assert main(args) == (2 if not args else 0)


def test_info_available_and_unavailable_json(
    tmp_path: Path,
    fake_runtime: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["info", "--json", "--output", str(tmp_path / "info")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is True
    assert payload["device"]["mlx_version"] == "0.32.3"
    assert not payload["memory_capabilities"]["supports_device_total"]
    from stormlog.mlx.runtime import MLXNotInstalledError

    def fail() -> None:
        raise MLXNotInstalledError("Install stormlog[mlx]")

    monkeypatch.setattr("stormlog.mlx.runtime.MLXRuntime", fail)
    assert main(["info", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["available"] is False


@pytest.mark.parametrize("command", ["monitor", "track", "diagnose"])
def test_capture_commands_reports_and_roundtrip(
    tmp_path: Path, fake_runtime: None, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    from stormlog import query

    assert (
        main(
            [command, "--duration", "0", "--json", "--output", str(tmp_path / command)]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    directory = payload["artifact_dir"]
    report = load_report(tmp_path / command / "report.json")
    assert report["verdict"]["exit_code"] == 0
    store = query.open([directory])
    sink_events = query.open([Path(directory) / "telemetry"]).query_events()
    assert sink_events
    assert len(store.query_events()) == len(sink_events)
    assert len(store.list_sessions()) == 1
    assert (
        main(
            [
                "analyze",
                directory,
                "--format",
                "json",
                "--output",
                str(tmp_path / "analysis"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["metrics"]["device_free_bytes"] is None
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/stormlog_report_v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(report, schema)
    jsonschema.validate(load_report(tmp_path / "analysis/report.json"), schema)


def test_monitor_prints_passive_live_labels(
    tmp_path: Path, fake_runtime: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["monitor", "--duration", "0", "--output", str(tmp_path / "monitor")]) == 0
    )
    text = capsys.readouterr().out
    assert "This process: active" in text and "lifetime peak" in text


def test_run_workload_arguments_and_failures(
    tmp_path: Path, fake_runtime: None, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "script.py"
    script.write_text(
        "import sys\nassert sys.argv[1:] == ['--duration', '999']\nraise SystemExit(12)"
    )
    assert (
        main(
            [
                "run",
                "--json",
                "--output",
                str(tmp_path / "run"),
                "--",
                str(script),
                "--duration",
                "999",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["workload"]["system_exit"] == 12
    assert load_report(tmp_path / "run/report.json")["verdict"]["exit_code"] == 1


@pytest.mark.parametrize(
    "args",
    [
        ["run"],
        ["track", "--interval", "0"],
        ["track", "--duration", "nan"],
        ["track", "--duration", "-1"],
        ["track", "--max-history", "0"],
        ["track", "--rank", "-1"],
        ["track", "--world-size", "0"],
        ["track", "--max-buffer-bytes", "1024"],
    ],
)
def test_usage_errors(args: list[str], fake_runtime: None, tmp_path: Path) -> None:
    if args == ["track", "--rank", "-1"]:
        args += ["--output", str(tmp_path / "capture")]
    assert main(args) == 2


def test_invalid_input_writes_invalid_input_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "analyze",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "analysis"),
                "--json",
            ]
        )
        == 5
    )
    assert json.loads(capsys.readouterr().out)["exit_code"] == 5
    assert (
        load_report(tmp_path / "analysis/report.json")["verdict"]["status"]
        == "invalid_input"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
def test_real_sigint_finalizes_script_capture(tmp_path: Path) -> None:
    script = tmp_path / "script.py"
    ready = tmp_path / "ready"
    script.write_text(
        f"from pathlib import Path\nimport time\nPath({str(ready)!r}).touch()\ntime.sleep(60)\n"
    )
    directory = tmp_path / "capture"
    code = """
import sys
from tests.mlx_fakes import make_runtime
import stormlog.mlx.artifacts as artifacts
artifacts.MLXRuntime = make_runtime
from stormlog.mlx.cli import main
raise SystemExit(main(['run', '--json', '--output', sys.argv[1], '--', sys.argv[2]]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(directory), str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stdout + stderr
        assert json.loads(stdout)["exit_code"] == 130
        report = load_report(directory / "report.json")
        assert report["verdict"]["status"] == "interrupted"
        manifest = json.loads((directory / "telemetry/manifest.json").read_text())
        assert manifest["sessions"][0]["status"] == "interrupted"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_analysis_output_failure_is_error_not_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pathlib import Path

    from tests.test_mlx_analysis import write_events

    source = tmp_path / "valid.json"
    write_events(source, [100, 100])
    original = Path.write_text

    def write(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name == "analysis.json":
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    assert (
        main(["analyze", str(source), "--output", str(tmp_path / "output"), "--json"])
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    report = json.loads((Path(payload["artifact_dir"]) / "report.json").read_text())
    assert report["verdict"]["exit_code"] == 1
    assert "disk full" in report["payload"]["error"]
    assert not report["artifacts"]


def test_capture_options_refuse_a_buffer_smaller_than_a_record() -> None:
    from stormlog.mlx.artifacts import MIN_BUFFER_BYTES, CaptureOptions

    with pytest.raises(ValueError, match="max_buffer_bytes must be at least"):
        CaptureOptions(max_buffer_bytes=MIN_BUFFER_BYTES - 1)
    assert CaptureOptions(max_buffer_bytes=MIN_BUFFER_BYTES).max_buffer_bytes


@pytest.mark.parametrize("name", ["interval", "alert_threshold_mb"])
def test_capture_options_name_the_invalid_setting(name: str) -> None:
    from stormlog.mlx.artifacts import CaptureOptions

    options: dict[str, Any] = {name: -1.0}
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        CaptureOptions(**options)


def test_text_mode_prints_an_analyze_error_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.json"
    assert main(["analyze", str(missing), "--output", str(tmp_path / "out")]) == 5
    captured = capsys.readouterr()
    assert (captured.out + captured.err).count(str(missing)) == 1


@pytest.mark.parametrize("command", ["info", "track", "analyze"])
def test_unexpected_value_error_is_an_internal_error(command, monkeypatch, capsys):
    import stormlog.mlx.cli as module

    def fail(*args, **kwargs):
        raise ValueError("internal invariant broke")

    monkeypatch.setattr(
        module,
        {"info": "_info", "track": "_capture", "analyze": "_analysis"}[command],
        fail,
    )
    args = [command] + (["input.json"] if command == "analyze" else [])
    assert main(args) == 1
    assert "ValueError: internal invariant broke" in capsys.readouterr().err


def test_capture_internal_value_error_is_not_a_usage_report(
    tmp_path, monkeypatch, capsys
):
    def fail(*args, **kwargs):
        raise ValueError("internal capture invariant")

    monkeypatch.setattr("stormlog.mlx.artifacts.CaptureBundle.start", fail)
    assert (
        main(
            [
                "track",
                "--duration",
                "0",
                "--json",
                "--output",
                str(tmp_path / "capture"),
            ]
        )
        == 1
    )
    output = json.loads(capsys.readouterr().out)
    report = load_report(Path(output["artifact_dir"]) / "report.json")
    assert report["verdict"]["status"] == "error"


@pytest.mark.parametrize(
    "args",
    [["--rank", "1"], ["--local-rank", "-1"], ["--world-size", "1", "--rank", "2"]],
)
def test_distributed_usage_errors_are_validated_before_capture(args, monkeypatch):
    def no_capture(*a, **kw):
        pytest.fail("invalid CLI must fail before creating artifacts")

    monkeypatch.setattr("stormlog.mlx.cli._capture", no_capture)
    assert main(["track", *args]) == 2
