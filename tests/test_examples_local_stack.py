"""local_stack.py: services started, killed and stopped by the pid it started."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from examples.observability import local_stack


def _fake_binary(tmp_path: Path) -> Path:
    script = tmp_path / "fake-otelcol"
    script.write_text(f"#!{sys.executable}\nimport time\nwhile True: time.sleep(1)\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_the_stack_script_starts_kills_and_stops_by_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("STORMLOG_OTELCOL", str(_fake_binary(tmp_path)))
    monkeypatch.setenv("STORMLOG_PROMETHEUS", str(tmp_path / "missing"))
    monkeypatch.setenv("STORMLOG_JAEGER", str(tmp_path / "missing"))
    state = ["--state-dir", str(tmp_path / "state")]
    assert local_stack.main(["start", "--x1", *state]) == 0
    out = capsys.readouterr().out
    assert "otelcol: started" in out and "prometheus: no prometheus binary" in out
    saved = json.loads((tmp_path / "state" / "otelcol.pid.json").read_text())
    assert saved["x1"] is True
    local_stack.main(["status", *state])
    assert "otelcol: running" in capsys.readouterr().out
    assert local_stack.main(["kill", "otelcol", *state]) == 0
    assert "otelcol: killed" in capsys.readouterr().out
    local_stack.main(["status", *state])
    assert "otelcol: not running" in capsys.readouterr().out
    with pytest.raises(OSError):
        os.kill(int(saved["pid"]), 0)


def test_the_stack_script_never_signals_a_reused_pid(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    # This test's own pid, with a start time that is not its own.
    (state / "otelcol.pid.json").write_text(
        json.dumps({"pid": os.getpid(), "started": 1.0, "x1": False})
    )
    assert local_stack.main(["stop", "--state-dir", str(state)]) == 0
    assert not (state / "otelcol.pid.json").exists()
