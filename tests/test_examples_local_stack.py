"""local_stack.py: services started, killed and stopped by the pid it started."""

import contextlib
import json
import os
import signal
import stat
import sys
from pathlib import Path

import psutil
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


def test_a_relative_binary_override_is_found_from_where_it_was_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A service starts in its state directory, so a path relative to the
    # caller's directory must be made absolute before then.
    _fake_binary(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORMLOG_OTELCOL", "./fake-otelcol")
    monkeypatch.setenv("STORMLOG_PROMETHEUS", "missing")
    monkeypatch.setenv("STORMLOG_JAEGER", "missing")
    state = ["--state-dir", str(tmp_path / "state")]
    try:
        assert local_stack.main(["start", *state]) == 0
        assert "otelcol: started" in capsys.readouterr().out
        saved = json.loads((tmp_path / "state" / "otelcol.pid.json").read_text())
        assert Path(saved["cmdline"][0]).is_absolute() or any(
            Path(part).is_absolute() and part.endswith("fake-otelcol")
            for part in saved["cmdline"]
        )
    finally:
        local_stack.main(["stop", *state])


def test_the_stack_script_never_signals_a_reused_pid(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    # This test's own pid, with a start time that is not its own.
    (state / "otelcol.pid.json").write_text(
        json.dumps({"pid": os.getpid(), "started": 1.0, "x1": False})
    )
    assert local_stack.main(["stop", "--state-dir", str(state)]) == 0
    assert not (state / "otelcol.pid.json").exists()


def _started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binary: Path
) -> tuple[list[str], int]:
    monkeypatch.setenv("STORMLOG_OTELCOL", str(binary))
    monkeypatch.setenv("STORMLOG_PROMETHEUS", str(tmp_path / "missing"))
    monkeypatch.setenv("STORMLOG_JAEGER", str(tmp_path / "missing"))
    state = ["--state-dir", str(tmp_path / "state")]
    assert local_stack.main(["start", *state]) == 0
    saved = json.loads((tmp_path / "state" / "otelcol.pid.json").read_text())
    return state, int(saved["pid"])


def test_a_service_that_ignores_sigterm_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubborn = tmp_path / "fake-otelcol"
    stubborn.write_text(
        f"#!{sys.executable}\nimport signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True: time.sleep(1)\n"
    )
    stubborn.chmod(stubborn.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(local_stack, "STOP_SECONDS", 0.5)
    state, pid = _started(tmp_path, monkeypatch, stubborn)
    assert local_stack.main(["stop", "otelcol", *state]) == 0
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_a_service_read_mid_exec_is_still_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Linux lets Popen return once exec has closed the child's close-on-exec
    # descriptors, a moment before the new program's arguments are set up;
    # a cmdline read in that window is empty. Saved so, the service later
    # looked like another process, and stop left it running.
    real_cmdline = psutil.Process.cmdline
    read: set[int] = set()

    def first_read_mid_exec(self: psutil.Process) -> list[str]:
        if self.pid not in read:
            read.add(self.pid)
            return []
        cmdline: list[str] = real_cmdline(self)
        return cmdline

    monkeypatch.setattr(psutil.Process, "cmdline", first_read_mid_exec)
    monkeypatch.setattr(local_stack, "STOP_SECONDS", 0.5)
    state, pid = _started(tmp_path, monkeypatch, _fake_binary(tmp_path))
    try:
        assert local_stack.main(["stop", "otelcol", *state]) == 0
        with pytest.raises(OSError):
            os.kill(pid, 0)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _saved(state: Path, pid: int, cmdline: list[str]) -> None:
    state.mkdir(parents=True, exist_ok=True)
    started = psutil.Process(pid).create_time()
    (state / "otelcol.pid.json").write_text(
        json.dumps({"pid": pid, "started": started, "cmdline": cmdline, "x1": False})
    )


def test_a_process_reading_no_command_line_is_not_the_saved_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate's probe P4: a pid with the saved start time that now reads
    # an empty command line, as a kernel thread does, is not the service a
    # non-empty saved command line names.
    _saved(tmp_path, os.getpid(), ["x"])
    monkeypatch.setattr(psutil.Process, "cmdline", lambda _self: [])
    assert local_stack._running(tmp_path, "otelcol") is None


def test_a_saved_empty_command_line_compares_the_start_time_only(
    tmp_path: Path,
) -> None:
    # A pid file from before the command line was read once set up: the
    # start time alone still names the live service.
    _saved(tmp_path, os.getpid(), [])
    running = local_stack._running(tmp_path, "otelcol")
    assert running is not None and running.pid == os.getpid()


def test_a_service_started_through_a_wrapper_that_execs_is_still_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An override may name a wrapper that execs the real binary, as a macOS
    # framework Python re-launches itself: the command line changes after
    # start, and the one saved must be the one that lasts.
    real = _fake_binary(tmp_path)
    wrapper = tmp_path / "otelcol-wrapper"
    wrapper.write_text(f"#!/bin/sh\nsleep 0.05\nexec '{real}'\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(local_stack, "STOP_SECONDS", 0.5)
    state, pid = _started(tmp_path, monkeypatch, wrapper)
    try:
        assert local_stack.main(["stop", "otelcol", *state]) == 0
        with pytest.raises(OSError):
            os.kill(pid, 0)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def test_a_zombie_service_is_not_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state, pid = _started(tmp_path, monkeypatch, _fake_binary(tmp_path))
    try:
        monkeypatch.setattr(
            psutil.Process, "status", lambda _self: psutil.STATUS_ZOMBIE
        )
        local_stack.main(["status", *state])
        assert "otelcol: not running" in capsys.readouterr().out
    finally:
        monkeypatch.undo()
        os.kill(pid, 9)


def test_a_clock_step_does_not_make_a_service_look_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # On Linux a process's start time is derived from the boot time, which
    # moves when the wall clock is stepped.
    state, pid = _started(tmp_path, monkeypatch, _fake_binary(tmp_path))
    real = psutil.Process.create_time
    try:
        monkeypatch.setattr(
            psutil.Process, "create_time", lambda self: real(self) + 0.5
        )
        local_stack.main(["status", *state])
        assert "otelcol: running" in capsys.readouterr().out
    finally:
        monkeypatch.undo()
        local_stack.main(["stop", "otelcol", *state])
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_a_pid_now_running_another_command_is_never_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pid reused within the start-time tolerance: the start time matches,
    # the command line does not, so the process is not the one started.
    state, pid = _started(tmp_path, monkeypatch, _fake_binary(tmp_path))
    pid_file = tmp_path / "state" / "otelcol.pid.json"
    saved = json.loads(pid_file.read_text())
    saved["cmdline"] = ["/usr/bin/some-other-program"]
    pid_file.write_text(json.dumps(saved))
    process = psutil.Process(pid)
    try:
        local_stack.main(["stop", "otelcol", *state])
        assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    finally:
        process.kill()
        process.wait(5)
