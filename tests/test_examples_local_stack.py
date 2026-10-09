"""local_stack.py: services started, killed and stopped by the pid it started."""

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
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


@contextlib.contextmanager
def _process(tmp_path: Path, *, own_session: bool) -> Iterator[psutil.Process]:
    """A running fake service, leading a session of its own or not."""
    child = subprocess.Popen(
        [str(_fake_binary(tmp_path))], start_new_session=own_session
    )
    try:
        yield psutil.Process(child.pid)
    finally:
        child.kill()
        child.wait(5)


@pytest.mark.parametrize("live", [["/usr/bin/some-other-program"], []])
def test_a_pid_outside_the_service_s_session_is_not_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live: list[str]
) -> None:
    # A pid reused within the start-time tolerance does not lead the session
    # start() gave the service: whatever its command line, even an empty
    # one as a kernel thread reads (the gate's probe P4), it is not the
    # service, and stop never signals it.
    with _process(tmp_path, own_session=False) as other:
        _saved(tmp_path / "state", other.pid, ["x"])
        monkeypatch.setattr(psutil.Process, "cmdline", lambda _self: live)
        assert local_stack._running(tmp_path / "state", "otelcol") is None
        local_stack.main(["stop", "otelcol", "--state-dir", str(tmp_path / "state")])
        assert other.is_running() and other.status() != psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("saved", [[], ["/usr/bin/what-it-was-started-as"]])
def test_the_service_is_known_by_its_start_time_and_session(
    tmp_path: Path, saved: list[str]
) -> None:
    # A process leading its own session, started at the saved time, is the
    # service whatever its command line has become since: a wrapper's exec
    # or an interpreter re-launching itself keeps the session.
    with _process(tmp_path, own_session=True) as service:
        _saved(tmp_path / "state", service.pid, saved)
        running = local_stack._running(tmp_path / "state", "otelcol")
        assert running is not None and running.pid == service.pid


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


def test_a_wrapper_that_execs_after_the_start_check_is_still_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate's case: the wrapper execs the real binary after start's
    # half-second check, so the command line saved is the wrapper's. The
    # service is still known by its start time and session.
    real = _fake_binary(tmp_path)
    wrapper = tmp_path / "otelcol-wrapper"
    wrapper.write_text(f"#!/bin/sh\nsleep 0.6\nexec '{real}'\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(local_stack, "STOP_SECONDS", 0.5)
    state, pid = _started(tmp_path, monkeypatch, wrapper)
    time.sleep(1.0)  # stop comes later, as a user's would, after the exec
    try:
        assert local_stack.main(["stop", "otelcol", *state]) == 0
        with pytest.raises(OSError):
            os.kill(pid, 0)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def test_an_empty_second_read_keeps_the_first_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A service that exits just after start's check reads an empty command
    # line at the second save; that must not replace the one already saved.
    real_settled = local_stack._settled_cmdline
    reads = 0

    def second_read_empty(process: psutil.Process, wait: float = 2.0) -> list[str]:
        nonlocal reads
        reads += 1
        return real_settled(process, wait) if reads == 1 else []

    monkeypatch.setattr(local_stack, "_settled_cmdline", second_read_empty)
    state, pid = _started(tmp_path, monkeypatch, _fake_binary(tmp_path))
    try:
        assert reads == 2
        saved = json.loads((tmp_path / "state" / "otelcol.pid.json").read_text())
        assert any(part.endswith("fake-otelcol") for part in saved["cmdline"])
    finally:
        local_stack.main(["stop", *state])


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
