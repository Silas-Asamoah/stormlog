"""The signal pulser, against the fake vLLM engine run as its own process."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
from typing import Iterator

import psutil
import pytest

from examples.qualification.fake_engine.process import FakeEngineProcess, _environment
from examples.qualification.pulser import (
    MAX_PULSE_SECONDS,
    Pulser,
    PulseRefused,
    Target,
    check_schedule,
    discover_roles,
)
from tests.qualification_fake_engine_helpers import get, wait_until

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(signal, "SIGSTOP"),
    reason="needs SIGSTOP and SIGCONT",
)


@pytest.fixture
def server() -> Iterator[FakeEngineProcess]:
    process = FakeEngineProcess(["--step-seconds", "0.001"])
    process.start()
    yield process
    process.stop()


def _answers(server: FakeEngineProcess, timeout: float = 0.2) -> bool:
    try:
        return get(f"{server.base_url}/health", timeout=timeout)[0] == 200
    except (TimeoutError, socket.timeout, urllib.error.URLError):
        return False


def test_a_pulse_stops_confirms_and_continues(server: FakeEngineProcess) -> None:
    target = Target.of(server.pid, "engine_core")
    seen_during: list[bool] = []
    with Pulser(target, watchdog=False) as pulser:
        pulse = pulser.pulse(0.3, during=lambda: seen_during.append(_answers(server)))
    assert seen_during == [False]
    assert not target.is_stopped()
    assert _answers(server, timeout=2)
    assert pulse.stop_sent_ns <= pulse.stopped_ns < pulse.continue_sent_ns
    assert pulse.confirm_latency_ns < 1_000_000_000
    assert pulse.continue_sent_ns - pulse.stopped_ns >= 290_000_000


def test_a_recycled_pid_is_never_signalled(server: FakeEngineProcess) -> None:
    impostor = Target(server.pid, Target.of(server.pid).start_time - 10)
    with Pulser(impostor, watchdog=False) as pulser:
        with pytest.raises(PulseRefused, match="no longer the target"):
            pulser.pulse(0.1)
    assert _answers(server)


def test_pulses_are_capped() -> None:
    with pytest.raises(PulseRefused):
        check_schedule(MAX_PULSE_SECONDS + 0.1, 10.0)
    with pytest.raises(PulseRefused):
        check_schedule(0.2, 0.3)  # a 67% duty cycle
    check_schedule(0.1, 2.0)


def test_the_target_is_continued_however_the_pulse_ends(
    server: FakeEngineProcess,
) -> None:
    target = Target.of(server.pid)

    def interrupted() -> None:
        raise KeyboardInterrupt

    with Pulser(target, watchdog=False) as pulser:
        with pytest.raises(KeyboardInterrupt):
            pulser.pulse(1.5, during=interrupted)
        assert not target.is_stopped()
    assert _answers(server, timeout=2)


def test_a_schedule_runs_its_pulses_one_per_period(server: FakeEngineProcess) -> None:
    with Pulser(Target.of(server.pid), watchdog=False) as pulser:
        pulses = pulser.run(0.05, 0.2, count=3)
    starts = [pulse.stop_sent_ns for pulse in pulses]
    assert len(pulses) == 3
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    assert all(gap >= 180_000_000 for gap in gaps)


def test_the_watchdog_continues_a_target_whose_harness_was_killed(
    server: FakeEngineProcess,
) -> None:
    # A harness that stops the target and is then killed outright: neither
    # its finally nor its atexit runs, so only the watchdog can continue it.
    script = textwrap.dedent(
        f"""
        import time
        from examples.qualification.pulser import Pulser, Target
        pulser = Pulser(Target.of({server.pid}), max_pulse_seconds=2.0)
        time.sleep(1.5)  # let the watchdog start
        print("ready", flush=True)
        pulser.pulse(2.0, during=lambda: time.sleep(60))
        """
    )
    harness = subprocess.Popen(
        [sys.executable, "-c", script],
        env=_environment(),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert harness.stdout is not None
        assert harness.stdout.readline().strip() == "ready"
        target = Target.of(server.pid)
        assert wait_until(target.is_stopped, timeout=5)
        os.kill(harness.pid, signal.SIGKILL)
        harness.wait(timeout=5)
        started = time.monotonic()
        assert wait_until(lambda: not target.is_stopped(), timeout=5)
        assert time.monotonic() - started < 2
    finally:
        if harness.poll() is None:
            harness.kill()
    assert _answers(server, timeout=2)


def test_roles_are_found_under_the_api_server_by_title() -> None:
    # A stand-in tree: an API server whose children carry vLLM's titles.
    script = textwrap.dedent(
        """
        import subprocess, sys, time
        children = [
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", title])
            for title in ("VLLM::EngineCore", "VLLM::Worker_TP0")
        ]
        print("ready", flush=True)
        time.sleep(30)
        """
    )
    server = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert server.stdout is not None
        assert server.stdout.readline().strip() == "ready"
        roles = discover_roles(server.pid)
        assert sorted(roles) == ["api_server", "engine_core", "worker_tp0"]
        assert all(target.is_alive() for target in roles.values())
        assert roles["api_server"].pid == server.pid
    finally:
        for child in psutil.Process(server.pid).children(recursive=True):
            child.kill()
        server.kill()
        server.wait()
