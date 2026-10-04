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
from typing import Callable, Iterator

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


TITLED = textwrap.dedent(
    """
    import subprocess, sys, time
    # argv[1:]: titles of children to start, as "title" or "title/child-title".
    for spec in sys.argv[1:]:
        title, _, child = spec.partition("/")
        script = (
            "import subprocess, sys, time\\n"
            + (f"subprocess.Popen([{child!r}, '-c', 'import time; time.sleep(30)'], executable=sys.executable)\\n" if child else "")
            + "time.sleep(30)"
        )
        subprocess.Popen([title, "-c", script], executable=sys.executable)
    print("ready", flush=True)
    time.sleep(30)
    """
)


@pytest.fixture
def api_server() -> Iterator[Callable[..., int]]:
    """A stand-in API server whose children carry the titles it's given."""
    servers: list[subprocess.Popen[str]] = []

    def start(*specs: str) -> int:
        server = subprocess.Popen(
            [sys.executable, "-c", TITLED, *specs], stdout=subprocess.PIPE, text=True
        )
        servers.append(server)
        assert server.stdout is not None
        assert server.stdout.readline().strip() == "ready"
        time.sleep(0.5)  # grandchildren
        return server.pid

    yield start
    for server in servers:
        for child in psutil.Process(server.pid).children(recursive=True):
            child.kill()
        server.kill()
        server.wait()


def test_roles_are_found_by_their_exact_titles(api_server: Callable[..., int]) -> None:
    # vLLM 0.30 retitles EngineCore under the API server, and its TP workers
    # under EngineCore. A helper that merely mentions EngineCore in its
    # arguments is not it, and Worker_TP1 is not Worker_TP10.
    pid = api_server(
        "python --about VLLM::EngineCore",
        "VLLM::EngineCore/VLLM::Worker_TP1",
    )
    engine = psutil.Process(pid).children()[1]
    tp1 = engine.children()[0]
    roles = discover_roles(pid)
    assert sorted(roles) == ["api_server", "engine_core", "worker_tp1"]
    assert roles["api_server"].pid == pid
    assert roles["engine_core"].pid == engine.pid
    assert roles["worker_tp1"].pid == tp1.pid
    assert all(target.is_alive() for target in roles.values())


def test_a_worker_rank_is_parsed_not_matched_by_prefix(
    api_server: Callable[..., int],
) -> None:
    pid = api_server("VLLM::EngineCore/VLLM::Worker_TP10")
    assert sorted(discover_roles(pid)) == ["api_server", "engine_core", "worker_tp10"]


@pytest.mark.parametrize(
    "specs",
    [("python -c VLLM::EngineCore",), ("VLLM::EngineCore", "VLLM::EngineCore")],
    ids=["missing", "ambiguous"],
)
def test_a_missing_or_ambiguous_engine_is_refused(
    api_server: Callable[..., int], specs: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="VLLM::EngineCore"):
        discover_roles(api_server(*specs))


# ------------------------------------------------------------------ the harness dies

LOOP = "import time\nwhile True:\n    time.sleep(0.01)\n"
HARNESS = textwrap.dedent(
    """
    import sys, time
    from examples.qualification.pulser import Pulser, Target
    pulser = Pulser(Target.of(int(sys.argv[1])), max_pulse_seconds=2.0)
    print(pulser.watchdog_pid, flush=True)

    def hold() -> None:
        print("stopped", flush=True)
        time.sleep(60)

    pulser.pulse(2.0, during=hold)
    """
)


@pytest.fixture
def loop() -> Iterator[subprocess.Popen[bytes]]:
    """A target in a session of its own, as a server the harness didn't start."""
    process = subprocess.Popen([sys.executable, "-c", LOOP], start_new_session=True)
    yield process
    os.kill(process.pid, signal.SIGCONT)
    process.kill()
    process.wait()


def _pulsing_harness(target: int) -> tuple[subprocess.Popen[str], int]:
    """A harness in its own session, mid-pulse; returns it and its watchdog."""
    harness = subprocess.Popen(
        [sys.executable, "-c", HARNESS, str(target)],
        env=_environment(),
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert harness.stdout is not None
    watchdog = int(harness.stdout.readline())
    assert harness.stdout.readline().strip() == "stopped"
    return harness, watchdog


def _running_within(pid: int, seconds: float) -> bool:
    return wait_until(
        lambda: psutil.Process(pid).status() != psutil.STATUS_STOPPED, timeout=seconds
    )


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_to_the_harnesss_group_continues_the_target(
    loop: subprocess.Popen[bytes], signum: int
) -> None:
    # A job's SIGTERM or an ssh disconnect's SIGHUP reaches the whole group:
    # the harness continues its target before exiting, and the watchdog,
    # in its own session, survives to see it go.
    harness, watchdog = _pulsing_harness(loop.pid)
    os.killpg(harness.pid, signum)
    assert harness.wait(timeout=10) == 128 + signum
    # Well inside the watchdog's 3 s stopped-too-long limit.
    assert _running_within(loop.pid, 1.0)
    assert wait_until(
        lambda: not psutil.pid_exists(watchdog)
        or psutil.Process(watchdog).status() == psutil.STATUS_ZOMBIE,
        timeout=5,
    )


def test_the_termination_handler_continues_before_an_earlier_handler_runs(
    loop: subprocess.Popen[bytes],
) -> None:
    # An earlier handler may exit at once (os._exit skips finally and
    # atexit), so the target is running before it is called.
    from examples.qualification import pulser as pulser_module

    target = Target.of(loop.pid)
    seen: list[bool] = []
    with Pulser(target, watchdog=False):
        os.kill(loop.pid, signal.SIGSTOP)
        assert wait_until(target.is_stopped)
        with pytest.raises(SystemExit):
            pulser_module._on_termination(
                lambda *_: seen.append(_running_within(loop.pid, 0.5)),
                signal.SIGTERM,
                None,
            )
    assert seen == [True]


def test_a_harness_group_killed_outright_leaves_the_watchdog_to_continue(
    loop: subprocess.Popen[bytes],
) -> None:
    harness, watchdog = _pulsing_harness(loop.pid)
    os.killpg(harness.pid, signal.SIGKILL)
    harness.wait(timeout=10)
    # The harness's pipe closed: the watchdog continues the target at once.
    assert _running_within(loop.pid, 1.0)


def test_a_dead_watchdog_is_replaced_before_the_next_stop(
    loop: subprocess.Popen[bytes],
) -> None:
    with Pulser(Target.of(loop.pid), max_pulse_seconds=0.5) as pulser:
        first = pulser.watchdog_pid
        assert first is not None
        os.kill(first, signal.SIGKILL)
        assert wait_until(
            lambda: pulser._watchdog is not None and pulser._watchdog.poll() is not None
        )
        pulser.pulse(0.1)
        second = pulser.watchdog_pid
        assert second not in (None, first)
        assert psutil.Process(second).status() != psutil.STATUS_ZOMBIE


def test_no_stop_without_a_watchdog_that_said_it_was_ready(
    loop: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.qualification import pulser as pulser_module

    monkeypatch.setattr(pulser_module.sys, "executable", "/usr/bin/false")
    with pytest.raises(PulseRefused, match="never said it was ready"):
        Pulser(Target.of(loop.pid))
    assert psutil.Process(loop.pid).status() != psutil.STATUS_STOPPED


def test_the_watchdog_continues_a_stop_held_past_the_longest_pulse(
    loop: subprocess.Popen[bytes],
) -> None:
    # The harness is alive but holds the stop far too long: the watchdog
    # continues the target 1 s after the longest pulse (0.3 s).
    target = Target.of(loop.pid)
    continued: list[bool] = []
    with Pulser(target, max_pulse_seconds=0.3) as pulser:
        pulser.pulse(
            0.3, during=lambda: continued.append(_running_within(loop.pid, 3.0))
        )
    assert continued == [True]


def test_close_and_exit_continue_a_target_stopped_outside_a_pulse(
    loop: subprocess.Popen[bytes],
) -> None:
    target = Target.of(loop.pid)
    pulser = Pulser(target, watchdog=False)
    os.kill(loop.pid, signal.SIGSTOP)
    assert wait_until(target.is_stopped)
    pulser.close()
    assert not target.is_stopped()
    # And at exit, through atexit, without a watchdog to fall back on.
    script = textwrap.dedent(
        f"""
        import os, signal
        from examples.qualification.pulser import Pulser, Target
        pulser = Pulser(Target.of({loop.pid}), watchdog=False)
        os.kill({loop.pid}, signal.SIGSTOP)
        """
    )
    subprocess.run([sys.executable, "-c", script], env=_environment(), check=True)
    assert _running_within(loop.pid, 1.0)


def test_a_recycled_pid_is_never_continued_either(
    loop: subprocess.Popen[bytes],
) -> None:
    # Another process now holds the pid: closing must not SIGCONT it.
    impostor = Target(loop.pid, Target.of(loop.pid).start_time - 10)
    pulser = Pulser(impostor, watchdog=False)
    os.kill(loop.pid, signal.SIGSTOP)
    assert wait_until(Target.of(loop.pid).is_stopped)
    pulser.close()
    time.sleep(0.2)
    assert psutil.Process(loop.pid).status() == psutil.STATUS_STOPPED


def test_a_pulse_past_its_cap_is_refused(loop: subprocess.Popen[bytes]) -> None:
    with Pulser(Target.of(loop.pid), watchdog=False, max_pulse_seconds=0.2) as pulser:
        with pytest.raises(PulseRefused, match="at most 0.2 s"):
            pulser.pulse(0.3)
    assert psutil.Process(loop.pid).status() != psutil.STATUS_STOPPED


def test_a_stop_that_never_takes_is_refused() -> None:
    # A zombie can't be stopped: the pulse confirms the stop, so it refuses
    # rather than recording a pulse that never happened.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert wait_until(
        lambda: psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE
    )
    try:
        with Pulser(Target.of(child.pid), watchdog=False) as pulser:
            with pytest.raises(PulseRefused, match="did not stop"):
                pulser.pulse(0.1)
    finally:
        child.wait()


# ------------------------------------------------------------------ pulse timing


def test_a_wall_clock_step_does_not_stretch_a_pulse(
    loop: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    # NTP steps the wall clock back 10 s once the stop is confirmed: the
    # pulse still ends on time, and records its true length.
    from examples.qualification import pulser as pulser_module

    real = time.time_ns
    stepped = {"now": False}
    monkeypatch.setattr(
        pulser_module.time,
        "time_ns",
        lambda: real() - (10_000_000_000 if stepped["now"] else 0),
    )
    target = Target.of(loop.pid)
    with Pulser(target, watchdog=False, max_pulse_seconds=0.5) as pulser:
        began = time.monotonic()
        pulse = pulser.pulse(0.5, during=lambda: stepped.update(now=True))
        lasted = time.monotonic() - began
    assert lasted < 0.8
    assert 0.5e9 <= pulse.held_ns < 0.8e9
    assert 0.4e9 < pulse.continue_sent_ns - pulse.stopped_ns < 0.8e9


def test_a_slow_confirmation_does_not_lengthen_the_pulse(
    loop: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seeing state T took 0.9 s, the process stopped all along: the pulse
    # still ends 2 s after SIGSTOP, not 2 s after the confirmation.
    real_stopped = Target.is_stopped
    first: dict[str, float] = {}

    def slow_to_see(self: Target) -> bool:
        first.setdefault("at", time.monotonic())
        return real_stopped(self) and time.monotonic() - first["at"] > 0.9

    monkeypatch.setattr(Target, "is_stopped", slow_to_see)
    with Pulser(Target.of(loop.pid), watchdog=False) as pulser:
        began = time.monotonic()
        pulse = pulser.pulse(2.0)
        lasted = time.monotonic() - began
    assert lasted <= 2.1
    assert pulse.confirm_latency_ns > 0.8e9
    assert pulse.held_ns <= 2.1e9


def test_a_schedule_that_falls_behind_keeps_the_duty_cap(
    loop: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each stop takes 0.3 s to confirm; 0.2 s pulses every 0.4 s must still
    # leave the target running at least half the time, not back to back.
    real_stopped = Target.is_stopped
    since: dict[str, float] = {}

    def slow_to_see(self: Target) -> bool:
        since.setdefault("at", time.monotonic())
        seen = real_stopped(self) and time.monotonic() - since["at"] > 0.3
        if seen:
            since.clear()
        return seen

    monkeypatch.setattr(Target, "is_stopped", slow_to_see)
    with Pulser(Target.of(loop.pid), watchdog=False, max_pulse_seconds=0.2) as pulser:
        pulses = pulser.run(0.2, 0.4, 5)
    # Each pulse held 0.3 s, so each is followed by at least 0.3 s running.
    assert all(pulse.held_ns >= 0.29e9 for pulse in pulses)
    for this, following in zip(pulses, pulses[1:]):
        assert following.stop_sent_ns - this.continue_sent_ns >= 0.95 * this.held_ns
