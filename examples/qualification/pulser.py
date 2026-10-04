"""Stop and continue a serving process, safely (#221 A.3, F4a/F4b/F5/H0).

A pulse is ``SIGSTOP``, a confirmed stop, a wait, and ``SIGCONT``. The
target is named by its pid *and* its start time, checked before every
signal, so a recycled pid is never signalled. A stopped target is always
continued: by ``finally`` around each pulse, by ``atexit``, by
``Pulser.close``, by SIGTERM and SIGHUP handlers, and by a watchdog
(``examples.qualification.watchdog``) in a session of its own, which a
signal to the harness's process group never reaches. No pulse starts unless
the watchdog is alive and has said it is ready. Pulses are at most 2 s, at a
duty cycle of at most 50%.
"""

from __future__ import annotations

import atexit
import functools
import os
import select
import signal
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from typing import IO, Any, Callable

import psutil

from .fake_engine.process import _environment

MAX_PULSE_SECONDS = 2.0
MAX_DUTY_CYCLE = 0.5
CONFIRM_TIMEOUT_SECONDS = 1.0
CONFIRM_POLL_SECONDS = 0.0005
# The watchdog continues a target stopped this long past the longest pulse.
WATCHDOG_SLACK_SECONDS = 1.0
WATCHDOG_READY_SECONDS = 30.0


class PulseRefused(RuntimeError):
    """A pulse that would break a safety rule, or reach the wrong process."""


@dataclass(frozen=True)
class Target:
    """A process named by its pid and start time."""

    pid: int
    start_time: float
    role: str = ""

    @classmethod
    def of(cls, pid: int, role: str = "") -> Target:
        return cls(pid, psutil.Process(pid).create_time(), role)

    def is_alive(self) -> bool:
        """The same process is still running: its pid was not reused."""
        try:
            started: float = psutil.Process(self.pid).create_time()
        except psutil.Error:
            return False
        return started == self.start_time

    def is_stopped(self) -> bool:
        try:
            status = psutil.Process(self.pid).status()
        except psutil.Error:
            return False
        return status in (psutil.STATUS_STOPPED, psutil.STATUS_TRACING_STOP)

    def to_record(self) -> dict[str, Any]:
        return {"role": self.role, "pid": self.pid, "start_time": self.start_time}


@dataclass(frozen=True)
class Pulse:
    """One pulse's actual times, in ``time.time_ns()``."""

    stop_sent_ns: int
    stopped_ns: int
    continue_sent_ns: int

    @property
    def confirm_latency_ns(self) -> int:
        return self.stopped_ns - self.stop_sent_ns

    def to_record(self) -> dict[str, int]:
        return {
            "stop_sent_ns": self.stop_sent_ns,
            "stopped_ns": self.stopped_ns,
            "confirm_latency_ns": self.confirm_latency_ns,
            "continue_sent_ns": self.continue_sent_ns,
        }


# vLLM 0.30 titles its processes; a role is found by its title.
ROLE_TITLES = {
    "engine_core": "EngineCore",
    "worker_tp0": "Worker_TP0",
    "worker_tp1": "Worker_TP1",
}


def discover_roles(api_server_pid: int) -> dict[str, Target]:
    """The API server and the processes under it, by role: each named by
    its pid and start time, so a later signal reaches the same process."""
    server = psutil.Process(api_server_pid)
    roles = {"api_server": Target.of(api_server_pid, "api_server")}
    for child in server.children(recursive=True):
        try:
            title = " ".join([child.name(), *child.cmdline()])
        except psutil.Error:
            continue
        for role, marker in ROLE_TITLES.items():
            if marker in title and role not in roles:
                roles[role] = Target.of(child.pid, role)
    return roles


def check_schedule(pulse_seconds: float, period_seconds: float) -> None:
    """Refuse a pulse longer than 2 s, or a duty cycle above 50%."""
    if not 0 < pulse_seconds <= MAX_PULSE_SECONDS:
        raise PulseRefused(f"a pulse lasts at most {MAX_PULSE_SECONDS} s")
    if pulse_seconds > MAX_DUTY_CYCLE * period_seconds:
        raise PulseRefused(f"a duty cycle above {MAX_DUTY_CYCLE:.0%}")


class Pulser:
    """Pulses one target, and guarantees it is continued."""

    def __init__(
        self,
        target: Target,
        *,
        watchdog: bool = True,
        max_pulse_seconds: float = MAX_PULSE_SECONDS,
    ) -> None:
        self.target = target
        self.max_pulse_seconds = max_pulse_seconds
        self.pulses: list[Pulse] = []
        self._lock = threading.Lock()
        self._closed = False
        self._watched = watchdog
        self._watchdog: subprocess.Popen[bytes] | None = None
        if watchdog:
            self._watchdog = _start_watchdog(target, max_pulse_seconds)
        atexit.register(self.close)
        _LIVE.add(self)
        _handle_termination()

    @property
    def watchdog_pid(self) -> int | None:
        return None if self._watchdog is None else self._watchdog.pid

    def pulse(
        self, seconds: float, *, during: Callable[[], None] | None = None
    ) -> Pulse:
        """Stop the target for ``seconds``; ``during`` runs while it is
        stopped. The target is continued however the pulse ends.

        Raises:
            PulseRefused: for a pulse too long, a closed pulser, a watchdog
                that can't be (re)started, a target that is no longer the
                same process, or a stop that was not confirmed within 1 s.
        """
        if not 0 < seconds <= self.max_pulse_seconds:
            raise PulseRefused(f"a pulse lasts at most {self.max_pulse_seconds} s")
        with self._lock:
            if self._closed:
                raise PulseRefused("the pulser is closed")
            self._ensure_watchdog()
            self._signal(signal.SIGSTOP)
            stop_sent = time.time_ns()
            try:
                stopped = self._confirm_stopped(stop_sent)
                if during is not None:
                    during()
                time.sleep(max(0.0, seconds - (time.time_ns() - stopped) / 1e9))
            finally:
                self._continue()
            pulse = Pulse(stop_sent, stopped, time.time_ns())
        self.pulses.append(pulse)
        return pulse

    def run(
        self,
        pulse_seconds: float,
        period_seconds: float,
        count: int,
        *,
        stop: threading.Event | None = None,
    ) -> list[Pulse]:
        """``count`` pulses, one per period, until ``stop`` is set."""
        check_schedule(pulse_seconds, period_seconds)
        done: list[Pulse] = []
        start = time.monotonic()
        for index in range(count):
            wait = start + index * period_seconds - time.monotonic()
            if stop is not None and stop.wait(max(0.0, wait)):
                break
            if stop is None:
                time.sleep(max(0.0, wait))
            done.append(self.pulse(pulse_seconds))
        return done

    def close(self) -> None:
        """Continue the target, and let the watchdog go: closing its pipe
        tells it the harness is done, and it continues the target once more
        and exits."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._continue()
        if self._watchdog is not None:
            _release(self._watchdog)
        atexit.unregister(self.close)
        _LIVE.discard(self)

    def continue_now(self) -> None:
        """Continue the target at once, without waiting for a pulse to end:
        for a signal handler, which may interrupt a pulse in progress."""
        self._continue()

    def _ensure_watchdog(self) -> None:
        """A watchdog that died (the OOM killer, a user) is replaced before
        the next stop; one that can't be is a refusal."""
        if not self._watched:
            return
        if self._watchdog is not None and self._watchdog.poll() is None:
            return
        self._watchdog = _start_watchdog(self.target, self.max_pulse_seconds)

    def __enter__(self) -> Pulser:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _signal(self, signum: signal.Signals) -> None:
        if not self.target.is_alive():
            raise PulseRefused(f"pid {self.target.pid} is no longer the target")
        os.kill(self.target.pid, signum)

    def _continue(self) -> None:
        if self.target.is_alive():
            os.kill(self.target.pid, signal.SIGCONT)

    def _confirm_stopped(self, stop_sent: int) -> int:
        deadline = stop_sent + int(CONFIRM_TIMEOUT_SECONDS * 1e9)
        while time.time_ns() < deadline:
            if self.target.is_stopped():
                return time.time_ns()
            time.sleep(CONFIRM_POLL_SECONDS)
        raise PulseRefused(f"pid {self.target.pid} did not stop within 1 s")


def _start_watchdog(
    target: Target, max_pulse_seconds: float
) -> subprocess.Popen[bytes]:
    """A watchdog in its own session, watching ``target``; returns once it
    says it is ready.

    Raises:
        PulseRefused: if it isn't ready within 30 s.
    """
    command = [
        sys.executable, "-m", "examples.qualification.watchdog",
        "--pid", str(target.pid),
        "--start-time", repr(target.start_time),
        "--limit", str(max_pulse_seconds + WATCHDOG_SLACK_SECONDS),
    ]  # fmt: skip
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=_environment(),
        start_new_session=True,
    )
    assert process.stdout is not None
    if _read_line(process.stdout, WATCHDOG_READY_SECONDS) != b"ready":
        process.kill()
        process.wait()
        raise PulseRefused("the pulse watchdog never said it was ready")
    return process


def _read_line(stream: IO[bytes], timeout_seconds: float) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    line = b""
    while not line.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            break
        chunk = os.read(stream.fileno(), 1)
        if not chunk:
            break
        line += chunk
    return line.strip()


def _release(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stdout is not None:
        process.stdout.close()


# Every open pulser, for the termination handlers.
_LIVE: weakref.WeakSet[Pulser] = weakref.WeakSet()
_HANDLED: set[int] = set()


def _handle_termination() -> None:
    """On SIGTERM or SIGHUP, continue every target, then exit: their default
    action would end the harness with no ``finally`` and no ``atexit``.
    Handlers can be set only from the main thread."""
    if threading.current_thread() is not threading.main_thread():
        return
    for signum in (signal.SIGTERM, signal.SIGHUP):
        if signum not in _HANDLED:
            previous = signal.getsignal(signum)
            signal.signal(signum, functools.partial(_on_termination, previous))
            _HANDLED.add(signum)


def _on_termination(previous: Any, signum: int, frame: Any) -> None:
    for pulser in list(_LIVE):
        pulser.continue_now()
    if callable(previous):
        previous(signum, frame)
    raise SystemExit(128 + signum)


__all__ = [
    "MAX_DUTY_CYCLE",
    "MAX_PULSE_SECONDS",
    "Pulse",
    "PulseRefused",
    "Pulser",
    "Target",
    "ROLE_TITLES",
    "check_schedule",
    "discover_roles",
]
