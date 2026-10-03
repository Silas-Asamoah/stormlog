"""Stop and continue a serving process, safely (#221 A.3, F4a/F4b/F5/H0).

A pulse is ``SIGSTOP``, a confirmed stop, a wait, and ``SIGCONT``. The
target is named by its pid *and* its start time, checked before every
signal, so a recycled pid is never signalled. A stopped target is always
continued: by ``finally`` around each pulse, by ``atexit``, by
``Pulser.close``, and by a watchdog process that outlives a harness killed
outright. Pulses are at most 2 s, at a duty cycle of at most 50%.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import psutil

MAX_PULSE_SECONDS = 2.0
MAX_DUTY_CYCLE = 0.5
CONFIRM_TIMEOUT_SECONDS = 1.0
CONFIRM_POLL_SECONDS = 0.0005
# The watchdog continues a target stopped this long past the longest pulse.
WATCHDOG_SLACK_SECONDS = 1.0


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
        self._watchdog: multiprocessing.process.BaseProcess | None = None
        if watchdog:
            self._watchdog = _start_watchdog(target, max_pulse_seconds)
        atexit.register(self.close)

    def pulse(
        self, seconds: float, *, during: Callable[[], None] | None = None
    ) -> Pulse:
        """Stop the target for ``seconds``; ``during`` runs while it is
        stopped. The target is continued however the pulse ends.

        Raises:
            PulseRefused: for a pulse too long, a closed pulser, a target
                that is no longer the same process, or a stop that was not
                confirmed within 1 s.
        """
        if not 0 < seconds <= self.max_pulse_seconds:
            raise PulseRefused(f"a pulse lasts at most {self.max_pulse_seconds} s")
        with self._lock:
            if self._closed:
                raise PulseRefused("the pulser is closed")
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
        """Continue the target, and stop the watchdog."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._continue()
        if self._watchdog is not None:
            self._watchdog.terminate()
            self._watchdog.join(timeout=5)
        atexit.unregister(self.close)

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
) -> multiprocessing.process.BaseProcess:
    process = multiprocessing.get_context("spawn").Process(
        target=watchdog,
        args=(os.getpid(), target.pid, target.start_time, max_pulse_seconds),
        name="stormlog-pulse-watchdog",
        daemon=True,
    )
    process.start()
    return process


def watchdog(
    harness_pid: int, pid: int, start_time: float, max_pulse_seconds: float
) -> None:
    """Continue the target if the harness dies, or if the target stays
    stopped well past the longest pulse; end once either process is gone."""
    target = Target(pid, start_time)
    stopped_since: float | None = None
    limit = max_pulse_seconds + WATCHDOG_SLACK_SECONDS
    while target.is_alive():
        if not psutil.pid_exists(harness_pid):
            os.kill(pid, signal.SIGCONT)
            return
        if target.is_stopped():
            stopped_since = stopped_since or time.monotonic()
            if time.monotonic() - stopped_since > limit:
                os.kill(pid, signal.SIGCONT)
                stopped_since = None
        else:
            stopped_since = None
        time.sleep(0.05)


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
    "watchdog",
]
