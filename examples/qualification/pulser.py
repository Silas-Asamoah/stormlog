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
import re
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
# How often a held stop checks that its watchdog still watches.
WATCHDOG_CHECK_SECONDS = 0.01


class PulseRefused(RuntimeError):
    """A pulse that would break a safety rule, or reach the wrong process."""


class TargetGone(PulseRefused):
    """The target exited, or its pid now names another process. Its message
    starts ``target_gone``, so a truth that records it says what happened,
    not that the watchdog stopped watching (which it does once its target
    is gone)."""


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
        return process_stopped(self.pid)

    def to_record(self) -> dict[str, Any]:
        return {"role": self.role, "pid": self.pid, "start_time": self.start_time}


@dataclass(frozen=True)
class Pulse:
    """One pulse as it happened. Lengths are measured on the monotonic
    clock; the wall-clock times are one wall reading at ``SIGSTOP`` plus
    those lengths, so a clock step can't bend them. ``held_ns`` runs from
    ``SIGSTOP`` sent to ``SIGCONT`` sent: the most the target was stopped.
    ``continued_by_other`` says the target ran while the pulse held it:
    already running when the pulser came to continue it (the watchdog's
    limit, or an operator), or seen running at one of the hold's checks
    and stopped again by someone else. The stop was then shorter than
    ``held_ns`` by an unknown amount."""

    stop_sent_ns: int
    stopped_ns: int
    continue_sent_ns: int
    continued_by_other: bool = False

    @classmethod
    def measured(
        cls,
        wall_ns: int,
        sent: int,
        stopped: int,
        continued: int,
        *,
        continued_by_other: bool = False,
    ) -> Pulse:
        """From the wall time at ``SIGSTOP`` and monotonic times."""
        return cls(
            wall_ns,
            wall_ns + stopped - sent,
            wall_ns + continued - sent,
            continued_by_other,
        )

    @property
    def confirm_latency_ns(self) -> int:
        return self.stopped_ns - self.stop_sent_ns

    @property
    def held_ns(self) -> int:
        return self.continue_sent_ns - self.stop_sent_ns

    def to_record(self) -> dict[str, int]:
        return {
            "stop_sent_ns": self.stop_sent_ns,
            "stopped_ns": self.stopped_ns,
            "confirm_latency_ns": self.confirm_latency_ns,
            "continue_sent_ns": self.continue_sent_ns,
            "held_ns": self.held_ns,
            "continued_by_other": self.continued_by_other,
            "completed": True,
        }


def seen_running(pid: int) -> bool:
    """Whether a status read succeeded and showed the process not stopped: a
    read that fails says nothing, so a hold's many checks never flag a
    pulse on a transient error."""
    try:
        status = psutil.Process(pid).status()
    except psutil.Error:
        return False
    return status not in (psutil.STATUS_STOPPED, psutil.STATUS_TRACING_STOP)


def process_stopped(pid: int) -> bool:
    """Whether the process is in the stopped state now."""
    try:
        status = psutil.Process(pid).status()
    except psutil.Error:
        return False
    return status in (psutil.STATUS_STOPPED, psutil.STATUS_TRACING_STOP)


# vLLM 0.30 retitles its processes (setproctitle replaces argv): EngineCore
# under the API server, and each TP worker under EngineCore.
ENGINE_TITLE = re.compile(r"^VLLM::EngineCore(?:_DP\d+)?$")
WORKER_TITLE = re.compile(r"^VLLM::Worker_TP(\d+)$")


def discover_roles(api_server_pid: int) -> dict[str, Target]:
    """The API server, its EngineCore and EngineCore's TP workers, by role,
    each named by its pid and start time so a later signal reaches the same
    process. Titles must match exactly: a helper whose arguments mention
    EngineCore is not it, and Worker_TP10 is rank 10, not rank 1.

    Raises:
        ValueError: when not exactly one child carries EngineCore's title,
            or two workers claim the same rank.
    """
    roles = {"api_server": Target.of(api_server_pid, "api_server")}
    engines = _titled(psutil.Process(api_server_pid).children(), ENGINE_TITLE)
    if len(engines) != 1:
        raise ValueError(
            f"expected one VLLM::EngineCore under pid {api_server_pid},"
            f" found {len(engines)}"
        )
    engine, _match = engines[0]
    roles["engine_core"] = Target.of(engine.pid, "engine_core")
    for worker, match in _titled(engine.children(), WORKER_TITLE):
        role = f"worker_tp{int(match.group(1))}"
        if role in roles:
            raise ValueError(f"two processes are titled {match.group(0)}")
        roles[role] = Target.of(worker.pid, role)
    return roles


def _titled(
    processes: list[psutil.Process], pattern: re.Pattern[str]
) -> list[tuple[psutil.Process, re.Match[str]]]:
    found = []
    for process in processes:
        match = pattern.match(_title(process))
        if match is not None:
            found.append((process, match))
    return found


def _title(process: psutil.Process) -> str:
    """A retitled process's title is its argv[0]; the 15-character comm
    would truncate it."""
    try:
        arguments: list[str] = process.cmdline()
    except psutil.Error:
        return ""
    return str(arguments[0]).strip() if arguments else ""


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
        # Never longer than the design's cap, whoever asks.
        self.max_pulse_seconds = min(max_pulse_seconds, MAX_PULSE_SECONDS)
        self.pulses: list[Pulse] = []
        # The last pulse a failure or an interruption cut short: its stop was
        # sent, so the target went through it, but it never completed.
        self.cut_short: dict[str, Any] | None = None
        # Whether the target was seen running during the current pulse's hold.
        self._ran_meanwhile = False
        self._lock = threading.Lock()
        self._closed = False
        self._watched = watchdog
        self._watchdog: subprocess.Popen[bytes] | None = None
        if watchdog:
            self._watchdog = _start_watchdog(target, self.max_pulse_seconds)
        atexit.register(self.close)
        _LIVE.add(self)
        handle_termination()

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
                that can't be (re)started or stops watching during the stop
                (the target is continued at once), a target that is no
                longer the same process, or a stop that was not confirmed
                within 1 s.
        """
        if not 0 < seconds <= self.max_pulse_seconds:
            raise PulseRefused(f"a pulse lasts at most {self.max_pulse_seconds} s")
        with self._lock:
            if self._closed:
                raise PulseRefused("the pulser is closed")
            self._ensure_watchdog()
            self._ran_meanwhile = False
            self._signal(signal.SIGSTOP)
            wall, sent = time.time_ns(), time.monotonic_ns()
            stopped = None
            try:
                stopped = self._confirm_stopped(sent)
                if during is not None:
                    during()
                # Ends ``seconds`` after SIGSTOP, however long it took to see
                # the stop, and whatever the wall clock does; at once if the
                # watchdog stops watching, which would leave a harness killed
                # in this stop nobody to continue its target.
                self._hold_until(sent + int(seconds * 1e9))
            except BaseException:
                continued_sent = self._continue()
                self.cut_short = _cut_short(
                    wall, sent, stopped, time.monotonic_ns() if continued_sent else None
                )
                raise
            finally:
                running = self._ran_meanwhile or not process_stopped(self.target.pid)
                self._continue()
                continued = time.monotonic_ns()
            pulse = Pulse.measured(
                wall, sent, stopped, continued, continued_by_other=running
            )
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
        """``count`` pulses, one per period, until ``stop`` is set. A pulse
        never follows the last one sooner than the period allows, nor sooner
        than the target was actually stopped, so a schedule that falls
        behind keeps the duty cap instead of catching up back to back."""
        check_schedule(pulse_seconds, period_seconds)
        done: list[Pulse] = []
        period_ns = int(period_seconds * 1e9)
        rest_ns = period_ns - int(pulse_seconds * 1e9)
        start = not_before = time.monotonic_ns()
        for index in range(count):
            due = max(start + index * period_ns, not_before)
            if _wait_until(due, stop):
                break
            pulse = self.pulse(pulse_seconds)
            done.append(pulse)
            not_before = time.monotonic_ns() + max(rest_ns, pulse.held_ns)
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

    def _hold_until(self, monotonic_ns: int) -> None:
        """Hold until ``monotonic_ns``, checking every
        ``WATCHDOG_CHECK_SECONDS`` that the target is still there and still
        stopped (one running at a check was continued by someone else, even
        if stopped again since) and that the watchdog still watches."""
        while True:
            if not self.target.is_alive():
                raise TargetGone(
                    f"target_gone: pid {self.target.pid} exited during the stop"
                )
            if seen_running(self.target.pid):
                self._ran_meanwhile = True
            if not self._watchdog_watching():
                raise PulseRefused("the watchdog stopped watching during the stop")
            left = monotonic_ns - time.monotonic_ns()
            if left <= 0:
                return
            time.sleep(min(WATCHDOG_CHECK_SECONDS, left / 1e9))

    def _watchdog_watching(self) -> bool:
        """The watchdog is alive and not itself stopped (or none was asked
        for)."""
        if not self._watched:
            return True
        process = self._watchdog
        if process is None or process.poll() is not None:
            return False
        try:
            status = psutil.Process(process.pid).status()
        except psutil.Error:
            return False
        return status not in (psutil.STATUS_STOPPED, psutil.STATUS_ZOMBIE)

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
            raise TargetGone(
                f"target_gone: pid {self.target.pid} is no longer the target"
            )
        os.kill(self.target.pid, signum)

    def _continue(self) -> bool:
        """Whether SIGCONT was sent: never to a target that is gone."""
        if not self.target.is_alive():
            return False
        os.kill(self.target.pid, signal.SIGCONT)
        return True

    def _confirm_stopped(self, stop_sent: int) -> int:
        """When the stop was seen, on the monotonic clock."""
        deadline = stop_sent + int(CONFIRM_TIMEOUT_SECONDS * 1e9)
        while time.monotonic_ns() < deadline:
            if self.target.is_stopped():
                return time.monotonic_ns()
            time.sleep(CONFIRM_POLL_SECONDS)
        raise PulseRefused(f"pid {self.target.pid} did not stop within 1 s")


def _cut_short(
    wall_ns: int, sent: int, stopped: int | None, continued: int | None
) -> dict[str, Any]:
    """A pulse that never completed, from the wall time at ``SIGSTOP`` and
    monotonic times; a time not reached (no stop confirmed, no ``SIGCONT``
    sent to a target that was gone) is None."""

    def wall(at: int | None) -> int | None:
        return None if at is None else wall_ns + at - sent

    return {
        "stop_sent_ns": wall_ns,
        "stopped_ns": wall(stopped),
        "continue_sent_ns": wall(continued),
        "completed": False,
    }


def _wait_until(monotonic_ns: int, stop: threading.Event | None) -> bool:
    """Wait for ``monotonic_ns``; whether ``stop`` was set meanwhile."""
    delay = max(0.0, (monotonic_ns - time.monotonic_ns()) / 1e9)
    if stop is None:
        time.sleep(delay)
        return False
    return stop.wait(delay)


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


def handle_termination() -> None:
    """On SIGTERM or SIGHUP, continue every target, then exit
    (``SystemExit``): their default action would end the harness with no
    ``finally`` and no ``atexit``. A run installs them as it starts, before
    any pulser exists, so that it is published however it ends. Handlers
    can be set only from the main thread."""
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
    "ENGINE_TITLE",
    "WORKER_TITLE",
    "check_schedule",
    "handle_termination",
    "discover_roles",
]
