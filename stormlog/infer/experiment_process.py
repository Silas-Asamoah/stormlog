"""Starting, watching and stopping the processes an experiment runs.

Every process the runner starts gets its own session and process group, so
the runner owns everything it starts, children included: vLLM's engine and
workers, and the ``pip`` and ``nvidia-smi`` its ``/server_info`` collector
runs. Stopping a process signals its whole group, and the runner then
checks that nothing it started is left: no process in the group or the
session, and none of the processes it remembered by PID and start time,
which also finds one that left the group with ``setsid``. Every launch also
carries a mark in its environment (``STORMLOG_RUN_MARK``), which every
descendant inherits unless it execs with a fresh environment, so a process
forked after the tree was remembered and then moved to a session of its own
is found too. An environment that cannot be read cannot be told unmarked;
the cleanup record counts them.

On Linux the checks read ``/proc`` (``server_process``). Elsewhere they use
``psutil`` and say so, since the session of another process cannot be read
there.
"""

from __future__ import annotations

import os
import platform
import secrets
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import psutil

from .server_process import (
    HELPER_ROLES,
    PROC,
    SERVER_ROLES,
    group_members,
    process_tree,
    read_process,
    still_running,
)

KILL_WAIT_SECONDS = 10.0
# Inherited by everything a launch starts; a fresh value per launch.
MARK_VARIABLE = "STORMLOG_RUN_MARK"
POLL_SECONDS = 0.1
EXPECTED_ROLES = frozenset(SERVER_ROLES) | frozenset(HELPER_ROLES)


@dataclass
class Launched:
    """A process the runner started, in a session of its own."""

    name: str
    process: subprocess.Popen[bytes]
    command: tuple[str, ...]
    started_at_ns: int
    log_path: Path | None
    mark: str = ""
    affinity: str | None = None
    affinity_applied: bool | None = None
    ended_at_ns: int | None = None
    exit_code: int | None = None
    stopped_by: str | None = None
    _log: IO[bytes] | None = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        code = self.process.poll()
        if code is not None and self.ended_at_ns is None:
            self.ended_at_ns = time.time_ns()
            self.exit_code = code
            self._close_log()
        return code

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pid": self.pid,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "exit_code": self.exit_code,
            "stopped_by": self.stopped_by,
            "affinity": self.affinity,
            "affinity_applied": self.affinity_applied,
            "log": None if self.log_path is None else str(self.log_path),
        }


def parse_cpu_list(text: str) -> set[int]:
    """``0-3,8`` as {0, 1, 2, 3, 8}; ValueError for anything else."""
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        start, end = int(low), int(high or low)
        if start < 0 or end < start:
            raise ValueError(f"CPU range {part!r} is not valid")
        cpus.update(range(start, end + 1))
    if not cpus:
        raise ValueError("a CPU list must name at least one CPU")
    return cpus


def launch(
    name: str,
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cpu_affinity: str | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> Launched:
    """Start a command in a session of its own, pinned to its CPUs if asked."""
    cpus = parse_cpu_list(cpu_affinity) if cpu_affinity else None
    log = log_path.open("ab") if log_path is not None else None
    mark = secrets.token_hex(16)
    process = subprocess.Popen(
        list(command),
        env={**os.environ, **(env or {}), MARK_VARIABLE: mark},
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=log if log is not None else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log is not None else subprocess.DEVNULL,
        start_new_session=True,
        preexec_fn=_pin(cpus) if cpus else None,
    )
    launched = Launched(
        name=name,
        process=process,
        command=tuple(command),
        started_at_ns=time.time_ns(),
        log_path=log_path,
        mark=mark,
        affinity=cpu_affinity,
        _log=log,
    )
    if cpus:
        launched.affinity_applied = affinity_matches(process.pid, cpus)
    return launched


def _pin(cpus: set[int]) -> Any:
    def pin() -> None:
        setter = getattr(os, "sched_setaffinity", None)
        if setter is not None:
            setter(0, cpus)

    return pin


def affinity_matches(pid: int, cpus: set[int]) -> bool | None:
    """Whether the process may run on exactly these CPUs; None where unknown."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return set(getter(pid)) == cpus
    except OSError:
        return None


def stop(
    launched: Launched,
    *,
    signals: Sequence[int] = (int(signal.SIGTERM),),
    timeout_s: float = 30.0,
) -> int | None:
    """Signal the process's whole group, escalating, and wait for it to end.

    Each signal in turn gets ``timeout_s``; SIGKILL follows the last. Returns
    the exit code, or None if even SIGKILL left it running.
    """
    for signum in (*signals, int(signal.SIGKILL)):
        if launched.poll() is not None:
            break
        _signal_group(launched.pid, signum)
        launched.stopped_by = signal.Signals(signum).name
        wait = KILL_WAIT_SECONDS if signum == signal.SIGKILL else timeout_s
        _wait(launched, wait)
    return launched.poll()


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        return


def _wait(launched: Launched, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and launched.poll() is None:
        time.sleep(POLL_SECONDS)


@dataclass(frozen=True)
class Cleanup:
    """What is left of a stopped process's group, session and remembered tree."""

    verified: bool
    method: str
    survivors: tuple[dict[str, Any], ...] = ()
    killed: tuple[int, ...] = ()
    # Environments the mark search could not read; None when there was no
    # mark to search for. The search is complete only when it read them all.
    unreadable: int | None = None

    def to_record(self) -> dict[str, Any]:
        search = None
        if self.unreadable is not None:
            search = {"complete": self.unreadable == 0, "unreadable": self.unreadable}
        return {
            "verified": self.verified,
            "method": self.method,
            "survivors": list(self.survivors),
            "killed": list(self.killed),
            "mark_search": search,
        }


def verify_cleanup(
    pgid: int,
    remembered: Iterable[tuple[int, int]] = (),
    *,
    wait_s: float = KILL_WAIT_SECONDS,
    proc: Path = PROC,
    mark: str | None = None,
) -> Cleanup:
    """Wait until nothing of a group, its session or remembered tree runs.

    ``mark`` is the launch's environment mark: a process that carries it
    is the launch's, wherever it went. Survivors are killed by PID once;
    whatever outlives that and ``wait_s`` is listed, and the cleanup is not
    verified. A process whose environment cannot be read cannot be told
    unmarked, so the record says how many there were.
    """
    keys = list(remembered)
    method = "proc" if _linux() else "psutil"
    deadline = time.monotonic() + wait_s
    killed: tuple[int, ...] = ()
    while True:
        marked, unreadable = _marked(mark, proc, method)
        survivors = _survivors(pgid, keys, proc, method) | marked
        if not survivors:
            return Cleanup(True, method, killed=killed, unreadable=unreadable)
        if not killed:
            killed = tuple(sorted(survivors))
            for pid in killed:
                _kill(pid)
        if time.monotonic() >= deadline:
            left = tuple(identify(pid, proc=proc) for pid in sorted(survivors))
            return Cleanup(False, method, left, killed, unreadable)
        time.sleep(POLL_SECONDS)


def identify(pid: int, *, proc: Path = PROC) -> dict[str, Any]:
    """A process by PID and start time, so a later check can tell it from
    another process given the same PID."""
    if _linux():
        info = read_process(pid, proc)
        return {"pid": pid, "start_ticks": None if info is None else info.start_ticks}
    try:
        return {"pid": pid, "create_time": psutil.Process(pid).create_time()}
    except psutil.Error:
        return {"pid": pid, "create_time": None}


def still_there(survivor: Mapping[str, Any], *, proc: Path = PROC) -> bool:
    """Whether a recorded survivor is still the same live process.

    One recorded without its start time cannot be told from a process that
    reused its PID, so it does not count.
    """
    pid = survivor.get("pid")
    if not isinstance(pid, int) or not _alive(pid):
        return False
    starts = {k: v for k, v in survivor.items() if k != "pid" and v is not None}
    now = identify(pid, proc=proc)
    return bool(starts) and all(now.get(k) == v for k, v in starts.items())


def _survivors(
    pgid: int, keys: list[tuple[int, int]], proc: Path, method: str
) -> set[int]:
    if method == "proc":
        found = {info.pid for info in group_members(pgid, pgid, proc)}
        found |= {info.pid for info in still_running(keys, proc)}
        return found
    return _psutil_group(pgid) | {pid for pid, _ in keys if _alive(pid)}


def _marked(mark: str | None, proc: Path, method: str) -> tuple[set[int], int | None]:
    """Live processes whose environment carries the launch's mark, and how
    many environments could not be read (None without a mark)."""
    if not mark:
        return set(), None
    needle = f"{MARK_VARIABLE}={mark}"
    if method == "proc":
        return _proc_marked(needle.encode() + b"\0", proc)
    return _psutil_marked(mark)


def _proc_marked(needle: bytes, proc: Path) -> tuple[set[int], int]:
    found, unreadable = set(), 0
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except PermissionError:
            unreadable += 1
            continue
        except OSError:
            continue  # gone
        if (b"\0" + environ).find(b"\0" + needle) >= 0 and _alive(int(entry.name)):
            found.add(int(entry.name))
    return found, unreadable


def _psutil_marked(mark: str) -> tuple[set[int], int]:
    found, unreadable = set(), 0
    for process in psutil.process_iter():
        try:
            environ = process.environ()
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        # psutil on macOS may raise SystemError for a process it cannot read.
        except (psutil.Error, OSError, SystemError):
            unreadable += 1
            continue
        if environ.get(MARK_VARIABLE) == mark and _alive(process.pid):
            found.add(process.pid)
    return found, unreadable


def _psutil_group(pgid: int) -> set[int]:
    found = set()
    for process in psutil.process_iter():
        try:
            if (
                os.getpgid(process.pid) == pgid
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                found.add(process.pid)
        except (OSError, psutil.Error):
            continue
    return found


def _alive(pid: int) -> bool:
    try:
        return bool(psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
    except psutil.Error:
        return False


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def unexpected_roles(pid: int, *, proc: Path = PROC) -> list[dict[str, Any]] | None:
    """Processes in the server's tree that are not vLLM's own; None off Linux.

    A ``pip`` or ``nvidia-smi`` left by ``/server_info``'s collector, or a
    shell, would compete with the measurement.
    """
    if not _linux():
        return None
    return [
        {"pid": info.pid, "role": info.role, "comm": info.comm}
        for info in process_tree(pid, proc)
        if info.role not in EXPECTED_ROLES
    ]


def process_key(pid: int, *, proc: Path = PROC) -> tuple[int, int] | None:
    """A process's PID and start ticks, its name for its lifetime; None off Linux."""
    if not _linux():
        return None
    info = read_process(pid, proc)
    return None if info is None else info.key


def remembered_tree(pid: int, *, proc: Path = PROC) -> list[tuple[int, int]]:
    """The server's tree by PID and start ticks, to check after stopping it."""
    if not _linux():
        return []
    return [info.key for info in process_tree(pid, proc)]


def _linux() -> bool:
    """Whether ``/proc`` is there to read; decided at run time, not by the type checker."""
    return platform.system() == "Linux"


def wait_for_file(
    path: Path, timeout_s: float, launched: Launched | None = None
) -> bool:
    """Whether the file appears in time, while its writer is still running."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        if launched is not None and launched.poll() is not None:
            return path.exists()
        time.sleep(POLL_SECONDS)
    return path.exists()


def run_step(
    name: str,
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cpu_affinity: str | None = None,
    timeout_s: float,
    log_path: Path | None = None,
) -> tuple[Launched, bool]:
    """Run a command to its end, or stop its group at the timeout.

    Returns the launched process and whether it timed out.
    """
    launched = launch(
        name, command, env=env, cpu_affinity=cpu_affinity, log_path=log_path
    )
    try:
        launched.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        stop(launched, timeout_s=KILL_WAIT_SECONDS)
        return launched, True
    launched.poll()
    return launched, False


__all__ = [
    "EXPECTED_ROLES",
    "Cleanup",
    "Launched",
    "affinity_matches",
    "identify",
    "launch",
    "parse_cpu_list",
    "remembered_tree",
    "run_step",
    "still_there",
    "stop",
    "unexpected_roles",
    "verify_cleanup",
    "wait_for_file",
]
