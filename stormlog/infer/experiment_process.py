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
is found too. A process whose environment cannot be read, or was emptied,
and that may be the launch's keeps the cleanup from verifying.

On Linux the checks read ``/proc`` (``server_process``). Elsewhere they use
``psutil`` and say so, since the session of another process cannot be read
there.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import re
import secrets
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import psutil

from .host_clock import host_boot_id
from .server_process import (
    HELPER_ROLES,
    PROC,
    SERVER_ROLES,
    group_members,
    listening_ports,
    process_tree,
    read_process,
    still_running,
)

KILL_WAIT_SECONDS = 10.0
# Inherited by everything a launch starts; a fresh value per launch, a
# random nonce of MARK_BYTES bytes, in hex.
MARK_VARIABLE = "STORMLOG_RUN_MARK"
MARK_BYTES = 16
# A journaled mark ties a process to its launch only as a nonce of at
# least 64 bits.
_NONCE = re.compile(r"[0-9a-f]{16,}")
POLL_SECONDS = 0.1
# A process started this long before a launch is not the launch's. Start
# times are compared in the processes' own clock (ticks since boot, or
# psutil's creation time), never against the wall clock.
START_SLACK_SECONDS = 2.0
EXPECTED_ROLES = frozenset(SERVER_ROLES) | frozenset(HELPER_ROLES)
# What ``identify`` records as a process's start: ticks since boot from
# /proc, or psutil's creation time.
_STARTS = ("start_ticks", "create_time")


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
    # Its PID and start time, read as it started (``identify``).
    identity: dict[str, Any] = field(default_factory=dict)
    # The monotonic clock at its start and at its end, for how long it ran.
    started_monotonic: float = field(default_factory=time.monotonic)
    # Where it was journaled, if it was.
    journal: Path | None = None
    ended_monotonic: float | None = None
    _log: IO[bytes] | None = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        code = self.process.poll()
        if code is not None and self.ended_at_ns is None:
            self.ended_at_ns = time.time_ns()
            self.ended_monotonic = time.monotonic()
            self.exit_code = code
            self._close_log()
        return code

    def lasted_s(self) -> float | None:
        """How long its leader ran; None while it runs."""
        if self.ended_monotonic is None:
            return None
        return self.ended_monotonic - self.started_monotonic

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
    journal: Path | None = None,
) -> Launched:
    """Start a command in a session of its own, pinned to its CPUs if asked.

    ``journal`` gets a line naming the launch (PID, group, start time and
    mark) as soon as it starts, so a runner that is killed leaves a record
    of what it left running.
    """
    cpus = parse_cpu_list(cpu_affinity) if cpu_affinity else None
    log = log_path.open("ab") if log_path is not None else None
    mark = secrets.token_hex(MARK_BYTES)
    process = _start(
        command, {**os.environ, **(env or {}), MARK_VARIABLE: mark}, cwd, log, cpus
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
    launched.identity = identify(process.pid)
    if journal is not None:
        launched.journal = journal
        _journal(journal, launched)
    if cpus:
        launched.affinity_applied = affinity_matches(process.pid, cpus)
    return launched


def _start(
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path | None,
    log: IO[bytes] | None,
    cpus: set[int] | None,
) -> subprocess.Popen[bytes]:
    """The process, in a session of its own. One that cannot start (its
    executable missing or not runnable) raises OSError, said in its log."""
    try:
        return subprocess.Popen(
            list(command),
            env=dict(env),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log if log is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log is not None else subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=_pin(cpus) if cpus else None,
        )
    except OSError as exc:
        if log is not None:
            log.write(f"launch failed: {exc}\n".encode())
            log.close()
        raise


def _journal(path: Path, launched: Launched) -> None:
    entry = {
        "name": launched.name,
        "pid": launched.pid,
        "pgid": launched.pid,
        "mark": launched.mark,
        "started_at_ns": launched.started_at_ns,
        "identity": launched.identity,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def end_journaled(launched: Launched, cleanup: Cleanup) -> None:
    """Journal a launch's end once its cleanup verified, so a resume does
    not judge it again; one whose cleanup did not verify stays for it."""
    if cleanup.verified and launched.journal is not None:
        end_launch(launched.journal, launched.mark)


def end_launch(journal: Path, mark: str) -> None:
    """Journal the end of the launch with this mark."""
    with journal.open("a") as handle:
        handle.write(json.dumps({"ended": mark}) + "\n")


def journaled(path: Path) -> list[dict[str, Any]]:
    """The launches a journal names that have not ended.

    An unreadable line is skipped. Only a whole line ends a launch: one a
    crash tore, unparsable or without its newline, ends nothing, so its
    launch is left to the resume.
    """
    found, ended = [], set()
    text = path.read_text() if path.is_file() else ""
    for line in text.splitlines(keepends=True):
        entry = _journal_line(line)
        if entry is None:
            continue
        if "ended" not in entry:
            found.append(entry)
        elif line.endswith("\n") and _journaled_mark(entry, "ended"):
            ended.add(entry["ended"])
    return [entry for entry in found if entry.get("mark") not in ended]


def _journal_line(line: str) -> dict[str, Any] | None:
    """A journal line's launch or end; None for anything else."""
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    if not isinstance(entry, dict):
        return None
    return entry if "ended" in entry or isinstance(entry.get("pgid"), int) else None


def stop_journaled(
    entry: Mapping[str, Any],
    *,
    timeout_s: float = KILL_WAIT_SECONDS,
    proc: Path = PROC,
) -> Cleanup:
    """Stop what a launch the runner no longer holds left running.

    The runner signals or kills only what it can tie to a launch it made:

    - A launch journaled in another boot left nothing running: nothing is
      signalled, searched or counted.
    - While the launch's leader is still that process (same boot, PID and
      start time), its group and session are the launch's: the group is
      signalled, then the group, the session and the mark are verified
      gone, as after any launch.
    - Otherwise the PID may now be another process's, so its group and
      session are neither signalled nor counted. A process that carries the
      launch's mark is still the launch's, and is killed, when all three
      hold: the journal's boot is this boot; the mark is a nonce of at least
      64 bits; and the process's environment could be read and holds the
      mark variable with exactly that value, never a value it merely starts
      with. A process whose environment cannot be read, or was emptied or
      overwritten, is never killed; one that may be the launch's keeps the
      cleanup from verifying, so a resume refuses while it runs.
    - A launch of no known boot ties nothing: what carries its mark is
      listed, never killed, and keeps the cleanup from verifying.
    """
    identity = entry.get("identity") or {}
    method = "proc" if _linux() else "psutil"
    boot, now = identity.get("boot_id"), current_boot()
    if boot is not None and now is not None and boot != now:
        return Cleanup(True, method)
    mark = _journaled_mark(entry)
    if boot is None or now is None:
        return _marked_only(mark, proc, method)
    pgid = int(entry["pgid"])
    held = still_there(identity, proc=proc)
    if held:
        _stop_group(pgid, identity, timeout_s, proc)
    return verify_cleanup(
        pgid if held else None,
        mark=mark,
        since=identity,
        wait_s=timeout_s,
        proc=proc,
    )


def _stop_group(
    pgid: int, leader: Mapping[str, Any], timeout_s: float, proc: Path
) -> None:
    """Signal the group, escalating, while its leader is still that process."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        if not still_there(leader, proc=proc):
            return
        _signal_group(pgid, signum)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and still_there(leader, proc=proc):
            time.sleep(POLL_SECONDS)


def _journaled_mark(entry: Mapping[str, Any], key: str = "mark") -> str | None:
    """A journal's mark, when it can tie a process to its launch."""
    mark = entry.get(key)
    return mark if isinstance(mark, str) and _NONCE.fullmatch(mark) else None


def _marked_only(mark: str | None, proc: Path, method: str) -> Cleanup:
    """What carries the mark, listed and never killed."""
    search = _marked(mark, proc, method)
    left = tuple(identify(pid, proc=proc) for pid in sorted(search.found))
    return Cleanup(not left, method, left, unreadable=search.unreadable)


def clean_up_after(launched: Launched, *, wait_s: float = KILL_WAIT_SECONDS) -> Cleanup:
    """After a step ends, stop whatever it left in its group and verify that
    nothing it started is left, as after a server."""
    _signal_group(launched.pid, signal.SIGTERM)
    cleanup = verify_cleanup(
        launched.pid,
        mark=launched.mark,
        since=launched.identity,
        lasted_s=launched.lasted_s(),
        wait_s=wait_s,
    )
    end_journaled(launched, cleanup)
    return cleanup


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
    # Processes that may be the launch's whose environment was unreadable or
    # held no variable: nothing shows them unmarked, so the cleanup is not
    # verified. Each by PID and start time, so a resume can tell it still runs.
    blind: tuple[dict[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        search = None
        if self.unreadable is not None:
            search = {
                "complete": self.unreadable == 0,
                "unreadable": self.unreadable,
                "blind": list(self.blind),
            }
        return {
            "verified": self.verified,
            "method": self.method,
            "survivors": list(self.survivors),
            "killed": list(self.killed),
            "mark_search": search,
        }


def verify_cleanup(
    pgid: int | None,
    remembered: Iterable[tuple[int, int]] = (),
    *,
    wait_s: float = KILL_WAIT_SECONDS,
    proc: Path = PROC,
    mark: str | None = None,
    since: Mapping[str, Any] | None = None,
    lasted_s: float | None = None,
) -> Cleanup:
    """Wait until nothing of a group, its session or remembered tree runs.

    ``pgid`` None counts no group or session: none is known to be the
    launch's (``stop_journaled``, once its leader is gone).
    ``mark`` is the launch's environment mark: a process that carries it
    is the launch's, wherever it went. Survivors are killed by PID once;
    whatever outlives that and ``wait_s`` is listed, and the cleanup is not
    verified. A process whose environment cannot be read, or holds no
    variable, cannot be shown unmarked: with ``since``, the launch's
    ``identify`` record, and ``lasted_s``, how long its leader ran, one that
    may be the launch's (``_may_be_launched``) keeps the cleanup from
    verifying, and is never killed, since it may be another's.

    Once a survivor or such a process is seen, it may start another and
    leave: ``lasted_s`` then clears no process for the rest of the check,
    and a poll that finds nothing left verifies only when the next one,
    which starts after anything that left during it, finds nothing either.
    """
    keys = list(remembered)
    method = "proc" if _linux() else "psutil"
    deadline = time.monotonic() + wait_s
    killed: tuple[int, ...] = ()
    settle = _Settle()
    while True:
        bound = None if settle.seen else lasted_s
        poll = _look(pgid, keys, proc, method, mark, since, bound)
        if settle.verifies(poll):
            return Cleanup(True, method, killed=killed, unreadable=poll.unreadable)
        killed = killed or _kill_all(poll.survivors)
        if time.monotonic() >= deadline and not settle.clean:
            left = tuple(identify(pid, proc=proc) for pid in sorted(poll.survivors))
            return Cleanup(False, method, left, killed, poll.unreadable, poll.blind)
        time.sleep(POLL_SECONDS)


@dataclass(frozen=True)
class _Poll:
    """One look at what is left of a launch."""

    survivors: set[int]
    blind: tuple[dict[str, Any], ...]
    unreadable: int | None

    @property
    def clear(self) -> bool:
        return not self.survivors and not self.blind


@dataclass
class _Settle:
    """Whether a poll that finds nothing left verifies: at once while nothing
    has been seen, and otherwise only after a clean poll before it."""

    seen: bool = False
    clean: bool = False

    def verifies(self, poll: _Poll) -> bool:
        if poll.clear and (self.clean or not self.seen):
            return True
        self.seen, self.clean = self.seen or not poll.clear, poll.clear
        return False


def _look(
    pgid: int | None,
    keys: list[tuple[int, int]],
    proc: Path,
    method: str,
    mark: str | None,
    since: Mapping[str, Any] | None,
    lasted_s: float | None,
) -> _Poll:
    """Survivors, and processes that may be the launch's unseen. Once any is
    found, a start after the leader's exit clears no process (``lasted_s``)."""
    search = _marked(mark, proc, method)
    survivors = _survivors(pgid, keys, proc, method) | search.found
    blind = _blind(search.unclear, since, lasted_s, proc, method)
    if lasted_s is not None and (survivors or blind):
        blind = _blind(search.unclear, since, None, proc, method)
    return _Poll(survivors, blind, search.unreadable)


def _kill_all(pids: set[int]) -> tuple[int, ...]:
    """Kill each by PID; the PIDs, sorted."""
    killed = tuple(sorted(pids))
    for pid in killed:
        _kill(pid)
    return killed


def identify(pid: int, *, proc: Path = PROC) -> dict[str, Any]:
    """A process by its host's boot, its PID and its start time, so a later
    check can tell it from another process given the same PID, in this boot
    or the next (Linux start ticks count from the boot)."""
    if _linux():
        info = read_process(pid, proc)
        start = {"start_ticks": None if info is None else info.start_ticks}
    else:
        try:
            start = {"create_time": psutil.Process(pid).create_time()}
        except psutil.Error:
            start = {"create_time": None}
    return {"pid": pid, **start, "boot_id": current_boot()}


@functools.cache
def current_boot() -> str | None:
    """This host's boot ID, read once: no process outlives its boot."""
    return host_boot_id()


def still_there(survivor: Mapping[str, Any], *, proc: Path = PROC) -> bool:
    """Whether a recorded survivor is still the same live process.

    One recorded without its start time cannot be told from a process that
    reused its PID, so it does not count; nor does one recorded in another
    boot, whatever now runs with its PID and start time.
    """
    pid = survivor.get("pid")
    if not isinstance(pid, int) or not _alive(pid):
        return False
    starts = {k: survivor[k] for k in _STARTS if survivor.get(k) is not None}
    if not starts:
        return False
    now = identify(pid, proc=proc)
    boot = survivor.get("boot_id")
    if boot is not None and boot != now["boot_id"]:
        return False
    return all(now.get(k) == v for k, v in starts.items())


def _survivors(
    pgid: int | None, keys: list[tuple[int, int]], proc: Path, method: str
) -> set[int]:
    if method == "proc":
        found = {info.pid for info in still_running(keys, proc)}
        if pgid is not None:
            found |= {info.pid for info in group_members(pgid, pgid, proc)}
        return found
    found = {pid for pid, _ in keys if _alive(pid)}
    return found if pgid is None else found | _psutil_group(pgid)


@dataclass(frozen=True)
class _Search:
    """What the mark search found: marked processes, those whose environment
    was unreadable or empty, and how many could not be read."""

    found: set[int] = field(default_factory=set)
    unclear: set[int] = field(default_factory=set)
    unreadable: int | None = None


def _marked(mark: str | None, proc: Path, method: str) -> _Search:
    """Live processes whose environment, read, holds the mark variable with
    exactly the launch's mark as its value."""
    if not mark:
        return _Search()
    needle = f"{MARK_VARIABLE}={mark}"
    if method == "proc":
        return _proc_marked(needle.encode() + b"\0", proc)
    return _psutil_marked(mark)


def _proc_marked(needle: bytes, proc: Path) -> _Search:
    search = _Search(unreadable=0)
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            environ = (entry / "environ").read_bytes()
        except PermissionError:
            search = _unclear(search, pid, unreadable=True)
            continue
        except OSError:
            continue  # gone
        if not any(b"=" in entry for entry in environ.split(b"\0")):
            # Emptied, or overwritten in place by a process title.
            search = _unclear(search, pid, unreadable=False)
        elif (b"\0" + environ).find(b"\0" + needle) >= 0 and _alive(pid):
            search.found.add(pid)
    return search


def _psutil_marked(mark: str) -> _Search:
    search = _Search(unreadable=0)
    for process in psutil.process_iter():
        try:
            environ = process.environ()
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        # psutil on macOS may raise SystemError for a process it cannot read.
        except (psutil.Error, OSError, SystemError):
            search = _unclear(search, process.pid, unreadable=True)
            continue
        if not environ:
            search = _unclear(search, process.pid, unreadable=False)
        elif environ.get(MARK_VARIABLE) == mark and _alive(process.pid):
            search.found.add(process.pid)
    return search


def _unclear(search: _Search, pid: int, *, unreadable: bool) -> _Search:
    if _alive(pid):
        search.unclear.add(pid)
    if not unreadable:
        return search
    return _Search(search.found, search.unclear, (search.unreadable or 0) + 1)


@dataclass(frozen=True)
class _Process:
    """What decides whether a process may be a launch's: its start, in the
    processes' own clock, its parent and its real user."""

    start: float | None
    ppid: int | None
    uid: int | None


def _blind(
    unclear: set[int],
    since: Mapping[str, Any] | None,
    lasted_s: float | None,
    proc: Path,
    method: str,
) -> tuple[dict[str, Any], ...]:
    if since is None:
        return ()
    adopters = _adopters(proc, method)
    found = sorted(
        pid
        for pid in unclear
        if _may_be_launched(
            pid, since, proc, method, lasted_s=lasted_s, adopters=adopters
        )
    )
    return tuple(identify(pid, proc=proc) for pid in found)


def _adopters(proc: Path, method: str) -> frozenset[int]:
    """What adopts an orphan of a launch: ``init``, or on Linux any of the
    runner's ancestors, since one may be a subreaper (``systemd --user``
    is, for a desktop session). macOS has no subreapers."""
    found = {1}
    pid = os.getppid() if method == "proc" else 1
    while pid > 1 and pid not in found:
        found.add(pid)
        parent = _view(pid, proc, method)
        pid = parent.ppid if parent is not None and parent.ppid is not None else 0
    return frozenset(found)


def _may_be_launched(
    pid: int,
    since: Mapping[str, Any],
    proc: Path,
    method: str,
    *,
    lasted_s: float | None = None,
    adopters: frozenset[int] = frozenset({1}),
) -> bool:
    """Whether a live process may be the launch's, by what is readable
    without its environment.

    It is not when another user runs it (which includes anything run under
    sudo), when it started more than ``START_SLACK_SECONDS`` before the
    launch or after its leader had exited (``lasted_s``, given only while
    nothing of the launch has been seen to outlive the leader: nothing of
    the launch was left to start it, but another orphan, itself judged), or
    when its parent is none of ``adopters`` (``_adopters``): a launch's
    process that left its group and session is an orphan, adopted by
    ``init`` or a subreaper among the runner's ancestors, and any other
    parent shows whose it is (a launch's own processes are found by their
    group, session and remembered tree). A process of the launch left as
    the child of a long-lived process that predates the launch (a ``tmux``
    server it asked to run something, an older shell) is missed.
    """
    seen = _view(pid, proc, method)
    if seen is None:
        return False
    if seen.uid is not None and seen.uid != os.getuid():
        return False
    if _older(seen.start, since, method) or _later(seen.start, since, method, lasted_s):
        return False
    return seen.ppid in adopters


def _older(start: float | None, since: Mapping[str, Any], method: str) -> bool:
    """Whether a start, in the processes' own clock, is before the launch's."""
    if method == "proc":
        launch, slack = since.get("start_ticks"), START_SLACK_SECONDS * _clock_ticks()
    else:
        launch, slack = since.get("create_time"), START_SLACK_SECONDS
    if start is None or not isinstance(launch, (int, float)):
        return False
    return start < launch - slack


def _later(
    start: float | None, since: Mapping[str, Any], method: str, lasted_s: float | None
) -> bool:
    """Whether a start, in the processes' own clock, is after the launch's
    leader had exited."""
    if lasted_s is None or start is None:
        return False
    if method == "proc":
        launch, unit = since.get("start_ticks"), _clock_ticks()
    else:
        launch, unit = since.get("create_time"), 1
    if not isinstance(launch, (int, float)):
        return False
    return start > launch + (lasted_s + START_SLACK_SECONDS) * unit


def _clock_ticks() -> int:
    return int(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100


def _view(pid: int, proc: Path, method: str) -> _Process | None:
    if method == "proc":
        info = read_process(pid, proc)
        if info is None:
            return None
        return _Process(info.start_ticks, info.ppid, _proc_uid(pid, proc))
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            return _Process(process.create_time(), process.ppid(), process.uids().real)
    except psutil.Error:
        return None


def _proc_uid(pid: int, proc: Path) -> int | None:
    """The real user ID, from ``status``, which is readable when ``environ``
    is not."""
    try:
        for line in (proc / str(pid) / "status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


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


def listens(pid: int, port: int, *, proc: Path = PROC) -> bool | None:
    """Whether the process or a descendant listens on the TCP port; None
    where that cannot be read."""
    if _linux():
        pids = [info.pid for info in process_tree(pid, proc)] or [pid]
        ports = listening_ports(pids, proc)
        return None if ports is None else port in ports
    try:
        root = psutil.Process(pid)
        return any(
            _psutil_listens(process, port)
            for process in [root, *root.children(recursive=True)]
        )
    except (psutil.Error, OSError):
        return None


def _psutil_listens(process: psutil.Process, port: int) -> bool:
    read = getattr(process, "net_connections", None) or process.connections
    return any(
        conn.status == psutil.CONN_LISTEN and conn.laddr.port == port
        for conn in read(kind="tcp")
    )


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
    journal: Path | None = None,
) -> tuple[Launched, bool]:
    """Run a command to its end, or stop its group at the timeout.

    Returns the launched process and whether it timed out.
    """
    launched = launch(
        name,
        command,
        env=env,
        cpu_affinity=cpu_affinity,
        log_path=log_path,
        journal=journal,
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
    "clean_up_after",
    "current_boot",
    "end_journaled",
    "end_launch",
    "identify",
    "journaled",
    "listens",
    "launch",
    "parse_cpu_list",
    "remembered_tree",
    "run_step",
    "still_there",
    "stop",
    "stop_journaled",
    "unexpected_roles",
    "verify_cleanup",
    "wait_for_file",
]
