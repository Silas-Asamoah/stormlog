"""A vLLM server's processes, read from Linux ``/proc``.

A process is identified by the host boot, its PID and its start time in
clock ticks since boot (``/proc/<pid>/stat`` field 22): a PID can be reused,
the triple cannot. Session and process group come from the same line, so a
caller can list every process left in a server's group or session, which a
parent's exit does not end.

vLLM 0.30.0 renames its processes (``setproctitle``): the engine core
becomes ``VLLM::EngineCore`` and each worker ``VLLM::Worker_TP0`` and so on.
Those names, the API server's command line and the helpers Python starts
(multiprocessing's resource tracker, torch's compile workers) give each
process a role. Anything else, such as a ``pip`` or ``nvidia-smi`` that
``collect_env`` started, is ``other``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROC = Path("/proc")

API_SERVER = "api_server"
ENGINE_CORE = "engine_core"
WORKER = "worker"
RESOURCE_TRACKER = "resource_tracker"
COMPILE_WORKER = "compile_worker"
OTHER = "other"
SERVER_ROLES = (API_SERVER, ENGINE_CORE, WORKER)
HELPER_ROLES = (RESOURCE_TRACKER, COMPILE_WORKER)

# VLLM_PROCESS_NAME_PREFIX, "::", the name, then "_DP0", "_TP1" and so on.
_TITLE = re.compile(r"^[A-Za-z0-9_.-]*::(EngineCore|Worker|APIServer)(?=$|[_\s])")
_STAT_FIELDS_AFTER_COMM = 20  # state (3) .. starttime (22)
# A zombie has exited and only waits to be reaped; it holds no GPU or memory.
_DEAD_STATES = frozenset({"Z", "X", "x"})


@dataclass(frozen=True)
class ProcessInfo:
    """One process as ``/proc`` showed it."""

    pid: int
    ppid: int
    pgid: int
    sid: int
    start_ticks: int
    state: str
    comm: str
    cmdline: tuple[str, ...]
    cpus_allowed_list: str | None
    role: str

    @property
    def key(self) -> tuple[int, int]:
        """PID and start ticks: one process within one boot."""
        return self.pid, self.start_ticks

    @property
    def alive(self) -> bool:
        return self.state not in _DEAD_STATES

    def to_record(self, *, boot_time_s: int | None = None) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "pgid": self.pgid,
            "sid": self.sid,
            "start_ticks": self.start_ticks,
            "start_ns": start_ns(self.start_ticks, boot_time_s),
            "comm": self.comm,
            "role": self.role,
            "cpus_allowed_list": self.cpus_allowed_list,
        }


def read_process(pid: int, proc: Path = PROC) -> ProcessInfo | None:
    """The process, or None when it is gone or cannot be read."""
    base = proc / str(pid)
    try:
        stat = (base / "stat").read_text()
        cmdline = (base / "cmdline").read_bytes()
    except OSError:
        return None
    parsed = _parse_stat(stat)
    if parsed is None:
        return None
    comm, state, ppid, pgid, sid, start_ticks = parsed
    arguments = tuple(part for part in cmdline.decode(errors="replace").split("\0"))
    arguments = tuple(part for part in arguments if part)
    return ProcessInfo(
        pid=pid,
        ppid=ppid,
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        state=state,
        comm=comm,
        cmdline=arguments,
        cpus_allowed_list=_status_field(base, "Cpus_allowed_list"),
        role=classify_role(comm, arguments),
    )


def read_environ(pid: int, proc: Path = PROC) -> dict[str, str] | None:
    """A process's environment at exec, or None when it cannot be read."""
    try:
        raw = (proc / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    environ: dict[str, str] = {}
    for entry in raw.decode(errors="replace").split("\0"):
        name, separator, value = entry.partition("=")
        if separator and name:
            environ[name] = value
    return environ


def all_processes(proc: Path = PROC) -> Iterator[ProcessInfo]:
    """Every live process ``/proc`` lists that can still be read."""
    try:
        entries = sorted(
            int(entry.name) for entry in proc.iterdir() if entry.name.isdigit()
        )
    except OSError:
        return
    for pid in entries:
        info = read_process(pid, proc)
        if info is not None and info.alive:
            yield info


def process_tree(pid: int, proc: Path = PROC) -> list[ProcessInfo]:
    """The process and its descendants, by parent PID, root first."""
    processes = list(all_processes(proc))
    root = next((info for info in processes if info.pid == pid), None)
    if root is None:
        return []
    children: dict[int, list[ProcessInfo]] = {}
    for info in processes:
        children.setdefault(info.ppid, []).append(info)
    tree, frontier = [root], [root]
    while frontier:
        parent = frontier.pop(0)
        for child in children.get(parent.pid, []):
            if child.pid != parent.pid:
                tree.append(child)
                frontier.append(child)
    return tree


def group_members(
    pgid: int, sid: int | None = None, proc: Path = PROC
) -> list[ProcessInfo]:
    """Every process in the process group, or else in the session."""
    return [
        info
        for info in all_processes(proc)
        if info.pgid == pgid or (sid is not None and info.sid == sid)
    ]


def still_running(
    keys: Iterable[tuple[int, int]], proc: Path = PROC
) -> list[ProcessInfo]:
    """The processes, by PID and start ticks, that are still alive.

    A PID now held by a process that started at another time is a reused
    PID, not a survivor. This finds a process that left its group and
    session (``setsid``), which a group listing cannot.
    """
    survivors = []
    for pid, start_ticks in keys:
        info = read_process(pid, proc)
        if info is not None and info.alive and info.start_ticks == start_ticks:
            survivors.append(info)
    return survivors


def classify_role(comm: str, cmdline: Iterable[str]) -> str:
    """What a vLLM 0.30.0 server process is, from its title or command line."""
    arguments = list(cmdline)
    title = arguments[0] if arguments else comm
    match = _TITLE.match(title.strip()) or _TITLE.match(comm)
    if match is not None:
        name = match.group(1)
        if name == "EngineCore":
            return ENGINE_CORE
        return WORKER if name == "Worker" else API_SERVER
    joined = " ".join(arguments)
    if "multiprocessing.resource_tracker" in joined:
        return RESOURCE_TRACKER
    if "torch._inductor.compile_worker" in joined:
        return COMPILE_WORKER
    if _is_api_server(arguments):
        return API_SERVER
    return OTHER


def boot_time_s(proc: Path = PROC) -> int | None:
    """The boot time in whole seconds since the epoch (``btime``)."""
    try:
        for line in (proc / "stat").read_text().splitlines():
            if line.startswith("btime "):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def start_ns(start_ticks: int, boot_time: int | None) -> int | None:
    """Wall start time, as ``psutil.Process.create_time`` computes it."""
    if boot_time is None:
        return None
    return int((boot_time + start_ticks / _clock_ticks()) * 1_000_000_000)


def _clock_ticks() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):
        return 100


def _parse_stat(stat: str) -> tuple[str, str, int, int, int, int] | None:
    """comm, state, ppid, pgrp, session and starttime.

    comm may hold spaces or ')', so the fields after it start at the last ')'.
    """
    opened, closed = stat.find("("), stat.rfind(")")
    if opened < 0 or closed < opened:
        return None
    fields = stat[closed + 1 :].split()
    if len(fields) < _STAT_FIELDS_AFTER_COMM:
        return None
    try:
        return (
            stat[opened + 1 : closed],
            fields[0],
            int(fields[1]),
            int(fields[2]),
            int(fields[3]),
            int(fields[19]),
        )
    except ValueError:
        return None


def _status_field(base: Path, name: str) -> str | None:
    try:
        for line in (base / "status").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key == name:
                return value.strip()
    except OSError:
        return None
    return None


def _is_api_server(arguments: list[str]) -> bool:
    """``vllm serve ...`` or the OpenAI server module run directly."""
    names = [Path(argument).name for argument in arguments[:3]]
    if "vllm" in names and "serve" in arguments[1:3]:
        return True
    return "vllm.entrypoints.openai.api_server" in arguments


_SOCKET = re.compile(r"^socket:\[(\d+)\]$")
_TCP_LISTEN = "0A"


def listening_ports(pids: Iterable[int], proc: Path = PROC) -> list[int] | None:
    """The TCP ports these processes listen on; None if no descriptor is readable.

    A listening socket is one of their descriptors whose inode the first
    readable process's network namespace lists in state LISTEN, so another
    process's listener in the same namespace is left out.
    """
    inodes: set[str] = set()
    tables: list[Path] = []
    for pid in pids:
        base = proc / str(pid)
        try:
            entries = list((base / "fd").iterdir())
        except OSError:
            continue
        tables.append(base / "net")
        inodes.update(_socket_inodes(entries))
    if not tables:
        return None
    ports: set[int] = set()
    for name in ("tcp", "tcp6"):
        ports |= _listening(tables[0] / name, inodes)
    return sorted(ports)


def _socket_inodes(entries: Iterable[Path]) -> set[str]:
    found = set()
    for entry in entries:
        try:
            match = _SOCKET.match(os.readlink(entry))
        except OSError:
            continue
        if match:
            found.add(match.group(1))
    return found


def _listening(table: Path, inodes: set[str]) -> set[int]:
    try:
        rows = table.read_text().splitlines()[1:]
    except OSError:
        return set()
    ports = set()
    for row in rows:
        fields = row.split()
        if len(fields) > 9 and fields[3] == _TCP_LISTEN and fields[9] in inodes:
            ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return ports


def network_namespace(pid: int | str, proc: Path = PROC) -> str | None:
    """A process's network namespace (``net:[inode]``), or None."""
    try:
        return os.readlink(proc / str(pid) / "ns" / "net")
    except OSError:
        return None


__all__ = [
    "API_SERVER",
    "COMPILE_WORKER",
    "ENGINE_CORE",
    "HELPER_ROLES",
    "OTHER",
    "PROC",
    "RESOURCE_TRACKER",
    "SERVER_ROLES",
    "WORKER",
    "ProcessInfo",
    "all_processes",
    "boot_time_s",
    "classify_role",
    "group_members",
    "listening_ports",
    "network_namespace",
    "process_tree",
    "read_environ",
    "read_process",
    "start_ns",
    "still_running",
]
