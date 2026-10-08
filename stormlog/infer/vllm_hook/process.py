"""The birth of the hook's process and of its parent, for the hello.

A pid can be reused, so a process is named by its pid and its start: the raw
start in clock ticks after boot (``/proc/<pid>/stat`` field 22), exact within
one boot, and the wall-clock start as psutil's ``create_time`` gives it, the
value Stormlog's server collector records. Each field is null where it cannot
be read.
"""

from __future__ import annotations

import os
from pathlib import Path


def process_fields() -> dict[str, int | None]:
    pid, parent = os.getpid(), os.getppid()
    return {
        "process_start_ns": start_ns(pid),
        "process_start_ticks": start_ticks(pid),
        "parent_pid": parent,
        "parent_process_start_ticks": start_ticks(parent),
        "parent_process_start_ns": start_ns(parent),
    }


def start_ticks(pid: int) -> int | None:
    try:
        return ticks_from_stat(Path(f"/proc/{pid}/stat").read_text())
    except OSError:
        return None


def ticks_from_stat(stat: str) -> int | None:
    """Field 22 of a ``/proc/<pid>/stat`` line. The command name in field 2
    may hold spaces and parentheses, so fields are counted from after its
    last ``)``: field 3 is the first of them, field 22 the 20th."""
    try:
        return int(stat[stat.rindex(")") + 2 :].split()[19])
    except (ValueError, IndexError):
        return None


def start_ns(pid: int) -> int | None:
    try:
        import psutil

        return int(psutil.Process(pid).create_time() * 1_000_000_000)
    except Exception:
        return _proc_start_ns(pid)


def _proc_start_ns(pid: int) -> int | None:
    """psutil's own arithmetic: the start ticks over CLK_TCK, plus boot time."""
    ticks, boot = start_ticks(pid), _boot_seconds()
    try:
        per_second = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        return None
    if ticks is None or boot is None or per_second <= 0:
        return None
    return int((ticks / per_second + boot) * 1_000_000_000)


def _boot_seconds() -> int | None:
    try:
        lines = Path("/proc/stat").read_text().splitlines()
        return next(int(line.split()[1]) for line in lines if line.startswith("btime"))
    except (OSError, ValueError, IndexError, StopIteration):
        return None


__all__ = ["process_fields", "start_ns", "start_ticks", "ticks_from_stat"]
