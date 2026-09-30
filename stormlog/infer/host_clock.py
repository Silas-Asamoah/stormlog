"""Identify one host boot when deciding if wall timestamps share a clock."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

WALL_CLOCK = "unix_epoch_ns"


def wall_clock_domain(host: str, boot_id: str | None) -> str:
    """Name the wall clock of one host boot.

    Hostnames repeat across machines, so the boot ID is part of the name: equal
    names then mean one clock. Without a boot ID the name cannot show that two
    timestamps share a clock; see :func:`is_boot_qualified`.
    """
    if boot_id:
        return f"{host}/{boot_id}/{WALL_CLOCK}"
    return f"{host}/{WALL_CLOCK}"


def is_boot_qualified(clock_domain: str) -> bool:
    """Whether a wall clock domain names a host boot, not only a hostname."""
    parts = clock_domain.split("/")
    return len(parts) == 3 and all(parts) and parts[2] == WALL_CLOCK


def host_boot_id() -> str | None:
    """Return a boot-scoped ID, or None when the host cannot provide one."""
    system = platform.system()
    if system == "Linux":
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
        except OSError:
            return None
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.bootsessionuuid"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    return None
