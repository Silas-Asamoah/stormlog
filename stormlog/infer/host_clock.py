"""Identify one host boot when deciding if wall timestamps share a clock."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


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
