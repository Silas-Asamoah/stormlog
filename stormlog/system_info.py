"""Framework-independent host information; no GPU runtime discovery."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import psutil


def get_system_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "pid": os.getpid(),
    }
    try:
        memory = psutil.virtual_memory()
        result.update(
            cpu_count=psutil.cpu_count(),
            cpu_count_logical=psutil.cpu_count(logical=True),
            memory_total=memory.total,
            memory_available=memory.available,
            memory_percent=memory.percent,
            process_rss_bytes=psutil.Process().memory_info().rss,
        )
    except Exception as exc:
        result["system_info_error"] = f"{type(exc).__name__}: {exc}"
    return result
