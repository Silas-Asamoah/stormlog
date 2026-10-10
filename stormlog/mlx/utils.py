"""Explicit runtime information and neutral host information."""

from __future__ import annotations

from typing import Any

from stormlog.system_info import get_system_info

from .runtime import MLXRuntime, Runtime

__all__ = ["get_device_info", "get_system_info"]


def get_device_info(runtime: Runtime | None = None) -> dict[str, Any]:
    adapter = runtime if runtime is not None else MLXRuntime()
    return {
        "framework": "mlx",
        "backend": "metal",
        "mlx_version": adapter.version,
        "device_id": 0,
        "memory_scope": "process_allocator",
        "memory_model": "unified",
        "raw_device_info": dict(adapter.device_info),
        "device_info_error": adapter.device_info_error,
    }
