"""Optional, process-local MLX memory instrumentation.

Importing this package or its classes does not initialize a native runtime.
"""

from __future__ import annotations

import importlib
from typing import Any

from stormlog import __version__

_EXPORTS = {
    "MLXMemoryProfiler": "profiler",
    "ProfileRegion": "profiler",
    "MemoryTracker": "tracker",
    "MemorySnapshot": "models",
    "ProfileResult": "models",
    "TrackingResult": "models",
    "MLXRuntime": "runtime",
    "profile_function": "context_profiler",
    "profile_context": "context_profiler",
    "get_global_profiler": "context_profiler",
    "clear_global_profiler": "context_profiler",
    "get_device_info": "utils",
    "get_system_info": "utils",
}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
