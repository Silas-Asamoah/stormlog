"""Stormlog - A comprehensive memory profiling tool for PyTorch."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"
__author__ = "Stormlog Team"

_TORCH_INSTALL_GUIDANCE = (
    "PyTorch is required for this feature. Install with "
    "`pip install 'stormlog[torch]'` "
    "or follow https://pytorch.org/get-started/locally/."
)
_VIZ_INSTALL_GUIDANCE = (
    "MemoryVisualizer requires optional visualization dependencies. "
    "Install with `pip install stormlog[viz]`."
)

_SYMBOL_TO_MODULE = {
    "GPUMemoryProfiler": (".profiler", "GPUMemoryProfiler"),
    "MemorySnapshot": (".profiler", "MemorySnapshot"),
    "ProfileResult": (".profiler", "ProfileResult"),
    "profile_context": (".context_profiler", "profile_context"),
    "profile_function": (".context_profiler", "profile_function"),
    "MemoryAnalyzer": (".analyzer", "MemoryAnalyzer"),
    "GapFinding": (".analyzer", "GapFinding"),
    "MemoryTracker": (".tracker", "MemoryTracker"),
    "OOMFlightRecorder": (".oom_flight_recorder", "OOMFlightRecorder"),
    "OOMFlightRecorderConfig": (".oom_flight_recorder", "OOMFlightRecorderConfig"),
    "OOMExceptionClassification": (
        ".oom_flight_recorder",
        "OOMExceptionClassification",
    ),
    "classify_oom_exception": (".oom_flight_recorder", "classify_oom_exception"),
    "TelemetryEventV2": (".telemetry", "TelemetryEventV2"),
    "DeviceMemoryCollector": (".device_collectors", "DeviceMemoryCollector"),
    "DeviceMemorySample": (".device_collectors", "DeviceMemorySample"),
    "build_device_memory_collector": (
        ".device_collectors",
        "build_device_memory_collector",
    ),
    "detect_torch_runtime_backend": (
        ".device_collectors",
        "detect_torch_runtime_backend",
    ),
    "CPUMemoryProfiler": (".cpu_profiler", "CPUMemoryProfiler"),
    "CPUMemoryTracker": (".cpu_profiler", "CPUMemoryTracker"),
    "telemetry_event_from_record": (".telemetry", "telemetry_event_from_record"),
    "telemetry_event_to_dict": (".telemetry", "telemetry_event_to_dict"),
    "validate_telemetry_record": (".telemetry", "validate_telemetry_record"),
    "load_telemetry_events": (".telemetry", "load_telemetry_events"),
    "resolve_distributed_identity": (".telemetry", "resolve_distributed_identity"),
    "get_gpu_info": (".utils", "get_gpu_info"),
    "format_bytes": (".utils", "format_bytes"),
    "convert_bytes": (".utils", "convert_bytes"),
}


def _is_torch_missing(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name == "torch":
            return True
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc
    return False


def _resolve_symbol(name: str) -> Any:
    module_name, symbol_name = _SYMBOL_TO_MODULE[name]
    try:
        module = importlib.import_module(module_name, __name__)
    except Exception as exc:
        if _is_torch_missing(exc):
            raise ImportError(_TORCH_INSTALL_GUIDANCE) from exc
        raise

    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def _resolve_memory_visualizer() -> Any:
    try:
        module = importlib.import_module(".visualizer", __name__)
        value = getattr(module, "MemoryVisualizer")
    except ImportError as exc:
        if _is_torch_missing(exc):
            raise ImportError(_TORCH_INSTALL_GUIDANCE) from exc
        import_error = exc

        class MemoryVisualizer:
            """Fallback placeholder when optional visualization dependencies are missing."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError(_VIZ_INSTALL_GUIDANCE) from import_error

        value = MemoryVisualizer
    globals()["MemoryVisualizer"] = value
    return value


def __getattr__(name: str) -> Any:
    if name == "MemoryVisualizer":
        return _resolve_memory_visualizer()
    if name in _SYMBOL_TO_MODULE:
        return _resolve_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


__all__ = [
    "GPUMemoryProfiler",
    "MemorySnapshot",
    "ProfileResult",
    "profile_context",
    "profile_function",
    "MemoryVisualizer",
    "MemoryAnalyzer",
    "GapFinding",
    "MemoryTracker",
    "OOMFlightRecorder",
    "OOMFlightRecorderConfig",
    "OOMExceptionClassification",
    "classify_oom_exception",
    "TelemetryEventV2",
    "DeviceMemoryCollector",
    "DeviceMemorySample",
    "build_device_memory_collector",
    "detect_torch_runtime_backend",
    "CPUMemoryProfiler",
    "CPUMemoryTracker",
    "telemetry_event_from_record",
    "telemetry_event_to_dict",
    "validate_telemetry_record",
    "load_telemetry_events",
    "resolve_distributed_identity",
    "get_gpu_info",
    "format_bytes",
    "convert_bytes",
]
