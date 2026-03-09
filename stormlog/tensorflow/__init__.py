"""TensorFlow support for Stormlog."""

from __future__ import annotations

import importlib
from typing import Any

from stormlog import __version__

__author__ = "Stormlog Team"
__email__ = "prince.agyei.tuffour@gmail.com"

_TF_INSTALL_GUIDANCE = (
    "TensorFlow is required for this feature. Install with "
    "`pip install 'stormlog[tf]'`."
)
_VIZ_INSTALL_GUIDANCE = (
    "TensorFlowVisualizer requires optional visualization dependencies. "
    "Install with `pip install 'stormlog[viz]'`."
)

_SYMBOL_TO_MODULE = {
    "TensorFlowGapFinding": (".analyzer", "GapFinding"),
    "TensorFlowAnalyzer": (".analyzer", "MemoryAnalyzer"),
    "TensorFlowProfiler": (".context_profiler", "TensorFlowProfiler"),
    "TFMemoryProfiler": (".profiler", "TFMemoryProfiler"),
    "TensorFlowMemoryTracker": (".tracker", "MemoryTracker"),
    "get_system_info": (".utils", "get_system_info"),
}


def _is_tensorflow_missing(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name == "tensorflow":
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
        if _is_tensorflow_missing(exc):
            raise ImportError(_TF_INSTALL_GUIDANCE) from exc
        raise

    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def _resolve_visualizer() -> Any:
    try:
        module = importlib.import_module(".visualizer", __name__)
        value = getattr(module, "MemoryVisualizer")
    except ImportError as exc:
        if _is_tensorflow_missing(exc):
            raise ImportError(_TF_INSTALL_GUIDANCE) from exc
        import_error = exc

        class TensorFlowVisualizer:
            """Fallback placeholder when visualization dependencies are missing."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError(_VIZ_INSTALL_GUIDANCE) from import_error

        value = TensorFlowVisualizer
    globals()["TensorFlowVisualizer"] = value
    return value


def __getattr__(name: str) -> Any:
    if name == "TensorFlowVisualizer":
        return _resolve_visualizer()
    if name in _SYMBOL_TO_MODULE:
        return _resolve_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


__all__ = [
    "TensorFlowProfiler",
    "TFMemoryProfiler",
    "TensorFlowMemoryTracker",
    "TensorFlowVisualizer",
    "TensorFlowAnalyzer",
    "TensorFlowGapFinding",
    "get_system_info",
]
