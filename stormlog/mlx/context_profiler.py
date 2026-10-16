"""Lazy singleton and transparent decorators; use outside MLX transforms."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

from .profiler import MLXMemoryProfiler, ProfileRegion

P = ParamSpec("P")
R = TypeVar("R")
_LOCK = threading.Lock()
_GLOBAL: MLXMemoryProfiler | None = None


def get_global_profiler() -> MLXMemoryProfiler:
    global _GLOBAL
    with _LOCK:
        if _GLOBAL is None:
            _GLOBAL = MLXMemoryProfiler()
        return _GLOBAL


def set_global_profiler(profiler: MLXMemoryProfiler | None) -> None:
    global _GLOBAL
    with _LOCK:
        _GLOBAL = profiler


def clear_global_profiler() -> None:
    set_global_profiler(None)


def profile_function(
    function: Callable[P, R] | None = None,
    *,
    profiler: MLXMemoryProfiler | None = None,
    name: str | None = None,
    state_getter: Callable[[], Any] | None = None,
    root_extractor: Callable[[Any], Any] | None = None,
    streams: tuple[Any, ...] | None = None,
) -> Any:
    """Decorate once or with options; construction does not initialize MLX."""

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        @wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            selected = profiler if profiler is not None else get_global_profiler()
            return cast(
                R,
                selected.profile_function(
                    lambda: target(*args, **kwargs),
                    name=name or target.__name__,
                    state_getter=state_getter,
                    root_extractor=root_extractor,
                    streams=streams,
                ),
            )

        return wrapper

    return decorate if function is None else decorate(function)


def profile_context(
    name: str = "context", *, profiler: MLXMemoryProfiler | None = None, **options: Any
) -> AbstractContextManager[ProfileRegion]:
    selected = profiler if profiler is not None else get_global_profiler()
    return selected.profile_context(name, **options)
