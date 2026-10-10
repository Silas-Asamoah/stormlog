"""Small injectable adapter for the MLX 0.32.3 core API.

Only construction discovers MLX. No allocation, default-device mutation,
setter probing, or import of another ML framework is performed.
"""

from __future__ import annotations

import importlib
import platform
import re
import threading
from importlib.metadata import version
from typing import Any, Protocol

MINIMUM_VERSION = (0, 32, 3)


class MLXRuntimeError(RuntimeError):
    """A requested native runtime is unavailable or incompatible."""


class MLXNotInstalledError(MLXRuntimeError):
    """The optional MLX package is missing."""


class MLXPlatformError(MLXRuntimeError):
    """The platform has not been qualified for this integration."""


class MLXInitializationError(MLXRuntimeError):
    """The native module or availability probe failed."""


class MLXMetalUnavailableError(MLXRuntimeError):
    """MLX loaded but no usable Metal runtime is present."""


class Runtime(Protocol):
    """Operations consumed by the collector and explicit profiler."""

    version: str
    device_info: dict[str, Any]
    device_info_error: str | None

    def supports(self, name: str) -> bool: ...
    def read_bytes(self, name: str) -> int: ...
    def evaluate(self, *roots: Any) -> None: ...
    def synchronize(self, streams: tuple[Any, ...]) -> None: ...
    def reset_peak(self) -> None: ...
    def default_stream(self) -> Any: ...


_DISCOVERY_LOCK = threading.Lock()
_CORE: Any = None
_CORE_VERSION: str | None = None


def _validate_version(value: str) -> None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", value)
    if match is None:
        raise MLXRuntimeError(f"Cannot determine MLX compatibility from {value!r}")
    release = tuple(int(match[i]) for i in (1, 2, 3))
    if release < MINIMUM_VERSION or (release == MINIMUM_VERSION and match[4]):
        raise MLXRuntimeError("MLX >= 0.32.3 is required; upgrade stormlog[mlx]")


def _check_core(core: Any) -> None:
    try:
        available = core.metal.is_available()
    except Exception as exc:
        raise MLXInitializationError("MLX Metal availability probe failed") from exc
    if not available:
        raise MLXMetalUnavailableError("MLX has no usable Metal device in this process")
    required = ("get_active_memory", "eval", "synchronize", "default_stream")
    missing = [name for name in required if not callable(getattr(core, name, None))]
    if missing:
        raise MLXRuntimeError(f"MLX core APIs unavailable: {', '.join(missing)}")


def _discover() -> tuple[Any, str]:
    global _CORE, _CORE_VERSION
    with _DISCOVERY_LOCK:
        if _CORE is not None and _CORE_VERSION is not None:
            return _CORE, _CORE_VERSION
        try:
            core = importlib.import_module("mlx.core")
            installed_version = version("mlx")
        except ModuleNotFoundError as exc:
            if exc.name in {"mlx", "mlx.core"}:
                raise MLXNotInstalledError(
                    "MLX is missing. Install with pip install 'stormlog[mlx]'"
                ) from exc
            raise MLXInitializationError("MLX native initialization failed") from exc
        except Exception as exc:
            if str(exc).startswith("[metal::load_device] No Metal device available."):
                raise MLXMetalUnavailableError(
                    "MLX has no usable Metal device in this process"
                ) from exc
            raise MLXInitializationError("MLX native initialization failed") from exc
        _validate_version(installed_version)
        _check_core(core)
        _CORE, _CORE_VERSION = core, installed_version
        return core, installed_version


def validate_bytes(value: Any, name: str) -> int:
    """Accept successful nonnegative integer bytes; bools are invalid."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must return nonnegative integer bytes, got {value!r}")
    return value


class MLXRuntime:
    """Native Apple Silicon Metal adapter; ``core`` is a test injection seam."""

    def __init__(
        self,
        device_id: int = 0,
        *,
        core: Any = None,
        runtime_version: str | None = None,
        platform_system: str | None = None,
        platform_machine: str | None = None,
    ) -> None:
        if type(device_id) is not int or device_id != 0:
            raise ValueError(
                "MLX supports only the local process allocator, device_id=0"
            )
        system = platform_system or platform.system()
        machine = platform_machine or platform.machine()
        if system != "Darwin" or machine != "arm64":
            raise MLXPlatformError(
                "Stormlog MLX requires native Apple Silicon macOS; "
                f"got {system}/{machine}"
            )
        if core is None:
            self.core, self.version = _discover()
        else:
            self.version = runtime_version or "unknown"
            _validate_version(self.version)
            _check_core(core)
            self.core = core
        self.device_info: dict[str, Any] = {}
        self.device_info_error: str | None = None
        try:
            self.device_info = dict(self.core.device_info(self.core.gpu))
        except Exception as exc:
            self.device_info_error = f"{type(exc).__name__}: {exc}"

    def supports(self, name: str) -> bool:
        return callable(getattr(self.core, name, None))

    def read_bytes(self, name: str) -> int:
        return validate_bytes(getattr(self.core, name)(), name)

    def evaluate(self, *roots: Any) -> None:
        """MLX eval traverses supported trees and ignores ordinary leaves."""
        self.core.eval(*roots)

    def default_stream(self) -> Any:
        return self.core.default_stream(self.core.gpu)

    def synchronize(self, streams: tuple[Any, ...]) -> None:
        for stream in streams:
            self.core.synchronize(stream)

    def reset_peak(self) -> None:
        if not self.supports("reset_peak_memory"):
            raise MLXRuntimeError("This MLX runtime cannot reset the peak counter")
        self.core.reset_peak_memory()
