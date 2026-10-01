"""Utility functions for JAX memory profiling.

This module provides helper functions for JAX device discovery,
memory formatting, system information, and environment validation.
"""

from __future__ import annotations

import functools
import logging
import os
import platform
from typing import Any, Callable, Dict, List, Optional, Union, cast

from .jax_env import configure_jax_logging

configure_jax_logging()

jax: Any

try:
    import jax as _jax  # noqa: E402

    jax = _jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


def normalize_jax_backend(backend: str) -> str:
    """Return Stormlog's stable name for a JAX runtime backend."""
    normalized = backend.strip().lower()
    if normalized in {"gpu", "cuda", "rocm"}:
        return "gpu"
    if normalized in {"metal", "mps"}:
        return "metal"
    if normalized in {"cpu", "tpu"}:
        return normalized
    return "unknown"


def get_device_memory_capability(device: Any) -> Dict[str, Any]:
    """Describe whether *device* exposes usable JAX allocator statistics."""
    try:
        raw_stats = device.memory_stats()
    except Exception as exc:
        return {
            "memory_stats_available": False,
            "memory_stats": {},
            "memory_stats_error": str(exc),
        }
    if not raw_stats or "bytes_in_use" not in raw_stats:
        return {
            "memory_stats_available": False,
            "memory_stats": dict(raw_stats or {}),
            "memory_stats_error": "JAX device does not expose bytes_in_use",
        }
    return {
        "memory_stats_available": True,
        "memory_stats": dict(raw_stats),
        "memory_stats_error": None,
    }


def resolve_jax_device(selector: Union[int, str] = 0) -> tuple[Any, int]:
    """Resolve a local-device index or a named JAX backend selector.

    Named selectors (``cpu``, ``gpu``, ``tpu``, and ``metal``) select the
    first device exposed by that backend. Numeric selectors preserve the
    historical local-device-index API.
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX not available. Install with `pip install 'stormlog[jax]'`."
        )
    if isinstance(selector, int) or str(selector).isdigit():
        index = int(selector)
        devices = _cached_local_devices()
        if index < 0 or index >= len(devices):
            raise ValueError(
                f"JAX device index {index} is out of range (found {len(devices)})"
            )
        return devices[index], index

    backend = normalize_jax_backend(str(selector))
    if backend == "unknown":
        raise ValueError("JAX device must be an index or one of cpu, gpu, tpu, metal")
    try:
        devices = tuple(jax.devices(backend=backend))
    except Exception as exc:
        raise ValueError(f"JAX backend {backend!r} is unavailable: {exc}") from exc
    if not devices:
        raise ValueError(f"JAX backend {backend!r} has no devices")
    device = devices[0]
    return device, int(getattr(device, "id", 0))


def _device_zero(device: Any) -> Any:
    """Create a scalar zero on a device using JAX's runtime-supported keyword."""

    zeros = cast(Callable[..., Any], jax.numpy.zeros)
    return zeros((), device=device)


@functools.lru_cache(maxsize=1)
def _cached_local_devices() -> tuple:
    """Return ``jax.local_devices()`` cached for the process lifetime.

    JAX device sets are fixed at initialisation, so caching avoids
    repeated runtime calls in utility functions that enumerate devices.
    """
    if not JAX_AVAILABLE:
        return ()
    try:
        return tuple(jax.local_devices())
    except Exception:
        return ()


def jax_is_available() -> bool:
    """Return True when JAX is importable."""
    return JAX_AVAILABLE


_cpu_warning_logged = False


def detect_jax_backend() -> str:
    """Return the active JAX backend name.

    Returns one of 'gpu', 'metal', 'tpu', 'cpu', or 'unknown'. Returns 'cpu'
    as a fallback if JAX is not installed or backend detection fails.
    """
    global _cpu_warning_logged
    if not JAX_AVAILABLE:
        return "cpu"
    try:
        backend = normalize_jax_backend(str(jax.default_backend()))
        if backend == "cpu" and not _cpu_warning_logged:
            logger.info(
                "JAX is running on CPU. Please download specific JAX types "
                "for CUDA or TPU if you want to work with those hardware accelerators."
            )
            _cpu_warning_logged = True
        return backend
    except Exception as exc:
        logger.debug("JAX backend detection failed: %s", exc)
        return "cpu"


def get_device_info(device_index: Union[int, str] = 0) -> Dict[str, Any]:
    """Return device kind, platform, and live memory statistics.

    Args:
        device_index: Local device index or named JAX backend selector.

    Returns:
        Dictionary with keys ``kind``, ``platform``, ``device_id``,
        ``process_index``, ``memory_stats`` (device statistics normalized to a
        dictionary), ``memory_stats_available``, ``memory_stats_error``, and
        ``client``.
    """
    if not JAX_AVAILABLE:
        return {
            "kind": "cpu",
            "platform": "cpu",
            "device_id": 0,
            "process_index": 0,
            "memory_stats": {},
            "memory_stats_available": False,
            "memory_stats_error": "JAX not available",
            "client": None,
            "error": "JAX not available",
        }

    try:
        device, resolved_index = resolve_jax_device(device_index)
        capability = get_device_memory_capability(device)

        return {
            "kind": str(getattr(device, "device_kind", "unknown")),
            "platform": str(device.platform),
            "device_id": getattr(device, "id", resolved_index),
            "process_index": getattr(device, "process_index", 0),
            **capability,
            "client": str(getattr(device, "client", None)),
        }
    except Exception as exc:
        logger.debug("get_device_info failed: %s", exc)
        return {
            "kind": "unknown",
            "platform": detect_jax_backend(),
            "device_id": device_index,
            "process_index": 0,
            "memory_stats": {},
            "memory_stats_available": False,
            "memory_stats_error": str(exc),
            "client": None,
            "error": str(exc),
        }


def get_backend_info() -> Dict[str, Any]:
    """Return backend diagnostics for JAX.

    Returns a dictionary with the JAX runtime backend classification
    and platform details.
    """
    raw_backend = "cpu"
    if JAX_AVAILABLE:
        try:
            raw_backend = str(jax.default_backend())
        except Exception as exc:
            logger.debug("Could not determine raw JAX backend: %s", exc)
    runtime_backend = normalize_jax_backend(raw_backend)
    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    info: Dict[str, Any] = {
        "runtime_backend": runtime_backend,
        "raw_runtime_backend": raw_backend,
        "jax_available": JAX_AVAILABLE,
        "is_gpu_build": runtime_backend in {"gpu", "metal"},
        "is_apple_silicon": is_apple_silicon,
        "jax_metal_active": runtime_backend == "metal",
        "device_count": 0,
        "devices": [],
    }

    if not JAX_AVAILABLE:
        return info

    try:
        devices = _cached_local_devices()
        info["device_count"] = len(devices)
        info["devices"] = [
            {
                "id": getattr(d, "id", i),
                "kind": str(getattr(d, "device_kind", "unknown")),
                "platform": str(d.platform),
            }
            for i, d in enumerate(devices)
        ]
    except Exception as exc:
        logger.debug("Could not enumerate JAX devices: %s", exc)

    return info


def get_system_info() -> Dict[str, Any]:
    """Return full system and JAX environment report.

    Includes JAX version, device list, platform, Python version,
    CPU count, and system memory statistics.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": "Not installed",
        "jax_available": JAX_AVAILABLE,
        "cpu_count": os.cpu_count(),
        "total_memory_gb": 0.0,
        "available_memory_gb": 0.0,
    }

    if JAX_AVAILABLE:
        info["jax_version"] = str(jax.__version__)

    # System memory
    if PSUTIL_AVAILABLE and psutil is not None:
        try:
            memory = psutil.virtual_memory()
            info["total_memory_gb"] = memory.total / (1024**3)
            info["available_memory_gb"] = memory.available / (1024**3)
            info["memory_percent_used"] = memory.percent
        except Exception as exc:
            logger.debug("psutil memory query failed: %s", exc)

    # Backend and device info
    info["backend"] = get_backend_info()
    info["device_info"] = get_device_info()

    return info


def format_memory(bytes_value: Optional[Union[int, float]]) -> str:
    """Format memory size in human-readable format.

    Delegates to :func:`stormlog.utils.format_bytes` when available,
    otherwise provides a standalone implementation.
    """
    if bytes_value is None:
        return "N/A"

    try:
        from stormlog.utils import format_bytes

        return format_bytes(int(bytes_value))
    except (ImportError, Exception):
        pass

    value = float(bytes_value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def validate_jax_environment() -> Dict[str, Any]:
    """Validate JAX environment for memory profiling.

    Returns a dictionary with validation results and a list of any
    issues found.
    """
    issues: List[str] = []
    validation: Dict[str, Any] = {
        "jax_available": JAX_AVAILABLE,
        "gpu_available": False,
        "tpu_available": False,
        "metal_available": False,
        "version_compatible": False,
        "issues": issues,
    }

    if not JAX_AVAILABLE:
        issues.append("JAX not installed")
        return validation

    _check_jax_version(validation, issues)

    # Check device availability
    try:
        backend = detect_jax_backend()
        devices = _cached_local_devices()

        if backend == "gpu":
            validation["gpu_available"] = True
        elif backend == "metal":
            validation["gpu_available"] = True
            validation["metal_available"] = True
        elif backend == "tpu":
            validation["tpu_available"] = True
        elif backend == "cpu":
            if len(devices) > 0:
                # CPU-only is valid but note it
                issues.append(
                    "Only CPU devices found — GPU/TPU memory profiling "
                    "will fall back to psutil"
                )
            else:
                issues.append("No JAX devices found")
        else:
            issues.append(f"Unrecognized JAX backend: {backend}")
    except Exception as exc:
        issues.append(f"Error checking device availability: {exc}")

    return validation


def _check_jax_version(validation: Dict[str, Any], issues: List[str]) -> None:
    # Check JAX version
    try:
        version = jax.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        # Require >= 0.4.0 (pip enforces >=0.4.20 at install time)
        if major > 0 or (major == 0 and minor >= 4):
            validation["version_compatible"] = True
        else:
            issues.append(
                f"JAX {version} may not be fully compatible " "(recommend 0.4.20+)"
            )
    except Exception as exc:
        logger.debug("JAX version check failed: %s", exc)
        issues.append("Could not determine JAX version")
