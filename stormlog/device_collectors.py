"""Backend-aware device memory collector abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union

import torch


@dataclass(frozen=True)
class DeviceMemoryCapabilities:
    """Immutable description of memory signals exposed by a collector."""

    backend: str
    telemetry_collector: str
    sampling_source: str
    supports_allocator_allocated: bool = False
    supports_allocator_reserved: bool = False
    supports_allocator_active: bool = False
    supports_allocator_inactive: bool = False
    supports_device_used: bool = False
    supports_device_free: bool = False
    supports_device_total: bool = False
    supports_native_allocator_history: bool = False
    supports_fragmentation_analysis: bool = False
    supports_allocator_attribution: bool = False
    supports_bounded_profiling: bool = False

    def __post_init__(self) -> None:
        for field_name in ("backend", "telemetry_collector", "sampling_source"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        has_allocator_core = (
            self.supports_allocator_allocated and self.supports_allocator_reserved
        )
        allocator_features = {
            "supports_native_allocator_history": (
                self.supports_native_allocator_history
            ),
            "supports_fragmentation_analysis": self.supports_fragmentation_analysis,
            "supports_allocator_attribution": self.supports_allocator_attribution,
            "supports_bounded_profiling": self.supports_bounded_profiling,
        }
        invalid_features = [
            name for name, enabled in allocator_features.items() if enabled
        ]
        if invalid_features and not has_allocator_core:
            raise ValueError(
                "allocator features require allocated and reserved counters: "
                + ", ".join(invalid_features)
            )
        if not (self.supports_allocator_allocated or self.supports_device_used):
            raise ValueError(
                "collector must support allocator allocated or device used memory"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeviceMemoryCapabilities":
        """Convert legacy capability mappings into the typed contract."""
        return cls(
            backend=str(value.get("backend", "unknown")),
            telemetry_collector=str(value.get("telemetry_collector", "legacy.unknown")),
            sampling_source=str(value.get("sampling_source", "unknown")),
            supports_allocator_allocated=bool(
                value.get("supports_allocator_allocated", True)
            ),
            supports_allocator_reserved=bool(
                value.get("supports_allocator_reserved", True)
            ),
            supports_allocator_active=bool(
                value.get("supports_allocator_active", True)
            ),
            supports_allocator_inactive=bool(
                value.get("supports_allocator_inactive", True)
            ),
            supports_device_used=bool(value.get("supports_device_used", True)),
            supports_device_free=bool(value.get("supports_device_free", False)),
            supports_device_total=bool(value.get("supports_device_total", False)),
            supports_native_allocator_history=bool(
                value.get("supports_native_allocator_history", False)
            ),
            supports_fragmentation_analysis=bool(
                value.get("supports_fragmentation_analysis", True)
            ),
            supports_allocator_attribution=bool(
                value.get("supports_allocator_attribution", True)
            ),
            supports_bounded_profiling=bool(
                value.get("supports_bounded_profiling", False)
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Return a JSON-safe capability mapping for telemetry metadata."""
        return {
            "backend": self.backend,
            "telemetry_collector": self.telemetry_collector,
            "sampling_source": self.sampling_source,
            "supports_allocator_allocated": self.supports_allocator_allocated,
            "supports_allocator_reserved": self.supports_allocator_reserved,
            "supports_allocator_active": self.supports_allocator_active,
            "supports_allocator_inactive": self.supports_allocator_inactive,
            "supports_device_used": self.supports_device_used,
            "supports_device_free": self.supports_device_free,
            "supports_device_total": self.supports_device_total,
            "supports_native_allocator_history": (
                self.supports_native_allocator_history
            ),
            "supports_fragmentation_analysis": self.supports_fragmentation_analysis,
            "supports_allocator_attribution": self.supports_allocator_attribution,
            "supports_bounded_profiling": self.supports_bounded_profiling,
        }


@dataclass(frozen=True)
class DeviceMemorySample:
    """Normalized device-memory sample produced by a backend collector."""

    allocated_bytes: Optional[int]
    reserved_bytes: Optional[int]
    used_bytes: Optional[int]
    free_bytes: Optional[int]
    total_bytes: Optional[int]
    active_bytes: Optional[int]
    inactive_bytes: Optional[int]
    device_id: int


@dataclass(frozen=True)
class DeviceMemorySampleResult:
    """Device-memory sample plus diagnostics about partial/core collection failures."""

    sample: Optional[DeviceMemorySample]
    partial_fields: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    core_error: Optional[str] = None

    @property
    def is_partial(self) -> bool:
        return self.sample is not None and bool(self.partial_fields)

    @property
    def is_core_failure(self) -> bool:
        return self.sample is None


class DeviceMemoryCollector(ABC):
    """Backend-specific collector contract for device memory signals."""

    @abstractmethod
    def name(self) -> str:
        """Return runtime backend name (cuda, rocm, mps)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this collector can sample in the current runtime."""

    @abstractmethod
    def sample(self) -> DeviceMemorySample:
        """Collect a single normalized memory sample."""

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        """Collect a sample while preserving core-failure diagnostics."""
        try:
            return DeviceMemorySampleResult(sample=self.sample())
        except Exception as exc:
            return DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": str(exc)},
                core_error=str(exc),
            )

    @abstractmethod
    def capabilities(self) -> DeviceMemoryCapabilities:
        """Describe backend capability signals for telemetry metadata."""


_SAMPLE_CAPABILITY_FIELDS = {
    "allocated_bytes": "supports_allocator_allocated",
    "reserved_bytes": "supports_allocator_reserved",
    "active_bytes": "supports_allocator_active",
    "inactive_bytes": "supports_allocator_inactive",
    "used_bytes": "supports_device_used",
    "free_bytes": "supports_device_free",
    "total_bytes": "supports_device_total",
}


_SAMPLE_TELEMETRY_FIELDS = {
    "allocated_bytes": "allocator_allocated_bytes",
    "reserved_bytes": "allocator_reserved_bytes",
    "active_bytes": "allocator_active_bytes",
    "inactive_bytes": "allocator_inactive_bytes",
    "used_bytes": "device_used_bytes",
    "free_bytes": "device_free_bytes",
    "total_bytes": "device_total_bytes",
}


def _validate_sample_value(
    value: int | None, supported: bool, telemetry_field: str, partial: set[str]
) -> bool:
    if value is not None:
        if value < 0:
            raise ValueError(f"{telemetry_field} must be >= 0 when provided")
        if not supported:
            raise ValueError(
                f"{telemetry_field} was provided by a collector that declares "
                "it unsupported"
            )
        if telemetry_field in partial:
            raise ValueError(f"{telemetry_field} is populated but marked partial")
        return True
    if supported and telemetry_field not in partial:
        raise ValueError(
            f"{telemetry_field} is missing without a partial-field diagnostic"
        )
    if not supported and telemetry_field in partial:
        raise ValueError(
            f"{telemetry_field} is unsupported and must not be marked partial"
        )
    return False


def validate_device_memory_sample(
    sample: DeviceMemorySample,
    capabilities: DeviceMemoryCapabilities,
    *,
    partial_fields: tuple[str, ...] = (),
) -> None:
    """Validate a sample against its collector's declared capabilities."""
    partial = set(partial_fields)
    known_telemetry_fields = set(_SAMPLE_TELEMETRY_FIELDS.values())
    unknown_partial_fields = sorted(partial - known_telemetry_fields)
    if unknown_partial_fields:
        raise ValueError(
            "unknown partial sample fields: " + ", ".join(unknown_partial_fields)
        )
    available_values = 0
    for sample_field, capability_field in _SAMPLE_CAPABILITY_FIELDS.items():
        value = getattr(sample, sample_field)
        supported = bool(getattr(capabilities, capability_field))
        telemetry_field = _SAMPLE_TELEMETRY_FIELDS[sample_field]
        if _validate_sample_value(value, supported, telemetry_field, partial):
            available_values += 1

    if available_values == 0:
        raise ValueError("device memory sample contains no supported measurements")
    if sample.total_bytes is not None:
        if sample.used_bytes is not None and sample.used_bytes > sample.total_bytes:
            raise ValueError("device_used_bytes cannot exceed device_total_bytes")
        if sample.free_bytes is not None and sample.free_bytes > sample.total_bytes:
            raise ValueError("device_free_bytes cannot exceed device_total_bytes")


def _is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return bool(mps_backend.is_available())
    except Exception:
        return False


def _is_rocm_runtime() -> bool:
    hip_version = getattr(torch.version, "hip", None)
    return bool(torch.cuda.is_available() and hip_version)


def detect_torch_runtime_backend() -> str:
    """Return the active torch runtime backend in this environment."""
    if torch.cuda.is_available():
        return "rocm" if _is_rocm_runtime() else "cuda"
    if _is_mps_available():
        return "mps"
    return "cpu"


def _resolve_device(device: Union[str, int, torch.device, None]) -> torch.device:
    if device is None:
        backend = detect_torch_runtime_backend()
        if backend in {"cuda", "rocm"}:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if backend == "mps":
            return torch.device("mps")
        raise RuntimeError("No supported GPU backend is available")
    if isinstance(device, int):
        backend = detect_torch_runtime_backend()
        if backend not in {"cuda", "rocm"}:
            raise ValueError(
                "Integer device IDs are only supported for CUDA/ROCm backends"
            )
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    return device


class CudaDeviceCollector(DeviceMemoryCollector):
    """Collector for NVIDIA CUDA runtime memory counters."""

    telemetry_collector = "stormlog.cuda_tracker"

    def __init__(self, device: Union[str, int, torch.device, None] = None) -> None:
        self.device = _resolve_device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA collector requires a CUDA device")

    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available() and not _is_rocm_runtime())

    def sample(self) -> DeviceMemorySample:
        result = self.sample_with_diagnostics()
        if result.sample is None:
            raise RuntimeError(result.core_error or "CUDA sample collection failed")
        return result.sample

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        try:
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
        except Exception as exc:
            return DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": str(exc)},
                core_error=str(exc),
            )

        used = max(allocated, reserved)
        total: Optional[int] = None
        free: Optional[int] = None
        active: Optional[int] = None
        inactive: Optional[int] = None
        partial_fields: list[str] = []
        errors: dict[str, str] = {}

        try:
            total = int(torch.cuda.get_device_properties(self.device).total_memory)
            free = max(total - used, 0)
        except Exception as exc:
            message = str(exc)
            partial_fields.extend(["device_total_bytes", "device_free_bytes"])
            errors["device_total_bytes"] = message
            errors["device_free_bytes"] = message

        try:
            stats = torch.cuda.memory_stats(self.device)
            active = int(stats.get("active_bytes.all.current", 0))
            inactive = int(stats.get("inactive_split_bytes.all.current", 0))
        except Exception as exc:
            message = str(exc)
            partial_fields.extend(
                ["allocator_active_bytes", "allocator_inactive_bytes"]
            )
            errors["allocator_active_bytes"] = message
            errors["allocator_inactive_bytes"] = message

        return DeviceMemorySampleResult(
            sample=DeviceMemorySample(
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                used_bytes=used,
                free_bytes=free,
                total_bytes=total,
                active_bytes=active,
                inactive_bytes=inactive,
                device_id=device_index,
            ),
            partial_fields=tuple(dict.fromkeys(partial_fields)),
            errors=errors,
        )

    def capabilities(self) -> DeviceMemoryCapabilities:
        return DeviceMemoryCapabilities(
            backend=self.name(),
            telemetry_collector=self.telemetry_collector,
            sampling_source="torch.cuda.memory_allocated/reserved",
            supports_allocator_allocated=True,
            supports_allocator_reserved=True,
            supports_allocator_active=True,
            supports_allocator_inactive=True,
            supports_device_used=True,
            supports_device_free=True,
            supports_device_total=True,
            supports_native_allocator_history=True,
            supports_fragmentation_analysis=True,
            supports_allocator_attribution=True,
            supports_bounded_profiling=True,
        )


class ROCmDeviceCollector(CudaDeviceCollector):
    """Collector for ROCm runtimes surfaced through torch.cuda APIs."""

    telemetry_collector = "stormlog.rocm_tracker"

    def name(self) -> str:
        return "rocm"

    def is_available(self) -> bool:
        return _is_rocm_runtime()

    def capabilities(self) -> DeviceMemoryCapabilities:
        return DeviceMemoryCapabilities(
            **{
                **super().capabilities().to_metadata(),
                "backend": self.name(),
                "sampling_source": "torch.cuda.memory_* (HIP runtime)",
                "telemetry_collector": self.telemetry_collector,
            }
        )


class MPSDeviceCollector(DeviceMemoryCollector):
    """Collector for Apple Metal (MPS) runtime counters."""

    telemetry_collector = "stormlog.mps_tracker"

    def __init__(self, device: Union[str, int, torch.device, None] = None) -> None:
        resolved = _resolve_device(device)
        if resolved.type != "mps":
            raise ValueError("MPS collector requires an MPS device")
        self.device = resolved

    def name(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        return _is_mps_available()

    def sample(self) -> DeviceMemorySample:
        result = self.sample_with_diagnostics()
        if result.sample is None:
            raise RuntimeError(result.core_error or "MPS sample collection failed")
        return result.sample

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        import torch.mps as torch_mps

        try:
            allocated = int(torch_mps.current_allocated_memory())
            reserved = int(torch_mps.driver_allocated_memory())
        except Exception as exc:
            return DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": str(exc)},
                core_error=str(exc),
            )

        used = max(allocated, reserved)

        total: Optional[int] = None
        partial_fields: list[str] = []
        errors: dict[str, str] = {}
        if hasattr(torch_mps, "recommended_max_memory"):
            try:
                # MPS does not expose a strict physical-total API here; this is the
                # best runtime approximation currently available from torch.
                raw_total = int(torch_mps.recommended_max_memory())
                total = raw_total if raw_total > 0 else None
                if total is None:
                    partial_fields.extend(["device_total_bytes", "device_free_bytes"])
            except Exception as exc:
                message = str(exc)
                partial_fields.extend(["device_total_bytes", "device_free_bytes"])
                errors["device_total_bytes"] = message
                errors["device_free_bytes"] = message
        free = max(total - used, 0) if total is not None else None

        return DeviceMemorySampleResult(
            sample=DeviceMemorySample(
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                used_bytes=used,
                free_bytes=free,
                total_bytes=total,
                active_bytes=None,
                inactive_bytes=None,
                device_id=0,
            ),
            partial_fields=tuple(dict.fromkeys(partial_fields)),
            errors=errors,
        )

    def capabilities(self) -> DeviceMemoryCapabilities:
        import torch.mps as torch_mps

        supports_total = hasattr(torch_mps, "recommended_max_memory")
        return DeviceMemoryCapabilities(
            backend=self.name(),
            telemetry_collector=self.telemetry_collector,
            sampling_source=(
                "torch.mps.current_allocated_memory/driver_allocated_memory"
            ),
            supports_allocator_allocated=True,
            supports_allocator_reserved=True,
            supports_device_used=True,
            supports_device_free=supports_total,
            supports_device_total=supports_total,
            supports_fragmentation_analysis=True,
        )


def build_device_memory_collector(
    device: Union[str, int, torch.device, None] = None,
) -> DeviceMemoryCollector:
    """Build a backend collector for CUDA/ROCm/MPS runtime environments."""
    resolved = _resolve_device(device)
    if resolved.type == "cuda":
        if _is_rocm_runtime():
            return ROCmDeviceCollector(resolved)
        return CudaDeviceCollector(resolved)
    if resolved.type == "mps":
        return MPSDeviceCollector(resolved)
    raise ValueError("Only CUDA/ROCm and MPS devices are supported for tracking")


__all__ = [
    "DeviceMemoryCollector",
    "DeviceMemoryCapabilities",
    "DeviceMemorySample",
    "DeviceMemorySampleResult",
    "CudaDeviceCollector",
    "ROCmDeviceCollector",
    "MPSDeviceCollector",
    "build_device_memory_collector",
    "detect_torch_runtime_backend",
    "validate_device_memory_sample",
]
