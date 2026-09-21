"""Portable contracts for optional, out-of-process native trace helpers.

This module does not load a native library or collect GPU activity. It defines
the boundary that a separately distributed helper can implement and reports
whether an installed host is a plausible target for such a helper.
"""

from __future__ import annotations

import ctypes.util
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .collector_health import CollectorHealthState

NATIVE_TRACE_FORMAT = "stormlog.native_trace"
NATIVE_TRACE_SCHEMA_VERSION = 1
NATIVE_HELPER_PROTOCOL_VERSION = 1

NativeBackend = Literal["cupti_activity", "rocprofiler"]
CapabilityStatus = Literal["available", "unavailable", "unsupported"]
HelperMessageType = Literal[
    "hello",
    "capabilities",
    "start",
    "started",
    "status",
    "flush",
    "stop",
    "stopped",
    "error",
]
HelperTermination = Literal[
    "completed",
    "partial",
    "cancelled",
    "timed_out",
    "failed",
    "incompatible",
]

_BACKENDS: tuple[NativeBackend, ...] = ("cupti_activity", "rocprofiler")
_HELPER_MESSAGE_TYPES = {
    "hello",
    "capabilities",
    "start",
    "started",
    "status",
    "flush",
    "stop",
    "stopped",
    "error",
}
_MESSAGE_REQUIRED_FIELDS = {
    "hello": {"helper_version", "executable", "supported_protocol_versions"},
    "capabilities": {"backends"},
    "start": {
        "capture_id",
        "backend",
        "output_directory",
        "max_bytes",
        "requested_activities",
        "target_selector",
    },
    "started": {"capture_id", "helper_pid"},
    "status": {"capture_id", "status", "delivered_records", "dropped_records"},
    "flush": {"capture_id"},
    "stop": {"capture_id"},
    "stopped": {"capture_id", "flush_outcome"},
    "error": {"code", "message", "retryable"},
}


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_helper_payload(
    message_type: HelperMessageType, payload: Mapping[str, Any]
) -> None:
    missing = _MESSAGE_REQUIRED_FIELDS[message_type] - payload.keys()
    if missing:
        raise ValueError(
            f"{message_type} payload is missing: {', '.join(sorted(missing))}"
        )
    for name in _message_string_fields(message_type):
        _require_non_empty(payload[name], name)
    validator = _payload_validators().get(message_type)
    if validator is not None:
        validator(payload)


def _message_string_fields(message_type: HelperMessageType) -> tuple[str, ...]:
    fields = {
        "hello": ("helper_version", "executable"),
        "capabilities": (),
        "start": ("capture_id", "backend", "output_directory", "target_selector"),
        "started": ("capture_id",),
        "status": ("capture_id", "status"),
        "flush": ("capture_id",),
        "stop": ("capture_id",),
        "stopped": ("capture_id", "flush_outcome"),
        "error": ("code", "message"),
    }
    return fields[message_type]


def _validate_status_payload(payload: Mapping[str, Any]) -> None:
    if payload["status"] not in {"healthy", "degraded", "unhealthy"}:
        raise ValueError("status payload has an unsupported health status")
    for name in ("delivered_records", "dropped_records"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _validate_start_payload(payload: Mapping[str, Any]) -> None:
    max_bytes = payload["max_bytes"]
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")


def _validate_started_payload(payload: Mapping[str, Any]) -> None:
    helper_pid = payload["helper_pid"]
    if (
        not isinstance(helper_pid, int)
        or isinstance(helper_pid, bool)
        or helper_pid <= 0
    ):
        raise ValueError("helper_pid must be a positive integer")


def _validate_error_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload["retryable"], bool):
        raise ValueError("retryable must be a boolean")


def _payload_validators() -> Mapping[str, Callable[[Mapping[str, Any]], None]]:
    return {
        "start": _validate_start_payload,
        "started": _validate_started_payload,
        "status": _validate_status_payload,
        "error": _validate_error_payload,
    }


@dataclass(frozen=True)
class NativeTraceCapability:
    """One backend's preflight result without loading its native library."""

    backend: NativeBackend
    status: CapabilityStatus
    reason: str
    operating_system: str
    architecture: str
    library_path: str | None = None
    required_privilege: str = "same-process-or-explicit-target-access"

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"unsupported native backend: {self.backend}")
        if self.status not in {"available", "unavailable", "unsupported"}:
            raise ValueError(f"unsupported capability status: {self.status}")
        for value, name in (
            (self.reason, "reason"),
            (self.operating_system, "operating_system"),
            (self.architecture, "architecture"),
            (self.required_privilege, "required_privilege"),
        ):
            _require_non_empty(value, name)
        if self.status == "available" and self.library_path is None:
            raise ValueError("available capability must identify a library path")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe capability record."""
        return {
            "backend": self.backend,
            "status": self.status,
            "reason": self.reason,
            "operating_system": self.operating_system,
            "architecture": self.architecture,
            "library_path": self.library_path,
            "required_privilege": self.required_privilege,
        }


@dataclass(frozen=True)
class NativeHelperMessage:
    """One versioned request or response on the native-helper control channel."""

    message_type: HelperMessageType
    request_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: int = NATIVE_HELPER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != NATIVE_HELPER_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported native helper protocol version: "
                f"{self.protocol_version}"
            )
        if self.message_type not in _HELPER_MESSAGE_TYPES:
            raise ValueError(f"unsupported helper message type: {self.message_type}")
        _require_non_empty(self.request_id, "request_id")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        _validate_helper_payload(self.message_type, self.payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe helper message."""
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type,
            "request_id": self.request_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeHelperMessage":
        """Parse a helper message and reject loose coercions."""
        protocol_version = payload.get("protocol_version")
        message_type = payload.get("message_type")
        request_id = payload.get("request_id")
        message_payload = payload.get("payload")
        if not isinstance(protocol_version, int) or isinstance(protocol_version, bool):
            raise ValueError("protocol_version must be an integer")
        if not isinstance(message_type, str):
            raise ValueError("message_type must be a string")
        if not isinstance(request_id, str):
            raise ValueError("request_id must be a string")
        if not isinstance(message_payload, Mapping):
            raise ValueError("payload must be an object")
        return cls(
            protocol_version=protocol_version,
            message_type=message_type,  # type: ignore[arg-type]
            request_id=request_id,
            payload=message_payload,
        )


@dataclass(frozen=True)
class NativeHelperOutcome:
    """Failure-isolated helper termination translated to collector health."""

    termination: HelperTermination
    error: str | None = None
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        supported = {
            "completed",
            "partial",
            "cancelled",
            "timed_out",
            "failed",
            "incompatible",
        }
        if self.termination not in supported:
            raise ValueError(f"unsupported helper termination: {self.termination}")
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must be >= 0")
        if self.termination == "completed" and self.error is not None:
            raise ValueError("completed helper outcome cannot contain an error")
        if self.termination != "completed" and not self.error:
            raise ValueError("non-completed helper outcome requires an error")

    def to_health(self) -> CollectorHealthState:
        """Map helper termination without raising into the default monitor."""
        if self.termination == "completed":
            return CollectorHealthState()
        status = (
            "degraded" if self.termination in {"partial", "cancelled"} else "unhealthy"
        )
        return CollectorHealthState(
            status=status,
            telemetry_partial=True,
            partial_fields=("native_trace",),
            last_error=self.error,
            consecutive_failures=max(self.consecutive_failures, 1),
        )


@dataclass(frozen=True)
class NativeTraceRecord:
    """Normalized evidence reference without assuming a single clock domain."""

    record_id: str
    activity_kind: str
    clock_domain: str
    provenance: str
    uncertainty: str
    cpu_start_ns: int | None = None
    cpu_end_ns: int | None = None
    device_start_ns: int | None = None
    device_end_ns: int | None = None
    correlation_id: str | None = None
    stream_id: str | None = None
    graph_id: str | None = None
    graph_node_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for required_value, name in (
            (self.record_id, "record_id"),
            (self.activity_kind, "activity_kind"),
            (self.clock_domain, "clock_domain"),
            (self.provenance, "provenance"),
            (self.uncertainty, "uncertainty"),
        ):
            _require_non_empty(required_value, name)
        for timestamp, name in (
            (self.cpu_start_ns, "cpu_start_ns"),
            (self.cpu_end_ns, "cpu_end_ns"),
            (self.device_start_ns, "device_start_ns"),
            (self.device_end_ns, "device_end_ns"),
        ):
            _require_non_negative(timestamp, name)
        _validate_interval(self.cpu_start_ns, self.cpu_end_ns, "CPU")
        _validate_interval(self.device_start_ns, self.device_end_ns, "device")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trace record."""
        return {
            "schema_version": NATIVE_TRACE_SCHEMA_VERSION,
            "record_id": self.record_id,
            "activity_kind": self.activity_kind,
            "clock_domain": self.clock_domain,
            "cpu_start_ns": self.cpu_start_ns,
            "cpu_end_ns": self.cpu_end_ns,
            "device_start_ns": self.device_start_ns,
            "device_end_ns": self.device_end_ns,
            "correlation_id": self.correlation_id,
            "stream_id": self.stream_id,
            "graph_id": self.graph_id,
            "graph_node_id": self.graph_node_id,
            "provenance": self.provenance,
            "uncertainty": self.uncertainty,
            "metadata": dict(self.metadata),
        }


def _validate_interval(start_ns: int | None, end_ns: int | None, name: str) -> None:
    if (start_ns is None) != (end_ns is None):
        raise ValueError(f"{name} interval requires both start and end")
    if start_ns is not None and end_ns is not None and end_ns < start_ns:
        raise ValueError(f"{name} interval end must be >= start")


def native_trace_preflight(
    *,
    system: str | None = None,
    architecture: str | None = None,
    library_search_paths: Sequence[str | Path] = (),
) -> tuple[NativeTraceCapability, ...]:
    """Report plausible native backends without loading any native code.

    A library's presence is not proof that its driver, device, ABI, permission,
    or requested activities work. The future helper must perform its own active
    capability handshake before capture.
    """
    host_system = system or platform.system()
    host_architecture = architecture or platform.machine()
    return tuple(
        _backend_preflight(
            backend,
            host_system,
            host_architecture,
            library_search_paths,
        )
        for backend in _BACKENDS
    )


def _backend_preflight(
    backend: NativeBackend,
    system: str,
    architecture: str,
    search_paths: Sequence[str | Path],
) -> NativeTraceCapability:
    supported_systems = {
        "cupti_activity": {"Linux", "Windows"},
        "rocprofiler": {"Linux"},
    }
    if system not in supported_systems[backend]:
        return NativeTraceCapability(
            backend=backend,
            status="unsupported",
            reason=f"{backend} is not supported by this contract on {system}",
            operating_system=system,
            architecture=architecture,
        )
    library = _find_native_library(backend, search_paths)
    if library is None:
        return NativeTraceCapability(
            backend=backend,
            status="unavailable",
            reason=(
                "native library was not found; install the matching toolkit and "
                "run the helper capability handshake"
            ),
            operating_system=system,
            architecture=architecture,
        )
    return NativeTraceCapability(
        backend=backend,
        status="available",
        reason=(
            "native library was found but device, driver, ABI, permission, and "
            "activity support remain unverified"
        ),
        operating_system=system,
        architecture=architecture,
        library_path=library,
    )


def _find_native_library(
    backend: NativeBackend, search_paths: Sequence[str | Path]
) -> str | None:
    names = {
        "cupti_activity": ("cupti", "libcupti.so", "cupti64.dll"),
        "rocprofiler": ("rocprofiler-sdk", "librocprofiler-sdk.so"),
    }[backend]
    for root_value in search_paths:
        root = Path(root_value)
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return str(candidate.resolve())
    discovered = ctypes.util.find_library(names[0])
    return discovered or None


__all__ = [
    "NATIVE_HELPER_PROTOCOL_VERSION",
    "NATIVE_TRACE_FORMAT",
    "NATIVE_TRACE_SCHEMA_VERSION",
    "NativeBackend",
    "NativeHelperMessage",
    "NativeHelperOutcome",
    "NativeTraceCapability",
    "NativeTraceRecord",
    "native_trace_preflight",
]
