"""Backend-neutral projection over the persisted telemetry event schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Optional, cast

# Bump this only with the docs, dataclass annotation, serialization tests, and
# compatibility behavior for the projected envelope.
TELEMETRY_PROJECTION_SCHEMA_VERSION: Literal[1] = 1

_INFERRED_COLLECTOR_SOURCE_KINDS = ("cuda", "rocm", "mps", "cpu", "tensorflow")

_SEVERITY_BY_EVENT_TYPE = {
    "critical": "critical",
    "error": "error",
    "warning": "warning",
    "collector_degraded": "warning",
    "collector_recovered": "info",
    "start": "info",
    "stop": "info",
    "phase_enter": "info",
    "phase_exit": "info",
    "sample": "info",
}


@dataclass(frozen=True)
class ProjectedTelemetryRecord:
    """Small immutable event envelope shared by live and loaded telemetry."""

    schema_version: Literal[1]
    record_id: str
    timestamp_ns: int
    observed_timestamp_ns: int
    session_id: str
    source_kind: str
    event_type: str
    stage: Optional[str]
    severity: Optional[str]
    severity_text: Optional[str]
    body: Optional[str]
    resource: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    correlation: Mapping[str, Any] = field(default_factory=dict)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_value(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in payload.items()}
    )


def _thaw_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _thaw_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(_thaw_value(payload), sort_keys=True, default=str))
    return cast(dict[str, Any], result)


def _record_id(record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(_jsonable_mapping(record), sort_keys=True).encode("utf-8")
    ).hexdigest()
    # The 32-character prefix keeps a compact 128-bit identity while preserving
    # deterministic grouping for the projected envelope.
    return f"telemetry-{digest[:32]}"


def _source_kind(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    backend = metadata.get("backend")
    if isinstance(backend, str) and backend.strip():
        return backend.strip().lower()

    collector = record.get("collector")
    if isinstance(collector, str):
        lowered = collector.lower()
        for candidate in _INFERRED_COLLECTOR_SOURCE_KINDS:
            if candidate in lowered:
                return candidate

    device_id = record.get("device_id")
    if isinstance(device_id, int) and device_id >= 0:
        return "gpu"
    return "other"


def _phase_stage(metadata: Mapping[str, Any]) -> Optional[str]:
    phase_scope = metadata.get("phase_scope")
    if not isinstance(phase_scope, Mapping):
        return None
    name = phase_scope.get("name")
    if isinstance(name, str) and name.strip():
        return name
    path = phase_scope.get("path")
    if isinstance(path, list) and path:
        tail = path[-1]
        if isinstance(tail, str) and tail.strip():
            return tail
    return None


def _severity(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> Optional[str]:
    explicit = metadata.get("severity")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    event_type = record.get("event_type")
    if isinstance(event_type, str):
        return _SEVERITY_BY_EVENT_TYPE.get(event_type.strip().lower())
    return None


def _resource(record: Mapping[str, Any], source_kind: str) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "collector": record.get("collector"),
        "host": record.get("host"),
        "pid": record.get("pid"),
        "device_id": record.get("device_id"),
        "job_id": record.get("job_id"),
        "rank": record.get("rank"),
        "local_rank": record.get("local_rank"),
        "world_size": record.get("world_size"),
    }


def _attributes(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    attributes = dict(metadata)
    attributes.update(
        {
            "sampling.interval_ms": record.get("sampling_interval_ms"),
            "memory.allocator.allocated_bytes": record.get("allocator_allocated_bytes"),
            "memory.allocator.reserved_bytes": record.get("allocator_reserved_bytes"),
            "memory.allocator.active_bytes": record.get("allocator_active_bytes"),
            "memory.allocator.inactive_bytes": record.get("allocator_inactive_bytes"),
            "memory.allocator.change_bytes": record.get("allocator_change_bytes"),
            "memory.device.used_bytes": record.get("device_used_bytes"),
            "memory.device.free_bytes": record.get("device_free_bytes"),
            "memory.device.total_bytes": record.get("device_total_bytes"),
        }
    )
    return attributes


def _correlation(
    record: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    correlation = {
        "session_id": record.get("session_id"),
        "job_id": record.get("job_id"),
        "rank": record.get("rank"),
        "local_rank": record.get("local_rank"),
        "world_size": record.get("world_size"),
    }
    phase_scope = metadata.get("phase_scope")
    if isinstance(phase_scope, Mapping):
        for key in ("scope_id", "parent_scope_id", "sequence"):
            value = phase_scope.get(key)
            if value is not None:
                correlation[f"phase.{key}"] = value
    return correlation


def project_telemetry_mapping(
    record: Mapping[str, Any],
    *,
    observed_timestamp_ns: int | None = None,
) -> ProjectedTelemetryRecord:
    """Project a normalized telemetry mapping into the projected envelope.

    Args:
        record: Existing normalized telemetry record, normally a
            `TelemetryEvent v3` dictionary.
        observed_timestamp_ns: Optional observation timestamp. Defaults to the
            source timestamp when no separate observation time is available.

    Returns:
        Backend-neutral projected telemetry record.

    Raises:
        ValueError: If required identity or timestamp fields are missing.
    """

    session_id, timestamp_ns = _projection_identity(record)

    _validate_observed_timestamp(observed_timestamp_ns)

    event_type = record.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("projected telemetry record requires event_type")

    metadata_value = record.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise ValueError("projected telemetry record metadata must be a mapping")
    metadata = dict(metadata_value)
    resolved_source_kind = _source_kind(record, metadata)
    body = record.get("context")

    return ProjectedTelemetryRecord(
        schema_version=TELEMETRY_PROJECTION_SCHEMA_VERSION,
        record_id=_record_id(record),
        timestamp_ns=timestamp_ns,
        observed_timestamp_ns=(
            timestamp_ns if observed_timestamp_ns is None else observed_timestamp_ns
        ),
        session_id=session_id,
        source_kind=resolved_source_kind,
        event_type=event_type,
        stage=_phase_stage(metadata),
        severity=_severity(record, metadata),
        severity_text=(
            str(metadata["severity_text"])
            if metadata.get("severity_text") is not None
            else None
        ),
        body=str(body) if body is not None else None,
        resource=_freeze_mapping(_resource(record, resolved_source_kind)),
        attributes=_freeze_mapping(_attributes(record, metadata)),
        correlation=_freeze_mapping(_correlation(record, metadata)),
    )


def projected_record_to_dict(record: ProjectedTelemetryRecord) -> dict[str, Any]:
    """Serialize a projected telemetry record to a deterministic dictionary."""

    return {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "timestamp_ns": record.timestamp_ns,
        "observed_timestamp_ns": record.observed_timestamp_ns,
        "session_id": record.session_id,
        "source_kind": record.source_kind,
        "event_type": record.event_type,
        "stage": record.stage,
        "severity": record.severity,
        "severity_text": record.severity_text,
        "body": record.body,
        "resource": _thaw_value(record.resource),
        "attributes": _thaw_value(record.attributes),
        "correlation": _thaw_value(record.correlation),
    }


def unique_projected_resources(
    records: Iterable[ProjectedTelemetryRecord],
) -> list[dict[str, Any]]:
    """Return stable unique resource dictionaries for projected telemetry records."""

    seen: set[str] = set()
    resources: list[dict[str, Any]] = []
    for record in records:
        resource = cast(dict[str, Any], _thaw_value(record.resource))
        key = json.dumps(_jsonable_mapping(resource), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        resources.append(resource)
    return resources


def unique_projected_correlations(
    records: Iterable[ProjectedTelemetryRecord],
) -> list[dict[str, Any]]:
    """Return stable unique correlation dictionaries for projected telemetry records."""

    seen: set[str] = set()
    correlations: list[dict[str, Any]] = []
    for record in records:
        correlation = cast(dict[str, Any], _thaw_value(record.correlation))
        key = json.dumps(_jsonable_mapping(correlation), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        correlations.append(correlation)
    return correlations


def _validate_observed_timestamp(observed_timestamp_ns: int | None) -> None:
    if observed_timestamp_ns is not None:
        if not isinstance(observed_timestamp_ns, int) or isinstance(
            observed_timestamp_ns, bool
        ):
            raise ValueError(
                "projected telemetry record requires integer observed_timestamp_ns"
            )
        if observed_timestamp_ns < 0:
            raise ValueError(
                "projected telemetry record requires non-negative observed_timestamp_ns"
            )


def _projection_identity(record: Mapping[str, Any]) -> tuple[str, int]:
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("projected telemetry record requires session_id")

    timestamp_ns = record.get("timestamp_ns")
    if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
        raise ValueError("projected telemetry record requires integer timestamp_ns")

    return session_id, timestamp_ns


__all__ = [
    "TELEMETRY_PROJECTION_SCHEMA_VERSION",
    "ProjectedTelemetryRecord",
    "project_telemetry_mapping",
    "projected_record_to_dict",
    "unique_projected_correlations",
    "unique_projected_resources",
]
