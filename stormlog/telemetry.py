"""Canonical telemetry event schema and legacy conversion helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Union

from .session import (
    SESSION_STATUS_INCOMPLETE,
    SessionSummary,
    infer_session_summary_from_events,
    select_default_loaded_session,
    sort_session_summaries,
    stable_legacy_session_id,
)
from .telemetry_model import (
    ProjectedTelemetryRecord,
    project_telemetry_mapping,
    unique_projected_correlations,
    unique_projected_resources,
)
from .telemetry_sink import (
    read_telemetry_sink_manifest,
    resolve_telemetry_sink_segment_paths,
)

SCHEMA_VERSION_V2: Literal[2] = 2
SCHEMA_VERSION_V3: Literal[3] = 3
SCHEMA_VERSION_V4: Literal[4] = 4
SCHEMA_VERSION_LATEST: Literal[4] = SCHEMA_VERSION_V4
UNKNOWN_PID = -1
UNKNOWN_HOST = "unknown"

_LEGACY_BACKEND_COLLECTORS = {
    "mps": "stormlog.mps_tracker",
    "rocm": "stormlog.rocm_tracker",
    "cuda": "stormlog.cuda_tracker",
    "cpu": "stormlog.cpu_tracker",
}

REQUIRED_V3_FIELDS = (
    "schema_version",
    "session_id",
    "timestamp_ns",
    "event_type",
    "collector",
    "sampling_interval_ms",
    "pid",
    "host",
    "device_id",
    "allocator_allocated_bytes",
    "allocator_reserved_bytes",
    "allocator_active_bytes",
    "allocator_inactive_bytes",
    "allocator_change_bytes",
    "device_used_bytes",
    "device_free_bytes",
    "device_total_bytes",
    "context",
    "metadata",
)
OPTIONAL_V3_FIELDS = (
    "job_id",
    "rank",
    "local_rank",
    "world_size",
)
REQUIRED_V2_FIELDS = tuple(
    field_name for field_name in REQUIRED_V3_FIELDS if field_name != "session_id"
)
OPTIONAL_V2_FIELDS = OPTIONAL_V3_FIELDS
KNOWN_V2_FIELD_SET = frozenset(REQUIRED_V2_FIELDS + OPTIONAL_V2_FIELDS)
KNOWN_V3_FIELD_SET = frozenset(REQUIRED_V3_FIELDS + OPTIONAL_V3_FIELDS)
REQUIRED_V4_FIELDS = REQUIRED_V3_FIELDS
OPTIONAL_V4_FIELDS = OPTIONAL_V3_FIELDS
KNOWN_V4_FIELD_SET = frozenset(REQUIRED_V4_FIELDS + OPTIONAL_V4_FIELDS)
_DISTRIBUTED_METADATA_KEYS = frozenset(OPTIONAL_V3_FIELDS)
_SESSION_METADATA_KEYS = frozenset({"session_id"})
_RANK_ENV_GROUPS = (
    ("RANK", "LOCAL_RANK", "WORLD_SIZE"),
    (
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_SIZE",
    ),
    ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"),
)
_JOB_ID_ENV_KEYS = ("TORCHELASTIC_RUN_ID", "SLURM_JOB_ID")
_MEMORY_CAPABILITY_BOOLEAN_FIELDS = (
    "supports_allocator_allocated",
    "supports_allocator_reserved",
    "supports_allocator_active",
    "supports_allocator_inactive",
    "supports_device_used",
    "supports_device_free",
    "supports_device_total",
    "supports_native_allocator_history",
    "supports_fragmentation_analysis",
    "supports_allocator_attribution",
    "supports_bounded_profiling",
)
_MEMORY_CAPABILITY_STRING_FIELDS = (
    "backend",
    "telemetry_collector",
    "sampling_source",
)
_MEMORY_CAPABILITY_FIELD_SET = frozenset(
    _MEMORY_CAPABILITY_STRING_FIELDS + _MEMORY_CAPABILITY_BOOLEAN_FIELDS
)
_COUNTER_CAPABILITY_FIELDS = {
    "allocator_allocated_bytes": "supports_allocator_allocated",
    "allocator_reserved_bytes": "supports_allocator_reserved",
    "allocator_active_bytes": "supports_allocator_active",
    "allocator_inactive_bytes": "supports_allocator_inactive",
    "device_used_bytes": "supports_device_used",
    "device_free_bytes": "supports_device_free",
    "device_total_bytes": "supports_device_total",
}


@dataclass
class TelemetryEventV2:
    """Legacy v2 telemetry event payload retained for backward-compatible writes/tests."""

    schema_version: Literal[2]
    timestamp_ns: int
    event_type: str
    collector: str
    sampling_interval_ms: int
    pid: int
    host: str
    device_id: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    allocator_active_bytes: Optional[int]
    allocator_inactive_bytes: Optional[int]
    allocator_change_bytes: int
    device_used_bytes: int
    device_free_bytes: Optional[int]
    device_total_bytes: Optional[int]
    context: Optional[str]
    job_id: Optional[str] = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_telemetry_record(telemetry_event_to_dict(self))


@dataclass
class TelemetryEventV3:
    """Legacy session-aware telemetry payload with required core counters."""

    schema_version: Literal[3]
    session_id: str
    timestamp_ns: int
    event_type: str
    collector: str
    sampling_interval_ms: int
    pid: int
    host: str
    device_id: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    allocator_active_bytes: Optional[int]
    allocator_inactive_bytes: Optional[int]
    allocator_change_bytes: int
    device_used_bytes: int
    device_free_bytes: Optional[int]
    device_total_bytes: Optional[int]
    context: Optional[str]
    job_id: Optional[str] = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_telemetry_record(telemetry_event_to_dict(self))


@dataclass
class TelemetryEventV4:
    """Capability-aware telemetry payload used by tracker exports and loaders."""

    schema_version: Literal[4]
    session_id: str
    timestamp_ns: int
    event_type: str
    collector: str
    sampling_interval_ms: int
    pid: int
    host: str
    device_id: int
    allocator_allocated_bytes: Optional[int]
    allocator_reserved_bytes: Optional[int]
    allocator_active_bytes: Optional[int]
    allocator_inactive_bytes: Optional[int]
    allocator_change_bytes: Optional[int]
    device_used_bytes: Optional[int]
    device_free_bytes: Optional[int]
    device_total_bytes: Optional[int]
    context: Optional[str]
    job_id: Optional[str] = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_telemetry_record(telemetry_event_to_dict(self))


TelemetryEvent = TelemetryEventV4
TelemetryEventLike = Union[TelemetryEventV2, TelemetryEventV3, TelemetryEventV4]


@dataclass
class LoadedTelemetrySession:
    """Grouped telemetry records and lifecycle metadata for one session."""

    summary: SessionSummary
    events: list[TelemetryEvent]
    sources_loaded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def telemetry_records(self) -> list[ProjectedTelemetryRecord]:
        """Return backend-neutral projected records for this loaded session."""

        return project_telemetry_events(self.events)

    def resources(self) -> list[dict[str, Any]]:
        """Return unique observed resources for this loaded session."""

        return unique_projected_resources(self.telemetry_records())

    def correlations(self) -> list[dict[str, Any]]:
        """Return unique correlation contexts for this loaded session."""

        return unique_projected_correlations(self.telemetry_records())


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_int(value: Any, field_name: str) -> int:
    if _is_int(value):
        return int(value)
    raise ValueError(f"{field_name} must be an integer")


def _coerce_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _coerce_int(value, field_name)


def _coerce_string(
    value: Any, field_name: str, *, allow_none: bool = False
) -> Optional[str]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip() and not allow_none:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _coerce_required_string(value: Any, field_name: str) -> str:
    coerced = _coerce_string(value, field_name)
    if coerced is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return coerced


def _coerce_optional_non_empty_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _coerce_required_string(value, field_name)


def _coerce_metadata_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return dict(value)


def _validate_memory_capabilities(metadata: Mapping[str, Any]) -> None:
    capabilities = metadata.get("memory_capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("metadata.memory_capabilities must be an object")
    missing = [
        name
        for name in (
            *_MEMORY_CAPABILITY_STRING_FIELDS,
            *_MEMORY_CAPABILITY_BOOLEAN_FIELDS,
        )
        if name not in capabilities
    ]
    if missing:
        raise ValueError("Missing memory capability fields: " + ", ".join(missing))
    unknown = sorted(set(capabilities) - _MEMORY_CAPABILITY_FIELD_SET)
    if unknown:
        raise ValueError("Unknown memory capability fields: " + ", ".join(unknown))
    for name in _MEMORY_CAPABILITY_STRING_FIELDS:
        _coerce_required_string(
            capabilities[name], f"metadata.memory_capabilities.{name}"
        )
    for name in _MEMORY_CAPABILITY_BOOLEAN_FIELDS:
        if not isinstance(capabilities[name], bool):
            raise ValueError(f"metadata.memory_capabilities.{name} must be a boolean")


def _validate_capability_counter_consistency(
    record: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    capabilities = metadata["memory_capabilities"]
    for counter_field, capability_field in _COUNTER_CAPABILITY_FIELDS.items():
        if record[counter_field] is not None and not capabilities[capability_field]:
            raise ValueError(
                f"{counter_field} must be null when "
                f"metadata.memory_capabilities.{capability_field} is false"
            )
    if (
        record["allocator_change_bytes"] is not None
        and not capabilities["supports_allocator_allocated"]
    ):
        raise ValueError(
            "allocator_change_bytes must be null when allocator allocated memory "
            "is unsupported"
        )


def _with_inferred_memory_capabilities(
    metadata: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(metadata)
    if "memory_capabilities" in result:
        return result
    collector = str(record.get("collector", "legacy.unknown"))
    backend_value = result.get("backend")
    if not isinstance(backend_value, str) or not backend_value.strip():
        backend_value = next(
            (
                candidate
                for candidate in ("cuda", "rocm", "mps", "cpu", "tensorflow", "jax")
                if candidate in collector.lower()
            ),
            "unknown",
        )
    allocated = record.get("allocator_allocated_bytes") is not None
    reserved = record.get("allocator_reserved_bytes") is not None
    capabilities = {
        "backend": str(backend_value),
        "telemetry_collector": collector,
        "sampling_source": str(result.get("sampling_source", "legacy")),
        "supports_allocator_allocated": allocated,
        "supports_allocator_reserved": reserved,
        "supports_allocator_active": record.get("allocator_active_bytes") is not None,
        "supports_allocator_inactive": (
            record.get("allocator_inactive_bytes") is not None
        ),
        "supports_device_used": record.get("device_used_bytes") is not None,
        "supports_device_free": record.get("device_free_bytes") is not None,
        "supports_device_total": record.get("device_total_bytes") is not None,
        "supports_native_allocator_history": False,
        "supports_fragmentation_analysis": allocated and reserved,
        "supports_allocator_attribution": allocated and reserved,
        "supports_bounded_profiling": False,
    }
    result["memory_capabilities"] = capabilities
    return result


def _extract_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    raw_metadata = record.get("metadata")
    if raw_metadata is None:
        pass
    elif isinstance(raw_metadata, Mapping):
        metadata.update(dict(raw_metadata))
    else:
        raise ValueError("metadata must be an object when provided")

    for key, value in record.items():
        if isinstance(key, str) and key.startswith("metadata_"):
            metadata[key.removeprefix("metadata_")] = value

    return metadata


def _first_env_value(env: Mapping[str, str], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = env.get(key)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _coerce_non_negative_int(value: Any, field_name: str) -> int:
    coerced = _coerce_int(value, field_name)
    if coerced < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return coerced


def _coerce_positive_int(value: Any, field_name: str) -> int:
    coerced = _coerce_int(value, field_name)
    if coerced <= 0:
        raise ValueError(f"{field_name} must be >= 1")
    return coerced


def _coerce_env_int(value: str, field_name: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _infer_distributed_identity_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    if env is None:
        return {"job_id": None, "rank": None, "local_rank": None, "world_size": None}

    raw_job_id = _first_env_value(env, _JOB_ID_ENV_KEYS)

    for rank_key, local_rank_key, world_size_key in _RANK_ENV_GROUPS:
        keys_present = any(
            key in env for key in (rank_key, local_rank_key, world_size_key)
        )
        if not keys_present:
            continue

        raw_rank = env.get(rank_key)
        raw_world_size = env.get(world_size_key)
        if raw_rank is None or raw_world_size is None:
            continue

        local_rank_value = env.get(local_rank_key)
        if local_rank_value is None or not local_rank_value.strip():
            local_rank_value = raw_rank

        rank_value = _coerce_env_int(raw_rank, "rank")
        local_rank_int = _coerce_env_int(local_rank_value, "local_rank")
        world_size_value = _coerce_env_int(raw_world_size, "world_size")

        return {
            "job_id": raw_job_id,
            "rank": _coerce_non_negative_int(rank_value, "rank"),
            "local_rank": _coerce_non_negative_int(local_rank_int, "local_rank"),
            "world_size": _coerce_positive_int(world_size_value, "world_size"),
        }

    return {"job_id": raw_job_id, "rank": None, "local_rank": None, "world_size": None}


def resolve_distributed_identity(
    *,
    job_id: Any = None,
    rank: Any = None,
    local_rank: Any = None,
    world_size: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Normalize distributed identity fields from explicit, metadata, or env inputs."""
    metadata_values = dict(metadata or {})
    raw = {
        "job_id": job_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }
    for name, value in raw.items():
        if value is None:
            raw[name] = metadata_values.get(name)
    _fill_identity_from_env(raw, env)
    return _normalize_distributed_identity(raw)


def _fill_identity_from_env(
    raw: dict[str, Any], env: Optional[Mapping[str, str]]
) -> None:
    """Fill only missing fields, without parsing rank env for complete identities."""
    needs_rank_env = any(
        raw[name] is None for name in ("rank", "local_rank", "world_size")
    )
    if needs_rank_env:
        inferred = _infer_distributed_identity_from_env(env)
        for name, value in raw.items():
            if value is None:
                raw[name] = inferred[name]
    elif raw["job_id"] is None and env is not None:
        raw["job_id"] = _first_env_value(env, _JOB_ID_ENV_KEYS)


def _normalize_distributed_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    raw_rank = raw["rank"]
    raw_local_rank = raw["local_rank"]
    raw_world_size = raw["world_size"]
    if raw_world_size is None:
        raw_world_size = 1
    if raw_rank is None:
        raw_rank = 0
    if raw_local_rank is None:
        raw_local_rank = raw_rank

    normalized_job_id = _coerce_optional_non_empty_string(raw["job_id"], "job_id")
    normalized_rank = _coerce_non_negative_int(raw_rank, "rank")
    normalized_local_rank = _coerce_non_negative_int(raw_local_rank, "local_rank")
    normalized_world_size = _coerce_positive_int(raw_world_size, "world_size")

    if normalized_rank >= normalized_world_size:
        raise ValueError("rank must be < world_size")
    if normalized_local_rank >= normalized_world_size:
        raise ValueError("local_rank must be < world_size")
    if normalized_world_size == 1 and normalized_rank != 0:
        raise ValueError("rank must be 0 when world_size is 1")
    if normalized_world_size == 1 and normalized_local_rank != 0:
        raise ValueError("local_rank must be 0 when world_size is 1")

    return {
        "job_id": normalized_job_id,
        "rank": normalized_rank,
        "local_rank": normalized_local_rank,
        "world_size": normalized_world_size,
    }


def _strip_distributed_identity_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _DISTRIBUTED_METADATA_KEYS
    }


def _strip_session_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _SESSION_METADATA_KEYS
    }


def _resolve_session_id(
    record: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    default_session_id: str | None = None,
) -> str:
    raw_session_id = record.get("session_id")
    if raw_session_id is None and metadata is not None:
        raw_session_id = metadata.get("session_id")
    if isinstance(raw_session_id, str) and raw_session_id.strip():
        return raw_session_id
    if default_session_id is not None:
        return default_session_id
    timestamp_value = record.get("timestamp_ns", record.get("timestamp", "unknown"))
    host_value = record.get("host", UNKNOWN_HOST)
    pid_value = record.get("pid", UNKNOWN_PID)
    return stable_legacy_session_id(timestamp_value, host_value, pid_value)


def _legacy_timestamp_ns(record: Mapping[str, Any]) -> int:
    if "timestamp_ns" in record:
        return _coerce_int(record["timestamp_ns"], "timestamp_ns")

    timestamp = record.get("timestamp")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        return int(float(timestamp) * 1_000_000_000)

    raise ValueError("Legacy record is missing a valid timestamp")


def _legacy_device_id(record: Mapping[str, Any]) -> int:
    if "device_id" in record:
        return _coerce_int(record["device_id"], "device_id")

    device = record.get("device")
    if isinstance(device, str):
        lowered = device.lower()
        if "cpu" in lowered:
            return -1
        if ":" in device:
            tail = device.rsplit(":", 1)[-1]
            if tail.isdigit():
                return int(tail)
        if lowered.startswith("/gpu"):
            return 0

    if "memory_mb" in record:
        return 0

    return -1


def _legacy_allocator_allocated_bytes(record: Mapping[str, Any]) -> int:
    if "allocator_allocated_bytes" in record:
        return _coerce_int(
            record["allocator_allocated_bytes"], "allocator_allocated_bytes"
        )

    if "memory_allocated" in record:
        return _coerce_int(record["memory_allocated"], "memory_allocated")

    memory_mb = record.get("memory_mb")
    if isinstance(memory_mb, (int, float)) and not isinstance(memory_mb, bool):
        return int(float(memory_mb) * (1024**2))

    if "device_used_bytes" in record:
        return _coerce_int(record["device_used_bytes"], "device_used_bytes")

    return 0


def _legacy_allocator_reserved_bytes(record: Mapping[str, Any], allocated: int) -> int:
    if "allocator_reserved_bytes" in record:
        return _coerce_int(
            record["allocator_reserved_bytes"], "allocator_reserved_bytes"
        )
    if "memory_reserved" in record:
        return _coerce_int(record["memory_reserved"], "memory_reserved")
    return allocated


def _legacy_allocator_change_bytes(record: Mapping[str, Any]) -> int:
    if "allocator_change_bytes" in record:
        return _coerce_int(record["allocator_change_bytes"], "allocator_change_bytes")
    if "memory_change" in record:
        return _coerce_int(record["memory_change"], "memory_change")
    return 0


def _legacy_optional_counter(record: Mapping[str, Any], key: str) -> Optional[int]:
    value = record.get(key)
    if value is None:
        return None
    return _coerce_int(value, key)


def _legacy_total_memory_bytes(
    record: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Optional[int]:
    if "device_total_bytes" in record:
        return _coerce_optional_int(
            record.get("device_total_bytes"), "device_total_bytes"
        )

    for key in ("total_memory", "device_total", "total_bytes"):
        if key in record:
            value = record[key]
            if value is None:
                return None
            return _coerce_int(value, key)

    for key in ("total_memory", "device_total", "total_bytes"):
        if key in metadata:
            value = metadata[key]
            if value is None:
                return None
            return _coerce_int(value, key)

    return None


def _legacy_device_used_bytes(record: Mapping[str, Any], allocated: int) -> int:
    if "device_used_bytes" in record:
        return _coerce_int(record["device_used_bytes"], "device_used_bytes")
    return allocated


def _legacy_device_free_bytes(
    record: Mapping[str, Any],
    used: int,
    total: Optional[int],
) -> Optional[int]:
    if "device_free_bytes" in record:
        return _coerce_optional_int(
            record.get("device_free_bytes"), "device_free_bytes"
        )

    if total is None:
        return None

    free = total - used
    return max(free, 0)


def _legacy_pid(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> int:
    if "pid" in record:
        return _coerce_int(record["pid"], "pid")
    if "pid" in metadata:
        return _coerce_int(metadata["pid"], "pid")
    return UNKNOWN_PID


def _legacy_host(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    if "host" in record:
        return _coerce_string(record["host"], "host") or UNKNOWN_HOST
    if "host" in metadata:
        return _coerce_string(metadata["host"], "host") or UNKNOWN_HOST
    return UNKNOWN_HOST


def _legacy_collector(
    record: Mapping[str, Any],
    default_collector: str,
    device_id: int,
    metadata: Mapping[str, Any],
) -> str:
    collector = record.get("collector")
    if isinstance(collector, str) and collector.strip():
        return collector

    backend_value = record.get("backend", metadata.get("backend"))
    if isinstance(backend_value, str):
        backend = backend_value.strip().lower()
        if backend in _LEGACY_BACKEND_COLLECTORS:
            return _LEGACY_BACKEND_COLLECTORS[backend]

    if "memory_mb" in record:
        return "stormlog.tensorflow.memory_tracker"

    if "memory_allocated" in record:
        return "stormlog.cpu_tracker" if device_id == -1 else "stormlog.cuda_tracker"

    return default_collector


def telemetry_event_to_dict(
    event: TelemetryEventV2 | TelemetryEventV3 | TelemetryEventV4,
) -> dict[str, Any]:
    """Serialize a telemetry event to a plain dictionary."""
    if isinstance(event, TelemetryEventV2):
        return {
            "schema_version": event.schema_version,
            "timestamp_ns": event.timestamp_ns,
            "event_type": event.event_type,
            "collector": event.collector,
            "sampling_interval_ms": event.sampling_interval_ms,
            "pid": event.pid,
            "host": event.host,
            "job_id": event.job_id,
            "rank": event.rank,
            "local_rank": event.local_rank,
            "world_size": event.world_size,
            "device_id": event.device_id,
            "allocator_allocated_bytes": event.allocator_allocated_bytes,
            "allocator_reserved_bytes": event.allocator_reserved_bytes,
            "allocator_active_bytes": event.allocator_active_bytes,
            "allocator_inactive_bytes": event.allocator_inactive_bytes,
            "allocator_change_bytes": event.allocator_change_bytes,
            "device_used_bytes": event.device_used_bytes,
            "device_free_bytes": event.device_free_bytes,
            "device_total_bytes": event.device_total_bytes,
            "context": event.context,
            "metadata": dict(event.metadata),
        }
    return {
        "schema_version": event.schema_version,
        "session_id": event.session_id,
        "timestamp_ns": event.timestamp_ns,
        "event_type": event.event_type,
        "collector": event.collector,
        "sampling_interval_ms": event.sampling_interval_ms,
        "pid": event.pid,
        "host": event.host,
        "job_id": event.job_id,
        "rank": event.rank,
        "local_rank": event.local_rank,
        "world_size": event.world_size,
        "device_id": event.device_id,
        "allocator_allocated_bytes": event.allocator_allocated_bytes,
        "allocator_reserved_bytes": event.allocator_reserved_bytes,
        "allocator_active_bytes": event.allocator_active_bytes,
        "allocator_inactive_bytes": event.allocator_inactive_bytes,
        "allocator_change_bytes": event.allocator_change_bytes,
        "device_used_bytes": event.device_used_bytes,
        "device_free_bytes": event.device_free_bytes,
        "device_total_bytes": event.device_total_bytes,
        "context": event.context,
        "metadata": dict(event.metadata),
    }


def validate_telemetry_record(record: Mapping[str, Any]) -> None:
    """Validate a v2, v3, or v4 telemetry record.

    Raises:
        ValueError: if the record is invalid or partial.
    """

    schema_version = _validate_telemetry_shape(record)
    nullable_core_counters = schema_version == SCHEMA_VERSION_V4
    _validate_telemetry_process(record)
    _validate_telemetry_identity_fields(record)
    _coerce_int(record["device_id"], "device_id")
    _validate_allocator_counters(record, nullable_core_counters=nullable_core_counters)
    _validate_device_counters(record, nullable_core_counters=nullable_core_counters)
    _coerce_string(record["context"], "context", allow_none=True)
    metadata = _coerce_metadata_dict(record["metadata"])
    if schema_version == SCHEMA_VERSION_V4:
        _validate_memory_capabilities(metadata)
        _validate_capability_counter_consistency(record, metadata)
    resolve_distributed_identity(
        job_id=record.get("job_id"),
        rank=record.get("rank"),
        local_rank=record.get("local_rank"),
        world_size=record.get("world_size"),
    )


def _telemetry_schema_fields(
    schema_version: int,
) -> tuple[tuple[str, ...], frozenset[str], bool]:
    required_fields: tuple[str, ...]
    known_fields: frozenset[str]
    if schema_version == SCHEMA_VERSION_V4:
        required_fields = REQUIRED_V4_FIELDS
        known_fields = KNOWN_V4_FIELD_SET
        require_session_id = True
    elif schema_version == SCHEMA_VERSION_V3:
        required_fields = REQUIRED_V3_FIELDS
        known_fields = KNOWN_V3_FIELD_SET
        require_session_id = True
    elif schema_version == SCHEMA_VERSION_V2:
        required_fields = REQUIRED_V2_FIELDS
        known_fields = KNOWN_V2_FIELD_SET
        require_session_id = False
    else:
        raise ValueError(f"Unsupported schema_version: {schema_version}")

    return required_fields, known_fields, require_session_id


def _validate_telemetry_shape(record: Mapping[str, Any]) -> int:
    schema_version = _coerce_int(record.get("schema_version"), "schema_version")
    required_fields, known_fields, require_session_id = _telemetry_schema_fields(
        schema_version
    )

    missing = [name for name in required_fields if name not in record]
    if missing:
        raise ValueError(f"Missing required telemetry fields: {', '.join(missing)}")

    unknown = sorted(str(name) for name in record if name not in known_fields)
    if unknown:
        raise ValueError(f"Unknown telemetry fields: {', '.join(unknown)}")

    if require_session_id:
        _coerce_required_string(record["session_id"], "session_id")
    return schema_version


def _validate_telemetry_process(record: Mapping[str, Any]) -> None:
    timestamp_ns = _coerce_int(record["timestamp_ns"], "timestamp_ns")
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be >= 0")

    _coerce_required_string(record["event_type"], "event_type")
    _coerce_required_string(record["collector"], "collector")

    sampling_interval_ms = _coerce_int(
        record["sampling_interval_ms"], "sampling_interval_ms"
    )
    if sampling_interval_ms < 0:
        raise ValueError("sampling_interval_ms must be >= 0")

    pid = _coerce_int(record["pid"], "pid")
    if pid < -1:
        raise ValueError("pid must be >= -1")

    _coerce_required_string(record["host"], "host")


def _validate_telemetry_identity_fields(record: Mapping[str, Any]) -> None:
    if "job_id" in record:
        _coerce_optional_non_empty_string(record["job_id"], "job_id")

    if "rank" in record:
        _coerce_non_negative_int(record["rank"], "rank")

    if "local_rank" in record:
        _coerce_non_negative_int(record["local_rank"], "local_rank")

    if "world_size" in record:
        _coerce_positive_int(record["world_size"], "world_size")


def _validate_counter_range(value: int | None, error: str) -> None:
    if value is not None and value < 0:
        raise ValueError(error)


def _validate_allocator_counters(
    record: Mapping[str, Any], *, nullable_core_counters: bool
) -> None:
    allocator_allocated_bytes = (
        _coerce_optional_int(
            record["allocator_allocated_bytes"], "allocator_allocated_bytes"
        )
        if nullable_core_counters
        else _coerce_int(
            record["allocator_allocated_bytes"], "allocator_allocated_bytes"
        )
    )
    allocator_reserved_bytes = (
        _coerce_optional_int(
            record["allocator_reserved_bytes"], "allocator_reserved_bytes"
        )
        if nullable_core_counters
        else _coerce_int(record["allocator_reserved_bytes"], "allocator_reserved_bytes")
    )
    allocator_active_bytes = _coerce_optional_int(
        record["allocator_active_bytes"], "allocator_active_bytes"
    )
    allocator_inactive_bytes = _coerce_optional_int(
        record["allocator_inactive_bytes"], "allocator_inactive_bytes"
    )
    allocator_change_bytes = (
        _coerce_optional_int(record["allocator_change_bytes"], "allocator_change_bytes")
        if nullable_core_counters
        else _coerce_int(record["allocator_change_bytes"], "allocator_change_bytes")
    )

    _validate_counter_range(
        allocator_allocated_bytes, "allocator_allocated_bytes must be >= 0"
    )
    _validate_counter_range(
        allocator_reserved_bytes, "allocator_reserved_bytes must be >= 0"
    )
    _validate_counter_range(
        allocator_active_bytes, "allocator_active_bytes must be >= 0 when provided"
    )
    _validate_counter_range(
        allocator_inactive_bytes, "allocator_inactive_bytes must be >= 0 when provided"
    )
    if allocator_change_bytes is not None and not _is_int(allocator_change_bytes):
        raise ValueError("allocator_change_bytes must be an integer when provided")


def _validate_device_counters(
    record: Mapping[str, Any], *, nullable_core_counters: bool
) -> None:
    device_used_bytes = (
        _coerce_optional_int(record["device_used_bytes"], "device_used_bytes")
        if nullable_core_counters
        else _coerce_int(record["device_used_bytes"], "device_used_bytes")
    )
    device_free_bytes = _coerce_optional_int(
        record["device_free_bytes"], "device_free_bytes"
    )
    device_total_bytes = _coerce_optional_int(
        record["device_total_bytes"], "device_total_bytes"
    )

    _validate_counter_range(device_used_bytes, "device_used_bytes must be >= 0")
    _validate_counter_range(
        device_free_bytes, "device_free_bytes must be >= 0 when provided"
    )
    _validate_counter_range(
        device_total_bytes, "device_total_bytes must be >= 0 when provided"
    )

    if device_total_bytes is None:
        return
    if device_used_bytes is not None and device_used_bytes > device_total_bytes:
        raise ValueError("device_used_bytes cannot exceed device_total_bytes")
    if device_free_bytes is not None and device_free_bytes > device_total_bytes:
        raise ValueError("device_free_bytes cannot exceed device_total_bytes")


def telemetry_event_from_record(
    record: Mapping[str, Any],
    permissive_legacy: bool = True,
    default_collector: str = "legacy.unknown",
    default_sampling_interval_ms: int = 0,
    default_session_id: str | None = None,
) -> TelemetryEvent:
    """Create a canonical telemetry event from v4, v3, v2, or legacy records."""

    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")

    if "schema_version" in record:
        schema_version = _coerce_int(record["schema_version"], "schema_version")
        if schema_version not in {
            SCHEMA_VERSION_V2,
            SCHEMA_VERSION_V3,
            SCHEMA_VERSION_V4,
        }:
            raise ValueError(f"Unsupported schema_version: {schema_version}")

        raw_metadata = record.get("metadata", {})
        metadata = _coerce_metadata_dict(raw_metadata)
        distributed_identity = resolve_distributed_identity(
            job_id=record.get("job_id"),
            rank=record.get("rank"),
            local_rank=record.get("local_rank"),
            world_size=record.get("world_size"),
        )
        session_id = _resolve_session_id(
            record,
            metadata=metadata,
            default_session_id=default_session_id,
        )
        metadata = _with_inferred_memory_capabilities(
            _coerce_metadata_dict(record["metadata"]),
            record,
        )
        normalized_record = dict(record)
        if schema_version != SCHEMA_VERSION_V2:
            normalized_record["session_id"] = session_id
        normalized_record["metadata"] = metadata
        validate_telemetry_record(normalized_record)
        nullable_core_counters = schema_version == SCHEMA_VERSION_V4

        def core_counter(field_name: str) -> Optional[int]:
            if nullable_core_counters:
                return _coerce_optional_int(record[field_name], field_name)
            return _coerce_int(record[field_name], field_name)

        return TelemetryEvent(
            schema_version=SCHEMA_VERSION_V4,
            session_id=session_id,
            timestamp_ns=_coerce_int(record["timestamp_ns"], "timestamp_ns"),
            event_type=_coerce_required_string(record["event_type"], "event_type"),
            collector=_coerce_required_string(record["collector"], "collector"),
            sampling_interval_ms=_coerce_int(
                record["sampling_interval_ms"], "sampling_interval_ms"
            ),
            pid=_coerce_int(record["pid"], "pid"),
            host=_coerce_required_string(record["host"], "host"),
            device_id=_coerce_int(record["device_id"], "device_id"),
            allocator_allocated_bytes=core_counter("allocator_allocated_bytes"),
            allocator_reserved_bytes=core_counter("allocator_reserved_bytes"),
            allocator_active_bytes=_coerce_optional_int(
                record["allocator_active_bytes"], "allocator_active_bytes"
            ),
            allocator_inactive_bytes=_coerce_optional_int(
                record["allocator_inactive_bytes"], "allocator_inactive_bytes"
            ),
            allocator_change_bytes=core_counter("allocator_change_bytes"),
            device_used_bytes=core_counter("device_used_bytes"),
            device_free_bytes=_coerce_optional_int(
                record["device_free_bytes"], "device_free_bytes"
            ),
            device_total_bytes=_coerce_optional_int(
                record["device_total_bytes"], "device_total_bytes"
            ),
            context=_coerce_string(record["context"], "context", allow_none=True),
            job_id=distributed_identity["job_id"],
            rank=distributed_identity["rank"],
            local_rank=distributed_identity["local_rank"],
            world_size=distributed_identity["world_size"],
            metadata=metadata,
        )

    if not permissive_legacy:
        raise ValueError("Legacy record conversion is disabled")

    metadata = _extract_metadata(record)
    timestamp_ns = _legacy_timestamp_ns(record)
    device_id = _legacy_device_id(record)

    allocator_allocated_bytes = _legacy_allocator_allocated_bytes(record)
    allocator_reserved_bytes = _legacy_allocator_reserved_bytes(
        record, allocator_allocated_bytes
    )
    allocator_change_bytes = _legacy_allocator_change_bytes(record)

    allocator_active_bytes = _legacy_optional_counter(record, "allocator_active_bytes")
    allocator_inactive_bytes = _legacy_optional_counter(
        record, "allocator_inactive_bytes"
    )

    device_used_bytes = _legacy_device_used_bytes(record, allocator_allocated_bytes)
    device_total_bytes = _legacy_total_memory_bytes(record, metadata)
    device_free_bytes = _legacy_device_free_bytes(
        record, device_used_bytes, device_total_bytes
    )

    event_type_value = record.get("event_type", record.get("type", "sample"))
    event_type = _coerce_string(event_type_value, "event_type") or "sample"

    sampling_interval_value = record.get(
        "sampling_interval_ms", default_sampling_interval_ms
    )
    sampling_interval_ms = _coerce_int(sampling_interval_value, "sampling_interval_ms")

    pid = _legacy_pid(record, metadata)
    host = _legacy_host(record, metadata)
    collector = _legacy_collector(record, default_collector, device_id, metadata)
    distributed_identity = resolve_distributed_identity(
        job_id=record.get("job_id"),
        rank=record.get("rank"),
        local_rank=record.get("local_rank"),
        world_size=record.get("world_size"),
        metadata=metadata,
    )
    metadata = _strip_distributed_identity_metadata(metadata)
    session_id = _resolve_session_id(
        record,
        metadata=metadata,
        default_session_id=default_session_id,
    )
    metadata = _strip_session_metadata(metadata)

    context_value = record.get("context", record.get("message"))
    context = _coerce_string(context_value, "context", allow_none=True)

    normalized_counters = {
        **record,
        "collector": collector,
        "allocator_allocated_bytes": allocator_allocated_bytes,
        "allocator_reserved_bytes": allocator_reserved_bytes,
        "allocator_active_bytes": allocator_active_bytes,
        "allocator_inactive_bytes": allocator_inactive_bytes,
        "device_used_bytes": device_used_bytes,
        "device_free_bytes": device_free_bytes,
        "device_total_bytes": device_total_bytes,
    }
    metadata = _with_inferred_memory_capabilities(metadata, normalized_counters)

    event = TelemetryEvent(
        schema_version=SCHEMA_VERSION_V4,
        session_id=session_id,
        timestamp_ns=timestamp_ns,
        event_type=event_type,
        collector=collector,
        sampling_interval_ms=sampling_interval_ms,
        pid=pid,
        host=host,
        device_id=device_id,
        allocator_allocated_bytes=allocator_allocated_bytes,
        allocator_reserved_bytes=allocator_reserved_bytes,
        allocator_active_bytes=allocator_active_bytes,
        allocator_inactive_bytes=allocator_inactive_bytes,
        allocator_change_bytes=allocator_change_bytes,
        device_used_bytes=device_used_bytes,
        device_free_bytes=device_free_bytes,
        device_total_bytes=device_total_bytes,
        context=context,
        job_id=distributed_identity["job_id"],
        rank=distributed_identity["rank"],
        local_rank=distributed_identity["local_rank"],
        world_size=distributed_identity["world_size"],
        metadata=metadata,
    )
    return event


def _looks_like_event_record(payload: Mapping[str, Any]) -> bool:
    candidate_keys = {
        "schema_version",
        "event_type",
        "type",
        "memory_allocated",
        "memory_mb",
        "timestamp",
        "timestamp_ns",
    }
    return any(key in payload for key in candidate_keys)


def _group_session_events(
    events: list[TelemetryEvent],
) -> dict[str, list[TelemetryEvent]]:
    grouped: dict[str, list[TelemetryEvent]] = {}
    for event in events:
        grouped.setdefault(event.session_id, []).append(event)
    for session_events in grouped.values():
        session_events.sort(key=lambda event: event.timestamp_ns)
    return grouped


def _assemble_loaded_sessions(
    *,
    grouped_events: dict[str, list[TelemetryEvent]],
    manifest_summaries: list[SessionSummary] | None = None,
    sources_by_session: Mapping[str, set[str]] | None = None,
    warnings_by_session: Mapping[str, list[str]] | None = None,
    default_source: str,
    default_source_path: str,
) -> list[LoadedTelemetrySession]:
    summary_by_id = {
        summary.session_id: summary for summary in (manifest_summaries or [])
    }
    session_ids = set(grouped_events) | set(summary_by_id)
    loaded_sessions: list[LoadedTelemetrySession] = []

    for session_id in session_ids:
        session_events = list(grouped_events.get(session_id, []))
        summary = summary_by_id.get(session_id)
        if summary is None:
            summary = infer_session_summary_from_events(
                session_id=session_id,
                events=session_events,
                source=default_source,
                fallback_status=SESSION_STATUS_INCOMPLETE,
            )
        loaded_sessions.append(
            LoadedTelemetrySession(
                summary=summary,
                events=session_events,
                sources_loaded=sorted(
                    (sources_by_session or {}).get(session_id, {default_source_path})
                ),
                warnings=list((warnings_by_session or {}).get(session_id, [])),
            )
        )

    ordered_summaries = sort_session_summaries(
        loaded.summary for loaded in loaded_sessions
    )
    order = {
        summary.session_id: index for index, summary in enumerate(ordered_summaries)
    }
    return sorted(
        loaded_sessions,
        key=lambda loaded: (
            order.get(loaded.summary.session_id, 999),
            loaded.summary.session_id,
        ),
    )


def _load_jsonl_events(
    path: Path,
    *,
    permissive_legacy: bool,
    default_session_id: str | None = None,
) -> list[TelemetryEvent]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[TelemetryEvent] = []

    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) and not line.endswith("\n"):
                break
            raise ValueError(
                f"Malformed telemetry JSONL record in {path} at line {index}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Telemetry record in {path} at line {index} must be an object"
            )
        output.append(
            telemetry_event_from_record(
                payload,
                permissive_legacy=permissive_legacy,
                default_session_id=default_session_id,
            )
        )
    return output


def _load_json_records(
    payload_path: Path,
    *,
    events_key: Optional[str],
) -> list[Mapping[str, Any]]:
    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: Any
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        if events_key is not None:
            records = payload.get(events_key)
            if not isinstance(records, list):
                raise ValueError(
                    f"Top-level key '{events_key}' must contain a list of events"
                )
        elif isinstance(payload.get("events"), list):
            records = payload["events"]
        elif _looks_like_event_record(payload):
            records = [payload]
        else:
            raise ValueError("JSON payload does not contain telemetry events")
    else:
        raise ValueError("Telemetry payload must be a JSON object or array")

    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Event at index {index} must be an object")
        normalized.append(record)
    return normalized


def load_telemetry_sessions(
    path: str | Path,
    permissive_legacy: bool = True,
    events_key: Optional[str] = None,
) -> list[LoadedTelemetrySession]:
    """Load grouped telemetry sessions from JSON, JSONL, or sink directories."""

    payload_path = Path(path)
    default_source_path = str(payload_path.resolve())
    default_source = f"artifact:{payload_path.name or payload_path.resolve()}"

    manifest = read_telemetry_sink_manifest(payload_path)
    segment_paths = resolve_telemetry_sink_segment_paths(payload_path)
    if segment_paths:
        segment_session_ids = {
            segment.filename: segment.session_id
            for segment in (manifest.segments if manifest is not None else [])
        }
        fallback_session_id = stable_legacy_session_id(default_source_path, "sink")
        grouped_events, sources_by_session = _group_sink_events(
            segment_paths, segment_session_ids, fallback_session_id, permissive_legacy
        )
        return _assemble_loaded_sessions(
            grouped_events=grouped_events,
            manifest_summaries=manifest.sessions if manifest is not None else None,
            sources_by_session=sources_by_session,
            default_source=default_source,
            default_source_path=default_source_path,
        )

    records = _load_json_records(payload_path, events_key=events_key)
    default_session_id = stable_legacy_session_id(
        default_source_path, events_key or "json"
    )
    loaded_events: list[TelemetryEvent] = [
        telemetry_event_from_record(
            record,
            permissive_legacy=permissive_legacy,
            default_session_id=default_session_id,
        )
        for record in records
    ]
    grouped_events = _group_session_events(loaded_events)
    sources_by_session = {
        session_id: {default_source_path} for session_id in grouped_events
    }
    return _assemble_loaded_sessions(
        grouped_events=grouped_events,
        manifest_summaries=None,
        sources_by_session=sources_by_session,
        default_source=default_source,
        default_source_path=default_source_path,
    )


def load_telemetry_events(
    path: str | Path,
    permissive_legacy: bool = True,
    events_key: Optional[str] = None,
    session_id: str | None = None,
) -> list[TelemetryEvent]:
    """Load telemetry events from JSON and return the selected session."""

    sessions = load_telemetry_sessions(
        path,
        permissive_legacy=permissive_legacy,
        events_key=events_key,
    )
    if not sessions:
        return []

    if session_id is not None:
        for loaded in sessions:
            if loaded.summary.session_id == session_id:
                return list(loaded.events)
        raise ValueError(f"Requested session_id not found: {session_id}")

    selected = select_default_loaded_session(sessions)
    return list(selected.events) if selected is not None else []


def project_telemetry_event(
    event: TelemetryEvent | Mapping[str, Any],
) -> ProjectedTelemetryRecord:
    """Project telemetry objects or compatible mappings into the shared model."""
    if isinstance(event, (TelemetryEventV3, TelemetryEventV4)):
        normalized = event
    else:
        normalized = telemetry_event_from_record(event)
    return project_telemetry_mapping(telemetry_event_to_dict(normalized))


def project_telemetry_events(
    events: Iterable[TelemetryEvent | Mapping[str, Any]],
) -> list[ProjectedTelemetryRecord]:
    """Project existing telemetry events into backend-neutral records."""

    return [project_telemetry_event(event) for event in events]


def _group_sink_events(
    segment_paths: list[Path],
    segment_session_ids: Mapping[str, str | None],
    fallback_session_id: str,
    permissive_legacy: bool,
) -> tuple[dict[str, list[TelemetryEvent]], dict[str, set[str]]]:
    grouped_events: dict[str, list[TelemetryEvent]] = {}
    sources_by_session: dict[str, set[str]] = {}
    for segment_path in segment_paths:
        hint_session_id = (
            segment_session_ids.get(segment_path.name) or fallback_session_id
        )
        segment_events = _load_jsonl_events(
            segment_path,
            permissive_legacy=permissive_legacy,
            default_session_id=hint_session_id,
        )
        session_groups = _group_session_events(segment_events)
        for session_id, events in session_groups.items():
            grouped_events.setdefault(session_id, []).extend(events)
            sources_by_session.setdefault(session_id, set()).add(str(segment_path))
        if not segment_events and hint_session_id:
            sources_by_session.setdefault(hint_session_id, set()).add(str(segment_path))
    for events in grouped_events.values():
        events.sort(key=lambda event: event.timestamp_ns)
    return grouped_events, sources_by_session


__all__ = [
    "SCHEMA_VERSION_V2",
    "SCHEMA_VERSION_V3",
    "SCHEMA_VERSION_V4",
    "SCHEMA_VERSION_LATEST",
    "ProjectedTelemetryRecord",
    "LoadedTelemetrySession",
    "TelemetryEvent",
    "TelemetryEventV2",
    "TelemetryEventV3",
    "TelemetryEventV4",
    "TelemetryEventLike",
    "project_telemetry_event",
    "project_telemetry_events",
    "load_telemetry_sessions",
    "telemetry_event_from_record",
    "telemetry_event_to_dict",
    "validate_telemetry_record",
    "load_telemetry_events",
    "resolve_distributed_identity",
]
