"""Canonical telemetry event schema and legacy conversion helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

SCHEMA_VERSION_V2: Literal[2] = 2
UNKNOWN_PID = -1
UNKNOWN_HOST = "unknown"

REQUIRED_V2_FIELDS = (
    "schema_version",
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
OPTIONAL_V2_FIELDS = (
    "job_id",
    "rank",
    "local_rank",
    "world_size",
)
KNOWN_V2_FIELD_SET = frozenset(REQUIRED_V2_FIELDS + OPTIONAL_V2_FIELDS)
_DISTRIBUTED_METADATA_KEYS = frozenset(OPTIONAL_V2_FIELDS)
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


@dataclass
class TelemetryEventV2:
    """Canonical telemetry event payload used by tracker exports."""

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
    raw_job_id = job_id if job_id is not None else metadata_values.get("job_id")
    raw_rank = rank if rank is not None else metadata_values.get("rank")
    raw_local_rank = (
        local_rank if local_rank is not None else metadata_values.get("local_rank")
    )
    raw_world_size = (
        world_size if world_size is not None else metadata_values.get("world_size")
    )

    needs_rank_env = (
        raw_rank is None or raw_local_rank is None or raw_world_size is None
    )
    if needs_rank_env:
        inferred = _infer_distributed_identity_from_env(env)
        if raw_rank is None:
            raw_rank = inferred["rank"]
        if raw_local_rank is None:
            raw_local_rank = inferred["local_rank"]
        if raw_world_size is None:
            raw_world_size = inferred["world_size"]
        if raw_job_id is None:
            raw_job_id = inferred["job_id"]
    elif raw_job_id is None and env is not None:
        raw_job_id = _first_env_value(env, _JOB_ID_ENV_KEYS)

    if raw_world_size is None:
        raw_world_size = 1
    if raw_rank is None:
        raw_rank = 0

    if raw_rank is not None and raw_local_rank is None:
        raw_local_rank = raw_rank

    normalized = {
        "job_id": _coerce_optional_non_empty_string(raw_job_id, "job_id"),
        "rank": _coerce_non_negative_int(raw_rank, "rank"),
        "local_rank": _coerce_non_negative_int(raw_local_rank, "local_rank"),
        "world_size": _coerce_positive_int(raw_world_size, "world_size"),
    }

    if normalized["rank"] >= normalized["world_size"]:
        raise ValueError("rank must be < world_size")
    if normalized["local_rank"] >= normalized["world_size"]:
        raise ValueError("local_rank must be < world_size")
    if normalized["world_size"] == 1 and normalized["rank"] != 0:
        raise ValueError("rank must be 0 when world_size is 1")
    if normalized["world_size"] == 1 and normalized["local_rank"] != 0:
        raise ValueError("local_rank must be 0 when world_size is 1")

    return normalized


def _strip_distributed_identity_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _DISTRIBUTED_METADATA_KEYS
    }


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
        if backend == "mps":
            return "gpumemprof.mps_tracker"
        if backend == "rocm":
            return "gpumemprof.rocm_tracker"
        if backend == "cuda":
            return "gpumemprof.cuda_tracker"
        if backend == "cpu":
            return "gpumemprof.cpu_tracker"

    if "memory_mb" in record:
        return "tfmemprof.memory_tracker"

    if "memory_allocated" in record:
        return (
            "gpumemprof.cpu_tracker" if device_id == -1 else "gpumemprof.cuda_tracker"
        )

    return default_collector


def telemetry_event_to_dict(event: TelemetryEventV2) -> dict[str, Any]:
    """Serialize a telemetry event to a plain dictionary."""
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


def validate_telemetry_record(record: Mapping[str, Any]) -> None:
    """Validate a v2 telemetry record.

    Raises:
        ValueError: if the record is invalid or partial.
    """

    missing = [name for name in REQUIRED_V2_FIELDS if name not in record]
    if missing:
        raise ValueError(f"Missing required telemetry fields: {', '.join(missing)}")

    unknown = sorted(str(name) for name in record if name not in KNOWN_V2_FIELD_SET)
    if unknown:
        raise ValueError(f"Unknown telemetry fields: {', '.join(unknown)}")

    schema_version = _coerce_int(record["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION_V2:
        raise ValueError("schema_version must be 2")

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

    if "job_id" in record:
        _coerce_optional_non_empty_string(record["job_id"], "job_id")

    if "rank" in record:
        _coerce_non_negative_int(record["rank"], "rank")

    if "local_rank" in record:
        _coerce_non_negative_int(record["local_rank"], "local_rank")

    if "world_size" in record:
        _coerce_positive_int(record["world_size"], "world_size")

    _coerce_int(record["device_id"], "device_id")

    allocator_allocated_bytes = _coerce_int(
        record["allocator_allocated_bytes"], "allocator_allocated_bytes"
    )
    allocator_reserved_bytes = _coerce_int(
        record["allocator_reserved_bytes"], "allocator_reserved_bytes"
    )
    allocator_active_bytes = _coerce_optional_int(
        record["allocator_active_bytes"], "allocator_active_bytes"
    )
    allocator_inactive_bytes = _coerce_optional_int(
        record["allocator_inactive_bytes"], "allocator_inactive_bytes"
    )
    _coerce_int(record["allocator_change_bytes"], "allocator_change_bytes")

    if allocator_allocated_bytes < 0:
        raise ValueError("allocator_allocated_bytes must be >= 0")
    if allocator_reserved_bytes < 0:
        raise ValueError("allocator_reserved_bytes must be >= 0")
    if allocator_active_bytes is not None and allocator_active_bytes < 0:
        raise ValueError("allocator_active_bytes must be >= 0 when provided")
    if allocator_inactive_bytes is not None and allocator_inactive_bytes < 0:
        raise ValueError("allocator_inactive_bytes must be >= 0 when provided")

    device_used_bytes = _coerce_int(record["device_used_bytes"], "device_used_bytes")
    device_free_bytes = _coerce_optional_int(
        record["device_free_bytes"], "device_free_bytes"
    )
    device_total_bytes = _coerce_optional_int(
        record["device_total_bytes"], "device_total_bytes"
    )

    if device_used_bytes < 0:
        raise ValueError("device_used_bytes must be >= 0")
    if device_free_bytes is not None and device_free_bytes < 0:
        raise ValueError("device_free_bytes must be >= 0 when provided")
    if device_total_bytes is not None and device_total_bytes < 0:
        raise ValueError("device_total_bytes must be >= 0 when provided")

    if device_total_bytes is not None and device_used_bytes > device_total_bytes:
        raise ValueError("device_used_bytes cannot exceed device_total_bytes")
    if (
        device_total_bytes is not None
        and device_free_bytes is not None
        and device_free_bytes > device_total_bytes
    ):
        raise ValueError("device_free_bytes cannot exceed device_total_bytes")

    _coerce_string(record["context"], "context", allow_none=True)

    metadata = _coerce_metadata_dict(record["metadata"])
    resolve_distributed_identity(
        job_id=record.get("job_id"),
        rank=record.get("rank"),
        local_rank=record.get("local_rank"),
        world_size=record.get("world_size"),
    )


def telemetry_event_from_record(
    record: Mapping[str, Any],
    permissive_legacy: bool = True,
    default_collector: str = "legacy.unknown",
    default_sampling_interval_ms: int = 0,
) -> TelemetryEventV2:
    """Create a v2 telemetry event from v2 or legacy tracker records."""

    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")

    if "schema_version" in record:
        schema_version = _coerce_int(record["schema_version"], "schema_version")
        if schema_version != SCHEMA_VERSION_V2:
            raise ValueError(f"Unsupported schema_version: {schema_version}")

        validate_telemetry_record(record)
        metadata = _coerce_metadata_dict(record["metadata"])
        distributed_identity = resolve_distributed_identity(
            job_id=record.get("job_id"),
            rank=record.get("rank"),
            local_rank=record.get("local_rank"),
            world_size=record.get("world_size"),
        )

        return TelemetryEventV2(
            schema_version=SCHEMA_VERSION_V2,
            timestamp_ns=_coerce_int(record["timestamp_ns"], "timestamp_ns"),
            event_type=_coerce_required_string(record["event_type"], "event_type"),
            collector=_coerce_required_string(record["collector"], "collector"),
            sampling_interval_ms=_coerce_int(
                record["sampling_interval_ms"], "sampling_interval_ms"
            ),
            pid=_coerce_int(record["pid"], "pid"),
            host=_coerce_required_string(record["host"], "host"),
            device_id=_coerce_int(record["device_id"], "device_id"),
            allocator_allocated_bytes=_coerce_int(
                record["allocator_allocated_bytes"], "allocator_allocated_bytes"
            ),
            allocator_reserved_bytes=_coerce_int(
                record["allocator_reserved_bytes"], "allocator_reserved_bytes"
            ),
            allocator_active_bytes=_coerce_optional_int(
                record["allocator_active_bytes"], "allocator_active_bytes"
            ),
            allocator_inactive_bytes=_coerce_optional_int(
                record["allocator_inactive_bytes"], "allocator_inactive_bytes"
            ),
            allocator_change_bytes=_coerce_int(
                record["allocator_change_bytes"], "allocator_change_bytes"
            ),
            device_used_bytes=_coerce_int(
                record["device_used_bytes"], "device_used_bytes"
            ),
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

    context_value = record.get("context", record.get("message"))
    context = _coerce_string(context_value, "context", allow_none=True)

    event = TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
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


def load_telemetry_events(
    path: str | Path,
    permissive_legacy: bool = True,
    events_key: Optional[str] = None,
) -> list[TelemetryEventV2]:
    """Load telemetry events from JSON and normalize to v2 payloads."""

    payload_path = Path(path)
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

    output: list[TelemetryEventV2] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Event at index {index} must be an object")
        event = telemetry_event_from_record(
            record,
            permissive_legacy=permissive_legacy,
        )
        output.append(event)

    return output


__all__ = [
    "SCHEMA_VERSION_V2",
    "TelemetryEventV2",
    "telemetry_event_from_record",
    "telemetry_event_to_dict",
    "validate_telemetry_record",
    "load_telemetry_events",
    "resolve_distributed_identity",
]
