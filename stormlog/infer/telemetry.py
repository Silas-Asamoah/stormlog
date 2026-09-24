"""Engine-neutral, scoped samples from an inference host.

These records live in a separate JSONL artifact. A client request does not own
device-wide memory merely because its time window overlaps a sample.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TELEMETRY_SCHEMA_VERSION = 1

METRIC_SCOPES = {
    "process_rss_bytes": "server_process",
    "process_gpu_used_bytes": "server_process",
    "allocator_allocated_bytes": "server_process",
    "allocator_reserved_bytes": "server_process",
    "engine_cache_occupied_bytes": "server_process",
    "device_memory_used_bytes": "gpu_device",
    "device_memory_reserved_bytes": "gpu_device",
    "instance_memory_used_bytes": "gpu_instance",
    "instance_memory_reserved_bytes": "gpu_instance",
}
METRIC_OWNERS = {
    "process_rss_bytes": "operating_system_process",
    "process_gpu_used_bytes": "gpu_process",
    "allocator_allocated_bytes": "allocator",
    "allocator_reserved_bytes": "allocator",
    "engine_cache_occupied_bytes": "engine_cache",
    "device_memory_used_bytes": "gpu_device",
    "device_memory_reserved_bytes": "gpu_device",
    "instance_memory_used_bytes": "gpu_instance",
    "instance_memory_reserved_bytes": "gpu_instance",
}
SAMPLE_STATES = {"valid", "missing", "stale", "invalid"}
PROVENANCE = {"observed", "reported", "estimated"}


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class ServerIdentity:
    """The process and accelerator that a collector actually observed."""

    host: str
    pid: int
    process_start_ns: int
    device_uuid: str | None = None
    gpu_instance_id: str | None = None
    replica_id: str | None = None
    rank: int | None = None
    boot_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not _nonempty_string(self.host)
            or not _positive_int(self.pid)
            or not _positive_int(self.process_start_ns)
        ):
            raise ValueError("host, positive pid, and process_start_ns are required")
        self._validate_optional_fields()

    def _validate_optional_fields(self) -> None:
        for value in (
            self.device_uuid,
            self.gpu_instance_id,
            self.replica_id,
            self.boot_id,
        ):
            if value is not None and not _nonempty_string(value):
                raise ValueError(
                    "optional server identity fields must be non-empty strings"
                )
        if self.gpu_instance_id and not self.device_uuid:
            raise ValueError("gpu_instance_id requires device_uuid")
        if self.rank is not None and (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 0
        ):
            raise ValueError("rank must be non-negative")


@dataclass(frozen=True)
class TelemetrySample:
    """One counter with one owner and one observation scope."""

    run_id: str
    identity: ServerIdentity
    observed_at_ns: int
    metric: str
    value_bytes: int | None
    state: str
    source: str
    interval_ms: int
    detail: str | None = None
    provenance: str = "observed"

    def __post_init__(self) -> None:
        self._validate_fields()
        self._validate_metadata()
        self._validate_value()
        self._validate_identity()

    def _validate_fields(self) -> None:
        if (
            not _nonempty_string(self.run_id)
            or not isinstance(self.metric, str)
            or self.metric not in METRIC_SCOPES
            or not _nonempty_string(self.source)
        ):
            raise ValueError("run_id, supported metric, and source are required")
        if not _positive_int(self.observed_at_ns) or not _positive_int(
            self.interval_ms
        ):
            raise ValueError("observed_at_ns and interval_ms must be positive")

    def _validate_metadata(self) -> None:
        if self.state not in SAMPLE_STATES:
            raise ValueError("unsupported telemetry state")
        if self.provenance not in PROVENANCE:
            raise ValueError("unsupported telemetry provenance")
        if self.detail is not None and not isinstance(self.detail, str):
            raise ValueError("detail must be a string or null")

    def _validate_value(self) -> None:
        if self.state == "valid":
            if (
                not isinstance(self.value_bytes, int)
                or isinstance(self.value_bytes, bool)
                or self.value_bytes < 0
            ):
                raise ValueError("valid samples require non-negative value_bytes")
        elif self.value_bytes is not None:
            raise ValueError("unavailable samples must have null value_bytes")

    def _validate_identity(self) -> None:
        if self.metric != "process_rss_bytes" and not self.identity.device_uuid:
            raise ValueError("GPU metrics require device_uuid")
        if self.scope == "gpu_instance" and not self.identity.gpu_instance_id:
            raise ValueError("instance metrics require gpu_instance_id")

    @property
    def scope(self) -> str:
        return METRIC_SCOPES[self.metric]

    @property
    def counter_owner(self) -> str:
        return METRIC_OWNERS[self.metric]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_type": "infer.telemetry_sample",
            "scope": self.scope,
            "clock_domain": f"{self.identity.host}/unix_epoch_ns",
            "counter_owner": self.counter_owner,
            **asdict(self),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TelemetrySample:
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != TELEMETRY_SCHEMA_VERSION
            or record.get("event_type") != "infer.telemetry_sample"
        ):
            raise ValueError("unsupported telemetry record")
        identity = ServerIdentity(**record["identity"])
        sample = cls(
            run_id=record["run_id"],
            identity=identity,
            observed_at_ns=record["observed_at_ns"],
            metric=record["metric"],
            value_bytes=record["value_bytes"],
            state=record["state"],
            source=record["source"],
            interval_ms=record["interval_ms"],
            detail=record.get("detail"),
            provenance=record.get("provenance", "observed"),
        )
        if record.get("scope") != sample.scope:
            raise ValueError("telemetry scope does not match metric")
        if record.get("clock_domain") != f"{identity.host}/unix_epoch_ns":
            raise ValueError("telemetry clock domain does not match host")
        if record.get("counter_owner") != sample.counter_owner:
            raise ValueError("telemetry counter owner does not match metric")
        return sample


def load_telemetry(path: str | Path) -> list[TelemetrySample]:
    """Read and validate every line before using a server sample."""
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("record must be an object")
                samples.append(TelemetrySample.from_record(payload))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid telemetry line {line_number}: {exc}"
                ) from exc
    if not samples:
        raise ValueError("telemetry artifact has no samples")
    return samples
