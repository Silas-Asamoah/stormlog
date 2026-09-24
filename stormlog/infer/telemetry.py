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
    "device_memory_used_bytes": "gpu_device",
    "device_memory_reserved_bytes": "gpu_device",
    "instance_memory_used_bytes": "gpu_instance",
    "instance_memory_reserved_bytes": "gpu_instance",
}
SAMPLE_STATES = {"valid", "missing", "stale", "invalid"}


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

    def __post_init__(self) -> None:
        if not self.host or self.pid <= 0 or self.process_start_ns <= 0:
            raise ValueError("host, positive pid, and process_start_ns are required")
        if self.gpu_instance_id and not self.device_uuid:
            raise ValueError("gpu_instance_id requires device_uuid")
        if self.rank is not None and self.rank < 0:
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

    def __post_init__(self) -> None:
        if not self.run_id or self.metric not in METRIC_SCOPES or not self.source:
            raise ValueError("run_id, supported metric, and source are required")
        if self.observed_at_ns <= 0 or self.interval_ms <= 0:
            raise ValueError("observed_at_ns and interval_ms must be positive")
        if self.state not in SAMPLE_STATES:
            raise ValueError("unsupported telemetry state")
        if self.state == "valid":
            if self.value_bytes is None or self.value_bytes < 0:
                raise ValueError("valid samples require non-negative value_bytes")
        elif self.value_bytes is not None:
            raise ValueError("unavailable samples must have null value_bytes")
        scope = METRIC_SCOPES[self.metric]
        if scope != "server_process" and not self.identity.device_uuid:
            raise ValueError("GPU metrics require device_uuid")
        if scope == "gpu_instance" and not self.identity.gpu_instance_id:
            raise ValueError("instance metrics require gpu_instance_id")

    @property
    def scope(self) -> str:
        return METRIC_SCOPES[self.metric]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_type": "infer.telemetry_sample",
            "scope": self.scope,
            "clock_domain": f"{self.identity.host}/unix_epoch_ns",
            "provenance": "observed",
            "counter_owner": (
                "server_process" if self.scope == "server_process" else self.scope
            ),
            **asdict(self),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TelemetrySample:
        if record.get("schema_version") != TELEMETRY_SCHEMA_VERSION or record.get(
            "event_type"
        ) != "infer.telemetry_sample":
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
        )
        if record.get("scope") != sample.scope:
            raise ValueError("telemetry scope does not match metric")
        if record.get("clock_domain") != f"{identity.host}/unix_epoch_ns":
            raise ValueError("telemetry clock domain does not match host")
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
                raise ValueError(f"invalid telemetry line {line_number}: {exc}") from exc
    if not samples:
        raise ValueError("telemetry artifact has no samples")
    return samples
