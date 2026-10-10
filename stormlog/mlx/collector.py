"""Passive, sequential MLX allocator reads mapped directly to telemetry v4."""

from __future__ import annotations

import time
from typing import Any, Callable

import psutil

from stormlog.session import SessionSummary
from stormlog.telemetry import validate_telemetry_record

from .models import MemorySnapshot
from .runtime import Runtime, validate_bytes

COLLECTOR = "stormlog.mlx.memory_tracker"
_NULL_COUNTERS = (
    "allocator_reserved_bytes",
    "allocator_active_bytes",
    "allocator_inactive_bytes",
    "device_used_bytes",
    "device_free_bytes",
    "device_total_bytes",
)
_UNSUPPORTED = (
    "allocator_reserved",
    "allocator_active",
    "allocator_inactive",
    "device_used",
    "device_free",
    "device_total",
    "native_allocator_history",
    "fragmentation_analysis",
    "allocator_attribution",
    "bounded_profiling",
)


class MLXCollector:
    def __init__(
        self,
        runtime: Runtime,
        *,
        host_reader: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self.runtime = runtime
        self.host_reader = host_reader or _host_memory
        self.capabilities = {
            "backend": "metal",
            "telemetry_collector": COLLECTOR,
            "sampling_source": "mlx_allocator",
            "supports_allocator_allocated": runtime.supports("get_active_memory"),
            **{f"supports_{name}": False for name in _UNSUPPORTED},
        }

    def _read(self, name: str, unavailable: dict[str, str]) -> int | None:
        api = f"get_{name}"
        if not self.runtime.supports(api):
            unavailable[name] = f"{api} unavailable in this runtime"
            return None
        try:
            return validate_bytes(self.runtime.read_bytes(api), api)
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
            return None

    def capture_snapshot(self, name: str = "sample") -> MemorySnapshot:
        timestamp = time.time_ns()
        start = time.perf_counter_ns()
        unavailable: dict[str, str] = {}
        active = self._read("active_memory", unavailable)
        cache = self._read("cache_memory", unavailable)
        peak = self._read("peak_memory", unavailable)
        limit = self._read("memory_limit", unavailable)
        host: dict[str, int] = {}
        try:
            host = {k: validate_bytes(v, k) for k, v in self.host_reader().items()}
        except Exception as exc:
            unavailable["host_memory"] = f"{type(exc).__name__}: {exc}"
        if self.runtime.device_info_error:
            unavailable["device_info"] = self.runtime.device_info_error
        mlx: dict[str, Any] = {
            "version": self.runtime.version,
            "cache_bytes": cache,
            "runtime_peak_bytes": peak,
            "runtime_peak_scope": "process_since_start_or_external_reset",
            "memory_limit_bytes": limit,
            "allocator_held_bytes": (
                None if active is None or cache is None else active + cache
            ),
            "snapshot_consistency": "sequential_non_atomic",
            "raw_device_info": dict(self.runtime.device_info),
            "recommended_working_set_bytes": self.runtime.device_info.get(
                "max_recommended_working_set_size"
            ),
            "capabilities": {
                "bounded_sampling": True,
                "synchronized_host_timing": True,
                "owned_peak_reset": self.runtime.supports("reset_peak_memory"),
            },
        }
        metadata = {
            "framework": "mlx",
            "backend": "metal",
            "sampling_source": "mlx_allocator",
            "memory_scope": "process_allocator",
            "memory_model": "unified",
            "memory_capabilities": dict(self.capabilities),
            "mlx": mlx,
            **host,
        }
        return MemorySnapshot(
            timestamp,
            time.perf_counter_ns() - start,
            name,
            active,
            cache,
            peak,
            host.get("process_rss_bytes"),
            metadata,
            unavailable,
        )

    def telemetry_record(
        self,
        snapshot: MemorySnapshot,
        session: SessionSummary,
        *,
        event_type: str = "sample",
        sampling_interval_ms: int = 0,
        previous_active_bytes: int | None = None,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        change = None
        if snapshot.active_bytes is not None and previous_active_bytes is not None:
            change = snapshot.active_bytes - previous_active_bytes
        record: dict[str, Any] = {
            "schema_version": 4,
            "session_id": session.session_id,
            "timestamp_ns": snapshot.timestamp_ns,
            "event_type": event_type,
            "collector": COLLECTOR,
            "sampling_interval_ms": sampling_interval_ms,
            "pid": session.pid,
            "host": session.host,
            "device_id": 0,
            "job_id": session.job_id,
            "rank": session.rank,
            "local_rank": session.local_rank,
            "world_size": session.world_size,
            "allocator_allocated_bytes": snapshot.active_bytes,
            "allocator_change_bytes": change,
            **{key: None for key in _NULL_COUNTERS},
            "context": context,
            "metadata": {
                **snapshot.metadata,
                "collection_duration_ns": snapshot.collection_duration_ns,
                "unavailable_fields": dict(snapshot.unavailable),
                **(metadata or {}),
            },
        }
        validate_telemetry_record(record)
        return record


def _host_memory() -> dict[str, int]:
    memory = psutil.virtual_memory()
    return {
        "system_memory_total_bytes": memory.total,
        "system_memory_available_bytes": memory.available,
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }
