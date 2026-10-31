"""MLX tracker view adapter; constructing it is an explicit runtime request."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from typing import Any

from stormlog.mlx.tracker import MemoryTracker


class MLXTrackerAdapter:
    backend = "mlx"
    device = "MLX Metal allocator (this process)"

    def __init__(
        self,
        *,
        sampling_interval: float,
        max_events: int,
        telemetry_sink_config: Any = None,
        **kwargs: Any,
    ) -> None:
        self.tracker = MemoryTracker(
            sampling_interval=sampling_interval,
            max_history=max_events,
            telemetry_sink_config=telemetry_sink_config,
            **kwargs,
        )
        self._cleared_at_ns = 0

    @property
    def is_tracking(self) -> bool:
        return self.tracker.is_tracking

    def start_tracking(self) -> None:
        self.tracker.start_tracking()

    def stop_tracking(self) -> None:
        self.tracker.stop_tracking()

    def get_session_summary(self) -> Any:
        return self.tracker.session_summary

    @property
    def events(self) -> list[Any]:
        return self.get_events()

    def get_events(self, since: float | None = None) -> list[Any]:
        return [
            self._event(record)
            for record in self.tracker.get_results().telemetry_events
            if record["timestamp_ns"] > self._cleared_at_ns
            and (since is None or record["timestamp_ns"] / 1e9 >= since)
        ]

    @staticmethod
    def _event(record: dict[str, Any]) -> Any:
        return SimpleNamespace(
            timestamp=record["timestamp_ns"] / 1e9,
            event_type=record["event_type"],
            context=record["context"],
            memory_allocated=record["allocator_allocated_bytes"],
            memory_reserved=None,
            memory_change=record["allocator_change_bytes"],
            record=record,
        )

    @staticmethod
    def _telemetry_record_from_event(event: Any) -> dict[str, Any]:
        return dict(event.record)

    def get_statistics(self) -> dict[str, Any]:
        result = self.tracker.get_results()
        latest = result.samples[-1] if result.samples else None
        return {
            "framework": "mlx",
            "backend": "metal",
            "memory_scope": "process_allocator",
            "current_memory_bytes": latest.active_bytes if latest else None,
            "cache_bytes": latest.cache_bytes if latest else None,
            "process_rss_bytes": latest.process_rss_bytes if latest else None,
            "runtime_lifetime_peak_bytes": (
                latest.runtime_peak_bytes if latest else None
            ),
            "max_memory_bytes": result.sampled_peak_bytes,
            "total_samples": result.total_samples,
            "total_alerts": result.total_alerts,
            "retained_samples": len(result.samples),
            "dropped_samples": result.history_dropped_samples,
            "collector_health_status": result.health["collector_health_status"],
            "collector_last_error": result.health.get("collector_last_error"),
            "mlx_version": self.tracker.runtime.version,
            "memory_capabilities": self.tracker.collector.capabilities,
        }

    def get_memory_timeline(self, interval: float = 1.0) -> dict[str, Any]:
        records = self.tracker.get_results().telemetry_events
        samples = [
            r
            for r in records
            if r["event_type"]
            in {"start", "sample", "stop", "phase_enter", "phase_exit"}
        ]
        return {
            "framework": "mlx",
            "timestamps": [r["timestamp_ns"] / 1e9 for r in samples],
            "allocated": [r["allocator_allocated_bytes"] for r in samples],
            "reserved": [None] * len(samples),
            "cache": [r["metadata"]["mlx"]["cache_bytes"] for r in samples],
            "rss": [r["metadata"].get("process_rss_bytes") for r in samples],
        }

    def clear_events(self) -> None:
        records = self.tracker.get_results().telemetry_events
        self._cleared_at_ns = records[-1]["timestamp_ns"] if records else 0

    def export_events(self, path: str, format: str = "csv") -> None:
        if format == "json":
            self.tracker.export(path)
            return
        if format != "csv":
            raise ValueError("MLX event export supports json or csv")
        records = self.tracker.get_results().telemetry_events
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]) if records else []
            )
            writer.writeheader()
            for record in records:
                writer.writerow({**record, "metadata": json.dumps(record["metadata"])})
