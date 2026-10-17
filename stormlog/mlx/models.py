"""JSON-safe MLX measurement types. Never retain workload array roots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from stormlog.session import SessionSummary


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp_ns: int
    collection_duration_ns: int
    name: str
    active_bytes: int | None
    cache_bytes: int | None
    runtime_peak_bytes: int | None
    process_rss_bytes: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileResult:
    name: str
    session_summary: SessionSummary
    started_at_ns: int
    ended_at_ns: int
    elapsed_ns: int
    completion_verified: bool
    status: str
    baseline: MemorySnapshot
    final: MemorySnapshot
    sampled_peak_bytes: int | None
    valid_sample_count: int
    peak_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackingResult:
    session_summary: SessionSummary
    total_samples: int
    valid_samples: int
    sampled_peak_bytes: int | None
    average_active_bytes: float | None
    min_active_bytes: int | None
    samples: list[MemorySnapshot]
    telemetry_events: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    total_events: int
    total_alerts: int
    history_window_limit: int
    health: dict[str, Any]
    sink_diagnostics: dict[str, Any]

    @property
    def history_dropped_samples(self) -> int:
        return self.total_samples - len(self.samples)

    @property
    def history_dropped_events(self) -> int:
        return self.total_events - len(self.telemetry_events)

    @property
    def history_dropped_alerts(self) -> int:
        return self.total_alerts - len(self.alerts)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            history_dropped_samples=self.history_dropped_samples,
            history_dropped_events=self.history_dropped_events,
            history_dropped_alerts=self.history_dropped_alerts,
        )
        return result
