"""Derived compact rollups for append-only telemetry sinks."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .session import SessionSummary, now_ns
from .telemetry import LoadedTelemetrySession, TelemetryEvent
from .telemetry_classification import (
    COLLECTOR_TRANSITION_TYPES,
    event_severity,
    is_alert_event,
    is_oom_event,
)
from .telemetry_sink import (
    MANIFEST_FILENAME,
    TelemetrySinkManifest,
    resolve_telemetry_sink_manifest_path,
)

TELEMETRY_ROLLUP_SCHEMA_VERSION = 1
ROLLUP_FILENAME = "rollups.json"
DEFAULT_ROLLUP_WINDOW_SECONDS = 60
DEFAULT_ROLLUP_WINDOW_NS = DEFAULT_ROLLUP_WINDOW_SECONDS * 1_000_000_000

_ROLLUP_FORMAT = "stormlog.telemetry_rollups"


@dataclass(frozen=True)
class PeakValue:
    """Peak value and source timestamp for a counter."""

    value: int | None = None
    timestamp_ns: int | None = None
    rank: int | None = None
    device_id: int | None = None


@dataclass(frozen=True)
class CounterSummary:
    """Peak memory counters for a session, rank, or window."""

    allocator_allocated_bytes: PeakValue = field(default_factory=PeakValue)
    allocator_reserved_bytes: PeakValue = field(default_factory=PeakValue)
    device_used_bytes: PeakValue = field(default_factory=PeakValue)


@dataclass(frozen=True)
class AlertSummary:
    """Alert counts by severity and event type."""

    total_count: int = 0
    severity_counts: dict[str, int] = field(default_factory=dict)
    event_type_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectorHealthSummary:
    """Collector transition counts and degraded duration estimate."""

    transition_count: int = 0
    degraded_count: int = 0
    recovered_count: int = 0
    degraded_time_ns: int = 0
    last_status: str | None = None


@dataclass(frozen=True)
class OomSummary:
    """OOM marker counts."""

    marker_count: int = 0
    event_count: int = 0
    bundle_path_count: int = 0


@dataclass(frozen=True)
class RollupCoverage:
    """Retained raw evidence covered by this rollup file."""

    retained_segment_filenames: list[str] = field(default_factory=list)
    retained_segment_count: int = 0
    retained_event_count: int = 0
    retained_bytes: int = 0
    pruned_segment_count: int | None = None
    pruned_bytes: int | None = None


@dataclass(frozen=True)
class RankRollup:
    """Per-rank compact summary."""

    rank: int
    local_rank: int
    world_size: int
    event_count: int
    sample_count: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None
    counters: CounterSummary
    hidden_gap_first_bytes: int | None = None
    hidden_gap_latest_bytes: int | None = None
    hidden_gap_peak_bytes: int | None = None
    hidden_gap_delta_bytes: int | None = None
    hidden_gap_drift_bytes_per_second: float | None = None


@dataclass(frozen=True)
class WindowRollup:
    """Fixed-duration session window summary."""

    index: int
    start_ns: int
    end_ns: int
    event_count: int
    sample_count: int
    rank_count: int
    counters: CounterSummary
    alert_count: int = 0
    collector_transition_count: int = 0
    oom_count: int = 0


@dataclass(frozen=True)
class SessionRollup:
    """Compact derived summary for one telemetry session."""

    session: SessionSummary
    event_count: int
    sample_count: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None
    source_count: int
    sources_loaded: list[str]
    warnings: list[str]
    counters: CounterSummary
    alerts: AlertSummary
    collector_health: CollectorHealthSummary
    oom: OomSummary
    ranks: list[RankRollup] = field(default_factory=list)
    windows: list[WindowRollup] = field(default_factory=list)


@dataclass(frozen=True)
class TelemetryRollupFile:
    """Top-level rollup sidecar payload."""

    schema_version: int
    format: str
    generated_at_ns: int
    source_manifest: str | None
    source_manifest_schema_version: int | None
    window_duration_ns: int
    coverage: RollupCoverage
    sessions: list[SessionRollup] = field(default_factory=list)


@dataclass
class _PeakAccumulator:
    value: int | None = None
    timestamp_ns: int | None = None
    rank: int | None = None
    device_id: int | None = None

    def observe(self, value: int, event: TelemetryEvent) -> None:
        if self.value is None or value > self.value:
            self.value = value
            self.timestamp_ns = event.timestamp_ns
            self.rank = event.rank
            self.device_id = event.device_id

    def to_peak(self) -> PeakValue:
        return PeakValue(
            value=self.value,
            timestamp_ns=self.timestamp_ns,
            rank=self.rank,
            device_id=self.device_id,
        )


@dataclass
class _CounterAccumulator:
    allocator_allocated_bytes: _PeakAccumulator = field(
        default_factory=_PeakAccumulator
    )
    allocator_reserved_bytes: _PeakAccumulator = field(default_factory=_PeakAccumulator)
    device_used_bytes: _PeakAccumulator = field(default_factory=_PeakAccumulator)

    def observe(self, event: TelemetryEvent) -> None:
        if event.allocator_allocated_bytes is not None:
            self.allocator_allocated_bytes.observe(
                event.allocator_allocated_bytes,
                event,
            )
        if event.allocator_reserved_bytes is not None:
            self.allocator_reserved_bytes.observe(
                event.allocator_reserved_bytes,
                event,
            )
        if event.device_used_bytes is not None:
            self.device_used_bytes.observe(event.device_used_bytes, event)

    def to_summary(self) -> CounterSummary:
        return CounterSummary(
            allocator_allocated_bytes=self.allocator_allocated_bytes.to_peak(),
            allocator_reserved_bytes=self.allocator_reserved_bytes.to_peak(),
            device_used_bytes=self.device_used_bytes.to_peak(),
        )


@dataclass
class _RankAccumulator:
    rank: int
    local_rank: int
    world_size: int
    event_count: int = 0
    sample_count: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    counters: _CounterAccumulator = field(default_factory=_CounterAccumulator)
    hidden_gap_first_timestamp_ns: int | None = None
    hidden_gap_first_bytes: int | None = None
    hidden_gap_latest_timestamp_ns: int | None = None
    hidden_gap_latest_bytes: int | None = None
    hidden_gap_peak_bytes: int | None = None

    def observe(self, event: TelemetryEvent) -> None:
        self.event_count += 1
        self.first_timestamp_ns = _min_optional(
            self.first_timestamp_ns,
            event.timestamp_ns,
        )
        self.last_timestamp_ns = _max_optional(
            self.last_timestamp_ns,
            event.timestamp_ns,
        )
        self.counters.observe(event)
        if event.event_type != "sample":
            return
        self.sample_count += 1
        if event.device_used_bytes is None or event.allocator_reserved_bytes is None:
            return
        gap_bytes = event.device_used_bytes - event.allocator_reserved_bytes
        if self.hidden_gap_first_timestamp_ns is None:
            self.hidden_gap_first_timestamp_ns = event.timestamp_ns
            self.hidden_gap_first_bytes = gap_bytes
        self.hidden_gap_latest_timestamp_ns = event.timestamp_ns
        self.hidden_gap_latest_bytes = gap_bytes
        self.hidden_gap_peak_bytes = (
            gap_bytes
            if self.hidden_gap_peak_bytes is None
            else max(self.hidden_gap_peak_bytes, gap_bytes)
        )

    def to_rollup(self) -> RankRollup:
        drift = _drift_bytes_per_second(
            first_timestamp_ns=self.hidden_gap_first_timestamp_ns,
            first_gap_bytes=self.hidden_gap_first_bytes,
            latest_timestamp_ns=self.hidden_gap_latest_timestamp_ns,
            latest_gap_bytes=self.hidden_gap_latest_bytes,
        )
        return RankRollup(
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            event_count=self.event_count,
            sample_count=self.sample_count,
            first_timestamp_ns=self.first_timestamp_ns,
            last_timestamp_ns=self.last_timestamp_ns,
            counters=self.counters.to_summary(),
            hidden_gap_first_bytes=self.hidden_gap_first_bytes,
            hidden_gap_latest_bytes=self.hidden_gap_latest_bytes,
            hidden_gap_peak_bytes=self.hidden_gap_peak_bytes,
            hidden_gap_delta_bytes=(
                self.hidden_gap_latest_bytes - self.hidden_gap_first_bytes
                if (
                    self.hidden_gap_first_bytes is not None
                    and self.hidden_gap_latest_bytes is not None
                )
                else None
            ),
            hidden_gap_drift_bytes_per_second=drift,
        )


@dataclass
class _WindowAccumulator:
    index: int
    start_ns: int
    end_ns: int
    event_count: int = 0
    sample_count: int = 0
    ranks: set[int] = field(default_factory=set)
    counters: _CounterAccumulator = field(default_factory=_CounterAccumulator)
    alert_count: int = 0
    collector_transition_count: int = 0
    oom_count: int = 0

    def observe(self, event: TelemetryEvent) -> None:
        self.event_count += 1
        self.ranks.add(event.rank)
        self.counters.observe(event)
        if event.event_type == "sample":
            self.sample_count += 1
        if is_alert_event(event):
            self.alert_count += 1
        if event.event_type in COLLECTOR_TRANSITION_TYPES:
            self.collector_transition_count += 1
        if is_oom_event(event):
            self.oom_count += 1

    def to_rollup(self) -> WindowRollup:
        return WindowRollup(
            index=self.index,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            event_count=self.event_count,
            sample_count=self.sample_count,
            rank_count=len(self.ranks),
            counters=self.counters.to_summary(),
            alert_count=self.alert_count,
            collector_transition_count=self.collector_transition_count,
            oom_count=self.oom_count,
        )


def build_telemetry_rollups(
    sessions: Sequence[LoadedTelemetrySession],
    manifest: TelemetrySinkManifest | None,
    *,
    window_duration_ns: int = DEFAULT_ROLLUP_WINDOW_NS,
    coverage: RollupCoverage | None = None,
) -> TelemetryRollupFile:
    """Build compact rollups from loaded telemetry sessions."""

    if window_duration_ns <= 0:
        raise ValueError("window_duration_ns must be >= 1")
    resolved_coverage = coverage or _coverage_from_manifest(manifest)
    return TelemetryRollupFile(
        schema_version=TELEMETRY_ROLLUP_SCHEMA_VERSION,
        format=_ROLLUP_FORMAT,
        generated_at_ns=now_ns(),
        source_manifest=MANIFEST_FILENAME if manifest is not None else None,
        source_manifest_schema_version=(
            manifest.schema_version if manifest is not None else None
        ),
        window_duration_ns=window_duration_ns,
        coverage=resolved_coverage,
        sessions=[
            _build_session_rollup(session, window_duration_ns)
            for session in sorted(
                sessions,
                key=lambda item: (item.summary.started_at_ns, item.summary.session_id),
            )
        ],
    )


def write_telemetry_rollups(
    root_dir: str | Path,
    rollups: TelemetryRollupFile,
) -> Path:
    """Write rollups atomically next to a sink manifest."""

    rollup_path = resolve_telemetry_rollup_path(root_dir)
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = rollup_path.with_name(
        f"{rollup_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            telemetry_rollup_file_to_dict(rollups),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(rollup_path)
    _fsync_directory(rollup_path.parent)
    return rollup_path


def read_telemetry_rollups(path: str | Path) -> TelemetryRollupFile | None:
    """Read a rollup sidecar if it exists and has a valid current shape."""

    rollup_path = resolve_telemetry_rollup_path(path)
    if not rollup_path.exists():
        return None
    try:
        payload = json.loads(rollup_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return None
        rollups = telemetry_rollup_file_from_dict(payload)
    except Exception:
        return None
    manifest = _read_manifest_for_rollup_path(path)
    if manifest is not None and not _rollup_matches_manifest(rollups, manifest):
        return None
    return rollups


def resolve_telemetry_rollup_path(path: str | Path) -> Path:
    """Resolve a sink directory, manifest, segment, or rollup path to rollups.json."""

    resolved = Path(path)
    if resolved.is_file() and resolved.name == ROLLUP_FILENAME:
        return resolved
    manifest_path = resolve_telemetry_sink_manifest_path(resolved)
    if manifest_path is not None:
        return manifest_path.parent / ROLLUP_FILENAME
    if resolved.is_file():
        return resolved.parent / ROLLUP_FILENAME
    return resolved / ROLLUP_FILENAME


def telemetry_rollup_file_to_dict(rollups: TelemetryRollupFile) -> dict[str, Any]:
    """Serialize a rollup payload into a deterministic JSON-safe dictionary."""

    return asdict(rollups)


def telemetry_rollup_file_from_dict(
    payload: Mapping[str, Any],
) -> TelemetryRollupFile:
    """Deserialize a rollup payload from JSON."""

    schema_version = _required_int(payload, "schema_version", minimum=1)
    if schema_version != TELEMETRY_ROLLUP_SCHEMA_VERSION:
        raise ValueError("unsupported telemetry rollup schema_version")
    fmt = payload.get("format")
    if fmt != _ROLLUP_FORMAT:
        raise ValueError("unsupported telemetry rollup format")
    coverage_payload = _required_mapping(payload, "coverage")
    return TelemetryRollupFile(
        schema_version=schema_version,
        format=str(fmt),
        generated_at_ns=_required_int(payload, "generated_at_ns", minimum=0),
        source_manifest=_required_optional_string(payload, "source_manifest"),
        source_manifest_schema_version=_required_optional_int(
            payload,
            "source_manifest_schema_version",
            minimum=1,
        ),
        window_duration_ns=_required_int(payload, "window_duration_ns", minimum=1),
        coverage=RollupCoverage(
            retained_segment_filenames=_required_string_list(
                coverage_payload,
                "retained_segment_filenames",
            ),
            retained_segment_count=_required_int(
                coverage_payload,
                "retained_segment_count",
                minimum=0,
            ),
            retained_event_count=_required_int(
                coverage_payload,
                "retained_event_count",
                minimum=0,
            ),
            retained_bytes=_required_int(
                coverage_payload,
                "retained_bytes",
                minimum=0,
            ),
            pruned_segment_count=_required_optional_int(
                coverage_payload,
                "pruned_segment_count",
                minimum=0,
            ),
            pruned_bytes=_required_optional_int(
                coverage_payload,
                "pruned_bytes",
                minimum=0,
            ),
        ),
        sessions=[
            _session_rollup_from_dict(_ensure_mapping(item, "sessions item"))
            for item in _required_list(payload, "sessions")
        ],
    )


def _build_session_rollup(
    loaded: LoadedTelemetrySession,
    window_duration_ns: int,
) -> SessionRollup:
    events = sorted(loaded.events, key=lambda event: event.timestamp_ns)
    counters = _CounterAccumulator()
    alert_summary = _AlertAccumulator()
    collector_health = _CollectorHealthAccumulator()
    oom_summary = _OomAccumulator()
    ranks: dict[int, _RankAccumulator] = {}
    windows: dict[int, _WindowAccumulator] = {}
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    sample_count = 0
    session_start_ns = loaded.summary.started_at_ns

    for event in events:
        first_timestamp_ns = _min_optional(first_timestamp_ns, event.timestamp_ns)
        last_timestamp_ns = _max_optional(last_timestamp_ns, event.timestamp_ns)
        counters.observe(event)
        alert_summary.observe(event)
        collector_health.observe(event)
        oom_summary.observe(event)
        if event.event_type == "sample":
            sample_count += 1

        rank_accumulator = ranks.setdefault(
            event.rank,
            _RankAccumulator(
                rank=event.rank,
                local_rank=event.local_rank,
                world_size=event.world_size,
            ),
        )
        rank_accumulator.observe(event)

        window_index = max(
            0, (event.timestamp_ns - session_start_ns) // window_duration_ns
        )
        window_start_ns = session_start_ns + window_index * window_duration_ns
        window = windows.setdefault(
            window_index,
            _WindowAccumulator(
                index=window_index,
                start_ns=window_start_ns,
                end_ns=window_start_ns + window_duration_ns,
            ),
        )
        window.observe(event)

    collector_summary = collector_health.to_summary(
        terminal_ns=loaded.summary.ended_at_ns or last_timestamp_ns
    )
    return SessionRollup(
        session=loaded.summary,
        event_count=len(events),
        sample_count=sample_count,
        first_timestamp_ns=first_timestamp_ns,
        last_timestamp_ns=last_timestamp_ns,
        source_count=len(loaded.sources_loaded),
        sources_loaded=sorted(loaded.sources_loaded),
        warnings=list(loaded.warnings),
        counters=counters.to_summary(),
        alerts=alert_summary.to_summary(),
        collector_health=collector_summary,
        oom=oom_summary.to_summary(),
        ranks=[
            rank.to_rollup()
            for rank in sorted(ranks.values(), key=lambda item: item.rank)
        ],
        windows=[
            window.to_rollup()
            for window in sorted(windows.values(), key=lambda item: item.index)
        ],
    )


@dataclass
class _AlertAccumulator:
    total_count: int = 0
    severity_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    event_type_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def observe(self, event: TelemetryEvent) -> None:
        if not is_alert_event(event):
            return
        self.total_count += 1
        self.severity_counts[event_severity(event)] += 1
        self.event_type_counts[event.event_type] += 1

    def to_summary(self) -> AlertSummary:
        return AlertSummary(
            total_count=self.total_count,
            severity_counts=dict(sorted(self.severity_counts.items())),
            event_type_counts=dict(sorted(self.event_type_counts.items())),
        )


@dataclass
class _CollectorHealthAccumulator:
    transition_count: int = 0
    degraded_count: int = 0
    recovered_count: int = 0
    degraded_since_ns: int | None = None
    degraded_time_ns: int = 0
    last_status: str | None = None

    def observe(self, event: TelemetryEvent) -> None:
        if event.event_type not in COLLECTOR_TRANSITION_TYPES:
            status = event.metadata.get("collector_health_status")
            if isinstance(status, str) and status.strip():
                self.last_status = status.strip().lower()
            return
        self.transition_count += 1
        if event.event_type == "collector_degraded":
            self.degraded_count += 1
            self.degraded_since_ns = event.timestamp_ns
            self.last_status = "degraded"
            return
        self.recovered_count += 1
        if self.degraded_since_ns is not None:
            self.degraded_time_ns += max(0, event.timestamp_ns - self.degraded_since_ns)
            self.degraded_since_ns = None
        self.last_status = "healthy"

    def to_summary(self, terminal_ns: int | None) -> CollectorHealthSummary:
        degraded_time_ns = self.degraded_time_ns
        if self.degraded_since_ns is not None and terminal_ns is not None:
            degraded_time_ns += max(0, terminal_ns - self.degraded_since_ns)
        return CollectorHealthSummary(
            transition_count=self.transition_count,
            degraded_count=self.degraded_count,
            recovered_count=self.recovered_count,
            degraded_time_ns=degraded_time_ns,
            last_status=self.last_status,
        )


@dataclass
class _OomAccumulator:
    marker_count: int = 0
    event_count: int = 0
    bundle_paths: set[str] = field(default_factory=set)

    def observe(self, event: TelemetryEvent) -> None:
        if not is_oom_event(event):
            return
        self.marker_count += 1
        if event.event_type == "error":
            self.event_count += 1
        bundle_path = event.metadata.get("oom_dump_path")
        if isinstance(bundle_path, str) and bundle_path.strip():
            self.bundle_paths.add(bundle_path)

    def to_summary(self) -> OomSummary:
        return OomSummary(
            marker_count=self.marker_count,
            event_count=self.event_count,
            bundle_path_count=len(self.bundle_paths),
        )


def rollup_coverage_from_manifest(
    manifest: TelemetrySinkManifest | None,
    *,
    pruned_segment_count: int | None = None,
    pruned_bytes: int | None = None,
) -> RollupCoverage:
    """Build sidecar coverage fields from a sink manifest."""

    if manifest is None:
        return RollupCoverage(
            pruned_segment_count=pruned_segment_count,
            pruned_bytes=pruned_bytes,
        )
    return RollupCoverage(
        retained_segment_filenames=[segment.filename for segment in manifest.segments],
        retained_segment_count=len(manifest.segments),
        retained_event_count=sum(segment.event_count for segment in manifest.segments),
        retained_bytes=sum(segment.size_bytes for segment in manifest.segments),
        pruned_segment_count=pruned_segment_count,
        pruned_bytes=pruned_bytes,
    )


def _rollup_matches_manifest(
    rollups: TelemetryRollupFile,
    manifest: TelemetrySinkManifest,
) -> bool:
    coverage = rollup_coverage_from_manifest(manifest)
    return (
        rollups.source_manifest_schema_version == manifest.schema_version
        and rollups.coverage.retained_segment_filenames
        == coverage.retained_segment_filenames
        and rollups.coverage.retained_event_count == coverage.retained_event_count
        and rollups.coverage.retained_bytes == coverage.retained_bytes
    )


def _read_manifest_for_rollup_path(path: str | Path) -> TelemetrySinkManifest | None:
    from .telemetry_sink import read_telemetry_sink_manifest

    resolved = Path(path)
    manifest_path = resolve_telemetry_sink_manifest_path(resolved)
    if manifest_path is None:
        if resolved.is_file() and resolved.name == ROLLUP_FILENAME:
            manifest_path = resolved.parent / MANIFEST_FILENAME
        elif resolved.is_dir():
            manifest_path = resolved / MANIFEST_FILENAME
    if manifest_path is None:
        return None
    return read_telemetry_sink_manifest(manifest_path)


def _session_rollup_from_dict(payload: Mapping[str, Any]) -> SessionRollup:
    from .session import session_summary_from_dict

    return SessionRollup(
        session=session_summary_from_dict(_required_mapping(payload, "session")),
        event_count=_required_int(payload, "event_count", minimum=0),
        sample_count=_required_int(payload, "sample_count", minimum=0),
        first_timestamp_ns=_required_optional_int(
            payload,
            "first_timestamp_ns",
            minimum=0,
        ),
        last_timestamp_ns=_required_optional_int(
            payload,
            "last_timestamp_ns",
            minimum=0,
        ),
        source_count=_required_int(payload, "source_count", minimum=0),
        sources_loaded=_required_string_list(payload, "sources_loaded"),
        warnings=_required_string_list(payload, "warnings"),
        counters=_counter_summary_from_dict(_required_mapping(payload, "counters")),
        alerts=_alert_summary_from_dict(_required_mapping(payload, "alerts")),
        collector_health=_collector_health_from_dict(
            _required_mapping(payload, "collector_health")
        ),
        oom=_oom_summary_from_dict(_required_mapping(payload, "oom")),
        ranks=[
            _rank_rollup_from_dict(_ensure_mapping(item, "ranks item"))
            for item in _required_list(payload, "ranks")
        ],
        windows=[
            _window_rollup_from_dict(_ensure_mapping(item, "windows item"))
            for item in _required_list(payload, "windows")
        ],
    )


def _rank_rollup_from_dict(payload: Mapping[str, Any]) -> RankRollup:
    return RankRollup(
        rank=_required_int(payload, "rank", minimum=0),
        local_rank=_required_int(payload, "local_rank", minimum=0),
        world_size=_required_int(payload, "world_size", minimum=1),
        event_count=_required_int(payload, "event_count", minimum=0),
        sample_count=_required_int(payload, "sample_count", minimum=0),
        first_timestamp_ns=_required_optional_int(
            payload,
            "first_timestamp_ns",
            minimum=0,
        ),
        last_timestamp_ns=_required_optional_int(
            payload,
            "last_timestamp_ns",
            minimum=0,
        ),
        counters=_counter_summary_from_dict(_required_mapping(payload, "counters")),
        hidden_gap_first_bytes=_required_optional_int(
            payload,
            "hidden_gap_first_bytes",
        ),
        hidden_gap_latest_bytes=_required_optional_int(
            payload,
            "hidden_gap_latest_bytes",
        ),
        hidden_gap_peak_bytes=_required_optional_int(
            payload,
            "hidden_gap_peak_bytes",
        ),
        hidden_gap_delta_bytes=_required_optional_int(
            payload,
            "hidden_gap_delta_bytes",
        ),
        hidden_gap_drift_bytes_per_second=_required_optional_float(
            payload,
            "hidden_gap_drift_bytes_per_second",
        ),
    )


def _window_rollup_from_dict(payload: Mapping[str, Any]) -> WindowRollup:
    return WindowRollup(
        index=_required_int(payload, "index", minimum=0),
        start_ns=_required_int(payload, "start_ns", minimum=0),
        end_ns=_required_int(payload, "end_ns", minimum=0),
        event_count=_required_int(payload, "event_count", minimum=0),
        sample_count=_required_int(payload, "sample_count", minimum=0),
        rank_count=_required_int(payload, "rank_count", minimum=0),
        counters=_counter_summary_from_dict(_required_mapping(payload, "counters")),
        alert_count=_required_int(payload, "alert_count", minimum=0),
        collector_transition_count=_required_int(
            payload,
            "collector_transition_count",
            minimum=0,
        ),
        oom_count=_required_int(payload, "oom_count", minimum=0),
    )


def _counter_summary_from_dict(payload: Mapping[str, Any]) -> CounterSummary:
    return CounterSummary(
        allocator_allocated_bytes=_peak_value_from_dict(
            _required_mapping(payload, "allocator_allocated_bytes")
        ),
        allocator_reserved_bytes=_peak_value_from_dict(
            _required_mapping(payload, "allocator_reserved_bytes")
        ),
        device_used_bytes=_peak_value_from_dict(
            _required_mapping(payload, "device_used_bytes")
        ),
    )


def _peak_value_from_dict(payload: Mapping[str, Any]) -> PeakValue:
    return PeakValue(
        value=_required_optional_int(payload, "value"),
        timestamp_ns=_required_optional_int(payload, "timestamp_ns", minimum=0),
        rank=_required_optional_int(payload, "rank", minimum=0),
        device_id=_required_optional_int(payload, "device_id"),
    )


def _alert_summary_from_dict(payload: Mapping[str, Any]) -> AlertSummary:
    return AlertSummary(
        total_count=_required_int(payload, "total_count", minimum=0),
        severity_counts=_required_int_mapping(payload, "severity_counts"),
        event_type_counts=_required_int_mapping(payload, "event_type_counts"),
    )


def _collector_health_from_dict(
    payload: Mapping[str, Any],
) -> CollectorHealthSummary:
    return CollectorHealthSummary(
        transition_count=_required_int(payload, "transition_count", minimum=0),
        degraded_count=_required_int(payload, "degraded_count", minimum=0),
        recovered_count=_required_int(payload, "recovered_count", minimum=0),
        degraded_time_ns=_required_int(payload, "degraded_time_ns", minimum=0),
        last_status=_required_optional_string(payload, "last_status"),
    )


def _oom_summary_from_dict(payload: Mapping[str, Any]) -> OomSummary:
    return OomSummary(
        marker_count=_required_int(payload, "marker_count", minimum=0),
        event_count=_required_int(payload, "event_count", minimum=0),
        bundle_path_count=_required_int(payload, "bundle_path_count", minimum=0),
    )


def _drift_bytes_per_second(
    *,
    first_timestamp_ns: int | None,
    first_gap_bytes: int | None,
    latest_timestamp_ns: int | None,
    latest_gap_bytes: int | None,
) -> float | None:
    if (
        first_timestamp_ns is None
        or first_gap_bytes is None
        or latest_timestamp_ns is None
        or latest_gap_bytes is None
    ):
        return None
    elapsed_ns = latest_timestamp_ns - first_timestamp_ns
    if elapsed_ns <= 0:
        return None
    return round((latest_gap_bytes - first_gap_bytes) / (elapsed_ns / 1_000_000_000), 6)


def _coverage_from_manifest(manifest: TelemetrySinkManifest | None) -> RollupCoverage:
    return rollup_coverage_from_manifest(manifest)


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _min_optional(current: int | None, candidate: int) -> int:
    return candidate if current is None else min(current, candidate)


def _max_optional(current: int | None, candidate: int) -> int:
    return candidate if current is None else max(current, candidate)


def _ensure_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in payload:
        raise ValueError(f"missing required {key}")
    return _ensure_mapping(payload[key], key)


def _required_list(payload: Mapping[str, Any], key: str) -> list[object]:
    if key not in payload:
        raise ValueError(f"missing required {key}")
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    values = _required_list(payload, key)
    strings: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain only strings")
        strings.append(item)
    return strings


def _required_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    if key not in payload:
        raise ValueError(f"missing required {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return int(value)


def _required_optional_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int | None:
    if key not in payload:
        raise ValueError(f"missing required {key}")
    if payload[key] is None:
        return None
    return _required_int(payload, key, minimum=minimum)


def _required_optional_float(
    payload: Mapping[str, Any],
    key: str,
) -> float | None:
    if key not in payload:
        raise ValueError(f"missing required {key}")
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


def _required_optional_string(
    payload: Mapping[str, Any],
    key: str,
) -> str | None:
    if key not in payload:
        raise ValueError(f"missing required {key}")
    value = payload[key]
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{key} must be a string or null")


def _required_int_mapping(payload: Mapping[str, Any], key: str) -> dict[str, int]:
    value = _required_mapping(payload, key)
    result: dict[str, int] = {}
    for item_key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{key} values must be non-negative integers")
        result[str(item_key)] = item
    return result


__all__ = [
    "DEFAULT_ROLLUP_WINDOW_NS",
    "DEFAULT_ROLLUP_WINDOW_SECONDS",
    "ROLLUP_FILENAME",
    "TELEMETRY_ROLLUP_SCHEMA_VERSION",
    "AlertSummary",
    "CollectorHealthSummary",
    "CounterSummary",
    "OomSummary",
    "PeakValue",
    "RankRollup",
    "RollupCoverage",
    "SessionRollup",
    "TelemetryRollupFile",
    "WindowRollup",
    "build_telemetry_rollups",
    "read_telemetry_rollups",
    "resolve_telemetry_rollup_path",
    "rollup_coverage_from_manifest",
    "telemetry_rollup_file_from_dict",
    "telemetry_rollup_file_to_dict",
    "write_telemetry_rollups",
]
