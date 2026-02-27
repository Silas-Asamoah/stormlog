"""Distributed diagnostics loaders and view-model builders for the TUI."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from gpumemprof.gap_analysis import analyze_hidden_memory_gaps
from gpumemprof.telemetry import (
    TelemetryEventV2,
    load_telemetry_events,
    telemetry_event_from_record,
)
from gpumemprof.utils import format_bytes

logger = logging.getLogger(__name__)

GAP_RATIO_THRESHOLD = 0.05
_GAP_THRESHOLDS = {
    "gap_ratio_threshold": GAP_RATIO_THRESHOLD,
    "gap_spike_zscore": 2.0,
    "gap_drift_r_squared": 0.6,
    "gap_fragmentation_ratio": 0.3,
}
_EMPTY_REMEDIATION: dict[str, list[str]] = {
    "transient_spike": [],
    "persistent_drift": [],
    "fragmentation_like": [],
}

_ALERT_TYPES = frozenset({"warning", "critical", "error"})
_SEVERITY_ORDER = {"info": 1, "warning": 2, "critical": 3}
_ALERT_SEVERITY = {"warning": "warning", "critical": "critical", "error": "critical"}

_CSV_INT_FIELDS = frozenset(
    {
        "schema_version",
        "timestamp_ns",
        "sampling_interval_ms",
        "pid",
        "rank",
        "local_rank",
        "world_size",
        "device_id",
        "allocator_allocated_bytes",
        "allocator_reserved_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
        "allocator_change_bytes",
        "device_used_bytes",
        "device_free_bytes",
        "device_total_bytes",
        "memory_allocated",
        "memory_reserved",
        "memory_change",
        "total_memory",
    }
)
_CSV_FLOAT_FIELDS = frozenset({"timestamp"})


@dataclass
class ArtifactLoadResult:
    """Result of loading one or more distributed artifact inputs."""

    events: list[TelemetryEventV2]
    warnings: list[str]
    sources_loaded: list[str]


@dataclass
class RankDiagnosticsRow:
    """Per-rank metrics rendered by the diagnostics table."""

    rank: int
    availability: str
    samples: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    hidden_gap_latest_bytes: int
    hidden_gap_peak_abs_bytes: int
    has_anomaly: bool
    first_anomaly_timestamp_ns: int | None = None
    first_anomaly_signal: str | None = None


@dataclass
class AnomalyIndicator:
    """Top-level first-cause indicator for distributed diagnostics."""

    kind: str
    rank: int
    severity: str
    timestamp_ns: int
    signal: str
    details: str


@dataclass
class DistributedDiagnosticsModel:
    """Aggregated distributed diagnostics state for TUI rendering."""

    rows: list[RankDiagnosticsRow]
    indicators: list[AnomalyIndicator]
    expected_ranks: list[int]
    present_ranks: list[int]
    missing_ranks: list[int]
    per_rank_timelines: dict[int, dict[str, list[int]]]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _AnomalyCandidate:
    rank: int
    severity: str
    timestamp_ns: int
    signal: str
    details: str


def parse_rank_filter(expr: str, available_ranks: list[int]) -> set[int]:
    """Parse a rank filter expression (e.g., ``all`` or ``0,2,4-7``)."""
    available_set = set(available_ranks)
    if not available_set:
        return set()

    text = (expr or "").strip().lower()
    if not text or text in {"all", "*"}:
        return set(available_set)

    selected: set[int] = set()
    for token in text.split(","):
        chunk = token.strip()
        if not chunk:
            continue
        if "-" in chunk:
            parts = chunk.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid rank range: {chunk!r}")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                raise ValueError(f"Invalid rank range (start>end): {chunk!r}")
            selected.update(range(start, end + 1))
            continue

        selected.add(int(chunk))

    return {rank for rank in selected if rank in available_set}


def load_distributed_artifacts(paths: list[Path]) -> ArtifactLoadResult:
    """Load telemetry events from JSON/CSV files and artifact directories."""
    events: list[TelemetryEventV2] = []
    warnings: list[str] = []
    sources_loaded: list[str] = []

    for input_path in paths:
        path = input_path.expanduser().resolve()
        if not path.exists():
            warnings.append(f"Path does not exist: {path}")
            continue

        if path.is_file():
            loaded, file_warnings = _load_artifact_file(path)
            events.extend(loaded)
            warnings.extend(file_warnings)
            if loaded:
                sources_loaded.append(str(path))
            continue

        if path.is_dir():
            loaded, dir_warnings, dir_sources = _load_artifact_directory(path)
            events.extend(loaded)
            warnings.extend(dir_warnings)
            sources_loaded.extend(dir_sources)
            continue

        warnings.append(f"Unsupported path type: {path}")

    deduped = _dedupe_events(events)
    deduped.sort(key=lambda event: event.timestamp_ns)
    return ArtifactLoadResult(
        events=deduped,
        warnings=warnings,
        sources_loaded=sorted(set(sources_loaded)),
    )


def build_distributed_model(
    events: list[TelemetryEventV2],
    selected_ranks: set[int] | None = None,
) -> DistributedDiagnosticsModel:
    """Build rank-level distributed diagnostics and first-cause indicators."""
    if not events:
        return DistributedDiagnosticsModel(
            rows=[],
            indicators=[],
            expected_ranks=[],
            present_ranks=[],
            missing_ranks=[],
            per_rank_timelines={},
            warnings=["No telemetry events loaded."],
        )

    grouped: dict[int, list[TelemetryEventV2]] = {}
    world_sizes: set[int] = set()
    for event in sorted(events, key=lambda item: item.timestamp_ns):
        grouped.setdefault(event.rank, []).append(event)
        if event.world_size > 0:
            world_sizes.add(event.world_size)

    present_ranks = sorted(grouped.keys())
    expected_world_size = max(world_sizes) if world_sizes else len(present_ranks)
    expected_ranks = (
        list(range(expected_world_size)) if expected_world_size > 0 else present_ranks
    )

    warnings: list[str] = []
    if len(world_sizes) > 1:
        warnings.append(
            "Inconsistent world_size values detected; using max observed world_size."
        )

    selected = set(selected_ranks) if selected_ranks is not None else None
    filtered_expected = (
        [rank for rank in expected_ranks if rank in selected]
        if selected is not None
        else expected_ranks
    )
    filtered_present = (
        [rank for rank in present_ranks if rank in selected]
        if selected is not None
        else present_ranks
    )
    filtered_missing = sorted(set(filtered_expected) - set(filtered_present))

    rows: list[RankDiagnosticsRow] = []
    timelines: dict[int, dict[str, list[int]]] = {}
    candidates: list[_AnomalyCandidate] = []

    for rank in filtered_expected:
        rank_events = grouped.get(rank, [])
        if not rank_events:
            rows.append(
                RankDiagnosticsRow(
                    rank=rank,
                    availability="missing",
                    samples=0,
                    allocated_delta_bytes=0,
                    reserved_delta_bytes=0,
                    hidden_gap_latest_bytes=0,
                    hidden_gap_peak_abs_bytes=0,
                    has_anomaly=False,
                )
            )
            continue

        row, rank_candidates = _build_rank_row(rank, rank_events)
        rows.append(row)
        candidates.extend(rank_candidates)
        timelines[rank] = {
            "timestamps_ns": [event.timestamp_ns for event in rank_events],
            "allocated": [event.allocator_allocated_bytes for event in rank_events],
            "reserved": [event.allocator_reserved_bytes for event in rank_events],
            "gap": [
                event.device_used_bytes - event.allocator_reserved_bytes
                for event in rank_events
            ],
        }

    return DistributedDiagnosticsModel(
        rows=rows,
        indicators=_build_first_cause_indicators(candidates),
        expected_ranks=filtered_expected,
        present_ranks=filtered_present,
        missing_ranks=filtered_missing,
        per_rank_timelines=timelines,
        warnings=warnings,
    )


def _build_rank_row(
    rank: int,
    rank_events: list[TelemetryEventV2],
) -> tuple[RankDiagnosticsRow, list[_AnomalyCandidate]]:
    first_event = rank_events[0]
    last_event = rank_events[-1]

    allocated_delta = (
        last_event.allocator_allocated_bytes - first_event.allocator_allocated_bytes
    )
    reserved_delta = (
        last_event.allocator_reserved_bytes - first_event.allocator_reserved_bytes
    )
    gaps = [
        event.device_used_bytes - event.allocator_reserved_bytes
        for event in rank_events
    ]
    gap_latest = gaps[-1] if gaps else 0
    gap_peak_abs = max((abs(value) for value in gaps), default=0)

    candidates = _derive_rank_anomaly_candidates(rank, rank_events)
    earliest = (
        min(candidates, key=lambda candidate: candidate.timestamp_ns)
        if candidates
        else None
    )
    row = RankDiagnosticsRow(
        rank=rank,
        availability="present",
        samples=len(rank_events),
        allocated_delta_bytes=allocated_delta,
        reserved_delta_bytes=reserved_delta,
        hidden_gap_latest_bytes=gap_latest,
        hidden_gap_peak_abs_bytes=gap_peak_abs,
        has_anomaly=bool(candidates),
        first_anomaly_timestamp_ns=earliest.timestamp_ns if earliest else None,
        first_anomaly_signal=earliest.signal if earliest else None,
    )
    return row, candidates


def _derive_rank_anomaly_candidates(
    rank: int,
    rank_events: list[TelemetryEventV2],
) -> list[_AnomalyCandidate]:
    candidates: list[_AnomalyCandidate] = []
    first_gap_breach_ts: int | None = None

    for event in rank_events:
        if event.event_type in _ALERT_TYPES:
            severity = _ALERT_SEVERITY.get(event.event_type, "warning")
            candidates.append(
                _AnomalyCandidate(
                    rank=rank,
                    severity=severity,
                    timestamp_ns=event.timestamp_ns,
                    signal=f"alert:{event.event_type}",
                    details=event.context or "Alert event",
                )
            )

        if event.device_total_bytes and event.device_total_bytes > 0:
            gap_value = event.device_used_bytes - event.allocator_reserved_bytes
            gap_ratio = abs(gap_value) / event.device_total_bytes
            if gap_ratio >= GAP_RATIO_THRESHOLD:
                if first_gap_breach_ts is None:
                    first_gap_breach_ts = event.timestamp_ns
                candidates.append(
                    _AnomalyCandidate(
                        rank=rank,
                        severity="warning",
                        timestamp_ns=event.timestamp_ns,
                        signal="gap_ratio_breach",
                        details=f"gap ratio {gap_ratio:.1%} exceeded threshold",
                    )
                )

    gap_findings = analyze_hidden_memory_gaps(
        events=rank_events,
        thresholds=_GAP_THRESHOLDS,
        format_memory=format_bytes,
        remediation_by_classification=_EMPTY_REMEDIATION,
    )
    for finding in gap_findings:
        fallback_ts = first_gap_breach_ts or rank_events[0].timestamp_ns
        candidates.append(
            _AnomalyCandidate(
                rank=rank,
                severity=finding.severity,
                timestamp_ns=fallback_ts,
                signal=f"gap:{finding.classification}",
                details=finding.description,
            )
        )

    return candidates


def _build_first_cause_indicators(
    candidates: list[_AnomalyCandidate],
) -> list[AnomalyIndicator]:
    if not candidates:
        return []

    earliest = min(candidates, key=lambda candidate: candidate.timestamp_ns)
    most_severe = min(
        candidates,
        key=lambda candidate: (
            -_SEVERITY_ORDER.get(candidate.severity, 0),
            candidate.timestamp_ns,
        ),
    )

    return [
        AnomalyIndicator(
            kind="earliest",
            rank=earliest.rank,
            severity=earliest.severity,
            timestamp_ns=earliest.timestamp_ns,
            signal=earliest.signal,
            details=earliest.details,
        ),
        AnomalyIndicator(
            kind="most_severe",
            rank=most_severe.rank,
            severity=most_severe.severity,
            timestamp_ns=most_severe.timestamp_ns,
            signal=most_severe.signal,
            details=most_severe.details,
        ),
    ]


def _load_artifact_file(path: Path) -> tuple[list[TelemetryEventV2], list[str]]:
    warnings: list[str] = []
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            return load_telemetry_events(path, permissive_legacy=True), warnings
        except Exception as exc:
            if path.name == "telemetry_timeline.json":
                synthesized, synth_warnings = _synthesize_events_from_timeline(path)
                return synthesized, synth_warnings
            warnings.append(f"Failed to parse JSON telemetry file {path}: {exc}")
            return [], warnings

    if suffix == ".csv":
        return _load_csv_events(path)

    warnings.append(f"Unsupported artifact file type: {path}")
    return [], warnings


def _load_csv_events(path: Path) -> tuple[list[TelemetryEventV2], list[str]]:
    events: list[TelemetryEventV2] = []
    warnings: list[str] = []

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                try:
                    normalized = _normalize_csv_record(row)
                    event = telemetry_event_from_record(
                        normalized,
                        permissive_legacy=True,
                        default_collector="legacy.csv",
                        default_sampling_interval_ms=0,
                    )
                    events.append(event)
                except Exception as exc:
                    warnings.append(f"CSV parse error {path}:{line_number}: {exc}")
    except OSError as exc:
        warnings.append(f"Failed to read CSV file {path}: {exc}")

    return events, warnings


def _normalize_csv_record(row: Mapping[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw_value in row.items():
        value: Any = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if value == "":
            normalized[key] = None
            continue

        if key == "metadata":
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = {}
                normalized[key] = parsed if isinstance(parsed, dict) else {}
            else:
                normalized[key] = {}
            continue

        if key in _CSV_INT_FIELDS:
            normalized[key] = int(float(value))
            continue

        if key in _CSV_FLOAT_FIELDS:
            normalized[key] = float(value)
            continue

        normalized[key] = value

    return normalized


def _load_artifact_directory(
    directory: Path,
) -> tuple[list[TelemetryEventV2], list[str], list[str]]:
    warnings: list[str] = []
    sources: list[str] = []
    events: list[TelemetryEventV2] = []

    candidate_files = _discover_candidate_files(directory)
    for file_path in candidate_files:
        if file_path.name == "telemetry_timeline.json":
            continue
        loaded, file_warnings = _load_artifact_file(file_path)
        events.extend(loaded)
        warnings.extend(file_warnings)
        if loaded:
            sources.append(str(file_path))

    if events:
        return events, warnings, sources

    timeline_files = sorted(directory.rglob("telemetry_timeline.json"))
    for timeline_file in timeline_files:
        synthesized, synth_warnings = _synthesize_events_from_timeline(timeline_file)
        events.extend(synthesized)
        warnings.extend(synth_warnings)
        if synthesized:
            sources.append(str(timeline_file))

    if not events:
        warnings.append(f"No telemetry event payloads found in directory: {directory}")
    return events, warnings, sources


def _discover_candidate_files(directory: Path) -> list[Path]:
    patterns = (
        "**/events.json",
        "**/*events*.json",
        "**/*track*.json",
        "**/*events*.csv",
        "**/*track*.csv",
        "**/telemetry_timeline.json",
    )
    discovered: set[Path] = set()
    for pattern in patterns:
        discovered.update(path for path in directory.rglob(pattern) if path.is_file())
    return sorted(discovered)


def _synthesize_events_from_timeline(
    timeline_file: Path,
) -> tuple[list[TelemetryEventV2], list[str]]:
    warnings: list[str] = []

    try:
        with timeline_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"Failed to read timeline payload {timeline_file}: {exc}"]

    if not isinstance(payload, Mapping):
        return [], [f"Timeline payload is not a JSON object: {timeline_file}"]

    timestamps = payload.get("timestamps")
    allocated = payload.get("allocated")
    reserved = payload.get("reserved")
    if not isinstance(timestamps, list) or not isinstance(allocated, list):
        return [], [f"Invalid timeline payload shape in {timeline_file}"]

    if not timestamps or not allocated:
        return [], [f"Timeline payload has no samples in {timeline_file}"]

    if not isinstance(reserved, list):
        reserved = allocated

    sample_count = min(len(timestamps), len(allocated), len(reserved))
    if sample_count == 0:
        return [], [f"Timeline payload has no aligned samples in {timeline_file}"]

    events: list[TelemetryEventV2] = []
    previous_allocated = 0
    previous_timestamp = float(timestamps[0])

    for index in range(sample_count):
        timestamp = float(timestamps[index])
        allocated_bytes = int(allocated[index])
        reserved_bytes = int(reserved[index])
        interval_ms = (
            int(round((timestamp - previous_timestamp) * 1000)) if index > 0 else 0
        )
        record = {
            "timestamp": timestamp,
            "event_type": "sample",
            "collector": "gpumemprof.diagnose.timeline",
            "sampling_interval_ms": max(0, interval_ms),
            "pid": -1,
            "host": "unknown",
            "device_id": 0,
            "memory_allocated": allocated_bytes,
            "memory_reserved": reserved_bytes,
            "memory_change": allocated_bytes - previous_allocated,
            "device_used_bytes": max(allocated_bytes, reserved_bytes),
            "device_total_bytes": None,
            "context": "diagnose timeline sample",
            "job_id": None,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "metadata": {"source": "diagnose.telemetry_timeline"},
        }
        event = telemetry_event_from_record(
            record,
            permissive_legacy=True,
            default_collector="gpumemprof.diagnose.timeline",
            default_sampling_interval_ms=max(0, interval_ms),
        )
        events.append(event)
        previous_allocated = allocated_bytes
        previous_timestamp = timestamp

    warnings.append(
        "Synthesized telemetry events from telemetry_timeline.json; "
        "distributed rank metadata may be incomplete."
    )
    return events, warnings


def _dedupe_events(events: Iterable[TelemetryEventV2]) -> list[TelemetryEventV2]:
    unique: list[TelemetryEventV2] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (
            event.timestamp_ns,
            event.rank,
            event.local_rank,
            event.world_size,
            event.event_type,
            event.allocator_allocated_bytes,
            event.allocator_reserved_bytes,
            event.device_used_bytes,
            event.context or "",
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


__all__ = [
    "AnomalyIndicator",
    "ArtifactLoadResult",
    "DistributedDiagnosticsModel",
    "RankDiagnosticsRow",
    "build_distributed_model",
    "load_distributed_artifacts",
    "parse_rank_filter",
]
