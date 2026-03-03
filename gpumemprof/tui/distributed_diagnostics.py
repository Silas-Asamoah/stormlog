"""Distributed diagnostics loaders and view-model builders for the TUI."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from gpumemprof.collective_attribution import (
    CollectiveAttributionResult,
    attribute_collective_memory,
    resolve_collective_attribution_config,
)
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
_COLLECTIVE_ATTRIBUTION_CONFIG = resolve_collective_attribution_config("medium")

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
_RANK_CONTEXT_PATTERN = re.compile(
    r"(?<!local_)(?:^|[^a-z0-9])rank[_-]?(?P<rank>\d+)(?:[^0-9]|$)",
    re.IGNORECASE,
)


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
    confidence: float | None = None
    reason_codes: list[str] = field(default_factory=list)


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
    confidence: float | None = None
    reason_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _TimelineRankContext:
    rank: int
    local_rank: int
    world_size: int
    source: str


@dataclass
class _TimelineRankAllocator:
    used_ranks: set[int] = field(default_factory=set)
    max_world_size: int = 1

    def register_events(self, events: Iterable[TelemetryEventV2]) -> None:
        for event in events:
            self.register_rank(event.rank, event.world_size)

    def register_rank(self, rank: int, world_size: int) -> None:
        self.used_ranks.add(rank)
        if world_size > self.max_world_size:
            self.max_world_size = world_size

    def next_available_rank(self) -> int:
        rank = 0
        while rank in self.used_ranks:
            rank += 1
        return rank


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
    rank_allocator = _TimelineRankAllocator()

    for input_path in paths:
        path = input_path.expanduser().resolve()
        if not path.exists():
            warnings.append(f"Path does not exist: {path}")
            continue

        if path.is_file():
            loaded, file_warnings = _load_artifact_file(
                path,
                rank_allocator=rank_allocator,
            )
            events.extend(loaded)
            warnings.extend(file_warnings)
            if loaded:
                sources_loaded.append(str(path))
            continue

        if path.is_dir():
            loaded, dir_warnings, dir_sources = _load_artifact_directory(
                path,
                rank_allocator=rank_allocator,
            )
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
    collective_by_rank = _group_collective_attribution_by_rank(events)

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

        row, rank_candidates = _build_rank_row(
            rank,
            rank_events,
            collective_by_rank.get(rank, []),
        )
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
    collective_attribution: list[CollectiveAttributionResult],
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

    candidates = _derive_rank_anomaly_candidates(
        rank,
        rank_events,
        collective_attribution,
    )
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
    collective_attribution: list[CollectiveAttributionResult],
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

    for attribution in collective_attribution:
        confidence = round(float(attribution.confidence), 3)
        reason_codes = sorted(set(attribution.reason_codes))
        reason_summary = ", ".join(reason_codes) if reason_codes else "no reason codes"
        attribution_ts = max(attribution.interval_start_ns, rank_events[0].timestamp_ns)
        candidates.append(
            _AnomalyCandidate(
                rank=rank,
                severity=_collective_severity(confidence),
                timestamp_ns=attribution_ts,
                signal=f"collective:{attribution.classification}",
                details=(
                    "Communication-attributed hidden-memory spike "
                    f"(confidence {confidence:.2f}; reasons: {reason_summary})."
                ),
                confidence=confidence,
                reason_codes=reason_codes,
            )
        )

    return candidates


def _group_collective_attribution_by_rank(
    events: list[TelemetryEventV2],
) -> dict[int, list[CollectiveAttributionResult]]:
    grouped: dict[int, list[CollectiveAttributionResult]] = {}
    attributions = attribute_collective_memory(
        events=events,
        config=_COLLECTIVE_ATTRIBUTION_CONFIG,
    )
    for attribution in attributions:
        grouped.setdefault(attribution.rank, []).append(attribution)
    return grouped


def _collective_severity(confidence: float) -> str:
    if confidence >= 0.8:
        return "critical"
    if confidence >= 0.6:
        return "warning"
    return "info"


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
            confidence=earliest.confidence,
            reason_codes=list(earliest.reason_codes),
        ),
        AnomalyIndicator(
            kind="most_severe",
            rank=most_severe.rank,
            severity=most_severe.severity,
            timestamp_ns=most_severe.timestamp_ns,
            signal=most_severe.signal,
            details=most_severe.details,
            confidence=most_severe.confidence,
            reason_codes=list(most_severe.reason_codes),
        ),
    ]


def _load_artifact_file(
    path: Path,
    *,
    rank_allocator: _TimelineRankAllocator | None = None,
) -> tuple[list[TelemetryEventV2], list[str]]:
    warnings: list[str] = []
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            events = load_telemetry_events(path, permissive_legacy=True)
            if rank_allocator is not None:
                rank_allocator.register_events(events)
            return events, warnings
        except Exception as exc:
            if path.name == "telemetry_timeline.json":
                synthesized, synth_warnings = _synthesize_events_from_timeline(
                    path,
                    rank_allocator=rank_allocator,
                )
                return synthesized, synth_warnings
            warnings.append(f"Failed to parse JSON telemetry file {path}: {exc}")
            return [], warnings

    if suffix == ".csv":
        events, csv_warnings = _load_csv_events(path)
        if rank_allocator is not None:
            rank_allocator.register_events(events)
        return events, csv_warnings

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
    *,
    rank_allocator: _TimelineRankAllocator | None = None,
) -> tuple[list[TelemetryEventV2], list[str], list[str]]:
    warnings: list[str] = []
    sources: list[str] = []
    events: list[TelemetryEventV2] = []
    allocator = rank_allocator or _TimelineRankAllocator()

    candidate_files = _discover_candidate_files(directory)
    for file_path in candidate_files:
        if file_path.name == "telemetry_timeline.json":
            continue
        loaded, file_warnings = _load_artifact_file(
            file_path,
            rank_allocator=allocator,
        )
        events.extend(loaded)
        warnings.extend(file_warnings)
        if loaded:
            sources.append(str(file_path))

    timeline_files = sorted(directory.rglob("telemetry_timeline.json"))
    for timeline_file in timeline_files:
        synthesized, synth_warnings = _synthesize_events_from_timeline(
            timeline_file,
            rank_allocator=allocator,
        )
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
    *,
    rank_allocator: _TimelineRankAllocator | None = None,
) -> tuple[list[TelemetryEventV2], list[str]]:
    warnings: list[str] = []
    allocator = rank_allocator or _TimelineRankAllocator()

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

    rank_context = _resolve_timeline_rank_context(
        timeline_file=timeline_file,
        payload=payload,
        rank_allocator=allocator,
    )

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
            "rank": rank_context.rank,
            "local_rank": rank_context.local_rank,
            "world_size": rank_context.world_size,
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
        f"assigned rank={rank_context.rank} world_size={rank_context.world_size} "
        f"(source={rank_context.source})."
    )
    return events, warnings


def _resolve_timeline_rank_context(
    *,
    timeline_file: Path,
    payload: Mapping[str, Any],
    rank_allocator: _TimelineRankAllocator,
) -> _TimelineRankContext:
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    identity = (
        payload.get("distributed_identity")
        if isinstance(payload.get("distributed_identity"), Mapping)
        else None
    )

    hinted_rank = _first_int(
        payload.get("rank"),
        metadata.get("rank") if isinstance(metadata, Mapping) else None,
        identity.get("rank") if isinstance(identity, Mapping) else None,
        minimum=0,
    )
    hinted_local_rank = _first_int(
        payload.get("local_rank"),
        metadata.get("local_rank") if isinstance(metadata, Mapping) else None,
        identity.get("local_rank") if isinstance(identity, Mapping) else None,
        minimum=0,
    )
    hinted_world_size = _first_int(
        payload.get("world_size"),
        metadata.get("world_size") if isinstance(metadata, Mapping) else None,
        identity.get("world_size") if isinstance(identity, Mapping) else None,
        minimum=1,
    )

    path_rank = _infer_rank_from_path(timeline_file)
    if hinted_rank is not None:
        rank = hinted_rank
        source = "payload"
    elif path_rank is not None:
        rank = path_rank
        source = "path"
    else:
        rank = rank_allocator.next_available_rank()
        source = "allocator"

    local_rank = hinted_local_rank if hinted_local_rank is not None else rank
    minimum_world_size = max(
        rank + 1,
        local_rank + 1,
        rank_allocator.max_world_size,
        len(rank_allocator.used_ranks | {rank}),
    )
    world_size = (
        hinted_world_size if hinted_world_size is not None else minimum_world_size
    )
    if world_size < minimum_world_size:
        world_size = minimum_world_size

    rank_allocator.register_rank(rank, world_size)
    return _TimelineRankContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        source=source,
    )


def _infer_rank_from_path(path: Path) -> int | None:
    for segment in [
        path.name,
        *(parent.name for parent in path.parents if parent.name),
    ]:
        match = _RANK_CONTEXT_PATTERN.search(segment)
        if match:
            try:
                return int(match.group("rank"))
            except (TypeError, ValueError):
                continue
    return None


def _first_int(*values: Any, minimum: int) -> int | None:
    for value in values:
        parsed = _coerce_optional_int(value, minimum=minimum)
        if parsed is not None:
            return parsed
    return None


def _coerce_optional_int(value: Any, *, minimum: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if parsed < minimum:
        return None
    return parsed


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
