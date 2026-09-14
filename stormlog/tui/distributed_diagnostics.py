"""Distributed diagnostics loaders and view-model builders for the TUI."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from stormlog.collective_attribution import (
    CollectiveAttributionResult,
    attribute_collective_memory,
    resolve_collective_attribution_config,
)
from stormlog.gap_analysis import analyze_hidden_memory_gaps

try:
    from stormlog.phases import (
        PhaseAttribution,
        PhaseReplayIndex,
        summarize_phase_attribution,
    )
except ImportError:  # pragma: no cover - phase package may land in another slice
    PhaseAttribution = Any  # type: ignore[assignment,misc]
    PhaseReplayIndex = Any  # type: ignore[assignment,misc]

    def summarize_phase_attribution(
        attribution: PhaseAttribution | None,
    ) -> str | None:
        return None


from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SessionSummary,
    create_session_summary,
    normalize_session_status,
    select_default_loaded_session,
    session_summary_from_dict,
    sort_session_summaries,
    stable_legacy_session_id,
    update_session_summary,
)
from stormlog.telemetry import (
    LoadedTelemetrySession,
    TelemetryEvent,
    TelemetryEventV2,
    load_telemetry_sessions,
    telemetry_event_from_record,
)
from stormlog.telemetry_sink import resolve_telemetry_sink_segment_paths
from stormlog.timeline_markers import TimelineMarker, derive_timeline_markers
from stormlog.utils import format_bytes

logger = logging.getLogger(__name__)

TelemetryCompatibleEvent = TelemetryEvent | TelemetryEventV2

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
    r"(?:^|[^a-z0-9])(?<!local[_-])rank[_-]?(?P<rank>\d+)(?:[^0-9]|$)",
    re.IGNORECASE,
)


@dataclass
class ArtifactLoadResult:
    """Result of loading one or more distributed artifact inputs."""

    events: list[TelemetryEvent]
    sessions: list[LoadedTelemetrySession]
    selected_session_id: str | None
    warnings: list[str]
    sources_loaded: list[str]
    markers: list[TimelineMarker] = field(default_factory=list)


@dataclass
class RankDiagnosticsRow:
    """Per-rank metrics rendered by the diagnostics table."""

    rank: int
    availability: str
    samples: int
    allocated_delta_bytes: int | None
    reserved_delta_bytes: int | None
    device_used_delta_bytes: int | None
    hidden_gap_latest_bytes: int | None
    hidden_gap_peak_abs_bytes: int | None
    has_anomaly: bool
    first_anomaly_timestamp_ns: int | None = None
    first_anomaly_signal: str | None = None
    first_anomaly_phase_path: str | None = None


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
    phase_path: str | None = None


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
    markers_by_rank: dict[int, list[TimelineMarker]] = field(default_factory=dict)


@dataclass(frozen=True)
class _AnomalyCandidate:
    rank: int
    severity: str
    timestamp_ns: int
    signal: str
    details: str
    confidence: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    phase_path: str | None = None


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

    def register_events(self, events: Iterable[TelemetryCompatibleEvent]) -> None:
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


@dataclass
class _LoadedSessionAccumulator:
    summary: SessionSummary
    events: list[TelemetryEvent] = field(default_factory=list)
    sources_loaded: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


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
        selected.update(_parse_rank_token(chunk))

    return {rank for rank in selected if rank in available_set}


def _session_source_for_path(path: Path) -> str:
    return f"artifact:{path.name or path.resolve()}"


def _merge_session_summaries(
    existing: SessionSummary,
    incoming: SessionSummary,
) -> SessionSummary:
    return sort_session_summaries([existing, incoming])[0]


def _legacy_flat_file_default_session_id(path: Path) -> str:
    suffix = path.suffix.lower()
    kind = "csv" if suffix == ".csv" else "json"
    return stable_legacy_session_id(str(path.resolve()), kind)


def _is_legacy_sessionless_flat_file(
    path: Path,
    loaded_sessions: Sequence[LoadedTelemetrySession],
) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
        return False
    if not loaded_sessions:
        return False
    default_session_id = _legacy_flat_file_default_session_id(path)
    return all(
        loaded.summary.session_id == default_session_id for loaded in loaded_sessions
    )


def _collect_common_optional_string(values: Iterable[str | None]) -> str | None:
    distinct = {value for value in values if value is not None}
    if len(distinct) == 1:
        return next(iter(distinct))
    return None


def _merge_legacy_flat_sessions(
    loaded_sessions: Sequence[LoadedTelemetrySession],
) -> LoadedTelemetrySession:
    if not loaded_sessions:
        raise ValueError("legacy flat session merge requires at least one session")

    source_paths = _merged_source_paths(loaded_sessions)
    merged_session_id = stable_legacy_session_id(
        "distributed.legacy",
        *source_paths,
    )
    return _merge_loaded_session_group(
        loaded_sessions,
        source_paths=source_paths,
        source="stormlog.diagnostics.legacy",
        session_id=merged_session_id,
    )


def _merge_distributed_path_sessions(
    loaded_sessions: Sequence[LoadedTelemetrySession],
) -> LoadedTelemetrySession:
    if not loaded_sessions:
        raise ValueError("distributed session merge requires at least one session")

    source_paths = _merged_source_paths(loaded_sessions)
    summaries = [loaded.summary for loaded in loaded_sessions]
    job_id = _collect_common_optional_string(summary.job_id for summary in summaries)
    if job_id is None:
        raise ValueError("distributed session merge requires a shared job_id")

    world_size = max(summary.world_size for summary in summaries)
    merged_session_id = stable_legacy_session_id(
        "distributed.rank_group",
        job_id,
        world_size,
        *sorted(summary.session_id for summary in summaries),
    )
    return _merge_loaded_session_group(
        loaded_sessions,
        source_paths=source_paths,
        source="stormlog.diagnostics.distributed",
        session_id=merged_session_id,
    )


def _merged_source_paths(
    loaded_sessions: Sequence[LoadedTelemetrySession],
) -> list[str]:
    return sorted(
        {source for loaded in loaded_sessions for source in loaded.sources_loaded}
    )


def _merge_loaded_session_group(
    loaded_sessions: Sequence[LoadedTelemetrySession],
    *,
    source_paths: list[str],
    source: str,
    session_id: str,
) -> LoadedTelemetrySession:
    merged_events = _dedupe_events(
        replace(event, session_id=session_id)
        for loaded in loaded_sessions
        for event in loaded.events
    )
    merged_events.sort(key=lambda event: event.timestamp_ns)
    return LoadedTelemetrySession(
        summary=_create_merged_session_summary(
            [loaded.summary for loaded in loaded_sessions],
            source=source,
            session_id=session_id,
        ),
        events=merged_events,
        sources_loaded=source_paths,
        warnings=_unique_strings(
            warning for loaded in loaded_sessions for warning in loaded.warnings
        ),
    )


def _merged_session_lifecycle(
    summaries: Sequence[SessionSummary],
) -> tuple[str, int, int | None]:
    all_completed = all(
        summary.status == SESSION_STATUS_COMPLETED for summary in summaries
    )
    status = SESSION_STATUS_COMPLETED if all_completed else SESSION_STATUS_INCOMPLETE
    ended_at_candidates = [
        summary.ended_at_ns for summary in summaries if summary.ended_at_ns is not None
    ]
    ended_at_ns = (
        max(ended_at_candidates) if all_completed and ended_at_candidates else None
    )
    return status, min(summary.started_at_ns for summary in summaries), ended_at_ns


def _create_merged_session_summary(
    summaries: Sequence[SessionSummary],
    *,
    source: str,
    session_id: str,
) -> SessionSummary:
    status, started_at_ns, ended_at_ns = _merged_session_lifecycle(summaries)
    host = _collect_common_optional_string(summary.host for summary in summaries)
    return create_session_summary(
        source=source,
        status=status,
        session_id=session_id,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        host=host or "multiple",
        pid=(
            summaries[0].pid if len({summary.pid for summary in summaries}) == 1 else -1
        ),
        job_id=_collect_common_optional_string(summary.job_id for summary in summaries),
        rank=min(summary.rank for summary in summaries),
        local_rank=min(summary.local_rank for summary in summaries),
        world_size=max(summary.world_size for summary in summaries),
    )


def _accumulate_path_loaded_sessions(
    accumulators: dict[str, _LoadedSessionAccumulator],
    legacy_flat_sessions: list[LoadedTelemetrySession],
    *,
    source_path: Path,
    loaded_sessions: Sequence[LoadedTelemetrySession],
) -> None:
    if _is_legacy_sessionless_flat_file(source_path, loaded_sessions):
        legacy_flat_sessions.extend(loaded_sessions)
        return
    _accumulate_loaded_sessions(accumulators, loaded_sessions)


def _load_sessions_from_events(
    events: list[TelemetryEvent],
    *,
    source_path: Path,
    warnings: list[str] | None = None,
    summary_by_id: Mapping[str, SessionSummary] | None = None,
) -> list[LoadedTelemetrySession]:
    grouped: dict[str, list[TelemetryEvent]] = {}
    for event in events:
        grouped.setdefault(event.session_id, []).append(event)

    sessions: list[LoadedTelemetrySession] = []
    for session_id, session_events in grouped.items():
        ordered_events = _dedupe_events(session_events)
        ordered_events.sort(key=lambda event: event.timestamp_ns)
        summary = (
            dict(summary_by_id or {}).get(session_id)
            if summary_by_id is not None
            else None
        )
        if summary is None:
            summary = create_session_summary(
                source=_session_source_for_path(source_path),
                status=SESSION_STATUS_INCOMPLETE,
                session_id=session_id,
                started_at_ns=ordered_events[0].timestamp_ns,
                ended_at_ns=(
                    ordered_events[-1].timestamp_ns
                    if ordered_events[-1].event_type == "stop"
                    else None
                ),
                host=ordered_events[0].host,
                pid=ordered_events[0].pid,
                job_id=ordered_events[0].job_id,
                rank=ordered_events[0].rank,
                local_rank=ordered_events[0].local_rank,
                world_size=ordered_events[0].world_size,
            )
            if ordered_events[-1].event_type == "stop":
                summary = update_session_summary(
                    summary,
                    status=SESSION_STATUS_COMPLETED,
                    ended_at_ns=ordered_events[-1].timestamp_ns,
                )
        sessions.append(
            LoadedTelemetrySession(
                summary=summary,
                events=ordered_events,
                sources_loaded=[str(source_path)],
                warnings=list(warnings or []),
            )
        )

    return sessions


def _accumulate_loaded_sessions(
    accumulators: dict[str, _LoadedSessionAccumulator],
    loaded_sessions: Iterable[LoadedTelemetrySession],
) -> None:
    for loaded in loaded_sessions:
        session_id = loaded.summary.session_id
        accumulator = accumulators.get(session_id)
        if accumulator is None:
            accumulator = _LoadedSessionAccumulator(summary=loaded.summary)
            accumulators[session_id] = accumulator
        else:
            accumulator.summary = _merge_session_summaries(
                accumulator.summary,
                loaded.summary,
            )
        accumulator.events.extend(loaded.events)
        accumulator.sources_loaded.update(loaded.sources_loaded)
        accumulator.warnings.extend(loaded.warnings)


def _finalize_loaded_sessions(
    accumulators: Mapping[str, _LoadedSessionAccumulator],
) -> list[LoadedTelemetrySession]:
    ordered_ids = [
        summary.session_id
        for summary in sort_session_summaries(
            accumulator.summary for accumulator in accumulators.values()
        )
    ]
    sessions: list[LoadedTelemetrySession] = []
    for session_id in ordered_ids:
        accumulator = accumulators[session_id]
        events = _dedupe_events(accumulator.events)
        events.sort(key=lambda event: event.timestamp_ns)
        sessions.append(
            LoadedTelemetrySession(
                summary=accumulator.summary,
                events=events,
                sources_loaded=sorted(accumulator.sources_loaded),
                warnings=_unique_strings(accumulator.warnings),
            )
        )
    return sessions


def _order_loaded_sessions(
    sessions: Sequence[LoadedTelemetrySession],
) -> list[LoadedTelemetrySession]:
    order = {
        summary.session_id: index
        for index, summary in enumerate(
            sort_session_summaries(loaded.summary for loaded in sessions)
        )
    }
    return sorted(
        sessions,
        key=lambda loaded: (
            order.get(loaded.summary.session_id, 999),
            loaded.summary.session_id,
        ),
    )


def _synthesize_distributed_sessions(
    sessions: Sequence[LoadedTelemetrySession],
    candidate_session_ids: set[str],
) -> list[LoadedTelemetrySession]:
    grouped = _group_distributed_sessions(sessions, candidate_session_ids)

    merged_sessions: list[LoadedTelemetrySession] = []
    for (_, world_size), group in grouped.items():
        if len(group) <= 1 or len(group) > world_size:
            continue

        ranks = [loaded.summary.rank for loaded in group]
        if len(set(ranks)) != len(ranks):
            continue

        merged_sessions.append(_merge_distributed_path_sessions(group))

    return merged_sessions


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _load_artifact_input(
    path: Path,
    rank_allocator: _TimelineRankAllocator,
) -> tuple[list[LoadedTelemetrySession], list[str], bool]:
    if not path.exists():
        return [], [f"Path does not exist: {path}"], False
    if path.is_file():
        sessions, warnings = _load_artifact_file(path, rank_allocator=rank_allocator)
        return sessions, warnings, _is_legacy_sessionless_flat_file(path, sessions)
    if path.is_dir():
        sessions, warnings = _load_artifact_directory(
            path, rank_allocator=rank_allocator
        )
        return sessions, warnings, False
    return [], [f"Unsupported path type: {path}"], False


def _collect_artifact_sessions(
    paths: list[Path],
    warnings: list[str],
    sources_loaded: set[str],
) -> tuple[list[LoadedTelemetrySession], set[str]]:
    accumulators: dict[str, _LoadedSessionAccumulator] = {}
    legacy_flat_sessions: list[LoadedTelemetrySession] = []
    single_session_input_ids: set[str] = set()
    rank_allocator = _TimelineRankAllocator()

    for input_path in paths:
        path = input_path.expanduser().resolve()
        loaded_sessions, path_warnings, is_legacy = _load_artifact_input(
            path, rank_allocator
        )
        if is_legacy:
            legacy_flat_sessions.extend(loaded_sessions)
        else:
            if len(loaded_sessions) == 1:
                single_session_input_ids.add(loaded_sessions[0].summary.session_id)
            _accumulate_loaded_sessions(accumulators, loaded_sessions)
        warnings.extend(path_warnings)
        for loaded in loaded_sessions:
            sources_loaded.update(loaded.sources_loaded)

    if legacy_flat_sessions:
        merged_legacy = _merge_legacy_flat_sessions(legacy_flat_sessions)
        _accumulate_loaded_sessions(accumulators, [merged_legacy])
        sources_loaded.update(merged_legacy.sources_loaded)

    return _finalize_loaded_sessions(accumulators), single_session_input_ids


def _select_artifact_session(
    sessions: list[LoadedTelemetrySession],
    synthetic_sessions: list[LoadedTelemetrySession],
    session_id: str | None,
) -> LoadedTelemetrySession | None:
    if session_id is not None:
        selected = next(
            (loaded for loaded in sessions if loaded.summary.session_id == session_id),
            None,
        )
        if selected is None:
            raise ValueError(f"Requested session_id not found: {session_id}")
        return selected
    if synthetic_sessions:
        selected = cast(
            LoadedTelemetrySession | None,
            select_default_loaded_session(synthetic_sessions),
        )
        if selected is not None:
            return selected
    return cast(LoadedTelemetrySession | None, select_default_loaded_session(sessions))


def load_distributed_artifacts(
    paths: list[Path],
    session_id: str | None = None,
) -> ArtifactLoadResult:
    """Load telemetry events from JSON/CSV files and artifact directories."""
    warnings: list[str] = []
    sources_loaded: set[str] = set()
    sessions, single_session_input_ids = _collect_artifact_sessions(
        paths, warnings, sources_loaded
    )
    synthetic_sessions = _synthesize_distributed_sessions(
        sessions,
        single_session_input_ids,
    )
    if synthetic_sessions:
        sessions = _order_loaded_sessions([*sessions, *synthetic_sessions])
    selected = _select_artifact_session(sessions, synthetic_sessions, session_id)

    selected_events = list(selected.events) if selected is not None else []
    selected_markers = (
        derive_timeline_markers(selected_events) if selected_events else []
    )
    combined_warnings = _unique_strings(
        list(warnings) + [warning for loaded in sessions for warning in loaded.warnings]
    )
    return ArtifactLoadResult(
        events=selected_events,
        sessions=sessions,
        selected_session_id=(
            selected.summary.session_id if selected is not None else None
        ),
        warnings=combined_warnings,
        sources_loaded=sorted(sources_loaded),
        markers=selected_markers,
    )


def _group_rank_events(
    events: Sequence[TelemetryCompatibleEvent],
) -> tuple[
    dict[int, list[TelemetryCompatibleEvent]],
    dict[int, list[TelemetryCompatibleEvent]],
    set[int],
]:
    grouped: dict[int, list[TelemetryCompatibleEvent]] = {}
    sample_grouped: dict[int, list[TelemetryCompatibleEvent]] = {}
    world_sizes: set[int] = set()
    for event in sorted(events, key=lambda item: item.timestamp_ns):
        grouped.setdefault(event.rank, []).append(event)
        if _is_sample_event(event):
            sample_grouped.setdefault(event.rank, []).append(event)
        if event.world_size > 0:
            world_sizes.add(event.world_size)
    return grouped, sample_grouped, world_sizes


def _expected_ranks(present_ranks: list[int], world_sizes: set[int]) -> list[int]:
    expected_world_size = max(world_sizes) if world_sizes else len(present_ranks)
    return sorted(
        set(present_ranks)
        | (
            set(range(expected_world_size))
            if expected_world_size > 0
            else set(present_ranks)
        )
    )


def _filter_ranks(ranks: list[int], selected_ranks: set[int] | None) -> list[int]:
    if selected_ranks is None:
        return ranks
    return [rank for rank in ranks if rank in selected_ranks]


def _build_rank_timeline(
    samples: Sequence[TelemetryCompatibleEvent],
) -> dict[str, list[int]]:
    allocator_samples = [
        event
        for event in samples
        if event.allocator_allocated_bytes is not None
        and event.allocator_reserved_bytes is not None
        and event.device_used_bytes is not None
    ]
    if allocator_samples:
        return {
            "timestamps_ns": [event.timestamp_ns for event in allocator_samples],
            "allocated": [
                event.allocator_allocated_bytes
                for event in allocator_samples
                if event.allocator_allocated_bytes is not None
            ],
            "reserved": [
                event.allocator_reserved_bytes
                for event in allocator_samples
                if event.allocator_reserved_bytes is not None
            ],
            "gap": [
                event.device_used_bytes - event.allocator_reserved_bytes
                for event in allocator_samples
                if event.device_used_bytes is not None
                and event.allocator_reserved_bytes is not None
            ],
        }
    else:
        device_samples = [
            event for event in samples if event.device_used_bytes is not None
        ]
        return {
            "timestamps_ns": [event.timestamp_ns for event in device_samples],
            "device_used": [
                event.device_used_bytes
                for event in device_samples
                if event.device_used_bytes is not None
            ],
        }


def build_distributed_model(
    events: Sequence[TelemetryCompatibleEvent],
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

    phase_resolver = (
        PhaseReplayIndex.from_events(events)
        if hasattr(PhaseReplayIndex, "from_events")
        else None
    )
    grouped, sample_grouped, world_sizes = _group_rank_events(events)
    present_ranks = sorted(grouped.keys())
    expected_ranks = _expected_ranks(present_ranks, world_sizes)

    warnings: list[str] = []
    if len(world_sizes) > 1:
        warnings.append(
            "Inconsistent world_size values detected; using max observed world_size."
        )
    allocator_samples = [
        event
        for rank_samples in sample_grouped.values()
        for event in rank_samples
        if event.allocator_allocated_bytes is not None
        and event.allocator_reserved_bytes is not None
    ]
    if sample_grouped and not allocator_samples:
        warnings.append(
            "Allocator-native diagnostics are unavailable; showing device memory "
            "usage only."
        )

    selected = set(selected_ranks) if selected_ranks is not None else None
    filtered_expected = _filter_ranks(expected_ranks, selected)
    filtered_present = _filter_ranks(present_ranks, selected)
    filtered_missing = sorted(set(filtered_expected) - set(filtered_present))

    markers_by_rank = _group_timeline_markers_by_rank(
        derive_timeline_markers(events),
        filtered_present,
    )
    rows: list[RankDiagnosticsRow] = []
    timelines: dict[int, dict[str, list[int]]] = {}
    candidates: list[_AnomalyCandidate] = []
    collective_by_rank = _group_collective_attribution_by_rank(
        events,
        phase_resolver=phase_resolver,
    )

    for rank in filtered_expected:
        rank_events = grouped.get(rank, [])
        rank_samples = sample_grouped.get(rank, [])
        if not rank_events:
            rows.append(
                RankDiagnosticsRow(
                    rank=rank,
                    availability="missing",
                    samples=0,
                    allocated_delta_bytes=None,
                    reserved_delta_bytes=None,
                    device_used_delta_bytes=None,
                    hidden_gap_latest_bytes=None,
                    hidden_gap_peak_abs_bytes=None,
                    has_anomaly=False,
                )
            )
            continue

        row, rank_candidates = _build_rank_row(
            rank,
            rank_events,
            rank_samples,
            collective_by_rank.get(rank, []),
            phase_resolver=phase_resolver,
        )
        rows.append(row)
        candidates.extend(rank_candidates)
        timelines[rank] = _build_rank_timeline(rank_samples)

    return DistributedDiagnosticsModel(
        rows=rows,
        indicators=_build_first_cause_indicators(candidates),
        expected_ranks=filtered_expected,
        present_ranks=filtered_present,
        missing_ranks=filtered_missing,
        per_rank_timelines=timelines,
        warnings=warnings,
        markers_by_rank=markers_by_rank,
    )


def _group_timeline_markers_by_rank(
    markers: Sequence[TimelineMarker],
    ranks: Sequence[int],
) -> dict[int, list[TimelineMarker]]:
    grouped: dict[int, list[TimelineMarker]] = {rank: [] for rank in ranks}
    rank_set = set(ranks)
    for marker in markers:
        if marker.rank is None:
            for rank in ranks:
                grouped[rank].append(marker)
            continue
        if marker.rank in rank_set:
            grouped.setdefault(marker.rank, []).append(marker)
    return {rank: values for rank, values in grouped.items() if values}


def _build_rank_row(
    rank: int,
    rank_events: Sequence[TelemetryCompatibleEvent],
    rank_samples: Sequence[TelemetryCompatibleEvent],
    collective_attribution: list[CollectiveAttributionResult],
    *,
    phase_resolver: PhaseReplayIndex | None = None,
) -> tuple[RankDiagnosticsRow, list[_AnomalyCandidate]]:
    allocated_delta, reserved_delta, device_used_delta = _rank_memory_deltas(
        rank_samples
    )
    gaps = [
        event.device_used_bytes - event.allocator_reserved_bytes
        for event in rank_samples
        if event.device_used_bytes is not None
        and event.allocator_reserved_bytes is not None
    ]
    gap_latest = gaps[-1] if gaps else None
    gap_peak_abs = max((abs(value) for value in gaps), default=None)

    candidates = _derive_rank_anomaly_candidates(
        rank,
        rank_events,
        rank_samples,
        collective_attribution,
        phase_resolver=phase_resolver,
    )
    earliest = (
        min(candidates, key=lambda candidate: candidate.timestamp_ns)
        if candidates
        else None
    )
    row = RankDiagnosticsRow(
        rank=rank,
        availability="present",
        samples=len(rank_samples),
        allocated_delta_bytes=allocated_delta,
        reserved_delta_bytes=reserved_delta,
        device_used_delta_bytes=device_used_delta,
        hidden_gap_latest_bytes=gap_latest,
        hidden_gap_peak_abs_bytes=gap_peak_abs,
        has_anomaly=bool(candidates),
        first_anomaly_timestamp_ns=earliest.timestamp_ns if earliest else None,
        first_anomaly_signal=earliest.signal if earliest else None,
        first_anomaly_phase_path=earliest.phase_path if earliest else None,
    )
    return row, candidates


def _derive_rank_anomaly_candidates(
    rank: int,
    rank_events: Sequence[TelemetryCompatibleEvent],
    rank_samples: Sequence[TelemetryCompatibleEvent],
    collective_attribution: list[CollectiveAttributionResult],
    *,
    phase_resolver: PhaseReplayIndex | None = None,
) -> list[_AnomalyCandidate]:
    candidates: list[_AnomalyCandidate] = []
    _append_alert_candidates(rank, rank_events, candidates, phase_resolver)

    first_gap_breach_ts = _append_gap_ratio_candidates(
        rank, rank_samples, candidates, phase_resolver
    )

    allocator_events = [
        event
        for event in rank_events
        if event.allocator_allocated_bytes is not None
        and event.allocator_reserved_bytes is not None
        and event.device_used_bytes is not None
    ]
    gap_findings = analyze_hidden_memory_gaps(
        events=cast(Sequence[TelemetryEventV2], allocator_events),
        thresholds=_GAP_THRESHOLDS,
        format_memory=format_bytes,
        remediation_by_classification=_EMPTY_REMEDIATION,
        phase_resolver=phase_resolver,
    )
    for finding in gap_findings:
        fallback_ts = first_gap_breach_ts or (
            rank_samples[0].timestamp_ns
            if rank_samples
            else rank_events[0].timestamp_ns
        )
        phase_path = summarize_phase_attribution(finding.phase_attribution)
        candidates.append(
            _AnomalyCandidate(
                rank=rank,
                severity=finding.severity,
                timestamp_ns=finding.evidence_timestamp_ns or fallback_ts,
                signal=f"gap:{finding.classification}",
                details=finding.description,
                phase_path=phase_path,
            )
        )

    for attribution in collective_attribution:
        confidence = round(float(attribution.confidence), 3)
        reason_codes = sorted(set(attribution.reason_codes))
        reason_summary = ", ".join(reason_codes) if reason_codes else "no reason codes"
        attribution_ts = max(
            attribution.interval_start_ns,
            (
                rank_samples[0].timestamp_ns
                if rank_samples
                else rank_events[0].timestamp_ns
            ),
        )
        phase_path = summarize_phase_attribution(attribution.phase_attribution)
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
                phase_path=phase_path,
            )
        )

    return candidates


def _is_sample_event(event: TelemetryCompatibleEvent) -> bool:
    return str(event.event_type).strip().lower() == "sample"


def _group_collective_attribution_by_rank(
    events: Sequence[TelemetryCompatibleEvent],
    *,
    phase_resolver: PhaseReplayIndex | None = None,
) -> dict[int, list[CollectiveAttributionResult]]:
    grouped: dict[int, list[CollectiveAttributionResult]] = {}
    allocator_events = [
        event
        for event in events
        if event.allocator_allocated_bytes is not None
        and event.allocator_reserved_bytes is not None
        and event.device_used_bytes is not None
    ]
    attributions = attribute_collective_memory(
        events=cast(Sequence[TelemetryEventV2], allocator_events),
        config=_COLLECTIVE_ATTRIBUTION_CONFIG,
        phase_resolver=phase_resolver,
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
            details=_format_indicator_details(earliest),
            confidence=earliest.confidence,
            reason_codes=list(earliest.reason_codes),
            phase_path=earliest.phase_path,
        ),
        AnomalyIndicator(
            kind="most_severe",
            rank=most_severe.rank,
            severity=most_severe.severity,
            timestamp_ns=most_severe.timestamp_ns,
            signal=most_severe.signal,
            details=_format_indicator_details(most_severe),
            confidence=most_severe.confidence,
            reason_codes=list(most_severe.reason_codes),
            phase_path=most_severe.phase_path,
        ),
    ]


def _format_indicator_details(candidate: _AnomalyCandidate) -> str:
    if not candidate.phase_path:
        return candidate.details
    return f"{candidate.details} Phase: {candidate.phase_path}."


def _load_artifact_file(
    path: Path,
    *,
    rank_allocator: _TimelineRankAllocator | None = None,
) -> tuple[list[LoadedTelemetrySession], list[str]]:
    warnings: list[str] = []
    suffix = path.suffix.lower()

    if suffix in {".json", ".jsonl"}:
        try:
            sessions = load_telemetry_sessions(path, permissive_legacy=True)
            if rank_allocator is not None:
                for loaded in sessions:
                    rank_allocator.register_events(loaded.events)
            return sessions, warnings
        except Exception as exc:
            if path.name == "telemetry_timeline.json":
                synthesized, synth_warnings = _synthesize_sessions_from_timeline(
                    path,
                    rank_allocator=rank_allocator,
                )
                return synthesized, synth_warnings
            warnings.append(f"Failed to parse JSON telemetry file {path}: {exc}")
            return [], warnings

    if suffix == ".csv":
        sessions, csv_warnings = _load_csv_sessions(path)
        if rank_allocator is not None:
            for loaded in sessions:
                rank_allocator.register_events(loaded.events)
        return sessions, csv_warnings

    warnings.append(f"Unsupported artifact file type: {path}")
    return [], warnings


def _load_csv_sessions(path: Path) -> tuple[list[LoadedTelemetrySession], list[str]]:
    events: list[TelemetryEvent] = []
    warnings: list[str] = []
    default_session_id = stable_legacy_session_id(str(path.resolve()), "csv")

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
                        default_session_id=default_session_id,
                    )
                    events.append(event)
                except Exception as exc:
                    warnings.append(f"CSV parse error {path}:{line_number}: {exc}")
    except OSError as exc:
        warnings.append(f"Failed to read CSV file {path}: {exc}")

    return (
        _load_sessions_from_events(
            events,
            source_path=path,
            warnings=warnings,
        ),
        warnings,
    )


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
) -> tuple[list[LoadedTelemetrySession], list[str]]:
    warnings: list[str] = []
    accumulators: dict[str, _LoadedSessionAccumulator] = {}
    legacy_flat_sessions: list[LoadedTelemetrySession] = []
    allocator = rank_allocator or _TimelineRankAllocator()

    sink_segment_paths = resolve_telemetry_sink_segment_paths(directory)
    if sink_segment_paths:
        try:
            sink_sessions = load_telemetry_sessions(directory, permissive_legacy=True)
            for loaded in sink_sessions:
                allocator.register_events(loaded.events)
            _accumulate_loaded_sessions(accumulators, sink_sessions)
        except Exception as exc:
            warnings.append(
                f"Failed to parse telemetry sink directory {directory}: {exc}"
            )

    candidate_files = _discover_candidate_files(directory)
    for file_path in candidate_files:
        if file_path.name == "telemetry_timeline.json":
            continue
        loaded_sessions, file_warnings = _load_artifact_file(
            file_path,
            rank_allocator=allocator,
        )
        _accumulate_path_loaded_sessions(
            accumulators,
            legacy_flat_sessions,
            source_path=file_path,
            loaded_sessions=loaded_sessions,
        )
        warnings.extend(file_warnings)

    timeline_files = sorted(directory.rglob("telemetry_timeline.json"))
    for timeline_file in timeline_files:
        synthesized_sessions, synth_warnings = _synthesize_sessions_from_timeline(
            timeline_file,
            rank_allocator=allocator,
        )
        _accumulate_loaded_sessions(accumulators, synthesized_sessions)
        warnings.extend(synth_warnings)

    if legacy_flat_sessions:
        _accumulate_loaded_sessions(
            accumulators,
            [_merge_legacy_flat_sessions(legacy_flat_sessions)],
        )

    sessions = _finalize_loaded_sessions(accumulators)
    if not sessions:
        warnings.append(f"No telemetry event payloads found in directory: {directory}")
    return sessions, warnings


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


def _created_iso_to_ns(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return int(datetime.fromisoformat(text).timestamp() * 1_000_000_000)
    except ValueError:
        return None


def _summary_with_rank_context(
    summary: SessionSummary,
    rank_context: _TimelineRankContext,
) -> SessionSummary:
    if (
        summary.rank == rank_context.rank
        and summary.local_rank == rank_context.local_rank
        and summary.world_size == rank_context.world_size
    ):
        return summary
    return SessionSummary(
        session_id=summary.session_id,
        status=summary.status,
        started_at_ns=summary.started_at_ns,
        ended_at_ns=summary.ended_at_ns,
        host=summary.host,
        pid=summary.pid,
        job_id=summary.job_id,
        rank=rank_context.rank,
        local_rank=rank_context.local_rank,
        world_size=rank_context.world_size,
        source=summary.source,
    )


def _load_diagnose_session_summary(
    timeline_file: Path,
    rank_context: _TimelineRankContext,
) -> SessionSummary | None:
    manifest_path = timeline_file.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None

    session_payload = payload.get("session")
    if isinstance(session_payload, Mapping):
        try:
            return _summary_with_rank_context(
                session_summary_from_dict(session_payload),
                rank_context,
            )
        except Exception:
            pass

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None

    try:
        status = normalize_session_status(
            payload.get("session_status", SESSION_STATUS_INCOMPLETE)
        )
    except ValueError:
        status = SESSION_STATUS_INCOMPLETE
    started_at_ns = _created_iso_to_ns(payload.get("created_iso"))
    ended_at_ns = started_at_ns if status != "running" else None
    return create_session_summary(
        source="stormlog.diagnose.bundle",
        status=status,
        session_id=session_id,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        host="unknown",
        pid=-1,
        rank=rank_context.rank,
        local_rank=rank_context.local_rank,
        world_size=rank_context.world_size,
    )


def _synthesize_sessions_from_timeline(
    timeline_file: Path,
    *,
    rank_allocator: _TimelineRankAllocator | None = None,
) -> tuple[list[LoadedTelemetrySession], list[str]]:
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
    session_summary = _timeline_session_summary(timeline_file, rank_context, timestamps)

    events = _timeline_events(
        timestamps, allocated, reserved, sample_count, session_summary
    )

    sessions = _load_sessions_from_events(
        events,
        source_path=timeline_file,
        warnings=[],
        summary_by_id={session_summary.session_id: session_summary},
    )
    warnings.append(
        "Synthesized telemetry events from telemetry_timeline.json; "
        f"assigned rank={rank_context.rank} world_size={rank_context.world_size} "
        f"(source={rank_context.source}, session_id={session_summary.session_id}, "
        f"status={session_summary.status})."
    )
    return sessions, warnings


def _resolve_timeline_rank_context(
    *,
    timeline_file: Path,
    payload: Mapping[str, Any],
    rank_allocator: _TimelineRankAllocator,
) -> _TimelineRankContext:
    hinted_rank, hinted_local_rank, hinted_world_size = _timeline_identity_hints(
        payload
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


def _dedupe_events(events: Iterable[TelemetryEvent]) -> list[TelemetryEvent]:
    unique: list[TelemetryEvent] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (
            event.session_id,
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


def _parse_rank_token(chunk: str) -> Iterable[int]:
    if "-" in chunk:
        parts = chunk.split("-", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid rank range: {chunk!r}")
        start = int(parts[0])
        end = int(parts[1])
        if start > end:
            raise ValueError(f"Invalid rank range (start>end): {chunk!r}")
        return range(start, end + 1)

    return [int(chunk)]


def _group_distributed_sessions(
    sessions: Sequence[LoadedTelemetrySession],
    candidate_session_ids: set[str],
) -> dict[tuple[str, int], list[LoadedTelemetrySession]]:
    grouped: dict[tuple[str, int], list[LoadedTelemetrySession]] = {}
    for loaded in sessions:
        summary = loaded.summary
        if summary.session_id not in candidate_session_ids:
            continue
        if summary.job_id is None or summary.world_size <= 1:
            continue
        if not loaded.events:
            continue
        grouped.setdefault((summary.job_id, summary.world_size), []).append(loaded)

    return grouped


def _rank_memory_deltas(
    rank_samples: Sequence[TelemetryCompatibleEvent],
) -> tuple[int | None, int | None, int | None]:
    if not rank_samples:
        return None, None, None
    first_event, last_event = rank_samples[0], rank_samples[-1]
    return (
        _nullable_memory_delta(
            first_event.allocator_allocated_bytes, last_event.allocator_allocated_bytes
        ),
        _nullable_memory_delta(
            first_event.allocator_reserved_bytes, last_event.allocator_reserved_bytes
        ),
        _nullable_memory_delta(
            first_event.device_used_bytes, last_event.device_used_bytes
        ),
    )


def _nullable_memory_delta(first: int | None, last: int | None) -> int | None:
    return last - first if first is not None and last is not None else None


def _append_alert_candidates(
    rank: int,
    rank_events: Sequence[TelemetryCompatibleEvent],
    candidates: list[_AnomalyCandidate],
    phase_resolver: PhaseReplayIndex | None,
) -> None:
    for event in rank_events:
        if event.event_type in _ALERT_TYPES:
            phase_path = (
                summarize_phase_attribution(phase_resolver.resolve_for_event(event))
                if phase_resolver is not None
                and hasattr(phase_resolver, "resolve_for_event")
                else None
            )
            severity = _ALERT_SEVERITY.get(event.event_type, "warning")
            candidates.append(
                _AnomalyCandidate(
                    rank=rank,
                    severity=severity,
                    timestamp_ns=event.timestamp_ns,
                    signal=f"alert:{event.event_type}",
                    details=event.context or "Alert event",
                    phase_path=phase_path,
                )
            )


def _append_gap_ratio_candidates(
    rank: int,
    rank_samples: Sequence[TelemetryCompatibleEvent],
    candidates: list[_AnomalyCandidate],
    phase_resolver: PhaseReplayIndex | None,
) -> int | None:
    first_gap_breach_ts: int | None = None
    for event in rank_samples:
        if (
            event.device_total_bytes
            and event.device_total_bytes > 0
            and event.device_used_bytes is not None
            and event.allocator_reserved_bytes is not None
        ):
            gap_value = event.device_used_bytes - event.allocator_reserved_bytes
            gap_ratio = abs(gap_value) / event.device_total_bytes
            if gap_ratio >= GAP_RATIO_THRESHOLD:
                if first_gap_breach_ts is None:
                    first_gap_breach_ts = event.timestamp_ns
                phase_path = (
                    summarize_phase_attribution(phase_resolver.resolve_for_event(event))
                    if phase_resolver is not None
                    and hasattr(phase_resolver, "resolve_for_event")
                    else None
                )
                candidates.append(
                    _AnomalyCandidate(
                        rank=rank,
                        severity="warning",
                        timestamp_ns=event.timestamp_ns,
                        signal="gap_ratio_breach",
                        details=f"gap ratio {gap_ratio:.1%} exceeded threshold",
                        phase_path=phase_path,
                    )
                )

    return first_gap_breach_ts


def _timeline_identity_hints(
    payload: Mapping[str, Any]
) -> tuple[int | None, int | None, int | None]:
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

    return hinted_rank, hinted_local_rank, hinted_world_size


def _timeline_session_summary(
    timeline_file: Path,
    rank_context: _TimelineRankContext,
    timestamps: list[Any],
) -> SessionSummary:
    session_summary = _load_diagnose_session_summary(timeline_file, rank_context)
    if session_summary is None:
        session_summary = create_session_summary(
            source="stormlog.diagnose.timeline",
            status=SESSION_STATUS_INCOMPLETE,
            session_id=stable_legacy_session_id(
                str(timeline_file.resolve()),
                "diagnose.timeline",
            ),
            started_at_ns=int(float(timestamps[0]) * 1_000_000_000),
            host="unknown",
            pid=-1,
            rank=rank_context.rank,
            local_rank=rank_context.local_rank,
            world_size=rank_context.world_size,
        )

    return session_summary


def _timeline_events(
    timestamps: list[Any],
    allocated: list[Any],
    reserved: list[Any],
    sample_count: int,
    session_summary: SessionSummary,
) -> list[TelemetryEvent]:
    events: list[TelemetryEvent] = []
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
            "session_id": session_summary.session_id,
            "timestamp": timestamp,
            "event_type": "sample",
            "collector": "stormlog.diagnose.timeline",
            "sampling_interval_ms": max(0, interval_ms),
            "pid": session_summary.pid,
            "host": session_summary.host,
            "device_id": 0,
            "memory_allocated": allocated_bytes,
            "memory_reserved": reserved_bytes,
            "memory_change": allocated_bytes - previous_allocated,
            "device_used_bytes": max(allocated_bytes, reserved_bytes),
            "device_total_bytes": None,
            "context": "diagnose timeline sample",
            "job_id": session_summary.job_id,
            "rank": session_summary.rank,
            "local_rank": session_summary.local_rank,
            "world_size": session_summary.world_size,
            "metadata": {"source": "diagnose.telemetry_timeline"},
        }
        event = telemetry_event_from_record(
            record,
            permissive_legacy=True,
            default_collector="stormlog.diagnose.timeline",
            default_sampling_interval_ms=max(0, interval_ms),
            default_session_id=session_summary.session_id,
        )
        events.append(event)
        previous_allocated = allocated_bytes
        previous_timestamp = timestamp

    return events


__all__ = [
    "AnomalyIndicator",
    "ArtifactLoadResult",
    "DistributedDiagnosticsModel",
    "RankDiagnosticsRow",
    "build_distributed_model",
    "load_distributed_artifacts",
    "parse_rank_filter",
]
