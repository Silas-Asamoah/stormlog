"""Distributed telemetry analysis helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Sequence, cast

try:
    from .phases import PhaseAttribution, PhaseReplayIndex, phase_attribution_to_payload
except ImportError:  # pragma: no cover - phase package may land in another slice
    PhaseAttribution = Any  # type: ignore[assignment,misc]
    PhaseReplayIndex = Any  # type: ignore[assignment,misc]

    def phase_attribution_to_payload(
        attribution: PhaseAttribution | None,
    ) -> dict[str, Any] | None:
        return None


from .telemetry import TelemetryEventLike

_SPIKE_MIN_BYTES = 64 * 1024**2
_SKEW_NOTE_MULTIPLIER = 5


@dataclass
class RankTimelinePoint:
    """A single telemetry sample in a rank-aligned timeline."""

    rank: int
    timestamp_ns: int
    aligned_timestamp_ns: int
    device_used_bytes: int
    allocator_reserved_bytes: int | None
    allocator_allocated_bytes: int | None
    allocator_change_bytes: int | None


@dataclass
class CrossRankMergeResult:
    """Merged distributed timeline state."""

    job_id: str | None
    world_size: int
    participating_ranks: list[int]
    missing_ranks: list[int]
    rank_sample_counts: dict[int, int]
    alignment_offsets_ns: dict[int, int]
    merged_points: list[RankTimelinePoint]
    notes: list[str] = field(default_factory=list)


@dataclass
class FirstCauseSuspect:
    """A ranked first-cause candidate."""

    rank: int
    first_spike_timestamp_ns: int
    aligned_first_spike_timestamp_ns: int
    peak_delta_bytes: int
    spike_window_samples: int
    lead_over_cluster_onset_ns: int
    confidence: str
    evidence: dict[str, int | str]
    phase_attribution: PhaseAttribution | None = None


@dataclass
class FirstCauseAnalysisResult:
    """The distributed first-cause detection result."""

    cluster_onset_timestamp_ns: int | None
    suspects: list[FirstCauseSuspect]
    notes: list[str] = field(default_factory=list)


@dataclass
class _RankSpikeCandidate:
    rank: int
    session_id: str | None
    first_spike_timestamp_ns: int
    aligned_first_spike_timestamp_ns: int
    peak_delta_bytes: int
    spike_window_samples: int


def _is_sample_event(event: TelemetryEventLike) -> bool:
    return event.event_type.casefold() == "sample"


def _group_events_by_rank(
    events: Sequence[TelemetryEventLike],
) -> dict[int, list[TelemetryEventLike]]:
    grouped: dict[int, list[TelemetryEventLike]] = defaultdict(list)
    for event in events:
        grouped[event.rank].append(event)
    for rank_events in grouped.values():
        rank_events.sort(key=lambda event: (event.timestamp_ns, event.pid, event.host))
    return dict(grouped)


def _select_job_id(events: Sequence[TelemetryEventLike]) -> str | None:
    job_ids = [event.job_id for event in events if event.job_id]
    if not job_ids:
        return None
    counts = Counter(job_ids)
    return counts.most_common(1)[0][0]


def _select_cross_rank_analysis_events(
    events: Sequence[TelemetryEventLike],
) -> tuple[list[TelemetryEventLike], str | None, list[str]]:
    notes: list[str] = []
    raw_sample_events = [event for event in events if _is_sample_event(event)]
    if len(raw_sample_events) != len(events):
        notes.append("Ignored non-sample events during cross-rank analysis.")
    sample_events = [
        event for event in raw_sample_events if event.device_used_bytes is not None
    ]
    if len(sample_events) != len(raw_sample_events):
        notes.append(
            "Ignored samples without device-used memory during cross-rank analysis."
        )
    if not sample_events:
        notes.append(
            "No sample telemetry events were available for distributed analysis."
        )
        return [], None, notes

    sample_events, job_id = _select_job_samples(sample_events, notes)

    return sample_events, job_id, notes


def _determine_world_size(
    events: Sequence[TelemetryEventLike], participating_ranks: Sequence[int]
) -> int:
    if not events:
        return 0
    claimed_sizes = [event.world_size for event in events if event.world_size > 1]
    declared = Counter(claimed_sizes).most_common(1)[0][0] if claimed_sizes else 1
    observed = (max(participating_ranks) + 1) if participating_ranks else declared
    if len(participating_ranks) > 1:
        return max(declared, observed)
    return max(declared, observed, 1)


def _median_sampling_interval_ns(
    grouped: dict[int, list[TelemetryEventLike]],
) -> int:
    intervals: list[int] = []
    for rank_events in grouped.values():
        for index in range(1, len(rank_events)):
            delta = (
                rank_events[index].timestamp_ns - rank_events[index - 1].timestamp_ns
            )
            if delta > 0:
                intervals.append(delta)
    if intervals:
        return int(median(intervals))

    fallback_intervals = [
        event.sampling_interval_ms * 1_000_000
        for rank_events in grouped.values()
        for event in rank_events
        if event.sampling_interval_ms > 0
    ]
    if fallback_intervals:
        return int(median(fallback_intervals))
    return 0


def merge_cross_rank_timelines(
    events: Sequence[TelemetryEventLike],
) -> CrossRankMergeResult:
    """Merge rank streams into a single aligned timeline."""

    if not events:
        return CrossRankMergeResult(
            job_id=None,
            world_size=0,
            participating_ranks=[],
            missing_ranks=[],
            rank_sample_counts={},
            alignment_offsets_ns={},
            merged_points=[],
            notes=["No telemetry events were provided."],
        )

    analysis_events, job_id, notes = _select_cross_rank_analysis_events(events)
    if not analysis_events:
        return CrossRankMergeResult(
            job_id=job_id,
            world_size=0,
            participating_ranks=[],
            missing_ranks=[],
            rank_sample_counts={},
            alignment_offsets_ns={},
            merged_points=[],
            notes=notes,
        )

    grouped = _group_events_by_rank(analysis_events)
    participating_ranks = sorted(grouped)
    world_size = _determine_world_size(analysis_events, participating_ranks)
    expected_ranks = set(range(world_size)) if world_size > 0 else set()
    missing_ranks = sorted(expected_ranks.difference(participating_ranks))
    rank_sample_counts = {
        rank: len(rank_events) for rank, rank_events in grouped.items()
    }

    if missing_ranks:
        notes.append(
            "Missing rank data for ranks: "
            + ", ".join(str(rank) for rank in missing_ranks)
            + "."
        )

    alignment_offsets_ns, merged_points = _align_rank_streams(
        grouped, participating_ranks, notes
    )

    return CrossRankMergeResult(
        job_id=job_id,
        world_size=world_size,
        participating_ranks=participating_ranks,
        missing_ranks=missing_ranks,
        rank_sample_counts=rank_sample_counts,
        alignment_offsets_ns=alignment_offsets_ns,
        merged_points=merged_points,
        notes=notes,
    )


def _find_rank_spike_candidate(
    rank_events: Sequence[TelemetryEventLike], offset_ns: int
) -> _RankSpikeCandidate | None:
    if len(rank_events) < 2:
        return None

    device_events = [
        event for event in rank_events if event.device_used_bytes is not None
    ]
    if len(device_events) < 2:
        return None

    spike_threshold = max(
        _SPIKE_MIN_BYTES,
        int(
            max(
                event.device_used_bytes
                for event in device_events
                if event.device_used_bytes is not None
            )
            * 0.10
        ),
    )
    window_start_index: int | None = None
    cumulative_delta = 0
    spike_window_samples = 0

    for index in range(1, len(device_events)):
        current_used = device_events[index].device_used_bytes
        previous_used = device_events[index - 1].device_used_bytes
        if current_used is None or previous_used is None:
            continue
        delta = current_used - previous_used
        if delta <= 0:
            window_start_index = None
            cumulative_delta = 0
            spike_window_samples = 0
            continue

        if window_start_index is None:
            window_start_index = index
            cumulative_delta = 0
            spike_window_samples = 0

        cumulative_delta += delta
        spike_window_samples += 1
        if cumulative_delta < spike_threshold:
            continue

        spike_event = device_events[index]
        return _RankSpikeCandidate(
            rank=spike_event.rank,
            session_id=getattr(spike_event, "session_id", None),
            first_spike_timestamp_ns=spike_event.timestamp_ns,
            aligned_first_spike_timestamp_ns=spike_event.timestamp_ns - offset_ns,
            peak_delta_bytes=cumulative_delta,
            spike_window_samples=spike_window_samples,
        )

    return None


def _detect_first_cause_spikes(
    grouped: dict[int, list[TelemetryEventLike]],
    merge_result: CrossRankMergeResult,
    phase_resolver: PhaseReplayIndex | None = None,
) -> FirstCauseAnalysisResult:
    if not grouped:
        return FirstCauseAnalysisResult(
            cluster_onset_timestamp_ns=None,
            suspects=[],
            notes=["No telemetry events were available for distributed analysis."],
        )

    if len(grouped) == 1:
        return FirstCauseAnalysisResult(
            cluster_onset_timestamp_ns=None,
            suspects=[],
            notes=[
                "At least two ranks are required for cross-rank first-cause analysis."
            ],
        )

    candidates, insufficient_sample_ranks = _collect_rank_spike_candidates(
        grouped, merge_result.alignment_offsets_ns
    )
    notes: list[str] = []
    if insufficient_sample_ranks:
        notes.append(
            "Some ranks have fewer than two samples and cannot contribute to spike detection: "
            + ", ".join(str(rank) for rank in sorted(insufficient_sample_ranks))
            + "."
        )

    if not candidates:
        notes.append("No qualifying cross-rank spikes were detected.")
        return FirstCauseAnalysisResult(
            cluster_onset_timestamp_ns=None,
            suspects=[],
            notes=notes,
        )

    ranked_suspects, cluster_onset_timestamp_ns = _select_first_cause_candidates(
        candidates
    )
    if cluster_onset_timestamp_ns is None:
        notes.append(
            "Only one rank produced a qualifying spike; confidence is limited."
        )

    suspects = _build_first_cause_suspects(
        ranked_suspects,
        cluster_onset_timestamp_ns=cluster_onset_timestamp_ns,
        candidate_count=len(candidates),
        median_interval_ns=_median_sampling_interval_ns(grouped),
        sparse_evidence=bool(merge_result.missing_ranks or insufficient_sample_ranks),
        phase_resolver=phase_resolver,
    )
    return FirstCauseAnalysisResult(
        cluster_onset_timestamp_ns=cluster_onset_timestamp_ns,
        suspects=suspects,
        notes=notes,
    )


def _collect_rank_spike_candidates(
    grouped: dict[int, list[TelemetryEventLike]],
    alignment_offsets_ns: dict[int, int],
) -> tuple[list[_RankSpikeCandidate], list[int]]:
    insufficient_sample_ranks = [
        rank for rank, rank_events in grouped.items() if len(rank_events) < 2
    ]
    candidates: list[_RankSpikeCandidate] = []
    for rank, rank_events in grouped.items():
        candidate = _find_rank_spike_candidate(
            rank_events, alignment_offsets_ns.get(rank, 0)
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates, insufficient_sample_ranks


def _select_first_cause_candidates(
    candidates: list[_RankSpikeCandidate],
) -> tuple[list[_RankSpikeCandidate], int | None]:
    """Order candidates and include every rank at or before the second spike."""
    candidates.sort(
        key=lambda candidate: (
            candidate.aligned_first_spike_timestamp_ns,
            -candidate.peak_delta_bytes,
            candidate.rank,
        )
    )

    cluster_onset_timestamp_ns: int | None = None
    if len(candidates) >= 2:
        cluster_onset_timestamp_ns = candidates[1].aligned_first_spike_timestamp_ns
    suspect_cutoff = (
        cluster_onset_timestamp_ns
        if cluster_onset_timestamp_ns is not None
        else candidates[0].aligned_first_spike_timestamp_ns
    )
    ranked_suspects = [
        candidate
        for candidate in candidates
        if candidate.aligned_first_spike_timestamp_ns <= suspect_cutoff
    ]

    return ranked_suspects, cluster_onset_timestamp_ns


def _first_cause_confidence(
    candidate: _RankSpikeCandidate,
    *,
    candidate_count: int,
    earliest_aligned_timestamp: int,
    earliest_count: int,
    sparse_evidence: bool,
    median_interval_ns: int,
    lead_over_cluster_onset_ns: int,
) -> str:
    if (
        candidate_count < 2
        or sparse_evidence
        or candidate.aligned_first_spike_timestamp_ns != earliest_aligned_timestamp
    ):
        return "low"
    if earliest_count == 1 and (
        median_interval_ns <= 0 or lead_over_cluster_onset_ns >= median_interval_ns
    ):
        return "high"
    return "medium"


def _resolve_candidate_phase(
    candidate: _RankSpikeCandidate, phase_resolver: PhaseReplayIndex | None
) -> PhaseAttribution | None:
    if (
        phase_resolver is not None
        and candidate.session_id is not None
        and hasattr(phase_resolver, "resolve")
    ):
        return cast(
            PhaseAttribution | None,
            phase_resolver.resolve(
                timestamp_ns=candidate.first_spike_timestamp_ns,
                session_id=candidate.session_id,
                rank=candidate.rank,
            ),
        )
    return None


def _build_first_cause_suspects(
    ranked_suspects: list[_RankSpikeCandidate],
    *,
    cluster_onset_timestamp_ns: int | None,
    candidate_count: int,
    median_interval_ns: int,
    sparse_evidence: bool,
    phase_resolver: PhaseReplayIndex | None,
) -> list[FirstCauseSuspect]:
    earliest_aligned_timestamp = ranked_suspects[0].aligned_first_spike_timestamp_ns
    earliest_count = sum(
        candidate.aligned_first_spike_timestamp_ns == earliest_aligned_timestamp
        for candidate in ranked_suspects
    )
    support_count = len(ranked_suspects)

    suspects: list[FirstCauseSuspect] = []
    for candidate in ranked_suspects:
        lead_over_cluster_onset_ns = (
            0
            if cluster_onset_timestamp_ns is None
            else cluster_onset_timestamp_ns - candidate.aligned_first_spike_timestamp_ns
        )

        confidence = _first_cause_confidence(
            candidate,
            candidate_count=candidate_count,
            earliest_aligned_timestamp=earliest_aligned_timestamp,
            earliest_count=earliest_count,
            sparse_evidence=sparse_evidence,
            median_interval_ns=median_interval_ns,
            lead_over_cluster_onset_ns=lead_over_cluster_onset_ns,
        )

        suspects.append(
            FirstCauseSuspect(
                rank=candidate.rank,
                first_spike_timestamp_ns=candidate.first_spike_timestamp_ns,
                aligned_first_spike_timestamp_ns=candidate.aligned_first_spike_timestamp_ns,
                peak_delta_bytes=candidate.peak_delta_bytes,
                spike_window_samples=candidate.spike_window_samples,
                lead_over_cluster_onset_ns=lead_over_cluster_onset_ns,
                confidence=confidence,
                evidence={
                    "device_used_delta_bytes": candidate.peak_delta_bytes,
                    "supporting_ranks_at_or_before_onset": support_count,
                },
                phase_attribution=_resolve_candidate_phase(candidate, phase_resolver),
            )
        )

    return suspects


def analyze_cross_rank_events(
    events: Sequence[TelemetryEventLike],
    *,
    phase_resolver: PhaseReplayIndex | None = None,
) -> tuple[CrossRankMergeResult, FirstCauseAnalysisResult]:
    """Analyze distributed telemetry for merged timelines and first-cause spikes."""

    merge_result = merge_cross_rank_timelines(events)
    analysis_events, _, selection_notes = _select_cross_rank_analysis_events(events)
    if not analysis_events:
        return merge_result, FirstCauseAnalysisResult(
            cluster_onset_timestamp_ns=None,
            suspects=[],
            notes=selection_notes or list(merge_result.notes),
        )
    grouped = _group_events_by_rank(analysis_events)
    first_cause_result = _detect_first_cause_spikes(
        grouped,
        merge_result,
        phase_resolver=phase_resolver,
    )
    return merge_result, first_cause_result


def summarize_cross_rank_analysis(
    events: Sequence[TelemetryEventLike],
    *,
    phase_resolver: PhaseReplayIndex | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable cross-rank analysis summary."""

    merge_result, first_cause_result = analyze_cross_rank_events(
        events,
        phase_resolver=phase_resolver,
    )
    notes = list(dict.fromkeys([*merge_result.notes, *first_cause_result.notes]))
    return {
        "job_id": merge_result.job_id,
        "world_size": merge_result.world_size,
        "participating_ranks": merge_result.participating_ranks,
        "missing_ranks": merge_result.missing_ranks,
        "rank_sample_counts": {
            str(rank): count for rank, count in merge_result.rank_sample_counts.items()
        },
        "alignment_offsets_ns": {
            str(rank): offset
            for rank, offset in merge_result.alignment_offsets_ns.items()
        },
        "cluster_onset_timestamp_ns": first_cause_result.cluster_onset_timestamp_ns,
        "first_cause_suspects": [
            _serialize_first_cause_suspect(suspect)
            for suspect in first_cause_result.suspects
        ],
        "notes": notes,
    }


def _serialize_first_cause_suspect(suspect: FirstCauseSuspect) -> dict[str, Any]:
    payload = asdict(suspect)
    payload["phase_attribution"] = phase_attribution_to_payload(
        suspect.phase_attribution
    )
    return payload


def _select_job_samples(
    sample_events: list[TelemetryEventLike], notes: list[str]
) -> tuple[list[TelemetryEventLike], str | None]:
    job_id = _select_job_id(sample_events)
    observed_job_ids = {event.job_id for event in sample_events if event.job_id}
    if len(observed_job_ids) > 1 and job_id is not None:
        notes.append(
            "Multiple job_id values were observed; filtering to the most common value."
        )
        sample_events = [event for event in sample_events if event.job_id == job_id]

    return sample_events, job_id


def _align_rank_streams(
    grouped: dict[int, list[TelemetryEventLike]],
    participating_ranks: list[int],
    notes: list[str],
) -> tuple[dict[int, int], list[RankTimelinePoint]]:
    anchor_rank = 0 if 0 in grouped else participating_ranks[0]
    anchor_timestamp = grouped[anchor_rank][0].timestamp_ns
    alignment_offsets_ns: dict[int, int] = {}
    merged_points: list[RankTimelinePoint] = []
    median_interval_ns = _median_sampling_interval_ns(grouped)

    for rank in participating_ranks:
        offset_ns = grouped[rank][0].timestamp_ns - anchor_timestamp
        alignment_offsets_ns[rank] = offset_ns
        if (
            median_interval_ns > 0
            and abs(offset_ns) > _SKEW_NOTE_MULTIPLIER * median_interval_ns
        ):
            notes.append(
                "Rank "
                f"{rank} starts {offset_ns} ns from the anchor; "
                "first-sample alignment may be approximate."
            )
        for event in grouped[rank]:
            device_used = event.device_used_bytes
            if device_used is None:
                continue
            merged_points.append(
                RankTimelinePoint(
                    rank=rank,
                    timestamp_ns=event.timestamp_ns,
                    aligned_timestamp_ns=event.timestamp_ns - offset_ns,
                    device_used_bytes=device_used,
                    allocator_reserved_bytes=event.allocator_reserved_bytes,
                    allocator_allocated_bytes=event.allocator_allocated_bytes,
                    allocator_change_bytes=event.allocator_change_bytes,
                )
            )

    merged_points.sort(
        key=lambda point: (point.aligned_timestamp_ns, point.rank, point.timestamp_ns)
    )

    return alignment_offsets_ns, merged_points


__all__ = [
    "CrossRankMergeResult",
    "FirstCauseAnalysisResult",
    "FirstCauseSuspect",
    "RankTimelinePoint",
    "analyze_cross_rank_events",
    "merge_cross_rank_timelines",
    "summarize_cross_rank_analysis",
]
