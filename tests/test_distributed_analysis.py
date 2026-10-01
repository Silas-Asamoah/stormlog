"""Tests for distributed timeline merge and first-cause spike detection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from stormlog.analyzer import MemoryAnalyzer
from stormlog.distributed_analysis import (
    analyze_cross_rank_events,
    merge_cross_rank_timelines,
)

PhaseReplayIndex: Any
try:
    from stormlog.phases import PhaseReplayIndex as _PhaseReplayIndex
except ImportError:  # pragma: no cover - phase package may land in another slice
    PhaseReplayIndex = None
else:
    PhaseReplayIndex = _PhaseReplayIndex
from stormlog.telemetry import telemetry_event_from_record, telemetry_event_to_dict
from tests.gap_test_helpers import BASE_NS, INTERVAL_NS, build_gap_event

_GB = 1024**3


def _build_rank_series(
    *,
    rank: int,
    device_used_values: list[int],
    world_size: int,
    timestamp_offset_ns: int = 0,
    job_id: str = "job-28",
) -> list:
    events = []
    for index, device_used in enumerate(device_used_values):
        allocator_reserved = max(device_used - 128 * 1024**2, 0)
        allocator_allocated = max(allocator_reserved - 64 * 1024**2, 0)
        events.append(
            build_gap_event(
                index=index,
                allocator_allocated=allocator_allocated,
                allocator_reserved=allocator_reserved,
                device_used=device_used,
                collector="stormlog.cuda_tracker",
                rank=rank,
                local_rank=rank % 2,
                world_size=world_size,
                job_id=job_id,
                host=f"host-{rank // 2}",
                timestamp_ns=BASE_NS + timestamp_offset_ns + index * INTERVAL_NS,
            )
        )
    return events


def _build_cross_rank_fixture(world_size: int = 4) -> list:
    events = []
    events.extend(
        _build_rank_series(
            rank=0,
            world_size=world_size,
            timestamp_offset_ns=0,
            device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, int(1.34 * _GB)],
        )
    )
    events.extend(
        _build_rank_series(
            rank=1,
            world_size=world_size,
            timestamp_offset_ns=20_000_000,
            device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, int(1.30 * _GB)],
        )
    )
    events.extend(
        _build_rank_series(
            rank=2,
            world_size=world_size,
            timestamp_offset_ns=40_000_000,
            device_used_values=[1 * _GB, 1 * _GB, int(1.38 * _GB), int(1.40 * _GB)],
        )
    )
    events.extend(
        _build_rank_series(
            rank=3,
            world_size=world_size,
            timestamp_offset_ns=10_000_000,
            device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, int(1.28 * _GB)],
        )
    )
    return events


def _canonicalize_phase_events(events: list[Any], *, session_id: str) -> list[Any]:
    canonical_events: list[Any] = []
    observed_ranks = sorted({event.rank for event in events})
    for offset, rank in enumerate(observed_ranks, start=1):
        canonical_events.append(
            telemetry_event_from_record(
                {
                    "schema_version": 3,
                    "session_id": session_id,
                    "timestamp_ns": BASE_NS - 1_000_000 + rank * 100_000,
                    "event_type": "phase_enter",
                    "collector": "stormlog.cuda_tracker",
                    "sampling_interval_ms": 100,
                    "pid": 1,
                    "host": f"host-{rank // 2}",
                    "job_id": "job-28",
                    "rank": rank,
                    "local_rank": rank % 2,
                    "world_size": len(observed_ranks),
                    "device_id": 0,
                    "allocator_allocated_bytes": 1 * _GB,
                    "allocator_reserved_bytes": 1 * _GB,
                    "allocator_active_bytes": None,
                    "allocator_inactive_bytes": None,
                    "allocator_change_bytes": 0,
                    "device_used_bytes": 1 * _GB,
                    "device_free_bytes": 15 * _GB,
                    "device_total_bytes": 16 * _GB,
                    "context": "Phase entered: train / forward",
                    "metadata": {
                        "phase_scope": {
                            "action": "enter",
                            "name": "forward",
                            "path": ["train", "forward"],
                            "depth": 2,
                            "scope_id": f"phase-{rank}",
                            "parent_scope_id": "phase-train",
                            "thread_id": 1,
                            "thread_name": "MainThread",
                            "sequence": offset,
                        }
                    },
                }
            )
        )
    for event in events:
        record = telemetry_event_to_dict(event)
        record["schema_version"] = 3
        record["session_id"] = session_id
        canonical_events.append(telemetry_event_from_record(record))
    for offset, rank in enumerate(observed_ranks, start=len(observed_ranks) + 1):
        canonical_events.append(
            telemetry_event_from_record(
                {
                    "schema_version": 3,
                    "session_id": session_id,
                    "timestamp_ns": BASE_NS + 5 * INTERVAL_NS + rank * 100_000,
                    "event_type": "phase_exit",
                    "collector": "stormlog.cuda_tracker",
                    "sampling_interval_ms": 100,
                    "pid": 1,
                    "host": f"host-{rank // 2}",
                    "job_id": "job-28",
                    "rank": rank,
                    "local_rank": rank % 2,
                    "world_size": len(observed_ranks),
                    "device_id": 0,
                    "allocator_allocated_bytes": 1 * _GB,
                    "allocator_reserved_bytes": 1 * _GB,
                    "allocator_active_bytes": None,
                    "allocator_inactive_bytes": None,
                    "allocator_change_bytes": 0,
                    "device_used_bytes": 1 * _GB,
                    "device_free_bytes": 15 * _GB,
                    "device_total_bytes": 16 * _GB,
                    "context": "Phase exited: train / forward",
                    "metadata": {
                        "phase_scope": {
                            "action": "exit",
                            "name": "forward",
                            "path": ["train", "forward"],
                            "depth": 2,
                            "scope_id": f"phase-{rank}",
                            "parent_scope_id": "phase-train",
                            "thread_id": 1,
                            "thread_name": "MainThread",
                            "sequence": offset,
                        }
                    },
                }
            )
        )
    return canonical_events


class TestCrossRankMerge:
    @pytest.mark.parametrize(
        ("rank_one_values", "world_size", "confidence", "onset_offset"),
        [
            ([1 * _GB, int(1.4 * _GB), int(1.4 * _GB)], 2, "medium", 1),
            ([1 * _GB, 1 * _GB, int(1.4 * _GB)], 2, "high", 2),
            ([1 * _GB, 1 * _GB, int(1.4 * _GB)], 3, "low", 2),
            ([1 * _GB, 1 * _GB, 1 * _GB], 2, "low", None),
            ([1 * _GB], 2, "low", None),
        ],
    )
    def test_first_cause_confidence_tracks_support_and_sampling(
        self,
        rank_one_values: list[int],
        world_size: int,
        confidence: str,
        onset_offset: int | None,
    ) -> None:
        events = _build_rank_series(
            rank=0,
            world_size=world_size,
            device_used_values=[1 * _GB, int(1.4 * _GB), int(1.4 * _GB)],
        ) + _build_rank_series(
            rank=1,
            world_size=world_size,
            device_used_values=rank_one_values,
        )

        _, analysis = analyze_cross_rank_events(events)

        first = analysis.suspects[0]
        assert first.rank == 0
        assert first.confidence == confidence
        assert analysis.cluster_onset_timestamp_ns == (
            BASE_NS + onset_offset * INTERVAL_NS if onset_offset is not None else None
        )
        assert first.evidence["supporting_ranks_at_or_before_onset"] == len(
            analysis.suspects
        )
        assert first.phase_attribution is None
        if onset_offset is None:
            assert first.lead_over_cluster_onset_ns == 0
            assert analysis.notes[-1] == (
                "Only one rank produced a qualifying spike; confidence is limited."
            )
        if len(rank_one_values) == 1:
            assert analysis.notes[0] == (
                "Some ranks have fewer than two samples and cannot contribute to spike detection: 1."
            )

    def test_merge_aligns_ranks_by_first_sample_offset(self) -> None:
        events = []
        events.extend(
            _build_rank_series(
                rank=0,
                world_size=2,
                device_used_values=[1 * _GB, int(1.05 * _GB)],
            )
        )
        events.extend(
            _build_rank_series(
                rank=1,
                world_size=2,
                timestamp_offset_ns=2 * INTERVAL_NS,
                device_used_values=[1 * _GB, int(1.05 * _GB)],
            )
        )

        merged = merge_cross_rank_timelines(events)

        assert merged.alignment_offsets_ns == {0: 0, 1: 2 * INTERVAL_NS}
        first_points: dict[int, int] = {}
        for point in merged.merged_points:
            first_points.setdefault(point.rank, point.aligned_timestamp_ns)
        assert first_points[0] == first_points[1] == BASE_NS

    def test_detects_injected_first_cause_rank(self) -> None:
        merge_result, first_cause = analyze_cross_rank_events(
            _build_cross_rank_fixture()
        )

        assert merge_result.participating_ranks == [0, 1, 2, 3]
        assert first_cause.cluster_onset_timestamp_ns is not None
        assert first_cause.suspects
        assert first_cause.suspects[0].rank == 2
        assert first_cause.suspects[0].confidence == "high"
        assert first_cause.suspects[0].lead_over_cluster_onset_ns == INTERVAL_NS

    def test_missing_rank_is_reported_but_analysis_still_runs(self) -> None:
        events = []
        events.extend(
            _build_rank_series(
                rank=0,
                world_size=4,
                device_used_values=[1 * _GB, 1 * _GB, int(1.40 * _GB), int(1.42 * _GB)],
            )
        )
        events.extend(
            _build_rank_series(
                rank=1,
                world_size=4,
                timestamp_offset_ns=15_000_000,
                device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, int(1.32 * _GB)],
            )
        )
        events.extend(
            _build_rank_series(
                rank=3,
                world_size=4,
                timestamp_offset_ns=25_000_000,
                device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, int(1.30 * _GB)],
            )
        )

        merge_result, first_cause = analyze_cross_rank_events(events)

        assert merge_result.missing_ranks == [2]
        assert first_cause.suspects
        assert first_cause.suspects[0].rank == 0

    def test_single_rank_input_returns_no_suspects(self) -> None:
        events = _build_rank_series(
            rank=0,
            world_size=1,
            device_used_values=[1 * _GB, 1 * _GB, int(1.40 * _GB)],
        )

        _, first_cause = analyze_cross_rank_events(events)

        assert first_cause.suspects == []
        assert "At least two ranks are required" in first_cause.notes[0]

    def test_no_spikes_returns_empty_suspects(self) -> None:
        events = []
        for rank, skew in ((0, 0), (1, 10_000_000)):
            events.extend(
                _build_rank_series(
                    rank=rank,
                    world_size=2,
                    timestamp_offset_ns=skew,
                    device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, 1 * _GB],
                )
            )

        _, first_cause = analyze_cross_rank_events(events)

        assert first_cause.suspects == []
        assert "No qualifying cross-rank spikes were detected." in first_cause.notes

    def test_tie_break_prefers_larger_delta_then_lower_rank(self) -> None:
        first_tie = []
        first_tie.extend(
            _build_rank_series(
                rank=1,
                world_size=3,
                device_used_values=[1 * _GB, 1 * _GB, int(1.42 * _GB), int(1.42 * _GB)],
            )
        )
        first_tie.extend(
            _build_rank_series(
                rank=2,
                world_size=3,
                timestamp_offset_ns=10_000_000,
                device_used_values=[1 * _GB, 1 * _GB, int(1.30 * _GB), int(1.30 * _GB)],
            )
        )
        first_tie.extend(
            _build_rank_series(
                rank=0,
                world_size=3,
                timestamp_offset_ns=20_000_000,
                device_used_values=[1 * _GB, 1 * _GB, 1 * _GB, int(1.30 * _GB)],
            )
        )

        _, first_cause = analyze_cross_rank_events(first_tie)
        assert first_cause.suspects[0].rank == 1

        second_tie = []
        second_tie.extend(
            _build_rank_series(
                rank=1,
                world_size=2,
                device_used_values=[1 * _GB, 1 * _GB, int(1.35 * _GB), int(1.35 * _GB)],
            )
        )
        second_tie.extend(
            _build_rank_series(
                rank=0,
                world_size=2,
                timestamp_offset_ns=10_000_000,
                device_used_values=[1 * _GB, 1 * _GB, int(1.35 * _GB), int(1.35 * _GB)],
            )
        )

        _, first_cause_same_delta = analyze_cross_rank_events(second_tie)
        assert first_cause_same_delta.suspects[0].rank == 0

    def test_ignores_non_sample_events_when_detecting_spikes(self) -> None:
        events = []
        rank_zero = _build_rank_series(
            rank=0,
            world_size=2,
            device_used_values=[1 * _GB, int(1.40 * _GB), int(1.40 * _GB)],
        )
        rank_zero[1] = replace(rank_zero[1], event_type="warning")
        events.extend(rank_zero)
        events.extend(
            _build_rank_series(
                rank=1,
                world_size=2,
                device_used_values=[1 * _GB, int(1.35 * _GB), int(1.35 * _GB)],
            )
        )

        merge_result, first_cause = analyze_cross_rank_events(events)

        assert merge_result.rank_sample_counts == {0: 2, 1: 3}
        assert (
            "Ignored non-sample events during cross-rank analysis."
            in merge_result.notes
        )
        assert first_cause.suspects[0].rank == 1

    def test_filters_to_most_common_job_id_for_cross_rank_analysis(self) -> None:
        events = []
        events.extend(
            _build_rank_series(
                rank=0,
                world_size=2,
                device_used_values=[1 * _GB, 1 * _GB, 1 * _GB],
                job_id="job-a",
            )
        )
        events.extend(
            _build_rank_series(
                rank=1,
                world_size=2,
                device_used_values=[1 * _GB, 1 * _GB, 1 * _GB],
                job_id="job-a",
            )
        )
        events.extend(
            _build_rank_series(
                rank=0,
                world_size=2,
                device_used_values=[1 * _GB, int(1.40 * _GB)],
                job_id="job-b",
            )
        )
        events.extend(
            _build_rank_series(
                rank=1,
                world_size=2,
                device_used_values=[1 * _GB, int(1.35 * _GB)],
                job_id="job-b",
            )
        )

        merge_result, first_cause = analyze_cross_rank_events(events)

        assert merge_result.job_id == "job-a"
        assert (
            "Multiple job_id values were observed; filtering to the most common value."
            in merge_result.notes
        )
        assert first_cause.suspects == []

    def test_ranks_spikes_by_threshold_crossing_timestamp(self) -> None:
        events = []
        events.extend(
            _build_rank_series(
                rank=0,
                world_size=2,
                device_used_values=[
                    1 * _GB,
                    int(1.05 * _GB),
                    int(1.10 * _GB),
                    int(1.15 * _GB),
                ],
            )
        )
        events.extend(
            _build_rank_series(
                rank=1,
                world_size=2,
                device_used_values=[
                    1 * _GB,
                    int(1.13 * _GB),
                    int(1.13 * _GB),
                    int(1.13 * _GB),
                ],
            )
        )

        _, first_cause = analyze_cross_rank_events(events)

        assert first_cause.suspects[0].rank == 1

    def test_first_cause_suspects_include_phase_attribution_when_available(
        self,
    ) -> None:
        if PhaseReplayIndex is None:
            pytest.skip("stormlog.phases is not available in this slice")

        events = _canonicalize_phase_events(
            _build_cross_rank_fixture(),
            session_id="session-cross-rank-phase",
        )

        _, first_cause = analyze_cross_rank_events(
            cast(Any, events),
            phase_resolver=PhaseReplayIndex.from_events(events),
        )

        assert first_cause.suspects
        assert first_cause.suspects[0].phase_attribution is not None
        assert first_cause.suspects[0].phase_attribution.phase_path == (
            "train / forward"
        )


class TestAnalyzerIntegration:
    def test_generate_optimization_report_includes_cross_rank_analysis(self) -> None:
        analyzer = MemoryAnalyzer()
        report = analyzer.generate_optimization_report(
            results=[],
            events=_build_cross_rank_fixture(),
        )

        assert "cross_rank_analysis" in report
        assert report["cross_rank_analysis"]["first_cause_suspects"][0]["rank"] == 2

    def test_generate_optimization_report_omits_cross_rank_analysis_for_single_rank(
        self,
    ) -> None:
        analyzer = MemoryAnalyzer()
        report = analyzer.generate_optimization_report(
            results=[],
            events=_build_rank_series(
                rank=0,
                world_size=1,
                device_used_values=[1 * _GB, 1 * _GB, int(1.35 * _GB)],
            ),
        )

        assert "cross_rank_analysis" not in report
