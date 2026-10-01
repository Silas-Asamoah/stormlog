"""Tests for hidden-memory gap analysis in MemoryAnalyzer."""

from __future__ import annotations

from typing import Any, cast

from stormlog.analyzer import MemoryAnalyzer
from stormlog.cli import _build_analyze_summary
from stormlog.phases import PhaseReplayIndex
from stormlog.telemetry import (
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V4,
    TelemetryEvent,
    TelemetryEventV2,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from tests.gap_test_helpers import build_gap_event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    index: int,
    allocator_allocated: int,
    allocator_reserved: int,
    device_used: int,
    device_total: int = 16 * 1024**3,
) -> TelemetryEventV2:
    """Build a minimal valid TelemetryEventV2 for gap analysis tests."""
    return build_gap_event(
        index=index,
        allocator_allocated=allocator_allocated,
        allocator_reserved=allocator_reserved,
        device_used=device_used,
        collector="stormlog.cuda_tracker",
        device_total=device_total,
    )


def _make_device_only_event(index: int) -> TelemetryEvent:
    used = 1_000_000_000 + index * 100_000_000
    return TelemetryEvent(
        schema_version=SCHEMA_VERSION_V4,
        session_id="device-only",
        timestamp_ns=1_700_000_000_000_000_000 + index * 100_000_000,
        event_type="sample",
        collector="stormlog.future_tracker",
        sampling_interval_ms=100,
        pid=1,
        host="test",
        device_id=0,
        allocator_allocated_bytes=None,
        allocator_reserved_bytes=None,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=None,
        device_used_bytes=used,
        device_free_bytes=4_000_000_000 - used,
        device_total_bytes=4_000_000_000,
        context=None,
        metadata={
            "memory_capabilities": {
                "backend": "future",
                "telemetry_collector": "stormlog.future_tracker",
                "sampling_source": "future.device_memory",
                "supports_allocator_allocated": False,
                "supports_allocator_reserved": False,
                "supports_allocator_active": False,
                "supports_allocator_inactive": False,
                "supports_device_used": True,
                "supports_device_free": True,
                "supports_device_total": True,
                "supports_native_allocator_history": False,
                "supports_fragmentation_analysis": False,
                "supports_allocator_attribution": False,
                "supports_bounded_profiling": False,
            }
        },
    )


def _make_collective_event(
    *,
    timestamp_ns: int,
    rank: int,
    world_size: int,
    allocator_reserved: int,
    device_used: int,
    event_type: str = "sample",
    context: str | None = None,
    metadata: dict | None = None,
) -> TelemetryEventV2:
    total_bytes = 16 * 1024**3
    return TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
        timestamp_ns=timestamp_ns,
        event_type=event_type,
        collector="stormlog.cuda_tracker",
        sampling_interval_ms=100,
        pid=1,
        host="test",
        device_id=0,
        allocator_allocated_bytes=max(0, allocator_reserved - 100_000_000),
        allocator_reserved_bytes=allocator_reserved,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=0,
        device_used_bytes=device_used,
        device_free_bytes=max(0, total_bytes - device_used),
        device_total_bytes=total_bytes,
        context=context,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        metadata=metadata or {},
    )


def _make_phase_sample_event(
    *,
    session_id: str,
    timestamp_ns: int,
    allocator_allocated: int,
    allocator_reserved: int,
    device_used: int,
    rank: int = 0,
    world_size: int = 1,
    event_type: str = "sample",
    context: str | None = None,
    metadata: dict | None = None,
) -> object:
    total_bytes = 16 * 1024**3
    return telemetry_event_from_record(
        {
            "schema_version": 3,
            "session_id": session_id,
            "timestamp_ns": timestamp_ns,
            "event_type": event_type,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1,
            "host": "test",
            "device_id": 0,
            "allocator_allocated_bytes": allocator_allocated,
            "allocator_reserved_bytes": allocator_reserved,
            "allocator_active_bytes": None,
            "allocator_inactive_bytes": None,
            "allocator_change_bytes": 0,
            "device_used_bytes": device_used,
            "device_free_bytes": max(0, total_bytes - device_used),
            "device_total_bytes": total_bytes,
            "context": context,
            "job_id": "job-phase",
            "rank": rank,
            "local_rank": rank,
            "world_size": world_size,
            "metadata": metadata or {},
        }
    )


def _make_phase_boundary_event(
    *,
    session_id: str,
    timestamp_ns: int,
    rank: int,
    world_size: int,
    event_type: str,
    action: str,
    sequence: int,
    scope_id: str,
) -> object:
    return _make_phase_sample_event(
        session_id=session_id,
        timestamp_ns=timestamp_ns,
        allocator_allocated=2_000_000_000,
        allocator_reserved=2_500_000_000,
        device_used=2_500_000_000,
        rank=rank,
        world_size=world_size,
        event_type=event_type,
        context=f"Phase {action}ed: train",
        metadata={
            "phase_scope": {
                "action": action,
                "name": "train",
                "path": ["train"],
                "depth": 1,
                "scope_id": scope_id,
                "parent_scope_id": None,
                "thread_id": 1,
                "thread_name": "MainThread",
                "sequence": sequence,
            }
        },
    )


# ---------------------------------------------------------------------------
# Scenario 1: allocator-only growth -- gap stays small, no findings expected
# ---------------------------------------------------------------------------


class TestAllocatorOnlyGrowth:
    def test_no_gap_findings(self) -> None:
        """When device_used tracks allocator_reserved closely, no gap is flagged."""
        events = []
        for i in range(20):
            alloc = 1_000_000_000 + i * 50_000_000  # 1 GB growing slowly
            reserved = alloc + 10_000_000  # tiny overhead
            device_used = reserved + 5_000_000  # negligible gap
            events.append(_make_event(i, alloc, reserved, device_used))

        analyzer = MemoryAnalyzer()
        findings = analyzer.analyze_memory_gaps(events)
        assert findings == []


# ---------------------------------------------------------------------------
# Scenario 2: non-allocator growth -- device_used drifts up, allocator flat
# ---------------------------------------------------------------------------


class TestPersistentDrift:
    def test_persistent_drift_detected(self) -> None:
        """Steady growth in device_used while allocator stays flat -> persistent_drift."""
        events = []
        alloc = 2_000_000_000
        reserved = 2_500_000_000
        for i in range(30):
            # device_used grows linearly well beyond allocator_reserved
            device_used = reserved + 500_000_000 + i * 100_000_000
            events.append(_make_event(i, alloc, reserved, device_used))

        analyzer = MemoryAnalyzer()
        findings = analyzer.analyze_memory_gaps(events)

        drift_findings = [f for f in findings if f.classification == "persistent_drift"]
        assert len(drift_findings) == 1

        finding = drift_findings[0]
        assert finding.confidence > 0
        assert finding.severity in ("warning", "critical")
        assert finding.evidence["r_squared"] >= 0.6
        assert finding.evidence["slope_bytes_per_sec"] > 0
        assert len(finding.remediation) > 0
        assert finding.description  # non-empty


# ---------------------------------------------------------------------------
# Scenario 3: transient spikes in device_used
# ---------------------------------------------------------------------------


class TestTransientSpike:
    def test_transient_spike_detected(self) -> None:
        """Occasional large spikes in device_used -> transient_spike."""
        events = []
        alloc = 2_000_000_000
        reserved = 2_500_000_000
        baseline_device = reserved + 100_000_000  # small steady gap

        for i in range(30):
            device_used = baseline_device
            # Inject large spikes at indices 10 and 20
            if i in (10, 20):
                device_used = reserved + 4_000_000_000  # huge spike
            events.append(_make_event(i, alloc, reserved, device_used))

        analyzer = MemoryAnalyzer()
        findings = analyzer.analyze_memory_gaps(events)

        spike_findings = [f for f in findings if f.classification == "transient_spike"]
        assert len(spike_findings) == 1

        finding = spike_findings[0]
        assert finding.confidence > 0
        assert finding.evidence["spike_count"] >= 2
        assert finding.evidence["max_zscore"] > 0
        assert len(finding.remediation) > 0
        assert finding.description


# ---------------------------------------------------------------------------
# Scenario 4: fragmentation-like -- high reserved-allocated ratio
# ---------------------------------------------------------------------------


class TestFragmentationLike:
    def test_disabled_fragmentation_is_not_reported_or_recommended(self) -> None:
        events = [
            telemetry_event_from_record(
                telemetry_event_to_dict(
                    _make_event(i, 100_000_000, 1_000_000_000, 2_500_000_000)
                )
            )
            for i in range(12)
        ]
        for event in events:
            event.metadata["memory_capabilities"][
                "supports_fragmentation_analysis"
            ] = False

        report = MemoryAnalyzer().generate_optimization_report(events=events)

        assert report["gap_analysis"] == []
        availability = report["analysis_availability"]["fragmentation_analysis"]
        assert availability["available"] is False
        assert "capabilities disable fragmentation analysis" in availability["reason"]
        summary = _build_analyze_summary("events.json", 0, report)
        assert "Capability warning:" in summary
        assert availability["reason"] in summary
        assert "empty_cache" not in str(report)
        assert "PYTORCH_CUDA_ALLOC_CONF" not in str(report)

    def test_fragmentation_like_detected(self) -> None:
        """High (reserved - allocated) / reserved ratio -> fragmentation_like."""
        events = []
        for i in range(20):
            alloc = 1_000_000_000
            # reserved grows much larger than allocated -> fragmentation
            reserved = 3_000_000_000 + i * 50_000_000
            # gap must exceed gap_ratio_threshold (5% of 16 GiB ≈ 858 MB)
            device_used = reserved + 1_500_000_000
            events.append(_make_event(i, alloc, reserved, device_used))

        analyzer = MemoryAnalyzer()
        findings = analyzer.analyze_memory_gaps(events)

        frag_findings = [
            f for f in findings if f.classification == "fragmentation_like"
        ]
        assert len(frag_findings) == 1

        finding = frag_findings[0]
        assert finding.confidence > 0
        assert finding.evidence["avg_fragmentation_ratio"] >= 0.3
        assert len(finding.remediation) > 0
        assert finding.description


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_events(self) -> None:
        analyzer = MemoryAnalyzer()
        assert analyzer.analyze_memory_gaps([]) == []

    def test_fewer_than_three_events(self) -> None:
        events = [_make_event(0, 1000, 2000, 3000), _make_event(1, 1000, 2000, 3000)]
        analyzer = MemoryAnalyzer()
        assert analyzer.analyze_memory_gaps(events) == []

    def test_device_only_report_explains_unavailable_allocator_analysis(self) -> None:
        events = [_make_device_only_event(index) for index in range(4)]

        report = MemoryAnalyzer().generate_optimization_report(events=events)

        assert report["gap_analysis"] == []
        assert report["collective_attribution"] == []
        availability = report["analysis_availability"]
        assert availability["gap_analysis"]["available"] is False
        assert (
            "does not expose allocator counters"
            in availability["gap_analysis"]["reason"]
        )

    def test_report_includes_gap_analysis_when_events_provided(self) -> None:
        """generate_optimization_report includes gap_analysis key when events are given."""
        events = []
        alloc = 2_000_000_000
        reserved = 2_500_000_000
        for i in range(30):
            device_used = reserved + 500_000_000 + i * 100_000_000
            events.append(_make_event(i, alloc, reserved, device_used))

        analyzer = MemoryAnalyzer()
        report = analyzer.generate_optimization_report(results=[], events=events)
        assert "gap_analysis" in report
        assert "collective_attribution" in report
        assert isinstance(report["gap_analysis"], list)
        assert isinstance(report["collective_attribution"], list)
        assert len(report["gap_analysis"]) > 0

    def test_report_omits_gap_analysis_when_no_events(self) -> None:
        """generate_optimization_report omits gap_analysis when events is None."""
        analyzer = MemoryAnalyzer()
        report = analyzer.generate_optimization_report(results=[])
        assert "gap_analysis" not in report
        assert "collective_attribution" not in report

    def test_report_collective_attribution_includes_confidence_and_reason_codes(
        self,
    ) -> None:
        events: list[TelemetryEventV2] = []
        base_ns = 1_700_000_000_000_000_000
        for rank in (0, 1):
            rank_offset = rank * 2_000_000
            for index in range(12):
                timestamp_ns = base_ns + index * 100_000_000 + rank_offset
                reserved = 2_000_000_000
                used = reserved + 40_000_000
                if index == 6:
                    used = reserved + 1_600_000_000
                events.append(
                    _make_collective_event(
                        timestamp_ns=timestamp_ns,
                        rank=rank,
                        world_size=2,
                        allocator_reserved=reserved,
                        device_used=used,
                    )
                )
                if index == 6:
                    events.append(
                        _make_collective_event(
                            timestamp_ns=timestamp_ns + 10_000_000,
                            rank=rank,
                            world_size=2,
                            allocator_reserved=reserved,
                            device_used=reserved,
                            event_type="collective",
                            context="NCCL all_reduce phase",
                            metadata={"phase": "communication.collective"},
                        )
                    )

        analyzer = MemoryAnalyzer()
        report = analyzer.generate_optimization_report(results=[], events=events)
        attributions = report["collective_attribution"]

        assert attributions
        for attribution in attributions:
            assert "confidence" in attribution
            assert "reason_codes" in attribution
            assert attribution["reason_codes"]

    def test_gap_findings_include_phase_attribution_when_available(self) -> None:
        session_id = "session-gap-phase"
        base_ts = 1_700_000_000_000_000_000
        events: list[Any] = [
            _make_phase_boundary_event(
                session_id=session_id,
                timestamp_ns=base_ts,
                rank=0,
                world_size=1,
                event_type="phase_enter",
                action="enter",
                sequence=1,
                scope_id="phase-1",
            )
        ]
        alloc = 2_000_000_000
        reserved = 2_500_000_000
        for index in range(12):
            events.append(
                _make_phase_sample_event(
                    session_id=session_id,
                    timestamp_ns=base_ts + (index + 1) * 100_000_000,
                    allocator_allocated=alloc,
                    allocator_reserved=reserved,
                    device_used=reserved + 500_000_000 + index * 120_000_000,
                )
            )
        events.append(
            _make_phase_boundary_event(
                session_id=session_id,
                timestamp_ns=base_ts + 13 * 100_000_000,
                rank=0,
                world_size=1,
                event_type="phase_exit",
                action="exit",
                sequence=2,
                scope_id="phase-1",
            )
        )

        analyzer = MemoryAnalyzer()
        resolver = PhaseReplayIndex.from_events(events)
        findings = analyzer.analyze_memory_gaps(
            cast(list[TelemetryEventV2], events),
            phase_resolver=resolver,
        )

        drift_findings = [f for f in findings if f.classification == "persistent_drift"]
        assert len(drift_findings) == 1
        finding = drift_findings[0]
        assert finding.evidence_timestamp_ns is not None
        assert finding.phase_attribution is not None
        assert finding.phase_attribution.phase_path == "train"
