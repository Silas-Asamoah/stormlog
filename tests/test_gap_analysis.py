"""Tests for hidden-memory gap analysis in MemoryAnalyzer."""

from __future__ import annotations

from gpumemprof.analyzer import MemoryAnalyzer
from gpumemprof.telemetry import SCHEMA_VERSION_V2, TelemetryEventV2
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
        collector="gpumemprof.cuda_tracker",
        device_total=device_total,
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
        collector="gpumemprof.cuda_tracker",
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
