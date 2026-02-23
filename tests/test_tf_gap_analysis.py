"""Tests for hidden-memory gap analysis in tfmemprof MemoryAnalyzer."""

from __future__ import annotations

from gpumemprof.telemetry import TelemetryEventV2
from tests.gap_test_helpers import build_gap_event
from tfmemprof.analyzer import MemoryAnalyzer

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
        collector="tfmemprof.memory_tracker",
        device_total=device_total,
    )


# ---------------------------------------------------------------------------
# Scenario 1: allocator-only growth -- gap stays small, no findings expected
# ---------------------------------------------------------------------------


class TestAllocatorOnlyGrowth:
    def test_no_gap_findings(self) -> None:
        """When device_used tracks allocator_reserved closely, no gap is flagged."""
        events = []
        for i in range(20):
            alloc = 1_000_000_000 + i * 50_000_000
            reserved = alloc + 10_000_000
            device_used = reserved + 5_000_000
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
        assert finding.description


# ---------------------------------------------------------------------------
# Scenario 3: transient spikes in device_used
# ---------------------------------------------------------------------------


class TestTransientSpike:
    def test_transient_spike_detected(self) -> None:
        """Occasional large spikes in device_used -> transient_spike."""
        events = []
        alloc = 2_000_000_000
        reserved = 2_500_000_000
        baseline_device = reserved + 100_000_000

        for i in range(30):
            device_used = baseline_device
            if i in (10, 20):
                device_used = reserved + 4_000_000_000
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
            reserved = 3_000_000_000 + i * 50_000_000
            # gap must exceed gap_ratio_threshold (5% of 16 GiB ~ 858 MB)
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

    def test_score_optimization_includes_gap_analysis_when_events_provided(
        self,
    ) -> None:
        """score_optimization includes gap_analysis key when events are given."""
        events = []
        alloc = 2_000_000_000
        reserved = 2_500_000_000
        for i in range(30):
            device_used = reserved + 500_000_000 + i * 100_000_000
            events.append(_make_event(i, alloc, reserved, device_used))

        class _Stub:
            peak_memory_mb = 100.0
            average_memory_mb = 80.0

        analyzer = MemoryAnalyzer()
        result = analyzer.score_optimization(_Stub(), events=events)
        assert "gap_analysis" in result
        assert isinstance(result["gap_analysis"], list)
        assert len(result["gap_analysis"]) > 0

    def test_score_optimization_omits_gap_analysis_when_no_events(self) -> None:
        """score_optimization omits gap_analysis when events is None."""

        class _Stub:
            peak_memory_mb = 100.0
            average_memory_mb = 80.0

        analyzer = MemoryAnalyzer()
        result = analyzer.score_optimization(_Stub())
        assert "gap_analysis" not in result
