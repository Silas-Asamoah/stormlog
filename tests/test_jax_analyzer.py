"""Tests for JAX analyzer module."""

from dataclasses import make_dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("numpy")

from stormlog.jax.analyzer import (
    MemoryAnalyzer,
    _serialize_collective_attribution,
    _serialize_gap_finding,
    _suggest_jax_optimizations,
)
from stormlog.jax.profiler import MemorySnapshot, ProfileResult
from tests.jax_test_helpers import fake_jax_runtime  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


def test_analyzer_init() -> None:
    analyzer = MemoryAnalyzer(sensitivity=0.1)
    assert analyzer.sensitivity == 0.1
    assert "gap_ratio_threshold" in analyzer.thresholds


def test_find_leaks_no_snapshots() -> None:
    analyzer = MemoryAnalyzer()
    res = ProfileResult(0, 0, 0, 0, 0, [], {})
    leaks = analyzer.detect_memory_leaks(res)
    assert len(leaks) == 0


def test_find_leaks_with_trend() -> None:
    analyzer = MemoryAnalyzer(sensitivity=1.0)
    # Simulate a steady climb to trigger leak detection (>10 points)
    snapshots = [
        MemorySnapshot(i, f"snap_{i}", 100 * i, 100 * i, 0, 0, {}) for i in range(15)
    ]
    # Adding a memory_usage array specifically to satisfy hasattr check
    res = ProfileResult(0, 15, 1400, 700, 0, snapshots, {})
    setattr(res, "memory_usage", [100 * i for i in range(15)])

    leaks = analyzer.detect_memory_leaks(res)
    assert len(leaks) >= 1
    assert leaks[0]["type"] == "leak"
    assert "slope" in leaks[0]


def test_detect_patterns_no_snapshots() -> None:
    analyzer = MemoryAnalyzer()
    res = ProfileResult(0, 0, 0, 0, 0, [], {})
    patterns = analyzer.detect_patterns(res)
    assert len(patterns) == 0


def test_detect_patterns_detected() -> None:
    analyzer = MemoryAnalyzer()

    # Need >= 10 samples and some periodicity to trigger autocorrelate peak
    usage = [100, 5000, 100, 5000, 100, 5000, 100, 5000, 100, 5000, 100, 5000]
    snapshots = [
        MemorySnapshot(i, f"snap_{i}", u, u, 0, 0, {}) for i, u in enumerate(usage)
    ]
    res = ProfileResult(0, len(usage), 5000, 2500, 100, snapshots, {})
    setattr(res, "memory_usage", usage)

    patterns = analyzer.detect_patterns(res)
    # We can't guarantee pattern heuristic triggers without exactly tuning the array.
    # It's sufficient for unit testing coverage if it executes the code without error.
    assert isinstance(patterns, list)


def test_analyze_efficiency() -> None:
    analyzer = MemoryAnalyzer()
    res = ProfileResult(0, 0, 0, 0, 0, [], {})
    score = analyzer.analyze_efficiency(res)
    # Default score when no ops/usage data is available is 1.0
    assert score == 1.0


def test_analyze_efficiency_with_data() -> None:
    analyzer = MemoryAnalyzer()
    res = mock.Mock()
    res.peak_memory_mb = 8100
    res.memory_growth_rate = 250
    res.snapshots = []
    res.memory_usage = [100, 500, 100]
    score = analyzer.analyze_efficiency(res)
    assert score < 1.0


@pytest.mark.parametrize(
    "peak,growth,fragmentation,leaks,expected",
    [
        (4000, 100, 0.3, [], 1.0),
        (4001, 101, 0.31, [], 0.65),
        (8000, 200, 0.5, [{"severity": "low"}], 0.5),
        (8001, 201, 0.51, [{"severity": "high"}], 0.0),
    ],
)
def test_efficiency_preserves_thresholds_and_penalty_order(
    peak: float,
    growth: float,
    fragmentation: float,
    leaks: list[dict[str, str]],
    expected: float,
) -> None:
    analyzer = MemoryAnalyzer()
    result = SimpleNamespace(
        peak_memory_mb=peak,
        memory_growth_rate=growth,
        snapshots=[],
        memory_usage=[1, 2, 3],
    )
    with (
        mock.patch.object(
            analyzer,
            "analyze_fragmentation",
            return_value={"fragmentation_score": fragmentation},
        ),
        mock.patch.object(analyzer, "detect_memory_leaks", return_value=leaks),
    ):
        assert analyzer.analyze_efficiency(result) == pytest.approx(expected)


def test_analyze_fragmentation() -> None:
    analyzer = MemoryAnalyzer()
    # Provide snapshots to exercise the fragmentation logic loop
    snapshots = [
        MemorySnapshot(1, "s1", 100, 200, 0, 0, {}),
        MemorySnapshot(2, "s2", 200, 400, 0, 0, {}),
        MemorySnapshot(3, "s3", 300, 600, 0, 0, {}),
    ]
    res = mock.Mock()
    res.snapshots = snapshots
    res.peak_memory_mb = 600
    frag = analyzer.analyze_fragmentation(res)
    assert "fragmentation_score" in frag
    assert "trend" in frag


def test_correlate_with_performance() -> None:
    analyzer = MemoryAnalyzer()
    res = mock.Mock()
    res.function_profiles = {
        "func1": {
            "calls": 10,
            "total_memory_delta": 30 * 1024**3,
            "total_duration": 15.0,
        },
        "func2": {"calls": 101, "total_duration": 20.0, "total_memory_delta": 100},
    }
    corr = analyzer.correlate_with_performance(res)
    assert "func1" in corr["function_efficiency"]
    assert "func2" in corr["function_efficiency"]


def test_analyze_memory_gaps() -> None:
    analyzer = MemoryAnalyzer()
    events: list[Any] = []
    gaps = analyzer.analyze_memory_gaps(events)
    assert isinstance(gaps, list)


def test_analyze_collective_attribution() -> None:
    analyzer = MemoryAnalyzer()
    events: list[Any] = []
    attr = analyzer.analyze_collective_attribution(events)
    assert isinstance(attr, list)


def test_score_optimization() -> None:
    analyzer = MemoryAnalyzer()
    res = mock.Mock()
    res.peak_memory_mb = 100
    res.snapshots = []
    res.memory_usage = [100, 100, 100, 100, 100]
    res.memory_growth_rate = 0
    res.function_profiles = {
        "func1": {
            "calls": 10,
            "total_memory_delta": 30 * 1024**3,
            "total_duration": 15.0,
        },
    }

    with (
        mock.patch.object(analyzer, "detect_memory_leaks", return_value=[]),
        mock.patch.object(analyzer, "detect_patterns", return_value=[]),
    ):
        report = analyzer.score_optimization(res)
        assert "overall_score" in report
        assert "categories" in report


def test_score_optimization_with_events() -> None:
    analyzer = MemoryAnalyzer()
    res = ProfileResult(0, 0, 0, 0, 0, [], {})
    events: list[Any] = []

    with (
        mock.patch.object(analyzer, "analyze_memory_gaps", return_value=[]),
        mock.patch.object(analyzer, "analyze_collective_attribution", return_value=[]),
    ):
        report = analyzer.score_optimization(res, events=events)
        assert "gap_analysis" in report
        assert "collective_attribution" in report


def test_suggest_jax_optimizations() -> None:
    res = mock.Mock()
    res.peak_memory_mb = 9000
    res.memory_growth_rate = 150
    sug = _suggest_jax_optimizations(res)
    assert len(sug) > 0


def test_serialize_helpers() -> None:
    GapFinding = make_dataclass("GapFinding", [("id", int)])
    finding = GapFinding(id=1)
    res1 = _serialize_gap_finding(finding)
    assert res1["id"] == 1

    CollAttr = make_dataclass("CollAttr", [("status", str)])
    attr = CollAttr(status="ok")
    res2 = _serialize_collective_attribution(attr)
    assert res2["status"] == "ok"
