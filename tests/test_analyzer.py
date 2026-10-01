"""Regression tests for PyTorch memory analysis."""

from __future__ import annotations

import pytest

from stormlog.analyzer import MemoryAnalyzer, MemoryPattern, PerformanceInsight
from stormlog.profiler import MemorySnapshot, ProfileResult


@pytest.mark.parametrize("events", [None, []])
def test_optimization_report_preserves_empty_telemetry_sections(
    events: list | None,
) -> None:
    result = _make_result(0.5, 3.0, 100)

    report = MemoryAnalyzer().generate_optimization_report([result], events=events)

    assert report["summary"] == {
        "total_functions_analyzed": 1,
        "total_function_calls": 1,
        "total_memory_allocated": 100,
        "total_execution_time": 0.5,
        "analysis_timestamp": 3.5,
    }
    assert list(report)[:7] == [
        "summary",
        "critical_issues",
        "high_impact_insights",
        "all_patterns",
        "all_insights",
        "recommendations",
        "optimization_score",
    ]
    assert ("gap_analysis" in report) == (events is not None)
    assert ("collective_attribution" in report) == (events is not None)
    assert "cross_rank_analysis" not in report


def test_empty_optimization_report_has_no_analysis_timestamp() -> None:
    report = MemoryAnalyzer().generate_optimization_report([])

    assert report["summary"] == {
        "total_functions_analyzed": 0,
        "total_function_calls": 0,
        "total_memory_allocated": 0,
        "total_execution_time": 0,
        "analysis_timestamp": None,
    }
    assert report["optimization_score"]["score"] == 100


def _make_snapshot(timestamp: float, allocated_memory: int) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=timestamp,
        allocated_memory=allocated_memory,
        reserved_memory=allocated_memory,
        max_memory_allocated=allocated_memory,
        max_memory_reserved=allocated_memory,
        active_memory=allocated_memory,
        inactive_memory=0,
        cpu_memory=0,
    )


def _make_result(
    execution_time: float,
    timestamp: float,
    memory_change: int = 0,
) -> ProfileResult:
    before_memory = max(0, -memory_change)
    after_memory = before_memory + memory_change
    before = _make_snapshot(timestamp, before_memory)
    after = _make_snapshot(timestamp + execution_time, after_memory)
    return ProfileResult(
        function_name="fast_function",
        execution_time=execution_time,
        memory_before=before,
        memory_after=after,
        memory_peak=after,
        memory_allocated=max(0, memory_change),
        memory_freed=max(0, -memory_change),
        tensors_created=0,
        tensors_deleted=0,
    )


@pytest.mark.parametrize(
    "method_name",
    ["generate_performance_insights", "generate_optimization_report"],
)
def test_zero_execution_times_do_not_crash_performance_analysis(
    method_name: str,
) -> None:
    analyzer = MemoryAnalyzer()
    results = [_make_result(0.0, float(index)) for index in range(5)]

    method = getattr(analyzer, method_name)

    assert method(results) is not None


@pytest.mark.parametrize(
    "severities,impacts,score,grade",
    [
        ([], [], 100, "A"),
        (["warning"], [], 90, "A"),
        (["critical"], [], 80, "B"),
        (["critical", "warning"], [], 70, "C"),
        (["critical", "critical"], [], 60, "D"),
        (["critical"] * 6, [], 0, "F"),
        (["other"], ["high", "medium", "other"], 69, "D"),
    ],
)
def test_optimization_score_preserves_penalties_and_grade_boundaries(
    severities: list[str], impacts: list[str], score: int, grade: str
) -> None:
    patterns = [
        MemoryPattern("test", "", severity, [], {}, []) for severity in severities
    ]
    insights = [
        PerformanceInsight("test", "", "", impact, 1.0, {}, []) for impact in impacts
    ]
    result = MemoryAnalyzer()._calculate_optimization_score(patterns, insights)
    assert result["score"] == score
    assert result["grade"] == grade
    assert result["issues_found"] == len(patterns) + len(insights)


def test_execution_insights_keep_slow_detection_before_variance() -> None:
    results = []
    for name, times in [("fast", [1.0] * 5), ("slow", [1.0] * 4 + [16.0])]:
        for index, duration in enumerate(times):
            result = _make_result(duration, float(index))
            result.function_name = name
            results.append(result)
    insights = MemoryAnalyzer()._analyze_execution_times(results)
    assert [item.title for item in insights] == [
        "Slow Function Detection",
        "Inconsistent Execution Times",
    ]
    assert insights[0].data["slow_functions"] == ["slow"]
    assert insights[0].data["slowest_time"] == 4.0
    assert insights[1].data["variable_functions"] == ["slow"]


def test_memory_leak_detection_ignores_mostly_freeing_function() -> None:
    mebibyte = 1024**2
    changes = [1024 * mebibyte, 1024 * mebibyte] + [-10 * mebibyte] * 11
    results = [
        _make_result(0.1, float(index), memory_change)
        for index, memory_change in enumerate(changes)
    ]

    patterns = MemoryAnalyzer().analyze_memory_patterns(results)

    assert all(pattern.pattern_type != "memory_leak" for pattern in patterns)


def test_memory_leak_detection_keeps_consistent_growth_signal() -> None:
    results = [_make_result(0.1, float(index), 40 * 1024**2) for index in range(3)]

    patterns = MemoryAnalyzer().analyze_memory_patterns(results)

    assert any(pattern.pattern_type == "memory_leak" for pattern in patterns)
