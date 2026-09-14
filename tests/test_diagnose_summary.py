"""Characterization tests for backend-specific diagnostic summaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import stormlog.diagnose as diagnose


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_gpu_summary_uses_allocator_counters_and_fragmentation_stats(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setattr(
        diagnose, "get_system_info", lambda: {"detected_backend": backend}
    )
    gpu_info = {
        "allocated_memory": 85,
        "reserved_memory": 90,
        "total_memory": 100,
        "max_memory_allocated": 0,
        "memory_stats": {"num_ooms": 2},
    }
    monkeypatch.setattr(diagnose, "get_gpu_info", lambda device: gpu_info)
    fragmentation = {"fragmentation_ratio": 0.3}
    collect_fragmentation = Mock(return_value=fragmentation)
    suggestions = Mock(return_value=["reuse buffers"])
    monkeypatch.setattr(diagnose, "check_memory_fragmentation", collect_fragmentation)
    monkeypatch.setattr(diagnose, "suggest_memory_optimization", suggestions)

    summary, risk = diagnose.build_diagnostic_summary(1)

    assert summary == {
        "backend": backend,
        "allocated_bytes": 85,
        "reserved_bytes": 90,
        "peak_bytes": 85,
        "total_bytes": 100,
        "allocator_gap_bytes": 5,
        "utilization_ratio": 0.85,
        "fragmentation_ratio": 0.3,
        "num_ooms": 2,
        "risk_flags": {
            "oom_occurred": True,
            "high_utilization": True,
            "fragmentation_warning": True,
        },
        "suggestions": ["reuse buffers"],
    }
    assert risk is True
    collect_fragmentation.assert_called_once_with(1)
    suggestions.assert_called_once_with(fragmentation)


@pytest.mark.parametrize(
    ("sample", "allocated", "reserved", "total", "risk"),
    [
        (None, 0, 0, 0, False),
        (
            SimpleNamespace(allocated_bytes=85, reserved_bytes=90, total_bytes=100),
            85,
            90,
            100,
            True,
        ),
        (
            SimpleNamespace(allocated_bytes=85, reserved_bytes=90, total_bytes=None),
            85,
            90,
            0,
            False,
        ),
    ],
)
def test_mps_summary_uses_collector_and_capacity_for_risk(
    monkeypatch: pytest.MonkeyPatch,
    sample: SimpleNamespace | None,
    allocated: int,
    reserved: int,
    total: int,
    risk: bool,
) -> None:
    monkeypatch.setattr(
        diagnose, "get_system_info", lambda: {"detected_backend": "mps"}
    )
    monkeypatch.setattr(diagnose, "get_gpu_info", lambda device: {"error": "no CUDA"})
    collector = Mock(return_value=("mps", sample))
    fragmentation = Mock(side_effect=AssertionError("MPS must not inspect CUDA stats"))
    monkeypatch.setattr(diagnose, "_collect_backend_sample", collector)
    monkeypatch.setattr(diagnose, "check_memory_fragmentation", fragmentation)

    summary, risk_detected = diagnose.build_diagnostic_summary(2)

    assert summary["allocated_bytes"] == allocated
    assert summary["reserved_bytes"] == reserved
    assert summary["total_bytes"] == total
    assert summary["peak_bytes"] == max(allocated, reserved)
    assert summary["num_ooms"] == 0
    assert summary["fragmentation_ratio"] == 0.0
    assert summary["risk_flags"]["high_utilization"] is risk
    assert bool(summary["suggestions"]) is risk
    assert risk_detected is risk
    collector.assert_called_once_with(2)


@pytest.mark.parametrize(
    ("system", "gpu_info", "num_ooms"),
    [
        ({}, {"memory_stats": {"num_ooms": 4}}, 0),
        (
            {"detected_backend": "cuda"},
            {"error": "unavailable", "memory_stats": {"num_ooms": 4}},
            4,
        ),
        (
            {"detected_backend": "cuda"},
            {"allocated_memory": None, "reserved_memory": None, "memory_stats": []},
            0,
        ),
    ],
)
def test_summary_preserves_missing_memory_and_backend_oom_fallbacks(
    monkeypatch: pytest.MonkeyPatch, system: dict, gpu_info: dict, num_ooms: int
) -> None:
    monkeypatch.setattr(diagnose, "get_system_info", lambda: system)
    monkeypatch.setattr(diagnose, "get_gpu_info", lambda device: gpu_info)
    monkeypatch.setattr(diagnose, "check_memory_fragmentation", lambda device: {})
    monkeypatch.setattr(diagnose, "suggest_memory_optimization", lambda info: [])

    summary, risk = diagnose.build_diagnostic_summary()

    assert summary["allocated_bytes"] == 0
    assert summary["reserved_bytes"] == 0
    assert summary["total_bytes"] == 0
    assert summary["utilization_ratio"] == 0.0
    assert summary["num_ooms"] == num_ooms
    assert risk is (num_ooms > 0)
