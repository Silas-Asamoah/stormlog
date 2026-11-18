"""Tests for selection-versus-baseline outlier comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from stormlog.outliers import (
    ELIGIBLE_FIELDS,
    MISSING_VALUE,
    ComparisonFilter,
    compare_events,
    describe_events,
    ks_statistic,
)
from stormlog.telemetry import TelemetryEvent, telemetry_event_from_record


def _event(
    *,
    timestamp_ns: int = 1_000,
    session_id: str = "session-a",
    rank: int = 0,
    reserved: int | None = 150,
    metadata: dict[str, Any] | None = None,
    host: str = "host-a",
    event_type: str = "sample",
) -> TelemetryEvent:
    return telemetry_event_from_record(
        {
            "schema_version": 4,
            "session_id": session_id,
            "timestamp_ns": timestamp_ns,
            "event_type": event_type,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 123,
            "host": host,
            "job_id": "job-a",
            "rank": rank,
            "local_rank": rank,
            "world_size": 2,
            "device_id": 0,
            "allocator_allocated_bytes": 100,
            "allocator_reserved_bytes": reserved,
            "allocator_active_bytes": None,
            "allocator_inactive_bytes": None,
            "allocator_change_bytes": None,
            "device_used_bytes": 200,
            "device_free_bytes": None,
            "device_total_bytes": 1000,
            "context": None,
            "metadata": {"backend": "cuda", **(metadata or {})},
        }
    )


def _events(count: int, **kwargs: Any) -> list[TelemetryEvent]:
    return [_event(timestamp_ns=1_000 + index, **kwargs) for index in range(count)]


def _field(result: Any, name: str) -> Any:
    return next(row for row in result.fields if row.field == name)


def test_ks_statistic_matches_scipy() -> None:
    rng = np.random.default_rng(114)
    first = rng.normal(0, 1, 200)
    second = rng.normal(0.5, 2, 150)
    expected = stats.ks_2samp(first, second).statistic
    assert ks_statistic(first, second) == pytest.approx(expected)


def test_ks_statistic_handles_ties_and_identical_samples() -> None:
    same = np.array([1.0, 1.0, 2.0, 3.0])
    assert ks_statistic(same, same.copy()) == 0.0
    assert ks_statistic(np.array([1.0]), np.array([2.0])) == 1.0


def test_identical_distributions_score_zero() -> None:
    sel = _events(40, metadata={"collector_health_status": "healthy"})
    sel += _events(40, metadata={"collector_health_status": "degraded"})
    result = compare_events(sel, list(sel), scope="window")
    assert all(row.score == 0.0 for row in result.fields)
    assert all(row.confidence == "high" for row in result.fields)


def test_shifted_counter_and_degraded_collector_rank_first() -> None:
    selection = _events(
        40, reserved=900, metadata={"collector_health_status": "degraded"}
    )
    baseline = _events(
        80, reserved=150, metadata={"collector_health_status": "healthy"}
    )
    result = compare_events(selection, baseline, scope="rank")
    top = {row.field for row in result.fields[:2]}
    assert top <= {
        "allocator_reserved_bytes",
        "collector_health_status",
        "allocator_gap_bytes",
        "allocator_cache_ratio",
        "hidden_gap_bytes",
        "allocator_change_bytes",
    }
    health = _field(result, "collector_health_status")
    assert health.kind == "categorical"
    assert health.score == pytest.approx(1.0)
    assert health.top_values[0].value in {"degraded", "healthy"}
    assert abs(health.top_values[0].delta) == pytest.approx(1.0)
    reserved = _field(result, "allocator_reserved_bytes")
    assert reserved.score == pytest.approx(1.0)
    assert reserved.selection_median == 900.0
    assert reserved.baseline_median == 150.0


def test_categorical_score_is_total_variation_distance() -> None:
    selection = _events(30, host="a") + _events(10, host="b")
    baseline = _events(10, host="a") + _events(30, host="b")
    host = _field(compare_events(selection, baseline, scope="window"), "host")
    assert host.score == pytest.approx(0.5)
    assert [share.value for share in host.top_values] == ["a", "b"]


def test_missing_values_drive_numeric_score() -> None:
    selection = _events(20, reserved=None) + _events(20, reserved=150)
    baseline = _events(40, reserved=150)
    reserved = _field(
        compare_events(selection, baseline, scope="window"),
        "allocator_reserved_bytes",
    )
    assert reserved.selection_missing_ratio == pytest.approx(0.5)
    assert reserved.baseline_missing_ratio == 0.0
    assert reserved.score == pytest.approx(0.5)
    assert "sparse_field" in reserved.reasons


def test_missing_categorical_value_is_its_own_category() -> None:
    selection = _events(40)
    baseline = _events(40, metadata={"collector_health_status": "healthy"})
    health = _field(
        compare_events(selection, baseline, scope="window"),
        "collector_health_status",
    )
    assert {share.value for share in health.top_values} == {MISSING_VALUE, "healthy"}


def test_high_cardinality_and_constant_fields_are_skipped() -> None:
    selection = [_event(host=f"host-{index}") for index in range(40)]
    baseline = [_event(host=f"host-{index + 100}") for index in range(40)]
    result = compare_events(selection, baseline, scope="window", max_values=10)
    skipped = {row.field: row for row in result.skipped}
    assert skipped["host"].reason == "high_cardinality"
    assert skipped["host"].distinct_values == 80
    assert skipped["event_type"].reason == "constant"
    assert skipped["allocator_active_bytes"].reason == "no_data"
    assert "host" not in {row.field for row in result.fields}


def test_small_baseline_is_low_confidence_with_warning() -> None:
    result = compare_events(
        _events(40, reserved=900), _events(3), scope="session", min_events=30
    )
    assert any("baseline has 3 events" in warning for warning in result.warnings)
    assert result.fields
    assert all(row.confidence == "low" for row in result.fields)
    assert all("small_baseline" in row.reasons for row in result.fields)


@pytest.mark.parametrize("empty_side", ["selection", "baseline"])
def test_empty_side_returns_no_fields(empty_side: str) -> None:
    events = _events(5)
    selection, baseline = ([], events) if empty_side == "selection" else (events, [])
    result = compare_events(selection, baseline, scope="window")
    assert result.fields == []
    assert result.skipped == []
    assert f"{empty_side} has no events; nothing to compare" in result.warnings


def test_rank_scope_excludes_rank_fields() -> None:
    result = compare_events(_events(5, rank=1), _events(5, rank=0), scope="rank")
    compared = {row.field for row in result.fields} | {
        row.field for row in result.skipped
    }
    assert "rank" not in compared
    assert "local_rank" not in compared


def test_field_restriction_and_limit() -> None:
    selection = _events(40, reserved=900, host="b")
    baseline = _events(40, reserved=150, host="a")
    result = compare_events(
        selection,
        baseline,
        scope="window",
        fields=("host", "allocator_reserved_bytes", "event_type"),
        limit=1,
    )
    assert [row.field for row in result.fields] == ["allocator_reserved_bytes"]
    assert [row.field for row in result.skipped] == ["event_type"]


def test_ordering_is_deterministic_on_ties() -> None:
    selection = _events(40, reserved=900, host="b")
    baseline = _events(40, reserved=150, host="a")
    first = compare_events(selection, baseline, scope="window").as_dict()
    second = compare_events(selection, baseline, scope="window").as_dict()
    assert first == second
    tied = [row["field"] for row in first["fields"] if row["score"] == 1.0]
    assert tied == sorted(tied)


def test_result_serializes_with_format_and_groups() -> None:
    result = compare_events(
        _events(2, session_id="s1", rank=1),
        _events(3, session_id="s2"),
        scope="session",
        criteria={"session_id": "s1"},
    ).as_dict()
    assert result["format"] == "stormlog.outlier_comparison"
    assert result["schema_version"] == 1
    assert result["criteria"] == {"session_id": "s1"}
    assert result["selection"] == {
        "event_count": 2,
        "session_ids": ["s1"],
        "ranks": [1],
        "start_ns": 1_000,
        "end_ns": 1_001,
    }
    assert describe_events([]).event_count == 0


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown comparison field"):
        compare_events(_events(1), _events(1), scope="window", fields=("pid",))
    assert "pid" not in ELIGIBLE_FIELDS
    assert "session_id" not in ELIGIBLE_FIELDS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "rank"}, "rank scope requires rank"),
        ({"scope": "window", "session_id": "s"}, "requires time_start_ns"),
        ({"scope": "session"}, "session scope requires session_id"),
        ({"scope": "fingerprint"}, "requires fingerprint_id"),
        ({"scope": "nope"}, "scope must be one of"),
        (
            {
                "scope": "window",
                "session_id": "s",
                "time_start_ns": 5,
                "time_end_ns": 5,
            },
            "earlier than",
        ),
        (
            {"scope": "rank", "rank": 0, "baseline_session_ids": ("s",)},
            "only apply to session scope",
        ),
        (
            {"scope": "session", "session_id": "s", "baseline_session_ids": ("s",)},
            "cannot also be a baseline",
        ),
        ({"scope": "rank", "rank": 0, "min_events": 0}, "min_events"),
        ({"scope": "rank", "rank": 0, "max_values": 0}, "max_values"),
        ({"scope": "rank", "rank": 0, "limit": -1}, "limit"),
        ({"scope": "fingerprint", "fingerprint_id": "x", "window_ns": 0}, "window"),
    ],
)
def test_comparison_filter_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ComparisonFilter(**kwargs)
