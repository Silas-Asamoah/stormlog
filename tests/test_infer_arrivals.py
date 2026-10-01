"""Seeded arrival schedules and arrival trace loading."""

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.arrivals import (
    ArrivalSpec,
    ArrivalTrace,
    arrival_offsets,
    load_arrival_trace,
)


def _offsets(spec: ArrivalSpec, **bounds: Any) -> list[float]:
    values: dict[str, Any] = {"count": None, "duration_seconds": None, "seed": 0}
    values.update(bounds)
    return arrival_offsets(spec, **values)


def test_fixed_rate_spaces_arrivals_evenly() -> None:
    spec = ArrivalSpec(mode="fixed-rate", rate_per_second=4.0)
    assert _offsets(spec, count=4) == [0.0, 0.25, 0.5, 0.75]
    # The window keeps arrivals before it closes.
    assert _offsets(spec, duration_seconds=1.0) == [0.0, 0.25, 0.5, 0.75]


def test_poisson_is_seeded_and_has_the_requested_mean_rate() -> None:
    spec = ArrivalSpec(mode="poisson", rate_per_second=5.0)
    first = _offsets(spec, count=2000, seed=7)
    assert first == _offsets(spec, count=2000, seed=7)
    assert first != _offsets(spec, count=2000, seed=8)
    assert first[0] == 0.0
    assert first == sorted(first)
    mean_gap = first[-1] / (len(first) - 1)
    assert mean_gap == pytest.approx(0.2, rel=0.05)


def test_bursts_send_groups_at_each_interval() -> None:
    spec = ArrivalSpec(mode="burst", burst_size=3, burst_interval_seconds=2.0)
    assert _offsets(spec, count=7) == [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0]
    assert _offsets(spec, duration_seconds=4.0) == [0.0] * 3 + [2.0] * 3


def test_replay_sends_the_whole_trace_unless_bounded() -> None:
    trace = ArrivalTrace(offsets_seconds=(0.0, 0.1, 0.5, 2.0), source="test")
    spec = ArrivalSpec(mode="replay", trace=trace)
    assert _offsets(spec) == [0.0, 0.1, 0.5, 2.0]
    assert _offsets(spec, count=2) == [0.0, 0.1]
    assert _offsets(spec, duration_seconds=1.0) == [0.0, 0.1, 0.5]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "steady"}, "arrival mode must be one of"),
        ({"mode": "poisson"}, "poisson arrivals need a rate"),
        ({"mode": "fixed-rate", "rate_per_second": 0.0}, "positive number"),
        ({"mode": "fixed-rate", "rate_per_second": float("inf")}, "positive"),
        ({"mode": "closed", "rate_per_second": 2.0}, "do not take a rate"),
        ({"mode": "burst", "burst_size": 2}, "need a burst interval"),
        (
            {"mode": "burst", "burst_size": 0, "burst_interval_seconds": 1.0},
            "burst size must be >= 1",
        ),
        ({"mode": "replay"}, "need an arrival trace"),
    ],
)
def test_specs_reject_settings_their_mode_cannot_use(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ArrivalSpec(**changes)


def test_open_loop_schedules_need_a_bound_and_closed_loops_have_none() -> None:
    with pytest.raises(ValueError, match="request count or duration"):
        _offsets(ArrivalSpec(mode="fixed-rate", rate_per_second=1.0))
    with pytest.raises(ValueError, match="closed loop has no arrival schedule"):
        _offsets(ArrivalSpec(), count=1)


def test_case_labels_name_the_arrival_shape() -> None:
    assert ArrivalSpec().case_label() == "closed"
    assert ArrivalSpec(mode="poisson", rate_per_second=0.5).case_label() == (
        "poisson0.5"
    )
    assert ArrivalSpec(mode="fixed-rate", rate_per_second=2.0).case_label() == (
        "fixed2"
    )
    burst = ArrivalSpec(mode="burst", burst_size=8, burst_interval_seconds=1.5)
    assert burst.case_label() == "burst8x1.5s"


def _write_lines(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_trace_offsets_are_sorted_and_start_at_zero(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "trace.jsonl",
        [{"offset_ms": 1500}, {"offset_ms": 500}, {"offset_ms": 750.5}],
    )
    trace = load_arrival_trace(path)
    assert trace.offsets_seconds == pytest.approx((0.0, 0.2505, 1.0))
    assert trace.source == "offset_ms trace"
    spec = ArrivalSpec(mode="replay", trace=trace)
    assert spec.to_record()["trace"]["arrivals"] == 3
    assert spec.to_record()["trace"]["digest"] == trace.digest()


def _request(case_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "event_type": "infer.request",
        "phase": "measured",
        "case_id": case_id,
        "status": "ok",
        **fields,
    }


def test_artifact_trace_replays_one_measured_case(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "infer.jsonl",
        [
            {"event_type": "infer.session"},
            _request("a", started_at_ns=5_000_000_000),
            _request("a", started_at_ns=5_400_000_000),
            # The intended arrival wins over the time the request was sent.
            _request("a", intended_at_ns=6_000_000_000, started_at_ns=6_900_000_000),
            _request("a", phase="warmup", started_at_ns=1),
            _request("a", status="dropped", intended_at_ns=7_000_000_000),
            _request("b", started_at_ns=9_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="choose one case .*: a, b"):
        load_arrival_trace(path)
    trace = load_arrival_trace(path, case_id="a")
    assert trace.offsets_seconds == (0.0, 0.4, 1.0)
    assert (trace.source, trace.case_id) == ("stormlog artifact", "a")
    with pytest.raises(ValueError, match="no measured case 'c'"):
        load_arrival_trace(path, case_id="c")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"offset_ms": -1}], "finite number >= 0"),
        ([{"offset_ms": "1"}], "numeric offset_ms"),
        ([{"offset_ms": 1}, {"other": 2}], "numeric offset_ms"),
        ([{"event_type": "infer.session"}], "no offset_ms lines or measured"),
    ],
)
def test_malformed_traces_are_rejected(
    tmp_path: Path, records: list[dict[str, Any]], message: str
) -> None:
    path = _write_lines(tmp_path / "trace.jsonl", records)
    with pytest.raises(ValueError, match=message):
        load_arrival_trace(path)


def test_an_absurd_schedule_is_refused() -> None:
    spec = ArrivalSpec(mode="fixed-rate", rate_per_second=1e9)
    with pytest.raises(ValueError, match="more than 1,000,000 requests"):
        _offsets(spec, duration_seconds=0.01)
    assert len(_offsets(spec, count=1_000_000)) == 1_000_000
