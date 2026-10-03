"""Engine-loop stalls from the execution hook's raw records."""

from __future__ import annotations

import time
from typing import Any

import pytest

from stormlog.infer.diagnosis_loop import (
    ATTRIBUTION_HOST,
    ATTRIBUTION_HOST_OR_GPU,
    BASELINE_FLOOR,
    LOCUS_BETWEEN_STEPS,
    LOCUS_IN_SCHEDULE,
    LOCUS_WITHIN_STEP,
    REASON_CAPPED,
    REASON_EPOCH_CHANGED,
    REASON_RECORDS_DROPPED,
    REASON_REQUIRES_HOOK,
    REASON_TOO_FEW_STEPS,
    REASON_WRITER_ERRORS,
    LoopGapConfig,
    engine_loop_gap,
    pause_intervals,
)
from stormlog.infer.diagnosis_thresholds import LOOP_STALL_FACTOR
from tests.vllm_execution_helpers import (
    WALL_OFFSET,
    completed,
    done,
    heartbeat,
    hello,
    member,
    scheduled,
)

MS = 1_000_000
T = 1_000_000 * MS  # the engine's monotonic clock when the window starts
CADENCE = 5 * MS
SCHEDULE = 200_000  # 0.2 ms inside schedule()
EPOCH = "engine-2600-1"


def _decode(*internals: str) -> list[dict[str, Any]]:
    return [
        member(name, scheduled=1, sighting="repeat", computed_before=8)
        for name in internals
    ]


def _loop(
    steps: int,
    *,
    stall_after: int | None = None,
    stall_ns: int = 0,
    locus: str = LOCUS_BETWEEN_STEPS,
    members: tuple[str, ...] = ("a", "b"),
    first_iteration: int = 0,
) -> list[dict[str, Any]]:
    """A synchronous loop: schedule, run, complete, schedule the next. One
    stall of ``stall_ns`` may follow step ``stall_after`` in ``locus``."""
    records: list[dict[str, Any]] = [hello("engine", 2600, 1)]
    clock = T
    for index in range(steps):
        stalled = stall_after is not None and index == stall_after + 1
        if stalled and locus == LOCUS_BETWEEN_STEPS:
            clock += stall_ns
        in_schedule = SCHEDULE + (
            stall_ns if stalled and locus == LOCUS_IN_SCHEDULE else 0
        )
        records.append(
            scheduled(
                first_iteration + index,
                clock,
                _decode(*members),
                duration_ns=in_schedule,
            )
        )
        run = (
            CADENCE
            - SCHEDULE
            + (stall_ns if stalled and locus == LOCUS_WITHIN_STEP else 0)
        )
        clock += in_schedule + run
        records.append(
            completed(first_iteration + index, clock, [done(n) for n in members])
        )
    return _sequenced(records)


def _sequenced(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for seq, record in enumerate(records):
        record.setdefault("epoch", EPOCH)
        record["seq"] = seq
    return records


def _end(records: list[dict[str, Any]]) -> int:
    return max(
        int(r.get("wall_ns", 0)) for r in records if r.get("kind") == "completed"
    )


# ------------------------------------------------------------------ loci
def test_a_steady_loop_has_no_stall() -> None:
    signal = engine_loop_gap(_loop(60))
    assert signal.sufficient and signal.exceeds is False
    assert signal.value is not None and signal.value < 50 * MS


@pytest.mark.parametrize(
    ("locus", "attribution"),
    [
        (LOCUS_BETWEEN_STEPS, ATTRIBUTION_HOST),
        (LOCUS_IN_SCHEDULE, ATTRIBUTION_HOST),
        (LOCUS_WITHIN_STEP, ATTRIBUTION_HOST_OR_GPU),
    ],
)
def test_a_stall_is_found_in_its_locus(locus: str, attribution: str) -> None:
    signal = engine_loop_gap(_loop(60, stall_after=40, stall_ns=200 * MS, locus=locus))
    assert signal.exceeds is True
    assert signal.detail["locus"] == locus
    assert signal.detail["attribution"] == attribution
    assert signal.value is not None and signal.value >= 200 * MS
    # 10 x the 5 ms cadence of 40 earlier steps with the same decode batch.
    assert signal.detail["baseline"] == "matched"
    assert signal.threshold == pytest.approx(50 * MS)


def test_an_idle_gap_without_continuing_work_is_not_a_stall() -> None:
    first = _loop(30, members=("a",))
    second = _loop(30, members=("z",), first_iteration=30)
    shift = _end(first) - WALL_OFFSET + 2_000 * MS - T
    for record in second[1:]:
        for key in ("start_mono_ns", "end_mono_ns", "mono_ns"):
            if key in record:
                record[key] += shift
        for key in ("start_wall_ns", "end_wall_ns", "wall_ns"):
            if key in record:
                record[key] += shift
    signal = engine_loop_gap(_sequenced(first + second[1:]))
    assert signal.exceeds is False


# ----------------------------------------------------------- no ready work
def test_a_scheduler_pause_with_the_capability_is_not_a_stall() -> None:
    records = _loop(60, stall_after=40, stall_ns=300 * MS)
    records[0]["observes"] = ["enqueued", "cache_reset", "pause"]
    completion = next(
        r["wall_ns"]
        for r in records
        if r["kind"] == "completed" and r["iteration"] == "40"
    )
    records += [
        {
            "kind": "pause",
            "from": "UNPAUSED",
            "to": "PAUSED_ALL",
            "wall_ns": completion + MS,
        },
        {
            "kind": "pause",
            "from": "PAUSED_ALL",
            "to": "UNPAUSED",
            "wall_ns": completion + 290 * MS,
        },
    ]
    signal = engine_loop_gap(_sequenced(records))
    assert signal.exceeds is False
    assert signal.detail["pause_capability"] is True


def test_pause_intervals_follow_paused_all_only() -> None:
    records = [
        {"kind": "pause", "from": "UNPAUSED", "to": "PAUSED_NEW", "wall_ns": 10},
        {"kind": "pause", "from": "PAUSED_NEW", "to": "PAUSED_ALL", "wall_ns": 20},
        {"kind": "pause", "from": "PAUSED_ALL", "to": "UNPAUSED", "wall_ns": 30},
        {"kind": "pause", "from": "UNPAUSED", "to": "PAUSED_ALL", "wall_ns": 40},
    ]
    assert pause_intervals(records) == [(20, 30), (40, 2**63 - 1)]


def test_an_excluded_interval_is_not_a_stall() -> None:
    records = _loop(60, stall_after=40, stall_ns=300 * MS)
    completion = next(
        r["wall_ns"]
        for r in records
        if r["kind"] == "completed" and r["iteration"] == "40"
    )
    config = LoopGapConfig(exclude_wall=[(completion, completion + 300 * MS)])
    signal = engine_loop_gap(records, config)
    assert signal.exceeds is False
    assert signal.detail["pause_capability"] is False


# -------------------------------------------------------------- ongoing
def test_an_ongoing_stall_counts_to_the_evaluation_time() -> None:
    records = _loop(60)
    end = _end(records)
    signal = engine_loop_gap(records, LoopGapConfig(now_wall_ns=end + 400 * MS))
    assert signal.exceeds is True
    assert signal.detail["ongoing"] is True
    assert signal.detail["locus"] == LOCUS_BETWEEN_STEPS


def test_an_ongoing_stall_with_a_step_in_flight_may_be_the_gpu() -> None:
    records = _loop(60)
    end = _end(records)
    records.append(scheduled(60, end - WALL_OFFSET + MS, _decode("a", "b")))
    signal = engine_loop_gap(
        _sequenced(records), LoopGapConfig(now_wall_ns=end + 400 * MS)
    )
    assert signal.detail["locus"] == LOCUS_WITHIN_STEP
    assert signal.detail["attribution"] == ATTRIBUTION_HOST_OR_GPU


def test_no_ongoing_stall_once_every_request_finished() -> None:
    records = _loop(60)
    last = records[-1]
    last["members"] = [done(name, finish_reason="stop") for name in ("a", "b")]
    signal = engine_loop_gap(
        records, LoopGapConfig(now_wall_ns=_end(records) + 900 * MS)
    )
    assert signal.exceeds is False


# ------------------------------------------------------------- baseline
def test_without_earlier_steps_the_floor_applies() -> None:
    """spec5's shape: the first step ran 1.3 s, with nothing before it."""
    records = _loop(4, stall_after=0, stall_ns=1_300 * MS, locus=LOCUS_WITHIN_STEP)
    signal = engine_loop_gap(records)
    assert signal.exceeds is True
    assert signal.detail["baseline"] == BASELINE_FLOOR
    assert signal.threshold == pytest.approx(500 * MS)


def test_a_threshold_override_is_reported() -> None:
    records = _loop(60, stall_after=40, stall_ns=200 * MS)
    signal = engine_loop_gap(
        records, LoopGapConfig(thresholds={LOOP_STALL_FACTOR: 100})
    )
    assert signal.threshold_overridden is True
    assert signal.exceeds is False


# ---------------------------------------------------------- sufficiency
def test_records_with_gaps_drops_caps_or_two_epochs_give_no_verdict() -> None:
    gap = _loop(60, stall_after=40, stall_ns=200 * MS)
    del gap[10]
    assert engine_loop_gap(gap).reason == REASON_RECORDS_DROPPED
    assert engine_loop_gap(gap).exceeds is None
    dropped = _loop(20)
    dropped += [
        heartbeat(T + 10 * MS, 5),
        heartbeat(T + 20 * MS, 6, dropped={"scheduled": 1}),
    ]
    assert engine_loop_gap(_sequenced(dropped)).reason == REASON_RECORDS_DROPPED
    capped = _loop(20) + [heartbeat(T + 10 * MS, 5, capped=True)]
    assert engine_loop_gap(_sequenced(capped)).reason == REASON_CAPPED
    two = _loop(20)
    two[-1]["epoch"] = "engine-2600-2"
    assert engine_loop_gap(two).reason == REASON_EPOCH_CHANGED


def test_no_steps_or_no_completed_step() -> None:
    assert engine_loop_gap([]).reason == REASON_REQUIRES_HOOK
    only_scheduled = _sequenced([r for r in _loop(5) if r["kind"] != "completed"])
    assert engine_loop_gap(only_scheduled).reason == REASON_TOO_FEW_STEPS


def _async_loop(steps: int, *, late_after: int | None = None) -> list[dict[str, Any]]:
    """Async scheduling: each step is scheduled while the previous one runs,
    3 ms before it completes. ``late_after`` delays one schedule() call
    until 300 ms after the previous completion."""
    records: list[dict[str, Any]] = [hello("engine", 2600, 1)]
    for index in range(steps):
        end = (
            T
            + (index + 1) * CADENCE
            + (300 * MS if late_after is not None and index > late_after else 0)
        )
        start = end - CADENCE - 3 * MS
        if late_after is not None and index == late_after + 1:
            start = end - CADENCE
        records.append(scheduled(index, start, _decode("a", "b")))
    for index in range(steps):
        end = (
            T
            + (index + 1) * CADENCE
            + (300 * MS if late_after is not None and index > late_after else 0)
        )
        records.append(completed(index, end, [done("a"), done("b")]))
    records.sort(key=lambda r: r.get("start_mono_ns", r.get("mono_ns", 0)))
    return _sequenced(records)


def test_async_scheduling_overlap_is_not_a_stall() -> None:
    signal = engine_loop_gap(_async_loop(60))
    assert signal.exceeds is False


def test_async_late_schedule_call_is_a_host_stall() -> None:
    signal = engine_loop_gap(_async_loop(60, late_after=40))
    assert signal.exceeds is True
    assert signal.detail["locus"] == LOCUS_BETWEEN_STEPS


def test_a_long_window_is_evaluated_in_linear_time() -> None:
    records = _loop(20_000, stall_after=15_000, stall_ns=200 * MS)
    started = time.perf_counter()
    signal = engine_loop_gap(records)
    assert time.perf_counter() - started < 5.0
    assert signal.exceeds is True


def test_a_long_prefill_step_is_not_measured_against_decode_steps() -> None:
    """A 2,048-token step that takes 150 ms is 30 x the 5 ms decode cadence,
    but no earlier step was that large, so only the absolute floor applies."""
    records = _loop(40, stall_after=30, stall_ns=150 * MS, locus=LOCUS_WITHIN_STEP)
    step = next(
        r for r in records if r["kind"] == "scheduled" and r["iteration"] == "31"
    )
    step["total_tokens"] = 2048
    signal = engine_loop_gap(records)
    assert signal.exceeds is False
    assert signal.detail["baseline"] == BASELINE_FLOOR


def test_writer_errors_rising_between_heartbeats_give_no_verdict() -> None:
    """A failed write is cut back and counted as an error, not always as a
    drop, so rising errors also mean the records may be incomplete."""
    records = _loop(20) + [
        heartbeat(T + 10 * MS, 5),
        heartbeat(T + 20 * MS, 6, errors=1),
    ]
    signal = engine_loop_gap(_sequenced(records))
    assert signal.reason == REASON_WRITER_ERRORS
    assert signal.exceeds is None
