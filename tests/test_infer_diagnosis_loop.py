"""Engine-loop stalls from the execution hook's raw records."""

from __future__ import annotations

import json
import time
from pathlib import Path
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
    REASON_COVERAGE_UNKNOWN,
    REASON_EPOCH_CHANGED,
    REASON_PAUSE_UNKNOWN,
    REASON_RECORDS_DROPPED,
    REASON_REQUIRES_HOOK,
    REASON_TOO_FEW_STEPS,
    REASON_WRITER_ERRORS,
    LoopGapConfig,
    engine_loop_gap,
    pause_intervals,
)
from stormlog.infer.diagnosis_thresholds import LOOP_STALL_FACTOR
from stormlog.infer.vllm_hook.writer import EpochWriter, WriterLimits
from tests.vllm_execution_helpers import (
    SECOND,
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


def _hello() -> dict[str, Any]:
    """An engine hello from a hook that records scheduler pauses."""
    return hello("engine", 2600, 1, observes=["cache_reset", "enqueued", "pause"])


def _decode(*internals: str) -> list[dict[str, Any]]:
    return [
        member(name, scheduled=1, sighting="repeat", computed_before=8)
        for name in internals
    ]


def _loop(
    steps: int,
    *,
    beats: bool = True,
    stall_after: int | None = None,
    stall_ns: int = 0,
    locus: str = LOCUS_BETWEEN_STEPS,
    members: tuple[str, ...] = ("a", "b"),
    first_iteration: int = 0,
) -> list[dict[str, Any]]:
    """A synchronous loop: schedule, run, complete, schedule the next. One
    stall of ``stall_ns`` may follow step ``stall_after`` in ``locus``."""
    records: list[dict[str, Any]] = [_hello()]
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
    return _sequenced(records, beats=beats)


def _sequenced(
    records: list[dict[str, Any]], *, beats: bool = True
) -> list[dict[str, Any]]:
    if beats:
        records = _with_heartbeats(records)
    for seq, record in enumerate(records):
        record.setdefault("epoch", EPOCH)
        record["seq"] = seq
    return records


def _mono(record: dict[str, Any]) -> int | None:
    for key in ("start_mono_ns", "mono_ns"):
        if isinstance(record.get(key), int):
            return int(record[key])
    clock = record.get("clock") or {}
    return clock.get("mono_ns")


def _with_heartbeats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The writer beats once a second whatever the engine does: heartbeats
    each second after the first record, and one just after the last, so a
    loop is whole from its hello to its end. Records keep their order; a
    window that already has heartbeats is left as it is."""
    times = [t for t in map(_mono, records) if t is not None]
    if not times or any(r.get("kind") == "heartbeat" for r in records):
        return records
    beaten: list[dict[str, Any]] = []
    due = min(times) + SECOND
    for record in records:
        at = _mono(record)
        while at is not None and at > due:
            beaten.append(heartbeat(due, 0))
            due += SECOND
        beaten.append(record)
    beaten.append(heartbeat(max(times) + MS, 0))
    return beaten


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


def test_without_the_pause_capability_a_stall_gives_no_verdict() -> None:
    """A hook that does not record pauses cannot tell a stall from vLLM
    pausing all running requests (an RL weight sync, say): a stall over its
    limit is no verdict. A pause only removes stalls, so no stall over its
    limit still is one."""
    stalled = _loop(60, stall_after=40, stall_ns=300 * MS)
    steady = _loop(60)
    for records in (stalled, steady):
        records[0].pop("observes")

    signal = engine_loop_gap(stalled)
    assert (signal.exceeds, signal.reason) == (None, REASON_PAUSE_UNKNOWN)
    assert signal.detail["pause_capability"] is False
    assert engine_loop_gap(steady).exceeds is False


def test_a_tail_window_carries_its_epoch_s_hello() -> None:
    """An online trigger evaluates the tail of a long log. Its hello says
    whether pauses are recorded; passed with the tail, or prepended to it,
    it is neither a sequence gap nor a zero point for the tail's coverage."""
    # 4 s of steps; the tail holds the last 2 s, the stall 3.5 s in.
    records = _loop(800, stall_after=700, stall_ns=300 * MS)
    first, tail = records[0], records[-800:]

    alone = engine_loop_gap(tail)
    given = engine_loop_gap(tail, LoopGapConfig(hello=first))
    prepended = engine_loop_gap([first, *tail])

    assert (alone.exceeds, alone.reason) == (None, REASON_PAUSE_UNKNOWN)
    for signal in (given, prepended):
        assert signal.sufficient and signal.exceeds is True
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


@pytest.mark.parametrize(
    ("covered", "remainder_ms"),
    [
        ((-1, 1), 3_003.8),  # only its last millisecond
        ((-2_000, -1_500), 1_500.0),  # a stretch in the middle: the longer side
    ],
)
def test_an_excluded_interval_removes_only_what_it_covers(
    covered: tuple[int, int], remainder_ms: float
) -> None:
    """A 3 s stall (3,004.8 ms with the step's own run) that a profiler
    window covers in part still stalled for the rest: only the covered part
    is removed. ``covered`` is in ms from the stalled step's completion."""
    records = _loop(60, stall_after=40, stall_ns=3_000 * MS, locus=LOCUS_WITHIN_STEP)
    completion = next(
        r["wall_ns"]
        for r in records
        if r["kind"] == "completed" and r["iteration"] == "41"
    )
    config = LoopGapConfig(
        exclude_wall=[(completion + covered[0] * MS, completion + covered[1] * MS)]
    )
    signal = engine_loop_gap(records, config)
    assert signal.exceeds is True
    assert signal.value == pytest.approx(remainder_ms * MS, abs=0.01 * MS)


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
    last = [r for r in records if r["kind"] == "completed"][-1]
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
def test_records_with_gaps_drops_or_two_epochs_give_no_verdict() -> None:
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
    two = _loop(20)
    two[-1]["epoch"] = "engine-2600-2"
    assert engine_loop_gap(two).reason == REASON_EPOCH_CHANGED


def _fields(record: dict[str, Any]) -> dict[str, Any]:
    """A record as the hook hands it to the writer, which adds its own kind,
    epoch and sequence."""
    return {k: v for k, v in record.items() if k not in ("kind", "epoch", "seq")}


def _capped_writer_log(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A loop written by the real writer until its disk cap: from then on it
    drops every record and writes no heartbeat, and only status.json says
    it is capped."""
    writer = EpochWriter(
        tmp_path,
        "engine",
        limits=WriterLimits(max_bytes=25_000, heartbeat_seconds=0.02),
    )
    writer.emit("hello", _fields(_hello()))
    for index, record in enumerate(_loop(60)[1:]):
        if record["kind"] == "heartbeat":
            continue
        writer.emit(record["kind"], _fields(record))
        if index % 10 == 0:
            time.sleep(0.03)  # heartbeats are written while the writer is live
    time.sleep(0.1)
    writer.close(goodbye=True)
    directory = next(tmp_path.glob("*/*"))
    records = [
        json.loads(line)
        for segment in sorted(directory.glob("0*.jsonl*"))
        for line in segment.read_text().splitlines()
    ]
    return records, json.loads((directory / "status.json").read_text())


def test_a_capped_writer_gives_no_verdict(tmp_path: Path) -> None:
    records, status = _capped_writer_log(tmp_path)
    beats = [r for r in records if r["kind"] == "heartbeat"]
    # Capped, it drops every record and writes no heartbeat; no heartbeat
    # ever says capped, only the status.
    assert status["capped"] and sum(status["dropped"].values()) > 0
    assert beats and not any(beat.get("capped") for beat in beats)

    signal = engine_loop_gap(records, LoopGapConfig(status=status))

    assert (signal.reason, signal.exceeds) == (REASON_CAPPED, None)


def _dropped_completions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A 60-step loop whose completed records of steps 41 to 59 were dropped,
    and the heartbeat that counts them."""
    body = [r for r in _loop(60, beats=False)[1:]]
    kept = [
        r for r in body if not (r["kind"] == "completed" and int(r["iteration"]) > 40)
    ]
    end = max(int(r.get("mono_ns", 0)) for r in body)
    return kept, heartbeat(end, 0, dropped={"completed": 19})


@pytest.mark.parametrize("shape", ["one_beat", "drops_before_both", "killed_tail"])
def test_drops_no_heartbeats_bracket_give_no_verdict(shape: str) -> None:
    """Missing completions make a false 90 ms gap; unless heartbeats on both
    sides show nothing lost across it, there is no verdict."""
    kept, counted = _dropped_completions()
    first = _hello()
    windows = {
        # The hello (nothing lost) and one heartbeat counting the drops.
        "one_beat": [first, *kept, counted],
        # Both heartbeats came after the drops.
        "drops_before_both": [
            first,
            *kept,
            counted,
            {
                **counted,
                "mono_ns": counted["mono_ns"] + MS,
                "wall_ns": counted["wall_ns"] + MS,
            },
        ],
        # A SIGKILLed epoch: heartbeats before the drops, none after.
        "killed_tail": [
            first,
            heartbeat(T, 0),
            *kept[:60],
            heartbeat(T + 150 * MS, 0),
            *kept[60:],
        ],
    }
    signal = engine_loop_gap(_sequenced(windows[shape], beats=False))
    assert signal.exceeds is None
    assert not signal.sufficient


def test_drops_heartbeats_bracket_are_a_counted_loss() -> None:
    kept, counted = _dropped_completions()
    records = [_hello(), heartbeat(T, 0), *kept, counted]
    signal = engine_loop_gap(_sequenced(records, beats=False))
    assert signal.reason == REASON_RECORDS_DROPPED


def test_an_ongoing_stall_needs_the_writer_heard_from_lately() -> None:
    """A capped or killed writer stops records and heartbeats together, so
    the engine running on unrecorded looks like a stall still going on."""
    records = _loop(60)
    end = _end(records)
    late = LoopGapConfig(now_wall_ns=end + 5 * SECOND)
    signal = engine_loop_gap(records, late)
    # The last heartbeat came just after the last completion, 5 s ago.
    assert (signal.exceeds, signal.reason) == (None, REASON_COVERAGE_UNKNOWN)


def test_no_steps_or_no_completed_step() -> None:
    assert engine_loop_gap([]).reason == REASON_REQUIRES_HOOK
    only_scheduled = _sequenced([r for r in _loop(5) if r["kind"] != "completed"])
    assert engine_loop_gap(only_scheduled).reason == REASON_TOO_FEW_STEPS


def _async_loop(steps: int, *, late_after: int | None = None) -> list[dict[str, Any]]:
    """Async scheduling: each step is scheduled while the previous one runs,
    3 ms before it completes. ``late_after`` delays one schedule() call
    until 300 ms after the previous completion."""
    records: list[dict[str, Any]] = [_hello()]
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


def _idle_then_a_new_request(idle_ns: int) -> list[dict[str, Any]]:
    """The shape of a real vLLM 0.30.0 run under async scheduling: request a
    runs and finishes, an empty step follows, the engine idles with nothing
    to run, then request b arrives, and each of its steps is scheduled 3 ms
    before the one before it completes."""
    records: list[dict[str, Any]] = [_hello()]
    ends = [T + (index + 1) * CADENCE for index in range(30)]
    for index, end in enumerate(ends):
        last = index == len(ends) - 1
        records.append(scheduled(index, end - CADENCE - 3 * MS, _decode("a")))
        finish = "length" if last else None
        records.append(completed(index, end, [done("a", finish_reason=finish)]))
    records.append(scheduled(30, ends[-1] - 3 * MS, []))
    records.append(completed(30, ends[-1] + MS, []))
    end = ends[-1] + MS + idle_ns + CADENCE
    records.append(scheduled(31, end - CADENCE, _decode("b")))
    records.append(completed(31, end, [done("b")]))
    for index in range(32, 62):
        records.append(scheduled(index, end - 3 * MS, _decode("b")))
        end += CADENCE
        records.append(completed(index, end, [done("b")]))
    records.sort(key=lambda r: r.get("start_mono_ns", r.get("mono_ns", 0)))
    return _sequenced(records)


def test_an_idle_engine_before_an_async_request_is_not_a_stall() -> None:
    """Request b's second step is scheduled before its first completes, so
    the latest completion before it is a's, before the idle stretch. Work is
    ready only when a request ran in the step whose completion starts the
    stretch: here none did, and 2 s of idling is no stall."""
    signal = engine_loop_gap(_idle_then_a_new_request(2_000 * MS))
    assert signal.exceeds is False
    assert signal.value is not None and signal.value < 50 * MS


def _two_runs_of(
    member_of: Any, gap_ns: int, *, steps: int = 30
) -> list[dict[str, Any]]:
    """One request runs ``steps`` steps, nothing runs for ``gap_ns``, then it
    runs ``steps`` more: synchronous steps of ``member_of(index)``."""
    records: list[dict[str, Any]] = [_hello()]
    clock = T
    for index in range(2 * steps):
        if index == steps:
            clock += gap_ns
        records.append(scheduled(index, clock, [member_of(index)]))
        clock += CADENCE
        records.append(completed(index, clock, [done("s")]))
    return _sequenced(records)


def test_a_streaming_request_waiting_for_input_is_not_ready() -> None:
    """A streaming-input request that ran out of input waits for more without
    being scheduled: its gap is the client's, not a stall."""
    records = _two_runs_of(
        lambda index: member("s", scheduled=1, sighting="repeat", resumable=True),
        2_000 * MS,
    )
    signal = engine_loop_gap(records)
    assert signal.exceeds is False


def test_a_prefill_split_across_steps_is_ready_between_them() -> None:
    """A long prompt prefilled in chunks runs in every step until its first
    token, so a host gap between two chunks is a stall."""
    records = _two_runs_of(
        lambda index: member(
            "s", scheduled=64, computed_before=64 * index, prompt_tokens=8192
        ),
        2_000 * MS,
    )
    signal = engine_loop_gap(records)
    assert signal.exceeds is True
    assert signal.detail["locus"] == LOCUS_BETWEEN_STEPS


def test_async_late_schedule_call_is_a_host_stall() -> None:
    signal = engine_loop_gap(_async_loop(60, late_after=40))
    assert signal.exceeds is True
    assert signal.detail["locus"] == LOCUS_BETWEEN_STEPS


def _fastest(records: list[dict[str, Any]], runs: int = 2) -> float:
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        engine_loop_gap(records)
        times.append(time.perf_counter() - started)
    return min(times)


def test_a_long_window_is_evaluated_in_linear_time() -> None:
    """Four times the steps take about four times as long, never the
    sixteen a quadratic search would: a ratio, so a loaded machine does not
    fail it."""
    short = _loop(5_000, stall_after=4_900, stall_ns=200 * MS)
    long = _loop(20_000, stall_after=19_900, stall_ns=200 * MS)

    assert engine_loop_gap(long).exceeds is True
    assert _fastest(long) < 8 * _fastest(short)


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


def _prefills_in_decode(
    steps: int, prefills: dict[int, tuple[int, int]]
) -> list[dict[str, Any]]:
    """A 5 ms decode loop in which the steps of ``prefills`` also schedule
    (tokens, duration ns) of prefill."""
    records: list[dict[str, Any]] = [_hello()]
    clock = T
    for index in range(steps):
        members = _decode("a", "b")
        tokens, run = prefills.get(index, (0, CADENCE))
        if tokens:
            members[0]["scheduled"] = tokens - 1
        records.append(scheduled(index, clock, members))
        clock += run
        records.append(completed(index, clock, [done("a"), done("b")]))
    return _sequenced(records)


@pytest.mark.parametrize("prefill_ms", [150, 60])
def test_a_second_long_prefill_is_not_measured_against_decode_steps(
    prefill_ms: int,
) -> None:
    """Two 2,048-token steps 2 s apart: the second finds the first, larger
    than every decode step, in its window, but one step is no cadence. Only
    enough earlier steps at least its size may set its limit."""
    long = (2_048, prefill_ms * MS)
    signal = engine_loop_gap(_prefills_in_decode(800, {100: long, 500: long}))
    assert signal.exceeds is False
    assert signal.detail["baseline"] == BASELINE_FLOOR


@pytest.mark.parametrize(("slow_ms", "exceeds"), [(300, False), (700, True)])
def test_larger_earlier_steps_set_the_limit_of_a_rarer_size(
    slow_ms: int, exceeds: bool
) -> None:
    """Thirty 4,096-token steps of 60 ms, then one 2,048-token step: too few
    of its own size, so the larger steps' cadence, 60 ms, sets its limit."""
    prefills = {index: (4_096, 60 * MS) for index in range(10, 310, 10)}
    prefills[330] = (2_048, slow_ms * MS)
    signal = engine_loop_gap(_prefills_in_decode(400, prefills))
    assert signal.exceeds is exceeds
    assert signal.detail["baseline"] == "unmatched"


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        ({"host_stall.stall_factr": 1.0}, "unknown"),
        ({LOOP_STALL_FACTOR: float("nan")}, "finite"),
        ({LOOP_STALL_FACTOR: 0.0}, "positive"),
        ({"host_stall.stall_floor_ns": 0.0}, "positive"),
        ({"host_stall.heartbeat_grace_ns": -1.0}, "positive"),
    ],
)
def test_a_threshold_override_that_could_not_decide_is_refused(
    thresholds: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LoopGapConfig(thresholds=thresholds)


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
