"""What the execution import derives from the hook's later records: the
clock alignment's bracket, enqueue times, source sequences, loss coverage,
and dated stages for preemptions, cache resets and pauses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stormlog.infer.correlation_accounting import (
    align_timestamp,
    alignment_offset_bounds,
    resolve_inference_events,
)
from stormlog.infer.correlation_events import (
    ClockAlignmentEvent,
    CorrelationContext,
    CorrelationEvent,
    EntityRef,
    IterationEvent,
    MembershipEvent,
    RequestEvent,
    StageEvent,
)
from stormlog.infer.vllm_execution import (
    ALIGNMENT_BASIS,
    ReduceResult,
    RunFacts,
    RunRequest,
    reduce_execution_log,
)
from stormlog.infer.vllm_execution_log import read_execution_log
from tests.vllm_execution_helpers import (
    BOOT,
    HOST,
    SECOND,
    WALL_OFFSET,
    alias,
    cache_reset,
    completed,
    done,
    enqueued,
    goodbye,
    heartbeat,
    hello,
    importer,
    member,
    pause,
    scheduled,
    stamp,
    terminal,
    write_epoch,
)

PID, START = 2600, 1_790_000_000_000_000_000
EPOCH = f"engine-{PID}-{START}"
T0 = 1_000 * SECOND
HERE = importer(T0 + 5 * SECOND)
RUN = "run-1"
REQUEST0 = "c1_in8_out4_measured_0_0"
X0 = f"stormlog-{RUN}-{REQUEST0}"
OWN0 = f"chatcmpl-{X0}-0f3a9c1d"
OTHER = "chatcmpl-stormlog-run-9-c1_x_0-deadbeef"


def _facts(**changes: Any) -> RunFacts:
    values: dict[str, Any] = {
        "run_id": RUN,
        "session_id": "session-1",
        "client_clock_domain": f"{HOST}/{BOOT}/unix_epoch_ns",
        "requests": {X0: RunRequest(REQUEST0, X0, "c1_in8_out4", "measured")},
    }
    values.update(changes)
    return RunFacts(**values)


def _reduce(
    root: Path, records: list[dict[str, Any]], facts: RunFacts | None = None
) -> ReduceResult:
    write_epoch(root, "engine", PID, START, records)
    return reduce_execution_log(
        read_execution_log(root, importer=HERE), facts or _facts()
    )


def _of(result: ReduceResult, kind: type[CorrelationEvent]) -> list[Any]:
    return [event for event in result.events if isinstance(event, kind)]


# ---------------------------------------------------------------- alignment


def test_the_alignment_is_the_midpoint_of_the_hello_s_bracket(tmp_path: Path) -> None:
    # Astra's counterexample: wall 100 before the monotonic read at 109, and
    # 110 after it, so the true offset lies in [100 - 109, 110 - 109].
    clock = {"wall_ns": 100, "mono_ns": 109, "wall_after_ns": 110, "gap_ns": 10}
    result = _reduce(tmp_path, [hello("engine", PID, START, clock=clock)])

    (alignment,) = _of(result, ClockAlignmentEvent)
    assert (alignment.offset_ns, alignment.uncertainty_ns) == (-4, 5)
    assert alignment_offset_bounds(alignment) == (-9, 1)
    assert alignment.metadata["alignment_basis"] == ALIGNMENT_BASIS
    assert alignment.metadata["bracket"] == {
        "wall_ns": 100,
        "mono_ns": 109,
        "wall_after_ns": 110,
    }
    placed = align_timestamp(
        200,
        from_clock_domain=alignment.from_clock_domain,
        to_clock_domain=alignment.to_clock_domain,
        alignments=[alignment],
    )
    assert (placed.value_ns - placed.uncertainty_ns, placed.value_ns) == (191, 196)


def test_a_hello_from_before_the_bracket_field_uses_its_gap(tmp_path: Path) -> None:
    clock = {"wall_ns": 100, "mono_ns": 109, "gap_ns": 10}
    (alignment,) = _of(
        _reduce(tmp_path, [hello("engine", PID, START, clock=clock)]),
        ClockAlignmentEvent,
    )
    assert alignment_offset_bounds(alignment) == (-9, 1)


def test_a_bracket_the_wall_clock_stepped_back_across_gives_no_alignment(
    tmp_path: Path,
) -> None:
    clock = {"wall_ns": 100, "mono_ns": 109, "wall_after_ns": 90, "gap_ns": -10}
    result = _reduce(tmp_path, [hello("engine", PID, START, clock=clock)])

    assert _of(result, ClockAlignmentEvent) == []


def test_a_legacy_alignment_is_read_as_its_bracket_never_rewritten() -> None:
    """Same offset, uncertainty and gap: an earlier import's record put the
    offset at the bracket's lower end, a corrected one at its midpoint."""
    context = CorrelationContext(
        run_id=RUN,
        session_id="session-1",
        producer_id="vllm:node-7",
        source="stormlog.infer.import_execution",
        clock_domain=f"{HOST}/{BOOT}/monotonic_ns",
        clock_kind="monotonic",
        collection_mode="imported",
        provenance="observed",
    )

    def alignment(**metadata: Any) -> ClockAlignmentEvent:
        return ClockAlignmentEvent(
            context=context,
            event_id="alignment:x",
            metadata={"gap_ns": 10, **metadata},
            from_clock_domain=f"{HOST}/{BOOT}/monotonic_ns",
            to_clock_domain=f"{HOST}/{BOOT}/unix_epoch_ns",
            offset_ns=-9,
            uncertainty_ns=5,
        )

    legacy, corrected = alignment(), alignment(alignment_basis=ALIGNMENT_BASIS)
    assert alignment_offset_bounds(legacy) == (-9, 1)
    assert alignment_offset_bounds(corrected) == (-14, -4)
    placed = align_timestamp(
        200,
        from_clock_domain=legacy.from_clock_domain,
        to_clock_domain=legacy.to_clock_domain,
        alignments=[legacy],
    )
    assert (placed.value_ns, placed.uncertainty_ns) == (196, 5)


# ---------------------------------------------------------------- enqueue


def _run(*records: dict[str, Any]) -> list[dict[str, Any]]:
    return [hello("engine", PID, START), *records]


def test_a_request_carries_when_it_entered_the_scheduler(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        _run(
            alias(OWN0, f"chatcmpl-{X0}", T0 - 30),
            enqueued(OWN0, T0 - 20, structured_output=True),
            alias(OTHER, "chatcmpl-other", T0 - 15),  # its enqueue was not recorded
            scheduled(0, T0, [member(OWN0, scheduled=8), member(OTHER, scheduled=8)]),
            completed(0, T0 + SECOND, [done(OWN0), done(OTHER)]),
            heartbeat(T0 + 2 * SECOND, 6),
        ),
    )

    requests = {r.metadata["ownership"]: r for r in _of(result, RequestEvent)}
    own = requests["run"].metadata
    assert (own["enqueued_mono_ns"], own["structured_output"]) == (T0 - 20, True)
    assert own["enqueued_wall_ns"] == T0 - 20 + WALL_OFFSET
    assert own["enqueued_wall_after_ns"] == T0 - 20 + WALL_OFFSET + 800
    other = requests["foreign"].metadata
    assert (other["enqueued_mono_ns"], other["structured_output"]) == (None, None)


def test_each_admission_of_a_reused_id_keeps_its_own_enqueue(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        _run(
            alias(OWN0, f"chatcmpl-{X0}", T0 - 30),
            enqueued(OWN0, T0 - 20),
            scheduled(0, T0, [member(OWN0, scheduled=8)]),
            completed(0, T0 + SECOND, [done(OWN0)]),
            terminal(OWN0, T0 + SECOND + 5),
            alias(OWN0, f"chatcmpl-{X0}", T0 + 2 * SECOND),
            enqueued(OWN0, T0 + 2 * SECOND + 10, structured_output=True),
            scheduled(1, T0 + 3 * SECOND, [member(OWN0, scheduled=8)]),
            completed(1, T0 + 4 * SECOND, [done(OWN0)]),
            heartbeat(T0 + 4 * SECOND + 1, 10),
        ),
    )

    enqueues = sorted(
        (r.metadata["enqueued_mono_ns"], r.metadata["structured_output"])
        for r in _of(result, RequestEvent)
    )
    assert enqueues == [(T0 - 20, False), (T0 + 2 * SECOND + 10, True)]


# ---------------------------------------------------------------- sources


def test_each_record_names_the_last_raw_record_it_needed(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        _run(
            alias(OWN0, f"chatcmpl-{X0}", T0 - 30),  # 1
            enqueued(OWN0, T0 - 20),  # 2
            scheduled(0, T0, [member(OWN0, scheduled=8)]),  # 3
            completed(0, T0 + SECOND, [done(OWN0)]),  # 4
            scheduled(1, T0 + SECOND + 10, [member(OWN0, scheduled=1)]),  # 5
            # Step 2's output never came; step 3's did, which makes 2 final.
            scheduled(2, T0 + SECOND + 20, [member(OWN0, scheduled=1)]),  # 6
            terminal(OWN0, T0 + SECOND + 30),  # 7: freed in step 1's update
            completed(1, T0 + SECOND + 40, [done(OWN0)]),  # 8
            scheduled(3, T0 + SECOND + 50, [member(OTHER, scheduled=1)]),  # 9
            completed(3, T0 + SECOND + 60, [done(OTHER)]),  # 10
            scheduled(4, T0 + SECOND + 70, [member(OWN0, scheduled=1)]),  # 11
            goodbye(T0 + 2 * SECOND, 12),  # 12: the epoch ends with step 4 open
        ),
    )

    iterations = {
        i.iteration_ref.id: i.metadata["source_seq_max"]
        for i in _of(result, IterationEvent)
    }
    assert iterations == {"0": 4, "1": 8, "2": 10, "4": 12}
    memberships = {
        m.iteration_ref.id: (m.metadata["epoch"], m.metadata["source_seq_max"])
        for m in _of(result, MembershipEvent)
    }
    # Step 1's membership carries the finish read at seq 7, before its output.
    assert memberships == {
        "0": (EPOCH, 4),
        "1": (EPOCH, 8),
        "2": (EPOCH, 10),
        "4": (EPOCH, 12),
    }
    (request,) = _of(result, RequestEvent)
    assert request.metadata["source_seq_max"] == 4  # its first final step
    (alignment,) = _of(result, ClockAlignmentEvent)
    # Known from the hello; valid until the goodbye, which is dated apart.
    assert alignment.metadata["source_seq_max"] == 0
    assert alignment.metadata["valid_to_source_seq"] == 12


def test_a_finish_read_after_its_step_raises_the_membership_s_source(
    tmp_path: Path,
) -> None:
    result = _reduce(
        tmp_path,
        _run(
            alias(OWN0, f"chatcmpl-{X0}", T0 - 30),  # 1
            scheduled(0, T0, [member(OWN0, scheduled=8)]),  # 2
            completed(0, T0 + SECOND, [done(OWN0)]),  # 3
            terminal(OWN0, T0 + 2 * SECOND),  # 4: aborted between steps
            heartbeat(T0 + 3 * SECOND, 4),  # 5
        ),
    )

    (membership,) = _of(result, MembershipEvent)
    assert membership.metadata["finish"]["in_step"] is False
    assert membership.metadata["source_seq_max"] == 4
    (alignment,) = _of(result, ClockAlignmentEvent)
    assert alignment.metadata["source_seq_max"] == 0  # the hello alone
    assert alignment.metadata["valid_to_source_seq"] is None


# ---------------------------------------------------------------- coverage


def _coverage(root: Path, records: list[dict[str, Any]], **kwargs: Any) -> Any:
    write_epoch(root, "engine", PID, START, records, **kwargs.pop("epoch", {}))
    (epoch,) = read_execution_log(root, importer=HERE, **kwargs).engines()
    return epoch.coverage()


def _spans(coverage: dict[str, Any]) -> list[tuple[int, int]]:
    return [(span["start_seq"], span["end_seq"]) for span in coverage["spans"]]


def test_coverage_is_where_heartbeats_show_nothing_lost(tmp_path: Path) -> None:
    observes = ["cache_reset", "enqueued", "pause"]
    coverage = _coverage(
        tmp_path,
        [
            hello("engine", PID, START, observes=observes),  # 0
            heartbeat(T0, 0),  # 1
            heartbeat(T0 + SECOND, 1),  # 2
            heartbeat(T0 + 2 * SECOND, 2),  # 3
            # A reset too large to queue: only its _oversized count rises.
            heartbeat(T0 + 3 * SECOND, 3, dropped={"cache_reset_oversized": 1}),  # 4
            heartbeat(T0 + 4 * SECOND, 4, dropped={"cache_reset_oversized": 1}),  # 5
            heartbeat(
                T0 + 5 * SECOND, 5, dropped={"cache_reset_oversized": 1}, errors=1
            ),
            heartbeat(
                T0 + 6 * SECOND, 6, dropped={"cache_reset_oversized": 1}, errors=1
            ),
            heartbeat(T0 + 7 * SECOND, 7, capped=True),  # 8
            heartbeat(T0 + 8 * SECOND, 8, capped=True),  # 9
        ],
    )

    assert coverage["observes"] == observes
    assert coverage["heartbeats"] == 9
    assert _spans(coverage) == [(1, 3), (4, 5), (6, 7)]
    first = coverage["spans"][0]
    assert (first["start_mono_ns"], first["end_mono_ns"]) == (T0, T0 + 2 * SECOND)
    assert first["start_wall_ns"] == T0 + WALL_OFFSET


def _backlog(
    *, read_to: int | None = None, dropped_after: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Heartbeat 2 was written ahead of two records accepted before its
    stamp (``pending``): they take seqs 3 and 4, then heartbeat 5."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, pending=0),  # 1
        heartbeat(T0 + SECOND, 1, pending=2),  # 2
        pause("UNPAUSED", "PAUSED_ALL", T0 + SECOND - 5),  # 3: accepted before 2
        pause("PAUSED_ALL", "UNPAUSED", T0 + SECOND - 4),  # 4
        heartbeat(
            T0 + 2 * SECOND,
            4,
            pending=0,
            dropped={"pause": 1} if dropped_after else {},
        ),  # 5
    ]
    return records[: read_to + 1] if read_to is not None else records, {}


def test_a_span_reaches_past_the_records_its_heartbeat_was_ahead_of(
    tmp_path: Path,
) -> None:
    records, _ = _backlog()
    coverage = _coverage(tmp_path, records)
    # 1 to 2 vouches for every record accepted by 2's stamp, through seq 4;
    # 2 to 5 joins it.
    assert _spans(coverage) == [(1, 5)]


def test_a_span_is_not_whole_until_its_pending_records_are_read(
    tmp_path: Path,
) -> None:
    """A live read that ends at heartbeat 2, before the pause it was ahead
    of, must not vouch that no pause happened before 2's stamp."""
    records, _ = _backlog(read_to=2)
    assert _spans(_coverage(tmp_path, records)) == []


def test_a_heartbeat_written_among_pending_records_does_not_take_their_place(
    tmp_path: Path,
) -> None:
    """Heartbeat 2 counts a pause reserved before its stamp, but the writer
    thread writes heartbeats 3 and 4 before the pause comes: the pause takes
    seq 5, not 3, so a live read that ends at heartbeat 4 has not read it
    and must not vouch that no pause happened before 2's stamp."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, pending=0),  # 1
        heartbeat(T0 + SECOND, 1, pending=1),  # 2
        heartbeat(T0 + 2 * SECOND, 2, pending=1),  # 3
        heartbeat(T0 + 3 * SECOND, 3, pending=1),  # 4
        pause("UNPAUSED", "PAUSED_ALL", T0 + SECOND - 5),  # 5: stamped before 2
        heartbeat(T0 + 4 * SECOND, 5, pending=0),  # 6
    ]

    assert _spans(_coverage(tmp_path / "live", records[:5])) == []
    assert _spans(_coverage(tmp_path / "whole", records)) == [(1, 6)]


def test_a_reserved_record_overtaken_by_another_thread_s_holds_the_span(
    tmp_path: Path,
) -> None:
    """The engine thread reserves a pause before heartbeat 2's stamp and is
    slow to emit it; meanwhile vLLM's input thread reserves and queues an
    alias, which takes seq 3. Heartbeat 2's one pending record is then the
    alias, not the pause: a live read through heartbeat 4 has read seq 3 but
    not the pause, so no span may vouch for 2's stamp until a heartbeat with
    nothing reserved bounds what is still to come."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, pending=0, reserved=0),  # 1
        heartbeat(T0 + SECOND, 1, pending=1, reserved=1),  # 2: the pause
        alias("a-1", "chatcmpl-a", T0 + SECOND + 5),  # 3: reserved after 2
        heartbeat(T0 + 2 * SECOND, 3, pending=1, reserved=1),  # 4
        pause("UNPAUSED", "PAUSED_ALL", T0 + SECOND - 5),  # 5: stamped before 2
        heartbeat(T0 + 3 * SECOND, 5, pending=0, reserved=0),  # 6
        heartbeat(T0 + 4 * SECOND, 6, pending=0, reserved=0),  # 7
    ]

    assert _spans(_coverage(tmp_path / "live", records[:5])) == []
    assert _spans(_coverage(tmp_path / "whole", records)) == [(1, 7)]


def test_several_reservations_hold_the_span_until_none_is_open(
    tmp_path: Path,
) -> None:
    """Heartbeat 2 counts two reserved records. One is queued and written
    (seq 3), an alias reserved later overtakes the other (seq 4), heartbeat
    5 still counts it reserved, and heartbeat 6 has it queued but not
    written: only heartbeat 6, the first with nothing reserved, bounds the
    span, through the record it is still ahead of (seq 7)."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, pending=0, reserved=0),  # 1
        heartbeat(T0 + SECOND, 1, pending=2, reserved=2),  # 2
        pause("UNPAUSED", "PAUSED_ALL", T0 + SECOND - 9),  # 3: reserved before 2
        alias("a-1", "chatcmpl-a", T0 + SECOND + 5),  # 4: reserved after 2
        heartbeat(T0 + 2 * SECOND, 4, pending=1, reserved=1),  # 5
        heartbeat(T0 + 3 * SECOND, 5, pending=1, reserved=0),  # 6
        pause("PAUSED_ALL", "UNPAUSED", T0 + SECOND - 5),  # 7: reserved before 2
        heartbeat(T0 + 4 * SECOND, 7, pending=0, reserved=0),  # 8
    ]

    for read_to in (6, 7):
        live = _coverage(tmp_path / f"live{read_to}", records[: read_to + 1])
        assert _spans(live) == []
    whole = _coverage(tmp_path / "whole", records)
    assert _spans(whole) == [(1, 8)]
    assert whole["held"] == {}


def test_a_hook_that_does_not_count_reservations_closes_no_live_span(
    tmp_path: Path,
) -> None:
    """A heartbeat without ``reserved`` (#217's hook, before reservations)
    cannot say whether its pending records keep their order, so it closes a
    span only on a read that holds every record the epoch wrote, never on
    a live or prefix read; the coverage block says why."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, queued=0),  # 1
        heartbeat(T0 + SECOND, 1, queued=0),  # 2
        heartbeat(T0 + 2 * SECOND, 2, queued=0),  # 3
    ]
    for record in records[1:]:
        del record["reserved"]

    live = _coverage(tmp_path / "live", records)
    assert _spans(live) == []
    assert live["held"] == {"reserved_unknown": 2}
    ended = _coverage(tmp_path / "ended", [*records, goodbye(T0 + 3 * SECOND, 3)])
    assert _spans(ended) == [(1, 3)]
    assert ended["held"] == {}


def test_a_span_reaches_past_every_heartbeat_among_its_pending_records(
    tmp_path: Path,
) -> None:
    """Heartbeats 3, 4 and 5 are all written before the pause heartbeat 2
    counts: a live read through 5, with 4 and 5 to witness, has still not
    read it, so no span may vouch for 2's stamp."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, pending=0),  # 1
        heartbeat(T0 + SECOND, 1, pending=1),  # 2
        heartbeat(T0 + 2 * SECOND, 2, pending=1),  # 3
        heartbeat(T0 + 3 * SECOND, 3, pending=1),  # 4
        heartbeat(T0 + 4 * SECOND, 4, pending=1),  # 5
        pause("UNPAUSED", "PAUSED_ALL", T0 + SECOND - 5),  # 6: accepted before 2
        heartbeat(T0 + 5 * SECOND, 6, pending=0),  # 7
    ]

    assert _spans(_coverage(tmp_path / "live", records[:6])) == []
    assert _spans(_coverage(tmp_path / "whole", records)) == [(1, 7)]


def test_a_pending_record_lost_after_heartbeats_between_breaks_the_span(
    tmp_path: Path,
) -> None:
    """Heartbeat 2 counts two pauses; heartbeats 3 and 5 are written among
    them, the first is written (seq 4) and the second is lost at write, so
    an alias written later takes the seq it was counted into. Only
    heartbeat 8, past that seq, shows the loss: an earlier witness, after 2
    plus its pending alone, would vouch for a span missing a pause."""
    records = [
        hello("engine", PID, START, observes=["pause"]),  # 0
        heartbeat(T0, 0, pending=0),  # 1
        heartbeat(T0 + SECOND, 1, pending=2),  # 2
        heartbeat(T0 + 2 * SECOND, 2, pending=2),  # 3
        pause("UNPAUSED", "PAUSED_NEW", T0 + SECOND - 9),  # 4: accepted before 2
        heartbeat(T0 + 3 * SECOND, 4, pending=1),  # 5
        heartbeat(T0 + 4 * SECOND, 5, pending=0, errors=1),  # 6: the second, lost
        alias("a-1", "chatcmpl-a", T0 + 4 * SECOND + 5),  # 7
        heartbeat(T0 + 5 * SECOND, 7, pending=0, errors=1),  # 8
    ]

    assert _spans(_coverage(tmp_path, records)) == [(6, 8)]


def test_a_pending_record_lost_at_write_breaks_the_span(tmp_path: Path) -> None:
    """A record lost while being written counts only in a later heartbeat."""
    records, _ = _backlog(dropped_after=True)
    assert _spans(_coverage(tmp_path, records)) == []


def test_coverage_needs_every_record_between_two_heartbeats(tmp_path: Path) -> None:
    records = [
        hello("engine", PID, START),  # 0: a hook that does not say what it observes
        heartbeat(T0, 0),  # 1
        alias(OWN0, f"chatcmpl-{X0}", T0 + 1),  # 2
        heartbeat(T0 + SECOND, 2),  # 3
        alias(OTHER, "chatcmpl-other", T0 + SECOND + 1),  # 4: not yet readable
        heartbeat(T0 + 2 * SECOND, 4),  # 5
    ]
    write_epoch(tmp_path, "engine", PID, START, records)
    part = next(tmp_path.glob("*/engine-*/000000.jsonl"))
    lines = part.read_text().splitlines(keepends=True)
    part.write_text("".join(lines[:4] + lines[5:]))  # seq 4 is missing

    (epoch,) = read_execution_log(tmp_path, importer=HERE).engines()
    coverage = epoch.coverage()
    assert coverage["observes"] is None
    assert _spans(coverage) == [(1, 3)]


def test_liveness_lists_the_stretches_no_heartbeat_was_heard(tmp_path: Path) -> None:
    # Seconds: a 2.3 s slip, as under load on a real run, then six silent
    # seconds from 5.3 s to 11.3 s.
    beats = [0.0, 1.0, 2.0, 4.3, 5.3, 11.3, 12.3]
    records = [hello("engine", PID, START)]
    records += [heartbeat(T0 + round(at * SECOND), n) for n, at in enumerate(beats)]
    write_epoch(tmp_path, "engine", PID, START, records)

    (epoch,) = read_execution_log(
        tmp_path, importer=HERE, high_water={EPOCH: 5}
    ).engines()
    liveness = epoch.summary()["liveness"]

    # Every heartbeat counts, those an earlier import consumed included.
    assert liveness["heartbeats"] == 7
    assert liveness["first"] == {"seq": 1, "mono_ns": T0, "wall_ns": T0 + WALL_OFFSET}
    assert liveness["last"] == {
        "seq": 7,
        "mono_ns": T0 + round(12.3 * SECOND),
        "wall_ns": T0 + round(12.3 * SECOND) + WALL_OFFSET,
    }
    # The slip is not a gap; the six silent seconds are.
    (gap,) = liveness["gaps"]
    assert (gap["start_mono_ns"], gap["end_mono_ns"]) == (
        T0 + round(5.3 * SECOND),
        T0 + round(11.3 * SECOND),
    )
    assert gap["start_wall_ns"] == T0 + round(5.3 * SECOND) + WALL_OFFSET
    assert liveness["gap_ns"] == 5 * SECOND
    assert liveness["max_interval_ns"] == 6 * SECOND


def test_an_epoch_without_heartbeats_has_no_liveness_bounds(tmp_path: Path) -> None:
    write_epoch(tmp_path, "engine", PID, START, [hello("engine", PID, START)])

    (epoch,) = read_execution_log(tmp_path, importer=HERE).engines()

    liveness = epoch.liveness()
    assert liveness["first"] is None and liveness["max_interval_ns"] is None
    assert liveness["gaps"] == []


def test_coverage_spans_what_earlier_imports_consumed(tmp_path: Path) -> None:
    records = [hello("engine", PID, START)]
    records += [heartbeat(T0 + n * SECOND, n) for n in range(4)]

    coverage = _coverage(tmp_path, records, high_water={EPOCH: 3})

    assert _spans(coverage) == [(1, 4)]


# ---------------------------------------------------------------- stages

REQUEST1 = "c1_in8_out4_measured_0_1"
X1 = f"stormlog-{RUN}-{REQUEST1}"
OWN1 = f"chatcmpl-{X1}-77aa00bb"
OBSERVES = ["cache_reset", "enqueued", "pause"]


def _two_requests(**changes: Any) -> RunFacts:
    return _facts(
        requests={
            X0: RunRequest(REQUEST0, X0, "c1_in8_out4", "measured"),
            X1: RunRequest(REQUEST1, X1, "c1_in8_out4", "measured"),
        },
        **changes,
    )


def _after(result: ReduceResult, **changes: Any) -> RunFacts:
    """The facts a later import reads back from an artifact holding ``result``."""
    graph = resolve_inference_events(result.events)
    return _two_requests(
        existing_iterations=frozenset(graph.iterations),
        existing_attempts=frozenset(
            r.attempt_ref for r in graph.requests.values() if r.attempt_ref is not None
        ),
        existing_alignments=frozenset(a.event_id for a in graph.alignments),
        existing_stages=frozenset(s.event_id for s in graph.stages.values()),
        **changes,
    )


def _admitted(*internals: str) -> list[dict[str, Any]]:
    """A hook that observes resets and pauses, with both run requests
    admitted and prefilled together in step 0."""
    return [
        hello("engine", PID, START, observes=OBSERVES),  # 0
        alias(OWN0, f"chatcmpl-{X0}", T0 - 30),  # 1
        alias(OWN1, f"chatcmpl-{X1}", T0 - 20),  # 2
        scheduled(0, T0, [member(name, scheduled=8) for name in internals]),  # 3
        completed(0, T0 + SECOND, [done(name) for name in internals]),  # 4
    ]


def _stages(result: ReduceResult) -> dict[str, list[StageEvent]]:
    found: dict[str, list[StageEvent]] = {}
    for stage in _of(result, StageEvent):
        found.setdefault(stage.name, []).append(stage)
    return found


def test_a_step_s_preemption_is_dated_by_its_schedule_call(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        [
            *_admitted(OWN0, OWN1),
            scheduled(
                1, T0 + SECOND + 10, [member(OWN0, scheduled=1)], preempted=[OWN1]
            ),
            completed(1, T0 + 2 * SECOND, [done(OWN0)]),  # 6
            heartbeat(T0 + 3 * SECOND, 6),
        ],
        _two_requests(),
    )

    (stage,) = _stages(result)["engine.preempted"]
    assert stage.iteration_ref is not None and stage.iteration_ref.id == "1"
    assert stage.request_ref == EntityRef("stormlog", REQUEST1)
    assert (stage.start_ns, stage.end_ns) == (T0 + SECOND + 10, T0 + SECOND + 200_010)
    metadata = stage.metadata
    assert (metadata["by"], metadata["reset_observed"]) == ("schedule", True)
    assert (metadata["epoch"], metadata["seq"], metadata["source_seq_max"]) == (
        EPOCH,
        5,
        6,
    )
    assert metadata["attempt"] == OWN1
    assert resolve_inference_events(result.events).unresolved == ()


def test_a_reset_s_preemptions_are_its_own_not_the_next_step_s(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        [
            *_admitted(OWN0, OWN1),
            cache_reset([OWN0, OWN1], T0 + SECOND + 5, succeeded=None),  # 5: raised
            # vLLM lists the reset's preemptions in the next step's output.
            scheduled(1, T0 + 2 * SECOND, [], preempted=[OWN0, OWN1]),  # 6
            completed(1, T0 + 2 * SECOND + 10, []),
            heartbeat(T0 + 3 * SECOND, 8),
        ],
        _two_requests(),
    )

    stages = _stages(result)
    assert "engine.preempted" not in stages
    # Step 1 preempted no run request itself, and ran none: it is empty.
    assert [i.iteration_ref.id for i in _of(result, IterationEvent)] == ["0"]
    (reset,) = stages["engine.cache_reset"]
    # Step 0 is the last one written before the reset; step 1 is empty.
    assert reset.iteration_ref == EntityRef(reset.context.producer_id, "0")
    assert (reset.metadata["succeeded"], reset.metadata["raised"]) == (None, True)
    assert (reset.start_ns, reset.end_ns) == (T0 + SECOND + 5, T0 + SECOND + 50_005)
    assert reset.metadata["start_wall_after_ns"] == T0 + SECOND + 5 + WALL_OFFSET + 800
    by_reset = stages["engine.preempted_by_reset"]
    assert sorted(s.request_ref.id for s in by_reset if s.request_ref) == [
        REQUEST0,
        REQUEST1,
    ]
    assert {s.metadata["reset_seq"] for s in by_reset} == {5}
    assert resolve_inference_events(result.events).unresolved == ()


def test_a_pause_with_no_step_before_it_is_a_dated_fact(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        [
            hello("engine", PID, START, observes=OBSERVES),
            pause("UNPAUSED", "PAUSED_ALL", T0 - 50),  # 1: nothing to refer to
            pause("PAUSED_ALL", "UNPAUSED", T0 - 40),  # 2
            alias(OWN0, f"chatcmpl-{X0}", T0 - 30),
            scheduled(0, T0, [member(OWN0, scheduled=8)]),
            completed(0, T0 + SECOND, [done(OWN0)]),
            pause("UNPAUSED", "PAUSED_NEW", T0 + SECOND + 5),  # 6
            heartbeat(T0 + 3 * SECOND, 6),
        ],
    )

    (stage,) = _stages(result)["engine.pause_transition"]
    assert (stage.metadata["from"], stage.metadata["to"]) == ("UNPAUSED", "PAUSED_NEW")
    assert stage.start_ns == stage.end_ns == T0 + SECOND + 5
    assert stage.iteration_ref is not None and stage.iteration_ref.id == "0"
    summary = result.summary["epochs"][EPOCH]
    assert [
        (f["seq"], f["from"], f["to"], f["start_mono_ns"])
        for f in summary["unanchored"]
    ] == [
        (1, "UNPAUSED", "PAUSED_ALL", T0 - 50),
        (2, "PAUSED_ALL", "UNPAUSED", T0 - 40),
    ]
    assert summary["stages"] == {"engine.pause_transition": 1, "unanchored": 2}


def test_other_clients_preemptions_follow_the_pseudonym_rules(tmp_path: Path) -> None:
    records = [
        *_admitted(OWN0),
        # Another client's request, in a step of its own that is not kept.
        alias(OTHER, "chatcmpl-other", T0 + SECOND + 1),
        scheduled(1, T0 + SECOND + 10, [member(OTHER, scheduled=8)]),
        completed(1, T0 + SECOND + 20, [done(OTHER)]),
        scheduled(2, T0 + 2 * SECOND, [member(OWN0, scheduled=1)], preempted=[OTHER]),
        completed(2, T0 + 2 * SECOND + 10, [done(OWN0)]),
        heartbeat(T0 + 3 * SECOND, 10),
    ]
    keyed = _reduce(tmp_path / "keyed", records)
    write_epoch(tmp_path / "keyless", "engine", PID, START, records, key=None)
    keyless = reduce_execution_log(
        read_execution_log(tmp_path / "keyless", importer=HERE), _facts()
    )

    # No record of the other request was written, so nothing can refer to it.
    assert _stages(keyed) == {}
    assert keyed.summary["epochs"][EPOCH]["stages"] == {"unreferenced": 1}
    assert keyless.summary["epochs"][EPOCH]["stages"] == {"withheld": 1}


def test_a_step_that_preempted_a_run_request_is_kept(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        [
            *_admitted(OWN0),
            alias(OTHER, "chatcmpl-other", T0 + SECOND + 1),
            # Only another client's request runs, outside every run window.
            scheduled(
                1, T0 + SECOND + 10, [member(OTHER, scheduled=8)], preempted=[OWN0]
            ),
            completed(1, T0 + SECOND + 20, [done(OTHER)]),
            heartbeat(T0 + 3 * SECOND, 8),
        ],
    )

    (stage,) = _stages(result)["engine.preempted"]
    assert stage.iteration_ref is not None and stage.iteration_ref.id == "1"
    assert "1" in [i.iteration_ref.id for i in _of(result, IterationEvent)]


def test_a_reset_waits_for_the_step_that_lists_its_preemptions(tmp_path: Path) -> None:
    first = [
        *_admitted(OWN0, OWN1),
        heartbeat(T0 + SECOND + 1, 4),  # 5
        cache_reset([OWN0, OWN1], T0 + SECOND + 5),  # 6
        heartbeat(T0 + 2 * SECOND, 6),  # 7
    ]
    rest = [
        scheduled(1, T0 + 2 * SECOND + 10, [], preempted=[OWN0, OWN1]),  # 8
        completed(1, T0 + 2 * SECOND + 20, []),  # 9
        heartbeat(T0 + 3 * SECOND, 9),  # 10
    ]
    before = _reduce(tmp_path, first, _two_requests())
    assert before.high_water == {EPOCH: 5}  # read again from the reset
    write_epoch(tmp_path, "engine", PID, START, [*first, *rest])
    after = reduce_execution_log(
        read_execution_log(tmp_path, importer=HERE, high_water=before.high_water),
        _after(before),
    )

    assert sorted(_stages(before)) == [
        "engine.cache_reset",
        "engine.preempted_by_reset",
    ]
    assert _stages(after) == {}  # the reset's stages are not written twice
    assert after.summary["epochs"][EPOCH]["stages"] == {"already_imported": 3}
    assert after.high_water == {EPOCH: 10}


def test_a_reset_waits_until_the_step_that_lists_it_is_final(tmp_path: Path) -> None:
    """Under async scheduling the log's tail nearly always holds a pending
    step. Read with the step after a reset while that step's output is not
    in, the import still holds the mark at the reset, so the next import
    reads the step with it and does not call the reset's preemptions the
    step's own."""
    resumed = [
        member(name, scheduled=8, sighting="repeat", phase="context")
        for name in (OWN0, OWN1)
    ]
    first = [
        *_admitted(OWN0, OWN1),  # 0..4
        cache_reset([OWN0, OWN1], T0 + SECOND + 5),  # 5
        scheduled(1, T0 + 2 * SECOND, resumed, preempted=[OWN0, OWN1]),  # 6
        heartbeat(T0 + 2 * SECOND + 5, 6),  # 7
    ]
    rest = [
        completed(1, T0 + 3 * SECOND, [done(OWN0), done(OWN1)]),  # 8
        heartbeat(T0 + 4 * SECOND, 8),  # 9
    ]
    before = _reduce(tmp_path, first, _two_requests())
    assert before.high_water == {EPOCH: 4}  # read again from the reset
    write_epoch(tmp_path, "engine", PID, START, [*first, *rest])
    after = reduce_execution_log(
        read_execution_log(tmp_path, importer=HERE, high_water=before.high_water),
        _after(before),
    )

    names = [
        stage.name
        for result in (before, after)
        for stage in [s for v in _stages(result).values() for s in v]
        if stage.name in ("engine.preempted", "engine.preempted_by_reset")
    ]
    # Each request was preempted once, by the reset.
    assert sorted(names) == ["engine.preempted_by_reset"] * 2


def test_a_reset_with_nothing_running_holds_no_mark(tmp_path: Path) -> None:
    """vLLM's default pause aborts every request, then resets the prefix
    cache with reset_running_requests and nothing running: no step will list
    its preemptions, so nothing waits for one."""
    records = [
        *_admitted(OWN0, OWN1),  # 0..4
        pause("UNPAUSED", "PAUSED_NEW", T0 + SECOND + 1),  # 5
        cache_reset([], T0 + SECOND + 5),  # 6
        pause("PAUSED_NEW", "UNPAUSED", T0 + SECOND + 9),  # 7
        scheduled(1, T0 + 2 * SECOND, [member(OWN0, scheduled=1, sighting="repeat")]),
        completed(1, T0 + 2 * SECOND + 10, [done(OWN0)]),  # 9
        scheduled(2, T0 + 3 * SECOND, [member(OWN0, scheduled=1, sighting="repeat")]),
        completed(2, T0 + 3 * SECOND + 10, [done(OWN0)]),  # 11
        heartbeat(T0 + 4 * SECOND, 11),  # 12
    ]

    result = _reduce(tmp_path, records, _two_requests())

    assert result.high_water == {EPOCH: 12}


def test_a_pause_s_empty_reset_holds_no_mark_while_nothing_is_scheduled(
    tmp_path: Path,
) -> None:
    """The default pause schedules nothing until it resumes, so no step
    closes a hold: an empty reset must not open one, or every import of the
    paused engine holds its mark below the reset."""
    records = [
        *_admitted(OWN0, OWN1),  # 0..4
        pause("UNPAUSED", "PAUSED_ALL", T0 + SECOND + 1),  # 5
        cache_reset([], T0 + SECOND + 5),  # 6
        heartbeat(T0 + 2 * SECOND, 6),  # 7
    ]

    result = _reduce(tmp_path, records, _two_requests())

    assert result.high_water == {EPOCH: 7}


def test_a_reset_s_preemption_waits_for_its_request_record(tmp_path: Path) -> None:
    first = [
        hello("engine", PID, START, observes=OBSERVES),  # 0
        alias(OWN0, f"chatcmpl-{X0}", T0 - 30),  # 1
        scheduled(0, T0, [member(OWN0, scheduled=8)]),  # 2: its output is not in
        cache_reset([OWN0], T0 + 5),  # 3
        scheduled(1, T0 + 10, [], preempted=[OWN0]),  # 4
        heartbeat(T0 + 20, 4),  # 5
    ]
    rest = [
        completed(0, T0 + SECOND, [done(OWN0)]),  # 6
        completed(1, T0 + SECOND + 10, []),  # 7
        heartbeat(T0 + 2 * SECOND, 7),  # 8
    ]
    before = _reduce(tmp_path, first)
    write_epoch(tmp_path, "engine", PID, START, [*first, *rest])
    after = reduce_execution_log(
        read_execution_log(tmp_path, importer=HERE, high_water=before.high_water),
        _after(before),
    )

    assert _stages(before) == {}
    assert before.high_water == {EPOCH: 0}
    assert before.summary["epochs"][EPOCH]["unanchored"][0]["seq"] == 3
    (stage,) = _stages(after)["engine.preempted_by_reset"]
    assert stage.request_ref == EntityRef("stormlog", REQUEST0)
    # Step 0 completed after the reset, so the reset itself has no step to
    # refer to: it stays a dated fact, and its preemption refers to the request.
    assert stage.iteration_ref is None
    assert after.summary["epochs"][EPOCH]["unanchored"][0]["seq"] == 3


def test_a_reset_stage_needs_the_records_of_its_request(tmp_path: Path) -> None:
    """A reset's preemption is written only with its request's record, which
    its first final step writes; that step completes after the reset, so the
    stage's source_seq_max is that completion: a prefix import that reaches
    it writes the stage."""
    records = [
        hello("engine", PID, START, observes=OBSERVES),  # 0
        alias(OWN0, f"chatcmpl-{X0}", T0 - 30),  # 1
        scheduled(0, T0, [member(OWN0, scheduled=8)]),  # 2: output not in yet
        cache_reset([OWN0], T0 + 5),  # 3
        scheduled(1, T0 + 10, [], preempted=[OWN0]),  # 4
        completed(0, T0 + SECOND, [done(OWN0)]),  # 5
        completed(1, T0 + SECOND + 10, []),  # 6
        heartbeat(T0 + 2 * SECOND, 6),  # 7
    ]
    full = _reduce(tmp_path / "full", records, _two_requests())
    (stage,) = _stages(full)["engine.preempted_by_reset"]
    source = stage.metadata["source_seq_max"]
    assert source == 5

    prefix = _reduce(tmp_path / "prefix", records[: source + 1], _two_requests())
    assert stage.event_id in {
        event.event_id for events in _stages(prefix).values() for event in events
    }


def test_a_step_that_resumes_a_reset_s_requests_keeps_none_of_its_preemptions(
    tmp_path: Path,
) -> None:
    result = _reduce(
        tmp_path,
        [
            *_admitted(OWN0, OWN1),
            cache_reset([OWN0, OWN1], T0 + SECOND + 5),  # 5
            # A second reset finds nothing running: the first preempted all.
            cache_reset([], T0 + SECOND + 7),  # 6
            # The next step lists both and resumes one; it is kept for it.
            scheduled(
                1,
                T0 + 2 * SECOND,
                [member(OWN0, scheduled=8)],
                preempted=[OWN0, OWN1],
            ),
            completed(1, T0 + 2 * SECOND + 10, [done(OWN0)]),  # 8
            heartbeat(T0 + 3 * SECOND, 8),
        ],
        _two_requests(),
    )

    stages = _stages(result)
    assert "1" in [i.iteration_ref.id for i in _of(result, IterationEvent)]
    assert "engine.preempted" not in stages
    by_reset = stages["engine.preempted_by_reset"]
    assert sorted(
        (s.request_ref.id, s.metadata["reset_seq"]) for s in by_reset if s.request_ref
    ) == [(REQUEST0, 5), (REQUEST1, 5)]


def test_a_reset_that_keeps_running_requests_preempts_none(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        [
            *_admitted(OWN0, OWN1),
            cache_reset(
                [OWN0, OWN1], T0 + SECOND + 5, reset_running_requests=False
            ),  # 5
            scheduled(
                1, T0 + 2 * SECOND, [member(OWN0, scheduled=1)], preempted=[OWN1]
            ),
            completed(1, T0 + 2 * SECOND + 10, [done(OWN0)]),  # 7
            heartbeat(T0 + 3 * SECOND, 7),
        ],
        _two_requests(),
    )

    stages = _stages(result)
    assert "engine.preempted_by_reset" not in stages
    (reset,) = stages["engine.cache_reset"]
    assert reset.metadata["reset_running_requests"] is False
    # The step's preemption is its own.
    (stage,) = stages["engine.preempted"]
    assert stage.request_ref == EntityRef("stormlog", REQUEST1)


def test_without_resets_observed_a_step_s_preemptions_say_so(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path,
        [
            hello("engine", PID, START),  # 0: a hook that does not list resets
            alias(OWN0, f"chatcmpl-{X0}", T0 - 30),
            alias(OWN1, f"chatcmpl-{X1}", T0 - 20),
            scheduled(0, T0, [member(OWN0, scheduled=8), member(OWN1, scheduled=8)]),
            completed(0, T0 + SECOND, [done(OWN0), done(OWN1)]),
            scheduled(
                1, T0 + SECOND + 10, [member(OWN0, scheduled=1)], preempted=[OWN1]
            ),
            completed(1, T0 + 2 * SECOND, [done(OWN0)]),  # 6
            heartbeat(T0 + 3 * SECOND, 6),
        ],
        _two_requests(),
    )

    (stage,) = _stages(result)["engine.preempted"]
    assert stage.metadata["reset_observed"] is False


def test_a_reset_waits_for_another_client_s_request_a_pending_step_writes(
    tmp_path: Path,
) -> None:
    """Another client's request whose only final step was not kept has no
    record yet, and no admission holds the mark for it: only the pending
    step that will write its record does, so the reset waits for it."""
    first = [
        hello("engine", PID, START, observes=OBSERVES),  # 0
        alias(OWN0, f"chatcmpl-{X0}", T0 - 30),  # 1
        alias(OTHER, "chatcmpl-other", T0 - 20),  # 2
        scheduled(0, T0, [member(OTHER, scheduled=8)]),  # 3: not kept
        completed(0, T0 + SECOND, [done(OTHER)]),  # 4
        cache_reset([OTHER], T0 + SECOND + 5),  # 5
        scheduled(
            1, T0 + SECOND + 10, [member(OWN0, scheduled=8)], preempted=[OTHER]
        ),  # 6
        completed(1, T0 + 2 * SECOND, [done(OWN0)]),  # 7
        # The other request resumes beside the run's, in a step still pending.
        scheduled(
            2,
            T0 + 2 * SECOND + 10,
            [member(OWN0, scheduled=1), member(OTHER, scheduled=8)],
        ),  # 8
        heartbeat(T0 + 3 * SECOND, 8),  # 9
    ]
    rest = [
        completed(2, T0 + 3 * SECOND + 10, [done(OWN0), done(OTHER)]),  # 10
        heartbeat(T0 + 4 * SECOND, 10),  # 11
    ]
    before = _reduce(tmp_path, first)
    write_epoch(tmp_path, "engine", PID, START, [*first, *rest])
    after = reduce_execution_log(
        read_execution_log(tmp_path, importer=HERE, high_water=before.high_water),
        _after(before),
    )

    assert "engine.preempted_by_reset" not in _stages(before)
    assert before.high_water == {EPOCH: 4}  # read again from the reset
    (stage,) = _stages(after)["engine.preempted_by_reset"]
    assert stage.metadata["reset_seq"] == 5
    assert stage.metadata["ownership"] == "foreign"


# ---------------------------------------------------------------- brackets


def test_records_keep_each_stamp_s_second_wall_read(tmp_path: Path) -> None:
    step = scheduled(0, T0, [member(OWN0, scheduled=8)])
    step.update(stamp(T0, "start_"), **stamp(T0 + 200_000, "end_"))
    result = _reduce(
        tmp_path,
        _run(
            {**alias(OWN0, f"chatcmpl-{X0}", T0 - 30), **stamp(T0 - 30)},
            step,
            {**terminal(OWN0, T0 + SECOND - 5), **stamp(T0 + SECOND - 5)},
            {**completed(0, T0 + SECOND, [done(OWN0)]), **stamp(T0 + SECOND)},
            heartbeat(T0 + 2 * SECOND, 5),
        ),
    )

    (iteration,) = _of(result, IterationEvent)
    after = WALL_OFFSET + 800
    assert iteration.metadata["start_wall_after_ns"] == T0 + after
    assert iteration.metadata["schedule_end_wall_after_ns"] == T0 + 200_000 + after
    assert iteration.metadata["completed_wall_after_ns"] == T0 + SECOND + after
    (request,) = _of(result, RequestEvent)
    assert request.metadata["admitted_wall_after_ns"] == T0 - 30 + after
    (membership,) = _of(result, MembershipEvent)
    assert membership.metadata["finish"]["wall_after_ns"] == T0 + SECOND - 5 + after
