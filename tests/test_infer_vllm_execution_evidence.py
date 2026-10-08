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
    assert alignment.metadata["source_seq_max"] == 12  # valid until the goodbye


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
