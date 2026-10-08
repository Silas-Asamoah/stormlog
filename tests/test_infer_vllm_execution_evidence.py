"""What the execution import derives from the hook's later records: the
clock alignment's bracket, enqueue times, source sequences, loss coverage,
and dated stages for preemptions, cache resets and pauses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stormlog.infer.correlation_accounting import (
    align_timestamp,
    alignment_offset_bounds,
)
from stormlog.infer.correlation_events import (
    ClockAlignmentEvent,
    CorrelationContext,
    CorrelationEvent,
    IterationEvent,
    MembershipEvent,
    RequestEvent,
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
    completed,
    done,
    enqueued,
    goodbye,
    heartbeat,
    hello,
    importer,
    member,
    scheduled,
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
