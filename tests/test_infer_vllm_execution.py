"""Reducing the execution hook's raw log into canonical records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stormlog.infer.correlation_accounting import resolve_inference_events
from stormlog.infer.correlation_events import (
    ClockAlignmentEvent,
    CorrelationEvent,
    EntityRef,
    IterationEvent,
    MembershipEvent,
    RequestEvent,
)
from stormlog.infer.vllm_execution import (
    FOREIGN,
    OWN,
    REASON_COMPLETED_MISSING,
    REASON_EPOCH_ENDED,
    UNRESOLVED,
    ReduceOptions,
    ReduceResult,
    RunFacts,
    RunRequest,
    Window,
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
    engine_log,
    failed,
    goodbye,
    heartbeat,
    member,
    producer,
    scheduled,
    terminal,
)

PID, START = 2600, 1_790_000_000_000_000_000
EPOCH = f"engine-{PID}-{START}"
PRODUCER = producer(PID, START)
T0 = 1_000 * SECOND
NOW = T0 + WALL_OFFSET + 5 * SECOND
RUN = "run-1"
REQUEST0 = "c1_in8_out4_measured_0_0"
REQUEST1 = "c1_in8_out4_measured_0_1"
X0 = f"stormlog-{RUN}-{REQUEST0}"
X1 = f"stormlog-{RUN}-{REQUEST1}"
OWN0 = f"chatcmpl-{X0}-0f3a9c1d"
OWN1 = f"chatcmpl-{X1}-77aa00bb"
CHILD = f"1_cmpl-{X1}-3-11223344"
OTHER = "chatcmpl-stormlog-run-9-c1_x_0-deadbeef"


def _facts(**changes: Any) -> RunFacts:
    requests = {
        X0: RunRequest(REQUEST0, X0, "c1_in8_out4", "measured"),
        X1: RunRequest(REQUEST1, X1, "c1_in8_out4", "measured"),
    }
    values: dict[str, Any] = {
        "run_id": RUN,
        "session_id": "session-1",
        "client_clock_domain": f"{HOST}/{BOOT}/unix_epoch_ns",
        "requests": requests,
    }
    values.update(changes)
    return RunFacts(**values)


def _facts_after(result: ReduceResult, **changes: Any) -> RunFacts:
    """The facts a later import reads back from an artifact holding ``result``."""
    graph = resolve_inference_events(result.events)
    return _facts(
        existing_iterations=frozenset(graph.iterations),
        existing_attempts=frozenset(
            r.attempt_ref for r in graph.requests.values() if r.attempt_ref is not None
        ),
        existing_alignments=frozenset(a.event_id for a in graph.alignments),
        **changes,
    )


def _two_shared_steps() -> list[dict[str, Any]]:
    """Two run requests prefilled in step 0, decoded in step 1; one finishes."""
    return [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OWN1, f"chatcmpl-{X1}", T0 - 5),
        scheduled(
            0,
            T0,
            [member(OWN0, scheduled=8, prompt_tokens=8), member(OWN1, scheduled=8)],
        ),
        completed(
            0,
            T0 + SECOND,
            [done(OWN0, computed_after=8), done(OWN1, computed_after=8)],
        ),
        scheduled(
            1,
            T0 + SECOND + 10,
            [
                member(OWN0, scheduled=1, computed_before=8, sighting="repeat"),
                member(OWN1, scheduled=1, computed_before=8, sighting="repeat"),
            ],
        ),
        terminal(OWN0, T0 + 2 * SECOND - 5, output_tokens=2),
        completed(
            1,
            T0 + 2 * SECOND,
            [done(OWN0, computed_after=9), done(OWN1, computed_after=9)],
        ),
        heartbeat(T0 + 3 * SECOND, 7),
    ]


def _reduce(root: Path, facts: RunFacts | None = None, **kwargs: Any) -> ReduceResult:
    read = read_execution_log(root, now_ns=NOW, **kwargs)
    return reduce_execution_log(read, facts or _facts())


def _by_type(events: list[CorrelationEvent]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for event in events:
        out.setdefault(event.EVENT_TYPE, []).append(event)
    return out


def _memberships(result: ReduceResult) -> list[MembershipEvent]:
    return _by_type(result.events).get("infer.membership", [])


def _attempt(event: MembershipEvent | RequestEvent) -> str:
    assert event.attempt_ref is not None
    return event.attempt_ref.id


def _requests(result: ReduceResult) -> dict[str, RequestEvent]:
    requests: list[RequestEvent] = _by_type(result.events).get("infer.request", [])
    return {r.attempt_ref.id: r for r in requests if r.attempt_ref is not None}


def _iteration_ids(result: ReduceResult) -> list[str]:
    return [
        i.iteration_ref.id for i in _by_type(result.events).get("infer.iteration", [])
    ]


def test_two_requests_sharing_two_steps_become_canonical_records(
    tmp_path: Path,
) -> None:
    engine_log(tmp_path, _two_shared_steps())
    result = _reduce(tmp_path)
    by_type = _by_type(result.events)
    iterations: list[IterationEvent] = by_type["infer.iteration"]
    assert [i.iteration_ref for i in iterations] == [
        EntityRef(PRODUCER, "0"),
        EntityRef(PRODUCER, "1"),
    ]
    first = iterations[0]
    assert first.context.clock_domain == f"{HOST}/{BOOT}/monotonic_ns"
    assert first.context.clock_kind == "monotonic"
    assert first.context.collection_mode == "imported"
    assert (first.context.host, first.context.pid) == (HOST, PID)
    assert (first.start_ns, first.end_ns) == (T0, T0 + SECOND)
    assert first.elapsed_ns == SECOND
    assert first.metadata["scheduler_residence_ns"] == SECOND
    assert first.metadata["state"] == "complete"
    assert first.metadata["run_members"] == 2 and first.metadata["total_tokens"] == 16
    memberships = _memberships(result)
    assert len(memberships) == 4
    prefill = [m for m in memberships if m.iteration_ref.id == "0"]
    assert {m.role for m in prefill} == {"prefill"}
    assert {m.request_ref for m in prefill} == {
        EntityRef("stormlog", REQUEST0),
        EntityRef("stormlog", REQUEST1),
    }
    assert {m.attempt_ref for m in prefill} == {
        EntityRef(PRODUCER, OWN0),
        EntityRef(PRODUCER, OWN1),
    }
    assert all(m.input_tokens == 8 and m.output_tokens == 1 for m in prefill)
    assert all(m.metadata["processed_prefill"] == 8 for m in prefill)
    decode = {_attempt(m): m for m in memberships if m.iteration_ref.id == "1"}
    assert {m.role for m in decode.values()} == {"decode"}
    # The request freed during step 1 carries its finish on that membership.
    finish = decode[OWN0].metadata["finish"]
    assert (finish["finish_reason"], finish["output_tokens"]) == ("stop", 2)
    assert finish["in_step"] is True
    assert "finish" not in decode[OWN1].metadata
    requests = _requests(result)
    assert set(requests) == {OWN0, OWN1}
    own0 = requests[OWN0]
    assert own0.request_ref == EntityRef("stormlog", REQUEST0)
    assert own0.backend_request_ref == EntityRef("vllm", OWN0)
    assert own0.start_ns == T0 - 10 and own0.input_tokens == 8
    assert own0.metadata["x_request_id"] == X0 and own0.metadata["bound_via"] == "alias"
    assert own0.end_ns is None  # admission facts only: the record never changes
    (alignment,) = by_type["infer.clock_alignment"]
    assert isinstance(alignment, ClockAlignmentEvent)
    assert alignment.from_clock_domain == f"{HOST}/{BOOT}/monotonic_ns"
    assert alignment.to_clock_domain == f"{HOST}/{BOOT}/unix_epoch_ns"
    assert alignment.offset_ns == WALL_OFFSET
    assert alignment.uncertainty_ns == 600 and alignment.valid_from_ns == T0
    assert alignment.valid_to_ns is None
    # Everything resolves through the existing correlation graph.
    graph = resolve_inference_events(result.events)
    assert graph.unresolved == ()
    assert len(graph.memberships) == 4
    epoch = result.summary["epochs"][EPOCH]
    assert (epoch["iterations_kept"], epoch["iterations_pending"]) == (2, 0)
    assert epoch["executions"] == {OWN: 2}
    assert result.high_water == {EPOCH: 8}


def test_a_pending_iteration_waits_and_holds_the_high_water_back(
    tmp_path: Path,
) -> None:
    records = _two_shared_steps()
    # Step 2 is scheduled but its output has not been processed; the epoch is
    # alive, so it waits. The terminal in the pending step waits with it.
    records += [
        scheduled(2, T0 + 3 * SECOND, [member(OWN1, scheduled=1, sighting="repeat")]),
        terminal(OWN1, T0 + 3 * SECOND + 5, output_tokens=3),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    assert _iteration_ids(result) == ["0", "1"]
    assert all(
        "finish" not in m.metadata for m in _memberships(result) if _attempt(m) == OWN1
    )
    epoch = result.summary["epochs"][EPOCH]
    assert epoch["iterations_pending"] == 1 and epoch["finish_unattached"] == 0
    # seq 9 is the pending scheduled record: the mark stops just before it.
    assert result.high_water == {EPOCH: 8}


def test_an_admission_no_step_has_shown_holds_the_high_water_back(
    tmp_path: Path,
) -> None:
    records = _two_shared_steps() + [
        alias("chatcmpl-x-00000000", "chatcmpl-x", T0 + 3 * SECOND)
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    assert result.high_water == {EPOCH: 8}  # the alias is seq 9
    assert _requests(result).keys() == {OWN0, OWN1}


def test_an_ended_epoch_makes_a_pending_iteration_incomplete(tmp_path: Path) -> None:
    records = _two_shared_steps() + [
        scheduled(2, T0 + 3 * SECOND, [member(OWN1, scheduled=1, sighting="repeat")]),
        alias("chatcmpl-late-00000000", "chatcmpl-late", T0 + 3 * SECOND + 1),
        goodbye(T0 + 4 * SECOND, 10),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    iterations = {
        i.iteration_ref.id: i for i in _by_type(result.events)["infer.iteration"]
    }
    assert iterations["2"].metadata["state"] == "incomplete"
    assert iterations["2"].metadata["incomplete_reason"] == REASON_EPOCH_ENDED
    assert iterations["2"].end_ns is None and iterations["2"].elapsed_ns is None
    (membership,) = [m for m in _memberships(result) if m.iteration_ref.id == "2"]
    assert membership.metadata["outcome"] == "unknown"
    assert (
        membership.output_tokens is None
        and membership.metadata["state"] == "incomplete"
    )
    (alignment,) = _by_type(result.events)["infer.clock_alignment"]
    assert alignment.valid_to_ns == T0 + 4 * SECOND
    epoch = result.summary["epochs"][EPOCH]
    assert (epoch["state"], epoch["iterations_incomplete"]) == ("ended", 1)
    # Nothing more can come: the mark covers the whole epoch, late alias included.
    assert result.high_water == {EPOCH: 11}


def test_a_step_whose_output_went_missing_is_final_once_a_later_one_completed(
    tmp_path: Path,
) -> None:
    records = _two_shared_steps() + [
        scheduled(2, T0 + 3 * SECOND, [member(OWN1, scheduled=1, sighting="repeat")]),
        scheduled(3, T0 + 4 * SECOND, [member(OWN1, scheduled=1, sighting="repeat")]),
        completed(3, T0 + 4 * SECOND + 10, [done(OWN1)]),  # 2's completed was dropped
        heartbeat(
            T0 + 4 * SECOND + 20,
            11,
            dropped={"scheduled": 0, "completed": 1, "alias": 0, "terminal": 0},
        ),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    iterations = {
        i.iteration_ref.id: i for i in _by_type(result.events)["infer.iteration"]
    }
    assert set(iterations) == {"0", "1", "2", "3"}
    assert iterations["2"].metadata["incomplete_reason"] == REASON_COMPLETED_MISSING
    assert iterations["3"].metadata["state"] == "complete"
    assert result.high_water == {EPOCH: 12}


def test_a_re_import_emits_nothing_the_artifact_already_holds(tmp_path: Path) -> None:
    engine_log(tmp_path, _two_shared_steps())
    first = _reduce(tmp_path)
    again = _reduce(tmp_path, _facts_after(first), high_water=first.high_water)
    assert again.events == []
    assert again.summary["epochs"][EPOCH]["high_water_seq"] == 8
    # Without the mark, the entities are still recognised and skipped.
    skipped = _reduce(tmp_path, _facts_after(first))
    assert skipped.events == []
    assert skipped.summary["epochs"][EPOCH]["iterations_already_imported"] == 2
    # And the same content reduced twice gives identical records.
    repeat = _reduce(tmp_path)
    assert [e.to_record() for e in repeat.events] == [
        e.to_record() for e in first.events
    ]


def test_a_later_import_continues_an_execution_it_already_knows(
    tmp_path: Path,
) -> None:
    engine_log(tmp_path, _two_shared_steps())
    first = _reduce(tmp_path)
    # Step 2 arrives later with OWN1 as a repeat member, and its terminal after
    # the step: no new alias, so it is the same execution, and the request
    # record is not written again.
    late = [
        scheduled(2, T0 + 3 * SECOND, [member(OWN1, scheduled=1, sighting="repeat")]),
        completed(2, T0 + 3 * SECOND + 10, [done(OWN1)]),
        terminal(OWN1, T0 + 3 * SECOND + 20, finish_reason="length", output_tokens=4),
        heartbeat(T0 + 4 * SECOND, 11),
    ]
    engine_log(tmp_path, _two_shared_steps() + late)
    second = _reduce(tmp_path, _facts_after(first), high_water=first.high_water)
    assert _iteration_ids(second) == ["2"]
    assert _requests(second) == {}
    (membership,) = _memberships(second)
    assert membership.attempt_ref == EntityRef(PRODUCER, OWN1)
    assert membership.request_ref == EntityRef("stormlog", REQUEST1)
    assert membership.metadata["finish"]["finish_reason"] == "length"
    assert membership.metadata["finish"]["in_step"] is False
    assert second.high_water == {EPOCH: 12}
    # Both imports together still resolve as one graph.
    assert resolve_inference_events(first.events + second.events).unresolved == ()


def test_binding_is_exact_and_other_ids_are_foreign_or_unresolved(
    tmp_path: Path,
) -> None:
    unparsable = "weird-id-7"
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 9),
        scheduled(
            0,
            T0,
            [
                member(OWN0, scheduled=8),
                member(OTHER, scheduled=8),
                member(CHILD, scheduled=4),  # no alias: the ID proposes a candidate
                member(unparsable, scheduled=2),
            ],
        ),
        completed(
            0, T0 + SECOND, [done(OWN0), done(OTHER), done(CHILD), done(unparsable)]
        ),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    requests = _requests(result)
    assert requests[OWN0].metadata["ownership"] == OWN
    child = requests[CHILD]
    assert (
        child.metadata["ownership"] == OWN and child.metadata["bound_via"] == "internal"
    )
    assert (child.metadata["child_index"], child.metadata["completion_index"]) == (1, 3)
    assert child.request_ref == EntityRef("stormlog", REQUEST1)
    assert child.metadata["admission_seen"] is False and child.start_ns is None
    others = [r for key, r in requests.items() if key not in (OWN0, CHILD)]
    assert {r.metadata["ownership"] for r in others} == {FOREIGN, UNRESOLVED}
    # Foreign and unresolved executions appear only under a keyed pseudonym.
    for record in others:
        assert record.attempt_ref is not None and len(record.attempt_ref.id) == 16
        int(record.attempt_ref.id, 16)
        text = str(record.to_record())
        assert OTHER not in text and unparsable not in text
        assert record.request_ref.producer_id == record.metadata["ownership"]
        assert record.backend_request_ref is None
    assert resolve_inference_events(result.events).unresolved == ()
    assert result.summary["epochs"][EPOCH]["executions"] == {
        OWN: 2,
        FOREIGN: 1,
        UNRESOLVED: 1,
    }


def test_an_alias_that_looks_like_the_run_but_is_not_stays_foreign(
    tmp_path: Path,
) -> None:
    # The alias is authoritative: an internal ID shaped like a run request
    # whose alias names another external ID is not ours.
    records = [
        alias(OWN0, "chatcmpl-someone-else", T0 - 10),
        scheduled(0, T0, [member(OWN0, scheduled=8), member(OWN1, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0), done(OWN1)]),
    ]
    engine_log(tmp_path, records)
    requests = _requests(_reduce(tmp_path))
    assert {r.metadata["ownership"] for r in requests.values()} == {FOREIGN, OWN}
    assert requests[OWN1].metadata["bound_via"] == "internal"


def test_raw_foreign_ids_are_an_opt_in(tmp_path: Path) -> None:
    records = [
        alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 9),
        scheduled(0, T0, [member(OWN0, scheduled=8), member(OTHER, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0), done(OTHER)]),
    ]
    engine_log(tmp_path, records)
    read = read_execution_log(tmp_path, now_ns=NOW)
    hidden = reduce_execution_log(read, _facts())
    shown = reduce_execution_log(read, _facts(), ReduceOptions(raw_foreign_ids=True))
    assert OTHER not in str([e.to_record() for e in hidden.events])
    assert _requests(shown)[OTHER].metadata["ownership"] == FOREIGN
    assert _requests(shown)[OTHER].request_ref == EntityRef(FOREIGN, OTHER)


def test_pseudonyms_are_keyed_by_the_epoch_key_and_the_run(tmp_path: Path) -> None:
    records = [
        alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 9),
        scheduled(0, T0, [member(OWN0, scheduled=8), member(OTHER, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0), done(OTHER)]),
    ]
    engine_log(tmp_path, records)
    other_root = tmp_path / "other"
    engine_log(other_root, records, key=bytes(32))
    foreign = lambda result: next(  # noqa: E731
        _attempt(r)
        for r in _requests(result).values()
        if r.metadata["ownership"] == FOREIGN
    )
    keyed = foreign(_reduce(tmp_path))
    assert keyed == foreign(_reduce(tmp_path))  # stable
    assert keyed != foreign(_reduce(tmp_path, _facts(run_id="run-2")))
    assert keyed != foreign(_reduce(other_root))
    missing = tmp_path / "missing"
    engine_log(missing, records, key=None)
    unkeyed = _reduce(missing)
    assert unkeyed.summary["epochs"][EPOCH]["pseudonym_key"] == "run_id_only"
    assert _reduce(tmp_path).summary["epochs"][EPOCH]["pseudonym_key"] == "epoch"


def test_foreign_only_iterations_are_kept_only_when_placed(tmp_path: Path) -> None:
    records = [
        alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 9),
        scheduled(0, T0, [member(OTHER, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OTHER)]),
        scheduled(1, T0 + 50 * SECOND, [member(OTHER, scheduled=1, sighting="repeat")]),
        completed(1, T0 + 51 * SECOND, [done(OTHER)]),
        scheduled(2, T0 + 80 * SECOND, [member(OTHER, scheduled=1, sighting="repeat")]),
        completed(2, T0 + 81 * SECOND, [done(OTHER)]),
        heartbeat(T0 + 82 * SECOND, 8),
    ]
    engine_log(tmp_path, records)
    window = Window(
        "phase",
        T0 + WALL_OFFSET - SECOND,
        T0 + WALL_OFFSET + 2 * SECOND,
        "c1",
        "measured",
    )
    placed = _reduce(
        tmp_path,
        _facts(
            windows=(window,),
            referenced_iterations=frozenset({EntityRef(PRODUCER, "2")}),
        ),
    )
    assert _iteration_ids(placed) == [
        "0",
        "2",
    ]  # 0 in the window, 2 referenced by a trace
    epoch = placed.summary["epochs"][EPOCH]
    assert (epoch["foreign_only_placed"], epoch["foreign_only_counted"]) == (1, 1)
    # Counted iterations consume their records: the mark moves past them.
    assert placed.high_water == {EPOCH: 8}
    # On another host the engine's wall clock is not the client's: no placing.
    elsewhere = _reduce(
        tmp_path,
        _facts(windows=(window,), client_clock_domain="laptop/boot-z/unix_epoch_ns"),
    )
    assert _iteration_ids(elsewhere) == []
    assert elsewhere.summary["epochs"][EPOCH]["foreign_only_counted"] == 3


def test_a_reused_internal_id_is_split_by_admission(tmp_path: Path) -> None:
    same = f"chatcmpl-{X0}"  # randomization off: internal == external
    records = [
        alias(same, same, T0 - 10),
        scheduled(0, T0, [member(same, scheduled=8)]),
        completed(0, T0 + SECOND, [done(same)]),
        terminal(same, T0 + SECOND + 1, output_tokens=1),
        alias(same, same, T0 + 2 * SECOND),
        scheduled(1, T0 + 3 * SECOND, [member(same, scheduled=8)]),
        completed(1, T0 + 4 * SECOND, [done(same)]),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    requests = _requests(result)
    assert set(requests) == {same, f"{same}#2"}
    assert all(r.metadata["reused_internal_id"] for r in requests.values())
    assert all(
        r.request_ref == EntityRef("stormlog", REQUEST0) for r in requests.values()
    )
    memberships = {m.iteration_ref.id: m for m in _memberships(result)}
    assert _attempt(memberships["0"]) == same
    assert _attempt(memberships["1"]) == f"{same}#2"
    # The first execution was freed between steps: its finish rides on its
    # last membership, marked as such, and never on the second execution's.
    assert memberships["0"].metadata["finish"]["output_tokens"] == 1
    assert memberships["0"].metadata["finish"]["in_step"] is False
    assert "finish" not in memberships["1"].metadata


def test_a_reuse_across_imports_takes_the_next_attempt(tmp_path: Path) -> None:
    same = f"chatcmpl-{X0}"
    first_half = [
        alias(same, same, T0 - 10),
        scheduled(0, T0, [member(same, scheduled=8)]),
        completed(0, T0 + SECOND, [done(same)]),
        terminal(same, T0 + SECOND + 1, output_tokens=1),
        heartbeat(T0 + SECOND + 2, 4),
    ]
    engine_log(tmp_path, first_half)
    first = _reduce(tmp_path)
    assert set(_requests(first)) == {same}
    second_half = [
        alias(same, same, T0 + 2 * SECOND),
        scheduled(1, T0 + 3 * SECOND, [member(same, scheduled=8)]),
        completed(1, T0 + 4 * SECOND, [done(same)]),
    ]
    engine_log(tmp_path, first_half + second_half)
    second = _reduce(tmp_path, _facts_after(first), high_water=first.high_water)
    assert set(_requests(second)) == {f"{same}#2"}
    assert _requests(second)[f"{same}#2"].metadata["reused_internal_id"] is True
    (membership,) = _memberships(second)
    assert membership.attempt_ref == EntityRef(PRODUCER, f"{same}#2")
    graph = resolve_inference_events(first.events + second.events)
    assert len(graph.requests) == 2 and graph.unresolved == ()


def test_a_terminal_with_no_step_to_carry_it_is_counted(tmp_path: Path) -> None:
    records = _two_shared_steps() + [
        alias("chatcmpl-aborted-00000000", "chatcmpl-aborted", T0 + 3 * SECOND),
        terminal(
            "chatcmpl-aborted-00000000",
            T0 + 3 * SECOND + 1,
            status="FINISHED_ABORTED",
            finish_reason="abort",
            output_tokens=0,
        ),
        heartbeat(T0 + 4 * SECOND, 10),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    assert set(_requests(result)) == {OWN0, OWN1}
    epoch = result.summary["epochs"][EPOCH]
    assert epoch["finish_unattached"] == 1
    # The admission still waits (a step could follow); the terminal does not.
    assert result.high_water == {EPOCH: 8}


def test_spec_decode_roles_and_outcomes_are_carried(tmp_path: Path) -> None:
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        scheduled(
            0,
            T0,
            [
                member(
                    OWN0,
                    scheduled=4,
                    computed_before=8,
                    prefill=0,
                    past_prompt=4,
                    drafts=3,
                    sighting="repeat",
                )
            ],
        ),
        completed(
            0,
            T0 + SECOND,
            [
                done(
                    OWN0,
                    sampled=3,
                    accepted=2,
                    retained=3,
                    computed_after=11,
                    stale=True,
                )
            ],
        ),
        scheduled(1, T0 + 2 * SECOND, [member(OWN0, scheduled=1, sighting="repeat")]),
        completed(
            1, T0 + 3 * SECOND, [done(OWN0, outcome="dropped_stale", retained=0)]
        ),
        scheduled(
            2,
            T0 + 4 * SECOND,
            [member(OWN0, scheduled=3, computed_before=2, prompt_tokens=8)],
            preempted=[OWN1],
        ),
        completed(2, T0 + 5 * SECOND, []),  # no outcome reported for the member
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    memberships = {m.iteration_ref.id: m for m in _memberships(result)}
    spec = memberships["0"]
    assert spec.role == "spec_decode"
    assert (spec.metadata["accepted_drafts"], spec.metadata["stale"]) == (2, True)
    assert spec.metadata["computed_after"] == 11 and spec.output_tokens == 3
    assert spec.metadata["processed_prefill"] == 0
    assert memberships["1"].metadata["outcome"] == "dropped_stale"
    assert memberships["1"].metadata["processed_prefill"] == 0
    assert memberships["2"].role == "prefill"
    assert memberships["2"].metadata["outcome"] == "unknown"
    assert memberships["2"].metadata["processed_prefill"] == 0
    iterations = {
        i.iteration_ref.id: i for i in _by_type(result.events)["infer.iteration"]
    }
    assert iterations["2"].metadata["preempted"] == 1
    assert OWN1 not in str(iterations["2"].to_record())


def test_the_role_follows_vllm_phase_not_the_token_counts(tmp_path: Path) -> None:
    resumed = member(  # preempted, resumed: recomputes past its prompt as context
        OWN0,
        scheduled=12,
        computed_before=0,
        sighting="repeat",
        phase="context",
        prefill=8,
        past_prompt=4,
        recompute=True,
        output_before=4,
    )
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OWN1, f"chatcmpl-{X1}", T0 - 9),
        scheduled(0, T0, [resumed, member(OWN1, scheduled=8, phase=None)]),
        completed(0, T0 + SECOND, [done(OWN0), done(OWN1)]),
        scheduled(
            1,
            T0 + 2 * SECOND,
            [
                member(
                    OWN0, scheduled=3, sighting="repeat", phase="generation", prefill=3
                )
            ],
        ),
        completed(1, T0 + 3 * SECOND, [done(OWN0)]),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    memberships = {(_attempt(m), m.iteration_ref.id): m for m in _memberships(result)}
    assert memberships[(OWN0, "0")].role == "prefill"
    assert memberships[(OWN0, "0")].metadata["recompute"] is True
    assert memberships[(OWN0, "0")].metadata["past_prompt_scheduled"] == 4
    assert memberships[(OWN0, "0")].metadata["phase"] == "context"
    assert memberships[(OWN1, "0")].role == "unknown"  # no phase: not guessed
    assert memberships[(OWN0, "1")].role == "decode"  # phase wins over counts
    assert result.summary["epochs"][EPOCH]["config"]["v2_model_runner"] is True


def test_an_idle_step_is_counted_as_empty_not_foreign(tmp_path: Path) -> None:
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        scheduled(0, T0, [member(OWN0, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0)]),
        scheduled(1, T0 + 2 * SECOND, []),  # vLLM's scheduler ran with nothing to do
        completed(1, T0 + 2 * SECOND + 10, []),
        scheduled(2, T0 + 3 * SECOND, []),
        completed(2, T0 + 3 * SECOND + 10, []),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    assert _iteration_ids(result) == ["0"]
    epoch = result.summary["epochs"][EPOCH]
    assert (epoch["empty_counted"], epoch["foreign_only_counted"]) == (2, 0)
    # Unless a trace's activity points at it, which keeps it as evidence.
    referenced = _reduce(
        tmp_path, _facts(referenced_iterations=frozenset({EntityRef(PRODUCER, "2")}))
    )
    assert _iteration_ids(referenced) == ["0", "2"]
    assert referenced.summary["epochs"][EPOCH]["empty_counted"] == 1


def test_a_failed_update_leaves_every_outcome_unknown(tmp_path: Path) -> None:
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OWN1, f"chatcmpl-{X1}", T0 - 9),
        scheduled(0, T0, [member(OWN0, scheduled=8), member(OWN1, scheduled=8)]),
        completed(0, T0 + SECOND, [failed(OWN0), failed(OWN1)], update_failed=True),
        scheduled(
            1,
            T0 + 2 * SECOND,
            [member(OWN0, scheduled=1, sighting="repeat", computed_before=8)],
        ),
        completed(1, T0 + 3 * SECOND, [done(OWN0, finish_reason="stop")]),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    iterations = {
        i.iteration_ref.id: i for i in _by_type(result.events)["infer.iteration"]
    }
    assert iterations["0"].metadata["update_failed"] is True
    assert iterations["0"].metadata["state"] == "complete"
    assert iterations["1"].metadata["update_failed"] is False
    memberships = {(_attempt(m), m.iteration_ref.id): m for m in _memberships(result)}
    for key in ((OWN0, "0"), (OWN1, "0")):
        failed_member = memberships[key]
        assert failed_member.metadata["outcome"] == "unknown"
        assert failed_member.metadata["update_failed"] is True
        assert failed_member.output_tokens is None
        assert failed_member.metadata["processed_prefill"] == 0
        assert failed_member.metadata["computed_after"] is None
    # The next step's output is read as usual, with vLLM's own finish reason.
    assert memberships[(OWN0, "1")].metadata["finish_reason"] == "stop"
    assert memberships[(OWN0, "1")].metadata["update_failed"] is False
    assert result.summary["epochs"][EPOCH]["iterations_update_failed"] == 1


def test_a_resumable_request_keeps_each_turns_prompt(tmp_path: Path) -> None:
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        scheduled(0, T0, [member(OWN0, scheduled=8, prompt_tokens=8, resumable=True)]),
        completed(0, T0 + SECOND, [done(OWN0)]),
        # A second turn of input: the live prompt grew to 12.
        scheduled(
            1,
            T0 + 2 * SECOND,
            [
                member(
                    OWN0,
                    scheduled=4,
                    computed_before=8,
                    prompt_tokens=12,
                    sighting="repeat",
                    phase="context",
                    resumable=True,
                )
            ],
        ),
        completed(1, T0 + 3 * SECOND, [done(OWN0)]),
    ]
    engine_log(tmp_path, records)
    result = _reduce(tmp_path)
    request = _requests(result)[OWN0]
    assert request.input_tokens == 8  # the prompt at the first sighting
    assert request.metadata["resumable"] is True
    memberships = {m.iteration_ref.id: m for m in _memberships(result)}
    assert [memberships[i].metadata["prompt_tokens"] for i in ("0", "1")] == [8, 12]
    assert memberships["1"].role == "prefill" and memberships["1"].metadata["resumable"]


def test_binding_follows_the_hello_s_request_id_randomization(tmp_path: Path) -> None:
    plain = f"chatcmpl-{X0}"  # randomization off: internal == external
    suffixed = f"chatcmpl-{X1}-0f3a9c1d"
    records = [
        scheduled(0, T0, [member(plain, scheduled=8), member(suffixed, scheduled=8)]),
        completed(0, T0 + SECOND, [done(plain), done(suffixed)]),
    ]
    engine_log(tmp_path, records, config={"request_id_randomization": False})
    off = _requests(_reduce(tmp_path))
    assert off[plain].metadata["ownership"] == OWN
    # With the suffix known to be absent, nothing is stripped: a suffixed ID
    # that is not a recorded X-Request-Id stays foreign.
    assert [
        r.metadata["ownership"]
        for r in off.values()
        if r.attempt_ref and r.attempt_ref.id != plain
    ] == [FOREIGN]
    engine_log(tmp_path / "on", records, config={"request_id_randomization": True})
    on = _requests(_reduce(tmp_path / "on"))
    assert (
        on[plain].metadata["ownership"] == OWN
        and on[suffixed].metadata["ownership"] == OWN
    )
    engine_log(tmp_path / "unknown", records, config={"request_id_randomization": None})
    unknown = _requests(_reduce(tmp_path / "unknown"))
    assert {r.metadata["ownership"] for r in unknown.values()} == {OWN}


def test_drafts_make_spec_decode_only_in_generation(tmp_path: Path) -> None:
    records = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OWN1, f"chatcmpl-{X1}", T0 - 9),
        scheduled(
            0,
            T0,
            [
                member(OWN0, scheduled=4, sighting="repeat", drafts=3),
                # Resumed with drafts in flight: still context for the step.
                member(OWN1, scheduled=4, sighting="repeat", phase="context", drafts=3),
            ],
        ),
        completed(0, T0 + SECOND, [done(OWN0), done(OWN1)]),
    ]
    engine_log(tmp_path, records)
    roles = {_attempt(m): m.role for m in _memberships(_reduce(tmp_path))}
    assert roles == {OWN0: "spec_decode", OWN1: "prefill"}
