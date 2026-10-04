"""Importing a vLLM execution log into an existing inference artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main
from stormlog.infer.correlation_accounting import resolve_inference_events
from stormlog.infer.correlation_capture import (
    CaptureCapabilities,
    EngineCapture,
    append_inference_capture,
)
from stormlog.infer.correlation_events import (
    ArtifactIdentityEvent,
    CapabilityEvent,
    CorrelationContext,
    EntityRef,
    IterationEvent,
    MembershipEvent,
    RequestEvent,
    load_inference_artifact,
)
from stormlog.infer.errors import InferInputError
from stormlog.infer.vllm_execution_import import (
    execution_high_water,
    import_execution_into_artifact,
    run_facts_from_records,
)
from stormlog.session import create_session_summary
from tests.vllm_execution_helpers import (
    BOOT,
    HOST,
    KEY,
    SECOND,
    WALL_OFFSET,
    alias,
    completed,
    done,
    engine_log,
    heartbeat,
    importer,
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
HERE = importer(NOW - WALL_OFFSET)  # on the server's host and boot
RUN, SESSION = "run-1", "session-1"
REQUEST0 = "c1_in8_out4_measured_0_0"
X0 = f"stormlog-{RUN}-{REQUEST0}"
OWN0 = f"chatcmpl-{X0}-0f3a9c1d"
OTHER = "chatcmpl-stormlog-run-9-c1_x_0-deadbeef"


def _artifact(path: Path, *, requests: bool = True) -> Path:
    identity = ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id=RUN,
            session_id=SESSION,
            producer_id="stormlog.infer.profile",
            source="stormlog.infer.profile",
            clock_domain=f"{HOST}/{BOOT}/unix_epoch_ns",
            clock_kind="wall",
            collection_mode="active",
            provenance="observed",
        ),
        event_id="artifact",
        artifact_kind="inference_jsonl",
        created_at_ns=1,
    )
    lines: list[dict[str, Any]] = [identity.to_record()]
    if requests:
        lines.append(
            {
                "schema_version": 1,
                "event_type": "infer.request",
                "session_id": SESSION,
                "request_id": REQUEST0,
                "case_id": "c1_in8_out4",
                "phase": "measured",
                "started_at_ns": T0 + WALL_OFFSET - 20,
                "ended_at_ns": T0 + WALL_OFFSET + 2 * SECOND,
                "status": "ok",
                "x_request_id": X0,
            }
        )
        lines.append(
            {
                "schema_version": 1,
                "event_type": "infer.phase_window",
                "session_id": SESSION,
                "case_id": "c1_in8_out4",
                "phase": "measured",
                "started_at_ns": T0 + WALL_OFFSET - SECOND,
                "window_ended_at_ns": T0 + WALL_OFFSET + 2 * SECOND,
                "drained_at_ns": T0 + WALL_OFFSET + 3 * SECOND,
            }
        )
    path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    return path


def _records() -> list[dict[str, Any]]:
    return [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 9),
        scheduled(0, T0, [member(OWN0, scheduled=8), member(OTHER, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0), done(OTHER)]),
        scheduled(1, T0 + SECOND + 10, [member(OWN0, scheduled=1, sighting="repeat")]),
        terminal(OWN0, T0 + 2 * SECOND - 5, output_tokens=2),
        completed(1, T0 + 2 * SECOND, [done(OWN0)]),
        # A foreign-only step inside the measured phase window: placed.
        scheduled(
            2, T0 + 2 * SECOND + 10, [member(OTHER, scheduled=1, sighting="repeat")]
        ),
        completed(2, T0 + 2 * SECOND + 20, [done(OTHER)]),
        # A foreign-only step outside every window: counted.
        scheduled(3, T0 + 40 * SECOND, [member(OTHER, scheduled=1, sighting="repeat")]),
        completed(3, T0 + 41 * SECOND, [done(OTHER)]),
        heartbeat(T0 + 42 * SECOND, 11),
    ]


def test_import_appends_the_reduced_records_and_the_high_water(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    engine_log(tmp_path / "hook", _records())

    capture = import_execution_into_artifact(artifact, tmp_path / "hook", importer=HERE)

    assert capture.capabilities.collected == (
        "iterations",
        "memberships",
        "requests",
        "clock_alignment",
    )
    records = load_inference_artifact(artifact)
    iterations = [r for r in records if isinstance(r, IterationEvent)]
    assert [i.iteration_ref for i in iterations] == [
        EntityRef(PRODUCER, "0"),
        EntityRef(PRODUCER, "1"),
        EntityRef(PRODUCER, "2"),
    ]
    memberships = [r for r in records if isinstance(r, MembershipEvent)]
    assert len(memberships) == 4
    requests = [r for r in records if isinstance(r, RequestEvent)]
    assert {r.metadata["ownership"] for r in requests} == {"run", "foreign"}
    own = next(r for r in requests if r.metadata["ownership"] == "run")
    assert own.request_ref == EntityRef("stormlog", REQUEST0)
    assert own.metadata["case_id"] == "c1_in8_out4"
    engine = next(
        r
        for r in records
        if isinstance(r, CapabilityEvent) and r.component == "engine_adapter"
    )
    execution = engine.metadata["summary"]["execution"]
    assert execution["high_water"] == {EPOCH: 12}
    assert execution["records"] == {
        "iterations": 3,
        "memberships": 4,
        "requests": 2,
        "clock_alignment": 1,
    }
    assert execution["epochs"][EPOCH]["foreign_only_counted"] == 1
    assert execution["epochs"][EPOCH]["foreign_only_placed"] == 1
    assert OTHER not in artifact.read_text(encoding="utf-8")
    # The raw log is not an attachment; the artifact itself is.
    envelope = json.loads((tmp_path / "stormlog_run.json").read_text(encoding="utf-8"))
    assert [row["kind"] for row in envelope["attachments"]] == ["inference_jsonl"]
    assert resolve_inference_events(records).unresolved == ()


def test_a_second_import_adds_only_what_became_final(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = tmp_path / "hook"
    engine_log(hook, _records())
    import_execution_into_artifact(artifact, hook, importer=HERE)
    lines = artifact.read_text(encoding="utf-8").count("\n")

    again = import_execution_into_artifact(artifact, hook, importer=HERE)

    assert again.events == ()
    assert again.capabilities.collected == ()
    # Only the capability records were appended; the mark is unchanged.
    assert artifact.read_text(encoding="utf-8").count("\n") == lines + 2
    assert execution_high_water(load_inference_artifact(artifact)) == {EPOCH: 12}
    late = _records() + [
        scheduled(4, T0 + 50 * SECOND, [member(OWN0, scheduled=1, sighting="repeat")]),
        completed(4, T0 + 51 * SECOND, [done(OWN0)]),
    ]
    engine_log(hook, late)
    third = import_execution_into_artifact(artifact, hook, importer=HERE)
    assert [e.EVENT_TYPE for e in third.events] == [
        "infer.iteration",
        "infer.membership",
    ]
    records = load_inference_artifact(artifact)
    assert execution_high_water(records) == {EPOCH: 14}
    assert resolve_inference_events(records).unresolved == ()


def test_an_alias_read_again_keeps_its_attempt(tmp_path: Path) -> None:
    """An admission that has not run yet holds the mark behind another
    request's alias; reading that alias again must not admit it twice
    (Astra #1)."""
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = tmp_path / "hook"
    waiting = "chatcmpl-other-11111111"
    first = [
        alias(waiting, "chatcmpl-other", T0 - 20),
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        scheduled(0, T0, [member(OWN0, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0)]),
    ]
    engine_log(hook, first)
    one = import_execution_into_artifact(artifact, hook, importer=HERE)
    assert one.summary is not None
    assert one.summary["execution"]["high_water"] == {EPOCH: 0}
    engine_log(
        hook,
        first
        + [
            scheduled(
                1, T0 + 2 * SECOND, [member(OWN0, scheduled=1, sighting="repeat")]
            ),
            completed(1, T0 + 3 * SECOND, [done(OWN0)]),
        ],
    )
    two = import_execution_into_artifact(artifact, hook, importer=HERE)
    assert [e.EVENT_TYPE for e in two.events] == ["infer.iteration", "infer.membership"]
    records = load_inference_artifact(artifact)
    requests = [r for r in records if isinstance(r, RequestEvent)]
    assert [r.attempt_ref for r in requests] == [EntityRef(PRODUCER, OWN0)]
    assert (requests[0].metadata["epoch"], requests[0].metadata["admission_seq"]) == (
        EPOCH,
        2,
    )
    memberships = [r for r in records if isinstance(r, MembershipEvent)]
    assert [(m.iteration_ref.id, m.attempt_ref) for m in memberships] == [
        ("0", EntityRef(PRODUCER, OWN0)),
        ("1", EntityRef(PRODUCER, OWN0)),
    ]
    assert resolve_inference_events(records).unresolved == ()
    # A third import, with nothing new, adds nothing.
    assert import_execution_into_artifact(artifact, hook, importer=HERE).events == ()


def _mixed_scheme_log(hook: Path, *, key: bytes | None = KEY) -> list[dict[str, Any]]:
    first = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 9),
        scheduled(0, T0, [member(OWN0, scheduled=8), member(OTHER, scheduled=8)]),
        completed(0, T0 + SECOND, [done(OWN0), done(OTHER)]),
        heartbeat(T0 + SECOND + 5, 5),
    ]
    engine_log(hook, first, key=key)
    return first


def _continue_log(
    hook: Path, first: list[dict[str, Any]], *, key: bytes | None = KEY
) -> None:
    engine_log(
        hook,
        first
        + [
            scheduled(
                1,
                T0 + 2 * SECOND,
                [
                    member(OWN0, scheduled=1, sighting="repeat"),
                    member(OTHER, scheduled=1, sighting="repeat"),
                ],
            ),
            completed(1, T0 + 3 * SECOND, [done(OWN0), done(OTHER)]),
            heartbeat(T0 + 3 * SECOND + 5, 8),
        ],
        key=key,
    )


@pytest.mark.parametrize("first_raw", [False, True])
def test_an_epochs_foreign_id_scheme_is_fixed_by_its_first_import(
    tmp_path: Path, first_raw: bool
) -> None:
    """Pseudonyms then raw IDs (or the reverse) for one epoch would give the
    same foreign execution two requests and split its memberships
    (Codex #2): the second import is refused before anything is appended."""
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = tmp_path / "hook"
    first = _mixed_scheme_log(hook)
    capture = import_execution_into_artifact(
        artifact, hook, raw_foreign_ids=first_raw, importer=HERE
    )
    assert capture.summary is not None
    assert capture.summary["execution"]["epochs"][EPOCH]["foreign_ids"] == (
        "raw" if first_raw else "pseudonym"
    )
    _continue_log(hook, first)
    before = artifact.read_text(encoding="utf-8")
    with pytest.raises(InferInputError, match="would duplicate their requests"):
        import_execution_into_artifact(
            artifact, hook, raw_foreign_ids=not first_raw, importer=HERE
        )
    assert artifact.read_text(encoding="utf-8") == before
    # The same scheme as before continues the execution, with one request.
    again = import_execution_into_artifact(
        artifact, hook, raw_foreign_ids=first_raw, importer=HERE
    )
    assert [e.EVENT_TYPE for e in again.events] == [
        "infer.iteration",
        "infer.membership",
        "infer.membership",
    ]
    records = load_inference_artifact(artifact)
    foreign = [
        r
        for r in records
        if isinstance(r, RequestEvent) and r.metadata["ownership"] == "foreign"
    ]
    assert len(foreign) == 1
    assert resolve_inference_events(records).unresolved == ()


def test_a_withheld_epoch_may_be_imported_under_any_later_scheme(
    tmp_path: Path,
) -> None:
    """Withholding wrote no other client's identity, so nothing can be
    duplicated when the key turns up or raw IDs are asked for later."""
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = tmp_path / "hook"
    first = _mixed_scheme_log(hook, key=None)
    capture = import_execution_into_artifact(artifact, hook, importer=HERE)
    assert capture.summary is not None
    assert capture.summary["execution"]["epochs"][EPOCH]["foreign_ids"] == "withheld"
    _continue_log(hook, first, key=KEY)  # the key file is there now
    later = import_execution_into_artifact(artifact, hook, importer=HERE)
    assert later.summary is not None
    assert later.summary["execution"]["epochs"][EPOCH]["foreign_ids"] == "pseudonym"
    assert "infer.request" in [e.EVENT_TYPE for e in later.events]


def test_cli_exits_invalid_input_for_a_changed_foreign_id_scheme(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = tmp_path / "hook"
    first = _mixed_scheme_log(hook)
    assert main(["import-execution", str(artifact), str(hook)]) == int(ExitCode.OK)
    _continue_log(hook, first)
    code = main(["import-execution", str(artifact), str(hook), "--raw-foreign-ids"])
    assert code == int(ExitCode.INVALID_INPUT)


def test_a_remote_import_leaves_pending_steps_for_later(tmp_path: Path) -> None:
    """With the server's clock 31 s behind the importer's, a live epoch's
    pending step must not be finalized as incomplete (Codex #1): from
    another host the import does not judge liveness, so the step waits
    and the completion that arrives later completes it."""
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = tmp_path / "hook"
    head = [
        alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
        scheduled(0, T0, [member(OWN0, scheduled=8)]),
        heartbeat(T0 + SECOND, 2),
    ]
    engine_log(hook, head)
    # The importer runs elsewhere; its own clock reads 31 s past the server's
    # last heartbeat, which under the old wall-clock rule meant "gone".
    elsewhere = importer(T0 + 32 * SECOND, host="laptop", boot_id="boot-zzzz")
    first = import_execution_into_artifact(artifact, hook, importer=elsewhere)
    # Only the hello's clock alignment: the pending step is not finalized.
    assert [e.EVENT_TYPE for e in first.events] == ["infer.clock_alignment"]
    assert first.summary is not None
    epoch = first.summary["execution"]["epochs"][EPOCH]
    assert (epoch["state"], epoch["state_reason"]) == ("unknown", "other_host")
    # The pending step (seq 2) and the admission no final step has shown
    # (seq 1) both wait, so the mark stays at the hello.
    assert (epoch["iterations_pending"], epoch["high_water_seq"]) == (1, 0)
    engine_log(
        hook,
        head
        + [completed(0, T0 + 2 * SECOND, [done(OWN0)]), heartbeat(T0 + 3 * SECOND, 4)],
    )
    second = import_execution_into_artifact(artifact, hook, importer=elsewhere)
    iterations = [e for e in second.events if isinstance(e, IterationEvent)]
    assert [(i.iteration_ref.id, i.metadata["state"]) for i in iterations] == [
        ("0", "complete")
    ]
    assert (
        sum(isinstance(r, IterationEvent) for r in load_inference_artifact(artifact))
        == 1
    )


def test_server_stopped_finalizes_an_epoch_without_goodbye(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = engine_log(
        tmp_path / "hook",
        [
            alias(OWN0, f"chatcmpl-{X0}", T0 - 10),
            scheduled(0, T0, [member(OWN0, scheduled=8)]),
            heartbeat(T0 + SECOND, 2),
        ],
    ).parent.parent
    # From another host, without the option, the step waits and the line says so.
    with mock.patch("stormlog.infer.vllm_execution_log.Importer.here") as here:
        here.return_value = importer(
            T0 + 32 * SECOND, host="laptop", boot_id="boot-zzzz"
        )
        assert main(["import-execution", str(artifact), str(hook)]) == int(ExitCode.OK)
        out = capsys.readouterr().out
        assert "unknown (other_host: pending steps wait; pass --server-stopped" in out
        assert "1 pending" in out
        code = main(["import-execution", str(artifact), str(hook), "--server-stopped"])
    assert code == int(ExitCode.OK)
    out = capsys.readouterr().out
    assert "gone; kept 1 steps (1 incomplete), 0 pending" in out
    iterations = [
        r for r in load_inference_artifact(artifact) if isinstance(r, IterationEvent)
    ]
    assert [i.metadata["incomplete_reason"] for i in iterations] == ["epoch_ended"]


def test_run_facts_come_from_the_artifact(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    facts = run_facts_from_records(load_inference_artifact(artifact), RUN, SESSION)
    assert facts.client_clock_domain == f"{HOST}/{BOOT}/unix_epoch_ns"
    assert facts.requests[X0].request_id == REQUEST0
    assert [w.kind for w in facts.windows] == ["phase"]
    assert facts.windows[0].end_ns == T0 + WALL_OFFSET + 3 * SECOND
    assert facts.existing_iterations == frozenset()


def test_cli_imports_and_reports_the_epochs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = engine_log(tmp_path / "hook", _records()).parent.parent

    code = main(["import-execution", str(artifact), str(hook)])

    assert code == int(ExitCode.OK)
    out = capsys.readouterr().out
    assert "3 iterations, 4 memberships, 2 requests, 1 clock alignments" in out
    assert f"{EPOCH}: " in out and "kept 3 steps" in out and "high-water seq 12" in out


def test_cli_exits_invalid_input_for_a_missing_or_empty_directory(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    assert main(["import-execution", str(artifact), str(tmp_path / "nope")]) == int(
        ExitCode.INVALID_INPUT
    )
    (tmp_path / "empty").mkdir()
    assert main(["import-execution", str(artifact), str(tmp_path / "empty")]) == int(
        ExitCode.INVALID_INPUT
    )
    assert artifact.read_text(encoding="utf-8").count("\n") == 3


def test_cli_exits_invalid_input_for_an_artifact_without_run_identity(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps({"schema_version": 1, "event_type": "infer.request"}) + "\n",
        encoding="utf-8",
    )
    hook = engine_log(tmp_path / "hook", _records()).parent.parent
    assert main(["import-execution", str(legacy), str(hook)]) == int(
        ExitCode.INVALID_INPUT
    )
    with pytest.raises(InferInputError, match="infer.artifact"):
        import_execution_into_artifact(legacy, hook)


def test_cli_exits_usage_without_a_directory(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    with pytest.raises(SystemExit) as excinfo:
        main(["import-execution", str(artifact)])
    assert excinfo.value.code == int(ExitCode.USAGE)


def test_raw_foreign_ids_flag_reaches_the_artifact(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl")
    hook = engine_log(tmp_path / "hook", _records()).parent.parent
    assert (
        main(["import-execution", str(artifact), str(hook), "--raw-foreign-ids"]) == 0
    )
    assert OTHER in artifact.read_text(encoding="utf-8")


def test_an_engine_adapter_summary_is_recorded_on_its_capability(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl", requests=False)

    class _Engine:
        def collect(self, *, run_id: str, session_id: str) -> EngineCapture:
            return EngineCapture(
                CaptureCapabilities(("iterations",), ("iterations",), ()),
                summary={"execution": {"high_water": {"engine-1-1": 5}}},
            )

    append_inference_capture(
        artifact,
        run_id=RUN,
        session=create_session_summary(source="test", session_id=SESSION),
        engine_adapter=_Engine(),
    )
    records = load_inference_artifact(artifact)
    engine = next(
        r
        for r in records
        if isinstance(r, CapabilityEvent) and r.component == "engine_adapter"
    )
    assert engine.metadata["summary"] == {
        "execution": {"high_water": {"engine-1-1": 5}}
    }
    assert execution_high_water(records) == {"engine-1-1": 5}
