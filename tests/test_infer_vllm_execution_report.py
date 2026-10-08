"""The execution coverage block: unions per device and clock, never sums."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from stormlog.infer.correlation_events import (
    ActivityReferenceEvent,
    ArtifactIdentityEvent,
    CapabilityEvent,
    CorrelationContext,
    EntityRef,
    MembershipEvent,
)
from stormlog.infer.vllm_execution import ReduceOptions, RunFacts, RunRequest
from stormlog.infer.vllm_execution_import import (
    import_execution_into_artifact,
    reduce_to_capture,
)
from stormlog.infer.vllm_execution_log import read_execution_log
from stormlog.infer.vllm_execution_report import execution_lines, execution_report
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
REQUEST_A = "c1_in8_out4_measured_0_0"
REQUEST_B = "c2_in8_out4_measured_0_0"
XA = f"stormlog-{RUN}-{REQUEST_A}"
XB = f"stormlog-{RUN}-{REQUEST_B}"
OWN_A = f"chatcmpl-{XA}-0f3a9c1d"
OWN_B = f"chatcmpl-{XB}-77aa00bb"
OTHER = "chatcmpl-stormlog-run-9-c1_x_0-deadbeef"
MS = 1_000_000


def _legacy_request(request_id: str, x_request_id: str, case_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infer.request",
        "session_id": SESSION,
        "request_id": request_id,
        "case_id": case_id,
        "phase": "measured",
        "status": "ok",
        "x_request_id": x_request_id,
    }


def _facts() -> RunFacts:
    return RunFacts(
        run_id=RUN,
        session_id=SESSION,
        client_clock_domain=f"{HOST}/{BOOT}/unix_epoch_ns",
        requests={
            XA: RunRequest(REQUEST_A, XA, "c1_in8_out4", "measured"),
            XB: RunRequest(REQUEST_B, XB, "c2_in8_out4", "measured"),
        },
        referenced_iterations=frozenset({EntityRef(PRODUCER, "2")}),
    )


def _execution_records(root: Path) -> list[dict[str, Any]]:
    """Three steps: A alone, A with B and a foreign request, foreign alone (an
    incomplete one, since the epoch ended before its output)."""
    engine_log(
        root,
        [
            alias(OWN_A, f"chatcmpl-{XA}", T0 - 10),
            alias(OWN_B, f"chatcmpl-{XB}", T0 - 9),
            alias(OTHER, "chatcmpl-stormlog-run-9-c1_x_0", T0 - 8),
            scheduled(0, T0, [member(OWN_A, scheduled=8)]),
            completed(0, T0 + SECOND, [done(OWN_A)]),
            scheduled(
                1,
                T0 + SECOND,
                [
                    member(OWN_A, scheduled=1, sighting="repeat"),
                    member(OWN_B, scheduled=8),
                    member(OTHER, scheduled=8),
                ],
            ),
            completed(1, T0 + 2 * SECOND, [done(OWN_A), done(OWN_B), done(OTHER)]),
            scheduled(
                2, T0 + 2 * SECOND, [member(OTHER, scheduled=1, sighting="repeat")]
            ),
            goodbye(T0 + 3 * SECOND, 8),
        ],
    )
    read = read_execution_log(root, importer=HERE)
    capture = reduce_to_capture(read, _facts(), ReduceOptions())
    records = [event.to_record() for event in capture.events]
    capability = CapabilityEvent(
        context=_context("stormlog.infer.capture", "wall"),
        event_id="engine_adapter:1",
        metadata={"summary": capture.summary},
        component="engine_adapter",
        available=True,
        supported=list(capture.capabilities.supported),
        enabled=list(capture.capabilities.enabled),
        collected=list(capture.capabilities.collected),
    )
    records.append(capability.to_record())
    return records


def _context(
    source: str, clock_kind: str, device_uuid: str | None = None
) -> CorrelationContext:
    return CorrelationContext(
        run_id=RUN,
        session_id=SESSION,
        producer_id=source,
        source=source,
        clock_domain="kineto:node-7:trace" if clock_kind == "device" else "node-7/wall",
        clock_kind=clock_kind,
        collection_mode="imported",
        provenance="observed",
        device_uuid=device_uuid,
    )


def _activity(
    name: str,
    start_ms: int,
    end_ms: int,
    *,
    iteration: str | None = None,
    device_uuid: str | None = "GPU-a",
    clock_kind: str = "device",
    reason: str = "launch_outside_iteration_range",
) -> dict[str, Any]:
    return ActivityReferenceEvent(
        context=_context("kineto", clock_kind, device_uuid),
        event_id=name,
        metadata={} if iteration else {"unresolved_reason": reason},
        activity_ref=EntityRef("kineto", name),
        activity_kind="kernel",
        activity_domain="gpu",
        attribution_status="linked" if iteration else "unresolved",
        iteration_ref=EntityRef(PRODUCER, iteration) if iteration else None,
        start_ns=start_ms * MS,
        end_ns=end_ms * MS,
    ).to_record()


def _records(root: Path) -> list[dict[str, Any]]:
    return [
        _legacy_request(REQUEST_A, XA, "c1_in8_out4"),
        _legacy_request(REQUEST_B, XB, "c2_in8_out4"),
        *_execution_records(root),
        # Step 0: two overlapping launches count once in the union.
        _activity("k0", 0, 10, iteration="0"),
        _activity("k1", 5, 15, iteration="0"),
        # Step 1, shared by both cases and a foreign request.
        _activity("k2", 20, 30, iteration="1"),
        # Step 2: foreign only, and incomplete.
        _activity("k3", 40, 44, iteration="2"),
        # Linked to a step that was never imported (still pending).
        _activity("k4", 50, 52, iteration="9"),
        # Unlinked, by reason.
        _activity("k5", 60, 63),
        _activity("k6", 70, 71, reason="no_launch"),
        # Unmeasured: no device UUID; and a wall-clock activity.
        _activity("k7", 80, 90, iteration="0", device_uuid=None),
        _activity("k8", 0, 1, iteration="0", clock_kind="wall"),
        # Another device, whose union is kept apart.
        _activity("k9", 0, 100, iteration="1", device_uuid="GPU-b"),
    ]


def test_coverage_is_a_union_per_device_and_clock(tmp_path: Path) -> None:
    report = execution_report(_records(tmp_path))

    assert report["available"] is True
    assert report["iterations"] == {
        "total": 3,
        "complete": 2,
        "incomplete": 1,
        "ownership": {"run": 1, "mixed": 1, "foreign": 1},
    }
    assert report["memberships"] == {"run": 3, "foreign": 2}
    assert report["requests"] == {
        "executions": {"run": 2, "foreign": 1},
        "run_requests_bound": 2,
        "run_requests_total": 2,
    }
    scope = report["gpu"]["GPU-a@kineto:node-7:trace"]
    # k0 and k1 overlap: 15 ms, not 20; k2 10, k3 4, k4 2 -> 31 ms linked.
    assert scope["linkage"] == {
        "linked_ns": 31 * MS,
        "unlinked_ns": 4 * MS,
        "unlinked_by_reason_ns": {
            "launch_outside_iteration_range": 3 * MS,
            "no_launch": 1 * MS,
        },
    }
    assert scope["measured_busy_ns"] == 35 * MS
    assert scope["membership_ns"] == {
        "complete": 25 * MS,
        "incomplete": 4 * MS,
        "none": 2 * MS,
    }
    assert scope["ownership_ns"] == {
        "run": 15 * MS,
        "mixed": 10 * MS,
        "foreign": 4 * MS,
        "none": 2 * MS,
    }
    assert (
        report["gpu"]["GPU-b@kineto:node-7:trace"]["linkage"]["linked_ns"] == 100 * MS
    )
    assert report["unmeasured"] == {
        "activity_records": 2,
        "linked_records": 2,
        "summed_activity_ns": 11 * MS,
    }
    assert report["cases"] == {
        "c1_in8_out4": {
            "iterations": 2,
            "shared_iterations": 1,
            "non_additive": True,
            "gpu": {
                "GPU-a@kineto:node-7:trace": {"linked_ns": 25 * MS},
                "GPU-b@kineto:node-7:trace": {"linked_ns": 100 * MS},
            },
        },
        "c2_in8_out4": {
            "iterations": 1,
            "shared_iterations": 1,
            "non_additive": True,
            "gpu": {
                "GPU-a@kineto:node-7:trace": {"linked_ns": 10 * MS},
                "GPU-b@kineto:node-7:trace": {"linked_ns": 100 * MS},
            },
        },
    }
    assert report["capture_loss"] == {
        "epochs": 1,
        "dropped": {},
        "missing_sequences": 0,
        "pending_iterations": 0,
        "incomplete_iterations": 1,
        "update_failed_iterations": 0,
        "range_misses": 0,
        "finish_unattached": 0,
        "startup_unranged": 0,
        "truncated_epochs": 0,
        "read_errors": 0,
    }
    assert report["imports"][0]["high_water"] == {EPOCH: 9}


def test_a_request_still_in_flight_counts_from_its_dispatch(tmp_path: Path) -> None:
    records = _records(tmp_path)
    # B was sent and has not ended: the client wrote only its dispatch.
    records[1] = {**records[1], "event_type": "infer.dispatch"}
    del records[1]["status"]

    report = execution_report(records)

    assert report["requests"]["run_requests_total"] == 2
    assert sorted(report["cases"]) == ["c1_in8_out4", "c2_in8_out4"]


def test_text_lines_name_every_dimension(tmp_path: Path) -> None:
    lines = execution_lines(execution_report(_records(tmp_path)))
    assert lines[0] == (
        "vLLM execution: 3 steps imported (2 complete, 1 incomplete), memberships "
        "foreign 2, run 3, 2 of 2 run requests bound"
    )
    assert lines[1] == (
        "  GPU-a@kineto:node-7:trace: linked 31.000 ms of 35.000 ms measured busy "
        "(unlinked: launch_outside_iteration_range 3.000 ms, no_launch 1.000 ms); "
        "membership complete 25.000 ms, incomplete 4.000 ms, none 2.000 ms; "
        "ownership foreign 4.000 ms, mixed 10.000 ms, none 2.000 ms, run 15.000 ms"
    )
    assert "  unmeasured: 2 GPU activity records" in lines[3]
    assert lines[4] == (
        "  capture loss: 0 records dropped by the hook, 0 missing, 0 steps pending, "
        "1 incomplete, 0 update failures, 0 range misses, 0 finishes unattached; "
        "0 start-up calls unranged (not loss)"
    )
    assert lines[5].startswith("  c1_in8_out4: 2 steps, 1 shared (non-additive); ")


def test_another_adapters_memberships_are_not_counted(tmp_path: Path) -> None:
    """The membership totals cover the execution import's steps only, like
    the step totals they sit beside (Codex #3)."""
    records = _records(tmp_path)
    other = MembershipEvent(
        context=_context("another.engine", "monotonic"),
        event_id="other-membership",
        metadata={"ownership": "run"},
        request_ref=EntityRef("stormlog", REQUEST_A),
        iteration_ref=EntityRef("another.engine", "step-9"),
        role="decode",
    )
    report = execution_report(records + [other.to_record()])
    assert report["memberships"] == {"run": 3, "foreign": 2}
    assert execution_lines(report)[0].endswith(
        "memberships foreign 2, run 3, 2 of 2 run requests bound"
    )


def test_nothing_imported_is_not_available() -> None:
    report = execution_report([_legacy_request(REQUEST_A, XA, "c1")])
    assert report == {"available": False}
    assert execution_lines(report) == []


def _client_artifact(path: Path) -> Path:
    identity = ArtifactIdentityEvent(
        context=_context("stormlog.infer.profile", "wall"),
        event_id="artifact",
        artifact_kind="inference_jsonl",
        created_at_ns=1,
    )
    lines = [identity.to_record(), _legacy_request(REQUEST_A, XA, "c1_in8_out4")]
    path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    return path


def test_unresolved_members_are_not_reported_as_mixed(tmp_path: Path) -> None:
    """``mixed`` means a run member and another client's are both
    established; an unresolved ID leaves the split unknown (Astra #6)."""
    opaque = "opaque-id-no-alias"
    engine_log(
        tmp_path,
        [
            alias(OWN_A, f"chatcmpl-{XA}", T0 - 10),
            alias(OTHER, "chatcmpl-another-client", T0 - 9),
            # foreign + unresolved: no run member is established.
            scheduled(0, T0, [member(OTHER, scheduled=8), member(opaque, scheduled=8)]),
            completed(0, T0 + SECOND, [done(OTHER), done(opaque)]),
            # run + unresolved: the other member may or may not be ours.
            scheduled(
                1,
                T0 + 2 * SECOND,
                [
                    member(OWN_A, scheduled=8),
                    member(opaque, scheduled=1, sighting="repeat"),
                ],
            ),
            completed(1, T0 + 3 * SECOND, [done(OWN_A), done(opaque)]),
            # run + foreign: both established, so mixed.
            scheduled(
                2,
                T0 + 4 * SECOND,
                [
                    member(OWN_A, scheduled=1, sighting="repeat"),
                    member(OTHER, scheduled=1, sighting="repeat"),
                ],
            ),
            completed(2, T0 + 5 * SECOND, [done(OWN_A), done(OTHER)]),
            goodbye(T0 + 6 * SECOND, 8),
        ],
    )
    facts = replace(
        _facts(), referenced_iterations=frozenset({EntityRef(PRODUCER, "0")})
    )
    capture = reduce_to_capture(read_execution_log(tmp_path, importer=HERE), facts)
    records = [
        _legacy_request(REQUEST_A, XA, "c1_in8_out4"),
        *[event.to_record() for event in capture.events],
        _activity("k0", 0, 10, iteration="0"),
        _activity("k1", 20, 30, iteration="1"),
        _activity("k2", 40, 50, iteration="2"),
    ]
    report = execution_report(records)
    assert report["iterations"]["ownership"] == {"unresolved": 2, "mixed": 1}
    assert report["memberships"] == {"foreign": 2, "unresolved": 2, "run": 2}
    scope = report["gpu"]["GPU-a@kineto:node-7:trace"]
    assert scope["ownership_ns"] == {"mixed": 10 * MS, "unresolved": 20 * MS}
    # Steps 1 and 2 each hold someone not established as this run's.
    assert report["cases"]["c1_in8_out4"]["shared_iterations"] == 2
    assert report["cases"]["c1_in8_out4"]["non_additive"] is True
    assert (
        "ownership mixed 10.000 ms, unresolved 20.000 ms" in execution_lines(report)[1]
    )


def test_a_re_import_never_improves_the_capture_loss(tmp_path: Path) -> None:
    """Step loss comes from the records the artifact holds and unattached
    finishes from their sequences, so reading an unchanged log again
    reports the same loss, not zero (Astra #5)."""
    artifact = _client_artifact(tmp_path / "infer.jsonl")
    engine_log(
        tmp_path / "hook",
        [
            alias(OWN_A, f"chatcmpl-{XA}", T0 - 10),
            scheduled(0, T0, [member(OWN_A, scheduled=8)]),
            completed(0, T0 + SECOND, [failed(OWN_A)], update_failed=True),
            scheduled(
                1, T0 + SECOND + 10, [member(OWN_A, scheduled=1, sighting="repeat")]
            ),
            terminal(OTHER, T0 + 2 * SECOND),
            goodbye(T0 + 3 * SECOND, 6),
        ],
    )
    expected = {
        "incomplete_iterations": 1,
        "update_failed_iterations": 1,
        "finish_unattached": 1,
    }
    seen = []
    for _ in range(2):
        capture = import_execution_into_artifact(
            artifact, tmp_path / "hook", importer=HERE
        )
        records = [
            json.loads(line)
            for line in artifact.read_text(encoding="utf-8").splitlines()
        ]
        loss = execution_report(records)["capture_loss"]
        seen.append((len(capture.events), {k: loss[k] for k in expected}))
    assert seen == [(6, expected), (0, expected)]
    # The text line says the same after both imports.
    assert "1 incomplete, 1 update failures, 0 range misses, 1 finishes unattached" in (
        execution_lines(execution_report(records))[-2]
    )


def test_capture_loss_keeps_worker_counters_and_unattached_finishes_apart() -> None:
    epochs = {
        "engine-1-1": {
            "role": "engine",
            "reduced": True,
            "dropped": {"scheduled": 1, "alias_oversized": 1, "terminal": 2},
            "gaps": 3,
            "iterations_pending": 1,
            "finish_unattached_seqs": [7, 9],
        },
        "worker-2-1": {
            "role": "worker",
            "reduced": False,
            "dropped": {},
            "range_misses": 1,
            "startup_unranged": 9,
            "pending_samples": 0,
        },
    }
    capability = CapabilityEvent(
        context=_context("stormlog.infer.capture", "wall"),
        event_id="engine_adapter:1",
        metadata={"summary": {"execution": {"directory": "/hook", "epochs": epochs}}},
        component="engine_adapter",
        available=True,
        supported=["iterations"],
        enabled=["iterations"],
        collected=[],
    )
    loss = execution_report([capability.to_record()])["capture_loss"]
    # An oversized record is a drop like any other.
    assert loss["dropped"] == {"scheduled": 1, "alias_oversized": 1, "terminal": 2}
    # Step loss is read from the records, of which this summary-only artifact
    # has none; the two unattached finishes come from their sequences.
    assert loss["update_failed_iterations"] == 0
    assert (loss["missing_sequences"], loss["pending_iterations"]) == (3, 1)
    assert (loss["range_misses"], loss["startup_unranged"]) == (1, 9)
    assert loss["finish_unattached"] == 2


def test_a_failed_import_is_reported_as_such() -> None:
    capability = CapabilityEvent(
        context=_context("stormlog.infer.capture", "wall"),
        event_id="engine_adapter:1",
        metadata={
            "summary": {"execution": {"directory": "/hook", "failed": "no epoch"}}
        },
        component="engine_adapter",
        available=True,
        supported=["iterations"],
        enabled=["iterations"],
        collected=[],
    )
    report = execution_report([capability.to_record()])
    assert report["available"] is True
    assert report["iterations"]["total"] == 0
    assert execution_lines(report) == ["vLLM execution: import failed (no epoch)"]
