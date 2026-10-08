"""Client admission, the API server, and the profiler's own stop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from stormlog.infer.diagnosis_client import (
    assess_api_server,
    assess_capture_pause,
    assess_client_admission,
)
from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_selection import SelectionOptions, Subject, select
from tests.diagnosis_scenarios import (
    MS,
    SESSION,
    Engine,
    SimRequest,
    build_run,
    poisson_free,
)
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET

AT = 90 * SECOND


def _context(
    tmp_path: Path,
    requests: list[SimRequest],
    engine: Engine,
    windows: list[dict[str, Any]] | None = None,
    declared: tuple[tuple[int, int], ...] = (),
) -> Context:
    view = join(read_input(build_run(tmp_path, requests, engine, windows=windows)))
    return Context(view, select(view, SelectionOptions(windows=declared)))


def _only(context: Context) -> Subject:
    (subject,) = context.subjects()
    return subject


# ------------------------------------------------------- client admission


def _held_burst() -> list[SimRequest]:
    """A calm run, then arrivals the client sent progressively late."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    late = [
        SimRequest(
            f"b{i}",
            sent_ns=AT + i * 5 * MS + i * 3 * MS,
            intended_ns=AT + i * 5 * MS,
            held_for_slot=True,
        )
        for i in range(100)
    ]
    return calm + late


def test_requests_held_at_the_client_are_its_admission(tmp_path: Path) -> None:
    start = AT + WALL_OFFSET
    context = _context(
        tmp_path,
        _held_burst(),
        Engine(max_num_seqs=256),
        declared=((start, start + SECOND),),
    )

    assessment = assess_client_admission(context, _only(context))

    (finding,) = assessment.findings
    assert finding.metrics["held_for_slot"] == 100
    assert finding.eligible and finding.cause == "instrumentation"
    assert finding.observations[2].metric == "dispatch_lag_excess_ms"
    assert finding.observations[2].ci is not None and finding.observations[2].ci[0] > 0


def test_a_client_that_only_sent_late_held_nothing_back(tmp_path: Path) -> None:
    """A starved client sends 150 ms after each intended arrival, holding
    nothing: lag alone says it sent late, not that its in-flight limit held
    requests, and a larger --max-in-flight cannot help."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    late = [
        SimRequest(
            f"b{i}", sent_ns=AT + i * 50 * MS + 150 * MS, intended_ns=AT + i * 50 * MS
        )
        for i in range(100)
    ]
    start = AT + WALL_OFFSET
    context = _context(
        tmp_path,
        calm + late,
        Engine(max_num_seqs=64),
        declared=((start, start + 5 * SECOND),),
    )

    (finding,) = assess_client_admission(context, _only(context)).findings

    assert finding.metrics["held_for_slot"] == 0 and finding.metrics["dropped"] == 0
    assert "held_or_dropped" in finding.failed_gates
    assert (finding.claim, finding.severity) == ("observation", "info")
    assert finding.title == "The client sent requests later than their arrivals"
    assert finding.experiment is not None
    assert "--max-in-flight" not in finding.experiment["change"]


def test_a_closed_loop_has_no_intended_arrivals(tmp_path: Path) -> None:
    requests = poisson_free(140, 10 * SECOND, 500 * MS, closed_loop=True)
    start = 60 * SECOND + WALL_OFFSET
    context = _context(
        tmp_path, requests, Engine(), declared=((start, start + 10 * SECOND),)
    )

    assessment = assess_client_admission(context, _only(context))

    assert (assessment.status, assessment.reasons) == (
        "unsupported",
        ["no_intended_arrivals"],
    )


# --------------------------------------------------------------- API server


def _slow_front() -> list[SimRequest]:
    """Background load keeps the engine stepping; then two thirds of the
    requests take 300 ms to reach it."""
    background = poisson_free(300, 10 * SECOND, 100 * MS, prefix="g")
    slow = poisson_free(100, 30 * SECOND, 50 * MS, prefix="s", ingress_ns=300 * MS)
    return background + slow


def test_requests_slow_to_reach_a_stepping_engine_are_a_frontend_stall(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _slow_front(), Engine(max_num_seqs=64))
    subject = context.subjects()[0]

    assessment = assess_api_server(context, subject)

    (finding,) = assessment.findings
    assert (finding.kind, finding.component) == ("host_stall", "api_server")
    assert finding.detail == {"form": "frontend", "attribution": "host"}
    assert finding.gates == {"engine_progress": True, "bounded_placement": True}
    assert (
        finding.metrics["send_to_ingress_placed"] == finding.metrics["subject_requests"]
    )
    statuses = {alt.kind: alt.status for alt in finding.alternatives}
    assert statuses == {"scheduler_paused": "ruled_out", "capture_pause": "ruled_out"}
    assert finding.eligible
    assert (
        finding.observations[0].value is not None
        and finding.observations[0].value > 100
    )


def test_a_pause_is_ruled_out_only_where_nothing_was_lost_over_the_stalls(
    tmp_path: Path,
) -> None:
    # The hook reports a dropped record from 30.5 s, inside the slow stretch.
    engine = Engine(max_num_seqs=64, dropped_from=30 * SECOND + 500 * MS)
    context = _context(tmp_path, _slow_front(), engine)

    (finding,) = assess_api_server(context, context.subjects()[0]).findings

    paused = {alt.kind: alt for alt in finding.alternatives}["scheduler_paused"]
    assert paused.status == "untestable"
    assert paused.reason.startswith("records may have been lost over")
    assert not finding.eligible


def test_a_frontend_stall_needs_bracketed_engine_stamps(tmp_path: Path) -> None:
    context = _context(
        tmp_path, _slow_front(), Engine(max_num_seqs=64, bracketed=False)
    )

    assessment = assess_api_server(context, context.subjects()[0])

    assert (assessment.status, assessment.reasons) == (
        "unsupported",
        ["clock_alignment_required"],
    )


# ------------------------------------------------------------ capture pause


def _trace_window(**fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infer.trace_window",
        "session_id": SESSION,
        "case_id": "c1",
        "phase": "measured",
        "requested_at_ns": AT + WALL_OFFSET - SECOND,
        "started": True,
        "started_at_ns": AT + WALL_OFFSET - SECOND,
        "timestamp_ns": AT + WALL_OFFSET + 700 * MS,
        **fields,
    }


@pytest.fixture(scope="module")
def burst_requests() -> list[SimRequest]:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    return calm + poisson_free(300, AT, 5 * MS, prefix="b")


def test_a_profiler_stop_across_waiting_requests_is_a_capture_pause(
    tmp_path: Path, burst_requests: list[SimRequest]
) -> None:
    stop = _trace_window(
        stop_requested_at_ns=AT + WALL_OFFSET + 200 * MS,
        stopped_at_ns=AT + WALL_OFFSET + 700 * MS,
    )
    context = _context(tmp_path, burst_requests, Engine(max_num_seqs=4), windows=[stop])

    assessment = assess_capture_pause(context, _only(context))

    (finding,) = assessment.findings
    assert finding.cause == "instrumentation"
    assert finding.metrics["stop_duration_ms"] == 500.0
    assert finding.metrics["requests_across_stop"] >= 100
    assert (
        finding.window is not None
        and finding.window["start_ns"] == AT + WALL_OFFSET + 200 * MS
    )


def test_without_the_stop_request_stamp_a_capture_pause_is_unsupported(
    tmp_path: Path, burst_requests: list[SimRequest]
) -> None:
    stop = _trace_window(stopped_at_ns=AT + WALL_OFFSET + 700 * MS)
    context = _context(tmp_path, burst_requests, Engine(max_num_seqs=4), windows=[stop])

    assessment = assess_capture_pause(context, _only(context))

    assert (assessment.status, assessment.reasons) == (
        "unsupported",
        ["no_stop_request_stamp"],
    )


def test_a_stop_that_held_the_waits_explains_them(tmp_path: Path) -> None:
    # The server blocks for the 2 s stop, as a real profiler stop does.
    calm = poisson_free(1000, 10 * SECOND, 100 * MS, prefix="a")
    stop = _trace_window(
        stop_requested_at_ns=AT + WALL_OFFSET,
        stopped_at_ns=AT + WALL_OFFSET + SECOND * 2,
    )
    context = _context(
        tmp_path,
        calm,
        Engine(max_num_seqs=64, stall=(AT, 2 * SECOND)),
        windows=[stop],
        declared=((AT + WALL_OFFSET, AT + WALL_OFFSET + 2 * SECOND),),
    )

    (finding,) = assess_capture_pause(context, _only(context)).findings

    assert "explains_ttft_excess" in finding.contribution.met
    assert finding.metrics["stop_duration_ms"] == 2000.0
    assert finding.observations[-1].value == pytest.approx(950.0, abs=60)


def test_a_brief_stop_explains_none_of_a_queue_s_waits(
    tmp_path: Path, burst_requests: list[SimRequest]
) -> None:
    """A 2 ms stop inside a burst that queued for hundreds of ms: it overlapped
    many requests, but held the median one back by no more than 2 ms."""
    stop = _trace_window(
        stop_requested_at_ns=AT + WALL_OFFSET + 700 * MS,
        stopped_at_ns=AT + WALL_OFFSET + 702 * MS,
    )
    context = _context(tmp_path, burst_requests, Engine(max_num_seqs=4), windows=[stop])

    (finding,) = assess_capture_pause(context, _only(context)).findings

    assert finding.metrics["requests_across_stop"] > 50
    assert "explains_ttft_excess" in finding.contribution.unmet
    assert finding.contribution_lower is not None and finding.contribution_lower <= 2.0


def test_an_instrumentation_warning_exits_3_without_a_fault_claim(
    tmp_path: Path,
) -> None:
    calm = poisson_free(1000, 10 * SECOND, 100 * MS, prefix="a")
    stop = _trace_window(
        stop_requested_at_ns=AT + WALL_OFFSET,
        stopped_at_ns=AT + WALL_OFFSET + 2 * SECOND,
    )
    artifact = build_run(
        tmp_path, calm, Engine(max_num_seqs=64, stall=(AT, 2 * SECOND)), windows=[stop]
    )
    window = (AT + WALL_OFFSET, AT + WALL_OFFSET + 2 * SECOND)

    report = diagnose_artifact(
        artifact, windows=[window], options=DiagnoseOptions(generated_at_ns=1)
    )

    details = report["payload"]["findings_detail"].values()
    (capture,) = [d for d in details if d["kind"] == "capture_pause"]
    assert (capture["severity"], capture["cause"], capture["claim"]) == (
        "warning",
        "instrumentation",
        "condition",
    )
    assert report["verdict"]["exit_code"] == 3


def test_a_frontend_stall_with_most_sends_unplaced_is_no_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only 60 of the subject's 160 sends can be placed (a wall clock
    discontinuity across the others' brackets, say), all of them slow: the
    excess of those that can is no bound on the subject's."""
    context = _context(tmp_path, _slow_front(), Engine(max_num_seqs=64))
    subject = context.subjects()[0]
    slow = [r for r in subject.requests if r.startswith("s")]
    unplaced = set(subject.requests) - set(slow[:60])
    values = context.segment_values

    def placed_only(request_ids: list[str], name: str) -> list[float]:
        if name == "send_to_ingress":
            request_ids = [r for r in request_ids if r not in unplaced]
        return values(request_ids, name)

    monkeypatch.setattr(context, "segment_values", placed_only)

    (finding,) = assess_api_server(context, subject).findings

    assert finding.gates["bounded_placement"] is False
    assert finding.claim == "observation"


def test_a_frontend_stall_says_how_many_requests_it_placed(tmp_path: Path) -> None:
    # The wall clock steps 50 ms back during the slow stretch: a send before
    # the step cannot be paired with an admission after it.
    engine = Engine(max_num_seqs=64, wall_jump=(32 * SECOND, -50 * MS))
    context = _context(tmp_path, _slow_front(), engine)

    (finding,) = assess_api_server(context, context.subjects()[0]).findings

    assert (
        finding.metrics["send_to_ingress_placed"],
        finding.metrics["subject_requests"],
    ) == (152, 160)
    assert finding.gates["bounded_placement"]  # most were placed
