"""Queue saturation against a toy engine that serves a burst."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis_context import Assessment, Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_model import Finding
from stormlog.infer.diagnosis_queue import assess_queue
from stormlog.infer.diagnosis_selection import SelectionOptions, select
from tests.diagnosis_scenarios import (
    MS,
    Engine,
    SimRequest,
    build_run,
    poisson_free,
)
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET, stamp

BURST_AT = 90 * SECOND


def _requests(**burst: Any) -> list[SimRequest]:
    """70 s at 2 requests/s, then 1.5 s at 200 requests/s."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    heavy = poisson_free(300, BURST_AT, 5 * MS, prefix="b", **burst)
    return calm + heavy


def _assess(tmp_path: Path, requests: list[SimRequest], engine: Engine) -> Assessment:
    view = join(read_input(build_run(tmp_path, requests, engine)))
    context = Context(view, select(view))
    (subject,) = context.subjects()
    return assess_queue(context, subject)


def _alternatives(finding: Finding) -> dict[str, str]:
    return {alt.kind: alt.status for alt in finding.alternatives}


@pytest.fixture(scope="module")
def burst(tmp_path_factory: pytest.TempPathFactory) -> Assessment:
    return _assess(tmp_path_factory.mktemp("q"), _requests(), Engine(max_num_seqs=4))


def test_a_full_engine_s_waits_are_a_queue_fault(burst: Assessment) -> None:
    (finding,) = burst.findings

    assert burst.status == "assessed" and finding.eligible
    assert (finding.severity, finding.cause, finding.claim) == (
        "warning",
        "fault",
        "fault",
    )
    assert finding.confidence_level == "high"
    assert set(_alternatives(finding).values()) == {"ruled_out"}
    assert finding.gates == {"capacity_witness": True, "usable_timing": True}
    wait = finding.observations[0]
    assert wait.metric == "scheduler_wait_excess_ms" and wait.ci is not None
    assert wait.ci[0] > 100  # hundreds of ms of queueing, against ~0
    assert finding.metrics["steps_at_capacity_share"] == pytest.approx(1.0, abs=0.01)
    assert finding.support and len(finding.display) <= 8


def test_a_queue_that_explains_little_of_the_ttft_rise_is_no_warning(
    tmp_path: Path,
) -> None:
    """The same burst, but each response takes 4 s to reach the client: the
    queue is real and eligible, yet explains a sixth of the TTFT rise."""
    assessment = _assess(
        tmp_path, _requests(delivery_ns=4 * SECOND), Engine(max_num_seqs=4)
    )

    (finding,) = assessment.findings
    assert finding.eligible
    assert finding.contribution.unmet == ("explains_ttft_excess",)
    assert (finding.severity, finding.cause, finding.claim) == (
        "info",
        "undetermined",
        "condition",
    )


@pytest.mark.parametrize(
    "enqueue_ms, status, claim",
    [
        (60, "ruled_out", "fault"),  # 8% of the wait excess
        (120, "contributing", "condition"),  # 16%: real, minor, contested
        (200, "not_ruled_out", "observation"),  # 26%
    ],
)
def test_time_before_the_queue_is_ruled_out_only_below_a_floor(
    tmp_path: Path, enqueue_ms: int, status: str, claim: str
) -> None:
    """The burst's requests also spend longer between reaching the engine
    and entering its queue; the wait excess stays 757 ms."""
    assessment = _assess(
        tmp_path, _requests(enqueue_ns=enqueue_ms * MS), Engine(max_num_seqs=4)
    )

    (finding,) = assessment.findings
    assert _alternatives(finding)["engine_ingress"] == status
    assert finding.claim == claim
    if status == "contributing":
        assert finding.eligible and finding.severity == "info"
        assert finding.contested == ["competitor:engine_ingress:contributing"]


def test_the_witness_counts_steps_while_someone_waited(tmp_path: Path) -> None:
    """A subject's window can begin in calm traffic before the burst, as the
    first real run's did: its early requests wait some microseconds for an
    idle engine to wake. Steps between those waits ran nobody waiting, and
    count neither for capacity nor against it."""
    requests = _requests()
    view = join(
        read_input(
            build_run(tmp_path, requests, Engine(max_num_seqs=4, wake_ns=20_000))
        )
    )
    start = BURST_AT + WALL_OFFSET - 30 * SECOND
    context = Context(
        view, select(view, SelectionOptions(windows=((start, start + 40 * SECOND),)))
    )
    (subject,) = context.subjects()

    (finding,) = assess_queue(context, subject).findings

    assert finding.gates["capacity_witness"]
    assert finding.metrics["steps_at_capacity_share"] == pytest.approx(1.0, abs=0.02)


def test_admissions_rule_out_a_pause_only_while_requests_waited(
    tmp_path: Path,
) -> None:
    """Without pause records, admissions that never stop for 500 ms while a
    request waits rule out a paused scheduler. A window that begins in calm
    traffic has seconds between admissions with nobody waiting: those gaps
    say nothing about a pause."""
    engine = Engine(max_num_seqs=4, wake_ns=20_000, observes=None)
    view = join(read_input(build_run(tmp_path, _requests(), engine)))
    start = BURST_AT + WALL_OFFSET - 30 * SECOND
    context = Context(
        view, select(view, SelectionOptions(windows=((start, start + 40 * SECOND),)))
    )
    (subject,) = context.subjects()

    (finding,) = assess_queue(context, subject).findings

    assert _alternatives(finding)["scheduler_paused"] == "ruled_out"


def test_an_older_log_cannot_tell_ingress_from_the_queue(tmp_path: Path) -> None:
    engine = Engine(max_num_seqs=4, enqueued_records=False, observes=None)
    assessment = _assess(tmp_path, _requests(), engine)

    (finding,) = assessment.findings
    assert finding.observations[0].metric == "engine_ingress_to_schedule_excess_ms"
    assert "usable_timing" in finding.failed_gates
    assert "competitor:engine_ingress:untestable" in finding.failed_gates
    assert "competitor:blocked_waiting:untestable" in finding.failed_gates
    assert (finding.claim, finding.severity, finding.cause) == (
        "observation",
        "info",
        "undetermined",
    )


def test_grammar_constrained_waiting_requests_are_not_ruled_out(
    tmp_path: Path,
) -> None:
    assessment = _assess(
        tmp_path, _requests(structured_output=True), Engine(max_num_seqs=4)
    )

    (finding,) = assessment.findings
    assert _alternatives(finding)["blocked_waiting"] == "not_ruled_out"
    assert finding.claim == "observation"


def test_a_pause_over_the_waits_fails_the_gate(tmp_path: Path) -> None:
    pauses = [
        (
            BURST_AT + 500 * MS,
            {
                "kind": "pause",
                "from": "UNPAUSED",
                "to": "PAUSED_NEW",
                **stamp(BURST_AT + 500 * MS),
            },
        ),
        (
            BURST_AT + 900 * MS,
            {
                "kind": "pause",
                "from": "PAUSED_NEW",
                "to": "UNPAUSED",
                **stamp(BURST_AT + 900 * MS),
            },
        ),
    ]
    assessment = _assess(tmp_path, _requests(), Engine(max_num_seqs=4, extra=pauses))

    (finding,) = assessment.findings
    assert _alternatives(finding)["scheduler_paused"] == "not_ruled_out"
    assert not finding.eligible


def test_a_stalled_engine_explains_the_waits_instead(tmp_path: Path) -> None:
    # Capacity to spare, but the loop stops for 2 s while requests arrive.
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    heavy = poisson_free(120, BURST_AT, 20 * MS, prefix="b")
    engine = Engine(max_num_seqs=256, stall=(BURST_AT + 50 * MS, 2 * SECOND))
    assessment = _assess(tmp_path, calm + heavy, engine)

    (finding,) = assessment.findings
    assert _alternatives(finding)["engine_stall"] == "not_ruled_out"
    assert "capacity_witness" in finding.failed_gates
    assert (
        assessment.status == "partial" and "no_capacity_witness" in assessment.reasons
    )
    assert finding.claim == "observation"


@pytest.mark.parametrize(
    "stall_ms, status, claim",
    [(150, "contributing", "condition"), (1000, "not_ruled_out", "observation")],
)
def test_a_stall_during_the_waits_is_judged_by_its_length(
    tmp_path: Path, stall_ms: int, status: str, claim: str
) -> None:
    """A stall holds a waiting request back by no more than its own length:
    150 ms is a minor second cause of a 907 ms excess; 1 s could explain all
    of a 1,757 ms one. Neither is ruled out, though neither covers half of
    the waiting time."""
    engine = Engine(max_num_seqs=4, stall=(BURST_AT + 500 * MS, stall_ms * MS))
    assessment = _assess(tmp_path, _requests(), engine)

    (finding,) = assessment.findings
    assert _alternatives(finding)["engine_stall"] == status
    assert finding.claim == claim and finding.severity == "info"


def _entering_after(requests: list[SimRequest], engine: Engine) -> None:
    """vLLM adds requests on the engine thread: one that reaches the engine
    during a stall enters its queue only when the loop resumes."""
    probe = Engine(max_num_seqs=engine.max_num_seqs, stall=engine.stall)
    starts = sorted(
        record["start_mono_ns"]
        for record in probe.run([SimRequest(r.request_id, r.sent_ns) for r in requests])
        if record["kind"] == "scheduled"
    )
    gap = max(zip(starts, starts[1:]), key=lambda pair: pair[1] - pair[0])
    for request in requests:
        if gap[0] < request.admitted_ns < gap[1]:
            request.enqueue_ns = gap[1] - 1 - request.admitted_ns


def test_a_stall_that_built_the_backlog_is_not_ruled_out(tmp_path: Path) -> None:
    """Load the engine carries (no witness without a stall), and a 600 ms
    stall half a second in: the requests that arrived during it enter the
    queue after it, so it covers none of their waiting time, yet the
    backlog it left is the whole excess."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    load = poisson_free(600, BURST_AT, 8 * MS, prefix="b")
    engine = Engine(max_num_seqs=8, stall=(BURST_AT + 500 * MS, 600 * MS))
    _entering_after(calm + load, engine)

    (finding,) = _assess(tmp_path, calm + load, engine).findings

    assert _alternatives(finding)["engine_stall"] == "not_ruled_out"
    assert finding.claim == "observation"


def test_without_engine_records_the_queue_is_unsupported(tmp_path: Path) -> None:
    path = build_run(tmp_path, _requests(), Engine(max_num_seqs=4))
    client_only = [
        line
        for line in path.read_text().splitlines()
        if '"schema_version": 2' not in line or '"infer.artifact"' in line
    ]
    path.write_text("\n".join(client_only) + "\n")
    view = join(read_input(path))
    context = Context(view, select(view))

    (subject,) = context.subjects()
    assessment = assess_queue(context, subject)

    assert (assessment.status, assessment.reasons) == (
        "unsupported",
        ["no_server_queue_signal"],
    )


def test_a_calm_window_shows_no_queue(tmp_path: Path) -> None:
    calm = poisson_free(200, 10 * SECOND, 500 * MS, prefix="a")
    view = join(read_input(build_run(tmp_path, calm, Engine(max_num_seqs=4))))
    start = 90 * SECOND + WALL_OFFSET
    context = Context(
        view, select(view, SelectionOptions(windows=((start, start + 20 * SECOND),)))
    )

    (subject,) = context.subjects()
    assessment = assess_queue(context, subject)

    assert (assessment.status, assessment.findings) == ("assessed", [])
    assert assessment.reasons == ["not_observed"]
