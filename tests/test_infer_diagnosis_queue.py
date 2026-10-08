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
