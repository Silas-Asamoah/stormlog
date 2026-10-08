"""KV preemption pressure against a toy engine with a small KV budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from stormlog.infer.diagnosis_context import Assessment, Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_kv import assess_kv
from stormlog.infer.diagnosis_selection import select
from stormlog.infer.diagnosis_text import render_text
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, stamp

AT = 90 * SECOND


def _requests() -> list[SimRequest]:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a", output=16)
    return calm + poisson_free(200, AT, 10 * MS, prefix="b", output=16)


def _assess(tmp_path: Path, **engine: Any) -> Assessment:
    view = join(
        read_input(
            build_run(
                tmp_path, _requests(), Engine(max_num_seqs=8, kv_tokens=120, **engine)
            )
        )
    )
    context = Context(view, select(view))
    (subject,) = context.subjects()
    return assess_kv(context, subject)


@pytest.fixture(scope="module")
def pressured(tmp_path_factory: pytest.TempPathFactory) -> Assessment:
    return _assess(tmp_path_factory.mktemp("kv"))


def test_allocation_preemptions_with_no_reset_are_kv_pressure(
    pressured: Assessment,
) -> None:
    (finding,) = pressured.findings

    assert pressured.status == "assessed"
    assert finding.gates == {"allocation_cause_established": True}
    assert [(a.kind, a.status) for a in finding.alternatives] == [
        ("prefix_cache_reset", "ruled_out")
    ]
    # The burst queued for seconds; the resume waits are a small part of it.
    assert finding.eligible and finding.severity == "info"
    assert finding.contribution.unmet == ("explains_e2e_excess",)
    metrics = finding.metrics
    assert metrics["allocation_preemptions"] and metrics["affected_requests"]
    assert metrics["recomputed_positions"] and metrics["resume_wait_p50_ms"]
    assert finding.observations[2].metric == "preemption_to_resume_entry_p50_ms"


def test_preemptions_that_explain_the_excess_are_a_kv_fault(tmp_path: Path) -> None:
    """Long outputs outgrow a small KV budget with slots to spare: requests
    are preempted again and again, and their resume waits make up the
    end-to-end excess."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a", output=100)
    heavy = poisson_free(100, AT, 200 * MS, prefix="b", output=100)
    view = join(
        read_input(
            build_run(tmp_path, calm + heavy, Engine(max_num_seqs=64, kv_tokens=300))
        )
    )
    context = Context(view, select(view))
    (subject,) = context.subjects()

    (finding,) = assess_kv(context, subject).findings

    assert "explains_e2e_excess" in finding.contribution.met
    assert (finding.severity, finding.claim) == ("warning", "fault")


def test_the_queue_behind_a_kv_fault_is_its_secondary(tmp_path: Path) -> None:
    """The queue, with an eligible KV finding upstream of it, is KV's
    consequence: secondary, no fault claim, ranked after the KV fault."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a", output=100)
    heavy = poisson_free(100, AT, 200 * MS, prefix="b", output=100)
    artifact = build_run(tmp_path, calm + heavy, Engine(max_num_seqs=64, kv_tokens=300))

    report = diagnose_artifact(artifact, options=DiagnoseOptions(generated_at_ns=1))

    details = report["payload"]["findings_detail"]
    by_kind = {detail["kind"]: (fid, detail) for fid, detail in details.items()}
    kv_id, kv = by_kind["kv_preemption_pressure"]
    _, queue = by_kind["queue_saturation"]
    assert (kv["role"], kv["claim"], kv["rank"]) == ("primary", "fault", 1)
    assert (queue["role"], queue["secondary_to"]) == ("secondary", [kv_id])
    assert queue["claim"] != "fault" and queue["rank"] > 1
    assert f"    secondary to {kv_id}" in render_text(report).splitlines()


def test_without_reset_records_the_cause_is_unknown(tmp_path: Path) -> None:
    # Astra's case: an older capture cannot rule out a reset, so a positive
    # excess with enough samples is still only an observation.
    assessment = _assess(tmp_path, observes=None)

    (finding,) = assessment.findings
    assert (assessment.status, assessment.reasons) == (
        "partial",
        ["preemption_cause_unknown"],
    )
    assert finding.alternatives[0].status == "untestable"
    assert (finding.claim, finding.severity, finding.cause) == (
        "observation",
        "info",
        "undetermined",
    )


def test_a_dropped_oversized_reset_leaves_no_reset_unestablished(
    tmp_path: Path,
) -> None:
    # The writer drops an oversized reset during the subject: dropped.cache_reset
    # stays 0 and nothing is capped, but cache_reset_oversized rises.
    assessment = _assess(tmp_path, dropped_from=AT + SECOND)

    (finding,) = assessment.findings
    assert finding.alternatives[0].status == "untestable"
    assert "records may have been lost" in finding.alternatives[0].reason
    assert finding.claim == "observation"


def test_a_reset_during_the_subject_is_not_ruled_out(tmp_path: Path) -> None:
    reset = {
        "kind": "cache_reset",
        "reset_running_requests": False,
        "reset_connector": False,
        "running": [],
        "succeeded": True,
        "raised": False,
        **stamp(AT + 500 * MS, "start_"),
        **stamp(AT + 500 * MS + 50_000, "end_"),
    }
    assessment = _assess(tmp_path, extra=[(AT + 500 * MS, reset)])

    (finding,) = assessment.findings
    assert finding.alternatives[0].status == "not_ruled_out"
    assert not finding.eligible


def test_no_preemption_is_not_observed(tmp_path: Path) -> None:
    view = join(read_input(build_run(tmp_path, _requests(), Engine(max_num_seqs=4))))
    context = Context(view, select(view))
    (subject,) = context.subjects()

    assessment = assess_kv(context, subject)

    assert (assessment.status, assessment.reasons, assessment.findings) == (
        "assessed",
        ["not_observed"],
        [],
    )


def test_a_reset_before_any_written_step_still_counts(tmp_path: Path) -> None:
    """A reset between the first request's admission and the first step has
    no step to refer to, so the import keeps it as a dated fact, not a
    stage; it still overlaps the request's lifetime."""
    first = SimRequest("a0", 10 * SECOND)
    at = first.admitted_ns + 50_000  # before it enters the queue
    reset = {
        "kind": "cache_reset",
        "reset_running_requests": False,
        "reset_connector": False,
        "running": [],
        "succeeded": True,
        "raised": False,
        **stamp(at, "start_"),
        **stamp(at + 50_000, "end_"),
    }
    calm = poisson_free(140, 10 * SECOND + SECOND, 500 * MS, prefix="a")
    view = join(
        read_input(build_run(tmp_path, [first, *calm], Engine(extra=[(at, reset)])))
    )
    context = Context(view, select(view))
    (producer,) = {e.producer for e in view.executions.values()}

    (epoch,) = view.engines.values()
    assert [fact["name"] for fact in epoch.unanchored] == ["engine.cache_reset"]
    alternative = context.reset_absent(producer, (at - MS, at + MS))
    assert alternative.status == "not_ruled_out"
