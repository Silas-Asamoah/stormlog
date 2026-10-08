"""KV preemption pressure against a toy engine with a small KV budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis_context import Assessment, Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_kv import assess_kv
from stormlog.infer.diagnosis_selection import select
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
    assert finding.eligible and finding.severity == "warning"
    metrics = finding.metrics
    assert metrics["allocation_preemptions"] and metrics["affected_requests"]
    assert metrics["recomputed_positions"] and metrics["resume_wait_p50_ms"]
    assert finding.observations[2].metric == "preemption_to_resume_entry_p50_ms"


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
