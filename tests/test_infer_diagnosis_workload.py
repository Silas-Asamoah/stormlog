"""What the workload asked for: load, lengths and shared prefixes."""

from __future__ import annotations

from pathlib import Path

import pytest

from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_selection import SelectionOptions, select
from stormlog.infer.diagnosis_workload import (
    assess_inputs,
    assess_load,
    assess_outputs,
    assess_sharing,
    rate_ratio_interval,
)
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET

AT = 90 * SECOND


def _context(tmp_path: Path, subject: list[SimRequest], **reference: object) -> Context:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a", **reference)
    view = join(
        read_input(build_run(tmp_path, calm + subject, Engine(max_num_seqs=64)))
    )
    start = AT + WALL_OFFSET
    return Context(
        view, select(view, SelectionOptions(windows=((start, start + 10 * SECOND),)))
    )


def test_the_rate_ratio_has_an_exact_interval() -> None:
    low, high = rate_ratio_interval(20, 1.0, 100, 50.0)
    assert low < 10.0 < high
    assert low > 5.0 and high < 17.0
    assert rate_ratio_interval(0, 1.0, 10, 1.0)[0] == 0.0


def test_faster_arrivals_are_a_load_increase(tmp_path: Path) -> None:
    context = _context(tmp_path, poisson_free(100, AT, 100 * MS, prefix="b"))

    (finding,) = assess_load(context, context.subjects()[0]).findings

    assert (finding.severity, finding.cause, finding.claim) == (
        "info",
        "workload_change",
        "condition",
    )
    observation = finding.observations[0]
    assert observation.ci is not None and observation.ci[0] > 1.25
    # What share of the incident the demand explains is the driver's to say.
    assert finding.contribution.as_dict() == {
        "level": None,
        "met": [],
        "unmet": ["not_determined"],
    }
    assert finding.confidence_level == finding.condition.level


def test_longer_prompts_are_longer_inputs_only(tmp_path: Path) -> None:
    context = _context(tmp_path, poisson_free(20, AT, 500 * MS, prefix="b", prompt=256))
    subject = context.subjects()[0]

    (finding,) = assess_inputs(context, subject).findings

    assert finding.observations[0].value == 248.0
    assert assess_outputs(context, subject).reasons == ["not_observed"]
    assert assess_load(context, subject).reasons == ["not_observed"]


@pytest.mark.parametrize("shared, observed", [(0, True), (64, False)])
def test_requests_that_stop_sharing_are_a_sharing_drop(
    tmp_path: Path, shared: int, observed: bool
) -> None:
    subject = poisson_free(
        20,
        AT,
        500 * MS,
        prefix="b",
        prefix_group=1,
        shared_prefix_tokens=shared or None,
    )
    context = _context(tmp_path, subject, prefix_group=1, shared_prefix_tokens=64)

    assessment = assess_sharing(context, context.subjects()[0])

    assert bool(assessment.findings) is observed
