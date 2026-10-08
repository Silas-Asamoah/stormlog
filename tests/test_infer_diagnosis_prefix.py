"""Prefix-cache loss: warm requests finding less of their prefix cached."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stormlog.infer.diagnosis_context import Assessment, Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_prefix import assess_prefix
from stormlog.infer.diagnosis_selection import SelectionOptions, select
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET, stamp

AT = 90 * SECOND
SHARED = {"prompt": 128, "prefix_group": 1, "shared_prefix_tokens": 64}


def _assess(
    tmp_path: Path, subject: list[SimRequest], engine: Engine | None = None
) -> Assessment:
    reference = poisson_free(
        140, 10 * SECOND, 500 * MS, prefix="a", cached=64, **SHARED
    )
    artifact = build_run(tmp_path, reference + subject, engine or Engine())
    view = join(read_input(artifact))
    start = AT + WALL_OFFSET
    context = Context(
        view, select(view, SelectionOptions(windows=((start, start + 20 * SECOND),)))
    )
    (declared,) = context.subjects()
    return assess_prefix(context, declared)


def _cold(**fields: Any) -> list[SimRequest]:
    values = {**SHARED, "cached": 0, **fields}
    return poisson_free(30, AT, 500 * MS, prefix="b", **values)


def test_warm_requests_finding_nothing_cached_are_a_prefix_loss(
    tmp_path: Path,
) -> None:
    assessment = _assess(tmp_path, _cold())

    (finding,) = assessment.findings
    assert finding.metrics["warm_requests"] == 30
    assert finding.metrics["uncached_tokens_excess"] == 64.0
    assert {a.kind: a.status for a in finding.alternatives} == {
        "prefix_cache_reset": "ruled_out",
        "prefix_sharing_drop": "ruled_out",
        "prefix_working_set_growth": "ruled_out",
    }
    assert finding.eligible
    assert finding.detail["token_count_provenance"] == "engine_cached_at_admission"
    # The toy engine prefills as fast without the cache: TTFT never rose, so
    # the loss explains no incident and is no warning.
    assert finding.contribution.unmet == ("ttft_rose",)
    assert (finding.severity, finding.claim) == ("info", "condition")


def test_concurrent_first_use_of_a_group_is_not_warm(tmp_path: Path) -> None:
    # A new group's requests all arrive before any of them finished prefill.
    first_use = poisson_free(
        30, AT, 100_000, prefix="n", prompt=128, prefix_group=2, shared_prefix_tokens=64
    )

    assessment = _assess(tmp_path, first_use)

    assert (assessment.status, assessment.reasons) == (
        "assessed",
        ["too_few_warm_requests"],
    )


def test_a_working_set_that_outgrew_the_cache_is_not_ruled_out(
    tmp_path: Path,
) -> None:
    """The reference shares 4 prefixes; the subject shares as much, but over
    40 groups that include those 4. An LRU cache evicts the old prefixes
    with no fault, so their loss cannot be called one."""
    reference = poisson_free(200, 10 * SECOND, 100 * MS, prefix="a", **SHARED)
    for index, request in enumerate(reference):
        request.prefix_group, request.cached = index % 4, 64 if index >= 4 else 0
    subject = poisson_free(400, AT, 50 * MS, prefix="b", **SHARED)
    for index, request in enumerate(subject):
        request.prefix_group = index % 40
    view = join(read_input(build_run(tmp_path, reference + subject, Engine())))
    start = AT + WALL_OFFSET
    context = Context(
        view, select(view, SelectionOptions(windows=((start, start + 20 * SECOND),)))
    )

    assessment = assess_prefix(context, context.subjects()[0])

    (finding,) = assessment.findings
    growth = {a.kind: a for a in finding.alternatives}["prefix_working_set_growth"]
    assert growth.status == "not_ruled_out" and not finding.eligible


def test_a_workload_that_shares_less_is_not_ruled_out(tmp_path: Path) -> None:
    assessment = _assess(tmp_path, _cold(shared_prefix_tokens=16))

    (finding,) = assessment.findings
    sharing = {a.kind: a for a in finding.alternatives}["prefix_sharing_drop"]
    assert sharing.status == "not_ruled_out" and not finding.eligible


def test_a_reset_while_the_group_was_warm_is_not_ruled_out(tmp_path: Path) -> None:
    reset = {
        "kind": "cache_reset",
        "reset_running_requests": False,
        "reset_connector": False,
        "running": [],
        "succeeded": True,
        "raised": False,
        **stamp(AT - 200 * MS, "start_"),
        **stamp(AT - 200 * MS + 50_000, "end_"),
    }

    assessment = _assess(tmp_path, _cold(), Engine(extra=[(AT - 200 * MS, reset)]))

    (finding,) = assessment.findings
    reset_alt = {a.kind: a for a in finding.alternatives}["prefix_cache_reset"]
    assert reset_alt.status == "not_ruled_out" and not finding.eligible


def test_requests_that_declare_no_sharing_are_unsupported(tmp_path: Path) -> None:
    reference = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    plain = poisson_free(30, AT, 500 * MS, prefix="b")
    view = join(read_input(build_run(tmp_path, reference + plain, Engine())))
    start = AT + WALL_OFFSET
    context = Context(
        view, select(view, SelectionOptions(windows=((start, start + 20 * SECOND),)))
    )

    assessment = assess_prefix(context, context.subjects()[0])

    assert (assessment.status, assessment.reasons) == (
        "unsupported",
        ["no_declared_sharing"],
    )
