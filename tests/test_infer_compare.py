"""Comparing a baseline arm of runs with a candidate arm."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.compare import ComparisonSpec, compare_runs
from stormlog.infer.compare_metrics import default_metrics
from stormlog.infer.comparison_stats import GateRule
from stormlog.infer.compatibility import RunField
from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.run_summary import RunSummary, summarize_run
from stormlog.infer.slo import parse_slo_flags
from tests.infer_workload_helpers import run_profile_with_fake_client

CASE = "c1"
E2E_GATE = (("client.e2e.p95", GateRule("non-inferiority", 0.05, "relative")),)


def _fields(**overrides: Any) -> dict[str, RunField]:
    values = {
        "model.weights_digest": "w" * 64,
        "engine.version": "0.30.0",
        "gpu.name": "NVIDIA A30",
        "gpu.driver_version": "580.82.07",
        "workload.spec_digest": "s" * 64,
        # The server answered /server_info, so its configuration was read.
        "scope.vllm_config": True,
        "vllm_config/scheduler_config/max_num_seqs": 256,
    }
    values.update(overrides)
    return {name: RunField(value, "test", "observed") for name, value in values.items()}


def _latency(p95: float, *, penalized: bool = False, sufficient: bool = True) -> dict:
    estimate = {
        "value_ms": None if penalized else p95,
        "sufficient": sufficient,
        "penalized": penalized,
    }
    levels = {level: dict(estimate) for level in ("p50", "p90", "p95", "p99")}
    return {"metrics": {"client.e2e": {"failure_penalized": levels}}}


def _case(
    e2e: float,
    *,
    goodput: float = 10.0,
    attainment: float = 0.99,
    coverage: float = 1.0,
    penalized: bool = False,
    slo_digest: str = "p" * 64,
) -> dict[str, Any]:
    return {
        "population": {"offered": 100, "successful": 99, "cohort_valid": True},
        "throughput": {"requests_per_second": 10.0, "output_tokens_per_second": 400.0},
        "latency": _latency(e2e, penalized=penalized),
        "slo": {
            "status": "evaluated",
            "goodput_lower_rps": goodput,
            "goodput_upper_rps": goodput,
            "attainment_lower": attainment,
            "attainment_upper": attainment,
            "evidence_coverage": coverage,
            "met": round(attainment * 100),
            "offered": 100,
            "slo_digest": slo_digest,
        },
        "cache": {},
    }


def _run(
    arm: str,
    block: int | None,
    e2e: float,
    *,
    fields: dict[str, RunField] | None = None,
    observers: dict[str, Any] | None = None,
    status: str = "completed",
    started: int = 0,
    **case: Any,
) -> RunSummary:
    labels = (
        {} if block is None else {"experiment": "e", "arm": arm, "block": str(block)}
    )
    return RunSummary(
        path=None,
        sha256=None,
        run_id=f"{arm}-{block}-{e2e}",
        session_id="s",
        session_status=status,
        labels=labels,
        fields=fields or _fields(),
        report={
            "cases": {CASE: _case(e2e, **case)},
            "observers": {"observers": observers or {}},
        },
        protocol_failures=() if status == "completed" else (f"session_{status}",),
        started_at_ns=started,
    )


BASE_E2E = [100.0, 102.0, 98.0, 101.0, 99.0, 100.0]


def _arms(
    candidate_e2e: list[float], *, paired: bool = True, **candidate: Any
) -> tuple[list[RunSummary], list[RunSummary]]:
    def block(i: int) -> int | None:
        return i if paired else None

    baseline = [
        _run("baseline", block(i), e2e, started=2 * i) for i, e2e in enumerate(BASE_E2E)
    ]
    candidates = [
        _run("candidate", block(i), e2e, started=2 * i + 1, **candidate)
        for i, e2e in enumerate(candidate_e2e)
    ]
    return baseline, candidates


SLOWER = [140.0, 143.0, 137.0, 141.0, 138.5, 140.5]
SAME = [100.5, 101.5, 98.5, 100.5, 99.5, 100.0]


def test_a_latency_regression_fails_its_gate_with_exit_4() -> None:
    comparison = compare_runs(*_arms(SLOWER), ComparisonSpec(gates=E2E_GATE))
    metric = comparison.cases[CASE]["metrics"]["client.e2e.p95"]

    assert comparison.design == "paired_blocks"
    assert metric.n_pairs == 6
    assert metric.gate is not None and metric.gate.status == "fail"
    assert (CASE, "client.e2e.p95") in comparison.failed
    assert comparison.exit_code == 4
    payload = comparison.to_payload()
    assert (payload["format"], payload["version"]) == ("stormlog.infer.comparison", 1)
    assert payload["verdict"]["exit_code"] == 4


def test_an_unchanged_candidate_passes() -> None:
    comparison = compare_runs(*_arms(SAME), ComparisonSpec(gates=E2E_GATE))
    assert comparison.exit_code == 0
    assert comparison.failed == []


def test_a_gate_that_matches_no_metric_cannot_be_evaluated() -> None:
    # The runs have no server spans: a gate on them gates nothing, and that
    # must not read as a pass while the candidate is 40% slower.
    gates = (("server.e2e.p95", GateRule("non-inferiority", 0.05, "relative")),)
    comparison = compare_runs(*_arms(SLOWER), ComparisonSpec(gates=gates))
    assert (CASE, "server.e2e.p95", "metric_absent") in comparison.not_evaluable
    assert comparison.exit_code == 4
    payload = comparison.to_payload()
    absent = payload["cases"][CASE]["absent_gates"]["server.e2e.p95"]
    assert (absent["status"], absent["reason"]) == ("not_evaluable", "metric_absent")


def test_a_case_no_run_has_is_invalid_input() -> None:
    with pytest.raises(InferInputError, match="case typo is in no run"):
        compare_runs(*_arms(SLOWER), ComparisonSpec(gates=E2E_GATE, cases=("typo",)))


def test_a_rate_without_an_interval_says_why() -> None:
    case = _case(100.0)
    case["throughput"]["requests_per_second"] = None
    case["slo"]["goodput_lower_rps"] = case["slo"]["goodput_upper_rps"] = None
    case["intervals"] = {"rate_reason": "overlapping_cohort"}
    metrics = {metric.name: metric for metric in default_metrics(case)}
    assert metrics["throughput_rps"].read(case) == (None, "overlapping_cohort")
    assert metrics["goodput_rps"].read(case) == (None, "overlapping_cohort")
    del case["intervals"]
    assert metrics["throughput_rps"].read(case) == (None, "rate_unavailable")


def test_every_value_a_run_cannot_give_has_a_reason() -> None:
    # A value read as plain missing was attrition: the gate went on with
    # the runs left. Each reader now says why.
    case = _case(100.0)
    estimate = case["latency"]["metrics"]["client.e2e"]["failure_penalized"]["p95"]
    estimate.update(value_ms=None, reason="successful_values_missing")
    case["population"] = {}
    metrics = {metric.name: metric for metric in default_metrics(case)}
    assert metrics["client.e2e.p95"].read(case) == (None, "successful_values_missing")
    assert metrics["failure_fraction"].read(case) == (None, "population_unrecorded")
    del estimate["reason"]
    assert metrics["client.e2e.p95"].read(case) == (None, "no_value")


def test_unlabelled_runs_are_independent_samples() -> None:
    comparison = compare_runs(*_arms(SLOWER, paired=False), ComparisonSpec())
    metric = comparison.cases[CASE]["metrics"]["client.e2e.p95"]
    assert comparison.design == "independent"
    assert metric.worst is not None and metric.worst.df == 5


def test_a_retried_block_keeps_the_attempt_that_finished() -> None:
    # The runner retries a run its protocol set aside; the first attempt is
    # listed, not a second run of the arm in that block.
    baseline, candidate = _arms(SAME)
    failed = _run("candidate", 2, 100.0, status="interrupted")
    comparison = compare_runs(baseline, [*candidate, failed], ComparisonSpec())
    assert [item["run"] for item in comparison.excluded] == [failed.name]


def test_an_empty_segment_cannot_be_gated() -> None:
    baseline, candidate = _arms(SLOWER)
    for run in [*baseline, *candidate]:
        run.report["cases"][CASE]["population"]["offered"] = 0
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and gate.reason == "empty_case"


def test_a_block_with_two_runs_of_an_arm_is_refused() -> None:
    baseline, candidate = _arms(SAME)
    candidate[1] = _run("candidate", 0, 100.0)
    with pytest.raises(InferInputError, match="has two candidate runs"):
        compare_runs(baseline, candidate, ComparisonSpec())


def test_an_unfinished_run_is_set_aside_and_listed() -> None:
    baseline, candidate = _arms(SLOWER)
    candidate[5] = _run("candidate", 5, 140.0, status="interrupted")
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    metric = comparison.cases[CASE]["metrics"]["client.e2e.p95"]
    assert metric.n_pairs == 5
    assert comparison.excluded[0]["reasons"] == ["session_interrupted"]
    strict = compare_runs(
        baseline, candidate, ComparisonSpec(gates=E2E_GATE, on_incomplete="fail")
    )
    gate = strict.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and (gate.status, gate.reason) == (
        "fail",
        "protocol_failure",
    )


def test_incompatible_arms_are_an_input_error_unless_the_difference_is_allowed() -> (
    None
):
    baseline, _ = _arms(SAME)
    changed = _fields(**{"vllm_config/scheduler_config/max_num_seqs": 64})
    candidate = [_run("candidate", i, e, fields=changed) for i, e in enumerate(SAME)]
    with pytest.raises(InferInputError, match="max_num_seqs"):
        compare_runs(baseline, candidate, ComparisonSpec())
    allowed = compare_runs(
        baseline, candidate, ComparisonSpec(allow=("engine.max_num_seqs",))
    )
    assert allowed.comparability.status == "compatible"


def test_unverified_runs_cannot_pass_a_gate() -> None:
    unknown = _fields()
    del unknown["model.weights_digest"]
    baseline = [_run("baseline", i, e, fields=unknown) for i, e in enumerate(BASE_E2E)]
    candidate = [_run("candidate", i, e, fields=unknown) for i, e in enumerate(SAME)]
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and (gate.status, gate.reason) == (
        "not_evaluable",
        "unverified",
    )
    assert comparison.exit_code == 4
    explore = compare_runs(
        baseline, candidate, ComparisonSpec(gates=E2E_GATE, allow_not_evaluable=True)
    )
    assert explore.exit_code == 0


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_one_run_that_cannot_be_verified_leaves_the_arms_unverified(arm: str) -> None:
    # Comparability is not transitive once a value is unknown: every run is
    # checked, not only each arm's first.
    baseline, candidate = _arms(SAME)
    unknown = _fields()
    del unknown["model.weights_digest"], unknown["gpu.name"]
    runs = baseline if arm == "baseline" else candidate
    runs[3] = _run(arm, 3, 100.0, fields=unknown, started=runs[3].started_at_ns or 0)
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    assert comparison.comparability.status == "unverified"
    assert {item.name for item in comparison.comparability.unverified} >= {
        "gpu.name",
        "model.weights_digest",
    }
    assert any(
        runs[3].name in pair for pair in comparison.diagnostics["unverified_pairs"]
    )
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and gate.reason == "unverified"


def test_a_candidate_arm_mixing_two_configurations_is_refused() -> None:
    baseline, candidate = _arms(SAME)
    other = _fields(**{"vllm_config/scheduler_config/max_num_seqs": 64})
    candidate[2] = _run("candidate", 2, 100.0, fields=other)
    with pytest.raises(InferInputError, match="candidate runs are incompatible"):
        compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))


def _observer(requested: bool, healthy: bool | None = True) -> dict[str, Any]:
    return {
        "requested": requested,
        "active": requested,
        "healthy": healthy if requested else None,
    }


def test_an_overhead_baseline_must_run_no_observers() -> None:
    watched = {"vllm_metrics": _observer(True)}
    baseline = [
        _run("baseline", i, e, observers=watched) for i, e in enumerate(BASE_E2E)
    ]
    candidate = [_run("candidate", i, e, observers=watched) for i, e in enumerate(SAME)]
    with pytest.raises(InferInputError, match="baseline without observers"):
        compare_runs(baseline, candidate, ComparisonSpec(mode="overhead"))


def test_an_unhealthy_observer_under_test_makes_gates_not_evaluable() -> None:
    baseline, _ = _arms(SAME)
    sick = {"vllm_metrics": _observer(True, healthy=False)}
    candidate = [_run("candidate", i, e, observers=sick) for i, e in enumerate(SAME)]
    comparison = compare_runs(
        baseline, candidate, ComparisonSpec(mode="overhead", gates=E2E_GATE)
    )
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and gate.reason == "observer_not_active"
    assert comparison.observer_issues


def test_an_incremental_candidate_declares_what_it_adds() -> None:
    baseline, _ = _arms(SAME)
    added = {"vllm_spans": _observer(True)}
    candidate = [_run("candidate", i, e, observers=added) for i, e in enumerate(SAME)]
    with pytest.raises(InferInputError, match="--added-observers: vllm_spans"):
        compare_runs(baseline, candidate, ComparisonSpec(mode="incremental"))
    declared = compare_runs(
        baseline,
        candidate,
        ComparisonSpec(mode="incremental", added_observers=("vllm_spans",)),
    )
    assert declared.observer_issues == []


def test_a_penalized_quantile_cannot_be_gated() -> None:
    baseline, _ = _arms(SAME)
    candidate = [
        _run("candidate", i, e, penalized=(i == 2)) for i, e in enumerate(SAME)
    ]
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and (gate.status, gate.reason) == (
        "not_evaluable",
        "penalized",
    )


def test_slo_metrics_need_full_evidence_unless_a_floor_is_declared() -> None:
    baseline, _ = _arms(SAME)
    candidate = [_run("candidate", i, e, coverage=0.97) for i, e in enumerate(SAME)]
    gates = (("goodput_rps", GateRule("non-inferiority", 0.05, "relative")),)
    strict = compare_runs(baseline, candidate, ComparisonSpec(gates=gates))
    gate = strict.cases[CASE]["metrics"]["goodput_rps"].gate
    assert gate is not None and gate.reason == "evidence_coverage_below_floor"
    lenient = compare_runs(
        baseline, candidate, ComparisonSpec(gates=gates, evidence_floor=0.95)
    )
    gate = lenient.cases[CASE]["metrics"]["goodput_rps"].gate
    assert gate is not None and gate.status == "pass"


@pytest.mark.parametrize(("missing", "status"), [(0, "pass"), (1, "fail")])
def test_min_attainment_is_a_claim_about_runs(missing: int, status: str) -> None:
    baseline, _ = _arms(SAME)
    candidate = [
        _run("candidate", i, e, attainment=0.98 if i < missing else 0.995)
        for i, e in enumerate(SAME)
    ]
    comparison = compare_runs(
        baseline, candidate, ComparisonSpec(min_attainment=0.99, min_run_pass=0.5)
    )
    gate = comparison.cases[CASE]["attainment_gate"]
    assert (gate["runs_meeting"], gate["runs"], gate["status"]) == (
        6 - missing,
        6,
        status,
    )
    assert gate["model"] == "independent_runs"
    assert comparison.cases[CASE]["attainment_mean"]["model"] == "normal_run_means"


def test_a_candidate_run_whose_slo_was_not_judged_does_not_meet_it() -> None:
    # Dropping the unmeasurable runs would keep only the ones that met it.
    baseline, _ = _arms(SAME + SAME[:4])
    candidate = [_run("candidate", i, 100.0, attainment=0.995) for i in range(6)] + [
        _run("candidate", 6 + i, 100.0, attainment=0.995) for i in range(4)
    ]
    for run in candidate[6:]:
        run.report["cases"][CASE]["slo"]["attainment_lower"] = None
    comparison = compare_runs(
        baseline, candidate, ComparisonSpec(min_attainment=0.99, min_run_pass=0.5)
    )
    gate = comparison.cases[CASE]["attainment_gate"]
    assert (gate["runs_meeting"], gate["runs"], gate["runs_unmeasurable"]) == (6, 10, 4)
    assert gate["status"] == "fail"


def test_an_attainment_gate_that_cannot_be_evaluated_exits_4() -> None:
    baseline, _ = _arms(SAME)
    candidate = [_run("candidate", i, e) for i, e in enumerate(SAME)]
    for run in candidate:
        del run.report["cases"][CASE]["slo"]
    comparison = compare_runs(baseline, candidate, ComparisonSpec(min_attainment=0.99))
    assert comparison.cases[CASE]["attainment_gate"]["status"] == "not_evaluable"
    assert (CASE, "min_attainment", "no_slo_evaluation") in comparison.not_evaluable
    assert comparison.exit_code == 4


def test_the_attainment_gate_obeys_the_same_blockers_as_the_others() -> None:
    # Arms not shown to measure one server cannot pass any gate.
    baseline, _ = _arms(SAME)
    unknown = _fields()
    del unknown["model.weights_digest"]
    candidate = [
        _run("candidate", i, e, fields=unknown, attainment=1.0)
        for i, e in enumerate(SAME)
    ]
    comparison = compare_runs(baseline, candidate, ComparisonSpec(min_attainment=0.99))
    gate = comparison.cases[CASE]["attainment_gate"]
    assert (gate["status"], gate["reason"]) == ("not_evaluable", "unverified")
    assert comparison.exit_code == 4


def test_slo_metrics_judged_by_different_policies_cannot_be_gated() -> None:
    # A candidate judged by a looser policy would meet it however much
    # slower it was.
    gates = (("goodput_rps", GateRule("non-inferiority", 0.05, "relative")),)
    baseline, candidate = _arms(SLOWER, slo_digest="q" * 64)
    comparison = compare_runs(
        baseline, candidate, ComparisonSpec(gates=gates, min_attainment=0.9)
    )
    case = comparison.cases[CASE]
    goodput = case["metrics"]["goodput_rps"]
    assert goodput.gate is not None
    assert (goodput.gate.status, goodput.gate.reason) == (
        "not_evaluable",
        "slo_policy_differs",
    )
    assert case["attainment_gate"]["reason"] == "slo_policy_differs"
    # Latency is not judged by a policy and is still gated.
    assert case["metrics"]["client.e2e.p95"].gate is None


def test_pooled_requests_are_labelled_model_based() -> None:
    baseline, _ = _arms(SAME)
    candidate = [_run("candidate", i, e, attainment=1.0) for i, e in enumerate(SAME)]
    comparison = compare_runs(
        baseline,
        candidate,
        ComparisonSpec(min_attainment=0.99, attainment_model="bernoulli"),
    )
    gate = comparison.cases[CASE]["attainment_gate"]
    assert gate["model"] == "independent_requests" and gate["model_based"] is True


def test_any_regression_needs_the_significant_rule_and_uses_holm() -> None:
    with pytest.raises(InferUsageError, match="significant"):
        ComparisonSpec(family="any_regression", gates=E2E_GATE)
    gates = (("client.e2e.*", GateRule("significant", 0.05, "relative")),)
    comparison = compare_runs(
        *_arms(SLOWER), ComparisonSpec(family="any_regression", gates=gates)
    )
    assert comparison.family is not None and comparison.family["method"] == "holm"
    assert comparison.family["tests"] == 4
    assert "c1/client.e2e.p95" in comparison.family["rejected"]
    assert (CASE, "client.e2e.p95") in comparison.failed


def _blocks_of(values: list[float], arm: str) -> list[RunSummary]:
    return [_run(arm, i, value, started=i) for i, value in enumerate(values)]


# Each block about 5% slower; a one-sided p of about 0.008 per test. Four
# identical tests (p50..p99 share the fixture's value): Holm at one-sided
# 0.025 needs p <= 0.00625 for the first, so none is rejected; unadjusted,
# or Holm at a two-sided 0.05 (0.0125), would reject all four.
HOLM_CANDIDATE = [108.2176, 102.1249, 106.6611, 103.6151, 109.7967, 100.6561]


def test_holm_adjusts_the_family_at_the_one_sided_level() -> None:
    gates = (("client.e2e.*", GateRule("significant", 0.0, "relative")),)
    comparison = compare_runs(
        _blocks_of([100.0] * 6, "baseline"),
        _blocks_of(HOLM_CANDIDATE, "candidate"),
        ComparisonSpec(gates=gates, family="any_regression"),
    )
    assert comparison.family is not None and comparison.family["tests"] == 4
    assert comparison.family["rejected"] == []
    assert comparison.exit_code == 0
    plain = compare_runs(
        _blocks_of([100.0] * 6, "baseline"),
        _blocks_of(HOLM_CANDIDATE, "candidate"),
        ComparisonSpec(gates=gates),
    )
    assert plain.exit_code == 4  # each alone is a significant regression


def test_holm_still_needs_the_estimate_beyond_the_budget() -> None:
    # Overwhelming evidence of a 5% slowdown, against a 10% budget.
    gates = (("client.e2e.*", GateRule("significant", 0.10, "relative")),)
    comparison = compare_runs(
        _blocks_of([100.0] * 6, "baseline"),
        _blocks_of([105.0, 105.1, 104.9, 105.05, 104.95, 105.0], "candidate"),
        ComparisonSpec(gates=gates, family="any_regression"),
    )
    assert comparison.family is not None and comparison.family["rejected"]
    assert comparison.exit_code == 0


def test_an_insufficient_tail_cannot_be_gated() -> None:
    baseline, candidate = _arms(SLOWER)
    for run in candidate:
        levels = run.report["cases"][CASE]["latency"]["metrics"]["client.e2e"]
        levels["failure_penalized"]["p95"]["sufficient"] = False
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None
    assert (gate.status, gate.reason) == ("not_evaluable", "insufficient_tail_samples")


def test_more_failures_in_the_candidate_is_worse() -> None:
    gates = (("failure_fraction", GateRule("non-inferiority", 0.01, "fraction")),)
    baseline, candidate = _arms(SAME)
    for i, run in enumerate(candidate):
        run.report["cases"][CASE]["population"]["successful"] = 90 + i % 2
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=gates))
    metric = comparison.cases[CASE]["metrics"]["failure_fraction"]
    assert metric.direction == "lower_is_better" and metric.unit == "fraction"
    assert metric.gate is not None and metric.gate.status == "fail"


def test_shares_are_compared_in_fraction_units() -> None:
    metrics = {metric.name: metric for metric in default_metrics(_case(100.0))}
    assert metrics["attainment"].unit == metrics["failure_fraction"].unit == "fraction"
    assert metrics["goodput_rps"].unit == "relative"


def test_arms_run_one_after_the_other_are_flagged() -> None:
    baseline, candidate = _arms(SAME)
    candidate = [_run("candidate", i, e, started=100 + i) for i, e in enumerate(SAME)]
    comparison = compare_runs(baseline, candidate, ComparisonSpec())
    assert any("confounded" in w for w in comparison.diagnostics["warnings"])
    interleaved = compare_runs(*_arms(SAME), ComparisonSpec())
    assert interleaved.diagnostics["warnings"] == []


def test_a_regression_that_crosses_the_slo_shows_in_goodput_and_attainment(
    tmp_path: Any,
) -> None:
    """End to end: profiled runs, a slower candidate, an SLO it misses."""
    slo = parse_slo_flags(["e2e:100"])

    def runs(arm: str, latency: float) -> list[RunSummary]:
        summaries = []
        for block in range(3):
            directory = tmp_path / f"{arm}{block}"
            directory.mkdir()
            run_profile_with_fake_client(
                directory,
                latency_seconds=latency,
                request_count=6,
                slo=slo,
                slo_source="flags",
                labels={"experiment": "e", "arm": arm, "block": str(block)},
            )
            summaries.append(summarize_run(directory / "infer.jsonl"))
        return summaries

    baseline, candidate = runs("baseline", 0.01), runs("candidate", 0.15)
    gates = (
        ("goodput_rps", GateRule("non-inferiority", 0.05, "relative")),
        ("attainment", GateRule("non-inferiority", 0.01, "fraction")),
    )
    comparison = compare_runs(
        baseline, candidate, ComparisonSpec(gates=gates, allow_not_evaluable=True)
    )
    (case_id,) = comparison.cases
    metrics = comparison.cases[case_id]["metrics"]
    goodput = metrics["goodput_rps"]
    assert goodput.reason == "candidate_zero"
    assert goodput.gate is not None and goodput.gate.status == "fail"
    attainment = metrics["attainment"]
    assert attainment.worst is not None and attainment.worst.effect == pytest.approx(
        -1.0
    )
    assert comparison.exit_code == 4
