"""Comparing a baseline arm of runs with a candidate arm."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.compare import ComparisonSpec, compare_runs
from stormlog.infer.compare_metrics import default_metrics
from stormlog.infer.compare_report import comparison_lines
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
        "workload.realization_digest": "r" * 64,
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
    protocol: tuple[str, ...] = (),
    evidence: dict[str, str] | None = None,
    **case: Any,
) -> RunSummary:
    labels = (
        {} if block is None else {"experiment": "e", "arm": arm, "block": str(block)}
    )
    return RunSummary(
        path=None,
        sha256=None,
        run_id=f"{arm}-{block}-{e2e}" if block is not None else f"{arm}-{started}",
        session_id="s",
        session_status=status,
        labels=labels,
        fields=fields or _fields(),
        report={
            "cases": {CASE: _case(e2e, **case)},
            "observers": {"observers": observers or {}},
        },
        protocol_failures=protocol,
        started_at_ns=started,
        outcome_failures=() if status == "completed" else (f"session_{status}",),
        external_evidence=evidence or {},
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


def test_a_case_no_run_has_is_a_usage_error() -> None:
    with pytest.raises(InferUsageError, match="case typo is in no run"):
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


def test_a_retried_block_keeps_the_last_attempt_and_marks_the_first() -> None:
    # The runner retries a run an external cause set aside; the first
    # attempt is listed, not a second run of the arm in that block.
    baseline, candidate = _arms(SAME)
    preempted = _run(
        "candidate",
        2,
        100.0,
        status="interrupted",
        protocol=("external:preempted",),
        evidence={"external:preempted": "paused"},
    )
    comparison = compare_runs(baseline, [preempted, *candidate], ComparisonSpec())
    assert comparison.excluded == [
        {
            "arm": "candidate",
            "run": preempted.name,
            "case": CASE,
            "reasons": ["superseded", "external:preempted"],
            "evidence": {"external:preempted": "paused"},
            "attempt_kept": candidate[2].name,
        }
    ]
    assert comparison.cases[CASE]["metrics"]["client.e2e.p95"].n_pairs == 6
    assert (
        f"Set aside: {preempted.name} for c1 (superseded, external:preempted; "
        f"evidence: paused; kept {candidate[2].name})"
    ) in comparison_lines(comparison)


def test_a_set_aside_lists_the_evidence_for_its_external_cause() -> None:
    baseline, candidate = _arms(SAME)
    candidate[2] = _run(
        "candidate",
        2,
        100.0,
        status="interrupted",
        started=5,
        protocol=("external:spot_preemption",),
        evidence={"external:spot_preemption": "box paused without a release"},
    )
    comparison = compare_runs(baseline, candidate, ComparisonSpec())
    item = next(i for i in comparison.excluded if i["run"] == candidate[2].name)
    assert item["evidence"] == {
        "external:spot_preemption": "box paused without a release"
    }
    assert (
        f"Set aside: {candidate[2].name} for c1 (external:spot_preemption; "
        "evidence: box paused without a release)"
    ) in comparison_lines(comparison)


def test_a_block_run_twice_keeps_the_last_attempt() -> None:
    baseline, candidate = _arms(SAME)
    again = _run("candidate", 0, 100.0, started=100)
    comparison = compare_runs(baseline, [*candidate, again], ComparisonSpec())
    assert [
        (item["run"], item["reasons"], item["attempt_kept"])
        for item in comparison.excluded
    ] == [(candidate[0].name, ["superseded"], again.name)]


def test_a_run_given_twice_is_refused() -> None:
    baseline, candidate = _arms(SAME)
    with pytest.raises(InferInputError, match="is given twice"):
        compare_runs(baseline, [*candidate, candidate[0]], ComparisonSpec())


def test_a_retry_never_replaces_an_outcome_failure() -> None:
    # Otherwise a treatment that crashes half its runs passes once retried.
    baseline, candidate = _arms(SAME)
    crashed = _run("candidate", 2, 140.0, status="interrupted", started=5)
    retry = _run("candidate", 2, 100.0, started=100)
    candidate[2] = crashed
    comparison = compare_runs(baseline, [*candidate, retry], ComparisonSpec())
    assert [
        (item["run"], item["reasons"], item["attempt_kept"])
        for item in comparison.excluded
    ] == [(retry.name, ["retry_of_outcome_failure"], crashed.name)]


def _with_segment(runs: list[RunSummary], membership: str) -> None:
    for run in runs:
        case = run.report["cases"][CASE]
        segment = {key: value for key, value in case.items() if key != "segments"}
        segment["intervals"] = {
            "membership": membership,
            "rate_reason": None if membership == "arrival" else "overlapping_cohort",
        }
        case["segments"] = {"early": segment}


@pytest.mark.parametrize("membership", ["arrival", "overlap"])
def test_an_overlap_segment_is_diagnostics_and_carries_no_gate(
    membership: str,
) -> None:
    # Requests in flight during a segment are length-biased: their rates
    # grow, and their quantiles stretch, with the arm's own latency.
    baseline, candidate = _arms(SAME)
    _with_segment(baseline, membership)
    _with_segment(candidate, membership)
    spec = ComparisonSpec(gates=E2E_GATE, min_attainment=0.9)
    comparison = compare_runs(baseline, candidate, spec)
    whole = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert whole is not None and whole.status == "pass"
    segment = comparison.cases[f"{CASE}/early"]
    gate = segment["metrics"]["client.e2e.p95"].gate
    if membership == "arrival":
        assert gate is not None and gate.status == "pass"
        assert "attainment_gate" in segment
        return
    assert segment["membership"] == "overlap" and segment["gated"] is False
    assert all(metric.gate is None for metric in segment["metrics"].values())
    assert segment["absent_gates"] == {} and "attainment_gate" not in segment
    lines = comparison_lines(comparison)
    assert f"- {CASE}/early (overlap: diagnostics only, not gated):" in lines


@pytest.mark.parametrize(
    "spec",
    [
        ComparisonSpec(gates=E2E_GATE, cases=(f"{CASE}/early",)),
        ComparisonSpec(min_attainment=0.9, cases=(f"{CASE}/early",)),
    ],
    ids=["gate", "min_attainment"],
)
def test_gates_asked_only_of_overlap_segments_never_pass_vacuously(
    spec: ComparisonSpec,
) -> None:
    baseline, candidate = _arms(SAME)
    _with_segment(baseline, "overlap")
    _with_segment(candidate, "overlap")
    with pytest.raises(InferUsageError, match="diagnostics only"):
        compare_runs(baseline, candidate, spec)


def test_a_case_no_run_offered_a_request_is_refused() -> None:
    # Such as a segment outside every run's measured phase.
    baseline, candidate = _arms(SLOWER)
    for run in [*baseline, *candidate]:
        run.report["cases"][CASE]["population"]["offered"] = 0
    with pytest.raises(InferInputError, match="c1 holds no request in any run"):
        compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))


def test_a_fallback_follows_a_pattern_gate_to_each_metric_it_names() -> None:
    gates = (("goodput*", GateRule("non-inferiority", 0.05, "relative")),)
    spec = ComparisonSpec(
        gates=gates, fallbacks=(("goodput_rps", 0.5, "requests_per_second"),)
    )
    rule = spec.gate_for("goodput_rps")
    assert rule is not None
    assert (rule.fallback_budget, rule.fallback_unit) == (0.5, "requests_per_second")


def test_an_infinite_candidate_latency_fails_as_the_worst_value() -> None:
    baseline, candidate = _arms(SLOWER[:3] + [100.0] * 3)
    for run in candidate[3:]:
        estimate = run.report["cases"][CASE]["latency"]["metrics"]["client.e2e"][
            "failure_penalized"
        ]["p95"]
        estimate["value_ms"] = float("inf")
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None
    assert (gate.status, gate.reason) == ("fail", "candidate_censored_worst")


def test_a_block_whose_runs_sent_different_workloads_is_set_aside() -> None:
    # Pairing assumes both runs of a block sent the same realization (the
    # block's seed); a block that did not is no pair.
    baseline, candidate = _arms(SLOWER)
    other = _fields(**{"workload.realization_digest": "other"})
    candidate[1] = _run("candidate", 1, SLOWER[1], fields=other, started=3)
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    reasons = {
        (item["arm"], item["run"]): item["reasons"] for item in comparison.excluded
    }
    assert reasons == {
        ("baseline", baseline[1].name): ["block_realization_differs"],
        ("candidate", candidate[1].name): ["block_realization_differs"],
    }
    assert comparison.cases[CASE]["metrics"]["client.e2e.p95"].n_pairs == 5


def test_an_unfinished_run_is_kept_as_data() -> None:
    # The treatment may have caused it: setting it aside would let a
    # treatment that crashes runs pass on the blocks left.
    baseline, candidate = _arms(SAME)
    candidate[5] = _run("candidate", 5, 100.0, status="interrupted", started=11)
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    metric = comparison.cases[CASE]["metrics"]["client.e2e.p95"]
    assert comparison.excluded == []
    assert metric.n_pairs == 6
    assert metric.gate is not None and metric.gate.status == "pass"


def _lose_case(run: RunSummary, how: str) -> None:
    if how == "missing":
        del run.report["cases"][CASE]
    else:
        run.report["cases"][CASE]["latency"] = _latency(0.0, penalized=True)


@pytest.mark.parametrize("how", ["missing", "unreadable"])
def test_an_outcome_the_candidate_lost_fails_its_gates(how: str) -> None:
    baseline, candidate = _arms(SAME)
    candidate[3] = _run("candidate", 3, 100.0, status="interrupted", started=7)
    _lose_case(candidate[3], how)
    spec = ComparisonSpec(gates=E2E_GATE, min_attainment=0.9)
    comparison = compare_runs(baseline, candidate, spec)
    metric = comparison.cases[CASE]["metrics"]["client.e2e.p95"]
    assert metric.gate is not None
    assert (metric.gate.status, metric.gate.reason) == ("fail", "outcome_unrecoverable")
    assert comparison.excluded == []
    if how == "missing":
        # It cannot have met the target either.
        assert comparison.cases[CASE]["attainment_gate"]["runs_unmeasurable"] == 1
    assert comparison.exit_code == 4


def test_a_metric_every_crashed_run_lost_says_so_gated_or_not() -> None:
    # With no candidate value left, the statistics alone would read
    # insufficient_blocks, hiding why.
    baseline, candidate = _arms(SAME)
    candidate = [
        _run("candidate", i, 100.0, status="interrupted", started=2 * i + 1)
        for i in range(6)
    ]
    for run in candidate:
        _lose_case(run, "unreadable")
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    metrics = comparison.cases[CASE]["metrics"]
    assert metrics["client.e2e.p50"].reason == "outcome_unrecoverable"
    assert metrics["client.e2e.p95"].reason == "outcome_unrecoverable"
    gate = metrics["client.e2e.p95"].gate
    assert gate is not None and gate.status == "fail"


def test_a_case_a_completed_candidate_run_lacks_is_an_outcome() -> None:
    baseline, candidate = _arms(SAME)
    del candidate[1].report["cases"][CASE]
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None
    assert (gate.status, gate.reason) == ("fail", "outcome_unrecoverable")


def test_an_outcome_the_baseline_lost_cannot_be_evaluated() -> None:
    baseline, candidate = _arms(SAME)
    baseline[3] = _run("baseline", 3, 100.0, status="interrupted", started=6)
    _lose_case(baseline[3], "missing")
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    metric = comparison.cases[CASE]["metrics"]["client.e2e.p95"]
    assert metric.gate is not None
    assert (metric.gate.status, metric.gate.reason) == (
        "not_evaluable",
        "control_failed",
    )
    assert metric.reason == "control_failed"


def test_a_broken_baseline_never_passes_the_candidate() -> None:
    # A baseline run that crashed slow makes any candidate look better: its
    # contrasts cannot be judged, even with every value present.
    baseline, candidate = _arms(SLOWER)
    baseline[3] = _run("baseline", 3, 400.0, status="interrupted", started=6)
    spec = ComparisonSpec(gates=E2E_GATE, min_attainment=0.9)
    comparison = compare_runs(baseline, candidate, spec)
    case = comparison.cases[CASE]
    gate = case["metrics"]["client.e2e.p95"].gate
    assert gate is not None
    assert (gate.status, gate.reason) == ("not_evaluable", "control_failed")
    # A claim about the candidate's runs alone does not lean on the baseline.
    assert case["attainment_gate"]["status"] == "pass"
    assert comparison.exit_code == 4


def test_a_bernoulli_attainment_gate_fails_on_a_lost_candidate_case() -> None:
    # Pooling requests would drop the run's requests from both counts.
    baseline, candidate = _arms(SAME)
    candidate[3] = _run("candidate", 3, 100.0, status="interrupted", started=7)
    _lose_case(candidate[3], "missing")
    spec = ComparisonSpec(min_attainment=0.9, attainment_model="bernoulli")
    gate = compare_runs(baseline, candidate, spec).cases[CASE]["attainment_gate"]
    assert (gate["status"], gate["reason"]) == ("fail", "outcome_unrecoverable")


@pytest.mark.parametrize("paired", [True, False])
def test_an_external_cause_sets_aside_its_block_and_the_preregistered_count_decides(
    paired: bool,
) -> None:
    baseline, candidate = _arms(SAME, paired=paired)
    block = 2 if paired else None
    candidate[2] = _run(
        "candidate",
        block,
        100.0,
        status="interrupted",
        started=5,
        protocol=("external:preempted",),
    )
    spec = ComparisonSpec(gates=E2E_GATE, min_attainment=0.9)
    one = compare_runs(baseline, candidate, spec)
    reasons = {(item["arm"], item["run"]): item["reasons"] for item in one.excluded}
    expected = {("candidate", candidate[2].name): ["external:preempted"]}
    if paired:
        expected[("baseline", baseline[2].name)] = ["block_set_aside"]
    assert reasons == expected
    metric = one.cases[CASE]["metrics"]["client.e2e.p95"]
    assert metric.gate is not None and metric.gate.status == "pass"
    assert one.cases[CASE]["set_aside"] == {
        "unit": "block" if paired else "run",
        "items": ["e/2" if paired else candidate[2].name],
    }
    # What a gate wrapper reads: the payload, not the object.
    payload = one.to_payload()["cases"][CASE]
    assert payload["set_aside"] == one.cases[CASE]["set_aside"]
    assert (payload["membership"], payload["gated"]) == (None, True)

    # Not a fixed count: the gate's pre-registered min_complete_blocks.
    baseline[4] = _run(
        "baseline",
        4 if paired else None,
        100.0,
        started=8,
        protocol=("external:operator_abort",),
    )
    two = compare_runs(baseline, candidate, spec)
    gate = two.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and gate.status == "pass"
    # Six planned: four complete pairs, or five runs an arm, are too few.
    planned = GateRule("non-inferiority", 0.05, "relative", min_complete_blocks=6)
    strict = compare_runs(
        baseline, candidate, ComparisonSpec(gates=(("client.e2e.p95", planned),))
    )
    gate = strict.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None
    assert (gate.status, gate.reason) == ("not_evaluable", "blocks_below_preregistered")


def test_a_block_given_for_one_arm_only_counts_as_lost() -> None:
    # Its other run is gone with no cause recorded: it is listed, and the
    # pre-registered block count keeps the contrast from shrinking quietly.
    baseline, candidate = _arms(SAME)
    one = compare_runs(baseline, candidate[:5], ComparisonSpec(gates=E2E_GATE))
    assert one.cases[CASE]["set_aside"]["items"] == ["e/5"]
    planned = GateRule("non-inferiority", 0.05, "relative", min_complete_blocks=6)
    spec = ComparisonSpec(gates=(("client.e2e.p95", planned),))
    gate = (
        compare_runs(baseline, candidate[:5], spec)
        .cases[CASE]["metrics"]["client.e2e.p95"]
        .gate
    )
    assert gate is not None
    assert (gate.status, gate.reason) == ("not_evaluable", "blocks_below_preregistered")


def test_a_set_aside_run_fails_its_contrasts_when_asked() -> None:
    baseline, candidate = _arms(SAME)
    candidate[5] = _run(
        "candidate", 5, 100.0, started=11, protocol=("probe_incomplete",)
    )
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


def test_an_allowed_difference_is_allowed_within_an_arm_too() -> None:
    # On the A30 box, the first launch of a model after a resume compiled
    # cold and got a smaller KV cache; the error says to allow the field,
    # so allowing it must work for the runs of one arm as well.
    baseline, candidate = _arms(SAME)
    cold = _fields(**{"effective.kv_cache_size_tokens": 855088})
    warm = _fields(**{"effective.kv_cache_size_tokens": 890960})
    baseline = [
        _run("baseline", i, e, fields=cold if i == 0 else warm, started=2 * i)
        for i, e in enumerate(BASE_E2E)
    ]
    candidate = [
        _run("candidate", i, e, fields=warm, started=2 * i + 1)
        for i, e in enumerate(SAME)
    ]
    with pytest.raises(InferInputError, match="allow a difference with --allow"):
        compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    allowed = ComparisonSpec(gates=E2E_GATE, allow=("effective.kv_cache_size_tokens",))
    comparison = compare_runs(baseline, candidate, allowed)
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and gate.status == "pass"


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
    baseline = [_run("baseline", i, 100.0, started=2 * i) for i in range(10)]
    candidate = [
        _run("candidate", i, 100.0, attainment=0.995, started=2 * i + 1)
        for i in range(10)
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


@pytest.mark.parametrize(
    "spec",
    [
        ComparisonSpec(
            gates=(("goodput_rps", GateRule("non-inferiority", 0.05, "relative")),)
        ),
        ComparisonSpec(min_attainment=0.9),
    ],
    ids=["goodput_gate", "min_attainment"],
)
def test_an_slo_gate_across_different_policies_is_invalid_input(
    spec: ComparisonSpec,
) -> None:
    # A candidate judged by a looser policy would meet it however much
    # slower it was: give one policy with --slo to judge both.
    baseline, candidate = _arms(SLOWER, slo_digest="q" * 64)
    with pytest.raises(InferInputError, match="judged by different SLO policies"):
        compare_runs(baseline, candidate, spec)


def test_ungated_slo_metrics_across_policies_say_why_they_are_not_compared() -> None:
    baseline, candidate = _arms(SLOWER, slo_digest="q" * 64)
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=E2E_GATE))
    assert comparison.cases[CASE]["metrics"]["goodput_rps"].reason == (
        "slo_policy_differs"
    )
    gate = comparison.cases[CASE]["metrics"]["client.e2e.p95"].gate
    assert gate is not None and gate.status == "fail"


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


FAILURE_GATE = (("failure_fraction", GateRule("non-inferiority", 0.01, "fraction")),)


def _served(runs: list[RunSummary], offered: int, failed: list[int]) -> None:
    for run, count in zip(runs, failed):
        population = run.report["cases"][CASE]["population"]
        population.update(offered=offered, successful=offered - count)


def test_more_failures_in_the_candidate_is_worse() -> None:
    # Judged as a claim about runs: each candidate run against its block.
    baseline, candidate = _arms(SAME)
    _served(baseline, 1000, [2] * 6)
    _served(candidate, 1000, [100, 101, 100, 101, 100, 101])
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=FAILURE_GATE))
    metric = comparison.cases[CASE]["metrics"]["failure_fraction"]
    assert metric.direction == "lower_is_better" and metric.unit == "fraction"
    assert metric.gate is not None
    assert (metric.gate.status, metric.gate.case) == ("fail", "run_level")
    assert metric.gate.claim is not None
    assert metric.gate.claim["runs_within_budget"] == 0
    assert metric.pooled is not None and metric.pooled["candidate"]["n"] == 6000
    # Stated as a claim about runs, never as a bound on the fraction.
    (line,) = [
        line
        for line in comparison_lines(comparison)
        if line.strip().startswith("failure_fraction")
    ]
    assert "(descriptive)" in line
    assert "0 of 6 candidate runs within 0.01 of the block baseline" in line


def test_a_fraction_gate_needs_enough_requests_in_every_run() -> None:
    # 100 requests per run: one failure is a whole point, past a 1% budget.
    baseline, candidate = _arms(SAME)
    comparison = compare_runs(baseline, candidate, ComparisonSpec(gates=FAILURE_GATE))
    gate = comparison.cases[CASE]["metrics"]["failure_fraction"].gate
    assert gate is not None
    assert (gate.status, gate.reason) == ("not_evaluable", "too_few_requests_per_run")


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
