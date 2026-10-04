"""Comparing one metric between two arms of runs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from scipy import stats

from stormlog.infer.comparison_stats import (
    GateRule,
    blocks_for_precision,
    clopper_pearson,
    compare_values,
    holm,
    run_pass_gate,
    skew_limit,
)
from stormlog.infer.errors import InferUsageError

BLOCKS = ["b1", "b2", "b3", "b4", "b5", "b6"]
LATENCY_A = [100.0, 104.0, 98.0, 101.0, 103.0, 99.0]
LATENCY_B = [110.0, 113.0, 107.0, 112.0, 111.0, 108.0]


def _latency(**options: Any) -> Any:
    return compare_values(
        "client.e2e.p95",
        options.pop("baseline", LATENCY_A),
        options.pop("candidate", LATENCY_B),
        direction=options.pop("direction", "lower_is_better"),
        scale=options.pop("scale", "log_ratio"),
        unit=options.pop("unit", "relative"),
        blocks=options.pop("blocks", (BLOCKS, BLOCKS)),
        **options,
    )


def _t(confidence: float, df: float) -> float:
    return float(stats.t.ppf(0.5 + confidence / 2, df))


def test_paired_log_ratios_give_a_relative_effect() -> None:
    result = _latency()
    d = np.log(LATENCY_B) - np.log(LATENCY_A)
    half = _t(0.95, 5) * d.std(ddof=1) / math.sqrt(6)

    assert (result.design, result.n_pairs) == ("paired_blocks", 6)
    assert result.worst is not None and result.worst == result.best
    assert result.worst.method == "paired_t_log_ratio"
    assert result.worst.estimand == "geometric_ratio"
    assert result.worst.effect == pytest.approx(math.expm1(d.mean()))
    assert result.worst.lower == pytest.approx(math.expm1(d.mean() - half))
    assert result.worst.upper == pytest.approx(math.expm1(d.mean() + half))
    assert result.worst.df == 5
    assert result.verdict == {"direction": "worse", "tolerance": "beyond"}
    # The arithmetic ratio, by Fieller, is only a cross-check.
    assert result.cross_check is not None
    assert result.cross_check.estimand == "arithmetic_ratio"
    assert result.cross_check.lower is not None
    # The difference is reported in the values' own unit, never gated.
    assert result.difference is not None
    assert result.difference.effect == pytest.approx(
        np.mean(LATENCY_B) - np.mean(LATENCY_A)
    )
    assert result.effect_size == {
        "name": "paired_superiority",
        "value": 1.0,
        "count": 6,
    }


def test_independent_runs_use_welch_with_the_smaller_arms_df() -> None:
    # An adversarial shape: 20 baseline runs against 3 candidate runs.
    rng = np.random.default_rng(20)
    baseline = list(np.exp(np.log(100) + 0.1 * rng.standard_normal(20)))
    candidate = [1.21, 0.95, 1.07]
    result = compare_values(
        "throughput",
        baseline,
        candidate,
        direction="higher_is_better",
        scale="log_ratio",
        unit="relative",
    )
    la, lb = np.log(baseline), np.log(candidate)
    se = math.sqrt(la.var(ddof=1) / 20 + lb.var(ddof=1) / 3)
    diff = lb.mean() - la.mean()

    assert result.design == "independent"
    assert result.worst is not None
    assert result.worst.method == "welch_t_min_df_log_ratio"
    assert result.worst.df == 2
    assert result.worst.lower == pytest.approx(math.expm1(diff - _t(0.95, 2) * se))
    assert result.effect_size is not None and result.effect_size["name"] == "a12"
    assert result.effect_size["count"] == 60


def test_a_difference_metric_stays_in_its_unit() -> None:
    attainment_a = [0.99, 0.985, 0.992, 0.99, 0.988, 0.991]
    attainment_b = [0.97, 0.972, 0.969, 0.975, 0.971, 0.968]
    result = _latency(
        baseline=attainment_a,
        candidate=attainment_b,
        direction="higher_is_better",
        scale="difference",
        unit="fraction",
        trials=([1000] * 6, [1000] * 6),
        gate=GateRule("non-inferiority", 0.01, "fraction"),
    )
    diffs = np.subtract(attainment_b, attainment_a)
    assert result.worst is not None
    assert result.worst.effect == pytest.approx(diffs.mean())
    assert result.gate is not None and result.gate.status == "fail"


def _fractions(candidate: Sequence[float | None], **options: Any) -> Any:
    n = len(candidate)
    blocks = [f"b{i}" for i in range(n)]
    return compare_values(
        "failure_fraction",
        options.pop("baseline", [0.0] * n),
        candidate,
        direction="lower_is_better",
        scale="difference",
        unit="fraction",
        blocks=options.pop("blocks", (blocks, blocks)),
        trials=options.pop("trials", ([300] * n, [300] * n)),
        gate=options.pop("gate", GateRule("non-inferiority", 0.01, "fraction")),
        **options,
    )


@pytest.mark.parametrize(
    ("within", "beyond", "status", "lower"),
    [
        (6, 0, "pass", 0.025 ** (1 / 6)),
        (5, 1, "fail", float(stats.beta.ppf(0.025, 5, 2))),
        (8, 0, "pass", 0.025 ** (1 / 8)),
        (7, 1, "fail", float(stats.beta.ppf(0.025, 7, 2))),
        (9, 1, "pass", float(stats.beta.ppf(0.025, 9, 2))),
    ],
)
def test_a_fraction_is_gated_on_the_share_of_runs_within_its_budget(
    within: int, beyond: int, status: str, lower: float
) -> None:
    # Exact under independent runs, whatever the correlation of failures
    # within a run (fable-213's ruling, opus-221's D32).
    result = _fractions([0.005] * within + [0.02] * beyond)
    gate = result.gate
    assert gate is not None and (gate.status, gate.case) == (status, "run_level")
    assert gate.reason == (None if status == "pass" else "too_few_runs_within_budget")
    assert gate.claim is not None
    assert (gate.claim["runs_within_budget"], gate.claim["runs"]) == (
        within,
        within + beyond,
    )
    assert gate.claim["run_pass_lower"] == pytest.approx(lower)
    assert gate.claim["model"] == "independent_runs"
    assert "run-pass rate" in gate.claim["statement"]


def test_a_run_is_judged_against_its_own_blocks_baseline() -> None:
    # 0.015 against a block baseline of 0.01 is within a 0.01 budget; 0.025
    # against 0.01 is not.
    within = _fractions([0.015] * 8, baseline=[0.01] * 8)
    assert within.gate is not None and within.gate.status == "pass"
    beyond = _fractions([0.015] * 7 + [0.025], baseline=[0.01] * 8)
    assert beyond.gate is not None and beyond.gate.status == "fail"
    # Without blocks, each candidate run against the baseline arm's mean.
    unpaired = _fractions([0.015] * 8, baseline=[0.0, 0.02] * 4, blocks=None)
    assert unpaired.gate is not None and unpaired.gate.status == "pass"
    assert unpaired.gate.claim is not None
    assert unpaired.gate.claim["reference"] == "baseline_mean"


def test_an_unmeasurable_candidate_run_is_a_miss() -> None:
    result = _fractions([0.0] * 7 + [None])
    assert result.gate is not None and result.gate.status == "fail"
    assert result.gate.claim is not None
    assert result.gate.claim["runs_unmeasurable"] == 1


def test_all_zero_fractions_pass_as_a_claim_about_runs_not_a_bound() -> None:
    # No failure anywhere: the claim is that runs stay within the budget, not
    # that the failure rate is bounded; the interval is floored, not [0, 0].
    result = _fractions([0.0] * 8)
    assert result.gate is not None and result.gate.status == "pass"
    assert result.worst is not None
    p = 1 / (2400 + 2)
    floor = math.sqrt(2 * p * (1 - p) / 2400)
    assert result.worst.effect == 0.0
    assert result.worst.upper == pytest.approx(_t(0.95, 7) * floor)
    assert result.worst.lower == pytest.approx(-_t(0.95, 7) * floor)
    assert "binomial_floor" in (result.worst.detail or "")


def test_the_fraction_interval_floors_its_se_at_the_pooled_binomial_se() -> None:
    baseline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    candidate = [0.01, 0.01, 0.0133, 0.0100, 0.01, 0.0067, 0.01, 0.01]
    candidate = [round(x * 300) / 300 for x in candidate]
    result = _fractions(candidate, baseline=baseline)
    d = np.subtract(candidate, baseline)
    x_b, x_c = 0, round(sum(candidate) * 300)
    p_b, p_c = (x_b + 1) / 2402, (x_c + 1) / 2402
    floor = math.sqrt(p_b * (1 - p_b) / 2400 + p_c * (1 - p_c) / 2400)
    se = max(d.std(ddof=1) / math.sqrt(8), floor)
    assert result.worst is not None
    assert result.worst.upper == pytest.approx(d.mean() + _t(0.95, 7) * se)


def test_a_fraction_records_each_runs_requests_beside_its_value() -> None:
    # A caller can check m >= 3 / b run by run, as the gate does.
    record = _fractions([0.0] * 7 + [None]).to_record()
    assert record["trials"] == {"baseline": [300] * 7, "candidate": [300] * 7}
    assert len(record["values"]["candidate"]) == 7
    assert _latency().to_record()["trials"] is None


def test_pooled_requests_are_reported_and_labelled_model_based() -> None:
    result = _fractions([0.0] * 8)
    assert result.pooled is not None
    assert result.pooled["model"] == "independent_requests"
    assert result.pooled["candidate"]["n"] == 2400
    assert result.pooled["n_eff_for_zero_event_bound"] == 368
    assert "1 + (m - 1)" in result.pooled["note"]


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"trials": ([200] * 8, [200] * 8)}, "too_few_requests_per_run"),
        ({"trials": None}, "requests_per_run_unrecorded"),
        ({"trials": ([300] * 8, [300] * 7 + [None])}, "requests_per_run_unrecorded"),
    ],
)
def test_the_claim_needs_three_over_the_budget_requests_per_run(
    options: dict[str, Any], reason: str
) -> None:
    # With fewer, a single failure breaches the budget.
    result = _fractions([0.0] * 8, **options)
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == ("not_evaluable", reason)


def test_too_few_runs_to_make_the_claim_cannot_be_evaluated() -> None:
    # Five of five within budget give a lower bound of 0.478 < 0.5: no
    # outcome of five runs could pass, so failing would say nothing.
    result = _fractions([0.0] * 5)
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "not_evaluable",
        "too_few_runs_for_claim",
    )


@pytest.mark.parametrize("rule", ["significant", "demonstrated"])
def test_a_regression_claim_on_a_fraction_is_a_usage_error(rule: str) -> None:
    # The message names the one claim v1 supports, and why.
    with pytest.raises(InferUsageError) as raised:
        _fractions([0.0] * 8, gate=GateRule(rule, 0.01, "fraction"))
    message = str(raised.value)
    assert f"{rule} is not supported" in message
    assert "only a run-level claim" in message and "non-inferiority" in message


def test_absolute_uses_the_candidates_runs_alone() -> None:
    result = compare_values(
        "trace.bytes",
        [0.0, 0.0, 0.0],
        [5.0, 6.0, 7.0, 6.0],
        direction="lower_is_better",
        scale="absolute",
        unit="bytes",
        blocks=(["1", "2", "3"], ["1", "2", "3", "4"]),
    )
    assert result.worst is not None
    assert result.worst.method == "one_sample_t"
    assert result.worst.effect == pytest.approx(6.0)
    assert result.worst.df == 3


@pytest.mark.parametrize(
    ("rule", "budget", "status"),
    [
        ("non-inferiority", 0.20, "pass"),
        ("non-inferiority", 0.05, "fail"),
        ("significant", 0.05, "fail"),
        ("significant", 0.12, "pass"),
        ("demonstrated", 0.05, "fail"),
        ("demonstrated", 0.09, "pass"),
    ],
)
def test_each_gate_rule_reads_the_interval_its_own_way(
    rule: str, budget: float, status: str
) -> None:
    # About +9.3%, interval +8.1% to +10.4%; no single block changes these.
    gate = _latency(gate=GateRule(rule, budget, "relative")).gate
    assert gate is not None and gate.status == status


def test_higher_is_better_passes_non_inferiority_when_the_lower_bound_is_above_minus_b() -> (
    None
):
    result = _latency(
        baseline=LATENCY_B,
        candidate=LATENCY_A,
        direction="higher_is_better",
        gate=GateRule("non-inferiority", 0.12, "relative"),
    )
    assert result.worst is not None and result.worst.lower is not None
    assert result.worst.lower >= -0.12
    assert result.gate is not None and result.gate.status == "pass"
    tighter = _latency(
        baseline=LATENCY_B,
        candidate=LATENCY_A,
        direction="higher_is_better",
        gate=GateRule("non-inferiority", 0.05, "relative"),
    )
    assert tighter.gate is not None and tighter.gate.status == "fail"


def test_missing_evidence_neither_hides_nor_invents_a_regression() -> None:
    # Goodput per run as (lower, upper): unknown SLO outcomes widen it.
    baseline = [(10.0, 10.2)] * 6
    candidate = [(9.7, 10.3), (9.75, 10.25), (9.72, 10.32)] * 2
    gate_safe = GateRule("non-inferiority", 0.01, "relative")
    gate_regression = GateRule("significant", 0.01, "relative")
    common: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "direction": "higher_is_better",
    }
    safe = _latency(gate=gate_safe, **common)
    regression = _latency(gate=gate_regression, **common)

    assert safe.interval_valued
    assert safe.worst is not None and safe.best is not None
    assert safe.worst.effect < 0 < safe.best.effect
    # Non-inferiority, a safety claim, is judged on the worst case: it fails.
    assert safe.gate is not None
    assert (safe.gate.status, safe.gate.case) == ("fail", "worst_case")
    # A regression must be shown in the best case too: here it is not.
    assert regression.gate is not None
    assert (regression.gate.status, regression.gate.case) == ("pass", "best_case")


def test_a_run_with_unknown_outcomes_is_judged_on_its_worst_case() -> None:
    # Attainment per run as (lower, upper): a run is within a 0.01 budget
    # only if its lower bound is, against its block baseline's upper bound.
    baseline = [(0.990, 0.992)] * 8
    within = [(0.985, 0.995)] * 8
    result = compare_values(
        "attainment",
        baseline,
        within,
        direction="higher_is_better",
        scale="difference",
        unit="fraction",
        blocks=(BLOCKS + ["b7", "b8"], BLOCKS + ["b7", "b8"]),
        trials=([1000] * 8, [1000] * 8),
        gate=GateRule("non-inferiority", 0.01, "fraction"),
    )
    assert result.gate is not None and result.gate.status == "pass"
    wider = compare_values(
        "attainment",
        baseline,
        within[:7] + [(0.975, 0.995)],
        direction="higher_is_better",
        scale="difference",
        unit="fraction",
        blocks=(BLOCKS + ["b7", "b8"], BLOCKS + ["b7", "b8"]),
        trials=([1000] * 8, [1000] * 8),
        gate=GateRule("non-inferiority", 0.01, "fraction"),
    )
    assert wider.gate is not None and wider.gate.status == "fail"


def test_a_zero_candidate_on_a_higher_is_better_metric_always_fails() -> None:
    result = _latency(
        baseline=[10.0, 11.0, 9.0, 10.0, 10.5, 9.5],
        candidate=[10.0, 0.0, 9.0, 10.0, 10.5, 9.5],
        direction="higher_is_better",
        gate=GateRule("non-inferiority", 0.5, "relative"),
    )
    assert result.reason == "candidate_zero"
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == ("fail", "candidate_zero")
    assert result.difference is not None


@pytest.mark.parametrize(
    ("baseline", "candidate", "direction"),
    [
        ([0.0, 1.0, 2.0, 1.0], [1.0, 1.0, 2.0, 1.0], "higher_is_better"),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "higher_is_better"),
        ([1.0, 1.0, 2.0, 1.0], [0.0, 1.0, 2.0, 1.0], "lower_is_better"),
    ],
)
def test_a_zero_elsewhere_leaves_a_log_ratio_undefined(
    baseline: list[float], candidate: list[float], direction: str
) -> None:
    result = compare_values(
        "goodput",
        baseline,
        candidate,
        direction=direction,  # type: ignore[arg-type]
        scale="log_ratio",
        unit="relative",
        blocks=(["1", "2", "3", "4"], ["1", "2", "3", "4"]),
        gate=GateRule("non-inferiority", 0.05, "relative"),
    )
    assert result.reason == "undefined_in_arm"
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "not_evaluable",
        "undefined_in_arm",
    )


def _goodput_with_fallback(per_second: float, unit: str) -> Any:
    """Goodput with a zero baseline run, in requests per second or minute."""
    baseline = [0.0, 2.0, 2.1, 1.9, 2.0, 2.05]
    candidate = [0.1, 2.0, 2.05, 1.95, 2.0, 2.0]
    return compare_values(
        "goodput",
        [x * per_second for x in baseline],
        [x * per_second for x in candidate],
        direction="higher_is_better",
        scale="log_ratio",
        unit="relative",
        value_unit=unit,
        blocks=(BLOCKS, BLOCKS),
        gate=GateRule(
            "non-inferiority",
            0.05,
            "relative",
            fallback_budget=0.2 * per_second,
            fallback_unit=unit,
        ),
    )


def test_a_preregistered_fallback_gates_the_difference_the_same_in_any_unit() -> None:
    per_second = _goodput_with_fallback(1.0, "requests_per_second")
    per_minute = _goodput_with_fallback(60.0, "requests_per_minute")
    for result in (per_second, per_minute):
        assert result.reason == "undefined_in_arm"
        assert result.gate is not None
        assert (result.gate.status, result.gate.case) == ("pass", "fallback")
    assert per_minute.difference is not None and per_second.difference is not None
    assert per_minute.difference.effect == pytest.approx(
        60 * per_second.difference.effect
    )


def test_all_zero_differences_report_a_run_level_bound_not_zero_to_zero() -> None:
    # Without request counts there is no floor: no interval, and a gate in
    # the metric's own unit cannot pass on the run-level bound alone.
    result = _latency(
        baseline=[0.0] * 6,
        candidate=[0.0] * 6,
        scale="difference",
        unit="seconds",
        gate=GateRule("non-inferiority", 0.01, "seconds"),
    )
    assert result.reason == "degenerate_zero"
    assert result.worst is None
    assert result.degenerate is not None
    bound = result.degenerate["run_departs_upper"]["candidate"]
    assert bound == pytest.approx(1 - 0.025 ** (1 / 6))
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "not_evaluable",
        "degenerate_zero",
    )


def test_fewer_than_three_pairs_cannot_gate() -> None:
    result = _latency(
        baseline=LATENCY_A[:2],
        candidate=LATENCY_B[:2],
        blocks=(BLOCKS[:2], BLOCKS[:2]),
        gate=GateRule("non-inferiority", 0.5, "relative"),
    )
    assert result.worst is not None  # the values are still reported
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "not_evaluable",
        "insufficient_blocks",
    )
    single = _latency(
        baseline=LATENCY_A[:1], candidate=LATENCY_B[:1], blocks=(BLOCKS[:1], BLOCKS[:1])
    )
    assert single.worst is None and single.reason == "insufficient_blocks"


def test_preregistered_block_counts_cannot_be_quietly_reduced() -> None:
    gate = GateRule("non-inferiority", 0.5, "relative", min_complete_blocks=6)
    result = _latency(candidate=[*LATENCY_B[:5], None], gate=gate)
    assert result.n_pairs == 5
    assert result.attrition == [{"block": "b6", "reason": "block_incomplete:candidate"}]
    assert result.gate is not None
    assert result.gate.reason == "blocks_below_preregistered"


def test_a_block_with_two_runs_of_one_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="two baseline runs"):
        _latency(blocks=(["b1", "b1", "b3", "b4", "b5", "b6"], BLOCKS))


def test_one_block_that_decides_the_gate_makes_it_unstable() -> None:
    baseline = [100.0] * 8
    candidate = [101.0, 100.5, 101.5, 100.0, 101.0, 100.5, 101.0, 116.0]
    blocks = [f"b{i}" for i in range(8)]
    result = _latency(
        baseline=baseline,
        candidate=candidate,
        blocks=(blocks, blocks),
        gate=GateRule("non-inferiority", 0.06, "relative"),
    )
    assert (
        result.guards.decision_flips is not None and result.guards.decision_flips >= 1
    )
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "not_evaluable",
        "decision_unstable",
    )


def test_the_skew_screen_is_reported_and_never_blocks() -> None:
    blocks = [f"b{i}" for i in range(8)]
    candidate = [101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 140.0]
    result = _latency(
        baseline=[100.0] * 8,
        candidate=candidate,
        blocks=(blocks, blocks),
        gate=GateRule("significant", 0.9, "relative"),
    )
    assert result.guards.skew_flagged is True
    assert result.guards.skew_limit == pytest.approx(1.509)
    assert result.gate is not None and result.gate.status == "pass"


def test_a_bootstrap_is_a_sensitivity_number_from_ten_pairs() -> None:
    blocks = [f"b{i}" for i in range(10)]
    rng = np.random.default_rng(3)
    baseline = list(100 + rng.normal(0, 2, 10))
    candidate = list(105 + rng.normal(0, 2, 10))
    result = _latency(baseline=baseline, candidate=candidate, blocks=(blocks, blocks))
    again = _latency(baseline=baseline, candidate=candidate, blocks=(blocks, blocks))
    assert result.sensitivity is not None
    assert result.sensitivity.method == "block_bootstrap_log_ratio"
    assert result.sensitivity == again.sensitivity  # seeded
    assert _latency().sensitivity is None


def test_a_gate_in_another_unit_is_a_usage_error() -> None:
    with pytest.raises(InferUsageError, match="unit"):
        _latency(gate=GateRule("non-inferiority", 0.05, "fraction"))
    with pytest.raises(InferUsageError, match="relative"):
        _latency(unit="seconds")


def test_zero_variance_differences_are_flagged() -> None:
    result = _latency(
        baseline=[1.0] * 4,
        candidate=[1.0] * 4,
        blocks=(BLOCKS[:4], BLOCKS[:4]),
        scale="difference",
        unit="seconds",
    )
    assert result.reason == "degenerate_constant"
    assert result.worst is None
    constant = _latency(
        baseline=[1.0, 2.0, 3.0, 4.0],
        candidate=[1.5, 2.5, 3.5, 4.5],
        blocks=(BLOCKS[:4], BLOCKS[:4]),
        scale="difference",
        unit="seconds",
    )
    assert constant.guards.zero_variance is True
    assert constant.worst is not None and constant.worst.lower == constant.worst.upper


def test_clopper_pearson_names_its_model() -> None:
    runs = clopper_pearson(0, 10, model="independent_runs")
    requests = clopper_pearson(3, 1000, model="independent_requests")
    assert runs.lower == 0.0
    assert runs.upper == pytest.approx(1 - 0.025 ** (1 / 10))
    assert not runs.model_based and requests.model_based


@pytest.mark.parametrize(("k", "status"), [(6, "pass"), (5, "fail")])
def test_the_run_level_attainment_gate_needs_every_run_at_six(
    k: int, status: str
) -> None:
    assert run_pass_gate(k, 6, 0.5)["status"] == status


def test_precision_planning_matches_a_brute_force_search() -> None:
    for sd in (0.02, 0.05, 0.08):
        plan = blocks_for_precision(sd, 0.05, n_min=6, n_max=12)
        target = math.log1p(0.05)
        brute = next(
            (n for n in range(6, 13) if _t(0.95, n - 1) * sd / math.sqrt(n) <= target),
            None,
        )
        assert plan.n == (brute if brute is not None else 12)
        assert plan.meets_target is (brute is not None)


def test_holm_steps_down() -> None:
    rejected = holm({"a": 0.001, "b": 0.02, "c": 0.06, "d": None})
    assert rejected == {"a": True, "b": True, "c": False, "d": False}
    # Each step compares with alpha over the hypotheses left: 0.04 <= 0.05/1.
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04})["c"] is True


def test_skew_limits_follow_the_table_then_the_standard_error() -> None:
    assert skew_limit(2) is None
    assert skew_limit(6) == 1.699
    assert skew_limit(40) == pytest.approx(
        1.96 * math.sqrt(6 * 40 * 39 / (38 * 41 * 43))
    )


@pytest.mark.parametrize(
    ("budget", "unit"), [(-0.01, "relative"), (math.inf, "relative"), (1.5, "fraction")]
)
def test_a_gate_budget_that_cannot_mean_anything_is_refused(
    budget: float, unit: str
) -> None:
    with pytest.raises(InferUsageError):
        GateRule("non-inferiority", budget, unit)


def _goodput(candidate: list[Any], rule: str, **options: Any) -> Any:
    return compare_values(
        "goodput_rps",
        [10.0] * 6,
        candidate,
        direction="higher_is_better",
        scale="log_ratio",
        unit="relative",
        blocks=(BLOCKS, BLOCKS),
        gate=GateRule(rule, 0.05, "relative"),
        **options,
    )


def test_a_zero_only_in_the_case_a_rule_does_not_use_is_no_regression() -> None:
    # Goodput known only as (0, 10.4): unknown outcomes. The significant
    # rule judges the best case, where the candidate served 10.4; missing
    # evidence must not invent a regression.
    candidate = [(0.0, 10.4)] * 6
    claim = _goodput(candidate, "significant")
    assert claim.gate is not None
    assert (claim.gate.status, claim.gate.reason) == (
        "not_evaluable",
        "undefined_in_arm",
    )
    # Non-inferiority judges the worst case, where it served nothing.
    safe = _goodput(candidate, "non-inferiority")
    assert safe.gate is not None
    assert (safe.gate.status, safe.gate.reason) == ("fail", "candidate_zero")


def test_a_regression_rule_on_a_zero_waits_for_its_blockers() -> None:
    result = _goodput([0.0] * 6, "significant", unavailable="unverified")
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == ("not_evaluable", "unverified")


def test_a_value_that_is_not_a_number_blocks_the_gate() -> None:
    # Dropping it as missing would decide the gate on the blocks that are left.
    candidate = [140.0, 143.0, 137.0, math.nan, math.nan, math.nan]
    result = _latency(
        candidate=candidate, gate=GateRule("non-inferiority", 0.05, "relative")
    )
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "not_evaluable",
        "non_finite_value",
    )


@pytest.mark.parametrize("rule", ["non-inferiority", "significant"])
def test_an_infinite_candidate_latency_is_the_worst_value(rule: str) -> None:
    candidate = [101.0, 104.0, 98.0, math.inf, 102.0, 99.0]
    result = _latency(candidate=candidate, gate=GateRule(rule, 0.05, "relative"))
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == (
        "fail",
        "candidate_censored_worst",
    )
    # A baseline that never finished says nothing about the candidate.
    baseline = [101.0, 104.0, 98.0, math.inf, 102.0, 99.0]
    other = _latency(baseline=baseline, gate=GateRule(rule, 0.05, "relative"))
    assert other.gate is not None and other.gate.reason == "non_finite_value"


def test_the_run_gate_is_one_sided_at_97_5_percent() -> None:
    # 6 of 6 runs: the one-sided 97.5% lower bound is 0.025^(1/6) = 0.541,
    # below a required 0.58; at 95% it would be 0.607, above it.
    gate = run_pass_gate(6, 6, 0.58)
    assert gate["lower_bound"] == pytest.approx(0.025 ** (1 / 6))
    assert gate["status"] == "fail"
