"""Comparing one metric between two arms of runs."""

from __future__ import annotations

import math
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
    # Astra's adversarial shape: 20 baseline runs against 3 candidate runs.
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
        gate=GateRule("non-inferiority", 0.01, "fraction"),
    )
    diffs = np.subtract(attainment_b, attainment_a)
    assert result.worst is not None
    assert result.worst.effect == pytest.approx(diffs.mean())
    assert result.gate is not None and result.gate.status == "fail"


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
    # Attainment per run as (lower, upper): unknown SLO outcomes widen it.
    baseline = [(0.990, 0.992)] * 6
    candidate = [(0.975, 0.995), (0.976, 0.994), (0.974, 0.996)] * 2
    gate_safe = GateRule("non-inferiority", 0.005, "fraction")
    gate_regression = GateRule("significant", 0.005, "fraction")
    common: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "direction": "higher_is_better",
        "scale": "difference",
        "unit": "fraction",
    }
    safe = _latency(gate=gate_safe, **common)
    regression = _latency(gate=gate_regression, **common)

    assert safe.interval_valued
    assert safe.worst is not None and safe.best is not None
    assert safe.worst.effect == pytest.approx(0.975 - 0.992, abs=0.002)
    # Non-inferiority, a safety claim, is judged on the worst case: it fails.
    assert safe.gate is not None
    assert (safe.gate.status, safe.gate.case) == ("fail", "worst_case")
    # A regression must be shown in the best case too: here it is not.
    assert regression.gate is not None
    assert (regression.gate.status, regression.gate.case) == ("pass", "best_case")


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
    result = _latency(
        baseline=[0.0] * 6,
        candidate=[0.0] * 6,
        scale="difference",
        unit="fraction",
        gate=GateRule("non-inferiority", 0.01, "fraction"),
    )
    assert result.reason == "degenerate_zero"
    assert result.worst is None
    assert result.degenerate is not None
    bound = result.degenerate["run_departs_upper"]["candidate"]
    assert bound == pytest.approx(1 - 0.025 ** (1 / 6))
    assert result.gate is not None
    assert (result.gate.status, result.gate.reason) == ("pass", "degenerate_zero")


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
