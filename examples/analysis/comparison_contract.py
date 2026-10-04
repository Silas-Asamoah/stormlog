"""Write ``tests/fixtures/infer/comparison_contract_v1.json``.

The fixture is the units contract between ``stormlog.infer.comparison_stats``
and its callers (#221's qualification gates): for each case, the inputs and
the effect, interval and gate outcome they must give. The expected numbers
here are computed from the textbook formulas directly, not by calling the
module, so the fixture checks the module rather than repeating it. The
expected gate outcomes are derived the same way, by the rules of the plan's
§0.4 (``derived_gate``), never assigned by hand.

    python -m examples.analysis.comparison_contract
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

OUTPUT = Path("tests/fixtures/infer/comparison_contract_v1.json")
BLOCKS6 = [f"b{i}" for i in range(1, 7)]
NUDGE = 1e-9  # keeps a boundary budget clear of floating-point ties


def _t(n_df: float) -> float:
    return float(stats.t.ppf(0.975, n_df))


def paired_log(baseline: list[float], candidate: list[float]) -> dict[str, float]:
    d = np.log(candidate) - np.log(baseline)
    half = _t(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    mean = float(d.mean())
    return {
        "effect": math.expm1(mean),
        "lower": math.expm1(mean - half),
        "upper": math.expm1(mean + half),
    }


def paired_difference(
    baseline: list[float], candidate: list[float]
) -> dict[str, float]:
    d = np.subtract(candidate, baseline)
    half = _t(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    mean = float(d.mean())
    return {"effect": mean, "lower": mean - half, "upper": mean + half}


def welch_log_min_df(baseline: list[float], candidate: list[float]) -> dict[str, float]:
    a, b = np.log(baseline), np.log(candidate)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    half = _t(min(len(a), len(b)) - 1) * se
    mean = float(b.mean() - a.mean())
    return {
        "effect": math.expm1(mean),
        "lower": math.expm1(mean - half),
        "upper": math.expm1(mean + half),
    }


def decide(
    rule: str,
    budget: float,
    direction: str,
    worst: dict[str, float],
    best: dict[str, float],
) -> str:
    """A rule's reading as a change in the bad direction: non-inferiority on
    the worst case, a regression claim on the best."""
    estimate = worst if rule == "non-inferiority" else best
    if direction == "lower_is_better":
        effect, lower, upper = estimate["effect"], estimate["lower"], estimate["upper"]
    else:
        effect, lower, upper = (
            -estimate["effect"],
            -estimate["upper"],
            -estimate["lower"],
        )
    if rule == "non-inferiority":
        failed = upper > budget
    elif rule == "significant":
        failed = lower > 0 and effect > budget
    else:
        failed = lower > budget
    return "fail" if failed else "pass"


def _bound(values: list[Any], index: int) -> list[float]:
    return [v[index] if isinstance(v, list) else v for v in values]


def intervals(
    baseline: list[Any],
    candidate: list[Any],
    *,
    paired: bool,
    scale: str,
    direction: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """The worst and the best case's intervals. A value may be a (low, high)
    pair: the worst case takes each arm's bound on its bad side."""
    if paired:
        method = paired_log if scale == "log_ratio" else paired_difference
    else:
        method = welch_log_min_df
    good, bad = (0, 1) if direction == "lower_is_better" else (1, 0)
    worst = method(_bound(baseline, good), _bound(candidate, bad))
    best = method(_bound(baseline, bad), _bound(candidate, good))
    return worst, best


def _leave_one_out(
    baseline: list[Any], candidate: list[Any], paired: bool
) -> list[tuple[list[Any], list[Any]]]:
    """Every sample with one pair (or one run of an arm) left out that still
    holds three, so that it could be gated."""
    if paired:
        if len(baseline) <= 3:
            return []
        return [
            (baseline[:i] + baseline[i + 1 :], candidate[:i] + candidate[i + 1 :])
            for i in range(len(baseline))
        ]
    samples = []
    if len(baseline) > 3:
        samples += [
            (baseline[:i] + baseline[i + 1 :], candidate) for i in range(len(baseline))
        ]
    if len(candidate) > 3:
        samples += [
            (baseline, candidate[:i] + candidate[i + 1 :])
            for i in range(len(candidate))
        ]
    return samples


def derived_gate(
    baseline: list[Any],
    candidate: list[Any],
    *,
    rule: str,
    budget: float,
    direction: str,
    scale: str = "log_ratio",
    paired: bool = True,
    min_complete_blocks: int | None = None,
) -> dict[str, Any]:
    """The gate by rule: fewer complete pairs than pre-registered, or than
    three, cannot be gated; a decision that any leave-one-out sample
    reverses is unstable; otherwise the rule decides."""
    if paired:
        kept = [
            (a, b)
            for a, b in zip(baseline, candidate)
            if a is not None and b is not None
        ]
        baseline, candidate = [a for a, _ in kept], [b for _, b in kept]
    complete = len(baseline) if paired else min(len(baseline), len(candidate))
    if min_complete_blocks is not None and complete < min_complete_blocks:
        return {"gate": "not_evaluable", "reason": "blocks_below_preregistered"}
    if complete < 3:
        reason = "insufficient_blocks" if paired else "insufficient_runs"
        return {"gate": "not_evaluable", "reason": reason}

    def decided(first: list[Any], second: list[Any]) -> str:
        bounds = intervals(
            first, second, paired=paired, scale=scale, direction=direction
        )
        return decide(rule, budget, direction, *bounds)

    decision = decided(baseline, candidate)
    for first, second in _leave_one_out(baseline, candidate, paired):
        if decided(first, second) != decision:
            return {"gate": "not_evaluable", "reason": "decision_unstable"}
    return {
        "gate": decision,
        "reason": None if decision == "pass" else "exceeds_budget",
        "case": "worst_case" if rule == "non-inferiority" else "best_case",
    }


def case(
    case_id: str,
    note: str,
    *,
    baseline: list[Any],
    candidate: list[Any],
    direction: str,
    scale: str,
    unit: str,
    gate: dict[str, Any] | None,
    expect: dict[str, Any],
    blocks: list[list[str]] | None = None,
    value_unit: str | None = None,
    unavailable: str | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "note": note,
        "input": {
            "baseline": baseline,
            "candidate": candidate,
            "blocks": blocks,
            "direction": direction,
            "scale": scale,
            "unit": unit,
            "value_unit": value_unit,
            "gate": gate,
            "unavailable": unavailable,
        },
        "expect": expect,
    }


def _fallback_gate(
    baseline: list[float], candidate: list[float], budget: float
) -> dict[str, Any]:
    """A pre-registered fallback gates the difference, higher_is_better,
    with no leave-one-out screen."""
    difference = paired_difference(baseline, candidate)
    decision = decide(
        "non-inferiority", budget, "higher_is_better", difference, difference
    )
    reason = "fallback_budget" if decision == "pass" else "exceeds_fallback_budget"
    return {"gate": decision, "reason": reason, "case": "fallback"}


def gate(rule: str, budget: float, unit: str, **extra: Any) -> dict[str, Any]:
    return {"rule": rule, "budget": budget, "unit": unit, **extra}


def build() -> list[dict[str, Any]]:
    cases = []
    # A 40% latency regression fails a 5% non-inferiority budget (the units contract with #221).
    base = [100.0, 102.0, 98.0, 101.0, 99.0, 100.0]
    slow = [140.0, 143.0, 137.0, 141.0, 138.5, 140.5]
    cases.append(
        case(
            "latency_regression_fails_non_inferiority",
            "lower_is_better: the effect is +40%, so upper > b fails",
            baseline=base,
            candidate=slow,
            blocks=[BLOCKS6, BLOCKS6],
            direction="lower_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", 0.05, "relative"),
            expect={
                **paired_log(base, slow),
                **derived_gate(
                    base,
                    slow,
                    rule="non-inferiority",
                    budget=0.05,
                    direction="lower_is_better",
                ),
            },
        )
    )
    same = [100.5, 101.5, 98.5, 100.5, 99.5, 100.0]
    cases.append(
        case(
            "unchanged_throughput_passes",
            "higher_is_better: an unchanged metric passes non-inferiority",
            baseline=base,
            candidate=same,
            blocks=[BLOCKS6, BLOCKS6],
            direction="higher_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", 0.05, "relative"),
            expect={
                **paired_log(base, same),
                **derived_gate(
                    base,
                    same,
                    rule="non-inferiority",
                    budget=0.05,
                    direction="higher_is_better",
                ),
            },
        )
    )
    # Boundaries: lower_is_better passes iff upper <= b; higher_is_better iff
    # lower >= -b. Three blocks, so that no leave-one-out sample can be gated
    # and only the comparison with the budget decides.
    slower = [104.0, 106.5, 101.5, 105.5, 103.0, 104.0]
    blocks3 = [BLOCKS6[:3], BLOCKS6[:3]]
    bounds3 = paired_log(base[:3], slower[:3])
    for label, budget in (
        ("just_above", bounds3["upper"] + NUDGE),
        ("just_below", bounds3["upper"] - NUDGE),
    ):
        cases.append(
            case(
                f"latency_boundary_{label}",
                "lower_is_better: non-inferiority compares the upper bound with b",
                baseline=base[:3],
                candidate=slower[:3],
                blocks=blocks3,
                direction="lower_is_better",
                scale="log_ratio",
                unit="relative",
                gate=gate("non-inferiority", budget, "relative"),
                expect={
                    **bounds3,
                    **derived_gate(
                        base[:3],
                        slower[:3],
                        rule="non-inferiority",
                        budget=budget,
                        direction="lower_is_better",
                    ),
                },
            )
        )
    lower_tp = [96.0, 97.5, 94.0, 96.5, 95.5, 96.0]
    tp_bounds = paired_log(base[:3], lower_tp[:3])
    for label, budget in (
        ("just_above", -tp_bounds["lower"] + NUDGE),
        ("just_below", -tp_bounds["lower"] - NUDGE),
    ):
        cases.append(
            case(
                f"throughput_boundary_{label}",
                "higher_is_better: non-inferiority compares the lower bound with -b",
                baseline=base[:3],
                candidate=lower_tp[:3],
                blocks=blocks3,
                direction="higher_is_better",
                scale="log_ratio",
                unit="relative",
                gate=gate("non-inferiority", budget, "relative"),
                expect={
                    **tp_bounds,
                    **derived_gate(
                        base[:3],
                        lower_tp[:3],
                        rule="non-inferiority",
                        budget=budget,
                        direction="higher_is_better",
                    ),
                },
            )
        )
    # At the boundary with six blocks, leaving one out changes the decision.
    bounds = paired_log(base, slower)
    cases.append(
        case(
            "boundary_decision_unstable",
            "a decision some single block would reverse is not evaluable",
            baseline=base,
            candidate=slower,
            blocks=[BLOCKS6, BLOCKS6],
            direction="lower_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", bounds["upper"] + NUDGE, "relative"),
            expect={
                **bounds,
                **derived_gate(
                    base,
                    slower,
                    rule="non-inferiority",
                    budget=bounds["upper"] + NUDGE,
                    direction="lower_is_better",
                ),
            },
        )
    )
    # Attainment budgets are fractions: 0.01 is one percentage point.
    att_a = [0.990, 0.988, 0.991, 0.989, 0.992, 0.990]
    att_b = [0.984, 0.983, 0.986, 0.982, 0.987, 0.985]
    cases.append(
        case(
            "attainment_budget_in_fraction_units",
            "a 0.6-point drop passes a 1-point (0.01) budget",
            baseline=att_a,
            candidate=att_b,
            blocks=[BLOCKS6, BLOCKS6],
            direction="higher_is_better",
            scale="difference",
            unit="fraction",
            gate=gate("non-inferiority", 0.01, "fraction"),
            expect={
                **paired_difference(att_a, att_b),
                **derived_gate(
                    att_a,
                    att_b,
                    rule="non-inferiority",
                    budget=0.01,
                    direction="higher_is_better",
                    scale="difference",
                ),
            },
        )
    )
    # The attainment boundary, in fraction units: three blocks, so only the
    # comparison of the lower bound with -b decides.
    att3_a = [0.990, 0.988, 0.991]
    att3_b = [0.984, 0.983, 0.986]
    att3 = paired_difference(att3_a, att3_b)
    for label, budget in (
        ("just_above", -att3["lower"] + NUDGE),
        ("just_below", -att3["lower"] - NUDGE),
    ):
        cases.append(
            case(
                f"attainment_boundary_{label}",
                "higher_is_better, difference in fractions: non-inferiority "
                "compares the lower bound with -b",
                baseline=att3_a,
                candidate=att3_b,
                blocks=[BLOCKS6[:3], BLOCKS6[:3]],
                direction="higher_is_better",
                scale="difference",
                unit="fraction",
                gate=gate("non-inferiority", budget, "fraction"),
                expect={
                    **att3,
                    **derived_gate(
                        att3_a,
                        att3_b,
                        rule="non-inferiority",
                        budget=budget,
                        direction="higher_is_better",
                        scale="difference",
                    ),
                },
            )
        )
    # Zero rules, and the same fallback decision in two units.
    for unit, per_second in (
        ("requests_per_second", 1.0),
        ("requests_per_minute", 60.0),
    ):
        zero_a = [x * per_second for x in (0.0, 2.0, 2.1, 1.9, 2.0, 2.05)]
        zero_b = [x * per_second for x in (0.1, 2.0, 2.05, 1.95, 2.0, 2.0)]
        cases.append(
            case(
                f"zero_baseline_fallback_{unit}",
                "a zero in the baseline leaves the log ratio undefined; the "
                "pre-registered fallback gates the difference, in any unit",
                baseline=zero_a,
                candidate=zero_b,
                blocks=[BLOCKS6, BLOCKS6],
                direction="higher_is_better",
                scale="log_ratio",
                unit="relative",
                value_unit=unit,
                gate=gate(
                    "non-inferiority",
                    0.05,
                    "relative",
                    fallback_budget=0.2 * per_second,
                    fallback_unit=unit,
                ),
                expect={
                    "effect": None,
                    "difference": paired_difference(zero_a, zero_b),
                    **_fallback_gate(zero_a, zero_b, 0.2 * per_second),
                },
            )
        )
    cases.append(
        case(
            "candidate_zero_always_fails",
            "a candidate that served nothing is the worst regression",
            baseline=[2.0, 2.1, 1.9, 2.0, 2.05, 1.95],
            candidate=[2.0, 0.0, 1.9, 2.0, 2.05, 1.95],
            blocks=[BLOCKS6, BLOCKS6],
            direction="higher_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", 0.5, "relative"),
            expect={"effect": None, "gate": "fail", "reason": "candidate_zero"},
        )
    )
    cases.append(
        case(
            "both_zero_is_not_evaluable",
            "zeros in both arms leave the log ratio undefined, with no fallback",
            baseline=[0.0] * 4,
            candidate=[0.0] * 4,
            blocks=[BLOCKS6[:4], BLOCKS6[:4]],
            direction="higher_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", 0.05, "relative"),
            expect={
                "effect": None,
                "gate": "not_evaluable",
                "reason": "undefined_in_arm",
            },
        )
    )
    cases.append(
        case(
            "remaining_blocks_below_preregistered",
            "six blocks were planned; one candidate run is missing",
            baseline=base,
            candidate=[*slower[:5], None],
            blocks=[BLOCKS6, BLOCKS6],
            direction="lower_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", 0.5, "relative", min_complete_blocks=6),
            expect={
                **paired_log(base[:5], slower[:5]),
                "n_pairs": 5,
                **derived_gate(
                    base,
                    [*slower[:5], None],
                    rule="non-inferiority",
                    budget=0.5,
                    direction="lower_is_better",
                    min_complete_blocks=6,
                ),
            },
        )
    )
    cases.append(
        case(
            "two_pairs_cannot_gate",
            "at two pairs df = 1 and the interval is always undetermined",
            baseline=base[:2],
            candidate=slower[:2],
            blocks=[BLOCKS6[:2], BLOCKS6[:2]],
            direction="lower_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("non-inferiority", 0.5, "relative"),
            expect={
                **paired_log(base[:2], slower[:2]),
                **derived_gate(
                    base[:2],
                    slower[:2],
                    rule="non-inferiority",
                    budget=0.5,
                    direction="lower_is_better",
                ),
            },
        )
    )
    cases.append(
        case(
            "censored_quantile_is_not_evaluable",
            "the caller knows the quantile is censored; the values are still compared",
            baseline=base,
            candidate=slower,
            blocks=[BLOCKS6, BLOCKS6],
            direction="lower_is_better",
            scale="log_ratio",
            unit="relative",
            unavailable="censored",
            gate=gate("non-inferiority", 0.5, "relative"),
            expect={**bounds, "gate": "not_evaluable", "reason": "censored"},
        )
    )
    # Missing-outcome bounds: worst case for safety, best case for regressions.
    att_bounds_a = [[0.990, 0.992]] * 6
    att_bounds_b = [[0.975, 0.995], [0.976, 0.994], [0.974, 0.996]] * 2
    worst = paired_difference(
        [x[1] for x in att_bounds_a], [x[0] for x in att_bounds_b]
    )
    best = paired_difference([x[0] for x in att_bounds_a], [x[1] for x in att_bounds_b])
    for rule in ("non-inferiority", "significant"):
        cases.append(
            case(
                f"missing_outcomes_{rule}",
                "unknown SLO outcomes: non-inferiority uses the worst case, "
                "regression claims the best",
                baseline=att_bounds_a,
                candidate=att_bounds_b,
                blocks=[BLOCKS6, BLOCKS6],
                direction="higher_is_better",
                scale="difference",
                unit="fraction",
                gate=gate(rule, 0.005, "fraction"),
                expect={
                    **worst,
                    "best": best,
                    **derived_gate(
                        att_bounds_a,
                        att_bounds_b,
                        rule=rule,
                        budget=0.005,
                        direction="higher_is_better",
                        scale="difference",
                    ),
                },
            )
        )
    cases.append(
        case(
            "all_zero_failure_rate_is_degenerate",
            "no failures anywhere: no [0, 0] interval, a run-level bound instead",
            baseline=[0.0] * 6,
            candidate=[0.0] * 6,
            blocks=[BLOCKS6, BLOCKS6],
            direction="lower_is_better",
            scale="difference",
            unit="fraction",
            gate=gate("non-inferiority", 0.01, "fraction"),
            expect={
                "effect": None,
                "run_departs_upper": 1 - 0.025 ** (1 / 6),
                "gate": "pass",
                "reason": "degenerate_zero",
            },
        )
    )
    independent_a = [100.0, 103.0, 97.0, 101.0, 99.0, 102.0, 98.0, 100.5]
    independent_b = [104.0, 107.0, 101.5]
    cases.append(
        case(
            "independent_runs_use_min_df",
            "8 against 3 runs: Welch on logs with df = min(nA, nB) - 1 = 2",
            baseline=independent_a,
            candidate=independent_b,
            direction="lower_is_better",
            scale="log_ratio",
            unit="relative",
            gate=gate("significant", 0.01, "relative"),
            expect={
                **welch_log_min_df(independent_a, independent_b),
                "df": 2,
                **derived_gate(
                    independent_a,
                    independent_b,
                    rule="significant",
                    budget=0.01,
                    direction="lower_is_better",
                    paired=False,
                ),
            },
        )
    )
    return cases


def main() -> None:
    cases = build()
    document = {
        "format": "stormlog.infer.comparison_contract",
        "version": 1,
        "payload": "stormlog.infer.comparison v1",
        "confidence": 0.95,
        "tolerance": {"relative": 1e-9, "absolute": 1e-12},
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"{len(cases)} cases written to {OUTPUT}")


if __name__ == "__main__":
    main()
