"""Exact one-sided bounds for the qualification's accuracy and false-positive
claims, and the victim-impact test."""

from __future__ import annotations

import math
from typing import Any

import pytest

from stormlog.infer.qualify.bounds import (
    clopper_pearson_interval,
    clopper_pearson_lower,
    clopper_pearson_upper,
    fisher_greater,
    poisson_rate_upper,
)


def _binomial_tail_at_least(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _binomial_tail_at_most(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def test_the_plans_accuracy_and_fpr_numbers() -> None:
    # #221 C.5: 15/15 passes a 0.78 floor, 14/15 does not, and 0/60 negative
    # runs bound the FPR at 0.0487.
    assert clopper_pearson_lower(15, 15) == pytest.approx(0.8190, abs=5e-5)
    assert clopper_pearson_lower(14, 15) == pytest.approx(0.7206, abs=5e-5)
    assert clopper_pearson_upper(0, 60) == pytest.approx(0.0487, abs=5e-5)


@pytest.mark.parametrize(("k", "n"), [(1, 15), (7, 15), (14, 15), (3, 60), (40, 60)])
def test_the_bounds_invert_the_binomial_tail(k: int, n: int) -> None:
    # At the lower bound, seeing k or more successes has probability 0.05;
    # at the upper bound, seeing k or fewer has probability 0.05.
    lower, upper = clopper_pearson_lower(k, n), clopper_pearson_upper(k, n)
    assert _binomial_tail_at_least(k, n, lower) == pytest.approx(0.05, rel=1e-6)
    assert _binomial_tail_at_most(k, n, upper) == pytest.approx(0.05, rel=1e-6)


def test_the_bounds_meet_the_edges() -> None:
    assert clopper_pearson_lower(0, 15) == 0.0
    assert clopper_pearson_upper(15, 15) == 1.0
    with pytest.raises(ValueError):
        clopper_pearson_lower(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0)


def test_the_poisson_rate_bound_inverts_the_poisson_tail() -> None:
    # #221 C.5: zero alarms over 4.9 negative hours bound the rate at 0.61/h.
    assert poisson_rate_upper(0, 4.9) == pytest.approx(0.611, abs=5e-4)
    rate = poisson_rate_upper(2, 4.9)
    mean = rate * 4.9
    tail = sum(math.exp(-mean) * mean**i / math.factorial(i) for i in range(3))
    assert tail == pytest.approx(0.05, rel=1e-6)
    with pytest.raises(ValueError):
        poisson_rate_upper(0, 0.0)


def test_the_impact_test_is_one_sided_fishers_exact() -> None:
    # More violations in the effect window than in the baseline: small p.
    strong = fisher_greater(
        violations=8, met=12, baseline_violations=3, baseline_met=57
    )
    # Fewer violations than the baseline: the one-sided p is large.
    weak = fisher_greater(violations=1, met=19, baseline_violations=6, baseline_met=54)
    assert strong < 0.01
    assert weak > 0.5
    # The hypergeometric upper tail, written out.
    total, drawn, marked = 80, 20, 11
    expected = sum(
        math.comb(marked, i) * math.comb(total - marked, drawn - i)
        for i in range(8, min(drawn, marked) + 1)
    ) / math.comb(total, drawn)
    assert strong == pytest.approx(expected, rel=1e-9)


def test_exploratory_claims_get_the_two_sided_interval() -> None:
    # C.5: DX-ON and TP2 are reported with two-sided 95% intervals: 6/6
    # gives 0.54 and 8/8 gives 0.63 at the bottom, where the one-sided bound
    # would say 0.61 and 0.69.
    six_lower, six_upper = clopper_pearson_interval(6, 6)
    assert six_lower == pytest.approx(0.5407, abs=5e-5)
    assert six_upper == 1.0
    assert clopper_pearson_interval(8, 8)[0] == pytest.approx(0.6306, abs=5e-5)
    lower, upper = clopper_pearson_interval(3, 10)
    assert _binomial_tail_at_least(3, 10, lower) == pytest.approx(0.025, rel=1e-6)
    assert _binomial_tail_at_most(3, 10, upper) == pytest.approx(0.025, rel=1e-6)


@pytest.mark.parametrize(
    "call",
    [
        lambda: clopper_pearson_lower(14.5, 15),  # type: ignore[arg-type]
        lambda: clopper_pearson_upper(True, 15),
        lambda: clopper_pearson_lower(14, 15, confidence=95),
        lambda: clopper_pearson_upper(0, 60, confidence=0.0),
        lambda: poisson_rate_upper(0, math.inf),
        lambda: poisson_rate_upper(0.5, 4.9),  # type: ignore[arg-type]
        lambda: poisson_rate_upper(0, 4.9, confidence=1.0),
    ],
    ids=[
        "fractional successes",
        "a bool as a count",
        "confidence as a percentage",
        "zero confidence",
        "infinite exposure",
        "fractional events",
        "certainty",
    ],
)
def test_inputs_that_would_give_a_meaningless_bound_are_refused(call: Any) -> None:
    with pytest.raises(ValueError):
        call()
