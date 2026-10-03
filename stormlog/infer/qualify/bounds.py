"""Exact one-sided bounds and tests for the qualification's claims.

Accuracy is claimed with a one-sided Clopper–Pearson lower bound, the
false-positive rate with its upper bound, and a negative-hour rate with the
exact Poisson upper bound. Victim impact is a one-sided Fisher exact test of
violations in an effect window against a baseline.
"""

from __future__ import annotations

from scipy import stats

CONFIDENCE = 0.95


def clopper_pearson_lower(
    successes: int, trials: int, confidence: float = CONFIDENCE
) -> float:
    """The one-sided lower bound on a binomial proportion: the p at which
    seeing ``successes`` or more has probability 1 − ``confidence``."""
    _check_counts(successes, trials)
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    return float(stats.beta.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(
    successes: int, trials: int, confidence: float = CONFIDENCE
) -> float:
    """The one-sided upper bound: the p at which seeing ``successes`` or
    fewer has probability 1 − ``confidence``."""
    _check_counts(successes, trials)
    if successes == trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def poisson_rate_upper(
    events: int, exposure: float, confidence: float = CONFIDENCE
) -> float:
    """The exact upper bound on an event rate per unit of ``exposure``: the
    rate at which seeing ``events`` or fewer has probability 1 − ``confidence``."""
    if events < 0:
        raise ValueError("events must be non-negative")
    if not exposure > 0:
        raise ValueError("exposure must be positive")
    mean = float(stats.chi2.ppf(confidence, 2 * events + 2)) / 2.0
    return mean / exposure


def fisher_greater(
    *, violations: int, met: int, baseline_violations: int, baseline_met: int
) -> float:
    """The one-sided Fisher exact p-value that violations are more likely in
    the effect window than in the baseline."""
    table = [[violations, met], [baseline_violations, baseline_met]]
    if min(violations, met, baseline_violations, baseline_met) < 0:
        raise ValueError("counts must be non-negative")
    _odds, p_value = stats.fisher_exact(table, alternative="greater")
    return float(p_value)


def _check_counts(successes: int, trials: int) -> None:
    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between 0 and trials")


__all__ = [
    "CONFIDENCE",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "fisher_greater",
    "poisson_rate_upper",
]
