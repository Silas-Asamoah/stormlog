"""Comparing one metric between a baseline and a candidate arm of runs.

The run is the unit of replication. When runs carry block labels the design
is paired: each block holds one baseline and one candidate run, and only
complete pairs count. Otherwise the two arms are independent samples.

One method gives the interval that verdicts and gates use:

| Design | ``log_ratio`` | ``difference`` |
| --- | --- | --- |
| paired | t on the block log ratios, df = n - 1 | t on the block differences |
| independent | Welch t on log values, df = min(nA, nB) - 1 | Welch t, same df |

``absolute`` is a one-sample t over the candidate's runs, for metrics the
baseline cannot have. A ``log_ratio`` effect is ``exp(mean log ratio) - 1``,
a signed fraction; a ``difference`` effect is in the metric's own unit.
Fieller's interval for the arithmetic ratio is a cross-check, and from
``bootstrap_min_n`` pairs a bootstrap is a sensitivity number; neither is
ever gated.

A run's value can be an interval ``(lower, upper)``, when evidence is
missing (unknown SLO outcomes). Each pair then has a worst and a best case
for the metric's direction: a non-inferiority gate, a safety claim, uses the
worst case; the regression claims ``significant`` and ``demonstrated`` use
the best. Missing evidence therefore can neither hide a regression nor
invent one.

The guards (skewness against normal noise, leave-one-out) are reported and
never certify coverage. Only a leave-one-out change of the gate's decision
makes a gate ``not_evaluable``.

A ``fraction`` metric (a failure fraction, an SLO attainment) is gated on a
claim about runs, not on an interval: k of n candidate runs stayed within
the budget of their block's baseline, and the one-sided Clopper-Pearson
lower bound of k/n is at least 0.5. With blocks that is exact when runs are
independent, however failures cluster within a run, which no interval on
the fraction is. Without blocks the gate is not evaluable: judging every
run against one estimated baseline mean correlates the judgements. Each run needs ``3 / budget`` requests, so that one failure cannot
breach the budget. The paired t, with its standard error floored at the
pooled binomial one, is the descriptive interval, and the pooled requests'
Clopper-Pearson bounds are reported, labelled model-based.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy import stats

from .errors import InferUsageError

LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"
LOG_RATIO = "log_ratio"
DIFFERENCE = "difference"
ABSOLUTE = "absolute"
PAIRED = "paired_blocks"
INDEPENDENT = "independent"
NON_INFERIORITY = "non-inferiority"
SIGNIFICANT = "significant"
DEMONSTRATED = "demonstrated"
GATE_RULES = (NON_INFERIORITY, SIGNIFICANT, DEMONSTRATED)
RELATIVE = "relative"

PASS = "pass"
FAIL = "fail"
NOT_EVALUABLE = "not_evaluable"
MIN_GATE_PAIRS = 3
BOOTSTRAP_RESAMPLES = 10_000

Direction = Literal["lower_is_better", "higher_is_better"]
Scale = Literal["log_ratio", "difference", "absolute"]
RunValue = float | tuple[float, float] | None

# The 95% quantile of |G1| (adjusted sample skewness) under normal noise,
# by n; from examples/analysis/skew_quantiles.py (seed 213, 10^6 draws).
SKEW_Q95_NORMAL = {
    3: 1.727,
    4: 1.852,
    5: 1.801,
    6: 1.699,
    7: 1.593,
    8: 1.509,
    9: 1.434,
    10: 1.375,
    11: 1.322,
    12: 1.275,
    13: 1.231,
    14: 1.193,
    15: 1.159,
    16: 1.128,
    17: 1.097,
    18: 1.069,
    19: 1.043,
    20: 1.021,
    21: 0.998,
    22: 0.977,
    23: 0.957,
    24: 0.939,
    25: 0.923,
    26: 0.907,
    27: 0.889,
    28: 0.877,
    29: 0.861,
    30: 0.848,
}


FRACTION_UNIT = "fraction"
# The fraction gate's claim: at least this share of runs within budget.
RUN_PASS_SHARE = 0.5
RUN_LEVEL = "run_level"
# A run needs 3 / budget requests, so that one failure cannot breach it.
REQUESTS_PER_BUDGET = 3
BINOMIAL_FLOOR = (
    "binomial_floor: se floored at the pooled binomial se; nominal under "
    "independent requests, under-covers when failures cluster within runs"
)


@dataclass(frozen=True)
class GateRule:
    """A pre-registered gate on one metric, on the metric's own scale and unit."""

    rule: str
    budget: float
    unit: str
    min_complete_blocks: int | None = None
    fallback_budget: float | None = None
    fallback_unit: str | None = None

    def __post_init__(self) -> None:
        if self.rule not in GATE_RULES:
            raise InferUsageError(f"gate rule must be one of {', '.join(GATE_RULES)}")
        if not math.isfinite(self.budget) or self.budget < 0:
            raise InferUsageError("a gate budget must be a finite number >= 0")
        if self.unit == FRACTION_UNIT and self.budget >= 1:
            raise InferUsageError(
                f"a fraction budget of {self.budget:g} is 1 or more: it can never fail"
            )

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Estimate:
    """One interval: its method, what it estimates, and on which scale."""

    method: str
    estimand: str
    effect: float | None
    lower: float | None
    upper: float | None
    df: float | None
    n: int
    p_worse: float | None = None
    detail: str | None = None

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class GateOutcome:
    status: str
    reason: str | None
    rule: GateRule
    case: str | None = None
    # A fraction gate's run-level claim: k of n runs, and what it certifies.
    claim: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "case": self.case,
            "claim": self.claim,
            **self.rule.to_record(),
        }


@dataclass(frozen=True)
class Guards:
    skewness: float | None = None
    skew_limit: float | None = None
    skew_flagged: bool | None = None
    leave_one_out: dict[str, float | None] | None = None
    decision_flips: int | None = None
    zero_variance: bool = False

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MetricComparison:
    """One metric, compared: every number a verdict or gate rests on."""

    name: str
    design: str
    scale: str
    unit: str
    direction: str
    n_baseline: int
    n_candidate: int
    n_pairs: int
    values: dict[str, list[Any]]
    attrition: list[dict[str, Any]]
    worst: Estimate | None
    best: Estimate | None
    cross_check: Estimate | None = None
    sensitivity: Estimate | None = None
    difference: Estimate | None = None
    guards: Guards = field(default_factory=Guards)
    effect_size: dict[str, Any] | None = None
    verdict: dict[str, str] = field(default_factory=dict)
    gate: GateOutcome | None = None
    reason: str | None = None
    degenerate: dict[str, Any] | None = None
    # A fraction's requests pooled per arm: model-based, never gated.
    pooled: dict[str, Any] | None = None
    # A fraction's requests in each kept run, aligned with ``values``.
    trials: dict[str, list[int | None]] | None = None

    @property
    def interval_valued(self) -> bool:
        return self.worst != self.best

    def to_record(self) -> dict[str, Any]:
        def record(item: Any) -> Any:
            return None if item is None else item.to_record()

        return {
            "name": self.name,
            "design": self.design,
            "scale": self.scale,
            "unit": self.unit,
            "direction": self.direction,
            "n_baseline": self.n_baseline,
            "n_candidate": self.n_candidate,
            "n_pairs": self.n_pairs,
            "values": self.values,
            "attrition": self.attrition,
            "effect": None if self.worst is None else self.worst.effect,
            "interval": _interval(self.worst),
            "estimates": {
                "worst_case": record(self.worst),
                "best_case": record(self.best),
            },
            "cross_check": record(self.cross_check),
            "sensitivity": record(self.sensitivity),
            "difference": record(self.difference),
            "guards": self.guards.to_record(),
            "effect_size": self.effect_size,
            "verdict": self.verdict,
            "gate": record(self.gate),
            "reason": self.reason,
            "degenerate": self.degenerate,
            "pooled": self.pooled,
            "trials": self.trials,
        }


@dataclass(frozen=True)
class ProportionBound:
    """A Clopper-Pearson interval, labelled with the model it assumes."""

    k: int
    n: int
    lower: float
    upper: float
    confidence: float
    model: str

    @property
    def model_based(self) -> bool:
        return self.model != "independent_runs"

    def to_record(self) -> dict[str, Any]:
        return {**self.__dict__, "model_based": self.model_based}


@dataclass(frozen=True)
class PrecisionPlan:
    n: int
    half_width_log: float
    meets_target: bool


# ------------------------------------------------------------------- pairs


@dataclass(frozen=True)
class _Pairs:
    design: str
    baseline: list[tuple[float, float]]
    candidate: list[tuple[float, float]]
    attrition: list[dict[str, Any]]
    n_baseline: int
    n_candidate: int
    # Requests behind each kept value, when a fraction's are known.
    trials: tuple[list[int | None], list[int | None]] = ([], [])


def _bounds(value: RunValue) -> tuple[float, float] | None:
    """A run's value as (lower, upper); a missing or non-finite one is None."""
    if value is None:
        return None
    low, high = value if isinstance(value, tuple) else (value, value)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        return None
    return float(low), float(high)


Trials = tuple[Sequence[int | None], Sequence[int | None]]
_Run = tuple[tuple[float, float] | None, int | None]


def _pair(
    baseline: Sequence[RunValue],
    candidate: Sequence[RunValue],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]] | None,
    trials: Trials | None = None,
) -> _Pairs:
    counts = trials or ([None] * len(baseline), [None] * len(candidate))
    if len(counts[0]) != len(baseline) or len(counts[1]) != len(candidate):
        raise ValueError("every run needs its request count")
    first = [(_bounds(v), m) for v, m in zip(baseline, counts[0])]
    second = [(_bounds(v), m) for v, m in zip(candidate, counts[1])]
    if blocks is None:
        return _unpaired(first, second)
    return _paired(first, second, blocks)


def _unpaired(first: list[_Run], second: list[_Run]) -> _Pairs:
    attrition = [
        {"arm": arm, "run": index, "reason": "missing_value"}
        for arm, runs in (("baseline", first), ("candidate", second))
        for index, (value, _) in enumerate(runs)
        if value is None
    ]
    (values_a, counts_a), (values_b, counts_b) = _split(first), _split(second)
    return _Pairs(
        INDEPENDENT,
        values_a,
        values_b,
        attrition,
        len(first),
        len(second),
        (counts_a, counts_b),
    )


def _split(
    runs: list[_Run],
) -> tuple[list[tuple[float, float]], list[int | None]]:
    """The runs with a value: their values, and their request counts."""
    kept = [(value, m) for value, m in runs if value is not None]
    return [value for value, _ in kept], [m for _, m in kept]


def _paired(
    first: list[_Run],
    second: list[_Run],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]],
) -> _Pairs:
    labels_a, labels_b = list(blocks[0]), list(blocks[1])
    if len(labels_a) != len(first) or len(labels_b) != len(second):
        raise ValueError("every run needs a block label")
    by_a = _by_block(labels_a, first, "baseline")
    by_b = _by_block(labels_b, second, "candidate")
    kept_a: list[_Run] = []
    kept_b: list[_Run] = []
    attrition = []
    for block in sorted(set(by_a) | set(by_b), key=str):
        a, b = by_a.get(block), by_b.get(block)
        if a is None or b is None:
            missing = "baseline" if a is None else "candidate"
            attrition.append({"block": block, "reason": f"block_incomplete:{missing}"})
            continue
        kept_a.append(a)
        kept_b.append(b)
    (values_a, counts_a), (values_b, counts_b) = _split(kept_a), _split(kept_b)
    return _Pairs(
        PAIRED,
        values_a,
        values_b,
        attrition,
        len(first),
        len(second),
        (counts_a, counts_b),
    )


def _by_block(
    labels: list[Hashable], runs: list[_Run], arm: str
) -> dict[Hashable, _Run]:
    found: dict[Hashable, _Run] = {}
    for label, run in zip(labels, runs):
        if label in found:
            raise ValueError(f"block {label!r} has two {arm} runs")
        found[label] = run
    return {label: run for label, run in found.items() if run[0] is not None}


def _cases(
    pairs: _Pairs, direction: str
) -> tuple[tuple[list[float], list[float]], tuple[list[float], list[float]]]:
    """The worst and the best case of every pair, for the metric's direction.

    For ``higher_is_better`` the worst case is the candidate's lower bound
    against the baseline's upper bound.
    """
    a_low = [low for low, _ in pairs.baseline]
    a_high = [high for _, high in pairs.baseline]
    b_low = [low for low, _ in pairs.candidate]
    b_high = [high for _, high in pairs.candidate]
    if direction == HIGHER_IS_BETTER:
        return (a_high, b_low), (a_low, b_high)
    return (a_low, b_high), (a_high, b_low)


# ----------------------------------------------------------------- methods


def _on_scale(values: Sequence[float], scale: str) -> np.ndarray:
    """Values as the method sees them: logs for a log ratio."""
    array: np.ndarray = np.asarray(values, dtype=float)
    if scale == LOG_RATIO:
        logged: np.ndarray = np.log(array)
        return logged
    return array


def _t_quantile(confidence: float, df: float) -> float:
    return float(stats.t.ppf(0.5 + confidence / 2, df))


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _one_sample(
    values: list[float],
    confidence: float,
    method: str,
    estimand: str,
    sign: float,
    floor: float | None = None,
) -> Estimate:
    """A t interval for the mean; ``sign`` turns the bad direction positive.

    ``floor`` is the smallest standard error the interval may use.
    """
    n = len(values)
    mean, sd = _mean_sd(values)
    se = max(sd / math.sqrt(n), floor or 0.0)
    half = _t_quantile(confidence, n - 1) * se
    p_worse = _p_positive(sign * mean, se, n - 1)
    detail = None if floor is None else BINOMIAL_FLOOR
    return Estimate(
        method, estimand, mean, mean - half, mean + half, n - 1, n, p_worse, detail
    )


def _welch(
    first: list[float],
    second: list[float],
    confidence: float,
    sign: float,
    floor: float | None = None,
) -> Estimate:
    """Welch t for mean(second) - mean(first), df = min(nA, nB) - 1."""
    mean_a, sd_a = _mean_sd(first)
    mean_b, sd_b = _mean_sd(second)
    se = max(math.sqrt(sd_a**2 / len(first) + sd_b**2 / len(second)), floor or 0.0)
    df = min(len(first), len(second)) - 1
    half = _t_quantile(confidence, df) * se
    diff = mean_b - mean_a
    p_worse = _p_positive(sign * diff, se, df)
    n = len(first) + len(second)
    detail = None if floor is None else BINOMIAL_FLOOR
    return Estimate(
        "welch_t_min_df", "", diff, diff - half, diff + half, df, n, p_worse, detail
    )


def _p_positive(mean: float, se: float, df: float) -> float | None:
    """One-sided p-value that the bad-oriented mean exceeds 0."""
    if se == 0:
        return None
    return float(stats.t.sf(mean / se, df))


def _primary(
    design: str,
    scale: str,
    first: list[float],
    second: list[float],
    confidence: float,
    sign: float,
    floor: float | None = None,
) -> Estimate:
    """The gated interval, on the effect scale; a fraction's is descriptive."""
    if scale == ABSOLUTE:
        return _one_sample(second, confidence, "one_sample_t", ABSOLUTE, sign)
    a = list(_on_scale(first, scale))
    b = list(_on_scale(second, scale))
    if design == PAIRED:
        diffs = [y - x for x, y in zip(a, b)]
        raw = _one_sample(diffs, confidence, "paired_t", "", sign, floor)
    else:
        raw = _welch(a, b, confidence, sign, floor)
    if scale == LOG_RATIO:
        return _relative(raw, "geometric_ratio")
    return Estimate(
        raw.method,
        DIFFERENCE,
        raw.effect,
        raw.lower,
        raw.upper,
        raw.df,
        raw.n,
        raw.p_worse,
        raw.detail,
    )


def _relative(raw: Estimate, estimand: str) -> Estimate:
    """A log-scale interval as a relative change: exp(x) - 1."""

    def rel(x: float | None) -> float | None:
        return None if x is None else math.expm1(x)

    return Estimate(
        raw.method + "_log_ratio",
        estimand,
        rel(raw.effect),
        rel(raw.lower),
        rel(raw.upper),
        raw.df,
        raw.n,
        raw.p_worse,
    )


def _fieller(
    design: str, first: list[float], second: list[float], confidence: float
) -> Estimate:
    """Fieller's interval for mean(second) / mean(first), as a relative change."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    n_a, n_b = len(a), len(b)
    if design == PAIRED:
        var_a, var_b = a.var(ddof=1) / n_a, b.var(ddof=1) / n_b
        cov = float(np.cov(a, b, ddof=1)[0, 1]) / n_a
        df = n_a - 1
    else:
        var_a, var_b = a.var(ddof=1) / n_a, b.var(ddof=1) / n_b
        cov = 0.0
        df = min(n_a, n_b) - 1
    t = _t_quantile(confidence, df)
    mean_a, mean_b = float(a.mean()), float(b.mean())
    quad = mean_a**2 - t**2 * var_a
    lin = -2 * (mean_a * mean_b - t**2 * cov)
    const = mean_b**2 - t**2 * var_b
    disc = lin**2 - 4 * quad * const
    effect = mean_b / mean_a - 1 if mean_a else None
    n = n_a if design == PAIRED else n_a + n_b
    if quad <= 0 or disc < 0:
        return Estimate(
            "fieller", "arithmetic_ratio", effect, None, None, df, n, None, "unbounded"
        )
    root = math.sqrt(disc)
    low, high = sorted(((-lin - root) / (2 * quad), (-lin + root) / (2 * quad)))
    return Estimate("fieller", "arithmetic_ratio", effect, low - 1, high - 1, df, n)


def _bootstrap(
    design: str,
    scale: str,
    first: list[float],
    second: list[float],
    confidence: float,
    seed: int,
) -> Estimate:
    """Percentile bootstrap of the effect: blocks in pairs, or runs within arms."""
    rng = np.random.default_rng(seed)
    a = _on_scale(first, scale)
    b = _on_scale(second, scale)
    if design == PAIRED:
        diffs = b - a
        picks = rng.integers(0, len(diffs), (BOOTSTRAP_RESAMPLES, len(diffs)))
        means = diffs[picks].mean(axis=1)
        method = "block_bootstrap"
    else:
        pick_a = rng.integers(0, len(a), (BOOTSTRAP_RESAMPLES, len(a)))
        pick_b = rng.integers(0, len(b), (BOOTSTRAP_RESAMPLES, len(b)))
        means = b[pick_b].mean(axis=1) - a[pick_a].mean(axis=1)
        method = "cluster_bootstrap"
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    effect = float(means.mean())
    raw = Estimate(
        method, "", effect, float(low), float(high), None, len(a), None, f"seed {seed}"
    )
    if scale == LOG_RATIO:
        return _relative(raw, "geometric_ratio")
    return Estimate(
        method, DIFFERENCE, effect, float(low), float(high), None, len(a), None
    )


# ------------------------------------------------------------------ guards


def skew_limit(n: int) -> float | None:
    """The 95% quantile of ``|G1|`` under normal noise at n; 1.96 SE beyond 30."""
    if n < 3:
        return None
    if n in SKEW_Q95_NORMAL:
        return SKEW_Q95_NORMAL[n]
    se = math.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    return 1.96 * se


def _skew(values: list[float]) -> tuple[float | None, float | None, bool | None]:
    if len(values) < 3 or np.ptp(values) == 0:
        return None, skew_limit(len(values)), None
    g1 = float(stats.skew(values, bias=False))
    limit = skew_limit(len(values))
    return g1, limit, bool(limit is not None and abs(g1) > limit)


def _guard_values(
    design: str, scale: str, first: list[float], second: list[float]
) -> list[float]:
    """What the skew screen looks at: block effects, or the candidate's values."""
    a = list(_on_scale(first, scale))
    b = list(_on_scale(second, scale))
    if design == PAIRED and scale != ABSOLUTE:
        return [y - x for x, y in zip(a, b)]
    return b


# ------------------------------------------------------------------- gates


def _bad(estimate: Estimate, direction: str) -> tuple[float, float, float] | None:
    """The estimate as a change in the bad direction: (effect, lower, upper)."""
    if estimate.effect is None or estimate.lower is None or estimate.upper is None:
        return None
    if direction == LOWER_IS_BETTER:
        return estimate.effect, estimate.lower, estimate.upper
    return -estimate.effect, -estimate.upper, -estimate.lower


def _decide(
    rule: str, budget: float, worst: Estimate, best: Estimate, direction: str
) -> str:
    """pass or fail under a gate rule, on the worst or the best case."""
    estimate = worst if rule == NON_INFERIORITY else best
    bad = _bad(estimate, direction)
    if bad is None:
        return NOT_EVALUABLE
    effect, lower, upper = bad
    if rule == NON_INFERIORITY:
        failed = upper > budget
    elif rule == SIGNIFICANT:
        failed = lower > 0 and effect > budget
    else:
        failed = lower > budget
    return FAIL if failed else PASS


def _verdict(
    worst: Estimate | None, best: Estimate | None, direction: str, tolerance: float
) -> dict[str, str]:
    """direction against 0 and tolerance against +-tau; regressions on the best case."""
    bad_best = _bad(best, direction) if best is not None else None
    bad_worst = _bad(worst, direction) if worst is not None else None
    if bad_best is None or bad_worst is None:
        return {"direction": "undetermined", "tolerance": "undetermined"}
    return {
        "direction": _moved(bad_best, bad_worst),
        "tolerance": _against_tolerance(bad_best, bad_worst, tolerance),
    }


def _moved(
    bad_best: tuple[float, float, float], bad_worst: tuple[float, float, float]
) -> str:
    if bad_best[1] > 0:
        return "worse"
    if bad_worst[2] < 0:
        return "better"
    return "no_detectable_change"


def _against_tolerance(
    bad_best: tuple[float, float, float],
    bad_worst: tuple[float, float, float],
    tolerance: float,
) -> str:
    if -tolerance < bad_worst[1] and bad_worst[2] < tolerance:
        return "within"
    if bad_best[1] > tolerance or bad_worst[2] < -tolerance:
        return "beyond"
    return "undetermined"


# -------------------------------------------------------------- the entry


@dataclass(frozen=True)
class _Request:
    name: str
    direction: str
    scale: str
    unit: str
    value_unit: str | None
    confidence: float
    tolerance: float
    gate: GateRule | None
    bootstrap_min_n: int
    seed: int
    unavailable: str | None
    trials: Trials | None = None


def compare_values(
    name: str,
    baseline: Sequence[RunValue],
    candidate: Sequence[RunValue],
    *,
    direction: Direction,
    scale: Scale,
    unit: str,
    value_unit: str | None = None,
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]] | None = None,
    confidence: float = 0.95,
    tolerance: float | None = None,
    gate: GateRule | None = None,
    bootstrap_min_n: int = 10,
    seed: int = 0,
    unavailable: str | None = None,
    trials: Trials | None = None,
) -> MetricComparison:
    """Compare one metric's per-run values between the two arms.

    ``unit`` is the effect's unit: ``relative`` for ``log_ratio``, the
    metric's own unit otherwise. ``value_unit`` is the unit of the raw
    values, for a ``log_ratio`` metric's difference and fallback budget.
    ``unavailable`` names why the caller already knows the metric cannot be
    gated (``censored``, ``unverified``, ...). An infinite candidate value of
    a lower-is-better metric is the worst value there is: the gate fails
    (``candidate_censored_worst``), read like a zero candidate on a
    higher-is-better one. Any other value that is not finite makes the gate
    ``non_finite_value``; dropping it as missing would decide the gate on
    the runs left. A duplicate block label is a ValueError.

    ``trials`` are the requests behind each run's value, for a ``fraction``
    metric: its gate is the run-level claim, and they floor its interval.
    """
    unavailable = unavailable or _non_finite_reason(direction, baseline, candidate)
    request = _check(
        _Request(
            name,
            direction,
            scale,
            unit,
            value_unit,
            confidence,
            gate.budget if tolerance is None and gate is not None else tolerance or 0.0,
            gate,
            bootstrap_min_n,
            seed,
            unavailable,
            trials,
        )
    )
    # The candidate alone is measured on an absolute scale: no pairing.
    pairs = _pair(baseline, candidate, None if scale == ABSOLUTE else blocks, trials)
    result = _compare(request, pairs)
    if request.unit == FRACTION_UNIT:
        outcome = _run_level_gate(request, baseline, candidate, blocks)
        return _with(result, gate=outcome, pooled=_pooled(request, pairs))
    if request.unavailable == CENSORED_WORST and request.gate is not None:
        outcome = _censored_gate(replace(request, unavailable=None), pairs)
        result = _with(result, gate=outcome)
    return result


CENSORED_WORST = "candidate_censored_worst"


def _non_finite_reason(
    direction: str, baseline: Sequence[RunValue], candidate: Sequence[RunValue]
) -> str | None:
    """Why a value that is not finite keeps the gate from its interval."""
    if not _non_finite([*baseline, *candidate]):
        return None
    if direction == LOWER_IS_BETTER and _positive_infinite(candidate):
        return CENSORED_WORST
    return "non_finite_value"


def _positive_infinite(values: Sequence[RunValue]) -> bool:
    for value in values:
        bounds = value if isinstance(value, tuple) else (value,)
        if any(bound == math.inf for bound in bounds if bound is not None):
            return True
    return False


def _censored_gate(request: _Request, pairs: _Pairs) -> GateOutcome:
    """A candidate run worse than any value: the gate fails.

    A regression rule waits for the gate's blockers first, as with a zero.
    """
    gate = request.gate
    assert gate is not None
    if gate.rule != NON_INFERIORITY:
        blocker = _gate_blocker(request, pairs, Guards())
        if blocker is not None:
            return GateOutcome(NOT_EVALUABLE, blocker, gate)
    return GateOutcome(FAIL, CENSORED_WORST, gate)


def _non_finite(values: Sequence[RunValue]) -> bool:
    for value in values:
        if value is None:
            continue
        bounds = value if isinstance(value, tuple) else (value,)
        if not all(math.isfinite(bound) for bound in bounds):
            return True
    return False


def _check(request: _Request) -> _Request:
    if request.direction not in (LOWER_IS_BETTER, HIGHER_IS_BETTER):
        raise InferUsageError(
            f"{request.name}: unknown direction {request.direction!r}"
        )
    if request.scale not in (LOG_RATIO, DIFFERENCE, ABSOLUTE):
        raise InferUsageError(f"{request.name}: unknown scale {request.scale!r}")
    if request.scale == LOG_RATIO and request.unit != RELATIVE:
        raise InferUsageError(f"{request.name}: a log_ratio effect's unit is relative")
    gate = request.gate
    if gate is not None and gate.unit != request.unit:
        raise InferUsageError(
            f"{request.name}: the gate's unit {gate.unit!r} is not the metric's "
            f"{request.unit!r}"
        )
    _check_fraction_gate(request)
    if gate is not None and gate.fallback_budget is not None:
        if gate.fallback_unit != request.value_unit:
            raise InferUsageError(
                f"{request.name}: the fallback budget's unit {gate.fallback_unit!r} "
                f"is not the values' {request.value_unit!r}"
            )
    return request


def _check_fraction_gate(request: _Request) -> None:
    gate = request.gate
    if gate is None or request.unit != FRACTION_UNIT:
        return
    if gate.rule != NON_INFERIORITY:
        raise InferUsageError(
            f"{request.name}: {gate.rule} is not supported for a fraction. A "
            "fraction supports only a run-level claim, gated with "
            "non-inferiority: k of n runs within the budget, which stays exact "
            "however failures cluster within a run, where no interval on the "
            "fraction does; it has no regression form in v1"
        )


def _compare(request: _Request, pairs: _Pairs) -> MetricComparison:
    base = _shell(request, pairs)
    n = len(pairs.candidate) if request.scale == ABSOLUTE else len(pairs.baseline)
    if pairs.design == INDEPENDENT and request.scale != ABSOLUTE:
        n = min(len(pairs.baseline), len(pairs.candidate))
    if n < 2:
        reason = (
            "insufficient_blocks" if pairs.design == PAIRED else "insufficient_runs"
        )
        return _with(base, reason=reason, gate=_not_evaluable(request, reason))
    worst_case, best_case = _cases(pairs, request.direction)
    if _degenerate_without_floor(request, pairs, worst_case):
        return _degenerate(base, request, pairs)
    if request.scale == LOG_RATIO:
        blocked = _nonpositive(request, worst_case, best_case)
        if blocked is not None:
            return _zero_rule(base, request, pairs, blocked)
    return _estimated(base, request, pairs, worst_case, best_case)


def _degenerate_without_floor(
    request: _Request, pairs: _Pairs, worst_case: tuple[list[float], list[float]]
) -> bool:
    """One value in every run, and no request counts to floor an interval with."""
    if request.scale != DIFFERENCE or _constant(pairs) is None:
        return False
    return _floor(request, pairs, worst_case) is None


def _shell(request: _Request, pairs: _Pairs) -> MetricComparison:
    return MetricComparison(
        name=request.name,
        design=pairs.design if request.scale != ABSOLUTE else ABSOLUTE,
        scale=request.scale,
        unit=request.unit,
        direction=request.direction,
        n_baseline=pairs.n_baseline,
        n_candidate=pairs.n_candidate,
        n_pairs=len(pairs.baseline) if pairs.design == PAIRED else 0,
        values={
            "baseline": [list(v) for v in pairs.baseline],
            "candidate": [list(v) for v in pairs.candidate],
        },
        attrition=pairs.attrition,
        worst=None,
        best=None,
        trials=(
            {"baseline": pairs.trials[0], "candidate": pairs.trials[1]}
            if request.unit == FRACTION_UNIT and request.trials is not None
            else None
        ),
    )


def _with(item: MetricComparison, **changes: Any) -> MetricComparison:
    return MetricComparison(**{**item.__dict__, **changes})


def _sign(direction: str) -> float:
    return 1.0 if direction == LOWER_IS_BETTER else -1.0


def _estimated(
    base: MetricComparison,
    request: _Request,
    pairs: _Pairs,
    worst_case: tuple[list[float], list[float]],
    best_case: tuple[list[float], list[float]],
) -> MetricComparison:
    sign = _sign(request.direction)
    worst, best = (
        _primary(
            pairs.design,
            request.scale,
            *case,
            request.confidence,
            sign,
            _floor(request, pairs, case),
        )
        for case in (worst_case, best_case)
    )
    guards = _guards(request, pairs, worst_case, worst, best)
    result = _with(
        base,
        worst=worst,
        best=best,
        cross_check=_cross_check(request, pairs, worst_case),
        sensitivity=_sensitivity(request, pairs, worst_case),
        difference=_companion(request, pairs, worst_case),
        guards=guards,
        effect_size=_effect_size(pairs),
        verdict=_verdict(worst, best, request.direction, request.tolerance),
    )
    return _with(result, gate=_gate(request, pairs, worst, best, guards))


def _cross_check(
    request: _Request, pairs: _Pairs, case: tuple[list[float], list[float]]
) -> Estimate | None:
    if request.scale != LOG_RATIO:
        return None
    return _fieller(pairs.design, *case, request.confidence)


def _sensitivity(
    request: _Request, pairs: _Pairs, case: tuple[list[float], list[float]]
) -> Estimate | None:
    if request.scale == ABSOLUTE or min(map(len, case)) < request.bootstrap_min_n:
        return None
    return _bootstrap(
        pairs.design, request.scale, *case, request.confidence, request.seed
    )


def _companion(
    request: _Request, pairs: _Pairs, case: tuple[list[float], list[float]]
) -> Estimate | None:
    """A log_ratio metric's difference, on the values' own scale; never gated."""
    if request.scale != LOG_RATIO:
        return None
    return _primary(
        pairs.design, DIFFERENCE, *case, request.confidence, _sign(request.direction)
    )


def _guards(
    request: _Request,
    pairs: _Pairs,
    case: tuple[list[float], list[float]],
    worst: Estimate,
    best: Estimate,
) -> Guards:
    looked_at = _guard_values(pairs.design, request.scale, *case)
    g1, limit, flagged = _skew(looked_at)
    # A fraction's gate is a claim about runs, which no single block decides.
    loo, flips = (
        (None, None)
        if request.unit == FRACTION_UNIT
        else _leave_one_out(request, pairs, worst, best)
    )
    return Guards(
        skewness=g1,
        skew_limit=limit,
        skew_flagged=flagged,
        leave_one_out=loo,
        decision_flips=flips,
        zero_variance=bool(np.ptp(looked_at) == 0) if looked_at else False,
    )


def _leave_one_out(
    request: _Request, pairs: _Pairs, worst: Estimate, best: Estimate
) -> tuple[dict[str, float | None] | None, int | None]:
    """The interval's range without each pair (or run), and decision changes."""
    reduced = _drop_one(pairs)
    if not reduced:
        return None, None
    decision = _rule_decision(request, worst, best)
    estimates = [_estimate_pair(request, subset) for subset in reduced]
    flips = sum(
        1
        for sub_worst, sub_best in estimates
        if _rule_decision(request, sub_worst, sub_best) != decision
    )
    lowers = [e.lower for e, _ in estimates if e.lower is not None]
    uppers = [e.upper for e, _ in estimates if e.upper is not None]
    span: dict[str, float | None] = {
        "lower_min": min(lowers, default=None),
        "lower_max": max(lowers, default=None),
        "upper_min": min(uppers, default=None),
        "upper_max": max(uppers, default=None),
    }
    return span, (flips if decision is not None else None)


def _estimate_pair(request: _Request, pairs: _Pairs) -> tuple[Estimate, Estimate]:
    """The worst- and best-case primary estimates of a sample."""
    sign = _sign(request.direction)
    worst_case, best_case = _cases(pairs, request.direction)
    return (
        _primary(pairs.design, request.scale, *worst_case, request.confidence, sign),
        _primary(pairs.design, request.scale, *best_case, request.confidence, sign),
    )


def _drop_one(pairs: _Pairs) -> list[_Pairs]:
    """Every sample with one pair (or one run of either arm) left out.

    Only samples that could still be gated count: with fewer than three
    pairs left, an interval is always undetermined, so a change of
    decision there says nothing about the block left out.
    """
    if pairs.design == PAIRED:
        if len(pairs.baseline) <= MIN_GATE_PAIRS:
            return []
        return [
            _Pairs(
                PAIRED,
                pairs.baseline[:i] + pairs.baseline[i + 1 :],
                pairs.candidate[:i] + pairs.candidate[i + 1 :],
                [],
                pairs.n_baseline,
                pairs.n_candidate,
            )
            for i in range(len(pairs.baseline))
        ]
    subsets = []
    for arm in ("baseline", "candidate"):
        values = getattr(pairs, arm)
        if len(values) <= MIN_GATE_PAIRS:
            continue
        for i in range(len(values)):
            kept = values[:i] + values[i + 1 :]
            first = kept if arm == "baseline" else pairs.baseline
            second = kept if arm == "candidate" else pairs.candidate
            subsets.append(
                _Pairs(
                    INDEPENDENT, first, second, [], pairs.n_baseline, pairs.n_candidate
                )
            )
    return subsets


def _rule_decision(request: _Request, worst: Estimate, best: Estimate) -> str | None:
    if request.gate is None:
        return None
    return _decide(
        request.gate.rule, request.gate.budget, worst, best, request.direction
    )


def _effect_size(pairs: _Pairs) -> dict[str, Any]:
    """Paired probability of superiority, or Vargha-Delaney A12 over run pairs."""
    first = [(low + high) / 2 for low, high in pairs.baseline]
    second = [(low + high) / 2 for low, high in pairs.candidate]
    if pairs.design == PAIRED:
        wins = sum(_beats(b, a) for a, b in zip(first, second))
        return {
            "name": "paired_superiority",
            "value": wins / len(first),
            "count": len(first),
        }
    count = len(first) * len(second)
    total = sum(_beats(b, a) for a in first for b in second)
    return {"name": "a12", "value": total / count if count else None, "count": count}


def _beats(candidate: float, baseline: float) -> float:
    """1 when the candidate is higher, 1/2 for a tie, 0 otherwise."""
    if candidate == baseline:
        return 0.5
    return 1.0 if candidate > baseline else 0.0


def _gate(
    request: _Request,
    pairs: _Pairs,
    worst: Estimate,
    best: Estimate,
    guards: Guards,
) -> GateOutcome | None:
    gate = request.gate
    if gate is None:
        return None
    reason = _gate_blocker(request, pairs, guards)
    if reason is not None:
        return GateOutcome(NOT_EVALUABLE, reason, gate)
    status = _decide(gate.rule, gate.budget, worst, best, request.direction)
    case = "worst_case" if gate.rule == NON_INFERIORITY else "best_case"
    reason = (
        None if status == PASS else "exceeds_budget" if status == FAIL else "unbounded"
    )
    return GateOutcome(status, reason, gate, case)


def _gate_blocker(request: _Request, pairs: _Pairs, guards: Guards) -> str | None:
    """Why a gate cannot be decided, before its interval is looked at."""
    gate = request.gate
    assert gate is not None
    complete = _complete(pairs, request.scale)
    if request.unavailable is not None:
        return request.unavailable
    if gate.min_complete_blocks is not None and complete < gate.min_complete_blocks:
        return "blocks_below_preregistered"
    if complete < MIN_GATE_PAIRS:
        return "insufficient_blocks" if pairs.design == PAIRED else "insufficient_runs"
    if guards.decision_flips:
        return "decision_unstable"
    return None


def _complete(pairs: _Pairs, scale: str) -> int:
    if scale == ABSOLUTE:
        return len(pairs.candidate)
    if pairs.design == PAIRED:
        return len(pairs.baseline)
    return min(len(pairs.baseline), len(pairs.candidate))


def _not_evaluable(request: _Request, reason: str) -> GateOutcome | None:
    if request.gate is None:
        return None
    return GateOutcome(NOT_EVALUABLE, request.unavailable or reason, request.gate)


# -------------------------------------------------------------- zero rules


def _constant(pairs: _Pairs) -> float | None:
    """The one value every run of both arms had, if there is one."""
    values = {x for pair in pairs.baseline + pairs.candidate for x in pair}
    return values.pop() if len(values) == 1 else None


def _degenerate(
    base: MetricComparison, request: _Request, pairs: _Pairs
) -> MetricComparison:
    """Every run of both arms had one value: no interval, only a run-level bound.

    That is a failure rate of 0, or an attainment of 1, in every run. A t
    interval would be a falsely certain [0, 0]. What the runs do show is
    that, with the confidence asked for, a run departs from that value (has
    any failure, misses any request) with probability at most
    1 - (alpha/2)^(1/n).
    """
    value = _constant(pairs)
    kind = "degenerate_zero" if value == 0 else "degenerate_constant"
    alpha = 1 - request.confidence
    bounds = {
        arm: 1 - (alpha / 2) ** (1 / count)
        for arm, count in (
            ("baseline", len(pairs.baseline)),
            ("candidate", len(pairs.candidate)),
        )
        if count
    }
    degenerate = {
        "kind": kind,
        "value": value,
        "run_departs_upper": bounds,
        "statement": "every run of both arms had this value; a run departs "
        "from it with probability at most the bound shown, per arm",
    }
    gate = None
    if request.gate is not None:
        # No interval: the run-level bound alone shows nothing within budget.
        blocker = _gate_blocker(request, pairs, Guards())
        gate = GateOutcome(NOT_EVALUABLE, blocker or kind, request.gate)
    verdict = {"direction": "no_detectable_change", "tolerance": "undetermined"}
    return _with(base, reason=kind, degenerate=degenerate, gate=gate, verdict=verdict)


def _nonpositive(
    request: _Request,
    worst_case: tuple[list[float], list[float]],
    best_case: tuple[list[float], list[float]],
) -> str | None:
    """Why a log ratio is undefined: a zero or negative value in an arm."""
    baseline = worst_case[0] + best_case[0]
    candidate = worst_case[1] + best_case[1]
    if any(value <= 0 for value in baseline):
        return "undefined_in_arm"
    if any(value <= 0 for value in candidate):
        return (
            "candidate_zero"
            if request.direction == HIGHER_IS_BETTER
            else "undefined_in_arm"
        )
    return None


def _zero_rule(
    base: MetricComparison, request: _Request, pairs: _Pairs, blocked: str
) -> MetricComparison:
    """A log ratio with a zero: the difference is reported, and gated only by a fallback.

    A candidate that serves nothing on a higher-is-better metric is the
    worst regression there is, so its gate fails whatever the budget.
    """
    worst_case, best_case = _cases(pairs, request.direction)
    sign = _sign(request.direction)
    worst = _primary(pairs.design, DIFFERENCE, *worst_case, request.confidence, sign)
    best = _primary(pairs.design, DIFFERENCE, *best_case, request.confidence, sign)
    result = _with(
        base, difference=worst, reason=blocked, effect_size=_effect_size(pairs)
    )
    gate = request.gate
    if gate is None:
        return result
    if blocked == "candidate_zero":
        outcome = _candidate_zero_gate(request, pairs, worst_case, best_case)
        if outcome is not None:
            return _with(result, gate=outcome)
        blocked = "undefined_in_arm"
    blocker = _gate_blocker(request, pairs, Guards())
    if blocker is not None or gate.fallback_budget is None:
        return _with(result, gate=GateOutcome(NOT_EVALUABLE, blocker or blocked, gate))
    status = _decide(gate.rule, gate.fallback_budget, worst, best, request.direction)
    reason = "fallback_budget" if status == PASS else "exceeds_fallback_budget"
    return _with(result, gate=GateOutcome(status, reason, gate, "fallback"))


def _candidate_zero_gate(
    request: _Request,
    pairs: _Pairs,
    worst_case: tuple[list[float], list[float]],
    best_case: tuple[list[float], list[float]],
) -> GateOutcome | None:
    """A candidate that served nothing, judged in the case its rule reads.

    Non-inferiority reads the worst case, a regression rule the best: a
    zero only in the case the rule does not read, from missing evidence,
    shows no regression (None: the log ratio is undefined there). A
    regression rule waits for the gate's blockers first.
    """
    gate = request.gate
    assert gate is not None
    regression = gate.rule != NON_INFERIORITY
    if regression:
        blocker = _gate_blocker(request, pairs, Guards())
        if blocker is not None:
            return GateOutcome(NOT_EVALUABLE, blocker, gate)
    used = best_case if regression else worst_case
    if any(value <= 0 for value in used[1]):
        return GateOutcome(FAIL, "candidate_zero", gate)
    return None


# ------------------------------------------------------- proportions, plans


# --------------------------------------------------------------- fractions


def _counted(pairs: _Pairs) -> bool:
    """Whether every kept run's request count is known."""
    counts = [*pairs.trials[0], *pairs.trials[1]]
    lengths = (len(pairs.trials[0]), len(pairs.trials[1]))
    return (
        lengths == (len(pairs.baseline), len(pairs.candidate))
        and bool(counts)
        and all(isinstance(m, int) and m > 0 for m in counts)
    )


def _floor(
    request: _Request, pairs: _Pairs, case: tuple[list[float], list[float]]
) -> float | None:
    """A fraction's pooled binomial standard error, with p = (x + 1) / (N + 2)."""
    if request.unit != FRACTION_UNIT or not _counted(pairs):
        return None
    variance = 0.0
    for values, counts in zip(case, pairs.trials):
        total = sum(m for m in counts if m)
        events = sum(v * (m or 0) for v, m in zip(values, counts))
        share = (events + 1) / (total + 2)
        variance += share * (1 - share) / total
    return math.sqrt(variance)


def _pooled(request: _Request, pairs: _Pairs) -> dict[str, Any] | None:
    """Each arm's requests pooled, as if independent: reported, never gated."""
    if not _counted(pairs):
        return None
    worst_case, _ = _cases(pairs, request.direction)
    bounds = {}
    for arm, values, counts in zip(("baseline", "candidate"), worst_case, pairs.trials):
        total = sum(m for m in counts if m)
        events = round(sum(v * (m or 0) for v, m in zip(values, counts)))
        bounds[arm] = clopper_pearson(
            events, total, request.confidence, model="independent_requests"
        ).to_record()
    if request.direction == LOWER_IS_BETTER:
        bad = bounds["candidate"]["upper"] - bounds["baseline"]["lower"]
    else:
        bad = bounds["baseline"]["upper"] - bounds["candidate"]["lower"]
    budget = request.tolerance
    alpha = (1 - request.confidence) / 2
    needed = (
        math.ceil(math.log(alpha) / math.log(1 - budget)) if 0 < budget < 1 else None
    )
    return {
        "model": "independent_requests",
        **bounds,
        "bad_direction_bound": bad,
        "n_eff_for_zero_event_bound": needed,
        "note": "exact only if requests are independent; with correlation rho "
        "within a run the effective sample is N / (1 + (m - 1) rho)",
    }


def _run_level_gate(
    request: _Request,
    baseline: Sequence[RunValue],
    candidate: Sequence[RunValue],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]] | None,
) -> GateOutcome | None:
    """k of n candidate runs within the budget; passes iff the one-sided
    Clopper-Pearson lower bound of k/n is at least ``RUN_PASS_SHARE``.

    A run is judged on its worst case against its block's baseline; a run
    with no value, or with no baseline to judge it by, is a miss. Without
    blocks there is no claim: judged against one estimated baseline mean,
    the runs' judgements are correlated and the bound is not exact.
    """
    gate = request.gate
    if gate is None:
        return None
    blocker = _run_level_blocker(request, baseline, candidate, blocks)
    if blocker is not None:
        return GateOutcome(NOT_EVALUABLE, blocker, gate, RUN_LEVEL)
    assert blocks is not None
    references = _references(baseline, blocks, request)
    within = _runs_within(request, gate.budget, candidate, references)
    k, n = sum(1 for ok in within if ok), len(within)
    bound = clopper_pearson(k, n, request.confidence, model="independent_runs")
    status = PASS if bound.lower >= RUN_PASS_SHARE else FAIL
    reference = "block_baseline"
    sided = 0.5 + request.confidence / 2
    claim = {
        "model": "independent_runs",
        "reference": reference,
        "runs_within_budget": k,
        "runs": n,
        "runs_unmeasurable": sum(1 for value in candidate if _bounds(value) is None),
        "run_pass_lower": bound.lower,
        "required_share": RUN_PASS_SHARE,
        "one_sided_confidence": sided,
        "statement": f"{k} of {n} candidate runs within {gate.budget:g} of the "
        f"{reference.replace('_', ' ')}; run-pass rate at least "
        f"{bound.lower:.3f} with {sided:.1%} confidence",
    }
    reason = None if status == PASS else "too_few_runs_within_budget"
    return GateOutcome(status, reason, gate, RUN_LEVEL, claim)


def _run_level_blocker(
    request: _Request,
    baseline: Sequence[RunValue],
    candidate: Sequence[RunValue],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]] | None,
) -> str | None:
    """Why the run-level claim cannot be made, before any run is judged."""
    gate = request.gate
    assert gate is not None
    if request.unavailable is not None:
        return request.unavailable
    if blocks is None:
        return "fraction_needs_blocks"
    counts = _measured_counts(request.trials, baseline, candidate)
    if counts is None:
        return "requests_per_run_unrecorded"
    if gate.budget <= 0 or min(counts, default=0) < REQUESTS_PER_BUDGET / gate.budget:
        return "too_few_requests_per_run"
    if (
        gate.min_complete_blocks is not None
        and len(candidate) < gate.min_complete_blocks
    ):
        return "blocks_below_preregistered"
    # n of n runs reach the share only when (alpha / 2) ** (1 / n) >= q.
    if (1 - request.confidence) / 2 < RUN_PASS_SHARE ** len(candidate):
        return "too_few_runs_for_claim"
    return None


def _measured_counts(
    trials: Trials | None, baseline: Sequence[RunValue], candidate: Sequence[RunValue]
) -> list[int] | None:
    """The request counts of every run with a value; None if any is unknown."""
    if trials is None:
        return None
    counts = []
    for values, arm in zip((baseline, candidate), trials):
        for value, m in zip(values, arm):
            if _bounds(value) is None:
                continue
            if not isinstance(m, int) or m <= 0:
                return None
            counts.append(m)
    return counts


def _runs_within(
    request: _Request,
    budget: float,
    candidate: Sequence[RunValue],
    references: Sequence[float | None],
) -> list[bool]:
    """Whether each candidate run's worst case is within the budget."""
    sign = _sign(request.direction)
    bad = 1 if request.direction == LOWER_IS_BETTER else 0
    within = []
    for value, reference in zip(candidate, references):
        run = _bounds(value)
        if run is None or reference is None:
            within.append(False)
            continue
        within.append(sign * (run[bad] - reference) <= budget)
    return within


def _references(
    baseline: Sequence[RunValue],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]],
    request: _Request,
) -> list[float | None]:
    """Each candidate run's reference: its block's baseline, on its good
    side, so that the comparison is the worst case."""
    good = 0 if request.direction == LOWER_IS_BETTER else 1
    by_block = {label: _bounds(v) for label, v in zip(blocks[0], baseline)}
    found = []
    for label in blocks[1]:
        partner = by_block.get(label)
        found.append(None if partner is None else partner[good])
    return found


def clopper_pearson(
    k: int,
    n: int,
    confidence: float = 0.95,
    *,
    model: Literal["independent_runs", "independent_requests"],
) -> ProportionBound:
    """The exact binomial interval for k of n, labelled with its model.

    ``independent_runs`` is exact when each run is independent. Counting
    requests as independent (``independent_requests``) ignores correlation
    within a run, so that bound is model-based.
    """
    if not 0 <= k <= n or n < 1:
        raise ValueError("need 0 <= k <= n and n >= 1")
    alpha = 1 - confidence
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return ProportionBound(k, n, lower, upper, confidence, model)


def run_pass_gate(k: int, n: int, q: float, confidence: float = 0.95) -> dict[str, Any]:
    """Whether at least a share q of runs meets the target.

    Passes iff the Clopper-Pearson lower bound of k/n is >= q; at the
    default 0.95 that bound is one-sided at 97.5%, like every other gate's
    side of a two-sided interval. With 6 runs and q = 0.5 it takes 6 of 6,
    and a true run-pass probability below 0.5 passes at most 0.5^6 = 1.6%
    of the time.
    """
    bound = clopper_pearson(k, n, confidence, model="independent_runs")
    return {
        "runs_meeting": k,
        "runs": n,
        "required_share": q,
        "lower_bound": bound.lower,
        "status": PASS if bound.lower >= q else FAIL,
        "model": "independent_runs",
    }


def blocks_for_precision(
    sd_log_ratio: float,
    h: float,
    confidence: float = 0.95,
    n_min: int = 6,
    n_max: int = 10,
) -> PrecisionPlan:
    """The fewest blocks whose t half-width on the log scale is <= log1p(h)."""
    target = math.log1p(h)
    half = math.inf
    for n in range(max(n_min, 2), n_max + 1):
        half = _t_quantile(confidence, n - 1) * sd_log_ratio / math.sqrt(n)
        if half <= target:
            return PrecisionPlan(n, half, True)
    return PrecisionPlan(n_max, half, False)


def holm(p_values: dict[str, float | None], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down: which hypotheses are rejected at family-wise alpha."""
    known = sorted(
        ((name, p) for name, p in p_values.items() if p is not None), key=lambda x: x[1]
    )
    rejected = {name: False for name in p_values}
    for rank, (name, p) in enumerate(known):
        if p > alpha / (len(known) - rank):
            break
        rejected[name] = True
    return rejected


def _interval(estimate: Estimate | None) -> list[float | None] | None:
    if estimate is None:
        return None
    return [estimate.lower, estimate.upper]


__all__ = [
    "ABSOLUTE",
    "DEMONSTRATED",
    "DIFFERENCE",
    "FAIL",
    "GATE_RULES",
    "HIGHER_IS_BETTER",
    "INDEPENDENT",
    "LOG_RATIO",
    "LOWER_IS_BETTER",
    "MIN_GATE_PAIRS",
    "NON_INFERIORITY",
    "NOT_EVALUABLE",
    "PAIRED",
    "PASS",
    "RELATIVE",
    "SIGNIFICANT",
    "SKEW_Q95_NORMAL",
    "Estimate",
    "GateOutcome",
    "GateRule",
    "Guards",
    "MetricComparison",
    "PrecisionPlan",
    "ProportionBound",
    "blocks_for_precision",
    "clopper_pearson",
    "compare_values",
    "holm",
    "run_pass_gate",
    "skew_limit",
]
