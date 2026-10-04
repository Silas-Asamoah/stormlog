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
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
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

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "case": self.case,
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


def _bounds(value: RunValue) -> tuple[float, float] | None:
    """A run's value as (lower, upper); a missing or non-finite one is None."""
    if value is None:
        return None
    low, high = value if isinstance(value, tuple) else (value, value)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        return None
    return float(low), float(high)


def _pair(
    baseline: Sequence[RunValue],
    candidate: Sequence[RunValue],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]] | None,
) -> _Pairs:
    first = [_bounds(value) for value in baseline]
    second = [_bounds(value) for value in candidate]
    if blocks is None:
        return _unpaired(first, second)
    return _paired(first, second, blocks)


def _unpaired(
    first: list[tuple[float, float] | None], second: list[tuple[float, float] | None]
) -> _Pairs:
    attrition = [
        {"arm": arm, "run": index, "reason": "missing_value"}
        for arm, values in (("baseline", first), ("candidate", second))
        for index, value in enumerate(values)
        if value is None
    ]
    return _Pairs(
        INDEPENDENT,
        [v for v in first if v is not None],
        [v for v in second if v is not None],
        attrition,
        len(first),
        len(second),
    )


def _paired(
    first: list[tuple[float, float] | None],
    second: list[tuple[float, float] | None],
    blocks: tuple[Sequence[Hashable], Sequence[Hashable]],
) -> _Pairs:
    labels_a, labels_b = list(blocks[0]), list(blocks[1])
    if len(labels_a) != len(first) or len(labels_b) != len(second):
        raise ValueError("every run needs a block label")
    by_a = _by_block(labels_a, first, "baseline")
    by_b = _by_block(labels_b, second, "candidate")
    pairs_a, pairs_b, attrition = [], [], []
    for block in sorted(set(by_a) | set(by_b), key=str):
        a, b = by_a.get(block), by_b.get(block)
        if a is None or b is None:
            missing = "baseline" if a is None else "candidate"
            attrition.append({"block": block, "reason": f"block_incomplete:{missing}"})
            continue
        pairs_a.append(a)
        pairs_b.append(b)
    return _Pairs(PAIRED, pairs_a, pairs_b, attrition, len(first), len(second))


def _by_block(
    labels: list[Hashable], values: list[tuple[float, float] | None], arm: str
) -> dict[Hashable, tuple[float, float] | None]:
    found: dict[Hashable, tuple[float, float] | None] = {}
    for label, value in zip(labels, values):
        if label in found:
            raise ValueError(f"block {label!r} has two {arm} runs")
        found[label] = value
    return {label: value for label, value in found.items() if value is not None}


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
    values: list[float], confidence: float, method: str, estimand: str, sign: float
) -> Estimate:
    """A t interval for the mean; ``sign`` turns the bad direction positive."""
    n = len(values)
    mean, sd = _mean_sd(values)
    se = sd / math.sqrt(n)
    half = _t_quantile(confidence, n - 1) * se
    p_worse = _p_positive(sign * mean, se, n - 1)
    return Estimate(method, estimand, mean, mean - half, mean + half, n - 1, n, p_worse)


def _welch(
    first: list[float], second: list[float], confidence: float, sign: float
) -> Estimate:
    """Welch t for mean(second) - mean(first), df = min(nA, nB) - 1."""
    mean_a, sd_a = _mean_sd(first)
    mean_b, sd_b = _mean_sd(second)
    se = math.sqrt(sd_a**2 / len(first) + sd_b**2 / len(second))
    df = min(len(first), len(second)) - 1
    half = _t_quantile(confidence, df) * se
    diff = mean_b - mean_a
    p_worse = _p_positive(sign * diff, se, df)
    n = len(first) + len(second)
    return Estimate(
        "welch_t_min_df", "", diff, diff - half, diff + half, df, n, p_worse
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
) -> Estimate:
    """The gated interval, on the effect scale."""
    if scale == ABSOLUTE:
        return _one_sample(second, confidence, "one_sample_t", ABSOLUTE, sign)
    a = list(_on_scale(first, scale))
    b = list(_on_scale(second, scale))
    if design == PAIRED:
        diffs = [y - x for x, y in zip(a, b)]
        raw = _one_sample(diffs, confidence, "paired_t", "", sign)
    else:
        raw = _welch(a, b, confidence, sign)
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
    """The 95% quantile of |G1| under normal noise at n; 1.96 SE beyond 30."""
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
) -> MetricComparison:
    """Compare one metric's per-run values between the two arms.

    ``unit`` is the effect's unit: ``relative`` for ``log_ratio``, the
    metric's own unit otherwise. ``value_unit`` is the unit of the raw
    values, for a ``log_ratio`` metric's difference and fallback budget.
    ``unavailable`` names why the caller already knows the metric cannot be
    gated (``censored``, ``unverified``, ...). A duplicate block label is a
    ValueError.
    """
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
        )
    )
    # The candidate alone is measured on an absolute scale: no pairing.
    pairs = _pair(baseline, candidate, None if scale == ABSOLUTE else blocks)
    return _compare(request, pairs)


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
    if gate is not None and gate.fallback_budget is not None:
        if gate.fallback_unit != request.value_unit:
            raise InferUsageError(
                f"{request.name}: the fallback budget's unit {gate.fallback_unit!r} "
                f"is not the values' {request.value_unit!r}"
            )
    return request


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
    if request.scale == DIFFERENCE and _constant(pairs) is not None:
        return _degenerate(base, request, pairs)
    if request.scale == LOG_RATIO:
        blocked = _nonpositive(request, worst_case, best_case)
        if blocked is not None:
            return _zero_rule(base, request, pairs, blocked)
    return _estimated(base, request, pairs, worst_case, best_case)


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
    worst = _primary(pairs.design, request.scale, *worst_case, request.confidence, sign)
    best = _primary(pairs.design, request.scale, *best_case, request.confidence, sign)
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
    loo, flips = _leave_one_out(request, pairs, worst, best)
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
        blocker = _gate_blocker(request, pairs, Guards())
        status, reason = (NOT_EVALUABLE, blocker) if blocker else (PASS, kind)
        gate = GateOutcome(status, reason, request.gate)
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
        return _with(result, gate=GateOutcome(FAIL, "candidate_zero", gate))
    blocker = _gate_blocker(request, pairs, Guards())
    if blocker is not None or gate.fallback_budget is None:
        return _with(result, gate=GateOutcome(NOT_EVALUABLE, blocker or blocked, gate))
    status = _decide(gate.rule, gate.fallback_budget, worst, best, request.direction)
    reason = "fallback_budget" if status == PASS else "exceeds_fallback_budget"
    return _with(result, gate=GateOutcome(status, reason, gate, "fallback"))


# ------------------------------------------------------- proportions, plans


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
