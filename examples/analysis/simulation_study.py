"""The comparison's method validation: how often each gate rule is wrong.

Each rule of ``stormlog.infer.comparison_stats`` is simulated where it has
to hold, with a fixed seed and 20,000 replications per cell (the Monte
Carlo standard error of a 2.5% rate is about 0.11 points):

1. **Paired t, non-inferiority, at a true effect equal to the budget.** The
   share of gates that pass is the false-safe rate; it must not exceed
   0.0283 (0.025 plus three standard errors). Normal and t3 noise, at 3,
   6, 8 and 10 blocks, with a treatment effect that varies by block
   (block-by-arm interaction, spread 0, 0.05 and 0.2): a block effect
   shared by both arms cancels in the log ratio, so it tests nothing.
   Leave-one-out is applied as the module applies it. The same for a
   higher-is-better metric at a true fall equal to the budget, and for
   the independent design (min-df Welch) at 3 and 6 runs per arm.
2. **The same with missing outcomes**, 0.5% and 2% of requests unknown, an
   attainment gate judged on the worst case.
3. **Min-df Welch on logs, independent runs:** coverage must be at least
   0.947 in an adversarial 20-against-3 case and four more.
4. **The run-level attainment gate:** at the supremum of a false claim
   (the true share of runs that meet the target equal to q), it must pass
   at most 0.0283 of the time, whatever the correlation within a run.
   Only cells where the gate can pass at all are kept: 6 runs need 6 of 6
   at q = 0.5, and no 10 runs can show q = 0.9.
5. **The skew screen:** under normal noise it must flag 0.05 +- 0.005.
6. **A fraction's descriptive interval** (failure fraction, 1% budget, a
   true difference equal to it): the paired t with its standard error
   floored at the pooled binomial one must be false-safe at most 0.0283
   when failures are independent, at 6-10 blocks of 300 requests and 8 of
   1000. Its gate is the run-level claim, exact under independent runs
   (item 4's arithmetic), so each cell also reports how often the claim
   passed and the true share of runs within the budget.

Three scenarios are outside what the rules claim, and are measured as
limits: paired t under strongly skewed noise (standardized lognormal, sigma
0.8), pooled requests (``--attainment-model bernoulli``) under correlation
within a run, and a fraction's intervals under failures correlated within
a run (beta-binomial, intra-run correlation 0.02 and 0.05): there the
floored t and the pooled requests' bound are false-safe far above 2.5%,
which is why neither gates.

The rules are restated here vectorized, for speed; ``check_against_module``
runs a sample of replications through ``compare_values`` and requires the
same decisions.

    python -m examples.analysis.simulation_study            # 20,000 per cell
    python -m examples.analysis.simulation_study --reps 2000
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

SEED = 213
REPS = 20_000
CRITERION_FALSE_SAFE = 0.0283
CRITERION_COVERAGE = 0.947
BUDGET = 0.05
SIGMA_RUN = 0.05
MIN_GATE_PAIRS = 3

Noise = Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]


# ------------------------------------------------------------------ noise


def normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape)


def t3(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Unit variance: a t3 variable has variance 3.
    return rng.standard_t(3, shape) / math.sqrt(3)


def skewed(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Standardized lognormal, sigma 0.8: the unsupported strong skew."""
    s = 0.8
    x = rng.lognormal(0, s, shape)
    mean = math.exp(s * s / 2)
    sd = math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
    return (x - mean) / sd


def skewed_left(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """The same, reflected: a long tail toward faster runs."""
    return -skewed(rng, shape)


NOISES: dict[str, Noise] = {
    "normal": normal,
    "t3": t3,
    "lognormal_0.8": skewed,
    "lognormal_0.8_reflected": skewed_left,
}


# ------------------------------------------------------- the rules, vectorized


def paired_upper(d: np.ndarray) -> np.ndarray:
    """Upper bound of the paired t interval of mean d, per row (log scale)."""
    n = d.shape[1]
    half = stats.t.ppf(0.975, n - 1) * d.std(axis=1, ddof=1) / math.sqrt(n)
    return d.mean(axis=1) + half


def paired_lower(d: np.ndarray) -> np.ndarray:
    n = d.shape[1]
    half = stats.t.ppf(0.975, n - 1) * d.std(axis=1, ddof=1) / math.sqrt(n)
    return d.mean(axis=1) - half


def non_inferiority_passes(log_d: np.ndarray, budget: float) -> np.ndarray:
    """lower_is_better: pass iff expm1(upper) <= budget, and no block flips it."""
    passes = np.expm1(paired_upper(log_d)) <= budget
    n = log_d.shape[1]
    if n <= MIN_GATE_PAIRS:
        return passes
    stable = np.ones(len(log_d), dtype=bool)
    for left_out in range(n):
        kept = np.delete(log_d, left_out, axis=1)
        stable &= (np.expm1(paired_upper(kept)) <= budget) == passes
    return passes & stable


# ------------------------------------------------------------- scenarios


def paired_false_safe(
    rng: np.random.Generator, noise: Noise, sigma_interaction: float, n: int, reps: int
) -> float:
    """At a true +budget latency change, how often non-inferiority passes.

    The treatment's effect varies by block by ``sigma_interaction`` around
    the budget, its mean. Skewed noise is put in the block log ratios
    themselves, with the same spread as the other cells: two equally
    skewed arms would cancel.
    """
    effect = math.log1p(BUDGET) + sigma_interaction * rng.standard_normal((reps, n))
    if noise in (skewed, skewed_left):
        spread = SIGMA_RUN * math.sqrt(2) * noise(rng, (reps, n))
        return float(non_inferiority_passes(effect + spread, BUDGET).mean())
    base = 5.0 + SIGMA_RUN * noise(rng, (reps, n))
    cand = 5.0 + effect + SIGMA_RUN * noise(rng, (reps, n))
    return float(non_inferiority_passes(cand - base, BUDGET).mean())


def higher_is_better_false_safe(rng: np.random.Generator, n: int, reps: int) -> float:
    """Goodput at a true fall equal to the budget: passes iff lower >= -budget."""
    base = 5.0 + SIGMA_RUN * rng.standard_normal((reps, n))
    cand = 5.0 + math.log1p(-BUDGET) + SIGMA_RUN * rng.standard_normal((reps, n))
    return float(_higher_passes(cand - base).mean())


def independent_false_safe(rng: np.random.Generator, n: int, reps: int) -> float:
    """Independent runs, n per arm, min-df Welch, at a true +budget change.

    Leave-one-out drops a run of either arm, as the module does.
    """
    a = 5.0 + SIGMA_RUN * rng.standard_normal((reps, n))
    b = 5.0 + math.log1p(BUDGET) + SIGMA_RUN * rng.standard_normal((reps, n))
    return float(_independent_passes(a, b).mean())


def _independent_passes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min-df Welch on logs, lower is better, leave-one-out over either arm."""

    def passes(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        na, nb = first.shape[1], second.shape[1]
        se = np.sqrt(first.var(axis=1, ddof=1) / na + second.var(axis=1, ddof=1) / nb)
        upper = second.mean(axis=1) - first.mean(axis=1)
        upper = upper + stats.t.ppf(0.975, min(na, nb) - 1) * se
        return np.expm1(upper) <= BUDGET

    result = passes(a, b)
    n = min(a.shape[1], b.shape[1])
    if n > MIN_GATE_PAIRS:
        stable = np.ones(len(a), dtype=bool)
        for left_out in range(a.shape[1]):
            stable &= passes(np.delete(a, left_out, axis=1), b) == result
        for left_out in range(b.shape[1]):
            stable &= passes(a, np.delete(b, left_out, axis=1)) == result
        result &= stable
    return result


def paired_power(
    rng: np.random.Generator, sigma_run: float, n: int, reps: int
) -> dict[str, float]:
    """With no change at all, how often non-inferiority passes, and how often
    leave-one-out alone stops a pass (the price of the stability guard)."""
    base = 5.0 + sigma_run * rng.standard_normal((reps, n))
    cand = 5.0 + sigma_run * rng.standard_normal((reps, n))
    d = cand - base
    plain = np.expm1(paired_upper(d)) <= BUDGET
    guarded = non_inferiority_passes(d, BUDGET)
    return {
        "pass": float(guarded.mean()),
        "pass_without_leave_one_out": float(plain.mean()),
        "stopped_by_leave_one_out": float((plain & ~guarded).mean()),
    }


def missing_outcome_false_safe(
    rng: np.random.Generator, unknown: float, n: int, reps: int, requests: int = 2000
) -> float:
    """Attainment at a true drop equal to the budget, judged on the worst case."""
    budget = 0.01
    p_base, p_cand = 0.99, 0.99 - budget

    def bounds(p: float) -> tuple[np.ndarray, np.ndarray]:
        met = rng.binomial(requests, p, (reps, n))
        hidden = rng.binomial(met, unknown)  # unknown outcomes, from met ones
        lost = rng.binomial(requests - met, unknown)
        lower = (met - hidden) / requests
        upper = (met - hidden + hidden + lost) / requests
        return lower, upper

    base_low, base_high = bounds(p_base)
    cand_low, cand_high = bounds(p_cand)
    worst = cand_low - base_high  # higher is better: the candidate's bad end
    # higher_is_better passes non-inferiority iff the lower bound >= -budget.
    passes = paired_lower(worst) >= -budget
    if n > MIN_GATE_PAIRS:
        for left_out in range(n):
            kept = np.delete(worst, left_out, axis=1)
            passes &= (paired_lower(kept) >= -budget) == (
                paired_lower(worst) >= -budget
            )
    return float(passes.mean())


INDEPENDENT_CASES = (
    ("20v3, cv 10%/20%, ratio 0.01 (adversarial)", 20, 3, 0.10, 0.20, 0.01),
    ("20v3, cv 5%/15%, ratio 1.0", 20, 3, 0.05, 0.15, 1.0),
    ("5v5, cv 5%/5%, ratio 1.05", 5, 5, 0.05, 0.05, 1.05),
    ("3v3, cv 3%/10%, ratio 1.0", 3, 3, 0.03, 0.10, 1.0),
    ("10v4, cv 20%/5%, ratio 0.9", 10, 4, 0.20, 0.05, 0.9),
)


def welch_min_df_coverage(
    rng: np.random.Generator,
    na: int,
    nb: int,
    sa: float,
    sb: float,
    ratio: float,
    reps: int,
) -> float:
    a = math.log(100) + sa * rng.standard_normal((reps, na))
    b = math.log(100 * ratio) + sb * rng.standard_normal((reps, nb))
    se = np.sqrt(a.var(axis=1, ddof=1) / na + b.var(axis=1, ddof=1) / nb)
    half = stats.t.ppf(0.975, min(na, nb) - 1) * se
    diff = b.mean(axis=1) - a.mean(axis=1)
    truth = math.log(ratio)
    return float(np.mean((diff - half <= truth) & (truth <= diff + half)))


def run_gate_reachable(n: int, q: float) -> bool:
    """Whether n runs of n can show a share q: 0.025^(1/n) >= q."""
    return bool(0.025 ** (1 / n) >= q)


def runs_to_show(q: float) -> int:
    """The fewest runs that can show a share q at all."""
    return next(n for n in range(1, 10_000) if run_gate_reachable(n, q))


def run_gate_cell(n: int, q: float) -> dict[str, Any]:
    """The run-level gate at the supremum of a false claim (p = q).

    A cell where n runs cannot show q is reported as one the gate cannot
    pass, with the fewest runs that could: its pass rate of 0 says nothing
    about the gate.
    """
    row: dict[str, Any] = {"runs": n, "p_run": q, "q": q}
    row["min_runs_to_pass"] = runs_to_show(q)
    row["can_pass"] = run_gate_reachable(n, q)
    row["false_pass"] = run_gate_false_pass(n, q, q) if row["can_pass"] else None
    return row


def run_gate_false_pass(n: int, p_run: float, q: float) -> float:
    """Exact: P(the Clopper-Pearson lower bound of k/n >= q) when k ~ Bin(n, p).

    It rises with p, so its supremum over the false claims (p < q) is its
    value at p = q.
    """
    passing = [
        k
        for k in range(n + 1)
        if (0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)) >= q
    ]
    return float(sum(stats.binom.pmf(k, n, p_run) for k in passing))


def clustered_run_gate(
    rng: np.random.Generator, n: int, reps: int, requests: int = 1000
) -> dict[str, float]:
    """Correlated failures within a run: the run is the unit that matters.

    Each run serves every request (attainment 1.0) with probability 0.7, and
    otherwise misses 2% of them (0.98). The claim under test is "a run meets
    0.99 with probability at least 0.8", which is false (0.7), and which 30
    or 60 runs can show (``run_gate_reachable``). The run-level gate should
    rarely pass; pooling requests as if they were independent sees a 99.4%
    pooled attainment and passes far more often.
    """
    good = rng.random((reps, n)) < 0.7
    k = good.sum(axis=1)
    lower_runs = np.where(
        k == 0, 0.0, stats.beta.ppf(0.025, np.maximum(k, 1), n - k + 1)
    )
    met = np.where(good, requests, round(0.98 * requests)).sum(axis=1)
    total = n * requests
    lower_pooled = stats.beta.ppf(0.025, met, total - met + 1)
    return {
        "runs": float(np.mean(lower_runs >= 0.8)),
        "pooled_requests": float(np.mean(lower_pooled >= 0.99)),
    }


def skew_screen_rate(rng: np.random.Generator, n: int, reps: int) -> float:
    from stormlog.infer.comparison_stats import skew_limit

    g1 = np.abs(stats.skew(rng.standard_normal((reps, n)), axis=1, bias=False))
    limit = skew_limit(n)
    return float(np.mean(g1 > limit)) if limit is not None else math.nan


# ------------------------------------------------------- module agreement


def check_against_module(rng: np.random.Generator, samples: int = 200) -> int:
    """Run sample replications through compare_values; return disagreements.

    Three configurations: paired latency at +budget, paired goodput at
    -budget (higher is better), and independent latency runs (min-df Welch).
    """
    disagreements = 0
    for n in (3, 6, 8):
        for _ in range(samples):
            disagreements += _agrees_paired(rng, n, higher=False)
            disagreements += _agrees_paired(rng, n, higher=True)
            disagreements += _agrees_independent(rng, n)
    return disagreements


def _module_passes(
    base: np.ndarray, cand: np.ndarray, *, higher: bool, paired: bool
) -> bool:
    from stormlog.infer.comparison_stats import GateRule, compare_values

    labels = [f"b{i}" for i in range(len(base))]
    result = compare_values(
        "x",
        list(base),
        list(cand),
        direction="higher_is_better" if higher else "lower_is_better",
        scale="log_ratio",
        unit="relative",
        blocks=(labels, labels) if paired else None,
        gate=GateRule("non-inferiority", BUDGET, "relative"),
    )
    assert result.gate is not None
    return result.gate.status == "pass"


def _agrees_paired(rng: np.random.Generator, n: int, *, higher: bool) -> int:
    shift = math.log1p(-BUDGET) if higher else math.log1p(BUDGET)
    base = 5.0 + SIGMA_RUN * rng.standard_normal(n)
    cand = 5.0 + shift + SIGMA_RUN * rng.standard_normal(n)
    d = (cand - base)[None, :]
    if higher:
        expected = bool(_higher_passes(d)[0])
    else:
        expected = bool(non_inferiority_passes(d, BUDGET)[0])
    found = _module_passes(np.exp(base), np.exp(cand), higher=higher, paired=True)
    return int(found != expected)


def _higher_passes(d: np.ndarray) -> np.ndarray:
    passes = np.expm1(paired_lower(d)) >= -BUDGET
    if d.shape[1] > MIN_GATE_PAIRS:
        stable = np.ones(len(d), dtype=bool)
        for left_out in range(d.shape[1]):
            kept = np.delete(d, left_out, axis=1)
            stable &= (np.expm1(paired_lower(kept)) >= -BUDGET) == passes
        passes &= stable
    return passes


def _agrees_independent(rng: np.random.Generator, n: int) -> int:
    a = 5.0 + SIGMA_RUN * rng.standard_normal(n)
    b = 5.0 + math.log1p(BUDGET) + SIGMA_RUN * rng.standard_normal(n)
    expected = bool(_independent_passes(a[None, :], b[None, :])[0])
    found = _module_passes(np.exp(a), np.exp(b), higher=False, paired=False)
    return int(found != expected)


# ----------------------------------------------------------------- report


# --------------------------------------------------------------- fractions

FRACTION_BUDGET = 0.01


def failure_counts(
    rng: np.random.Generator, shape: tuple[int, int], m: int, p: float, rho: float
) -> np.ndarray:
    """Failures per run: binomial, or beta-binomial with correlation rho."""
    if p == 0:
        return np.zeros(shape, dtype=int)
    if rho == 0:
        return rng.binomial(m, p, shape)
    a, b = p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho
    return rng.binomial(m, rng.beta(a, b, shape))


def _floored_upper(xb: np.ndarray, xc: np.ndarray, m: int) -> np.ndarray:
    """The upper bound of the paired t on fractions, se floored at the
    pooled binomial se with p = (x + 1) / (N + 2) per arm."""
    n = xb.shape[1]
    d = (xc - xb) / m
    total = n * m
    shares = [(x.sum(axis=1) + 1) / (total + 2) for x in (xb, xc)]
    floor = np.sqrt(sum(p * (1 - p) / total for p in shares))
    se = np.maximum(d.std(axis=1, ddof=1) / math.sqrt(n), floor)
    return np.asarray(d.mean(axis=1) + stats.t.ppf(0.975, n - 1) * se)


def _run_claim(xb: np.ndarray, xc: np.ndarray, m: int) -> np.ndarray:
    """k of n runs within the budget of their block; CP lower bound >= 0.5."""
    n = xb.shape[1]
    k = ((xc - xb) / m <= FRACTION_BUDGET).sum(axis=1)
    lower = np.where(k == 0, 0.0, stats.beta.ppf(0.025, np.maximum(k, 1), n - k + 1))
    return np.asarray(lower >= 0.5)


def fraction_cell(
    rng: np.random.Generator,
    n: int,
    m: int,
    rho: float,
    reps: int,
    baseline: float = 0.0,
) -> dict[str, Any]:
    """At a true difference equal to the budget: how often each method
    passes, and the true share of runs within the budget."""
    xb = failure_counts(rng, (reps, n), m, baseline, rho)
    xc = failure_counts(rng, (reps, n), m, baseline + FRACTION_BUDGET, rho)
    total = n * m
    upper_c = stats.beta.ppf(0.975, xc.sum(axis=1) + 1, total - xc.sum(axis=1))
    lower_b = np.where(
        xb.sum(axis=1) == 0,
        0.0,
        stats.beta.ppf(
            0.025, np.maximum(xb.sum(axis=1), 1), total - xb.sum(axis=1) + 1
        ),
    )
    return {
        "blocks": n,
        "requests_per_run": m,
        "rho": rho,
        "baseline": baseline,
        "floored_t_false_safe": float(
            np.mean(_floored_upper(xb, xc, m) <= FRACTION_BUDGET)
        ),
        "pooled_requests_false_safe": float(
            np.mean(upper_c - lower_b <= FRACTION_BUDGET)
        ),
        "run_claim_passes": float(np.mean(_run_claim(xb, xc, m))),
        "runs_within_budget": float(np.mean((xc - xb) / m <= FRACTION_BUDGET)),
    }


FRACTION_INDEPENDENT = ((6, 300, 0.0), (8, 300, 0.0), (10, 300, 0.0), (8, 1000, 0.0))
FRACTION_CLUSTERED = ((8, 300), (8, 1000), (10, 300))


def check_fractions_against_module(rng: np.random.Generator, samples: int) -> int:
    """The module's run-level gate and floored interval against the rules here."""
    from stormlog.infer.comparison_stats import GateRule, compare_values

    disagreements = 0
    for n in (5, 6, 8, 10):
        for _ in range(samples):
            xb = failure_counts(rng, (1, n), 300, 0.002, 0.05)
            xc = failure_counts(rng, (1, n), 300, 0.008, 0.05)
            blocks = [f"b{i}" for i in range(n)]
            result = compare_values(
                "failure_fraction",
                list(xb[0] / 300),
                list(xc[0] / 300),
                direction="lower_is_better",
                scale="difference",
                unit="fraction",
                blocks=(blocks, blocks),
                trials=([300] * n, [300] * n),
                gate=GateRule("non-inferiority", FRACTION_BUDGET, "fraction"),
            )
            expected = (
                "not_evaluable"
                if 0.025 < 0.5**n
                else ("pass" if _run_claim(xb, xc, 300)[0] else "fail")
            )
            assert result.gate is not None and result.worst is not None
            upper = float(_floored_upper(xb, xc, 300)[0])
            disagreements += int(
                result.gate.status != expected
                or not math.isclose(result.worst.upper or 0.0, upper, rel_tol=1e-9)
            )
    return disagreements


def run(reps: int, seed: int = SEED, module_samples: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"seed": seed, "reps": reps}
    results["paired_false_safe"] = [
        {
            "noise": name,
            "sigma_interaction": sigma,
            "blocks": n,
            "false_safe": paired_false_safe(rng, NOISES[name], sigma, n, reps),
            "supported": not name.startswith("lognormal"),
        }
        for name in NOISES
        for sigma in (0.0, 0.05, 0.2)
        for n in (3, 6, 8, 10)
    ]
    results["higher_is_better_false_safe"] = [
        {"blocks": n, "false_safe": higher_is_better_false_safe(rng, n, reps)}
        for n in (3, 6, 10)
    ]
    results["independent_false_safe"] = [
        {"runs_per_arm": n, "false_safe": independent_false_safe(rng, n, reps)}
        for n in (3, 6)
    ]
    results["paired_power"] = [
        {"sigma_run": sigma, "blocks": n, **paired_power(rng, sigma, n, reps)}
        for sigma in (0.02, 0.05)
        for n in (3, 6, 8, 10)
    ]
    results["missing_outcomes_false_safe"] = [
        {
            "unknown": u,
            "blocks": n,
            "false_safe": missing_outcome_false_safe(rng, u, n, reps),
        }
        for u in (0.005, 0.02)
        for n in (3, 6, 8, 10)
    ]
    results["independent_coverage"] = [
        {
            "case": label,
            "coverage": welch_min_df_coverage(rng, na, nb, sa, sb, ratio, reps),
        }
        for label, na, nb, sa, sb, ratio in INDEPENDENT_CASES
    ]
    results["run_gate_false_pass"] = [
        run_gate_cell(n, q) for n in (6, 8, 10, 30) for q in (0.5, 0.6, 0.8)
    ]
    results["clustered_runs"] = {
        str(n): clustered_run_gate(rng, n, reps) for n in (30, 60)
    }
    results["skew_screen"] = [
        {"blocks": n, "rate": skew_screen_rate(rng, n, reps)}
        for n in (3, 6, 8, 10, 20, 30)
    ]
    results["module_disagreements"] = check_against_module(rng, module_samples)
    # Drawn after every other cell, so the cells above stay as published.
    results["fraction_independent"] = [
        fraction_cell(rng, n, m, rho, reps) for n, m, rho in FRACTION_INDEPENDENT
    ] + [fraction_cell(rng, 8, 300, 0.0, reps, baseline=0.005)]
    results["fraction_clustered_limit"] = [
        fraction_cell(rng, n, m, rho, reps)
        for rho in (0.02, 0.05)
        for n, m in FRACTION_CLUSTERED
    ]
    results["fraction_module_disagreements"] = check_fractions_against_module(
        rng, module_samples // 4
    )
    results["verdict"] = verdict(results)
    return results


def verdict(results: dict[str, Any]) -> dict[str, bool]:
    return {
        "paired_false_safe": all(
            row["false_safe"] <= CRITERION_FALSE_SAFE
            for row in results["paired_false_safe"]
            if row["supported"]
        ),
        "missing_outcomes": all(
            row["false_safe"] <= CRITERION_FALSE_SAFE
            for row in results["missing_outcomes_false_safe"]
        ),
        "higher_is_better_and_independent": all(
            row["false_safe"] <= CRITERION_FALSE_SAFE
            for key in ("higher_is_better_false_safe", "independent_false_safe")
            for row in results[key]
        ),
        "independent_coverage": all(
            row["coverage"] >= CRITERION_COVERAGE
            for row in results["independent_coverage"]
        ),
        "run_gate": all(
            row["false_pass"] <= CRITERION_FALSE_SAFE
            for row in results["run_gate_false_pass"]
            if row["can_pass"]
        )
        and all(
            v["runs"] <= CRITERION_FALSE_SAFE
            for v in results["clustered_runs"].values()
        ),
        "skew_screen": all(
            abs(row["rate"] - 0.05) <= 0.005 for row in results["skew_screen"]
        ),
        "module_agrees": results["module_disagreements"] == 0
        and results["fraction_module_disagreements"] == 0,
        "fraction_floor_independent": all(
            row["floored_t_false_safe"] <= CRITERION_FALSE_SAFE
            for row in results["fraction_independent"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = run(args.reps)
    text = json.dumps(results, indent=1)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
