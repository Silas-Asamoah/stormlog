"""The comparison's method validation: how often each gate rule is wrong.

Each rule of ``stormlog.infer.comparison_stats`` is simulated where it has
to hold, with a fixed seed and 20,000 replications per cell (the Monte
Carlo standard error of a 2.5% rate is about 0.11 points):

1. **Paired t, non-inferiority, at a true effect equal to the budget.** The
   share of gates that pass is the false-safe rate; it must not exceed
   0.0283 (0.025 plus three standard errors). Normal and t3 noise, block
   spread 0, 0.05 and 0.2, at 3, 6, 8 and 10 blocks. Leave-one-out is
   applied as the module applies it.
2. **The same with missing outcomes**, 0.5% and 2% of requests unknown, an
   attainment gate judged on the worst case.
3. **Min-df Welch on logs, independent runs:** coverage must be at least
   0.947 in an adversarial 20-against-3 case and four more.
4. **The run-level attainment gate:** with the true share of runs that meet
   the target below q, it must pass at most 0.0283 of the time, whatever
   the correlation within a run.
5. **The skew screen:** under normal noise it must flag 0.05 +- 0.005.

Two scenarios are outside what the rules claim, and are measured as limits:
paired t under strongly skewed noise (standardized lognormal, sigma 0.8),
and pooled requests (``--attainment-model bernoulli``) under correlation
within a run.

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
    rng: np.random.Generator, noise: Noise, sigma_block: float, n: int, reps: int
) -> float:
    """At a true +budget latency change, how often non-inferiority passes.

    Skewed noise is put in the block log ratios themselves, with the same
    spread as the other cells: two equally skewed arms would cancel.
    """
    block = sigma_block * rng.standard_normal((reps, n))
    if noise in (skewed, skewed_left):
        base = 5.0 + block
        spread = SIGMA_RUN * math.sqrt(2) * noise(rng, (reps, n))
        cand = 5.0 + block + math.log1p(BUDGET) + spread
    else:
        base = 5.0 + block + SIGMA_RUN * noise(rng, (reps, n))
        cand = 5.0 + block + math.log1p(BUDGET) + SIGMA_RUN * noise(rng, (reps, n))
    return float(non_inferiority_passes(cand - base, BUDGET).mean())


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


def run_gate_false_pass(n: int, p_run: float, q: float) -> float:
    """Exact: P(the Clopper-Pearson lower bound of k/n >= q) when k ~ Bin(n, p)."""
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

    Each run serves every request (attainment 1.0) with probability 0.9, and
    otherwise misses 5% of them (0.95). The claim under test is "a run meets
    0.99 with probability at least 0.95", which is false (0.9). The run-level
    gate should rarely pass; pooling requests as if they were independent
    sees a 99.5% pooled attainment and passes far more often.
    """
    good = rng.random((reps, n)) < 0.9
    k = good.sum(axis=1)
    lower_runs = np.where(
        k == 0, 0.0, stats.beta.ppf(0.025, np.maximum(k, 1), n - k + 1)
    )
    met = np.where(good, requests, round(0.95 * requests)).sum(axis=1)
    total = n * requests
    lower_pooled = stats.beta.ppf(0.025, met, total - met + 1)
    return {
        "runs": float(np.mean(lower_runs >= 0.95)),
        "pooled_requests": float(np.mean(lower_pooled >= 0.99)),
    }


def skew_screen_rate(rng: np.random.Generator, n: int, reps: int) -> float:
    from stormlog.infer.comparison_stats import skew_limit

    g1 = np.abs(stats.skew(rng.standard_normal((reps, n)), axis=1, bias=False))
    limit = skew_limit(n)
    return float(np.mean(g1 > limit)) if limit is not None else math.nan


# ------------------------------------------------------- module agreement


def check_against_module(rng: np.random.Generator, samples: int = 200) -> int:
    """Run sample replications through compare_values; return disagreements."""
    from stormlog.infer.comparison_stats import GateRule, compare_values

    disagreements = 0
    for n in (3, 6, 8):
        blocks = [f"b{i}" for i in range(n)]
        for _ in range(samples):
            base = np.exp(5.0 + SIGMA_RUN * rng.standard_normal(n))
            cand = np.exp(5.0 + math.log1p(BUDGET) + SIGMA_RUN * rng.standard_normal(n))
            expected = bool(
                non_inferiority_passes((np.log(cand) - np.log(base))[None, :], BUDGET)[
                    0
                ]
            )
            result = compare_values(
                "x",
                list(base),
                list(cand),
                direction="lower_is_better",
                scale="log_ratio",
                unit="relative",
                blocks=(blocks, blocks),
                gate=GateRule("non-inferiority", BUDGET, "relative"),
            )
            assert result.gate is not None
            disagreements += (result.gate.status == "pass") != expected
    return disagreements


# ----------------------------------------------------------------- report


def run(reps: int, seed: int = SEED, module_samples: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"seed": seed, "reps": reps}
    results["paired_false_safe"] = [
        {
            "noise": name,
            "sigma_block": sigma,
            "blocks": n,
            "false_safe": paired_false_safe(rng, NOISES[name], sigma, n, reps),
            "supported": not name.startswith("lognormal"),
        }
        for name in NOISES
        for sigma in (0.0, 0.05, 0.2)
        for n in (3, 6, 8, 10)
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
        {"runs": n, "p_run": p, "q": q, "false_pass": run_gate_false_pass(n, p, q)}
        for n in (6, 8, 10)
        for p, q in ((0.49, 0.5), (0.89, 0.9))
    ]
    results["clustered_runs"] = {
        str(n): clustered_run_gate(rng, n, reps) for n in (6, 10, 30)
    }
    results["skew_screen"] = [
        {"blocks": n, "rate": skew_screen_rate(rng, n, reps)}
        for n in (3, 6, 8, 10, 20, 30)
    ]
    results["module_disagreements"] = check_against_module(rng, module_samples)
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
        "independent_coverage": all(
            row["coverage"] >= CRITERION_COVERAGE
            for row in results["independent_coverage"]
        ),
        "run_gate": all(
            row["false_pass"] <= CRITERION_FALSE_SAFE
            for row in results["run_gate_false_pass"]
        )
        and all(
            v["runs"] <= CRITERION_FALSE_SAFE
            for v in results["clustered_runs"].values()
        ),
        "skew_screen": all(
            abs(row["rate"] - 0.05) <= 0.005 for row in results["skew_screen"]
        ),
        "module_agrees": results["module_disagreements"] == 0,
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
