[← Back to docs](index.md)

# Comparing inference runs

Whether a change made inference faster, slower or no different is a
question about repeated runs: one run of each arm says little, because two
runs of one server already differ. Stormlog compares a **baseline** arm with
a **candidate** arm, run by run, and says which method it used and how far
the answer can be trusted.

The run is the unit of replication. A request inside a run is not an
independent sample of the server, so no interval here treats requests as
independent unless it says so.

## Command line

```bash
stormlog infer compare \
  --baseline runs/base-*.jsonl --candidate runs/cand-*.jsonl \
  --gate 'client.e2e.p95=non-inferiority:0.05' \
  --gate 'goodput_rps=non-inferiority:0.05' \
  --gate 'attainment=non-inferiority:0.01' \
  --report comparison.json
```

| Option | Meaning |
| --- | --- |
| `--baseline`, `--candidate` | Each arm's artifacts |
| `--case ID` | Only this case; repeatable |
| `--slo KEY:MS`, `--slo-file FILE` | Judge both arms by this policy instead of each run's own |
| `--mode config\|overhead\|incremental` | What may differ between the arms (see Modes) |
| `--design auto\|paired_blocks\|independent` | `auto` pairs runs by block when every run is labelled |
| `--allow FIELD` | A field (`engine.max_num_seqs`) or `vllm_config` JSON pointer (`/scheduler_config`) that may differ |
| `--added-observers NAME,...` | The observers an `incremental` candidate adds |
| `--gate METRIC=RULE:BUDGET` | Gate a metric, or every metric a pattern names (`client.*.p99`). A latency, goodput or throughput budget is relative (0.05 is 5%); an attainment or failure-fraction budget is a fraction (0.01 is one point) |
| `--fallback METRIC=BUDGET:UNIT` | A pre-registered budget on the difference, for when a zero leaves the log ratio undefined |
| `--min-complete-blocks N` | Every gate needs at least N complete pairs |
| `--min-attainment X`, `--min-run-pass Q` | At least a share Q (0.5) of candidate runs reach attainment X: a claim about runs. `--attainment-model bernoulli` pools requests instead, labelled model-based |
| `--family all_budgets\|any_regression` | `any_regression` adjusts the regression tests with Holm's method; its gates use `significant` |
| `--on-incomplete exclude\|fail` | A run set aside by a protocol failure is listed (`exclude`), or fails its contrasts (`fail`) |
| `--allow-not-evaluable` | Exit 0 although a gate could not be evaluated: for exploration, and recorded |
| `--evidence-floor F` | The `evidence_coverage` SLO metrics need in every run (1.0) |
| `--format txt\|json`, `--report FILE` | Text, or the report envelope on stdout; `--report` writes the envelope too |

Metrics, for every case: `goodput_rps`, `attainment` (bounds when outcomes
are unknown), `throughput_rps`, `output_tps`, `failure_fraction`, and p50,
p90, p95 and p99 of each latency metric the runs have (`client.ttft`,
`client.e2e`, `client.tpot`, `client.e2e_from_intended`, and `server.ttft`
and `server.e2e` with spans). Latency quantiles are the failure-penalized
estimand: a quantile that falls among failed requests (`penalized`) or rests
on too few requests (`insufficient_tail_samples`) cannot be gated.

### Modes

- `config`: only allowed fields may differ; observers must match.
- `overhead`: the cost of observing. The baseline runs no observers; the
  candidate's must be active and healthy.
- `incremental`: the cost of added observers. Shared observers are active
  and healthy in both arms; the added ones are declared and healthy in the
  candidate.

### Exit codes and the report

| Code | When |
| --- | --- |
| 0 | No gate, or every gate passed |
| 4 | A gate failed, or one could not be evaluated without `--allow-not-evaluable` |
| 5 | The runs could not be compared: an artifact missing, a block with two runs of one arm, incompatible arms, an arm with no usable run, an observer contract broken |
| 2 | Flags it cannot read |

`--format json` and `--report` give a `stormlog.report` v1 envelope with
`report_kind: inference_comparison` and the `stormlog.infer.comparison` v1
payload. A failed gate is a `regression` finding, a gate that cannot be
evaluated a `not_evaluable` one, and a run set aside an `excluded_run` one.
Each points at its metric with a JSON pointer into the payload
(`/payload/cases/<case>/metrics/<metric>`): on stdout with no `path`, so it
resolves inside the report itself; in a written report, with `path` set to
the report's file name. A comparison that exits 5 still writes its report,
with one `invalid_input` finding, so a CI job keeps the reason.

## Runs

Each run is summarized again from its artifact's raw records
(`stormlog.infer.run_summary.summarize_run`), never from the summary the
run wrote about itself. A summary carries the run's report, its experiment
labels, its comparable fields (see
[Inference server descriptions](inference_server.md)) and its **protocol
failures**: faults of the measurement that exclude it, with a reason.

| Protocol failure | When |
| --- | --- |
| `session_<status>` | The run did not finish: `interrupted`, `incomplete`, or no terminal record |
| `identity_changed` | The server's identity changed between its before and after descriptions |
| `description_mismatch` | The before description disagrees with the server the probe reached: its model, vLLM version or driver |
| `probe_incomplete` | The server probe's `/server_info` did not answer in time, or the server dropped it unanswered |
| `cohort_invalid` (a case) | The case's requests are not one whole cohort |
| `cache_reset_not_acknowledged` (a case) | A cold cache was asked for and no reset was acknowledged |

Failed requests, timeouts, or a candidate that served nothing are outcomes,
never protocol failures: they are compared, not excluded.

## Designs

When runs carry block labels, the design is **paired**: each block holds one
baseline run and one candidate run, run close together, so that drift
between blocks cancels. Only complete pairs count; a block missing an arm's
run is listed with `block_incomplete:<arm>`, and the design stays paired. A
block with two runs of one arm is refused.

Without block labels, the arms are **independent** samples of runs.

## Methods

One method gives the interval every verdict and gate uses:

| Design | `log_ratio` | `difference` |
| --- | --- | --- |
| paired | t on the block log ratios, df = n − 1 | t on the block differences, df = n − 1 |
| independent | Welch t on log run values, df = min(nA, nB) − 1 | Welch t, df = min(nA, nB) − 1 |

`absolute` is a one-sample t over the candidate's runs, for a metric the
baseline cannot have, such as the bytes a profiler writes.

The independent design's df is the smaller arm's size minus one, not
Welch–Satterthwaite's: with 20 runs against 3, Welch–Satterthwaite's
interval covered 92.9% of the time where it should cover 95%, and the
smaller df covered 96.9%. It is conservative by construction.

Also reported, never gated:
- **Fieller's interval** for the ratio of arithmetic means, a cross-check of
  the log-ratio interval. It is unbounded when the baseline's mean is not
  clearly away from zero.
- **A bootstrap**, from 10 pairs (or runs per arm): blocks resampled in
  pairs, or runs within each arm, 10,000 times with a recorded seed. Below 10
  it under-covers badly (75–82% at 3–5 blocks), so it is not computed.
- **For a `log_ratio` metric, its difference** in the values' own unit.

## Scales and units

Every result carries `scale`, `unit`, `effect` and `interval` on one scale:

| Scale | `effect` and interval bounds | `unit` | A budget `b` |
| --- | --- | --- | --- |
| `log_ratio` | The relative change `exp(mean log ratio) − 1`: 0.05 is +5%. The bounds are `exp(d̄ ± t·se) − 1`, so they are not symmetric | `relative` | Latency (lower is better) passes non-inferiority iff `upper ≤ b`; throughput and goodput (higher is better) iff `lower ≥ −b`. Never compared with `1 ± b` |
| `difference` | `mean(candidate − baseline)` | The metric's own: `fraction` for attainment and failure rate (0.01 is one percentage point), `requests_per_second`, `cores`, `bytes` | In the same unit |
| `absolute` | The candidate's mean | The metric's own | In the same unit |

A gate's unit must equal the metric's, or the gate is refused.

## Zeros

A log ratio needs positive values:
- a candidate value of 0 on a higher-is-better metric (a candidate that
  served nothing) fails its gate: `candidate_zero`, the worst regression;
- a baseline value of 0, both zero, or a candidate value of 0 on a
  lower-is-better metric leaves the log ratio undefined: the gate is
  `not_evaluable: undefined_in_arm`, unless a fallback budget was declared
  beforehand, in the values' own unit, which then gates the difference.

The difference is always reported, in its own unit, and never gated with the
relative budget. A fallback in requests per minute gives the same decision
as the same fallback in requests per second.

When every run of both arms has the same value (a failure rate of 0, an
attainment of 1), a t interval would be a falsely certain [0, 0]. The result
is `degenerate_zero` (or `degenerate_constant`) with no interval, and the
bound that the runs do support: a run departs from that value with
probability at most `1 − (α/2)^(1/n)`, per arm. Such a metric's gate passes,
with that reason.

## Missing evidence

A run's value can be an interval `(lower, upper)`, when some of its evidence
is missing: SLO attainment and goodput with unknown outcomes. Each pair then
has a **worst case**, the candidate's bad end against the baseline's good
end, and a **best case**, the other way round. Both are reported.

- **Non-inferiority**, a claim that the candidate is safe, is judged on the
  worst case.
- **Significant** and **demonstrated**, claims that the candidate regressed,
  are judged on the best case.

Missing evidence can therefore neither hide a regression nor invent one.

## Verdicts and gates

Each metric gets two descriptive verdicts:
- `direction`: `worse` when the best case's interval excludes 0 on the bad
  side, `better` when the worst case's excludes it on the good side, else
  `no_detectable_change`;
- `tolerance`: `within` when the worst case's interval lies inside ±τ,
  `beyond` when one lies entirely outside it, else `undetermined`. τ is the
  gate's budget unless given.

A gate (`GateRule`) has a rule, a budget and a unit:

| Rule | Fails when |
| --- | --- |
| `non-inferiority` | The upper bound of the change in the bad direction exceeds the budget |
| `significant` | The interval excludes 0 on the bad side, and the estimate exceeds the budget |
| `demonstrated` | The interval lies entirely beyond the budget |

A gate is `not_evaluable`, never passed, when:
- fewer than 3 complete pairs (or runs per arm) remain: at 2, df = 1 and the
  interval is always undetermined;
- fewer remain than the gate's pre-registered `min_complete_blocks`, so an
  excluded run cannot quietly turn six blocks into two;
- the log ratio is undefined (see Zeros);
- removing any single block (or run) changes the gate's decision
  (`decision_unstable`); only samples that could still be gated count, so
  this is checked from 4 pairs;
- the caller already knows why, such as unverified comparability or a
  censored quantile.

## Guards

Reported for every metric, never certifying anything:
- **The skew screen:** the adjusted skewness G1 of the block log ratios (or
  of the candidate's values), against the 95% quantile of |G1| under normal
  noise at the same n. The table (1.699, 1.509 and 1.375 at 6, 8 and 10)
  comes from `examples/analysis/skew_quantiles.py`, with 1.96 times G1's
  standard error beyond 30. It flags; it never blocks: a fixed limit of 1
  flagged 23.5% of 6-block comparisons of normal noise.
- **Leave one out:** the range of the interval's bounds without each pair,
  and the number of decision flips, the one guard that blocks a gate.
- **Zero variance:** every pair moved by exactly the same amount.

## Effect sizes

Descriptive, with their counts: for paired runs, the share of blocks in
which the candidate is higher (ties count ½); for independent runs, Vargha
and Delaney's A12 over every pair of runs.

## The contract

`tests/fixtures/infer/comparison_contract_v1.json` is the binding contract
for callers that gate on these results, such as release qualification. Each
case gives the inputs and the effect, interval and gate outcome they must
give. The cases cover:
- effect units and signs in both directions;
- a 40% latency regression against a 5% non-inferiority budget;
- the gate at its budget boundary;
- attainment budgets in fraction units;
- the zero rules and their fallback in two units;
- missing-outcome bounds;
- pre-registered block counts;
- degenerate metrics;
- the independent design's df.

The expected numbers come from the formulas themselves, in
`examples/analysis/comparison_contract.py`, not from this module, and a
test checks the fixture is what that script writes.

No float in the fixture is exact: numpy and scipy differ in a float's last
bits between platforms. The fixture states its tolerance (`tolerance`,
relative 1e-9 and absolute 1e-12) and the rule that applies it
(`tolerance_rule`): a float, given or expected, matches when
|actual − expected| ≤ max(relative × |expected|, absolute), and anything
else, such as a gate's status or a missing value, matches exactly. Its own
tests, and the check that it is what the script writes, apply that rule as
written; an implementation checked against it should too.

## Other tools

| Function | Gives |
| --- | --- |
| `clopper_pearson(k, n, model=...)` | The exact binomial interval, labelled `independent_runs` (exact when runs are independent) or `independent_requests` (model-based: requests within a run are correlated) |
| `run_pass_gate(k, n, q)` | Whether at least a share q of runs meets a target: k of n runs did, and the Clopper–Pearson lower bound of k/n is at least q. With 6 runs and q = 0.5 it takes 6 of 6 |
| `blocks_for_precision(sd_log_ratio, h)` | The fewest blocks, from 6 to 10, whose log-scale half-width `t·sd/√n` is at most `log1p(h)`; 10 with `meets_target: false` when none is |
| `holm(p_values)` | Which of a family's one-sided regression tests Holm's step-down rejects, for a claim that any regression is detected |

## Limits

- Paired t is nominal for roughly symmetric block log ratios. Under strongly
  skewed noise it covers 86–88% instead of 95%, with one-sided false safety
  of 12–13%, and 6–7% even when the guards pass.
- Guards flag; they do not certify coverage.
- The independent design's intervals are conservative by construction.
- The order-statistic screens on latency quantiles assume independent
  requests, which queueing violates.

## Python API

```python
from stormlog.infer.comparison_stats import GateRule, compare_values

result = compare_values(
    "client.e2e.p95",
    baseline_values,
    candidate_values,
    direction="lower_is_better",
    scale="log_ratio",
    unit="relative",
    blocks=(baseline_blocks, candidate_blocks),
    gate=GateRule("non-inferiority", 0.05, "relative"),
)
result.to_record()
```

## Related pages

- [Inference Profiling](inference.md)
- [Inference SLOs and goodput](inference_slo.md)
- [Inference server descriptions](inference_server.md)
