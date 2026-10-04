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
| `--slo KEY:MS`, `--slo-file FILE` | Judge both arms by this policy instead of each run's own. Without it, a goodput or attainment gate, or `--min-attainment`, over runs that recorded different policies (`slo_digest`) is invalid input (exit 5), since a candidate judged by a looser policy would meet it however slow it was; ungated, those metrics read `slo_policy_differs` |
| `--mode config\|overhead\|incremental` | What may differ between the arms (see Modes) |
| `--design auto\|paired_blocks\|independent` | `auto` pairs runs by block when every run is labelled |
| `--allow FIELD` | A field (`engine.max_num_seqs`) or `vllm_config` JSON pointer (`/scheduler_config`) that may differ |
| `--added-observers NAME,...` | The observers an `incremental` candidate adds |
| `--gate METRIC=RULE:BUDGET` | Gate a metric, or every metric a pattern names (`client.*.p99`). A latency, goodput or throughput budget is relative (0.05 is 5%); an attainment or failure-fraction budget is a fraction (0.01 is one point). A budget that can never fail is a usage error: a fraction above 1, or a fall of 100% or more in a rate |
| `--fallback METRIC=BUDGET:UNIT` | A pre-registered budget on the difference, for when a zero leaves the log ratio undefined. `METRIC` is a name or pattern, and the budget applies to every gated metric it matches, whichever gate pattern named it; one that matches no gated metric is a usage error |
| `--min-complete-blocks N` | Every gate needs at least N complete pairs |
| `--min-attainment X`, `--min-run-pass Q` | At least a share Q (0.5) of candidate runs reach attainment X: a claim about runs. Both are in (0, 1]. A candidate run whose SLO could not be judged counts as not reaching X, and the gate obeys the same blockers as the metric gates (unverified comparability, observers, `--on-incomplete fail`); when it cannot be evaluated, it exits 4 like they do. `--attainment-model bernoulli` pools requests instead, labelled model-based |
| `--family all_budgets\|any_regression` | `any_regression` adjusts the regression tests with Holm's method; its gates use `significant` |
| `--on-incomplete exclude\|fail` | A block (or run) lost to a protocol failure, or given for one arm only, is listed (`exclude`), or fails its contrasts (`fail`) |
| `--allow-not-evaluable` | Exit 0 although a gate could not be evaluated: for exploration, and recorded. The summary then says how many could not be evaluated, never that every gate passed |
| `--evidence-floor F` | The `evidence_coverage` SLO metrics need in every run (1.0), in [0, 1] |
| `--segment NAME=START:END` | Also compare this slice of each case's measured phase, in seconds from its start; repeatable |
| `--segment-membership arrival\|overlap` | A segment's requests: those that arrived in it (the default), or that overlap it |
| `--format txt\|json`, `--report FILE` | Text, or the report envelope on stdout; `--report` writes the envelope too |

Metrics, for every case: `goodput_rps`, `attainment` (bounds when outcomes
are unknown), `throughput_rps`, `output_tps`, `failure_fraction`, and p50,
p90, p95 and p99 of each latency metric the runs have (`client.ttft`,
`client.e2e`, `client.tpot`, `client.e2e_from_intended`, and `server.ttft`
and `server.e2e` with spans). Latency quantiles are the failure-penalized
estimand: a quantile that falls among failed requests (`penalized`) or rests
on too few requests (`insufficient_tail_samples`) cannot be gated.

### Segments

A segment is a slice of each case's measured phase, such as the seconds
around a profiler's stop. With `--segment`, each segment is compared as a
case of its own, named `<case>/<segment>`, with the same metrics: its
requests (by arrival, or by overlap), and its rates per second of the
segment, clipped to the phase. A segment fails with its case.

Under `--segment-membership overlap`, a segment has no rates (goodput,
throughput, output tokens): its requests are those in flight during it,
and their count per second grows with their latency, so a slower candidate
would look faster. Those metrics read `overlapping_cohort`, and a rate gate
with overlap segments is a usage error; gate rates by arrival. Its shares
(attainment, failure fraction) and latency are compared as usual. Any rate
a run cannot give is read with its interval's reason (`rate_reason`), or
`rate_unavailable`.

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
| 5 | The runs could not be compared: an artifact missing, one artifact given twice in an arm, incompatible arms, an arm with no usable run, an observer contract broken |
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
[Inference server descriptions](inference_server.md)), its **protocol
failures**, faults of the measurement that set it aside with a reason, and
its **outcome failures**, which never do.

| Protocol failure | When |
| --- | --- |
| `external:<reason>` | An experiment runner recorded an external cause in an `infer.run_state` record with state `protocol_failure`: a preemption, an operator abort, a server that never became healthy, a failed prelude |
| `identity_changed` | The server's identity changed between its before and after descriptions |
| `description_mismatch` | The before description disagrees with the server the probe reached: its model, vLLM version or driver |
| `probe_incomplete` | The server probe's `/server_info` did not answer in time, or the server dropped it unanswered |
| `cohort_invalid` (a case) | The case's requests are not one whole cohort |
| `cache_reset_not_acknowledged` (a case) | A cold cache was asked for and no reset was acknowledged |

A run that did not finish (`session_<status>`: `interrupted`, `incomplete`,
or no terminal record) is an **outcome**, like failed requests, timeouts, a
candidate that served nothing, or an outcome a runner recorded
(`runner:<reason>`, from `infer.run_state` with state `outcome_failure`: a
server that exited, a step that failed or timed out, a treatment that
stopped early): the treatment may have caused it, so it is compared, not
set aside. Outcome beats protocol: unless an external cause is recorded,
such a run's run-level faults (`identity_changed`, `probe_incomplete`,
`description_mismatch`) are outcomes too, as is a cohort an unfinished run
cut short (`phase_window_missing`, `records_missing`). A case a
run lacks is an outcome as well. Where such a run has no value for a
metric, the outcome cannot be recovered: a candidate's gate on it fails
(`outcome_unrecoverable`), and a baseline's leaves it `not_evaluable`
(`baseline_outcome_unrecoverable`). Under `--min-attainment`, such a
candidate run counts as not reaching the target, and a `bernoulli` gate
fails. Otherwise a treatment that crashed runs could pass on the blocks
left.

A protocol failure sets aside the whole block, both arms' runs, for the
cases it touches; the partner is listed with `block_set_aside`. A block
given for one arm only also counts as lost. Each case lists what it lost
under `set_aside` (blocks, or runs in an independent design); more than one
leaves every gate of the case `not_evaluable` (`blocks_set_aside` or
`runs_set_aside`), so attrition cannot quietly shrink a contrast.

A block an arm ran more than once keeps its last attempt (by start time)
and lists the others with `superseded` and `attempt_kept`, so a block
retried after a protocol failure (`--attempt 2`) is compared once. A retry
never replaces an outcome failure: the first attempt that failed as an
outcome stands, and later ones are listed with `retry_of_outcome_failure`.
The same artifact given twice in an arm is invalid input.

The kept runs must have measured one server. Every run is checked against
its arm's first run, and every run against the other arm's first, because
comparability is not transitive once a value is unknown. A difference that
blocks is invalid input (exit 5). If any pair is `unverified`, the
comparison is, with every field that could not be shown equal, and
`diagnostics.unverified_pairs` names each pair; every gate is then
`not_evaluable: unverified`.

## Designs

When runs carry block labels, the design is **paired**: each block holds one
baseline run and one candidate run, run close together, so that drift
between blocks cancels. Only complete pairs count; a block missing an arm's
run is listed with `block_incomplete:<arm>`, counts as lost (see Runs), and
the design stays paired. A pair stands for one block's conditions, the same
workload realization (the block's seed) in both arms: a block whose runs'
`workload.realization_digest` differ is set aside, both runs listed with
`block_realization_differs`.

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
  served nothing) fails its gate: `candidate_zero`, the worst regression.
  It is read in the case the rule judges: the worst case for
  non-inferiority, the best case for `significant` and `demonstrated`. A
  zero only in the case the rule does not read (goodput known only as
  `(0, 10.4)`, from unknown outcomes) shows no regression, and the gate is
  `not_evaluable: undefined_in_arm`; a regression rule also waits for its
  blockers (unverified comparability, too few blocks) first;
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

A run with no value for a metric says why, and that reason makes the
metric's gate `not_evaluable`: a latency quantile with a successful
request that lacks it (`successful_values_missing`), a rate with no
interval (its `rate_reason`, or `rate_unavailable`), a population not
recorded. An infinite candidate value of a lower-is-better metric, such as
a latency that never finished, is the worst value there is: its gate fails
(`candidate_censored_worst`), read like a zero candidate on a
higher-is-better metric. Any other value that is not finite makes the gate
`not_evaluable: non_finite_value`; dropping it as missing would decide the
gate on the runs left.

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
- more than one block (or run) of the case was set aside or lost
  (`blocks_set_aside`, see Runs);
- the log ratio is undefined (see Zeros);
- removing any single block (or run) changes the gate's decision
  (`decision_unstable`); only samples that could still be gated count, so
  this is checked from 4 pairs;
- the caller already knows why, such as unverified comparability or a
  censored quantile;
- the case has no metric the gate's name or pattern matches, such as a
  server latency gate on runs without spans (`metric_absent`, listed under
  the case's `absent_gates`): a gate on something never measured gates
  nothing, and must not read as a pass.

A `--case` that no run has is a usage error (exit 2). A case no run offered
a request, such as a segment outside every run's phase, is invalid input
(exit 5). A metric that is not compared says why in its `reason`, gated or
not.

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
- the gate at its budget boundary, for latency, throughput and attainment;
- attainment budgets in fraction units;
- the zero rules and their fallback in two units;
- missing-outcome bounds;
- pre-registered block counts;
- degenerate metrics;
- the independent design's df.

The expected numbers come from the formulas themselves, in
`examples/analysis/comparison_contract.py`, not from this module, and a
test checks the fixture is what that script writes. So do the gate
outcomes: the script applies each rule to its own interval, after the
blockers (too few or fewer than pre-registered pairs) and the leave-one-out
screen, rather than assigning them by hand.

## Other tools

| Function | Gives |
| --- | --- |
| `clopper_pearson(k, n, model=...)` | The exact binomial interval, labelled `independent_runs` (exact when runs are independent) or `independent_requests` (model-based: requests within a run are correlated) |
| `run_pass_gate(k, n, q)` | Whether at least a share q of runs meets a target: k of n runs did, and the Clopper–Pearson lower bound of k/n is at least q. With 6 runs and q = 0.5 it takes 6 of 6 |
| `blocks_for_precision(sd_log_ratio, h)` | The fewest blocks, from 6 to 10, whose log-scale half-width `t·sd/√n` is at most `log1p(h)`; 10 with `meets_target: false` when none is |
| `holm(p_values)` | Which of a family's one-sided regression tests Holm's step-down rejects, for a claim that any regression is detected |

## Method validation

`examples/analysis/simulation_study.py` simulates each rule where it has to
hold, with seed 213 and 20,000 replications per cell; the Monte Carlo
standard error of a 2.5% rate is about 0.11 points. Its results are in
`examples/analysis/simulation_results.json`, and a sample of its
replications goes through `compare_values` too, which must make the same
decisions: paired latency, paired goodput (higher is better) and
independent runs (it does, in every one).

**Non-inferiority at a true change equal to the budget** (+5% latency, run
spread 0.05 on the log scale, paired blocks, leave-one-out applied as the
module applies it). Every pass here is a false "safe": the rule must stay at
or below 2.83% (2.5% plus three standard errors) wherever it is supported.
The treatment's effect varies by block around its mean of +5% (block-by-arm
interaction): a block effect both arms share cancels in the log ratio, so
it would test nothing.

| Noise in the block log ratios | Effect spread across blocks | 3 blocks | 6 | 8 | 10 |
| --- | --- | --- | --- | --- | --- |
| normal | 0.0 | 2.54% | 0.77% | 0.98% | 1.05% |
| normal | 0.05 | 2.40% | 0.91% | 0.92% | 1.09% |
| normal | 0.2 | 2.62% | 0.85% | 1.04% | 1.13% |
| t, 3 df | 0.0 | 1.90% | 0.60% | 0.77% | 0.94% |
| t, 3 df | 0.05 | 2.31% | 0.64% | 0.84% | 0.97% |
| t, 3 df | 0.2 | 2.51% | 0.84% | 1.23% | 1.06% |
| lognormal 0.8, right tail (unsupported) | 0.0 | 11.91% | 8.33% | 8.88% | 8.36% |
| lognormal 0.8, right tail (unsupported) | 0.05 | 4.00% | 1.90% | 2.54% | 2.97% |
| lognormal 0.8, right tail (unsupported) | 0.2 | 2.50% | 0.95% | 1.10% | 1.20% |
| lognormal 0.8, left tail (unsupported) | 0.0 | 0.46% | 0.05% | 0.03% | 0.04% |
| lognormal 0.8, left tail (unsupported) | 0.05 | 1.33% | 0.18% | 0.24% | 0.26% |
| lognormal 0.8, left tail (unsupported) | 0.2 | 2.31% | 0.74% | 0.84% | 1.11% |

Below 4 blocks no leave-one-out sample can be gated, so 3 blocks show the
interval alone. Strongly right-skewed block log ratios are outside what the
rule claims: there it passes 8–12% of the time instead of at most 2.5%.

**A higher-is-better metric, and the independent design**, at a true change
equal to the budget:

| Design and metric | Runs | False-safe |
| --- | --- | --- |
| paired, goodput at -5% (higher is better) | 3 blocks | 2.31% |
| paired, goodput at -5% (higher is better) | 6 blocks | 0.78% |
| paired, goodput at -5% (higher is better) | 10 blocks | 1.19% |
| independent, latency at +5% (min-df Welch) | 3 per arm | 0.67% |
| independent, latency at +5% (min-df Welch) | 6 per arm | 0.40% |

**The same for attainment with unknown outcomes**, a one-point budget judged
on the worst case: missing evidence only makes the gate more careful.

| Unknown outcomes per run | 3 blocks | 6 | 8 | 10 |
| --- | --- | --- | --- | --- |
| 0.50% | 0.04% | 0.00% | 0.00% | 0.00% |
| 2.00% | 0.00% | 0.00% | 0.00% | 0.00% |

**Independent runs**, coverage of the min-df Welch interval on logs; it must
be at least 0.947:

| Independent runs | Coverage |
| --- | --- |
| 20v3, cv 10%/20%, ratio 0.01 (adversarial) | 0.9686 |
| 20v3, cv 5%/15%, ratio 1.0 | 0.9603 |
| 5v5, cv 5%/5%, ratio 1.05 | 0.9747 |
| 3v3, cv 3%/10%, ratio 1.0 | 0.9740 |
| 10v4, cv 20%/5%, ratio 0.9 | 0.9914 |

**The run-level attainment gate**, at the supremum of a false claim: the
pass rate rises with the true share of runs that meet the target, so its
largest value below q is at q itself (exact binomial pass rates). n runs
show at most a share of `0.025^(1/n)`, so 6 runs need 6 of 6 at q = 0.5 and
8 runs 8 of 8 (7 of 8 gives 0.47). A cell where the gate cannot pass at all
measures nothing, so it is marked, with the fewest runs that could pass:

| Runs | q, and the true share of runs meeting the target | Pass rate |
| --- | --- | --- |
| 6 | 0.5 | 1.56% |
| 6 | 0.6 | cannot pass (needs 8 runs) |
| 6 | 0.8 | cannot pass (needs 17 runs) |
| 8 | 0.5 | 0.39% |
| 8 | 0.6 | 1.68% |
| 8 | 0.8 | cannot pass (needs 17 runs) |
| 10 | 0.5 | 1.07% |
| 10 | 0.6 | 0.60% |
| 10 | 0.8 | cannot pass (needs 17 runs) |
| 30 | 0.5 | 2.14% |
| 30 | 0.6 | 1.72% |
| 30 | 0.8 | 1.05% |

With failures correlated inside a run (each run serves everything with
probability 0.7 and otherwise misses 2% of requests; the claim that a run
meets 0.99 with probability 0.8 is false), the run-level gate almost never
passes; pooling requests as if they were independent passes most of the
time, which is why `--attainment-model bernoulli` is labelled model-based.
Fewer than 17 runs cannot show a share of 0.8 at all:

| Runs | Run-level gate | Pooled requests (`bernoulli`) |
| --- | --- | --- |
| 30 | 0.03% | 95.89% |
| 60 | 0.01% | 99.48% |

**The skew screen** must flag 5% ± 0.5 points of normal noise:

| Blocks | Skew screen rate under normal noise |
| --- | --- |
| 3 | 4.93% |
| 6 | 4.76% |
| 8 | 4.90% |
| 10 | 4.92% |
| 20 | 4.87% |
| 30 | 5.17% |

**What leave-one-out costs.** With no change at all, the share of
non-inferiority gates that pass, at a 5% budget:

| Run spread | Blocks | Passes | Passes without leave-one-out | Share of those passes it stops |
| --- | --- | --- | --- | --- |
| 0.02 | 3 | 38.68% | 38.68% | 0% |
| 0.02 | 6 | 72.95% | 91.47% | 20% |
| 0.02 | 8 | 93.84% | 98.56% | 5% |
| 0.02 | 10 | 99.09% | 99.83% | 1% |
| 0.05 | 3 | 11.24% | 11.24% | 0% |
| 0.05 | 6 | 12.76% | 28.09% | 55% |
| 0.05 | 8 | 24.20% | 39.72% | 39% |
| 0.05 | 10 | 33.84% | 49.45% | 32% |

The stability guard stops about a fifth of the passes a plain interval would
give at 6 blocks with a run spread of 0.02, and more than half at 0.05;
at 0.05 more blocks recover them only slowly (two fifths at 8 blocks, a
third at 10).

## Limits

- Paired t is nominal for roughly symmetric block log ratios. Under strongly
  right-skewed block log ratios a non-inferiority gate at a true change
  equal to its budget passes 8–12% of the time instead of at most 2.5% (see
  Method validation).
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
