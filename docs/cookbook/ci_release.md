[← Back to Production Cookbook](index.md)

# CI and Release Qualification

Use this page when the goal is not just to diagnose one run, but to qualify
Stormlog's operational behavior before release or as part of CI.

> **Source checkout only below.** The commands on this page use the repository
> `examples/` package and benchmark assets under `docs/benchmarks/`.

Audience: release owners, CI maintainers.
Difficulty: intermediate.

## Prerequisites

- install the checkout with the needed extras first:
  `pip install -e ".[dev,test,all]"`
- start from [Installation](../installation.md), [Examples Guide](../examples.md),
  and [Benchmark Harness](../benchmark_harness.md) if the workflow is unfamiliar
- a source checkout with the `examples/` package available
- benchmark asset files under `docs/benchmarks/`
- enough time and artifact space for the chosen harness profile

Success signal:

- the chosen smoke or benchmark command completes and writes its expected report artifact

## Choose the qualification path

| If the main goal is... | Start with... |
| --- | --- |
| fast CLI sanity signal | `examples.cli.quickstart` |
| broad smoke coverage across surfaces | `examples.cli.capability_matrix` |
| always-on operability qualification | `examples.cli.benchmark_harness --mode all` |
| enforce regression or budget gates | benchmark harness `--check` modes |

## Fast smoke validation

```bash
python -m examples.cli.quickstart
```

Use this when you want one fast signal that the installed CLI entrypoints still
behave as expected.

## Cross-surface smoke validation

```bash
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

Use this when you want one command that touches the major PyTorch and
TensorFlow validation paths.

## Always-on operability qualification

```bash
python -m examples.cli.benchmark_harness \
  --profile pr \
  --mode all \
  --output artifacts/benchmarks/latest_v0.4.json
```

This is the fastest source-checkout qualification path for always-on tracking,
bounded history, and retention behavior.

## Regression-gated benchmark run

```bash
python -m examples.cli.benchmark_harness \
  --check \
  --profile pr \
  --mode all \
  --gate-mode regression \
  --iterations 5000 \
  --baseline docs/benchmarks/v0.4_baseline.json \
  --tolerances docs/benchmarks/v0.4_tolerances.json \
  --output artifacts/benchmarks/latest_v0.4_regression.json
```

On Linux CI runners add `--overhead-scratch-root /dev/shm/stormlog-benchmark`
so the overhead trials run on RAM-backed storage; otherwise the runner's disk
latency enters `runtime_overhead_pct`.

## Budget-gated benchmark run

```bash
python -m examples.cli.benchmark_harness \
  --check \
  --profile pr \
  --mode all \
  --gate-mode budget \
  --iterations 5000 \
  --budgets docs/benchmarks/v0.4_operating_budget.json \
  --output artifacts/benchmarks/latest_v0.4_budget.json
```

## What to archive from CI

- benchmark harness JSON output
- sink directories or diagnose bundles for failed runs
- any saved analysis reports used during triage

## What to do next

- If the harness fails on collector health or retention metrics, move to
  [Always-on Tracking](always_on.md).
- If the failure centers on a specific runtime incident, move to
  [PyTorch Production Recipes](pytorch.md) or
  [TensorFlow Production Recipes](tensorflow.md).
- If the failure is distributed and rank-specific, move to
  [Distributed Diagnostics Recipes](distributed.md).

## Troubleshooting

### Symptom: a benchmark command is too heavy for local iteration

Likely cause: the current profile is intended for PR or nightly gating.
Fix: start with smoke validation or a single harness mode before running the full gate.
Verify: the shorter command completes and writes a usable report artifact.

---

[← Back to Production Cookbook](index.md)
