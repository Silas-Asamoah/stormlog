[← Back to main docs](index.md)

# Benchmark Harness (v0.2)

This benchmark harness measures current CPU tracker overhead and artifact growth with explicit pass/fail budgets.

## Run the Harness

```bash
python -m examples.cli.benchmark_harness \
  --iterations 200 \
  --output artifacts/benchmarks/latest.json
```

## Enforce Budgets

```bash
python -m examples.cli.benchmark_harness \
  --check \
  --iterations 200 \
  --budgets docs/benchmarks/v0.2_budgets.json \
  --output artifacts/benchmarks/latest.json
```

Use `--check` in local validation scripts to fail fast when budget regressions are introduced.
The default iterations are tuned for stable local/CI signals; keep `--iterations 200`
unless you are intentionally profiling harness variability.

## What It Measures

- `runtime_overhead_pct`: wall-time overhead of a tracked run vs an unprofiled run.
- `cpu_overhead_pct`: CPU-time overhead of a tracked run vs an unprofiled run.
- `sampling_impact_pct`: extra wall-time cost of default sampling vs lower-frequency sampling.
- `artifact_growth_bytes`: additional artifact size from the tracked run vs the unprofiled run.

The current implementation uses `CPUMemoryTracker` and a deterministic CPU workload in `examples/cli/benchmark_harness.py`. Treat it as a budget harness for tracking overhead, not as a full-framework performance benchmark.

## Output Format

The JSON report includes:

- `config`: benchmark configuration and paths.
- `budgets`: loaded threshold values.
- `scenarios`: per-scenario runtime, CPU time, event count, and artifact size.
- `metrics`: computed deltas used for gating.
- `budget_checks`: per-metric value/max/passed results.
- `passed`: overall gate status.

## Versioned Budgets

The v0.2 thresholds live in:

`docs/benchmarks/v0.2_budgets.json`

Update this file in versioned commits when budget policy changes.
