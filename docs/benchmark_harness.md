[← Back to main docs](index.md)

# Benchmark Harness (v0.4)

> **Source checkout only.** `python -m examples.cli.benchmark_harness` requires
> the repository `examples/` package and `docs/benchmarks/`. It is not shipped
> in the PyPI package.

The v0.4 benchmark harness measures always-on monitoring as a benchmarked
operability budget, not just as a point-in-time benchmark.

It still supports the same two gate modes:

- `budget`: compare current metrics to absolute max thresholds
- `regression`: compare current metrics to a checked-in baseline plus allowed deltas

On top of that, v0.4 extends the benchmark coverage to:

- `gpumemprof` CPU fallback (`gpumemprof_cpu`)
- `tfmemprof --device /CPU:0` (`tfmemprof_cpu`)
- accelerated soak qualification
- rollover and retention validation
- history truncation diagnostics
- actionable failure reporting

## Default operating assumptions

The harness models the always-on default runtime mode as:

- `track` with append-only sink enabled
- `flush_every_seconds=2.0`
- `rollover_max_bytes=64 MB`
- `retention_max_files=8`
- `retention_max_total_bytes=512 MB`

Retention validation also runs a forced-churn subtest with tighter limits so
rollover and pruning are exercised even in fast local runs.

## Profiles

- `pr`: accelerated `6h`-equivalent soak plus default-interval overhead checks
- `nightly`: accelerated `24h`-equivalent soak plus the same overhead checks

“24h-equivalent” means the harness does not sleep between samples. Instead, it
collects the same number of samples that a 24-hour run would emit at the
runtime’s default interval:

- `gpumemprof_cpu`: `864000` samples at `0.1s`
- `tfmemprof_cpu`: `86400` samples at `1.0s`

## Modes

- `overhead`: run only the unprofiled vs tracked overhead comparison
- `soak`: run only the accelerated soak and retention validation
- `all`: run both

## Run the harness

```bash
python -m examples.cli.benchmark_harness \
  --profile pr \
  --mode all \
  --output artifacts/benchmarks/latest_v0.4.json
```

## Enforce Regression Gate

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

This is the policy used by the pull-request memory gate in CI.
The checked-in regression assets intentionally cover only the default `pr`
profile.

## Enforce Budgets

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

Use budget mode when you want a short benchmark run checked against absolute
operating thresholds rather than baseline deltas.

## Nightly operating-budget gate

```bash
python -m examples.cli.benchmark_harness \
  --check \
  --profile nightly \
  --mode all \
  --gate-mode budget \
  --iterations 5000 \
  --budgets docs/benchmarks/v0.4_operating_budget.json \
  --output artifacts/benchmarks/latest_v0.4_nightly.json
```

This keeps budget enforcement in place for the longer nightly soak profile, so
the short-run checks and long-run checks use the same benchmark policy model.

## What it measures

- `runtime_overhead_pct`: wall-clock overhead of the tracked default mode vs the unprofiled workload.
- `cpu_overhead_pct`: CPU-time overhead of the tracked default mode vs the unprofiled workload.
- `artifact_growth_bytes`: tracked-output size minus the unprofiled output size.
- `rss_growth_per_24h_equiv`: RSS delta normalized to a 24-hour-equivalent run.
- `max_rss_delta_bytes`: largest observed RSS increase above the soak baseline.
- `final_retained_files`: retained append-only segment count after pruning.
- `final_retained_bytes`: retained append-only bytes after pruning.
- `rollover_count`, `pruned_segment_count`, `pruned_bytes`: sink churn under sustained load.
- `history_dropped_*`: bounded-history eviction counts surfaced by the runtime.
- `collector_failure_event_count`: degraded/recovered collector transitions seen during the run.

## Output format

The v0.4 report includes:

- `profile`, `mode`, `gate_mode`
- `config`: comparison config plus runtime-specific sample counts
- `runtimes`: per-runtime overhead, soak, retention-validation, and diagnostic data
- `metrics`: flattened per-runtime metrics used for gating
- `budget_checks` or `regression_checks`
- `failure_diagnostics`: actionable failures with collector, sink, and history context
- `passed`

## Interpreting failures

Failure lines are intentionally verbose. A budget or regression failure includes:

- the failing metric and threshold
- the runtime name
- collector health state
- collector failure count
- rollover and prune counts
- retained file and byte totals
- retained and dropped history counters

Typical examples:

- overhead regression: runtime or CPU overhead jumped materially above baseline
- retention failure: retained files or bytes exceeded the configured sink budget
- collector failure: degraded-mode transitions occurred during the soak
- history drift: dropped-event or dropped-sample counts grew beyond the expected envelope

## Tuning order

When a run fails, adjust knobs in this order:

1. sampling interval
2. sink flush cadence
3. rollover size or rollover event count
4. retention file and byte limits
5. TensorFlow `max_history` if sample/event windows are too large for the deployment

## Versioned assets

The v0.4 harness reads:

- `docs/benchmarks/v0.4_operating_budget.json`
- `docs/benchmarks/v0.4_baseline.json`
- `docs/benchmarks/v0.4_tolerances.json`

Telemetry v4 intentionally repeats the complete memory-capability declaration
in each canonical event so standalone JSONL records remain self-describing. The
v0.4 PR tolerances include that bounded serialization and retained-record cost
for the PyTorch and TensorFlow CPU reference lanes; event-count, file-count,
rollover, and CPU-overhead gates remain unchanged.

Update these files only with an intentional benchmark refresh. Run the harness
with the same profile and config as CI, inspect the new metrics, then commit the
asset update separately from unrelated code.
