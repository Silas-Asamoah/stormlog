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

This is the policy used by the pull-request memory gate in CI, which also
passes `--overhead-scratch-root /dev/shm/stormlog-benchmark` so the overhead
trials run on RAM-backed storage (see "Overhead measurement" below).
The checked-in regression assets intentionally cover only the default `pr`
profile.

With `--check`, a failed gate exits `4` (`GATE_FAILED`); without it the
report is still written and the command exits `0`. A missing, malformed,
non-object, wrong-version, or incomplete budget, baseline, or tolerance
file exits `5` (`INVALID_INPUT`) with a one-line message. Problems with a
file itself (missing, unparsable, not an object, wrong version, non-numeric,
baseline config mismatch) are caught before any scenario runs, so a typo
does not cost the run; a budget for a metric the run never produced is
caught after it. An
`--artifact-root` or `--output` that cannot be written exits `1`. See the
[Report and Exit-Code Contract](report_contract.md).

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
- `rss_growth_per_24h_equiv`: in-loop RSS delta (last soak sample minus the
  warmup baseline) normalized to a 24-hour-equivalent run.
- `max_rss_delta_bytes`: largest RSS increase above the warmup baseline seen at
  any soak checkpoint.
- `final_retained_files`: retained append-only segment count after pruning.
- `final_retained_bytes`: retained append-only bytes after pruning.
- `rollover_count`, `pruned_segment_count`, `pruned_bytes`: sink churn under sustained load.
- `history_dropped_*`: bounded-history eviction counts surfaced by the runtime.
- `collector_failure_event_count`: degraded/recovered collector transitions seen during the run.

### Overhead measurement

Each overhead trial times the unprofiled workload and then the same workload
with the runtime emitting one sample per iteration; the trial at the 25th
percentile of wall overhead is reported so a single runner stall does not win.
The tracked run emits its samples synchronously, so every sink flush (a write,
an `fsync`, and a manifest rewrite every 50 events) lands on the workload's
critical path. The unprofiled reference does no I/O at all, which means the
storage latency of the trial directory enters `runtime_overhead_pct` directly
and dominates it on shared runners with slow disks, while `cpu_overhead_pct`
is unaffected.

Pass `--overhead-scratch-root` pointing at a RAM-backed filesystem (CI uses
`/dev/shm/stormlog-benchmark`) to keep the trials' sink I/O out of the wall
measurement. In real use the tracker flushes from its own thread, off the
workload's path, so this is the more faithful comparison. The selected trial
is still promoted into `--artifact-root`, and the scratch root is recorded in
the report as `config.overhead_scratch_root` and `overhead.scratch_root`.
Regressions that add wall time per sample (a sleep, a lock wait) still fail
the gate; see the injected-sleep test in `tests/test_benchmark_harness.py`.

The v0.4 baseline was recorded before `--overhead-scratch-root` existed, so
its `runtime_overhead_pct` includes about 43 points of disk wait (778.51
runtime versus 735.28 CPU). Record the next baseline with the same flag CI
uses; issue #255 tracks that re-baseline.

### Soak RSS measurement window

The soak reads process RSS at 50 evenly spaced checkpoints while samples are
emitted. The first checkpoint (2% of the samples) is the warmup boundary and
becomes `rss_baseline_bytes`; the last checkpoint is the final sample and
becomes `rss_final_bytes`. Both RSS gates are computed from those in-loop
readings only, and the 24-hour extrapolation uses the measured window
(`rss_measured_equivalent_seconds`), not the whole soak.

Session finalization is deliberately outside that window. Closing the sink
loads every retained segment back into memory to build rollups and the runtime
then exports its bounded history, which is a one-shot transient of a few
hundred megabytes on the PR profile. How much of it stays resident afterwards
depends on the allocator, not on tracker growth, so the harness reports it as
`finalization_rss_delta_bytes` (and `rss_after_finish_bytes`) for inspection
without gating it. A genuine leak shows up as a positive slope across the
checkpoints and still fails the gate; see the synthetic leak test in
`tests/test_benchmark_harness.py`.

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
