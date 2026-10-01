[← Back to main docs](index.md)

# Command Line Guide

Stormlog currently exposes four console scripts:

- `gpumemprof`
- `tfmemprof`
- `jaxmemprof`
- `stormlog`

Use `gpumemprof`, `tfmemprof`, and `jaxmemprof` for framework memory
automation. Use `stormlog` with no arguments when you want the Textual TUI,
`stormlog query` when you want local artifact queries, or `stormlog infer` when
you want OpenAI-compatible inference endpoint profiling.

If you want task-oriented operational recipes instead of option-by-option
guidance, use the [Production Cookbook](cookbook/index.md), especially
[Always-on Tracking](cookbook/always_on.md),
[Incident Playbooks](cookbook/incidents.md), and
[Distributed Diagnostics Recipes](cookbook/distributed.md).

## Verify the installed commands

```bash
gpumemprof --help
tfmemprof --help
jaxmemprof --help
stormlog --help
stormlog query --help
stormlog infer --help
```

If you are working from a repository checkout, `pip install -e .` also exposes
the source-only `examples/` package used in a few release-validation flows.

Install and launch the TUI with the current dependency set:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

## `stormlog`

The top-level `stormlog` command is TUI-first for compatibility:

```bash
stormlog       # launch the TUI
stormlog tui   # launch the TUI explicitly
```

The command also dispatches non-TUI workflows without importing Textual:

```bash
stormlog query --help
stormlog infer --help
```

### Profile OpenAI-compatible inference

Use `stormlog infer profile` for active endpoint profiling:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,4,8 \
  --input-tokens 512,2048 \
  --output-tokens 128,512 \
  --requests 20 \
  --output artifacts/infer_qwen.jsonl
```

The artifact is newline-delimited JSON with one session record, request traces,
optional system samples, and a summary record. Analyze it with:

```bash
stormlog infer analyze artifacts/infer_qwen.jsonl
stormlog infer analyze artifacts/infer_qwen.jsonl --format json --output report.json
```

For token fallback support beyond server usage metadata, install:

```bash
pip install "stormlog[infer-tokenizers]"
```

See [Inference Profiling](inference.md) for the full endpoint profiling guide.
For server memory, run `stormlog infer collect-server` on the serving host,
pass the same `--run-id` to `infer profile`, and import the collector JSONL with
`infer analyze --server-telemetry`. The guide explains the direct-route and
clock evidence required before a server sample enters a case report.

## `gpumemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
gpumemprof info
gpumemprof info --device 0 --detailed
```

`gpumemprof info` still reports the active PyTorch runtime first. When no
supported PyTorch GPU runtime is active, it now falls back to a best-effort host
GPU hardware probe so the command can show detected device names separately from
runtime availability. Supported PyTorch GPU runtimes remain NVIDIA CUDA, AMD
ROCm-backed PyTorch on Linux, and Apple MPS. In that unsupported-runtime mode,
`--device` is ignored because it only applies to an active PyTorch GPU runtime.

### Capture a bounded monitoring window

```bash
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.csv --format csv
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.json --format json
```

### Track events over time

```bash
gpumemprof track --duration 30 --interval 0.5 --output track.json --format json
gpumemprof track --warning-threshold 75 --critical-threshold 90 --output alerts.csv
gpumemprof track --job-id train-42 --rank 1 --local-rank 1 --world-size 8 --output rank1.json --format json
gpumemprof track --telemetry-sink-dir ./live_sink --telemetry-rollover-mb 32 --telemetry-retention-total-mb 256
```

Every `gpumemprof track` run now creates exactly one session identity. The
session begins after tracker startup succeeds and before the first record is
persisted, and it is marked `completed` only after clean shutdown finalization
finishes.

If your Python workload instruments phases with `tracker.phase(...)` or
`tracker.enter_phase(...)`, `track` persists the emitted `phase_enter` /
`phase_exit` records alongside the regular telemetry samples. The CLI does not
invent phase records on its own; it only preserves the structured phase events
your workload emitted. Phase records remain optional and do not change the
`track` CLI surface in v1.

For long-running tracking sessions, Stormlog now degrades gracefully when a
collector becomes unhealthy:

- the tracked workload keeps running
- exported telemetry includes `collector_degraded` / `collector_recovered` events
- per-event metadata marks partial or unhealthy collector state
- retries use bounded exponential backoff instead of crashing the tracker loop

For always-on sessions, `gpumemprof track` can also stream append-only
telemetry into a sink directory during the run instead of waiting for shutdown.
The sink writes JSONL segments plus a manifest, rolls segments when they hit the
configured size limit, and prunes the oldest closed segments to stay within the
retention budget.

The sink manifest also keeps a session ledger, so multiple runs can safely
share the same sink directory without merging captures. If a previous run was
still `running` when the process died, the next startup recovers it as
`interrupted` and starts a fresh session for the new run.

Useful sink options:

- `--telemetry-sink-dir`
- `--telemetry-flush-seconds`
- `--telemetry-rollover-mb`
- `--telemetry-retention-files`
- `--telemetry-retention-total-mb`

The always-on qualification harness assumes the default sink settings:

- flush every `2.0s`
- roll segments at `64 MB`
- retain at most `8` files
- retain at most `512 MB` total

When you inspect `track` output after a long run, look for these diagnostics in
the JSON or CLI summary:

- `rollover_count`
- `pruned_segment_count`
- `pruned_bytes`
- `final_retained_files`
- `final_retained_bytes`
- `history_retained_*`
- `history_dropped_*`

Optional OOM flight-recorder support:

```bash
gpumemprof track \
  --oom-flight-recorder \
  --oom-dump-dir ./oom_dumps \
  --oom-max-dumps 10 \
  --oom-max-total-mb 1024 \
  --output track.json --format json
```

### Analyze saved telemetry

```bash
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof analyze track.json --visualization --plot-dir plots
gpumemprof analyze ./live_sink
gpumemprof analyze ./live_sink --session-id 2b30f4a4-7d2d-48f7-a9f6-7d40c14eb95e
```

`gpumemprof analyze` uses a positional input file. It can now read a normal JSON
telemetry export, a sink JSONL segment, or a sink directory containing the
current and rolled append-only outputs. If you add `--visualization`, plots are
written to the directory passed via `--plot-dir` or to `plots/` by default.

When the telemetry stream includes structured phase boundaries, the text summary
also includes phase-aware hints such as:

- `Top gap phase: train / forward`
- `Suspect phase: train / communication`

When Stormlog cannot prove a unique phase but can still surface a useful winner,
the summary uses a heuristic marker instead of pretending certainty:

- `Top gap phase: (likely) train / communication`

In JSON report payloads, that distinction is preserved as:

- canonical `phase_attribution.phase_resolution`
- canonical `phase_attribution.phase_source`
- optional `phase_attribution.phase_summary` only when the displayed winner is heuristic

When multiple sessions are present, `gpumemprof analyze` selects:

1. the newest `completed` session
2. otherwise the newest `interrupted` session
3. otherwise the newest `incomplete` session

Use `--session-id` to analyze a specific capture instead of the default one.

When phase records are present, `gpumemprof analyze` also reports the top
phase-attributed gap finding and the top first-cause suspect phase in the text
summary. The JSON report keeps the structured `phase_attribution` payload next
to:

- `gap_analysis`
- `collective_attribution`
- `cross_rank_analysis.first_cause_suspects`

For always-on deployment posture and incident response checklists, continue with
[Always-on Tracking](cookbook/always_on.md) and
[Incident Playbooks](cookbook/incidents.md).

### Produce a diagnose bundle

```bash
gpumemprof diagnose --duration 5 --interval 0.5 --output ./diag_bundle
gpumemprof diagnose --duration 0 --output ./diag_bundle_quick
gpumemprof diagnose --native-history --duration 0 --output ./diag_bundle_native
```

Use `--duration 0` when you want a fast artifact bundle without a new tracking window.

For task-oriented recipes that combine `track`, `analyze`, `diagnose`, and the
TUI into one workflow, continue with
[Incident Playbooks](cookbook/incidents.md) and
[Distributed Diagnostics Recipes](cookbook/distributed.md).

Each standalone diagnose bundle also owns its own session id. The bundle
manifest records whether the run finished `completed` or was left
`incomplete`, and synthesized timeline telemetry inherits that same session id
when reloaded later.

`--native-history` is a CUDA-only debug mode. It records allocator history for
the current `gpumemprof diagnose` process, then writes native snapshot artifacts
such as `cuda_allocator_snapshot.pickle`,
`cuda_allocator_state_history.html`,
`cuda_allocator_state_history_annotated.html`, and tensor-attribution JSON
alongside the normal diagnose bundle files. The annotated HTML is the
Stormlog-native view that exposes the timeline trace, segment explorer, and
active-memory table in one file. For a maintained workflow example of that
artifact, continue with [PyTorch Production Recipes](cookbook/pytorch.md). On
MPS, ROCm, or CPU-only runtimes, the command fails explicitly instead of
pretending support.

## `stormlog query`

`stormlog query` asks structured questions over local artifact directories. It
uses the same canonical telemetry loaders as `gpumemprof analyze`, but exposes
rows that are easier to filter, export, and reuse from automation.

The query surface is local-first and file-backed. It reads sink manifests and
bundle manifests before loading raw events, so listing sessions or OOM bundles
does not require parsing every JSONL segment in a large sink directory.

List top-level runs:

```bash
stormlog query runs ./artifacts --json
stormlog query runs ./distributed_runs --job-id train-42 --table
```

When `stormlog_run.json` exists, this command lists explicit run envelopes.
Without an envelope, Stormlog synthesizes runs from sessions: rank-local
sessions with the same non-null `job_id` become one distributed run, while
sessions without `job_id` stay separate.

List run-indexed attachments:

```bash
stormlog query attachments ./artifacts --run-id run-train-42 --table
stormlog query attachments ./artifacts --source-namespace wandb --json
stormlog query attachments ./live_sink --kind telemetry_sink_segment --csv
```

Attachments include local telemetry artifacts, OOM and diagnose bundles, rollup
sidecars, `stormlog_attachments.json` entries, and external references declared
by a run envelope.

List sessions:

```bash
stormlog query sessions ./live_sink --status interrupted --json
stormlog query sessions ./artifacts --has-oom-bundle --table
```

Query events:

```bash
stormlog query events ./live_sink \
  --session-id 2b30f4a4-7d2d-48f7-a9f6-7d40c14eb95e \
  --rank 0 \
  --event-type collector_degraded \
  --limit 50
```

List OOM bundles:

```bash
stormlog query ooms ./artifacts --backend cuda --table
stormlog query ooms ./artifacts --created-after 2026-05-12T00:00:00Z --json
```

List grouped recurring issues:

```bash
stormlog query issues ./live_sink ./oom_dumps --kind oom --json
stormlog query issues ./artifacts --severity warning --session-id session-123
```

Run built-in summaries:

```bash
stormlog query summary ./live_sink \
  --metric peak_allocator_reserved_bytes \
  --group-by session

stormlog query summary ./live_sink \
  --metric hidden_memory_gap_growth \
  --group-by session-rank
```

Find correlated evidence around a timestamp or projected telemetry record:

```bash
stormlog query correlate ./live_sink ./artifacts \
  --at-ns 1800000000000000010 \
  --session-id session-123 \
  --window-ns 60000000000 \
  --json

stormlog query correlate ./distributed_runs \
  --at-ns 1800000000000000010 \
  --job-id train-42 \
  --scope distributed \
  --kind alert \
  --kind oom_bundle
```

Correlation returns nearby evidence with confidence reasons instead of only
listing artifacts. Evidence can include telemetry events, timeline markers,
alerts, OOM bundles, diagnose bundles, rollup windows, and local attachment
sidecars. Use `--record-id` instead of `--at-ns` to anchor on a projected
telemetry record id.

Supported output formats:

- `--table`: readable table output, used by default
- `--json`: machine-readable rows
- `--csv`: row-query exports for `runs`, `attachments`, `sessions`, `events`,
  and `ooms`

The Python API behind the CLI is available as:

```python
import stormlog.query

store = stormlog.query.open(["./live_sink", "./oom_dumps"])
runs = store.list_runs()
attachments = store.list_run_attachments()
sessions = store.list_sessions()
events = store.query_events()
ooms = store.list_oom_bundles()
issues = store.list_issues()
```

For engine-choice details and follow-on work, see
[Local Query Layer](query_layer.md).
For issue grouping rules and schema details, see
[Durable Issue Fingerprinting](issue_fingerprinting.md).

## `tfmemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
tfmemprof info
```

### Monitor TensorFlow memory usage

```bash
tfmemprof monitor --interval 0.5 --duration 30 --output tf_monitor.json
tfmemprof monitor --interval 0.5 --duration 30 --threshold 4096 --device /GPU:0 --output tf_monitor_threshold.json
```

For CPU-only TensorFlow or when the GPU backend is unavailable, use
`--device /CPU:0`:

```bash
tfmemprof monitor --interval 0.5 --duration 30 --device /CPU:0 --output tf_monitor.json
tfmemprof track --interval 0.5 --threshold 4096 --device /CPU:0 --output tf_track.json
```

### Track TensorFlow memory usage

```bash
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json
tfmemprof track --interval 0.5 --threshold 4096 --job-id train-42 --rank 3 --local-rank 1 --world-size 8 --output tf_rank3.json
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json --telemetry-sink-dir ./tf_live_sink
```

`tfmemprof track` follows the same degraded-mode rules as the PyTorch tracker:
collector failures pause new sample emission, status events remain visible in the
artifact stream, and normal sampling resumes automatically after recovery.
The same append-only sink options are available when you need bounded,
interrupt-tolerant TensorFlow telemetry during a long-running session.

TensorFlow tracking also keeps only a bounded recent history in memory. The
current CLI output and JSON exports surface the retained vs dropped sample,
event, and alert counts so long-running jobs can distinguish expected eviction
from a silent memory-growth regression.

The same session rules apply to TensorFlow tracking:

- one session id per `tfmemprof track` run
- sink recovery marks old running sessions as `interrupted`
- loaders and diagnostics separate same-host runs by `session_id`, not by job or rank alone

Like the PyTorch and CPU trackers, TensorFlow tracking also preserves optional
structured phase companion records when you instrument the tracker through the
Python API.

### Analyze TensorFlow results

```bash
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize --visualize --report tf_report.txt
```

Unlike `gpumemprof analyze`, the TensorFlow analyzer uses `--input`.

### Produce a diagnose bundle

```bash
tfmemprof diagnose --duration 5 --interval 0.5 --output ./tf_diag
tfmemprof diagnose --duration 0 --output ./tf_diag_quick
```

## `jaxmemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
jaxmemprof info
```

### Monitor JAX memory usage

```bash
jaxmemprof monitor --interval 0.5 --duration 30 --output jax_monitor.json
jaxmemprof monitor --interval 0.5 --duration 30 --device gpu --output jax_monitor_gpu.json
```

`--device` accepts a local-device index or the named `cpu`, `gpu`, `tpu`, and
`metal` backends. For CPU-only execution, use `--device cpu`:

```bash
jaxmemprof monitor --interval 0.5 --duration 30 --device cpu --output jax_monitor.json
jaxmemprof track --interval 0.5 --device cpu --output jax_track.json
```

JAX memory sample events are emitted only when the selected backend exposes
`memory_stats().bytes_in_use`. On runtimes without allocator counters,
`jaxmemprof` marks device memory unavailable and suppresses sample events.
Monitor and track results report process RSS separately. Collector status events
retain the numeric allocator fields required by the telemetry schema, but mark
those fields as partial in `metadata` rather than presenting them as samples.

### Track JAX memory usage

```bash
jaxmemprof track --interval 0.5 --output jax_track.json
jaxmemprof track --interval 0.5 --job-id train-42 --rank 2 --local-rank 0 --world-size 8 --output jax_rank2.json
jaxmemprof track --interval 0.5 --output jax_track.json --telemetry-sink-dir ./jax_live_sink
```

`jaxmemprof track` shares the same robust, degraded-mode semantics as the PyTorch and TensorFlow trackers, allowing it to gracefully handle long-running runs, collector interruptions, and append-only sink persistence.

JAX tracking also logs structured phase boundaries when using `jaxmemprof.MemoryTracker` instrumentation and emits telemetry streams that are compatible with `gpumemprof analyze` and the Textual TUI.

### Analyze JAX results

```bash
jaxmemprof analyze --input jax_monitor.json --detect-leaks --optimize
jaxmemprof analyze --input jax_monitor.json --detect-leaks --optimize --visualize --report jax_report.txt
```

### Produce a diagnose bundle

```bash
jaxmemprof diagnose --duration 5 --interval 0.5 --output ./jax_diag
jaxmemprof diagnose --duration 0 --output ./jax_diag_quick
```

## TUI launch

```bash
pip install "stormlog[tui,torch]"
stormlog
```

Inside the TUI, the `CLI & Actions` tab exposes quick actions for:

- `gpumemprof info`
- `gpumemprof monitor`
- `tfmemprof monitor`
- `jaxmemprof monitor`
- `gpumemprof diagnose`
- sample workloads
- OOM scenario runner
- capability matrix smoke run

## Release-validation shortcuts

**Source checkout only.** These commands require the `examples/` package from a
repository clone:

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

**Pip users** should use this CLI-only sequence instead:

```bash
gpumemprof info
gpumemprof track --duration 2 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag

tfmemprof info
tfmemprof diagnose --duration 0 --output ./tf_diag

jaxmemprof info
jaxmemprof diagnose --duration 0 --output ./jax_diag
```

## Choosing the right command

### Use `monitor` when

- you want a bounded sample window
- you only need a simple CSV or JSON output

### Use `track` when

- you want event streams and alert thresholds
- you want later exports or distributed identity fields
- you want a reconstructable session that owns its sink, diagnose, or OOM artifacts

### Use `analyze` when

- you already have saved telemetry
- you want a report or plot output

### Use `diagnose` when

- you need a portable artifact bundle to archive or share
- you plan to inspect the output later in the TUI diagnostics flow

---

[← Back to main docs](index.md)
