[← Back to Production Cookbook](index.md)

# Always-on Tracking

Use this recipe when you want Stormlog to behave like an operational service:
bounded history in memory, append-only sink artifacts on disk, and enough
session metadata to reconstruct one run later.

Audience: operators, platform owners.
Difficulty: intermediate.

## Prerequisites

- install the package first with [Installation](../installation.md)
- use `pip install "stormlog[torch]"` for `gpumemprof track`
- use `pip install "stormlog[tf]"` for `tfmemprof track`
- use [Command Line Guide](../cli.md) if you need per-flag reference
- a writable artifact directory for sink files
- enough runtime permissions to inspect the target device backend

Success signal:

- a sink manifest is created
- `analyze` can reload the sink
- collector health and retention counters are visible in the output

## When this is the right recipe

- you want long-running `track` sessions
- you need rollover and retention limits on artifacts
- you want the run to stay alive when a collector becomes unhealthy
- you want a stable path from live telemetry to later analysis or TUI loading

## PyTorch always-on baseline

```bash
gpumemprof track \
  --interval 0.5 \
  --warning-threshold 75 \
  --critical-threshold 90 \
  --telemetry-sink-dir ./live_sink \
  --telemetry-flush-seconds 2.0 \
  --telemetry-rollover-mb 64 \
  --telemetry-retention-files 8 \
  --telemetry-retention-total-mb 512
```

What this gives you:

- append-only JSONL sink segments plus a manifest
- one session identity for the run
- rollover and pruning under a bounded artifact budget
- `collector_degraded` and `collector_recovered` events instead of synthetic zero samples

## Add workload phases when timestamps are not enough

Use structured phases when you want long-running artifacts to answer what part
of the workload was active when a hidden-memory anomaly appeared.

```python
from stormlog import MemoryTracker

tracker = MemoryTracker(
    sampling_interval=0.5,
    # telemetry_sink_config=...,  # Optional: configure the append-only sink.
)

tracker.start_tracking()

for epoch in range(num_epochs):
    with tracker.phase("train", metadata={"epoch": epoch}):
        with tracker.phase("load_batch"):
            batch = next(loader)
        with tracker.phase("forward"):
            loss = model(batch).sum()
        with tracker.phase("backward"):
            loss.backward()
        with tracker.phase("optimizer_step"):
            optimizer.step()

tracker.stop_tracking()
```

What changes when phases are present:

- `track` writes companion `phase_enter` / `phase_exit` records with deterministic nested paths
- `gpumemprof analyze` adds phase-aware summaries beside timestamps
- the TUI Diagnostics tab shows the first anomaly phase path for each rank
- when you omit instrumentation entirely, the same workflow stays valid and low-overhead

## TensorFlow always-on baseline

```bash
tfmemprof track \
  --interval 1.0 \
  --threshold 4096 \
  --device /CPU:0 \
  --output ./tf_track.json \
  --telemetry-sink-dir ./tf_live_sink \
  --telemetry-flush-seconds 2.0 \
  --telemetry-rollover-mb 64 \
  --telemetry-retention-files 8 \
  --telemetry-retention-total-mb 512
```

Use `/GPU:0` instead of `/CPU:0` when the TensorFlow runtime has a GPU device
available.
Stop the command cleanly with `Ctrl+C` when you want it to flush the final
output file and session summary.

## Inspect the latest clean session

```bash
gpumemprof analyze ./live_sink --format txt --output ./live_analysis.txt
tfmemprof analyze --input ./tf_track.json --detect-leaks --optimize --report ./tf_report.txt
```

For PyTorch sink directories, default session selection prefers the newest clean
completed session and falls back to interrupted or incomplete sessions only when
needed.

## What to watch during long runs

Treat these values as operational signals, not just debug trivia:

- `rollover_count`
- `pruned_segment_count`
- `pruned_bytes`
- `final_retained_files`
- `final_retained_bytes`
- `history_retained_*`
- `history_dropped_*`
- `collector_failure_event_count`
- `session_status`

## How to interpret degraded mode

If the collector becomes unhealthy during `track`:

- the process keeps running
- new sample emission pauses until recovery
- status events remain visible in the artifact stream
- the final report should still show the collector-health transition history

Treat either of these as actionable:

- non-zero `collector_failure_event_count`
- any final collector state other than `healthy`

## Troubleshooting

### Symptom: sink files grow too quickly

Likely cause: retention is too loose for the deployment budget.
Fix: tighten retention and rollover settings before lowering sample fidelity.
Verify: `final_retained_*`, `pruned_*`, and `rollover_count` stabilize.

### Symptom: the sink's disk fills up or starts failing

Likely cause: a full or failing volume under the sink directory. By default a
failed flush raises from the call that triggered it, and the records stay in
memory until a flush succeeds.
Fix: when the process must keep running on a bad disk, give the sink a memory
bound, `TelemetrySinkConfig(..., max_buffer_bytes=8 * 1024 * 1024)`. Then a
record that would go over the bound is dropped and counted, a failed flush is
counted and retried after a backoff that doubles from
`failure_backoff_seconds` up to `failure_backoff_max_seconds`, and nothing
raises. In either mode a write that fails partway is cut back, so the segment
never keeps a partial line.
Verify: `sink.failure_diagnostics()` reports `buffered_bytes`,
`dropped_records`, `flush_failures` and `last_flush_error`.

### Symptom: tracking stays alive but live telemetry looks partial

Likely cause: the collector entered degraded mode.
Fix: inspect `collector_failure_event_count` and the emitted status events.
Verify: `collector_health_status` returns to `healthy` and sampling resumes.

### Symptom: analysis loads the wrong run from a reused sink directory

Likely cause: more than one session is present in the sink.
Fix: inspect the discovered sessions and target the session you want explicitly.
Verify: the selected session id matches the intended run metadata.

## What to do next

- If the artifact budget is too high, tighten retention before you lower the
  sampling interval.
- If `collector_failure_event_count` is non-zero, move to the
  [Incident Playbooks](incidents.md) degraded-collector checklist.
- If the run needs to be qualified for CI or release use, move to the
  [CI and Release Qualification](ci_release.md) harness workflow.
- If the next question is rank-aware diagnosis, move to the
  [Distributed Diagnostics Recipes](distributed.md).

---

[← Back to Production Cookbook](index.md)
