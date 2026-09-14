[← Back to main docs](index.md)

# Correlation Workflow

Stormlog correlation is a derived investigation view over existing artifacts. It
does not change `TelemetryEvent v4`; instead, the query layer projects telemetry,
markers, rollups, OOM bundles, diagnose manifests, and external attachments into
one evidence list around an investigation anchor.

For broader navigation, `stormlog query runs` and `stormlog query attachments`
provide a run-centric catalog. Correlation remains the timestamp or record
pivot; run envelopes answer which sessions and artifacts belong to the same
top-level investigation.

## Minimum Contract

A trustworthy correlation row needs enough identity, time, and provenance to let
users understand why it appeared:

- `session_id`: the strongest single-run grouping key
- `job_id`: the distributed-run grouping key when ranks write separate sessions
- `rank` and `world_size`: rank-local evidence placement
- `start_ns` and `end_ns`: Unix epoch nanosecond bounds for point or interval evidence
- `source_path` and `source_kind`: where the evidence came from
- `metadata`: source-specific details such as event type, rollup counters, or
  attachment attributes
- `confidence` and `reasons`: a plain explanation of the match quality

Correlation uses both identifiers and timestamps. Timestamp-only matches are
kept as low-confidence fallbacks because they can be useful during partial
recovery, but Stormlog labels them clearly instead of presenting them as proven
links.

## Single-Run And Distributed Correlation

For a single run, `session_id` is the primary pivot. Evidence with the same
session and an overlapping or nearby timestamp window is high confidence. When a
rank is supplied, same-rank evidence is preferred; cross-rank evidence in the
same session is still shown with a lower confidence reason.

For distributed investigations, `job_id` links evidence across sessions and
ranks. Stormlog keeps the same Unix epoch nanosecond clock domain used by
telemetry records and reports cross-rank evidence when it overlaps the anchor
window. If evidence lacks `session_id` and `job_id`, it can only match by time
and is downgraded.

Stormlog does not currently rewrite cross-host clocks. Producers should keep
host clocks synchronized; consumers should inspect `observed_timestamp_ns`,
source paths, and confidence reasons when debugging rank-to-rank timing.

## Correlation Versus Listing

Artifact listing answers "what exists?" Correlation answers "what evidence is
near this suspicious point, and why is it related?" A correlated result is
anchored by `--at-ns` or a projected telemetry `record_id`, filters evidence by
identity and time, then sorts by confidence and distance from the anchor.

This makes correlation suitable for pivots such as:

- memory spikes to nearby phase markers
- degraded collector periods to alert rows
- OOM events to dump bundles and diagnose manifests
- rollup windows to nearby raw telemetry
- telemetry windows to external profiler or experiment-tracking links

## External Attachments

External attachments are discovered from `stormlog_attachments.json` sidecars.
The sidecar is local JSON; Stormlog records URLs or local paths but does not
fetch remote content during correlation.

Minimal shape:

```json
{
  "schema_version": 1,
  "format": "stormlog.attachments",
  "attachments": [
    {
      "attachment_id": "wandb-run-123",
      "title": "W&B run",
      "kind": "experiment",
      "url": "https://wandb.ai/example/project/runs/run-123",
      "session_id": "session-123",
      "job_id": "job-42",
      "rank": 0,
      "start_ns": 1700000000000000000,
      "end_ns": 1700000060000000000,
      "updated_at_utc": "2026-06-14T21:00:00Z",
      "metadata": {"owner": "training"}
    }
  ]
}
```

Relative `path` values resolve against the sidecar directory. Each attachment
must provide either `url` or `path`, and should provide identifiers and time
bounds whenever possible. `attachment_id` is optional but recommended for
sidecars that are updated over time because it gives correlation output a stable
evidence identity.

Sidecars may also include `run_id`, `storage`, `source_namespace`, and
`source_ref`. Those fields are used by the run attachment catalog and remain
optional for existing correlation sidecars.
