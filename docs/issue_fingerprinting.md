[← Back to docs](index.md)

# Durable Issue Fingerprinting

Stormlog issue fingerprinting is a deterministic summarization layer over
existing artifacts. It groups repeated failures across sessions without mutating
`TelemetryEvent v4`, append-only sink segments, diagnose bundles, or OOM flight
recorder bundles.

The v1 implementation follows two outside patterns:

- [Sentry SDK Fingerprinting](https://docs.sentry.io/platforms/dotnet/guides/apple/usage/sdk-fingerprinting/)
  treats fingerprints as explicit grouping dimensions.
- [OpenTelemetry Trace Concepts](https://opentelemetry.io/docs/concepts/signals/traces/)
  keeps evidence linkable through context, events, attributes, and links.

## Issue Object

`stormlog.issues.StormlogIssue` is the grouped issue row returned by
`stormlog.query.QueryStore.list_issues()`.

Canonical fields:

- `fingerprint_id`: deterministic hash of schema version, issue kind, and
  normalized fingerprint dimensions
- `fingerprint`: `kind`, `schema_version`, `fingerprint_id`, and `dimensions`
- `kind`: `oom`, `collector_degradation`, `alert`, or `hidden_memory_anomaly`
- `state`: `open`, `resolved`, `ignored`, or `regressed`
- `severity`: `info`, `warning`, or `critical`
- `title`
- `hit_count`
- `first_seen_ns`, `last_seen_ns`
- `affected_sessions`
- `representative_evidence`
- `evidence`
- `details`

`representative_evidence` and each entry in `evidence` can link back to:

- `session_id`
- `timestamp_ns`
- `rank`
- `source_path`
- `source_kind`
- `event_type`
- `bundle_path`
- low-cardinality `metadata`

Current query output defaults derived issues to `open`. State overrides are
accepted by fingerprint id in the Python API so a future persisted sidecar can
restore `resolved`, `ignored`, or `regressed` state without changing raw
telemetry.

## Fingerprint Rules

Fingerprints contain stable grouping dimensions. They intentionally exclude
session ids, timestamps, raw file paths, full exception messages, and metric
magnitudes unless the value is converted into a stable category.

OOM fingerprints use:

- `backend`
- `reason`

OOM details and evidence keep volatile or inconsistently available fields such
as exception module/type, collector, device id, rank, bundle path, event count,
session status, context, and exact timestamps. This lets the same OOM group
together whether it is discovered from an OOM bundle manifest or from telemetry
events.

Collector degradation fingerprints use:

- `collector`
- `backend`
- `health_status`
- sorted `partial_fields`
- normalized `error_stem`

Collector details keep retry timestamps, consecutive failure counts, and source
event metadata.

Alert fingerprints use:

- `event_type`
- `severity`
- `collector`
- `backend`
- normalized alert `category`

High-fragmentation alerts use the stable category `high_fragmentation`, so
`High fragmentation: 40.0%` and `High fragmentation: 51.5%` group together.

Hidden-memory anomaly fingerprints use:

- `classification`: `transient_spike`, `persistent_drift`, or
  `fragmentation_like`
- `severity`
- stable phase summary when available
- `collector`
- `backend`

Hidden-memory details keep confidence, z-score, slope, gap bytes,
fragmentation ratios, sample counts, and phase-attribution payloads.

## Worked Examples

OOM bundle:

```json
{
  "kind": "oom",
  "dimensions": {
    "backend": "cuda",
    "reason": "message_pattern:out of memory"
  }
}
```

Collector degradation:

```json
{
  "kind": "collector_degradation",
  "dimensions": {
    "backend": "cuda",
    "collector": "stormlog.cuda_tracker",
    "error_stem": "runtimeerror",
    "health_status": "degraded",
    "partial_fields": ["device_free_bytes"]
  }
}
```

High-fragmentation alert:

```json
{
  "kind": "alert",
  "dimensions": {
    "backend": "cuda",
    "category": "high_fragmentation",
    "collector": "stormlog.cuda_tracker",
    "event_type": "warning",
    "severity": "warning"
  }
}
```

Hidden-memory drift:

```json
{
  "kind": "hidden_memory_anomaly",
  "dimensions": {
    "backend": "cuda",
    "classification": "persistent_drift",
    "collector": "stormlog.cuda_tracker",
    "phase": "train / forward",
    "severity": "critical"
  }
}
```

## Where Issues Live

In v1, grouped issues are derived at query time:

```bash
stormlog query issues ./live_sink ./oom_dumps --json
```

Python callers can use:

```python
import stormlog.query

store = stormlog.query.open(["./live_sink", "./oom_dumps"])
issues = store.list_issues()
```

A future persistence pass should write a derived artifact-level `issues.json`
sidecar next to the artifact set. That sidecar should contain grouped issue
state and cached summaries only. It should not rewrite telemetry events, sink
segments, diagnose manifests, or OOM bundle manifests.

## Follow-On Tasks

- Persist and reload `issues.json` state overrides by `fingerprint_id`.
- Add TUI issue tables and issue-detail panes backed by `list_issues()`.
- Add issue-oriented report schema fields for agent automation.
- Add regression detection by comparing current issue fingerprints with a
  previous persisted sidecar.
- Add controls for ignoring known noisy fingerprints from CLI/TUI surfaces.
