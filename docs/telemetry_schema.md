[← Back to main docs](index.md)

# TelemetryEvent v4 Schema

`TelemetryEvent v4` is the canonical event format for tracker exports,
append-only sink segments, synthesized diagnose timelines, and loader output.
The backend-neutral `ProjectedTelemetryRecord` described in
[Stormlog Telemetry Projection](telemetry_projection.md) is an internal view
derived from this persisted schema; it is not a replacement artifact format.

Schema files:

- `docs/schemas/telemetry_event_v4.schema.json`
- legacy compatibility: `docs/schemas/telemetry_event_v3.schema.json` and
  `docs/schemas/telemetry_event_v2.schema.json`

## Required fields

- `schema_version` (`4`)
- `session_id`
- `timestamp_ns`
- `event_type`
- `collector`
- `sampling_interval_ms`
- `pid`
- `host`
- `device_id`
- `allocator_allocated_bytes`
- `allocator_reserved_bytes`
- `allocator_active_bytes`
- `allocator_inactive_bytes`
- `allocator_change_bytes`
- `device_used_bytes`
- `device_free_bytes`
- `device_total_bytes`
- `context`
- `metadata`

Every memory key remains present. Allocator and device counters are
`integer | null`: `null` means the backend does not expose that measurement, or
the collector declared the supported field temporarily unavailable in its
diagnostics. Loaders upgrade v2, v3, and recognized legacy records to v4 while
preserving measured values.

## Session identity and lifecycle

Every long-running capture now belongs to exactly one session.

- `session_id` is an opaque UUID string generated once per `track` run, TUI live session, or standalone `diagnose` bundle
- session identity is the authoritative grouping key for telemetry and attached artifacts
- host, job, and rank remain descriptive metadata, but they do not define capture ownership

Session lifecycle is recorded separately from the event stream:

- `running`: session started but no terminal state has been persisted yet
- `completed`: clean shutdown finished and terminal session state was written
- `interrupted`: a previously running append-only sink session was recovered on the next startup
- `incomplete`: loaders found partial or orphaned artifacts that cannot prove a clean or recovered stop

Default session selection when multiple sessions are present:

1. newest `completed`
2. newest `interrupted`
3. newest `incomplete`

## Distributed identity fields

`TelemetryEvent v4` also recognizes these top-level distributed identity fields:

- `job_id` (`string | null`)
- `rank` (`integer`)
- `local_rank` (`integer`)
- `world_size` (`integer`)

New exports always emit these fields. For single-process runs, the defaults are:

- `job_id` -> `null`
- `rank` -> `0`
- `local_rank` -> `0`
- `world_size` -> `1`

`TelemetryEvent v4` validation is strict:

- unknown top-level fields are rejected
- `metadata` must be a JSON object (`dict` in Python terms)
- `rank` and `local_rank` must be >= `0`
- `world_size` must be >= `1`
- `rank` and `local_rank` must be < `world_size`

## Collector values

- `stormlog.cuda_tracker`
- `stormlog.rocm_tracker`
- `stormlog.mps_tracker`
- `stormlog.cpu_tracker`
- `stormlog.tensorflow.memory_tracker`
- `stormlog.jax.memory_tracker`

## Backend capability metadata

Every v4 event includes the complete capability contract under
`metadata["memory_capabilities"]`:

- identity: `backend`, `telemetry_collector`, and `sampling_source`
- allocator counters: `supports_allocator_allocated`,
  `supports_allocator_reserved`, `supports_allocator_active`, and
  `supports_allocator_inactive`
- device counters: `supports_device_used`, `supports_device_free`, and
  `supports_device_total`
- features: `supports_native_allocator_history`,
  `supports_fragmentation_analysis`, `supports_allocator_attribution`, and
  `supports_bounded_profiling`

The older flat `backend`, `supports_device_total`, `supports_device_free`, and
`sampling_source` metadata keys remain in tracker exports for compatibility.
Unsupported counters remain `null`. A supported counter that is temporarily
unavailable is also `null`, and its field name appears in
`collector_partial_fields` with an explanation in collector diagnostics.

## Collector health metadata

Always-on tracker exports annotate collector degradation in `metadata`:

- `collector_health_status` (`healthy`, `degraded`, `unhealthy`)
- `telemetry_partial` (`bool`)
- `collector_partial_fields` (`list[str]`)
- `collector_last_error` (`string | null`)
- `collector_consecutive_failures` (`integer`)
- `collector_next_retry_epoch_s` (`float | null`)

Tracker exports may also emit these lifecycle events:

- `collector_degraded`
- `collector_recovered`
- `start`
- `stop`
- `phase_enter`
- `phase_exit`

When the collector cannot produce core metrics, Stormlog pauses sample emission
until the next retry window and records the degraded state instead of exporting
synthetic zero-valued samples.

## Structured phase metadata

Phase-aware trackers store workload boundaries under `metadata["phase_scope"]`.
This payload is attached to `phase_enter` and `phase_exit` events and is replayed
later by the analyzer and TUI.

The current shape is:

- `action` (`enter` or `exit`)
- `name` (leaf phase label)
- `path` (`list[str]`)
- `depth` (`int`)
- `scope_id` (`string`)
- `parent_scope_id` (`string | null`)
- `thread_id` (`int`)
- `thread_name` (`string`)
- `sequence` (`int`)
- `attributes` (`object`, optional)

Example:

```json
{
  "event_type": "phase_enter",
  "metadata": {
    "phase_scope": {
      "action": "enter",
      "name": "forward",
      "path": ["train", "forward"],
      "depth": 2,
      "scope_id": "session-1:2",
      "parent_scope_id": "session-1:1",
      "thread_id": 88,
      "thread_name": "MainThread",
      "sequence": 2,
      "attributes": {"epoch": 3}
    }
  }
}
```

## Timeline markers

Timeline markers are a derived view over canonical telemetry, not a new
`TelemetryEvent` top-level schema. This keeps v4 event validation strict while
letting CLI, TUI, and query surfaces align important landmarks on one timeline.

The first marker contract is exposed through `stormlog.timeline_markers`:

- `TimelineMarker`
- `derive_timeline_markers(events)`
- `derive_session_timeline_markers(loaded_session)`
- `timeline_marker_to_dict(marker)`

TUI artifact loading exposes the same derived view without mutating session
events: `load_distributed_artifacts(...).markers` contains markers for the
selected session, and `build_distributed_model(...).markers_by_rank` groups them
for rank timeline rendering.

Canonical marker fields:

- `session_id`
- `start_ns`
- `end_ns` (`null` for point markers)
- `kind` (`lifecycle`, `collector`, `alert`, `oom`, or `phase`)
- `source` (`telemetry_event` or `phase_replay`)
- `severity` (`info`, `warning`, or `critical`)
- `label`
- `rank`, `local_rank`, `world_size`
- `event_type`
- `metadata`

System-generated point markers are promoted from existing telemetry events:

- `start`
- `stop`
- `collector_degraded`
- `collector_recovered`
- `warning`
- `critical`
- `error`

`error` events with OOM metadata such as `metadata["oom_reason"]` are promoted
as `oom` markers. Structured phases are promoted as interval markers by replaying
matching `phase_enter` and `phase_exit` records through `PhaseReplayIndex`.

User-authored annotations should remain separate from raw telemetry events in a
future sidecar or catalog layer. They can share the `TimelineMarker` shape, but
they should use an annotation-specific `source` rather than mutating historical
telemetry records.

## External attachment sidecars

Correlation queries can also discover local `stormlog_attachments.json`
sidecars. These sidecars are not raw telemetry and do not change
`TelemetryEvent v4`; they are catalog/query-layer metadata that links external
evidence such as experiment-tracking URLs, profiler traces, or runbooks to a
session/job/rank/time window.

Schema file:

- `docs/schemas/stormlog_attachments_v1.schema.json`

Top-level fields:

- `schema_version` (`1`)
- `format` (`stormlog.attachments`)
- `attachments`

Each attachment includes:

- `title`
- `kind`
- `attachment_id` for stable lifecycle-managed sidecars
- `url` or `path`
- optional `run_id`
- `session_id`, `job_id`, and `rank` when known
- `start_ns` and `end_ns` when the attachment is time-bounded
- `created_at_utc` and `updated_at_utc`
- `storage` (`reference` or `copy`) when the producer knows whether the file was
  copied into the local artifact set
- optional `source_namespace` and `source_ref` for integration navigation
- `metadata`

Relative `path` values resolve against the sidecar directory. Stormlog records
remote URLs but does not fetch them during local correlation.

Run-level grouping is described by `stormlog_run.json`, whose schema is
`docs/schemas/stormlog_run_envelope_v1.schema.json`. See
[Run Envelopes](run_envelopes.md) for the distinction between sessions and
top-level run envelopes.

## Analyzer phase attribution payloads

Analyze and Diagnostics outputs may also include a derived
`phase_attribution` object. This is not part of the raw event schema above; it
is report-layer data produced after replaying the `phase_scope` boundaries.

Canonical fields:

- `phase_resolution` (`unique`, `ambiguous`, or omitted when no attribution exists)
- `phase_source` (`exact`, `thread_local`, `heuristic`, or omitted)
- `phase_path` (`string`, only for unique attributions)
- `phase_paths` (`list[str]`, one or more candidate labels)
- `scope_id`, `thread_id`, `thread_name` (present only when uniquely tied to one scope)

Optional presentation field:

- `phase_summary`
  - `phase_path`
  - `source`

`phase_summary` is emitted only when the product wants one useful display label
even though the canonical attribution remains ambiguous. For example, the CLI
may show `(likely) train / communication` while the underlying
`phase_attribution.phase_resolution` still stays `ambiguous`.

If ambiguity collapses to only one distinct label, Stormlog keeps the canonical
ambiguity and omits `phase_summary` because there is no materially different
winner to show.

## Legacy compatibility

Legacy conversion is permissive by default in
`stormlog.telemetry.telemetry_event_from_record`. Legacy conversion is attempted
only when `schema_version` is absent.

If `schema_version` is present:

- it must be an integer
- it must be a supported schema version
- any other value is rejected without legacy fallback

Legacy defaults:

- missing `pid` -> `-1`
- missing `host` -> `"unknown"`
- missing `device_id` -> inferred from `device` if possible, otherwise `-1`
- missing `allocator_reserved_bytes` -> `allocator_allocated_bytes`
- missing `allocator_change_bytes` -> `0`
- missing `device_used_bytes` -> `allocator_allocated_bytes`
- missing `device_total_bytes` and `device_free_bytes` -> `null`
- missing `event_type` -> `type` field if present, else `"sample"`
- missing distributed identity -> `job_id: null`, `rank: 0`, `local_rank: 0`, `world_size: 1`
- legacy `metadata_*` fields are folded into the canonical `metadata` object
- legacy artifacts without `session_id` receive a deterministic synthetic session id during load

If a legacy record is missing a valid timestamp, conversion fails.

## Distributed env inference

Tracker constructors can infer distributed identity from common launcher env vars:

- PyTorch / `torchrun`: `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `TORCHELASTIC_RUN_ID`
- Open MPI: `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`, `OMPI_COMM_WORLD_SIZE`
- Slurm: `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_JOB_ID`

CLI and Python API callers can override these values explicitly.

## Python API

Use the public conversion, validation, and loading helpers in `stormlog.telemetry`:

```python
from stormlog.telemetry import (
    load_telemetry_events,
    load_telemetry_sessions,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)
```

- `load_telemetry_sessions(path, permissive_legacy=True, events_key=None)`
- `load_telemetry_events(path, permissive_legacy=True, events_key=None, session_id=None)`
- `telemetry_event_from_record(record, permissive_legacy=True, ...)`
- `validate_telemetry_record(record)`

These APIs normalize legacy records to canonical `schema_version: 3` events and
enforce required fields.

## Append-only sink layout

Always-on `track` sessions can also write append-only telemetry into a sink
directory during the run:

```text
telemetry_sink/
  manifest.json
  rollups.json
  segment-000001.jsonl
  segment-000002.jsonl
```

- each JSONL line is one canonical telemetry record
- `manifest.json` is schema `v2` and tracks segment ordering, per-segment `session_id`, and a session ledger
- `rollups.json` is schema `v1` and stores compact derived summaries for fast browsing
- closed segments may be pruned when file-count or total-size retention limits are hit
- on recovery, previously running sessions are closed as `interrupted` and new writes start in a new session

## Compact rollup sidecars

`rollups.json` is a derived sidecar for append-only sink directories. It is safe
to delete and rebuild because raw JSONL segments and `manifest.json` remain the
durable evidence. The sidecar schema is documented in
`docs/schemas/telemetry_rollup_v1.schema.json`.

Stormlog precomputes only low-cardinality summaries that are broadly useful for
long-running monitor sessions:

- session and rank peak allocator/device counters
- per-rank hidden-gap first/latest/peak/delta/drift summaries
- alert counts by severity and event type
- collector degraded/recovered transition counts and degraded-time estimates
- OOM marker counts promoted from telemetry metadata
- fixed-duration session windows (60s by default) with sample/event counts and peak counters

Rollups are written after forced sink flush and terminal manifest persistence on
`close()`. Sink recovery marks previously running sessions `interrupted` and
rewrites the manifest, but it does not replay retained JSONL during construction;
readers safely fall back to raw events when the sidecar is missing or stale. They
are not computed on the append hot path or periodic flush path. Future
post-processing commands can call the same builder to regenerate sidecars for
existing sink directories.

Rollup coverage is explicit. The sidecar records retained segment filenames,
retained event counts, retained bytes, and sink prune diagnostics when available.
If retention has removed older closed segments, the rollup describes only the raw
segments still present in the sink. Interrupted and incomplete sessions keep
their lifecycle status in each embedded session summary.

Consumers must treat rollups as an optimization:

- use a current rollup for overview tables, session comparison, and built-in
  summaries that can be answered exactly from the sidecar
- fall back to raw events for exact timelines, diagnostics drilldown, stale or
  malformed sidecars, unsupported filters, and window/rank views that need more
  detail than the sidecar carries

The rebuild cost is `O(events)` CPU because the builder scans retained raw
events once. Storage is `O(sessions + ranks + windows)`. A 24-hour session at
the default 60-second window size produces about 1,440 window entries before
session and rank metadata. This is intentionally much smaller than replaying
every raw sample for browsing.

Follow-on work:

- expand TUI overview panels to prefer fresh rollups before raw replay
- add a post-processing CLI for rebuilding rollups on existing sinks
- benchmark rollup rebuild cost against large retained sink directories
- evaluate optional DuckDB or persistent indexing only after real directory
  sizes justify it
- keep grouped issue state in a separate future sidecar instead of mixing it
  into raw telemetry or rollups

`load_telemetry_events()` can read:

- a normal JSON export
- a sink directory
- `manifest.json`
- an individual JSONL segment

If the final JSONL line is truncated because the process was interrupted during a
write, the loader ignores that incomplete tail and still returns the fully
written records ahead of it. If the artifacts cannot prove recovered ownership,
the loader classifies the older capture as `incomplete`.

## Diagnose and OOM manifests

Standalone `diagnose` bundles now write manifest schema `v2` and include:

- `session_id`
- `session_status`
- `session`

OOM flight-recorder bundles now write manifest schema `v2` and metadata that
reference the owning tracking `session_id` directly. That makes it possible to
tie a bundle back to the exact capture that emitted the OOM.

## Reconstructing a capture

```python
from stormlog.telemetry import load_telemetry_events, load_telemetry_sessions

sessions = load_telemetry_sessions("./live_sink")
selected = sessions[0]
events = load_telemetry_events(
    "./live_sink",
    session_id=selected.summary.session_id,
)

print(selected.summary.session_id)
print(selected.summary.status)
print(len(events))
```
