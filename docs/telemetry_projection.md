[← Back to main docs](index.md)

# Stormlog Telemetry Projection

Stormlog uses a telemetry-first internal projection to give runtime capture,
append-only sink persistence, artifact loading, live display, and offline
analysis one shared event model.

`TelemetryEvent v4` remains Stormlog's canonical persisted event schema for
artifacts, append-only sink segments, exports, and loader output. The
backend-neutral envelope described here is an internal projection over that
schema, implemented in `stormlog.telemetry_model` and exposed through
`stormlog.telemetry`.

## Decision

Stormlog should use a telemetry-first internal projection as the shared
internal model for analysis and UI code. Runtime trackers should keep emitting
compact tracker-local events on the hot path, then normalize at tracker or
loader edges before persistence, display, or analysis fan-out.

The current implementation takes the first migration step: it keeps persisted
records on `TelemetryEvent v4`, then projects those records into
`ProjectedTelemetryRecord` for live and loaded sessions. Future sink migrations
can persist the projected envelope directly after benchmark gates prove the
runtime cost is acceptable.

## Current State Map

Stormlog currently has several event shapes and normalization boundaries:

- `TelemetryEventV4` is the stable memory telemetry contract used by artifact
  exports and loader output.
- Tracker `TrackingEvent` values stay lightweight for runtime capture and are
  normalized through `_telemetry_record_from_event(...)`.
- TensorFlow capture builds compatible telemetry dictionaries in the
  TensorFlow tracker path before exporting or loading them.
- Append-only sink JSONL entries persist normalized telemetry records and keep
  recovery, rollover, pruning, and manifest behavior separate from the record
  projection.
- Artifact loaders group loaded records into `LoadedTelemetrySession` objects
  after legacy V2, V3, JSON, and JSONL compatibility handling.
- The live TUI uses `TrackerSession` as the display-facing session model.

The projected telemetry envelope does not remove those boundaries yet. It gives
them a shared target shape so migration can happen one layer at a time without
changing the existing artifact schema.

## Implemented Model

The projected telemetry envelope is `ProjectedTelemetryRecord`. It is a small
immutable event envelope with these fields:

- `schema_version`: internal projected telemetry envelope version.
- `record_id`: deterministic identifier derived from the source telemetry
  record.
- `timestamp_ns`: event time from the source.
- `observed_timestamp_ns`: time Stormlog observed or normalized the event.
- `session_id`: capture/session identity.
- `source_kind`: backend family such as `cpu`, `cuda`, `rocm`, `mps`,
  `tensorflow`, generic `gpu`, or `other`.
- `event_type`: generic classification such as `sample`, `start`, `stop`,
  `phase_enter`, `phase_exit`, `warning`, `critical`, or `error`.
- `stage`: optional lifecycle or workload stage.
- `severity` and `severity_text`: normalized severity when meaningful.
- `body`: primary message or payload.
- `resource`: runtime identity such as host, process, backend, device, job, and
  rank.
- `attributes`: extensible metadata and backend-specific measurements.
- `correlation`: session, job, rank, phase, and future trace/span alignment
  fields.

The projection keeps backend-specific details out of top-level fields. Memory
counters from `TelemetryEvent v4`, collector health metadata, phase metadata,
and future backend details are represented as attributes, resources, or
correlation fields.

The smallest useful contract for runtime, sink, loader, TUI, and exported
artifact views is:

- identity: `schema_version`, `record_id`, and `session_id`,
- time: `timestamp_ns` and `observed_timestamp_ns`,
- classification: `source_kind`, `event_type`, `stage`, and severity fields,
- payload: `body`,
- projection data: `resource`, `attributes`, and `correlation`.

`source_kind` must stay a backend family. Distributed identity such as host,
process id, device id, rank, local rank, world size, and future node ids belongs
in `resource` or `correlation`, not in `source_kind`. Future backend families
can be added when Stormlog adds collector support for them.

Some identity values intentionally appear in more than one place. For example,
`session_id` is top-level and also available in `correlation`, while
`source_kind` is top-level and also available in `resource`. These copies are
derived from the same normalized source record and must not drift; adapters that
change one identity value must update every projected copy in the same
projection step.

## Projection Versioning

`TELEMETRY_PROJECTION_SCHEMA_VERSION` versions only the internal projected
envelope. It is separate from `TelemetryEvent v4` and does not change artifact
compatibility by itself.

When the projection shape changes incompatibly, update the version constant,
the `ProjectedTelemetryRecord.schema_version` type annotation, serialization
tests, this document, and compatibility behavior together. Compatibility code
should be explicit about whether it can read both projection versions or must
re-project from the persisted `TelemetryEvent v4` source.

## Data Flow

The current flow is:

1. Trackers capture compact runtime events or tracker-local records.
2. Existing normalizers produce `TelemetryEvent v4` records for exports, sink
   writes, loaders, and TUI adapters.
3. `project_telemetry_event(...)` projects V3 records into
   `ProjectedTelemetryRecord`.
4. `LoadedTelemetrySession.telemetry_records()` exposes projected telemetry
   records for loaded artifacts.
5. `TrackerSession.telemetry_records()` exposes projected telemetry records for
   live TUI sessions.

Because projection currently happens after existing V3 normalization, this keeps
the capture path unchanged while giving downstream code one stable
backend-neutral view. A future sink migration that persists the projected
envelope directly still needs benchmark gates for capture latency, allocations,
queue depth, and sink throughput before it can claim the same hot-path cost.

## Compatibility

Stormlog preserves existing compatibility boundaries:

- Existing V2, V3, and legacy artifacts remain loadable.
- `TelemetryEvent v4` remains the persisted artifact and append-only sink
  record format.
- Legacy artifact upcasting stays in loader and normalizer code.
- Telemetry projection is additive and does not change CLI, TUI, or Python API
  behavior.
- Legacy export shapes remain explicit compatibility paths.

This lets analysis and UI code adopt the projected telemetry envelope without
breaking older artifacts or changing the on-disk schema.

## Live and Loaded Sessions

Live and loaded data now share the same projected telemetry record shape:

- Loaded artifacts use `LoadedTelemetrySession.telemetry_records()`.
- Loaded artifacts use `LoadedTelemetrySession.resources()` for unique observed
  resources.
- Loaded artifacts use `LoadedTelemetrySession.correlations()` for unique
  correlation contexts.
- Live TUI sessions use `TrackerSession.telemetry_records()`.

The TUI can keep rendering lightweight view models, while analysis and future
query code can use the projected telemetry records regardless of whether the
source is a live tracker or an artifact.

## Performance Policy

Always-on capture protects application liveness first. The hot path should do
only the work required to capture local fields, timestamp events, look up
session/correlation identity, and append to bounded in-memory history or a
queue.

Capture code should avoid:

- blocking on sink persistence,
- heap-heavy serialization,
- reflection-heavy shaping,
- formatting strings before normalization when raw fields are enough.

Sink persistence should remain batched and append-only. Under pressure,
Stormlog should prefer explicit sampling or low-priority event shedding over
blocking the application being measured.

Pressure means any bounded buffer, queue, history, sink, or long-running session
cannot accept events at the current production rate without unbounded memory
growth or blocking the measured application. Examples include bounded history
overflow, sink queue backlog, slow disk flushes, bursty alert streams, and
always-on sessions that run longer than the retention budget.

Any sampling or shedding policy must be visible in telemetry and diagnostics.
At minimum, adapters and sinks should expose the active policy, queue or history
depth, dropped sample count, dropped event count, dropped alert count when
alerts are handled separately, and a reason for each class of drop.

## Benchmark Plan

Benchmark validation should compare the existing V3 flow against the projection
path before any persisted-format migration.

Measure:

- events per second sustained,
- average and p95 event capture latency,
- allocations per event,
- bytes allocated per second,
- queue or bounded-history depth under burst load,
- sink flush throughput,
- TUI update latency in live mode,
- artifact load time for large sessions,
- memory growth over long always-on runs.

Coverage should include:

- CPU-only workloads,
- GPU-heavy workloads,
- mixed backend workloads,
- quiet long-running always-on sessions,
- bursty error-heavy sessions.

The acceptance bar is that the projected telemetry envelope must not materially
harm the hot path. Any extra projection cost must be offset by simpler shared
sinks, loaders, UI adapters, and analysis code.

## Migration Plan

### Phase 1: Telemetry Projection

- Keep `TelemetryEvent v4` as the persisted artifact format.
- Project V3 records into `ProjectedTelemetryRecord`.
- Expose projected telemetry records from loaded sessions and live TUI sessions.
- Cover projection behavior with deterministic tests.

### Phase 2: Tracker Adapters

- Keep tracker-local runtime events compact.
- Move PyTorch, CPU, and TensorFlow normalization into shared adapter helpers.
- Centralize timestamp rules, session identity, severity mapping, backend tags,
  collector health, and distributed correlation.
- Add capture and enqueue counters for benchmark visibility.

### Phase 3: Sink Migration

- Add a future sink schema version for persisted projected telemetry records.
- Keep append-only JSONL semantics and deterministic serialization.
- Preserve rollover, pruning, recovery, and manifest behavior.
- Keep old sink loading paths for existing artifacts.

### Phase 4: Loader Migration

- Dispatch by artifact and sink schema version.
- Parse V2, V3, legacy JSON, JSONL, and future persisted projected telemetry
  records into the same internal stream.
- Keep compatibility transforms separate from primary parsing.
- Preserve old fixtures through loader adapters instead of rewriting user data.

### Phase 5: TUI and Session Unification

- Render live monitoring and diagnostics from projected telemetry records or
  derived view models built from projected telemetry records.
- Keep TUI-specific formatting outside the core telemetry model.
- Reuse the same query and marker logic for live and offline sessions.

### Phase 6: Benchmarks and Regression Gates

- Extend the benchmark harness with capture-latency, allocation, queue-depth,
  sink-throughput, load-time, and TUI-latency metrics.
- Compare current V3 normalization against the projected telemetry envelope and
  future persisted-envelope writes.
- Require regression and budget gates before enabling new persisted formats.

## Follow-On Tasks

- Tracker layer: introduce shared runtime-event-to-projection adapter helpers for
  PyTorch, CPU, and TensorFlow paths.
- Sink layer: add projected-record sink versioning while preserving append-only
  recovery and retention behavior.
- Loader layer: add version dispatch that upcasts V2, V3, and future persisted
  projected telemetry records into one internal stream.
- TUI/session layer: move more live and loaded views onto projected telemetry
  records.
- Benchmark layer: extend `examples.cli.benchmark_harness` and docs benchmark
  assets with overhead, allocation, queue-depth, load-time, and TUI-latency
  checks.
- Compatibility layer: add explicit legacy import/export adapters and document
  retirement criteria.

## Non-Goals

- Rewriting every tracker to emit fully normalized records immediately.
- Changing the existing `TelemetryEvent v4` JSON schema.
- Designing a new external protocol.
- Changing user-facing CLI, TUI, or Python API behavior.
- Optimizing for every future backend-specific field up front.
