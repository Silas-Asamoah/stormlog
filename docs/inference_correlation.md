# Inference execution correlation

Stormlog's endpoint profiler already emits v1 `infer.request` records. Those
records describe **client-observed** latency, response chunks, and token counts.
They do not prove which server iteration or GPU activity served a request. The
v2 records in `stormlog.infer.correlation_events` add server-side evidence to
the same JSONL stream. The schema is
[`inference_correlation_v2.schema.json`](schemas/inference_correlation_v2.schema.json).

## Identity and relationships

Every v2 event has a `run_id`, `session_id`, `producer_id`, and `event_id`. An
event ID is unique within its producer and session; consumers deduplicate on
that tuple. The run and session refer to the existing run envelope and capture
session, not a new artifact namespace.

Entity references have `{"producer_id": "...", "id": "..."}`. Backend request,
iteration, batch, stage, and activity IDs are therefore scoped to the producer
that issued them. A logical request, its attempts, and the events that report
them are separate identities. Adapters must emit the same logical request
reference in membership records when they can correlate it. Similar ID strings
from different producers do not establish a join.

| v2 event | Meaning |
| --- | --- |
| `infer.artifact` | Run and session identity for the JSONL artifact, including its producing Stormlog version. |
| `infer.request` | A logical request observation, with optional attempt and backend request references. |
| `infer.iteration` | One execution iteration, recorded once even when many requests participate. An optional batch reference can group iterations. |
| `infer.stage` | A named span attached to a request, an iteration, or both. Names are generic; token and KV fields are optional. |
| `infer.membership` | A request's participation in an iteration, with its own role (for example, prefill or decode). |
| `infer.activity_ref` | A trace activity with optional iteration and trace attachment links, plus available runtime/CUDA, stream, and graph IDs. |
| `infer.capabilities` | Whether an optional source was present, what it supported and enabled, and what it actually collected. |
| `infer.clock_alignment` | An offset and uncertainty between two clock domains. |

An iteration can mix prefill for one request and decode for another. The
`infer.membership.role` records each request's role; it does not assert a
separately measured duration or grant the request ownership of the full batch.
An activity's `activity_domain` is `gpu`, `runtime`, `cpu`, or `unknown`. Only
explicitly classified GPU activities enter GPU time calculations.
An activity with no proved iteration link uses `attribution_status: unresolved`
and `iteration_ref: null`. Optional numeric fields use `null` for unavailable
data, never zero as a substitute.

## Provenance and clocks

The context records source and version, engine/backend and versions, host and
process, device UUID, rank dimensions, clock domain, collection mode, and
whether the event was observed, reported, or estimated. Unknown optional
identity values remain `null`. A monotonic `start_ns`/`end_ns` pair defines a
local elapsed duration. Wall and device timestamps do not become local
durations merely because they are numbers in nanoseconds.

Clock domains must identify comparable timestamps, for example a monotonic
clock on one host and boot. Cross-host ordering requires a clock alignment
event: add its `offset_ns` to a timestamp in `from_clock_domain` to express it
in `to_clock_domain`, carrying `uncertainty_ns` with the result. An alignment
may have a validity range. Without a valid alignment, cross-domain subtraction
is undefined.

## GPU time and request shares

`resolve_inference_events` deduplicates event identities and physical entity
references before resolving request, iteration, stage, and activity links. It
accepts out-of-order delivery and reports links whose targets never arrived.
Conflicting duplicates fail explicitly. Legacy v1 observations remain separate
from server evidence.

`account_gpu_time` reports three distinct quantities:

- **Iteration elapsed time:** the local monotonic interval of an iteration,
  when both endpoints were recorded.
- **Summed activity duration:** the sum of complete GPU activity spans. This
  can exceed elapsed time when streams overlap.
- **Observed GPU activity union:** the length of the merged activity intervals,
  grouped by device UUID, clock domain, and clock kind. For `[0, 8)` and
  `[2, 10)`, the sum is 16 units and the union is 10. This is a union of
  observed trace intervals, not a hardware utilization measurement.

An activity without a device UUID, complete span, or usable clock domain stays
unmeasured. Unlinked activities can still contribute to the device total when
their intervals are valid, but not to an iteration total. Totals from different
devices or clock domains are not silently added. Summing per-iteration unions
can also double-count intervals that overlap across iterations; use the run's
device total for that question.

There is no automatically measured per-request GPU duration. A caller may
submit `RequestShareEstimate` values with a named model. For a chosen
iteration/device/clock budget, `validate_request_shares` requires the
estimated shares plus an explicit unattributed remainder to equal the measured
union. Shares in one budget must use the same named estimation model.
Membership proves participation; it does not itself choose a share.

## Compatibility

New endpoint profiles write one v2 `infer.artifact` record with a generated
`run_id` alongside their unchanged v1 client observations. The `run_id` is
also available as `InferenceProfiler.run_id` when a caller needs to coordinate
optional capture. Existing v1-only files remain valid.
Pass `--run-id` to `stormlog infer profile` when coordinating a separate
on-host server telemetry artifact. The scoped `infer.telemetry_sample` records
are ingested by `stormlog infer analyze --server-telemetry`; they are separate
from the v2 execution events and do not establish request-to-iteration links.
See [Inference Profiling](inference.md#optional-server-telemetry) for the
identity, route, and clock requirements for a case-window memory observation.
`load_inference_artifact` reads a stream containing both v1 and v2 records.
It returns v1 records as `LegacyInferenceRecord`, preserving their original
fields; it does not invent a server iteration, host clock, or GPU attribution.
The existing `analyze_inference_events` path still analyzes v1 request records.
Unsupported schema versions fail explicitly instead of being interpreted as
v1. New fields that change record meaning require a new schema version.

Raw profiler traces are linked by `trace_attachment_id` through the existing
run envelope attachment catalog. The activity record is a pointer and
correlation fact, not a copy of the raw trace.

## Optional collection and run catalog

`EngineAdapter` and `TraceCollector` are separate optional protocols. An engine
adapter returns server events; a trace collector returns activity events and
raw trace attachments. `append_inference_capture` validates their run/session
identity, appends the v2 records to an existing inference JSONL artifact, and
registers that artifact and any traces in the existing `stormlog_run.json`
envelope. It preserves a compatible existing envelope and rejects a conflicting
run ID. An absent source produces an `infer.capabilities` record with
`available: false`; supported, enabled, and successfully collected capabilities
are separate lists.

Automatic joining uses a shared `correlation_scope`, the host, process, device,
session, and a runtime or CUDA correlation ID. A stream or graph ID is retained
as evidence but is not enough by itself to prove a request-to-activity link.
An ambiguous or unscoped match remains unresolved. Adapters may also provide
an explicit iteration link when they can prove it.
