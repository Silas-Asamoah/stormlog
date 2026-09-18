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
| `infer.request` | A logical request observation, with optional attempt and backend request references. |
| `infer.iteration` | One execution iteration, recorded once even when many requests participate. An optional batch reference can group iterations. |
| `infer.stage` | A named span attached to a request, an iteration, or both. Names are generic; token and KV fields are optional. |
| `infer.membership` | A request's participation in an iteration, with its own role (for example, prefill or decode). |
| `infer.activity_ref` | A trace activity with optional iteration and trace attachment links, plus available runtime/CUDA, stream, and graph IDs. |
| `infer.clock_alignment` | An offset and uncertainty between two clock domains. |

An iteration can mix prefill for one request and decode for another. The
`infer.membership.role` records each request's role; it does not assert a
separately measured duration or grant the request ownership of the full batch.
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

## Compatibility

The existing profiler continues to write v1 client observations unchanged.
`load_inference_artifact` reads a stream containing both v1 and v2 records.
It returns v1 records as `LegacyInferenceRecord`, preserving their original
fields; it does not invent a server iteration, host clock, or GPU attribution.
The existing `analyze_inference_events` path still analyzes v1 request records.
Unsupported schema versions fail explicitly instead of being interpreted as
v1. New fields that change record meaning require a new schema version.

Raw profiler traces are linked by `trace_attachment_id` through the existing
run envelope attachment catalog. The activity record is a pointer and
correlation fact, not a copy of the raw trace.
