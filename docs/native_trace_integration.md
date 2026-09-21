[← Back to main docs](index.md)

# Native Trace Helper Integration

This page documents the portable scaffolding from
[issue #234](https://github.com/Silas-Asamoah/stormlog/issues/234). It implements
the process and artifact boundary recommended by
[issue #118](https://github.com/Silas-Asamoah/stormlog/issues/118), but it does
**not** ship or invoke a native collector.

Hardware qualification remains in
[issue #235](https://github.com/Silas-Asamoah/stormlog/issues/235). Contract
tests and synthetic traces are not evidence that CUPTI, ROCProfiler, a GPU,
driver, runtime, engine, or container configuration works.

## Guarantees

The scaffolding provides these platform-independent guarantees:

- CUPTI and ROCProfiler are optional. Importing Stormlog does not load either
  native library.
- Preflight reports `available`, `unavailable`, or `unsupported`. `available`
  means only that a candidate library was found; the helper must still verify
  its ABI, driver, device, permissions, and requested activities.
- Helper control messages carry an exact protocol version and request identity.
- CPU launch and device execution intervals are separate nullable fields.
- Every normalized record carries a clock domain, provenance, and uncertainty.
- Trace files are bounded, owner-only, and checksum-addressed. The writer drops
  an entire opaque record when it would exceed the limit, so it never publishes
  a locally truncated record as valid.
- Interrupted captures can preserve an owner-only `.partial` artifact.
- Manifests are written atomically and link to raw trace files by safe relative
  paths.
- Only the manifest is registered in `stormlog_attachments.json`. Raw trace
  records do not become `TelemetryEvent v4` memory samples.
- Helper or trace failures use the existing `healthy`, `degraded`, and
  `unhealthy` vocabulary. Dropped records and truncation remain separate,
  explicit loss facts.

These guarantees protect the local artifact boundary. They do not measure
collection overhead or prove the semantic accuracy of native records.

## Contracts

The machine-readable contracts are:

- [`native_helper_protocol_v1.schema.json`](schemas/native_helper_protocol_v1.schema.json)
- [`native_trace_record_v1.schema.json`](schemas/native_trace_record_v1.schema.json)
- [`native_trace_manifest_v1.schema.json`](schemas/native_trace_manifest_v1.schema.json)
- [`stormlog_attachments_v1.schema.json`](schemas/stormlog_attachments_v1.schema.json)

The schemas reject unknown protocol versions, unsafe manifest paths, missing
uncertainty, and unknown required fields. Python constructors additionally
validate interval ordering, requested/enabled activity relationships, health
and loss consistency, and attachment identity conflicts.

Adding optional fields is compatible only when readers can ignore them safely.
Changing field meaning, required behavior, clock interpretation, or lifecycle
semantics requires a new schema or protocol version.

## Capability preflight

```python
from stormlog.native_trace import native_trace_preflight

for capability in native_trace_preflight():
    print(capability.backend, capability.status, capability.reason)
```

Preflight never imports a discovered native library. On unsupported operating
systems it returns `unsupported`. On a supported operating system without a
discoverable library it returns `unavailable`. Finding a library returns
`available` with an explicitly unverified reason.

A future helper must repeat capability detection in its own process before
capture and return the actual toolkit, driver, device, activity, privilege, and
coexistence outcome. Stormlog must not promote library discovery to supported
hardware.

## Helper protocol

`NativeHelperMessage` models newline-delimited JSON control messages. Version 1
allows these message types:

- `hello` and `capabilities` for negotiation;
- `start` and `started` for an explicitly bounded capture;
- `status` for health and loss reporting;
- `flush` for a bounded drain;
- `stop` and `stopped` for orderly shutdown; and
- `error` for an attributable failure response.

Every response must retain the request ID it answers. A future launcher must
pass an executable and argument vector directly to the operating system. It
must never construct a shell command from user-controlled values. The current
scaffolding intentionally does not launch a binary.

Each message type has required payload fields in the protocol schema and in the
Python parser. `NativeHelperOutcome` maps completed, partial, cancelled, timed
out, failed, and incompatible termination into collector health. Cancellation
and readable partial output are degraded; timeout, crash, and incompatibility
are unhealthy. These outcomes remain data for the optional source and do not
raise through or disable the default monitor.

## Record semantics

`NativeTraceRecord` is a normalized evidence record, not a memory telemetry
event. It supports:

- independent CPU and device intervals;
- explicit source clock domain;
- correlation, stream, graph, and graph-node identifiers;
- provenance and uncertainty; and
- backend-specific metadata that does not change the common field meanings.

An interval must contain both its start and end and its end cannot precede its
start. A record may contain a CPU interval, a device interval, or both. Importers
must not synthesize a missing device interval from CPU launch duration.

## Bounded artifact flow

```python
from stormlog.collector_health import CollectorHealthState
from stormlog.native_trace_store import (
    BoundedTraceWriter,
    NativeTraceIdentity,
    NativeTraceManifest,
    register_native_trace_attachment,
    write_native_trace_manifest,
)

capture_dir = "artifacts/native/capture-1"
writer = BoundedTraceWriter(
    capture_dir,
    "raw/activity.ndjson",
    max_bytes=64 * 1024 * 1024,
)

# A future helper adapter supplies complete bytes or NativeTraceRecord values.
# The example deliberately does not pretend to collect GPU evidence.
writer.write(b'{"external_native_record":true}\n')
artifact = writer.finalize(content_type="application/x-ndjson")
loss = writer.loss(flush_outcome="complete")

manifest = NativeTraceManifest(
    capture_id="capture-1",
    backend="cupti_activity",
    identity=NativeTraceIdentity(session_id="session-1", pid=1234),
    helper_executable="stormlog-cupti-helper",
    helper_version="0.1.0",
    started_ns=1,
    ended_ns=2,
    clock_domains=("host/monotonic_ns", "gpu-0/device_ns"),
    requested_activities=("runtime", "kernel"),
    enabled_activities=("runtime", "kernel"),
    max_bytes=64 * 1024 * 1024,
    privilege="same-process",
    target_selector="pid:1234",
    health=CollectorHealthState(),
    loss=loss,
    artifacts=(artifact,),
)
manifest_path = write_native_trace_manifest(capture_dir, manifest)
register_native_trace_attachment(capture_dir, manifest_path, manifest)
```

When a record does not fit, `write` returns `False` and increments dropped
record and byte counters. The final manifest must then use partial health and a
non-complete flush outcome. When a helper crashes or is cancelled, call
`preserve_partial` and publish degraded or unhealthy manifest state only if the
remaining metadata is trustworthy.

## Security boundary

- Artifact paths must be POSIX-style relative paths without `..`, absolute
  roots, Windows drive roots, or backslashes.
- Resolved paths must stay beneath the capture directory, including through
  existing symlinks.
- Capture directories use owner-only permissions and files use mode `0600`.
- Temporary manifests are flushed and atomically replaced in the destination
  directory.
- Checksums are streamed so a bounded but large trace is not loaded fully into
  Python memory.
- Manifests declare sensitive field classes such as symbols, addresses, stacks,
  paths, or user data.
- A future launcher must record the helper executable identity, privilege, and
  target selector and must reject ambiguous process scope.

The helper remains a separate trust boundary. Packaging provenance, binary
signing, native sandboxing, injection behavior, and runtime privileges require
review with the actual implementation.

## Synthetic fixtures

The deterministic fixture in
`tests/fixtures/native_trace/overlap_graph_records.jsonl` exercises:

- distinct CPU and device clock domains;
- overlapping device intervals on two streams;
- graph and graph-node identity;
- repeated graph execution; and
- explicit synthetic provenance and uncertainty.

It validates parsing, schema, storage, correlation fields, and loss handling.
It does not validate GPU timestamps, runtime correlation, graph semantics,
kernel coverage, event loss, or overhead. Those claims require #235.

## Integration with inference correlation

[PR #233](https://github.com/Silas-Asamoah/stormlog/pull/233) defines the
request, iteration, shared-execution, capability, and activity-reference
contracts. Native import should adapt completed manifests and normalized records
to that contract only when scoped correlation IDs and identity make the join
unambiguous. Timestamp proximity alone is insufficient.

This module does not duplicate #233's request accounting and remains mergeable
whether #233 or the native-probe research lands first.
