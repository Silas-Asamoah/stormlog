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

An activity whose `metadata.intervals` lists `[offset_ns, duration_ns]` busy
intervals inside its span contributes those intervals, not the whole span; see
[Record detail](#record-detail). Such a record is `schema_version: 3` (see
[Compatibility](#compatibility)), and malformed intervals are rejected when
the record is built or read rather than measured as the span or as nothing.
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
The join places their timestamps on the client clock through the same
`infer.clock_alignment` records and `align_timestamp` described here.
See [Inference Profiling](inference.md#optional-server-telemetry) for the
identity, route, and clock requirements for a case-window memory observation.
`load_inference_artifact` reads a stream containing both v1 and v2 records.
It returns v1 records as `LegacyInferenceRecord`, preserving their original
fields; it does not invent a server iteration, host clock, or GPU attribution.
The existing `analyze_inference_events` path still analyzes v1 request records.
Unsupported schema versions fail explicitly instead of being interpreted as
v1. New fields that change record meaning require a new schema version.

`infer.activity_ref` records that carry `metadata.intervals` are written with
`schema_version: 3`, because the intervals change what the span means: the
device is busy inside them, not from `start_ns` to `end_ns`. A reader that
knows only v2 rejects such a record with `unsupported inference
schema_version: 3` instead of counting the idle gaps inside a launch as busy
time. Every other record stays v2, and a v3 record of any other type is
rejected. A v3 activity record must carry valid intervals (a non-empty,
sorted, disjoint list of `[offset_ns, duration_ns]` pairs inside its span) and
a v2 activity record must carry none; the loader rejects both mismatches.

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

## Importing profiler traces

`stormlog infer import-trace` reads PyTorch profiler (Kineto) Chrome traces,
including the `rank*.pt.trace.json.gz` files vLLM writes after
`/start_profile` and `/stop_profile`, and appends their GPU work to an existing
inference artifact as `infer.activity_ref` records. The artifact must contain
an `infer.artifact` record, which supplies the run and session. Each trace is
registered by reference in the run envelope. A trace this artifact already
imported from the same file is skipped, so running the command twice does not
duplicate records; a trace that was only registered, for example over a size
bound, can still be imported.

```bash
stormlog infer import-trace infer.jsonl rank0.pt.trace.json.gz \
  --device-uuid 0=GPU-6d1f0c5e-... --detail launch
```

### Linking GPU work to iterations

A CUDA kernel, copy, or memset carries the correlation ID of the CPU runtime or
driver call that launched it; a CUDA graph launch is one call and many GPU
events. An engine marks each iteration by wrapping the code that launches its
GPU work in a profiler range named
`stormlog.iteration/<producer_id>/<iteration_id>`:

```python
from stormlog.infer.trace_ranges import iteration_range

with iteration_range("my-engine", str(step)):
    run_one_step()
```

The importer finds each GPU event's launch call, then the iteration range that
contains that call on the same thread. It compares CPU timestamps on one thread
only, never GPU timestamps against CPU ones. A linked event gets
`iteration_ref` and `attribution_status: linked`. Anything else stays
`unresolved`, with `metadata.unresolved_reason`:

| Reason | Meaning |
| --- | --- |
| `no_launch_record` | The trace has no CPU call with the event's correlation ID. |
| `launch_outside_iteration_range` | The launch call is not inside any iteration range on its thread, for example work done between iterations. |
| `ambiguous_iteration_range` | More than one iteration range on that thread contains the launch call. |

A trace without iteration ranges imports with every GPU activity unresolved.
That is the expected result for an engine that does not emit them yet.

### What is recorded

- **GPU work only.** Kernels, copies, and memsets become GPU activities. Launch
  calls are CPU work and are never counted as GPU time.
- **Identity.** CUDA correlation ID, stream, CUDA graph ID, rank and world size,
  and the CUPTI version. The engine version is recorded when the trace carries
  it; vLLM 0.30.0 stamps it only for scheduled profiles, not for traces taken
  with `/start_profile` and `/stop_profile`.
- **Device.** Pass `--device-uuid INDEX=UUID` for each device. The index is the
  CUDA device ordinal inside the traced process, after `CUDA_VISIBLE_DEVICES`,
  which is not necessarily the host's NVML index. Two processes can both call
  their GPU device 0, so `--device-uuid TRACE_FILE:INDEX=UUID` scopes a UUID to
  one trace. `TRACE_FILE` is the trace's path as passed to the command, or its
  file name when only one imported trace has that name; a prefix that names no
  trace, or a file name that several traces share, is refused before anything
  is imported. An unscoped ordinal that traces from different processes (by
  host name, rank, and launching pid) use is refused rather than applied to
  both. That check cannot tell apart two containers that report the same host
  name, rank, and pid, so give per-trace UUIDs whenever traces come from
  separate containers or hosts, or from vLLM's Ray backend, which gives each
  worker its own `CUDA_VISIBLE_DEVICES`. Without a UUID, GPU activities are
  kept but stay unmeasured: they never enter a GPU time total under a guessed
  identity.
- **Attachment.** Each trace is registered as `kineto:<file name>:<digest>`,
  where the digest comes from the resolved path, so traces with the same file
  name from different directories stay distinct.
- **Clock.** Timestamps are Kineto's host-calibrated device times, in a clock
  domain scoped to the host and the trace. No alignment to other clocks is
  implied.
- **Event loss.** Kineto traces do not report dropped CUPTI records, so the
  import summary reports loss as unknown (`null`), not zero.
- **Summary.** The trace collector's `infer.capabilities` record carries a
  `summary`: GPU events, records written, linked and unresolved counts by
  reason, CUDA-graph events, and per device the exact busy time, the busy time
  covered by the records, and the summed event time.

### Record detail

`--detail launch` (the default) writes one record per launch call, device, and
activity kind; a CUDA graph launch that ran kernels and a memset gives two
records. The record spans the launch's first GPU event to its last. When the launch has several
events, `metadata.intervals` lists the exact busy intervals inside that span as
`[offset_ns, duration_ns]` pairs from `start_ns`, and `metadata.busy_ns` their
total; such a record is written as `schema_version: 3`, while a single-event
launch record and every `--detail kernel` record stay v2 (see
[Compatibility](#compatibility)). `account_gpu_time` unions those intervals instead of the span, so idle
gaps inside a CUDA graph replay are not counted as busy and the device total is
the same as with one record per event. `--detail kernel` writes one record per
GPU event. Both keep every event's link.

The accounting's other two numbers follow the records. At launch detail,
`summed_activity_ns` adds each record's busy intervals, so overlap between
streams inside one launch is already merged, and `activity_count` counts
records, not GPU events. The per-event sum is in each record's
`metadata.summed_duration_ns` and in the import summary's `summed_ns`. Busy
time is the same at both details.

| Trace | GPU events | Launch records | Span time beyond busy time |
| --- | ---: | ---: | ---: |
| vLLM 0.30.0, Qwen2.5-7B, 64 requests, 30 s | 218,012 | 35,380 | 0.015% |
| vLLM 0.30.0, Qwen2.5-0.5B, 64 requests, 3.5 s | 189,918 | 31,843 | 0.19% |
| Two overlapping streams + a CUDA graph, 200 steps | 3,601 | 1,601 | 1.1% |

The last column is what a consumer that ignored `metadata.intervals` and used
the spans would overcount. The import summary reports `busy_ns` and
`launch_span_ns` per device. Each record is about 1.3 KB of JSONL, most of it
the repeated context, so a 30-second vLLM trace at kernel detail adds roughly
280 MB to the artifact and about 45 MB at launch detail. Capture short windows.

`examples/scenarios/trace_import_scenario.py` checks the import on a CUDA
device. On an NVIDIA A30 it linked every GPU event launched inside an
iteration (3,600 of 3,601). The one copy made after the loop stayed
`launch_outside_iteration_range`. It joined all 2,400 graph-replay events
through their launch calls, counted the two streams' overlap once (busy
305.8 ms against 318.3 ms summed), and produced identical results with the
profiler on and off.

### Nsight Systems reports

`import-trace` also reads Nsight Systems reports exported to SQLite. The same
linking, record detail, and accounting apply:

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o run python serve.py
nsys export --type sqlite run.nsys-rep
stormlog infer import-trace infer.jsonl run.sqlite
```

- **Iteration ranges are NVTX ranges** with the same
  `stormlog.iteration/<producer_id>/<iteration_id>` name. Emit them with
  `iteration_range(..., nvtx=True)`; a `record_function` range alone is not
  visible to Nsight Systems.
- **Record CUDA graphs per node** with `--cuda-graph-trace=node`. At the default
  graph-level tracing Nsight Systems writes one row per graph launch instead of
  its kernels. Those rows are not imported; the summary counts them under
  `not_imported`, and the import says to re-record.
- **Several processes in one report** are handled: launches are matched to GPU
  work by process and correlation ID, and each device is reported per process
  (`"<pid>/<device>"`).
- **GPUs are named from the report when it can name them.** A GPU event's
  device is the CUDA device number inside its process, after
  `CUDA_VISIBLE_DEVICES`, and nsys numbers GPUs in its own order, not
  `nvidia-smi`'s. Exports from nsys 2025.1 and later map each process's device
  numbers to GPUs, so renumbering needs no `--device-uuid`. Older exporters,
  such as nsys 2024.4's, do not write that mapping. A report from one of them
  that lists one GPU is still named, since nsys lists every GPU on the host.
  With several GPUs, the devices stay unmeasured and the import says to
  re-export the report with a newer nsys, which adds the mapping.
- **`--device-uuid` only fills gaps.** It is used for a device the report does
  not name. A UUID that contradicts the report's is refused. So is one ordinal
  given for several processes in a report, since `TRACE_FILE:` cannot pick one
  process inside a report.
- **A `.nsys-rep` file is registered, not read.** Its format is not public, so
  the import notes that it must be exported to SQLite first.
- **Event loss is unknown** (`null`), as for Kineto traces: the exported tables
  this reader uses do not report dropped CUPTI records.
- **Attachment.** The report is registered as `nsys:<file name>:<digest>`.

Run the scenario with `--external` to get a workload with NVTX iteration ranges
and no PyTorch profiler, since two CUPTI clients cannot run at once. On an
NVIDIA A30, exported with nsys 2024.4 (200 steps after 20 warmup steps, each
in its own iteration range), the import linked 3,960 of 3,979 GPU events to
the 220 iterations, including all 2,640 graph-replay events, and wrote 1,779
launch records. The other 19 events were setup and teardown work launched
outside any iteration range, such as initializing the inputs and the copy
after the loop. The report listed one GPU, which named the work, and
`account_gpu_time` on the records gave 384.74 ms busy, the same to the
nanosecond as the union of the GPU events, against 400.72 ms summed.

On a host with two A30s, three processes ran under one capture. One saw only
the second GPU, one saw both in reverse order and used its device 1, and one
saw both in the usual order. In each, the kernels' device was the
process's own CUDA device number, and nsys's GPU ids did not follow
`nvidia-smi`'s order. An nsys 2025.6 export mapped all three to the GPU each
process reported for itself, and so did the same nsys 2024.3 report once
re-exported with nsys 2025.6.

The same workload under the PyTorch profiler gave the same structure. In both
captures each of the 200 steps had 12 kernels, all from the graph replay, and
6 memsets, and the two streams overlapped by 4% of the summed time. Median
kernel durations agreed within 3% (the matrix multiply took 246 µs under
Kineto and 251 µs under Nsight), and median busy time per step within 0.6%
(1.535 ms and 1.544 ms). The first 42 Nsight steps were about 1.5× slower in
every kernel on both streams. That fits a GPU still raising its clocks after
20 warmup steps; the Kineto capture ran after five unprofiled repeats.
