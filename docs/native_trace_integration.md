[← Back to main docs](index.md)

# Native CUPTI Trace Capture

Stormlog implements an **opt-in, Linux-only CUPTI Activity prototype** for
commands that Stormlog starts. It is disabled by default, is built separately
from the Python package, and adds no mandatory native or Python dependency.
AMD ROCProfiler remains a separate, unsupported backend rather than an implied
equivalent.

The implementation consists of the CUDA injection library in `native/cupti`,
the `stormlog.native_trace_capture` launcher/importer, bounded trace sidecars,
and the `stormlog native-trace` command. It follows the feasibility decision in
[issue #118](https://github.com/Silas-Asamoah/stormlog/issues/118) and the
portable boundary in
[issue #234](https://github.com/Silas-Asamoah/stormlog/issues/234). Hardware
qualification remains in
[issue #235](https://github.com/Silas-Asamoah/stormlog/issues/235). Compiling
the helper and passing synthetic tests are not GPU accuracy or overhead
evidence.

## Build the native component

Prerequisites are Linux, CMake 3.20 or newer, a C++17 compiler, and a CUDA 12+
Toolkit containing CUPTI headers and `libcupti`.

```bash
cmake \
  -S native/cupti \
  -B build/native-cupti \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build/native-cupti --config Release
```

The build produces `libstormlog_cupti_injection.so`. It is deliberately not
included in the Stormlog wheel. CI compiles the source against pinned NVIDIA
CUDA 12.9 runtime, compiler, and CUPTI headers. A portable configuration check
that does not require CUDA is also available:

```bash
cmake \
  -S native/cupti \
  -B build/native-cupti-contract \
  -DSTORMLOG_CUPTI_VALIDATE_ONLY=ON
```

## Capture a command

Use a new capture ID for each run. The argument separator keeps target flags
out of Stormlog's parser.

```bash
stormlog native-trace \
  --injection-library "$PWD/build/native-cupti/libstormlog_cupti_injection.so" \
  --output-dir "$PWD/artifacts/native" \
  --session-id inference-session-1 \
  --capture-id warmup-1 \
  --max-bytes 67108864 \
  --timeout 300 \
  -- python -m my_inference_server --model example/model
```

The default activity set is driver API, runtime API, concurrent kernel,
memcpy, and memset. Repeating `--activity` replaces that set:

```bash
stormlog native-trace \
  --injection-library /opt/stormlog/libstormlog_cupti_injection.so \
  --output-dir ./artifacts/native \
  --session-id session-1 \
  --activity runtime \
  --activity kernel \
  --activity synchronization \
  -- ./serve-model --port 8000
```

Optional `--run-id`, `--job-id`, `--rank`, and `--device-id` values preserve
distributed identity. The command returns the target's exit code. A timeout
returns `124` after terminating the launched process group. Collector failure
does not replace a successful target exit code; inspect `health` and the
manifest for capture completeness.

## Execution and data flow

1. Stormlog verifies Linux support, a regular non-symlink injection library,
   and that the library is not group- or world-writable.
2. It creates an owner-only capture directory and rejects pre-existing
   `CUDA_INJECTION64_PATH` or reserved `STORMLOG_CUPTI_*` settings.
3. It starts the argument vector with `shell=False` and a new process group.
4. CUDA loads the library and calls its exported `InitializeInjection` entry
   point when the target initializes CUDA.
5. The library registers asynchronous CUPTI buffers and enables the requested
   activities. `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` preserves observable
   concurrency; the serializing kernel activity kind is never enabled.
6. Completion callbacks normalize supported records into bounded NDJSON. A
   record that cannot fit is dropped in full and counted.
7. Exit handling force-flushes CUPTI, records CUPTI and local drops, and
   publishes a complete trace only after the final timestamp, file flush,
   close, rename, and directory synchronization succeed.
8. Python validates the bounded status document, PID, request, ownership,
   permissions, link count, and reported trace size. It hashes the trace,
   writes the manifest, and securely registers it in
   `stormlog_attachments.json`.

The machine-readable contracts are:

- [`native_cupti_status_v1.schema.json`](schemas/native_cupti_status_v1.schema.json)
- [`native_trace_record_v1.schema.json`](schemas/native_trace_record_v1.schema.json)
- [`native_trace_manifest_v1.schema.json`](schemas/native_trace_manifest_v1.schema.json)
- [`stormlog_attachments_v1.schema.json`](schemas/stormlog_attachments_v1.schema.json)

High-volume records remain trace sidecars. They are not converted into
`TelemetryEvent v4` memory samples.

## Record semantics

Runtime and driver records populate CPU intervals. Kernels, copies, and
memsets populate device intervals. Correlation IDs are strings so launch and
device records can be joined without numeric-width loss. Kernel records retain
stream, graph, graph-node, device, context, and symbol identity.
Synchronization records retain CUPTI type and stream/context identity.

Native record timestamps use the explicit `cupti_timestamp_ns` clock domain.
Capture bounds use `unix_epoch_ns`. Stormlog does not pretend those domains are
interchangeable or infer a conversion. Records identify `cupti_activity`
provenance and the remaining timestamp uncertainty.

Status metadata records the compiled CUDA and CUPTI API versions and the
loaded CUPTI version. Driver and CUDA Runtime versions are queried from symbols
already loaded in the target and remain explicit `null` values when the target
does not expose those APIs. GPU architecture and authoritative device identity
remain qualification metadata for #235 rather than inferred values.

## Bounds, health, and failure behavior

The producer enforces `--max-bytes` before every complete NDJSON record. It
separately reports delivered records, CUPTI buffer drops, local record and byte
drops, and shutdown/flush completion.

Zero drops and validated final status produce `healthy`. Unsupported requested
activities or dropped data produce `degraded` with partial telemetry. Missing,
malformed, permission-unsafe, PID-mismatched, or unfinalized output produces
`unhealthy`. A failed flush retains `activity.ndjson.partial`; it is never
renamed to a complete trace. The importer independently treats any retained
`.partial` artifact or trace-size disagreement as partial or unhealthy evidence,
even if producer status claims success.

Target crashes, `SIGKILL`, `exec`, and `_exit` can bypass exit flushing. The
importer treats missing final status as unhealthy and preserves only validated
partial evidence. A target's nonzero exit alone does not mean the collector
failed, and healthy collection does not mean the target succeeded.

## Security and operational constraints

- Native injection runs inside the target CUDA process. A defect in the
  library can affect that process; full process-level crash isolation is not
  possible with CUPTI startup injection.
- Stormlog launches an argument vector and never builds a shell command.
- Capture directories use mode `0700`; trace, status, manifest, and attachment
  files use mode `0600`.
- Native output uses directory-relative `openat` with `O_NOFOLLOW`. Import
  rejects symlink components, hard-linked files, unsafe paths, files owned by
  another user, non-owner-only files, and oversized status/attachment metadata.
- Captures can contain symbols, process/thread IDs, graph identity, and workload
  timing. Treat the whole capture directory as sensitive.
- Existing CUDA injection settings are rejected instead of overwritten.
  Containers must mount the library and a compatible CUPTI runtime explicitly.
- No root access is needed when Stormlog launches a process owned by the caller.
  The prototype does not attach to existing or unrelated processes.
- CUPTI and another profiler may compete for resources. Coexistence is not
  assumed; qualify every profiler/runtime combination through #235.

## Current limitations

- Linux startup injection only. Windows CUPTI support is not implemented, and
  macOS has no CUDA path.
- NVIDIA only. No ROCProfiler library or AMD support claim is shipped.
- No late attach, system-wide tracing, eBPF or USDT transport, programmable
  instruction probes, or device metrics.
- CUPTI v1 Activity callbacks preserve CUDA 12 compatibility. A future CUDA
  13.3+ path may adopt subscriber-scoped v2 APIs after compatibility testing.
- Graph identity is recorded for kernels. Graph replay correctness,
  overlapping-stream fidelity, framework correlation coverage, coexistence,
  event loss, and overhead still require the matched hardware protocol in
  #235.
- Abrupt process termination can leave only partial evidence.

These limitations are explicit unsupported or partial states, not silent
support claims.
