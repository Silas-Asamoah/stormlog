[← Back to main docs](index.md)

# Native Probe Primary-Source Review

This review supports the source-backed capability matrix at
`research/native_probes/matrices/source_backed.json`. It records what primary
sources establish and separates those statements from unmeasured Stormlog
claims. Sources were last checked on 23 September 2026.

## Evidence rules

The machine-readable matrix uses these statuses:

- `CONFIRMED_BY_PRIMARY_SOURCE`: directly documented by a vendor, kernel,
  project source repository, or official engine documentation.
- `PAPER_RESULT_NOT_REPRODUCED`: reported by a paper for its evaluated setup,
  but not reproduced by Stormlog.
- `INFERENCE`: a reasoned architecture or operational conclusion.
- `VERSION_DEPENDENT`: behavior explicitly varies with tool, driver, runtime,
  kernel, platform, or configuration.
- `UNKNOWN`: no acceptable evidence or experiment is available.
- `UNSUPPORTED`: the reviewed source explicitly excludes the combination.
- `STORMLOG_VALIDATED`: reserved for completed Stormlog experiments and absent
  from the source-backed matrix.

Published overhead is never relabeled as Stormlog overhead.

## Existing public and engine profiler surfaces

### PyTorch and Kineto

The current [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
describes CPU and supported accelerator activities, scheduled capture, and
Chrome-trace export. The profiler is the required framework baseline because it
can preserve framework operator context that a generic CUDA activity collector
does not create.

Configuration changes evidence quality and cost. Stack capture, shapes,
memory profiling, schedules, and accelerator activities must be recorded in the
trial manifest. A public trace is not automatically complete, and lack of an
observable loss denominator must be represented as `UNKNOWN`, not zero loss.

### vLLM and Triton Proton

Current [vLLM profiling documentation](https://docs.vllm.ai/en/latest/contributing/profiling/)
materially changes the comparison that PR #236 made. The supported profiler
configuration now includes Triton Proton with:

- NVIDIA timing through a CUPTI backend;
- aggregate tree and Chrome-trace output modes;
- optional Triton launch metadata;
- repeated start and stop profiling through vLLM controls;
- CUDA graph attribution when the session observes graph capture;
- Triton 3.7+ phase handling for repeated graph-aware profiles.

The same source states that Proton's current vLLM path does not support ROCm.
PC sampling synchronizes the CUDA context and requires eager execution, so it
is not a faithful serving-timeline substitute.

The required experimental question is not whether a custom collector can emit
CUPTI records. It is whether it supplies material evidence or operating cost
advantages beyond Proton at the intended bounded or continuous operating point.

### Nsight Systems

The current [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
documents CUDA API, kernel, memory, context, stream, graph, and event timelines.
It also documents graph-level versus node-level tracing, hardware versus legacy
software trace, focused capture, and the possibility of losing buffered data
when a target terminates without a flush.

Nsight Systems is the NVIDIA trusted timeline reference. It is not presumed to
be an always-on Stormlog collector. Graph granularity, CUDA trace method,
fallbacks, event tracing, backtraces, and capture boundaries must be immutable
within matched comparisons.

### Nsight Compute

The current [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
documents kernel, application, and range replay; memory save and restore;
cache-control behavior; clock control; software patching; and kernel
serialization. These mechanisms can materially perturb the workload.

Nsight Compute is therefore a detailed-counter or offline diagnostic baseline.
Its duration cannot be treated as an unperturbed serving timeline, and it must
not run concurrently with modes except in the explicit coexistence experiment.

### ROCProfiler and rocprofv3

The current [ROCProfiler SDK documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
and [quick reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html)
document HIP API tracing, kernel dispatches, memory-copy activity, counters, and
buffered services. The current quick guide identifies PC sampling as beta.

AMD qualification is an independent lane. HIP graph behavior, queues, loss,
coexistence, permissions, containers, and engine support must be measured on a
supported AMD host. No NVIDIA result is carried into the AMD matrix.

PyTorch's current
[HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html) explicitly
reuses the `torch.cuda` interfaces on ROCm. AMD's current
[HIP graph documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
documents stream capture, graph instantiation, and replay. The W3 harness must
therefore attempt the public PyTorch graph API on a compatible ROCm build and
retain an exact runtime rejection if unsupported, rather than blanket-skipping
all HIP hosts. ROCProfiler buffered callbacks expose a `drop_count`; that is a
distinct AMD producer-loss input, not evidence about BPF transport or artifact
storage loss.

## Direct NVIDIA Activity API

The current [CUPTI documentation](https://docs.nvidia.com/cupti/main/main.html)
defines the Activity API as asynchronous collection of CPU and GPU activity
through client-supplied buffers. Important constraints are:

- records can arrive out of order because some are produced lazily;
- the client must provide buffers promptly;
- the client must query dropped records;
- runtime or driver activity must be enabled to generate the correlation IDs
  that join API and device records;
- device activity carries context, stream, correlation, and version-dependent
  graph fields;
- unsupported activity kinds can return compatibility errors.

The [CUPTI release notes](https://docs.nvidia.com/cupti/release-notes/release-notes.html)
must be read with the installed toolkit and driver. CUDA 13.3 introduced
multiple Activity subscribers with subscriber-scoped v2 APIs and requires a
compatible r610+ driver. This does not mean arbitrary NVIDIA developer tools or
profiling APIs can always coexist.

The pinned direct prototype is PR #237 commit
`d4ea783b7ef98aa2f0e3bfbcffc169e4612fd3a9`. It uses the CUDA 12-compatible
v1 Activity callback surface. It is retained as an experiment, not selected as
the final architecture.

## Host runtime semantic probes

### USDT and uprobes

The [USDT documentation](https://docs.ebpf.io/linux/concepts/usdt/) explains
that intentionally defined userspace tracepoints are represented by ELF notes
and NOP sites. A loader such as libbpf resolves the notes and attaches uprobes.
USDT gives a discoverable, intentional contract when the runtime owns the probe
point. Attaching to arbitrary internal symbols or offsets does not create that
stability.

A host probe can report that a runtime function was entered, returned, and
received selected arguments. It does not by itself establish when asynchronous
GPU work started or ended.

### BPF transport and privileges

The [Linux BPF ring-buffer design](https://www.kernel.org/doc/html/next/bpf/ringbuf.html)
documents shared memory, cross-CPU reservation ordering, and nonblocking
reservation failure when capacity is unavailable. A correct experiment must
count failed reservations and consumer lag. It cannot infer zero loss because
the consumer produced a file.

Linux capabilities documentation defines `CAP_BPF` and `CAP_PERFMON` as
separate least-privilege capabilities for BPF and performance-monitoring
operations. Kernel policy, `perf_event_paranoid`, unprivileged BPF policy,
namespaces, target ownership, and container configuration still determine the
actual deployed requirement. Root should not be used as a vague substitute for
documenting the necessary authority.

### ProfInfer

[ProfInfer](https://arxiv.org/abs/2601.20755) reports semantic function probes
and hardware trend collection for its evaluated llama.cpp architecture. Its
reported overhead is `PAPER_RESULT_NOT_REPRODUCED`.

The paper does not establish stable vLLM, SGLang, TensorRT-LLM, TensorRT core,
ROCm, or general CUDA probe points. The vLLM experiment must first inventory
intentional probes and exported symbols for the exact build, then explain what
semantic evidence survives compilation, stripping, Python/C++ boundaries, and
version changes.

## CUPTI-to-USDT/eBPF hybrid

The [Parca GPU design](https://www.polarsignals.com/blog/posts/2025/10/22/gpu-profiling)
and [parcagpu source](https://github.com/parca-dev/parcagpu) show a distinct
architecture:

1. CUPTI produces CUDA launch and device activity.
2. An injected library exposes selected records through USDT.
3. eBPF attaches to those USDT sites.
4. A privileged agent transports, aggregates, and joins records.

This is not evidence that eBPF independently supplies GPU execution timing.
CUPTI remains the GPU evidence source. The experiment separately measures
target CPU, helper or agent CPU, ring/perf-buffer loss, file I/O, bytes, and
post-processing so that a transport advantage cannot hide elsewhere.

The hybrid reference is pinned to parcagpu commit
`1e7e8da62513fd188c121716fc5028b4bd8ac47c`, verified on 23 September 2026.
Its published continuous-profiler results remain external until reproduced.

## Programmable GPU probes

[Neutrino](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)
introduces assembly-level programmable time and value probes for evaluated
Linux NVIDIA and AMD systems. Its strength is instruction-level investigation,
not replacement of a request-to-device activity timeline.

The artifact is pinned to commit
`4a82cd22f474c31ac2fecfa174d381a19bb3f469`. At least one lightweight named
question must be tested before a Stormlog claim is made. Heavy memory or
instruction probes remain targeted offline diagnostics unless measured serving
trials establish otherwise. One probe's overhead does not generalize to all
probes.

## No new native collector

This candidate combines:

- existing Stormlog metrics and inference semantic events;
- the #233 request, iteration, membership, and activity-reference contract;
- PyTorch/Kineto or supported engine profiler orchestration;
- Proton for current vLLM NVIDIA graph-aware profiling;
- imported Nsight or ROCProfiler traces for trusted offline diagnosis.

It avoids a Stormlog-owned GPU ABI and privileged BPF agent. It still incurs
format-adapter, orchestration, version, capture, and artifact support costs. It
wins only if the experiments show that supported surfaces answer the intended
questions at an acceptable operating point.

## Current research conclusions

**Confirmed:** polling cannot reconstruct asynchronous device intervals,
overlapping streams, or complete graph replay.

**Confirmed:** host eBPF probes do not independently provide GPU execution
timing.

**Confirmed:** the current vLLM Proton path creates a stronger public baseline
than PR #236 evaluated.

**Strong inference:** a custom collector has a high burden of proof because it
must beat supported public surfaces in evidence or cost while also accepting
native packaging, version, coexistence, and security obligations.

**Unknown:** whether direct CUPTI, the hybrid, semantic-only eBPF, programmable
probes, ROCProfiler ownership, or no new collector best fits Stormlog.

No architecture is selected until the hardware experiment turns the important
unknown cells into evidence-linked `STORMLOG_VALIDATED` cells.
