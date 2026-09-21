[← Back to main docs](index.md)

# Optional Native Probe Feasibility

This investigation answers [issue #118](https://github.com/Silas-Asamoah/stormlog/issues/118):
whether native probes can add inference evidence or reduce collection cost without
changing Stormlog's local-first, portable default experience.

## Decision

**Prototype CUPTI Activity collection first, after the capture and qualification
foundations in issues #216 and #221 are available.** The prototype should be an
opt-in, separately built native helper that writes bounded local trace sidecars.
Stormlog should import and correlate those sidecars instead of loading native
profiling code into its main Python process.

Defer eBPF/USDT transport and programmable GPU probes. Reconsider them only if a
direct CUPTI prototype proves useful but cannot meet an empirically selected
overhead or transport budget. Evaluate AMD ROCProfiler independently. Do not
infer AMD support from NVIDIA results, or vLLM and SGLang support from llama.cpp
results.

This is an **adopt for a bounded prototype**, not an adoption decision for a
production collector. The separately reviewed follow-up in issue #234 now
implements that opt-in prototype, CLI, and sidecar schemas without changing
default behavior or mandatory dependencies. See
[Native CUPTI Trace Capture](native_trace_integration.md).

## Evidence status

The labels below distinguish source-backed facts from decisions and open
questions. Sources were reviewed on 2026-09-21.

- **Confirmed:** behavior stated by official documentation, source code, or a
  cited paper for its evaluated configuration.
- **Strong inference:** the recommended Stormlog design based on confirmed
  constraints. It still needs qualification in Stormlog.
- **Hypothesis:** a claim that the experiment protocol must test.
- **Unknown:** no acceptable evidence is available yet.

This investigation contains **no Stormlog native-probe performance
measurements**. The development host is Apple Silicon macOS and has neither an
NVIDIA nor AMD GPU toolchain. Published overhead results belong to their cited
systems and workloads; they are not Stormlog results.

## Problems worth solving

Polling and framework summaries remain the right default for bounded memory
monitoring. Native evidence is justified only where it answers questions that
those surfaces cannot answer reliably.

| Problem | Evidence needed | Why current polling is insufficient |
| --- | --- | --- |
| Separate CPU launch cost from device execution | Host API interval, device start/end timestamps, and a correlation identifier | A short launch call can enqueue long-running asynchronous work. |
| Preserve overlapping execution | Context, device, and stream identity for each device interval | Interval samples cannot reconstruct concurrent streams. |
| Recognize CUDA graph replay | Graph and graph-node identity, plus replay activity | Repeated graph execution may not reproduce the framework operator sequence seen during capture. |
| Attribute GPU time to a bounded capture | Kernel, copy, synchronization, and correlation records with capture ownership | Utilization and memory samples reveal pressure but not which device work occupied the interval. |
| Detect incomplete evidence | Producer drop counts, buffer pressure, flush status, and unsupported activity kinds | A trace with silent loss can produce a confident but false explanation. |
| Compare detailed probes with trusted traces | Device timestamps and kernel identity under matched runtime settings | CPU timers and replayed counter measurements are not substitutes for an asynchronous execution trace. |

Native probes do not by themselves establish per-request GPU cost under shared
batching. That requires the request-to-iteration and shared-execution accounting
owned by issue #211. A kernel may serve multiple requests, so copying its full
duration onto every request would double-count work.

## Candidate capability matrix

"Privilege" describes the typical tracing requirement, not a promise for every
kernel, driver, container, or cloud configuration. Every future collector must
run a capability check on the target host.

| Surface | Evidence | OS and accelerator | Runtime or engine reach | Typical privilege and deployment | Graph, streams, and loss | Status for Stormlog |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch/Kineto | Framework operations, CPU activities, CUDA runtime, kernels, optional shapes/stacks/memory | PyTorch-supported platforms; CUDA evidence requires NVIDIA | Strong for PyTorch and engines that expose compatible profiler controls | In-process opt-in; no separate system tracer, but detail can add overhead | CUDA activities and flows are available; export behavior and settings must be recorded | Public-profiler baseline and import source under #216 |
| Nsight Systems | System-wide CPU/runtime/device timeline | NVIDIA platforms supported by the installed tool | Framework-independent CUDA visibility | External tool, target access, installation, and container configuration | Trusted timeline baseline for streams, graph activity, and overlap | Baseline only, not an always-on dependency |
| Nsight Compute | Kernel metrics and instruction/source detail | Supported NVIDIA GPUs and platforms | Kernel-focused, not a serving timeline | Performance-counter permission may require administrator capabilities | Replay, serialization, cache control, and software patching can perturb execution | Detailed-counter baseline only |
| CUPTI Activity | Asynchronous CUDA API, kernel, copy, synchronization, context, stream, graph, and correlation records | NVIDIA CUDA on supported Linux, Windows, WSL, QNX; not macOS | Framework-independent once CUDA work reaches the runtime/driver | Native library in the target process or supported injection/orchestration; activity tracing differs from privileged counter collection | Explicit CPU/device timestamps, correlation, stream/graph IDs, and dropped-record query | **First bounded prototype candidate** |
| USDT plus eBPF | Intentional user-space probe arguments transported through Linux tracing | Linux only; CPU-side transport rather than a GPU API | Only binaries with stable probe points, symbols, or known offsets | Commonly requires `CAP_BPF` and `CAP_PERFMON`, root, or delegated BPF token policy | BPF ring buffer is bounded and nonblocking; reservation failure and user-space lag must be counted | Defer unless CUPTI transport is the measured bottleneck |
| Parca GPU architecture | CUPTI activity emitted by an injected shim through USDT and eBPF | Linux NVIDIA | Demonstrated by the project on CUDA workloads, including a vLLM example | Privileged agent plus injected native library and BPF toolchain | Exposes correlation, kernel timing, streams, graphs, errors, and rate controls | Implementation reference, not a dependency or Stormlog result |
| Neutrino | Programmable assembly-level timing and memory probes | Linux; paper evaluates NVIDIA and AMD configurations | Kernel and build-path dependent; documented integrations can require custom builds | Native compiler/toolchain and GPU-specific instrumentation | Instruction-level evidence; coverage and perturbation depend on kernel/toolchain support | Defer until activity traces leave a concrete unanswered question |
| ProfInfer approach | Semantic runtime functions and hardware trends via uprobes/eBPF | Linux | Paper evaluates llama.cpp; vLLM and SGLang compatibility is unproven | Symbol/offset stability, BPF privileges, and per-version qualification | Semantic detail depends on runtime functions remaining discoverable and meaningful | Research reference only |
| ROCProfiler SDK | HIP/HSA runtime calls, kernel dispatches, memory moves, counters, and sampling services | AMD ROCm on Linux | HIP/ROCm workloads | Separate AMD native toolchain and supported hardware; PC sampling is documented as beta | Buffered callbacks expose drop counts; individual services have distinct support and stability | Separate AMD feasibility and qualification lane |
| Metal counters and Xcode tools | Command-buffer timing, counter samples, captures, and shader profiling | Apple platforms and Metal GPUs | Metal applications; framework mapping depends on what the framework exposes | Application instrumentation or Apple developer tooling | Counter availability is device-specific; this is not CUPTI or eBPF parity | Keep in the MPS/Metal investigation, not the CUDA prototype |
| ETW | Kernel and application-defined Windows events | Windows | Only providers that emit useful events | Windows session permissions and provider configuration | Does not create CUDA semantic or device evidence by itself | Supporting transport context, not a native GPU collector |

### Source-backed findings

- **Confirmed:** the CUPTI Activity API delivers CPU and GPU activity through
  asynchronous client-supplied buffers. Records may arrive out of order, and
  clients must query dropped records and provide buffers promptly. CUDA API and
  device records use correlation IDs; device, context, stream, graph, and graph
  node identifiers refine the relationship. See the
  [CUPTI Activity documentation](https://docs.nvidia.com/cupti/main/main.html#cupti-activity-api).
- **Confirmed:** current CUPTI release notes list Linux, Windows, WSL 2, and QNX
  support, but not macOS. API and architecture availability varies by CUPTI,
  driver, and GPU version. See
  [CUPTI platform support](https://docs.nvidia.com/cupti/release-notes/release-notes.html#support).
- **Confirmed:** recent CUPTI versions add multiple Activity API subscribers,
  but coexistence has version and driver requirements and does not extend to all
  profiling APIs or NVIDIA developer tools. A future helper must probe the
  installed version instead of assuming coexistence.
- **Confirmed:** Nsight Compute can replay or serialize work and documents
  overhead from metric collection, memory save/restore, cache control, software
  patching, and repeated application execution. See the
  [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- **Confirmed:** Linux tracing eBPF programs generally require `CAP_BPF` plus
  `CAP_PERFMON` or equivalent administrator authority, subject to kernel policy.
  See the [BPF capability model](https://docs.ebpf.io/linux/concepts/token/)
  and the Linux
  [`unprivileged_bpf_disabled` documentation](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/kernel.html#unprivileged-bpf-disabled).
- **Confirmed:** BPF ring buffers are bounded, preserve ordering across CPUs,
  and fail reservation rather than blocking when full. See the
  [Linux BPF ring-buffer design](https://www.kernel.org/doc/html/latest/bpf/ringbuf.html).
- **Confirmed:** USDT exposes intentionally placed user-space tracepoints that
  eBPF tooling attaches as uprobes. This does not make arbitrary runtime
  internals a stable interface. See the [USDT concepts](https://docs.ebpf.io/linux/concepts/usdt/).
- **Confirmed:** ROCProfiler SDK is Linux-only and offers buffered tracing of
  runtime calls and asynchronous device activities. Its PC sampling service is
  documented as beta with stability warnings. See the
  [ROCProfiler SDK documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
  and [PC sampling documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/pc_sampling.html).
- **Confirmed:** Apple exposes device-specific Metal counter sets and counter
  sample buffers, plus capture and profiling workflows. See
  [Metal GPU counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers).

### Research results that require reproduction

- [Neutrino](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)
  reports programmable assembly-level probing on evaluated Linux NVIDIA and AMD
  systems. Its repository describes integration-specific build requirements.
  Stormlog has not reproduced its coverage or overhead.
- [ProfInfer](https://arxiv.org/abs/2601.20755) reports fine-grained eBPF
  profiling for llama.cpp. Its results do not establish stable probe points or
  semantic equivalence for vLLM, SGLang, TensorRT-LLM, or TensorRT core.
- [Parca's CUDA design](https://www.polarsignals.com/blog/posts/2025/10/22/gpu-profiling)
  combines CUPTI, an injected USDT shim, eBPF, and a privileged agent. It proves
  that the architecture is implementable, but its workload results and support
  model are not Stormlog results.

## Implemented prototype boundary

The follow-up prototype has four explicit boundaries:

1. A native helper owns CUPTI initialization, activity selection, bounded
   buffers, drop accounting, flush, and shutdown.
2. The helper writes a versioned, bounded local trace sidecar and a small capture
   manifest. It does not call back into Stormlog Python on the capture hot path.
3. Stormlog imports the completed or recovered sidecar and registers it through
   the existing attachment catalog using session, job, rank, process, device,
   and time bounds.
4. Analysis joins the trace with request/iteration evidence only when clock,
   identity, and correlation coverage support the claimed relationship.

This boundary is a **strong inference**. It isolates native packaging and ABI
dependencies from mandatory Python dependencies, preserves the local-first
artifact workflow, and permits capture failure without taking down the default
monitor. CUPTI startup injection still runs native callbacks inside the target,
so it cannot isolate the target from an injected-library crash. Qualification
must measure whether the evidence is worth that residual risk and complexity.

The design breaks first when capture must begin after a server has already
initialized CUDA but the selected attachment mechanism requires startup-time
injection. It also breaks when another profiler owns an incompatible CUPTI
surface, when buffers cannot drain without material perturbation, or when a
runtime provides no stable semantic correlation beyond CUDA launches.

## Telemetry and collector-health fit

High-volume native activity must **not** be encoded as memory samples in
`TelemetryEvent v4`. Doing so would overload a strict memory schema, inflate
always-on artifacts, and obscure device clock semantics.

Instead:

- Store raw native records in a bounded trace sidecar with its own future schema.
- Link the sidecar through `stormlog_attachments.json` using `session_id`,
  `job_id`, `rank`, `start_ns`, `end_ns`, and a local path.
- Preserve the source clock domain, host observation time, CPU API interval,
  device execution interval, process/thread/device/context/stream identity,
  correlation ID, graph identity, event provenance, and uncertainty.
- Record requested and enabled activity kinds, buffer policy, bytes/events
  written, dropped-record counts, truncation, flush outcome, helper version,
  CUPTI/driver/toolkit versions, and capture ownership in the manifest.
- Reuse `healthy`, `degraded`, and `unhealthy` plus the existing partial-field,
  last-error, consecutive-failure, and retry vocabulary for helper lifecycle.
  Trace-specific loss remains an explicit manifest fact, not a synthetic memory
  value.
- Treat helper absence or unsupported capability as an optional unavailable
  source. Never fail the default monitor or silently substitute CPU launch time
  for device execution time.

Issue #211's correlation contract is not present on this branch. A prototype
must align with it after it lands rather than defining a competing request,
batch, or GPU-work schema here.

## Packaging and deployment risks

| Risk | Consequence | Required mitigation before adoption |
| --- | --- | --- |
| CUDA, CUPTI, and driver mismatch | Initialization failure, missing fields, or unsupported activity kinds | Discover installed versions and capabilities; fail closed with actionable diagnostics. |
| Native binary distribution | Platform and architecture build matrix, signing, and supply-chain burden | Ship separately from the pure-Python package; publish provenance and checksums; keep source builds reproducible. |
| Startup-time injection | Cannot attach safely to an existing server | Declare startup requirements in the manifest and distinguish them from attachable modes. |
| Profiler ownership conflict | Capture failure or changed workload behavior | Probe subscriber/tool coexistence, refuse unsafe combinations, and record the conflicting tool. |
| Buffer exhaustion | Biased or incomplete trace | Bound buffers and files, count all loss, mark the artifact partial, and never claim complete attribution. |
| Helper crash or hang | Lost evidence or target disruption | Minimize injected code, bound shutdown/flush, preserve partial artifacts, and state that startup injection cannot provide full crash isolation. |
| Clock conversion error | Incorrect ordering or latency | Retain raw timestamps and clock metadata; validate conversion against trusted markers. |
| Container or cloud restrictions | Missing libraries, injection denial, or counter permission failure | Publish a capability preflight and least-privilege deployment examples; do not require privileged defaults. |
| Symbol and address exposure | Sensitive implementation details in artifacts | Use restrictive file permissions, configurable symbolization, scrubbing, retention, and explicit sharing guidance. |
| Cross-process tracing authority | Unintended observation of other tenants | Scope by target identity, use least privilege, audit attachment, and reject ambiguous targets. |
| Runtime probe drift | Wrong semantic labels after upgrades | Prefer intentional versioned probes; pin and record runtime builds; qualify each adapter independently. |

## Security position

The isolated helper does not eliminate risk. Profiling data can reveal model
structure, kernel names, timing, memory behavior, addresses, paths, and process
identity. eBPF authority can expose activity beyond the intended target. A
future implementation therefore must:

- default to same-user, explicitly selected target processes;
- request no elevated capability unless a chosen mode demonstrably needs it;
- refuse broad host or container scope by default;
- create artifacts with owner-only permissions and local paths;
- keep symbolization and stack capture disabled unless requested;
- apply the artifact scrubbing and retention work coordinated with issue #111;
- include the exact privilege, target selector, and collected fields in the
  capture manifest; and
- clean up subscriptions, probes, temporary files, and injected state on normal,
  cancelled, and failed exits.

## Decision gates

The bounded CUPTI prototype is implemented as follow-up issue #234. Reuse issue
#216 capture work and align with issue #211 shared-execution correlation when
those contracts land. Use
[Native Probe Qualification](native_probe_qualification.md) with issue #221
before any production-readiness claim.

Adopt a production collector only if all of the following are demonstrated:

- It adds actionable device evidence that the public-profiler baseline cannot
  provide at the required operating point, or achieves materially lower measured
  collection cost for the same evidence.
- Correlation, graph replay, overlapping streams, and event loss remain explicit
  and accurate against a trusted trace.
- The helper remains optional, bounded, local-first, and failure-isolated.
- Packaging, security, support, and coexistence are acceptable for each declared
  platform and runtime.
- NVIDIA and AMD claims have independent artifacts and qualification results.

Defer the collector if evidence is useful but overhead, packaging, privileges,
or runtime coverage miss the budgets selected in issue #221. Reject it if it
cannot improve on imported public-profiler evidence without privileged defaults,
silent loss, misleading attribution, or unacceptable target perturbation.

## References

- [Issue #118](https://github.com/Silas-Asamoah/stormlog/issues/118)
- [Inference roadmap #210](https://github.com/Silas-Asamoah/stormlog/issues/210)
- [Capture work #216](https://github.com/Silas-Asamoah/stormlog/issues/216)
- [Qualification protocol #221](https://github.com/Silas-Asamoah/stormlog/issues/221)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [vLLM profiling controls](https://docs.vllm.ai/en/stable/contributing/profiling/)
- [NVIDIA CUPTI](https://docs.nvidia.com/cupti/main/main.html)
- [NVIDIA profiling-counter permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Linux BPF ring buffer](https://www.kernel.org/doc/html/latest/bpf/ringbuf.html)
- [USDT concepts](https://docs.ebpf.io/linux/concepts/usdt/)
- [ROCProfiler SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [Metal GPU counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
- [Event Tracing for Windows](https://learn.microsoft.com/en-us/windows/win32/etw/about-event-tracing)
- [Neutrino](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)
- [ProfInfer](https://arxiv.org/abs/2601.20755)
- [Parca GPU profiling design](https://www.polarsignals.com/blog/posts/2025/10/22/gpu-profiling)
