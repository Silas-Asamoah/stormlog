[← Back to main docs](index.md)

# Native Probe Security and Deployment Review

This review treats operational cost as a decision dimension equal to profiling
fidelity and runtime overhead. No production deployment is approved by this
document.

## Protected assets

Native traces may expose:

- model and kernel structure;
- symbol names, addresses, stacks, paths, and build identifiers;
- process, thread, device, stream, graph, and request identities;
- request timing, batching patterns, cache behavior, and workload volume;
- model inputs or application data if a probe captures values;
- host-wide activity when tracing scope is broader than intended.

Raw traces are sensitive local artifacts. Share checksums and controlled durable
locations when raw files are too large or sensitive for Git.

## Trust boundaries

### In-process native code

CUPTI injection and programmable GPU hooks execute in or affect the target
process. A memory-safety bug, ABI mismatch, deadlock, callback stall, or cleanup
failure can crash or corrupt the workload. A sidecar file does not create
process isolation when the producer library is in-process.

Required controls:

- explicit opt-in and exact library path;
- pinned source, toolchain, and checksums;
- no group- or world-writable library or output path;
- owner-only directories and artifacts;
- byte, time, buffer, and flush bounds;
- fail-closed initialization and explicit partial status;
- no silent fallback from device timing to host timing;
- recovery tests for signals, cancellation, helper failure, and target crash.

### eBPF agent

An eBPF loader or agent crosses the userspace/kernel boundary and may observe
processes beyond the intended target. `CAP_BPF`, `CAP_PERFMON`, ptrace policy,
namespace visibility, perf policy, lockdown, and BPF token delegation must be
recorded. Do not request `CAP_SYS_ADMIN` merely because it is convenient.

Required controls:

- restrict attachment by verified PID identity, executable build ID, user,
  cgroup, and namespace;
- reject PID reuse and ambiguous targets;
- load only pinned BPF objects through a constrained loader;
- avoid writable host mounts and broad container privileges;
- record probe sites and arguments in the manifest;
- count ring/perf-buffer reservation and consumer drops;
- detach probes and remove pinned objects on success, failure, and cancellation;
- audit which processes were observable, not only which records were emitted.

### External profiler tools

Nsight, ROCProfiler, Proton, and detailed-counter tools have their own capture,
injection, temporary-file, process-control, and counter-permission boundaries.
Stormlog orchestration must preserve tool output and diagnostics without
claiming stronger isolation than the source tool provides.

## Artifact controls

- Use a new `0700` trial directory and `0600` files.
- Refuse symlinks, hard-link ambiguity, path traversal, and overwrite.
- Stream checksums while copying or finalize before hashing.
- Record byte size and SHA-256 for every raw artifact.
- Persist exact argv but reject secret-like environment overrides.
- Redact authorization headers, API keys, bearer tokens, model inputs, and
  query parameters before any artifact is committed or shared.
- Keep symbolization optional and record whether symbols, addresses, stacks,
  or user data are present.
- Define retention and secure deletion outside this research branch before any
  production feature is proposed.

## Deployment matrix

| Candidate | Startup/attach | Typical authority | Container cost | Failure scope |
| --- | --- | --- | --- | --- |
| PyTorch/Kineto | In-process configured capture | Target user | Framework and GPU device access | Target overhead or profiler failure |
| vLLM Proton | Profiler configured at server startup; bounded capture later | Target user plus CUDA permissions | Compatible vLLM, Triton, CUPTI, output mount | Worker overhead or profiler failure |
| Nsight Systems | Launch or supported attach/session | Same user in common process-tree mode; broader modes vary | Tool install, injection visibility, output mount | Target control and buffered data loss |
| Direct CUPTI | PR #237 prototype uses startup injection | Target user for bounded activity path | Compatible library and CUPTI mounted in target | In-process target crash or stall |
| Semantic eBPF | Late attach possible if probe sites and policy allow | Commonly delegated BPF/perf authority | Kernel features, BPF loader, namespace visibility | Agent/probe scope can exceed target |
| CUPTI/eBPF hybrid | CUDA injection plus agent attachment | Target user plus delegated/privileged agent | Native shim, BPF objects, agent, mounts, capabilities | Combined in-process and host-agent risk |
| Neutrino | Instrumented runtime/build path | Toolchain and hook dependent | Custom build and GPU runtime interception | Kernel coverage or target perturbation |
| ROCProfiler | Launch or version-dependent attach | ROCm device and service dependent | ROCm toolchain, devices, output mount | Tool or target perturbation |
| No native collector | Source profiler dependent; import after capture | Import itself is unprivileged | Lowest additional Stormlog authority | Format/orchestration failure |

These are source-backed expectations, not deployed demonstrations.

## Packaging implications

The default Stormlog wheel must remain pure Python and usable without CUDA,
ROCm, eBPF, vLLM, Triton, or vendor tools. Any adopted native path requires a
separate distribution decision covering:

- source and binary provenance;
- supported OS, architecture, libc, driver, toolkit, and runtime matrix;
- reproducible builds and software bill of materials;
- signing and checksums;
- optional dependency isolation;
- ABI and record-version compatibility policy;
- vulnerability response and update cadence;
- license compatibility for vendored or redistributed code;
- failure behavior when the component is absent or incompatible.

Reusing a supported public profiler avoids Stormlog-owned native binaries but
does not eliminate format, version, or orchestration support.

## Required deployment demonstrations

For every claimed platform:

1. run as a normal same-user process with no unnecessary capability;
2. record the exact denial when authority is insufficient;
3. run in the declared container configuration;
4. demonstrate target scoping and PID-reuse protection;
5. fill buffers and storage bounds;
6. crash the producer or consumer;
7. cancel and time out capture;
8. verify cleanup and partial artifact readability;
9. run profiler coexistence cases;
10. inspect committed artifacts for secrets and sensitive values.

Unsupported deployment combinations remain explicit. A root-only success does
not establish support for ordinary Stormlog users.
