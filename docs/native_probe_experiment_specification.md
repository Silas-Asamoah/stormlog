[← Back to main docs](index.md)

# Native Probe Experiment Specification

This protocol defines the fair experiment for issue #118. The implementation
lives in `research/native_probes/`. It is a research harness, not a Stormlog
production API.

## Objective and decision

The objective is to identify the lowest-cost, operationally acceptable way to
obtain evidence that existing Stormlog telemetry and supported public profilers
cannot provide. The result feeds an adopt, defer, reject, or layered-architecture
decision. A negative result is valid.

No arbitrary weighted score is used. Raw observations are published first.
Candidates must pass critical correctness gates before cost comparisons can
support a decision.

## Candidate modes

| Mode | Candidate | Purpose |
| --- | --- | --- |
| `off` | No profiler | Performance and resource baseline |
| `public-pytorch` | PyTorch/Kineto | Framework-visible public baseline |
| `public-engine` | vLLM supported profiler control | Engine-supported baseline |
| `proton` | vLLM Triton Proton | Current CUPTI-backed engine baseline and graph attribution |
| `trusted` | Nsight Systems or rocprofv3 | Independent device timeline reference |
| `ebpf-semantic` | USDT/uprobe/eBPF | Host semantic value and transport cost |
| `direct-cupti` | Pinned PR #237 native helper | Direct NVIDIA Activity prototype |
| `hybrid-cupti-ebpf` | Pinned parcagpu | CUPTI evidence with USDT/eBPF transport |
| `programmable` | Pinned Neutrino | Named instruction-level question only |
| `amd-rocprofiler` | ROCProfiler SDK | Independent AMD activity experiment |
| `detailed-counter` | Nsight Compute or AMD counters | Perturbed offline diagnostic baseline |

Do not run mutually incompatible combinations outside the coexistence phase.

## Immutable configurations

A comparison configuration freezes:

- repository and prototype revisions;
- OS, kernel, CPU, GPU, architecture, driver, and firmware;
- CUDA or ROCm, CUPTI or ROCProfiler, PyTorch, Triton, vLLM, and profiler
  versions;
- container image digest, runtime, mounts, devices, seccomp, capabilities, and
  user namespace;
- model and tokenizer immutable revisions;
- quantization, attention backend, compilation, graph, parallelism, power,
  clock, and cache settings;
- workload seed, shapes, arrivals, concurrency, timeouts, and token targets;
- capture activities, detail options, bounds, flush policy, and symbolization;
- all source clocks and clock-alignment method.

Changing one of these values creates a new configuration. Results from distinct
configurations are not pooled.

## Required environments

### NVIDIA

- Supported Linux x86-64 or Arm server host.
- Supported NVIDIA GPU with exclusive access for correctness trials.
- Compatible driver, CUDA toolkit, CUPTI, PyTorch, vLLM, Triton, Nsight
  Systems, and Nsight Compute.
- Normal-user and container trials where support is claimed.
- Exact performance-counter policy and competing processes recorded.

### AMD

- Supported Linux host and AMD GPU.
- Compatible ROCm, HIP, ROCProfiler SDK, rocprofv3, PyTorch, and engine build.
- Stable tracing separated from beta sampling services.
- Normal-user and container trials where support is claimed.

### Linux semantic and hybrid

- Kernel release and configuration recorded.
- `unprivileged_bpf_disabled`, `perf_event_paranoid`, BPF JIT policy, BTF,
  tracefs, and lockdown state recorded.
- Effective `CAP_BPF`, `CAP_PERFMON`, `CAP_SYS_PTRACE`, and `CAP_SYS_ADMIN`
  recorded without assuming root is necessary.
- Target mount, PID, user, and cgroup namespaces recorded.

### Control host

The Apple M1/macOS control host may validate schemas, reports, source pinning,
command construction, trial retention, and analysis. It cannot validate CUDA,
ROCm, Linux eBPF, vLLM GPU behavior, privileges, containers, or performance.

## Microbenchmarks

All workloads record the seed, warmup, measured iterations, tensor shapes,
dtype, expected operations, synchronization points, and explicit marker range.

### W1: asynchronous eager launch

Run a fixed matrix operation and elementwise operation on one stream. Preserve:

- per-iteration host enqueue duration;
- host duration through final synchronization;
- CUDA/HIP event duration for the measured device region;
- expected operation and iteration counts.

Pass condition: CPU API and device intervals are separate fields, eligible
device records join to API evidence, and no tool substitutes one duration for
the other.

### W2: overlapping streams

Run the same independent matrix operations on two streams in two variants:

- `w2-overlap`, with no inter-stream serialization;
- `w2-serialized`, synchronizing the first stream before the second launch.

Pass condition: the trusted trace confirms that the variants differ; the
candidate preserves stream identity, per-stream order, and the overlap result.
If profiling serializes the overlap variant, mark the trial perturbed and fail
the fidelity gate.

### W3: graph replay

Capture a fixed operation sequence once and replay it after warmup. Compare
with the eager equivalent. Record capture count, replay count, graph and node
identity where exposed, expected device activities, and unmatched work.

Pass condition: graph replay remains identifiable, repeated GPU work is not
mistaken for repeated framework capture, and missing node identity is explicit.

### W4: high-event-rate stress

Issue 100 small elementwise launches per iteration and increase iterations or
reduce buffer bounds until loss occurs. Record expected launches, delivered
records, all producer and transport drops, truncation, buffer capacity,
occupancy where exposed, flush result, and artifact bytes.

Pass condition: induced loss is visible. A collector that silently loses
records fails even if its surviving trace parses.

## Failure experiments

Run these outside performance comparisons:

- helper or agent crash;
- target cancellation and timeout;
- full output capacity and deliberately small ring buffer;
- failed or partial flush;
- unsupported activity;
- toolkit, driver, helper, or runtime mismatch;
- permission denial;
- target crash, restart, and PID reuse;
- invalid or inaccessible output path;
- consumer delay and post-processing failure.

Record target correctness and liveness, cleanup, partial artifact readability,
loss state, exit status, and failure blast radius. Never discard a failed trial.

## vLLM experiment

Do not begin vLLM trials until W1 through W4 establish candidate correctness.
Use #233 identities:

```text
request
  -> membership in a shared iteration
  -> generic stage
  -> runtime/device activity references
```

Record measured iteration elapsed time, summed activity duration, merged GPU
interval duration, and unresolved activity separately. A request share is an
`ESTIMATE` with a named accounting model and an unattributed remainder.

At minimum run:

1. eager single request;
2. eager concurrent requests;
3. CUDA graphs with required profiler attribution settings;
4. continuous/shared batching;
5. reproducible mixed prefill/decode iteration;
6. cache-stable traffic;
7. controlled cache-state change.

Use the same model revision, server arguments, runtime, graph, cache, arrival,
request, and token settings across modes. First qualify vLLM. Do not generalize
to SGLang, TensorRT-LLM, TensorRT core, or llama.cpp.

### Proton baseline

Configure vLLM at startup with a local profile directory:

```bash
.venv/bin/vllm serve <immutable-model-revision> \
  --profiler-config '{
    "profiler": "proton",
    "proton_profiler_dir": "./artifacts/proton",
    "proton_output_format": "hatchet",
    "proton_hook": "triton",
    "proton_graph_attribution": true
  }'
```

Use the same `/start_profile` and `/stop_profile` capture boundaries for each
trial. Record `proton_data`, backend, mode, hook, graph attribution, output
format, and Triton version. For trace mode, follow vLLM's documented graph
limitations rather than silently changing the deployed graph setting.

## Prototype acquisition

### Direct CUPTI

Extract the native helper from PR #237 commit
`d4ea783b7ef98aa2f0e3bfbcffc169e4612fd3a9` using
`research.native_probes.references.extract_cupti_reference`. Build it against
the declared CUDA toolkit. Do not merge or exercise its production Stormlog
integration during the comparison.

### Hybrid

Clone [parcagpu](https://github.com/parca-dev/parcagpu) at
`1e7e8da62513fd188c121716fc5028b4bd8ac47c`, initialize its pinned submodules,
build its injection library and BPF consumer, and record every dependency.
Measure direct CUPTI output and hybrid transport separately. The hybrid must
report CUPTI drops and BPF/perf or ring-buffer drops independently.

### Semantic eBPF

Before attaching probes, inventory the exact vLLM, PyTorch, CUDA/HIP, and
extension binaries for intentional USDT notes and exported symbols. If no
stable semantic point exists, record that as the result. A function-name or
offset experiment must pin the binary build ID and remains version-fragile.

### Programmable

Clone [Neutrino](https://github.com/open-neutrino/neutrino) at
`4a82cd22f474c31ac2fecfa174d381a19bb3f469` and install it into an isolated
environment using that source. The first named question is:

> When W1's activity timeline identifies one unexpectedly long matrix kernel,
> can a lightweight entry/exit device-clock probe localize intra-kernel warp
> tail behavior that the activity timeline cannot explain?

Test only the minimal entry/exit clock probe first. Report supported kernel
coverage and rejected kernels. Do not infer heavy memory-probe cost from it.

## Trial discipline

- Warm up once before measurement.
- Restart the target between independent trials.
- Run at least five measured trials per condition.
- Use at least 1,000 measured microbenchmark iterations.
- Follow #221's final inference sample-sufficiency policy. Until operational,
  use at least 1,000 successful measured requests per condition and report all
  failures separately.
- Generate a reproducible counterbalanced order with a recorded seed.
- Reset declared cache and graph state between trials.
- Record thermal, power, clock, and competing-process state where available.
- Retain unsupported, failed, partial, and timed-out runs.
- Repeat claimed comparisons after a clean process restart.

## Measurements

### Evidence fidelity

```text
host_device_correlation_coverage =
  correlated_eligible_device_records / eligible_device_records

kernel_count_coverage =
  matched_expected_kernel_instances / expected_kernel_instances

graph_replay_coverage =
  identified_replays / expected_replays

record_loss_rate =
  reported_dropped_records /
  (delivered_records + reported_dropped_records)
```

Also report stream attribution accuracy, ordering, timestamp agreement with the
trusted trace, semantic-phase coverage, request/iteration join coverage,
unmatched activities, and unknown remainder. If the denominator is unavailable,
the result is `UNKNOWN`.

### Workload perturbation

Measure throughput, SLO goodput, p50/p95/p99 latency, TTFT, valid TPOT,
microbenchmark makespan, and capture start, stop, flush, and post-processing
pauses. Compute matched deltas as:

```text
100 * (profiled - off) / off
```

State direction for throughput and goodput because a negative delta is a
degradation.

### Collector and resource cost

Measure target, helper, and system CPU; RSS; threads; wakeups when practical;
disk I/O; records and bytes per second; peak buffer occupancy; raw and
compressed bytes; symbolization; and all post-processing time.

### Operational cost

Demonstrate privileges, container configuration, injection or attach timing,
driver/toolkit coupling, native build, binary distribution, supported OS and
architectures, security exposure, cleanup, coexistence, and failure blast
radius.

## Coexistence matrix

Explicitly test each applicable candidate with:

- PyTorch/Kineto;
- vLLM engine profiler controls;
- Proton;
- Nsight Systems;
- Nsight Compute;
- another CUPTI Activity subscriber;
- ROCProfiler service combinations on AMD.

Record installed CUPTI version and driver. CUDA 13.3+ multiple-subscriber
support is version-gated and does not prove arbitrary tool coexistence.

## Decision gates

A candidate cannot win on cost if it fails any required gate:

1. CPU and device time are represented separately.
2. Required correlation, stream, graph, and kernel evidence matches ground
   truth and the trusted trace within declared tolerances.
3. All observed loss is explicit; silent loss is a failure.
4. Target correctness is preserved for the candidate's intended failure model.
5. Shared execution is not double-counted.
6. Privilege, packaging, container, and support obligations are acceptable for
   the claimed operating point.

Only after those gates pass may evidence value, perturbation, resource cost,
and operational cost support an adopt/defer/reject decision.
