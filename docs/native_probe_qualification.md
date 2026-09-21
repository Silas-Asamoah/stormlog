[← Back to native probe feasibility](native_probe_feasibility.md)

# Native Probe Qualification Protocol

This protocol makes the recommendation in
[Optional Native Probe Feasibility](native_probe_feasibility.md) reproducible.
It defines evidence and artifact requirements for issue #118. It does not claim
that the experiments have run.

The protocol extends the inference qualification work in
[issue #221](https://github.com/Silas-Asamoah/stormlog/issues/221). Use its
declared inference workloads and sample-sufficiency policy when available. The
microbenchmarks below isolate trace correctness before an inference engine adds
scheduling, batching, cache, and distributed effects.

## Rules

1. Keep profiler-off and profiler-on workload settings identical.
2. Separate CPU API intervals from device execution intervals.
3. Preserve asynchronous execution, deployed graph settings, cache state, and
   stream concurrency.
4. Compare NVIDIA and AMD in separate result sets. Never fill an unsupported
   cell with a result from the other vendor.
5. Retain raw artifacts, commands, manifests, and tool output for every result.
6. Treat dropped, truncated, unflushed, or clock-ambiguous evidence as partial.
7. Do not set an overhead budget from literature. Issue #221 selects budgets
   from matched Stormlog measurements.
8. A lane that skips every applicable GPU scenario is not a successful run.

## Experiment modes

Run each supported mode independently. Run coexistence combinations only in the
separate coexistence phase so one profiler does not silently change another
mode's result.

| ID | Mode | Purpose | Required output |
| --- | --- | --- | --- |
| `off` | No profiler; Stormlog client and required workload telemetry only | Performance and resource baseline | Request results, workload ground truth, system samples |
| `public` | PyTorch/Kineto or the engine's documented public profiler | Framework-visible baseline | Original trace, profiler configuration, export log |
| `trusted` | Nsight Systems for NVIDIA or `rocprofv3` tracing for AMD | Independent timeline reference | Original vendor trace and export summary |
| `host-probe` | Declared USDT/uprobe/eBPF implementation | Evaluate semantic probes and ring-buffer transport | Probe inventory, raw events, BPF/kernel policy, loss counters |
| `cupti-activity` | Proposed bounded CUPTI Activity helper | Evaluate direct host/device activity collection | Raw activity sidecar, helper manifest, dropped-record counts |
| `programmable` | Neutrino or another declared programmable GPU probe | Test instruction-level evidence for a specific unanswered question | Probe source/configuration, supported-kernel inventory, raw output |
| `detailed-counter` | Nsight Compute or vendor-equivalent detailed counters | Offline diagnostic comparison, not live-serving latency | Replay/configuration metadata, report, original output |

`cupti-activity` and Nsight modes are NVIDIA-only. The AMD lane substitutes
ROCProfiler services and records each service separately. A Linux eBPF transport
is not an AMD GPU API and must not be described as vendor parity.

## Required environments

### NVIDIA lane

- Linux x86-64 or Linux Arm server host with a supported NVIDIA GPU.
- Installed driver, CUDA Toolkit, CUPTI, PyTorch, and selected inference engine.
- Nsight Systems and Nsight Compute versions compatible with the driver.
- Container runtime, capability, and performance-counter policy recorded when
  the workload is containerized.
- Exclusive GPU access for correctness trials where another workload would make
  trace comparison ambiguous.

Windows and WSL can be recorded as separate capability trials after the Linux
prototype. They do not inherit Linux injection, eBPF, packaging, or privilege
results.

### AMD lane

- Supported Linux host, AMD GPU, ROCm, HIP runtime, and ROCProfiler SDK.
- `rocprofv3` and the exact tracing or sampling services under test.
- Independent engine and framework support declaration.
- PC sampling disabled unless the host and experiment explicitly accept its
  documented beta risk.

### Control host

A CPU-only or macOS host may validate manifests, parsers, reports, and static
artifacts. It cannot produce NVIDIA or AMD performance, coverage, privilege, or
coexistence results.

## Workload suite

Every workload records deterministic seeds, warmup, measured iterations,
tensor shapes, data types, synchronization points, and expected activity. A
native fixture should use explicit marker ranges around the measured region so
tools can select the same work.

### W1: Eager launch and execution

Run a fixed sequence of named kernels and copies from one host thread on one
stream. Include at least one kernel long enough to distinguish CPU launch
duration from device execution duration.

Ground truth:

- ordered kernel and copy sequence;
- expected launch count;
- CUDA or HIP event duration for the measured device region; and
- host monotonic timestamps around enqueue and final synchronization.

Acceptance observation: the report shows CPU API and device intervals as
different fields and correlates eligible device work without substituting one
duration for the other.

### W2: Overlapping streams

Launch independent, duration-controlled kernels on at least two non-default
streams with event dependencies that create both overlap and a known ordering
edge. Run the same topology with overlap disabled as a control.

Ground truth:

- stream assignment for every launch;
- explicit event dependency;
- expected overlapping kernel pair; and
- overlapped and serialized device makespans.

Acceptance observation: the trace preserves stream identity, the dependency,
and the presence or absence of overlap. A serialized profiler run is reported
as perturbed rather than accepted as faithful.

### W3: CUDA or HIP graph replay

Capture a fixed kernel/copy sequence into a graph, instantiate it once, and
replay it repeatedly after warmup. Record capture and replay counts. Compare it
with a semantically equivalent eager sequence.

Ground truth:

- graph and node topology;
- capture count;
- replay count; and
- expected device activities per replay.

Acceptance observation: graph replay is identifiable, repeated device work is
not mistaken for repeated framework operator capture, and missing graph/node
identity remains explicit.

### W4: Declared inference workload

Use the controlled arrival, prefix-cache, model, and request-shape definition
owned by issue #212 and the capture lifecycle from issue #216. The first target
is vLLM; SGLang, TensorRT-LLM, and TensorRT core receive distinct later runs.

At minimum record:

- model identifier and immutable revision;
- engine, framework, compiler, runtime, driver, and toolkit versions;
- server arguments and environment variables;
- GPU topology and distributed ranks;
- eager or graph mode and compile settings;
- arrival process, concurrency, warmup, request count, input/output token
  targets, streaming mode, and timeout;
- cache state and how it was established; and
- profiler start, stop, flush, and ownership lifecycle.

Do not execute W4 until its request and server identities can join the
request-to-iteration contract from issue #211. Kernel correlation alone is not
per-request attribution under shared batching.

## Trial design

For each supported workload and mode:

1. Run one unmeasured environment and model warmup.
2. Run at least five measured trials per condition.
3. Use at least 1,000 measured microbenchmark iterations. For inference, follow
   issue #221's sample-sufficiency result; until defined, collect at least 1,000
   successful measured requests per condition and report failures separately.
4. Randomize or counterbalance mode order to reduce thermal and temporal bias.
5. Reset the declared cache and graph state between trials.
6. Record device clocks, power policy, competing processes, and thermal state
   when the platform exposes them.
7. Preserve failures and timeouts in the denominator.
8. Repeat any claimed comparison on a second run after a clean process restart.

Do not combine results across different GPUs, drivers, toolkit versions, model
revisions, or engine configurations. Report those as separate configurations.

## Required measurements

### Workload perturbation

- request throughput and SLO goodput for inference;
- p50, p95, and p99 end-to-end latency, plus TTFT when valid;
- measured-region wall time for microbenchmarks;
- CPU process time and utilization;
- peak and delta RSS;
- GPU memory values already available through the declared public surface;
- capture start, stop, and flush pause; and
- process exit, cancellation, and recovery behavior.

For metric `x`, report the paired perturbation as:

```text
100 * (x_profiled - x_off) / x_off
```

For throughput and goodput, also state the direction explicitly because a
negative delta represents degradation. Publish individual trials, median,
dispersion, and the selected confidence interval method. Do not report only the
best trial.

### Collection cost

- raw and compressed artifact bytes;
- bytes and records per second;
- peak queue or ring-buffer occupancy when observable;
- requested, delivered, discarded, truncated, and dropped record counts;
- helper CPU time, RSS, threads, and wakeup/poll strategy; and
- symbolization and post-processing time measured separately from capture.

### Evidence correctness

Define eligible records before running the comparison. Then report:

```text
host_device_correlation_coverage =
    correlated_eligible_device_records / eligible_device_records

graph_replay_coverage =
    identified_replays / expected_replays

kernel_count_coverage =
    matched_expected_kernel_instances / expected_kernel_instances

record_loss_rate =
    reported_dropped_records / (delivered_records + reported_dropped_records)
```

If a tool cannot expose a denominator, mark the metric `unknown`; do not encode
it as zero. Compare kernel name or stable identity, device, context, stream,
start/end order, graph metadata, and activity kind against the trusted trace.
Explain matching tolerances and unmatched records.

For overlap, compare the expected concurrent pair, per-stream ordering, and
device makespan. Timestamp differences between tools are expected; compare
clock domains and conversion before interpreting drift.

### Evidence value

For each mode, list every diagnosis question answered uniquely beyond the
`public` and `trusted` baselines. A new field is not automatically useful. It
must change or substantiate an incident conclusion while retaining inspectable
raw evidence.

## Coexistence and failure matrix

Run these as explicit negative and compatibility tests rather than mixing them
into performance trials:

- helper plus PyTorch/Kineto;
- helper plus engine-native profiler controls;
- helper plus Nsight Systems;
- helper plus Nsight Compute or another counter owner;
- unsupported activity kind;
- incompatible helper/CUPTI/driver versions;
- denied injection or missing native library;
- insufficient tracing or performance-counter permission;
- full output buffer and full disk budget;
- helper crash, target crash, cancellation, and forced timeout;
- failed or partial flush; and
- target restart or PID reuse.

For each case, record whether the target remained correct and live, whether the
helper cleaned up, whether a partial artifact stayed readable, and which health
and loss fields explain the failure. A profiler conflict must be refused or
clearly degraded, never silently ignored.

## Capture manifest requirements

Each trial manifest must contain these groups. A future implementation may
version a machine-readable schema after the experiment proves the shape; this
research issue does not create a persisted contract.

- `identity`: run, trial, session, job, rank, host, process, device, and helper
  identifiers.
- `environment`: OS/kernel, CPU, GPU, driver, toolkit, CUPTI or ROCProfiler,
  Python, framework, engine, container, and source revisions.
- `workload`: suite ID, model and revision where applicable, seeds, shapes,
  arrival/cache/graph/compile settings, warmup, measured work, and ground truth.
- `collector`: mode ID, executable and version, activity/probe configuration,
  buffer/file bounds, privilege, target selector, start/stop/flush ownership,
  and coexistence state.
- `time`: every source clock domain, conversion or synchronization method,
  capture bounds, and observation bounds.
- `health`: status, partial fields, last error, consecutive failures, retry
  state, delivered/dropped/truncated counts, and flush outcome.
- `artifacts`: path, kind, byte size, checksum, storage mode, and whether the
  artifact contains symbols, addresses, stacks, or user data.
- `result`: trial status, unsupported reasons, measured metrics, analysis
  version, and limitations.

Unknown and unsupported values remain explicit. Do not omit them in a way that
could be read as successful support.

## Artifact layout

Use one immutable directory per configuration and trial:

```text
native-probe-qualification/
  README.md
  environment.json
  configurations/
    <configuration-id>/
      manifest.json
      commands.txt
      trials/
        <trial-id>/
          manifest.json
          raw/
          normalized/
          metrics.json
          logs/
      report.md
  comparison.json
```

- `README.md` identifies the protocol revision and explains reproduction.
- `environment.json` records host-wide facts and capability preflight output.
- `commands.txt` contains exact commands with secrets redacted, plus immutable
  source and image references.
- `raw/` preserves original tool output without rewriting it.
- `normalized/` contains derived records and matching diagnostics.
- `metrics.json` contains individual observations, not only aggregates.
- `report.md` uses the template below.
- `comparison.json` indexes only configurations that are valid to compare.

Raw GPU traces can be large or sensitive. Publish checksums and a durable access
location when repository storage is inappropriate. Never commit credentials,
model inputs, or unreviewed addresses and paths.

## Result-report template

```markdown
# Native Probe Qualification: <configuration>

## Status

- Protocol revision:
- Date:
- Result: pass | fail | partial | unsupported
- Evidence completeness:
- Reviewer:

## Environment

<Link to manifest and concise hardware/runtime summary.>

## Workloads and modes

<List exact workload IDs, trial counts, and supported/skipped modes.>

## Correctness

<Correlation, graph replay, stream overlap, kernel coverage, event loss,
clock validation, and differences from the trusted trace.>

## Perturbation and resource cost

<Individual-trial links, aggregates, uncertainty, CPU/RAM, artifact growth,
capture/flush pauses, and profiler-off comparison.>

## Evidence gained

<Questions answered beyond public-profiler and trusted-trace baselines.>

## Privilege, packaging, and coexistence

<Required capabilities, startup requirements, failures, and tool conflicts.>

## Limitations

<Unsupported hardware, kernels, engines, activities, and uncertain claims.>

## Decision

Adopt | defer | reject, with evidence-linked reasons and next action.
```

## Acceptance decision

The CUPTI helper advances beyond prototype only when the artifacts demonstrate
all feasibility gates in the companion report and satisfy budgets selected by
issue #221. A lower overhead number is insufficient if correlation, event loss,
security, or deployment makes the evidence unreliable.

Defer eBPF transport unless the direct helper's measured handoff or storage path
is a dominant cost that a bounded ring-buffer design can plausibly reduce.
Evaluate Neutrino only for a named instruction-level question that activity
traces and detailed public tools cannot answer. A negative or unsupported result
is valid when the manifest and raw evidence make the limitation reproducible.
