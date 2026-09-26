# Native Probe Comparative Evaluation

## Research question

What is the lowest-cost, operationally acceptable way for Stormlog to obtain
evidence that its telemetry and supported profiler imports cannot provide?

This report does not select a native collector. The current host cannot execute
the required NVIDIA, AMD, Linux eBPF, or vLLM trials. Selecting a winner from
documentation alone would repeat the sequencing error this investigation is
intended to correct.

## Existing Stormlog baseline

Stormlog already has bounded captures, imported profiler artifacts, inference
events, and the shared-execution correlation contract delivered by PR #233.
That contract keeps a request's membership in shared GPU execution separate
from any modeled allocation of shared work. Issue #211 is therefore no longer
an absent prerequisite.

The baseline can answer memory, request, batch, stage, and imported-trace
questions. It does not by itself establish CPU-launch-to-device-execution
correlation, stream overlap, graph replay identity, kernel coverage, or loss
denominators. Public profilers may already answer many of those questions and
must be measured before Stormlog owns another collector.

## Candidate architectures

The comparison retains all serious candidates:

1. PyTorch/Kineto, vLLM profiler controls, Triton Proton, and imported vendor
   traces;
2. direct CUPTI Activity on NVIDIA;
3. intentional USDT or host-runtime uprobes transported with eBPF;
4. CUPTI activity exported through USDT and eBPF;
5. Neutrino-style programmable GPU probes for named deep-diagnostic questions;
6. ROCProfiler SDK as an independent AMD implementation; and
7. no new native collector.

These are not interchangeable. Host probes observe host-side events, while a
GPU activity API supplies device execution evidence. The hybrid candidate is
therefore CUPTI plus a transport architecture, not eBPF replacing CUPTI.

## Experimental environments

The checked-in control manifest records macOS 26.6.2 on Apple arm64, Python
3.11.4, and source revision `96a65142eda78e0def8630f3824107cdc7eeaca4`.
It detected no CUDA/CUPTI, NVIDIA tools, ROCProfiler, Linux eBPF tools, vLLM,
Proton viewer, or Neutrino executable.

Consequently:

- NVIDIA: **UNTESTED - HARDWARE/TOOLCHAIN UNAVAILABLE**
- AMD: **UNTESTED - HARDWARE/TOOLCHAIN UNAVAILABLE**
- eBPF and USDT transport: **UNTESTED - LINUX HOST UNAVAILABLE**
- vLLM and Proton: **UNTESTED - ENGINE/TOOLCHAIN UNAVAILABLE**

The final framework-revision environment artifact and its checksum are under
`research/native_probes/artifacts/control-macos-arm64-f1384f2/`. It records
framework revision `f1384f2e693c49af0f28c05defaf1909ea43f9d1`; the following
evidence-only commit adds that immutable output without changing the harness.
No accelerator trial
was run, so there is no raw GPU trace and the trial dataset is intentionally
empty. Unsupported preflight results are retained rather than replaced by
synthetic results.

## Methodology

The experiment specification defines five controlled microbenchmarks before
any inference workload: asynchronous eager launch, overlapping versus
serialized streams, graph capture/replay, high-rate buffer stress, and
collector failure. Only after correctness gates pass does it run the vLLM
matrix for eager/graph execution, single/concurrent requests, shared batching,
mixed prefill/decode, and controlled cache state.

Trials use identical workload arguments, warmup, at least five measured
repetitions, process restarts, deterministic counterbalancing, and retained
failures/timeouts. Analysis reports individual values, medians, dispersion,
bootstrap confidence intervals, failure counts, and matched deltas against
profiler-off. Unknown denominators remain unknown.

The required gates are correctness before cost: host/device separation,
overlap preservation, graph attribution, explicit loss accounting, workload
correctness, shared-execution safety, and acceptable deployment constraints.
No weighted score can compensate for failing one of these gates.

## Correctness results

No candidate correctness trial ran on compatible hardware. CPU API timing,
device timing, correlation, stream overlap, graph replay, kernel coverage,
loss behavior, failure isolation, and request/iteration joins therefore remain
**UNKNOWN** in the Stormlog-validated matrix.

Primary documentation establishes what experiments are plausible, not what
Stormlog has validated. CUPTI defines activity records and correlation IDs;
PyTorch/Kineto and Proton expose public profiling surfaces; ROCProfiler exposes
HIP/HSA and kernel dispatch tracing; eBPF can observe selected host functions;
and Neutrino demonstrates programmable device probes in its evaluated setup.
None of those source facts substitutes for the required Stormlog runs.

## Performance/perturbation results

No throughput, goodput, latency, TTFT, TPOT, makespan, capture-pause, or
flush-pause measurements are available. Every comparative perturbation result
is **UNKNOWN**. Published overhead from ProfInfer, Neutrino, or Parca remains an
external result and is not copied into the Stormlog result matrix.

## Resource-cost results

Target CPU, helper CPU, system CPU, RSS, helper RSS, threads, wakeups, event
rate, bytes per second, buffer occupancy, disk I/O, post-processing, and
symbolization costs are all **UNKNOWN**. The harness is able to retain process
CPU/RSS samples and artifact sizes, but only a compatible run can produce a
meaningful comparison.

## Evidence-value comparison

Source review supports these hypotheses for testing:

- public PyTorch/Kineto and current vLLM Proton may already satisfy bounded
  timeline diagnosis without a Stormlog-owned native ABI;
- direct CUPTI may offer a smaller purpose-built record set or different
  capture lifecycle, but no material advantage has been demonstrated;
- intentional engine USDT points may add semantic phase boundaries, while
  arbitrary internal symbols and offsets are too fragile to claim support;
- eBPF alone does not provide asynchronous GPU execution intervals;
- the CUPTI/eBPF hybrid may move transport work out of the target, but adds a
  privileged Linux agent and two independently lossy buffers;
- programmable probes can answer instruction, memory, or value questions that
  activity timelines cannot, but no primary Stormlog workflow has yet shown
  that this granularity is required; and
- ROCProfiler must be tested and supported independently from the NVIDIA path.

These are hypotheses in the source-backed matrix, not experiment outcomes.

## Privilege and deployment results

No privilege or container demonstration ran. Source review indicates that
ordinary in-process activity tracing and a bounded sidecar have a different
security model from Linux BPF. eBPF commonly requires delegated BPF/perf
authority, appropriate kernel policy, process visibility, and device/mount
configuration. A CUPTI/USDT/eBPF hybrid also requires startup injection.

Normal unprivileged Linux and container support is therefore **UNKNOWN** for
every proposed native mode until demonstrated. A root-only success would not
qualify ordinary Stormlog use. Packaging must keep research code out of the
runtime distribution and avoid making native toolchains mandatory dependencies.

## Coexistence results

Coexistence with Kineto, Proton, Nsight Systems, Nsight Compute, and another
CUPTI subscriber is **UNKNOWN**. Current CUPTI documentation describes multiple
Activity subscribers for CUDA 13.3 with an R610-or-newer driver, but that is a
version-specific source claim, not a result for Stormlog or for every profiler
combination. The protocol therefore tests each pair explicitly and never runs
incompatible combinations outside that experiment.

## NVIDIA results

**UNTESTED - NVIDIA HARDWARE/TOOLCHAIN UNAVAILABLE.** Direct CUPTI, Proton,
Kineto CUDA activity, Nsight Systems, Nsight Compute, the hybrid transport, and
Neutrino's NVIDIA path were not executed. PR #237 remains a pinned experimental
reference and is not merged or treated as qualification evidence.

## AMD results

**UNTESTED - AMD HARDWARE/TOOLCHAIN UNAVAILABLE.** ROCProfiler SDK/rocprofv3,
HIP graph behavior, queue overlap, buffer pressure, privilege, and coexistence
were not executed. No NVIDIA result may be generalized to AMD.

## Engine/runtime limitations

Only vLLM is in the proposed first qualification. A vLLM result would not
qualify SGLang, TensorRT-LLM, TensorRT core, or llama.cpp. Current vLLM profiling
documentation makes Proton with CUPTI and CUDA-graph attribution a mandatory
public baseline. ProfInfer's llama.cpp results do not establish stable vLLM
probe points. Arbitrary private symbols or offsets remain unsupported unless a
versioned, intentional interface is available.

Shared batching is a further boundary: one device activity can serve several
requests. Stormlog must retain measured shared execution separately. Any
request allocation is an `ESTIMATE` with a named model and an unattributed
remainder, never measured per-request GPU time.

## Candidate capability matrix

Two machine-readable matrices are committed:

- `research/native_probes/matrices/source_backed.json` records documentation,
  papers, repositories, dates, versions, confidence, and assumptions.
- `research/native_probes/matrices/stormlog_validated.json` resets every cell
  to `UNKNOWN` and links it to the control-host evidence explaining why no
  compatible trial ran.

The validated matrix is generated reproducibly from the source matrix. Only a
real experiment with an environment manifest, exact command, raw artifact,
trial output, and analysis result may promote a cell to `STORMLOG_VALIDATED`.

## Evidence gaps

All hardware-dependent acceptance criteria remain open: public baselines,
trusted vendor traces, direct activity APIs, semantic probes, hybrid transport,
programmable probes, overlap, graphs, induced loss, correlation and activity
coverage, coexistence, privilege/container behavior, failure recovery, and
inference sample sufficiency.

The current work provides the audit, source review, protocol, minimal research
harness, schemas, immutable control manifest, explicit empty trial inventory,
analysis code, both matrices, security analysis, and follow-up plan. It does
not complete issue #118 because the decisive comparative evidence is absent.

## Decision

**DEFER - REQUIRED HARDWARE UNAVAILABLE.**

Do not merge PR #237 as a production integration. Do not select CUPTI, eBPF,
the hybrid, programmable probes, or ROCProfiler from source claims. Keep the
existing Stormlog telemetry and public-profiler import architecture as the safe
interim behavior. This is a defer decision, not evidence that the public path
has won.

Answers to the required comparative questions:

1. **Q1:** Unknown. No custom collector versus public-path experiment ran.
2. **Q2:** Unknown. Candidate-specific incremental evidence is not measured.
3. **Q3:** Unknown. Direct CUPTI was not compared with Proton, Kineto, or Nsight.
4. **Q4:** Possible source-backed value at intentional semantic boundaries;
   unique value for vLLM is unproven.
5. **Q5:** No. A host eBPF probe alone does not measure asynchronous device
   execution; it needs CUPTI, ROCProfiler, or another GPU evidence source.
6. **Q6:** Unknown. Transport cost and privilege trade-offs were not measured.
7. **Q7:** Unknown. Bounded direct CUPTI did not run.
8. **Q8:** Unknown. ProfInfer's llama.cpp design was not reproduced on vLLM.
9. **Q9:** No stable vLLM intentional probe surface was established; internal
   function names and offsets must be treated as version-fragile.
10. **Q10:** Unknown experimentally. Graph attribution must be compared with
    trusted traces and the current public Proton baseline.
11. **Q11:** Unknown. Overlap preservation was not executed.
12. **Q12:** Shared work cannot be copied to each request; #233's membership
    model and an explicit estimated allocation are required.
13. **Q13:** Unknown. Buffer exhaustion and both hybrid loss domains were not
    induced.
14. **Q14:** Unknown. Every profiler pairing still requires a real test.
15. **Q15:** Source review suggests instruction/value-level evidence; its added
    value for Stormlog is unmeasured.
16. **Q16:** No named primary workflow currently proves that value is required.
17. **Q17:** Unknown for production; retain programmable probes as targeted
    offline research until probe-specific perturbation is measured.
18. **Q18:** Separately unqualified. AMD requires ROCProfiler experiments.
19. **Q19:** Unknown. Do not claim normal-user or container support until the
    exact capability and device configuration is demonstrated.
20. **Q20:** No. Without comparative evidence, prior CUPTI investment is not a
    valid reason to select CUPTI.

## Why each alternative was accepted/deferred/rejected

- Public/engine profilers are **accepted only as the interim existing path**.
  Their sufficiency remains to be measured.
- Direct CUPTI is **deferred**, not rejected: it has plausible device evidence
  but no measured incremental value, cost, loss behavior, or coexistence.
- Semantic USDT/eBPF is **deferred**: it may add intentional semantic events but
  cannot replace a device evidence source and has an unproven vLLM surface.
- CUPTI/USDT/eBPF is **deferred**: no transport advantage has been measured to
  justify privileged deployment and added failure/loss domains.
- Neutrino is **deferred to targeted offline diagnosis**: no named primary
  workflow or probe-specific cost result justifies production serving use.
- ROCProfiler is **deferred independently** because AMD hardware is absent.
- A production native integration is **rejected at this stage** because it
  would precede the critical evidence gates.

Evidence that could reverse the defer decision includes a reproducible result
showing a public-profiler evidence gap, a candidate closing it correctly, an
acceptable matched perturbation/resource envelope, explicit loss reporting,
safe failure behavior, viable unprivileged or documented privileged deployment,
and compatible profiler coexistence.

## Recommended Stormlog architecture

For now, retain a layered architecture without new production-native code:

1. default: existing local-first Stormlog telemetry and #233 correlation;
2. bounded diagnosis: orchestrate or import supported engine/public profiler
   artifacts;
3. advanced native research: run direct CUPTI and ROCProfiler prototypes only
   in isolated qualification environments;
4. semantic research: test intentional USDT boundaries separately from device
   timing; and
5. deep offline diagnosis: use detailed counters or programmable probes only
   for named questions.

After comparative trials, adopt only the narrowest layer that passes every
critical gate. NVIDIA and AMD may legitimately require separate implementations.

## Follow-up implementation plan

Do not open a production implementation issue yet. The next issues should be
qualification tasks, split by available laboratory environment:

1. NVIDIA microbenchmark and public-baseline qualification;
2. Linux semantic and hybrid transport qualification on the same NVIDIA host;
3. vLLM inference and shared-execution qualification after microbench gates;
4. independent AMD ROCProfiler qualification; and
5. cross-profiler coexistence, container, privilege, and failure qualification.

Each issue must use the committed protocol and return immutable artifacts. Only
after those results support a decision should an implementation issue name the
selected architecture.

## Reproduction instructions

Use the repository `.venv` and run from the repository root:

```bash
.venv/bin/python -m research.native_probes.cli preflight \
  --host-id <immutable-host-id> \
  --repository "$PWD" \
  --output research/native_probes/artifacts/<host-id>/environment.json

.venv/bin/python -m research.native_probes.cli plan \
  --configuration-id <configuration-id> \
  --vendor <nvidia-or-amd> \
  --workload w1-eager --workload w2-overlap --workload w2-serialized \
  --mode off --mode trusted --repetitions 5 \
  --environment-artifact \
    research/native_probes/artifacts/<host-id>/environment.json \
  --artifact-root research/native_probes/artifacts/<host-id> \
  --output research/native_probes/artifacts/<host-id>/plan.json

.venv/bin/python -m research.native_probes.cli run \
  --plan research/native_probes/artifacts/<host-id>/plan.json \
  --output research/native_probes/artifacts/<host-id>/run-index.json

.venv/bin/python -m research.native_probes.cli initialize-matrix \
  --source research/native_probes/matrices/source_backed.json \
  --environment-artifact \
    research/native_probes/artifacts/<host-id>/environment.json \
  --output /tmp/stormlog-validated-matrix.json
```

Follow `docs/native_probe_experiment_specification.md` for mode-specific exact
commands, immutable configurations, trial count, gates, and workload order.
Run one trial with the research runner, retain stdout/stderr and checksums, then
aggregate the resulting JSONL without removing failures:

```bash
.venv/bin/python -m research.native_probes.cli analyze \
  --input <all-trials.jsonl> \
  --output <analysis.json> \
  --bootstrap-samples 10000
```

The checked-in control commands are preserved verbatim in
`research/native_probes/artifacts/control-macos-arm64/commands.txt`.
