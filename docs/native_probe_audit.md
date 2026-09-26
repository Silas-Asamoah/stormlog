[← Back to main docs](index.md)

# Native Probe Research Audit

This audit resets [issue #118](https://github.com/Silas-Asamoah/stormlog/issues/118)
to its intended evidence-first sequence. It was prepared from the issue and pull
request bodies, comments, commits, diffs, current `release/dev` tree, and primary
sources reviewed on 23 September 2026.

## Status vocabulary

- **Confirmed:** directly established by repository state, a GitHub record, or
  a cited primary source.
- **Strong inference:** the best explanation of confirmed facts, but not a
  measured Stormlog result.
- **Hypothesis:** a proposition owned by the comparative experiments.
- **Unknown:** evidence is unavailable or the required experiment has not run.

## A. What issue #118 requires

**Confirmed:** #118 is a research and technology-selection issue. Its original
deliverables are to define concrete problems, compare capability and operational
risk, explain schema and collector-health fit, and recommend prototype, defer,
or reject. The later inference criteria require experimental comparison of a
public profiler, host probes, CUPTI activity, and programmable probes. They also
require separate CPU and GPU time, stream and graph behavior, event loss,
perturbation, resource cost, coexistence, deployment requirements, and separate
NVIDIA and AMD claims.

The issue does not authorize selecting or productionizing CUPTI before those
comparisons. It explicitly states that a negative result is valid and that
native probes are optional research, not a blocker for the inference roadmap.

The concrete decision question is:

> What is the lowest-cost, operationally acceptable way for Stormlog to obtain
> the specific evidence its current telemetry and supported public profiler
> paths cannot provide?

## B. What PR #236 proved

PR #236 contains three documentation commits:

| Commit | Scope |
| --- | --- |
| `b878ac3a640f9bb2c14f7070752304f50176a565` | Feasibility review and preliminary capability table |
| `b0473e1f9b76b756ea0d0f8cf8e6b679cbb3c23a` | Qualification protocol |
| `152bcb79bc1fdd820368426c200f5457aa6db3b4` | Documentation regression checks |

**Confirmed:** the PR established the following useful foundations:

- CPU API intervals and device execution intervals are different evidence.
- Stream overlap, graph replay, loss accounting, and clock domains require
  explicit validation.
- Shared batching prevents treating a whole kernel duration as measured
  per-request GPU time.
- Native activity should remain a bounded sidecar rather than enter
  `TelemetryEvent v4`.
- NVIDIA and AMD require independent toolchains and support claims.
- Public profiler, trusted vendor trace, host probe, activity, programmable,
  and detailed-counter modes need different evidence labels.
- At least five measured trials, raw artifacts, exact commands, immutable
  manifests, failures, and unsupported modes should be retained.

**Confirmed:** PR #236 contained no Stormlog native-probe performance,
correctness, event-loss, graph, stream, deployment, privilege, or coexistence
measurement. Its own PR body acknowledges this boundary.

## C. What PR #236 inferred

The document selected a direct CUPTI prototype as the first implementation and
deferred eBPF/USDT, programmable probes, and AMD implementation before a fair
comparison was run.

The following were reasonable hypotheses, not proved decisions:

- direct CUPTI would add material evidence beyond Kineto, Proton, or imported
  Nsight traces;
- bounded userspace file output would be operationally preferable to a
  CUPTI-to-USDT/eBPF transport;
- eBPF transport should be considered only if direct CUPTI output was a
  measured bottleneck;
- programmable probes did not answer a sufficiently important Stormlog
  question;
- a separately shipped native helper was worth its build, ABI, distribution,
  security, and support burden.

The new source-backed matrix preserves these as `INFERENCE`, `UNKNOWN`, or
`VERSION_DEPENDENT`, never as Stormlog-validated support.

## D. What PR #237 implemented without empirical qualification

PR #237 is an 11-commit stack ending at
`d4ea783b7ef98aa2f0e3bfbcffc169e4612fd3a9`. Relative to PR #236 it adds 4,269
lines across 26 files. The implementation includes:

- a C++17 CUPTI startup-injection library;
- driver, runtime, concurrent-kernel, memory, synchronization, and graph
  activity normalization;
- bounded native trace and status formats;
- Python capture, timeout, cancellation, import, health, attachment, and CLI
  integration;
- four persisted schemas;
- wheel and source-distribution decisions;
- Linux native CI and 179 focused tests;
- extensive security and failure hardening.

**Confirmed:** the code validated compilation, static contracts, synthetic
lifecycles, packaging, and import security. It did not validate on real GPU
hardware:

- device timestamp fidelity;
- host/device correlation coverage;
- stream overlap preservation;
- CUDA graph replay and node identity;
- induced activity loss and loss denominators;
- workload perturbation or helper resource cost;
- public-profiler evidence equivalence;
- direct-versus-hybrid transport cost;
- profiler coexistence;
- vLLM shared-batch joins;
- normal-user or container deployment;
- any AMD capability.

PR #237 is therefore preserved as the pinned direct-CUPTI research prototype,
not treated as the selected production design. The comparative framework
extracts only its `native/cupti` source at the exact revision without merging
the production Python, CLI, schema, or packaging integration.

## E. Experimentally unanswered acceptance criteria

Every material empirical criterion from #118 remains open:

| Criterion | Current status |
| --- | --- |
| Existing public profiler evidence measured | **Unknown** |
| Current vLLM Proton path compared | **Unknown** |
| Trusted NVIDIA timeline established | **Unknown** |
| Trusted AMD timeline established | **Unknown** |
| Host semantic probes tested on vLLM | **Unknown** |
| Direct CUPTI tested on supported NVIDIA hardware | **Unknown** |
| Direct ROCProfiler tested on supported AMD hardware | **Unknown** |
| CUPTI-to-USDT/eBPF hybrid compared | **Unknown** |
| Named programmable probe evaluated | **Unknown** |
| CPU and device intervals verified against ground truth | **Unknown** |
| Overlap and graph replay fidelity measured | **Unknown** |
| Loss deliberately induced and surfaced | **Unknown** |
| Coexistence matrix executed | **Unknown** |
| Privilege and container behavior demonstrated | **Unknown** |
| vLLM shared-execution joins measured | **Unknown** |
| Matched perturbation and resource-cost trials completed | **Unknown** |
| Evidence-linked architecture selected | **Unknown** |

This control host is Apple M1/macOS. It has no CUDA, CUPTI, NVIDIA GPU,
ROCm, AMD GPU, Linux eBPF, vLLM, Nsight, or ROCProfiler installation. The
hardware cells must remain `UNTESTED - HARDWARE UNAVAILABLE` here.

## F. Reusable qualification work

The useful portions of PR #236 are preserved conceptually and strengthened in
the new experiment package:

- W1 eager launch and execution separation;
- W2 overlap plus serialized control;
- W3 graph capture and replay plus eager control;
- W4 high-event-rate stress and explicit expected counts;
- failure, timeout, cancellation, capacity, partial-flush, permission, and
  version-mismatch cases;
- warmup, process restarts, five or more measured trials, counterbalanced order,
  retained failures, and individual observations;
- distinct fidelity, perturbation, collector-resource, and operational-cost
  dimensions;
- explicit denominators and `UNKNOWN` when a denominator is unavailable;
- raw, normalized, metric, log, environment, and command artifacts;
- a trusted vendor trace used as a correctness reference, not an always-on
  deployment dependency.

The new implementation adds immutable writes, deterministic identities,
argv-only execution, process-tree sampling, exact checksums, matched off/on
comparisons, and schema-validated source and experimental matrices.

## G. Premature conclusions in the previous feasibility document

The following prior statements are removed or downgraded:

- `Prototype CUPTI Activity collection first` was a selection before the
  public-profiler, hybrid, semantic, programmable, and no-collector candidates
  were measured.
- `Defer eBPF/USDT transport unless CUPTI transport is a measured bottleneck`
  assumed the direct architecture should be the default comparator. Direct and
  hybrid must instead be measured as peer candidates.
- `Defer programmable probes` lacked a named-question experiment or an
  evidence-based finding that no Stormlog question needs instruction detail.
- The capability table mixed primary-source facts, published external results,
  and design inferences in one presentation.
- The decision section used a proposed architecture as if the comparative
  research had already narrowed the field.

## H. Stale statements

The previous document says that #211's correlation contract is not present.
That statement is now false.

**Confirmed:** PR #233 merged into `release/dev` as commit
`208f300d2c03d86188ca546ef3a12400651b8eb9`. The repository now contains:

- versioned request, iteration, membership, stage, activity-reference, and
  artifact identity records;
- separate engine-adapter and trace-collector interfaces;
- shared-execution accounting that records an iteration once;
- merged GPU interval duration distinct from summed activity duration;
- named estimated request shares and an unattributed remainder;
- optional raw trace attachments and explicit unresolved links.

The experiment must reuse this contract. It must never copy a shared kernel's
full duration to every participating request.

Other stale statements include the assumption that vLLM's public profiler path
was limited to the previously reviewed surface. Current vLLM documentation now
describes Triton Proton on NVIDIA through CUPTI, repeated bounded profiling, and
CUDA graph attribution with Triton 3.7+. That makes Proton a required baseline
and creates a direct burden of proof for a custom CUPTI collector.

## Disposition

- Do not merge PR #236 as the issue #118 decision.
- Do not merge PR #237 as a selected production collector.
- Preserve both branches and commits for audit and prototype reference.
- Treat #234 as a premature implementation selection.
- Supersede #235's CUPTI-first qualification framing with the all-candidate
  experiment specified in this branch.
- Do not manufacture a revert in `release/dev`; neither PR is merged there.
- Do not open a new production implementation issue until measured evidence
  supports the final architecture.

## Audit sources

- [Issue #118](https://github.com/Silas-Asamoah/stormlog/issues/118)
- [Issue #234](https://github.com/Silas-Asamoah/stormlog/issues/234)
- [Issue #235](https://github.com/Silas-Asamoah/stormlog/issues/235)
- [PR #236](https://github.com/Silas-Asamoah/stormlog/pull/236)
- [PR #237](https://github.com/Silas-Asamoah/stormlog/pull/237)
- [Issue #211](https://github.com/Silas-Asamoah/stormlog/issues/211)
- [PR #233](https://github.com/Silas-Asamoah/stormlog/pull/233)
- [Issue #216](https://github.com/Silas-Asamoah/stormlog/issues/216)
- [Issue #221](https://github.com/Silas-Asamoah/stormlog/issues/221)
- [Inference roadmap #210](https://github.com/Silas-Asamoah/stormlog/issues/210)
