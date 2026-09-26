[← Back to main docs](index.md)

# Native Probe Follow-up Issue Drafts

Do not open an implementation issue until hardware results support one of the
conditional drafts below. Replace every `[EVIDENCE]` marker with immutable
artifact links and remove alternatives that did not pass their gates.

## Hardware experiment issue

### Title

Run the issue #118 all-candidate native probe comparison on NVIDIA and AMD

### Scope

- Provision one supported NVIDIA Linux and one supported AMD Linux environment.
- Execute W1 through W4 and the declared vLLM workloads for all feasible modes.
- Preserve exact environments, commands, raw artifacts, checksums, individual
  trials, failures, unsupported cases, and analysis.
- Run privilege, container, failure, and coexistence matrices.
- Publish separate NVIDIA and AMD findings and an evidence-linked decision.

### Completion

Complete only when the validated capability matrix links every changed cell to
an environment manifest, exact command, raw artifact, trial manifest, and
analysis result. A lane that skips every GPU case is not complete.

## Conditional public-profiler integration issue

Open only if `[EVIDENCE]` shows that Kineto, Proton, and vendor trace import
answer the intended questions without a custom collector.

### Title

Add bounded public-profiler orchestration and trace import for inference

### Scope

- implement #216's bounded capture/import lifecycle;
- support current PyTorch and vLLM profiler controls, including Proton where
  qualified;
- import trusted vendor traces with explicit clocks, loss, and provenance;
- join only through #233 identities and preserve shared remainder;
- add no native default dependency.

## Conditional direct native issue

Open only if `[EVIDENCE]` shows that direct CUPTI or ROCProfiler provides unique
required evidence or materially lower cost than public imports.

### Title

Implement qualified bounded native GPU activity capture

### Scope

- define separate NVIDIA CUPTI and AMD ROCProfiler backends;
- adopt only configurations that passed fidelity, loss, perturbation,
  privilege, container, packaging, and coexistence gates;
- keep high-volume activity in bounded sidecars;
- preserve explicit unavailability and failure isolation;
- do not claim unsupported vendor, OS, architecture, or engine parity.

## Conditional hybrid issue

Open only if `[EVIDENCE]` shows that CUPTI-to-USDT/eBPF reduces total collection
or transport cost enough to justify its authority and deployment burden.

### Title

Implement qualified Linux CUPTI-to-USDT/eBPF transport

### Scope

- retain CUPTI as the GPU evidence source;
- define a stable versioned USDT record contract;
- use least-privilege BPF/perf delegation and narrow target scoping;
- report CUPTI and BPF transport loss separately;
- keep a direct bounded fallback for users without BPF authority if supported by
  the evidence.

## Conditional programmable-probe issue

Open only for the named question that `[EVIDENCE]` proves cannot be answered by
an activity timeline or detailed vendor tool.

### Title

Add targeted offline programmable GPU probe workflow for [NAMED QUESTION]

### Scope

- support only the qualified kernels, builds, probes, and hardware;
- label it offline/developer-only unless production trials pass;
- preserve original activity traces alongside instruction-level output;
- state probe-specific perturbation and rejected coverage.

## Defer or reject outcome

If no candidate passes critical gates, close #118 with an evidence-linked defer
or reject decision. Document what future change could reverse it, such as a
stable engine probe surface, new public profiler capability, lower-overhead
vendor API, or acceptable deployment model.
