[← Back to main docs](index.md)

# Native Probe Integration Contract

This contract defines the hardware-independent handoff for issue #118. It does
not approve a collector and does not turn source documentation into measured
support. The current decision remains **DEFER - REQUIRED HARDWARE UNAVAILABLE**.

## Boundary with Stormlog inference records

High-volume profiler activity stays in an immutable sidecar. It does not enter
`TelemetryEvent` and it does not change the correlation records delivered by
#233. A normalized activity may reference an existing request, iteration,
stage, membership, or artifact identity, but must not copy a shared kernel's
full duration to every request. Allocation models remain named estimates with
an unattributed remainder.

The research harness emits five evidence layers:

1. environment manifest;
2. immutable execution plan and exact argv;
3. raw profiler artifacts plus hashes and producer metadata;
4. trial and normalized manifests;
5. aggregate analysis and capability-matrix promotions.

A `STORMLOG_VALIDATED` matrix cell requires links for all five roles:
environment, command, raw artifact, trial, and analysis. The validator rejects
partial bundles, duplicate roles, path escapes, symlinks, remote-only files,
checksum mismatches, wrong document kinds, and cross-file identity or revision
mismatches. Every promoted source must be locally readable while validation
runs. The matrix cell preserves each role, path, and recomputed SHA-256 in its
`evidence_roles` field.

Promotion also requires a `pass` trial with exit code zero, a valid flushed
measurement window, all required artifacts present, and analysis
`claim_evidence[claim_id]` containing `status: "pass"`, a nonempty `basis`, and
nonempty `criterion` and `observed` fields, plus the linked trial ID and raw
artifact ID. The generic analyzer does not infer graph attribution,
overlap preservation, or loss completeness from a successful process. Add
claim-specific evidence only after reviewing the raw evidence. If that proof
is unavailable, keep the matrix cell `UNKNOWN`.

## Clock and measurement contract

Warmup finishes before the named measurement range begins. Each result records
the range ID, marker mechanism, warmup and measured counts, host start/end
timestamps, clock domain, and flush completion. A missing, reversed, duplicate,
or unflushed boundary makes the normalized trial partial. Host enqueue,
host-through-synchronize, device-event, and profiler clocks remain distinct
until a validated alignment establishes a conversion.

W2-A is only *eligible* for overlap by design. The trusted vendor trace must
show a positive concurrent interval before another collector can be graded for
overlap preservation. W2-B uses identical work and an explicit dependency.
Natural serialization of W2-A makes that configuration unsuitable; it is not a
profiler failure.

PyTorch deliberately reuses `torch.cuda` interfaces on ROCm, and HIP supports
runtime graph capture and replay. W3 therefore attempts the public PyTorch graph
API on both backends. An absent API or runtime rejection is retained as an
explicit unsupported result; NVIDIA graph behavior is never copied into AMD.

## Loss and resource contract

Loss is reported independently for vendor activity production, BPF transport,
profiler export, and artifact storage. Missing denominators remain `null` and
`unknown`; they are never interpreted as zero loss. W4 records its launch count
and pressure controls. Each adapter must additionally record the buffer/output
bounds, consumer or reader delay, flush interval, and any unsupported pressure
control it cannot apply.

Resource sampling separates target, profiler wrapper, helper/agent,
postprocessor, and system roles. A role that cannot be discovered is `unknown`,
not zero. Wrapper cost must not be labeled target cost. Postprocessing runs as a
separate stage so its CPU, memory, I/O, and wall time cannot disappear from the
comparison.

## Adapter ownership

Each mode adapter owns its command, expected raw artifact, producer, format,
sensitivity, and loss-metadata expectation. The runner downgrades an otherwise
successful command to partial when a required artifact is absent. Normalizers
must preserve unknown and unsupported source semantics and may not invent a
universal timeline field that the raw format cannot support.

- NVIDIA trusted timeline: Nsight Systems.
- AMD trusted timeline: `rocprofv3` / ROCProfiler SDK.
- NVIDIA detailed counters: Nsight Compute, treated as perturbed offline data.
- AMD counters: ROCProfiler configuration qualified independently.
- Direct CUPTI: pinned PR #237 helper, experiment-only.
- eBPF and hybrid modes: Linux-only adapters with explicit capability and
  policy evidence.

## End-to-end workflow

Use only the repository `.venv` for Python commands. Every output path is new;
the writer refuses replacement.

```bash
.venv/bin/python -m research.native_probes.cli preflight \
  --host-id <host-id> --repository "$PWD" --output <environment.json>

.venv/bin/python -m research.native_probes.cli plan \
  --configuration-id <configuration> --vendor nvidia \
  --workload w1-eager --workload w2-overlap --workload w2-serialized \
  --mode off --mode trusted --repetitions 5 \
  --environment-artifact <environment.json> \
  --artifact-root <artifact-root> --output <plan.json>

.venv/bin/python -m research.native_probes.cli run \
  --plan <plan.json> --output <run-index.json>

.venv/bin/python -m research.native_probes.cli normalize \
  --trial <trial-manifest.json> --output <normalized.json>

.venv/bin/python -m research.native_probes.cli analyze \
  --input <normalized.jsonl> --output <analysis.json>

.venv/bin/python -m research.native_probes.cli validate-matrix \
  --matrix <unvalidated-matrix.json> --promotions <promotions.json> \
  --repository "$PWD" --output <validated-matrix.json>
```

The plan is the reviewable command inventory. `run` executes only its argv and
declared environment. It retains every trial and writes the run index even if
a command is missing, fails, times out, or produces a partial result; its exit
status is nonzero if any trial is not `pass`. `normalize` validates its input
and preserves raw artifact identities and loss domains. `analyze` accepts only
schema-valid trial/normalized JSONL and groups by configuration, workload, and
mode. Matrix promotion is last and fails closed. CPU fixture integration tests
exercise this chain without promoting hardware claims; see the research README
for the opt-in NVIDIA runtime smoke command.

## Compatibility and versioning

Environment schema version 2 separates installed tooling from accessible
hardware and runtime initialization. Trial, run-index, and analysis records
use schema version 2. Trial v2 uses a canonical directory manifest containing
normalized relative paths, member byte sizes, and per-file SHA-256 values.
Run-index v2 stores trial manifest entries as objects, and analysis v2 uses
configuration/workload/mode group keys with a defined evidence structure.
Promotion validates environment, plan, trial, and analysis against their
schemas, rejects unsupported versions, and requires a structured executed
command in both the linked plan trial and trial manifest. Plan, capability
matrix, and normalized records remain schema version 1. Any incompatible field
change requires a schema-version bump and adapter update; consumers must reject
unknown versions. Raw vendor formats retain their own versions and remain
authoritative evidence.

Version 1 trial, run-index, and analysis evidence is immutable. To obtain v2
evidence, users rerun the current experiment pipeline into fresh immutable
paths. Preserve historical v1 artifacts; do not relabel them in place. No
automatic metadata conversion is supplied, and old directory hashes are not
reinterpreted using the version 2 manifest contract.

The default Stormlog package remains pure Python and gains no CUDA, ROCm, eBPF,
vLLM, or native runtime dependency from this research harness.
