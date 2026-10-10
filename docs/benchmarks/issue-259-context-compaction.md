---
orphan: true
---

# Issue #259 context compaction

## Real vLLM capture on Modal

On 2026-10-10 the maintainer authorized a fresh small-model Modal capture as
real-trace evidence in place of waiting for the contributor-held #216/#257
artifact. The original audit remains unreproduced; its historical 218,012 GPU
events and 35,380 launch records describe a different workload.

The capture used `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, vLLM 0.30.0,
PyTorch 2.13.0+cu130, Python 3.12.10 and one NVIDIA L4 on Modal.
Driver: 580.95.05. Device UUID recorded with `nvidia-smi`:
`GPU-4518df78-7fac-78ae-8e7d-c45614f29ec4`.

One unprofiled warmup batch preceded two profiled batches of 16 prompts,
using greedy decoding, 64 output tokens per request and EOS ignored.
All 32 requests completed, generating 2,048 tokens. The engine used BF16,
tensor parallelism 1, maximum length and batch tokens 1,024, maximum sequences
16, GPU memory utilization 0.5, seed 259 and prefix caching disabled.
Compilation and CUDA graphs were enabled; stack, shape, memory and FLOP
profiling were disabled. Full prompts, outputs, settings and packages are saved.

The capture function completed and the app stopped.
The script limits each capture to one L4, no retries and a 900-second timeout.

## Paired results

The raw trace contains 50,642 kernels, 574 copies and 280 memsets:
**51,496 GPU events**, including **46,060** from CUDA graphs.
Both details were imported locally through production APIs using the same
trace path, fixed run/session IDs and the actual UUID override.
Stormlog base: `7c94fb01c90e6dd9c48257aea8f3fdc53f91bcd6`, with implementation
changes in the dirty working tree. Local comparison Python: 3.14.7.

| Detail | GPU activity records | Semantic events | Embedded bytes | Compact bytes | Saved bytes | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Launch | 5,562 | 5,565 | 9,598,674 | 6,846,215 | 2,752,459 | 28.68% |
| Kernel | 51,496 | 51,499 | 88,251,421 | 62,761,632 | 25,489,789 | 28.88% |

Each compact file has three context definitions, included in its measured
bytes and excluded from semantic counts. Physical lines: 5,568 launch and
51,502 kernel, against 5,565 and 51,499 embedded lines.

Both comparisons passed exact equality of original imported versus reloaded
compact events, complete embedded versus compact semantic records, resolved
graphs and GPU accounting. Per-device/clock totals also agree between details:
**busy 0.990239929 seconds**, **summed activity 0.990479635 seconds**. Clock domain:
`kineto:modal:kineto:rank0.1791657897067102020.pt.trace.json.gz:9b0a03e873b8`.
Launch spans cover 0.993423839 seconds; interval-based accounting excludes the gaps
and retains the same busy time as kernel detail.

This capture uses vLLM's standard profiler without Stormlog iteration hooks.
All GPU events remain `launch_outside_iteration_range`: zero linked activities,
5,562 unresolved launch records and 51,496 unresolved kernel records. Both files
have zero unmeasured GPU activities. These are preserved attribution limits,
not evidence of request/iteration joining. Dropped CUPTI records are not reported,
so event loss is unknown.

The embedded baseline comes from the **same complete production import**, using
the semantic `to_record()` representation. Both JSONL files use identical
whitespace and no compression. This measures paired encoding size, not runtime
performance or peak reader memory. Recorded profile wall duration includes
export overhead and is not a throughput benchmark. Single-use contexts can grow.

## Artifacts and reproduction

[Machine-readable provenance and results](issue-259-modal-results.json) contain
the configuration, portable reproduction commands, counts, checksums and
invariants. Account-specific storage identifiers and absolute paths are omitted
from this published export; original provenance remains in the evidence bundle.
Large artifacts are ignored by Git under `artifacts/issue-259/<capture-id>/`:

- `traces/rank0.1791657897067102020.pt.trace.json.gz`: unchanged raw trace.
- `capture.json`, `local-provenance.json`, `workload.json`, `outputs.json`:
  provenance and complete workload.
- `packages.txt`, `nvidia-smi.txt`, `nvcc.txt`, `modal-capture.log`: environment.
- `launch/` and `kernel/`: paired JSONL files, run envelopes and `results.json`.
- `scripts/`: exact capture and comparison scripts.
- `issue-259-trace-evidence.tar.gz`: review bundle containing the above evidence.

Raw gzip trace: **2,571,040 bytes**; SHA-256:
`fb0230d22d498cb179567481cb2272893c26234b483a8e0dc4531a0df47268e7`.
Review bundle: **6,505,711 bytes**; SHA-256:
`e6ddecc96cfa9b04adc79ff8fbefdff9b96b773d267177aaade80bfa42d071ff`.
The capture script retains raw traces in a dedicated Modal volume. The review
bundle contains the original provenance and paired encodings.

Capture again with the separately installed Modal CLI:

```bash
modal run --profile YOUR_PROFILE examples/scenarios/modal_context_trace.py
```

The script pins CUDA JIT compiler, CRT and NVVM to 13.0.88, matching PyTorch's
CUDA 13.0 headers, configures `CUDA_HOME` and shared-library links, then compiles
and links a CUDA program during image build. This avoids the missing CUDA path,
compiler/header mismatch and missing unversioned runtime link encountered during
setup. Failed startup attempts produced no benchmark traces. Profiling follows
the pinned [vLLM offline example](https://github.com/vllm-project/vllm/blob/v0.30.0/examples/features/profiling/simple_profiling_offline.py).

Reproduce the offline comparison against the saved trace:

```bash
UV_CACHE_DIR=/tmp/stormlog-uv-cache uv run --no-project --python .venv/bin/python python \
  -m examples.scenarios.context_compaction_scenario \
  --trace traces/rank0.1791657897067102020.pt.trace.json.gz \
  --detail launch \
  --device-uuid 0=GPU-4518df78-7fac-78ae-8e7d-c45614f29ec4 \
  --output-dir /tmp/stormlog-context-compaction-launch
```

Repeat with `--detail kernel` and another fresh output directory. The script
refuses existing outputs, verifies the input checksum remains unchanged and
never overwrites the trace. Retrieve the bundle:

```bash
modal volume get --profile YOUR_PROFILE stormlog-issue-259-traces \
  CAPTURE_DIRECTORY/issue-259-trace-evidence.tar.gz \
  /tmp/issue-259-trace-evidence.tar.gz
```

For regression coverage, reader audit and local checks, see
[Issue #259 implementation validation](issue-259-validation.md).
