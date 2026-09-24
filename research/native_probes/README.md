# Native Probe Comparative Research

This directory contains research-only code and evidence for Stormlog issue
[#118](https://github.com/Silas-Asamoah/stormlog/issues/118). It does not ship
in the Stormlog Python package and does not select a production collector.

## Layout

- `preflight.py`: secret-minimized environment and capability inventory.
- `planning.py`: reproducible counterbalanced trial ordering and identities.
- `runner.py`: argv-only execution, timeout handling, retained logs, process
  resource sampling, checksums, and immutable trial manifests.
- `workloads/cuda_microbench.py`: W1 eager, W2 overlapping/serialized streams,
  W3 graph replay, and W4 high-event-rate workloads.
- `mode_commands.py`: matched wrappers for profiler-off, PyTorch, Nsight,
  direct CUPTI, ROCProfiler, and detailed counters.
- `references.py`: exact prototype revisions, including extraction of PR #237's
  CUPTI helper without merging its production integration.
- `analysis.py`: individual values, medians, dispersion, bootstrap intervals,
  retained failures, unknown denominators, and matched perturbation.
- `schemas/`: environment, trial, and capability-matrix contracts.
- `matrices/`: theoretical and Stormlog-validated matrices kept separate.
- `artifacts/`: immutable control-host and future hardware evidence.

## Invariants

- Use the repository `.venv` for every Python command.
- Never run commands through a shell.
- Never put credentials in command environment overrides or artifacts.
- Never overwrite an environment, trial, raw trace, or analysis artifact.
- Keep failures, timeouts, partial runs, and unsupported modes.
- Use `null` and `UNKNOWN` when a denominator is unavailable.
- Do not relabel CPU launch time as GPU execution time.
- Do not relabel a shared iteration as measured per-request GPU time.
- Do not copy NVIDIA claims into AMD cells or one engine's results into another.

## Control-host preflight

From the repository root:

```bash
.venv/bin/python -m research.native_probes.cli preflight \
  --host-id control-macos-arm64 \
  --repository "$PWD" \
  --output research/native_probes/artifacts/control-macos-arm64/environment.json
```

The writer uses exclusive creation. Choose a new immutable path for a new
revision or environment.

## Hardware run order

1. Create a clean checkout at the exact research revision.
2. Record preflight and vendor environment details before installing or
   changing profiler dependencies.
3. Pin model, engine, framework, driver, toolkit, profiler, and prototype
   revisions.
4. Build the direct, hybrid, and programmable references outside the `.venv`;
   use `.venv/bin/python` for all Python setup and execution.
5. Run one unmeasured warmup.
6. Run at least five matched repetitions in the generated counterbalanced
   order, restarting target processes between trials.
7. Preserve unsupported cases and failures instead of changing the denominator.
8. Analyze only configurations whose immutable fields match.
9. Review raw measurements before applying decision gates.

## Direct CUPTI reference

PR #237 remains unmerged and is used only as an experimental reference. Extract
its native helper from the pinned commit:

```python
from pathlib import Path

from research.native_probes.references import extract_cupti_reference

source = extract_cupti_reference(Path.cwd(), Path("/tmp/stormlog-cupti-reference"))
print(source)
```

Then build with the CUDA toolkit selected for the environment:

```bash
cmake -S /tmp/stormlog-cupti-reference/native/cupti \
  -B /tmp/stormlog-cupti-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/stormlog-cupti-build --config Release
```

The build is not evidence that runtime behavior is correct.

## Analysis

Concatenate schema-valid trial manifests into JSONL without editing individual
records, then run:

```bash
.venv/bin/python -m research.native_probes.cli analyze \
  --input /path/to/trials.jsonl \
  --output /path/to/analysis.json
```

The analysis output retains individual measurements and failure counts. Raw
traces remain outside Git when large or sensitive; commit cryptographic
checksums and a durable access location instead.
