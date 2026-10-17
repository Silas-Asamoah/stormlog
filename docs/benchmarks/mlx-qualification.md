# MLX qualification

Qualified locally on 2026-10-10. This evidence covers the runtime adapter,
Python profiling/tracking, telemetry and optional installation in PR 1.
The PR 1 evidence below is retained as the initial runtime gate. The PR 2/3
section records the completed CLI/TUI rollout and comparative timing evidence.

## Environment and source identity

- Native Apple M5 Pro, arm64, macOS 27.0.1.
- Python 3.14.7, MLX and mlx-metal 0.32.3; the proposed lower bound and the
  current release resolved by `stormlog[mlx]` were both 0.32.3.
- Device recommendation: 42,949,672,960 bytes. Host memory: 51,539,607,552 bytes.
  These are device/host descriptors, not total/free GPU capacity counters.
- An isolated editable `stormlog[mlx]` installation plus pytest/jsonschema had
  no PyTorch, TensorFlow or JAX installed. Package/class/offline imports were
  also tested with imports of all four frameworks explicitly blocked.
- Qualification began at revision `2498e07` with the PR 1 tracking/packaging
  changes in the working tree. Per-module SHA-256 digests are recorded in
  `artifacts/mlx-pr1-qualification/qualification.json` to identify the exact
  tested implementation. Generated artifacts are ignored by Git.

The filesystem sandbox could not initialize a Metal device. The same native
probe and tests succeeded with native device access outside that sandbox.
Hardware tests do not substitute OS/architecture labels for an availability
probe, and an explicitly selected hardware gate fails rather than skips when
Metal is unavailable.

## Controlled memory boundaries

The harness used float32 256 × 256 matrices, explicitly evaluated inputs,
GPU-default-stream completion, and a quiescent process at comparison
boundaries. Cache clearing and GC occurred only in the test harness.
No collector performed these mutations. No model downloads or OOM-sized
allocations were used.

| Boundary | Active bytes | Cache bytes | Lifetime runtime peak bytes |
| --- | ---: | ---: | ---: |
| Idle after explicit cache clear | 0 | 0 | 0 |
| Evaluated input retained | 262,144 | 4 | 262,148 |
| Arrays released | 0 | 786,436 | 786,432 |
| Cache explicitly cleared | 0 | 0 | 786,432 |

Stormlog active/cache readings equalled direct MLX getters at controlled
boundaries. These byte values are observations of this allocator/version,
not fixed assertions for every machine.

Constructing the lazy matrix result left active memory at 262,144 bytes.
Transparent function profiling evaluated that result, returned the same array,
and recorded `completion_verified=true`. The sampled region maximum was
524,288 bytes. The single synchronized host duration was 1,586,958 ns; this is
one functional smoke measurement, not a performance/overhead benchmark.

A sustained small tracking loop recorded 16 samples/events, retained 3 samples,
and counted 13 dropped in-memory samples. All 16 persisted events reloaded into
the same completed session. Disk retention and in-memory retention are separate.

## Reproduction and gates

```bash
python3 -m venv /tmp/stormlog-mlx-isolation
/tmp/stormlog-mlx-isolation/bin/python -m pip install -e '.[mlx]' pytest jsonschema packaging
/tmp/stormlog-mlx-isolation/bin/python -m pytest tests/ -o 'python_files=test_mlx*.py' -m 'not mlx_hardware' -q
/tmp/stormlog-mlx-isolation/bin/python -m pytest tests/test_mlx_hardware.py -m mlx_hardware -v
```

The isolated contract suite passed, and all four native hardware tests passed:
allocator activity/cache lifecycle; returned/state-root evaluation with declared
multiple streams; tracker/sink/session round-trip with bounded retention; and a compiled callable
profiled outside its transform.
The tests record the declared scope and assert semantic behavior without
requiring specific allocator page sizes. Linux CI runs the fake contract suite
on Python 3.10–3.12 without native MLX or any other framework.

Python 3.10–3.12 CI jobs are configured, but were not executed remotely by this
local qualification. The PR 2/3 qualification below covers script/report/query round-trips and
mean/p90 timing comparisons.


## PR 2/3 core rollout qualification

Qualified locally on 2026-10-11 on the same Apple M5 Pro / macOS 27.0.1,
Python 3.14.7 and MLX + mlx-metal 0.32.3. The minimum supported and current
qualified MLX versions coincide in this run. Native operations required access
outside the filesystem sandbox. No PyTorch, TensorFlow or JAX was installed in
the isolated MLX/TUI environment.

[Raw qualification data](mlx-metal-qualification.json) records source revision
`89c2f9119d236f5f083ac97dbf532dc81ee0274f`, per-module source SHA-256 values,
all raw timing windows, controlled memory boundaries and CLI captures. The
scenario was then uncommitted, identified by SHA-256
`a6c9d4dd4c3d17cc621c987b4c603c504799bc5450f8cec52c338fe992c8b028`.
The installed local wheel reported version `0.0.0` (the source-build fallback);
commit and hashes identify the measured source rather than a released package.
Artifact paths in the published JSON are relative to the repository; generated
capture bundles remain under ignored `artifacts/`.

### Matched timing windows

The workload is eager float32 128 × 128 matrix multiplication with evaluated
input, default-GPU-stream completion, warm allocator cache and no cache clears
inside timing windows. Each window uses 20 warmups and 200 completed operations.
Three rounds rotate mode order: off/passive/bounded, passive/bounded/off,
bounded/off/passive. There are 600 measured operations per mode.

| Mode | Host mean (µs) | Host p90 (µs) | Samples | Native window peak (bytes) | Reported `ru_maxrss` (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 181.249 | 221.000 | 0 | 131,072 | 53,051,392 |
| Passive, 2 ms polling | 156.525 | 187.042 | 51 | 131,072 | 53,133,312 |
| Bounded profile per operation | 276.817 | 294.875 | 1,200 | 131,072 | 52,969,472 |

These are synchronized host durations, including explicit completion; they are
not GPU kernel timestamps. Bounded instrumentation intentionally adds completion
boundaries and changes scheduling. Native peaks are explicitly reset by the
standalone harness for each window and include the evaluated-input baseline.
RSS uses the macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss` field, rather
than an isolated per-mode RSS maximum. The observed values are nonmonotonic
across these windows, so they are recorded as platform-reported observations
and are not used to infer each mode's memory overhead. Raw windows also include
independent boundary RSS from psutil; neither RSS measure is additive GPU memory.

The signed passive mean difference is −13.64%. Even with counterbalanced order,
this short workload is sensitive to scheduling/warmup/noise; the negative value
is **not evidence of a speedup or a universal overhead budget**. These results
qualify instrumentation and provide a reproducible local comparison. Repeat
with the target workload/configuration before making performance claims.

### Memory and CLI outcomes

Controlled 128 × 128 boundaries observed active/cache bytes of 0/0 at idle,
65,536/4 for evaluated input and its lazy output, 131,072/4 after profiling
evaluated the output, 0/131,076 after array release, and 0/0 after explicit harness
cache clearing. Direct native getter comparisons matched. Lifetime peak remained
131,072 after release/clear; the profiler returned the original output and
verified its completion.

The real same-interpreter CLI workload produced 70 samples, retained 3, and
reloaded all 70 through `stormlog.query`. Its session completed and offline CLI
analysis returned 0. Separate completed-array workloads verified these cases:

| Workload outcome | CLI/report exit | Session status |
| --- | ---: | --- |
| `SystemExit(7)` | 1 | incomplete |
| Unexpected exception | 1 | incomplete |
| Simulated versioned allocator-OOM signature | 3 | incomplete |
| Actual self-delivered SIGINT | 130 | interrupted |

OOM classification uses a synthetic exception, not a memory-exhausting allocation.
Every failure retained its report and analyzable telemetry. Fake contract tests
also cover disk failure, unavailable/partial counters and recovery boundaries.

### UI, packaging and regression gates

- 120 framework-independent MLX contract tests passed; four required Metal
  hardware tests passed. MLX-only startup and saved-artifact browsing were
  also tested with imports of all four frameworks explicitly blocked.
- 130 TUI/entrypoint tests passed, preserving the default PyTorch selection;
  337 shared packaging/query/telemetry/session/report/utility tests passed.
  A further 168 derived-field/import/docs/capability-matrix/JAX-utility checks
  passed during the final scope audit.
- Native Runtime Info, evaluated MLX Sample, live tracking and shutdown passed
  in the MLX/TUI-only environment. Both runnable cookbook examples (training
  loss/model/optimizer roots and completed token generation) executed natively.
  The [reviewed MLX panel](../tui-mlx-current.svg) comes from this native run.
- A wheel built and included the MLX CLI, adapter and console script. Installed
  `mlxmemprof --help` and `stormlog tui --help` succeeded. The final wheel also
  launched the MLX TUI from outside the checkout with all four framework imports
  blocked; its documentation and artifact controls were visible. Changed Python sources
  parsed with Python 3.10 grammar; this is not execution on Python 3.10.
- Changed-source Black/isort/flake8 and mypy checks passed; the repository Radon
  gate passed with no exceptions and maximum permitted complexity 10.
- Sphinx emitted HTML. Its strict warning gate still reports existing inference
  cross-reference/highlighting issues and the untracked planning documents;
  these are outside this rollout. Python 3.10–3.12 Linux CI is configured, not
  claimed as remotely executed by this local qualification.

```bash
python3 -m examples.scenarios.mlx_memory_scenario --output artifacts/mlx-contracts
python3 -m examples.scenarios.mlx_memory_scenario --hardware --repetitions 200 --output artifacts/mlx-metal
python3 -m examples.cli.capability_matrix --target mlx --mode smoke
```

PRs 1–3 now supply the core Python/CLI/TUI memory integration. The native
optional capability matrix reported PASS (1), SKIP (0), FAIL (0). The separate
PR4 serving milestone covers oMLX, MLX-LM and MLX-Audio endpoint support and
scoped server evidence; it is not implemented by this qualification. Owned
Metal trace capture, array-buffer inspection, Linux MLX CPU/CUDA and distributed
qualification remain optional follow-ups. Endpoint client profiling does not
read an external server's process allocator.

## Base-branch merge verification (2026-10-16)

Merged `release/dev` at `330f7fe` into `mlx-integration2`. The execution-import
test keeps the upstream newer-hook evidence cases and exercises repeat imports
with both embedded and compact context seeds. The affected execution, codec,
MLX, CLI/TUI, telemetry, query, session and documentation selection passed
417 tests; the separate framework-independent MLX suite passed 140 tests.
Repository-wide CI-pinned formatting, lint and mypy (264 source files), plus
the complexity gate, passed. These are portable regression checks; native
measurements above retain their recorded source revisions.
