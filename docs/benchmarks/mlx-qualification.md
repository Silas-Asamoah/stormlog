# MLX PR 1 qualification

Qualified locally on 2026-10-10. This evidence covers the runtime adapter,
Python profiling/tracking, telemetry and optional installation in PR 1.
CLI, dedicated TUI and comparative overhead qualification remain later rollout
work. No claim of full MLX integration completion is made.

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
local qualification. Full CLI script/report/query and mean/p90 passive versus
bounded overhead comparisons are deferred to the rollout that ships those
features.
