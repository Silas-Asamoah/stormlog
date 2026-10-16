# MLX memory instrumentation

The initial target is native Apple Silicon macOS with an available MLX Metal
runtime. Linux CPU/CUDA MLX and Intel macOS are outside this qualification.
Stormlog observes the MLX allocator **in the current process**, across its
application threads. It does not attach to another training process or server.

Importing `stormlog.mlx` and retrieving its class definitions does not initialize
MLX, PyTorch, TensorFlow, or JAX. Constructing a runtime-dependent object checks
platform, MLX version (minimum 0.32.3), required core APIs, and Metal availability.
Missing MLX, an unsupported platform, a broken native loader, and unavailable
Metal produce distinct exceptions with chained causes. Successful runtime
discovery is cached; sampled counters and failed discovery are not cached.

## Counter meanings

| Value | Meaning |
| --- | --- |
| `active_bytes` | Native MLX active allocator bytes; telemetry `allocator_allocated_bytes`. |
| `cache_bytes` | Reusable cached buffers; namespaced `metadata.mlx.cache_bytes`. |
| `runtime_peak_bytes` | Native lifetime peak since startup or the most recent external reset. |
| `allocator_held_bytes` | Active + cache from sequential reads; not an atomic snapshot. |
| `memory_limit_bytes` | Read-only runtime limit, when the getter exists. |
| `recommended_working_set_bytes` | Device recommendation; not free capacity or total VRAM. |
| `process_rss_bytes` | Independent host RSS; do not add it to allocator counters. |

Unified memory can share CPU/GPU pages. Generic reserved/active/inactive block
counters and device used/free/total counters remain null. MLX active memory is
not CUDA allocator block-active memory. Cache is not fragmentation. No GPU-only
array attribution, live array census, allocation stack history, or global GPU
memory utilization is implied.

Telemetry uses the existing v4 schema with explicit complete capability
metadata. Generic `supports_bounded_profiling` remains false under its existing
native allocator contract; MLX bounded sampling and host timing are advertised
separately in `metadata.mlx.capabilities`. Optional getter failures preserve
successful fields and record reasons. An active-getter failure is unavailable,
never zero. Sampling is sequential and records collection duration.

## Bounded Python profiling

```python
import mlx.core as mx
from stormlog.mlx import MLXMemoryProfiler

profiler = MLXMemoryProfiler()
x = mx.ones((256, 256), stream=mx.gpu)
mx.eval(x)  # input preparation and warmup belong outside the region
output = profiler.profile_function(lambda: x @ x.T, name="matmul")
print(profiler.get_results()[0].to_dict())
profiler.export("mlx-profile.json")
```

`profile_function` executes once, evaluates its returned tree, and returns the
original output unchanged. Optional `state_getter=lambda: (model.state,
optimizer.state)` roots are evaluated after the call. For a custom result
container, supply `root_extractor`; arbitrary object graphs are not traversed.
The profiler retains scalar results/snapshots, never output arrays or state.

```python
with profiler.profile_context("train_step") as region:
    loss, grads = loss_and_grad(model, batch)  # illustrative application code
    optimizer.update(model, grads)
    region.evaluate(loss, model.state, optimizer.state)
```

Contexts cannot discover local lazy arrays. Without `region.evaluate(...)` or
an explicit `already_evaluated=True` declaration, `completion_verified` is false.
Evaluation covers the supplied roots and their dependencies; unrelated stream
work is outside that claim. Pass `streams=(stream_a, stream_b)` to synchronize
those streams before and after the measured region. The default is the GPU's
default stream. Timing uses `perf_counter_ns()` and measures synchronized host
elapsed time, including explicit root evaluation, rather than GPU kernel time.
Place wrappers outside `mx.compile` and `mx.value_and_grad` transforms.

The default `peak_mode="sampled"` does not reset native runtime counters. The
sampled-window maximum includes baseline/final and passive intermediate reads;
short-lived peaks can be missed. It is separate from the lifetime native peak.
`peak_mode="reset"` explicitly resets the shared native counter under an
exclusive process-local lease. Nested/overlapping reset scopes fail. Unrelated
MLX threads and external resets are not isolated. The reported reset-window
peak includes baseline active memory even if the runtime reset writes zero.
Cleanup preserves the original exception, including `SystemExit` and SIGINT.

`sampling_interval` must be positive and finite, and `max_history` bounds
retained samples without truncating peak/count aggregates. Retrieve results
with `get_results()` and discard them with `clear_results()`. Convenience
`profile_function` decorators and `profile_context` use a lazily constructed
global profiler; clearing that helper never resets native peak state.

Profile exports use `format="stormlog.mlx.profile"`, `schema_version=1` and
[the published schema](schemas/mlx_profile_v1.schema.json). Offline validation
with `stormlog.mlx.profile_artifact.load_profiles` requires no native MLX.
Tracker exports use telemetry v4; these are different artifact contracts.

## Passive tracking and persistence

Install with `pip install 'stormlog[mlx]'`. The optional dependency is included
only on Darwin arm64; on other platforms the extra installs no native MLX wheel.
Base/offline functionality still works there. This first rollout ships Python
APIs; `mlxmemprof` commands and dedicated TUI support are later PRs.

```python
from pathlib import Path
from stormlog.mlx import MemoryTracker
from stormlog.telemetry_sink import TelemetrySinkConfig

tracker = MemoryTracker(
    sampling_interval=0.1,
    max_history=1000,
    telemetry_sink_config=TelemetrySinkConfig(
        Path("mlx-telemetry"), max_buffer_bytes=4 * 1024**2,
    ),
)
with tracker.tracking():
    with tracker.phase("train", {"epoch": 1}):
        train_or_generate()  # the application owns evaluation and its streams
tracker.export("mlx-tracking.json")
```

Manual `start_tracking()` / `stop_tracking()` are supported. Starting twice
keeps one sampler; restarting after stop creates a fresh session. Sampling
uses monotonic deadlines and interruptible waits. Core collection failures
produce nullable events, exponential retry backoff, collector health changes,
and recovery evidence. First samples and recovery after a gap have null
allocator deltas. Optional-counter failures preserve valid active measurements.

Sampling never calls evaluation, synchronization, peak reset, cache clearing,
GC, or allocator limit setters. Byte thresholds configured through
`alert_threshold_mb` emit bounded, alert-only records; they are disabled by
default. Session/job/rank identity comes from the shared utilities without
initializing MLX distributed execution.

Sample/event/alert windows are bounded by `max_history`. The returned
`TrackingResult` includes full-session aggregates and retained/dropped counts.
Append-only sinks have independent disk retention/rotation policies and expose
flush/drop diagnostics. Use a bounded sink buffer for resilience to disk
failures. The tracker stops its sampler and closes its sink on context exit;
user exceptions and interrupts propagate with incomplete/interrupted sessions.
Phase handles cannot append after their session stops.

Optional `enable_oom_flight_recorder=True` records a bounded pre-failure window.
The MLX classifier recognizes the versioned native allocator's
`[malloc] Unable to allocate ... bytes.` signature and host `MemoryError`.
Resource-count exhaustion, oversized single-buffer requests and compiler
errors remain distinct from memory exhaustion. No real OOM allocation is used
in tests. Recorder limits and diagnostic failures cannot replace a workload's
original exception. The MLX path supplies neutral host information; existing
PyTorch/TF classification and environment payloads are preserved.

Saved telemetry remains readable through `stormlog.telemetry` without MLX.
For a sink directory use `load_telemetry_sessions("mlx-telemetry")`; for tracker
JSON exports use the same loader. Default session selection follows the shared
completed/interrupted/incomplete policy.

## Verification

```bash
python3 -m pytest tests/ -o 'python_files=test_mlx*.py' -m 'not mlx_hardware' -q
python3 -m pytest tests/test_mlx_hardware.py -m mlx_hardware -v
```

Ordinary tests use an injectable runtime and need no framework installation.
Hardware tests are opt-in; selecting them requires usable Metal and fails if
it is unavailable. Linux CI runs fake contracts with all four frameworks absent.
See [PR 1 hardware qualification](benchmarks/mlx-qualification.md) for the tested
environment and limits of the evidence. Full CLI/TUI and overhead qualification
belongs to later rollout PRs.
