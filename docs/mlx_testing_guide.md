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

## Feature matrix

| Surface | Core support | Limits |
| --- | --- | --- |
| Python | Snapshots, function/context profiles, passive tracking, phases | Explicit lazy roots; process allocator scope |
| CLI | Info, monitor, track, same-process run, offline analyze, diagnose | Script mode only; no attachment to server allocators |
| TUI | Explicit MLX actions, live tracking, saved profiles and nullable timelines | Live counters belong to the TUI process |
| Persistence | Telemetry v4, profile v1, bounded sinks/OOM bundles, shared reports/query | No native allocation stacks or fragmentation attribution |
| Qualification | Framework-free contracts and controlled Apple Silicon Metal gates | Linux MLX CPU/CUDA and distributed runtimes unqualified |
| PR4 serving | oMLX/MLX-LM text, MLX-Audio TTS/STT, scoped process evidence and offline joins | Selected live model/version pairs qualified; authenticated oMLX stats unqualified |
| Optional follow-ups | Metal traces and array-buffer inspection | Outside the core rollout |

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
Base/offline functionality still works there. The Python APIs and dedicated `mlxmemprof` CLI are available.

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

Tracker JSON and the TUI's MLX JSON export retain a `telemetry_events` array
alongside tracking aggregates. The shared loader recognizes this envelope;
`mlxmemprof analyze mlx-tracking.json` and `stormlog query` can read these
exports directly, including nullable counters and interrupted captures.
The exported summary preserves session status even when the retained window
omits the start event. Offline sample metrics describe those retained events;
use an append-only sink when the full history is needed.

## Verification

```bash
python3 -m pytest tests/ -o 'python_files=test_mlx*.py' -m 'not mlx_hardware' -q
python3 -m pytest tests/test_mlx_hardware.py -m mlx_hardware -v
```

Ordinary tests use an injectable runtime and need no framework installation.
Hardware tests are opt-in; selecting them requires usable Metal and fails if
it is unavailable. Linux CI runs fake contracts with all four frameworks absent.
See [hardware qualification](benchmarks/mlx-qualification.md) for the tested
environment and limits of the evidence. CLI round-trips and comparative timing evidence are recorded there as well.

## Offline analysis (PR 2)

`stormlog.mlx.analyzer.analyze(path, session_id=None)` accepts telemetry JSON,
JSONL, sink directories and MLX profile JSON without importing a framework.
It uses the shared default session policy or selects the requested session.
Active, cache and native lifetime peaks remain separate; unknown values remain
null in the timeline. Growth windows stop at failed samples, recovery boundaries
and long sampling gaps. Positive active growth asks for workload review; it is
not proof of a leak. Cache growth is not a fragmentation finding.

Querying MLX telemetry through `stormlog.query` also requires no framework
initialization. The legacy `stormlog.utils.format_bytes` import remains available
through a neutral formatting helper used by query consumers.

## Script capture and diagnostic bundles (PR 2)

The same-process runner uses `runpy.run_path(..., run_name="__main__")` and
sets the script's directory as the first import path. It passes all workload
arguments through and restores `sys.argv`/`sys.path`, retaining the invocation
working directory. Sampling begins before user code and ends in a `finally`
block. The application owns evaluation of lazy arrays, including final work.
Separate subprocess allocators need their own instrumentation.

Capture bundles contain an environment/capability snapshot, bounded tracking
summary, rotating telemetry sink, diagnostic summary, session manifest and
shared `report.json`. Existing output directories receive a unique child
directory. Active thresholds are opt-in; the watchdog is alert-only. OOM dumps
use bounded shared flight recording with the narrow MLX allocator classifier.

The report stores workload exit status separately: `SystemExit(None/0)` passes;
other integer or text values remain in the payload and map to exit 1. Unexpected
user failure maps to 1, missing scripts to 5, unavailable runtime to 2, observed
OOM to 3, and interruption to 130. Partial bundles are finalized best-effort;
diagnostic failure cannot turn a failed workload into success. An idle diagnose
probe describes this command's own allocator, not another application's memory.

## Dedicated CLI

```bash
mlxmemprof info --json
mlxmemprof monitor --duration 2 --interval 0.1
mlxmemprof track --duration 2 --output artifacts/mlx-track
mlxmemprof run --output artifacts/mlx-run -- script.py --application-flag value
mlxmemprof analyze artifacts/mlx-run --output artifacts/mlx-analysis --json
mlxmemprof diagnose --duration 0.2 --output artifacts/mlx-diagnose --json
```

`monitor`, `track` and `diagnose` observe their own process. Use `run` for a
real script workload. `run` passes arguments after the script untouched; module
mode (`-m`) is deferred. `--json` or `--format json` prints a machine-readable
result. Workload stdout/stderr stays with the workload, so scripts that print
can mix their output with the final CLI JSON; `report.json` is the durable
machine-readable verdict. `analyze` accepts `--session-id` and writes
`analysis.json` plus the shared report. It can return 3 for candidate growth,
gaps or unverified completion; inspect findings rather than treating growth
as a confirmed leak.

Capture options include `--interval`, `--max-history`, `--rollover-max-events`,
`--retention-max-files`, `--max-buffer-bytes`, `--job-id`, `--rank`,
`--local-rank` and `--world-size`. The first two history/retention limits apply
independently to memory and disk. Disk retention can remove old segments.
`--alert-threshold-mb` is an opt-in active allocation alert in MiB. Flight
recording limits are `--oom-max-dumps` and `--oom-max-total-mb`. No cleanup,
evaluation or peak reset flags are supplied for passive capture.


## TUI and saved artifacts

```bash
pip install 'stormlog[mlx,tui]'
stormlog tui --framework mlx
```

The MLX tab exposes explicit runtime information, a small evaluated sample,
profile refresh/clear, and loading saved profile JSON, tracker JSON, JSONL,
telemetry directories or capture bundles. Startup and offline browsing do not
initialize any framework. Runtime Info, Sample, and Start Live Tracking are
explicit native actions; unavailable Metal produces a useful error and never
falls back to a CPU tracker. The default framework remains PyTorch.

Select **Use MLX for Tracking** or start with `--framework mlx` to use MLX in
Monitoring. Live counters belong to the TUI process. Active allocator bytes,
reusable cache, host RSS, and native lifetime peak have separate labels. Sampled
profile peaks and completion verification are shown separately. Unknown values
remain N/A or gaps; GPU capacity percentages, fragmentation and automatic
cleanup are unavailable for this adapter. Stopping or closing the app stops its
sampler. Saved artifacts display without native MLX even on other platforms.

For runnable training and generation recipes, see the [MLX cookbook](cookbook/mlx.md).
From a source checkout, run `python3 -m examples.basic.mlx_demo --output artifacts/mlx-demo`.
Serving integration is a separate planned PR4 milestone for oMLX, MLX-LM and
MLX-Audio. Metal trace capture and array-buffer attribution remain optional
follow-ups. This core memory rollout introduces no inference wire or memory
schema bump.


## Inference server qualification (PR4)

The [inference guide](mlx_inference.md) documents endpoint profiling and
same-process capture. The [qualification matrix and artifacts](benchmarks/mlx-inference-qualification.md)
record actual versions, models, fixtures, serving PIDs, off/on measurements
and unsupported combinations. Client profiling and offline joins require no
MLX or server imports. Only native capture needs the serving interpreter's
MLX >=0.32.3, Apple Silicon and a usable Metal device.

Protocol/audio/transport/statistics/evidence tests use hardware-free fixtures:

```bash
python3 -m pytest tests/test_infer_mlx_text.py tests/test_infer_http_transport.py tests/test_infer_audio*.py tests/test_infer_omlx_stats.py tests/test_infer_mlx_evidence.py tests/test_infer_mlx_report.py tests/test_infer_mlx_server.py tests/test_infer_mlx_qualification_helpers.py -q
```

The source-only scenario client needs explicit already running endpoints,
pre-provisioned model IDs and authored manifests. It never launches a service,
downloads weights or resets state. The separate server qualification example
reads bounded model-call checkpoints in the serving PID; use identical settings
for off/on pairs. Keep warmup, request counts, seed, fixtures, cache policy and
dependency versions fixed. Record clock/identity provenance and sampled scope.

For the full live gate, set `STORMLOG_INFER_MLX_QUALIFICATION_CONFIG` to a JSON
file mapping `omlx`, `mlx-lm`, `mlx-audio` to lists of scenario CLI arguments
(omit `--engine`, supplied by the test). Include `--base-url`, `--server-version`,
`--run-id`, a fresh `--output`, `--telemetry-state` and each model/manifest.
oMLX additionally needs `--omlx-stats-url` and a separately supplied normally
issued admin credential via `--omlx-stats-cookie-env` or the explicit gateway
bearer environment option. Credential values must never appear in the file.

```bash
python3 -m pytest tests/test_infer_mlx_qualification.py -m infer_mlx_qualification -q -ra
```

Requested qualification fails on missing runtime, Metal, configuration, endpoint
or valid stats; it never reports success with every case skipped. Ordinary CI
runs the framework-free contracts. The manual `MLX inference qualification`
workflow uses a dedicated self-hosted macOS/ARM64 runner whose operator has
prepared the environment and endpoints. It does not install models or reset
services. Successful authenticated stats are still unqualified locally at the
user's request; the complete live gate remains strict for future qualification.
