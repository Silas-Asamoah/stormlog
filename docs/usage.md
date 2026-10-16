[← Back to main docs](index.md)

# Usage Guide

This guide focuses on the workflows that are stable in the current codebase:

- profile a bounded section of code
- profile an OpenAI-compatible inference endpoint
- track memory over time
- export telemetry and plots
- move from runtime data to a diagnose bundle or TUI session

If you want task-oriented production recipes instead of a capability guide, use
the [Production Cookbook](cookbook/index.md), especially
[Always-on Tracking](cookbook/always_on.md),
[PyTorch Production Recipes](cookbook/pytorch.md), and
[TensorFlow Production Recipes](cookbook/tensorflow.md).

Install the distribution as `stormlog`, then import the Python APIs from
`stormlog`, `stormlog.tensorflow`, `stormlog.jax`, or `stormlog.mlx`. The framework memory CLI
automation commands are `gpumemprof`, `tfmemprof`, `jaxmemprof`, and `mlxmemprof`; local
artifact queries live under `stormlog query`, and endpoint inference profiling
lives under `stormlog infer`.

## Choose the right tool

### `GPUMemoryProfiler`

Use when:

- you have a PyTorch runtime exposed through `torch.cuda` (CUDA or ROCm)
- you want per-call or per-context profiling
- you care about allocated vs reserved GPU memory during a bounded operation

### `TFMemoryProfiler`

Use when:

- you are profiling TensorFlow code directly
- you want snapshots plus aggregated TensorFlow profiling results

### `JAXMemoryProfiler`

Use when:

- you are profiling JAX code directly
- you want memory usage insights taking XLA compilation into account
- you need snapshots of JAX device allocations over time

### `MemoryTracker`

Use when:

- you need telemetry over time rather than one profiled call
- you want backend-aware tracking on CUDA, ROCm, MPS, or CPU fallback paths
- you want alerts, exported event streams, or diagnose bundles

### `stormlog infer`

Use when:

- you want to profile an OpenAI-compatible Chat Completions endpoint
- the backend may be vLLM, SGLang, TensorRT-LLM, MLX-LM, a hosted gateway, or
  another OpenAI-compatible server
- you care about client-observed TTFT, latency, throughput, token rates,
  failures, and optional host/GPU telemetry

### `CPUMemoryProfiler` / `CPUMemoryTracker`

Use when:

- you are on a CPU-only machine
- you want the same workflow shape without CUDA-specific profiling

## Canonical workflow

For a new environment or a new project, use this sequence:

```bash
gpumemprof info
gpumemprof track --duration 2 --interval 0.5 --output /tmp/gpumemprof_track.json --format json
gpumemprof analyze /tmp/gpumemprof_track.json --format txt --output /tmp/gpumemprof_analysis.txt
gpumemprof diagnose --duration 0 --output /tmp/gpumemprof_diag

tfmemprof info
tfmemprof diagnose --duration 0 --output /tmp/tf_diag

jaxmemprof info
jaxmemprof monitor --duration 2 --interval 0.5 --output /tmp/jaxmemprof_monitor.json
jaxmemprof analyze --input /tmp/jaxmemprof_monitor.json --detect-leaks --optimize --report /tmp/jaxmemprof_analysis.txt
jaxmemprof diagnose --duration 0 --output /tmp/jax_diag
```

This gives you:

- environment visibility
- a short telemetry sample
- a readable analysis artifact
- a portable diagnose bundle you can load later

For long-running captures, Stormlog now treats every `track` run or standalone
`diagnose` bundle as its own session. That makes it possible to reuse the same
sink directory across runs and still reconstruct or audit one capture at a
time.

If you are working from a source checkout, you can optionally add the maintained
example smoke paths:

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

The `examples/` package is not included in the PyPI distribution.

## OpenAI-compatible inference profiling

Use `stormlog infer profile` to send controlled traffic to a Chat Completions
endpoint and write a JSONL artifact:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,4,8 \
  --input-tokens 512,2048 \
  --output-tokens 128,512 \
  --requests 20 \
  --output /tmp/stormlog_infer.jsonl
```

Analyze the artifact later:

```bash
stormlog infer analyze /tmp/stormlog_infer.jsonl
```

Install optional tokenizer dependencies when server usage metadata is missing
and you need stronger fallback token counts:

```bash
pip install "stormlog[infer-tokenizers]"
```

See the [Inference Profiling Guide](inference.md) for metric boundaries,
token count provenance, streaming behavior, and system telemetry options.

## PyTorch profiling

`GPUMemoryProfiler` currently targets `torch.cuda` runtimes. That includes
NVIDIA CUDA builds and ROCm-backed PyTorch builds surfaced through
`torch.cuda`. If you are on Apple MPS or CPU-only hardware, move to
`MemoryTracker`, the CLI, or the CPU-only path below.

```python
import torch
from stormlog import GPUMemoryProfiler

profiler = GPUMemoryProfiler(track_tensors=True)
device = profiler.device
model = torch.nn.Linear(1024, 256).to(device)

def train_step() -> torch.Tensor:
    inputs = torch.randn(64, 1024, device=device)
    outputs = model(inputs)
    return outputs.sum()

profile = profiler.profile_function(train_step)
summary = profiler.get_summary()

print(profile.function_name)
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
```

### Context profiling

```python
import torch
from stormlog import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
device = profiler.device

with profiler.profile_context("forward_pass"):
    x = torch.randn(32, 1024, device=device)
    y = torch.nn.Linear(1024, 128).to(device)(x)
```

## TensorFlow profiling

```python
from stormlog.tensorflow import TFMemoryProfiler

profiler = TFMemoryProfiler(enable_tensor_tracking=True)

with profiler.profile_context("training"):
    model.fit(x_train, y_train, epochs=1, batch_size=32)

results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
print(f"Snapshots captured: {len(results.snapshots)}")
```

For CPU-only TensorFlow or when the GPU backend is unavailable, initialize the
profiler with `TFMemoryProfiler(device="/CPU:0")`.

## MLX profiling and script capture

`stormlog.mlx.MLXMemoryProfiler` evaluates returned roots or explicit context
roots and records synchronized host elapsed time and sampled allocator peaks.
`stormlog.mlx.MemoryTracker` observes passive, process-local active/cache counters
with bounded history and shared telemetry sinks. Native support requires Apple
Silicon Metal and MLX >=0.32.3. Unknown device metrics remain null.

```bash
mlxmemprof run --output artifacts/mlx-run -- workload.py --application-flag value
mlxmemprof analyze artifacts/mlx-run --output artifacts/mlx-analysis --json
stormlog tui --framework mlx
```

The workload runs in the command's interpreter. Separate workers/servers require
their own instrumentation. Analysis and saved-artifact TUI browsing work without
native MLX. See the [MLX guide](mlx_testing_guide.md) and runnable
[training/generation cookbook](cookbook/mlx.md) for state and stream boundaries.

## JAX profiling

```python
import jax.numpy as jnp
from stormlog.jax import JAXMemoryProfiler

profiler = JAXMemoryProfiler()

with profiler.profile_context("training"):
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    # Block to ensure accurate profile boundaries due to async dispatch
    y.block_until_ready()

results = profiler.get_results()
if results.device_memory_available:
    print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
else:
    print(
        "Device memory unavailable: "
        f"{results.device_memory_unavailable_reason}"
    )
print(f"Snapshots captured: {len(results.snapshots)}")
```

For CPU-only JAX profiling, initialize
`JAXMemoryProfiler(device_index="cpu")`; from the CLI, use `--device cpu`.
Device-memory metrics are available only when the selected runtime exposes JAX
allocator stats. Check `device_memory_available` before reading profiler
metrics. Monitor and track exports report process RSS separately when device
memory is unavailable.

### JAX OOM Flight Recorder and Graph Visualization

Enable the OOM flight recorder when a tracked workload may raise an OOM:

```bash
jaxmemprof track \
  --oom-flight-recorder \
  --oom-dump-dir ./oom_dumps \
  --output jax_track.json
```

After a recognized OOM, Stormlog attempts to save an XLA device-memory profile
when the runtime exposes `jax.profiler.save_device_memory_profile`. If profile
capture and protobuf parsing succeed, Stormlog writes
`jax-device-memory-graph.html` beside the `.prof` file.

The HTML file is self-contained after generation. Creating its directed graph
requires the Graphviz `dot` executable at generation time. Without Graphviz,
the file still includes the allocation table and summary, while the graph tab
shows a dependency message.

You can view the resulting diagnostic graph in two ways:

1. **Stormlog HTML:** Open `jax-device-memory-graph.html` in a browser. The
   browser does not need Go, `protoc`, or Graphviz after Stormlog has generated
   the file.
2. **Official Go Tool:** If you prefer the standard Go toolchain, run
   `go tool pprof -http=:8080 <path_to_.prof>`.

## CPU-only workflow

Use this when PyTorch CUDA profiling is unavailable but you still want a local validation path:

```python
from stormlog import CPUMemoryProfiler

profiler = CPUMemoryProfiler()

with profiler.profile_context("cpu_step"):
    values = [i * i for i in range(100_000)]
    values.reverse()

summary = profiler.get_summary()
print(summary["mode"])
print(summary["snapshots_collected"])
```

## Tracking over time

### PyTorch tracker

`MemoryTracker` follows the PyTorch tracker stack. Install
`stormlog[torch]` before using this example. On Apple Silicon, this is the
MPS-aware tracker path once the PyTorch extra is installed.

```python
from stormlog import MemoryTracker

tracker = MemoryTracker(
    sampling_interval=0.5,
    enable_alerts=True,
)

tracker.start_tracking()
# run workload here
tracker.stop_tracking()

stats = tracker.get_statistics()
print(f"Peak memory: {stats.get('peak_memory', 0) / (1024**3):.2f} GB")
print(f"Events: {stats.get('total_events', 0)}")
```

### Structured workload phases

All long-running trackers can emit explicit workload-phase boundaries while
tracking is active:

```python
from stormlog import CPUMemoryTracker

tracker = CPUMemoryTracker(sampling_interval=0.1)
tracker.start_tracking()

with tracker.phase("train", metadata={"epoch": 1}):
    with tracker.phase("forward", metadata={"microbatch": 4}):
        _ = [i * i for i in range(50_000)]

tracker.stop_tracking()
```

This writes `phase_enter` and `phase_exit` telemetry records with structured
`metadata["phase_scope"]` payloads. The same API shape is available on
`MemoryTracker`, `CPUMemoryTracker`, `stormlog.tensorflow.MemoryTracker`, and `stormlog.jax.MemoryTracker`.

When those phase boundaries are present, `gpumemprof analyze` and the TUI
Diagnostics tab can attach hidden-memory gaps, collective spikes, and
cross-rank first-cause suspects back to the active phase path.

If the underlying collector becomes unstable, `MemoryTracker` keeps the tracking
session alive, exposes collector health through `get_statistics()`, and emits
`collector_degraded` / `collector_recovered` events instead of exporting
synthetic zero-valued samples.

The tracker session lifecycle is:

- session begins after tracker startup succeeds and before the first persisted record
- clean `stop_tracking()` finalization marks the session `completed`
- recovered append-only runs are marked `interrupted`
- partial artifacts without provable shutdown remain `incomplete`

For always-on deployments, treat the tracker as an operating-budgeted service:

- keep the default sink retention enabled so artifacts stay bounded
- watch `rollover_count`, `pruned_segment_count`, and `final_retained_*`
- watch `history_dropped_*` to confirm bounded in-memory windows are evicting as expected
- treat any non-zero `collector_failure_event_count` or non-`healthy` collector state as actionable

The maintained source-checkout qualification path for those guarantees is the
v0.4 operability harness in `examples/cli/benchmark_harness.py`.

For CUDA-only OOM debugging, enable native allocator-history capture and wrap
the risky block with `capture_oom()`:

```python
from stormlog import MemoryTracker

tracker = MemoryTracker(
    enable_oom_flight_recorder=True,
    enable_native_cuda_history=True,
)

with tracker.capture_oom(context="train-step"):
    run_training_step()
```

If that block raises a CUDA OOM, the OOM dump bundle is extended with native
snapshot artifacts and pointer-attribution summaries.
The OOM manifest and metadata also include the owning `session_id`, so the dump
can be traced back to the exact tracking run that produced it.

### CPU tracker

```python
from stormlog import CPUMemoryTracker

tracker = CPUMemoryTracker(sampling_interval=0.5)
tracker.start_tracking()
# run workload here
tracker.stop_tracking()
stats = tracker.get_statistics()
print(stats["total_events"])
print(stats["final_retained_files"])
print(stats["history_dropped_events"])
```

## Reusing a sink directory across runs

You can keep a long-lived sink directory and still separate captures
deterministically:

```bash
gpumemprof track --telemetry-sink-dir ./live_sink --duration 30
gpumemprof track --telemetry-sink-dir ./live_sink --duration 30
gpumemprof analyze ./live_sink
gpumemprof analyze ./live_sink --session-id "<session-id-from-report>"
```

The default loader behavior is:

1. newest `completed`
2. newest `interrupted`
3. newest `incomplete`

Use explicit session selection when you want an older or partial run.

## Plot exports

`MemoryVisualizer` is for saved plots and exported data once you already have profiler results or monitoring snapshots.

```python
import torch
from stormlog import GPUMemoryProfiler, MemoryVisualizer

profiler = GPUMemoryProfiler()
device = profiler.device


def sample_workload() -> torch.Tensor:
    x = torch.randn(32, 64, device=device)
    return x.sum()

profile = profiler.profile_function(sample_workload)
visualizer = MemoryVisualizer(profiler)
visualizer.plot_memory_timeline(interactive=False, save_path="timeline.png")
```

Install `stormlog[viz]` before relying on PNG or HTML exports.

## When to switch to the TUI

Launch the default TUI with either `stormlog` or `stormlog tui`; neither needs
options. Use `stormlog tui --framework mlx` to select MLX explicitly.

Use the TUI when you want:

- a live monitoring session without writing custom code
- quick CSV/JSON export from an active tracker
- PNG or HTML timeline export from the current session
- artifact loading and distributed diagnostics in one place
- session-aware switching between multiple captures loaded from the same artifact root

See the [TUI Guide](tui.md) for the TUI flow and the [CLI Guide](cli.md) for scriptable automation.

## Related examples

The example modules below are available in the source repository only, not in
the pip package. Pip users should use the snippets above.

- [examples/basic/pytorch_demo.py](../examples/basic/pytorch_demo.py)
- [examples/basic/tensorflow_demo.py](../examples/basic/tensorflow_demo.py)
- [examples/advanced/tracking_demo.py](../examples/advanced/tracking_demo.py)
- [examples/cli/quickstart.py](../examples/cli/quickstart.py)

---

[← Back to main docs](index.md)
