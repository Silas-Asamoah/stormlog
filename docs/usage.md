[← Back to main docs](index.md)

# Usage Guide

This guide focuses on the workflows that are stable in the current codebase:

- profile a bounded section of code
- track memory over time
- export telemetry and plots
- move from runtime data to a diagnose bundle or TUI session

Install the distribution as `stormlog`, then import the Python APIs from
`stormlog` or `stormlog.tensorflow`. The CLI automation commands remain
`gpumemprof` and `tfmemprof`.

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

### `MemoryTracker`

Use when:

- you need telemetry over time rather than one profiled call
- you want backend-aware tracking on CUDA, ROCm, MPS, or CPU fallback paths
- you want alerts, exported event streams, or diagnose bundles

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
```

This gives you:

- environment visibility
- a short telemetry sample
- a readable analysis artifact
- a portable diagnose bundle you can load later

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

### CPU tracker

```python
from stormlog import CPUMemoryTracker

tracker = CPUMemoryTracker(sampling_interval=0.5)
tracker.start_tracking()
# run workload here
tracker.stop_tracking()
print(tracker.get_statistics()["total_events"])
```

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

Use the TUI when you want:

- a live monitoring session without writing custom code
- quick CSV/JSON export from an active tracker
- PNG or HTML timeline export from the current session
- artifact loading and distributed diagnostics in one place

See [tui.md](tui.md) for the TUI flow and [cli.md](cli.md) for scriptable automation.

## Related examples

- [examples/basic/pytorch_demo.py](../examples/basic/pytorch_demo.py)
- [examples/basic/tensorflow_demo.py](../examples/basic/tensorflow_demo.py)
- [examples/advanced/tracking_demo.py](../examples/advanced/tracking_demo.py)
- [examples/cli/quickstart.py](../examples/cli/quickstart.py)

---

[← Back to main docs](index.md)
