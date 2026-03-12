[← Back to main docs](index.md)

# Usage Guide

This guide covers how to use Stormlog for both PyTorch and TensorFlow applications.

## Canonical Workflow (v0.2)

Use this path when onboarding a new project:

1. Validate install and runtime backend:

```bash
gpumemprof --help
gpumemprof info
tfmemprof --help
tfmemprof info
```

2. Capture baseline telemetry:

```bash
gpumemprof track --duration 2 --interval 0.5 --output /tmp/gpumemprof_track.json --format json --watchdog
gpumemprof analyze /tmp/gpumemprof_track.json --format txt --output /tmp/gpumemprof_analysis.txt
```

3. Collect a diagnose bundle for reproducibility:

```bash
gpumemprof diagnose --duration 0 --output /tmp/gpumemprof_diag
tfmemprof diagnose --duration 0 --output /tmp/tf_diag
```

4. Optional source-checkout smoke tests:

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

Steps 1 through 3 work for pip installs on CPU-only, MPS, CUDA, and ROCm
systems. Step 4 requires a source checkout because the `examples/` package is
not included in the PyPI distribution.

## Quick Start

### PyTorch Usage

> **MPS and CPU-only users**: `GPUMemoryProfiler` requires CUDA or ROCm. On
> Apple MPS or CPU-only systems, it raises `RuntimeError: CUDA is not
> available`. Use the CPU-only and tracking snippets below instead.

```python
from gpumemprof import GPUMemoryProfiler

# Create profiler instance
profiler = GPUMemoryProfiler()

# Function to profile
def train_step(model, data, target):
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    return loss

# Profile call + get summary
profile = profiler.profile_function(train_step, model, data, target)
summary = profiler.get_summary()
print(f"Profiled: {profile.function_name}")
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
```

### CPU-only workflow

```python
from gpumemprof import CPUMemoryProfiler

profiler = CPUMemoryProfiler()
with profiler.profile_context("cpu_step"):
    values = [i * i for i in range(100_000)]
    values.reverse()
print(profiler.get_summary())
```

### TensorFlow Usage

```python
from tfmemprof import TFMemoryProfiler

# Create profiler instance
profiler = TFMemoryProfiler()

# Profile training
with profiler.profile_context("training"):
    model.fit(x_train, y_train, epochs=5)

# Get results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

For CPU-only TensorFlow or when the GPU backend is unavailable, initialize the
profiler with `TFMemoryProfiler(device="/CPU:0")`.

## Advanced Usage

### Real-time Monitoring

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Start monitoring
profiler.start_monitoring(interval=1.0)  # Sample every second

# Your training code here
for epoch in range(10):
    for batch in dataloader:
        train_step(model, batch)

# Stop and get results
profiler.stop_monitoring()
results = profiler.get_summary()
```

### Tracking over time

```python
from gpumemprof import CPUMemoryTracker

cpu_tracker = CPUMemoryTracker(sampling_interval=0.5)
cpu_tracker.start_tracking()
# Run a CPU-only workload here.
cpu_tracker.stop_tracking()
print(cpu_tracker.get_statistics()["total_events"])
```

If you installed the PyTorch extra (`pip install "stormlog[torch]"`) and want
Apple Silicon MPS telemetry, use `MemoryTracker` instead:

```python
from gpumemprof import MemoryTracker

mps_tracker = MemoryTracker(sampling_interval=0.5, enable_alerts=True)
mps_tracker.start_tracking()
# Run an MPS workload here.
mps_tracker.stop_tracking()
print(mps_tracker.get_statistics())
```

### Memory leak detection

> **Requires the PyTorch extra**: `MemoryTracker` follows the PyTorch tracker
> stack. Install `stormlog[torch]` before using this example.

```python
from gpumemprof import MemoryTracker

tracker = MemoryTracker(
    sampling_interval=0.5,
    enable_alerts=True,
)
tracker.start_tracking()

# Run your code
for i in range(100):
    train_step(model, data)

# Inspect tracking statistics
tracker.stop_tracking()
stats = tracker.get_statistics()
print(f"Peak memory: {stats.get('peak_memory', 0) / (1024**3):.2f} GB")
```

### Visualization

> **Note**: `MemoryVisualizer` requires `GPUMemoryProfiler`, which is
> CUDA/ROCm-only. On MPS or CPU, use the TUI Visualizations tab with live
> tracking instead, or run `gpumemprof analyze --visualization` on saved
> telemetry from `gpumemprof track`.

```python
from gpumemprof import GPUMemoryProfiler, MemoryVisualizer

profiler = GPUMemoryProfiler()

# Profile your code
with profiler.profile_context("training"):
    train_model()

# Generate visualizations
visualizer = MemoryVisualizer(profiler)
visualizer.plot_memory_timeline(interactive=False, save_path="timeline.png")
visualizer.export_data(format="json", save_path="profile_export")
```

## CLI Usage

### Basic Monitoring

```bash
# Monitor for 60 seconds
gpumemprof monitor --duration 60 --output monitoring.csv

# Track with telemetry output + watchdog
gpumemprof track --duration 30 --interval 0.5 --output tracking.json --format json --watchdog
```

### Analysis

```bash
# Analyze results
gpumemprof analyze tracking.json --format txt --output analysis.txt

# Build a diagnose bundle
gpumemprof diagnose --duration 0 --output ./diag_bundle
```

## Configuration

### Profiler Settings

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler(
    device=0,
    track_tensors=True,
    track_cpu_memory=True,
    collect_stack_traces=False,
)
```

### Context Profiling

```python
from gpumemprof import profile_function, profile_context

# Function decorator
@profile_function
def my_function():
    pass

# Context manager
with profile_context("my_context"):
    pass
```

## Best Practices

1. **Start Early**: Begin profiling early in development
2. **Use Contexts**: Use context managers for better organization
3. **Monitor Regularly**: Set up continuous monitoring in production
4. **Set Alerts**: Configure appropriate thresholds
5. **Export Data**: Save results for later analysis

## Related examples

The following example scripts are available in the **source repository** only,
not in the pip package:

- `examples/basic/pytorch_demo.py` for CUDA-gated `GPUMemoryProfiler` usage
- `examples/basic/tensorflow_demo.py` for `TFMemoryProfiler` context profiling
- `examples/advanced/tracking_demo.py` for `MemoryTracker` with alerts and export
- `examples/cli/quickstart.py` for CLI smoke validation

Pip users should use the [CPU-only workflow](#cpu-only-workflow), the
[Tracking over time](#tracking-over-time) snippets above, and the
[Examples Guide](examples.md) for equivalent Python and CLI commands.

---

[← Back to main docs](index.md)
