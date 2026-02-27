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

4. Run curated examples:

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

This sequence works for CPU-only and MPS/CUDA systems; unsupported checks are
reported as explicit `SKIP` instead of silent failures.

## Quick Start

### PyTorch Usage

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

### Memory Leak Detection

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

## Examples

See the [examples directory](../examples/) for complete working examples:

- [Basic PyTorch profiling](../examples/basic/pytorch_demo.py)
- [Advanced tracking](../examples/advanced/tracking_demo.py)
- [TensorFlow profiling](../examples/basic/tensorflow_demo.py)
- [CLI quickstart](../examples/cli/quickstart.py)

---

[← Back to main docs](index.md)
