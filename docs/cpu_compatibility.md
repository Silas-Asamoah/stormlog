[← Back to main docs](index.md)

# CPU Compatibility Guide

Stormlog is still useful on machines without CUDA. The main difference is that GPU-specific PyTorch profiling is replaced by CPU-backed monitoring and profiling helpers.

## What still works on CPU-only machines

- `gpumemprof info`
- `gpumemprof monitor`
- `gpumemprof track`
- `gpumemprof analyze`
- `gpumemprof diagnose`
- `CPUMemoryProfiler`
- `CPUMemoryTracker`
- the TUI overview, monitoring, diagnostics, and CLI tabs
- TensorFlow flows when you have a CPU-backed TensorFlow install

## What does not work without CUDA

- `GPUMemoryProfiler`
- CUDA memory allocator statistics
- CUDA-specific PyTorch sample workloads

If you need the CUDA path, see [gpu_setup.md](gpu_setup.md).

## Fast local validation

Use the CLI first:

```bash
export CUDA_VISIBLE_DEVICES=
gpumemprof info
python -m examples.cli.quickstart
```

On Windows, clear `CUDA_VISIBLE_DEVICES` with the shell-appropriate syntax before running the same steps.

## CPU profiling in Python

```python
from stormlog import CPUMemoryProfiler

profiler = CPUMemoryProfiler()

with profiler.profile_context("cpu_task"):
    values = [i * i for i in range(250_000)]
    values.reverse()

summary = profiler.get_summary()
print(summary["mode"])
print(summary["peak_memory_usage"])
```

## CPU tracking over time

```python
from stormlog import CPUMemoryTracker

tracker = CPUMemoryTracker(sampling_interval=0.5)
tracker.start_tracking()

# run workload here

tracker.stop_tracking()
stats = tracker.get_statistics()
print(stats["total_events"])
```

## CLI workflows on CPU-only hosts

### Bounded monitoring

```bash
gpumemprof monitor --duration 30 --interval 0.5 --output cpu_monitor.json --format json
```

### Event tracking

```bash
gpumemprof track --duration 30 --interval 0.5 --output cpu_track.json --format json
```

### Analysis and artifact capture

```bash
gpumemprof analyze cpu_track.json --format txt --output cpu_analysis.txt
gpumemprof diagnose --duration 0 --output ./cpu_diag
```

## TUI workflows on CPU-only hosts

The TUI remains useful even when CUDA is not available:

- `Overview` still shows system information
- `Monitoring` still runs the CPU tracker
- `Visualizations` can export plots once tracking has collected samples
- `Diagnostics` can load saved telemetry or diagnose bundles
- `CLI & Actions` remains available for quick commands

Launch it with:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

The current TUI startup path still imports PyTorch immediately, even if you only plan to use CPU monitoring and diagnostics flows once the app is open.

## Recommended CPU-only checklist

1. verify the install with `gpumemprof info`
2. run `python -m examples.cli.quickstart`
3. run `python -m examples.scenarios.cpu_telemetry_scenario`
4. capture a short `track` output
5. load the result into the TUI if you need an interactive review

## Common confusion points

### "Why does the PyTorch demo skip itself?"

`examples.basic.pytorch_demo` intentionally requires CUDA because it demonstrates `GPUMemoryProfiler`. Use the CLI quickstart or `CPUMemoryProfiler` on CPU-only hosts.

### "Why do I still see `gpumemprof` on a CPU machine?"

The CLI is broader than CUDA-only profiling. It still supports CPU-backed monitoring, tracking, analysis, and diagnose flows.

### "Can I still use TensorFlow?"

Yes, as long as TensorFlow itself is installed. `TFMemoryProfiler` works on CPU-backed TensorFlow environments too.

---

[← Back to main docs](index.md)
