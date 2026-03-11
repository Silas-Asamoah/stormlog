[← Back to main docs](index.md)

# Examples Guide

This guide shows the fastest safe validation paths for both pip installs and
source checkouts.

> **Note for pip users**: The example modules
> (`examples.cli.quickstart`, `examples.basic.pytorch_demo`, and related
> scripts) are **only available when you install from a source checkout**.
> If you installed via `pip install stormlog`, these commands will raise
> `ModuleNotFoundError`. Use the
> [CLI-only validation](#cli-only-validation-for-pip-users) and
> [Python snippets for pip users](#python-snippets-for-pip-users) below
> instead, or clone the repository for full example coverage.

## CLI-only validation for pip users

If you installed Stormlog from PyPI, use this sequence instead of
`examples.cli.quickstart`:

```bash
gpumemprof info
gpumemprof track --duration 2 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag

tfmemprof info
tfmemprof diagnose --duration 0 --output ./tf_diag
```

This validates the installed CLI and produces artifacts you can load in the
TUI Diagnostics tab.

## Python snippets for pip users

If you do not have the example modules, use these snippets in place of the
source-only demos.

### CPU profiling

```python
from gpumemprof import CPUMemoryProfiler

profiler = CPUMemoryProfiler()
with profiler.profile_context("cpu_step"):
    values = [i * i for i in range(100_000)]
    values.reverse()
print(profiler.get_summary())
```

### CPU tracking

```python
from gpumemprof import CPUMemoryTracker

tracker = CPUMemoryTracker(sampling_interval=0.5)
tracker.start_tracking()
# Run workload here.
tracker.stop_tracking()
print(tracker.get_statistics()["total_events"])
```

### MPS tracking

```python
from gpumemprof import MemoryTracker

tracker = MemoryTracker(sampling_interval=0.5, enable_alerts=True)
tracker.start_tracking()
# Run workload here.
tracker.stop_tracking()
print(tracker.get_statistics())
```

### TensorFlow

```python
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler(device="/CPU:0", enable_tensor_tracking=True)
with profiler.profile_context("training"):
    # Your model.fit() or TensorFlow ops go here.
    pass
print(profiler.get_results())
```

## Source-checkout examples

The workflows below require a repository clone plus `pip install -e .`.

### Launch QA scenario matrix

Source checkout only.

```bash
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

This command runs:

- CPU telemetry export and schema validation
- MPS telemetry export and analyze roundtrip when MPS is available
- OOM flight-recorder scenario in safe simulated mode
- TensorFlow monitor, track, analyze, and diagnose end-to-end coverage
- `gpumemprof diagnose` artifact checks
- benchmark harness budget gates
- optional TUI PTY smoke when `--skip-tui` is omitted

Switch to full mode to include extra demos:

```bash
python -m examples.cli.capability_matrix --mode full --target both --oom-mode simulated
```

### Individual scenario modules

Source checkout only.

```bash
python -m examples.scenarios.cpu_telemetry_scenario
python -m examples.scenarios.mps_telemetry_scenario
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
python -m examples.scenarios.tf_end_to_end_scenario
```

### Repository example scripts

These scripts are available in the source repository only, not in the pip
package:

- `examples/basic/pytorch_demo.py`
- `examples/advanced/tracking_demo.py`
- `examples/basic/tensorflow_demo.py`
- `examples/cli/quickstart.py`

### Markdown test guides

The Markdown test guides are in the source repository at
`docs/examples/test_guides/README.md`. They are not shipped with the pip
package. If you have a source checkout, use them for manual checklists. Pip
users should follow the CLI-only validation and Python snippets above plus the
[Usage Guide](usage.md).

## Best Practices

1. **Profile Early**: Start profiling during development, not just in production
2. **Use Contexts**: Organize profiling with meaningful context names
3. **Set Thresholds**: Configure appropriate memory thresholds for your hardware
4. **Monitor Continuously**: Use CLI tools for continuous monitoring
5. **Export Data**: Save results for later analysis and comparison
6. **Visualize**: Use built-in visualization tools to understand patterns

## Troubleshooting Examples

### Memory Leak Detection

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.5)

# Run your code
for i in range(100):
    profiler.profile_function(train_step, model, data)

profiler.stop_monitoring()
summary = profiler.get_summary()
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
print(f"Memory change: {summary['memory_change_from_baseline'] / (1024**3):.2f} GB")
```

### Performance Optimization

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Profile different batch sizes
for batch_size in [16, 32, 64, 128]:
    with profiler.profile_context(f"batch_size_{batch_size}"):
        train_with_batch_size(model, dataloader, batch_size)

    summary = profiler.get_summary()
    peak_gb = summary['peak_memory_usage'] / (1024**3)
    print(f"Batch size {batch_size}: Peak memory {peak_gb:.2f} GB")
```

---

[← Back to main docs](index.md)
