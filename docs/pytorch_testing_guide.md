[← Back to main docs](index.md)

# PyTorch Guide

This guide is for the current PyTorch-backed Stormlog workflow: profiling
`torch.cuda` workloads, tracking long-running jobs across supported backends,
and exporting artifacts for later diagnostics.

## Before you start

Validate the environment:

```bash
gpumemprof info
python -m examples.basic.pytorch_demo
```

`python -m examples.basic.pytorch_demo` is source-checkout only. If you
installed from PyPI, use `gpumemprof info`, `gpumemprof track`, and the Python
snippets in [usage.md](usage.md) instead.

If no supported `torch.cuda` backend is available, the example script will skip
the bounded profiling path. In that case, use the tracker, CLI, or CPU-only
flows in [cpu_compatibility.md](cpu_compatibility.md).

## Daily workflow: ML engineer

Use this when you want to understand the memory cost of a training or inference step.

```python
import torch
from stormlog import GPUMemoryProfiler

profiler = GPUMemoryProfiler(track_tensors=True)
device = profiler.device
model = torch.nn.Linear(1024, 256).to(device)

def train_step() -> torch.Tensor:
    x = torch.randn(64, 1024, device=device)
    y = model(x)
    return y.sum()

profile = profiler.profile_function(train_step)
summary = profiler.get_summary()

print(profile.function_name)
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
```

Use `profile_context` when you want one named block rather than one function call:

```python
import torch
from stormlog import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
device = profiler.device
model = torch.nn.Linear(1024, 256).to(device)

with profiler.profile_context("forward_pass"):
    outputs = model(torch.randn(32, 1024, device=device))
```

## Daily workflow: debugging growth over time

Use the tracker when one profiled call is not enough. This is the backend-aware
PyTorch path for CUDA, ROCm, and MPS telemetry.

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
print(stats["total_events"])
print(stats["peak_memory"])
```

If you want exports without writing custom code, use the CLI instead:

```bash
gpumemprof track --duration 30 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag_bundle
```

## Daily workflow: release or CI triage

Use these commands when you need a reproducible signal rather than an ad hoc notebook session:

> **Source checkout only.** These commands require the repository `examples/`
> package.

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

If the run produces telemetry or a diagnose bundle, load the artifacts into the TUI Diagnostics tab to compare ranks or inspect anomaly indicators.

## TUI-assisted PyTorch workflow

The current TUI PyTorch-related tabs are:

- `PyTorch` for sample workloads and collected profile summaries
- `Monitoring` for live tracking and exports
- `Visualizations` for timeline export
- `Diagnostics` for artifact review

Launch:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

## Recommended validation sequence

Use this when changing PyTorch behavior or reviewing a regression:

> **Source checkout only.** Replace the example-module steps below with the
> CLI-only validation from [cli.md](cli.md) if you installed from PyPI.

```bash
gpumemprof info
python -m examples.basic.pytorch_demo
python -m examples.advanced.tracking_demo
gpumemprof track --duration 10 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag_bundle
```

## Common issues

### `GPUMemoryProfiler` raises because no `torch.cuda` backend is available

That is expected on CPU-only or MPS-only hosts. Use `MemoryTracker`, the CLI,
or the CPU profiler classes instead.

### `gpumemprof analyze` rejects `--input`

The current CLI uses a positional input file:

```bash
gpumemprof analyze track.json --format txt --output analysis.txt
```

### Plot export fails

Install the visualization extra:

```bash
pip install "stormlog[viz]"
```

## Related docs

- [usage.md](usage.md)
- [cli.md](cli.md)
- [tui.md](tui.md)
- [troubleshooting.md](troubleshooting.md)

---

[← Back to main docs](index.md)
