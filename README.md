# Stormlog

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Silas-Asamoah/stormlog/actions)
[![PyPI Version](https://img.shields.io/pypi/v/stormlog.svg)](https://pypi.org/project/stormlog/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org/)
[![Textual TUI](https://img.shields.io/badge/TUI-Textual-blueviolet)](docs/tui.md)

<p align="center">
  <img src="docs/gpu-profiler-overview.gif" alt="Stormlog terminal UI overview" width="900">
  <br/>
  <em>Current Textual workflow: monitoring, visualization export, diagnostics, and CLI actions.</em>
</p>

Stormlog is a memory-profiling toolkit for day-to-day PyTorch and TensorFlow work. It combines Python APIs, CLI commands, and a Textual TUI so you can move from "what is using memory?" to saved artifacts and shareable diagnostics without switching tools.

## Why use this tool

- Catch memory growth before it becomes an OOM.
- Compare allocated vs reserved usage during training and inference.
- Export telemetry and diagnose bundles for CI or release triage.
- Load the same artifacts into a terminal UI for faster debugging.
- Keep workflows available on CPU-only and MPS systems, not just CUDA boxes.

## Installation

### From PyPI

```bash
pip install stormlog
pip install "stormlog[viz]"
pip install "stormlog[tui,torch]"
pip install "stormlog[torch]"
pip install "stormlog[tf]"
pip install "stormlog[all]"
```

### Package and import names

`stormlog` is the distribution name on PyPI. The installed Python modules remain
backend-specific:

| Task | Use |
| --- | --- |
| Install the package | `pip install stormlog` |
| Launch the TUI | `stormlog` |
| Import PyTorch APIs | `from gpumemprof import GPUMemoryProfiler, MemoryTracker` |
| Import TensorFlow APIs | `from tfmemprof import TFMemoryProfiler` |

There is no top-level `stormlog` Python module today.

### From source

```bash
git clone https://github.com/Silas-Asamoah/stormlog.git
cd stormlog
pip install -e .
pip install -e ".[viz,tui,torch]"
```

If you want both framework extras in a development checkout:

```bash
pip install -e ".[dev,test,all,tui,viz]"
```

## Quick start

### CLI-first workflow

This is the fastest path to verify an environment and produce an artifact you can inspect later:

```bash
gpumemprof info
gpumemprof track --duration 2 --interval 0.5 --output /tmp/gpumemprof_track.json --format json
gpumemprof analyze /tmp/gpumemprof_track.json --format txt --output /tmp/gpumemprof_analysis.txt
gpumemprof diagnose --duration 0 --output /tmp/gpumemprof_diag

tfmemprof info
tfmemprof diagnose --duration 0 --output /tmp/tf_diag
```

### PyTorch API workflow

`GPUMemoryProfiler` currently targets PyTorch runtimes exposed through
`torch.cuda`, which covers NVIDIA CUDA and ROCm-backed builds. On Apple MPS or
CPU-only systems, use `MemoryTracker`, the CLI, or `CPUMemoryProfiler` instead.

```python
import torch
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
device = profiler.device
model = torch.nn.Linear(1024, 128).to(device)

def train_step() -> torch.Tensor:
    x = torch.randn(64, 1024, device=device)
    y = model(x)
    return y.sum()

profile = profiler.profile_function(train_step)
summary = profiler.get_summary()

print(profile.function_name)
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
```

### TensorFlow API workflow

`TFMemoryProfiler` works on GPU or CPU-backed TensorFlow runtimes.

```python
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler(enable_tensor_tracking=True)

with profiler.profile_context("training"):
    model.fit(x_train, y_train, epochs=1, batch_size=32)

results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
print(f"Snapshots captured: {len(results.snapshots)}")
```

## Daily workflows

### ML engineer

- instrument a training step with `GPUMemoryProfiler` or `TFMemoryProfiler`
- switch to `track` when you need telemetry over time
- export plots or analyze saved telemetry later

### Researcher debugging regressions

- capture `track` output or a `diagnose` bundle
- open the same artifacts in the TUI diagnostics and visualizations tabs
- compare growth, gaps, and per-rank behavior before changing model code

### CI or release owner

- run `python -m examples.cli.quickstart`
- run `python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated`
- archive the emitted artifacts for later triage in the TUI

## Terminal UI

Install the optional TUI dependencies and launch:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

The current TUI startup path imports PyTorch immediately, so `stormlog[tui]` alone is not enough yet.

The current TUI tabs are:

- `Overview`
- `PyTorch`
- `TensorFlow`
- `Monitoring`
- `Visualizations`
- `Diagnostics`
- `CLI & Actions`

<p align="center">
  <img src="docs/tui-overview-current.png" alt="Stormlog overview tab" width="700">
  <br/>
  <em>Overview tab with current system summary and navigation guidance.</em>
</p>

<p align="center">
  <img src="docs/tui-diagnostics-current.png" alt="Stormlog diagnostics tab" width="700">
  <br/>
  <em>Diagnostics tab with the current artifact loader, rank table, and timeline panes.</em>
</p>

Use the Monitoring tab to start live tracking, export CSV or JSON events to `./exports`, and tune warning or critical thresholds. In Visualizations, refresh the live timeline and save PNG or HTML exports under `./visualizations`. In Diagnostics, load live telemetry or artifact paths and rebuild rank-level diagnostics without leaving the terminal.

For screen-by-screen details, see [docs/tui.md](docs/tui.md).

## Examples and walkthroughs

- [Documentation home](docs/index.md)
- [Installation guide](docs/installation.md)
- [Usage guide](docs/usage.md)
- [CLI guide](docs/cli.md)
- [Examples guide](docs/examples.md)
- [Testing guide](docs/testing.md)
- [PyTorch guide](docs/pytorch_testing_guide.md)
- [TensorFlow guide](docs/tensorflow_testing_guide.md)
- [Long-form article](docs/article.md)

## Launch QA scenarios

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
python -m examples.scenarios.cpu_telemetry_scenario
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
```

## CPU-only and laptop workflows

If CUDA is not available, Stormlog still supports:

- `gpumemprof info`
- `gpumemprof monitor`
- `gpumemprof track`
- `CPUMemoryProfiler`
- `CPUMemoryTracker`
- the TUI overview, monitoring, diagnostics, and CLI tabs

See [docs/cpu_compatibility.md](docs/cpu_compatibility.md) for the CPU-only path.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[MIT License](LICENSE)
