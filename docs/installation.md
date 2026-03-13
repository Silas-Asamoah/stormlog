[← Back to main docs](index.md)

# Installation Guide

Use the smallest install that matches your workflow, then validate the console scripts you plan to use.

## Choose an install profile

## Install name vs import names

Install the distribution as `stormlog`, then import the Python APIs from
`stormlog` or `stormlog.tensorflow`:

| Task | Use |
| --- | --- |
| Install from PyPI | `pip install stormlog` |
| Launch the TUI | `stormlog` |
| Import PyTorch APIs | `from stormlog import GPUMemoryProfiler, MemoryTracker` |
| Import TensorFlow APIs | `from stormlog.tensorflow import TFMemoryProfiler` |
| Run CLI automation | `gpumemprof` or `tfmemprof` |

### Core package

```bash
pip install stormlog
```

Includes:

- `stormlog`
- `stormlog.tensorflow`
- core telemetry and analysis utilities
- CPU-compatible monitoring and tracking

### Visualization extras

```bash
pip install "stormlog[viz]"
```

Adds the dependencies used by:

- `MemoryVisualizer`
- TUI HTML export from the Visualizations tab
- richer plot generation in example scripts

### TUI extras

```bash
pip install "stormlog[tui,torch]"
```

Installs the Textual stack plus the current PyTorch runtime dependency required by TUI startup. The `stormlog` console script is declared by the package; these extras make the app runnable.

### Framework extras

```bash
pip install "stormlog[torch]"
pip install "stormlog[tf]"
pip install "stormlog[all]"
```

## Source checkout

```bash
git clone https://github.com/Silas-Asamoah/stormlog.git
cd stormlog
pip install -e .
```

For a contributor setup with all common extras:

```bash
pip install -e ".[dev,test,all,tui,viz]"
pre-commit install
```

### Source-only examples and guides

The `examples/` package and the Markdown test guides under
`docs/examples/test_guides/README.md` are only available from a repository
checkout. A plain `pip install stormlog` does not include them.

## Verification

### Core verification

```bash
python3 -c "import stormlog; print(stormlog.__version__)"
gpumemprof --help
gpumemprof info
```

### TensorFlow verification

If you installed the TensorFlow extra:

```bash
tfmemprof --help
tfmemprof info
```

### TUI verification

If you installed the TUI dependencies:

```bash
stormlog
```

If TUI dependencies are missing after a source checkout, reinstall with:

```bash
pip install -e ".[tui,torch]"
```

## Platform notes

### CPU-only systems

You do not need CUDA to use Stormlog. The CLI, TUI monitoring flows, and CPU profiler helpers remain usable on CPU-only machines. See [cpu_compatibility.md](cpu_compatibility.md).

### macOS and MPS

The TUI and CLI support Apple Silicon workflows. Use the CLI or TUI monitoring
and diagnostics flows even when `GPUMemoryProfiler` is unavailable on MPS.

### Linux and CUDA

Install the correct framework build for your `torch.cuda` runtime before using
`GPUMemoryProfiler`. For NVIDIA systems that means matching the CUDA toolchain.
See [gpu_setup.md](gpu_setup.md) for the full checklist.

## Common install failures

### `gpumemprof: command not found`

```bash
pip install -e .
hash -r
gpumemprof --help
```

### `stormlog: command not found`

```bash
pip install -e ".[tui,torch]"
hash -r
stormlog
```

### Missing framework imports

Install the matching extra instead of trying to work around import errors manually:

```bash
pip install "stormlog[torch]"
pip install "stormlog[tf]"
```

## Next steps

1. Read [usage.md](usage.md) for the Python API and CLI-first workflow.
2. Read [cli.md](cli.md) if you need telemetry, plots, or diagnose bundles.
3. Read [tui.md](tui.md) if you want an interactive terminal workflow.

---

[← Back to main docs](index.md)
