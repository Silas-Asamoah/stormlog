[← Back to main docs](index.md)

# Troubleshooting Guide

This guide focuses on the failure modes that show up in the current codebase and workflows.

## Installation and entrypoints

### `gpumemprof: command not found`

Reinstall the package into the active environment:

```bash
pip install -e .
hash -r
gpumemprof --help
```

### `tfmemprof: command not found`

Install the TensorFlow extra:

```bash
pip install -e ".[tf]"
hash -r
tfmemprof --help
```

### `stormlog: command not found`

Install the TUI dependencies:

```bash
pip install -e ".[tui,torch]"
hash -r
stormlog
```

## Missing dependencies

### `ModuleNotFoundError: No module named 'torch'`

Install the PyTorch extra instead of trying to use CUDA-specific profiling without the framework:

```bash
pip install "stormlog[torch]"
```

### `ModuleNotFoundError: No module named 'tensorflow'`

```bash
pip install "stormlog[tf]"
```

### Visualization export errors

PNG and HTML exports depend on the visualization stack:

```bash
pip install "stormlog[viz]"
```

## Runtime mismatches

### `GPUMemoryProfiler` fails on a non-CUDA machine

That class is for CUDA-backed PyTorch profiling. On CPU-only or MPS-only systems, use:

- `gpumemprof monitor`
- `gpumemprof track`
- `CPUMemoryProfiler`
- `CPUMemoryTracker`

If you need setup guidance for real CUDA profiling, see the [GPU Setup Guide](gpu_setup.md).

### TensorFlow CLI is installed but reports no GPU

Start with:

```bash
tfmemprof info
```

If it still reports no GPU devices, treat it as an environment problem first:

- TensorFlow build may be CPU-only
- device visibility may be restricted
- the current host may genuinely be CPU-only

The profiler still supports CPU-backed TensorFlow runs.

## TUI issues

### Monitoring starts but Visualizations stays empty

The Visualizations tab only renders after timeline samples exist.

Use this sequence:

1. open `Monitoring`
2. click `Start Live Tracking`
3. let the workload run long enough to create samples
4. open `Visualizations`
5. click `Refresh Timeline`

### Diagnostics loads but shows no rank data

Check the source you loaded:

- live diagnostics require an active tracker session with telemetry events
- artifact diagnostics require real JSON, CSV, or diagnose paths
- after changing artifact paths, click `Refresh`

### PNG or HTML export appears blank

This usually means there were no timeline samples, not that the export code failed.

Validate in order:

1. start tracking
2. confirm the monitoring log is receiving events
3. refresh the Visualizations tab
4. export again

### The TUI layout looks broken

The app can run in a small terminal, but it is easier to use with a wider window. The deterministic snapshot coverage uses roughly `140x44`.

## CLI workflow issues

### `gpumemprof analyze` rejects `--input`

That is expected. The current PyTorch CLI uses a positional input file:

```bash
gpumemprof analyze track.json --format txt --output analysis.txt
```

### `tfmemprof analyze` rejects the positional input style

That is also expected. The current TensorFlow CLI uses `--input`:

```bash
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize
```

### Diagnose bundle feels too slow

Use `--duration 0` when you only need the bundle structure and not a new sampling window:

```bash
gpumemprof diagnose --duration 0 --output ./diag_bundle
tfmemprof diagnose --duration 0 --output ./tf_diag
```

## CI and docs issues

### Sphinx build fails locally

Install the docs extra and rebuild:

```bash
pip install -e ".[docs]"
python3 -m sphinx -W --keep-going -b html docs docs/_build/html
```

### A docs snippet looks suspicious

Use the code and `--help` output as the source of truth, then run:

```bash
python3 -m pytest tests/test_docs_regressions.py -v
```

## Recommended debugging path

If you are unsure where a failure belongs:

1. verify installation and entrypoints
2. verify framework availability with `info`
3. reproduce with the smallest matching example under `examples/`
4. capture telemetry or a diagnose bundle
5. inspect the result in the TUI or analyzer

If you installed from PyPI and do not have the `examples/` package, use the
CLI-first validation paths in the [Usage Guide](usage.md), [Examples Guide](examples.md),
or [CLI Guide](cli.md) instead.

---

[← Back to main docs](index.md)
