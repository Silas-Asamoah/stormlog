[← Back to main docs](index.md)

# TensorFlow Guide

This guide covers the current TensorFlow workflow in Stormlog: profiling TensorFlow code directly, tracking TensorFlow memory usage from the CLI, and exporting artifacts for later review.

## Before you start

Validate the environment:

```bash
tfmemprof info
python -m examples.basic.tensorflow_demo
```

`python -m examples.basic.tensorflow_demo` is source-checkout only. If you
installed from PyPI, validate with `tfmemprof info`, the CLI commands below,
and the TensorFlow snippets in [usage.md](usage.md).

These checks work on CPU-backed TensorFlow installs as well as GPU-backed ones.

## Daily workflow: ML engineer

Use `TFMemoryProfiler` when you want snapshots and aggregate results around a real TensorFlow workload.

```python
from stormlog.tensorflow import TFMemoryProfiler

profiler = TFMemoryProfiler(enable_tensor_tracking=True)

with profiler.profile_context("training"):
    model.fit(x_train, y_train, epochs=1, batch_size=32)

results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
print(f"Snapshots captured: {len(results.snapshots)}")
```

Decorator-style profiling also exists through the TensorFlow profiler API, but the context-manager flow above is the clearest daily-workflow path.

## Daily workflow: investigate sustained growth

The current TensorFlow CLI is the simplest way to capture longer-running telemetry:

```bash
tfmemprof monitor --interval 0.5 --duration 30 --output tf_monitor.json
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize --report tf_report.txt
tfmemprof diagnose --duration 0 --output ./tf_diag
```

For CPU-backed TensorFlow or when the GPU backend is unavailable, add
`--device /CPU:0` to `monitor` and `track`.

Key difference from `gpumemprof`:

- TensorFlow analysis uses `--input`
- PyTorch analysis uses a positional file argument

## Daily workflow: release or CI triage

Use the maintained example and scenario modules instead of inventing a one-off shell script:

> **Source checkout only.** These commands require the repository `examples/`
> package.

```bash
python -m examples.cli.quickstart
python -m examples.scenarios.tf_end_to_end_scenario
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

This is the fastest route to a reproducible TensorFlow artifact set.

## TUI-assisted TensorFlow workflow

The current TUI can help after or during a TensorFlow run:

- `TensorFlow` tab for sample workloads and collected summaries
- `Monitoring` for live tracking
- `Visualizations` for plot export
- `Diagnostics` for artifact review
- `CLI & Actions` for command-driven flows

Launch:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

The current TUI startup path imports PyTorch immediately, so TensorFlow-only environments still need the `torch` extra for `stormlog` to launch.

## Recommended validation sequence

Use this when you need a compact TensorFlow confidence pass:

> **Source checkout only.** Replace `python -m examples.basic.tensorflow_demo`
> with `tfmemprof monitor --interval 0.5 --duration 15 --device /CPU:0 --output tf_monitor.json`
> if you installed from PyPI on a CPU-backed TensorFlow setup.

```bash
tfmemprof info
python -m examples.basic.tensorflow_demo
tfmemprof monitor --interval 0.5 --duration 15 --output tf_monitor.json
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize --report tf_report.txt
tfmemprof diagnose --duration 0 --output ./tf_diag
```

## Common issues

### `tfmemprof` is installed but no GPU devices appear

Run:

```bash
tfmemprof info
```

If it still shows no GPU devices, treat it as an environment issue first. The TensorFlow flow still supports CPU-backed runs.

### `tfmemprof analyze` rejects positional input

That is expected. Use:

```bash
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize
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
