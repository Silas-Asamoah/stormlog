[← Back to main docs](index.md)

# Examples Guide

This page maps documented workflows to the example modules that are actually maintained in the repo.

> **Source checkout only below.** The example modules on this page live under
> `examples/` and are not included in the PyPI distribution. If you installed
> with `pip install stormlog`, use the CLI-only validation below and the Python
> snippets in the [Usage Guide](usage.md) instead.

## CLI-only validation for pip users

Use this when you installed from PyPI and do not have the `examples/` package:

```bash
gpumemprof info
gpumemprof track --duration 2 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag

tfmemprof info
tfmemprof diagnose --duration 0 --output ./tf_diag
```

This validates the installed CLI and produces artifacts you can load in the TUI
Diagnostics tab.

## Start here

### CLI smoke and environment validation

```bash
python -m examples.cli.quickstart
```

Use this when you want a fast signal that the installed console scripts work.

### Release-style capability sweep

```bash
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

Use this when you want one command that touches the major launch-validation flows.

## Python API examples

### PyTorch

```bash
python -m examples.basic.pytorch_demo
```

This demo shows:

- CUDA-gated `GPUMemoryProfiler` usage
- `profile_function`
- `profile_context`
- summary reporting

### TensorFlow

```bash
python -m examples.basic.tensorflow_demo
```

This demo shows:

- `TFMemoryProfiler`
- context profiling
- TensorFlow result summaries
- snapshot-driven reporting

### Advanced tracking

```bash
python -m examples.advanced.tracking_demo
```

This demo shows:

- `MemoryTracker`
- alert callbacks
- watchdog cleanup flow
- exported CSV and JSON tracker events

## Scenario modules

These are the closest examples to real operational workflows:

```bash
python -m examples.scenarios.cpu_telemetry_scenario
python -m examples.scenarios.mps_telemetry_scenario
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
python -m examples.scenarios.tf_end_to_end_scenario
```

### When to use them

- `cpu_telemetry_scenario`: validate CPU-only telemetry export
- `mps_telemetry_scenario`: validate Apple Silicon / MPS flows
- `oom_flight_recorder_scenario`: rehearse OOM artifact capture safely
- `tf_end_to_end_scenario`: validate TensorFlow monitor, track, analyze, and diagnose flow together

## Daily workflow mapping

### ML engineer

Run:

```bash
python -m examples.basic.pytorch_demo
python -m examples.basic.tensorflow_demo
```

Then move to the [Usage Guide](usage.md) if you want the same patterns embedded inside your own code.

### Researcher or debugger

Run:

```bash
python -m examples.advanced.tracking_demo
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
```

Then move to the [TUI Guide](tui.md) and [Troubleshooting Guide](troubleshooting.md).

### CI or release owner

Run:

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

Then move to the [Testing and Validation Guide](testing.md) for the current CI mapping.

## Markdown-only test guides

The old executable guides were replaced by Markdown checklists:

- [Example Test Guides](examples/test_guides/README.md)

Those Markdown guides are also source-checkout only. Pip users should follow
the CLI-only validation above and the Python API snippets in the [Usage Guide](usage.md).

## Notes

- Example modules are preferred over large inline doc snippets whenever a maintained script already exists.
- Some examples are environment-gated. For example, `examples.basic.pytorch_demo` skips itself when CUDA is unavailable.

---

[← Back to main docs](index.md)
