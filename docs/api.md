[← Back to main docs](index.md)

# API Reference

This page is the human-first API guide for the package. The canonical function and class signatures are generated from source code and docstrings during the docs build.

Install the distribution as `stormlog`, then import the Python APIs from
`stormlog` or `stormlog.tensorflow`.

## Generated Reference

- [Generated API Modules](reference/index.md)

The pages under `docs/reference/generated/` are build output, not hand-maintained source files.

## Package Surface

### `stormlog` (PyTorch + shared collectors)

Use this package for:

- Profiling memory usage in PyTorch workflows
- Running background tracking and telemetry export
- OOM flight-recorder capture and diagnostics
- CPU-only profiling utilities

Core modules documented in generated pages:

- `stormlog`
- `stormlog.profiler`
- `stormlog.tracker`
- `stormlog.telemetry`
- `stormlog.cpu_profiler`

Primary exported symbols:

- `GPUMemoryProfiler`
- `MemoryTracker`
- `CPUMemoryProfiler`
- `CPUMemoryTracker`
- `MemoryVisualizer`
- `TelemetryEventV2`

### `stormlog.tensorflow` (TensorFlow workflows)

Use this package for:

- TensorFlow memory profiling and context instrumentation
- Tracking, analysis, and diagnose flows for TF workloads

Core modules documented in generated pages:

- `stormlog.tensorflow`
- `stormlog.tensorflow.profiler`
- `stormlog.tensorflow.tracker`
- `stormlog.tensorflow.context_profiler`

Primary exported symbols:

- `TFMemoryProfiler`
- `TensorFlowProfiler`
- `TensorFlowMemoryTracker`
- `TensorFlowVisualizer`
- `TensorFlowAnalyzer`
- `TensorFlowGapFinding`

## Usage Paths

For task-oriented guidance:

- [Usage Guide](usage.md)
- [CLI Guide](cli.md)
- [Examples](examples.md)

---

[← Back to main docs](index.md)
