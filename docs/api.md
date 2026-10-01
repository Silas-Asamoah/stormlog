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
- `stormlog.telemetry_model`
- `stormlog.cpu_profiler`

Primary exported symbols:

- `GPUMemoryProfiler`
- `MemoryTracker`
- `CPUMemoryProfiler`
- `CPUMemoryTracker`
- `MemoryVisualizer`
- `TelemetryEventV2`
- `TelemetryEventV3`
- `TelemetryEventV4`
- `DeviceMemoryCapabilities`

`TelemetryEventV4` is the latest canonical event type. V2 and v3 remain public
for compatibility with existing artifacts; loading either version produces v4
events.

`MemoryTracker` accepts an optional collector extension seam. Pass either
`device` or `collector`, never both:

```python
from stormlog import MemoryTracker
from stormlog.device_collectors import (
    DeviceMemoryCapabilities,
    DeviceMemoryCollector,
    DeviceMemorySample,
)


class RuntimeCollector(DeviceMemoryCollector):
    def name(self) -> str:
        return "custom"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> DeviceMemoryCapabilities:
        return DeviceMemoryCapabilities(
            backend="custom",
            telemetry_collector="example.custom_tracker",
            sampling_source="custom.runtime.memory_info",
            supports_device_used=True,
            supports_device_free=True,
            supports_device_total=True,
        )

    def sample(self) -> DeviceMemorySample:
        used, free, total = read_runtime_memory_info()
        return DeviceMemorySample(
            device_id=0,
            allocated_bytes=None,
            reserved_bytes=None,
            active_bytes=None,
            inactive_bytes=None,
            used_bytes=used,
            free_bytes=free,
            total_bytes=total,
        )


tracker = MemoryTracker(collector=RuntimeCollector())
```

The example deliberately does not map device usage to allocator counters.
`GPUMemoryProfiler` remains allocator-native and rejects collectors whose
capabilities do not declare bounded profiling support.

Telemetry model symbols exported from `stormlog.telemetry`:

- `ProjectedTelemetryRecord`
- `project_telemetry_event`
- `project_telemetry_events`

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

### `stormlog.jax` (JAX workflows)

Use this package for:

- JAX memory profiling, XLA compilation tracking, and context instrumentation
- Tracking, analysis, and diagnose flows for JAX workloads on CUDA, TPU, and CPU

Core modules documented in generated pages:

- `stormlog.jax`
- `stormlog.jax.profiler`
- `stormlog.jax.tracker`
- `stormlog.jax.context_profiler`
- `stormlog.jax.analyzer`

Primary exported symbols:

- `JAXMemoryProfiler`
- `JAXProfiler`
- `MemoryTracker`
- `MemoryAnalyzer`
- `MemoryVisualizer`
- `profile_function`
- `profile_context`

JAX snapshots and profile results retain numeric fields for compatibility.
`MemorySnapshot` and `ProfileResult` expose `device_memory_available` and
`device_memory_unavailable_reason`. `TrackingResult` additionally exposes
`memory_source` and `process_memory_bytes`. Consumers must check
`device_memory_available` before interpreting a numeric device-memory value as
a measurement.

`JAXProfiler.profile_training` accepts re-iterable datasets, finite one-shot
iterators, and zero-argument dataset factories.  Use a factory for streaming
multi-epoch inputs, and combine large or infinite streams with
`steps_per_epoch`.  Callable objects that are already iterable are iterated
directly; wrap them in a zero-argument callable to force factory behavior.

## JAX Result Units

`stormlog.jax.profiler.ProfileResult.memory_growth_rate` reports memory growth
in MB/second.

Stormlog versions before `0.3.6` reported this property in bytes/second;
`0.3.6` and later report MB/second.

## Usage Paths

For task-oriented guidance:

- [Usage Guide](usage.md)
- [CLI Guide](cli.md)
- [Examples](examples.md)

---

[← Back to main docs](index.md)
