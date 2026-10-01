[← Back to main docs](index.md)

# Architecture Guide

This page describes the current code-level architecture of Stormlog. It is a source-of-truth guide for how the repo is organized today, not a roadmap.

## Repository surfaces

Stormlog has four user-facing surfaces:

- Python APIs for bounded profiling and background tracking
- CLI entrypoints for capture, analysis, and diagnose flows
- OpenAI-compatible inference endpoint profiling
- a Textual TUI for live monitoring, visualization export, and artifact review

Those surfaces are implemented under one package root:

- `stormlog` for PyTorch, CPU fallback utilities, telemetry normalization, inference profiling, local artifact queries, and the TUI
- `stormlog.tensorflow` for TensorFlow profiling, tracking, and TensorFlow-specific analysis helpers
- `stormlog.jax` for JAX profiling, tracking, diagnostics, and JAX-specific analysis helpers

## Package boundaries

### `stormlog`

The `stormlog` package owns:

- `GPUMemoryProfiler` for bounded PyTorch profiling
- `MemoryTracker` for capability-aware device tracking on CUDA, ROCm, MPS, or
  an injected collector; CPU tracking uses the dedicated fallback path
- `CPUMemoryProfiler` and `CPUMemoryTracker` for CPU-only workflows
- `MemoryVisualizer` for PNG, HTML, heatmap, and dashboard-style exports
- `MemoryAnalyzer`, `GapFinding`, and collective-attribution helpers
- `TelemetryEventV4` plus v2/v3 conversion and validation utilities
- device collector abstractions in `device_collectors.py`
- OpenAI-compatible inference profiling under `stormlog.infer`
- local artifact queries under `stormlog.query` and `stormlog.query_cli`
- the Textual TUI under `stormlog.tui`

### `stormlog.tensorflow`

The `stormlog.tensorflow` subpackage owns:

- `TFMemoryProfiler` for bounded TensorFlow profiling
- `TensorFlowProfiler` and `ProfiledLayer` in `context_profiler.py`
- `TensorFlowMemoryTracker` (an exported alias of `stormlog.tensorflow.tracker.MemoryTracker`)
- `TensorFlowVisualizer`
- `TensorFlowAnalyzer` and `TensorFlowGapFinding`
- TensorFlow runtime and backend diagnostics in `stormlog.tensorflow.utils`

The TensorFlow package does not ship a separate TUI. The shared terminal UI
lives in `stormlog.tui`; `stormlog.entrypoint` dispatches to that TUI or to
top-level command groups such as `stormlog query` and `stormlog infer`.

### `stormlog.jax`

The `stormlog.jax` subpackage owns:

- `JAXMemoryProfiler` for bounded JAX profiling
- JAX tracking, diagnostics, visualization, and analyzer helpers
- JAX runtime and device discovery helpers in `stormlog.jax.utils`

## High-level layering

```text
User code / shell
    |
    +-- Python APIs
    |     +-- stormlog.GPUMemoryProfiler
    |     +-- stormlog.MemoryTracker / CPUMemoryTracker
    |     +-- stormlog.tensorflow.TFMemoryProfiler
    |     +-- stormlog.tensorflow.TensorFlowMemoryTracker
    |     +-- stormlog.jax.JAXMemoryProfiler
    |
    +-- CLI entrypoints
    |     +-- gpumemprof
    |     +-- tfmemprof
    |     +-- jaxmemprof
    |     +-- stormlog / stormlog query / stormlog infer
    |
    +-- Shared artifact layer
          +-- TelemetryEventV4 JSON/CSV exports
          +-- inference JSONL exports
          +-- diagnose bundles
          +-- PNG / HTML visualization outputs
```

## Core modules and responsibilities

### Profilers

Bounded profilers are for "what happened inside this call or context?" questions.

- `stormlog.profiler.GPUMemoryProfiler`
- `stormlog.cpu_profiler.CPUMemoryProfiler`
- `stormlog.tensorflow.profiler.TFMemoryProfiler`

They expose:

- `profile_function(...)`
- `profile_context(...)`
- summary/result accessors such as `get_summary()` or `get_results()`
- optional live monitoring helpers such as `start_monitoring(...)`

### Trackers

Trackers are for "what happened over time?" questions.

- `stormlog.tracker.MemoryTracker`
- `stormlog.cpu_profiler.CPUMemoryTracker`
- `stormlog.tensorflow.tracker.MemoryTracker` exported as `TensorFlowMemoryTracker`

Trackers are responsible for:

- background sampling
- event generation
- threshold-triggered alerts
- timeline aggregation
- exportable telemetry events

### Telemetry

`stormlog.telemetry` is the shared interchange layer used by trackers, CLI tools, diagnostics, and the TUI.

Key responsibilities:

- normalize v2, v3, and recognized legacy records into canonical
  `TelemetryEventV4`
- validate event shape
- load saved event streams from disk
- group saved artifacts into session-aware capture units
- resolve distributed identity defaults from environment variables or explicit inputs

`stormlog.session` is the shared lifecycle contract used by trackers,
append-only sinks, diagnose bundles, OOM bundles, CLI analysis, and TUI
diagnostics.

It defines:

- unique `session_id` generation
- lifecycle states: `running`, `completed`, `interrupted`, `incomplete`
- session summaries with host, pid, distributed identity, and source metadata
- default session selection order for multi-session artifact roots

This shared schema is what allows Stormlog tracker exports, TensorFlow tracker
exports, diagnose bundles, and TUI diagnostics loading to operate on the same
underlying event model.

### Device collectors

`stormlog.device_collectors` is the backend-aware abstraction for PyTorch-side device memory sampling.

Current collector contract:

- `sample()` returns a normalized `DeviceMemorySample` whose allocator and
  device counters may be `None`
- `capabilities()` returns a frozen `DeviceMemoryCapabilities` describing each
  supported counter and allocator-native feature
- `name()` identifies the runtime backend (`cuda`, `rocm`, `mps`)

The tracker validates each sample against its capability declaration. Populated
unsupported fields are collector failures. Supported fields may be temporarily
missing only when collector diagnostics name them as partial. This makes the
distinction between allocator counters and whole-device used/free/total memory
explicit.

Current concrete collectors:

- `CudaDeviceCollector`
- `ROCmDeviceCollector`
- `MPSDeviceCollector`

Third-party runtimes can inject a `DeviceMemoryCollector` into
`MemoryTracker(collector=...)` without passing a torch device. No global backend
registry is required. Device-only collectors retain session, distributed,
append-only sink, query, and TUI behavior while allocator allocation events,
fragmentation, attribution, native history, and bounded profiling remain
unavailable.

### Inference profiling

`stormlog.infer` owns active OpenAI-compatible endpoint profiling. It is kept
separate from `gpumemprof` and `tfmemprof` because the endpoint backend may be
PyTorch, vLLM, SGLang, TensorRT-LLM, MLX-LM, a hosted gateway, or another
server.

Current responsibilities:

- build workload cases from concurrency, prompt token targets, and output caps
- send Chat Completions requests and parse streaming or non-streaming responses
- record client-observed request traces with token count provenance
- write inference-specific JSONL artifacts
- analyze those artifacts into latency, throughput, failure, and memory summaries

### Analyzers

Analyzers turn raw or normalized memory data into higher-level findings.

- `stormlog.analyzer.MemoryAnalyzer`
- `stormlog.tensorflow.analyzer.MemoryAnalyzer`
- gap-analysis and collective-attribution helpers in `stormlog`
- common metric formulas centralized in `stormlog.derived_fields`

These modules power:

- leak and growth heuristics
- hidden-memory gap analysis
- distributed diagnostics summaries
- recommendation text in CLI or artifact flows

### Visualizers

Visualizers convert profiler or tracker output into human-readable plots.

- `stormlog.visualizer.MemoryVisualizer`
- `stormlog.tensorflow.visualizer.MemoryVisualizer`

The PyTorch-side visualizer also underpins the TUI plot export path for:

- PNG timeline plots
- HTML timeline plots
- heatmaps
- multi-panel dashboard exports

## TUI architecture

The `stormlog` console script points to `stormlog.entrypoint:main`. Running
`stormlog` with no arguments still launches `stormlog.tui:run_app`; command
groups such as `stormlog query` and `stormlog infer` dispatch through the same
top-level entrypoint.

The TUI is assembled from:

- `stormlog.tui.app` for the main Textual application
- `stormlog.tui.monitor.TrackerSession` for adapting tracker data into the UI
- `stormlog.tui.distributed_diagnostics` for artifact loading and rank-level summaries
- `stormlog.tui.widgets.*` for tables, panels, and timeline rendering

Current tabs are:

- `Overview`
- `PyTorch`
- `TensorFlow`
- `Monitoring`
- `Visualizations`
- `Diagnostics`
- `CLI & Actions`

The TUI is not a separate analysis engine. It reuses:

- tracker sessions for live data
- TelemetryEventV4 records for artifact loading
- session-aware artifact loading so users can switch between discovered captures
- `MemoryVisualizer`-style plot generation for PNG/HTML export

## Main runtime flows

### 1. Bounded profiling flow

```text
User code
  -> profiler.profile_function(...) or profiler.profile_context(...)
  -> framework/runtime-specific snapshots
  -> in-memory result object
  -> summary/report accessors
```

### 2. Tracking flow

```text
Tracker start
  -> create session summary
  -> periodic sampling
  -> alert evaluation
  -> event storage
  -> append-only sink / OOM artifacts reference session_id
  -> statistics / timeline / export helpers
  -> clean stop marks session completed
```

### 3. Diagnose flow

```text
CLI diagnose
  -> create session summary
  -> runtime/system info
  -> telemetry capture
  -> manifest records session_id and terminal status
  -> artifact bundle on disk
  -> later reload in TUI Diagnostics or analyzer paths
```

### 4. TUI flow

```text
TrackerSession or artifact path input
  -> normalized telemetry events
  -> timeline / rank-table rendering
  -> optional PNG or HTML export
```

## Configuration model

Stormlog configuration is currently local to:

- constructor arguments
- method parameters
- CLI flags
- distributed identity environment inference inside telemetry helpers

There is no repo-level persistent config file format today.

## Error-handling model

The codebase prefers capability-gated behavior over silent fallback.

Examples:

- PyTorch-specific APIs raise import/runtime errors when `torch` is missing
- `GPUMemoryProfiler` is for CUDA-backed profiling, while CPU-only workflows use separate CPU profiler classes
- TUI startup currently hard-imports `torch`, so `stormlog` requires the current TUI plus PyTorch dependency path
- telemetry loaders collect warnings when artifact payloads are malformed or incomplete

## Test architecture

The repo test layout is split by behavior, not by package alone:

```text
tests/
  test_*.py          # core, CLI, telemetry, analyzer, and framework tests
  tui/               # Textual pilot and snapshot coverage
  e2e/               # PTY smoke coverage
```

Current marker families used in the repo:

- `slow`
- `integration`
- `unit`
- `tui_pilot`
- `tui_pty`
- `tui_snapshot`

The operational guide for running those slices lives in the [Testing and Validation Guide](testing.md).

## Extensibility points that exist today

The repo currently exposes a few real extension seams:

- backend collection through `DeviceMemoryCollector`
- telemetry normalization through `stormlog.telemetry`
- new CLI/documentation workflows through example modules and diagnose artifacts
- new TUI tables or views through `stormlog.tui.widgets`

Anything beyond those seams should be treated as new feature work, not assumed architecture.

## Related guides

- [Usage Guide](usage.md)
- [CLI Guide](cli.md)
- [TUI Guide](tui.md)
- [API Reference](api.md)

---

[← Back to main docs](index.md)
