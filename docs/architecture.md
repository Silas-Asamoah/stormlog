[← Back to main docs](index.md)

# Architecture Guide

This document describes the architecture and design principles of Stormlog.

## Overview

Stormlog is designed with a modular, extensible architecture that supports both PyTorch and TensorFlow while maintaining clean separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stormlog                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   PyTorch   │  │ TensorFlow  │  │     CLI     │         │
│  │  Profiler   │  │  Profiler   │  │   Tools     │         │
│  │ (gpumemprof)│  │(tfmemprof)  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Core Components                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Profiler  │  │  Tracker    │  │ Visualizer  │         │
│  │             │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Analyzer   │  │   Utils     │  │   Context   │         │
│  │             │  │             │  │  Profiler   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Framework Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   PyTorch   │  │ TensorFlow  │  │    CPU      │         │
│  │   Memory    │  │   Memory    │  │   Memory    │         │
│  │  Interface  │  │  Interface  │  │  Interface  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Profiler (`profiler.py`)

The main profiling engine that coordinates memory monitoring and data collection.

**Responsibilities:**

- Initialize profiling sessions
- Coordinate data collection from framework layers
- Manage profiling state and configuration
- Provide high-level API for users

**Key Classes:**

- `GPUMemoryProfiler` (PyTorch -- `gpumemprof.profiler`)
- `TFMemoryProfiler` (TensorFlow -- `tfmemprof.profiler`)

### 2. Tracker (`tracker.py`)

Real-time memory tracking with background monitoring capabilities.

**Responsibilities:**

- Continuous memory monitoring
- Alert system for memory thresholds
- Background data collection
- Memory leak detection

**Key Classes:**

- `MemoryTracker` (exported from both packages)
- `TrackingEvent` (`gpumemprof`) / `TrackingResult` (`tfmemprof`)
- `MemoryWatchdog` (internal — not re-exported from package `__init__`)

### 3. Visualizer (`visualizer.py`)

Data visualization and reporting capabilities.

**Responsibilities:**

- Generate memory timeline plots
- Create heatmaps and charts
- Interactive dashboards
- Export visualizations

**Key Classes:**

- `MemoryVisualizer` (requires `[viz]` extra; uses matplotlib, seaborn, plotly internally)

### 4. Analyzer (`analyzer.py`)

Advanced analysis and optimization recommendations.

**Responsibilities:**

- Memory leak detection algorithms
- Performance analysis
- Optimization suggestions
- Pattern recognition

**Key Classes:**

- `MemoryAnalyzer`
- `GapFinding` (hidden-memory gap analysis)

### 5. Context Profiler (`context_profiler.py`)

Context-aware profiling with decorators and context managers.

**Responsibilities:**

- Function-level profiling
- Context manager support
- Decorator implementations
- Scope-based memory tracking

**Key Classes/Functions:**

- `profile_function` (decorator)
- `profile_context` (context manager)
- `MemoryProfiler` / `ProfiledModule` (`gpumemprof`)
- `TensorFlowProfiler` / `ProfiledLayer` (`tfmemprof`)

### 6. Utils (`utils.py`)

Utility functions and system information gathering.

**Responsibilities:**

- System information collection
- Memory formatting
- Framework detection
- Error handling

**Key Functions:**

- `get_gpu_info()` (`gpumemprof`) / `get_system_info()` (`tfmemprof`)
- `format_bytes()`, `convert_bytes()`
- `detect_torch_runtime_backend()` (`gpumemprof`)

### 7. CLI (`cli.py`)

Command-line interface for standalone usage.

**Responsibilities:**

- Command-line argument parsing
- Real-time monitoring interface
- Data export and analysis
- System information display

**Key Commands:**

- `info` - System information
- `monitor` - Real-time monitoring
- `track` - Background tracking
- `analyze` - Results analysis
- `diagnose` - Diagnostic bundle generation

### 8. OOM Flight Recorder (`oom_flight_recorder.py`)

Captures memory state before out-of-memory crashes for post-mortem analysis.

**Key Classes:**

- `OOMFlightRecorder`
- `OOMFlightRecorderConfig`
- `OOMExceptionClassification`

### 9. Device Collectors (`device_collectors.py`)

Backend-aware device memory sampling across CUDA, ROCm, and MPS.

**Key Classes:**

- `DeviceMemoryCollector` (abstract base)
- `CudaDeviceCollector`, `ROCmDeviceCollector`, `MPSDeviceCollector`
- `DeviceMemorySample`

### 10. Telemetry (`telemetry.py`)

Structured telemetry event schema for profiling data interchange.

**Key Classes:**

- `TelemetryEventV2`

## Framework-Specific Architecture

### PyTorch Profiler (`gpumemprof`)

```
┌─────────────────────────────────────────┐
│              gpumemprof                 │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Profiler  │  │  Context    │      │
│  │             │  │  Profiler   │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Tracker   │  │ Visualizer  │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  Analyzer   │  │    Utils    │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────┤
│              PyTorch Layer              │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ torch.cuda  │  │   Memory    │      │
│  │   Memory    │  │  Allocator  │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

**PyTorch-Specific Features:**

- Tensor lifecycle tracking
- CUDA memory management integration
- PyTorch-specific optimizations
- Autograd memory profiling

### TensorFlow Profiler (`tfmemprof`)

```
┌─────────────────────────────────────────┐
│              tfmemprof                  │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Profiler  │  │  Context    │      │
│  │             │  │  Profiler   │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Tracker   │  │ Visualizer  │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  Analyzer   │  │    Utils    │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────┤
│            TensorFlow Layer             │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Session   │  │   Graph     │      │
│  │  Memory     │  │ Execution   │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

**TensorFlow-Specific Features:**

- Session-based memory tracking
- Graph execution monitoring
- Keras model profiling
- Mixed precision support

## Data Flow

### 1. Initialization Flow

```
User Code → Profiler Init → Framework Detection → System Info → Ready
```

### 2. Profiling Flow

```
User Code → Context/Decorator → Memory Snapshot → Data Collection → Analysis
```

### 3. Monitoring Flow

```
Background Thread → Memory Sampling → Alert Check → Data Storage → Visualization
```

### 4. Analysis Flow

```
Collected Data → Pattern Detection → Leak Analysis → Optimization Suggestions → Reports
```

## Design Principles

### 1. Modularity

Each component has a single responsibility and can be used independently:

```python
# Use only the profiler
from gpumemprof import GPUMemoryProfiler
profiler = GPUMemoryProfiler()

# Use only the tracker
from gpumemprof import MemoryTracker
tracker = MemoryTracker()

# Use only the visualizer
from gpumemprof import MemoryVisualizer
visualizer = MemoryVisualizer()
```

### 2. Extensibility

The architecture supports easy extension through the device-collector abstraction:

```python
from gpumemprof.device_collectors import DeviceMemoryCollector, DeviceMemorySample

class NewBackendCollector(DeviceMemoryCollector):
    def collect(self) -> DeviceMemorySample:
        # Backend-specific memory sampling
        pass
```

### 3. Thread Safety

All components are designed to be thread-safe for concurrent usage:

```python
# Safe to use in multi-threaded environments
profiler = GPUMemoryProfiler()
profiler.start_monitoring()  # Background thread
# Main thread continues...
```

### 4. Performance

Minimal overhead design with configurable sampling:

```python
# Low overhead mode
profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=5.0)

# High precision mode
profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.1)
```

## Configuration Management

Configuration is handled through constructor arguments and CLI flags. There is no
external configuration file or environment variable interface at this time.

## Error Handling

### Graceful Degradation

```python
try:
    profiler = GPUMemoryProfiler()
except CUDAError:
    # Fall back to CPU mode
    from gpumemprof import CPUMemoryProfiler

    profiler = CPUMemoryProfiler()
```

## Testing Architecture

### Test Structure

Tests live in a flat `tests/` directory with framework-specific prefixes:

```
tests/
├── test_profiler.py             # Core PyTorch profiler
├── test_core_profiler.py        # Profiler integration
├── test_cpu_profiler.py         # CPU-only profiler
├── test_device_collectors.py    # Backend collectors
├── test_gap_analysis.py         # PyTorch gap analysis
├── test_oom_flight_recorder.py  # OOM recorder
├── test_telemetry_v2.py         # Telemetry schema
├── test_cli_info.py             # CLI info command
├── test_cli_diagnose.py         # CLI diagnose command
├── test_tf_*.py                 # TensorFlow-specific tests
├── test_utils.py                # Utility tests
├── test_benchmark_harness.py    # Performance budgets
├── test_docs_regressions.py     # Doc drift guard
├── tui/                         # TUI snapshot & pilot tests
└── e2e/                         # End-to-end tests
```

**Pytest markers** (defined in `pyproject.toml`): `unit`, `integration`, `slow`, `tui_pilot`, `tui_pty`, `tui_snapshot`.

### Mock Strategy

```python
# Mock CUDA for testing
@pytest.fixture
def mock_cuda():
    with patch('torch.cuda.is_available', return_value=True):
        yield
```

## Future Extensibility

### Plugin System

```python
class ProfilerPlugin:
    def on_memory_snapshot(self, snapshot):
        pass

    def on_leak_detected(self, leak):
        pass
```

### Custom Visualizations

```python
class CustomVisualizer(MemoryVisualizer):
    def create_custom_plot(self, data):
        # Custom visualization logic
        pass
```

### Framework Support

```python
# New frameworks can implement a DeviceMemoryCollector
# and integrate with the existing profiling pipeline.
```

---

[← Back to main docs](index.md)
