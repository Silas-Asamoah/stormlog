[← Back to main docs](index.md)

# PyTorch Testing & Usage Guide

**Authors:** Prince Agyei Tuffour and Silas Bempong
**Last Updated:** June 2025

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Quick Start](#quick-start)
5. [GPU Profiling (CUDA Available)](#gpu-profiling-cuda-available)
6. [CPU Profiling (No CUDA Required)](#cpu-profiling-no-cuda-required)
7. [Testing Framework](#testing-framework)
8. [Usage Examples](#usage-examples)
9. [Command Line Interface](#command-line-interface)
10. [Troubleshooting](#troubleshooting)
11. [Performance Benchmarks](#performance-benchmarks)
12. [Advanced Features](#advanced-features)

---

> **Note:** Legacy scripts under `examples/test_guides/` have been
> replaced by curated Markdown workflows.
> Use `docs/examples/test_guides/README.md` plus the modules in
> `examples/basic`, `examples/advanced`, and `examples/cli` for hands-on tests.

## Overview

The PyTorch GPU Memory Profiler is a comprehensive tool for monitoring and optimizing memory usage during PyTorch model training and inference. This guide provides complete instructions for testing and using the profiler in both GPU and CPU environments.

### Key Features

✅ **GPU Memory Profiling** (CUDA required)

-   Real-time GPU memory tracking
-   CUDA memory allocation monitoring
-   GPU memory fragmentation analysis
-   CUDA-specific optimizations

✅ **CPU Memory Profiling** (No CUDA required)

-   System RAM usage monitoring
-   Process memory tracking
-   CPU-based model profiling
-   Cross-platform compatibility

✅ **Universal Features** (Both GPU and CPU)

-   Function execution timing
-   Context-based profiling
-   Memory leak detection
-   Real-time monitoring
-   Data visualization
-   Export capabilities

---

## System Requirements

### Minimum Requirements

-   **Python**: 3.10 or higher
-   **PyTorch**: 1.8.0 or higher
-   **System Memory**: 4GB RAM
-   **Storage**: 100MB free space

### For GPU Profiling

-   **CUDA**: 10.2 or higher
-   **GPU**: NVIDIA GPU with CUDA support
-   **GPU Memory**: 2GB or higher recommended
-   **NVIDIA Drivers**: Compatible with CUDA version

### For CPU-Only Profiling

-   **CPU**: Any modern CPU (2+ cores recommended)
-   **RAM**: 4GB or higher
-   **OS**: Windows, macOS, or Linux

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/Silas-Asamoah/gpu-memory-profiler.git
cd gpu-memory-profiler
```

### Step 2: Choose Your Installation

#### For GPU Systems (CUDA Available)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install profiler dependencies
pip install -r requirements.txt

# Install the profiler package
pip install -e .
```

#### For CPU-Only Systems

```bash
# Install CPU-only PyTorch (smaller download)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install profiler dependencies
pip install psutil matplotlib numpy

# Install the profiler package
pip install -e .
```

### Step 3: Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check profiler installation
python -c "from gpumemprof import GPUMemoryProfiler; print('Profiler installed successfully!')"
```

---

## Quick Start

### 🚀 Ultra-Quick Test (30 seconds)

Choose your scenario:

#### With GPU:

```bash
python -m examples.basic.pytorch_demo
# Optional advanced tracker demo
python -m examples.advanced.tracking_demo
```

#### CPU Only:

```bash
python -m examples.cli.quickstart
```

### Expected Output:

```
⚡ Quick Test - Basic Profiler Functionality
✅ Quick test passed!
✅ Result: 1234567.89
✅ Peak memory: 15.28 MB
✅ Functions profiled: 1
```

If you see ✅ indicators, you're ready to go!

---

## GPU Profiling (CUDA Available)

### Prerequisites Check

First, verify your GPU setup:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check GPU information
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

# Check GPU memory
nvidia-smi
```

### Testing the GPU Profiler

Run the curated demos (these are the same ones exercised in CI):

```bash
python -m examples.basic.pytorch_demo
python -m examples.advanced.tracking_demo
```

For scenario-by-scenario instructions (context profiling, leak detection,
watchdog tuning, exports, CLI workflows), see
`docs/examples/test_guides/README.md#pytorch-gpu-checklist`.

### Basic GPU Profiling Usage

#### Function Profiling

```python
import torch
from gpumemprof import GPUMemoryProfiler

# Initialize profiler
profiler = GPUMemoryProfiler()

def gpu_operation():
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)
    return torch.sum(y)

# Run profiled function
profile = profiler.profile_function(gpu_operation)

# Get summary
summary = profiler.get_summary()
print(f"Peak GPU memory: {summary['peak_memory_usage'] / (1024**2):.2f} MB")
print(f"Execution time: {profile.execution_time:.4f} seconds")
```

#### Context Profiling

```python
import torch
import torch.nn as nn
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

with profiler.profile_context("model_training"):
    # Create model
    model = nn.Linear(1000, 100).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # Training step
    inputs = torch.randn(64, 1000, device='cuda')
    targets = torch.randn(64, 100, device='cuda')

    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# View results
summary = profiler.get_summary()
for context, stats in summary["function_summaries"].items():
    print(f"{context}: {stats['total_time']:.3f}s, {stats['total_memory_allocated'] / (1024**2):.2f} MB")
```

#### Real-Time Monitoring

```python
from gpumemprof import MemoryTracker

# Setup tracker with alerts
tracker = MemoryTracker(
    sampling_interval=0.1,     # Sample every 100ms
    enable_alerts=True
)

# Start tracking
tracker.start_tracking()

try:
    # Your GPU operations here
    for i in range(10):
        x = torch.randn(500, 500, device='cuda')
        y = torch.matmul(x, x.T)
        print(f"Iteration {i+1}/10")
finally:
    # Stop tracking
    tracker.stop_tracking()

# Analyze results
stats = tracker.get_statistics()
alert_events = tracker.get_events(event_type='warning') + tracker.get_events(event_type='critical')
print(f"Peak memory: {stats.get('peak_memory', 0) / (1024**2):.2f} MB")
print(f"Alerts: {len(alert_events)}")
```

### GPU Command Line Tools

```bash
# Show GPU information
python -m gpumemprof.cli info

# Real-time monitoring (30 seconds)
python -m gpumemprof.cli monitor --interval 1.0 --duration 30

# Background tracking
python -m gpumemprof.cli track --output gpu_results.json --duration 60

# Analyze results
python -m gpumemprof.cli analyze gpu_results.json --visualization
```

---

## CPU Profiling (No CUDA Required)

### When to Use CPU Profiling

-   CUDA is not available
-   Development and testing
-   Learning memory profiling concepts
-   CPU-based inference optimization
-   Educational purposes

### Testing the CPU Profiler

Use the CLI and basic demo to validate CPU-only environments:

```bash
CUDA_VISIBLE_DEVICES="" gpumemprof info
python -m examples.cli.quickstart
```

For deeper CPU scenarios, follow the CPU section in
`docs/examples/test_guides/README.md`.

### Basic CPU Profiling Usage

#### Simple CPU Profiler

```python
import torch
import psutil
import time

class CPUMemoryProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.results = []

    def profile_function(self, func):
        def wrapper(*args, **kwargs):
            before_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
            start_time = time.time()

            result = func(*args, **kwargs)

            after_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
            duration = time.time() - start_time

            self.results.append({
                'function': func.__name__,
                'memory_used_mb': after_mem - before_mem,
                'duration_s': duration
            })

            return result
        return wrapper

    def get_summary(self):
        for result in self.results:
            print(f"{result['function']}: {result['duration_s']:.4f}s, {result['memory_used_mb']:.2f}MB")

# Usage
profiler = CPUMemoryProfiler()

@profiler.profile_function
def cpu_operation():
    x = torch.randn(1000, 1000)  # CPU tensor
    y = torch.matmul(x, x.T)
    return torch.mean(y)

result = cpu_operation()
profiler.get_summary()
```

#### CPU Model Training

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_cpu_model():
    # Create CPU model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )  # No .cuda() call

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training on CPU...")
    for epoch in range(5):
        # Generate batch data (CPU)
        inputs = torch.randn(64, 784)
        targets = torch.randint(0, 10, (64,))

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

train_cpu_model()
```

#### CPU Memory Tracking

```python
import psutil
import time
import threading

class CPUMemoryTracker:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.tracking = False
        self.samples = []

    def start_tracking(self):
        self.tracking = True
        self.samples = []

        def track():
            while self.tracking:
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                self.samples.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                time.sleep(self.interval)

        threading.Thread(target=track, daemon=True).start()

    def stop_tracking(self):
        self.tracking = False

    def get_results(self):
        if not self.samples:
            return {}

        memories = [s['memory_mb'] for s in self.samples]
        return {
            'peak_memory_mb': max(memories),
            'average_memory_mb': sum(memories) / len(memories),
            'samples_count': len(self.samples)
        }

# Usage
tracker = CPUMemoryTracker()
tracker.start_tracking()

# Your CPU operations here
for i in range(5):
    data = torch.randn(1000, 1000)
    result = torch.fft.fft(data, dim=0)
    time.sleep(0.5)

tracker.stop_tracking()
results = tracker.get_results()
print(f"Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
```

---

## Testing Framework

### Automated Test Execution

Both test suites provide comprehensive validation:

#### GPU & CPU Checklists

See `docs/examples/test_guides/README.md` for the full set of scenarios. Each
checklist maps to the curated modules:

- `examples/basic/pytorch_demo.py`
- `examples/advanced/tracking_demo.py`
- `examples/cli/quickstart.py`

These replace the monolithic `pytorch_profiler_test_guide.py` and
`cpu_profiler_test_guide.py` scripts.

### Expected Test Results

#### Successful GPU Test Output

```
🔍 Checking Requirements...
✅ CUDA Available: 11.8
✅ GPU Device: NVIDIA GeForce RTX 3080
✅ GPU Memory: 10.0 GB
✅ PyTorch Version: 2.0.1
✅ All requirements satisfied!

🧪 Test 1: Basic Function Profiling
✅ Peak memory usage: 195.31 MB
✅ Functions profiled: 2

🧪 Test 2: Context-Based Profiling
✅ Context blocks profiled: 3
   - data_preparation: 0.002s, 3.94 MB
   - model_creation: 0.015s, 2.18 MB
   - training_step: 0.008s, 0.51 MB

🎉 All Tests Completed Successfully!
```

#### Successful CPU Test Output

```
🔍 Checking System Configuration...
✅ PyTorch Version: 2.0.1
✅ CUDA Available: False
✅ CPU Count: 8
✅ System Memory: 16.0 GB

🧪 Test 1: Basic CPU Function Profiling
✅ Peak CPU memory: 125.45 MB
✅ Functions profiled: 2

🧪 Test 2: CPU Model Training Profiling
✅ Peak CPU memory: 89.23 MB
✅ Context blocks profiled: 7

🎉 All CPU Tests Completed Successfully!
```

---

## Usage Examples

### Example 1: Basic Model Training Profiling

#### GPU Version

```python
import torch
import torch.nn as nn
from gpumemprof import GPUMemoryProfiler, MemoryVisualizer

# Initialize profiler
profiler = GPUMemoryProfiler(track_tensors=True)

# Create model
with profiler.profile_context("model_setup"):
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    with profiler.profile_context(f"epoch_{epoch}"):

        # Data generation
        with profiler.profile_context("data_gen"):
            inputs = torch.randn(128, 1000, device='cuda')
            targets = torch.randint(0, 10, (128,), device='cuda')

        # Forward pass
        with profiler.profile_context("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass
        with profiler.profile_context("backward"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

# Generate report
summary = profiler.get_summary()
print(f"\nTraining completed!")
print(f"Peak GPU memory: {summary['peak_memory_usage'] / (1024**2):.2f} MB")
print(f"Function calls: {summary['total_function_calls']}")

# Create visualizations
visualizer = MemoryVisualizer(profiler)
visualizer.plot_memory_timeline(interactive=False, save_path='training_memory.png')
visualizer.export_data(format='json', save_path='training_results')
```

#### CPU Version

```python
import torch
import torch.nn as nn

# Initialize CPU profiler
profiler = CPUMemoryProfiler()

# Create model (CPU)
with profiler.profile_context("model_setup"):
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )  # No .cuda()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    with profiler.profile_context(f"epoch_{epoch}"):

        # Data generation (CPU)
        inputs = torch.randn(64, 1000)  # Smaller batch for CPU
        targets = torch.randint(0, 10, (64,))

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

# Generate report
results = profiler.get_summary()
print(f"\nCPU Training completed!")
print(f"Peak CPU memory: {results['peak_memory_usage'] / (1024**2):.2f} MB")

# Show context breakdown
for context, stats in results['function_summaries'].items():
    print(f"{context}: {stats['total_time']:.3f}s, {stats['total_memory_allocated'] / (1024**2):.2f} MB")
```

### Example 2: Memory Leak Detection

#### GPU Version

```python
from gpumemprof import MemoryTracker

# Setup tracking
tracker = MemoryTracker(sampling_interval=0.05)
tracker.start_tracking()

try:
    # Simulate memory leak
    leaked_tensors = []

    for i in range(20):
        # Create tensor without proper cleanup
        tensor = torch.randn(1000, 1000, device='cuda')
        leaked_tensors.append(tensor)

        # Some computation
        result = torch.matmul(tensor, tensor.T)

        if i % 5 == 0:
            print(f"Iteration {i+1}/20")

finally:
    tracker.stop_tracking()

# Analyze for suspicious growth
events = tracker.get_events()
if len(events) > 1:
    growth_mb = (events[-1].memory_allocated - events[0].memory_allocated) / (1024**2)
    print(f"Memory growth during run: {growth_mb:.2f} MB")
    if growth_mb > 100:
        print("Potential memory leak detected")
    else:
        print("No significant memory growth")

# Cleanup (fix the leak)
del leaked_tensors
torch.cuda.empty_cache()
```

#### CPU Version

```python
from gpumemprof.cpu_profiler import CPUMemoryTracker

# Setup CPU tracking
tracker = CPUMemoryTracker(sampling_interval=0.05)
tracker.start_tracking()

try:
    # Simulate CPU memory leak
    leaked_arrays = []

    for i in range(15):
        # Create large array without cleanup
        import numpy as np
        array = np.random.randn(1000000)  # ~8MB
        leaked_arrays.append(array)

        # Some computation
        result = np.sum(array)

        if i % 3 == 0:
            print(f"Iteration {i+1}/15")

finally:
    tracker.stop_tracking()

# Analyze results
samples = tracker.get_events(event_type='allocation')

if len(samples) > 1:
    initial_memory = samples[0].memory_allocated / (1024**2)
    final_memory = samples[-1].memory_allocated / (1024**2)
    growth = final_memory - initial_memory

    print(f"Memory growth: {growth:.2f} MB")
    if growth > 50:  # 50MB threshold
        print("🔴 Potential memory leak detected!")
    else:
        print("✅ No significant memory leak")

# Cleanup
del leaked_arrays
```

### Example 3: Performance Comparison

```python
import torch
import time

def compare_operations():
    """Compare different PyTorch operations."""

    operations = [
        ("Small Matrix Multiply", lambda: torch.matmul(torch.randn(100, 100), torch.randn(100, 100))),
        ("Large Matrix Multiply", lambda: torch.matmul(torch.randn(1000, 1000), torch.randn(1000, 1000))),
        ("Convolution", lambda: torch.nn.functional.conv2d(torch.randn(1, 3, 224, 224), torch.randn(64, 3, 3, 3))),
        ("FFT", lambda: torch.fft.fft(torch.randn(1000000))),
    ]

    # GPU version (if available)
    if torch.cuda.is_available():
        print("GPU Performance:")
        for name, op in operations:
            # Move to GPU
            def gpu_op():
                return op().cuda() if hasattr(op(), 'cuda') else op()

            start_time = time.time()
            result = gpu_op()
            torch.cuda.synchronize()  # Wait for GPU
            duration = time.time() - start_time

            memory_used = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  {name}: {duration:.4f}s, {memory_used:.2f}MB")

            torch.cuda.empty_cache()

    # CPU version
    print("CPU Performance:")
    for name, op in operations:
        start_time = time.time()
        result = op()
        duration = time.time() - start_time
        print(f"  {name}: {duration:.4f}s")

compare_operations()
```

---

## Command Line Interface

### GPU CLI Commands

```bash
# System information
gpumemprof info
# Output: GPU details, CUDA version, memory info

# Real-time monitoring
gpumemprof monitor --interval 1.0 --duration 30
# Monitor for 30 seconds, sample every 1 second

# Background tracking
gpumemprof track --output results.json --warning-threshold 70 --critical-threshold 90
# Track with warning/critical thresholds, save to file

# Analysis
gpumemprof analyze results.json --visualization --plot-dir analysis_plots
# Analyze saved results and create plots
```

### CPU CLI Alternative

```bash
# For CPU monitoring, use system tools:

# Monitor system memory
watch -n 1 'free -h'

# Monitor process memory
top -p $(pgrep python)

# Profile Python script
python -m memory_profiler your_script.py

# Custom monitoring script
python -c "
import psutil
import time
for i in range(30):
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.used/1e9:.1f}GB ({mem.percent:.1f}%)')
    time.sleep(1)
"
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms:**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

```python
# 1. Reduce batch size
batch_size = 32  # Instead of 128

# 2. Use gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)

# 3. Clear cache regularly
torch.cuda.empty_cache()

# 4. Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Issue 2: Import Errors

**Symptoms:**

```
ImportError: No module named 'gpumemprof'
```

**Solutions:**

```bash
# Reinstall package
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
pip list | grep gpumemprof
```

#### Issue 3: Visualization Errors

**Symptoms:**

```
ImportError: No module named 'matplotlib'
```

**Solutions:**

```bash
# Install visualization dependencies
pip install matplotlib plotly

# For headless servers
export MPLBACKEND=Agg

# Alternative: export data only
visualizer.export_data(results, format='csv')
```

#### Issue 4: CPU Performance Issues

**Symptoms:**

-   Very slow execution on CPU
-   High memory usage

**Solutions:**

```python
# 1. Reduce problem size
inputs = torch.randn(32, 784)  # Instead of 128

# 2. Use CPU-optimized operations
torch.set_num_threads(4)  # Limit CPU threads

# 3. Enable MKLDNN (if available)
torch.backends.mkldnn.enabled = True

# 4. Monitor memory usage
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

#### Issue 5: Permission Errors

**Symptoms:**

```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

```bash
# Use user installation
pip install --user -e .

# Check permissions
ls -la

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Debug Mode

Enable debug mode for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# For GPU profiler
profiler = GPUMemoryProfiler(collect_stack_traces=True)

# For CPU profiler
import traceback
traceback.print_stack()
```

---

## Performance Benchmarks

### Expected Performance

#### GPU Profiling Overhead

| Operation                   | Baseline | With Profiling | Overhead |
| --------------------------- | -------- | -------------- | -------- |
| Matrix Multiply (1000x1000) | 0.005s   | 0.006s         | 20%      |
| CNN Forward Pass            | 0.050s   | 0.055s         | 10%      |
| Memory Tracking             | -        | 0.001s         | Minimal  |
| Visualization               | -        | 0.100s         | One-time |

#### CPU Profiling Overhead

| Operation                   | Baseline | With Profiling | Overhead |
| --------------------------- | -------- | -------------- | -------- |
| Matrix Multiply (1000x1000) | 0.100s   | 0.102s         | 2%       |
| Model Forward Pass          | 0.500s   | 0.510s         | 2%       |
| Memory Tracking             | -        | 0.001s         | Minimal  |

#### Memory Usage

| Component         | GPU Memory | CPU Memory |
| ----------------- | ---------- | ---------- |
| Profiler Overhead | ~10MB      | ~5MB       |
| Data Storage      | ~1MB/hour  | ~1MB/hour  |
| Visualization     | ~50MB      | ~50MB      |

### Optimization Tips

#### For GPU:

```python
# 1. Reduce profiling frequency
profiler = GPUMemoryProfiler(
    track_tensors=False  # Disable detailed tracking
)

# 2. Use sampling
tracker = MemoryTracker(sampling_interval=1.0)  # Sample every second

# 3. Batch operations
with profiler.profile_context("batch_operations"):
    for i in range(100):
        # Multiple operations in one context
        pass
```

#### For CPU:

```python
# 1. Increase sampling interval
tracker = CPUMemoryTracker(sampling_interval=0.5)  # Sample every 500ms

# 2. Limit tracked operations
def important_function():  # Only profile critical functions
    pass

profile = profiler.profile_function(important_function)

# 3. Use process-level tracking
import psutil
process = psutil.Process()
memory_info = process.memory_info()  # Single measurement
```

---

## Advanced Features

### Custom Memory Tracking

#### GPU Custom Tracking

```python
from gpumemprof import GPUMemoryProfiler

class CustomGPUProfiler(GPUMemoryProfiler):
    def __init__(self):
        super().__init__()
        self.custom_metrics = {}

    def track_custom_metric(self, name, value):
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append({
            'timestamp': time.time(),
            'value': value,
            'gpu_memory': torch.cuda.memory_allocated()
        })

    def get_custom_report(self):
        return self.custom_metrics

# Usage
profiler = CustomGPUProfiler()

for epoch in range(5):
    # Your training code
    loss = train_one_epoch()

    # Track custom metrics
    profiler.track_custom_metric('loss', loss.item())
    profiler.track_custom_metric('learning_rate', optimizer.param_groups[0]['lr'])

report = profiler.get_custom_report()
```

#### CPU Custom Tracking

```python
import psutil
import time

class AdvancedCPUProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.metrics = {
            'memory': [],
            'cpu_percent': [],
            'io_counters': []
        }

    def capture_metrics(self, label=""):
        timestamp = time.time()

        # Memory
        memory_info = self.process.memory_info()
        self.metrics['memory'].append({
            'timestamp': timestamp,
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'label': label
        })

        # CPU
        cpu_percent = self.process.cpu_percent()
        self.metrics['cpu_percent'].append({
            'timestamp': timestamp,
            'percent': cpu_percent,
            'label': label
        })

        # I/O
        try:
            io_counters = self.process.io_counters()
            self.metrics['io_counters'].append({
                'timestamp': timestamp,
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'label': label
            })
        except AttributeError:
            pass  # Not available on all platforms

    def get_summary(self):
        summary = {}

        # Memory summary
        memories = [m['rss'] for m in self.metrics['memory']]
        summary['memory'] = {
            'peak_mb': max(memories) / (1024 * 1024),
            'average_mb': sum(memories) / len(memories) / (1024 * 1024),
            'samples': len(memories)
        }

        # CPU summary
        cpu_values = [c['percent'] for c in self.metrics['cpu_percent'] if c['percent'] > 0]
        if cpu_values:
            summary['cpu'] = {
                'peak_percent': max(cpu_values),
                'average_percent': sum(cpu_values) / len(cpu_values)
            }

        return summary

# Usage
profiler = AdvancedCPUProfiler()

for epoch in range(5):
    profiler.capture_metrics(f"epoch_{epoch}_start")

    # Your training code
    train_one_epoch()

    profiler.capture_metrics(f"epoch_{epoch}_end")

summary = profiler.get_summary()
print(f"Peak memory: {summary['memory']['peak_mb']:.2f} MB")
print(f"Average CPU: {summary.get('cpu', {}).get('average_percent', 0):.1f}%")
```

### Integration with ML Frameworks

#### MLflow Integration

```python
import mlflow
from gpumemprof import GPUMemoryProfiler, MemoryVisualizer

# Start MLflow run
with mlflow.start_run():
    profiler = GPUMemoryProfiler()

    # Your training code with profiling
    for epoch in range(10):
        with profiler.profile_context(f"epoch_{epoch}"):
            loss = train_one_epoch()

        # Log metrics to MLflow
        results = profiler.get_summary()
        mlflow.log_metric("gpu_memory_mb", results["peak_memory_usage"] / (1024**2), step=epoch)
        mlflow.log_metric("train_loss", loss, step=epoch)

    # Log final profiling results
    final_results = profiler.get_summary()
    mlflow.log_metric("peak_gpu_memory", final_results["peak_memory_usage"] / (1024**2))
    mlflow.log_metric("total_profiled_calls", final_results["total_function_calls"])

    # Export and log profiling data
    visualizer = MemoryVisualizer(profiler)
    export_path = visualizer.export_data(format="json", save_path="profiling_results")
    mlflow.log_artifact(export_path)
```

#### Weights & Biases Integration

```python
import wandb
from gpumemprof import GPUMemoryProfiler, MemoryVisualizer

# Initialize wandb
wandb.init(project="memory-profiling")

profiler = GPUMemoryProfiler()

for epoch in range(10):
    with profiler.profile_context(f"epoch_{epoch}"):
        loss = train_one_epoch()

    # Log to wandb
    results = profiler.get_summary()
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "gpu_memory_mb": results["peak_memory_usage"] / (1024**2),
        "profiled_calls": results["total_function_calls"],
    })

# Upload profiling visualization
visualizer = MemoryVisualizer(profiler)
visualizer.plot_memory_timeline(interactive=False, save_path='memory_plot.png')
wandb.log({"memory_timeline": wandb.Image('memory_plot.png')})
```

### Automated Optimization

```python
from gpumemprof import GPUMemoryProfiler, MemoryAnalyzer

class AutoOptimizer:
    def __init__(self):
        self.profiler = GPUMemoryProfiler()
        self.analyzer = MemoryAnalyzer()

    def optimize_batch_size(self, model, initial_batch_size=64):
        """Automatically find optimal batch size."""
        best_batch_size = initial_batch_size
        max_throughput = 0

        for batch_size in [16, 32, 64, 128, 256]:
            try:
                print(f"Testing batch size: {batch_size}")

                with self.profiler.profile_context(f"batch_size_{batch_size}"):
                    # Test training step
                    inputs = torch.randn(batch_size, 784, device='cuda')
                    outputs = model(inputs)
                    loss = outputs.sum()
                    loss.backward()

                results = self.profiler.get_summary()
                memory_used = results["peak_memory_usage"] / (1024**2)
                duration = max(results["total_execution_time"], 1e-6)
                throughput = batch_size / duration

                print(f"  Memory: {memory_used:.1f}MB, Throughput: {throughput:.1f} samples/s")

                if throughput > max_throughput and memory_used < 8000:  # 8GB limit
                    max_throughput = throughput
                    best_batch_size = batch_size

                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at batch size {batch_size}")
                    break
                raise

        print(f"Optimal batch size: {best_batch_size}")
        return best_batch_size

# Usage
optimizer = AutoOptimizer()
model = MyModel().cuda()
optimal_batch_size = optimizer.optimize_batch_size(model)
```

---

## Conclusion

This comprehensive guide provides everything you need to test and use the PyTorch GPU Memory Profiler in both GPU and CPU environments. Whether you're debugging memory issues, optimizing performance, or learning about memory management, the profiler provides valuable insights and tools.

### Quick Reference

**GPU Users:**

```bash
python -m examples.basic.pytorch_demo          # Quick test
python -m examples.advanced.tracking_demo      # Tracker/watchdog demo
```

**CPU Users:**

```bash
python -m examples.cli.quickstart      # Quick test
gpumemprof info                        # System summary
```

**Getting Help:**

-   Check the [API Reference](api.md) for detailed API documentation
-   Review the [Troubleshooting Guide](troubleshooting.md) for common issues
-   Run tests with `--help` for usage options
-   Examine example files in `examples/` directory

### Next Steps

1. **Start Testing**: Run the appropriate test suite for your system
2. **Explore Examples**: Try the provided usage examples
3. **Integrate**: Add profiling to your existing PyTorch projects
4. **Optimize**: Use insights to improve memory efficiency
5. **Contribute**: Share your improvements with the community

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/Silas-Asamoah/gpu-memory-profiler).

---

**Happy Profiling!** 🚀

_Authors: Prince Agyei Tuffour and Silas Bempong_
