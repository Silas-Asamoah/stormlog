[← Back to main docs](index.md)

# CPU Compatibility Guide

This guide explains how to use Stormlog on systems without CUDA/GPU support.

## Built-in CPU Mode

As of the latest release, the CLI, Python API, and Textual TUI automatically
fallback to a psutil-powered `CPUMemoryProfiler`/`CPUMemoryTracker` whenever
`torch.cuda.is_available()` returns `False`. No extra flags are required—the same
commands just switch to RSS-based metrics:

- `gpumemprof monitor` and `gpumemprof track` display CPU RSS data (MB) and can
  still export CSV/JSON logs.
- The TUI monitoring tab starts a CPU tracker so you can stream events/exports
  without a GPU.
- PyTorch sample workloads automatically run on CPU tensors and display RSS
  deltas in the log.

If you only care about CPU memory, simply run the normal CLI/TUI commands and
they will do the right thing.

## 🚫 No CUDA? Still Want Custom Tweaks?

If you need lower-level control (or are using an older version), the sections
below show how to roll your own CPU profilers or integrate other tooling.

## What Works Without CUDA

✅ **Available Features:**

- CPU memory profiling
- Function execution timing
- Context-based profiling
- Real-time memory tracking
- Memory leak detection (for CPU memory)
- Basic visualization
- Data export functionality
- Command-line interface (modified)

❌ **Limited Features:**

- GPU memory tracking (obviously)
- CUDA-specific optimizations
- GPU tensor lifecycle tracking
- GPU memory fragmentation analysis

## Quick Solutions

### Option 1: Run the Markdown-Based Checklists

Follow the steps in `docs/examples/test_guides/README.md`:

```bash
CUDA_VISIBLE_DEVICES="" gpumemprof info
python -m examples.cli.quickstart
```

### Option 2: Modify Existing Code for CPU (Legacy Approach)

If you want to customize the CPU profiler beyond the built-in behavior, here are
some patterns you can borrow:

The built-in `gpumemprof.CPUMemoryProfiler` now exposes `get_summary()` for
aggregate CPU-memory metrics and is recommended for most workflows.

#### Modify the PyTorch Profiler for CPU

```python
# Replace CUDA device setup with CPU
import torch
from gpumemprof import GPUMemoryProfiler

# Option 1: Mock GPU profiler for CPU
class CPUMemoryProfiler:
    def __init__(self):
        self.results = []
        self.function_profiles = {}
        self.snapshots = []
        import psutil
        self.process = psutil.Process()

    def _get_memory_usage(self):
        """Get CPU memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def profile_function(self, func):
        """Profile function CPU memory usage."""
        def wrapper(*args, **kwargs):
            before_mem = self._get_memory_usage()
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()
            after_mem = self._get_memory_usage()

            # Store results
            func_name = func.__name__
            if func_name not in self.function_profiles:
                self.function_profiles[func_name] = {
                    'calls': 0,
                    'total_duration': 0.0,
                    'total_memory_used': 0.0
                }

            profile = self.function_profiles[func_name]
            profile['calls'] += 1
            profile['total_duration'] += (end_time - start_time)
            profile['total_memory_used'] += (after_mem - before_mem)

            return result
        return wrapper

    def get_results(self):
        """Get profiling results."""
        return {
            'function_profiles': self.function_profiles,
            'peak_memory_mb': max([self._get_memory_usage()]),
            'snapshots': self.snapshots
        }

# Usage
profiler = CPUMemoryProfiler()

@profiler.profile_function
def cpu_test():
    x = torch.randn(1000, 1000)  # CPU tensor
    y = torch.matmul(x, x.T)
    return y.sum()

result = cpu_test()
print(f"CPU Memory Usage: {profiler.get_results()}")
```

#### Modify PyTorch Operations for CPU

```python
# Instead of .cuda(), use CPU tensors
x = torch.randn(1000, 1000)  # CPU tensor
y = torch.randn(1000, 1000)  # CPU tensor

# Model on CPU
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100)
)  # No .cuda() call

# Training on CPU
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    inputs = torch.randn(32, 1000)  # CPU
    targets = torch.randint(0, 100, (32,))  # CPU

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Option 3: Use Alternative Memory Profiling

```python
import psutil
import time
import matplotlib.pyplot as plt
from memory_profiler import profile

# Simple memory tracking
def track_memory(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()

        before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        result = func(*args, **kwargs)

        after = process.memory_info().rss / 1024 / 1024  # MB
        duration = time.time() - start_time

        print(f"Function: {func.__name__}")
        print(f"Memory before: {before:.2f} MB")
        print(f"Memory after: {after:.2f} MB")
        print(f"Memory diff: {after - before:.2f} MB")
        print(f"Duration: {duration:.4f} seconds")

        return result
    return wrapper

# Usage
@track_memory
def memory_intensive_function():
    # Large tensor operations
    data = torch.randn(2000, 2000)
    result = torch.matmul(data, data.T)
    return torch.mean(result)

memory_intensive_function()
```

## Running Tests Without CUDA

### Method 1: Skip CUDA Tests

```python
import torch

def run_tests():
    if torch.cuda.is_available():
        print("Running GPU tests...")
        # GPU test code
    else:
        print("CUDA not available, running CPU tests...")
        # CPU test code

run_tests()
```

### Method 2: Mock CUDA Functions

```python
# Create mock CUDA functions
class MockCUDA:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def memory_allocated(device=None):
        return 0

    @staticmethod
    def memory_reserved(device=None):
        return 0

    @staticmethod
    def empty_cache():
        pass

# Replace torch.cuda with mock if needed
if not torch.cuda.is_available():
    torch.cuda = MockCUDA()
```

## Working Examples

### Example 1: Basic CPU Profiling

```python
import torch
import time
import psutil

class SimpleCPUProfiler:
    def __init__(self):
        self.results = []

    def profile(self, name, func, *args, **kwargs):
        process = psutil.Process()

        before_mem = process.memory_info().rss
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        after_mem = process.memory_info().rss

        self.results.append({
            'name': name,
            'duration': end_time - start_time,
            'memory_diff_mb': (after_mem - before_mem) / 1024 / 1024,
            'memory_before_mb': before_mem / 1024 / 1024,
            'memory_after_mb': after_mem / 1024 / 1024
        })

        return result

    def summary(self):
        for result in self.results:
            print(f"{result['name']}: {result['duration']:.4f}s, {result['memory_diff_mb']:.2f}MB")

# Usage
profiler = SimpleCPUProfiler()

# Profile tensor operations
result1 = profiler.profile("create_tensor", torch.randn, 1000, 1000)
result2 = profiler.profile("matrix_multiply", torch.matmul, result1, result1)

profiler.summary()
```

### Example 2: PyTorch Model Training on CPU

```python
import torch
import torch.nn as nn
import torch.optim as optim

# CPU-only model training with profiling
def train_cpu_model():
    # Create model (CPU)
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training data (CPU)
    batch_size = 64

    for epoch in range(5):
        # Generate random batch
        inputs = torch.randn(batch_size, 784)
        targets = torch.randint(0, 10, (batch_size,))

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

## Installation for CPU-Only

```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install profiler dependencies
pip install psutil matplotlib numpy

# Install the profiler
pip install -e .
```

## Command Line Usage (CPU Mode)

```bash
# Run CPU-compatible tests
python -m examples.cli.quickstart

# Profile existing Python script
python -m memory_profiler your_script.py

# Monitor system memory
watch -n 1 'free -h'
```

## Benefits of CPU Profiling

Even without GPU profiling, you still get:

1. **Memory Usage Tracking**: Monitor how much system RAM your models use
2. **Performance Profiling**: Measure execution time of different operations
3. **Memory Leak Detection**: Find CPU memory leaks in your code
4. **Resource Optimization**: Optimize CPU usage and memory allocation
5. **Educational Value**: Learn memory management concepts

## Performance Expectations

| Operation                   | CPU (16 cores) | Notes                |
| --------------------------- | -------------- | -------------------- |
| Matrix Multiply (1000x1000) | ~0.1s          | Using optimized BLAS |
| CNN Forward Pass            | ~0.5s          | Small models only    |
| Memory Tracking             | ~0.01s         | Very fast            |
| Data Export                 | ~0.1s          | JSON/CSV export      |

## Limitations and Alternatives

### Limitations:

- No GPU memory insights
- Slower training/inference
- Limited model sizes
- No CUDA-specific optimizations

### Alternatives:

- Use Google Colab (free GPU)
- AWS EC2 with GPU instances
- Local GPU rental services
- CPU-optimized model architectures

## Getting Started (CPU Only)

1. **Quick Start:**

```bash
python -m examples.cli.quickstart
gpumemprof info
```

2. **Full Test Suite:**

Follow the CPU checklist in `docs/examples/test_guides/README.md`.

3. **Specific Tests:**

Use the snippets in this guide (or the Markdown checklists) to build targeted
CPU profiling scenarios.

## Conclusion

While GPU profiling provides the most insights for deep learning, CPU profiling is still valuable for:

- Development and testing
- Understanding memory patterns
- Learning profiling concepts
- Debugging memory issues
- CPU-based inference optimization

The CPU-compatible version gives you ~70% of the profiler's functionality and is perfect for learning and development!
