[← Back to main docs](index.md)

# Troubleshooting Guide

This guide helps you resolve common issues with GPU Memory Profiler.

## Common Issues

### Import Errors

#### Problem: `ModuleNotFoundError: No module named 'gpumemprof'`

**Solution:**

```bash
# Install the package
pip install -e .

# Or install from PyPI
pip install stormlog
```

#### Problem: `ModuleNotFoundError: No module named 'torch'`

**Solution:**

```bash
# Install PyTorch
pip install torch

# Or install with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Problem: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**

```bash
# Install TensorFlow (GPU support is included automatically)
pip install tensorflow
```

### CUDA Issues

#### Problem: `CUDA not available`

**Symptoms:**

- Error: `CUDA not available`
- Profiler falls back to CPU mode

**Solutions:**

1. **Check CUDA installation:**

```bash
nvidia-smi
nvcc --version
```

2. **Verify PyTorch CUDA:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

3. **Verify TensorFlow CUDA:**

```python
import tensorflow as tf
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
```

4. **Install CUDA-compatible versions:**

```bash
# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# TensorFlow (GPU support is included automatically)
pip install tensorflow
```

#### Problem: `CUDA out of memory`

**Symptoms:**

- Error: `CUDA out of memory`
- Training crashes

**Solutions:**

1. **Reduce batch size:**

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=16)  # Instead of 64
```

2. **Clear cache:**

```python
import torch
torch.cuda.empty_cache()
```

3. **Use gradient checkpointing:**

```python
from torch.utils.checkpoint import checkpoint
# Wrap memory-heavy layers with checkpoint()
```

4. **Monitor memory usage:**

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.5)

# Your training code here
profiler.stop_monitoring()
```

### Memory Leak Issues

#### Problem: Memory usage keeps increasing

**Symptoms:**

- Memory usage grows over time
- Profiler detects memory leaks

**Solutions:**

1. **Check for unreleased tensors:**

```python
# Ensure tensors are properly deleted
del tensor
torch.cuda.empty_cache()
```

2. **Use context managers:**

```python
with torch.no_grad():
    # Inference code here
    pass
```

3. **Monitor with profiler:**

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.5)

# Your code here
with profiler.profile_context("training_step"):
    train_step()

profiler.stop_monitoring()
summary = profiler.get_summary()
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
print(f"Memory change: {summary['memory_change_from_baseline'] / (1024**3):.2f} GB")
```

### CLI Issues

#### Problem: `gpumemprof: command not found`

**Solution:**

```bash
# Reinstall the package
pip install -e .

# Check if entry points are installed
pip show gpu-memory-profiler
```

#### Problem: CLI commands fail

**Solutions:**

1. **Check Python path:**

```bash
which python
which gpumemprof
```

2. **Reinstall with entry points:**

```bash
pip uninstall gpu-memory-profiler
pip install -e .
```

3. **Use Python module directly:**

```bash
python -m gpumemprof.cli info
python -m tfmemprof.cli info
```

### Visualization Issues

#### Problem: Plots don't display

**Symptoms:**

- No plots appear
- Error: `No display name and no $DISPLAY environment variable`

**Solutions:**

1. **Use non-interactive backend:**

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

2. **Save plots to files:**

```python
from gpumemprof import MemoryVisualizer

visualizer = MemoryVisualizer(profiler)
visualizer.plot_memory_timeline(interactive=False, save_path='timeline.png')
```

3. **Use Plotly for web-based plots:**

```python
from gpumemprof import MemoryVisualizer

visualizer = MemoryVisualizer(profiler)
visualizer.export_data(format='json', save_path='dashboard_data')
```

#### Problem: Dash visualization fails

**Symptoms:**

- Error: `ImportError: No module named 'dash'`

**Solution:**

```bash
pip install dash
```

### Performance Issues

#### Problem: Profiler adds too much overhead

**Symptoms:**

- Training is significantly slower
- High CPU usage

**Solutions:**

1. **Increase sampling interval:**

```python
profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=2.0)  # Sample every 2 seconds
```

2. **Disable visualization during training:**

```python
profiler = GPUMemoryProfiler(track_tensors=False)
```

3. **Use context profiling selectively:**

```python
# Only profile specific functions
from gpumemprof import profile_function

@profile_function
def critical_function():
    pass
```

### Dependency Conflicts

#### Problem: `typing_extensions` version conflict

**Symptoms:**

- Error with TensorFlow CLI
- Version conflicts between packages

**Solutions:**

1. **Check versions:**

```bash
pip list | grep typing
```

2. **Install compatible version:**

```bash
pip install typing-extensions==4.5.0
```

3. **Use virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Platform-Specific Issues

#### macOS Issues

**Problem: CUDA not available on macOS**

**Solution:**

- Use CPU mode or MPS (Metal Performance Shaders)
- Install PyTorch with MPS support

**Problem: TensorFlow issues on Apple Silicon**

**Solution:**

```bash
# Install TensorFlow (Apple Silicon is supported natively since TF 2.13)
pip install tensorflow

# For Metal GPU acceleration, also install:
pip install tensorflow-metal
```

#### Windows Issues

**Problem: Path issues**

**Solution:**

```bash
# Use forward slashes or raw strings
python -m gpumemprof.cli info
```

**Problem: Permission issues**

**Solution:**

```bash
# Run as administrator or use --user flag
pip install --user -e .
```

## Debug Mode

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from gpumemprof import GPUMemoryProfiler
profiler = GPUMemoryProfiler()
```

### Verbose CLI Output

```bash
# Use detailed/system output commands
gpumemprof info --detailed
gpumemprof monitor --duration 10
```

### Check System Information

```bash
# Quickest way to check environment health
gpumemprof info --detailed
tfmemprof info
```

Or from Python:

```python
from gpumemprof import get_gpu_info
info = get_gpu_info()  # Returns GPU details, or {"error": ...} on non-CUDA hosts
print(info)
```

## Getting Help

### Before Asking for Help

1. **Check the documentation:**

   - [Installation guide](installation.md)
   - [Usage guide](usage.md)
   - [API reference](api.md)

2. **Run diagnostics:**

```bash
gpumemprof info --detailed
gpumemprof diagnose --duration 0 --output ./diag_bundle
tfmemprof info
tfmemprof diagnose --duration 0 --output ./tf_diag_bundle
```

3. **Test with minimal example:**

```python
from gpumemprof import GPUMemoryProfiler
import torch

profiler = GPUMemoryProfiler()

def test():
    return torch.randn(100, 100).cuda()

profile = profiler.profile_function(test)
summary = profiler.get_summary()
print(profile.to_dict())
print(summary)
```

### Reporting Issues

When reporting issues, include:

1. **System information:**

   - OS and version
   - Python version
   - PyTorch/TensorFlow versions
   - CUDA version (if applicable)

2. **Error messages:**

   - Full error traceback
   - Any warning messages

3. **Reproduction steps:**

   - Minimal code example
   - Expected vs actual behavior

4. **Environment:**
   - Virtual environment details
   - Package versions (`pip freeze`)

### Community Support

- **GitHub Issues**: [Create an issue](https://github.com/Silas-Asamoah/gpu-memory-profiler/issues)
- **Documentation**: Check the [docs](index.md)
- **Examples**: See the [examples directory](../examples/)

---

[← Back to main docs](index.md)
