[← Back to main docs](index.md)

# Examples Guide

This guide provides comprehensive examples for using Stormlog with both PyTorch and TensorFlow.

## PyTorch Examples

### Basic Profiling

```python
import torch
import torch.nn as nn
from gpumemprof import GPUMemoryProfiler

# Create a simple model
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100)
)

# Initialize profiler
profiler = GPUMemoryProfiler()

# Profile training step
def train_step(model, data, target):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return loss

# Run training
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(dataloader):
        profile = profiler.profile_function(train_step, model, data, target)

# Get summary
summary = profiler.get_summary()
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
```

### Advanced Monitoring

```python
from gpumemprof import GPUMemoryProfiler, MemoryVisualizer

profiler = GPUMemoryProfiler()

# Start monitoring
profiler.start_monitoring(interval=0.5)

# Your training loop
for epoch in range(10):
    for batch in dataloader:
        profiler.profile_function(train_step, model, batch[0], batch[1])

# Stop monitoring
profiler.stop_monitoring()

# Generate visualizations
visualizer = MemoryVisualizer(profiler)
visualizer.plot_memory_timeline(interactive=False, save_path="timeline.png")
visualizer.export_data(format="json", save_path="training_profile")
```

### Context Profiling

```python
from gpumemprof import profile_context

# Profile different phases
with profile_context("data_loading"):
    train_data = load_dataset()
    val_data = load_validation_data()

with profile_context("model_creation"):
    model = create_model()

with profile_context("training"):
    for epoch in range(10):
        train_epoch(model, train_data)

with profile_context("validation"):
    validate_model(model, val_data)
```

## TensorFlow Examples

### Basic TensorFlow Profiling

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Initialize profiler
profiler = TFMemoryProfiler()

# Profile training
with profiler.profile_context("training"):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, epochs=5, batch_size=32)

# Get results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### Keras Model Profiling

```python
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler()

# Profile model creation
with profiler.profile_context("model_creation"):
    model = create_complex_model()

# Profile data preprocessing
with profiler.profile_context("data_preprocessing"):
    x_train, y_train = preprocess_data()

# Profile training
with profiler.profile_context("training"):
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )

# Profile evaluation
with profiler.profile_context("evaluation"):
    test_loss, test_acc = model.evaluate(x_test, y_test)

# Analyze results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### Custom Training Loop

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler()

@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Profile custom training
with profiler.profile_context("custom_training"):
    for epoch in range(10):
        for batch_x, batch_y in train_dataset:
            loss = train_step(model, optimizer, batch_x, batch_y)
```

## CLI Examples

### Real-time Monitoring

```bash
# Monitor for 5 minutes
gpumemprof monitor --duration 300 --output monitoring.json

# Monitor with alerts
gpumemprof track --warning-threshold 75 --critical-threshold 90 --output tracking.json

# Analyze results
gpumemprof analyze monitoring.json --visualization
```

### TensorFlow CLI

```bash
# Get system info
tfmemprof info

# Monitor TensorFlow training
tfmemprof monitor --duration 600 --output tf_monitoring.json

# Track with alerts
tfmemprof track --threshold 3000 --output tf_tracking.json
```

## Launch QA Scenario Matrix

For v0.2 launch validation, use the scenario matrix orchestrator:

```bash
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

This command runs:

- CPU telemetry export + schema validation
- MPS telemetry export + analyze roundtrip (when MPS is available)
- OOM flight-recorder scenario (safe simulated mode by default)
- TensorFlow monitor/track/analyze/diagnose end-to-end scenario
- `gpumemprof diagnose` artifact check
- benchmark harness budget gate
- optional TUI PTY smoke (skip with `--skip-tui`)

Switch to full mode to include extra demos:

```bash
python -m examples.cli.capability_matrix --mode full --target both --oom-mode simulated
```

### Individual Scenario Modules

```bash
python -m examples.scenarios.cpu_telemetry_scenario
python -m examples.scenarios.mps_telemetry_scenario
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
python -m examples.scenarios.tf_end_to_end_scenario
```

## Complete Working Examples

### PyTorch Training Example

See [examples/basic/pytorch_demo.py](../examples/basic/pytorch_demo.py) for a complete PyTorch training example with profiling.

### Advanced Tracking Example

See [examples/advanced/tracking_demo.py](../examples/advanced/tracking_demo.py) for advanced memory tracking with alerts and visualization.

### CLI Quickstart

See [examples/cli/quickstart.py](../examples/cli/quickstart.py) for runnable samples of the `gpumemprof` and `tfmemprof` command-line interfaces.

### Testing Guides

Legacy testing flows have been migrated to Markdown. See [docs/examples/test_guides/README.md](examples/test_guides/README.md) for CPU-only, PyTorch, TensorFlow, and CLI smoke-test checklists.

### TensorFlow Example

See [examples/basic/tensorflow_demo.py](../examples/basic/tensorflow_demo.py) for a complete TensorFlow example.

## Best Practices

1. **Profile Early**: Start profiling during development, not just in production
2. **Use Contexts**: Organize profiling with meaningful context names
3. **Set Thresholds**: Configure appropriate memory thresholds for your hardware
4. **Monitor Continuously**: Use CLI tools for continuous monitoring
5. **Export Data**: Save results for later analysis and comparison
6. **Visualize**: Use built-in visualization tools to understand patterns

## Troubleshooting Examples

### Memory Leak Detection

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.5)

# Run your code
for i in range(100):
    profiler.profile_function(train_step, model, data)

profiler.stop_monitoring()
summary = profiler.get_summary()
print(f"Peak memory: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
print(f"Memory change: {summary['memory_change_from_baseline'] / (1024**3):.2f} GB")
```

### Performance Optimization

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Profile different batch sizes
for batch_size in [16, 32, 64, 128]:
    with profiler.profile_context(f"batch_size_{batch_size}"):
        train_with_batch_size(model, dataloader, batch_size)

    summary = profiler.get_summary()
    peak_gb = summary['peak_memory_usage'] / (1024**3)
    print(f"Batch size {batch_size}: Peak memory {peak_gb:.2f} GB")
```

---

[← Back to main docs](index.md)
