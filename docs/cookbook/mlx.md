[← Cookbook](index.md)

# MLX training and local generation

Install `pip install 'stormlog[mlx]'` on native Apple Silicon with usable Metal.
These recipes instrument the allocator in the workload's own process. The
[MLX guide](../mlx_testing_guide.md) explains counter scope, nullable values,
stream completion and the difference between sampled and lifetime peaks.

## A small training loop

Save this as `train_mlx.py`. It uses generated inputs and needs no model download.
The profiler wraps the training step outside the differentiation transform;
explicit roots include the loss, model state and optimizer state.

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from stormlog.mlx import MLXMemoryProfiler, MemoryTracker

model = nn.Linear(8, 8)
optimizer = optim.Adam(learning_rate=0.001)
x = mx.ones((16, 8))
mx.eval(x, model.state)


def loss_fn(model, x):
    return mx.mean(model(x) ** 2)


loss_and_grad = nn.value_and_grad(model, loss_fn)
profiler = MLXMemoryProfiler()
tracker = MemoryTracker(sampling_interval=0.01, max_history=100)
with tracker.tracking():
    with tracker.phase("train"):
        for _ in range(10):
            with profiler.profile_context("train_step") as region:
                loss, grads = loss_and_grad(model, x)
                optimizer.update(model, grads)
                region.evaluate(loss, model.state, optimizer.state)
profiler.export("train-profiles.json")
tracker.export("train-tracking.json")
print(float(loss.item()))
```

```bash
mlxmemprof run --interval 0.01 --output artifacts/mlx-training -- train_mlx.py
mlxmemprof analyze artifacts/mlx-training --output artifacts/mlx-training-analysis --json
```

The CLI adds a passive tracker in the same interpreter; the explicit Python
profiler controls the bounded training regions. Instrumentation does not
change global cache or memory limits. A separate default stream is synchronized
by the profiler; declare `streams=(...)` if the application uses other streams.

## A completed local generation loop

This small toy autoregressive loop demonstrates evaluation boundaries without
model downloads. Replace the toy logits calculation with your model while
keeping token/cache state as explicit roots. It is not a language-model quality
benchmark or an MLX-LM adapter.

```python
import mlx.core as mx
from stormlog.mlx import MLXMemoryProfiler, MemoryTracker

embedding = mx.ones((64, 16))
projection = mx.ones((16, 64))
current = mx.array(1)
mx.eval(embedding, projection, current)
profiler = MLXMemoryProfiler()
tracker = MemoryTracker(sampling_interval=0.01, max_history=100)
tokens = []
with tracker.tracking():
    with tracker.phase("decode"):
        for _ in range(20):
            with profiler.profile_context("token_step") as region:
                logits = embedding[current] @ projection
                current = mx.argmax(logits)
                region.evaluate(current)
            tokens.append(int(current.item()))
profiler.export("generation-profiles.json")
tracker.export("generation-tracking.json")
print(tokens)
```

For a real model, evaluate returned token roots and the model's evolving KV
cache/state. Avoid serializing arrays into profile artifacts. Profiling an HTTP
client cannot observe an oMLX server's allocator: instrument inside each server
worker or export its telemetry with PID/session identity. Serving integration for oMLX, MLX-LM and MLX-Audio is a separate planned
PR4 milestone after this core memory rollout.

## Read growth and artifacts correctly

Compare active memory after warmup at matched workload boundaries. Compilation,
optimizer initialization, retained outputs and KV cache can explain active
growth. The analyzer reports candidate growth for investigation, not proof of
a leak. Reusable allocator cache can remain high after arrays are released;
cache growth does not establish fragmentation. Host RSS can overlap unified
allocator pages and must not be added to active/cache bytes.

`mlxmemprof monitor`, `track` and `diagnose` observe their own command process.
Use `run` for a workload, and preserve its bundle's environment, telemetry,
manifest and `report.json`. Scripts that print can mix output with CLI JSON;
read the durable report for automation. `stormlog tui --framework mlx` can load
those bundles or the explicit profile exports without any installed framework.

## Qualification from a source checkout

```bash
python3 -m examples.scenarios.mlx_memory_scenario --output artifacts/mlx-contracts
python3 -m examples.scenarios.mlx_memory_scenario --hardware --repetitions 200 --output artifacts/mlx-metal
python3 -m examples.cli.capability_matrix --target mlx --mode smoke
```

Fake mode checks contracts without claiming hardware timings. The required
`--hardware` scenario fails if native Metal is unavailable. The optional
capability matrix reports SKIP with a reason for an unavailable MLX runtime;
unexpected scenario errors report FAIL. Existing MPS checks remain separate.
See [published qualification](../benchmarks/mlx-qualification.md) for measured
mean/p90 timing, peak-memory scopes and evidence limits.
