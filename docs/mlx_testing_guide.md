# MLX memory instrumentation

The initial target is native Apple Silicon macOS with an available MLX Metal
runtime. Linux CPU/CUDA MLX and Intel macOS are outside this qualification.
Stormlog observes the MLX allocator **in the current process**, across its
application threads. It does not attach to another training process or server.

Importing `stormlog.mlx` and retrieving its class definitions does not initialize
MLX, PyTorch, TensorFlow, or JAX. Constructing a runtime-dependent object checks
platform, MLX version (minimum 0.32.3), required core APIs, and Metal availability.
Missing MLX, an unsupported platform, a broken native loader, and unavailable
Metal produce distinct exceptions with chained causes. Successful runtime
discovery is cached; sampled counters and failed discovery are not cached.

## Counter meanings

| Value | Meaning |
| --- | --- |
| `active_bytes` | Native MLX active allocator bytes; telemetry `allocator_allocated_bytes`. |
| `cache_bytes` | Reusable cached buffers; namespaced `metadata.mlx.cache_bytes`. |
| `runtime_peak_bytes` | Native lifetime peak since startup or the most recent external reset. |
| `allocator_held_bytes` | Active + cache from sequential reads; not an atomic snapshot. |
| `memory_limit_bytes` | Read-only runtime limit, when the getter exists. |
| `recommended_working_set_bytes` | Device recommendation; not free capacity or total VRAM. |
| `process_rss_bytes` | Independent host RSS; do not add it to allocator counters. |

Unified memory can share CPU/GPU pages. Generic reserved/active/inactive block
counters and device used/free/total counters remain null. MLX active memory is
not CUDA allocator block-active memory. Cache is not fragmentation. No GPU-only
array attribution, live array census, allocation stack history, or global GPU
memory utilization is implied.

Telemetry uses the existing v4 schema with explicit complete capability
metadata. Generic `supports_bounded_profiling` remains false under its existing
native allocator contract; MLX bounded sampling and host timing are advertised
separately in `metadata.mlx.capabilities`. Optional getter failures preserve
successful fields and record reasons. An active-getter failure is unavailable,
never zero. Sampling is sequential and records collection duration.
