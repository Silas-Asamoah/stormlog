[← Back to main docs](index.md)

# Inference Profiling

Stormlog can actively profile OpenAI-compatible Chat Completions endpoints with
the top-level `stormlog infer` command group. This surface is intentionally
separate from `gpumemprof` and `tfmemprof`: the endpoint may be backed by
PyTorch, vLLM, SGLang, TensorRT-LLM, MLX-LM, a hosted gateway, or another
server that accepts the Chat Completions request shape.

## Profile an endpoint

```bash
stormlog infer profile \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,4,8 \
  --input-tokens 512,2048 \
  --output-tokens 128,512 \
  --requests 20 \
  --output artifacts/infer_qwen.jsonl
```

You can pass a `/v1` base URL instead of the full endpoint:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 8 \
  --input-tokens 2048 \
  --output-tokens 512 \
  --duration 120 \
  --output artifacts/infer_steady_state.jsonl
```

The profiler sends controlled traffic for each workload case in the matrix:

- `concurrency`
- prompt token target
- output token cap
- streaming or non-streaming mode

`--requests` is the total measured request count per workload case, shared
across the configured workers. `--duration` instead runs each workload case for
the requested wall-clock window.

Warmup requests are recorded but excluded from analysis:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --warmup-requests 8 \
  --requests 50 \
  --output artifacts/infer_with_warmup.jsonl
```

## Analyze an artifact

```bash
stormlog infer analyze artifacts/infer_qwen.jsonl
stormlog infer analyze artifacts/infer_qwen.jsonl --format json --output report.json
```

The report includes:

- end-to-end latency percentiles
- TTFT percentiles for streaming responses
- first streamed chunk latency
- requests/sec
- output tokens/sec and total tokens/sec
- failure rate
- highest recorded client-local device memory when system telemetry is available
- scoped server memory observations when a matching on-host collector artifact is supplied

## Token accounting

Server usage metadata is preferred whenever the endpoint returns it. If usage is
missing, Stormlog falls back to the configured tokenizer and records the source
on every request event:

- `server_usage`
- `tiktoken`
- `transformers`
- `estimated`
- `unknown`

When streaming is enabled, Stormlog requests OpenAI-style streaming usage
metadata with `stream_options.include_usage` by default. Use
`--no-stream-usage` for endpoints that reject that request field. If streaming
usage is unavailable, output token counts fall back to the configured tokenizer
or estimate and the request event records that provenance.

Core endpoint profiling does not require tokenizer packages. Install tokenizer
extras when you want better prompt sizing and fallback counts:

```bash
pip install "stormlog[infer-tokenizers]"
```

Useful tokenizer options:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tokenizer transformers \
  --tokenizer-model Qwen/Qwen2.5-7B-Instruct \
  --output artifacts/infer_qwen.jsonl
```

For OpenAI-model tokenizers:

```bash
stormlog infer profile \
  --endpoint https://api.openai.com/v1/chat/completions \
  --model gpt-4o-mini \
  --tokenizer tiktoken \
  --tiktoken-encoding o200k_base \
  --output artifacts/infer_openai.jsonl
```

## Metric boundaries

Stormlog reports client-observed metrics in v1:

- TTFT is measured from request start to the first non-empty streamed content
  delta.
- Non-streaming responses do not have TTFT or chunk timing.
- Chunk inter-arrival timing is chunk-level timing. It is not treated as
  token-level ITL unless a future engine adapter can prove token-level events.
- Token throughput uses server usage when available; otherwise the configured
  tokenizer or estimate is clearly recorded.

## Client-local telemetry

Use `--system-sampler` to choose best-effort telemetry:

```bash
stormlog infer profile ... --system-sampler nvidia-smi
stormlog infer profile ... --system-sampler psutil
stormlog infer profile ... --system-sampler none
```

These samplers run where `stormlog infer profile` runs. Even if the endpoint is
remote, `nvidia-smi` reads the **client's** GPU and `psutil` reads the
**Stormlog client process**. Each new `infer.system_sample` has
`observation_scope: client_local`; older v1 samples are interpreted the same
way. The JSON report puts these values under `memory.observation_scope:
client_local` and retains the existing `peak_device_used_bytes` and
`peak_process_rss_bytes` keys for compatibility. Neither key is a server
memory claim. An unavailable reading stays `null`, never synthetic zero.

## Optional server telemetry

Run the collector **on the inference host** while profiling. Give both commands
the same run ID. The server PID must be the process serving the directly
addressed endpoint. This example assumes the profiler and server share a host;
for separate hosts, use the clock alignment options described below.

```bash
RUN_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')
stormlog infer collect-server \
  --run-id "$RUN_ID" --pid 12345 --device-index 0 \
  --interval 0.1 --duration 60 --output artifacts/server.jsonl &
stormlog infer profile \
  --run-id "$RUN_ID" --base-url http://127.0.0.1:8000/v1 \
  --model my-model --requests 20 --output artifacts/client.jsonl
wait
stormlog infer analyze artifacts/client.jsonl \
  --server-telemetry artifacts/server.jsonl --direct-server --format json
```

The collector requires `psutil` (a core dependency) and, for GPU counters, an
NVIDIA driver exposing NVML v2. Use `--no-gpu` for process RSS only. Pass
`--device-uuid` when device index alone is ambiguous; a MIG UUID selects an
instance. `--replica-id` and `--rank` attach additional identity. The collector
records the actual host, boot ID, PID, process start, GPU UUID, and MIG identity on
each counter. It stops if the process restarts or the GPU identity changes.
An NVML read failure produces a `missing` sample with a null value.
For servers with separate HTTP and GPU worker processes, target the worker PID
that owns the GPU work. The direct-route assertion then includes the operator's
knowledge that the addressed HTTP server uses that worker. Stormlog does not
infer this relationship from a matching GPU index.

For a profiler on another host, copy the server JSONL to the analysis host and
provide a measured server-to-client clock offset and an uncertainty bound:

```bash
stormlog infer analyze artifacts/client.jsonl \
  --server-telemetry artifacts/server.jsonl --direct-server \
  --clock-offset-ns 1200000 --clock-uncertainty-ns 300000 \
  --format json
```

`server timestamp + offset = client timestamp`. Derive the bound from a clock
synchronization service or a two-way timestamp probe near the run. If the
hosts or boot IDs differ, or either boot ID is unavailable, and no alignment is
supplied, the report lists server targets
but does not join their samples to client request windows. The uncertainty
must fit inside the request window for a sample to count. `--direct-server`
is an explicit assertion that every request in the profile reached this one
serving process. Do not use it for a load balancer that can route to multiple
replicas. Multiple server identities, a different run ID, or a restart prevent
the case-window join. These checks do not prove per-request memory ownership:
other requests and processes can use the GPU during the same window.

The server artifact uses [versioned `infer.telemetry_sample` records](schemas/inference_telemetry_v1.schema.json),
one counter per record. `scope` identifies `server_process`, `gpu_device`, or `gpu_instance`;
`counter_owner`, `source`, `provenance`, `interval_ms`, `state`, and the
identity explain what the number means. The report's
`memory.server_observations` gives `maximum_recorded_bytes` and counts of
valid, missing, stale, and invalid samples for each metric. A maximum is the
largest recorded value at the chosen cadence, not the true peak. The 100 ms
default is a starting point for short requests with direct NVML reads; it is
not a universal sampling rule. Slower collectors and exporters may miss short
peaks or report cached/averaged values.

| Counter | Scope and owner | Collector support |
| --- | --- | --- |
| `process_rss_bytes` | Server process; operating system | On-host `psutil` |
| `device_memory_used_bytes`, `device_memory_reserved_bytes` | Whole GPU device; NVML | On-host NVML v2 |
| `instance_memory_used_bytes`, `instance_memory_reserved_bytes` | MIG instance; NVML | On-host NVML v2 when supported |
| `process_gpu_used_bytes` | Server process; GPU process accounting | Contract for optional sources; not collected by this command |
| `allocator_allocated_bytes`, `allocator_reserved_bytes` | Server process; allocator | Contract for optional sources; not collected by this command |
| `engine_cache_occupied_bytes` | Server process; engine cache | Contract for optional sources; not collected by this command |

Allocator, device, process RSS, and engine cache numbers must not be added or
substituted for each other. NVIDIA documents the NVML v2 `used` and `reserved`
fields separately in its [NVML memory structure](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlMemory__v2__t.html).
Exporters such as DCGM may report interval averages or cached values; this
collector currently reads NVML directly and does not ingest DCGM metrics.

## Execution correlation and future adapters

The v1 request path is engine-agnostic. Future adapters can enrich the same run
with engine-native telemetry such as vLLM scheduler metrics, SGLang cache
metrics, TensorRT-LLM inflight batching metrics, or MLX Metal runtime stats
without changing the core `stormlog infer profile` artifact shape.

The versioned request, iteration, stage, membership, and GPU activity contract
is described in [Inference execution correlation](inference_correlation.md).
It preserves v1 client observations while allowing optional server evidence
to be appended to the same JSONL stream. New profiles also include a v2
artifact identity record that names their run and capture session.
