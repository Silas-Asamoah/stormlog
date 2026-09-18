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
- peak sampled device memory when system telemetry is available

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

## System telemetry

Use `--system-sampler` to choose best-effort telemetry:

```bash
stormlog infer profile ... --system-sampler nvidia-smi
stormlog infer profile ... --system-sampler psutil
stormlog infer profile ... --system-sampler none
```

Remote endpoints may not expose memory. In that case memory fields are omitted,
not filled with synthetic zeroes.

## Future engine adapters

The v1 request path is engine-agnostic. Future adapters can enrich the same run
with engine-native telemetry such as vLLM scheduler metrics, SGLang cache
metrics, TensorRT-LLM inflight batching metrics, or MLX Metal runtime stats
without changing the core `stormlog infer profile` artifact shape.

The versioned request, iteration, stage, membership, and GPU activity contract
is described in [Inference execution correlation](inference_correlation.md).
It preserves v1 client observations while allowing optional server evidence
to be appended to the same JSONL stream.
