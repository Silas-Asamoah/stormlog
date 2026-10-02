[← Back to main docs](index.md)

# vLLM native telemetry

`stormlog infer profile` can enrich a run with what vLLM already knows about
itself: the scheduler and cache metrics it exposes on `/metrics`, and the
OpenTelemetry span it emits for each request. The run then shows how much of
a latency change is queueing, how busy the KV cache was, how many prompt
tokens the prefix cache served, and what the engine's own stopwatches say
about each phase, without installing vLLM on the client.

Everything on this page is aggregate or engine-side evidence. A scrape
describes every request the engine served, from every client. A span
describes one request's residency in the scheduler's phases. Neither says
which request used which GPU time; that join belongs to the worker
instrumentation issue and the GPU trace work, not to this page.

Verified against vLLM 0.30.0 (CUDA, V1 engine). The names and meanings below
come from its source and from live scrapes; a later vLLM may add, rename or
retire series, and the capability record of each run says what it found.

## Collect

Scraping metrics needs nothing beyond Stormlog. Receiving vLLM's spans needs
the `infer-otlp` extra, which brings the generated OpenTelemetry protobuf
classes the receiver decodes with:

```bash
pip install "stormlog[infer-otlp]"
```

Start vLLM with its metrics on (the default) and, for spans, with
`--otlp-traces-endpoint` pointing at the receiver that `infer profile` runs:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
  --otlp-traces-endpoint http://127.0.0.1:4318/v1/traces

stormlog infer profile \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,8,32 --input-tokens 512 --output-tokens 128 --requests 64 \
  --prompt-mode unique \
  --vllm-metrics --vllm-metrics-interval 1 \
  --vllm-spans-listen 127.0.0.1:4318 \
  --output artifacts/infer_vllm.jsonl
```

| Flag | What it does |
| --- | --- |
| `--vllm-metrics [URL]` | Scrape `/metrics` just before each phase's first send, after its drain, and every `--vllm-metrics-interval` seconds (default 1) in between. Without a URL, the endpoint's origin plus `/metrics`. |
| `--vllm-spans-listen [HOST:PORT]` | Run an OTLP/HTTP receiver (default `127.0.0.1:4318`) for the length of the run and keep every span vLLM exports to it. |
| `--vllm-spans-drain SECONDS` | Keep the receiver listening this long after the last phase (default 6), because vLLM's exporter flushes spans in batches every 5 s (`OTEL_BSP_SCHEDULE_DELAY`), so the last requests' spans leave after the last answer. |
| `infer analyze --vllm-spans FILE` | Load spans someone else collected: an OTLP JSON export, or one span per line. May be given more than once. |

Scrapes are stamped on the client's clock, like the phase windows, so no
clock alignment is involved. A scrape that fails or returns something that
is not Prometheus text is recorded with its reason and the run goes on; the
first failure is printed once. The receiver binds before any request is
sent; a port it cannot bind is recorded as unavailable, with the error, and
the run goes on without spans. Without the `infer-otlp` extra the receiver
still listens and accepts OTLP JSON, but vLLM's exporter sends protobuf, so
those exports are refused with a 415, counted, and the capability record
lists `otlp_http_protobuf` as supported but not enabled; the run warns once.

The receiver exists only while the profile runs. Spans vLLM exports when no
receiver is listening, such as its startup spans or the spans of traffic
between runs, fail on the server side and are dropped there; the exporter
retries a failed batch with backoff for up to about a minute, which can hold
back the next batch. For short runs, start vLLM with
`OTEL_BSP_SCHEDULE_DELAY=1000` so batches leave every second, and keep
`--vllm-spans-drain` at or above that delay. The report's `spans` block
counts requests without a span, so a late batch is visible, never silent.

Every request is sent with `X-Request-Id: stormlog-<run_id>-<request_id>`,
recorded on its `infer.request` event as `x_request_id`. vLLM embeds that
header in its own request id and in the span's `gen_ai.request.id`, so a span
joins to a request by the recorded value and never by a rebuilt string.

## What the artifact holds

- `infer.vllm_scrape`: one record per scrape, holding every series in a
  compact form: label sets shared across series, native names, native
  histogram bucket boundaries, the `# TYPE` of each family, and a discovery
  block that says what the catalog recognised. A failed scrape records the
  error instead. About 10 KB per scrape for 0.30.0.
- `infer.vllm_span`: one record per span, attributes under their native
  names, timestamps on the exporter's wall clock (named by its host, never
  taken as the client's clock), and the recovered `request_id`.
- `infer.capabilities` for `vllm.metrics` and `vllm.spans`: what was
  supported, enabled and collected, with the engine's unknown, retired and
  removed series, scrape and span counts, and the receiver's decode failures.

Both records are described by
[`inference_vllm_v1.schema.json`](schemas/inference_vllm_v1.schema.json).

## What the report says

`infer analyze` adds `telemetry.vllm` to the JSON report and a few lines to
the text report. Per measured case and per `engine` label:

| Block | Content | Source |
| --- | --- | --- |
| `counters` | delta over the case window, per extra label (`finished_reason`, `source`, ...) | phase-start and phase-end scrapes |
| `histograms` | count and sum deltas, the mean, and per-bucket deltas under the native `le` boundaries | the same two scrapes |
| `gauges` | min, mean, max, last and sample count over every scrape in the window | all scrapes of the case |
| `derived.rates` | prompt and generated tokens per second, finished requests per second | counters over the window |
| `derived.prefix_cache` | queries, hits and the hit ratio, in tokens | counters |
| `derived.kv_cache` | peak and mean usage as a fraction, and in blocks when `cache_config_info` names `num_gpu_blocks` | gauge plus info labels |
| `derived.mfu` | the estimated FLOPs and bytes per GPU, or `not_enabled` | counters |
| `spans` | requests with a span, and p50/p95/mean of each native latency attribute | joined spans |

The window runs from the scrape before the first send to the scrape after
the drain, and the report says `includes_drain: true`, so rates are over the
whole phase and not only the arrival window. Engines (the `engine` label,
one per data-parallel engine core) are reported one by one and never added
together.

Nothing is defaulted to zero. A field is `unresolved` with one of these
reasons when the two scrapes cannot be compared:

| Reason | Meaning |
| --- | --- |
| `scrape_missing:<marker>`, `scrape_failed:<marker>` | The phase has no successful start or end scrape |
| `engine_restart` | The exporter's `process_start_time_seconds` changed between the two scrapes |
| `engine_set_changed` | The set of `engine` labels changed |
| `counter_reset` | The counter went backwards |
| `counter_recreated` | The counter's `*_created` timestamp changed |
| `series_missing` | One scrape lacks the series |
| `bucket_boundaries_changed` | The histogram's `le` set differs between the scrapes |
| `not_enabled` | The MFU counters stayed at zero while tokens were generated |

A series the catalog does not name is kept under its native name and listed
as unknown. A retired name is normalised under its successor, kept as
`deprecated_alias_of`, and listed as retired.

## Metric map

Residency means wall-clock time in a scheduler phase, measured on the
engine's clock. It is not execution time and not GPU time: under continuous
batching every running request's residency covers the same forward passes.

| Field | Native series (0.30.0) | Kind, unit | Meaning |
| --- | --- | --- | --- |
| `queue_depth` | `vllm:num_requests_waiting` | gauge, requests | WAITING at scrape time |
| `queue_depth_by_reason` | `vllm:num_requests_waiting_by_reason{reason}` | gauge, requests | why they wait (`capacity`, `deferred`) |
| `running_requests` | `vllm:num_requests_running` | gauge, requests | RUNNING at scrape time |
| `queue_time` | `vllm:request_queue_time_seconds` | histogram, s | WAITING residency per finished request |
| `preemptions` | `vllm:num_preemptions_total` | counter | requests preempted |
| `preemptions_per_request` | `vllm:request_num_preemptions` | histogram | preemptions a finished request went through |
| `kv_cache_usage` | `vllm:kv_cache_usage_perc` | gauge, fraction 0–1 | KV blocks in use; logical occupancy, not device memory |
| `cache_config` | `vllm:cache_config_info` | info labels | `num_gpu_blocks`, `block_size`, ... |
| `prefix_cache_queries`, `prefix_cache_hits` | `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` | counter, tokens | local prefix cache lookups and hits |
| `external_prefix_cache_*` | `vllm:external_prefix_cache_*_total` | counter, tokens | an external KV cache, when a connector is configured |
| `prompt_tokens`, `prompt_tokens_cached`, `prompt_tokens_by_source` | `vllm:prompt_tokens_total`, `vllm:prompt_tokens_cached_total`, `vllm:prompt_tokens_by_source_total{source}` | counter, tokens | prefill tokens, and where they came from |
| `generation_tokens` | `vllm:generation_tokens_total` | counter, tokens | tokens generated |
| `iteration_tokens` | `vllm:iteration_tokens_total` | histogram, tokens | tokens scheduled per engine step (a histogram despite the suffix) |
| `request_success` | `vllm:request_success_total{finished_reason}` | counter, requests | finished requests |
| `time_to_first_token`, `inter_token_latency`, `time_per_output_token`, `e2e_latency` | `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds` | histogram, s | residency |
| `inference_time`, `prefill_time`, `decode_time` | `vllm:request_inference_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` | histogram, s | residency; see below |
| request shape | `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, `vllm:request_max_num_generation_tokens`, `vllm:request_params_n`, `vllm:request_params_max_tokens`, `vllm:request_prefill_kv_computed_tokens` | histogram | finished requests' sizes and parameters |
| `spec_decode_*` | `vllm:spec_decode_num_drafts_total`, `..._draft_tokens_total`, `..._accepted_tokens_total`, `..._accepted_tokens_per_pos_total{position}` | counter | only with speculative decoding |
| `kv_transfer` group | `vllm:kv_offload_*` | mixed | only with a KV connector or offloading |
| `estimated_*_per_gpu` | `vllm:estimated_flops_per_gpu_total`, `vllm:estimated_read_bytes_per_gpu_total`, `vllm:estimated_write_bytes_per_gpu_total` | counter, estimated | advance only with `--enable-mfu-metrics`; estimates from model shapes, per GPU |
| `kv_block_*` | `vllm:kv_block_lifetime_seconds`, `..._idle_before_evict_seconds`, `..._reuse_gap_seconds` | histogram, s | only with `--kv-cache-metrics`; sampled |

Retired names recognised: `vllm:gpu_cache_usage_perc` (now
`kv_cache_usage_perc`), `vllm:gpu_prefix_cache_queries_total` and
`vllm:gpu_prefix_cache_hits_total` (now `prefix_cache_*`),
`vllm:time_in_queue_requests` (now `request_queue_time_seconds`), and the
removed `vllm:model_forward_time_milliseconds`,
`vllm:model_execute_time_milliseconds`, `vllm:num_requests_swapped`,
`vllm:cpu_cache_usage_perc` and `vllm:request_params_best_of`, which have no
successor.

### Three facts about the inference-time fields

1. `vllm:request_inference_time_seconds` and the span attribute
   `gen_ai.latency.time_in_model_inference` are one expression:
   last token processed minus first schedule. The Prometheus histogram and
   the span are the same number exported twice, not two sources.
2. That stopwatch starts the first time a request is scheduled and stops
   when the output of its last step has been processed. With async
   scheduling (the default) the engine processes a step's output only after
   it has scheduled and launched the next step, so every request is charged
   one step it was not in. Under a steady load that is under 1%; when
   requests finish in waves it can be a few percent.
3. `--collect-detailed-traces all` is accepted and documented as costly, but
   in 0.30.0 nothing sets `gen_ai.latency.time_in_model_forward` or
   `gen_ai.latency.time_in_model_execute`. The capability matrix lists them
   as unsupported; no run will ever carry them.

## Spans

A span's attributes are kept as exported: `gen_ai.request.id`,
`gen_ai.latency.time_in_queue`, `time_to_first_token`,
`time_in_model_prefill`, `time_in_model_decode`, `time_in_model_inference`,
`e2e`, `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens` and the
request parameters. The receiver accepts OTLP/HTTP protobuf bodies (with the
`infer-otlp` extra) and JSON bodies. Spans of other names, such as vLLM's
startup spans, are kept but not joined.

For the v2 correlation model, `spans_to_correlation_events` maps each request
span onto an `infer.request` record with vLLM's own timestamps (provenance
`reported`) and `infer.stage` records for `queue`, `prefill`, `decode` and
`inference`, placed from the span start by adding the reported durations in
scheduler order (provenance `estimated`, native attribute in metadata). They
remain residency; see [Inference execution correlation](inference_correlation.md)
for how GPU time is accounted separately.

## Capability matrix

| Component | Verified | Supported | Not supported |
| --- | --- | --- | --- |
| `vllm.metrics` | vLLM 0.30.0, CUDA backend, single and data-parallel engines (`engine` label) | the fields in the metric map; unknown series kept raw | per-request attribution from scrapes |
| `vllm.spans` | vLLM 0.30.0 over OTLP/HTTP protobuf (`infer-otlp` extra, `opentelemetry-proto>=1.20`) and JSON; OTLP JSON and JSONL files | `llm_request` spans and their native attributes; join by `X-Request-Id` | `time_in_model_forward`, `time_in_model_execute` (never set in 0.30.0); gRPC export; protobuf without the extra |
| vLLM profiler controls | see [Inference execution correlation](inference_correlation.md) | `/start_profile`, `/stop_profile` orchestration lives with the GPU trace capture work | starting a profile without `--profiler-config` at server start |

Other engines, tensor-parallel scrapes from several API servers behind a load
balancer, and vLLM versions other than 0.30.0 are untested. A scrape taken
through a load balancer can mix engines; the `engine` label and
`process_start_time_seconds` tell the report when that happened, and it
leaves the window unresolved rather than mixing them.

## Example

`examples/cli/vllm_native_telemetry.py` runs the same workload with unique
prompts, with shared-prefix prompts, and at a concurrency that saturates the
scheduler, then prints the telemetry blocks side by side: prefix-cache hits
rise with shared prefixes, queue depth and queue time rise under saturation,
and the KV occupancy follows. The differences are what the engine reports;
the example does not claim that one caused the other.
