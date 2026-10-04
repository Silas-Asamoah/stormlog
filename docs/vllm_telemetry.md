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
`--otlp-traces-endpoint` pointing at the receiver that `infer profile` runs.
vLLM 0.30.0 exports spans over gRPC unless
`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` is set
(`vllm/tracing/otel.py`, `get_span_exporter`), so the variable is part of
every server line below:

```bash
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
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
| `--vllm-metrics [URL]` | Scrape `/metrics` just before each phase's first send, after its drain, and every `--vllm-metrics-interval` seconds (default 1) in between. Without a URL, the endpoint's origin plus `/metrics`. The endpoint's bearer token (`--api-key`, `OPENAI_API_KEY`) goes with the scrape only when the URL has the endpoint's scheme, host and port; any other origin is scraped without credentials, and the run says so once. A redirect is never followed: that scrape fails and names the target, so the token cannot be forwarded off its origin. |
| `--vllm-spans-listen [HOST:PORT]` | Run an OTLP/HTTP receiver (default `127.0.0.1:4318`) for the length of the run and keep every span vLLM exports to it. |
| `--vllm-spans-drain SECONDS` | Keep the receiver listening this long after the last phase (default 6), because vLLM's exporter flushes spans in batches every 5 s (`OTEL_BSP_SCHEDULE_DELAY`), so the last requests' spans leave after the last answer. |
| `infer analyze --vllm-spans FILE` | Load spans someone else collected: an OTLP JSON export, or one span per line. May be given more than once. |

Scrapes are stamped on the client's clock, like the phase windows, so no
clock alignment is involved. A scrape that fails or returns something that
is not Prometheus text is recorded with its reason and the run goes on; the
first failure is printed once. A run stopped with Ctrl+C waits at most
2 s for an interval scrape still in flight, then leaves it to its thread
and records nothing from it; takes the phase-end scrape under a 2 s
deadline for the whole scrape, not per byte, so a server that stopped
answering, or one that dribbles its answer, cannot hold the stop back;
keeps the spans received so far; and writes the capability records before
the session's last word. That holds on Python 3.10 too, where Ctrl+C
cancels every task of the run at once, and when the stop lands in the
`--vllm-spans-drain` wait after the last phase: the wait is given up,
the receiver is still stopped and its spans still kept. The receiver
binds before any request is
sent; a port it cannot bind is recorded as unavailable, with the error, and
the run goes on without spans. Without the `infer-otlp` extra the receiver
still listens and accepts OTLP JSON, but vLLM's exporter sends protobuf, so
those exports are refused with a 415, counted, and the capability record
lists `otlp_http_protobuf` as supported but not enabled; the run warns once.
A server left on its gRPC default opens HTTP/2 connections instead; the
receiver counts those as `grpc_attempts` in the capability record, with the
variable to set, so an empty span count explains itself.

The receiver exists only while the profile runs. When it stops it closes
every connection it accepted, waits briefly for an export already being
read, and answers anything that still arrives with a 503, counted as
`after_stop` and never kept. Spans vLLM exports when no receiver is
listening, such as its startup spans or the spans of traffic between
runs, fail on the server side and are dropped there; the exporter
retries a failed batch with backoff for up to its export timeout
(`OTEL_EXPORTER_OTLP_TRACES_TIMEOUT`, 10 s by default in opentelemetry-sdk
1.44.0), which can hold back the next batch. For short runs, start vLLM with
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
  error instead. About 30 KB per scrape for 0.30.0, most of it the sixteen
  histograms' bucket lists. NaN and infinite sample values are written as
  the strings `NaN`, `+Inf` and `-Inf`, so every line is strict JSON.
- `infer.vllm_span`: one record per span, attributes under their native
  names, timestamps on the exporter's wall clock, and the recovered
  `request_id`. The clock domain is named by the exporter's `host.name`
  resource attribute when it has one; vLLM's resource has none, so spans the
  receiver collects are named by the peer address the export came from,
  such as `127.0.0.1/unix_epoch_ns`, which never counts as the client's
  clock. Bodies may be gzip-encoded; a body over 32 MiB, as sent or once
  inflated, is refused with 413 before it is held whole in memory.
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
| `histograms` | count and sum deltas, the mean, and per-bucket deltas under the native `le` boundaries, per extra label set (`by_label`, each set differenced against itself) with the total over the sets when every one resolves | the same two scrapes |
| `gauges` | min, mean, max, last and sample count over every scrape in the window | all scrapes of the case |
| `derived.rates` | prompt and generated tokens per second, finished requests per second | counters over the window |
| `derived.prefix_cache` | queries, hits and the hit ratio, in tokens | counters |
| `derived.kv_cache` | peak and mean usage as a fraction, and in blocks when `cache_config_info` names `num_gpu_blocks` | gauge plus info labels |
| `derived.mfu` | the estimated FLOPs and bytes per GPU, or `not_enabled`; zeros are `unresolved` while the generated token delta is | counters |
| `spans` | requests with a span, and p50/p95/mean of each native latency attribute | joined spans, each once by trace and span id: a repeated delivery (an exporter retry, two overlapping files) is counted as a duplicate, one that differs from the first as conflicting, and the first is the one kept |

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
| `engine_restart` | The exporter's `process_start_time_seconds` differs on any scrape of the window, the interval ones included; a scrape another exporter answered is counted as `foreign_scrapes` and kept out of the gauge summaries |
| `engine_set_changed` | The set of `engine` labels differs on any scrape of the window |
| `exporter_identity_unknown` | A scrape has no `process_start_time_seconds`, so a restart cannot be told from a quiet window |
| `counter_reset` | The counter, or a histogram's count, sum or bucket, went backwards |
| `counter_recreated` | The counter's or histogram's `*_created` timestamp changed (a counter's stamp drops `_total`, a histogram's keeps its full name) |
| `series_missing` | One scrape lacks the series, or a histogram's `_sum` or `_count` sample (`missing` names which); neither is ever taken as zero |
| `bucket_boundaries_changed` | The histogram's `le` set differs between the scrapes |
| `not_enabled` | The MFU counters stayed at zero while tokens were generated |
| `non_finite_sample` | A sample was `NaN` or infinite: a counter or histogram with one is not differenced, a gauge's statistics cover its finite samples and count the others, and nothing derived from the value resolves; the rest of the report completes |

A series the catalog does not name is kept under its native name and listed
as unknown, unless it belongs to an optional subsystem the catalog knows by
prefix (`vllm:kv_offload_*`, `vllm:spec_decode_*`, `vllm:diffusion_*`): those
are listed as `optional_present` and kept raw. A retired name is normalised
under its successor, kept as `deprecated_alias_of`, and listed as retired.

### Windows of scrapes

`stormlog.infer.scrape_window` aggregates any window of scrapes the caller
chooses, for one `engine` label, and is what windowed analyses use (online
triggers and the incident diagnoser). It follows the rules above and adds
two:

- **Consecutive deltas.** A counter's or histogram's change over the window
  is the sum of its changes between consecutive scrapes. A value that went
  backwards between two interior scrapes is `counter_reset` even when the
  window's first and last values look consistent; the case blocks above,
  which compare only the phase-start and phase-end scrapes, cannot see that.
- **Sample intervals.** A scrape sampled the server at some instant between
  its stamp (`observed_at_ns`) and its response. A window's duration is
  measured between the first and last scrapes' interval midpoints, and rates
  carry the bounds the intervals allow. A scrape bounded only by its
  `duration_ms` misses the fetch thread's start delay, so its window says
  `placement: approximate`.

A series that more than one label set matches, such as
`vllm:num_requests_waiting_by_reason` without a `reason`, is
`ambiguous_series` rather than a sum, and a scrape with several engines needs
the engine named (`engine_required`). The window is checked as a whole, not
family by family: a signal that divides one family by another would otherwise
read each from whichever engine exported it. A series whose labels change
between scrapes (another model name, say) is a different series, so it is
never differenced across the change (`series_labels_changed`). A gauge's
samples, like a counter's, must all come from one exporter: samples from
both sides of a restart are not one gauge (`engine_restart`), and a family
that is not a gauge (a histogram, say) gives no gauge samples
(`not_a_gauge`).

Scrapes must be given in strictly increasing stamp order
(`scrapes_out_of_order`, `duplicate_scrape_time`), and nothing is differenced
when they are not. Their sample midpoints must increase too: a quick scrape
inside a slow one's interval may have sampled first, so it is
`scrapes_out_of_order` as well. A failed scrape sampled nothing, so only its
stamp takes part in the order, not its midpoint. A failed scrape inside a
window leaves fewer samples, and counters are differenced across it; a failed
first or last scrape shortens the window the caller chose, so the window is
`scrape_failed`. Each change of a histogram between consecutive scrapes must
itself be a histogram: cumulative counts that never fall as the boundary
rises, none above the change in `_count`, and the `+Inf` bucket equal to it.
Two scrapes that are each valid can differ by a change that is not one, and
its shares would fall outside 0 to 1, so such a window is
`histogram_inconsistent`. A histogram's share of observations above a value,
and the bucket holding a quantile, are reported as bounds between bucket
boundaries; a quantile in the `+Inf` bucket has no upper bound
(`quantile_in_overflow_bucket`).

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
2. That stopwatch starts when the scheduler first admits the request
   (`scheduled_ts`, stamped at the top of `Scheduler.schedule`) and stops
   when the engine core has processed the output of the step that produced
   its last token (`last_token_ts`, the `EngineCoreOutputs` timestamp). With
   async scheduling, which vLLM turns on when the executor supports it, the
   admitting `schedule()` runs while the previous step is still on the GPU,
   so the interval starts up to one step before the request's first kernel.
   The step's output is processed only after the next step has been
   scheduled and launched, so when the CPU is the bottleneck the interval
   also runs past the last kernel by that time. On an A30 with vLLM 0.30.0
   the overhang was under 1% of a request's time at one request in flight
   and 3–4% at 64 (up to 6% at 16 in flight on a 7B model). A request that
   stops at `max_tokens` is not scheduled into the following step, so no
   step after its last token is included.
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
| `vllm.metrics` | vLLM 0.30.0, CUDA backend, one engine | the fields in the metric map, one block per `engine` label (data-parallel engines are kept apart by construction, not verified on a DP server); unknown series kept raw | per-request attribution from scrapes |
| `vllm.spans` | vLLM 0.30.0 over OTLP/HTTP protobuf (`infer-otlp` extra, `opentelemetry-proto>=1.20`) and JSON; OTLP JSON and JSONL files | `llm_request` spans and their native attributes; join by `X-Request-Id` | `time_in_model_forward`, `time_in_model_execute` (never set in 0.30.0); gRPC export; protobuf without the extra |
| vLLM profiler controls | see [Inference execution correlation](inference_correlation.md) | `/start_profile`, `/stop_profile` orchestration lives with the GPU trace capture work | starting a profile without `--profiler-config` at server start |

Other engines, data-parallel servers, tensor-parallel scrapes from several
API servers behind a load balancer, and vLLM versions other than 0.30.0 are
untested. A scrape taken
through a load balancer can mix engines; the `engine` label and
`process_start_time_seconds` tell the report when that happened, and it
leaves the window unresolved rather than mixing them.

## Example

`examples/cli/vllm_native_telemetry.py` runs the same workload with unique
prompts, with shared-prefix prompts, and at a concurrency that saturates the
scheduler, each run under its own seed so no run warms the prefix cache for
the next, then prints the telemetry blocks side by side: prefix-cache hits
rise with shared prefixes, queue depth and queue time rise under saturation,
and the KV occupancy follows. The differences are what the engine reports;
the example does not claim that one caused the other.
