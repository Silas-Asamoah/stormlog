[← Back to main docs](index.md)

# Exporting inference metrics and spans

`stormlog infer profile` can expose what it measures to Prometheus, and
send its own spans to an OpenTelemetry collector, while the run lasts. The
JSONL artifact is still written in full, and stays the record the analysis
reads. Each export is optional and off unless you ask for it. Standard
`OTEL_*` environment variables never turn one on.

**What is exported:** what Stormlog itself measured or decided, such as
client-observed latencies, request outcomes, token counts, scrape health
and the exporter's own health; and, as spans, the capture, its phases and
every request it sent.

**What is not exported:** anything Stormlog collected from the engine. A
`vllm:*` series scraped from vLLM's `/metrics` is never re-exposed, a span
received from vLLM's exporter is never forwarded, and no engine quantity is
re-aggregated. Prometheus can scrape vLLM directly, and a copy would count
the same events twice. A Stormlog metric that overlaps an engine metric in
meaning has its own name. The table below says which pairs must never be
added together.

## Start

Serve `/metrics` while the run lasts, and keep it up 30 s afterwards for a
final scrape:

```bash
stormlog infer profile \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,8 --requests 64 --output artifacts/infer.jsonl \
  --prometheus-listen 127.0.0.1:9900 --prometheus-linger 30
```

Or write a textfile, with no server at all, for node_exporter's textfile
collector or to read yourself:

```bash
stormlog infer profile ... \
  --prometheus-textfile-dir /var/lib/node_exporter/textfile \
  --prometheus-slot bench-a
```

There is no default port. Choose one that is free on your host; the
registry of Prometheus exporter ports is crowded around 9100–9999.

| Flag | What it does |
| --- | --- |
| `--prometheus-listen HOST:PORT` | Serve `GET /metrics` (text format 0.0.4) while the run lasts. Loopback is the safe choice: the endpoint has no authentication, and a non-loopback address prints one warning saying so. |
| `--prometheus-linger SECONDS` | Keep `/metrics` up this long after the run, serving the final values (default 0, at most 3600). The run says where and for how long when the linger starts, and prints its report after it. Skipped when the run ends with Ctrl+C or an error; a Ctrl+C during the linger ends only the linger. |
| `--prometheus-textfile-dir DIR` | Write `DIR/stormlog-<slot>.prom` at start, every `--prometheus-textfile-interval` seconds (default 15, from 1 to 3600), and once more at the end. |
| `--prometheus-slot NAME` | This producer's name (default `default`), 1–64 of `A-Z a-z 0-9 _ . -`. It is the `stormlog_producer` label on every series, at the endpoint and in the textfile, and it names the textfile and its lock. |
| `--prometheus-textfile-remove-on-exit` | Remove the textfile at the end instead of keeping its final values. |
| `--prometheus-max-series N` | Refuse a run whose metrics need more than N samples per scrape (default 50,000). |
| `--prometheus-max-bytes BYTES` | Refuse a run whose metrics could exceed BYTES per scrape (default 16 MiB). |
| `--prometheus-series-headroom N` | Series a family may add beyond the configured ones (default 0 for a profile, whose label values all come from its configuration). |
| `--prometheus-case-label off` | Every series gets `case="all"`, for a matrix too large for the budget. |

A setting that cannot be used exits 2 before anything is sent, and no
artifact is written (see the [exit-code contract](report_contract.md)):
- an address that is not `HOST:PORT`;
- an export flag given without `--prometheus-listen` or `--prometheus-textfile-dir`, even with its default value;
- a span flag (`--otlp-flush-timeout`, `--otlp-probe-interval`, `--otlp-sample-ratio` and the like) given without `--otlp-endpoint` or `--otlp-file`, even with its default value; `--otlp-sample-ratio` also applies with `--trace-context follow-sampling`;
- a slot another live writer holds, or whose lock is not a plain file (a
  link, a directory);
- a textfile directory that is missing, holds the artifact, or cannot be
  written;
- a matrix over the budget.

Both outputs may be on at once. Then let Prometheus read only one of them:
scraping the endpoint and node_exporter's textfile collector together
gives every series twice, and a query that joins on `stormlog_run_info`
then matches two series per producer.

## What is exported

Units are base units: seconds and counts. Every series also carries
`stormlog_producer`.

| Metric | Kind | Labels | Meaning | Never add to |
| --- | --- | --- | --- | --- |
| `stormlog_infer_requests_total` | counter | model, server, case, phase, status | Requests by outcome, as the client saw them. `status` is `ok`, `timeout`, `rejected`, `error`, `dropped`, `cancelled`, `unreachable` (it never reached the server) or `delivery_unknown` (it may have). | `vllm:request_success_total`, which counts every client's finished requests and never sees a request that did not reach it |
| `stormlog_infer_request_duration_seconds` | histogram | model, server, case, phase | End-to-end latency of completed requests, from send to last byte | `vllm:e2e_request_latency_seconds`, the engine's residency, which leaves out HTTP and the network |
| `stormlog_infer_time_to_first_token_seconds` | histogram | same | Send to the first non-empty content delta, streaming only | `vllm:time_to_first_token_seconds` |
| `stormlog_infer_time_to_first_chunk_seconds` | histogram | same | Send to the first streamed chunk | — |
| `stormlog_infer_chunk_interarrival_seconds` | histogram | same | Gaps between streamed chunks. This is chunk timing, not token inter-token latency | `vllm:inter_token_latency_seconds` |
| `stormlog_infer_e2e_from_intended_seconds` | histogram | same | Open-loop arrivals only: completed requests measured from their intended arrival, so a held arrival's wait is included | — |
| `stormlog_infer_dispatch_lag_seconds` | histogram | case, arrival_mode | How late each request was sent against its schedule | — |
| `stormlog_infer_tokens_total` | counter | model, server, case, phase, direction, source | Prompt and output tokens of completed requests. `source` is where each count came from: `server_usage`, `tiktoken`, `transformers`, `estimated` or `unknown`. A count no counter can take, negative or past 2^53, is left out and counted as `tokens_rejected` in the capability record | `vllm:prompt_tokens_total` and `vllm:generation_tokens_total` (every client) |
| `stormlog_infer_requests_held_for_slot_total` | counter | case | Open-loop arrivals that found every in-flight slot taken: they waited for one, or with `--overflow drop` were dropped | — |
| `stormlog_infer_phases_total` | counter | case, phase | Completed phase windows | — |
| `stormlog_infer_abandoned_requests_total` | counter | case, phase | Requests from an earlier drain still running when a phase was ready to start | — |
| `stormlog_engine_scrapes_total` | counter | server, outcome | Scrapes of vLLM's `/metrics` (with `--vllm-metrics`), `ok` or `error` | — |
| `stormlog_engine_scrape_duration_seconds` | histogram | server | How long each of those scrapes took | — |
| `stormlog_engine_last_scrape_timestamp_seconds` | gauge | server, outcome | When the engine was last scraped, by outcome, in Unix seconds; absent until a scrape has that outcome | — |
| `stormlog_engine_metrics_source_changes_total` | counter | server | Times the process answering `/metrics` changed: an engine restart, or a load balancer reaching another process | — |
| `stormlog_engine_span_receiver_requests_total` | counter | outcome | Span exports the receiver refused or could not read (with `--vllm-spans-listen`). These count HTTP requests, not spans | — |
| `stormlog_engine_span_receiver_spans_total` | counter | — | Spans the receiver decoded and kept for the artifact. Never forwarded | — |
| `stormlog_trace_windows_total` | counter | stop_reason, started | Profiler trace windows (with `--trace`) by how they ended | — |
| `stormlog_run_info` | gauge | run_id, session_id, version, command | 1 for the run being exported | — |
| `stormlog_run_start_time_seconds` | gauge | — | When the run started, in Unix seconds | — |

`model` is the `--model` given. `server` is the endpoint's origin
(scheme, host and port), never its path or credentials. `case` is the case
ID, and `phase` is `warmup` or `measured`. Exclude warmup in a query with
`phase="measured"`.

The run's identity is in one series, `stormlog_run_info`, so it does not
multiply the others. To label other series with it, join on the producer:

```promql
sum by (stormlog_producer, status) (rate(stormlog_infer_requests_total[1m]))
  * on (stormlog_producer) group_left (run_id) stormlog_run_info
```

Histogram bucket bounds are fixed, in seconds:

| Histogram | Bounds |
| --- | --- |
| request duration, latency from intended | 0.05 0.1 0.25 0.5 1 2 5 10 15 20 30 45 60 90 120 180 300 600 |
| time to first token, time to first chunk | 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5 10 30 |
| chunk interarrival | 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.15 0.2 0.25 0.5 1 2.5 5 10 30 |
| dispatch lag | 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 |
| scrape duration | 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5 |

## Labels and the budget

Labels are of two kinds:
- **Closed sets:** `status`, `phase`, `direction`, `source`, `outcome`, `stop_reason`.
- **Values your configuration names:** `model`, `server`, `case`, `arrival_mode`.

No label ever holds a request ID, a trace ID or a prompt ID. Those belong in
the artifact (and in spans).

Because every value is known when the run starts, the exact set of series is
created then, at 0, and so are the number of samples per scrape and the
largest size a scrape can reach. A run over `--prometheus-max-series` or
`--prometheus-max-bytes` is refused before it sends anything, and the message
gives both figures. For a closed-loop matrix with scraping and traces on:

| Cases | Samples | Largest scrape |
| --- | --- | --- |
| 2 | about 780 | about 0.3 MiB |
| 20 | about 5,000 | about 1.2 MiB |
| 200 | about 48,000 | about 9.7 MiB |
| 300 | about 71,000: refused at the default | about 14 MiB |

Most of a case's samples are its five client histograms: 15 to 21 samples
each, per phase. For a larger matrix, raise the limits or pass
`--prometheus-case-label off`.

A configured value longer than 64 characters, such as a long model path,
becomes its first 55 characters, `~` and 8 hex characters of its SHA-256. Two
long values that share a prefix therefore stay two series.

If a series beyond a family's cap still appears, the outcome depends on the
metric kind:
- **A counter or histogram** keeps its closed-set labels and has its
  configuration labels replaced by `__overflow__`. Totals over the closed
  sets stay exact; only the per-case breakdown of the extra series is lost.
- **A gauge's** extra series is dropped.

Either way, `stormlog_metrics_series_overflow_total` or
`stormlog_metrics_series_rejected_total` counts it. A profile should never
see either, since its values are all known.

## The textfile

One slot names three things:
- the label `stormlog_producer="<slot>"`;
- the file `DIR/stormlog-<slot>.prom`;
- the lock `DIR/stormlog-<slot>.lock`.

Two runs in one directory with the same slot therefore contend for one
lock: the second is refused, so the two files can never hold identical
series (node_exporter would fail the scrape). Give concurrent runs
different slots.

The writer holds the lock with `flock` for the whole run, so it is released
when the run's process ends, however it ends, and the next run with that
slot simply takes it. The lock file names its writer's pid and host, for
the refusal's message. A lock file naming another host, as on a shared
filesystem, is never taken over, since `flock` may not reach across hosts.
Delete it by hand once you know its writer is gone. On a system without
`flock`, a lock is taken over only when its writer is certainly gone: its
process no longer exists, or that pid now belongs to a process with a
different start time.

Neither the lock nor the temporary file below is ever opened through a
link: whoever else can write the directory cannot make a write land in
another file. A lock that is a link, or not a regular file, refuses the
run; a temporary file that is one fails that write.

Each write goes to a temporary file in the same directory, which then
replaces the real one, so a reader never sees half a file. A failed write
leaves the previous file in place. The file carries two freshness series:
- `stormlog_textfile_updated_timestamp_seconds`, when it was last written;
- `stormlog_run_active`, which is 1 while the run lasts and 0 in the final
  write.

After the run, the file keeps its final values until the next run with that
slot replaces it, unless `--prometheus-textfile-remove-on-exit` is given.
node_exporter's own `node_textfile_mtime_seconds` shows how old each file is.
A directory that holds the artifact is refused; give the textfile a
directory of its own.

## The endpoint

- At most 4 connections are served at once. A fifth gets a 503 straight
  away, before any thread starts.
- Each connection serves one request, and is closed after 10 s whatever the
  client does, including sending nothing, dribbling its headers or never
  reading the answer.
- A request line and headers over 16 KiB get a 431 and the connection is
  closed; a scraper sends a few hundred bytes.
- Scrapes share one rendering, rebuilt at most once a second, so a scrape
  can be up to a second behind.
- After the run, the endpoint serves the final, frozen values until the
  linger ends.

## The exporter's own health

These series sit beside the exported metrics:
- `stormlog_metrics_records_applied_total`, and
  `stormlog_metrics_records_dropped_total{reason}` with `reason` one of:
  - `queue_full`: the exporter's queue was full: 8 MiB, counted as the
    memory each queued record holds, or 65,536 records, whichever comes
    first. A request record holds about 1.8 to 2.5 KiB, so the bytes bind
    first, at about 3,400 to 4,800 records;
  - `shutdown`: not yet applied when the exporter's close stopped waiting
    for it;
  - `error`: its update failed and was undone whole, which is an exporter
    bug, counted in `stormlog_exporter_internal_errors_total` too;
  - `closed`: refused by the queue after it closed. The exporter stops
    taking records first, so this stays 0 in a profile, and the records it
    no longer takes (the run's own capability and session records) are
    never mapped.

  While every reason is 0, no record failed to be read
  (`stormlog_exporter_internal_errors_total{entry="observe"}`) and no token
  count was left out (`tokens_rejected`), the exported totals are exact:
  the capability record's `exact` says so. They are
  exact for the records offered to the exporter, which it is handed as
  each is written; a record written just as a second Ctrl+C lands may not
  be.
- `stormlog_metrics_scrapes_total{outcome}`, with `outcome` one of `ok`,
  `rejected_busy`, `timeout`, `not_found`, `bad_request` and `error`, and
  `stormlog_metrics_textfile_writes_total{outcome}`.
- `stormlog_exporter_internal_errors_total{entry}`: exporter failures that
  were caught before they could reach the run.
- `stormlog_health_snapshot_age_seconds`: how long ago the exporter last
  read its health sources, such as the span receiver; absent until the
  first read.

The artifact keeps the same figures, final, in an `infer.capabilities` record
for `export.prometheus`:
- `supported`, `enabled` and `collected` name the endpoint and the textfile.
- `metadata.summary` holds the record counts (and whether they are `exact`),
  the budget, series and overflow per family (only here: the overflow and
  rejection counters above have no family label), scrape outcomes, the
  textfile's writes (with `final_stale` true if its final write could only
  use values from before they froze), the slot, and any endpoint error.

That record is written when the capture ends, after the exporter has
stopped and frozen its values, and before the session's last record.

## Exporting spans

Send spans over OTLP/HTTP to a collector, or write them to a file:

```bash
stormlog infer profile ... --otlp-endpoint http://127.0.0.1:4318
stormlog infer profile ... --otlp-file artifacts/spans.jsonl
```

| Flag | What it does |
| --- | --- |
| `--otlp-endpoint URL` | POST spans to this OTLP/HTTP traces URL. A bare origin gets `/v1/traces`. Requests are protobuf with the `infer-otlp` extra installed, JSON without it, and always gzip-compressed. Credentials in the URL are refused: use `--otlp-header`. |
| `--otlp-file PATH` | Append spans to PATH as OTLP JSON, one export request per line, up to 256 MiB: the format of the OpenTelemetry Collector's file exporter, which its `otlpjsonfile` receiver reads. A line is written whole or not at all, and a missing directory is created. Use this or `--otlp-endpoint`, not both. The file may not be the artifact, nor sit in `--prometheus-textfile-dir` or `--vllm-execution-dir`, however the path is spelled: through a link, or in other case, which a case-insensitive disk folds. Such a run exits 2. A name that differs from the artifact's only in case is refused on every disk. |
| `--otlp-file-fsync` | fsync the file after each line. |
| `--otlp-header NAME=VALUE` | A request header, such as an API key; repeatable. `OTEL_EXPORTER_OTLP_HEADERS` and `OTEL_EXPORTER_OTLP_TRACES_HEADERS` are read too, flags winning. Values are sent, never recorded. |
| `--otlp-allow-insecure-headers` | Send the headers over plain `http://` to a host other than this one. Without it, headers from the flags or the variables never go in clear text off the host: such a run exits 2. This host is a loopback address (127.0.0.0/8 or ::1) or the name `localhost`; any other name, even one that starts with `127.`, is another host. |
| `--otlp-resource-attribute KEY=VALUE` | A resource attribute; repeatable. See "The resource" below. |
| `--otlp-resource-attribute-allow KEY` | Accept one more resource key. |
| `--otlp-sample-ratio RATIO` | Keep this fraction of the successful request spans sent without trace context (default 1). Failed and cancelled requests, and every request sent with a `traceparent`, are always kept. Under `--trace-context follow-sampling` it is also the flag's ratio; see "Trace context". |
| `--otlp-flush-timeout SECONDS` | How long the end of the run waits for spans to leave (default 5; at most 2 after Ctrl+C). |
| `--otlp-probe-interval SECONDS` | While the collector is down, how often it is retried (default 8; 0.5 to 30). |
| `--export-content ITEMS` | Free text spans may carry, comma-separated: `digests`, `errors`, `prompts`, `outputs`. Default none; see "What a span holds". |

Spans and Prometheus can be used together, or either alone. With both, the
span exporter's own figures are also metrics (see "The span exporter's
health").

### The spans

| Span | Kind | Where | One per |
| --- | --- | --- | --- |
| `stormlog.infer.capture` | INTERNAL | the root of the capture's trace | run, sent as the capture ends |
| `stormlog.infer.phase` | INTERNAL | child of the capture | phase window (warmup and measured, per case) |
| `stormlog.infer.trace_window` | INTERNAL | child of the capture | profiler window (with `--trace`) |
| `stormlog.infer.request` | CLIENT | its own trace, linked to its phase | request sent: never one dropped before sending |

A request span with `--trace-context` keeps the trace and span IDs sent in
its `traceparent`, so a tracing vLLM's span for that request is its child.
Without it, a request's IDs, like every other span's, are derived from the
session and the record: the same artifact always maps to the same spans.

`--otlp-sample-ratio` keeps a successful request span when the lowest 56
bits of its trace ID are at least (1 − ratio) · 2⁵⁶; left-out spans are
counted as `sampled_out`. A failed or cancelled request is always kept, and
so is every request sent with a `traceparent`: the server may have
recorded a child of it, which without it would point at a parent no
backend ever receives.

### What a span holds

Every attribute is on a fixed list, and every value is a configuration
identifier, an ID Stormlog made, a value from a closed set, or a number:

- **Request spans:** `gen_ai.operation.name` (`chat`),
  `gen_ai.request.model`, `gen_ai.request.max_tokens`, `server.address`,
  `server.port`, `http.request.method`, `url.path`,
  `http.response.status_code`, `error.type`,
  `gen_ai.response.time_to_first_chunk`, and `stormlog.*`: the run, session,
  request and `X-Request-Id`, the case and phase, `request.status`, the
  arrival mode and index, dispatch lag, time to first token (also a
  `stormlog.first_token` event), prompt and output token counts with their
  sources, the prompt ID, prefix group and shared prefix, and the chunk
  count.
- **Status:** `timeout`, `rejected`, `error`, `unreachable` and
  `delivery_unknown` are ERROR, with `error.type` set to the status, the HTTP
  status code, or the exception's class name. `cancelled`, where Stormlog
  stopped waiting at its own drain deadline, leaves the status unset;
  `stormlog.request.status` says what happened. A trace window whose
  profiler call failed is ERROR. Only an ERROR span has a status message, as
  OpenTelemetry asks.
- **A server's error** is reduced to its OpenAI-style `type` and `code`,
  each mapped to a known value or `other`: `stormlog.error.api_type` and
  `stormlog.error.api_code`. Its message and `param` are left out.
- **URLs:** only the endpoint's host and port. `url.path` is the path only
  when it is a standard one (`/v1/chat/completions`, `/v1/completions`,
  `/metrics`, `/v1/traces`), and `<redacted>` otherwise. `url.full` is
  never exported.
- No `gen_ai.usage.*`: token counts come from Stormlog's records, with
  their sources.
- **The `gen_ai.*` names** follow the OpenTelemetry GenAI conventions as of
  commit `e07f4eb` of
  [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai/tree/e07f4ebacb08f56db8c4c882d117720333fbca04)
  (2026-10-02), where they moved from the core conventions; the scope's
  `schema_url` (1.44.0) covers the rest, and marks `gen_ai.*` deprecated
  there for that reason. The GenAI conventions are still in development.
  Two of their recommendations are not followed: the span keeps Stormlog's
  name, `stormlog.infer.request`, rather than `{operation} {model}`, so it
  is found the same way whatever the model; and `gen_ai.provider.name` is
  left out, since Stormlog cannot know what serves an OpenAI-compatible
  endpoint.

`--export-content` adds free text, each item cut to 1 KiB:
- `digests`: the prompt's digest, from the artifact, and the output's;
- `errors`: the server's error text, as a failed request span's status
  message, and a failed trace window's control error. A server's error can
  echo the request, so
  this is consent to export echoed prompt text even without `prompts`;
- `prompts`, `outputs`: the request's text.

Consented text passes through pattern scrubbing (bearer tokens, `key=`
pairs, URL credentials, common key shapes). Every exported string, consented
or not, has every credential the run was given replaced with `<redacted>`:
the API key, the OTLP header values (and, after an auth scheme such as
`Bearer`, the credential alone; for `Basic`, its decoded user and
password), and the user names, passwords and query values of the run's
URLs, in raw, percent-encoded, JSON-escaped and base64 forms.

### The resource

Every span's resource has `service.name` (`stormlog`), `service.version`,
`service.instance.id` (the session ID), `host.name`, `process.pid` and
`stormlog.run_id`. More attributes can come from `OTEL_RESOURCE_ATTRIBUTES`,
`OTEL_SERVICE_NAME` and `--otlp-resource-attribute`, in that order, a later
one winning. Only these keys are accepted:

`service.name`, `service.namespace`, `service.version`,
`service.instance.id`, `deployment.environment.name`,
`deployment.environment`, `host.name`, `host.id`, `host.arch`, `os.type`,
`k8s.cluster.name`, `k8s.namespace.name`, `k8s.pod.name`, `k8s.pod.uid`,
`k8s.node.name`, `k8s.deployment.name`, `k8s.statefulset.name`,
`k8s.container.name`, `cloud.provider`, `cloud.platform`, `cloud.region`,
`cloud.availability_zone`, `container.name`, `container.id`.

`--otlp-resource-attribute-allow KEY` accepts another key, unless its name
contains `pass`, `pwd`, `secret`, `token`, `key`, `auth`, `bearer`, `cred`,
`cookie`, `session`, `signature` or `private`: that is refused (exit 2). A
value must be printable and at most 128 characters. Any key left out is
listed by name, never with its value, in the `export.otlp` record.

### Delivery and accounting

Spans are batched (512 spans, 4 MiB, or one second after the first) and
sent one batch at a time, each attempt on its own connection within 5 s.
Each attempt is classified by what is known:

| Attempt | When |
| --- | --- |
| `confirmed` | a readable 200, which says how many spans were rejected |
| `refused` | a 3xx (never followed), a 4xx, 429 or 503 |
| `ambiguous` | the body was sent, then a timeout or reset; a 5xx other than 503; any 2xx other than 200; or a 200 whose body is unreadable, over 64 KiB, or claims an impossible rejection count |
| `not_sent` | the name did not resolve, the connection or TLS failed, or the body did not all leave |

Timeouts, resets, 429, 502, 503, 504 and connection failures are retried,
with exponential backoff from 0.5 s to 8 s and full jitter, within 5
attempts or 30 s per batch, honoring `Retry-After`. A `Retry-After` beyond
that budget ends the batch at once. Every span offered then ends in exactly
one disposition:

| Disposition | Meaning |
| --- | --- |
| `exported` | confirmed stored, or written whole to the file |
| `rejected` | the collector confirmed it rejected them |
| `refused{status_class}` | definitely not taken: `http_4xx`, `throttled` or `redirect` |
| `dropped{reason}` | never sent: the queue (2,048 spans or 8 MiB) was full, the exporter had closed or was shutting down, it could not be encoded, or the destination never answered (`connect_refused`, `connect_timeout`, `dns`, `tls`, `send_failed`, `file_full`, `file_error`, `file_disabled`) |
| `unknown{reason}` | sent, but the collector may or may not have stored them: `timeout_after_send`, `reset_after_send`, `http_5xx`, `unreadable_response`, `nonconformant_response`, `shutdown_in_flight`, `file_partial`, or `rejected_after_ambiguous` / `refused_after_ambiguous`, when an earlier attempt of the same batch was ambiguous |

At every instant, `offered` = `exported` + `rejected` + `refused` +
`dropped` + `unknown` + `queued` + `in_flight`, and when the run ends both
`queued` and `in_flight` are 0: the end of the capture waits at most
`--otlp-flush-timeout`, then settles whatever is left (`dropped{shutdown}`,
or `unknown{shutdown_in_flight}` for a batch being sent). An answer that
arrives later is counted under `late_results` and changes nothing.

A collector can store a batch more than once when its answer is lost and the
batch is sent again. `max_extra_copies` bounds the extra spans that can
cause. Compared with what a collector really received, the counts always
satisfy:
- `exported` ≤ unique spans received ≤ `exported` + `unknown`;
- raw spans received − unique spans received ≤ `max_extra_copies`.

After 3 batches in a row end without a confirmed attempt, the destination is
marked down. While it is down, new spans queue up to the bounds, and the
batch at the head is retried once per `--otlp-probe-interval`; one
confirmed attempt marks it up again. Retries within a batch back off
exponentially, but never wait longer than the probe interval, unless the
collector asks for longer with `Retry-After`: so a collector that comes
back is used within one probe interval, whether or not it was marked down.
The `export.otlp` record lists the
transitions (`first_failure`, `breaker_open`, `first_success`,
`breaker_closed`, at most 64), each with its time and reason.

The endpoint's name is resolved when the capture starts, and again after 3
attempts in a row reach none of its addresses, so a collector whose address
changes, such as a recreated service, is found again. An attempt waits at
most 2 s for a resolution, which keeps running in its own thread.

A proxy set in `HTTP_PROXY` or `HTTPS_PROXY` is not used: spans go straight
to the endpoint, and a non-loopback endpoint with a proxy variable set
prints one warning.

### The span exporter's health

The `export.otlp` record in the artifact holds the destination (its origin,
or the file), the encoding, the header names (never their values), the
resource keys and the keys left out, the sampling ratio, the trace-context
policy and declared server sampler, the content items, and the final
summary: the dispositions above, `sampled_out`, `late_results`, attempts by
kind and category, retries, batches, encoded and sent bytes, the first
error's kind, category and status, `warnings` (confirmations that rejected
nothing but carried a message), the transitions, any stage stuck (a
`resolve` running over 2 s, or a file `write` over 5 s), how long the flush
at the end took, and the worker's CPU time. A collector's own message is kept, as
`collector_message`, only with `--export-content errors`: it can echo what
was sent, so it is scrubbed and cut to 256 bytes first.

With Prometheus on too, the same figures are metrics:
`stormlog_export_spans_{offered,exported,rejected,sampled_out,max_extra_copies}_total`,
`stormlog_export_spans_refused_total{status_class}`,
`stormlog_export_spans_dropped_total{reason}`,
`stormlog_export_spans_unknown_total{reason}`,
`stormlog_export_in_flight_spans`, `stormlog_export_queue_spans`,
`stormlog_export_queue_bytes`, `stormlog_export_queue_capacity_spans`,
`stormlog_export_queue_capacity_bytes`,
`stormlog_export_queue_high_water_spans`, `stormlog_export_stalled{stage}`,
`stormlog_export_requests_total{outcome}`, `stormlog_export_retries_total`,
`stormlog_export_destination_up`,
`stormlog_export_last_success_timestamp_seconds`,
`stormlog_export_sent_bytes_total` and
`stormlog_export_late_results_total{outcome}`.

## Collector health

`stormlog infer collect-server` takes the same Prometheus flags, to report its
own health while it runs:

```bash
stormlog infer collect-server --run-id "$RUN_ID" --pid 2600 \
  --device-uuid "$GPU_UUID_0" --group-id tp --rank 0 --world-size 2 \
  --output artifacts/rank0.jsonl \
  --prometheus-textfile-dir /var/lib/node_exporter/textfile \
  --prometheus-slot collector-rank0
```

It exports the collector's health, never its measurements: the memory
values stay in its output, where the analysis joins them to the run, and a
DCGM or node exporter already reports device and process memory.

| Metric | Kind | Labels | Meaning |
| --- | --- | --- | --- |
| `stormlog_collector_info` | gauge | run_id, host, boot_id, pid, process_start_ns, device_uuid, gpu_instance_id, replica_id, group_id, rank, world_size, version | 1, labelled with the identity the collector confirmed. A part that is not known (no GPU, no MIG instance, no group) is empty, never guessed |
| `stormlog_collector_running` | gauge | — | 1 while it polls, 0 in the final values |
| `stormlog_collector_start_time_seconds` | gauge | — | When it started, in Unix seconds |
| `stormlog_collector_polls_total` | counter | — | Polls written to its output |
| `stormlog_collector_last_poll_timestamp_seconds` | gauge | — | When the last poll was taken |
| `stormlog_collector_samples_total` | counter | metric, state | Samples by what was read (`process_rss_bytes`, `device_memory_used_bytes`, and so on) and whether it could be: `valid`, `missing`, `stale` or `invalid` |
| `stormlog_collector_stops_total` | counter | reason | 1 for why it stopped, in the final values: `duration_elapsed`, `stop_requested` (Ctrl+C or SIGTERM), `server_process_ended`, `gpu_identity_changed`, or `error` |

The identity is known once the collector has found its process and GPU, so
the series and the budget are fixed then, before anything is collected. A
budget overrun, a held slot or a bad address exits 2 with no output written.
The collector closes the export in its stop path, so the final textfile
holds the stop reason, and `--prometheus-linger` keeps `/metrics` up after
it. Its exit codes are unchanged: 0, or 3 when the GPU identity changed.
Span export and trace context do not apply to it.

## Trace context

`--trace-context` sends a W3C `traceparent` header with each request, beside
its `X-Request-Id`, so a server that traces records its span as a child of
the request's own trace. It is off by default, and no other flag turns it
on: runs compared with and without it would otherwise differ in what the
server receives.

```bash
stormlog infer profile ... \
  --trace-context preserve-engine --server-trace-sampler parentbased_always_on
```

Each request sent gets a random trace ID and span ID, recorded on its
`infer.request` record as `trace_id` and `span_id`. A request never sent,
such as one dropped at the in-flight limit, has neither. The session
record's `config.trace_context` holds the policy, the ratio and the
declared server sampler.

vLLM 0.30 uses the header only while its own tracing is on
(`--otlp-traces-endpoint`); otherwise it logs one warning and ignores it.
Its tracer has no sampler of its own, so the server's `OTEL_TRACES_SAMPLER`
decides what the sampled flag does:

| Server sampler | `off` (no header) | `preserve-engine` (always sampled) | `follow-sampling` |
| --- | --- | --- | --- |
| `parentbased_always_on` (the default) | every request, in separate traces | every request, in Stormlog's traces | only the requests Stormlog sampled |
| `parentbased_traceidratio:p` | a fraction p | every Stormlog request: its volume rises from p to all of them | Stormlog's ratio instead of p |
| `always_on`, `traceidratio:p` | all, or p | the same; the header sets only the parent | the same; the flag is ignored |
| `always_off` | none | none | none |

- **`preserve-engine`** is the mode to use when propagation is on and the
  server keeps its default sampler: it records what it would have recorded
  without the header, so [vLLM span ingestion](vllm_telemetry.md) keeps
  every span, and every request span Stormlog exports has its server child.
  Under a parent-based ratio sampler it records every Stormlog request
  instead of its ratio; when `--server-trace-sampler` declares such a
  sampler, the run warns so. `--otlp-sample-ratio` has no effect here and is
  refused: every request carries a sampled `traceparent`, and every such
  span is exported.
- **`follow-sampling`** sends Stormlog's own decision at
  `--otlp-sample-ratio`. When `--server-trace-sampler` declares a
  parent-based sampler with a known share, the ratio defaults to that share
  (else 1), and a higher one is refused, since it would raise the server's
  volume; a sampler that is not parent-based ignores the flag, so the ratio
  is left alone. The server then
  drops its spans for every request Stormlog did not sample, including the
  failed and slow ones that Stormlog keeps after the fact.

Stormlog's decision keeps a trace when the lowest 56 bits of its trace ID
are at least (1 − ratio) · 2⁵⁶, the OpenTelemetry ProbabilitySampler's
predicate, so it can be recomputed from the recorded ID. Stormlog sends no
`th` threshold, so it claims no agreement with other samplers downstream.

Stormlog cannot read the server's sampler. `--server-trace-sampler
NAME[:ARG]` records it as you declare it, unverified, after checking NAME
against the OpenTelemetry SDK's `OTEL_TRACES_SAMPLER` names (`always_on`,
`always_off`, `traceidratio`, `parentbased_always_on`,
`parentbased_always_off`, `parentbased_traceidratio`, `jaeger_remote`,
`parentbased_jaeger_remote`, `xray`) and ARG as a ratio from 0 to 1 for the
ratio samplers: a typo would otherwise make two compared runs differ
silently. Anything else exits 2.

## Deployment examples

`examples/observability/` holds configs and scripts for three setups, from
no services at all to a collector in front of a trace store.

**No services.** Write both exports to files. The textfile directory must
already exist, as for any textfile run; the span file's directory is
created:

```bash
mkdir -p artifacts/metrics
stormlog infer profile ... --output artifacts/infer.jsonl \
  --prometheus-textfile-dir artifacts/metrics \
  --otlp-file artifacts/spans.jsonl
```

Each line of the span file is one OTLP JSON export request. To list the
slowest requests:

```bash
jq -r '.resourceSpans[].scopeSpans[].spans[]
       | select(.name == "stormlog.infer.request")
       | [((.endTimeUnixNano|tonumber) - (.startTimeUnixNano|tonumber)) / 1e6,
          (.attributes[] | select(.key == "stormlog.request_id") | .value.stringValue)]
       | @tsv' artifacts/spans.jsonl | sort -rn | head
```

To browse the traces, replay the file into a collector with its
`otlpjsonfile` receiver and send them to Jaeger. `infer analyze
--vllm-spans` reads the file too, but it looks for vLLM's request spans:
it counts Stormlog's as `not_a_request_span`, and is not a viewer for them.

**Local services.** `local_stack.py` runs `otelcol-contrib`, `prometheus` and
`jaeger` from their binaries, each with the config beside it, and stops or
kills them by the pid it started, never by name:

```bash
python -m examples.observability.local_stack start
python -m examples.observability.local_stack status
python -m examples.observability.local_stack stop
```

A binary is found on `PATH`, or named by `STORMLOG_OTELCOL`,
`STORMLOG_PROMETHEUS` or `STORMLOG_JAEGER`; a missing one is skipped.
`prometheus.yml` scrapes vLLM and Stormlog as separate jobs. Jaeger takes OTLP
over gRPC on 4317, leaving 4318 to the collector, and serves its UI on
16686; it keeps port 8888 for its own metrics, since both collector
configs turn theirs off. `docker-compose.yml` runs the same three services,
but has not been run yet and is marked so.

**A collector in front.** `otelcol.yaml` takes vLLM's spans and Stormlog's on
one OTLP/HTTP receiver (127.0.0.1:4318) and feeds two pipelines:

- **`traces/stormlog-analysis`** forwards only vLLM's spans, unchanged and
  unsampled, to the receiver Stormlog runs for its engine-side analysis
  (`--vllm-spans-listen 127.0.0.1:4319`). It selects them by the resource
  attribute `vllm.instrumenting_module_name`, which vLLM's tracer always
  sets, so neither Stormlog's spans nor another service's reach the analysis,
  whatever their `service.name`. Its queue and retries are bounded; a retry
  can deliver a batch twice, which the analysis counts as duplicates and
  keeps once.
- **`traces/backend`** sends everything to Jaeger and to a file, through tail
  sampling that keeps every failed trace and one in ten others. Remove the
  sampler to keep all.

The two pipelines share one receiver, so a refusal in either, such as the
process-wide memory limiter while the backend's tail sampler buffers, is
the receiver's answer: the sender retries, and the other pipeline gets the
batch again. The analysis keeps each span once; Jaeger may show it twice.

```bash
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
OTEL_RESOURCE_ATTRIBUTES=host.name=$(hostname) \
vllm serve MODEL --otlp-traces-endpoint http://127.0.0.1:4318/v1/traces

stormlog infer profile ... --vllm-spans-listen 127.0.0.1:4319 \
  --otlp-endpoint http://127.0.0.1:4318 --trace-context preserve-engine \
  --prometheus-listen 127.0.0.1:9900 --prometheus-linger 30
```

**Outage episodes.** For #221's qualification:
- `otelcol-x1.yaml` writes straight to a file, with no batching and with
  rotation set, so each export is written as it arrives rather than
  buffered for a second. The collector can still answer just before it
  writes, so what it acknowledged in the moment it was killed may be
  missing, and the collector-side bounds checked on this file are
  approximate. Kill and restart it with `local_stack.py kill otelcol` and
  `local_stack.py start --x1 otelcol`, and run Stormlog with
  `--otlp-probe-interval 1`.
- For exact bounds, run the episode against `fake_collector.py` instead.
- `fake_collector.py` stores each export's spans, fsynced, before it
  answers, after `--delay-seconds`, with `--status`, in the request's
  encoding. `--partial-rejected N` keeps all but the last N spans of each
  export, as a partial success, and `--retry-after S` adds that header to
  its 429 and 503 answers, in whole seconds as RFC 9110 has it. Like a collector, it answers 404 off
  `/v1/traces` and 415 for a body that is neither protobuf nor JSON.
  `GET /counts`, or `--count FILE` after it has gone, gives the raw and
  unique spans to check Stormlog's collector-side bounds. Slower than
  Stormlog's 5 s attempt deadline, it gives `unknown{timeout_after_send}`.

These configs were run on an A30 with vLLM 0.30.0 and Qwen2.5-0.5B:
- `otelcol.yaml`, with tail sampling on and off, and `otelcol-x1.yaml` ran
  under otelcol-contrib 0.162.0. The analysis filter passed vLLM's spans
  and nothing else, even with Stormlog's `service.name` set to `vllm` and an
  unrelated service sending. So the request spans Stormlog joined were the
  ones a direct receiver gives.
- `jaeger.yaml` ran under Jaeger 2.21.0, beside the collector, which
  exported to it without an error.
- `prometheus.yml` scraped vLLM and Stormlog under Prometheus 3.15.0, and
  `promtool check config` accepts it.

`docker-compose.yml` has not been run.

## When something goes wrong

| Situation | What happens | Exit code |
| --- | --- | --- |
| The port is in use | One warning; no endpoint; `export.prometheus` says `available: false` with the error; the run goes on | unchanged |
| A non-loopback address | One warning that the endpoint has no authentication | unchanged |
| A slow or stuck scraper | Cut after 10 s; at most 4 at once, a fifth gets 503 | unchanged |
| A request head over 16 KiB | 431, and the connection closed; counted as `bad_request` | unchanged |
| A textfile write fails | The previous file stays; `writes_failed` counts it | unchanged |
| The textfile writer is stuck in I/O at the end | Left to finish, with its lock kept until the process exits | unchanged |
| The exporter's queue is full | The record is dropped from the metrics only and counted; the artifact has it | unchanged |
| The matrix is over the budget, or the slot is held | Refused before anything is sent | 2 |
| Ctrl+C | The exporter stops within 2 s, writes its final textfile and capability record, and skips the linger. A second Ctrl+C while it stops ends its waiting, not its steps: the values still freeze and the final file and record are still written. A Ctrl+C while the record is written takes effect once it is, and a second one at once, so a write stuck on a hung filesystem can still be broken off | 130, as before |
| The run fails after the exporter started | The exporter is stopped the same way, within 2 s, so the final file says the run ended | unchanged |
| The collector is down, slow or refusing | Spans are retried, then counted as `dropped`, `unknown` or `refused`; the run waits at most `--otlp-flush-timeout` at its end | unchanged |
| The span file cannot be opened | One warning; `export.otlp` says `available: false`; every span is `dropped{file_disabled}` | unchanged |
| The span queue is full | The span is dropped and counted; the artifact has the record | unchanged |
| A bad OTLP URL, header or content item, a forbidden `--otlp-resource-attribute-allow`, both `--otlp-endpoint` and `--otlp-file`, or headers that would go in clear text to another host | Refused before anything is sent | 2 |

## Cost

The exporter runs beside the requests it measures:
- **On the event loop**, each record costs a fixed list of field reads and
  one append to a bounded queue. It never takes the metrics lock or does I/O.
- **Chunk gaps** are summarized on the request's own thread, so a long
  response costs the loop nothing extra.
- **Metric updates, rendering and the textfile** run on their own threads,
  and so do span building, batching and delivery. A span's extras (the
  server's error type, consented content) are prepared on the request's own
  thread too.

Under Python's GIL, any other running thread can still delay the event loop
by up to a switch interval (5 ms by default). The design limits how often
that happens, not how long it lasts.

Memory is bounded, by figures the tests measure with `tracemalloc` on the
real parts. With M the largest scrape (the budget's bytes) and S its
samples:

| Holder | Bound |
| --- | --- |
| Metric queue | 8 MiB, counted as the memory each queued record holds, or 65,536 records, whichever comes first. A request record holds about 1.8 to 2.5 KiB, so about 3,400 to 4,800 request records fit: at 1,000 requests/s, a stalled worker drops records after 3 to 5 s |
| Registry | M plus about 350 bytes per sample |
| Renders | at most 3 alive, so 3 M, and 3.25 M at the peak while the next is built |
| Textfile | shares the render it writes; one write at a time |
| Endpoint | 4 connections, each holding at most 16 KiB of request head |
| Span queue | 2,048 spans or 8 MiB, counted as the memory each queued span holds |
| Span batch | its spans encoded, at most 4 MiB; the request built from them; a gzip copy of it while it is sent; a response read up to 64 KiB |

That is at most about 8 MiB + 4.25 M + 350 B × S: about 15 MiB for 20 cases,
and about 93 MiB at the default limits (16 MiB, 50,000 samples). Span
export adds at most about 20 MiB for the queue and the batch being sent.

`scripts/benchmark_export_observe.py` reports the per-record cost on your
machine. Its numbers depend heavily on machine load, and they are not a
claim about serving overhead. Matched exporter-off and exporter-on runs on
real servers are part of #221.
