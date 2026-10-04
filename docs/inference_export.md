[← Back to main docs](index.md)

# Exporting inference metrics

`stormlog infer profile` can expose what it measures to Prometheus while the
run lasts. The JSONL artifact is still written in full, and stays the
record the analysis reads. The export is optional and off unless you ask
for it. Standard `OTEL_*` environment variables never turn it on.

**What is exported:** what Stormlog itself measured or decided, such as
client-observed latencies, request outcomes, token counts, scrape health
and the exporter's own health.

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
- a slot another live writer holds;
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
| `stormlog_infer_tokens_total` | counter | model, server, case, phase, direction, source | Prompt and output tokens of completed requests. `source` is where each count came from: `server_usage`, `tiktoken`, `transformers`, `estimated` or `unknown` | `vllm:prompt_tokens_total` and `vllm:generation_tokens_total` (every client) |
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

  While every reason is 0, and no record failed to be read
  (`stormlog_exporter_internal_errors_total{entry="observe"}`), the exported
  totals are exact: the capability record's `exact` says so. They are
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

## Cost

The exporter runs beside the requests it measures:
- **On the event loop**, each record costs a fixed list of field reads and
  one append to a bounded queue. It never takes the metrics lock or does I/O.
- **Chunk gaps** are summarized on the request's own thread, so a long
  response costs the loop nothing extra.
- **Metric updates, rendering and the textfile** run on their own threads.

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

That is at most about 8 MiB + 4.25 M + 350 B × S: about 15 MiB for 20 cases,
and about 93 MiB at the default limits (16 MiB, 50,000 samples).

`scripts/benchmark_export_observe.py` reports the per-record cost on your
machine. Its numbers depend heavily on machine load, and they are not a
claim about serving overhead. Matched exporter-off and exporter-on runs on
real servers are part of #221.
