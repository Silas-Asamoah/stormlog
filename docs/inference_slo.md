[← Back to docs](index.md)

# Inference SLOs and goodput

An SLO policy declares the latency each request should meet. Stormlog keeps
two kinds of latency apart and never relabels one as the other:

- **client** criteria are measured by Stormlog's own client, from the send (or
  the intended arrival) to the first content delta or the end of the stream;
- **server** criteria are what vLLM reports about itself, per request through
  its span attributes, or in aggregate through its Prometheus histograms.

A criterion is named `boundary.metric`, such as `client.ttft` or
`server.ttft`. There is no client inter-token latency: a streamed chunk can
carry several tokens, so chunk gaps are reported as chunk statistics and never
as ITL.

This page describes the policy format and the Python API in
`stormlog.infer.slo`.

## Criteria

| Key | Measures | Per request, from | Aggregate, from |
| --- | --- | --- | --- |
| `client.ttft` | Send to the first non-empty content delta | `infer.request` `ttft_ms` | none |
| `client.ttft_from_intended` | Intended arrival to the first content delta, which counts any wait before the send | `ttft_ms + dispatch_lag_ms` | none |
| `client.e2e` | Send to the end of the response, after `[DONE]` is parsed | `e2e_latency_ms` | none |
| `client.e2e_from_intended` | Intended arrival to the end of the response | `ended_at_ns − intended_at_ns` | none |
| `client.tpot` | `(e2e − ttft) / (output_tokens − 1)`, the formula vLLM's benchmark uses. It needs the output token count the server reported, and it is a per-request mean, not ITL | `infer.request` with `output_token_source: server_usage` | none |
| `server.ttft` | vLLM's own time to first token | span `gen_ai.latency.time_to_first_token` | `vllm:time_to_first_token_seconds` |
| `server.e2e` | vLLM's own end-to-end latency | span `gen_ai.latency.e2e` | `vllm:e2e_request_latency_seconds` |
| `server.queue` | vLLM's first wait in the scheduler queue | span `gen_ai.latency.time_in_queue` | `vllm:request_queue_time_seconds` |
| `server.itl` | vLLM's inter-token latency, as vLLM defines it | none: aggregate only | `vllm:inter_token_latency_seconds` |
| `server.tpot` | vLLM's time per output token | none: aggregate only | `vllm:request_time_per_output_token_seconds` |

A request passes a criterion when its value is **at most** `max_ms`, which is
inclusive, as in vLLM's `--goodput`.

## Policy files

A policy is a JSON document with `format: "stormlog.infer.slo"` and
`version: 1`:

```json
{
  "format": "stormlog.infer.slo",
  "version": 1,
  "name": "chat_interactive",
  "criteria": [
    {"metric": "ttft", "boundary": "client", "max_ms": 500},
    {"metric": "tpot", "boundary": "client", "max_ms": 50, "attainment_target": 0.99}
  ],
  "attainment_target": 0.98,
  "population": "offered",
  "unknown_policy": "bounds",
  "interval": {"kind": "measured_window"}
}
```

| Field | Meaning |
| --- | --- |
| `name` | Lowercase letters, digits and underscores, starting with a letter, at most 64 characters. It is safe as a metric label. |
| `criteria` | At least one criterion, each key at most once. `max_ms` is a positive number of milliseconds. |
| `attainment_target` (policy) | Optional. The **joint** target: the share of requests that pass every criterion. |
| `attainment_target` (criterion) | Optional. A **marginal** target for that criterion alone. MLPerf's separate p99 limits on TTFT and TPOT are marginal targets; they are not the same claim as a joint target. |
| `population` | `offered`, the only value in version 1: attainment is a share of every request that was offered, including dropped and failed ones. |
| `unknown_policy` | `bounds`, the only value in version 1: a request whose criteria cannot be judged widens the reported bounds and is never counted as met or missed without saying so. |
| `interval` | `{"kind": "measured_window"}` judges a case's declared measurement interval. `{"kind": "sliding", "seconds": N}` is for an online watcher. |

Unknown fields are refused, so a policy written for a later version is never
read as if it were version 1.

## Flags

A short policy can be written as `KEY:MS` flags. A key without a boundary is
a client criterion:

```text
ttft:500 server.ttft:400 tpot:50
```

Flags have no targets and judge the measured window.

## On the command line

`stormlog infer profile` and `stormlog infer analyze` both take a policy:

```bash
stormlog infer profile ... --slo ttft:500 --slo server.ttft:400
stormlog infer analyze infer.jsonl --slo-file slo.json
```

| Option | Meaning |
| --- | --- |
| `--slo KEY:MS` | One criterion, written as under Flags above. Repeat it for more. |
| `--slo-file FILE` | A policy file. Not together with `--slo`. |

`profile` writes the policy into the artifact as an `infer.slo` record and
judges the run by it in its closing report. `analyze` judges by the policy
its options give, or, without them, by the one the artifact recorded; the
report's `slo.source` says which (`flags`, `file` or `artifact`). With
neither, the report's `slo` is `null` and no case has an `slo` block.

A malformed flag, an unknown criterion, or both options at once exits `2`.
A policy file that is missing, unreadable or invalid exits `5`; `profile`
refuses both before it sends anything.

The report gains a top-level `slo` block, with the policy's `name`, `digest`,
`source` and the `policy` document itself, and each case gains an `slo` block:
the evaluation described under Attainment and goodput below, over the case's
rate interval. The text report prints one line per case:

```text
  SLO interactive: attainment 96.0%-98.5% of 200 offered, goodput 9.60-9.85 req/s, 5 unknown
```

## Python API

```python
from stormlog.infer.slo import load_slo, parse_slo_flags, slo_from_artifact

spec = load_slo("slo.json")
spec = parse_slo_flags(["ttft:500", "server.ttft:400"])
spec = slo_from_artifact(records)   # the artifact's infer.slo record, or None
spec.digest()                       # SHA-256 of the canonical document
```

| Function | Errors |
| --- | --- |
| `load_slo(path)` | `InferInputError` (the CLI exits 5) for a missing, unreadable or invalid file |
| `parse_slo_flags(items)` | `InferUsageError` (exit 2) for a malformed flag, an unknown criterion, `client.itl`, a repeated key or a limit that is not positive |
| `slo_from_artifact(records)` | `InferInputError` (exit 5) for more than one `infer.slo` record, or one that is not a valid policy |

`slo_record(spec, session_id=..., source=...)` builds the `infer.slo` record
an artifact carries: the policy document, its digest, and whether it came from
a file or from flags.

## Judging requests and spans

`evaluate_request(record, spec, span=None)` judges one `infer.request` record.
Client criteria read the record. Server criteria read only `span`, the
attributes of the request's joined vLLM span. A missing span leaves the server
criteria unknown; they are never filled from client values.

Each criterion is `pass`, `fail`, `not_applicable` or `unknown`:
- `not_applicable` is TPOT on a response with at most one output token. It
  counts as passing, as in vLLM's benchmark.
- `unknown` comes with a reason, for example:
  - `no_client_ttft` for a non-streaming response;
  - `no_joined_span`;
  - `output_tokens_not_server_reported`: a local tokenizer's count is
    deterministic but is not the count the server generated;
  - `aggregate_only`, for `server.itl` and `server.tpot`.

The request's outcome is decided by the first rule that applies:

| Rule | Outcome |
| --- | --- |
| The status is anything but `ok`: `dropped`, `unreachable`, `delivery_unknown`, `rejected`, `error`, `timeout`, `cancelled` | `missed` |
| Any criterion fails | `missed` |
| Any criterion is unknown | `unknown` |
| Otherwise | `met` |

For a request that did not succeed, its criteria are still judged on what was
recorded, for diagnosis.

`slo_attained(record, spec, span=None)` returns `True`, `False` or `None` for
`met`, `missed` and `unknown`.

`evaluate_span(span_attributes, spec)` judges an engine-finished vLLM span
against the policy's server criteria; client criteria are `unknown`
(`client_boundary`) on a span. Its outcome is `criteria_met`,
`criteria_missed` or `unknown`, and `service_success` is always `unverified`.
vLLM 0.30.0 spans carry no finish reason, and vLLM emits a span for aborted,
failed and ignored requests too, so a span shows whether the criteria held,
never whether the request succeeded.

`evaluate_criteria(values, spec, boundary=...)` is the shared judge underneath
both. It takes values keyed by criterion, in milliseconds.

`span_attributes_by_request(records, span_paths=())` joins each measured
request to its vLLM span, from the artifact and from span files, by
`X-Request-Id`. It never guesses between deliveries. A request whose span
arrived again with different content, or that has several spans, is left out
and listed in `quarantined` with the reason. `request_span(record, spans)`
returns a request's span and, when it has none, the reason. Pass that reason
to `evaluate_request(..., missing_span_reason=...)` so the unknown server
criteria say why.

## Populations and intervals

`stormlog.infer.populations.case_populations(records)` describes each
measured case's request cohort, and the interval its rates divide by.

### Populations

Every measured request lands in exactly one count, by status:

| Field | Requests |
| --- | --- |
| `offered` | All of the case's measured requests |
| `scheduled` | The arrivals an open loop scheduled (`infer.phase_window`, or recomputed from the seeded workload when the phase was cut short); `null` for a closed loop |
| `dropped` | Never sent |
| `sent` | `offered − dropped` |
| `unreachable` | `connect()` failed; no byte was sent |
| `delivery_unknown` | Sending failed after the connection completed |
| `rejected` | HTTP 429 or 503 |
| `accepted` | `sent − unreachable − delivery_unknown − rejected`: the client's view that nothing refused them |
| `successful`, `failed`, `timed_out`, `cancelled` | Statuses `ok`, `error`, `timeout`, `cancelled` |
| `other` | Any other status, by name; never dropped |
| `censored` | `timed_out + cancelled`: latency known only to exceed what was observed |
| `server_admitted` | Requests the server confirmed it saw: a joined span (conflicting spans included), or an execution hook record with the request's `X-Request-Id`. `null` when the run has no server source; `0` when it ran a span receiver that received nothing |
| `server_evidence_coverage` | `server_admitted / accepted`; `null` without a server source |

### Cohort checks

The cohort is checked for the records a run should have:
- unique `request_id` and `x_request_id`;
- every scheduled arrival's `request_index` exactly once, so a duplicated
  record cannot stand in for a missing one;
- one session;
- every request's times inside its phase, start to drain end;
- the phase's window record, with its start and drain end. A run that
  records its workload records each measured phase's window once the phase
  drains, so a case without one was cut short (`phase_window_missing`), and
  its scheduled arrivals are recomputed from the seeded workload to show how
  many are missing. A window without its bounds is `phase_window_incomplete`;
- without request indexes, one record per scheduled arrival
  (`offered_differs_from_scheduled`).

A failed check sets `cohort_valid: false` and names the problem in `issues`.
Two notes don't invalidate the cohort:
- requests an earlier phase left running (`abandoned_requests_at_start`);
- artifacts written before requests carried an index
  (`request_index_unrecorded`).

### Intervals

| Interval | What it is |
| --- | --- |
| `scheduled_window` | An open loop's own schedule, from the phase start to `scheduled_endpoint_offset_ns`. It doesn't depend on when requests were actually sent. |
| `dispatch_window` | The first send to the last send. |
| `drain` | The window end to the drain end. |
| `measured_span` | The phase start to the drain end. |
| `request_span` | First start to last end over **every** measured request, failed ones included. Only for artifacts older than phase windows. |
| `segment` | A caller-defined slice; see below. |

The **rate** interval, which every rate divides by, depends on the run:

| Run | Rate interval | Numerator |
| --- | --- | --- |
| Open loop | `scheduled_window` | The arrival cohort: requests scheduled in the window, however late they finished |
| Closed loop | `measured_span` | Every measured request |
| Older than phase windows | `request_span` | Every measured request |
| Cut short, or a window without bounds | none (`phase_window_missing`, `phase_window_incomplete`) | |

An old open-loop artifact without a recorded endpoint has it recomputed from
its seeded workload record. A replay without a duration has none, so it has
no rate interval: its rates and goodput are `null`, with `rate_reason:
endpoint_undeclared`. Its attainment is still judged. Dividing by
`measured_span` instead would make an open loop's rate depend on when its
requests finished.

The intervals also give the configured rate and the realized offered rate
(scheduled arrivals per second of the scheduled window).

### Segments

`Segment(name, start_offset_ns, end_offset_ns)` slices a case by offsets from
its measured phase's start. A segment is clipped to the phase.
`membership="arrival"`, the default, counts a request in the segment its
intended arrival (or, without one, its send) falls in. `membership="overlap"`
counts every request whose span meets the segment, for example the requests
in flight during a profiler's stop.

## Attainment and goodput

`stormlog.infer.populations.goodput(requests, spec, interval, spans=None)`
judges a case's offered requests (dropped ones included) against a policy. It
returns a `stormlog.infer.slo_evaluation` v1 result:

| Field | Meaning |
| --- | --- |
| `offered`, `met`, `missed`, `unknown` | Request outcomes, as in `evaluate_request` |
| `attainment_lower` | `met / offered`: unknown outcomes counted as missed |
| `attainment_upper` | `(met + unknown) / offered`: unknown outcomes counted as met |
| `evidence_coverage` | The share of successful requests whose every criterion could be judged |
| `goodput_lower_rps`, `goodput_upper_rps` | Met (and met plus unknown) requests per second of the case's rate interval |
| `goodput_lower_output_tps` | Output tokens of met requests per second |
| `per_criterion` | For each criterion: pass, fail, not-applicable and unknown counts among successful requests, and its own marginal attainment bounds over offered requests |
| `population_declared`, `population_evaluated` | The policy's population and the one judged; both `offered` here. An online watcher that sees only engine-finished spans says so. |

Missing evidence widens the bounds instead of moving a single figure. A lost
span therefore cannot look like an SLO violation, and cannot hide one either.

This is **SLO goodput at the offered load**: good requests per second at one
offered load, as vLLM's benchmark computes it. It is not DistServe's goodput,
which is the highest request rate that still meets an attainment target, and
one run does not establish that capacity.

A case where no request succeeded is still measurable: every request is
missed, and goodput is 0. The evaluation is `unmeasurable` (`null`, never 0),
with a `reason`, when a criterion cannot be judged per request at all:
- it is aggregate-only, such as `server.itl`;
- no successful request could be judged on it, such as client TTFT without
  streaming, or a server criterion without spans.

## Latency quantiles and how much data they need

`stormlog.infer.quantiles` estimates latency quantiles and says whether a
case has enough requests to trust them.

### Sufficiency

A quantile from n observations has a distribution-free confidence interval
made of two order statistics: `[X(j), X(k)]` covers the p-quantile with
probability `B(k−1) − B(j−1)`, where B is the Binomial(n, p) distribution
function (Le Boudec, *Performance Evaluation of Computer and Communication
Systems*, Theorem 2.1). The ranks come from n and p alone, never from the
data.

A quantile is **`sufficient`** when the equal-tailed 95% interval exists
(each tail at most 2.5%) with at least 5 order statistics above its upper
rank. That needs at least:

| Quantile | p50 | p75 | p90 | p95 | p99 | p99.9 |
| --- | --- | --- | --- | --- | --- | --- |
| `sufficient` (symmetric, margin 5) | 20 | 44 | 114 | 230 | 1,164 | 11,665 |
| symmetric, margin 0 | 6 | 13 | 36 | 72 | 368 | 3,688 |
| `n_min_exists` (narrowest, margin 0) | 6 | | | 59 | 299 | 2,995 |

`n_min_exists` is the smallest n for which any interval exists; it matches
Le Boudec's own tables. `sufficient` certifies that statement and nothing
else. It is not a precision in milliseconds, and it assumes independent
requests, which queueing does not give. Every estimate reports:
- the ranks;
- the coverage the ranks achieve;
- the interval's width in milliseconds.

`quantile_minimum_n(p, confidence, margin=..., tails=...)` computes any of
these minimums.

### Two estimands

- **`successful`**: the latency of the requests that succeeded.
- **`failure_penalized`**: every offered request, with each one that did not
  succeed ranked worst. That is a policy penalty, not an observed latency.
  - The p-quantile is the successful values' quantile at level `p / (1 − f)`,
    where `f` is the share of offered requests that did not succeed. That is
    the estimand's definition. Interpolating over a sample padded with
    infinite values differs from it by less than one gap between order
    statistics, and is infinite at level 1 where this is not.
  - With 95 successes and 5 failures, p95 is the largest success.
  - Above level 1 the quantile falls in the failure mass (`penalized: true`)
    and has no value. Only a case with failures is ever penalized.
  - Its `n` is the offered count. When a successful request has no value for
    the metric (no joined span, a non-streamed TTFT, a TPOT without server
    usage), it cannot be ranked, so the estimate has no value and its
    `reason` is `successful_values_missing`. A case with no requests at all
    has reason `no_values`.
  - It gets an observed lower bound only when every failure was a real
    timeout: the quantile had each timeout ended when it was abandoned. A
    timed-out request took at least that long, so the bound holds under the
    same interpolation. A request cancelled after 1 ms is not evidence of a
    long latency.

Both use the same linear interpolation between order statistics as the rest
of the inference report.

What the requests that did not succeed were observed to take is kept apart,
in the `latency` block's `unsuccessful` entry: for each status, the count,
the minimum, median and maximum elapsed milliseconds, and how many have no
elapsed time. That is the time until Stormlog saw each request end, not a
latency it would have had; only for a timeout is it a lower bound on one. A
dropped request was never sent, so it has no elapsed time.

## Related pages

- [Inference Profiling](inference.md)
- [vLLM native telemetry](vllm_telemetry.md)
- [Report and Exit-Code Contract](report_contract.md)
