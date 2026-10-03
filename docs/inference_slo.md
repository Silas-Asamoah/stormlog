[← Back to docs](index.md)

# Inference SLO policies

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

## Related pages

- [Inference Profiling](inference.md)
- [vLLM native telemetry](vllm_telemetry.md)
- [Report and Exit-Code Contract](report_contract.md)
