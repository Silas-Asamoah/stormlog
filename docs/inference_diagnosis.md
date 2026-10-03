[← Back to main docs](index.md)

# Inference diagnosis

Stormlog is building a diagnoser that explains a slow request or a slow
window of an inference run from the evidence the run captured: client
latency, vLLM's own metrics and spans, the scheduler steps of the
[execution hook](vllm_execution.md), and imported GPU traces. This page
documents the parts that exist today: the vocabulary every diagnosis uses, the
threshold table, and the cheap signals an online trigger evaluates over a
window of `/metrics` scrapes.

## Finding kinds

Every finding names one kind, located at one component, with one cause. The
vocabulary is closed (`stormlog.infer.diagnosis_vocabulary`): a new kind is a
change to the diagnosis payload's schema.

| Kind | Component | What it means |
| --- | --- | --- |
| `queue_saturation` | `scheduler` | Requests wait to be scheduled because the engine is at capacity |
| `kv_preemption_pressure` | `kv_cache` | Running requests are preempted because KV blocks ran out, and are recomputed |
| `prefix_cache_loss` | `prefix_cache` | Requests that should reuse a cached prefix find less of it cached |
| `mixed_prefill_interference` | `scheduler` | Decode steps slow down because they share steps with other requests' prefill |
| `host_stall` | `engine_core`, `worker` or `api_server` | The process that should make progress does not: the engine loop, a worker's launches, or the API server |
| `rank_delay` | `worker` | One tensor-parallel rank arrives late, and the others wait in the collective |
| `transfer_degradation` | `interconnect` | Collective or copy time grows on every rank at the same work |
| `capture_pause` | `profiler` | A stall caused by Stormlog's own profiler window |
| `client_admission` | `client` | Stormlog's client held requests back (`--max-in-flight`) |
| `load_increase`, `longer_inputs`, `longer_outputs`, `prefix_sharing_drop` | `workload` | The workload asked for more; always reported at `info` |

A cause is one of `fault`, `workload_change`, `instrumentation` and
`undetermined`.

## Thresholds

Online triggers and the diagnoser read thresholds from one versioned table,
`stormlog.infer.diagnosis_thresholds` (version `diagnosis_thresholds_v1`), so
they cannot disagree about what a threshold is. A caller may override an
entry; every result records the table version and whether it did. The values
are provisional until they are read from real runs.

| Key | Value | Meaning |
| --- | --- | --- |
| `queue_saturation.median_waiting_requests` | 1 | requests waiting in the window's median scrape |
| `kv_preemption_pressure.preemptions` | 1 | preemptions counted in the window |
| `prefix_cache_loss.hit_ratio_drop` | 0.2 | fall of the prefix-cache hit ratio below the caller's reference |

## Online signals

`stormlog.infer.diagnosis_signals.evaluate_signal(kind, scrapes, config)`
evaluates one kind over a window of consecutive scrapes the caller chose,
using the [scrape windows](vllm_telemetry.md#windows-of-scrapes) rules. It
returns a `SignalValue`:

| Field | Meaning |
| --- | --- |
| `value` | the kind's measure over the window, or `None` |
| `sufficient` | whether the window had enough data to decide |
| `reason` | the first reason it did not; all of them are in `detail["reasons"]` |
| `exceeds` | the verdict against the threshold; `None` whenever `sufficient` is False |
| `threshold`, `thresholds_version`, `threshold_overridden` | which threshold decided |
| `detail` | supporting figures, and `scope: engine_global` |

| Kind | `value` | Supporting detail |
| --- | --- | --- |
| `queue_saturation` | median of `vllm:num_requests_waiting` | its maximum; the median per waiting reason (`capacity`, `deferred`); bounds on the window's p90 queue time |
| `kv_preemption_pressure` | increase of `vllm:num_preemptions_total` | the preemption rate; the maximum KV usage |
| `prefix_cache_loss` | prefix-cache hits per queried token | the hit and query counts; the drop below `config.reference` |

`/metrics` describes everything an engine served, from every client, so a
signal over its threshold means a mechanism is *suspected* in the engine's
traffic; it does not say whose requests it hurt. A prefix-cache signal in
particular cannot tell another client's prompts lowering the ratio from the
cache losing a victim's prefixes, and it gives no verdict without a reference
ratio (`requires_reference`); a reference outside 0-1 is refused. Kinds that metrics alone cannot decide answer
`requires_hook`, `requires_trace` or `requires_client`.
