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
entry; every result records the table version and whether it did. An override
of a key the table lacks, or with a value that is not a finite number (a NaN,
a string, a bool), is refused, as is a window floor (`min_scrapes`) below 2.
The values are provisional until they are read from real runs.

| Key | Value | Meaning |
| --- | --- | --- |
| `queue_saturation.median_waiting_requests` | 1 | requests waiting in the window's median scrape |
| `kv_preemption_pressure.preemptions` | 1 | preemptions counted in the window |
| `prefix_cache_loss.hit_ratio_drop` | 0.2 | fall of the prefix-cache hit ratio below the caller's reference |
| `prefix_cache_loss.min_queried_tokens` | 2048 | tokens the window must have queried the prefix cache for before its ratio decides |
| `host_stall.stall_factor` | 10 | an engine-loop stall is at least this many times the median completion cadence before it |
| `host_stall.stall_floor_ns` | 50 ms | and at least this long |
| `host_stall.no_baseline_floor_ns` | 500 ms | or, with no earlier busy steps to compare with, at least this long |
| `host_stall.baseline_window_ns` | 30 s | the window of earlier busy steps the cadence is taken from |
| `host_stall.min_busy_steps` | 20 | busy steps at least as large as the stall's that window needs |
| `host_stall.matched_bin_min_steps` | 20 | steps of the stall's own work bucket (scheduled tokens within a factor of two) needed to compare it with steps of its size |
| `host_stall.heartbeat_grace_ns` | 2 s | how recently the hook's writer must have been heard from to judge a stall still going on |

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
| `detail` | supporting figures; `scope: engine_global`; the window's successful `scrapes`, `failed_scrapes` inside it, `window_seconds` and `placement` |

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
ratio (`requires_reference`); a reference outside 0-1 is refused. Nor does it
decide over fewer than `min_queried_tokens` queried tokens
(`too_few_queried_tokens`): vLLM counts every prompt token of a new request
as a query, so a quiet second with one short, unseen prompt has a ratio of
0 by construction. Every figure of one signal comes from one engine: behind an
exporter with several, `config.engine` names it, and without it the window is
`engine_required`.
Kinds that metrics alone cannot decide answer `requires_hook`,
`requires_trace` or `requires_client`.

## Engine-loop stalls

`stormlog.infer.diagnosis_loop.engine_loop_gap(records, config)` reads one
engine epoch's raw [execution hook](vllm_execution.md) records (the
`scheduled`, `completed`, `heartbeat` and, from hooks that record them,
`pause` records, in `seq` order) and returns a `SignalValue` for the longest
stretch in which the engine made no progress while it had work it could run.
It needs no import, so an online trigger can run it on the records it tails;
the diagnoser runs the same rules on imported steps. `LoopGapConfig` refuses
a threshold override with a key the table lacks, a value that is not a
finite number, or a loop threshold that is not positive.

Work is *ready* during a stretch when a request ran in the step before it
and in the step after it. The step before a gap between steps is the one
whose completion starts it, without the request finishing there: under
async scheduling a request's second step is scheduled before its first
completes, so the step scheduled just before it may come after an idle
stretch in which nothing was ready. A request prefilled in chunks is ready
between them; a streaming-input request (`resumable`) never makes a gap
ready, since between steps it may be waiting for its client's next input. A
stretch the scheduler spent paused with
`PAUSED_ALL` (from the hook's `pause` records), or one the caller excludes
with `exclude_wall` (for example its own profiler stop), has no ready work;
only the part of a stall such an interval covers is removed, and what
remains on either side is still a stall.
Where a stall sits decides what it can be blamed on:

| `detail["locus"]` | Stretch | `detail["attribution"]` |
| --- | --- | --- |
| `between_steps` | a step's completion to the next `schedule()` entry | `host` |
| `in_schedule` | inside `schedule()` | `host` |
| `within_step` | a step's own time after `schedule()` returned (or after the previous completion, under async scheduling) | `host_or_gpu`: without a GPU trace the two cannot be told apart |

A stall exceeds when it is at least `stall_factor` times the median
completion cadence of the busy steps that completed in the `baseline_window`
before it, and at least `stall_floor_ns`. Steps are compared with steps of
their own size: the cadence is taken over earlier steps whose scheduled
tokens lie within a factor of two of the stall's step when enough share it
(`detail["baseline"]` is `matched`), so a step running a long prefill is not
measured against decode-only steps. Otherwise it is taken over the earlier
busy steps at least as large as the stall's (`unmatched`), when there are
`min_busy_steps` of them, so a step is never measured against smaller ones;
otherwise it is replaced by `no_baseline_floor_ns` (`floor`). A second long
prefill a few seconds after the first therefore meets the floor, not the
decode cadence. Only earlier steps count, so the decision never depends on
what happened after the stall. With `config.now_wall_ns`, a stall still
going on counts from the last completion (`detail["ongoing"]`).

A stall is judged only where the records are known to be whole: between
two heartbeats (the hello counting as one with nothing lost) whose drop
counts and errors did not change, one at or before the stall's start and one
at or after its end. A stall still going on also needs a heartbeat since it
began, the last within `heartbeat_grace_ns` (2 s, about two of the writer's
one-second beats) of the evaluation time. A capped or killed writer stops
writing records and heartbeats alike, so the engine running on unrecorded
looks like a stall with no heartbeat around it. A stall over its limit
outside that coverage gives no verdict (`hook_coverage_unknown`) instead of
exceeding; `detail["covered"]` says which the reported stall was.

The records also give no verdict (`sufficient` is False) when they come from
two epochs (`epoch_changed`), skip a `seq` or show drop counts (of any kind,
oversized records included) rising between heartbeats
(`hook_records_dropped`), show write errors rising between heartbeats
(`hook_writer_errors`), hold no completed step (`too_few_steps`) or are
absent (`requires_hook`). Write errors also count failed seals and
`status.json` writes, which lose nothing, so this abstains more than it
must. Only the epoch's `status.json`, passed as `config.status`, says the
writer is capped (`hook_capped`): no heartbeat ever does.
`detail["pause_capability"]` says whether the hook records pauses. Without
it a pause of every running request (vLLM's `PAUSED_ALL`, as for an RL
weight sync) looks like a stall with ready work, so a stall over its limit is
no verdict (`pause_state_unknown`). A pause can only remove stalls, so
records with no stall over its limit still say none exceeded.
