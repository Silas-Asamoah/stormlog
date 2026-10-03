# vLLM execution hook

The vLLM execution hook records which requests vLLM ran together in each
scheduler step, and marks each step's GPU launches with a profiler range. A
trace imported with `stormlog infer import-trace` can then link GPU work to the
step that launched it, and `stormlog infer import-execution` can link the step
to the requests that shared it.

The hook runs inside the vLLM server. It writes a raw log on the server host and
never writes to the inference artifact itself. The import turns that log into
`infer.iteration`, `infer.membership`, `infer.request` and
`infer.clock_alignment` records (see
[Execution correlation](inference_correlation.md)).

## Enable it

Install Stormlog into the vLLM server's environment and set the output
directory before starting the server:

```bash
pip install stormlog  # into the vLLM environment
STORMLOG_VLLM_HOOK_DIR=/var/tmp/stormlog-vllm vllm serve Qwen/Qwen2.5-7B-Instruct
```

Stormlog registers the hook as a `vllm.general_plugins` entry point. Without
`STORMLOG_VLLM_HOOK_DIR` it does nothing. `VLLM_PLUGINS` can allow or exclude it
like any other plugin.

| Variable | Meaning |
| --- | --- |
| `STORMLOG_VLLM_HOOK_DIR` | Where the raw log is written. Unset: the hook is off. |
| `STORMLOG_VLLM_HOOK_NVTX` | `1` also emits each step's range through NVTX, for Nsight Systems. |
| `STORMLOG_VLLM_HOOK_RETAIN_HOURS` | Epoch directories older than this are deleted at start (default 24). |
| `STORMLOG_VLLM_HOOK_MAX_BYTES` | Disk cap per epoch (default 256 MiB). |

## What is supported

The hook checks vLLM's configured objects when they are built and enables
itself only for this matrix. Anything else is refused: vLLM runs untouched, and
the hook's `hello` record names the reason.

| Supported in this version | Refused |
| --- | --- |
| vLLM 0.30.0 | any other version |
| uniproc and multiproc executors | Ray, external launcher |
| tensor parallelism of any size | pipeline parallelism, data parallelism > 1 |
| `Scheduler` and `AsyncScheduler` | other scheduler classes |
| the V2 GPU model runner (0.30.0's default) and the V1 runner it falls back to | other runners |
| async scheduling on or off | |
| CUDA graphs, chunked prefill | |
| n-gram speculative decoding | draft-model speculation, adaptive verification |
| | KV or encoder-cache connectors, pooling models |

A failure inside the hook is counted and never reaches vLLM: every patched call
runs vLLM's own code exactly once, and its exceptions pass through unchanged.

## Raw log format, version 1

### Layout

```text
$STORMLOG_VLLM_HOOK_DIR/
  <host>-<boot id>/
    <role>-<pid>-<start ns>/        one epoch: one process lifetime
      key                           32 random bytes, mode 0600
      status.json                   overwritten every second
      000000.jsonl                  sealed segment
      000001.jsonl.part             open segment
```

`<role>` is `engine` or `worker`. A uniproc server has one process holding both
roles and writes one epoch per role.

A segment is sealed by renaming `.part` to `.jsonl`. Sealing happens at 8 MiB,
after 60 s, when the process exits, and when a file named `flush` appears in the
epoch directory; the writer deletes it once sealed. Files are created with mode
`0600`. A reader takes sealed segments whole and only complete lines of an open
segment.

### Records

Every line is one JSON object with these common fields:

| Field | Meaning |
| --- | --- |
| `format` | `"stormlog.vllm_hook/1"` |
| `kind` | the record kind, below |
| `epoch` | the epoch directory name, `<role>-<pid>-<start ns>` |
| `seq` | 0, 1, 2, … within the epoch, with no gaps unless records were dropped |

Times are pairs: `*_wall_ns` from `time.time_ns()` and `*_mono_ns` from
`time.monotonic_ns()`, read in the same process.

| Kind | Written by | When |
| --- | --- | --- |
| `hello` | every epoch | first record |
| `alias` | engine | a request is admitted |
| `scheduled` | engine | `Scheduler.schedule` returns |
| `completed` | engine | `Scheduler.update_from_output` returns |
| `terminal` | engine | `Scheduler._free_request` runs |
| `heartbeat` | every epoch | every second |
| `goodbye` | every epoch | the process exits cleanly |

**`hello`**

```json
{"kind": "hello", "role": "engine", "host": "node-7", "boot_id": "…",
 "pid": 2600, "start_ns": 1790000000000000000,
 "vllm_version": "0.30.0", "enabled": true, "refused": null,
 "producer": "vllm:node-7:<boot>:2600:1790000000000000000",
 "config": {"executor": "mp", "scheduler": "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
            "runner": null, "tp": 2, "pp": 1, "dp": 1, "async_scheduling": true,
            "speculative": "ngram", "max_num_batched_tokens": 2048,
            "request_id_randomization": true},
 "clock": {"wall_ns": 1790000000000123456, "mono_ns": 123456789000, "gap_ns": 1200}}
```

`refused` is null when enabled, else a short reason. `producer` names the
engine's iterations and is the producer ID in every range. A worker's `hello`
adds `rank` (`tp`, `pp`, `dp`), `local_rank`, `cuda_ordinal`, `device_uuid`
(`GPU-…`), and `trace_rank_suffix`, the suffix vLLM puts in that worker's
trace file name. The `config.runner` field is filled by workers.

**`alias`**

```json
{"kind": "alias", "internal": "chatcmpl-stormlog-r1-q0-0f3a9c1d",
 "external": "chatcmpl-stormlog-r1-q0", "wall_ns": …, "mono_ns": …}
```

Written from vLLM's input thread, so it may come before or after the request's
first `scheduled` record.

**`scheduled`**

```json
{"kind": "scheduled", "iteration": "41",
 "start_wall_ns": …, "start_mono_ns": …, "end_wall_ns": …, "end_mono_ns": …,
 "total_tokens": 2048, "zero_token": false, "preempted": ["…"],
 "members": [
   {"internal": "…", "sighting": "first", "phase": "context", "scheduled": 2000,
    "computed_before": 0, "prompt_tokens": 4000,
    "prefill_scheduled": 2000, "past_prompt_scheduled": 0,
    "drafts_scheduled": 0, "cached_at_admission": 0,
    "recompute": false, "output_before": 0}]}
```

`iteration` counts the engine's `schedule()` calls from 0. `computed_before` is
the context before this step, read from the scheduler output. `phase` is vLLM's
own classification: `context` for a request new in this output (including one
resumed after preemption) or still in its context phase, else `generation`.
`prefill_scheduled` is the part of the step below the prompt length;
`past_prompt_scheduled` is the rest, which is decoding for a running request and
recomputation for a resumed one. `sighting` is
`first` the first time the hook sees an internal ID and `repeat` after, whatever
vLLM's own field calls it. `cached_at_admission` is set on the first sighting
only.

**`completed`**

```json
{"kind": "completed", "iteration": "41", "wall_ns": …, "mono_ns": …,
 "members": [
   {"internal": "…", "outcome": "kept", "stale": false,
    "sampled": 1, "accepted_drafts": 0, "retained": 1,
    "computed_after": 2000}]}
```

`outcome` is `kept`, `dropped_stale`, `discarded_finished` or `unknown`.
`sampled` is the number of tokens the step sampled for the request, counted
before vLLM trims at a stop string. `retained` is how many of them were kept.
`computed_after` is derived as
`computed_before + scheduled - (drafts_scheduled - accepted_drafts)`, the
rollback applying only when the output is not stale.

**`terminal`**

```json
{"kind": "terminal", "internal": "…", "status": "FINISHED_STOPPED",
 "finish_reason": "stop", "output_tokens": 128, "wall_ns": …, "mono_ns": …}
```

**`heartbeat`** and `status.json`

```json
{"kind": "heartbeat", "wall_ns": …, "mono_ns": …, "last_seq": 1234,
 "dropped": {"scheduled": 0, "completed": 0, "alias": 0, "terminal": 0},
 "errors": 0, "bytes": 1048576, "capped": false}
```

A worker's heartbeat adds `range_misses` (serving calls that ran without an
iteration range), `startup_unranged` (warm-up, dummy and CUDA-graph capture
calls before the first serving step, which never have one), and
`pending_samples`.

`status.json` holds the latest heartbeat's fields and is still updated after
the disk cap stops record writing, so loss stays visible.

**`goodbye`** has `wall_ns`, `mono_ns` and `last_seq`.

## Iteration ranges

Each worker wraps the model runner's `execute_model` call for a serving step,
and the `sample_tokens` call that completes it, in a profiler range named
`stormlog.iteration/<producer>/<iteration>` (see
[Importing profiler traces](inference_correlation.md#importing-profiler-traces)).
Internal dummy runs, CUDA-graph capture and warm-up get no range; they run
before the first serving step.
