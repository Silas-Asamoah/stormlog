# vLLM execution hook

The vLLM execution hook records which requests vLLM ran together in each
scheduler step, and marks each step's GPU launches with a profiler range. A
trace imported with `stormlog infer import-trace` can then link GPU work to the
step that launched it, and `stormlog infer import-execution` can link the step
to the requests that shared it.

The hook runs inside the vLLM server. It writes a raw log on the server host and
never writes to the inference artifact itself. The import turns that log into
`infer.iteration`, `infer.membership`, `infer.request` and
`infer.clock_alignment` records, with `infer.stage` records for preemptions,
cache resets and pauses (see
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

By default vLLM 0.30.0 stops its engine process with SIGKILL as soon as the
server shuts down (`--shutdown-timeout` defaults to 0). The hook then cannot
finish its log: each epoch ends without a `goodbye` record, with its last
segment still a `.part` file, and records not yet written are lost. Records
already written are kept, and a reader takes the complete lines of a `.part`
segment. For a clean end, serve with `--shutdown-timeout 5` or more, or create
the `flush` file in each epoch directory (see below) and wait for it to
disappear before stopping the server.

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

The hook never makes vLLM wait on a disk. Records go into a queue that a
background thread writes out. Each record's fields are serialized once, when the
record is queued, and the queue holds at most 20,000 records and 32 MiB of that
JSON, counted at its exact size until the record is written. Each queued record
takes about 113 bytes of memory beyond its JSON, so the queue holds at most
about 34 MiB. A record that does not fit is dropped and counted, and so is a
single record over 4 MiB. Two kinds of record are dropped before they are
serialized, so they cost vLLM no encoding: any record while the queue already
holds 20,000 records or the disk cap has stopped record writing, and a record
whose request IDs alone, which clients choose, already pass 4 MiB. The thread
checks its heartbeat, flush and sealing deadlines after every record, so a
backlog delays them by one write at most.

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
epoch directory; the writer deletes `flush` only after a seal succeeds. A seal
that fails is counted as an error, and retried; until then the segment stays a
`.part` file and later records are appended to it. A process exit includes a
forked multiprocessing worker's, which skips `atexit`. Each line is written
whole or not at all: a failed write is cut back off the file and counted as an
error and a dropped record. Files are created with mode `0600`. A reader takes
sealed segments whole and only complete lines of a `.part` segment.

### Records

Every line is one JSON object with these common fields:

| Field | Meaning |
| --- | --- |
| `format` | `"stormlog.vllm_hook/1"` |
| `kind` | the record kind, below |
| `epoch` | the epoch directory name, `<role>-<pid>-<start ns>` |
| `seq` | 0, 1, 2, … within the epoch, with no gaps; a dropped record takes no number |

Times are bracketed reads in the writing process: `*_wall_ns` from
`time.time_ns()`, then `*_mono_ns` from `time.monotonic_ns()`, then
`*_wall_after_ns` from `time.time_ns()` again. The wall time at the
monotonic read lies between `*_wall_ns` and `*_wall_after_ns`, so the pair
is known to within their difference. The hello's `clock` is one such read,
also giving that difference as `gap_ns`. Logs from before
`*_wall_after_ns` have it only in the hello, and their other pairs have no
stated bound.

| Kind | Written by | When |
| --- | --- | --- |
| `hello` | every epoch | first record |
| `alias` | engine | a request is admitted |
| `enqueued` | engine | a request enters the scheduler |
| `scheduled` | engine | `Scheduler.schedule` returns |
| `completed` | engine | `Scheduler.update_from_output` returns |
| `terminal` | engine | `Scheduler._free_request` runs |
| `pause` | engine | `Scheduler.set_pause_state` returns |
| `cache_reset` | engine | `Scheduler.reset_prefix_cache` returns or raises |
| `heartbeat` | every epoch | every second |
| `goodbye` | every epoch | the process exits cleanly |

**`hello`**

```json
{"kind": "hello", "role": "engine", "host": "node-7", "boot_id": "…",
 "pid": 2600, "start_ns": 1790000000000000000,
 "process_start_ns": 1789999990120000000, "process_start_ticks": 81234567,
 "parent_pid": 2512, "parent_process_start_ticks": 81230011,
 "parent_process_start_ns": 1789999944560000000,
 "vllm_version": "0.30.0", "enabled": true, "refused": null,
 "producer": "vllm:node-7:<boot>:2600:1790000000000000000",
 "observes": ["cache_reset", "enqueued", "pause"],
 "config": {"executor": "mp", "scheduler": "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
            "runner": null, "tp": 2, "pp": 1, "dp": 1, "async_scheduling": true,
            "speculative": "ngram", "max_num_batched_tokens": 2048,
            "request_id_randomization": true,
            "max_num_seqs": 256, "num_gpu_blocks": 9000, "kv_cache_groups": 1,
            "block_size": 16, "cudagraph_mode": "FULL_AND_PIECEWISE",
            "gpu_memory_utilization": 0.9, "enable_cumem_allocator": false,
            "enable_sleep_mode": false,
            "profiler": {"profiler": "torch", "torch_profiler_dir": "/traces",
                         "torch_profiler_with_stack": false,
                         "torch_profiler_dump_cuda_time_total": false,
                         "ignore_frontend": true, "max_iterations": 40,
                         "delay_iterations": 0, "warmup_iterations": 0,
                         "active_iterations": 5, "wait_iterations": 0}},
 "clock": {"wall_ns": 1790000000000123456, "mono_ns": 123456789000,
           "wall_after_ns": 1790000000000124656, "gap_ns": 1200}}
```

`start_ns` is when the hook's writer started. The process fields name the
process across pid reuse: `process_start_ticks` is field 22 of
`/proc/<pid>/stat` (clock ticks after boot, exact within one boot), and
`process_start_ns` is the wall-clock start as psutil's `create_time` gives
it, the value Stormlog's server collector records. `parent_pid` and the two
`parent_process_*` fields say the same of the parent process; a multiproc
worker's parent is the engine core. Each is null where it cannot be read.

`observes` lists the optional record kinds this epoch's hook writes, of
those described below; only kinds the installed vLLM can produce are
listed, and the list is empty when the hook is refused and on a worker. A
listed kind's absence over an interval means none happened only when no
record was lost across it: `seq` has no gap, and heartbeats on either side
show every `dropped` count and `errors` unchanged and the writer not
`capped`. For a kind not listed, absence says nothing.

`config` records vLLM's settings as JSON values (an enum by its name), each
null when vLLM does not have it. `max_num_seqs`, `num_gpu_blocks`,
`kv_cache_groups` (how many), `block_size`, `cudagraph_mode`,
`gpu_memory_utilization`, `enable_cumem_allocator` and `enable_sleep_mode`
describe capacity and memory; an engine's are what its scheduler was built
with, after vLLM sized the KV cache, while a worker records the configured
values before that. `profiler` holds ten of vLLM's profiler settings, which
decide how long a profiler stop pauses the server and whether a window stops
by itself, or is null without a profiler configuration.

`refused` is null when enabled, else a short reason.
`config.request_id_randomization` is false when vLLM was told not to add a random
suffix to request IDs, so each internal ID equals its external ID, and null when
vLLM's setting could not be read. `producer` names the
engine's iterations and is the producer ID in every range. A worker's `hello`
adds `rank` (`tp`, `pp`, `dp`), `local_rank`, `cuda_ordinal`, `device_uuid`
(`GPU-…`), and `trace_rank_suffix`, the suffix vLLM puts in that worker's
trace file name. The `config.runner` field is filled by workers.

**`alias`**

```json
{"kind": "alias", "internal": "chatcmpl-stormlog-r1-q0-0f3a9c1d",
 "external": "chatcmpl-stormlog-r1-q0",
 "wall_ns": …, "mono_ns": …, "wall_after_ns": …}
```

Written from vLLM's input thread, so it may come before or after the request's
first `scheduled` record.

**`enqueued`**

```json
{"kind": "enqueued", "internal": "chatcmpl-stormlog-r1-q0-0f3a9c1d",
 "structured_output": false, "resumable": false,
 "wall_ns": …, "mono_ns": …, "wall_after_ns": …}
```

Stamped just before `Scheduler.add_request` puts the request in the waiting
queue, on the engine's own thread; the wait from here to the request's first
`scheduled` record is spent in the scheduler, while the time from `alias`
to here is spent getting into the engine. One per internal ID while it is
live: a streaming-input request's later inputs reuse its ID and write none.
`structured_output` says whether the request is constrained by a grammar,
which can hold it waiting while others run, and `resumable` whether it is a
streaming-input request; each is null when vLLM's request does not say. A
call that raises is not recorded. Listed in `observes` as `enqueued`.

**`scheduled`**

```json
{"kind": "scheduled", "iteration": "41",
 "start_wall_ns": …, "start_mono_ns": …, "start_wall_after_ns": …,
 "end_wall_ns": …, "end_mono_ns": …, "end_wall_after_ns": …,
 "total_tokens": 2048, "zero_token": false, "preempted": ["…"],
 "pause_state": "UNPAUSED",
 "members": [
   {"internal": "…", "sighting": "first", "phase": "context", "scheduled": 2000,
    "computed_before": 0, "prompt_tokens": 4000,
    "prefill_scheduled": 2000, "past_prompt_scheduled": 0,
    "drafts_scheduled": 0, "cached_at_admission": 0,
    "recompute": false, "output_before": 0, "resumable": false}]}
```

`iteration` counts the engine's `schedule()` calls from 0. `pause_state` is
the scheduler's pause state when the call returned (see `pause` below). `computed_before` is
the context before this step, read from the scheduler output. `phase` is vLLM's
own classification: `context` for a request new in this output (including one
resumed after preemption) or still in its context phase, else `generation`.
`prefill_scheduled` is the part of the step below the prompt length;
`past_prompt_scheduled` is the rest, which is decoding for a running request and
recomputation for a resumed one. `prompt_tokens` is read from the live request:
a `resumable` (streaming-input) request's prompt grows with each new input, and
each turn's prefill is measured against the prompt of that turn. `sighting` is
`first` the first time the hook sees an internal ID and `repeat` after, whatever
vLLM's own field calls it. `cached_at_admission` is set on the first sighting
only.

**`completed`**

```json
{"kind": "completed", "iteration": "41",
 "wall_ns": …, "mono_ns": …, "wall_after_ns": …,
 "members": [
   {"internal": "…", "outcome": "kept", "stale": false,
    "sampled": 1, "accepted_drafts": 0, "retained": 1,
    "finish_reason": null, "computed_after": 2000}]}
```

`outcome` is `kept`, `dropped_stale`, `discarded_finished` or `unknown`.
`sampled` is the number of tokens the step sampled for the request, counted
before vLLM trims at a stop string. `retained` is how many tokens vLLM's own
output for this step carried for the request, and `finish_reason` is the reason
that output gave, or null if the request did not finish. Both are read from the
output `update_from_output` returns, so they stay right when vLLM trims at a
stop, frees the request inside the update, or resets a streaming-input
request's counters. They are null unless the outcome is `kept`.
`computed_after` is derived as
`computed_before + scheduled - (drafts_scheduled - accepted_drafts)`, the
rollback applying only when the output is not stale.

When `update_from_output` raises, the record has `"update_failed": true`, every
member's `outcome` is `unknown`, and its other fields are null: vLLM's
exception passes through, and the step's fate was not seen.

**`terminal`**

```json
{"kind": "terminal", "internal": "…", "status": "FINISHED_STOPPED",
 "finish_reason": "stop", "output_tokens": 128,
 "wall_ns": …, "mono_ns": …, "wall_after_ns": …}
```

**`pause`**

```json
{"kind": "pause", "from": "UNPAUSED", "to": "PAUSED_ALL",
 "wall_ns": …, "mono_ns": …, "wall_after_ns": …}
```

One per call to `set_pause_state`, vLLM's single place for changing the
scheduler's pause state, stamped after the call returns; `from` and `to` are
the states before and after (`UNPAUSED`, `PAUSED_NEW` or `PAUSED_ALL`).
`PAUSED_ALL` stops the engine from stepping, so no `scheduled` record shows
it; `PAUSED_NEW` holds new requests while running ones keep stepping. A call
that raises is not recorded. Listed in `observes` as `pause`.

**`cache_reset`**

```json
{"kind": "cache_reset", "reset_running_requests": true, "reset_connector": false,
 "running": ["…", "…"], "succeeded": true, "raised": false,
 "start_wall_ns": …, "start_mono_ns": …, "start_wall_after_ns": …,
 "end_wall_ns": …, "end_mono_ns": …, "end_wall_after_ns": …}
```

One per call to `reset_prefix_cache`, which empties vLLM's prefix cache.
`running` lists the internal IDs running when the call started. With
`reset_running_requests`, vLLM preempts every one of them before it resets
the cache, so they are preempted even when the reset then fails; without it,
the reset succeeds only when no running request holds cache blocks.
`succeeded` is the call's return value, or null when it raised (`raised`
true; vLLM's exception passes through). The start stamp is read before the
call and the end stamp after it. Listed in `observes` as `cache_reset`.

**`heartbeat`** and `status.json`

```json
{"kind": "heartbeat", "wall_ns": …, "mono_ns": …, "wall_after_ns": …,
 "last_seq": 1234,
 "dropped": {"alias": 3, "alias_oversized": 1},
 "errors": 0, "bytes": 1048576, "capped": false, "queued": 0}
```

`dropped` counts dropped records by kind, and `<kind>_oversized` counts single
records over 4 MiB. A record dropped unserialized because the queue was full or
record writing had stopped counts under its kind, whatever its size. `queued`
is the number of records waiting to be written.

A worker's heartbeat adds `range_misses` (serving calls that ran without an
iteration range), `startup_unranged` (warm-up, dummy and CUDA-graph capture
calls before the first serving step, which never have one), and
`pending_samples`.

`status.json` holds the latest heartbeat's fields and is still updated after
the disk cap stops record writing, so loss stays visible.

**`goodbye`** has `wall_ns`, `mono_ns`, `wall_after_ns` and `last_seq`. A process that is killed,
including by vLLM's default shutdown, writes no `goodbye`.

## Iteration ranges

Each worker wraps the model runner's `execute_model` call for a serving step,
and the `sample_tokens` call that completes it, in a profiler range named
`stormlog.iteration/<producer>/<iteration>` (see
[Importing profiler traces](inference_correlation.md#importing-profiler-traces)).
Internal dummy runs, CUDA-graph capture and warm-up get no range; they run
before the first serving step.

## Import

```bash
stormlog infer import-execution infer.jsonl /var/tmp/stormlog-vllm
stormlog infer import-execution infer.jsonl copied-log/ --server-stopped
stormlog infer profile ... --vllm-execution-dir /var/tmp/stormlog-vllm
```

`import-execution` reduces the raw log into canonical records and appends
them to an artifact that holds an `infer.artifact` record. `infer profile
--vllm-execution-dir` does the same when the run ends: it asks every live
epoch to seal its open segment (a `flush` file), waits up to 10 s for the
writer's heartbeat, indexes the worker hellos, imports the traces with the
GPU UUIDs those hellos name, reduces the execution log, and only then
analyzes the artifact and writes the report. A log that cannot be read or
imported is a warning and a capability record with nothing collected; the
run itself does not fail. Exit codes are in the
[report contract](report_contract.md).

**Final steps only.** A step is reduced once it is final: its `completed`
record arrived, or it never will because the epoch ended or is gone, or a
later step completed, which makes it `incomplete`. A pending step waits for
a later import. Nothing is corrected after it is written: each epoch's
high-water sequence is kept in the engine adapter's capability summary, and
a later import starts from there, so running the command twice adds nothing
twice.

**Liveness.** An epoch has ended when it wrote `goodbye`. Without one, the
import may call it gone only on evidence from the clock that wrote its
stamps: when the import runs on the epoch's own host and boot (the hello's
`host` and `boot_id` equal the importer's own), it compares the epoch's last
heartbeat `mono_ns` with its own monotonic clock, and 30 s of silence means
gone; that clock is immune to wall-clock steps. From any other host, or
when the importer's boot ID is unknown, the two clocks are unrelated, so
liveness is not judged: the epoch is `unknown`, its pending steps wait, its
mark is held back, and the summary says why. For a log copied from a
server that has since stopped, `--server-stopped` says so, and every epoch
without `goodbye` is then gone and its pending steps final. `infer profile`
imports while its server may still be up and needs no such flag.

**Loss coverage.** Each epoch's summary has a `coverage` block saying where
its log is known to be whole: `spans` from one heartbeat to a later one
(their `seq`, `mono_ns` and `wall_ns`) with every record between them read,
every `dropped` count (of any kind, `<kind>_oversized` included) and
`errors` unchanged, and the writer not capped; `observes` is the hello's
list, or null for a hook that does not give one. Every import computes it
from all the heartbeats it read, including ones an earlier import consumed,
under `basis` `heartbeat_counters/1`. A kind the hello observes is known
not to have happened in an interval only when a span covers it.

**Records.** One `infer.iteration` per step on the engine's monotonic clock
(`<host>/<boot id>/monotonic_ns`, `clock_kind` `monotonic`), whose
`elapsed_ns` is the scheduler residence from `schedule()` to the processed
output; one `infer.membership` per (request, attempt, iteration, role),
whose role follows vLLM's `phase` (`context` is `prefill`, `generation` is
`decode` or, with drafts scheduled, `spec_decode`; no phase is `unknown`), with
the step's token counts, outcome and, for the step a request was freed in,
its finish; one `infer.request` per backend execution, holding admission
facts only, among them `enqueued_mono_ns`, `enqueued_wall_ns`,
`enqueued_wall_after_ns` and `structured_output` from its `enqueued` record
(null without one); and one `infer.clock_alignment` per epoch from the hello's
bracketed read: the wall clock at the monotonic read lies between the wall
reads before and after it, so the offset is that bracket's midpoint and the
uncertainty half its width. Its metadata gives `alignment_basis`
(`hello_bracket_midpoint/1`) and the raw `bracket`. A hello the wall clock
stepped back across gives no alignment. Earlier imports wrote the offset at
the bracket's lower end, with no `alignment_basis`; `align_timestamp` reads
such a record as the interval from its offset to its offset plus `gap_ns`,
and an existing record is never rewritten. The raw
log is never registered as an attachment. A step whose `update_from_output`
raised (`update_failed`) keeps its memberships with outcome `unknown`, no
output tokens and nothing read from its counters; the coverage block counts
such steps. A request's `input_tokens` is its prompt at the first sighting;
a `resumable` request's later prompts are on its memberships.

Every record the import writes names its epoch and its `source_seq_max` in
its metadata: the `seq` of the last raw record it was derived from, so an
import whose read had reached that record could have written it. For a step
that is the record that made it final (its `completed` record, the first
completion of a later step when its own output never came, or the last
record of an ended epoch); a membership adds the `terminal` record of a
finish it carries; a request takes its first final step, its alias and its
`enqueued` record; a clock alignment takes the hello, or the goodbye that
bounds it.

**Binding.** A request is the run's when its alias `external` is
`chatcmpl-<x_request_id>` or `cmpl-<x_request_id>-<i>` for an
`x_request_id` the artifact recorded, in an `infer.dispatch` record (at the
send) or an `infer.request` record (at the end), so an import of a run still
under way binds the requests in flight; without an alias, the internal ID's
shape only proposes the same exact match, with vLLM's random suffix
stripped only when the hello says `request_id_randomization` is on (both
forms are tried when it is unknown). Every other ID is foreign, or
unresolved when it has no alias and does not parse. Foreign and unresolved
executions appear only as `HMAC-SHA256(HMAC(epoch key, run_id), id)[:16]`,
stable within a run and different in every other artifact;
`--raw-foreign-ids` records their IDs as vLLM saw them. An epoch whose key
file is missing or unreadable cannot key a pseudonym, so without
`--raw-foreign-ids` other clients' identities from that epoch are withheld:
their memberships and requests are not written, steps that held only their
requests are not kept, the import summary counts what was withheld and
says `withheld` instead of `hmac-sha256-keyed`, and a kept step still counts
them in `withheld_members`. The scheme an epoch was first imported with is
recorded in the import summary and fixed: a later import of that epoch that
would switch between pseudonyms and `--raw-foreign-ids` is refused before
anything is appended, since the same execution would get a second request;
a withheld epoch wrote no such identity, so any later scheme may follow it.
A reused internal
ID (randomization off) is split into one execution per admission, also
across imports.

**Which steps are kept.** Steps an imported GPU activity references, steps
with a run member, and steps that preempted a run request are always kept. A step with only other clients'
requests is kept when the engine's wall clock is the client's (same host
and boot) and the step lies inside a run phase or trace window; otherwise
it is counted in the summary, not written. A phase that has an
`infer.phase_start` record but no `infer.phase_window` yet runs from its
start to the session's terminal `infer.session` record, or with no end while
the session is still running. A step that scheduled no request
(an idle scheduler call) is counted as empty, not written.

**Stages.** Preemptions, cache resets and pause changes are written as
`infer.stage` records on the engine's monotonic clock, each with its
`epoch`, the raw `seq` and its `source_seq_max` in the metadata, and the
bracketing wall reads:

| `name` | One per | Refers to | Dated by |
| --- | --- | --- | --- |
| `engine.preempted` | attempt a step preempted to free memory | the attempt's request (`metadata.attempt`) and that step | the step's `schedule()` call |
| `engine.cache_reset` | `cache_reset` record | the last step written before it | the reset call |
| `engine.preempted_by_reset` | request running at a reset with `reset_running_requests` | its request, and the reset's step if it has one | the reset call |
| `engine.pause_transition` | `pause` record, with `from` and `to` | the last step written before it | its stamp |

vLLM lists a reset's preemptions in the next step's `preempted` along with
the step's own, so of a step's preemptions those a reset since the previous
step made are the reset's; until that next step is read, the import holds
its mark below the reset and reads it again. A reset's preemptions stand
even when the reset failed, since vLLM preempts before it checks.
`engine.preempted` says in `reset_observed` whether the hook records resets
at all: without them, a reset's preemptions look like the step's own. A
request preempted by a reset whose request record a pending step will
write waits for it in the same way; another client's that no written step
ran is counted as `unreferenced`, and one in an epoch without its key as
`withheld`, in the epoch's `stages` counts. A reset or pause with no step
written before it goes in the epoch's `unanchored` list with its `seq`, its
times and its fields. Each stage's ID is fixed by the epoch and the raw
`seq`, so no import writes one twice.

**Device binding for traces.** A worker hello names the worker's host, pid,
CUDA ordinal and GPU UUID. `import-trace --vllm-execution-dir DIR` and the
profile's own import bind a trace to the worker epoch that was alive on the
trace's host, with its launching pid, across the trace's time window; a pid
that no epoch covers, or that several cover, is reported and left to
`--device-uuid`, which always wins.

## Coverage

`stormlog infer analyze` reports the import under `telemetry.execution`,
and the text report prints it after the vLLM telemetry lines. The block
keeps five questions apart, each a union of busy intervals per device and
clock scope, never a sum:

1. **Linkage.** Measured GPU busy time with an iteration link against
   without, by the trace importer's reason.
2. **Membership.** Linked time whose step has complete, incomplete or no
   membership (a step the import has not written yet).
3. **Ownership.** Linked time in steps that ran only this run's requests
   (`run`), only other clients' (`foreign`), or both (`mixed`, claimed only
   when a run member and another client's are both established). A step
   with an unresolved member, an ID that had no alias and does not parse,
   is `unresolved` whether or not a run member sits beside it, since the
   split is unknown. A step is shared, and its case figure non-additive,
   when any member is not established as this run's.
4. **Measurement.** GPU activity without a device UUID or device clock is
   counted with its summed duration and never added to a device's union.
5. **Capture loss.** Records the hook dropped, missing sequences, steps
   still pending or incomplete, worker serving calls that ran without a
   range, and finishes no step could carry. The warm-up and graph-capture
   calls before the first serving step (`startup_unranged`) are shown apart:
   they never have a range, so they are not loss.

A case's figure covers every step one of its requests shared, so a case
that shared a batch with another case or with other clients is labelled
`non_additive`. No per-request GPU cost is computed.
