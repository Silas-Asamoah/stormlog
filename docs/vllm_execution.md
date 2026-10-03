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
background thread writes out. The queue holds at most 20,000 records and 32 MiB,
estimated from each record's content and counted until the record is written; a
record that does not fit is dropped and counted, and so is a single record over
4 MiB. The thread checks its heartbeat, flush and sealing deadlines after every
record, so a backlog delays them by one write at most.

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
    "recompute": false, "output_before": 0, "resumable": false}]}
```

`iteration` counts the engine's `schedule()` calls from 0. `computed_before` is
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
{"kind": "completed", "iteration": "41", "wall_ns": …, "mono_ns": …,
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
 "finish_reason": "stop", "output_tokens": 128, "wall_ns": …, "mono_ns": …}
```

**`heartbeat`** and `status.json`

```json
{"kind": "heartbeat", "wall_ns": …, "mono_ns": …, "last_seq": 1234,
 "dropped": {"alias": 3, "alias_oversized": 1},
 "errors": 0, "bytes": 1048576, "capped": false, "queued": 0}
```

`dropped` counts dropped records by kind, and `<kind>_oversized` counts single
records over 4 MiB. `queued` is the number of records waiting to be written.

A worker's heartbeat adds `range_misses` (serving calls that ran without an
iteration range), `startup_unranged` (warm-up, dummy and CUDA-graph capture
calls before the first serving step, which never have one), and
`pending_samples`.

`status.json` holds the latest heartbeat's fields and is still updated after
the disk cap stops record writing, so loss stays visible.

**`goodbye`** has `wall_ns`, `mono_ns` and `last_seq`. A process that is killed,
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
record arrived, or it never will because the epoch ended (a `goodbye`, or
30 s without a heartbeat) or a later step completed, which makes it
`incomplete`. A pending step waits for a later import. Nothing is corrected
after it is written: each epoch's high-water sequence is kept in the engine
adapter's capability summary, and a later import starts from there, so
running the command twice adds nothing twice.

**Records.** One `infer.iteration` per step on the engine's monotonic clock
(`<host>/<boot id>/monotonic_ns`, `clock_kind` `monotonic`), whose
`elapsed_ns` is the scheduler residence from `schedule()` to the processed
output; one `infer.membership` per (request, attempt, iteration, role),
whose role follows vLLM's `phase` (`context` is `prefill`, `generation` is
`decode` or, with drafts scheduled, `spec_decode`; no phase is `unknown`), with
the step's token counts, outcome and, for the step a request was freed in,
its finish; one `infer.request` per backend execution, holding admission
facts only; and one `infer.clock_alignment` per epoch from the hello's
wall/monotonic pair, with half the sampling gap as its uncertainty. The raw
log is never registered as an attachment. A step whose `update_from_output`
raised (`update_failed`) keeps its memberships with outcome `unknown`, no
output tokens and nothing read from its counters; the coverage block counts
such steps. A request's `input_tokens` is its prompt at the first sighting;
a `resumable` request's later prompts are on its memberships.

**Binding.** A request is the run's when its alias `external` is
`chatcmpl-<x_request_id>` or `cmpl-<x_request_id>-<i>` for an
`x_request_id` the artifact recorded; without an alias, the internal ID's
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
them in `withheld_members`. A reused internal
ID (randomization off) is split into one execution per admission, also
across imports.

**Which steps are kept.** Steps an imported GPU activity references and
steps with a run member are always kept. A step with only other clients'
requests is kept when the engine's wall clock is the client's (same host
and boot) and the step lies inside a run phase or trace window; otherwise
it is counted in the summary, not written. A step that scheduled no request
(an idle scheduler call) is counted as empty, not written.

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
3. **Ownership.** Linked time in steps that ran only this run's requests,
   only other clients' (`foreign`), both (`mixed`), or unresolved IDs.
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
