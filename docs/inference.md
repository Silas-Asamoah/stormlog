[← Back to main docs](index.md)

# Inference Profiling

Stormlog can actively profile OpenAI-compatible Chat Completions endpoints with
the top-level `stormlog infer` command group. This surface is intentionally
separate from `gpumemprof` and `tfmemprof`: the endpoint may be backed by
PyTorch, vLLM, SGLang, TensorRT-LLM, MLX-LM, a hosted gateway, or another
server that accepts the Chat Completions request shape.

## Profile an endpoint

```bash
stormlog infer profile \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,4,8 \
  --input-tokens 512,2048 \
  --output-tokens 128,512 \
  --requests 20 \
  --output artifacts/infer_qwen.jsonl
```

You can pass a `/v1` base URL instead of the full endpoint:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 8 \
  --input-tokens 2048 \
  --output-tokens 512 \
  --duration 120 \
  --output artifacts/infer_steady_state.jsonl
```

The profiler sends controlled traffic for each workload case in the matrix:

- `concurrency`
- prompt token target
- output token cap
- streaming or non-streaming mode

`--requests` is the total measured request count per workload case, shared
across the configured workers (default: 1). `--duration` instead runs each
workload case for the requested wall-clock window.

Warmup requests are recorded but excluded from analysis:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --warmup-requests 8 \
  --requests 50 \
  --output artifacts/infer_with_warmup.jsonl
```

## Control the workload

By default, the profiler runs a closed loop and repeats one prompt per case.
Each worker sends its next request when its previous one finishes, so a slower
server receives less traffic, and after the first request most of the prompt
can come from the engine's prefix cache. Both behaviours are useful, but a
comparison between engine configurations needs to choose them on purpose. The
options below control arrivals, prompts, and cache state. Every run records
what it used.

### Arrivals

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --arrival poisson --rate 2,4,8 --duration 120 \
  --prompt-mode unique \
  --output artifacts/infer_poisson.jsonl
```

| `--arrival` | When requests are sent | Case ID prefix |
| --- | --- | --- |
| `closed` (default) | When a worker is free; `--concurrency` sets the workers | `c4` |
| `fixed-rate` | Evenly spaced at each `--rate` (requests/second) | `fixed2` |
| `poisson` | Exponential gaps at each `--rate`, drawn from `--seed` | `poisson2` |
| `burst` | `--burst-size` requests every `--burst-interval` seconds | `burst8x1s` |
| `replay` | The offsets in `--arrival-trace` | `replay` |

In the open-loop modes, the schedule is fixed before the run starts and the
first request goes out at 0. A phase can schedule at most 1,000,000 arrivals.
`--requests` caps the number of arrivals, `--duration` keeps the arrivals
before the window closes, and a replay with neither sends the whole trace. A
replay trace is JSON lines with `offset_ms` (milliseconds), or a Stormlog
inference artifact. From an artifact, the measured requests of one case are
replayed: pick the case with `--arrival-trace-case` when there are several.

`--max-in-flight` (default 128) limits how many requests can be outstanding
at once. When every slot is busy, `--overflow wait` (default) holds the
arrival until a slot frees up, and the arrivals behind it fall behind too; in
a `--duration` run the hold ends at the drain deadline (see below).
`--overflow drop` records the arrival as `dropped` and never sends it. A
closed loop takes `--concurrency` instead; `--max-in-flight` and `--overflow`
apply only to the open-loop modes.

Every request records its `arrival_mode`, `request_index` and
`intended_at_ns`. It also records its `dispatch_lag_ms` (sent minus
intended), whether it was `held_for_slot`, and its `in_flight_at_dispatch`.
Each case's `arrivals` block in the report counts what was offered, sent,
completed, dropped and held. It also gives failures by status, peak
in-flight, the offered rate and dispatch-lag percentiles. The offered rate is
measured from the arrivals actually scheduled, so a Poisson case shows the
rate it drew rather than `--rate`. It is null for a closed loop, and for a
case whose arrivals all came at one instant, such as a single request or a
single burst. When requests were held, latency measured from the send leaves
out the time they waited.
`latency_ms.e2e_from_intended_*` measures from each request's intended
arrival, so that delay stays visible. The text report shows its p95 for every
open-loop case.

### Measured window and drain

Each phase has a window, when requests arrive, and a drain after it. The
window ends when `--duration` runs out, or when the last scheduled or counted
request is sent. During the drain, requests that are still running may finish
for up to `--drain-timeout` seconds (default: `--timeout`), measured from the
window end. Any request still running at that deadline is recorded as
`cancelled`, and an open-loop arrival that `--overflow wait` was still holding
for a slot is recorded as `dropped`: the window has closed, so it is never
sent. An open loop counted with `--requests` sends every arrival, held ones
included, before its window closes, so its drain starts after the last send.
A closed loop counted with `--requests` has no drain deadline: every request
in it is measured, so each one finishes or times out. Every phase writes an
`infer.phase_window` record, and each case's `arrivals` block reports
`window_seconds` and `drain_seconds`.

`cancelled` means Stormlog stopped waiting, not that the request stopped. The
HTTP call keeps running, on the server and on a client thread, until it
finishes or reaches `--timeout`. So the next phase waits for those calls
before it starts, rather than measuring their load as its own. Its
`infer.phase_window` record says how many it waited for and for how long, under
`abandoned_requests`. A lower `--drain-timeout` ends a phase sooner, but not
the run.

Ctrl+C stops a profile with exit code 130. Requests still running are
recorded as `cancelled`, and the artifact ends with an `infer.session` record
whose status is `interrupted`. A run that fails for another reason ends with
status `incomplete`. Either way, the requests recorded before the stop can
still be analyzed.

`infer profile` returns codes from the
[exit-code contract](report_contract.md):

- `0` when the run completes and at least one measured request succeeds.
- `3` when none succeeds; the artifact records each failure.
- `2` for a setting it cannot use, before anything is sent.
- `5` for an `--arrival-trace` it cannot read.
- `1` for anything unexpected.

Request outcomes:

| `status` | Meaning |
| --- | --- |
| `ok` | The request completed |
| `timeout` | The client gave up after `--timeout` |
| `rejected` | The server answered HTTP 429 or 503; `http_status` says which |
| `error` | Any other failure, with `http_status` when there was one |
| `dropped` | Never sent: `--overflow drop` turned the arrival away, or the drain deadline passed while `--overflow wait` held it; `error_message` says which |
| `cancelled` | Still running when the drain deadline passed; the call itself runs on until it finishes or times out |

### Prompts and prefix sharing

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input-tokens 2048 --requests 200 \
  --prompt-mode shared-prefix --shared-prefix-ratio 0.75 --prefix-groups 4 \
  --output artifacts/infer_shared_prefix.jsonl
```

| `--prompt-mode` | What requests share |
| --- | --- |
| `repeat` (default) | One prompt for the whole case, warmup included, as in earlier versions |
| `unique` | Nothing: each request starts with its own nonce |
| `shared-prefix` | A group prefix covering `--shared-prefix-ratio` of the tokens, one of `--prefix-groups` groups chosen by seed |

Nonces come from the seed, the case and the phase, so a run can be repeated
exactly. Neither another case nor the warmup shares a prefix with the measured
requests, except in `repeat` mode, where cases with the same input length send
the same text. The server's chat template still adds the same tokens to every
request, so even `unique` prompts share those. Because a repeated run sends the
same prompts, running the same workload twice against one server makes the
second run start with those prompts already cached. To measure each run from a
cold cache, reset the cache before each case (see below) or change `--seed`
between runs; the CLI warns when neither is done.

Each request records its `prompt_mode`, `prompt_id`, `prefix_group`,
`shared_prefix_tokens` and `prompt_digest`. Each case reports how many distinct
prompts and prefix groups it used, and the range of shared prefix lengths. Each
phase window's `prompts_digest` covers the prompts of every scheduled arrival in
schedule order, dropped ones included, so two runs of one schedule share it
whatever their overflow policy. In a closed loop with `--duration`, it covers
the prompts actually sent.

The nonce takes about eight subword tokens, so `unique` and `shared-prefix` need
`--input-tokens` of at least 32 to hit the target length and prefix share; the
CLI warns below that. Prompts land within a token of the target, and each
request records the length it actually sent: the server's count when it reports
one, otherwise Stormlog's own count. A dropped or cancelled request records the
planned length, marked as not exact. A prompt is built when its request is sent
and its text is dropped once the request is done, so a long schedule neither
waits for all of its prompts to be built nor keeps them in memory.

### Cache state

`--cache-state cold` records that each case should start with an empty prefix
cache. `--cache-reset-url` is POSTed before each case to clear it, with the API
key when one is set. Examples are
vLLM's `/reset_prefix_cache`, which vLLM serves only when started with
`VLLM_SERVER_DEV_MODE=1`, and SGLang's `/flush_cache`:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-mode unique \
  --cache-state cold --cache-reset-url http://localhost:8000/reset_prefix_cache \
  --output artifacts/infer_cold.jsonl
```

Asking for a cold cache is not proof that the cache was empty. No engine
adapter can read the cache yet, so each case's `infer.cache_state` record and
the report's `cache` block say `unverified`, with the reason. The reason is
one of:

- the reset succeeded but cannot be confirmed;
- the reset failed, with its HTTP status or error;
- nothing reset the cache.

A failed reset is recorded and the run continues. Each case also has a
`run_kind`, which names how the run was designed:

- `cold_start`: a cold cache was requested, no warmup ran, and no reset
  failed;
- `steady_state`: warmup ran;
- `unspecified`: anything else, including a cold start whose reset failed.

The label is not evidence about the cache. One warmup request is enough to
make a case `steady_state`. Compare runs of the same kind.

### Workload record

Every run writes an `infer.workload` record. It holds the seed, the prompt
generator version, the cases and their arrival shapes, and the measurement and
warmup settings. It also holds the decoding settings sent, the requested cache
state, and the tokenizer that sized the prompts, with its name, revision and
library version when known.

The record's `workload_digest` covers what decides the requests a run sends:
the cases, arrivals, prompts, warmup, decoding settings, seed, tokenizer and
requested cache state. For open-loop arrivals it also covers
`--max-in-flight` and `--overflow`. It leaves out the endpoint, model,
timeouts and reset URL, so the same workload sent to two engine
configurations has the same digest. `--extra-body` numbers are compared by
value, so `0` and `0.0` count as the same.

The API key is never recorded, and the reset URL is recorded without its
credentials or query string. The server applies the chat template, and the
record says so. When a local transformers tokenizer has a template, the
record keeps that template's digest as a hint only.

Pass sampling settings with `--extra-body`, a JSON object merged into every
request:

```bash
stormlog infer profile ... --extra-body '{"temperature": 0, "ignore_eos": true}'
```

`--extra-body` cannot replace the fields Stormlog sets itself: `model`,
`messages`, `stream`, `stream_options` and the output cap. Any setting that
isn't passed is recorded as a server default. For each case, the report gives
the prompt and output token distributions of the completed requests.

## Analyze an artifact

```bash
stormlog infer analyze artifacts/infer_qwen.jsonl
stormlog infer analyze artifacts/infer_qwen.jsonl --format json --output report.json
```

The report includes:

- end-to-end latency percentiles
- TTFT percentiles for streaming responses
- first streamed chunk latency
- requests/sec
- output tokens/sec and total tokens/sec
- failure rate
- highest recorded client-local device memory when system telemetry is available
- scoped server memory observations when a matching on-host collector artifact is supplied

`infer analyze` exits `5` when the artifact or a `--server-telemetry` file is
missing, unparsable, or invalid, which includes an artifact with no
`infer.session` or `infer.request` records. Otherwise it exits `0`, even when
every request in the artifact failed: analysis reports findings without
failing.

## Token accounting

Server usage metadata is preferred whenever the endpoint returns it. If usage is
missing, Stormlog falls back to the configured tokenizer and records the source
on every request event:

- `server_usage`
- `tiktoken`
- `transformers`
- `estimated`
- `unknown`

When streaming is enabled, Stormlog requests OpenAI-style streaming usage
metadata with `stream_options.include_usage` by default. Use
`--no-stream-usage` for endpoints that reject that request field. If streaming
usage is unavailable, output token counts fall back to the configured tokenizer
or estimate and the request event records that provenance.

Fallback counts are made on each request's own thread, not on the thread that
keeps the arrival schedule. A slow tokenizer can still compete with the
schedule for the CPU when hundreds of long prompts a second need counting, so
for open-loop runs at high rates prefer an endpoint that reports usage, and
check the case's dispatch-lag percentiles.

Core endpoint profiling does not require tokenizer packages. Install tokenizer
extras when you want better prompt sizing and fallback counts:

```bash
pip install "stormlog[infer-tokenizers]"
```

Useful tokenizer options:

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tokenizer transformers \
  --tokenizer-model Qwen/Qwen2.5-7B-Instruct \
  --output artifacts/infer_qwen.jsonl
```

For OpenAI-model tokenizers:

```bash
stormlog infer profile \
  --endpoint https://api.openai.com/v1/chat/completions \
  --model gpt-4o-mini \
  --tokenizer tiktoken \
  --tiktoken-encoding o200k_base \
  --output artifacts/infer_openai.jsonl
```

## Metric boundaries

Stormlog reports client-observed metrics in v1:

- TTFT is measured from request start to the first non-empty streamed content
  delta.
- Non-streaming responses do not have TTFT or chunk timing.
- Chunk inter-arrival timing is chunk-level timing. It is not treated as
  token-level ITL unless a future engine adapter can prove token-level events.
- Token throughput uses server usage when available; otherwise the configured
  tokenizer or estimate is clearly recorded.

## Client-local telemetry

Use `--system-sampler` to choose best-effort telemetry:

```bash
stormlog infer profile ... --system-sampler nvidia-smi
stormlog infer profile ... --system-sampler psutil
stormlog infer profile ... --system-sampler none
```

These samplers run where `stormlog infer profile` runs. Even if the endpoint is
remote, `nvidia-smi` reads the **client's** GPU and `psutil` reads the
**Stormlog client process**. Each new `infer.system_sample` has
`observation_scope: client_local`; older v1 samples are interpreted the same
way. The JSON report puts these values under `memory.observation_scope:
client_local` and retains the existing `peak_device_used_bytes` and
`peak_process_rss_bytes` keys for compatibility. Neither key is a server
memory claim. An unavailable reading stays `null`, never synthetic zero.

## Optional server telemetry

Run the collector **on the inference host** while profiling. Give both commands
the same run ID. The server PID must be the process serving the directly
addressed endpoint. This example assumes the profiler and server share a host;
for separate hosts, use the clock alignment options described below.

```bash
RUN_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')
# The UUID of the GPU that the server's worker process uses; see below.
GPU_UUID=GPU-00000000-0000-0000-0000-000000000000
stormlog infer collect-server \
  --run-id "$RUN_ID" --pid 12345 --device-uuid "$GPU_UUID" \
  --interval 0.1 --duration 60 --output artifacts/server.jsonl &
stormlog infer profile \
  --run-id "$RUN_ID" --base-url http://127.0.0.1:8000/v1 \
  --model my-model --requests 20 --output artifacts/client.jsonl
wait
stormlog infer analyze artifacts/client.jsonl \
  --server-telemetry artifacts/server.jsonl --direct-server --format json
```

The collector requires `psutil` (a core dependency) and, for GPU counters, an
NVIDIA driver exposing NVML v2. Use `--no-gpu` for process RSS only. The
collector records the actual host, boot ID, PID, process start, GPU UUID, and
MIG identity on each counter. `--replica-id` and `--rank` attach additional
identity. For servers with separate HTTP and GPU worker processes, target the
worker PID that owns the GPU work. The direct-route assertion then includes the
operator's knowledge that the addressed HTTP server uses that worker. Stormlog
does not infer this relationship from a matching GPU index.

Select the GPU with `--device-uuid`. `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv`
lists the GPU that each process uses, and a MIG UUID selects an instance.
`--device-index` is NVML's index, which follows PCI bus order. It is not the
server's CUDA device number: CUDA orders GPUs fastest first by default, and
`CUDA_VISIBLE_DEVICES` renumbers them, so a server's `cuda:0` can be NVML index
3. When collection starts, the collector checks that NVML lists the server PID
on the selected GPU and prints a warning if it does not. If a child of the
server process owns the GPU work, the warning names that PID. Inside a container
NVML can report host PIDs, so treat the warning as a prompt to check rather than
proof of a wrong GPU.

A server that spreads a model across GPUs, such as vLLM with tensor parallelism,
runs one worker process per GPU. Run one collector per worker, with that
worker's PID and GPU UUID, and declare the collectors as one group: the same
`--group-id` and `--world-size`, and a distinct `--rank` from 0 to N-1. In vLLM
0.30 the workers are children of the `VLLM::EngineCore` process and are titled
`VLLM::Worker_TP0`, `VLLM::Worker_TP1`, and so on; the API server and engine
core hold no GPU memory.

```bash
ps -eo pid,args | grep "[V]LLM::Worker_TP"   # worker PIDs and ranks
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader
stormlog infer collect-server --run-id "$RUN_ID" --pid 2600 \
  --device-uuid "$GPU_UUID_0" --group-id tp --rank 0 --world-size 2 \
  --output artifacts/rank0.jsonl &
stormlog infer collect-server --run-id "$RUN_ID" --pid 2601 \
  --device-uuid "$GPU_UUID_1" --group-id tp --rank 1 --world-size 2 \
  --output artifacts/rank1.jsonl &
# ... run `stormlog infer profile --run-id "$RUN_ID" ...`, then stop the collectors
stormlog infer analyze artifacts/client.jsonl --direct-server \
  --server-telemetry artifacts/rank0.jsonl --server-telemetry artifacts/rank1.jsonl
```

Read process titles with `ps -o args`; `/proc/<pid>/comm` cuts them at 15
characters. The same flags describe one process that spreads a model over
several GPUs: run one collector per GPU with the same `--pid`.

The analyzer joins a group only when every rank from 0 to N-1 appears exactly
once. Otherwise the report stays unjoined with one of these reasons:

| Reason | Meaning |
| --- | --- |
| `group_member_missing` | A rank below `--world-size` has no collector artifact |
| `group_member_changed` | One rank appears with two identities, such as a restarted worker or another GPU |
| `undeclared_server_identity` | An identity without the group ID sits next to the group |
| `multiple_server_groups` | The artifacts name more than one group ID |
| `inconsistent_group_size` | The members disagree on `--world-size` |
| `clock_flags_ambiguous` | Clock flags were given, but members run on more than one other host |

Each member keeps its own clock evidence and invalidation point. Members on the
client's host and boot share its clock; members on other hosts each need an
`infer.clock_alignment` record, because `--clock-offset-ns` can describe only
one remote host. For a group, each case lists `memory.server_members`, one
entry per rank with its PID, GPU, observations, and coverage, and
`memory.server_observations` stays empty. `memory.server_coverage` is
`observed` when every member is observed, `empty` when none is, and `partial`
otherwise. Values from different members are never added together: separate
collectors poll at different instants, so a sum of their maxima is not the peak
of the combined usage.

Collection ends when `--duration` elapses, on Ctrl+C or SIGTERM, when the server
process exits, or when the GPU UUID changes. Every completed poll is kept, and
the command prints the stop reason. Ctrl+C and SIGTERM are a normal stop. When
the server process exits, the collector writes one `invalid` sample, prints a
warning, and exits 0, because stopping the server after a run is routine. A
changed GPU UUID also writes an `invalid` sample, and the command exits 3
(`FINDINGS`): the earlier polls are sound, but later case windows are not
observed. Options it cannot use, a `--pid` with no running process, a
`--device-index` or `--device-uuid` the host does not have, and a host without
NVML (pass `--no-gpu` there) exit 2 before collection starts. A
reading that fails without evidence of a different process or GPU, such as an
NVML error or a psutil permission error, is recorded as `missing` with a null
value, and collection continues.

Timestamps are joined through clock domains. A wall clock domain names one host
boot, `{host}/{boot_id}/unix_epoch_ns`, because hostnames can repeat across
machines. The client artifact records its domain in the `infer.artifact`
context, and each telemetry record carries the collector's. Equal domains are
one clock, so on the same host and boot no alignment is needed; a
`--clock-uncertainty-ns` given alone is applied as given. A host that cannot
report a boot ID, such as a Windows host, gets `{host}/unix_epoch_ns`, which
never counts as a shared clock, even with the same hostname on both sides. Such
a pair can still be joined with `--clock-offset-ns` and `--clock-uncertainty-ns`
(use an offset of 0 when both commands ran on one machine); the report then
names the server side `{host}/unix_epoch_ns#server`. An `infer.clock_alignment`
record cannot describe that pair, because both of its domain names would be
equal. Artifacts written before domains named the boot are read with the boot
ID from their `metadata`.

For a profiler on another host, copy the server JSONL to the analysis host and
provide a measured server-to-client clock offset and an uncertainty bound:

```bash
stormlog infer analyze artifacts/client.jsonl \
  --server-telemetry artifacts/server.jsonl --direct-server \
  --clock-offset-ns 1200000 --clock-uncertainty-ns 300000 \
  --format json
```

`server timestamp + offset = client timestamp`. Derive the bound from a clock
synchronization service or a two-way timestamp probe near the run. Instead of
flags, a tool can append `infer.clock_alignment` records to the client artifact
(see [Inference execution correlation](inference_correlation.md)) with
`from_clock_domain` set to the server's domain and `to_clock_domain` set to the
client's. Such records are used without retyping and may carry
`valid_from_ns`/`valid_to_ns` windows, for example to follow clock drift during
a long run; each sample then uses the one record whose window covers its server
timestamp. The flags replace records for the same pair of domains, and the
report lists the replaced records under `overridden_clock_alignments`. The
flags and records are validated by the same rules. A record whose
`context.run_id` differs from the client artifact's run is never used, so a
calibration copied from another run cannot place samples; a joined report lists
such records under `ignored_clock_alignments`.

Without clock evidence the report lists server targets but does not join their
samples to client request windows, and `telemetry.server_join.reason` says why:

| Reason | Meaning |
| --- | --- |
| `clock_alignment_required` | The domains differ and no flag or record connects them |
| `clock_uncertainty_required` | `--clock-offset-ns` was given without `--clock-uncertainty-ns` |
| `clock_offset_required` | `--clock-uncertainty-ns` was given alone, but the server is on another host or boot |
| `clock_offset_on_shared_clock` | A nonzero `--clock-offset-ns` was given for one host and boot, which is one clock |
| `clock_domain_unverified` | Both sides have the same hostname and no boot ID, and no clock flags were given |
| `clock_alignment_uncovered` | No record's validity window covers any sample |
| `clock_alignment_ambiguous` | Several records cover the same samples |
| `clock_alignment_from_another_run` | The only records for these clocks name a different run ID |

A joined report lists every alignment it applied in
`telemetry.server_join.clock_alignments` (its source, `event_id`, offset,
uncertainty, window and number of samples), and counts the samples that no
single record covered under `unaligned_samples`; those samples are left out.
`clock_offset_ns` is set when one alignment placed every joined sample, and
`clock_uncertainty_ns` is the largest uncertainty applied. Each sample keeps
the uncertainty of the alignment that placed it and counts only if that
uncertainty fits inside the request window, so a loose alignment late in a run
does not remove samples that a precise earlier alignment placed.

`--direct-server` is an explicit assertion that every request in the
profile reached this one serving process. Do not use it for a load balancer that
can route to multiple replicas. Multiple server identities or a different run ID
prevent the case-window join; the client report is still produced, with the
reason under `telemetry.server_join`. An `invalid` sample marks the point where
the observed process or GPU stopped being the one the collector started with.
Samples before it still count, and a case is left out only if its window could
extend past the last poll that confirmed the identity, allowing for the clock
uncertainty. These checks do not prove per-request memory ownership: other
requests and processes can use the GPU during the same window.

The server artifact uses [versioned `infer.telemetry_sample` records](schemas/inference_telemetry_v1.schema.json),
one counter per record. `scope` identifies `server_process`, `gpu_device`, or `gpu_instance`;
`counter_owner`, `source`, `provenance`, `interval_ms`, `state`, and the
identity explain what the number means. The report's
`memory.server_observations` gives `maximum_recorded_bytes` and counts of
valid, missing, stale, and invalid samples for each metric, using only samples
inside the case's counted window. A maximum is the
largest recorded value at the chosen cadence, not the true peak. The 100 ms
default is a starting point for short requests with direct NVML reads; it is
not a universal sampling rule. Slower collectors and exporters may miss short
peaks or report cached/averaged values.

A record that passes the schema can still fail to load, because JSON Schema
cannot compare two values or tell `100` from `100.0`. The loader also requires
that `clock_domain` is exactly `{host}/{boot_id}/unix_epoch_ns`, or
`{host}/unix_epoch_ns` when `boot_id` is null; that a group member's `rank` is
below its `world_size`; and that integers are written without a decimal point
or exponent. A float cannot hold a nanosecond timestamp exactly: near the
current time, adjacent float64 values are 256 ns apart.

Each case also has `memory.server_coverage`. Its `status` is `observed`,
`partial`, or `empty`, and `counted_window_ns` gives the client-clock span in
which samples count. A `partial` or `empty` case carries a `reason`:

| Reason | Meaning |
| --- | --- |
| `window_shorter_than_uncertainty` | The case is shorter than twice the clock uncertainty, so no sample is certainly inside it |
| `identity_invalidated` | The case could extend past the last poll that confirmed the server process and GPU |
| `no_collector_coverage` | No collector poll falls inside the counted window |
| `clock_alignment_uncovered` | Polls that likely fell in this case were not placed, because no alignment record covers them |
| `clock_alignment_ambiguous` | Polls that likely fell in this case were not placed, because several alignment records cover them |
| `collector_started_after_window_start` | The collector's first poll came more than one interval after the window began |
| `collector_stopped_before_window_end` | The collector's last poll came more than one interval before the window ended |

`telemetry.server_join.case_coverage` counts the cases in each status, and the
text report prints the status for every case.

| Counter | Scope and owner | Collector support |
| --- | --- | --- |
| `process_rss_bytes` | Server process; operating system | On-host `psutil` |
| `device_memory_used_bytes`, `device_memory_reserved_bytes` | Whole GPU device; NVML | On-host NVML v2 |
| `instance_memory_used_bytes`, `instance_memory_reserved_bytes` | MIG instance; NVML | On-host NVML v2 when supported |
| `process_gpu_used_bytes` | Server process; GPU process accounting | Contract for optional sources; not collected by this command |
| `allocator_allocated_bytes`, `allocator_reserved_bytes` | Server process; allocator | Contract for optional sources; not collected by this command |
| `engine_cache_occupied_bytes` | Server process; engine cache | Contract for optional sources; not collected by this command |

Allocator, device, process RSS, and engine cache numbers must not be added or
substituted for each other. NVIDIA documents the NVML v2 `used` and `reserved`
fields separately in its [NVML memory structure](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlMemory__v2__t.html).
Exporters such as DCGM may report interval averages or cached values; this
collector currently reads NVML directly and does not ingest DCGM metrics.

## Profiler traces

`stormlog infer profile --trace vllm-torch` opens a vLLM torch-profiler window
around one phase of each case, then imports the GPU work in the traces it wrote
(see [Importing profiler traces](inference_correlation.md#importing-profiler-traces)).

What must be set when the server starts, and what Stormlog does while it runs:

| When | Setting | Who sets it |
| --- | --- | --- |
| Server start | `--profiler-config.profiler=torch` and `--profiler-config.torch_profiler_dir=DIR` | you |
| Server start | `--profiler-config.torch_profiler_with_stack=false` (Python stacks per operator are the largest overhead) and `--profiler-config.ignore_frontend=true` (no second profiler in the API server) | you; recommended |
| During the run | `POST /start_profile` and `/stop_profile` around the window | Stormlog |
| After the run | read the new `rank*.pt.trace.json*` files in `DIR` and import them | Stormlog, when `--trace-dir` is readable from the client |

```bash
stormlog infer profile --base-url http://server:8000/v1 --model MODEL \
  --concurrency 1,8 --requests 64 \
  --trace vllm-torch --trace-dir /shared/vllm-traces \
  --trace-max-seconds 30 --trace-device-uuid 0=GPU-6d1f0c5e-...
```

- **Window.** By default each case's measured phase is profiled
  (`--trace-phase warmup` profiles warmup instead). The window closes when the
  phase ends, when `--trace-max-seconds` elapses (the phase keeps running
  unprofiled), or when the run is cancelled; the profiler is stopped in every
  case.
- **Ownership.** Stormlog calls `/stop_profile` only after its own
  `/start_profile` succeeded; a failed start (for example, a server started
  without a profiler) is recorded as not started and nothing is stopped. vLLM
  0.30.0 answers 200 to a second `/start_profile` and to `/stop_profile` with
  nothing running, so Stormlog cannot tell from HTTP whether another profile
  was already active. Do not point two profilers at one server. If a worker
  trace appears before Stormlog's stop, for example from a profile configured
  with `max_iterations`, the window's `stop_reason` is `stopped_by_server`.
- **Record.** Each window writes an `infer.trace_window` event, also when the
  run is cancelled: case, phase, control URL (credentials and query removed),
  when the start was requested and when it was confirmed, start and stop HTTP
  status or error, why it stopped, and the trace files found. A cancelled run
  does not import its traces; import the listed files with
  `stormlog infer import-trace`. If `--trace` was requested and no trace could
  be imported, the trace collector's `infer.capabilities` record says so, with
  each window's reason.
- **Files.** Only worker traces (`rank*.pt.trace.json*`) that appear during the
  window are imported; the API server's `*.async_llm.*` trace is ignored. A file
  larger than `--trace-max-bytes` is registered in the run envelope but not
  parsed. Without `--trace-dir`, the traces stay on the server; import them
  later with `stormlog infer import-trace`. vLLM writes the trace while handling
  `/stop_profile`, so if no new file has appeared about 5 seconds after the
  stop, Stormlog stops waiting for that window.
- **Clock and settings.** Imported timestamps are Kineto's host-calibrated
  device times, not a raw GPU clock. The CUDA-graph and compile settings the
  server ran with are not in the trace; they stay unknown unless recorded
  elsewhere.
- **Coverage.** With an engine that emits iteration ranges, GPU work launched
  outside them stays unresolved by design. On vLLM 0.30.0 with Qwen2.5-0.5B on
  an NVIDIA A30, two profiled cases produced 367,192 GPU events in 29,578
  launch records: all unresolved without ranges, and all linked with a test
  plugin that wrapped each step in `stormlog.iteration/...` ranges.
- **Cost.** The profiler adds no synchronization per request, but it does CPU
  work per operator, and `/stop_profile` blocks while the server writes the
  trace (tens of seconds for a 30-second window). On vLLM 0.30.0 on an NVIDIA
  A30, with stacks off and `ignore_frontend=true`, output tokens per second fell
  0.4–1.3% for Qwen2.5-7B and 4.4–6.5% for Qwen2.5-0.5B, and requests per step
  were unchanged within 0.5%. A trace import never treats CPU launch time as GPU
  time.

For a PyTorch program that is not behind a server, `capture_torch_trace`
profiles a block in-process and writes a trace for `import-trace`:

```python
from stormlog.infer.trace_ranges import iteration_range
from stormlog.infer.trace_torch import capture_torch_trace

with capture_torch_trace("traces/run.pt.trace.json"):
    for step in range(100):
        with iteration_range("my-loop", str(step)):
            train_or_serve_one_step()
```

It refuses to start while another PyTorch profiler is running.

## Execution correlation and future adapters

The v1 request path is engine-agnostic. Future adapters can enrich the same run
with engine-native telemetry such as vLLM scheduler metrics, SGLang cache
metrics, TensorRT-LLM inflight batching metrics, or MLX Metal runtime stats
without changing the core `stormlog infer profile` artifact shape.

The versioned request, iteration, stage, membership, and GPU activity contract
is described in [Inference execution correlation](inference_correlation.md).
It preserves v1 client observations while allowing optional server evidence
to be appended to the same JSONL stream. New profiles also include a v2
artifact identity record that names their run and capture session.
