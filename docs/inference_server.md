[← Back to docs](index.md)

# Inference server descriptions

Two runs can only be compared when they measured the same server, so
Stormlog describes the server a run measured: its vLLM configuration, its
environment, its runtime and its GPUs. A description is meant to be shared
with the run's results, so it never keeps a credential in a field vLLM
0.30.0 defines to hold one, in an environment variable named as a secret,
or in a URL; the rules are below.

## Describing a server

Run `describe-server` on the host that serves vLLM, with the API server's
PID:

```bash
stormlog infer describe-server --pid "$(pgrep -f 'vllm serve' | head -1)" \
  --server-log /var/log/vllm/server.log \
  --output artifacts/server-before.json
```

| Option | Meaning |
| --- | --- |
| `--pid PID` | vLLM's API server process. Another process is described with an issue, and only it and its descendants are covered. |
| `--run-id ID` | The run the description belongs to, as passed to `infer profile`. `attach-manifest` refuses a description of another run. |
| `--output FILE` | Where to write the description. |
| `--server-log FILE` | The server's log, for the choices it made at start-up. |
| `--python auto\|PATH\|none` | The interpreter asked for the Python and package versions (torch, triton, flashinfer, transformers, vllm). `auto` (the default) is the interpreter on the server's command line: a bare `python` is found on the server's own `PATH`, and a relative path from its working directory; one that cannot be found is not run. `auto` runs that interpreter as the describing user, so describing as root a server you do not trust, pass `--python none` or a path you chose. |
| `--hash-weights` | Hash every file of a local model directory. |
| `--verify-model-files` | Hash every hub-cache blob to check it against its name. |
| `--digest-cache FILE` | Where `--hash-weights` keeps digests between runs. |
| `--no-gpu` | Describe without NVML. |

The description is one JSON document, `stormlog.infer.server_description`
version 1:

| Field | Content |
| --- | --- |
| `observed_at_ns` | When it was taken |
| `host` | Hostname, boot ID, boot time, `nproc`, and the describing process's CPU affinity |
| `server` | The root PID and start ticks, the command line's model arguments, every process in the tree, the worker start method, and the kept environment |
| `gpus` | The driver and each NVML device, with the server's processes on it |
| `model` | The model files, their digests and `identity_evidence` |
| `log` | The start-up choices from `--server-log`, or `null` |
| `runtime` | The interpreter's Python and package versions, or `null` |
| `nvidia_smi` | SHA-256 and size of `nvidia-smi -q -x`, never its text |
| `issues` | What could not be described, such as an unreadable environment |
| `sha256` | SHA-256 of the canonical JSON of everything else |

A description whose `sha256` does not match its content is refused when it
is read back.

`describe-server` exits `0` when the description is written, and `2` when
the PID is not a process this host can read (it needs Linux `/proc`) or NVML
is missing without `--no-gpu`. A `--server-log` that cannot be read exits
`5`.

## The run manifest

An artifact records which server it measured in `infer.manifest` records,
which are only ever appended:

| Role | Written by | Holds |
| --- | --- | --- |
| `before` | `infer profile --describe-server FILE` | A description taken before the run |
| `after` | `infer attach-manifest ARTIFACT FILE` | A description taken after the run |
| `declared` | `infer profile --declare FILE` | What the operator states; nothing observed |

```bash
# On the server host, before the run:
stormlog infer describe-server --pid "$SERVER_PID" --run-id "$RUN_ID" --output before.json
# On the client:
stormlog infer profile ... --run-id "$RUN_ID" --describe-server before.json
# On the server host, after the run, then on the client:
stormlog infer describe-server --pid "$SERVER_PID" --run-id "$RUN_ID" --output after.json
stormlog infer attach-manifest artifacts/infer.jsonl after.json
```

`profile` refuses (exit `5`) a `before` description that names another
run than its `--run-id`, and warns when it was taken more than an hour
before the run started.

`attach-manifest` refuses (exit `5`) an `after` description when:
- the artifact has no `before` description;
- it names another run;
- it was taken on another host, or after a reboot;
- its API server process is not the `before` one, by PID and start time:
  the server was restarted;
- it is the `before` description itself, or is already attached;
- it was taken before the last measured phase ended (the two hosts' clocks
  must agree, as with NTP);
- it was taken no later than the `before` one, or sooner after it than the
  measured phases took. Each interval is read on its own clock, the
  descriptions' on the server's and the phases' on the client's, so this
  holds however far apart the clocks are; one second of shortfall is
  allowed for clock rates. Without phase windows (an interrupted run), the
  measured requests' first start to last end stands in for the phases.

A declarations file is `stormlog.infer.declared` version 1:

```json
{"format": "stormlog.infer.declared", "version": 1,
 "fields": {"engine.version": "0.30.0", "host.purpose": "baseline"}}
```

The report's `manifest` block lists the `before` and `after` descriptions
(digest, time, server process and GPUs) and the declared fields.

The `before` description is checked against what the server told the probe
when the run began: the model it serves (`/v1/models` and `/server_info`),
its vLLM version (`/version`) and the GPU driver (`/server_info`'s
`system_env`). A disagreement is listed under `description_mismatches` and
makes `protocol_failure: description_mismatch`: the description is of
another server, or is stale.

With both descriptions, it compares them:

- `identity_changes`: settings that identify the server and changed during
  the run. Those are the driver and CUDA driver versions, the server's GPU
  UUIDs and each one's settings, the model snapshot, weights and chat
  template digests, the launch arguments, the Python and package versions,
  and the start-up choices from the log. Any change makes
  `protocol_failure: identity_changed`: the run did not measure one server.
- `identity_unverified`: settings only one description could read, such as
  an `after` taken without `--server-log` or `--python`, or a GPU field
  NVML did not answer. Missing evidence is not a change, so these are not a
  protocol failure; describe both sides with the same options.
- `drift`: for each of the server's GPUs, the SM clock, temperature and
  clock event reasons, before and after. Drift is reported, not a failure.

## Comparing two runs

`stormlog.infer.compatibility.compatible(a, b)` says whether two runs
measured the same thing. Each run's fields come from its artifact
(`run_fields(records)`): the `before` description, what the server reported
to the probe, the workload record, the observers the session configured,
and the declarations. Each field keeps its value, its source and its
provenance:

| Provenance | Meaning |
| --- | --- |
| `observed` | Stormlog saw it: the description, the workload record |
| `reported` | The server said so: `/version`, `/server_info` |
| `inferred` | Evidence that cannot show it, such as a model digest not bound to the launch |
| `declared` | The operator said so |

Observed outranks reported, and a declaration only fills a field nothing
observed or reported. An inferred or declared value, a redacted one and an
NVML field that could not be read are all unknown: they never verify a
required field, and two of them are never equal.

A `null` in vLLM's configuration or `vllm_env` is a setting, not missing
evidence: `quantization: null` against `fp8` is a difference. Where both
runs read a source (`/server_info`'s `vllm_config` or `vllm_env`, or the
server's process environment), a field only one has is a difference too,
named `only_in_a` or `only_in_b`: a section that is `null` in one run and
set in the other (speculative decoding turned on), a list cut short, or
`CUDA_LAUNCH_BLOCKING` set on one side. Where one run did not read the
source, its fields there are unknown. Only the server's own answer makes
`vllm_config` known; a declared configuration name does not.

Every field has a class:

| Class | A difference |
| --- | --- |
| `identity` | Makes the runs incompatible, unless it is allowed |
| `launch` | Is a covariate: ports, instance IDs, cache directories, which GPU, the workload's seed |
| `observation` | Which observers ran: allowed in `overhead` and `incremental` comparisons only |
| `label` | Is ignored: names, and credentials such as `hf_token`, which say who fetched the weights, not what ran |

The classes of vLLM's configuration are a versioned table,
`config_classes_v1`, keyed by JSON pointer into `/server_info`'s
`vllm_config`; the longest pointer that covers a leaf decides. A leaf no
pointer covers is `unclassified`, and a difference in it blocks. The
table's launch entries come from vLLM 0.30.0's source. Two identical
launches of vLLM 0.30.0 with Qwen2.5-0.5B on an A30 had 406 configuration
leaves, every one classified, and only `/instance_id` differed.

The result is one of:

| Status | When |
| --- | --- |
| `incompatible` | An identity or unclassified field differs and is not allowed |
| `unverified` | A required field is unknown on either or both sides: the model's weights digest, the vLLM version, the GPU name, the driver version, the workload's spec digest, or `vllm_config` itself. Or an identity or unclassified field is unknown on one side or both, so a difference in it could not be seen. Unknown launch and observation fields are listed under `unknown` without blocking |
| `compatible` | Otherwise |

`allowed` takes canonical names (`engine.max_num_seqs`), or JSON pointers
into `vllm_config` (`/scheduler_config`), which cover their subtree. The
result lists each difference with its class and reason: blocking,
unverified (`unknown`, or `differs_unverified` when two unverified values
disagree), allowed, covariates and observation.

The workload record carries two digests: `workload_digest`, the
realization, which includes the seed, and `spec_digest`, which leaves it
out. Runs of one workload with different seeds share a spec digest, which
must be equal for a comparison; the realization is a covariate.

## Observers

The report's `observers` block says, for each observer, whether the run
asked for it (`requested`), whether the server was set up to feed it
(`configured`, where that can be seen), whether it produced evidence
(`active`) and whether that evidence was good enough (`healthy`). Both are
judged in every compared phase (each case's measured phase, from its start
to the end of its drain), not once for the run:

| Observer | Active in a phase | Healthy in a phase |
| --- | --- | --- |
| `system_sampler` | A sample in it | At least 90% of the samples its interval expects. The sampler keeps a fixed grid, so a slow sample delays the next one, not the rate; a phase shorter than one interval is not judged |
| `vllm_metrics` | An ok scrape for it | Ok scrapes at its start and end, no gap between ok scrapes over twice the interval, and a metrics window that resolved (the vLLM block's case state) |
| `vllm_spans` | A span joined to one of its requests | Spans joined for at least 99% of its accepted requests, and no decode or receiver errors in the run |
| `trace` | A trace started for it | The trace stopped cleanly, wrote a file, and that file was imported |
| `execution` | An iteration of the hook in it | No dropped records, errors or disk cap in any epoch |

The execution hook is `requested` when the client imports its log
(`--vllm-execution-dir`), and also when the `before` description shows it
enabled in the server's environment (`STORMLOG_VLLM_HOOK_DIR`): it observes
the server either way, so an `overhead` baseline that must run without
observers sees it.

An observer is `healthy` only when it is in every compared phase it could
be judged in. What the artifact cannot show is `null`, with the reason in
`unjudged` (or, for one phase, in that phase's reasons): the hook's
heartbeat times are not kept in the artifact, so the execution hook is at
best `null` (not shown unhealthy, not shown healthy). The session record
now keeps the system sampler's interval and the trace settings, which these
judgments need.

## The server's processes

A description reads the server's processes from Linux `/proc`, so it runs on
the host that serves vLLM. Each process is identified by the host boot, its
PID and its start time in clock ticks since boot: a PID can be reused, the
three together cannot.

vLLM 0.30.0 renames its processes, and each one gets a role from its name or
command line:

| Role | Process |
| --- | --- |
| `api_server` | `vllm serve ...`, or `python -m vllm.entrypoints.openai.api_server` |
| `engine_core` | `VLLM::EngineCore`, or `VLLM::EngineCore_DP<n>` |
| `worker` | `VLLM::Worker`, or `VLLM::Worker_TP<n>` and the like |
| `resource_tracker` | Python multiprocessing's resource tracker |
| `compile_worker` | torch inductor's compile workers |
| `other` | Anything else, such as the `pip` or `nvidia-smi` that vLLM's `/server_info` starts |

`VLLM` is vLLM's default `VLLM_PROCESS_NAME_PREFIX`; another prefix is
recognized too. Each process also records its parent, process group,
session and `Cpus_allowed_list`.

A server's parent exiting does not end its children, and a child can leave
its process group and session. So a check that a server is gone looks at the
whole group and session, and at every process the description listed, by
PID and start time. A zombie has exited and counts as gone.

## The server's GPUs

The server's GPUs are the devices on which NVML lists one of the server's
processes as a compute process. `CUDA_VISIBLE_DEVICES` renumbers devices for
the server, so a device index alone cannot name them. Inside a container
NVML may list host PIDs, and then no device matches; the description says
so instead of guessing.

For each device NVML lists, the description keeps:

| Kind | Fields |
| --- | --- |
| Settings, which identify the hardware | name, power limit and enforced power limit (W), application clocks (graphics, SM, memory, MHz), persistence mode, ECC, MIG mode, compute mode |
| A reading of what drifts during a run | SM clock (MHz), temperature (°C), clock event reasons (formerly throttle reasons, such as `sw_power_cap` or `hw_thermal_slowdown`) |

plus the host's driver version and the CUDA version the driver supports.
A field NVML cannot read records why (`{"unavailable": "NVML code 3"}`) and
is never filled in.

## The model's weights

The server's command line names the model and its revision
(`vllm serve MODEL --revision R`, or `--model`). Where the files are depends
on the model:

- **A Hugging Face repository.** The server's hub cache (`--download-dir`,
  else its `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`,
  `XDG_CACHE_HOME` or home directory, as huggingface_hub looks) links each
  file of a snapshot to a blob named by a digest of the file: SHA-256 for a
  file stored in LFS, such as the weights, and git's SHA-1 for a small one,
  such as `config.json`. The description records each file's algorithm,
  digest and size. With blob verification, it hashes each blob, and records
  what the blob holds. A snapshot of copies rather than links (where links
  are unsupported) has no digests by name, so its files have none, and no
  `weights_digest`, unless verification hashes them. A link that leads
  nowhere, or in a loop, has no digest either.
- **A local directory.** Files have no digest of their own. With weight
  hashing, each file's SHA-256 is computed, and cached by path, size,
  `mtime_ns`, `ctime_ns` and inode so the next description does not read the
  weights again. The change time catches a rewrite that kept the size and
  modification time. Without hashing, only sizes are known.

`weights_digest` is a SHA-256 over the sorted list of files, each with its
algorithm, digest and size. The description also keeps the digest of the
chat template (`--chat-template`, else the snapshot's `chat_template.jinja`
or its `tokenizer_config.json`) and the snapshot's `generation_config.json`.

None of this shows what the server loaded: the cache or the directory may
have changed since it started. So the description only names its evidence:

| `identity_evidence` | When |
| --- | --- |
| `pinned_commit` | `--revision` is a 40-character commit, found in the cache |
| `inferred` | A branch or tag (`main` when none is given), resolved through the cache afterwards |
| `post_launch_digest` | A local directory, hashed after the server started |
| `size_only` | A local directory, not hashed |
| `unresolved` | The revision or the cache could not be found |

None of these verifies the model's identity on its own; only a launch the
experiment runner controls can do that.

## The server's log

Some settings are only decided once the engine runs, and vLLM's reported
configuration does not hold all of them. vLLM logs them, so a description
can read the server's log:

| Field | vLLM 0.30.0 log line |
| --- | --- |
| `attention_backend`, `attention_candidates` | `Using FLASH_ATTN attention backend out of potential backends: [...]`, or `Using ... backend.` for a backend chosen by name |
| `kv_cache_size_tokens`, `max_concurrency` | `GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: Xx` |
| `num_gpu_blocks_override` | `Overriding num_gpu_blocks=... with num_gpu_blocks_override=N` |
| `cudagraph_captures` | `Capturing CUDA graphs (decode, FULL)` and the like, as `decode:FULL`; Model Runner V2 names only the mode, `Capturing CUDA graphs (FULL)`, kept as `FULL` |
| `graph_capture_gib` | `Graph capturing finished in N secs, took X GiB`, from the last capture: with CUDA graph memory profiling on (vLLM's default) a first capture only measures memory |

The patterns are vLLM 0.30.0's own log statements (`patterns:
vllm_0_30_0`); another version may word them differently, and then a field
is not found. A log file can hold several start-ups, and only the last one
counts. When workers disagree, for example on the attention backend, every
value is kept with an issue.

## What the server reports

`infer profile` asks the server about itself over HTTP before the first case
and after the last (`--server-probe`, see [Inference Profiling](inference.md)).
Each route's answer is recorded with its status (`ok`, `http_error`,
`unreachable`, `delivery_unknown`, `failed`, `timeout`, `too_large`,
`invalid_json` or `skipped`), HTTP status, time and size:

- every answer is capped at 4 MiB, and redirects are never followed;
- the API key goes only to the endpoint's own origin;
- `/version` and `/v1/models` get 60 seconds each, `/server_info` one 120-second deadline. A deadline bounds the whole exchange, from connecting to the last byte, however slowly the server sends;
- when the server cannot be reached, or a route gets no answer in time, the other routes are skipped instead of each waiting out its deadline.

`/server_info` is never retried. When it times out, or the server takes the
request and drops it before a byte of answer (`delivery_unknown`), vLLM's
environment collector may still be running, so `profile` exits `5` before
it measures.

A URL anywhere in the `/version` and `/v1/models` answers, such as a
model's `root`, loses its credentials and query. `/server_info`'s answer is
kept redacted, by the rules below: its
`vllm_config`, its `vllm_env`, and a summary of `system_env`. vLLM caches
`system_env`, so after the run it is labelled `cached` and says nothing new.
Its package listing comes from `pip` in vLLM's environment and is empty
when that environment has none; `describe-server --python` asks the
interpreter instead.

## What a description keeps

Redaction follows vLLM's configuration schema, never a substring. A field
is removed because it is a known credential field of vLLM 0.30.0, not
because its name contains "token": `max_num_batched_tokens`,
`long_prefill_token_threshold` and `tokenizer` are kept as they are.

| Source | Kept | Removed |
| --- | --- | --- |
| `vllm_config` | Every field, with URLs stripped of credentials and query | The fields in `credential_paths_v1` that hold a value: `hf_token` (of the model, and of a speculative target or draft model); the free-form `model_loader_extra_config`, `kv_connector_extra_config`, `ec_connector_extra_config`, the cache manager's `manager_config` and the platform plugins' `additional_config`; and Ray's `ray_runtime_env`, whose `env_vars` can carry any secret of the job |
| Process environment | Names starting `VLLM_`, `NCCL_`, `OTEL_`, `STORMLOG_`, `CUDA_` or `PYTORCH_`, plus `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`. A URL loses its credentials and query, whatever space surrounds it, and `user:password@host` without a scheme loses its credentials | Everything else, any kept name with a secret word in it, and `OTEL_` settings other than the exporter's endpoint, protocol, timeout, compression and batching, and the sampler and service name: resource attributes are free-form |
| vLLM's `vllm_env` | Every variable, URLs stripped wherever they sit in its value | Any name with a secret word in it, and free-form `OTEL_` settings |
| vLLM's `system_env` | Allowlisted scalars (torch, CUDA, cuDNN, driver, Python, OS and vLLM versions, among others), and the versions of torch, triton, flashinfer and transformers read from its package listing | `env_vars`, the package listing itself, `cpu_info`, `gpu_topo`, and anything that is not a scalar |

A secret word is a whole `_`-separated word of the name: `TOKEN`,
`ACCESS_TOKENS`, `SECRET(S)`, `PASSWORD(S)`, `PASSWD`, `PASS`, `KEY`,
`APIKEY(S)`, `API_KEYS`, `CREDENTIAL(S)`, `HEADER(S)`, `AUTH`,
`AUTHORIZATION`, `COOKIE` or `BEARER`. So `VLLM_API_KEY` and
`OTEL_EXPORTER_OTLP_HEADERS` (which carries the exporter's `Authorization`)
are removed, and `VLLM_MAX_TOKENS_PER_EXPERT` is kept. vLLM 0.30.0's
`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` and
`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` are integer thresholds, named
in `NOT_SECRET_NAMES_V1`, and kept: removing them would hide a change in
them from every comparison.

The server's command line is never kept. Of its loading options
(`--model`, `--revision`, `--tokenizer` and the rest), a URL loses its
credentials and query, and any other value its query. When the Python
probe of `--python` fails, the description keeps its exit code, not its
output.

An unset credential field (`null`, `false`, or empty) holds no secret and is
kept. A removed value is replaced by a marker:

```json
{"redacted": true, "path": "/model_config/hf_token"}
```

A removed value is unavailable evidence. A comparison that requires it
cannot be verified, and two removed values are never treated as equal.

## Python API

```python
from stormlog.infer.server_privacy import (
    redact_environ,
    redact_vllm_config,
    redact_vllm_env,
    system_env_summary,
)
```

| Function | Returns |
| --- | --- |
| `redact_vllm_config(config)` | `vllm_config` with credential fields replaced by markers and URLs stripped |
| `redact_environ(environ)` | The kept part of a process environment |
| `redact_vllm_env(vllm_env)` | `vllm_env` by the same rules |
| `system_env_summary(system_env)` | The allowlisted scalars, and `packages` with the four runtime versions |

```python
from stormlog.infer.server_process import group_members, process_tree, still_running
```

| Function | Returns |
| --- | --- |
| `process_tree(pid)` | The process and its live descendants, root first, each with its role |
| `group_members(pgid, sid=None)` | Every live process in the group, or in the session too |
| `still_running(keys)` | The processes, given as `(pid, start_ticks)`, that are still alive |

```python
from stormlog.infer.server_gpu import NvmlGpuReader, describe_gpus, read_series
```

| Function | Returns |
| --- | --- |
| `describe_gpus(reader, server_pids)` | Every device with its settings, one drift reading, and the server processes on it |
| `read_series(reader, uuids)` | A fresh drift reading of those devices |

```python
from stormlog.infer.server_model import describe_model, launch_arguments
```

| Function | Returns |
| --- | --- |
| `launch_arguments(cmdline)` | The model, revision, tokenizer, chat template and download directory the command line names |
| `describe_model(launch, hub_cache=..., cwd=..., hash_weights=False, verify_blobs=False)` | The files, digests and `identity_evidence` above |
| `stormlog.infer.server_log.read_server_log(path)` | The last start-up's choices from a server log |
| `stormlog.infer.server_probe.probe_server(endpoint, mode="auto", ...)` | What the server reports about itself, as a `ServerProbe` |
| `stormlog.infer.compatibility.run_fields(records)` | One run's comparable fields |
| `stormlog.infer.compatibility.compatible(a, b, allowed=(), mode="config")` | `compatible`, `unverified` or `incompatible`, with every difference |

## Related pages

- [Inference Profiling](inference.md)
- [Inference SLOs and goodput](inference_slo.md)
- [vLLM native telemetry](vllm_telemetry.md)
