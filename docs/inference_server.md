[← Back to docs](index.md)

# Inference server descriptions

Two runs can only be compared when they measured the same server, so
Stormlog describes the server a run measured: its vLLM configuration, its
environment, its runtime and its GPUs. A description is meant to be shared
with the run's results, so it never keeps a credential.

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
  else its `HF_HUB_CACHE`, `HF_HOME` or home directory) links each file of a
  snapshot to a blob named by a digest of the file: SHA-256 for a file stored
  in LFS, such as the weights, and git's SHA-1 for a small one, such as
  `config.json`. The description records each file's algorithm, digest and
  size. With blob verification, it hashes each blob to check its name.
- **A local directory.** Files have no digest of their own. With weight
  hashing, each file's SHA-256 is computed, and cached by path, size,
  `mtime_ns` and inode so the next description does not read the weights
  again. Without it, only sizes are known.

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
| `cudagraph_captures` | `Capturing CUDA graphs (decode, FULL)` and the like, as `decode:FULL` |
| `graph_capture_gib` | `Graph capturing finished in N secs, took X GiB` |

The patterns are vLLM 0.30.0's own log statements (`patterns:
vllm_0_30_0`); another version may word them differently, and then a field
is not found. A log file can hold several start-ups, and only the last one
counts. When workers disagree, for example on the attention backend, every
value is kept with an issue.

## What a description keeps

Redaction follows vLLM's configuration schema, never a substring. A field
is removed because it is a known credential field of vLLM 0.30.0, not
because its name contains "token": `max_num_batched_tokens`,
`long_prefill_token_threshold` and `tokenizer` are kept as they are.

| Source | Kept | Removed |
| --- | --- | --- |
| `vllm_config` | Every field, with URLs stripped of credentials and query | The fields in `credential_paths_v1` that hold a value: `hf_token` (of the model, and of a speculative target or draft model), and the free-form `model_loader_extra_config`, `kv_connector_extra_config` and `ec_connector_extra_config` |
| Process environment | Names starting `VLLM_`, `NCCL_`, `OTEL_`, `STORMLOG_`, `CUDA_` or `PYTORCH_`, plus `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`; URL values are stripped | Everything else, and any kept name with a secret word in it |
| vLLM's `vllm_env` | Every variable, URL values stripped | Any name with a secret word in it |
| vLLM's `system_env` | Allowlisted scalars (torch, CUDA, cuDNN, driver, Python, OS and vLLM versions, among others), and the versions of torch, triton, flashinfer and transformers read from its package listing | `env_vars`, the package listing itself, `cpu_info`, `gpu_topo`, and anything that is not a scalar |

A secret word is a whole `_`-separated word of the name: `TOKEN`, `SECRET`,
`PASSWORD`, `PASSWD`, `KEY`, `CREDENTIAL(S)`, `HEADER(S)`, `AUTH`,
`AUTHORIZATION` or `COOKIE`. So `VLLM_API_KEY` and
`OTEL_EXPORTER_OTLP_HEADERS` (which carries the exporter's `Authorization`)
are removed, and `VLLM_MAX_TOKENS_PER_EXPERT` is kept.

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

## Related pages

- [Inference Profiling](inference.md)
- [Inference SLOs and goodput](inference_slo.md)
- [vLLM native telemetry](vllm_telemetry.md)
