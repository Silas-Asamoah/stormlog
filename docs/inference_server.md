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

## Related pages

- [Inference Profiling](inference.md)
- [Inference SLOs and goodput](inference_slo.md)
- [vLLM native telemetry](vllm_telemetry.md)
