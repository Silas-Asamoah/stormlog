"""How each part of a run's description counts when two runs are compared.

Every field has a class:

- ``identity``: it decides what was measured, so a difference makes two runs
  incompatible;
- ``launch``: it differs between two launches of the same server (ports,
  instance IDs, cache directories, which physical GPU); a covariate, never
  blocking;
- ``observation``: which observers ran and how; it may differ only in
  ``overhead`` and ``incremental`` comparisons;
- ``label``: a name, ignored.

``CONFIG_CLASSES_V1`` classifies vLLM 0.30.0's ``/server_info`` configuration
by JSON pointer; the longest pointer that covers a leaf decides. A leaf that
no pointer covers is ``unclassified`` and blocks a comparison it differs in:
a configuration hash alone never allows a difference. The launch entries
come from vLLM 0.30.0's source. Two identical launches (Qwen2.5-0.5B on an
A30) had 406 leaves, all classified, and differed only in ``/instance_id``.
The table is versioned so that a change to it is visible.
"""

from __future__ import annotations

from collections.abc import Mapping

IDENTITY = "identity"
LAUNCH = "launch"
OBSERVATION = "observation"
LABEL = "label"
UNCLASSIFIED = "unclassified"
CLASSES = (IDENTITY, LAUNCH, OBSERVATION, LABEL)

CONFIG_CLASSES_VERSION = "config_classes_v1"

CONFIG_CLASSES_V1: Mapping[str, str] = {
    "/model_config": IDENTITY,
    "/model_config/served_model_name": LABEL,
    "/model_config/allowed_local_media_path": LAUNCH,
    "/cache_config": IDENTITY,
    "/parallel_config": IDENTITY,
    "/parallel_config/data_parallel_master_ip": LAUNCH,
    "/parallel_config/data_parallel_master_port": LAUNCH,
    "/parallel_config/data_parallel_rpc_port": LAUNCH,
    "/parallel_config/master_addr": LAUNCH,
    "/parallel_config/master_port": LAUNCH,
    "/parallel_config/_coord_store_port": LAUNCH,
    "/parallel_config/distributed_timeout_seconds": LAUNCH,
    "/parallel_config/cpu_distributed_timeout_seconds": LAUNCH,
    "/scheduler_config": IDENTITY,
    "/device_config": IDENTITY,
    "/load_config": IDENTITY,
    "/load_config/download_dir": LAUNCH,
    "/offload_config": IDENTITY,
    "/attention_config": IDENTITY,
    "/engram_config": IDENTITY,
    "/mamba_config": IDENTITY,
    "/kernel_config": IDENTITY,
    "/lora_config": IDENTITY,
    "/speculative_config": IDENTITY,
    "/watermark_config": IDENTITY,
    "/diffusion_config": IDENTITY,
    "/structured_outputs_config": IDENTITY,
    "/observability_config": OBSERVATION,
    "/quant_config": IDENTITY,
    "/compilation_config": IDENTITY,
    "/compilation_config/cache_dir": LAUNCH,
    "/compilation_config/local_cache_dir": LAUNCH,
    "/compilation_config/debug_dump_path": LAUNCH,
    "/compilation_config/compilation_time": LAUNCH,
    "/compilation_config/encoder_compilation_time": LAUNCH,
    "/profiler_config": OBSERVATION,
    "/kv_transfer_config": IDENTITY,
    "/kv_transfer_config/kv_ip": LAUNCH,
    "/kv_transfer_config/kv_port": LAUNCH,
    "/kv_transfer_config/engine_id": LAUNCH,
    "/kv_events_config": OBSERVATION,
    "/ec_transfer_config": IDENTITY,
    "/ec_transfer_config/ec_ip": LAUNCH,
    "/ec_transfer_config/ec_port": LAUNCH,
    "/ec_transfer_config/engine_id": LAUNCH,
    "/ec_manager_config": IDENTITY,
    "/reasoning_config": IDENTITY,
    "/additional_config": IDENTITY,
    "/instance_id": LAUNCH,
    "/optimization_level": IDENTITY,
    "/performance_mode": IDENTITY,
    "/weight_transfer_config": IDENTITY,
    "/shutdown_timeout": LAUNCH,
}

# vLLM's own environment variables that only place or observe a launch.
VLLM_ENV_CLASSES_V1: Mapping[str, str] = {
    "VLLM_PORT": LAUNCH,
    "VLLM_HOST_IP": LAUNCH,
    "VLLM_RPC_BASE_PATH": LAUNCH,
    "VLLM_CACHE_ROOT": LAUNCH,
    "VLLM_CONFIG_ROOT": LAUNCH,
    "VLLM_ASSETS_CACHE": LAUNCH,
    "VLLM_SERVER_DEV_MODE": OBSERVATION,
    "VLLM_LOGGING_LEVEL": OBSERVATION,
    "VLLM_LOGGING_PREFIX": OBSERVATION,
    "VLLM_LOGGING_CONFIG_PATH": OBSERVATION,
    "VLLM_CONFIGURE_LOGGING": OBSERVATION,
    "VLLM_LOG_STATS_INTERVAL": OBSERVATION,
    "VLLM_TRACE_FUNCTION": OBSERVATION,
    "VLLM_TORCH_PROFILER_DIR": OBSERVATION,
    "VLLM_PROCESS_NAME_PREFIX": LABEL,
}

# Canonical fields: exact names, and prefixes ending in "." or "_".
FIELD_CLASSES_V1: Mapping[str, str] = {
    "model.": IDENTITY,
    "model.configured": LABEL,
    "tokenizer.": IDENTITY,
    "server.chat_template_digest": IDENTITY,
    "server.generation_config": IDENTITY,
    "server.served_models": LABEL,
    "engine.": IDENTITY,
    "effective.": IDENTITY,
    "runtime.": IDENTITY,
    "gpu.": IDENTITY,
    "gpu.uuids": LAUNCH,
    "host.": LAUNCH,
    "host.start_method": IDENTITY,
    "environ.": IDENTITY,
    "environ.CUDA_VISIBLE_DEVICES": LAUNCH,
    "environ.HF_HOME": LAUNCH,
    "environ.HF_HUB_CACHE": LAUNCH,
    "environ.OTEL_": OBSERVATION,
    "environ.STORMLOG_": OBSERVATION,
    "workload.": IDENTITY,
    "workload.realization_digest": LAUNCH,
    "observer.": OBSERVATION,
    "experiment.": LABEL,
}


def config_class(pointer: str) -> str:
    """The class of a ``vllm_config`` leaf, by its JSON pointer."""
    return _longest(CONFIG_CLASSES_V1, pointer, separator="/") or UNCLASSIFIED


def vllm_env_class(name: str) -> str:
    return VLLM_ENV_CLASSES_V1.get(name, IDENTITY)


def field_class(name: str) -> str:
    """The class of a canonical field: its own entry, else its longest prefix.

    An entry that ends in "." or "_" is a prefix; any other names one field.
    """
    if name in FIELD_CLASSES_V1:
        return FIELD_CLASSES_V1[name]
    prefixes = [
        prefix
        for prefix in FIELD_CLASSES_V1
        if prefix.endswith((".", "_")) and name.startswith(prefix)
    ]
    return FIELD_CLASSES_V1[max(prefixes, key=len)] if prefixes else UNCLASSIFIED


def _longest(table: Mapping[str, str], pointer: str, *, separator: str) -> str | None:
    path = pointer
    while path:
        if path in table:
            return table[path]
        path = path.rsplit(separator, 1)[0] if separator in path[1:] else ""
    return None


__all__ = [
    "CLASSES",
    "CONFIG_CLASSES_V1",
    "CONFIG_CLASSES_VERSION",
    "FIELD_CLASSES_V1",
    "IDENTITY",
    "LABEL",
    "LAUNCH",
    "OBSERVATION",
    "UNCLASSIFIED",
    "VLLM_ENV_CLASSES_V1",
    "config_class",
    "field_class",
    "vllm_env_class",
]
