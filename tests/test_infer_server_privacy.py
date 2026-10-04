"""Schema-aware redaction of vLLM's configuration and environment."""

from __future__ import annotations

import json
from typing import Any

import pytest

from stormlog.infer.server_privacy import (
    is_redacted,
    package_versions,
    redact_environ,
    redact_vllm_config,
    redact_vllm_env,
    redacted,
    secret_name,
    system_env_summary,
)

PLANTED = "hf_plantedSecret0123456789"


def _vllm_config() -> dict[str, Any]:
    return {
        "model_config": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "abc123",
            "hf_token": PLANTED,
        },
        "scheduler_config": {
            "max_num_batched_tokens": 8192,
            "long_prefill_token_threshold": 0,
            "max_num_seqs": 256,
        },
        "load_config": {"model_loader_extra_config": {"s3_secret": PLANTED}},
        "kv_transfer_config": {
            "kv_connector": "LMCacheConnectorV1",
            "kv_connector_extra_config": {"password": PLANTED},
        },
        "speculative_config": {
            "draft_model_config": {"hf_token": PLANTED, "model": "draft"},
        },
        "observability_config": {
            "otlp_traces_endpoint": f"https://user:{PLANTED}@collector:4318/v1?k={PLANTED}"
        },
    }


def test_credential_fields_go_and_lookalike_names_stay() -> None:
    redacted_config = redact_vllm_config(_vllm_config())

    assert PLANTED not in json.dumps(redacted_config)
    model = redacted_config["model_config"]
    assert model["hf_token"] == redacted("/model_config/hf_token")
    assert model["tokenizer"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert redacted_config["scheduler_config"] == {
        "max_num_batched_tokens": 8192,
        "long_prefill_token_threshold": 0,
        "max_num_seqs": 256,
    }
    assert redacted_config["kv_transfer_config"]["kv_connector"] == "LMCacheConnectorV1"
    assert is_redacted(
        redacted_config["kv_transfer_config"]["kv_connector_extra_config"]
    )
    assert is_redacted(redacted_config["load_config"]["model_loader_extra_config"])
    draft = redacted_config["speculative_config"]["draft_model_config"]
    assert is_redacted(draft["hf_token"]) and draft["model"] == "draft"
    endpoint = redacted_config["observability_config"]["otlp_traces_endpoint"]
    assert endpoint == "https://collector:4318/v1?<redacted>"


def test_plugin_configs_and_the_ray_runtime_env_are_credential_paths() -> None:
    # Free-form dicts handed to plugins and to Ray, any of which can carry a
    # token or an environment of secrets.
    config = {
        "additional_config": {"remote_cache": {"auth_token": PLANTED}},
        "ec_manager_config": {
            "encoder_cache_manager_cls": None,
            "manager_config": {"password": PLANTED},
        },
        "parallel_config": {
            "ray_runtime_env": {"env_vars": {"HF_TOKEN": PLANTED}},
            "tensor_parallel_size": 1,
        },
    }
    kept = redact_vllm_config(config)
    assert PLANTED not in json.dumps(kept)
    assert is_redacted(kept["additional_config"])
    assert is_redacted(kept["ec_manager_config"]["manager_config"])
    assert kept["ec_manager_config"]["encoder_cache_manager_cls"] is None
    assert is_redacted(kept["parallel_config"]["ray_runtime_env"])
    assert kept["parallel_config"]["tensor_parallel_size"] == 1


@pytest.mark.parametrize("value", [None, False, True, {}, ""])
def test_an_unset_credential_field_is_kept_as_it_is(value: Any) -> None:
    # Redacting an empty field would make every comparison on it unverified.
    config = {"model_config": {"hf_token": value}}
    assert redact_vllm_config(config) == config


def test_json_pointer_tokens_are_escaped() -> None:
    config = {"a/b": {"c~d": f"http://u:{PLANTED}@h:1/x"}}
    assert redact_vllm_config(config) == {"a/b": {"c~d": "http://h:1/x"}}


@pytest.mark.parametrize(
    "name",
    [
        "HF_TOKEN",
        "VLLM_API_KEY",
        "STORMLOG_ACCESS_KEY",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "NCCL_SECRET",
        "CUDA_CREDENTIALS_FILE",
        "PYTORCH_PASSWORD",
        "VLLM_APIKEY",
        "VLLM_API_KEYS",
        "VLLM_ACCESS_TOKENS",
        "STORMLOG_OTLP_BEARER",
        "VLLM_PASS",
        "NCCL_SECRETS",
    ],
)
def test_secret_names_match_whole_words(name: str) -> None:
    assert secret_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_MAX_TOKENS_PER_EXPERT",
        "VLLM_TOKENIZER_POOL_SIZE",
        "NCCL_SOCKET_IFNAME",
        "VLLM_KEYEPOCH",
        "PYTORCH_CUDA_ALLOC_CONF",
        # vLLM 0.30.0's own knobs: thresholds and counts, not secrets.
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD",
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
        "VLLM_MAX_BAD_WORDS_TOTAL_TOKENS",
        "VLLM_ENABLE_PREGRAD_PASSES",
    ],
)
def test_names_that_only_contain_a_secret_word_are_not_secrets(name: str) -> None:
    assert not secret_name(name)


def test_settings_that_change_performance_are_kept_outside_the_prefixes() -> None:
    # A profiler preloaded into one arm, or another thread count, compiler
    # or kernel cache, changes what a run measures; dropping them hid it.
    environ = {
        "TORCH_COMPILE_DISABLE": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
        "TRITON_CACHE_DIR": "/home/cache/triton",
        "OMP_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4",
        "TORCH_HOME": "/root/.cache/torch",
        "HOME": "/root",
    }
    kept = redact_environ(environ)
    assert set(kept) == set(environ) - {"TORCH_HOME", "HOME"}
    assert kept["LD_PRELOAD"] == environ["LD_PRELOAD"]
    # Only a path that looks like a credential is redacted.
    secret = redact_environ({"LD_PRELOAD": f"/opt/{PLANTED}/lib.so"})
    assert is_redacted(secret["LD_PRELOAD"])


def test_an_environment_keeps_only_its_settings_without_secrets() -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "HF_TOKEN": PLANTED,
        "HF_HUB_OFFLINE": "1",
        "VLLM_API_KEY": PLANTED,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Bearer {PLANTED}",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://u:{PLANTED}@127.0.0.1:4318",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    kept = redact_environ(environ)

    assert PLANTED not in json.dumps(kept)
    assert set(kept) == {
        "HF_HUB_OFFLINE",
        "VLLM_API_KEY",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "CUDA_VISIBLE_DEVICES",
    }
    assert kept["VLLM_API_KEY"] == redacted("/environ/VLLM_API_KEY")
    assert kept["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == "http://127.0.0.1:4318"
    assert kept["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_other_otel_settings_are_redacted() -> None:
    # Resource attributes are free-form, and can carry a token.
    environ = {
        "OTEL_RESOURCE_ATTRIBUTES": f"service.name=x,api.token={PLANTED}",
        "OTEL_SERVICE_NAME": "vllm",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://127.0.0.1:4318/v1/traces",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
    }
    kept = redact_environ(environ)
    assert PLANTED not in json.dumps(kept)
    assert is_redacted(kept["OTEL_RESOURCE_ATTRIBUTES"])
    assert kept["OTEL_SERVICE_NAME"] == "vllm"
    assert kept["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert kept["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
        "http://127.0.0.1:4318/v1/traces"
    )


@pytest.mark.parametrize(
    "value",
    [
        f" http://user:{PLANTED}@collector:4317/v1",
        f"user:{PLANTED}@collector:4317",
    ],
    ids=["leading_space", "no_scheme"],
)
def test_credentials_in_a_url_of_any_spelling_go(value: str) -> None:
    kept = redact_environ({"VLLM_ENDPOINT": value})
    assert PLANTED not in json.dumps(kept)
    assert "collector:4317" in json.dumps(kept)


def test_urls_nested_in_vllm_env_values_are_stripped() -> None:
    kept = redact_vllm_env(
        {
            "VLLM_LIST": [f"http://u:{PLANTED}@h:1/x"],
            "VLLM_DICT": {"a": f"http://u:{PLANTED}@h:2"},
        }
    )
    assert PLANTED not in json.dumps(kept)
    assert kept["VLLM_LIST"] == ["http://h:1/x"]


def test_vllms_token_thresholds_are_kept() -> None:
    values = {
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": 1024,
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": 256,
    }
    assert redact_vllm_env(values) == values
    environ = {name: str(value) for name, value in values.items()}
    assert redact_environ(environ) == environ


def test_vllm_env_follows_the_same_rules() -> None:
    kept = redact_vllm_env(
        {"VLLM_USE_V1": True, "VLLM_HTTP_TOKEN": PLANTED, "VLLM_PORT": 8000}
    )
    assert kept == {
        "VLLM_HTTP_TOKEN": redacted("/vllm_env/VLLM_HTTP_TOKEN"),
        "VLLM_PORT": 8000,
        "VLLM_USE_V1": True,
    }


def test_system_env_keeps_scalars_and_named_package_versions_only() -> None:
    system_env = {
        "torch_version": "2.9.0+cu128",
        "cuda_runtime_version": "12.8.93",
        "python_version": "3.12.3 (main) [GCC 13.2.0] (64-bit runtime)",
        "vllm_version": "0.30.0",
        "is_cuda_available": True,
        "env_vars": f"HF_TOKEN={PLANTED}",
        "cpu_info": "Architecture: x86_64\n...",
        "gpu_topo": "GPU0 X",
        "pip_packages": (
            "numpy==2.2.6\ntorch==2.9.0+cu128\ntriton==3.5.0\n"
            f"transformers==4.57.1\nflashinfer-python==0.5.2\nsecret-pkg=={PLANTED}"
        ),
        "nvidia_gpu_models": ["not", "a", "scalar"],
    }
    summary = system_env_summary(system_env)

    assert PLANTED not in json.dumps(summary)
    assert "env_vars" not in summary and "cpu_info" not in summary
    assert "pip_packages" not in summary and "gpu_topo" not in summary
    assert "nvidia_gpu_models" not in summary
    assert summary["torch_version"] == "2.9.0+cu128"
    assert summary["packages"] == {
        "flashinfer-python": "0.5.2",
        "torch": "2.9.0+cu128",
        "transformers": "4.57.1",
        "triton": "3.5.0",
    }


def test_package_versions_ignore_anything_that_is_not_a_listing() -> None:
    assert package_versions(None) == {}
    assert package_versions("torch 2.9.0\ntriton") == {}
