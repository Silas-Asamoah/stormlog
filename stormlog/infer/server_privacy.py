"""What a server description may keep of vLLM's configuration and environment.

Redaction follows the schema, never a substring: a configuration field is
removed because its JSON pointer is a known credential path in vLLM 0.30.0,
not because its name contains "token" (``max_num_batched_tokens`` and
``long_prefill_token_threshold`` must survive). Environment names are
matched on whole ``_``-separated words.

A removed value is replaced by ``{"redacted": true, "path": ...}``. It is
unavailable evidence: a comparison that requires it is unverified, and two
redacted values are never equal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .cache_state import redact_url

CREDENTIAL_PATHS_VERSION = "credential_paths_v1"

# Fields of vLLM 0.30.0's VllmConfig that can hold a credential: Hugging Face
# tokens, and the free-form extra configs handed to loaders and connectors.
CREDENTIAL_PATHS_V1: tuple[str, ...] = (
    "/model_config/hf_token",
    "/speculative_config/target_model_config/hf_token",
    "/speculative_config/draft_model_config/hf_token",
    "/load_config/model_loader_extra_config",
    "/speculative_config/draft_load_config/model_loader_extra_config",
    "/kv_transfer_config/kv_connector_extra_config",
    "/ec_transfer_config/ec_connector_extra_config",
)

# Environment the description keeps: vLLM's and its libraries' settings,
# plus the Hugging Face settings that say where and how weights were loaded.
ENVIRON_PREFIXES = ("VLLM_", "NCCL_", "OTEL_", "STORMLOG_", "CUDA_", "PYTORCH_")
ENVIRON_NAMES = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# A whole word of the name, so VLLM_MAX_TOKENS_PER_X survives and
# OTEL_EXPORTER_OTLP_HEADERS (which carries Authorization) does not.
_SECRET_NAME = re.compile(
    r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIALS?|HEADERS?|AUTH|"
    r"AUTHORIZATION|COOKIE)(_|$)"
)
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

# Scalars of vLLM's collect_env worth keeping; never env_vars or pip output.
SYSTEM_ENV_FIELDS = (
    "torch_version",
    "is_debug_build",
    "cuda_compiled_version",
    "cuda_runtime_version",
    "cudnn_version",
    "nvidia_driver_version",
    "nvidia_gpu_models",
    "python_version",
    "python_platform",
    "os",
    "libc_version",
    "is_cuda_available",
    "caching_allocator_config",
    "vllm_version",
)
# Versions read out of collect_env's ``pip list --format=freeze`` lines.
RUNTIME_PACKAGES = ("torch", "triton", "flashinfer-python", "transformers")


def redacted(path: str) -> dict[str, Any]:
    """The marker that replaces a removed value."""
    return {"redacted": True, "path": path}


def is_redacted(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("redacted") is True


def secret_name(name: str) -> bool:
    """Whether an environment variable's name says it holds a secret."""
    return _SECRET_NAME.search(name.upper()) is not None


def strip_url(value: str, path: str) -> Any:
    """A URL without credentials or query; a malformed one is removed."""
    try:
        return redact_url(value)
    except ValueError:
        return redacted(path)


def redact_vllm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """``vllm_config`` with credential fields removed and URLs stripped."""
    result = _redact_tree(config, "", frozenset(CREDENTIAL_PATHS_V1))
    return result if isinstance(result, dict) else {}


def redact_environ(environ: Mapping[str, str]) -> dict[str, Any]:
    """The kept part of an environment, secrets removed and URLs stripped."""
    kept: dict[str, Any] = {}
    for name in sorted(environ):
        if not (name.startswith(ENVIRON_PREFIXES) or name in ENVIRON_NAMES):
            continue
        path = f"/environ/{_escape(name)}"
        value = environ[name]
        if secret_name(name):
            kept[name] = redacted(path)
        elif _URL.match(value):
            kept[name] = strip_url(value, path)
        else:
            kept[name] = value
    return kept


def redact_vllm_env(vllm_env: Mapping[str, Any]) -> dict[str, Any]:
    """``/server_info``'s ``vllm_env``, by the same rules as an environment."""
    kept: dict[str, Any] = {}
    for name in sorted(vllm_env):
        path = f"/vllm_env/{_escape(name)}"
        value = vllm_env[name]
        if secret_name(name):
            kept[name] = redacted(path)
        elif isinstance(value, str) and _URL.match(value):
            kept[name] = strip_url(value, path)
        else:
            kept[name] = value
    return kept


def system_env_summary(system_env: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlisted scalars of collect_env, plus a few package versions."""
    summary: dict[str, Any] = {
        name: system_env[name]
        for name in SYSTEM_ENV_FIELDS
        if isinstance(system_env.get(name), (str, int, float, bool))
    }
    summary["packages"] = package_versions(system_env.get("pip_packages"))
    return summary


def package_versions(
    pip_packages: Any, names: Sequence[str] = RUNTIME_PACKAGES
) -> dict[str, str]:
    """``name==version`` lines for ``names``; nothing else of the listing."""
    if not isinstance(pip_packages, str):
        return {}
    wanted = {name.lower() for name in names}
    versions: dict[str, str] = {}
    for line in pip_packages.splitlines():
        name, separator, version = line.strip().partition("==")
        if separator and name.lower() in wanted and version:
            versions[name.lower()] = version
    return dict(sorted(versions.items()))


def _redact_tree(value: Any, path: str, credentials: frozenset[str]) -> Any:
    if path in credentials and _holds_something(value):
        return redacted(path)
    if isinstance(value, Mapping):
        return {
            str(key): _redact_tree(item, f"{path}/{_escape(str(key))}", credentials)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_tree(item, f"{path}/{index}", credentials)
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and _URL.match(value):
        return strip_url(value, path)
    return value


def _holds_something(value: Any) -> bool:
    """A credential field that is unset, empty or a flag holds no secret."""
    if value is None or isinstance(value, bool):
        return False
    return not (isinstance(value, (Mapping, list, str)) and not value)


def _escape(key: str) -> str:
    """A JSON pointer reference token (RFC 6901)."""
    return key.replace("~", "~0").replace("/", "~1")


__all__ = [
    "CREDENTIAL_PATHS_V1",
    "CREDENTIAL_PATHS_VERSION",
    "ENVIRON_NAMES",
    "ENVIRON_PREFIXES",
    "RUNTIME_PACKAGES",
    "SYSTEM_ENV_FIELDS",
    "is_redacted",
    "package_versions",
    "redact_environ",
    "redact_vllm_config",
    "redact_vllm_env",
    "redacted",
    "secret_name",
    "strip_url",
    "system_env_summary",
]
