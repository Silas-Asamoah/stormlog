"""What a server description may keep of vLLM's configuration and environment.

Redaction follows the schema, never a substring: a configuration field is
removed because its JSON pointer is a known credential path in vLLM 0.30.0,
not because its name contains "token" (``max_num_batched_tokens`` and
``long_prefill_token_threshold`` must survive). Environment names are
matched on whole ``_``-separated words, except vLLM 0.30.0's own knobs that
the words would take for secrets. OpenTelemetry settings are kept only by
name: the rest, such as resource attributes, are free-form.

A removed value is replaced by ``{"redacted": true, "path": ...}``. It is
unavailable evidence: a comparison that requires it is unverified, and two
redacted values are never equal.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .cache_state import redact_url

CREDENTIAL_PATHS_VERSION = "credential_paths_v1"

# Fields of vLLM 0.30.0's VllmConfig that can hold a credential: Hugging Face
# tokens, the free-form configs handed to loaders, connectors, cache managers
# and platform plugins, and Ray's runtime environment, whose env_vars can
# carry any secret of the job.
CREDENTIAL_PATHS_V1: tuple[str, ...] = (
    "/model_config/hf_token",
    "/speculative_config/target_model_config/hf_token",
    "/speculative_config/draft_model_config/hf_token",
    "/load_config/model_loader_extra_config",
    "/speculative_config/draft_load_config/model_loader_extra_config",
    "/kv_transfer_config/kv_connector_extra_config",
    "/ec_transfer_config/ec_connector_extra_config",
    "/ec_manager_config/manager_config",
    "/additional_config",
    "/parallel_config/ray_runtime_env",
)

# Environment the description keeps: vLLM's and its libraries' settings,
# plus the Hugging Face settings that say where and how weights were loaded.
ENVIRON_PREFIXES = (
    "VLLM_",
    "NCCL_",
    "OTEL_",
    "STORMLOG_",
    "CUDA_",
    "PYTORCH_",
    # The compiler, kernels and BLAS a run uses change what it measures.
    "TORCHINDUCTOR_",
    "TRITON_",
    "CUBLAS_",
)
ENVIRON_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TORCH_COMPILE_DISABLE",
    "OMP_NUM_THREADS",
    # A library preloaded into one arm, such as a profiler or an allocator.
    "LD_PRELOAD",
)
# A path segment shaped like a credential: a token, or a long encoded run.
_SECRET_SHAPED = re.compile(
    r"(hf_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|[A-Za-z0-9+/=]{40,})"
)

# A whole word of the name, so VLLM_MAX_TOKENS_PER_X survives and
# OTEL_EXPORTER_OTLP_HEADERS (which carries Authorization) does not.
_SECRET_NAME = re.compile(
    r"(^|_)(TOKEN|ACCESS_TOKENS|SECRETS?|PASSWORDS?|PASSWD|PASS|KEY|APIKEYS?|"
    r"API_KEYS|CREDENTIALS?|HEADERS?|AUTH|AUTHORIZATION|COOKIE|BEARER)(_|$)"
)
# vLLM 0.30.0's envs.py names that the words above would take for secrets.
NOT_SECRET_NAMES_V1 = frozenset(
    {
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD",
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
    }
)
# OpenTelemetry settings kept as they are; any other OTEL_ value, such as
# OTEL_RESOURCE_ATTRIBUTES, is free-form and is removed.
OTEL_NAMES_KEPT = frozenset(
    {
        "OTEL_SDK_DISABLED",
        "OTEL_SERVICE_NAME",
        "OTEL_TRACES_EXPORTER",
        "OTEL_TRACES_SAMPLER",
        "OTEL_TRACES_SAMPLER_ARG",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_INSECURE",
        "OTEL_EXPORTER_OTLP_TRACES_INSECURE",
        "OTEL_EXPORTER_OTLP_TIMEOUT",
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
        "OTEL_BSP_SCHEDULE_DELAY",
        "OTEL_BSP_EXPORT_TIMEOUT",
        "OTEL_BSP_MAX_QUEUE_SIZE",
        "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
    }
)
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
# ``user:password@host`` without a scheme.
_USERINFO = re.compile(r"^[A-Za-z0-9._~%!$&'()*+,;=-]+:[^\s/@]*@(?=[A-Za-z0-9.\[-])")

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
    upper = name.upper()
    return upper not in NOT_SECRET_NAMES_V1 and _SECRET_NAME.search(upper) is not None


def scrub_value(value: str, path: str) -> Any:
    """An environment value as it may be kept: credentials out of any URL.

    A URL loses its userinfo and query, whatever space surrounds it; a
    ``user:password@host`` without a scheme loses its userinfo.
    """
    trimmed = value.strip()
    if _URL.match(trimmed):
        return strip_url(trimmed, path)
    found = _USERINFO.match(trimmed)
    if found:
        return "<redacted>@" + trimmed[found.end() :]
    return value


# A scheme-less URL (host/path?token=...): no spaces, no template braces.
_URL_LIKE = re.compile(r"^[^\s{}%]+$")


def scrub_argument(value: str) -> str:
    """A command-line value as it may be kept: no credentials, no query."""
    trimmed = value.strip()
    if _URL.match(trimmed):
        try:
            return redact_url(trimmed) or "<redacted>"
        except ValueError:
            return "<redacted>"
    found = _USERINFO.match(trimmed)
    if found:
        return "<redacted>@" + trimmed[found.end() :]
    if "?" in trimmed and _URL_LIKE.match(trimmed):
        return trimmed.partition("?")[0] + "?<redacted>"
    return value


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


def _secret_preload(name: str, value: str) -> bool:
    """A preloaded library's path is kept, unless a segment looks like a key."""
    return name == "LD_PRELOAD" and _SECRET_SHAPED.search(value) is not None


def redact_environ(environ: Mapping[str, str]) -> dict[str, Any]:
    """The kept part of an environment, secrets removed and URLs stripped."""
    kept: dict[str, Any] = {}
    for name in sorted(environ):
        if not (name.startswith(ENVIRON_PREFIXES) or name in ENVIRON_NAMES):
            continue
        path = f"/environ/{_escape(name)}"
        if _removed_name(name) or _secret_preload(name, environ[name]):
            kept[name] = redacted(path)
        else:
            kept[name] = scrub_value(environ[name], path)
    return kept


def redact_vllm_env(vllm_env: Mapping[str, Any]) -> dict[str, Any]:
    """``/server_info``'s ``vllm_env``, by the same rules as an environment."""
    kept: dict[str, Any] = {}
    for name in sorted(vllm_env):
        path = f"/vllm_env/{_escape(name)}"
        if _removed_name(name):
            kept[name] = redacted(path)
        else:
            kept[name] = _redact_tree(vllm_env[name], path, frozenset(), scrub_value)
    return kept


def _removed_name(name: str) -> bool:
    return secret_name(name) or (
        name.upper().startswith("OTEL_") and name.upper() not in OTEL_NAMES_KEPT
    )


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


def _redact_tree(
    value: Any,
    path: str,
    credentials: frozenset[str],
    strings: Callable[[str, str], Any] | None = None,
) -> Any:
    if path in credentials and _holds_something(value):
        return redacted(path)
    if isinstance(value, Mapping):
        return {
            str(key): _redact_tree(
                item, f"{path}/{_escape(str(key))}", credentials, strings
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_tree(item, f"{path}/{index}", credentials, strings)
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        return (strings or _config_string)(value, path)
    return value


def _config_string(value: str, path: str) -> Any:
    """A configuration string: a URL loses its credentials and query."""
    trimmed = value.strip()
    return strip_url(trimmed, path) if _URL.match(trimmed) else value


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
    "NOT_SECRET_NAMES_V1",
    "OTEL_NAMES_KEPT",
    "RUNTIME_PACKAGES",
    "SYSTEM_ENV_FIELDS",
    "is_redacted",
    "package_versions",
    "redact_environ",
    "redact_vllm_config",
    "redact_vllm_env",
    "redacted",
    "scrub_argument",
    "scrub_value",
    "secret_name",
    "strip_url",
    "system_env_summary",
]
