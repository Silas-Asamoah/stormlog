"""Decide, from vLLM's configured objects, whether the hook may record.

The gate runs where the configuration is known: in the scheduler's constructor
on the engine side and after ``Worker.init_device`` on the worker side. Both
sides apply the same rules to the same ``vllm_config``, so they agree. Anything
outside the supported matrix in ``docs/vllm_execution.md`` is refused with a
reason, and vLLM then runs untouched.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

SUPPORTED_VERSIONS = frozenset({"0.30.0"})
SUPPORTED_EXECUTORS = frozenset({"uni", "mp"})
SUPPORTED_SCHEDULERS = frozenset(
    {
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
    }
)
# Model Runner V2 is 0.30.0's default; it falls back to V1 for n-gram
# speculation. The hook wraps either through the same execute_model and
# sample_tokens shapes.
SUPPORTED_RUNNERS = frozenset(
    {
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
    }
)
SUPPORTED_SPECULATION = frozenset({"ngram"})
# vLLM's ProfilerConfig settings that decide what a profiler window costs the
# server (how long a stop pauses it) and whether one stops by itself.
PROFILER_FIELDS = (
    "profiler",
    "torch_profiler_dir",
    "torch_profiler_use_gzip",
    "torch_profiler_with_stack",
    "torch_profiler_record_shapes",
    "torch_profiler_with_memory",
    "torch_profiler_with_flops",
    "torch_profiler_dump_cuda_time_total",
    "capture_torch_profiler",
    "ignore_frontend",
    "max_iterations",
    "delay_iterations",
    "warmup_iterations",
    "active_iterations",
    "wait_iterations",
)


@dataclass(frozen=True)
class GateResult:
    """Whether to record, why not, and what was configured."""

    enabled: bool
    refused: str | None
    config: dict[str, Any] = field(default_factory=dict)


def vllm_version() -> str | None:
    try:
        return package_version("vllm")
    except PackageNotFoundError:
        return None


def check(vllm_config: Any, *, scheduler: Any = None, runner: Any = None) -> GateResult:
    """Apply the supported matrix; ``scheduler`` and ``runner`` are instances."""
    summary = config_summary(vllm_config, scheduler=scheduler, runner=runner)
    reasons = [
        reason
        for reason in (
            _version_reason(summary),
            _parallel_reason(summary),
            _connector_reason(vllm_config),
            _model_reason(vllm_config),
            _speculation_reason(vllm_config),
            _class_reason(summary.get("scheduler"), SUPPORTED_SCHEDULERS, "scheduler"),
            _class_reason(summary.get("runner"), SUPPORTED_RUNNERS, "model runner"),
        )
        if reason
    ]
    refused = "; ".join(reasons) or None
    return GateResult(enabled=refused is None, refused=refused, config=summary)


def config_summary(
    vllm_config: Any, *, scheduler: Any = None, runner: Any = None
) -> dict[str, Any]:
    parallel = getattr(vllm_config, "parallel_config", None)
    scheduling = getattr(vllm_config, "scheduler_config", None)
    speculative = getattr(vllm_config, "speculative_config", None)
    return {
        "vllm_version": vllm_version(),
        "executor": _executor_name(
            getattr(parallel, "distributed_executor_backend", None)
        ),
        "tp": getattr(parallel, "tensor_parallel_size", None),
        "pp": getattr(parallel, "pipeline_parallel_size", None),
        "dp": getattr(parallel, "data_parallel_size", None),
        "async_scheduling": getattr(scheduling, "async_scheduling", None),
        "max_num_batched_tokens": getattr(scheduling, "max_num_batched_tokens", None),
        "speculative": getattr(speculative, "method", None) if speculative else None,
        "v2_model_runner": getattr(vllm_config, "use_v2_model_runner", None),
        "request_id_randomization": _request_id_randomization(),
        "scheduler": _qualname(scheduler),
        "runner": _qualname(runner),
        **_runtime_layout(vllm_config, scheduler),
    }


def _runtime_layout(vllm_config: Any, scheduler: Any) -> dict[str, Any]:
    """Capacity and memory settings; a scheduler's own resolved values win,
    since the engine sets the KV layout only after profiling memory."""
    cache = getattr(vllm_config, "cache_config", None)
    model = getattr(vllm_config, "model_config", None)
    groups = getattr(
        getattr(scheduler, "kv_cache_config", None), "kv_cache_groups", None
    )
    block_size = getattr(scheduler, "block_size", None)
    return {
        "max_num_seqs": _primitive(
            getattr(
                getattr(vllm_config, "scheduler_config", None), "max_num_seqs", None
            )
        ),
        "num_gpu_blocks": _primitive(getattr(cache, "num_gpu_blocks", None)),
        "kv_cache_groups": len(groups) if isinstance(groups, (list, tuple)) else None,
        "block_size": _primitive(
            getattr(cache, "block_size", None) if block_size is None else block_size
        ),
        "cudagraph_mode": _primitive(
            getattr(
                getattr(vllm_config, "compilation_config", None), "cudagraph_mode", None
            )
        ),
        "gpu_memory_utilization": _primitive(
            getattr(cache, "gpu_memory_utilization", None)
        ),
        "enable_cumem_allocator": _primitive(
            getattr(model, "enable_cumem_allocator", None)
        ),
        "enable_sleep_mode": _primitive(getattr(model, "enable_sleep_mode", None)),
        "profiler": _profiler(getattr(vllm_config, "profiler_config", None)),
    }


def _profiler(profiler_config: Any) -> dict[str, Any] | None:
    if profiler_config is None:
        return None
    return {
        name: _primitive(getattr(profiler_config, name, None))
        for name in PROFILER_FIELDS
    }


def _primitive(value: Any) -> Any:
    """A JSON primitive: an enum by its name, any other object by its text."""
    if isinstance(value, enum.Enum):
        return value.name
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _request_id_randomization() -> bool | None:
    """Whether vLLM suffixes request IDs; ``None`` when vLLM cannot say."""
    try:
        from vllm import envs

        return not bool(envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION)
    except Exception:
        return None


def _version_reason(summary: dict[str, Any]) -> str | None:
    found = summary["vllm_version"]
    if found in SUPPORTED_VERSIONS:
        return None
    return f"vLLM {found or 'not found'} is not supported"


def _parallel_reason(summary: dict[str, Any]) -> str | None:
    if summary["executor"] not in SUPPORTED_EXECUTORS:
        return f"executor {summary['executor']} is not supported"
    if (summary["pp"] or 1) > 1:
        return "pipeline parallelism is not supported"
    if (summary["dp"] or 1) > 1:
        return "data parallelism is not supported"
    return None


def _connector_reason(vllm_config: Any) -> str | None:
    for name in ("kv_transfer_config", "ec_transfer_config"):
        if getattr(vllm_config, name, None) is not None:
            return f"{name} is set; connectors are not supported"
    return None


def _model_reason(vllm_config: Any) -> str | None:
    runner_type = getattr(
        getattr(vllm_config, "model_config", None), "runner_type", None
    )
    if runner_type == "pooling":
        return "pooling models are not supported"
    return None


def _speculation_reason(vllm_config: Any) -> str | None:
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None:
        return None
    method = getattr(speculative, "method", None)
    if method not in SUPPORTED_SPECULATION:
        return f"speculative method {method} is not supported"
    if getattr(speculative, "enable_adaptive_verification", False):
        return "adaptive verification is not supported"
    return None


def _class_reason(name: str | None, supported: frozenset[str], what: str) -> str | None:
    if name is None or name in supported:
        return None
    return f"{what} {name} is not supported"


def _executor_name(backend: Any) -> str | None:
    if backend is None or isinstance(backend, str):
        return backend
    return _qualname_of(backend)


def _qualname(instance: Any) -> str | None:
    return None if instance is None else _qualname_of(type(instance))


def _qualname_of(cls: Any) -> str:
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', '?')}"


__all__ = [
    "GateResult",
    "PROFILER_FIELDS",
    "SUPPORTED_VERSIONS",
    "check",
    "config_summary",
    "vllm_version",
]
