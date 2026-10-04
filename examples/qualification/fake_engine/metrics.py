"""``/metrics`` in vLLM 0.30.0's Prometheus names, labels and buckets.

Rendered from the snapshot the step loop takes after each step, so while the
loop is paused a scrape still answers, with the values of the last step.
"""

from __future__ import annotations

import math
from typing import Iterable

from .config import MAX_MODEL_LEN, FakeEngineConfig
from .stats import FINISH_REASONS, Histogram, StatsSnapshot

_COUNTERS = (
    ("vllm:prefix_cache_queries_total", "prefix_queries", "Prefix cache queries"),
    ("vllm:prefix_cache_hits_total", "prefix_hits", "Prefix cache hits"),
    ("vllm:num_preemptions_total", "preemptions", "Cumulative preemptions"),
    ("vllm:prompt_tokens_total", "prompt_tokens", "Prefill tokens processed"),
    ("vllm:prompt_tokens_cached_total", "prompt_tokens_cached", "Cached prefill"),
    ("vllm:generation_tokens_total", "generation_tokens", "Generation tokens"),
)
# Counters vLLM 0.30 always exports that this engine never moves: performance
# estimates (off by default), a KV connector's cache, the multi-modal cache.
_ZERO_COUNTERS = (
    "vllm:estimated_flops_per_gpu_total",
    "vllm:estimated_read_bytes_per_gpu_total",
    "vllm:estimated_write_bytes_per_gpu_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:mm_cache_queries_total",
    "vllm:mm_cache_hits_total",
)
# vLLM 0.30's CacheConfig labels that this engine's settings don't decide,
# with the values of a default single-GPU server.
_CACHE_CONFIG_DEFAULTS = {
    "_block_size_resolved": "True",
    "cache_dtype": "auto",
    "enable_mamba_fine_grained_prefix_cache": "False",
    "gpu_memory_utilization": "0.9",
    "is_attention_free": "False",
    "kv_cache_dtype_skip_layers": "[]",
    "kv_cache_layout": "None",
    "kv_cache_memory_bytes": "None",
    "kv_offloading_backend": "native",
    "kv_offloading_size": "None",
    "kv_sharing_fast_prefill": "False",
    "mamba_block_size": "None",
    "mamba_cache_dtype": "auto",
    "mamba_cache_mode": "none",
    "mamba_page_size_padded": "None",
    "mamba_ssm_cache_dtype": "auto",
    "num_cpu_blocks": "None",
    "num_gpu_blocks_override": "None",
    "prefix_cache_retention_interval": "0",
    "prefix_caching_hash_algo": "sha256",
    "prefix_match_unit": "None",
    "replayssm_buffer_len": "16",
    "skip_page_size_padded": "None",
    "sliding_window": "None",
    "use_kda_recoverssm": "False",
    "use_replayssm": "False",
    "user_specified_block_size": "False",
    "user_specified_mamba_block_size": "False",
}


def render_metrics(
    snapshot: StatsSnapshot, config: FakeEngineConfig, process_start_s: float
) -> str:
    engine = {"engine": "0", "model_name": config.model}
    lines: list[str] = []
    _family(lines, "process_start_time_seconds", "gauge", "Process start time")
    lines.append(f"process_start_time_seconds {_number(process_start_s)}")
    _gauges(lines, snapshot, engine)
    _cache_config(lines, config, engine)
    _counters(lines, snapshot, engine)
    for name, histogram in snapshot.stats.histograms.items():
        _histogram(lines, name, histogram, engine, snapshot.stats.created_s)
    return "\n".join(lines) + "\n"


def _gauges(lines: list[str], snapshot: StatsSnapshot, engine: dict[str, str]) -> None:
    gauges = (
        ("vllm:num_requests_running", snapshot.running),
        ("vllm:num_requests_waiting", snapshot.waiting),
        ("vllm:kv_cache_usage_perc", snapshot.kv_usage),
    )
    for name, value in gauges:
        _family(lines, name, "gauge", name)
        lines.append(_sample(name, engine, value))
    name = "vllm:num_requests_waiting_by_reason"
    _family(lines, name, "gauge", "Waiting requests by reason")
    # vLLM labels its ordinary waiting queue capacity, and deferred only the
    # requests its scheduler skipped, which this engine never does.
    for reason, value in (("capacity", snapshot.waiting), ("deferred", 0)):
        lines.append(_sample(name, {**engine, "reason": reason}, value))
    # Never asleep: this engine has no sleep mode.
    name = "vllm:engine_sleep_state"
    _family(lines, name, "gauge", "Engine sleep state")
    for state, value in (("awake", 1), ("weights_offloaded", 0), ("discard_all", 0)):
        lines.append(_sample(name, {**engine, "sleep_state": state}, value))


def _cache_config(
    lines: list[str], config: FakeEngineConfig, engine: dict[str, str]
) -> None:
    name = "vllm:cache_config_info"
    _family(lines, name, "gauge", "Information of the LLMEngine CacheConfig")
    tokens = config.num_gpu_blocks * config.block_size
    labels = {
        **_CACHE_CONFIG_DEFAULTS,
        "block_size": str(config.block_size),
        "enable_prefix_caching": str(config.enable_prefix_caching),
        "engine": engine["engine"],
        "kv_cache_max_concurrency": repr(tokens / MAX_MODEL_LEN),
        "kv_cache_size_tokens": str(tokens),
        "num_gpu_blocks": str(config.num_gpu_blocks),
    }
    lines.append(_sample(name, labels, 1.0))


def _counters(
    lines: list[str], snapshot: StatsSnapshot, engine: dict[str, str]
) -> None:
    stats = snapshot.stats
    for name, attribute, help_text in _COUNTERS:
        _counter(lines, name, help_text, [(engine, float(getattr(stats, attribute)))])
        _created(lines, name[: -len("_total")] + "_created", [engine], stats.created_s)
    for name in _ZERO_COUNTERS:
        _counter(lines, name, name, [(engine, 0.0)])
        _created(lines, name[: -len("_total")] + "_created", [engine], stats.created_s)
    _prompt_sources(lines, snapshot, engine)
    name = "vllm:request_success_total"
    series = [
        ({**engine, "finished_reason": reason}, float(stats.success.get(reason, 0)))
        for reason in FINISH_REASONS
    ]
    _counter(lines, name, "Finished requests", series)
    _created(
        lines,
        "vllm:request_success_created",
        [labels for labels, _value in series],
        stats.created_s,
    )


def _prompt_sources(
    lines: list[str], snapshot: StatsSnapshot, engine: dict[str, str]
) -> None:
    """Prompt tokens by where their KV came from, counted at each request's
    first token: computed here, or this engine's prefix cache."""
    stats = snapshot.stats
    series = [
        ({**engine, "source": source}, float(value))
        for source, value in (
            ("local_compute", stats.prompt_tokens - stats.prompt_tokens_cached),
            ("local_cache_hit", stats.prompt_tokens_cached),
            ("external_kv_transfer", 0),
        )
    ]
    name = "vllm:prompt_tokens_by_source_total"
    _counter(lines, name, "Number of prompt tokens by source", series)
    _created(
        lines,
        "vllm:prompt_tokens_by_source_created",
        [labels for labels, _value in series],
        stats.created_s,
    )


def _counter(
    lines: list[str],
    name: str,
    help_text: str,
    series: Iterable[tuple[dict[str, str], float]],
) -> None:
    _family(lines, name, "counter", help_text)
    for labels, value in series:
        lines.append(_sample(name, labels, value))


def _created(
    lines: list[str], name: str, label_sets: list[dict[str, str]], created_s: float
) -> None:
    _family(lines, name, "gauge", name)
    for labels in label_sets:
        lines.append(_sample(name, labels, created_s))


def _histogram(
    lines: list[str],
    name: str,
    histogram: Histogram,
    engine: dict[str, str],
    created_s: float,
) -> None:
    _family(lines, name, "histogram", name)
    bounds = [*map(_number, histogram.bounds), "+Inf"]
    for bound, count in zip(bounds, histogram.cumulative()):
        lines.append(_sample(f"{name}_bucket", {**engine, "le": bound}, count))
    lines.append(_sample(f"{name}_count", engine, histogram.count))
    lines.append(_sample(f"{name}_sum", engine, histogram.total))
    _created(lines, f"{name}_created", [engine], created_s)


def _family(lines: list[str], name: str, kind: str, help_text: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {kind}")


def _sample(name: str, labels: dict[str, str], value: float) -> str:
    body = ",".join(f'{key}="{_escape(labels[key])}"' for key in sorted(labels))
    return f"{name}{{{body}}} {_number(value)}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value: float) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


__all__ = ["render_metrics"]
