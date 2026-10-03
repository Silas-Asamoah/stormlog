"""``/metrics`` in vLLM 0.30.0's Prometheus names, labels and buckets.

Rendered from the snapshot the step loop takes after each step, so while the
loop is paused a scrape still answers, with the values of the last step.
"""

from __future__ import annotations

import math
from typing import Iterable

from .config import FakeEngineConfig
from .stats import FINISH_REASONS, Histogram, StatsSnapshot

_COUNTERS = (
    ("vllm:prefix_cache_queries_total", "prefix_queries", "Prefix cache queries"),
    ("vllm:prefix_cache_hits_total", "prefix_hits", "Prefix cache hits"),
    ("vllm:num_preemptions_total", "preemptions", "Cumulative preemptions"),
    ("vllm:prompt_tokens_total", "prompt_tokens", "Prefill tokens processed"),
    ("vllm:prompt_tokens_cached_total", "prompt_tokens_cached", "Cached prefill"),
    ("vllm:generation_tokens_total", "generation_tokens", "Generation tokens"),
)


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
    deferred = snapshot.waiting - snapshot.waiting_capacity
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
    for reason, value in (
        ("capacity", snapshot.waiting_capacity),
        ("deferred", deferred),
    ):
        lines.append(_sample(name, {**engine, "reason": reason}, value))


def _cache_config(
    lines: list[str], config: FakeEngineConfig, engine: dict[str, str]
) -> None:
    name = "vllm:cache_config_info"
    _family(lines, name, "gauge", "Information of the LLMEngine CacheConfig")
    labels = {
        "block_size": str(config.block_size),
        "enable_prefix_caching": str(config.enable_prefix_caching),
        "engine": engine["engine"],
        "gpu_memory_utilization": "0.9",
        "num_gpu_blocks": str(config.num_gpu_blocks),
        "num_gpu_blocks_override": "None",
    }
    lines.append(_sample(name, labels, 1.0))


def _counters(
    lines: list[str], snapshot: StatsSnapshot, engine: dict[str, str]
) -> None:
    stats = snapshot.stats
    for name, attribute, help_text in _COUNTERS:
        _counter(lines, name, help_text, [(engine, float(getattr(stats, attribute)))])
        _created(lines, name[: -len("_total")] + "_created", [engine], stats.created_s)
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
