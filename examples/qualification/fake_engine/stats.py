"""The engine's own statistics, named and bucketed as vLLM 0.30.0 exports them."""

from __future__ import annotations

import copy
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

# Bucket boundaries copied from a real vLLM 0.30.0 /metrics response.
_LATENCY: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
)
_LATENCY += (1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0)
_ITL: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
)
_ITL += (2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)
_REQUEST: tuple[float, ...] = (
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
)
_REQUEST += (40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0)
_TOKENS: tuple[float, ...] = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
)
_STEP_TOKENS: tuple[float, ...] = (
    1.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
)
_STEP_TOKENS += (2048.0, 4096.0, 8192.0, 16384.0)
_PREEMPTIONS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
_N: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)

HISTOGRAM_BUCKETS: dict[str, tuple[float, ...]] = {
    "vllm:time_to_first_token_seconds": _LATENCY,
    "vllm:inter_token_latency_seconds": _ITL,
    "vllm:request_time_per_output_token_seconds": _ITL,
    "vllm:e2e_request_latency_seconds": _REQUEST,
    "vllm:request_queue_time_seconds": _REQUEST,
    "vllm:request_inference_time_seconds": _REQUEST,
    "vllm:request_prefill_time_seconds": _REQUEST,
    "vllm:request_decode_time_seconds": _REQUEST,
    "vllm:iteration_tokens_total": _STEP_TOKENS,
    "vllm:request_prompt_tokens": _TOKENS,
    "vllm:request_generation_tokens": _TOKENS,
    "vllm:request_num_preemptions": _PREEMPTIONS,
    "vllm:request_prefill_kv_computed_tokens": _TOKENS,
    "vllm:request_max_num_generation_tokens": _TOKENS,
    "vllm:request_params_n": _N,
    "vllm:request_params_max_tokens": _TOKENS,
}
FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")


@dataclass
class Histogram:
    bounds: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.bounds) + 1)

    def observe(self, value: float) -> None:
        self.counts[bisect_left(self.bounds, value)] += 1
        self.count += 1
        self.total += value

    def cumulative(self) -> list[int]:
        running = 0
        out: list[int] = []
        for value in self.counts:
            running += value
            out.append(running)
        return out


@dataclass
class EngineStats:
    """Cumulative counters and histograms; gauges come from the engine state."""

    created_s: float
    prefix_queries: int = 0
    prefix_hits: int = 0
    preemptions: int = 0
    prompt_tokens: int = 0
    prompt_tokens_cached: int = 0
    generation_tokens: int = 0
    success: Counter[str] = field(default_factory=Counter)
    histograms: dict[str, Histogram] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, bounds in HISTOGRAM_BUCKETS.items():
            self.histograms.setdefault(name, Histogram(bounds))

    def observe(self, name: str, value: float) -> None:
        self.histograms[name].observe(value)

    def observe_all(self, name: str, values: Sequence[float]) -> None:
        for value in values:
            self.histograms[name].observe(value)


@dataclass(frozen=True)
class StatsSnapshot:
    """What /metrics shows: taken at the end of a step, so it freezes while the
    step loop is paused, as vLLM's scheduler stats do."""

    taken_ns: int
    running: int
    waiting: int
    waiting_capacity: int
    kv_usage: float
    stats: EngineStats


def snapshot(
    *,
    taken_ns: int,
    running: int,
    waiting: int,
    waiting_capacity: int,
    kv_usage: float,
    stats: EngineStats,
) -> StatsSnapshot:
    return StatsSnapshot(
        taken_ns=taken_ns,
        running=running,
        waiting=waiting,
        waiting_capacity=waiting_capacity,
        kv_usage=kv_usage,
        stats=copy.deepcopy(stats),
    )


__all__ = [
    "FINISH_REASONS",
    "HISTOGRAM_BUCKETS",
    "EngineStats",
    "Histogram",
    "StatsSnapshot",
    "snapshot",
]
