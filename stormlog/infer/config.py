"""Configuration models for inference profiling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .arrivals import BURST, CLOSED, RATE_MODES, REPLAY, ArrivalSpec, ArrivalTrace
from .cache_state import RESET_RETRY_SECONDS, UNSPECIFIED
from .prompts import REPEAT, PromptSpec

if TYPE_CHECKING:
    from .slo import SloSpec
    from .trace_capture import TraceCaptureConfig

DEFAULT_ENDPOINT_PATH = "/chat/completions"


def parse_int_list(value: str, *, field_name: str) -> list[int]:
    """Parse a comma-separated positive integer list."""
    parsed: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain integers") from exc
        if number <= 0:
            raise ValueError(f"{field_name} values must be >= 1")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{field_name} must contain at least one value")
    return parsed


def resolve_endpoint(*, endpoint: str | None, base_url: str | None) -> str:
    """Resolve either a full chat-completions endpoint or a /v1 base URL."""
    if endpoint and base_url:
        raise ValueError("Use either --endpoint or --base-url, not both")
    if endpoint:
        return endpoint
    if not base_url:
        raise ValueError("One of --endpoint or --base-url is required")
    return base_url.rstrip("/") + DEFAULT_ENDPOINT_PATH


def parse_float_list(value: str, *, field_name: str) -> list[float]:
    """Parse a comma-separated list of positive, finite numbers."""
    parsed: list[float] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            number = float(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain numbers") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{field_name} values must be > 0")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{field_name} must contain at least one value")
    return parsed


@dataclass(frozen=True)
class WorkloadCase:
    """One inference profiling workload shape.

    For an open-loop case, ``concurrency`` is the in-flight limit: the most
    requests that may be outstanding at once.
    """

    case_id: str
    concurrency: int
    input_tokens: int
    output_tokens: int
    arrival: ArrivalSpec = field(default_factory=ArrivalSpec)


@dataclass(frozen=True)
class ProfileConfig:
    """Resolved configuration for one inference profiling run."""

    endpoint: str
    model: str
    concurrency: tuple[int, ...]
    input_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    output_path: str
    duration_seconds: float | None = None
    request_count: int | None = 1
    stream: bool = True
    stream_include_usage: bool = True
    timeout_seconds: float = 60.0
    warmup_requests: int = 0
    seed: int = 0
    api_key: str | None = None
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    tokenizer: str = "auto"
    tokenizer_model: str | None = None
    tiktoken_encoding: str | None = None
    strict_token_counts: bool = False
    system_sampler: str = "auto"
    sample_interval_seconds: float = 1.0
    run_id: str | None = None
    arrival_mode: str = CLOSED
    rates: tuple[float, ...] = ()
    burst_size: int | None = None
    burst_interval_seconds: float | None = None
    arrival_trace: ArrivalTrace | None = None
    max_in_flight: int = 128
    overflow: Literal["wait", "drop"] = "wait"
    # After the measured window, how long in-flight requests may finish.
    # None means the request timeout.
    drain_timeout_seconds: float | None = None
    prompt_mode: str = REPEAT
    shared_prefix_ratio: float | None = None
    prefix_groups: int | None = None
    cache_state: str = UNSPECIFIED
    # Extra request fields, e.g. {"temperature": 0, "ignore_eos": true}.
    extra_body: dict[str, Any] | None = None
    # POSTed before each case, e.g. vLLM /reset_prefix_cache or SGLang /flush_cache.
    cache_reset_url: str | None = None
    # How long a reset the server refuses (vLLM's success false) is retried.
    cache_reset_timeout_seconds: float = RESET_RETRY_SECONDS
    # vLLM's Prometheus endpoint, scraped at phase boundaries and on a cadence;
    # None leaves native telemetry off.
    vllm_metrics_url: str | None = None
    vllm_metrics_interval_seconds: float = 1.0
    # HOST:PORT for an OTLP/HTTP receiver that collects vLLM's request spans
    # while the profile runs; None runs no receiver.
    vllm_spans_listen: str | None = None
    # How long the receiver keeps listening after the last phase, so the
    # exporter's final batch (flushed every 5 s by default) still arrives.
    vllm_spans_drain_seconds: float = 6.0
    # Optional bounded profiler windows; None records no trace.
    trace: TraceCaptureConfig | None = None
    # The vLLM execution hook's STORMLOG_VLLM_HOOK_DIR as this host sees it;
    # its final steps are imported when the run ends. None imports nothing.
    vllm_execution_dir: Path | None = None
    # The SLO policy to record in the artifact as infer.slo, and whether it
    # came from a file or from --slo flags. None records none.
    slo: SloSpec | None = None
    slo_source: str | None = None
    # What to ask the server about itself before and after the run: auto,
    # basic or none (see server_probe). /server_info is asked of a public
    # host only when allow_remote_probe is set.
    server_probe: str = "auto"
    allow_remote_probe: bool = False

    def prompt_spec(self) -> PromptSpec:
        return PromptSpec(
            mode=self.prompt_mode,
            shared_prefix_ratio=self.shared_prefix_ratio,
            prefix_groups=self.prefix_groups,
        )

    def arrival_specs(self) -> list[ArrivalSpec]:
        """One arrival shape per case group: per rate, or a single shape."""
        if self.arrival_mode in RATE_MODES:
            return [
                ArrivalSpec(mode=self.arrival_mode, rate_per_second=rate)
                for rate in self.rates
            ]
        return [
            ArrivalSpec(
                mode=self.arrival_mode,
                burst_size=self.burst_size if self.arrival_mode == BURST else None,
                burst_interval_seconds=(
                    self.burst_interval_seconds if self.arrival_mode == BURST else None
                ),
                trace=self.arrival_trace if self.arrival_mode == REPLAY else None,
            )
        ]

    def cases(self) -> list[WorkloadCase]:
        cases = self._matrix()
        ids = [case.case_id for case in cases]
        repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if repeated:
            # Two cases with one ID would merge their requests and records.
            raise ValueError(
                f"two workload cases would share the ID {repeated[0]}; list each "
                "concurrency, rate and token length once"
            )
        return cases

    def _matrix(self) -> list[WorkloadCase]:
        if self.arrival_mode == CLOSED:
            return [
                self._case(f"c{concurrency}", concurrency, ArrivalSpec(), tokens)
                for concurrency in self.concurrency
                for tokens in self._token_shapes()
            ]
        return [
            self._case(spec.case_label(), self.max_in_flight, spec, tokens)
            for spec in self.arrival_specs()
            for tokens in self._token_shapes()
        ]

    def _token_shapes(self) -> list[tuple[int, int]]:
        return [(i, o) for i in self.input_tokens for o in self.output_tokens]

    @staticmethod
    def _case(
        label: str, concurrency: int, arrival: ArrivalSpec, tokens: tuple[int, int]
    ) -> WorkloadCase:
        input_tokens, output_tokens = tokens
        return WorkloadCase(
            case_id=f"{label}_in{input_tokens}_out{output_tokens}",
            concurrency=concurrency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            arrival=arrival,
        )
