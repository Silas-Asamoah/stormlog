"""Configuration models for inference profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class WorkloadCase:
    """One inference profiling workload shape."""

    case_id: str
    concurrency: int
    input_tokens: int
    output_tokens: int


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

    def cases(self) -> list[WorkloadCase]:
        cases: list[WorkloadCase] = []
        for concurrency in self.concurrency:
            for input_tokens in self.input_tokens:
                for output_tokens in self.output_tokens:
                    case_id = f"c{concurrency}_in{input_tokens}_out{output_tokens}"
                    cases.append(
                        WorkloadCase(
                            case_id=case_id,
                            concurrency=concurrency,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )
                    )
        return cases
