"""Validated audio workload configuration, independent of any ML runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .arrivals import BURST, CLOSED, RATE_MODES, REPLAY, ArrivalSpec, ArrivalTrace
from .engine_adapters import get_adapter
from .http_transport import validate_http_endpoint

TTS_SETTINGS: dict[str, type] = {
    "voice": str,
    "instruct": str,
    "speed": float,
    "gender": str,
    "pitch": float,
    "lang_code": str,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "repetition_penalty": float,
    "streaming_interval": float,
    "max_tokens": int,
}
STT_SETTINGS: dict[str, type] = {
    "language": str,
    "max_tokens": int,
    "chunk_duration": float,
    "frame_threshold": int,
    "context": str,
    "prefill_step_size": int,
    "text": str,
    "word_timestamps": bool,
}


def validate_settings(task: str, settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("Fixture settings must be an object")
    allowed = TTS_SETTINGS if task == "tts" else STT_SETTINGS
    if set(settings) - set(allowed):
        raise ValueError(
            "Unknown or unsupported audio setting (reference audio is not uploaded)"
        )
    for key, value in settings.items():
        _setting_type(key, value, allowed[key])
        _setting_bounds(key, value, allowed[key])
    return dict(settings)


def _setting_type(key: str, value: Any, expected: type) -> None:
    valid = type(value) is expected
    if expected is float:
        valid = type(value) in (float, int) and math.isfinite(value)
    if not valid:
        raise ValueError(f"Invalid type for audio setting {key}")
    if isinstance(value, str) and (not value or len(value.encode()) > 65536):
        raise ValueError(f"Invalid string bound for audio setting {key}")


def _setting_bounds(key: str, value: Any, expected: type) -> None:
    if expected in (int, float) and value < 0:
        raise ValueError(f"Audio setting {key} must be nonnegative")
    positive = {
        "speed",
        "pitch",
        "repetition_penalty",
        "streaming_interval",
        "chunk_duration",
        "prefill_step_size",
        "max_tokens",
    }
    if key in positive and value <= 0:
        raise ValueError(f"Audio setting {key} must be positive")
    if key == "top_p" and value > 1:
        raise ValueError("Audio setting top_p must be in [0, 1]")


@dataclass(frozen=True)
class AudioCase:
    case_id: str
    concurrency: int
    arrival: ArrivalSpec = field(default_factory=ArrivalSpec)


@dataclass(frozen=True)
class AudioProfileConfig:
    endpoint: str
    model: str
    manifest_path: str
    output_path: str
    task: str
    engine: str = "mlx-audio"
    concurrency: tuple[int, ...] = (1,)
    request_count: int | None = 1
    duration_seconds: float | None = None
    warmup_requests: int = 0
    stream: bool = False
    response_format: str | None = None
    timeout_seconds: float = 60
    max_response_bytes: int = 64 * 1024**2
    max_fixture_bytes: int = 32 * 1024**2
    max_manifest_bytes: int = 8 * 1024**2
    max_total_fixture_bytes: int = 64 * 1024**2
    max_audio_seconds: float = 300
    seed: int = 0
    api_key: str | None = None
    run_id: str | None = None
    server_version: str | None = None
    arrival_mode: str = CLOSED
    rates: tuple[float, ...] = ()
    burst_size: int | None = None
    burst_interval_seconds: float | None = None
    arrival_trace: ArrivalTrace | None = None
    max_in_flight: int = 128
    overflow: Literal["wait", "drop"] = "wait"
    drain_timeout_seconds: float | None = None
    retain_media_dir: str | None = None
    max_retained_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        get_adapter(self.engine, self.task)
        validate_http_endpoint(self.endpoint)
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be nonempty")
        fmt = self.response_format or (
            "wav" if self.task == "tts" else "ndjson" if self.stream else "json"
        )
        expected = "wav" if self.task == "tts" else "ndjson" if self.stream else "json"
        if fmt != expected:
            raise ValueError(
                f"{self.task} with stream={self.stream} requires response format {expected}"
            )
        object.__setattr__(self, "response_format", fmt)
        self._validate_bounds()
        self.cases()

    def _validate_bounds(self) -> None:
        for key in (
            "max_response_bytes",
            "max_fixture_bytes",
            "max_manifest_bytes",
            "max_total_fixture_bytes",
            "max_retained_bytes",
            "max_in_flight",
        ):
            _positive_integer(getattr(self, key), key)
        for key in (
            "timeout_seconds",
            "max_audio_seconds",
            "duration_seconds",
            "drain_timeout_seconds",
        ):
            _positive_number(getattr(self, key), key)
        if self.request_count is not None:
            _positive_integer(self.request_count, "requests")
        _nonnegative_integer(self.warmup_requests, "warmup")
        self._validate_policy()
        if not self.concurrency:
            raise ValueError("concurrency must contain positive integers")
        for concurrency in self.concurrency:
            _positive_integer(concurrency, "concurrency")

    def _validate_policy(self) -> None:
        if (
            self.request_count is None
            and self.duration_seconds is None
            and self.arrival_mode != REPLAY
        ):
            raise ValueError("Specify requests or duration")
        if self.overflow not in {"wait", "drop"}:
            raise ValueError("overflow must be wait or drop")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    def cases(self) -> list[AudioCase]:
        if self.arrival_mode == CLOSED:
            cases = [AudioCase(f"c{c}_{self.task}", c) for c in self.concurrency]
        else:
            cases = [
                AudioCase(f"{spec.case_label()}_{self.task}", self.max_in_flight, spec)
                for spec in self._arrival_specs()
            ]
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("Duplicate audio cases")
        return cases

    def _arrival_specs(self) -> list[ArrivalSpec]:
        options: dict[str, Any] = dict(
            mode=self.arrival_mode,
            burst_size=self.burst_size if self.arrival_mode == BURST else None,
            burst_interval_seconds=(
                self.burst_interval_seconds if self.arrival_mode == BURST else None
            ),
            trace=self.arrival_trace if self.arrival_mode == REPLAY else None,
        )
        if self.arrival_mode not in RATE_MODES:
            return [ArrivalSpec(**options)]
        if not self.rates:
            raise ValueError("Rate arrival requires at least one rate")
        return [ArrivalSpec(**options, rate_per_second=rate) for rate in self.rates]

    @property
    def drain_timeout(self) -> float:
        return (
            self.drain_timeout_seconds
            if self.drain_timeout_seconds is not None
            else self.timeout_seconds
        )


def _nonnegative_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value: Any, name: str) -> None:
    if value is not None and (
        type(value) not in (float, int) or not math.isfinite(value) or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
