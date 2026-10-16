"""Versioned audio events, deliberately separate from text/token telemetry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

AUDIO_FAMILIES = frozenset(
    {"infer.audio.request", "infer.audio.workload", "infer.audio.summary"}
)
STATUSES = frozenset(
    {
        "ok",
        "error",
        "timeout",
        "cancelled",
        "dropped",
        "unreachable",
        "delivery_unknown",
        "rejected",
    }
)
OBSERVATIONS = (
    "e2e_latency_ms",
    "first_response_byte_ms",
    "first_audio_ms",
    "first_transcript_ms",
    "input_audio_seconds",
    "output_audio_seconds",
    "rtf",
)


def validate_audio_record(record: dict[str, Any]) -> None:
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported audio event schema version")
    family = record.get("event_type")
    if family not in AUDIO_FAMILIES:
        raise ValueError("Unknown audio event family")
    if record.get("task") not in {"tts", "stt"} or record.get("engine") != "mlx-audio":
        raise ValueError("Invalid audio engine/task identity")
    if not isinstance(record.get("session_id"), str) or not record["session_id"]:
        raise ValueError("Audio event requires session identity")
    if family == "infer.audio.request":
        _request(record)
    elif family == "infer.audio.workload":
        _workload(record)


def _workload(record: dict[str, Any]) -> None:
    if not isinstance(record.get("cases"), list) or not isinstance(
        record.get("fixtures"), list
    ):
        raise ValueError("Audio workload requires cases and fixtures")
    if (
        not isinstance(record.get("workload_digest"), str)
        or len(record["workload_digest"]) != 64
    ):
        raise ValueError("Audio workload requires digest")


def _request(record: dict[str, Any]) -> None:
    if record.get("status") not in STATUSES or record.get("phase") not in {
        "warmup",
        "measured",
    }:
        raise ValueError("Invalid audio request status/phase")
    for name in ("request_id", "case_id", "fixture_id", "fixture_digest"):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f"Missing audio request {name}")
    _request_times(record)
    _observations(record)
    _completion(record)
    allowed = {item.name for item in fields(AudioRequestEvent)} | {
        "schema_version",
        "event_type",
    }
    if set(record) - allowed:
        raise ValueError("Unknown audio request fields")


def _request_times(record: dict[str, Any]) -> None:
    for name in ("started_at_ns", "ended_at_ns", "request_index", "response_bytes"):
        if type(record.get(name)) is not int or record[name] < 0:
            raise ValueError(f"Invalid audio request {name}")
    if record["ended_at_ns"] < record["started_at_ns"]:
        raise ValueError("Audio request ends before it starts")


def _observations(record: dict[str, Any]) -> None:
    for name in OBSERVATIONS:
        value = record.get(name)
        if value is not None and (
            type(value) not in (float, int) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"Invalid audio observation {name}")
    if any(
        key in record
        for key in (
            "prompt_tokens",
            "output_tokens",
            "target_input_tokens",
            "ttft_ms",
            "tpot_ms",
            "text",
            "transcript",
            "audio",
        )
    ):
        raise ValueError("Audio events cannot contain token metrics or raw media/text")


def _completion(record: dict[str, Any]) -> None:
    if record["status"] != "ok" and (
        record.get("rtf") is not None or record.get("output_audio_seconds") is not None
    ):
        raise ValueError(
            "Failed response cannot contribute successful audio duration/RTF"
        )
    if record["status"] == "ok":
        duration = record.get(
            "output_audio_seconds" if record["task"] == "tts" else "input_audio_seconds"
        )
        if duration is None or duration <= 0 or record.get("rtf") is None:
            raise ValueError("Successful audio response requires duration and RTF")


@dataclass(frozen=True)
class AudioRequestEvent:
    session_id: str
    request_id: str
    case_id: str
    phase: str
    task: str
    model: str
    fixture_id: str
    fixture_digest: str
    request_index: int
    started_at_ns: int
    ended_at_ns: int
    status: str
    endpoint: str
    stream: bool
    arrival_mode: str
    intended_at_ns: int
    response_bytes: int = 0
    engine: str = "mlx-audio"
    x_request_id: str | None = None
    concurrency: int | None = None
    max_in_flight: int | None = None
    dispatch_lag_ms: float | None = None
    held_for_slot: bool | None = None
    in_flight_at_dispatch: int | None = None
    e2e_latency_ms: float | None = None
    first_response_byte_ms: float | None = None
    first_audio_ms: float | None = None
    first_transcript_ms: float | None = None
    input_audio_seconds: float | None = None
    output_audio_seconds: float | None = None
    rtf: float | None = None
    response_digest: str | None = None
    audio_segments: list[dict[str, Any]] = field(default_factory=list)
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    retained_media: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "event_type": "infer.audio.request",
            **asdict(self),
        }
        validate_audio_record(record)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> AudioRequestEvent:
        validate_audio_record(record)
        return cls(
            **{
                key: value
                for key, value in record.items()
                if key not in {"schema_version", "event_type"}
            }
        )
