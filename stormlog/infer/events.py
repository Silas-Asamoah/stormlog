"""Inference profiling event records and JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

INFER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InferenceRequestEvent:
    """Client-observed request trace for one chat completion."""

    session_id: str
    request_id: str
    case_id: str
    phase: str
    started_at_ns: int
    ended_at_ns: int
    endpoint: str
    model: str
    concurrency: int
    target_input_tokens: int
    target_output_tokens: int
    stream: bool
    status: str
    e2e_latency_ms: float | None
    ttft_ms: float | None
    first_chunk_latency_ms: float | None
    chunk_interarrival_ms: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    prompt_token_source: str = "unknown"
    prompt_token_exact: bool = False
    output_tokens: int | None = None
    output_token_source: str = "unknown"
    output_token_exact: bool = False
    total_tokens: int | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "schema_version": INFER_SCHEMA_VERSION,
                "event_type": "infer.request",
                "timestamp_ns": self.started_at_ns,
            }
        )
        return record


@dataclass(frozen=True)
class InferenceSystemSample:
    """Best-effort telemetry from the machine running the endpoint client."""

    session_id: str
    timestamp_ns: int
    sampler: str
    device_id: int | None = None
    device_name: str | None = None
    device_used_bytes: int | None = None
    device_free_bytes: int | None = None
    device_total_bytes: int | None = None
    gpu_utilization_percent: float | None = None
    process_rss_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_scope: str = "client_local"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "schema_version": INFER_SCHEMA_VERSION,
                "event_type": "infer.system_sample",
            }
        )
        return record


@dataclass(frozen=True)
class InferenceSummaryEvent:
    """Aggregate summary emitted at the end of a profiling run."""

    session_id: str
    timestamp_ns: int
    summary: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": INFER_SCHEMA_VERSION,
            "event_type": "infer.summary",
            "session_id": self.session_id,
            "timestamp_ns": self.timestamp_ns,
            "summary": self.summary,
        }


class JsonlEventWriter:
    """Append inference profiling records to a JSONL artifact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None

    def __enter__(self) -> "JsonlEventWriter":
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def append(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlEventWriter is not open")
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
