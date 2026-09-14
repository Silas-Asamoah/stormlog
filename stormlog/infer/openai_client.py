"""OpenAI-compatible Chat Completions client used by inference profiling."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal


@dataclass(frozen=True)
class ChatCompletionResult:
    """Parsed response and client-observed timing metadata."""

    text: str
    started_at_ns: int
    ended_at_ns: int
    e2e_latency_ms: float
    ttft_ms: float | None
    first_chunk_latency_ms: float | None
    chunk_interarrival_ms: list[float] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class OpenAIChatCompletionsClient:
    """Minimal OpenAI-compatible Chat Completions HTTP client."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
        max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
    ) -> None:
        _validate_http_endpoint(endpoint)
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.max_tokens_field = max_tokens_field

    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        stream: bool,
        stream_include_usage: bool,
    ) -> ChatCompletionResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            self.max_tokens_field: output_tokens,
        }
        if stream and stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        _validate_http_endpoint(self.endpoint)
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        started_at_ns = time.time_ns()
        started_perf = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                if stream:
                    return self._read_streaming_response(
                        response=response,
                        started_at_ns=started_at_ns,
                        started_perf=started_perf,
                    )
                return self._read_json_response(
                    response=response,
                    started_at_ns=started_at_ns,
                    started_perf=started_perf,
                )
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {message}") from exc

    def _read_json_response(
        self,
        *,
        response: Any,
        started_at_ns: int,
        started_perf: float,
    ) -> ChatCompletionResult:
        payload = json.loads(response.read().decode("utf-8"))
        ended_at_ns = time.time_ns()
        ended_perf = time.perf_counter()
        choice = _first_choice(payload)
        message = choice.get("message") if isinstance(choice, dict) else None
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        finish_reason = (
            str(choice.get("finish_reason"))
            if isinstance(choice, dict) and choice.get("finish_reason") is not None
            else None
        )
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        return ChatCompletionResult(
            text=text,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            e2e_latency_ms=(ended_perf - started_perf) * 1000.0,
            ttft_ms=None,
            first_chunk_latency_ms=None,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _read_streaming_response(
        self,
        *,
        response: Any,
        started_at_ns: int,
        started_perf: float,
    ) -> ChatCompletionResult:
        content_parts: list[str] = []
        first_chunk_perf: float | None = None
        first_content_perf: float | None = None
        previous_content_perf: float | None = None
        chunk_interarrival_ms: list[float] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None

        for data in _stream_data(response):
            now_perf = time.perf_counter()
            if first_chunk_perf is None:
                first_chunk_perf = now_perf

            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choice = _first_choice(chunk)
            chunk_finish_reason, piece = _stream_choice_fields(choice)
            if chunk_finish_reason is not None:
                finish_reason = chunk_finish_reason
            if not piece:
                continue
            content_parts.append(piece)
            if first_content_perf is None:
                first_content_perf = now_perf
            if previous_content_perf is not None:
                chunk_interarrival_ms.append(
                    (now_perf - previous_content_perf) * 1000.0
                )
            previous_content_perf = now_perf

        ended_at_ns = time.time_ns()
        ended_perf = time.perf_counter()
        return ChatCompletionResult(
            text="".join(content_parts),
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            e2e_latency_ms=(ended_perf - started_perf) * 1000.0,
            ttft_ms=(
                (first_content_perf - started_perf) * 1000.0
                if first_content_perf is not None
                else None
            ),
            first_chunk_latency_ms=(
                (first_chunk_perf - started_perf) * 1000.0
                if first_chunk_perf is not None
                else None
            ),
            chunk_interarrival_ms=chunk_interarrival_ms,
            usage=usage,
            finish_reason=finish_reason,
        )


def _first_choice(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, dict) else {}


def _validate_http_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must use http:// or https://")


def _stream_data(response: Any) -> Iterator[str]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        yield data


def _stream_choice_fields(choice: dict[str, Any]) -> tuple[str | None, str]:
    finish_reason: str | None = None
    if isinstance(choice, dict) and choice.get("finish_reason") is not None:
        finish_reason = str(choice.get("finish_reason"))
    delta = choice.get("delta") if isinstance(choice, dict) else None
    piece = ""
    if isinstance(delta, dict):
        piece = str(delta.get("content") or "")
    return finish_reason, piece
