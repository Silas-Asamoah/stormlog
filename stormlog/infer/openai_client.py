"""OpenAI-compatible Chat Completions client used by inference profiling."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal

from .http_transport import (
    BoundedResponse,
)
from .http_transport import ConnectError as ConnectError
from .http_transport import CutResponseError as CutResponseError
from .http_transport import EndpointHTTPError as EndpointHTTPError
from .http_transport import NoResponseError as NoResponseError
from .http_transport import ignored_proxies as ignored_proxies
from .http_transport import inference_opener as inference_opener
from .http_transport import safe_error as safe_error
from .http_transport import validate_http_endpoint as _validate_http_endpoint

__all__ = [
    "ChatCompletionResult",
    "OpenAIChatCompletionsClient",
    "ConnectError",
    "CutResponseError",
    "NoResponseError",
    "EndpointHTTPError",
    "inference_opener",
    "ignored_proxies",
    "validate_extra_body",
]


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
    response_id: str | None = None
    system_fingerprint: str | None = None
    # The response's HTTP status, as received.
    http_status: int | None = None


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
        extra_body: dict[str, Any] | None = None,
        max_response_bytes: int = 16 * 1024**2,
        task: str = "chat",
    ) -> None:
        _validate_http_endpoint(endpoint)
        if task not in {"chat", "completion"}:
            raise ValueError("Text client requires chat or completion task")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("response byte limit must be a positive integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("request timeout must be positive and finite")
        self.task = task
        self.max_response_bytes = max_response_bytes
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.max_tokens_field = max_tokens_field
        self.extra_body = validate_extra_body(extra_body, max_tokens_field, task=task)
        self._opener = inference_opener()

    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        stream: bool,
        stream_include_usage: bool,
        request_id: str | None = None,
        headers: dict[str, str] | None = None,
        on_sent: Callable[[int], None] | None = None,
        on_first_content: Callable[[int], None] | None = None,
    ) -> ChatCompletionResult:
        """Send one chat completion.

        ``request_id`` goes out as ``X-Request-Id``, which vLLM embeds in its
        own request id and in the ``gen_ai.request.id`` of the request span.
        ``headers`` adds others, such as ``traceparent``; they cannot replace
        the ones the client sets.
        ``on_sent`` is called with the send's wall stamp just before the
        request goes out, and ``on_first_content`` with the first streamed
        content piece's arrival on that clock (the send stamp plus the TTFT);
        both run on the calling thread, so they must return quickly.
        """
        payload = {
            **self.extra_body,
            "model": self.model,
            **self._prompt_payload(prompt),
            "stream": stream,
            self.max_tokens_field: output_tokens,
        }
        if stream and stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload).encode("utf-8")

        _validate_http_endpoint(self.endpoint)
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=self._request_headers(stream, request_id, headers),
            method="POST",
        )

        started_at_ns = time.time_ns()
        started_perf = time.perf_counter()
        if on_sent is not None:
            on_sent(started_at_ns)
        try:
            with self._opener.open(
                request,
                timeout=self.timeout_seconds,
            ) as raw_response:
                with BoundedResponse(
                    raw_response,
                    deadline=started_perf + self.timeout_seconds,
                    max_bytes=self.max_response_bytes,
                ) as response:
                    return self._parse_response(
                        response,
                        stream,
                        started_at_ns,
                        started_perf,
                        on_first_content=on_first_content,
                    )
        except urllib.error.HTTPError as exc:
            try:
                with BoundedResponse(
                    exc, deadline=started_perf + self.timeout_seconds, max_bytes=65536
                ) as response:
                    message = safe_error(
                        response.read().decode("utf-8", errors="replace"),
                        secrets=(self.api_key,),
                    )
            except Exception:
                message = "Error response body unavailable or exceeded limits"
            finally:
                exc.close()
            raise EndpointHTTPError(exc.code, message) from exc

    def _prompt_payload(self, prompt: str) -> dict[str, Any]:
        return (
            {"prompt": prompt}
            if self.task == "completion"
            else {"messages": [{"role": "user", "content": prompt}]}
        )

    def _parse_response(
        self,
        response: Any,
        stream: bool,
        started_at_ns: int,
        started_perf: float,
        *,
        on_first_content: Callable[[int], None] | None = None,
    ) -> ChatCompletionResult:
        if stream:
            return self._read_streaming_response(
                response=response,
                started_at_ns=started_at_ns,
                started_perf=started_perf,
                on_first_content=on_first_content,
            )
        return self._read_json_response(
            response=response,
            started_at_ns=started_at_ns,
            started_perf=started_perf,
        )

    def _request_headers(
        self, stream: bool, request_id: str | None, headers: dict[str, str] | None
    ) -> dict[str, str]:
        """The client's own headers, then any of ``headers`` they leave."""
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        if request_id:
            request_headers["X-Request-Id"] = request_id
        for name, value in (headers or {}).items():
            request_headers.setdefault(name, value)
        return request_headers

    def _read_json_response(
        self,
        *,
        response: Any,
        started_at_ns: int,
        started_perf: float,
    ) -> ChatCompletionResult:
        payload = json.loads(response.read().decode("utf-8"))
        _check_payload(payload, secrets=(self.api_key,))
        ended_at_ns = time.time_ns()
        ended_perf = time.perf_counter()
        choice = _first_choice(payload)
        text = self._choice_text(choice, stream=False)
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
            response_id=_optional_string(payload, "id"),
            system_fingerprint=_optional_string(payload, "system_fingerprint"),
            http_status=_status(response),
        )

    def _read_streaming_response(
        self,
        *,
        response: Any,
        started_at_ns: int,
        started_perf: float,
        on_first_content: Callable[[int], None] | None = None,
    ) -> ChatCompletionResult:
        content_parts: list[str] = []
        first_chunk_perf: float | None = None
        first_content_perf: float | None = None
        previous_content_perf: float | None = None
        chunk_interarrival_ms: list[float] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        response_id = fingerprint = None

        for data in _stream_data(response):
            now_perf = time.perf_counter()
            if first_chunk_perf is None:
                first_chunk_perf = now_perf

            chunk = json.loads(data)
            _check_payload(chunk, secrets=(self.api_key,))
            response_id = _latest_string(chunk, "id", response_id)
            fingerprint = _latest_string(chunk, "system_fingerprint", fingerprint)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choice = _first_choice(chunk)
            chunk_finish_reason, _legacy_piece = _stream_choice_fields(choice)
            piece = self._choice_text(choice, stream=True)
            if chunk_finish_reason is not None:
                finish_reason = chunk_finish_reason
            if not piece:
                continue
            content_parts.append(piece)
            if first_content_perf is None:
                first_content_perf = now_perf
                if on_first_content is not None:
                    on_first_content(
                        started_at_ns + round((now_perf - started_perf) * 1e9)
                    )
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
            ttft_ms=_elapsed_ms(started_perf, first_content_perf),
            first_chunk_latency_ms=_elapsed_ms(started_perf, first_chunk_perf),
            chunk_interarrival_ms=chunk_interarrival_ms,
            usage=usage,
            finish_reason=finish_reason,
            response_id=response_id,
            system_fingerprint=fingerprint,
            http_status=_status(response),
        )

    def _choice_text(self, choice: dict[str, Any], *, stream: bool) -> str:
        if self.task == "completion":
            return str(choice.get("text") or "")
        value = choice.get("delta" if stream else "message")
        return str(value.get("content") or "") if isinstance(value, dict) else ""


def validate_extra_body(
    extra_body: dict[str, Any] | None, max_tokens_field: str, *, task: str = "chat"
) -> dict[str, Any]:
    """Extra request fields may add settings but not replace the ones Stormlog sets."""
    if extra_body is None:
        return {}
    if not isinstance(extra_body, dict):
        raise ValueError("extra request fields must be a JSON object")
    owned = {
        "model",
        "messages",
        "prompt",
        "stream",
        "stream_options",
        max_tokens_field,
    }
    clashes = sorted(owned & set(extra_body))
    if clashes:
        raise ValueError(f"extra request fields cannot set {', '.join(clashes)}")
    return dict(extra_body)


def _first_choice(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, dict) else {}


def _status(response: Any) -> int | None:
    status = getattr(response, "status", None)
    return status if isinstance(status, int) else None


def _elapsed_ms(started_perf: float, perf: float | None) -> float | None:
    return None if perf is None else (perf - started_perf) * 1000.0


def _stream_data(response: Any) -> Iterator[str]:
    for data in _sse_frames(response):
        if data == "[DONE]":
            return
        yield data


def _data_lines(response: Any) -> Iterator[str | None]:
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            yield None
        elif line.startswith("data:"):
            yield line[5:].lstrip(" ")


def _sse_frames(response: Any) -> Iterator[str]:
    parts: list[str] = []
    for fragment in _data_lines(response):
        if fragment is None:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif fragment == "[DONE]":
            if parts:
                yield "\n".join(parts)
            yield fragment
            return
        else:
            # Older compatible servers omit separators after complete JSON.
            if parts and _complete_frame("\n".join(parts)):
                yield "\n".join(parts)
                parts = []
            parts.append(fragment)
    if parts:
        yield "\n".join(parts)


def _stream_choice_fields(choice: dict[str, Any]) -> tuple[str | None, str]:
    finish_reason: str | None = None
    if isinstance(choice, dict) and choice.get("finish_reason") is not None:
        finish_reason = str(choice.get("finish_reason"))
    delta = choice.get("delta") if isinstance(choice, dict) else None
    piece = ""
    if isinstance(delta, dict):
        piece = str(delta.get("content") or "")
    return finish_reason, piece


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _check_payload(payload: Any, *, secrets: tuple[str | None, ...]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON response object")
    if payload.get("error") is not None:
        raise EndpointHTTPError(200, safe_error(str(payload["error"]), secrets=secrets))


def _complete_frame(data: str) -> bool:
    if data == "[DONE]":
        return True
    try:
        json.loads(data)
        return True
    except ValueError:
        return False


def _latest_string(
    payload: dict[str, Any], key: str, previous: str | None
) -> str | None:
    return _optional_string(payload, key) or previous
