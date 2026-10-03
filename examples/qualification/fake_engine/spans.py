"""vLLM's per-request ``llm_request`` span, exported as OTLP/HTTP JSON.

Spans leave in batches on a timer, like the OpenTelemetry batch processor
vLLM uses. An incoming W3C ``traceparent`` makes the request's span its
child. A failed export is counted and dropped, as vLLM's exporter drops it
once its retries run out.
"""

from __future__ import annotations

import gzip
import http.client
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .config import Controls
from .engine import EngineObserver, FakeRequest

# Bigger than the 32 MiB Stormlog's receiver accepts, raw or inflated.
ABUSIVE_BYTES = 33 * 1024 * 1024


class SpanExporter(EngineObserver):
    def __init__(self, endpoint: str, controls: Controls, *, interval: float) -> None:
        self.endpoint = endpoint
        self.controls = controls
        self.interval = interval
        self.statuses: list[int] = []
        self.failed = 0
        self._pending: list[tuple[int, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="fake-engine-spans", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        """Send what is still queued, whatever its delay, then stop."""
        self._stop.set()
        self._thread.join(timeout=10)
        self._flush(everything=True)

    def on_free(self, request: FakeRequest) -> None:
        ready = time.time_ns() + int(self.controls.span_delay_seconds * 1e9)
        with self._lock:
            self._pending.append((ready, request_span(request)))

    def send_abusive(self, kind: str) -> int:
        """(f) One export the receiver must refuse: ``oversized`` or ``gzip_bomb``."""
        if kind == "oversized":
            return self._post_declared_oversize()
        if kind == "gzip_bomb":
            return self._post(gzip.compress(b"0" * ABUSIVE_BYTES), encoding="gzip")
        raise ValueError(f"unknown abusive body {kind!r}")

    def _post_declared_oversize(self) -> int:
        """Declare a body over the cap and read the answer before sending it:
        a receiver that checks Content-Length first refuses at once."""
        target = urlparse(self.endpoint)
        connection = http.client.HTTPConnection(
            target.hostname or "127.0.0.1", target.port or 80, timeout=10
        )
        try:
            connection.putrequest("POST", target.path or "/")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(ABUSIVE_BYTES))
            connection.endheaders()
            status = int(connection.getresponse().status)
        except (OSError, http.client.HTTPException):
            status = 0
        finally:
            connection.close()
        self.statuses.append(status)
        return status

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._flush(everything=False)

    def _flush(self, *, everything: bool) -> None:
        now = time.time_ns()
        with self._lock:
            ready = [span for at, span in self._pending if everything or at <= now]
            self._pending = [
                item for item in self._pending if not (everything or item[0] <= now)
            ]
        if not ready:
            return
        body = json.dumps(_document(ready)).encode()
        for _ in range(2 if self.controls.span_duplicates else 1):
            self._post(body, encoding=None)

    def _post(self, body: bytes, *, encoding: str | None) -> int:
        headers = {"Content-Type": "application/json"}
        if encoding is not None:
            headers["Content-Encoding"] = encoding
        request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = int(response.status)
        except urllib.error.HTTPError as error:
            status = int(error.code)
        except (urllib.error.URLError, OSError):
            status = 0
        self.statuses.append(status)
        if status != 200:
            self.failed += 1
        return status


def request_span(request: FakeRequest) -> dict[str, Any]:
    trace_id, parent = _parent(request.traceparent)
    end = request.finished_ns or time.time_ns()
    attributes: dict[str, Any] = {
        "gen_ai.request.id": request.external_id,
        "gen_ai.usage.prompt_tokens": request.prompt_len,
        "gen_ai.usage.completion_tokens": request.output_tokens,
        "gen_ai.request.max_tokens": request.max_tokens,
        "gen_ai.request.n": 1,
        "gen_ai.request.top_p": 1.0,
        "gen_ai.latency.e2e": (end - request.arrival_ns) / 1e9,
    }
    attributes.update(_latencies(request, end))
    return {
        "traceId": trace_id,
        "spanId": secrets.token_hex(8),
        "parentSpanId": parent or "",
        "name": "llm_request",
        "kind": 2,
        "startTimeUnixNano": str(request.arrival_ns),
        "endTimeUnixNano": str(end),
        "attributes": [_attribute(key, value) for key, value in attributes.items()],
        "status": {"code": 0},
    }


def _latencies(request: FakeRequest, end: int) -> dict[str, float]:
    scheduled, first = request.first_scheduled_ns, request.first_token_ns
    if scheduled is None or first is None:
        return {}
    return {
        "gen_ai.latency.time_in_queue": (scheduled - request.arrival_ns) / 1e9,
        "gen_ai.latency.time_to_first_token": (first - request.arrival_ns) / 1e9,
        "gen_ai.latency.time_in_model_prefill": (first - scheduled) / 1e9,
        "gen_ai.latency.time_in_model_decode": (end - first) / 1e9,
        "gen_ai.latency.time_in_model_inference": (end - scheduled) / 1e9,
    }


def _parent(traceparent: str | None) -> tuple[str, str | None]:
    """The W3C trace ID and parent span ID, or a new trace for a root span."""
    parts = (traceparent or "").split("-")
    if len(parts) == 4 and len(parts[1]) == 32 and len(parts[2]) == 16:
        return parts[1], parts[2]
    return secrets.token_hex(16), None


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        wrapped: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, str):
        wrapped = {"stringValue": value}
    elif isinstance(value, int):
        wrapped = {"intValue": str(value)}
    else:
        wrapped = {"doubleValue": float(value)}
    return {"key": key, "value": wrapped}


def _document(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [_attribute("service.name", "vllm")],
                },
                "scopeSpans": [
                    {"scope": {"name": "vllm.llm_engine"}, "spans": spans},
                ],
            }
        ]
    }


__all__ = ["ABUSIVE_BYTES", "SpanExporter", "request_span"]
