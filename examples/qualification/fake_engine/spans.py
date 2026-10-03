"""vLLM's per-request ``llm_request`` span, exported over OTLP/HTTP.

Spans leave in batches on a timer, like the OpenTelemetry batch processor
vLLM uses, as protobuf, the only encoding the SDK's HTTP exporter sends
(OTLP/JSON on request). An incoming W3C ``traceparent`` makes the request's
span its child. A failed export is retried as the SDK's exporter retries it,
then counted and dropped.
"""

from __future__ import annotations

import gzip
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from stormlog.infer.vllm_spans import (
    JSON_MEDIA,
    OTLP_EXTRA_HINT,
    PROTOBUF_MEDIA,
    otlp_protobuf_available,
)

from .config import Controls
from .engine import EngineObserver, FakeRequest
from .identities import Identities

# Bigger than the 32 MiB Stormlog's receiver accepts, raw or inflated.
ABUSIVE_BYTES = 33 * 1024 * 1024
# opentelemetry-exporter-otlp-proto-http 1.44.0: at most six attempts, 2**n s
# apart with 20% jitter, all inside one export timeout.
MAX_ATTEMPTS = 6
# What a post reports when no HTTP answer came. Like requests'
# ConnectionError, a failed connect, send or read is retried; like its
# ReadTimeout, no answer in time after the body went is final.
CONNECTION_ERROR = 0
READ_TIMEOUT = -1


class SpanExporter(EngineObserver):
    def __init__(
        self,
        endpoint: str,
        controls: Controls,
        *,
        interval: float,
        encoding: str = "protobuf",
        timeout: float = 10.0,
        ids: Identities | None = None,
    ) -> None:
        if encoding not in ("protobuf", "json"):
            raise ValueError(f"unknown span encoding {encoding!r}")
        if encoding == "protobuf" and not otlp_protobuf_available():
            raise RuntimeError(OTLP_EXTRA_HINT)
        self.endpoint = endpoint
        self.controls = controls
        self.interval = interval
        self.encoding = encoding
        self.ids = ids or Identities()
        # OTEL_EXPORTER_OTLP_TRACES_TIMEOUT: one export's whole retry window.
        self.timeout = timeout
        self.media = PROTOBUF_MEDIA if encoding == "protobuf" else JSON_MEDIA
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
        if self._thread.ident is not None:  # started
            self._thread.join(timeout=10)
        self._flush(everything=True)

    def on_free(self, request: FakeRequest) -> None:
        if request.finish_reason == "abort":
            # vLLM makes the span when its front end sees the request finish
            # (OutputProcessor.do_tracing); a client abort drops the
            # request's state first, so it never gets one.
            return
        ready = time.time_ns() + int(self.controls.span_delay_seconds * 1e9)
        with self._lock:
            self._pending.append((ready, request_span(request, self.ids)))

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
            connection.putheader("Content-Type", self.media)
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
        ready = self._take_ready(everything)
        if not ready:
            return
        if self.encoding == "protobuf":
            body = _protobuf(ready)
        else:
            body = json.dumps(_document(ready)).encode()
        for _ in range(2 if self.controls.span_duplicates else 1):
            self._export(body)

    def _take_ready(self, everything: bool) -> list[dict[str, Any]]:
        """The spans due now, or every queued one, taken off the queue."""
        now = time.time_ns()
        with self._lock:
            ready = [span for at, span in self._pending if everything or at <= now]
            self._pending = [
                item for item in self._pending if not (everything or item[0] <= now)
            ]
        return ready

    def _export(self, body: bytes) -> None:
        """One batch, as the SDK's exporter sends it: 408, 5xx and connection
        errors are retried with backoff inside the timeout; any other answer,
        429 included, and a read timeout are final. A close abandons the
        retries."""
        deadline = time.monotonic() + self.timeout
        for attempt in range(MAX_ATTEMPTS):
            backoff = 2**attempt * random.uniform(0.8, 1.2)
            status = self._attempt(body, deadline - time.monotonic())
            if 200 <= status < 400:
                return
            if not _retryable(status) or attempt + 1 == MAX_ATTEMPTS:
                break
            if backoff > deadline - time.monotonic() or self._stop.wait(backoff):
                break
        self.failed += 1

    def _attempt(self, body: bytes, timeout: float) -> int:
        status = self._post(body, encoding=None, timeout=timeout)
        if status == CONNECTION_ERROR:
            # The SDK posts once more at once when a kept-alive connection
            # breaks, before counting the attempt as failed.
            status = self._post(body, encoding=None, timeout=timeout)
        return status

    def _post(self, body: bytes, *, encoding: str | None, timeout: float = 10.0) -> int:
        headers = {"Content-Type": self.media}
        if encoding is not None:
            headers["Content-Encoding"] = encoding
        request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=max(timeout, 0.01)) as answer:
                status = int(answer.status)
        except urllib.error.HTTPError as error:
            status = int(error.code)
        except urllib.error.URLError:
            # Raised while connecting or sending.
            status = CONNECTION_ERROR
        except TimeoutError:
            # Raised while waiting for the answer.
            status = READ_TIMEOUT
        except OSError:
            # A reset or close while reading the answer.
            status = CONNECTION_ERROR
        self.statuses.append(status)
        return status


def _retryable(status: int) -> bool:
    """The SDK's rule: a connection error, 408 or any 5xx."""
    return status in (CONNECTION_ERROR, 408) or 500 <= status <= 599


def request_span(request: FakeRequest, ids: Identities | None = None) -> dict[str, Any]:
    ids = ids or Identities()
    trace_id, parent = _parent(request.traceparent, ids)
    end = request.finished_ns or time.time_ns()
    attributes: dict[str, Any] = {
        "gen_ai.request.id": request.external_id,
        "gen_ai.usage.prompt_tokens": request.prompt_len,
        "gen_ai.usage.completion_tokens": request.output_tokens,
        "gen_ai.latency.e2e": (end - request.arrival_ns) / 1e9,
    }
    # vLLM adds each sampling parameter only when it is set and not zero.
    for key, value in (
        ("gen_ai.request.top_p", request.top_p),
        ("gen_ai.request.max_tokens", request.max_tokens),
        ("gen_ai.request.temperature", request.temperature),
        ("gen_ai.request.n", 1),
    ):
        if value:
            attributes[key] = value
    attributes.update(_latencies(request, end))
    return {
        "traceId": trace_id,
        "spanId": ids.hex(8),
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


def _parent(traceparent: str | None, ids: Identities) -> tuple[str, str | None]:
    """The W3C trace ID and parent span ID, or a new trace for a root span."""
    parts = (traceparent or "").split("-")
    if len(parts) == 4 and len(parts[1]) == 32 and len(parts[2]) == 16:
        return parts[1], parts[2]
    return ids.hex(16), None


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


def _protobuf(spans: list[dict[str, Any]]) -> bytes:
    """The same document as the SDK's exporter sends it: an
    ``ExportTraceServiceRequest`` in protobuf."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

    message = ExportTraceServiceRequest()
    resource_spans = message.resource_spans.add()
    resource_spans.resource.attributes.append(
        _key_value(_attribute("service.name", "vllm"))
    )
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "vllm.llm_engine"
    for span in spans:
        scope_spans.spans.append(
            Span(
                trace_id=bytes.fromhex(span["traceId"]),
                span_id=bytes.fromhex(span["spanId"]),
                parent_span_id=bytes.fromhex(span["parentSpanId"]),
                name=span["name"],
                kind=span["kind"],
                start_time_unix_nano=int(span["startTimeUnixNano"]),
                end_time_unix_nano=int(span["endTimeUnixNano"]),
                attributes=[_key_value(item) for item in span["attributes"]],
                status=Status(code=span["status"]["code"]),
            )
        )
    return bytes(message.SerializeToString())


def _key_value(item: dict[str, Any]) -> Any:
    """An OTLP/JSON attribute as the protobuf ``KeyValue``."""
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

    ((kind, value),) = item["value"].items()
    if kind == "intValue":
        wrapped = AnyValue(int_value=int(value))
    elif kind == "stringValue":
        wrapped = AnyValue(string_value=value)
    elif kind == "boolValue":
        wrapped = AnyValue(bool_value=value)
    else:
        wrapped = AnyValue(double_value=value)
    return KeyValue(key=item["key"], value=wrapped)


__all__ = ["ABUSIVE_BYTES", "SpanExporter", "request_span"]
