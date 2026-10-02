"""Ingest vLLM's OpenTelemetry request spans.

Spans arrive in three ways: an OTLP/HTTP receiver that ``stormlog infer
profile`` runs for the length of a run, an OTLP JSON file written by a
collector's file exporter, or the one-span-per-line JSONL that a small sink
writes. Every path produces the same ``infer.vllm_span`` record with the
native attributes untouched.

The receiver decodes OTLP protobuf with the wire format alone, so no
OpenTelemetry package is needed on the client. The latency attributes a span
carries are phase residency measured on the engine's clock; the mapping to
the v2 correlation model keeps them as reported durations and marks the
derived stage windows as estimates.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .correlation_events import (
    CapabilityEvent,
    CorrelationContext,
    CorrelationEvent,
    EntityRef,
    RequestEvent,
    StageEvent,
)
from .host_clock import wall_clock_domain
from .vllm_telemetry import (
    SPAN_SOURCE_JSONL,
    SPAN_SOURCE_OTLP_JSON,
    SPAN_SOURCE_RECEIVER,
    VllmSpanRecord,
    request_id_from_span_id,
)

OTLP_TRACES_PATH = "/v1/traces"
PROTOBUF_MEDIA = "application/x-protobuf"
JSON_MEDIA = "application/json"
DEFAULT_SPANS_LISTEN = "127.0.0.1:4318"
CAPABILITY_COMPONENT = "vllm.spans"
SPAN_KINDS = {
    0: "UNSPECIFIED",
    1: "INTERNAL",
    2: "SERVER",
    3: "CLIENT",
    4: "PRODUCER",
    5: "CONSUMER",
}
STATUS_CODES = {0: "UNSET", 1: "OK", 2: "ERROR"}

# Stage names and the native attribute that holds each duration, in order.
STAGE_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("queue", "gen_ai.latency.time_in_queue"),
    ("prefill", "gen_ai.latency.time_in_model_prefill"),
    ("decode", "gen_ai.latency.time_in_model_decode"),
)
INFERENCE_ATTRIBUTE = "gen_ai.latency.time_in_model_inference"
REQUEST_ID_ATTRIBUTE = "gen_ai.request.id"
SPAN_CAPABILITIES = (
    "otlp_http_protobuf",
    "otlp_http_json",
    "otlp_json_file",
    "jsonl_file",
)
_RECEIVER_CAPABILITIES = {
    PROTOBUF_MEDIA: "otlp_http_protobuf",
    JSON_MEDIA: "otlp_http_json",
}
_DETAILED_TRACE_NOTE = {
    "latency_attributes": "phase residency on the engine clock, not GPU time",
    "time_in_model_forward": "unsupported: never set by vLLM 0.30.0",
    "time_in_model_execute": "unsupported: never set by vLLM 0.30.0",
}


@dataclass(frozen=True)
class RawSpan:
    """A decoded span before it is stamped with run identity."""

    name: str
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    kind: str | None = None
    start_unix_ns: int | None = None
    end_unix_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    resource: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] | None = None
    dropped: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ protobuf wire
class ProtobufDecodeError(ValueError):
    """The bytes are not an OTLP trace export request."""


Payload = bytes | int


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def done(self) -> bool:
        return self.position >= len(self.data)

    def varint(self) -> int:
        result = 0
        shift = 0
        while True:
            if self.position >= len(self.data):
                raise ProtobufDecodeError("truncated varint")
            byte = self.data[self.position]
            self.position += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 70:
                raise ProtobufDecodeError("varint too long")

    def read(self, length: int) -> bytes:
        if length < 0 or self.position + length > len(self.data):
            raise ProtobufDecodeError("truncated field")
        chunk = self.data[self.position : self.position + length]
        self.position += length
        return chunk

    def fields(self) -> Iterable[tuple[int, Payload]]:
        """Yield (field number, payload); groups and unknown types are errors."""
        while not self.done():
            key = self.varint()
            number, wire = key >> 3, key & 0x7
            if wire == 0:
                yield number, self.varint()
            elif wire == 1:
                yield number, self.read(8)
            elif wire == 2:
                yield number, self.read(self.varint())
            elif wire == 5:
                yield number, self.read(4)
            else:
                raise ProtobufDecodeError(f"unsupported wire type {wire}")


def _as_bytes(payload: Payload) -> bytes:
    if not isinstance(payload, bytes):
        raise ProtobufDecodeError("expected a length-delimited field")
    return payload


def _as_int(payload: Payload) -> int:
    if isinstance(payload, bytes):
        if len(payload) == 8:
            return int(struct.unpack("<Q", payload)[0])
        raise ProtobufDecodeError("expected a varint field")
    return payload


def _text(payload: Payload) -> str:
    return _as_bytes(payload).decode("utf-8", errors="replace")


def _decode_any_value(data: bytes) -> Any:
    """An ``AnyValue``: the one field set says which kind of value it holds."""
    for number, payload in _Reader(data).fields():
        if number == 5:
            return [
                _decode_any_value(_as_bytes(item))
                for num, item in _Reader(_as_bytes(payload)).fields()
                if num == 1
            ]
        if number == 6:
            return _decode_key_value_list(_as_bytes(payload))
        return _decode_scalar_value(number, payload)
    return None


def _decode_scalar_value(number: int, payload: Payload) -> Any:
    if number == 1:
        return _text(payload)
    if number == 2:
        return bool(_as_int(payload))
    if number == 3:
        value = _as_int(payload)
        return value - (1 << 64) if value >= 1 << 63 else value
    if number == 4:
        return float(struct.unpack("<d", _as_bytes(payload))[0])
    if number == 7:
        return _as_bytes(payload).hex()
    return None


def _decode_key_value(data: bytes) -> tuple[str, Any]:
    """One ``KeyValue`` message: a key and an ``AnyValue``."""
    key, value = "", None
    for number, payload in _Reader(data).fields():
        if number == 1:
            key = _text(payload)
        elif number == 2:
            value = _decode_any_value(_as_bytes(payload))
    return key, value


def _decode_key_value_list(data: bytes) -> dict[str, Any]:
    """A ``KeyValueList`` message, whose field 1 repeats ``KeyValue``."""
    values: dict[str, Any] = {}
    for number, payload in _Reader(data).fields():
        if number == 1:
            key, value = _decode_key_value(_as_bytes(payload))
            values[key] = value
    return values


def _decode_status(data: bytes) -> dict[str, Any]:
    status: dict[str, Any] = {"code": "UNSET"}
    for number, payload in _Reader(data).fields():
        if number == 2:
            status["message"] = _text(payload)
        elif number == 3:
            code = _as_int(payload)
            status["code"] = STATUS_CODES.get(code, str(code))
    return status


_SPAN_ID_FIELDS = {1: "trace_id", 2: "span_id", 4: "parent_span_id"}
_SPAN_DROPPED_FIELDS = {10: "attributes", 12: "events", 14: "links"}


def _decode_span(
    data: bytes, resource: dict[str, Any], scope: dict[str, Any]
) -> RawSpan:
    attributes: dict[str, Any] = {}
    dropped: dict[str, int] = {}
    values: dict[str, Any] = {}
    for number, payload in _Reader(data).fields():
        if number in _SPAN_ID_FIELDS:
            values[_SPAN_ID_FIELDS[number]] = _as_bytes(payload).hex() or None
        elif number in _SPAN_DROPPED_FIELDS:
            dropped[_SPAN_DROPPED_FIELDS[number]] = _as_int(payload)
        elif number == 9:
            key, value = _decode_key_value(_as_bytes(payload))
            attributes[key] = value
        else:
            _decode_span_scalar(number, payload, values)
    if not values.get("name"):
        raise ProtobufDecodeError("span without a name")
    return RawSpan(
        resource=resource, scope=scope, attributes=attributes, dropped=dropped, **values
    )


def _decode_span_scalar(number: int, payload: Payload, values: dict[str, Any]) -> None:
    if number == 5:
        values["name"] = _text(payload)
    elif number == 6:
        kind = _as_int(payload)
        values["kind"] = SPAN_KINDS.get(kind, str(kind))
    elif number == 7:
        values["start_unix_ns"] = _as_int(payload)
    elif number == 8:
        values["end_unix_ns"] = _as_int(payload)
    elif number == 15:
        values["status"] = _decode_status(_as_bytes(payload))


def _decode_scope(data: bytes) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    for number, payload in _Reader(data).fields():
        if number == 1:
            scope["name"] = _text(payload)
        elif number == 2:
            scope["version"] = _text(payload)
    return scope


def _decode_resource(data: bytes) -> dict[str, Any]:
    """A ``Resource`` message, whose field 1 repeats ``KeyValue``."""
    return _decode_key_value_list(data)


def _decode_scope_spans(data: bytes, resource: dict[str, Any]) -> list[RawSpan]:
    scope: dict[str, Any] = {}
    span_bytes: list[bytes] = []
    for number, payload in _Reader(data).fields():
        if number == 1:
            scope = _decode_scope(_as_bytes(payload))
        elif number == 2:
            span_bytes.append(_as_bytes(payload))
    return [_decode_span(item, resource, scope) for item in span_bytes]


def _decode_resource_spans(data: bytes) -> list[RawSpan]:
    resource: dict[str, Any] = {}
    scope_bytes: list[bytes] = []
    for number, payload in _Reader(data).fields():
        if number == 1:
            resource = _decode_resource(_as_bytes(payload))
        elif number == 2:
            scope_bytes.append(_as_bytes(payload))
    spans: list[RawSpan] = []
    for item in scope_bytes:
        spans.extend(_decode_scope_spans(item, resource))
    return spans


def decode_otlp_protobuf(data: bytes) -> list[RawSpan]:
    """Decode an ``ExportTraceServiceRequest`` without generated classes."""
    spans: list[RawSpan] = []
    for number, payload in _Reader(data).fields():
        if number == 1:
            spans.extend(_decode_resource_spans(_as_bytes(payload)))
    return spans


# ------------------------------------------------------------------ OTLP JSON
def _json_any_value(value: Any) -> Any:
    """Unwrap OTLP JSON's ``{"stringValue": ...}`` style values."""
    if not isinstance(value, dict):
        return value
    for key, raw in value.items():
        if key in {"stringValue", "boolValue", "doubleValue", "bytesValue"}:
            return raw
        if key == "intValue":
            return int(raw)
        if key == "arrayValue":
            return [_json_any_value(item) for item in raw.get("values", [])]
        if key == "kvlistValue":
            return _json_key_values(raw.get("values", []))
    return value


def _json_key_values(items: Any) -> dict[str, Any]:
    if isinstance(items, dict):
        return {str(k): v for k, v in items.items()}
    values: dict[str, Any] = {}
    for item in items or []:
        if isinstance(item, dict) and "key" in item:
            values[str(item["key"])] = _json_any_value(item.get("value"))
    return values


def _json_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_kind(kind: Any) -> str | None:
    if isinstance(kind, int):
        return SPAN_KINDS.get(kind, str(kind))
    return str(kind) if kind is not None else None


def _json_status(status: Any) -> dict[str, Any] | None:
    if not isinstance(status, dict):
        return None
    code = status.get("code")
    if isinstance(code, int):
        return {**status, "code": STATUS_CODES.get(code, str(code))}
    return dict(status)


def _json_dropped(raw: dict[str, Any]) -> dict[str, int]:
    keys = (
        ("attributes", "droppedAttributesCount"),
        ("events", "droppedEventsCount"),
        ("links", "droppedLinksCount"),
    )
    return {name: int(raw[key]) for name, key in keys if isinstance(raw.get(key), int)}


def _json_span(
    raw: dict[str, Any], resource: dict[str, Any], scope: dict[str, Any]
) -> RawSpan:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("span without a name")
    return RawSpan(
        name=name,
        trace_id=raw.get("traceId") or None,
        span_id=raw.get("spanId") or None,
        parent_span_id=raw.get("parentSpanId") or None,
        kind=_json_kind(raw.get("kind")),
        start_unix_ns=_json_int(raw.get("startTimeUnixNano")),
        end_unix_ns=_json_int(raw.get("endTimeUnixNano")),
        attributes=_json_key_values(raw.get("attributes")),
        resource=resource,
        scope=scope,
        status=_json_status(raw.get("status")),
        dropped=_json_dropped(raw),
    )


def _json_scope_spans(
    scope_spans: dict[str, Any], resource: dict[str, Any]
) -> list[RawSpan]:
    scope = {
        k: v
        for k, v in (scope_spans.get("scope") or {}).items()
        if k in {"name", "version"}
    }
    return [_json_span(raw, resource, scope) for raw in scope_spans.get("spans") or []]


def decode_otlp_json(document: Any) -> list[RawSpan]:
    """Decode the JSON form of ``ExportTraceServiceRequest`` (file exporter output)."""
    if not isinstance(document, dict) or "resourceSpans" not in document:
        raise ValueError("not an OTLP JSON trace document")
    spans: list[RawSpan] = []
    for resource_spans in document.get("resourceSpans") or []:
        resource = _json_key_values(
            (resource_spans.get("resource") or {}).get("attributes")
        )
        for scope_spans in resource_spans.get("scopeSpans") or []:
            spans.extend(_json_scope_spans(scope_spans, resource))
    return spans


def decode_jsonl_span(raw: dict[str, Any]) -> RawSpan:
    """One line of the sink format: name, start/end unix ns, flat attributes."""
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("span without a name")
    attributes = raw.get("attributes")
    return RawSpan(
        name=name,
        trace_id=raw.get("trace_id"),
        span_id=raw.get("span_id"),
        parent_span_id=raw.get("parent_span_id"),
        kind=raw.get("kind"),
        start_unix_ns=_json_int(raw.get("start_unix_ns")),
        end_unix_ns=_json_int(raw.get("end_unix_ns")),
        attributes=dict(attributes) if isinstance(attributes, dict) else {},
        resource=dict(raw.get("resource") or {}),
        scope=dict(raw.get("scope") or {}),
        status=raw.get("status") if isinstance(raw.get("status"), dict) else None,
    )


def read_span_file(path: str | Path) -> tuple[str, list[RawSpan]]:
    """Read OTLP JSON (one document, or one per line) or sink JSONL; say which."""
    stripped = Path(path).read_text(encoding="utf-8").strip()
    if not stripped:
        raise ValueError("span file is empty")
    try:
        document = json.loads(stripped)
    except ValueError:
        document = None
    if isinstance(document, dict) and "resourceSpans" in document:
        return SPAN_SOURCE_OTLP_JSON, decode_otlp_json(document)
    return _read_span_lines(stripped.splitlines())


def _read_span_lines(lines: list[str]) -> tuple[str, list[RawSpan]]:
    spans: list[RawSpan] = []
    source = SPAN_SOURCE_JSONL
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("record must be an object")
            if "resourceSpans" in raw:
                source = SPAN_SOURCE_OTLP_JSON
                spans.extend(decode_otlp_json(raw))
            else:
                spans.append(decode_jsonl_span(raw))
        except ValueError as exc:
            raise ValueError(f"invalid span line {line_number}: {exc}") from exc
    return source, spans


# ------------------------------------------------------------------ records
def span_clock_domain(resource: dict[str, Any], fallback_host: str) -> str:
    """The exporter's wall clock, named by its host; never a shared clock."""
    host = resource.get("host.name")
    name = host if isinstance(host, str) and host else fallback_host
    return wall_clock_domain(name.replace("/", "_") or "unknown-host", None)


def span_record(
    raw: RawSpan,
    *,
    session_id: str,
    run_id: str,
    source: str,
    clock_domain: str,
    received_at_ns: int | None = None,
) -> VllmSpanRecord:
    return VllmSpanRecord(
        session_id=session_id,
        run_id=run_id,
        source=source,
        name=raw.name,
        clock_domain=clock_domain,
        received_at_ns=received_at_ns,
        trace_id=raw.trace_id,
        span_id=raw.span_id,
        parent_span_id=raw.parent_span_id,
        kind=raw.kind,
        start_unix_ns=raw.start_unix_ns,
        end_unix_ns=raw.end_unix_ns,
        attributes=dict(raw.attributes),
        resource=dict(raw.resource),
        scope=dict(raw.scope),
        status=raw.status,
        dropped=dict(raw.dropped),
        request_id=request_id_from_span_id(raw.attributes.get(REQUEST_ID_ATTRIBUTE)),
    )


# ------------------------------------------------------------------ receiver
@dataclass
class ReceiverStats:
    requests: int = 0
    spans: int = 0
    decode_failures: int = 0
    unsupported_media: int = 0
    by_media: dict[str, int] = field(default_factory=dict)


class OtlpSpanReceiver:
    """Accept OTLP/HTTP trace exports on a local port while a profile runs."""

    def __init__(self, *, listen: str, session_id: str, run_id: str) -> None:
        host, port = parse_listen_address(listen)
        self.session_id = session_id
        self.run_id = run_id
        self.stats = ReceiverStats()
        self._queue: queue.SimpleQueue[VllmSpanRecord] = queue.SimpleQueue()
        self._lock = threading.Lock()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                receiver._handle(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return None

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="stormlog-otlp", daemon=True
        )

    @property
    def listen(self) -> str:
        host, port = self._server.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else str(host)
        return f"{name}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()

    def drain(self) -> list[VllmSpanRecord]:
        records: list[VllmSpanRecord] = []
        while True:
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                return records

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != OTLP_TRACES_PATH:
            _respond(handler, 404, b"", "text/plain")
            return
        media = (handler.headers.get("Content-Type") or "").split(";")[0].strip()
        body = handler.rfile.read(int(handler.headers.get("Content-Length") or 0))
        with self._lock:
            self.stats.requests += 1
        if media not in _RECEIVER_CAPABILITIES:
            self._count("unsupported_media")
            _respond(handler, 415, b"", "text/plain")
            return
        try:
            spans = _decode_export(body, media)
        except (ValueError, UnicodeDecodeError):
            self._count("decode_failures")
            _respond(handler, 400, b"", "text/plain")
            return
        self._enqueue(spans, media, handler.client_address[0])
        # An empty ExportTraceServiceResponse is valid in either encoding.
        _respond(handler, 200, b"" if media == PROTOBUF_MEDIA else b"{}", media)

    def _count(self, name: str) -> None:
        with self._lock:
            setattr(self.stats, name, getattr(self.stats, name) + 1)

    def _enqueue(self, spans: list[RawSpan], media: str, peer: str) -> None:
        received_at_ns = time.time_ns()
        for raw in spans:
            self._queue.put(
                span_record(
                    raw,
                    session_id=self.session_id,
                    run_id=self.run_id,
                    source=SPAN_SOURCE_RECEIVER,
                    clock_domain=span_clock_domain(raw.resource, peer),
                    received_at_ns=received_at_ns,
                )
            )
        with self._lock:
            self.stats.spans += len(spans)
            self.stats.by_media[media] = self.stats.by_media.get(media, 0) + len(spans)

    def config_record(self) -> dict[str, Any]:
        return {"listen": self.listen, "path": OTLP_TRACES_PATH}

    def capability_metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "listen": self.listen,
                "requests": self.stats.requests,
                "spans": self.stats.spans,
                "decode_failures": self.stats.decode_failures,
                "unsupported_media": self.stats.unsupported_media,
                "spans_by_media": dict(self.stats.by_media),
            }


def _decode_export(body: bytes, media: str) -> list[RawSpan]:
    if media == PROTOBUF_MEDIA:
        return decode_otlp_protobuf(body)
    return decode_otlp_json(json.loads(body.decode("utf-8")))


def _respond(
    handler: BaseHTTPRequestHandler, status: int, body: bytes, media: str
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", media)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def span_capability_event(
    context: CorrelationContext,
    *,
    receiver: OtlpSpanReceiver | None,
    listen: str,
    error: str | None,
) -> CapabilityEvent:
    """The span capability record for a run that asked for a receiver.

    ``supported`` names every ingest path, ``enabled`` the two the receiver
    serves, and ``collected`` the ones that delivered at least one span. A
    receiver that could not listen is unavailable, with the error kept.
    """
    if receiver is None:
        return CapabilityEvent(
            context=context,
            event_id=f"capability:{CAPABILITY_COMPONENT}",
            component=CAPABILITY_COMPONENT,
            available=False,
            metadata={"listen": listen, "error": error, **_DETAILED_TRACE_NOTE},
        )
    metadata = receiver.capability_metadata()
    by_media = metadata["spans_by_media"]
    collected = [
        name for media, name in _RECEIVER_CAPABILITIES.items() if by_media.get(media)
    ]
    return CapabilityEvent(
        context=context,
        event_id=f"capability:{CAPABILITY_COMPONENT}",
        component=CAPABILITY_COMPONENT,
        available=True,
        supported=list(SPAN_CAPABILITIES),
        enabled=list(_RECEIVER_CAPABILITIES.values()),
        collected=collected,
        metadata={**metadata, **_DETAILED_TRACE_NOTE},
    )


def parse_listen_address(listen: str) -> tuple[str, int]:
    """``HOST:PORT`` for the receiver; the host defaults to loopback."""
    host, separator, port_text = listen.rpartition(":")
    if not separator or not port_text.isdigit():
        raise ValueError("--vllm-spans-listen must be HOST:PORT")
    port = int(port_text)
    if port > 65535:
        raise ValueError("--vllm-spans-listen port must be at most 65535")
    return host.strip("[]") or "127.0.0.1", port


# ------------------------------------------------------------------ v2 mapping
def spans_to_correlation_events(
    spans: Iterable[VllmSpanRecord], *, producer_id: str = "vllm.otel"
) -> list[CorrelationEvent]:
    """Map request spans onto v2 request and stage records.

    The request record carries the span's own timestamps as reported by
    vLLM. Each stage window is placed from the span start by adding the
    reported durations in scheduler order, so the windows are estimates;
    the native duration attribute is kept in each stage's metadata.
    """
    events: list[CorrelationEvent] = []
    for span in spans:
        if span.name != "llm_request" or span.start_unix_ns is None:
            continue
        events.extend(_request_events(span, producer_id))
    return events


def _context(
    span: VllmSpanRecord, producer_id: str, provenance: str
) -> CorrelationContext:
    return CorrelationContext(
        run_id=span.run_id,
        session_id=span.session_id,
        producer_id=producer_id,
        source=span.source,
        clock_domain=span.clock_domain,
        clock_kind="wall",
        collection_mode="passive",
        provenance=provenance,
        engine="vllm",
    )


def _request_events(span: VllmSpanRecord, producer_id: str) -> list[CorrelationEvent]:
    native_id = span.attributes.get(REQUEST_ID_ATTRIBUTE)
    native = native_id if isinstance(native_id, str) and native_id else None
    request_name = span.request_id or native
    if request_name is None:
        return []
    request_ref = EntityRef(producer_id, request_name)
    base_id = span.span_id or request_name
    request = RequestEvent(
        context=_context(span, producer_id, "reported"),
        event_id=f"span:{base_id}",
        request_ref=request_ref,
        backend_request_ref=EntityRef(producer_id, native) if native else None,
        start_ns=span.start_unix_ns,
        end_ns=span.end_unix_ns,
        input_tokens=_count(span.attributes.get("gen_ai.usage.prompt_tokens")),
        output_tokens=_count(span.attributes.get("gen_ai.usage.completion_tokens")),
        metadata={"name": span.name, "native_request_id": native},
    )
    stages = _stage_events(span, producer_id, base_id, request_ref)
    return [request, *stages]


def _stage_events(
    span: VllmSpanRecord, producer_id: str, base_id: str, request_ref: EntityRef
) -> list[StageEvent]:
    start = span.start_unix_ns or 0
    cursor = start
    stages: list[StageEvent] = []
    for stage, attribute in STAGE_ATTRIBUTES:
        duration_ns = _duration_ns(span.attributes.get(attribute))
        if duration_ns is None:
            continue
        end = cursor + duration_ns
        stages.append(
            _stage(
                span, producer_id, base_id, request_ref, stage, attribute, cursor, end
            )
        )
        cursor = end
    inference_ns = _duration_ns(span.attributes.get(INFERENCE_ATTRIBUTE))
    if inference_ns is not None:
        begin = start + (_duration_ns(span.attributes.get(STAGE_ATTRIBUTES[0][1])) or 0)
        stages.append(
            _stage(
                span,
                producer_id,
                base_id,
                request_ref,
                "inference",
                INFERENCE_ATTRIBUTE,
                begin,
                begin + inference_ns,
            )
        )
    return stages


def _stage(
    span: VllmSpanRecord,
    producer_id: str,
    base_id: str,
    request_ref: EntityRef,
    stage: str,
    attribute: str,
    start_ns: int,
    end_ns: int,
) -> StageEvent:
    return StageEvent(
        context=_context(span, producer_id, "estimated"),
        event_id=f"span:{base_id}:{stage}",
        stage_ref=EntityRef(producer_id, f"{base_id}:{stage}"),
        name=stage,
        request_ref=request_ref,
        start_ns=start_ns,
        end_ns=end_ns,
        metadata={
            "native_attribute": attribute,
            "native_seconds": span.attributes.get(attribute),
            "meaning": "wall-clock residency in a scheduler phase, not GPU time",
            "placement": "span start plus reported durations in scheduler order",
        },
    )


def _duration_ns(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds: float = float(value)
    return max(0, round(seconds * 1e9))


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)
