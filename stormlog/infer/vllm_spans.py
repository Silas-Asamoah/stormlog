"""Ingest vLLM's OpenTelemetry request spans.

Spans arrive in three ways: an OTLP/HTTP receiver that ``stormlog infer
profile`` runs for the length of a run, an OTLP JSON file written by a
collector's file exporter, or the one-span-per-line JSONL that a small sink
writes. Every path produces the same ``infer.vllm_span`` record with the
native attributes untouched.

OTLP protobuf bodies, which is what vLLM's exporter sends, are decoded with
the generated classes from ``opentelemetry-proto`` (the ``infer-otlp``
extra). Without that package the receiver still runs, accepts OTLP JSON,
and records protobuf as supported but not enabled. The latency attributes a
span carries are phase residency measured on the engine's clock; the
mapping to the v2 correlation model keeps them as reported durations and
marks the derived stage windows as estimates.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import zlib
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
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
OTLP_EXTRA_HINT = (
    "install stormlog[infer-otlp] (opentelemetry-proto) to decode OTLP protobuf"
)
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


# ------------------------------------------------------------------ OTLP protobuf
class ProtobufDecodeError(ValueError):
    """The bytes are not an OTLP trace export request."""


class OtlpProtobufUnavailable(RuntimeError):
    """The ``infer-otlp`` extra is not installed."""


def _otlp_request_class() -> Any | None:
    """The generated ``ExportTraceServiceRequest``, or None without the extra."""
    # The package is optional and ships no py.typed marker, so mypy is told
    # to ignore it in pyproject rather than here: an inline ignore cannot
    # name both the missing-package and the untyped-package codes without
    # one of them being unused.
    try:
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
    except ImportError:
        return None
    return trace_service_pb2.ExportTraceServiceRequest


def otlp_protobuf_available() -> bool:
    return _otlp_request_class() is not None


def protobuf_parse_bytes() -> int:
    """What parsing one body byte costs with the installed protobuf backend."""
    try:
        from google.protobuf.internal import api_implementation
    except ImportError:
        return PROTOBUF_PARSE_BYTES_OTHER
    return PROTOBUF_PARSE_BYTES.get(
        api_implementation.Type(), PROTOBUF_PARSE_BYTES_OTHER
    )


def decode_otlp_protobuf(data: bytes) -> list[RawSpan]:
    """Decode an ``ExportTraceServiceRequest`` with the generated classes."""
    return spans_from_message(parse_otlp_protobuf(data))


def parse_otlp_protobuf(data: bytes | bytearray) -> Any:
    """The ``ExportTraceServiceRequest`` message, before any span is built."""
    request_class = _otlp_request_class()
    if request_class is None:
        raise OtlpProtobufUnavailable(OTLP_EXTRA_HINT)
    try:
        return request_class.FromString(bytes(data))
    except Exception as exc:  # google.protobuf.message.DecodeError and friends
        raise ProtobufDecodeError(f"not an OTLP trace export: {exc}") from exc


def message_span_count(message: Any) -> int:
    return sum(
        len(scope_spans.spans)
        for resource_spans in message.resource_spans
        for scope_spans in resource_spans.scope_spans
    )


def message_value_count(message: Any) -> int:
    """Attribute values the spans will hold, nested ones included."""
    values = 0
    for resource_spans in message.resource_spans:
        values += _value_count(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            values += sum(_value_count(span.attributes) for span in scope_spans.spans)
    return values


def _value_count(key_values: Any) -> int:
    return sum(1 + _nested_value_count(item.value) for item in key_values)


def _nested_value_count(value: Any) -> int:
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return sum(1 + _nested_value_count(item) for item in value.array_value.values)
    if kind == "kvlist_value":
        return _value_count(value.kvlist_value.values)
    return 0


def spans_from_message(message: Any) -> list[RawSpan]:
    spans: list[RawSpan] = []
    for resource_spans in message.resource_spans:
        resource = _message_attributes(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            scope = _message_scope(scope_spans.scope)
            spans.extend(
                _message_span(span, resource, scope) for span in scope_spans.spans
            )
    return spans


def _message_attributes(key_values: Any) -> dict[str, Any]:
    return {item.key: _message_any_value(item.value) for item in key_values}


def _message_any_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "array_value":
        return [_message_any_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return _message_attributes(value.kvlist_value.values)
    if kind == "bytes_value":
        return bytes(value.bytes_value).hex()
    return getattr(value, kind)


def _message_scope(scope: Any) -> dict[str, Any]:
    return {
        key: getattr(scope, key) for key in ("name", "version") if getattr(scope, key)
    }


def _message_span(
    span: Any, resource: dict[str, Any], scope: dict[str, Any]
) -> RawSpan:
    if not span.name:
        raise ProtobufDecodeError("span without a name")
    status = None
    if span.HasField("status"):
        status = {
            "code": STATUS_CODES.get(span.status.code, str(span.status.code)),
            "message": span.status.message,
        }
    return RawSpan(
        name=span.name,
        trace_id=bytes(span.trace_id).hex() or None,
        span_id=bytes(span.span_id).hex() or None,
        parent_span_id=bytes(span.parent_span_id).hex() or None,
        kind=SPAN_KINDS.get(span.kind, str(span.kind)),
        start_unix_ns=int(span.start_time_unix_nano) or None,
        end_unix_ns=int(span.end_time_unix_nano) or None,
        attributes=_message_attributes(span.attributes),
        resource=resource,
        scope=scope,
        status=status,
        dropped={
            "attributes": int(span.dropped_attributes_count),
            "events": int(span.dropped_events_count),
            "links": int(span.dropped_links_count),
        },
    )


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
    """The exporter's wall clock, named by its host; never a shared clock.

    vLLM's resource carries no ``host.name``, so for spans the receiver
    collects the domain is named by the peer address the export came from,
    such as ``127.0.0.1/unix_epoch_ns``. Without a boot ID it never counts as
    the client's clock, even on one machine.
    """
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
    protobuf_unavailable: int = 0
    grpc_attempts: int = 0
    oversized: int = 0
    too_large: int = 0
    bad_requests: int = 0
    handler_errors: int = 0
    after_stop: int = 0
    refused_connections: int = 0
    busy: int = 0
    header_timeouts: int = 0
    body_timeouts: int = 0
    too_many_spans: int = 0
    dropped_queue_full: int = 0
    by_media: dict[str, int] = field(default_factory=dict)


MAX_BODY_BYTES = 32 * 1024 * 1024
# What decoding an export costs in memory, so it is charged before it
# happens. Measured on CPython 3.10 with protobuf 4.24, and rounded up:
# json.loads peaked at 90 bytes a structural token, and a protobuf parse at
# 19 bytes a body byte with the upb backend and 135 with the pure-Python
# one; a decoded span with its record but without attributes took 2 KB, and
# an attribute value 190 bytes. Text decoded out of a body takes at most 4
# bytes a byte (a str holding one character outside the Basic Multilingual
# Plane stores every character in 4 bytes), and JSON is decoded to text
# whole before it is parsed.
JSON_TOKEN_BYTES = 128
PROTOBUF_PARSE_BYTES = {"upb": 24}
# Any other backend is charged as the pure-Python one.
PROTOBUF_PARSE_BYTES_OTHER = 160
SPAN_BYTES = 2048
VALUE_BYTES = 256
TEXT_BYTES = 4
# DEFLATE's largest expansion: a gzip body inflates to at most this many
# times its own length.
GZIP_MAX_RATIO = 1032


@dataclass(frozen=True)
class ReceiverLimits:
    """Admission bounds, applied before a body is read, decoded or queued.

    A connection over ``max_connections`` is answered 503 and closed without a
    handler thread. Each request, its request line, headers and body, must
    arrive within ``request_deadline_seconds`` of when the receiver starts
    waiting for it; a kept-alive connection idle that long is closed. The
    exports being read and decoded at once may be charged at most
    ``max_inflight_bytes``, each step charged before it runs: the body, with
    the most a gzip body can inflate to, then an estimate of decoding it
    (``json_decode_estimate``, or for protobuf its parse and then the spans
    it holds). An export that does not fit now is answered 503; one that
    never could, 413. A body with more than ``max_spans_per_body`` spans is
    refused, a protobuf one before any span is built, and one that does not
    fit the queue whole is refused with 503, so the exporter can resend it;
    spans are never queued in part.
    """

    max_connections: int = 8
    request_deadline_seconds: float = 10.0
    max_inflight_bytes: int = 128 * 1024 * 1024
    max_spans_per_body: int = 10_000
    max_queued_spans: int = 100_000
    max_queued_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"receiver {name} must be > 0")
        if self.max_inflight_bytes < 2 * MAX_BODY_BYTES:
            raise ValueError(
                "receiver max_inflight_bytes must hold one full gzip body and "
                "its inflation"
            )


_BUSY_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\n"
    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
)
# How long stop() waits for an export already being read to finish.
STOP_GRACE_SECONDS = 2.0


def json_decode_estimate(content: bytes | bytearray) -> int:
    """An upper bound on what decoding this OTLP JSON export holds at once.

    Counted on the bytes, before anything is parsed: structural tokens
    (every value is opened by one, or follows a comma or colon), spans (each
    has a ``"name"``), and attribute values (each has one ``*Value`` key).
    A count can only be too high: inside a JSON string a quote is escaped.
    """
    tokens = 1 + sum(content.count(mark) for mark in (b"{", b"[", b",", b":"))
    spans = content.count(b'"name"')
    values = content.count(b'Value"')
    return (
        2 * TEXT_BYTES * len(content)
        + tokens * JSON_TOKEN_BYTES
        + spans * SPAN_BYTES
        + values * VALUE_BYTES
    )


def gunzip_capped(body: bytes | bytearray, cap: int) -> bytes | None:
    """Inflate a gzip body, or None when its output would exceed ``cap``.

    A gzip member a few hundred kilobytes long can hold gigabytes of zeros,
    so the decoder is asked for at most ``cap + 1`` bytes: one byte over the
    cap, or input left unconsumed, refuses the body without inflating it
    whole. A stream cut before its trailer raises ``ValueError``; a
    malformed one raises ``zlib.error``.
    """
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = decoder.decompress(body, cap + 1)
    if len(out) > cap or decoder.unconsumed_tail:
        return None
    if not decoder.eof:
        raise ValueError("truncated gzip body")
    return out


_GRPC_PREFACE = b"PRI * HTTP/2.0"
GRPC_HINT = (
    "an export arrived as gRPC (HTTP/2 preface); start vLLM with "
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf"
)


class _DeadlineReader(io.RawIOBase):
    """A connection's reads, each bounded by what is left of one deadline.

    A socket timeout applies to each read, so a sender trickling a byte just
    within it can hold a request open for as long as it likes, in the
    request line and headers as well as the body. Before each read this sets
    the socket's timeout to the time left, and once the deadline has passed
    it raises ``TimeoutError``, however the bytes arrive. Writes get the
    connection's own timeout back.
    """

    def __init__(self, sock: socket.socket, timeout: float) -> None:
        super().__init__()
        self._raw = sock.makefile("rb", buffering=0)
        self._sock = sock
        self._timeout = timeout
        self.deadline: float | None = None
        self.received = 0
        self.timed_out = False

    def start(self, seconds: float) -> None:
        """A new request: the deadline is ``seconds`` from now."""
        self.deadline = time.monotonic() + seconds
        self.received = 0
        self.timed_out = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int | None:
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                self.timed_out = True
                raise TimeoutError("request deadline passed")
            self._sock.settimeout(left)
        try:
            count = self._raw.readinto(buffer)
        except TimeoutError:
            self.timed_out = True
            raise
        finally:
            self._sock.settimeout(self._timeout)
        self.received += count or 0
        return count

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


class OtlpSpanReceiver:
    """Accept OTLP/HTTP trace exports on a local port while a profile runs."""

    def __init__(
        self,
        *,
        listen: str,
        session_id: str,
        run_id: str,
        limits: ReceiverLimits | None = None,
    ) -> None:
        host, port = parse_listen_address(listen)
        self.session_id = session_id
        self.run_id = run_id
        self.limits = limits or ReceiverLimits()
        self.protobuf_available = otlp_protobuf_available()
        self._parse_bytes = protobuf_parse_bytes()
        self.stats = ReceiverStats()
        # Each queued span with the bytes it is charged; guarded by _lock.
        self._queue: deque[tuple[VllmSpanRecord, int]] = deque()
        self._queued_bytes = 0
        self._inflight_bytes = 0
        self._lock = threading.Lock()
        self._stopped = False
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = self.limits.request_deadline_seconds

            def setup(self) -> None:
                super().setup()
                self.rfile.close()
                self.reader = _DeadlineReader(self.connection, self.timeout)
                self.rfile = io.BufferedReader(self.reader)
                self.reading_body = False

            def handle_one_request(self) -> None:
                # One deadline from the wait for the request line to the
                # body's last byte; the base class closes the connection when
                # a read raises TimeoutError.
                self.reader.start(receiver.limits.request_deadline_seconds)
                self.reading_body = False
                super().handle_one_request()
                timed_out_in_head = self.reader.timed_out and not self.reading_body
                if timed_out_in_head and self.reader.received:
                    receiver._count("header_timeouts")

            def parse_request(self) -> bool:
                # vLLM's default exporter is gRPC: its HTTP/2 connection
                # preface is not a request we can serve, but it is a fact
                # worth recording.
                line = getattr(self, "raw_requestline", b"")
                if isinstance(line, bytes) and line.startswith(_GRPC_PREFACE):
                    receiver._count("grpc_attempts")
                    self.close_connection = True
                    return False
                return bool(super().parse_request())

            def do_POST(self) -> None:  # noqa: N802
                self.reading_body = True
                if receiver.stopped:
                    # A request on a connection accepted before the stop:
                    # refused and counted, never queued behind the final
                    # drain where no run would see it.
                    receiver._count("after_stop")
                    self.close_connection = True
                    _try_respond(self, 503)
                    return
                try:
                    receiver._handle(self)
                except Exception:  # a bug must not take the receiver down
                    receiver._count("handler_errors")
                    _try_respond(self, 400)

            def log_message(self, _format: str, *_args: object) -> None:
                return None

        class Server(ThreadingHTTPServer):
            """Knows its accepted connections, so stop() can close them.

            Closing the listener alone leaves every kept-alive HTTP/1.1
            connection and its handler thread alive; this server shuts
            those sockets down on request and can wait for the handlers
            still inside a request to finish.
            """

            address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._connections: set[Any] = set()
                self._idle = threading.Condition()

            def process_request(self, request: Any, client_address: Any) -> None:
                with self._idle:
                    admitted = len(self._connections) < receiver.limits.max_connections
                    if admitted:
                        self._connections.add(request)
                if not admitted:
                    # Refused before a handler thread exists for it.
                    receiver._count("refused_connections")
                    _refuse_connection(request)
                    self.shutdown_request(request)
                    return
                super().process_request(request, client_address)

            def shutdown_request(self, request: Any) -> None:
                try:
                    super().shutdown_request(request)
                finally:
                    with self._idle:
                        self._connections.discard(request)
                        self._idle.notify_all()

            def close_connections(self) -> None:
                """Shut down every accepted socket; idle handlers see EOF."""
                with self._idle:
                    sockets = list(self._connections)
                for sock in sockets:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

            def wait_idle(self, timeout: float) -> bool:
                """True once every handler has finished, or False at the timeout."""
                with self._idle:
                    return bool(
                        self._idle.wait_for(lambda: not self._connections, timeout)
                    )

        self._server = Server((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="stormlog-otlp", daemon=True
        )

    @property
    def listen(self) -> str:
        host, port = self._server.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else str(host)
        if ":" in name:
            name = f"[{name}]"
        return f"{name}:{port}"

    @property
    def enabled(self) -> list[str]:
        """The receiver paths this process can serve."""
        return [
            name
            for media, name in _RECEIVER_CAPABILITIES.items()
            if media != PROTOBUF_MEDIA or self.protobuf_available
        ]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop for good: nothing is queued after this returns.

        New requests are refused first, then the listener closes, every
        accepted connection is shut down, and handlers still inside a
        request get a short grace to finish, so what they decoded is in
        the queue for the final drain and nothing can arrive after it.
        """
        self._stopped = True
        self._server.shutdown()
        self._server.close_connections()
        self._server.wait_idle(STOP_GRACE_SECONDS)
        self._thread.join(timeout=5)
        self._server.server_close()

    @property
    def stopped(self) -> bool:
        """True once ``stop`` has run: the listener is closed for good."""
        return self._stopped

    def drain(self) -> list[VllmSpanRecord]:
        with self._lock:
            records = [record for record, _size in self._queue]
            self._queue.clear()
            self._queued_bytes = 0
        return records

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != OTLP_TRACES_PATH:
            _respond(handler, 404, b"", "text/plain")
            return
        self._count("requests")
        reservation = _Reservation()
        try:
            if self._reserve(handler, reservation, _admission_bytes(handler)):
                self._handle_admitted(handler, reservation)
        finally:
            with self._lock:
                self._inflight_bytes -= reservation.bytes

    def _reserve(
        self, handler: BaseHTTPRequestHandler, reservation: _Reservation, amount: int
    ) -> bool:
        """Charge ``amount`` more to this request before it is held; False
        after answering 503 when it does not fit now, or 413 when it never
        could."""
        never = reservation.bytes + amount > self.limits.max_inflight_bytes
        with self._lock:
            fits = (
                not never
                and self._inflight_bytes + amount <= self.limits.max_inflight_bytes
            )
            if fits:
                self._inflight_bytes += amount
                reservation.bytes += amount
            elif never:
                self.stats.too_large += 1
            else:
                self.stats.busy += 1
        if never:
            handler.close_connection = True
            _try_respond(handler, 413)
        elif not fits:
            _respond_busy(handler)
        return fits

    def _settle(self, reservation: _Reservation, held: int) -> None:
        with self._lock:
            self._inflight_bytes -= reservation.bytes - held
            reservation.bytes = held

    def _handle_admitted(
        self, handler: BaseHTTPRequestHandler, reservation: _Reservation
    ) -> None:
        media = (handler.headers.get("Content-Type") or "").split(";")[0].strip()
        body = self._read_body(handler)
        if body is None:
            return
        # Only the content is held now: a body shorter than announced, the
        # compressed bytes and unused inflation are let go.
        self._settle(reservation, len(body))
        if not self._media_served(handler, media):
            return
        try:
            spans = self._decode(handler, body, media, reservation)
        except (ValueError, UnicodeDecodeError):
            self._count("decode_failures")
            _respond(handler, 400, b"", "text/plain")
            return
        if spans is None:
            return
        if len(spans) > self.limits.max_spans_per_body:
            self._refuse_too_many(handler)
            return
        if not self._enqueue(spans, media, handler.client_address[0], len(body)):
            _respond_busy(handler)
            return
        # An empty ExportTraceServiceResponse is valid in either encoding.
        _respond(handler, 200, b"" if media == PROTOBUF_MEDIA else b"{}", media)

    def _decode(
        self,
        handler: BaseHTTPRequestHandler,
        body: bytes | bytearray,
        media: str,
        reservation: _Reservation,
    ) -> list[RawSpan] | None:
        """The export's spans, each step charged before it runs; None after
        answering an export that does not fit."""
        if media != PROTOBUF_MEDIA:
            if not self._reserve(handler, reservation, json_decode_estimate(body)):
                return None
            return decode_otlp_json(json.loads(body.decode("utf-8")))
        if not self._reserve(handler, reservation, self._parse_bytes * len(body)):
            return None
        message = parse_otlp_protobuf(body)
        spans = message_span_count(message)
        if spans > self.limits.max_spans_per_body:
            self._refuse_too_many(handler)  # before a single span is built
            return None
        built = (
            spans * SPAN_BYTES
            + message_value_count(message) * VALUE_BYTES
            + TEXT_BYTES * len(body)
        )
        if not self._reserve(handler, reservation, built):
            return None
        return spans_from_message(message)

    def _media_served(self, handler: BaseHTTPRequestHandler, media: str) -> bool:
        """True, or False after answering 415 for a media type not served."""
        if media not in _RECEIVER_CAPABILITIES:
            self._count("unsupported_media")
            _respond(handler, 415, b"", "text/plain")
            return False
        if media == PROTOBUF_MEDIA and not self.protobuf_available:
            self._count("protobuf_unavailable")
            _respond(handler, 415, OTLP_EXTRA_HINT.encode(), "text/plain")
            return False
        return True

    def _refuse_too_many(self, handler: BaseHTTPRequestHandler) -> None:
        self._count("too_many_spans")
        _respond(handler, 413, b"too many spans in one body", "text/plain")

    def _body_length(self, handler: BaseHTTPRequestHandler) -> int | None:
        """The announced length, or None after answering one we cannot take."""
        if "Transfer-Encoding" in handler.headers or (
            "Content-Length" not in handler.headers
        ):
            # A chunked body, or none announced: not read, so the
            # connection cannot carry another request.
            self._count("bad_requests")
            handler.close_connection = True
            _respond(handler, 411, b"Content-Length required", "text/plain")
            return None
        try:
            length = int(handler.headers.get("Content-Length") or 0)
            if length < 0:
                raise ValueError("negative length")
        except ValueError:
            self._count("bad_requests")
            _respond(handler, 400, b"bad Content-Length", "text/plain")
            return None
        if length > MAX_BODY_BYTES:
            self._count("oversized")
            _respond(handler, 413, b"", "text/plain")
            return None
        return length

    def _read_body(self, handler: BaseHTTPRequestHandler) -> bytes | bytearray | None:
        """The decoded body, or None after answering a request we cannot take."""
        length = self._body_length(handler)
        if length is None:
            return None
        try:
            body = self._read_within_deadline(handler, length)
        except TimeoutError:
            self._count("body_timeouts")
            handler.close_connection = True
            _try_respond(handler, 408)
            return None
        encoding = (handler.headers.get("Content-Encoding") or "").strip().lower()
        if encoding in {"", "identity"}:
            return body
        if encoding != "gzip":
            self._count("unsupported_media")
            _respond(
                handler, 415, b"only gzip or identity Content-Encoding", "text/plain"
            )
            return None
        return self._gunzip(handler, body)

    def _read_within_deadline(
        self, handler: BaseHTTPRequestHandler, length: int
    ) -> bytearray:
        """The body, or ``TimeoutError`` once the request's deadline passed.

        The handler's reader holds the deadline, so a sender trickling bytes
        cannot hold a connection and its reserved bytes open by meeting a
        per-read timeout. The bytes go straight into one buffer of the
        announced length, already charged, whatever the size of each read.
        """
        body = bytearray(length)
        with memoryview(body) as view:
            received = 0
            while received < length:
                count = handler.rfile.readinto(view[received:])  # type: ignore[attr-defined]
                if not count:
                    break  # a short body fails to decode and is answered 400
                received += count
        del body[received:]
        return body

    def _gunzip(
        self, handler: BaseHTTPRequestHandler, body: bytes | bytearray
    ) -> bytes | None:
        """The inflated body, within the same cap as a plain one."""
        try:
            inflated = gunzip_capped(body, MAX_BODY_BYTES)
        except (zlib.error, ValueError):
            self._count("decode_failures")
            _respond(handler, 400, b"bad gzip body", "text/plain")
            return None
        if inflated is None:
            self._count("oversized")
            _respond(handler, 413, b"decompressed body over the cap", "text/plain")
            return None
        return inflated

    def _count(self, name: str) -> None:
        with self._lock:
            setattr(self.stats, name, getattr(self.stats, name) + 1)

    def _enqueue(
        self, spans: list[RawSpan], media: str, peer: str, body_bytes: int
    ) -> bool:
        """Queue every span of one body, or none; False when it does not fit.

        Each span is charged an equal share of the decoded body's bytes.
        """
        received_at_ns = time.time_ns()
        records = [
            span_record(
                raw,
                session_id=self.session_id,
                run_id=self.run_id,
                source=SPAN_SOURCE_RECEIVER,
                clock_domain=span_clock_domain(raw.resource, peer),
                received_at_ns=received_at_ns,
            )
            for raw in spans
        ]
        size = max(1, body_bytes // max(1, len(records)))
        with self._lock:
            fits = (
                len(self._queue) + len(records) <= self.limits.max_queued_spans
                and self._queued_bytes + size * len(records)
                <= self.limits.max_queued_bytes
            )
            if not fits:
                self.stats.dropped_queue_full += len(records)
                return False
            self._queue.extend((record, size) for record in records)
            self._queued_bytes += size * len(records)
            self.stats.spans += len(spans)
            self.stats.by_media[media] = self.stats.by_media.get(media, 0) + len(spans)
        return True

    def config_record(self) -> dict[str, Any]:
        return {
            "listen": self.listen,
            "path": OTLP_TRACES_PATH,
            "protobuf": self.protobuf_available,
            "limits": asdict(self.limits),
        }

    def capability_metadata(self) -> dict[str, Any]:
        with self._lock:
            metadata: dict[str, Any] = {
                "listen": self.listen,
                "requests": self.stats.requests,
                "spans": self.stats.spans,
                "decode_failures": self.stats.decode_failures,
                "unsupported_media": self.stats.unsupported_media,
                "protobuf_unavailable": self.stats.protobuf_unavailable,
                "grpc_attempts": self.stats.grpc_attempts,
                "oversized": self.stats.oversized,
                "too_large": self.stats.too_large,
                "bad_requests": self.stats.bad_requests,
                "handler_errors": self.stats.handler_errors,
                "after_stop": self.stats.after_stop,
                "refused_connections": self.stats.refused_connections,
                "busy": self.stats.busy,
                "header_timeouts": self.stats.header_timeouts,
                "body_timeouts": self.stats.body_timeouts,
                "too_many_spans": self.stats.too_many_spans,
                "dropped_queue_full": self.stats.dropped_queue_full,
                "queued": len(self._queue),
                "spans_by_media": dict(self.stats.by_media),
            }
        if not self.protobuf_available:
            metadata["otlp_http_protobuf"] = OTLP_EXTRA_HINT
        if metadata["grpc_attempts"]:
            metadata["grpc"] = GRPC_HINT
        return metadata


@dataclass
class _Reservation:
    """What one request has been charged against ``max_inflight_bytes``."""

    bytes: int = 0


def _admission_bytes(handler: BaseHTTPRequestHandler) -> int:
    """The body's own bytes and, for gzip, the most it can inflate to."""
    try:
        length = max(0, int(handler.headers.get("Content-Length") or 0))
    except ValueError:
        length = 0  # _read_body answers the malformed length
    encoding = (handler.headers.get("Content-Encoding") or "").strip().lower()
    reserve = min(length, MAX_BODY_BYTES)
    if encoding == "gzip":
        reserve += min(MAX_BODY_BYTES, GZIP_MAX_RATIO * reserve)
    return reserve


def _respond(
    handler: BaseHTTPRequestHandler, status: int, body: bytes, media: str
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", media)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _respond_busy(handler: BaseHTTPRequestHandler) -> None:
    """503 with ``Retry-After``; the connection closes, since the request
    body may be unread."""
    handler.close_connection = True
    try:
        handler.send_response(503)
        handler.send_header("Retry-After", "1")
        handler.send_header("Content-Length", "0")
        handler.send_header("Connection", "close")
        handler.end_headers()
    except (OSError, ValueError):
        pass


def _refuse_connection(sock: Any) -> None:
    """Answer an over-limit connection with a bare 503 before closing it."""
    try:
        sock.settimeout(1.0)
        sock.sendall(_BUSY_RESPONSE)
    except OSError:
        pass


def _try_respond(handler: BaseHTTPRequestHandler, status: int) -> None:
    """Answer if the connection still allows it; a failed answer is not an error."""
    try:
        _respond(handler, status, b"", "text/plain")
    except (OSError, ValueError):
        handler.close_connection = True


def span_capability_event(
    context: CorrelationContext,
    *,
    receiver: OtlpSpanReceiver | None,
    listen: str,
    error: str | None,
) -> CapabilityEvent:
    """The span capability record for a run that asked for a receiver.

    ``supported`` names every ingest path, ``enabled`` the ones the receiver
    could serve (protobuf only with the ``infer-otlp`` extra), and
    ``collected`` the ones that delivered at least one span. A receiver that
    could not listen is unavailable, with the error kept.
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
        enabled=receiver.enabled,
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
        input_tokens=_count_value(span.attributes.get("gen_ai.usage.prompt_tokens")),
        output_tokens=_count_value(
            span.attributes.get("gen_ai.usage.completion_tokens")
        ),
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


def _count_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)
