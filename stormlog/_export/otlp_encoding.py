"""OTLP trace export requests, in protobuf or JSON, and their responses.

Protobuf uses the generated classes from ``opentelemetry-proto`` (the
``infer-otlp`` extra). Without them, requests are OTLP JSON, written here:
IDs as hex, enums as integers, 64-bit integers as strings. Both encode a
span on its own first, so a batch can be closed by its size before the
request is assembled.

Responses are decoded strictly: a ``partial_success`` must give a whole
number of rejected spans between 0 and the number sent, or the response
says nothing reliable about what was stored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .spans import Attributes, Scope, Span, Value

PROTOBUF_MEDIA = "application/x-protobuf"
JSON_MEDIA = "application/json"
PROTOBUF = "protobuf"
JSON = "json"
# What a span adds to a request beyond its own encoding: a field tag and a
# length prefix in protobuf, a comma in JSON.
_SPAN_OVERHEAD = 6


class NonconformantResponse(ValueError):
    """A response body that does not say how many spans were accepted."""


@dataclass(frozen=True)
class ExportResult:
    """What a 200 response said: spans rejected, and the collector's message."""

    rejected: int = 0
    message: str = ""


class SpanEncoding(Protocol):
    name: str
    media_type: str

    def unit(self, span: Span) -> tuple[Any, int]:
        """``span`` encoded on its own, and what it adds to a request in bytes."""

    def request(
        self, resource: Attributes, scope: Scope, units: Sequence[Any]
    ) -> bytes:
        """One export request holding ``units``."""


def protobuf_available() -> bool:
    return _pb2() is not None


def encoding_for(name: str | None = None) -> SpanEncoding:
    """Protobuf when the extra is installed, otherwise JSON; or the one named."""
    if name == JSON or (name is None and not protobuf_available()):
        return JsonEncoding()
    return ProtobufEncoding()


def decode_response(body: bytes, media_type: str, *, sent: int) -> ExportResult:
    """The ``partial_success`` of a 200 response to an export of ``sent`` spans.

    An empty body is a success with nothing rejected. Raises
    ``NonconformantResponse`` for a body that cannot be read, or whose
    rejection count is not a whole number between 0 and ``sent``.
    """
    if not body:
        return ExportResult()
    if media_type == JSON_MEDIA:
        rejected, message = _json_partial_success(body)
    else:
        rejected, message = _protobuf_partial_success(body)
    if not 0 <= rejected <= sent:
        raise NonconformantResponse(f"rejected {rejected} of {sent} spans")
    return ExportResult(rejected=rejected, message=message)


def status_message(body: bytes, media_type: str) -> str | None:
    """The ``message`` of a ``google.rpc.Status`` error body, if it has one."""
    try:
        if media_type == JSON_MEDIA:
            document = json.loads(body)
            message = document.get("message") if isinstance(document, dict) else None
            return message if isinstance(message, str) else None
        for value in _field_values(_wire_fields(body), 2, 2):
            return bytes(value).decode("utf-8", errors="replace")
    except ValueError:
        return None
    return None


class ProtobufEncoding:
    name = PROTOBUF
    media_type = PROTOBUF_MEDIA

    def __init__(self) -> None:
        pb2 = _pb2()
        if pb2 is None:
            raise RuntimeError(
                "install stormlog[infer-otlp] (opentelemetry-proto) for protobuf"
            )
        self._trace, self._common, self._service = pb2

    def unit(self, span: Span) -> tuple[Any, int]:
        message = self._span(span)
        return message, message.ByteSize() + _SPAN_OVERHEAD

    def request(
        self, resource: Attributes, scope: Scope, units: Sequence[Any]
    ) -> bytes:
        request = self._service.ExportTraceServiceRequest()
        resource_spans = request.resource_spans.add()
        self._fill(resource_spans.resource.attributes, resource)
        scope_spans = resource_spans.scope_spans.add()
        scope_spans.scope.name = scope.name
        scope_spans.scope.version = scope.version
        scope_spans.schema_url = scope.schema_url
        scope_spans.spans.extend(units)
        return bytes(request.SerializeToString())

    def _span(self, span: Span) -> Any:
        message = self._trace.Span(
            trace_id=bytes.fromhex(span.trace_id),
            span_id=bytes.fromhex(span.span_id),
            parent_span_id=bytes.fromhex(span.parent_span_id or ""),
            name=span.name,
            kind=span.kind,
            start_time_unix_nano=span.start_ns,
            end_time_unix_nano=span.end_ns,
            dropped_attributes_count=span.dropped_attributes,
            dropped_events_count=span.dropped_events,
            dropped_links_count=span.dropped_links,
        )
        self._fill(message.attributes, span.attributes)
        for event in span.events:
            added = message.events.add(time_unix_nano=event.time_ns, name=event.name)
            self._fill(added.attributes, event.attributes)
        for link in span.links:
            linked = message.links.add(
                trace_id=bytes.fromhex(link.trace_id),
                span_id=bytes.fromhex(link.span_id),
            )
            self._fill(linked.attributes, link.attributes)
        message.status.code = span.status
        if span.status_message:
            message.status.message = span.status_message
        return message

    def _fill(self, target: Any, attributes: Attributes) -> None:
        for key, value in attributes:
            pair = target.add(key=key)
            _set_any_value(pair.value, value)


class JsonEncoding:
    name = JSON
    media_type = JSON_MEDIA

    def unit(self, span: Span) -> tuple[Any, int]:
        text = _compact(_json_span(span))
        return text, len(text.encode("utf-8")) + 1

    def request(
        self, resource: Attributes, scope: Scope, units: Sequence[Any]
    ) -> bytes:
        # The spans are already encoded, so the request is assembled around them.
        resource_part = _compact({"attributes": _json_attributes(resource)})
        scope_part = _compact({"name": scope.name, "version": scope.version})
        if scope.schema_url:
            scope_part += ',"schemaUrl":' + _compact(scope.schema_url)
        return (
            '{"resourceSpans":[{"resource":'
            + resource_part
            + ',"scopeSpans":[{"scope":'
            + scope_part
            + ',"spans":['
            + ",".join(units)
            + "]}]}]}"
        ).encode("utf-8")


def _pb2() -> tuple[Any, Any, Any] | None:
    # The package is optional and ships no py.typed marker; pyproject tells
    # mypy to ignore it.
    try:
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
        from opentelemetry.proto.common.v1 import common_pb2
        from opentelemetry.proto.trace.v1 import trace_pb2
    except ImportError:
        return None
    return trace_pb2, common_pb2, trace_service_pb2


def _set_any_value(target: Any, value: Value) -> None:
    if isinstance(value, bool):
        target.bool_value = value
    elif isinstance(value, int):
        target.int_value = value
    elif isinstance(value, float):
        target.double_value = value
    elif isinstance(value, str):
        target.string_value = value
    else:
        array = target.array_value
        for item in value:
            _set_any_value(array.values.add(), item)


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _json_span(span: Span) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": span.kind,
        "startTimeUnixNano": str(span.start_ns),
        "endTimeUnixNano": str(span.end_ns),
        "attributes": _json_attributes(span.attributes),
        "status": {"code": span.status},
    }
    if span.parent_span_id:
        raw["parentSpanId"] = span.parent_span_id
    if span.status_message:
        raw["status"]["message"] = span.status_message
    if span.events:
        raw["events"] = [
            {
                "timeUnixNano": str(event.time_ns),
                "name": event.name,
                "attributes": _json_attributes(event.attributes),
            }
            for event in span.events
        ]
    if span.links:
        raw["links"] = [
            {
                "traceId": link.trace_id,
                "spanId": link.span_id,
                "attributes": _json_attributes(link.attributes),
            }
            for link in span.links
        ]
    for key, count in (
        ("droppedAttributesCount", span.dropped_attributes),
        ("droppedEventsCount", span.dropped_events),
        ("droppedLinksCount", span.dropped_links),
    ):
        if count:
            raw[key] = count
    return raw


def _json_attributes(attributes: Attributes) -> list[dict[str, Any]]:
    return [{"key": key, "value": _json_value(value)} for key, value in attributes]


def _json_value(value: Value) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": _json_double(value)}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"arrayValue": {"values": [_json_value(item) for item in value]}}


def _json_double(value: float) -> float | str:
    # Protobuf's JSON mapping spells the values JSON has no number for.
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _json_partial_success(body: bytes) -> tuple[int, str]:
    try:
        document = json.loads(body)
    except ValueError as exc:
        raise NonconformantResponse("unreadable JSON response") from exc
    if not isinstance(document, dict):
        raise NonconformantResponse("the response is not a JSON object")
    partial = document.get("partialSuccess", document.get("partial_success"))
    if partial is None:
        return 0, ""
    if not isinstance(partial, dict):
        raise NonconformantResponse("partialSuccess is not an object")
    raw = partial.get("rejectedSpans", partial.get("rejected_spans", 0))
    message = partial.get("errorMessage", partial.get("error_message", ""))
    return _whole_number(raw), message if isinstance(message, str) else ""


def _whole_number(raw: Any) -> int:
    # int64 is a string in canonical JSON, but a number is accepted too.
    if isinstance(raw, bool):
        raise NonconformantResponse("rejectedSpans is not a number")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.lstrip("-").isdigit():
        return int(raw)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    raise NonconformantResponse(f"rejectedSpans is not a whole number: {raw!r}")


def _protobuf_partial_success(body: bytes) -> tuple[int, str]:
    # ExportTraceServiceResponse: partial_success = 1, holding
    # rejected_spans = 1 (int64) and error_message = 2 (string).
    rejected, message = 0, ""
    try:
        for partial in _field_values(_wire_fields(body), 1, 2):
            inner = _wire_fields(bytes(partial))
            for value in _field_values(inner, 1, 0):
                rejected = _int64(int(value))
            for value in _field_values(inner, 2, 2):
                message = bytes(value).decode("utf-8", errors="replace")
    except ValueError as exc:
        raise NonconformantResponse("unreadable protobuf response") from exc
    return rejected, message


def _field_values(
    fields: list[tuple[int, int, int | bytes]], number: int, wire_type: int
) -> list[int | bytes]:
    return [v for n, t, v in fields if n == number and t == wire_type]


def _int64(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def _wire_fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """The fields of one protobuf message, read from its wire format.

    Only what a response needs: varints, fixed-width values and
    length-delimited bytes. Raises ``ValueError`` for anything malformed.
    """
    fields: list[tuple[int, int, int | bytes]] = []
    position = 0
    while position < len(data):
        key, position = _varint(data, position)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise ValueError("field number 0")
        value: int | bytes
        if wire_type == 0:
            value, position = _varint(data, position)
        elif wire_type == 2:
            length, position = _varint(data, position)
            if position + length > len(data):
                raise ValueError("truncated field")
            value, position = data[position : position + length], position + length
        elif wire_type in (1, 5):
            width = 8 if wire_type == 1 else 4
            if position + width > len(data):
                raise ValueError("truncated field")
            value = int.from_bytes(data[position : position + width], "little")
            position += width
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        fields.append((number, wire_type, value))
    return fields


def _varint(data: bytes, position: int) -> tuple[int, int]:
    result = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("truncated varint")
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
    raise ValueError("varint too long")
