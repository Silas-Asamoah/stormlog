"""Counting an OTLP protobuf export's messages on the wire, before parsing."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.otlp_wire import MAX_DEPTH, count_trace_request
from tests.otlp_test_helpers import (
    MessageField,
    export_with,
    message_fields,
    messages_in,
    spans_in,
)

UNBOUNDED = {"max_messages": 10**9, "max_spans": 10**9}


def _modules() -> tuple[Any, Any]:
    trace_service = pytest.importorskip(
        "opentelemetry.proto.collector.trace.v1.trace_service_pb2"
    )
    return trace_service, trace_service.ExportTraceServiceRequest


def _rich_request() -> Any:
    """Every message type the scan follows, nested values included."""
    _service, request_class = _modules()
    request = request_class()
    resource_spans = request.resource_spans.add()
    resource_spans.resource.attributes.add(key="host.name").value.string_value = "h"
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "vllm"
    scope_spans.scope.attributes.add(key="s").value.int_value = 1
    span = scope_spans.spans.add(name="llm_request", kind=2, trace_id=b"t" * 16)
    span.start_time_unix_nano = 1
    span.end_time_unix_nano = 2
    span.status.code = 1
    span.attributes.add(key="a").value.double_value = 0.5
    array = span.attributes.add(key="arr").value.array_value
    array.values.add().string_value = "x"
    array.values.add().kvlist_value.values.add(key="k").value.bool_value = True
    nested = span.attributes.add(key="kv").value.kvlist_value
    nested.values.add(key="inner").value.array_value.values.add().int_value = 3
    event = span.events.add(name="e", time_unix_nano=3)
    event.attributes.add(key="ea").value.string_value = "v"
    link = span.links.add(trace_id=b"l" * 16, span_id=b"s" * 8)
    link.attributes.add(key="la").value.string_value = "w"
    scope_spans.spans.add()  # an empty span: 2 bytes on the wire
    request.resource_spans.add()  # an empty resource_spans
    return request


def test_the_counts_are_what_the_parse_builds() -> None:
    request = _rich_request()
    counts = count_trace_request(request.SerializeToString(), **UNBOUNDED)
    assert counts.messages == messages_in(request)
    assert counts.spans == 2
    # Attributes everywhere (resource, scope, span, event, link and inside
    # kvlists: 9) and array elements (3).
    assert counts.values == 12


@pytest.mark.parametrize(
    "target", [pytest.param(f, id=f.name) for f in message_fields()]
)
def test_every_message_field_the_installed_schema_defines_is_counted(
    target: MessageField,
) -> None:
    """Built field by field from the installed descriptors: a field a newer
    opentelemetry-proto adds, such as Resource.entity_refs, is counted as
    the parse builds it."""
    _service, request_class = _modules()
    body = export_with(target, 3).SerializeToString()
    parsed = request_class.FromString(body)
    counts = count_trace_request(body, **UNBOUNDED)
    assert counts.messages == messages_in(parsed)
    assert counts.spans == spans_in(parsed)


@pytest.mark.parametrize("count", [0, 1, 1000])
def test_empty_spans_are_counted_one_message_each(count: int) -> None:
    _service, request_class = _modules()
    request = request_class()
    spans = request.resource_spans.add().scope_spans.add().spans
    for _ in range(count):
        spans.add()
    counts = count_trace_request(request.SerializeToString(), **UNBOUNDED)
    assert (counts.messages, counts.spans) == (3 + count, count)


def test_unknown_fields_are_skipped_whatever_their_wire_type() -> None:
    # A span (field 2 of ScopeSpans) holding field 99 as a varint, a fixed64,
    # a fixed32 and a length-delimited field, none of them a message.
    unknown = (
        bytes([0x98, 0x06, 0x96, 0x01])  # 99, varint 150
        + bytes([0x99, 0x06])
        + bytes(8)  # 99, fixed64
        + bytes([0x9D, 0x06])
        + bytes(4)  # 99, fixed32
        + bytes([0x9A, 0x06, 0x03])
        + b"\x0a\x01\x00"  # 99, 3 bytes: not entered
    )
    span = bytes([0x12, len(unknown)]) + unknown
    scope_spans = bytes([0x12, len(span)]) + span
    body = bytes([0x0A, len(scope_spans)]) + scope_spans
    assert count_trace_request(body, **UNBOUNDED).messages == 4


def test_the_scan_stops_once_a_count_passes_its_cap() -> None:
    _service, request_class = _modules()
    request = request_class()
    spans = request.resource_spans.add().scope_spans.add().spans
    for _ in range(1000):
        spans.add()
    body = request.SerializeToString()
    by_spans = count_trace_request(body, max_messages=10**9, max_spans=10)
    assert by_spans.spans == 11
    by_messages = count_trace_request(body, max_messages=50, max_spans=10**9)
    assert by_messages.messages == 51


@pytest.mark.parametrize(
    "body",
    [
        bytes([0x0A]),  # a length that never comes
        bytes([0x0A, 0x05, 0x12]),  # a message that runs past the body
        bytes([0x0A, 0x02, 0x12, 0x05]),  # a child that runs past its parent
        bytes([0x0B]),  # wire type 3: a group
        bytes([0x0E]),  # wire type 6: no such type
        bytes([0x02, 0x00]),  # field 0
        bytes([0x08]) + bytes([0xFF] * 11),  # a varint over ten bytes
        bytes([0x09, 0x00]),  # a fixed64 cut short: no field may follow it
    ],
    ids=[
        "truncated-length",
        "past-the-body",
        "past-the-parent",
        "group",
        "bad-wire-type",
        "field-zero",
        "long-varint",
        "short-fixed64",
    ],
)
def test_a_body_protobuf_could_not_parse_is_refused(body: bytes) -> None:
    with pytest.raises(ValueError):
        count_trace_request(body, **UNBOUNDED)


def test_nesting_deeper_than_protobuf_allows_is_refused() -> None:
    """Arrays nest without end in AnyValue; protobuf stops at 100 levels."""
    value = b""
    for _ in range(MAX_DEPTH):
        array = _field(1, value)  # ArrayValue.values
        value = _field(5, array)  # AnyValue.array_value
    key_value = _field(2, value)  # KeyValue.value
    span = _field(9, key_value)  # Span.attributes
    body = _field(1, _field(2, _field(2, span)))
    with pytest.raises(ValueError, match="nested too deep"):
        count_trace_request(body, **UNBOUNDED)


def _nested(levels: int) -> bytes:
    """An export with ``levels`` messages nested below the request: resource
    spans, scope spans, a span, an attribute, its value, then arrays and
    values in turn."""
    value = b""
    for level in range(levels - 5):
        number = 5 if level % 2 == levels % 2 else 1  # array_value / values
        value = _field(number, value)
    attribute = _field(2, value)  # KeyValue.value
    return _field(1, _field(2, _field(2, _field(9, attribute))))


def test_the_scan_nests_exactly_as_deep_as_upb() -> None:
    """upb parses 100 levels below the request and refuses 101; so does
    the scan, whichever backend is installed."""
    _service, request_class = _modules()
    deepest, too_deep = _nested(MAX_DEPTH), _nested(MAX_DEPTH + 1)
    assert count_trace_request(deepest, **UNBOUNDED).messages == MAX_DEPTH + 1
    with pytest.raises(ValueError, match="nested too deep"):
        count_trace_request(too_deep, **UNBOUNDED)
    from google.protobuf.internal import api_implementation

    if api_implementation.Type() == "upb":
        request_class.FromString(deepest)
        with pytest.raises(Exception):
            request_class.FromString(too_deep)


def _field(number: int, payload: bytes) -> bytes:
    """A length-delimited field, its length as a varint."""
    length = len(payload)
    encoded = bytearray()
    while True:
        byte = length & 0x7F
        length >>= 7
        if length:
            encoded.append(byte | 0x80)
        else:
            encoded.append(byte)
            break
    return bytes([(number << 3) | 2]) + bytes(encoded) + payload
