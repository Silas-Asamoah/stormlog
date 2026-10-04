"""Count an OTLP/HTTP protobuf trace export's messages before it is parsed.

Parsing an ``ExportTraceServiceRequest`` costs memory per message far more
than per byte: an empty span is 2 bytes on the wire and over a hundred in
upb's arena, a kilobyte as a pure-Python message. Unknown fields, and the
elements of a repeated string or number, are kept one by one as well. So
the receiver charges a parse by what it builds, counted here first by a
linear scan of the wire format that builds nothing. The scan reads its
schema from the installed ``opentelemetry-proto`` descriptors, so it
follows every field the parser will build, those a newer version adds
included, and knows which fields the parser keeps as unknown. The same
scan counts the spans, so an export with too many is refused before
``ParseFromString`` runs.

The scan stops as soon as a count passes its cap; what it returns then is
over that cap, which is all the receiver needs to refuse the export.
Anything protobuf could not parse as this message is a ``ValueError``. So
are two things the pure-Python backend of protobuf 4 would parse: groups
(wire types 3 and 4), which no OTLP message uses, and messages nested more
than 100 levels below the request, protobuf's default limit, where upb and
later pure-Python versions stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

# The types counted apart from other messages, by their full names.
_SPAN = "opentelemetry.proto.trace.v1.Span"
_KEY_VALUE = "opentelemetry.proto.common.v1.KeyValue"
_ANY_VALUE = "opentelemetry.proto.common.v1.AnyValue"
# protobuf's own default recursion limit: how many levels of messages may
# nest below the root.
MAX_DEPTH = 100
_VARINT = 0
_FIXED64 = 1
_LENGTH_DELIMITED = 2
_GROUP = 3
_FIXED32 = 5
# Each field type's wire type, by its FieldDescriptor.TYPE_* number, which
# descriptor.proto fixes: double, fixed64 and sfixed64 are 64-bit; float,
# fixed32 and sfixed32 32-bit; string, message and bytes length-delimited;
# a group is wire type 3; every other scalar is a varint.
_WIRE_OF_TYPE = {
    1: _FIXED64,
    6: _FIXED64,
    16: _FIXED64,
    2: _FIXED32,
    7: _FIXED32,
    15: _FIXED32,
    9: _LENGTH_DELIMITED,
    11: _LENGTH_DELIMITED,
    12: _LENGTH_DELIMITED,
    10: _GROUP,
}
# Every byte of a varint but its last has the high bit set.
_CONTINUATION_BYTES = bytes(range(0x80, 0x100))


@dataclass(frozen=True)
class WireCounts:
    """What parsing the export would build.

    ``values`` counts every attribute (``KeyValue``) and every element of an
    array value, wherever they sit: at least the attribute values the spans
    keep. ``elements`` counts what the parser keeps one by one outside a
    message's own fields: each unknown field (a field number the schema
    does not define, or a wire type its field does not take), and each
    element of a repeated string, bytes or number field. ``unknown_bytes``
    is the size of the unknown fields, tags included.
    """

    messages: int
    spans: int
    values: int
    elements: int
    unknown_bytes: int


class _Field(NamedTuple):
    """A field the schema defines, as the parser reads it."""

    wires: frozenset[int]  # the wire types it is read in; others are unknown
    child: int  # the message type it holds, or -1
    element: int  # a repeated non-message field's element wire type, or -1


@dataclass(frozen=True)
class _Schema:
    """The message types under a root, read from protobuf descriptors.

    The root is type 0; ``fields[t]`` maps each field of type ``t`` by its
    number.
    """

    fields: tuple[dict[int, _Field], ...]
    span: int
    key_value: int
    any_value: int


@lru_cache(maxsize=None)
def message_schema(root: Any) -> _Schema:
    """The schema of ``root``, a message ``Descriptor``, and of every
    message type it can hold."""
    types = [root]
    index = {root.full_name: 0}
    fields: list[dict[int, _Field]] = []
    for descriptor in types:  # grows while it is walked
        known: dict[int, _Field] = {}
        for field in descriptor.fields:
            child = -1
            if field.type == field.TYPE_MESSAGE:
                name = field.message_type.full_name
                if name not in index:
                    index[name] = len(types)
                    types.append(field.message_type)
                child = index[name]
            known[field.number] = _field_spec(field, child)
        fields.append(known)
    return _Schema(
        tuple(fields),
        index.get(_SPAN, -1),
        index.get(_KEY_VALUE, -1),
        index.get(_ANY_VALUE, -1),
    )


def _field_spec(field: Any, child: int) -> _Field:
    wire = _WIRE_OF_TYPE.get(field.type, _VARINT)
    if child >= 0 or not _is_repeated(field):
        return _Field(frozenset({wire}), child, -1)
    # A repeated number is read packed or one element at a time.
    return _Field(frozenset({wire, _LENGTH_DELIMITED}), -1, wire)


def _is_repeated(field: Any) -> bool:
    """``is_repeated`` from protobuf 6; ``label`` before it, gone in 7."""
    is_repeated = getattr(field, "is_repeated", None)
    if is_repeated is None:
        return bool(field.label == field.LABEL_REPEATED)
    return bool(is_repeated)


def trace_request_descriptor() -> Any:
    """The installed ``ExportTraceServiceRequest`` descriptor; ``ImportError``
    without the ``infer-otlp`` extra."""
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    return trace_service_pb2.ExportTraceServiceRequest.DESCRIPTOR


def count_trace_request(
    data: bytes | bytearray,
    *,
    max_messages: int,
    max_spans: int,
    max_elements: int,
) -> WireCounts:
    """Count what parsing an encoded export would build.

    Stops once ``messages``, ``spans`` or ``elements`` passes its cap.
    """
    return count_message(
        data,
        trace_request_descriptor(),
        max_messages=max_messages,
        max_spans=max_spans,
        max_elements=max_elements,
    )


def count_message(
    data: bytes | bytearray,
    root: Any,
    *,
    max_messages: int,
    max_spans: int,
    max_elements: int,
) -> WireCounts:
    """:func:`count_trace_request` for a message of any type, by its
    ``Descriptor``."""
    walk = _Walk(data, message_schema(root), (max_messages, max_spans, max_elements))
    try:
        walk.run()
    except IndexError:
        raise ValueError("a protobuf field is cut short") from None
    return WireCounts(
        walk.messages, walk.spans, walk.values, walk.elements, walk.unknown_bytes
    )


class _Walk:
    def __init__(
        self, data: bytes | bytearray, schema: _Schema, caps: tuple[int, int, int]
    ):
        self.data = data
        self.schema = schema
        self.max_messages, self.max_spans, self.max_elements = caps
        self.messages = 1  # the root itself
        self.spans = 0
        self.values = 0
        self.elements = 0
        self.unknown_bytes = 0
        # The open messages: where each ends, and its type.
        self.ends = [len(data)]
        self.kinds = [0]

    def run(self) -> None:
        fields = self.schema.fields
        pos = 0
        while self.ends:
            end = self.ends[-1]
            if pos < end:
                pos = self._field(pos, end, fields[self.kinds[-1]])
            elif pos == end:
                self.ends.pop()
                self.kinds.pop()
            else:
                raise ValueError("a protobuf field runs past its message")
            if (
                self.messages > self.max_messages
                or self.spans > self.max_spans
                or self.elements > self.max_elements
            ):
                return

    def _field(self, pos: int, end: int, fields: dict[int, _Field]) -> int:
        """Read one field at ``pos``: enter it if it is a message, else
        count what the parser keeps of it and skip it."""
        start = pos
        tag, pos = _varint(self.data, pos)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("a protobuf field numbered 0")
        known = fields.get(number)
        if known is None or wire not in known.wires:
            pos = _skip(self.data, pos, end, wire)
            self.elements += 1
            self.unknown_bytes += pos - start
            return pos
        if wire != _LENGTH_DELIMITED:
            if known.element >= 0:  # one element of a repeated number
                self.elements += 1
            return _skip(self.data, pos, end, wire)
        length, pos = _varint(self.data, pos)
        child_end = pos + length
        if child_end > end:
            raise ValueError("a protobuf field runs past its message")
        if known.child >= 0:
            self._enter(known.child, child_end)
            return pos
        if known.element >= 0:
            self.elements += _elements(self.data, pos, child_end, known.element)
        return child_end

    def _enter(self, kind: int, end: int) -> None:
        if len(self.ends) > MAX_DEPTH:
            raise ValueError("protobuf messages nested too deep")
        self.ends.append(end)
        self.kinds.append(kind)
        self.messages += 1
        schema = self.schema
        if kind == schema.span:
            self.spans += 1
        elif kind == schema.key_value or (
            kind == schema.any_value and self.kinds[-2] != schema.key_value
        ):
            self.values += 1


def _elements(data: bytes | bytearray, start: int, end: int, element: int) -> int:
    """How many elements a repeated field's length-delimited value holds:
    one string or bytes, or every number packed in it."""
    if element == _LENGTH_DELIMITED:
        return 1
    if element == _FIXED64:
        return (end - start) // 8
    if element == _FIXED32:
        return (end - start) // 4
    return len(data[start:end].translate(None, _CONTINUATION_BYTES))


def _varint(data: bytes | bytearray, pos: int) -> tuple[int, int]:
    """A base-128 varint at ``pos``, and where the next field starts."""
    result = 0
    for shift in range(0, 70, 7):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
    raise ValueError("a protobuf varint over ten bytes")


def _skip(data: bytes | bytearray, pos: int, end: int, wire: int) -> int:
    """Where the value of a field not entered ends."""
    if wire == _VARINT:
        return _varint(data, pos)[1]
    if wire == _FIXED64:
        return pos + 8
    if wire == _FIXED32:
        return pos + 4
    if wire == _LENGTH_DELIMITED:
        length, pos = _varint(data, pos)
        if pos + length > end:
            raise ValueError("a protobuf field runs past its message")
        return pos + length
    raise ValueError(f"protobuf wire type {wire} is not used by OTLP")


__all__ = [
    "MAX_DEPTH",
    "WireCounts",
    "count_message",
    "count_trace_request",
    "message_schema",
    "trace_request_descriptor",
]
