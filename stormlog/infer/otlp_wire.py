"""Count an OTLP/HTTP protobuf trace export's messages before it is parsed.

Parsing an ``ExportTraceServiceRequest`` costs memory per message far more
than per byte: an empty span is 2 bytes on the wire and over a hundred in
upb's arena, a kilobyte as a pure-Python message. So the receiver charges a
parse by its messages, counted here first by a linear scan of the wire
format that follows only message-typed fields and builds nothing. The scan
reads its schema from the installed ``opentelemetry-proto`` descriptors, so
it follows every message field the parser will build, those a newer
version adds included. The same scan counts the spans, so an export with
too many is refused before ``ParseFromString`` runs.

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
from typing import Any

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
_FIXED32 = 5


@dataclass(frozen=True)
class WireCounts:
    """What parsing the export would build.

    ``values`` counts every attribute (``KeyValue``) and every element of an
    array value, wherever they sit: at least the attribute values the spans
    keep.
    """

    messages: int
    spans: int
    values: int


@dataclass(frozen=True)
class _Schema:
    """The message types under a root, read from protobuf descriptors.

    The root is type 0; ``children[t]`` maps each message-typed field of
    type ``t``, by number, to the type it holds. Every other field is
    skipped over by its wire type.
    """

    children: tuple[dict[int, int], ...]
    span: int
    key_value: int
    any_value: int


@lru_cache(maxsize=None)
def message_schema(root: Any) -> _Schema:
    """The schema of ``root``, a message ``Descriptor``, and of every
    message type it can hold."""
    types = [root]
    index = {root.full_name: 0}
    children: list[dict[int, int]] = []
    for descriptor in types:  # grows while it is walked
        fields: dict[int, int] = {}
        for field in descriptor.fields:
            if field.type != field.TYPE_MESSAGE:
                continue
            child = field.message_type
            if child.full_name not in index:
                index[child.full_name] = len(types)
                types.append(child)
            fields[field.number] = index[child.full_name]
        children.append(fields)
    return _Schema(
        tuple(children),
        index.get(_SPAN, -1),
        index.get(_KEY_VALUE, -1),
        index.get(_ANY_VALUE, -1),
    )


def trace_request_descriptor() -> Any:
    """The installed ``ExportTraceServiceRequest`` descriptor; ``ImportError``
    without the ``infer-otlp`` extra."""
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    return trace_service_pb2.ExportTraceServiceRequest.DESCRIPTOR


def count_trace_request(
    data: bytes | bytearray, *, max_messages: int, max_spans: int
) -> WireCounts:
    """Count the messages, spans and values of an encoded export.

    Stops once ``messages`` passes ``max_messages`` or ``spans`` passes
    ``max_spans``.
    """
    return count_message(
        data,
        trace_request_descriptor(),
        max_messages=max_messages,
        max_spans=max_spans,
    )


def count_message(
    data: bytes | bytearray, root: Any, *, max_messages: int, max_spans: int
) -> WireCounts:
    """:func:`count_trace_request` for a message of any type, by its
    ``Descriptor``."""
    walk = _Walk(data, message_schema(root), max_messages, max_spans)
    try:
        walk.run()
    except IndexError:
        raise ValueError("a protobuf field is cut short") from None
    return WireCounts(walk.messages, walk.spans, walk.values)


class _Walk:
    def __init__(
        self,
        data: bytes | bytearray,
        schema: _Schema,
        max_messages: int,
        max_spans: int,
    ):
        self.data = data
        self.schema = schema
        self.max_messages = max_messages
        self.max_spans = max_spans
        self.messages = 1  # the root itself
        self.spans = 0
        self.values = 0
        # The open messages: where each ends, and its type.
        self.ends = [len(data)]
        self.kinds = [0]

    def run(self) -> None:
        children = self.schema.children
        pos = 0
        while self.ends:
            end = self.ends[-1]
            if pos < end:
                pos = self._field(pos, end, children[self.kinds[-1]])
            elif pos == end:
                self.ends.pop()
                self.kinds.pop()
            else:
                raise ValueError("a protobuf field runs past its message")
            if self.messages > self.max_messages or self.spans > self.max_spans:
                return

    def _field(self, pos: int, end: int, children: dict[int, int]) -> int:
        """Read one field at ``pos``: enter it if it is a message, else skip it."""
        tag, pos = _varint(self.data, pos)
        field, wire = tag >> 3, tag & 7
        if field == 0:
            raise ValueError("a protobuf field numbered 0")
        if wire != _LENGTH_DELIMITED:
            return _skip(self.data, pos, wire)
        length, pos = _varint(self.data, pos)
        child_end = pos + length
        if child_end > end:
            raise ValueError("a protobuf field runs past its message")
        child = children.get(field)
        if child is None:
            return child_end
        self._enter(child, child_end)
        return pos

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


def _skip(data: bytes | bytearray, pos: int, wire: int) -> int:
    """Where a field of a non-message wire type ends."""
    if wire == _VARINT:
        return _varint(data, pos)[1]
    if wire == _FIXED64:
        return pos + 8
    if wire == _FIXED32:
        return pos + 4
    raise ValueError(f"protobuf wire type {wire} is not used by OTLP")


__all__ = [
    "MAX_DEPTH",
    "WireCounts",
    "count_message",
    "count_trace_request",
    "message_schema",
    "trace_request_descriptor",
]
