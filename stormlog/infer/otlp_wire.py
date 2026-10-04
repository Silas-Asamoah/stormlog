"""Count an OTLP/HTTP protobuf trace export's messages before it is parsed.

Parsing an ``ExportTraceServiceRequest`` costs memory per message far more
than per byte: an empty span is 2 bytes on the wire and over a hundred in
upb's arena, a kilobyte as a pure-Python message. So the receiver charges a
parse by its messages, counted here first by a linear scan of the wire
format that follows only the schema's message-typed fields and builds
nothing. The same scan counts the spans, so an export with too many is
refused before ``ParseFromString`` runs.

The scan stops as soon as a count passes its cap; what it returns then is
over that cap, which is all the receiver needs to refuse the export.
Anything protobuf could not parse as this message is a ``ValueError``. So
are two things the pure-Python backend would parse: groups (wire types 3
and 4), which no OTLP message uses, and messages nested deeper than
protobuf's default limit of 100 levels, where upb stops too.
"""

from __future__ import annotations

from dataclasses import dataclass

# The message types of opentelemetry/proto/collector/trace/v1 and the
# common, resource and trace protos it nests.
(
    _REQUEST,
    _RESOURCE_SPANS,
    _RESOURCE,
    _SCOPE_SPANS,
    _SCOPE,
    _SPAN,
    _EVENT,
    _LINK,
    _STATUS,
    _KEY_VALUE,
    _ANY_VALUE,
    _ARRAY_VALUE,
    _KEY_VALUE_LIST,
) = range(13)
# Each type's message-typed fields, by field number; every other field is
# skipped over by its wire type.
_CHILDREN: tuple[dict[int, int], ...] = (
    {1: _RESOURCE_SPANS},  # ExportTraceServiceRequest.resource_spans
    {1: _RESOURCE, 2: _SCOPE_SPANS},  # ResourceSpans
    {1: _KEY_VALUE},  # Resource.attributes
    {1: _SCOPE, 2: _SPAN},  # ScopeSpans
    {3: _KEY_VALUE},  # InstrumentationScope.attributes
    {9: _KEY_VALUE, 11: _EVENT, 13: _LINK, 15: _STATUS},  # Span
    {3: _KEY_VALUE},  # Span.Event.attributes
    {4: _KEY_VALUE},  # Span.Link.attributes
    {},  # Status
    {2: _ANY_VALUE},  # KeyValue.value
    {5: _ARRAY_VALUE, 6: _KEY_VALUE_LIST},  # AnyValue
    {1: _ANY_VALUE},  # ArrayValue.values
    {1: _KEY_VALUE},  # KeyValueList.values
)
# protobuf's own default recursion limit.
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


def count_trace_request(
    data: bytes | bytearray, *, max_messages: int, max_spans: int
) -> WireCounts:
    """Count the messages, spans and values of an encoded export.

    Stops once ``messages`` passes ``max_messages`` or ``spans`` passes
    ``max_spans``.
    """
    walk = _Walk(data, max_messages, max_spans)
    try:
        walk.run()
    except IndexError:
        raise ValueError("a protobuf field is cut short") from None
    return WireCounts(walk.messages, walk.spans, walk.values)


class _Walk:
    def __init__(self, data: bytes | bytearray, max_messages: int, max_spans: int):
        self.data = data
        self.max_messages = max_messages
        self.max_spans = max_spans
        self.messages = 1  # the request itself
        self.spans = 0
        self.values = 0
        # The open messages: where each ends, and its type.
        self.ends = [len(data)]
        self.kinds = [_REQUEST]

    def run(self) -> None:
        pos = 0
        while self.ends:
            end = self.ends[-1]
            if pos < end:
                pos = self._field(pos, end, _CHILDREN[self.kinds[-1]])
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
        if len(self.ends) >= MAX_DEPTH:
            raise ValueError("protobuf messages nested too deep")
        self.ends.append(end)
        self.kinds.append(kind)
        self.messages += 1
        if kind == _SPAN:
            self.spans += 1
        elif kind == _KEY_VALUE or (
            kind == _ANY_VALUE and self.kinds[-2] != _KEY_VALUE
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


__all__ = ["MAX_DEPTH", "WireCounts", "count_trace_request"]
