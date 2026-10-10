"""Spans as an exporter builds them, capped before they are encoded.

A span here is plain data: IDs as lowercase hex, times in Unix
nanoseconds, attributes as ordered key-value pairs. The caps bound what one
span can hold, whatever the record it came from, so the encoded size of a
batch follows from the number of spans in it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Union

from ..scrub import truncate_utf8

# OTLP's SpanKind and StatusCode values.
KIND_INTERNAL = 1
KIND_CLIENT = 3
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2

Scalar = Union[str, bool, int, float]
# An array holds values of one type, as OTLP requires.
Value = Union[Scalar, tuple[str, ...], tuple[int, ...], tuple[float, ...]]
Attributes = tuple[tuple[str, Value], ...]


@dataclass(frozen=True)
class SpanEvent:
    name: str
    time_ns: int
    attributes: Attributes = ()


@dataclass(frozen=True)
class SpanLink:
    trace_id: str
    span_id: str
    attributes: Attributes = ()


@dataclass(frozen=True)
class Span:
    name: str
    trace_id: str
    span_id: str
    kind: int
    start_ns: int
    end_ns: int
    parent_span_id: str | None = None
    attributes: Attributes = ()
    events: tuple[SpanEvent, ...] = ()
    links: tuple[SpanLink, ...] = ()
    status: int = STATUS_UNSET
    status_message: str | None = None
    dropped_attributes: int = 0
    dropped_events: int = 0
    dropped_links: int = 0


@dataclass(frozen=True)
class Scope:
    """The instrumentation scope, and the semantic-conventions version it follows."""

    name: str
    version: str
    schema_url: str = ""


@dataclass(frozen=True)
class SpanLimits:
    max_attributes: int = 64
    max_events: int = 4
    max_links: int = 32
    max_array: int = 32
    # Bytes of UTF-8 per string value.
    max_string: int = 256
    max_resource: int = 32


def capped(span: Span, limits: SpanLimits) -> Span:
    """``span`` within ``limits``; what was cut off is counted on the span."""
    attributes = _capped_pairs(span.attributes, limits)
    events = tuple(
        replace(event, attributes=_capped_pairs(event.attributes, limits))
        for event in span.events[: limits.max_events]
    )
    links = tuple(
        replace(link, attributes=_capped_pairs(link.attributes, limits))
        for link in span.links[: limits.max_links]
    )
    message = span.status_message
    return replace(
        span,
        attributes=attributes,
        events=events,
        links=links,
        status_message=(
            truncate_utf8(message, limits.max_string) if message is not None else None
        ),
        dropped_attributes=span.dropped_attributes
        + len(span.attributes)
        - len(attributes),
        dropped_events=span.dropped_events + len(span.events) - len(events),
        dropped_links=span.dropped_links + len(span.links) - len(links),
    )


def _capped_pairs(attributes: Attributes, limits: SpanLimits) -> Attributes:
    return tuple(
        (key, _capped_value(value, limits))
        for key, value in attributes[: limits.max_attributes]
    )


def _capped_value(value: Value, limits: SpanLimits) -> Value:
    if isinstance(value, str):
        return truncate_utf8(value, limits.max_string)
    if isinstance(value, tuple):
        items = value[: limits.max_array]
        if items and isinstance(items[0], str):
            return tuple(truncate_utf8(str(item), limits.max_string) for item in items)
        return items
    return value
