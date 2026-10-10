"""Capped, immutable copies of the fields an exporter needs from a record.

An exporter never queues the record itself: a record can hold an unbounded
error body or a per-chunk list. A mapper reads a fixed list of fields into
an envelope, and the caps here bound what it can hold whatever the record
contained. The work is proportional to the number of fields, never to the
size of the record.

An envelope's size is the memory it holds, as ``sys.getsizeof`` measures
it: the envelope and its tuple of fields, a pair per field, and each value.
Field names are not counted: they come from the mapper's fixed list and are
shared by every envelope.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

from ..scrub import truncate_utf8

Scalar = Union[str, int, float, bool, None]
Value = Union[Scalar, tuple[Scalar, ...]]

# A tuple before its items, and the pointer each item adds.
_TUPLE = sys.getsizeof(())
_POINTER = sys.getsizeof((None,)) - _TUPLE
# A field: its pointer in the envelope's tuple, and its (name, value) pair.
_FIELD_BYTES = _POINTER + sys.getsizeof((None, None))
# Ints are clamped to 64 bits, so no int holds more than this.
INT_MIN, INT_MAX = -(2**63), 2**63 - 1


@dataclass(frozen=True)
class EnvelopeLimits:
    """What one envelope may hold.

    ``content_fields`` are the opt-in content fields (an error body, a
    prompt) that may be up to ``max_content_bytes`` of UTF-8; every other
    string is cut to ``max_string`` characters.
    """

    max_fields: int = 32
    max_string: int = 256
    max_tuple: int = 32
    max_bytes: int = 4096
    content_fields: frozenset[str] = field(default_factory=frozenset)
    max_content_bytes: int = 1024


@dataclass(frozen=True, slots=True)
class Envelope:
    """The capped fields of one record, in the order the mapper read them."""

    kind: str
    fields: tuple[tuple[str, Value], ...]
    size: int
    # Fields cut short or left out to stay inside the limits.
    truncated: int = 0

    def get(self, key: str, default: Value = None) -> Value:
        for name, value in self.fields:
            if name == key:
                return value
        return default

    def as_dict(self) -> dict[str, Value]:
        return dict(self.fields)


def make_envelope(
    kind: str, items: Iterable[tuple[str, Value]], limits: EnvelopeLimits
) -> Envelope:
    """An envelope of ``items`` cut to ``limits``.

    Fields past ``max_fields``, and trailing fields that would take the
    size past ``max_bytes``, are left out; each field cut or left out counts
    once in ``truncated``.
    """
    fields: list[tuple[str, Value]] = []
    size = _ENVELOPE_BYTES
    truncated = 0
    for name, value in items:
        if len(fields) >= limits.max_fields:
            truncated += 1
            continue
        capped, cut = _cap(name, value, limits)
        field_size = _FIELD_BYTES + _value_size(capped)
        if size + field_size > limits.max_bytes:
            truncated += 1
            continue
        fields.append((name, capped))
        size += field_size
        truncated += cut
    return Envelope(kind=kind, fields=tuple(fields), size=size, truncated=truncated)


def _cap(name: str, value: Value, limits: EnvelopeLimits) -> tuple[Value, int]:
    if isinstance(value, str):
        if name in limits.content_fields:
            # Every character is at least one byte, so the first cap's worth
            # of characters holds the answer: a long value is never encoded
            # whole on the producer.
            cap = limits.max_content_bytes
            capped = truncate_utf8(value[:cap], cap)
        else:
            capped = value[: limits.max_string]
        return capped, int(capped != value)
    if isinstance(value, tuple):
        items = tuple(_cap_scalar(item, limits) for item in value[: limits.max_tuple])
        return items, int(items != value)
    # Only an int past 64 bits is cut here, into a new object.
    scalar = _cap_scalar(value, limits)
    return scalar, int(scalar is not value)


def _cap_scalar(value: Scalar, limits: EnvelopeLimits) -> Scalar:
    if isinstance(value, str):
        return value[: limits.max_string]
    if isinstance(value, int) and not isinstance(value, bool):
        return min(max(value, INT_MIN), INT_MAX)  # the same object when in range
    if value is None or isinstance(value, (bool, float)):
        return value
    raise TypeError(f"an envelope holds scalars, not {type(value).__name__}")


def _value_size(value: Value) -> int:
    """The memory ``value`` holds; ``None`` and the bools are shared."""
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, tuple):
        return sys.getsizeof(value) + sum(_value_size(item) for item in value)
    return sys.getsizeof(value)


# The envelope and its tuple of fields, before any field.
_ENVELOPE_BYTES = sys.getsizeof(Envelope("", (), 0)) + _TUPLE
