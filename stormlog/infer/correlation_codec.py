"""Artifact-local v4 context interning; models retain embedded v2/v3 values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .correlation_events import (
    CONTEXT_REFERENCE_SCHEMA_VERSION,
    CorrelationContext,
    CorrelationEvent,
    InferenceRecord,
    is_correlation_event_type,
    parse_inference_record,
)

_T = TypeVar("_T")


def _context(payload: object) -> CorrelationContext:
    if not isinstance(payload, dict):
        raise ValueError("context must be an object")
    return CorrelationContext(**payload)


def _context_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("context_id must be a non-empty string")
    return value


# Every reference repeats the ID, so keep it short. A file holds a handful of
# contexts, which makes a clash in 16 hex digits vanishingly rare, and a clash
# fails as a conflicting definition instead of merging two contexts.
_ID_HEX_DIGITS = 16


def _generated_id(context: CorrelationContext) -> str:
    canonical = json.dumps(
        asdict(context), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "ctx-" + digest[:_ID_HEX_DIGITS]


class ContextRegistry:
    """Validated immutable contexts, with the first ID retained for aliases."""

    def __init__(self) -> None:
        self._contexts: dict[str, CorrelationContext] = {}
        self._ids: dict[CorrelationContext, str] = {}

    def register(self, identity: str, context: CorrelationContext) -> None:
        _context_id(identity)
        previous = self._contexts.get(identity)
        if previous is not None and previous != context:
            raise ValueError(f"conflicting context definition: {identity}")
        self._contexts[identity] = context
        self._ids.setdefault(context, identity)

    def lookup(self, identity: str) -> CorrelationContext:
        try:
            return self._contexts[identity]
        except KeyError as exc:
            raise ValueError(f"unknown context_id: {identity}") from exc

    def identity(self, context: CorrelationContext) -> str | None:
        return self._ids.get(context)

    def check(self, identity: str, context: CorrelationContext) -> None:
        previous = self._contexts.get(identity)
        if previous is not None and previous != context:
            raise ValueError(f"conflicting context definition: {identity}")

    def copy(self) -> ContextRegistry:
        result = ContextRegistry()
        result._contexts = self._contexts.copy()
        result._ids = self._ids.copy()
        return result


class CorrelationRecordDecoder:
    """Consume definitions before references, without validating legacy rows."""

    def __init__(self) -> None:
        self.registry = ContextRegistry()
        # Each context as a dict, built once per ID: asdict on every reference
        # cost about a third of reading a compact artifact.
        self._context_dicts: dict[str, dict[str, Any]] = {}

    def decode(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """The record with its context restored; None for a definition."""
        expanded, compact = self._expand(record)
        if expanded is not None and compact:
            _validate_expanded(expanded)
        return expanded

    def decode_event(self, record: Mapping[str, Any]) -> InferenceRecord | None:
        """The record as a typed event, parsed once; None for a definition."""
        expanded, compact = self._expand(record)
        if expanded is None:
            return None
        if compact:
            _reject_null_intervals(expanded)
        return parse_inference_record(expanded)

    def _expand(self, record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        """The record with its context restored, and whether it was compact."""
        if not isinstance(record, Mapping):
            raise ValueError("record must be an object")
        version = record.get("schema_version", 1)
        definition = record.get("event_type") == "infer.context"
        transport = definition or "context_id" in record or version == 4
        if not transport:
            return dict(record), False
        if type(version) is not int or version != CONTEXT_REFERENCE_SCHEMA_VERSION:
            raise ValueError("context transport requires schema_version 4")
        identity = _context_id(record.get("context_id"))
        if definition:
            self._define(record, identity)
            return None, False
        if "context" in record:
            raise ValueError("v4 event must have context_id and no context")
        expanded = dict(record)
        del expanded["context_id"]
        expanded["context"] = self._context_dict(identity)
        expanded["schema_version"] = _embedded_version(record)
        return expanded, True

    def _context_dict(self, identity: str) -> dict[str, Any]:
        cached = self._context_dicts.get(identity)
        if cached is None:
            cached = asdict(self.registry.lookup(identity))
            self._context_dicts[identity] = cached
        # Context values are strings, integers or None, so a shallow copy
        # leaves the cache untouched when a caller edits the result.
        return dict(cached)

    def _define(self, record: Mapping[str, Any], identity: str) -> None:
        if set(record) != {"schema_version", "event_type", "context_id", "context"}:
            raise ValueError(
                "infer.context requires exactly version, type, ID, context"
            )
        self.registry.register(identity, _context(record["context"]))


def _embedded_version(record: Mapping[str, Any]) -> int:
    metadata = record.get("metadata")
    if (
        record.get("event_type") == "infer.activity_ref"
        and isinstance(metadata, dict)
        and "intervals" in metadata
    ):
        return 3
    return 2


def _null_intervals(record: Mapping[str, Any]) -> bool:
    """An activity naming ``intervals`` with no value; v4 cannot express it."""
    metadata = record.get("metadata")
    return (
        record.get("event_type") == "infer.activity_ref"
        and isinstance(metadata, dict)
        and "intervals" in metadata
        and metadata["intervals"] is None
    )


def _reject_null_intervals(expanded: Mapping[str, Any]) -> None:
    if _null_intervals(expanded):
        raise ValueError(
            "metadata.intervals must be a non-empty list of [offset_ns, duration_ns] "
            "pairs"
        )


def _validate_expanded(expanded: Mapping[str, Any]) -> None:
    _reject_null_intervals(expanded)
    parse_inference_record(expanded)


def _is_transport(record: Mapping[str, Any]) -> bool:
    return (
        record.get("event_type") == "infer.context"
        or "context_id" in record
        or record.get("schema_version") == CONTEXT_REFERENCE_SCHEMA_VERSION
    )


def _carries_context(record: Mapping[str, Any]) -> bool:
    """A known correlation type past v1; everything else is written as given."""
    version = record.get("schema_version", 1)
    legacy = type(version) is int and version == 1
    return is_correlation_event_type(record.get("event_type")) and not legacy


class CorrelationRecordEncoder:
    """Stage emissions; commit definitions only after successful serialization."""

    def __init__(self, registry: ContextRegistry | None = None) -> None:
        self.registry = registry.copy() if registry is not None else ContextRegistry()

    def prepare(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        if _is_transport(record):
            raise ValueError("writer accepts embedded semantic records only")
        if not _carries_context(record):
            return [dict(record)]
        event = parse_inference_record(record)
        if not isinstance(event, CorrelationEvent) or _null_intervals(record):
            return [dict(record)]
        context = event.context
        identity = self.registry.identity(context)
        rows = []
        if identity is None:
            identity = _generated_id(context)
            self.registry.check(identity, context)
            rows.append(
                {
                    "schema_version": 4,
                    "event_type": "infer.context",
                    "context_id": identity,
                    "context": asdict(context),
                }
            )
        compact = dict(record)
        del compact["context"]
        compact.update(schema_version=4, context_id=identity)
        return [*rows, compact]

    def commit(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            if row.get("event_type") == "infer.context":
                self.registry.register(
                    _context_id(row["context_id"]), _context(row["context"])
                )

    def encode(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Encode a row for in-memory use; file writers stage before committing."""
        rows = self.prepare(record)
        self.commit(rows)
        return rows


def expand_inference_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decoder = CorrelationRecordDecoder()
    expanded = []
    for index, record in enumerate(records, 1):
        try:
            value = decoder.decode(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {index}: {exc}") from exc
        if value is not None:
            expanded.append(value)
    return expanded


def _decoded_lines(
    path: str | Path, decode: Callable[[Mapping[str, Any]], _T | None]
) -> Iterator[tuple[int, _T]]:
    """Each line's decoded value with its physical line number. A line nested
    past ``MAX_NESTING`` is not JSON, as ``diagnose`` and Studio read it:
    json.loads's own limit moves with the Python and the thread's stack."""
    from .diagnosis_inputs import MAX_NESTING, nested_too_deep  # imports us

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                if nested_too_deep(line.encode()):
                    raise ValueError(f"not JSON (nested deeper than {MAX_NESTING})")
                value = decode(json.loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            if value is not None:
                yield line_number, value


def iter_inference_records(
    path: str | Path, decoder: CorrelationRecordDecoder | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Expanded record dicts, definitions consumed, with physical line numbers."""
    decoder = decoder if decoder is not None else CorrelationRecordDecoder()
    return _decoded_lines(path, decoder.decode)


def iter_inference_events(
    path: str | Path, decoder: CorrelationRecordDecoder | None = None
) -> Iterator[tuple[int, InferenceRecord]]:
    """Typed events, each parsed once, with physical line numbers."""
    decoder = decoder if decoder is not None else CorrelationRecordDecoder()
    return _decoded_lines(path, decoder.decode_event)


def read_inference_records(path: str | Path) -> list[dict[str, Any]]:
    return [record for _, record in iter_inference_records(path)]


__all__ = [
    "ContextRegistry",
    "CorrelationRecordDecoder",
    "CorrelationRecordEncoder",
    "expand_inference_records",
    "iter_inference_events",
    "iter_inference_records",
    "read_inference_records",
]
