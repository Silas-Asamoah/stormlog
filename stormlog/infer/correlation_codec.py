"""Artifact-local v4 context interning; models retain embedded v2/v3 values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .correlation_events import (
    CONTEXT_REFERENCE_SCHEMA_VERSION,
    CorrelationContext,
    CorrelationEvent,
    is_correlation_event_type,
    parse_inference_record,
)


def _context(payload: object) -> CorrelationContext:
    if not isinstance(payload, dict):
        raise ValueError("context must be an object")
    return CorrelationContext(**payload)


def _context_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("context_id must be a non-empty string")
    return value


def _generated_id(context: CorrelationContext) -> str:
    canonical = json.dumps(
        asdict(context), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "ctx-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    def decode(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        version = record.get("schema_version", 1)
        definition = record.get("event_type") == "infer.context"
        transport = definition or "context_id" in record or version == 4
        if not transport:
            return dict(record)
        if type(version) is not int or version != CONTEXT_REFERENCE_SCHEMA_VERSION:
            raise ValueError("context transport requires schema_version 4")
        identity = _context_id(record.get("context_id"))
        if definition:
            self._define(record, identity)
            return None
        if "context" in record:
            raise ValueError("v4 event must have context_id and no context")
        expanded = dict(record)
        del expanded["context_id"]
        expanded["context"] = asdict(self.registry.lookup(identity))
        expanded["schema_version"] = _embedded_version(record)
        _validate_expanded(expanded)
        return expanded

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


def _validate_expanded(expanded: Mapping[str, Any]) -> None:
    if _null_intervals(expanded):
        raise ValueError(
            "metadata.intervals must be a non-empty list of [offset_ns, duration_ns] "
            "pairs"
        )
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


def iter_inference_records(
    path: str | Path, decoder: CorrelationRecordDecoder | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    decoder = decoder if decoder is not None else CorrelationRecordDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                expanded = decoder.decode(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            if expanded is not None:
                yield line_number, expanded


def read_inference_records(path: str | Path) -> list[dict[str, Any]]:
    return [record for _, record in iter_inference_records(path)]
