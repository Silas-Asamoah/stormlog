"""Read an inference artifact for diagnosis, line by physical line.

A diagnosis cites the records behind each finding by physical line, so a
reader can find them again; and by ``record_id`` and the line's SHA-256, so a
reader can tell when the file changed and find each record by what it is
rather than where it was. This module reads the one file it is given, once,
and keeps for every line its zero-based number (counting blank lines), the
parsed record, its ID and its hash. See ``docs/inference_diagnosis.md`` for
how each record type's ID is derived.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .correlation_events import InferenceRecord, parse_inference_record

INPUT_KIND = "inference_jsonl"

# How a v1 record's ID is built: from these fields, joined by "/" after the
# event type. A v2 record's ID is its event_id.
_ID_FIELDS: dict[str, tuple[str, ...]] = {
    "infer.request": ("request_id",),
    "infer.dispatch": ("request_id",),
    "infer.first_content": ("request_id",),
    "infer.phase_start": ("case_id", "phase"),
    "infer.phase_window": ("case_id", "phase"),
    "infer.trace_window": ("case_id", "phase", "started_at_ns"),
    "infer.vllm_span": ("trace_id", "span_id"),
}
# Records identified by when they were taken: "<event type>@<stamp>".
_STAMP_FIELDS: dict[str, str] = {
    "infer.vllm_scrape": "observed_at_ns",
    "infer.system_sample": "timestamp_ns",
    "infer.telemetry_sample": "observed_at_ns",
    "infer.session": "timestamp_ns",
    "infer.summary": "timestamp_ns",
    "infer.cache_state": "timestamp_ns",
    "infer.workload": "timestamp_ns",
}


@dataclass(frozen=True)
class Line:
    """One physical line of the artifact."""

    number: int  # zero-based, counting every raw line, blank ones included
    sha256: str  # of the line's bytes, without its newline
    raw: dict[str, Any] | None  # None for a blank or unparseable line
    record: InferenceRecord | None  # None when the record does not parse
    record_id: str | None

    @property
    def event_type(self) -> str | None:
        value = (self.raw or {}).get("event_type")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class InputFile:
    """An artifact as diagnosis read it: its identity and every line."""

    path: Path
    size: int
    sha256: str
    lines: tuple[Line, ...]
    problems: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        """The ``payload.inputs`` entry a reader checks before resolving."""
        return {
            "kind": INPUT_KIND,
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
            "lines": len(self.lines),
        }

    def records(self) -> list[Line]:
        return [line for line in self.lines if line.raw is not None]


def read_input(path: str | Path) -> InputFile:
    """Read ``path`` once; a line that is not a record is noted, not fatal.

    Raises:
        OSError: when the file cannot be read.
    """
    source = Path(path)
    data = source.read_bytes()
    chunks = data.split(b"\n")
    if chunks and chunks[-1] == b"":
        chunks.pop()  # the newline ending the last line starts no line
    lines: list[Line] = []
    problems: list[str] = []
    for number, chunk in enumerate(chunks):
        line, problem = _line(number, chunk)
        lines.append(line)
        if problem is not None:
            problems.append(f"line {number}: {problem}")
    return InputFile(
        path=source,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        lines=tuple(lines),
        problems=tuple(problems),
    )


def line_digest(chunk: bytes) -> str:
    return hashlib.sha256(chunk).hexdigest()


def record_id(raw: dict[str, Any]) -> str | None:
    """A record's ID: its ``event_id`` for a v2 record, otherwise built from
    the fields that identify it; None when it has none of them."""
    event_type = raw.get("event_type")
    if not isinstance(event_type, str):
        return None
    if raw.get("schema_version", 1) != 1:
        event_id = raw.get("event_id")
        return event_id if isinstance(event_id, str) and event_id else None
    fields = _ID_FIELDS.get(event_type)
    if fields is None:
        stamp = raw.get(_STAMP_FIELDS.get(event_type, "timestamp_ns"))
        return None if stamp is None else f"{event_type}@{stamp}"
    values = [raw.get(name) for name in fields]
    if None in values:
        return None
    return "/".join([event_type, *(str(value) for value in values)])


def _line(number: int, chunk: bytes) -> tuple[Line, str | None]:
    digest = line_digest(chunk)
    if not chunk.strip():
        return Line(number, digest, None, None, None), None
    try:
        raw = json.loads(chunk)
    except ValueError as exc:
        return Line(number, digest, None, None, None), f"not JSON ({exc})"
    if not isinstance(raw, dict):
        return Line(number, digest, None, None, None), "not a JSON object"
    try:
        record: InferenceRecord | None = parse_inference_record(raw)
        problem = None
    except ValueError as exc:
        record, problem = None, str(exc)
    return Line(number, digest, raw, record, record_id(raw)), problem


__all__ = ["INPUT_KIND", "InputFile", "Line", "line_digest", "read_input", "record_id"]
