"""Records for vLLM native telemetry inside an inference artifact.

A scrape record holds one ``/metrics`` response in the compact form from
:mod:`stormlog.infer.vllm_metrics`, stamped on the client's clock. A span
record holds one OpenTelemetry span exactly as vLLM exported it. Both are
appended to the client artifact next to the request events, and both are
aggregate or engine-side evidence: nothing in them says which request used
which GPU time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .vllm_metrics import CompactScrape, Discovery

VLLM_SCHEMA_VERSION = 1
SCRAPE_EVENT_TYPE = "infer.vllm_scrape"
SPAN_EVENT_TYPE = "infer.vllm_span"
OBSERVATION_SCOPE = "engine_aggregate"

MARKER_PHASE_START = "phase_start"
MARKER_INTERVAL = "interval"
MARKER_PHASE_END = "phase_end"
MARKER_MANUAL = "manual"
MARKERS = (MARKER_PHASE_START, MARKER_INTERVAL, MARKER_PHASE_END, MARKER_MANUAL)

SCRAPE_OK = "ok"
SCRAPE_ERROR = "error"
SCRAPE_STATES = (SCRAPE_OK, SCRAPE_ERROR)

SPAN_SOURCE_RECEIVER = "otlp_http_receiver"
SPAN_SOURCE_OTLP_JSON = "otlp_json_file"
SPAN_SOURCE_JSONL = "jsonl_file"
SPAN_SOURCES = (SPAN_SOURCE_RECEIVER, SPAN_SOURCE_OTLP_JSON, SPAN_SOURCE_JSONL)


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _optional_nonempty(value: object, name: str) -> None:
    if value is not None:
        _nonempty(value, name)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, name: str) -> None:
    if not _is_int(value) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _optional_nonnegative_int(value: object, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")


def _string_dict(value: object, name: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be an object with string keys")


@dataclass(frozen=True)
class VllmScrapeRecord:
    """One ``/metrics`` response, kept whole, stamped on the client clock."""

    session_id: str
    run_id: str
    observed_at_ns: int
    source_url: str
    marker: str
    interval_ms: int
    status: str
    clock_domain: str
    case_id: str | None = None
    phase: str | None = None
    duration_ms: float | None = None
    http_status: int | None = None
    error: str | None = None
    content_digest: str | None = None
    content_bytes: int | None = None
    scrape: CompactScrape | None = None
    discovery: Discovery | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "run_id", "source_url", "clock_domain"):
            _nonempty(getattr(self, name), name)
        _positive_int(self.observed_at_ns, "observed_at_ns")
        _positive_int(self.interval_ms, "interval_ms")
        if self.marker not in MARKERS:
            raise ValueError("unsupported scrape marker")
        if self.status not in SCRAPE_STATES:
            raise ValueError("unsupported scrape status")
        _optional_nonempty(self.case_id, "case_id")
        _optional_nonempty(self.phase, "phase")
        _optional_nonempty(self.error, "error")
        _optional_nonempty(self.content_digest, "content_digest")
        _optional_nonnegative_int(self.http_status, "http_status")
        _optional_nonnegative_int(self.content_bytes, "content_bytes")
        self._validate_duration()
        self._validate_outcome()

    def _validate_duration(self) -> None:
        duration = self.duration_ms
        if duration is None:
            return
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError("duration_ms must be a non-negative number or null")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_ms must be a non-negative number or null")

    def _validate_outcome(self) -> None:
        if self.status == SCRAPE_OK:
            if self.scrape is None or self.discovery is None:
                raise ValueError("an ok scrape carries its series and discovery")
            if self.error is not None:
                raise ValueError("an ok scrape has no error")
        elif self.scrape is not None or self.discovery is not None:
            raise ValueError("a failed scrape carries no series")
        elif self.error is None:
            raise ValueError("a failed scrape says why")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": VLLM_SCHEMA_VERSION,
            "event_type": SCRAPE_EVENT_TYPE,
            "observation_scope": OBSERVATION_SCOPE,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "observed_at_ns": self.observed_at_ns,
            "timestamp_ns": self.observed_at_ns,
            "source_url": self.source_url,
            "marker": self.marker,
            "interval_ms": self.interval_ms,
            "status": self.status,
            "clock_domain": self.clock_domain,
            "case_id": self.case_id,
            "phase": self.phase,
            "duration_ms": self.duration_ms,
            "http_status": self.http_status,
            "error": self.error,
            "content_digest": self.content_digest,
            "content_bytes": self.content_bytes,
            "scrape": self.scrape.to_record() if self.scrape is not None else None,
            "discovery": (
                self.discovery.to_record() if self.discovery is not None else None
            ),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> VllmScrapeRecord:
        _check_envelope(record, SCRAPE_EVENT_TYPE)
        if record.get("observation_scope") != OBSERVATION_SCOPE:
            raise ValueError("scrape records are engine-aggregate evidence")
        scrape = record.get("scrape")
        discovery = record.get("discovery")
        return cls(
            session_id=record["session_id"],
            run_id=record["run_id"],
            observed_at_ns=record["observed_at_ns"],
            source_url=record["source_url"],
            marker=record["marker"],
            interval_ms=record["interval_ms"],
            status=record["status"],
            clock_domain=record["clock_domain"],
            case_id=record.get("case_id"),
            phase=record.get("phase"),
            duration_ms=record.get("duration_ms"),
            http_status=record.get("http_status"),
            error=record.get("error"),
            content_digest=record.get("content_digest"),
            content_bytes=record.get("content_bytes"),
            scrape=CompactScrape.from_record(scrape) if scrape is not None else None,
            discovery=(
                _discovery_from_record(discovery) if discovery is not None else None
            ),
        )


def _discovery_from_record(record: dict[str, Any]) -> Discovery:
    return Discovery(
        present=tuple(record["present"]),
        absent=tuple(record["absent"]),
        optional_absent=tuple(record["optional_absent"]),
        optional_present=tuple(record.get("optional_present", ())),
        deprecated_present=tuple(record["deprecated_present"]),
        removed_present=tuple(record["removed_present"]),
        unknown=tuple(record["unknown"]),
        engines=tuple(record["engines"]),
        model_names=tuple(record["model_names"]),
        process_start_ns=record["process_start_ns"],
    )


def _check_envelope(record: dict[str, Any], event_type: str) -> None:
    if (
        type(record.get("schema_version")) is not int
        or record.get("schema_version") != VLLM_SCHEMA_VERSION
        or record.get("event_type") != event_type
    ):
        raise ValueError(f"unsupported {event_type} record")


@dataclass(frozen=True)
class VllmSpanRecord:
    """One span as exported, with its attributes under their native names.

    The timestamps are the exporter's wall clock, named by ``clock_domain``.
    ``request_id`` is the client-chosen ``X-Request-Id`` recovered from
    ``gen_ai.request.id`` when the span carries one; the join to a Stormlog
    request uses it, never a rebuilt string.
    """

    session_id: str
    run_id: str
    source: str
    name: str
    clock_domain: str
    received_at_ns: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    kind: str | None = None
    start_unix_ns: int | None = None
    end_unix_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    resource: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] | None = None
    dropped: dict[str, int] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "run_id", "name", "clock_domain"):
            _nonempty(getattr(self, name), name)
        if self.source not in SPAN_SOURCES:
            raise ValueError("unsupported span source")
        for name in ("trace_id", "span_id", "parent_span_id", "kind", "request_id"):
            _optional_nonempty(getattr(self, name), name)
        for name in ("received_at_ns", "start_unix_ns", "end_unix_ns"):
            _optional_nonnegative_int(getattr(self, name), name)
        if (
            self.start_unix_ns is not None
            and self.end_unix_ns is not None
            and self.end_unix_ns < self.start_unix_ns
        ):
            raise ValueError("end_unix_ns must be >= start_unix_ns")
        for name in ("attributes", "resource", "scope", "dropped"):
            _string_dict(getattr(self, name), name)
        if self.status is not None:
            _string_dict(self.status, "status")

    @property
    def duration_ns(self) -> int | None:
        if self.start_unix_ns is None or self.end_unix_ns is None:
            return None
        return self.end_unix_ns - self.start_unix_ns

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": VLLM_SCHEMA_VERSION,
            "event_type": SPAN_EVENT_TYPE,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "source": self.source,
            "name": self.name,
            "clock_domain": self.clock_domain,
            "received_at_ns": self.received_at_ns,
            "timestamp_ns": self.start_unix_ns,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind,
            "start_unix_ns": self.start_unix_ns,
            "end_unix_ns": self.end_unix_ns,
            "attributes": dict(self.attributes),
            "resource": dict(self.resource),
            "scope": dict(self.scope),
            "status": dict(self.status) if self.status is not None else None,
            "dropped": dict(self.dropped),
            "request_id": self.request_id,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> VllmSpanRecord:
        _check_envelope(record, SPAN_EVENT_TYPE)
        return cls(
            session_id=record["session_id"],
            run_id=record["run_id"],
            source=record["source"],
            name=record["name"],
            clock_domain=record["clock_domain"],
            received_at_ns=record.get("received_at_ns"),
            trace_id=record.get("trace_id"),
            span_id=record.get("span_id"),
            parent_span_id=record.get("parent_span_id"),
            kind=record.get("kind"),
            start_unix_ns=record.get("start_unix_ns"),
            end_unix_ns=record.get("end_unix_ns"),
            attributes=dict(record.get("attributes") or {}),
            resource=dict(record.get("resource") or {}),
            scope=dict(record.get("scope") or {}),
            status=record.get("status"),
            dropped=dict(record.get("dropped") or {}),
            request_id=record.get("request_id"),
        )


def request_id_from_span_id(native_id: object) -> str | None:
    """Recover the ``X-Request-Id`` vLLM embedded in ``gen_ai.request.id``.

    vLLM names a chat completion ``chatcmpl-<X-Request-Id>`` and a text
    completion ``cmpl-<X-Request-Id>-<index>``, with the per-prompt index only
    on the completions path. A trailing ``-<digits>`` is stripped only when it
    is one; the request ids Stormlog sends end in ``_<n>``, never ``-<n>``.
    Without the client's header vLLM uses a random id, which no Stormlog
    request owns.
    """
    if not isinstance(native_id, str):
        return None
    for prefix in ("chatcmpl-", "cmpl-"):
        if native_id.startswith(prefix):
            body = native_id[len(prefix) :]
            head, _sep, tail = body.rpartition("-")
            return head if head and tail.isdigit() else body
    return None


def load_vllm_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[VllmScrapeRecord], list[VllmSpanRecord]]:
    """Parse the vLLM records of an artifact; an invalid one is an error."""
    scrapes: list[VllmScrapeRecord] = []
    spans: list[VllmSpanRecord] = []
    for index, record in enumerate(records):
        event_type = record.get("event_type")
        try:
            if event_type == SCRAPE_EVENT_TYPE:
                scrapes.append(VllmScrapeRecord.from_record(record))
            elif event_type == SPAN_EVENT_TYPE:
                spans.append(VllmSpanRecord.from_record(record))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {event_type} record {index}: {exc}") from exc
    scrapes.sort(key=lambda item: item.observed_at_ns)
    return scrapes, spans
