"""The spans of an inference profile: which records become spans, and what they hold.

A profile exports a root ``stormlog.infer.capture`` span, a
``stormlog.infer.phase`` child for each phase window, a
``stormlog.infer.trace_window`` child for each profiler window, and a
``stormlog.infer.request`` span for each request sent, in its own trace and
linked to its phase. A request sent with ``traceparent`` keeps the IDs it
was sent with, so the server's span is its child; every other ID is derived
from the session and the record, so mapping an artifact again gives the
same spans. Nothing Stormlog collected from the engine becomes a span.

What a span may hold is an allowlist. Its values are configuration
identifiers, Stormlog's own IDs, closed enums and numbers. A server's error
text, prompts and outputs leave only with ``--export-content`` consent, and
then pass through ``scrub_text``. Every string also passes known-secret
redaction.

As with the metrics, the work is split: ``envelope`` copies a fixed list of
fields on the producer's thread, and ``to_span`` builds the span on the
exporter's worker.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._export.envelope import Envelope, EnvelopeLimits, Value, make_envelope
from .._export.spans import (
    KIND_CLIENT,
    KIND_INTERNAL,
    STATUS_ERROR,
    STATUS_UNSET,
    Attributes,
    Scope,
    Span,
    SpanEvent,
    SpanLink,
)
from ..scrub import REDACTED, KnownSecrets, scrub_text
from .trace_context import derived_ids, keeps

SCOPE_NAME = "stormlog.infer"
# The semantic-conventions version the attribute names were checked against.
SCHEMA_URL = "https://opentelemetry.io/schemas/1.44.0"
CAPTURE_SPAN = "stormlog.infer.capture"
PHASE_SPAN = "stormlog.infer.phase"
TRACE_WINDOW_SPAN = "stormlog.infer.trace_window"
REQUEST_SPAN = "stormlog.infer.request"
FIRST_TOKEN_EVENT = "stormlog.first_token"

DIGESTS = "digests"
ERRORS = "errors"
PROMPTS = "prompts"
OUTPUTS = "outputs"
CONTENT_ITEMS = (DIGESTS, ERRORS, PROMPTS, OUTPUTS)
MAX_CONTENT_BYTES = 1024

# Paths that are safe to export as url.path; any other is redacted.
FIXED_PATHS = ("/v1/chat/completions", "/v1/completions", "/metrics", "/v1/traces")
# The error type and code an OpenAI-compatible server returns, when known.
API_ERROR_TYPES = (
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_error",
    "server_error",
    "api_error",
    "BadRequestError",
    "NotFoundError",
    "InternalServerError",
    "ServiceUnavailableError",
)
API_ERROR_CODES = (
    "context_length_exceeded",
    "rate_limit_exceeded",
    "model_not_found",
    "invalid_api_key",
    "insufficient_quota",
    "server_error",
)
OTHER = "other"
# Statuses that end a request span in error; cancelled is Stormlog's choice.
ERROR_STATUSES = ("timeout", "rejected", "error", "unreachable")
# The most of an error body read for its type and code, on the pool thread.
_DIAGNOSTIC_BYTES = 64 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,63}\Z")

SPAN_LIMITS = EnvelopeLimits(
    max_fields=32,
    max_string=128,
    max_bytes=8192,
    content_fields=frozenset({"error_message", "prompt", "output"}),
    max_content_bytes=MAX_CONTENT_BYTES,
)


def scope(version: str) -> Scope:
    return Scope(SCOPE_NAME, version, SCHEMA_URL)


def error_diagnostics(body: str) -> tuple[str, str] | None:
    """The mapped ``type`` and ``code`` of an OpenAI-style error body.

    Runs on the request's pool thread. Each is one of a closed set, or
    ``other``; nothing else of the body is kept. None when the body is not
    such an error.
    """
    text = body[:_DIAGNOSTIC_BYTES]
    start = text.find("{")
    if start < 0:
        return None
    try:
        document = json.loads(text[start:])
    except ValueError:
        return None
    if not isinstance(document, dict):
        return None
    error = document.get("error")
    # vLLM puts the fields at the top level; OpenAI nests them in "error".
    fields = error if isinstance(error, dict) else document
    if "type" not in fields and "code" not in fields:
        return None
    return _mapped(fields.get("type"), API_ERROR_TYPES), _mapped_code(
        fields.get("code")
    )


def _mapped(value: Any, known: tuple[str, ...]) -> str:
    return value if isinstance(value, str) and value in known else OTHER


def _mapped_code(value: Any) -> str:
    # vLLM puts the numeric HTTP status in "code".
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return str(value)
    return _mapped(value, API_ERROR_CODES)


@dataclass(frozen=True)
class SpanIdentity:
    """What every span of one profile shares."""

    run_id: str
    session_id: str
    model: str
    endpoint: str
    sample_ratio: float = 1.0
    content: frozenset[str] = frozenset()
    server_address: str = field(init=False)
    server_port: int | None = field(init=False)
    url_path: str = field(init=False)

    def __post_init__(self) -> None:
        parts = urllib.parse.urlsplit(self.endpoint)
        object.__setattr__(self, "server_address", parts.hostname or "")
        try:
            port = parts.port
        except ValueError:
            port = None
        if port is None and parts.scheme in ("http", "https"):
            port = 443 if parts.scheme == "https" else 80
        object.__setattr__(self, "server_port", port)
        path = parts.path if parts.path in FIXED_PATHS else REDACTED
        object.__setattr__(self, "url_path", path)


class ProfileSpans:
    """Map a profile's records to envelopes, and envelopes to spans."""

    def __init__(self, identity: SpanIdentity, secrets: KnownSecrets) -> None:
        self.identity = identity
        self.secrets = secrets
        self.capture = derived_ids(identity.session_id, CAPTURE_SPAN)
        # Counted on the producer's thread, which is the only writer.
        self.sampled_out = 0
        self._builders = {
            "infer.request": self._request_envelope,
            "infer.phase_window": _phase_envelope,
            "infer.trace_window": self._trace_window_envelope,
        }

    # ------------------------------------------------------------- producer
    def envelope(
        self, record: dict[str, Any], extras: Mapping[str, Any] | None
    ) -> Envelope | None:
        """The envelope of a record that becomes a span, or None."""
        builder = self._builders.get(str(record.get("event_type")))
        return builder(record, extras) if builder is not None else None

    def capture_envelope(
        self, *, started_ns: int, ended_ns: int, outcome: str, error_type: str | None
    ) -> Envelope:
        return make_envelope(
            "capture",
            [
                ("started_ns", started_ns),
                ("ended_ns", ended_ns),
                ("outcome", outcome),
                ("error_type", error_type),
            ],
            SPAN_LIMITS,
        )

    def _request_envelope(
        self, record: dict[str, Any], extras: Mapping[str, Any] | None
    ) -> Envelope | None:
        if record.get("x_request_id") is None:
            return None  # never sent: there is nothing to trace
        sent_trace, sent_span = record.get("trace_id"), record.get("span_id")
        # Sent with a traceparent: the server may have recorded a child of
        # this span, so it is exported whatever the ratio.
        carried = isinstance(sent_trace, str) and isinstance(sent_span, str)
        if carried:
            trace_id, span_id = str(sent_trace), str(sent_span)
        else:
            ids = derived_ids(self.identity.session_id, str(record.get("request_id")))
            trace_id, span_id = ids.trace_id, ids.span_id
        status = record.get("status")
        if (
            status == "ok"
            and not carried
            and not keeps(trace_id, self.identity.sample_ratio)
        ):
            self.sampled_out += 1
            return None
        return make_envelope(
            "request",
            _request_fields(record, extras or {}, trace_id, span_id)
            + self._content_fields(record, extras or {}),
            SPAN_LIMITS,
        )

    def _content_fields(
        self, record: dict[str, Any], extras: Mapping[str, Any]
    ) -> list[tuple[str, Value]]:
        content = self.identity.content
        if not content:
            return []
        items: list[tuple[str, Value]] = []
        if DIGESTS in content:
            items.append(("prompt_digest", record.get("prompt_digest")))
            items.append(("output_digest", extras.get("output_digest")))
        if ERRORS in content:
            items.append(("error_message", record.get("error_message")))
        if PROMPTS in content:
            items.append(("prompt", extras.get("prompt")))
        if OUTPUTS in content:
            items.append(("output", extras.get("output")))
        return items

    def _trace_window_envelope(
        self, record: dict[str, Any], _extras: Mapping[str, Any] | None
    ) -> Envelope:
        items: list[tuple[str, Value]] = [
            ("case", record.get("case_id")),
            ("phase", record.get("phase")),
            ("requested_ns", record.get("requested_at_ns")),
            ("started_ns", record.get("started_at_ns")),
            ("stopped_ns", record.get("stopped_at_ns")),
            ("started", bool(record.get("started"))),
            ("stop_reason", record.get("stop_reason") or "not_started"),
            ("start_status", record.get("start_status")),
            ("stop_status", record.get("stop_status")),
        ]
        if ERRORS in self.identity.content:
            error = record.get("start_error") or record.get("stop_error")
            items.append(("error_message", error))
        return make_envelope("trace_window", items, SPAN_LIMITS)

    # ------------------------------------------------------------- worker
    def to_span(self, envelope: Envelope) -> Span:
        fields = envelope.as_dict()
        if envelope.kind == "request":
            return self._request_span(fields)
        if envelope.kind == "phase":
            return self._phase_span(fields)
        if envelope.kind == "trace_window":
            return self._trace_window_span(fields)
        return self._capture_span(fields)

    def _common(self) -> list[tuple[str, Value]]:
        identity = self.identity
        return [
            ("stormlog.run_id", identity.run_id),
            ("stormlog.session_id", identity.session_id),
        ]

    def _phase_ids(self, case: Value, phase: Value) -> tuple[str, str]:
        ids = derived_ids(self.identity.session_id, PHASE_SPAN, str(case), str(phase))
        return self.capture.trace_id, ids.span_id

    def _request_span(self, fields: Mapping[str, Value]) -> Span:
        identity = self.identity
        status = str(fields.get("status"))
        started = _int(fields.get("started_ns"))
        ttft_ms = fields.get("ttft_ms")
        events: tuple[SpanEvent, ...] = ()
        if isinstance(ttft_ms, (int, float)):
            events = (SpanEvent(FIRST_TOKEN_EVENT, started + round(ttft_ms * 1e6)),)
        phase_trace, phase_span = self._phase_ids(
            fields.get("case"), fields.get("phase")
        )
        error = status in ERROR_STATUSES
        attributes = [
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.request.model", identity.model),
            ("gen_ai.request.max_tokens", fields.get("max_tokens")),
            ("server.address", identity.server_address),
            ("server.port", identity.server_port),
            ("http.request.method", "POST"),
            ("url.path", identity.url_path),
            ("http.response.status_code", fields.get("http_status")),
            ("error.type", _error_type(status, fields) if error else None),
            ("gen_ai.response.time_to_first_chunk", _seconds(fields, "first_chunk_ms")),
            *self._common(),
            ("stormlog.request_id", fields.get("request_id")),
            ("stormlog.x_request_id", fields.get("x_request_id")),
            ("stormlog.case_id", fields.get("case")),
            ("stormlog.phase", fields.get("phase")),
            ("stormlog.request.status", status),
            ("stormlog.arrival_mode", fields.get("arrival")),
            ("stormlog.request_index", fields.get("index")),
            ("stormlog.dispatch_lag_seconds", _seconds(fields, "lag_ms")),
            ("stormlog.time_to_first_token_seconds", _seconds(fields, "ttft_ms")),
            ("stormlog.prompt_tokens", fields.get("prompt_tokens")),
            ("stormlog.prompt_token_source", fields.get("prompt_source")),
            ("stormlog.output_tokens", fields.get("output_tokens")),
            ("stormlog.output_token_source", fields.get("output_source")),
            ("stormlog.prompt_id", fields.get("prompt_id")),
            ("stormlog.prefix_group", fields.get("prefix_group")),
            ("stormlog.shared_prefix_tokens", fields.get("shared_prefix")),
            ("stormlog.chunk_count", fields.get("chunk_count")),
            ("stormlog.error.api_type", fields.get("api_type")),
            ("stormlog.error.api_code", fields.get("api_code")),
            *self._content_attributes(fields),
        ]
        return Span(
            name=REQUEST_SPAN,
            trace_id=str(fields.get("trace_id")),
            span_id=str(fields.get("span_id")),
            kind=KIND_CLIENT,
            start_ns=started,
            end_ns=_int(fields.get("ended_ns"), started),
            attributes=self._clean(attributes),
            events=events,
            links=(SpanLink(phase_trace, phase_span),),
            status=STATUS_ERROR if error else STATUS_UNSET,
            status_message=self._message(fields),
        )

    def _content_attributes(
        self, fields: Mapping[str, Value]
    ) -> list[tuple[str, Value]]:
        return [
            ("stormlog.prompt.digest", fields.get("prompt_digest")),
            ("stormlog.output.digest", fields.get("output_digest")),
            ("stormlog.prompt.text", self._free_text(fields.get("prompt"))),
            ("stormlog.output.text", self._free_text(fields.get("output"))),
        ]

    def _phase_span(self, fields: Mapping[str, Value]) -> Span:
        trace_id, span_id = self._phase_ids(fields.get("case"), fields.get("phase"))
        started = _int(fields.get("started_ns"))
        window_end = _int(fields.get("window_ended_ns"), started)
        ended = _int(fields.get("drained_ns"), window_end)
        return Span(
            name=PHASE_SPAN,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=self.capture.span_id,
            kind=KIND_INTERNAL,
            start_ns=started,
            end_ns=max(ended, started),
            attributes=self._clean(
                [
                    *self._common(),
                    ("stormlog.case_id", fields.get("case")),
                    ("stormlog.phase", fields.get("phase")),
                    ("stormlog.arrival_mode", fields.get("arrival")),
                    ("stormlog.scheduled_arrivals", fields.get("scheduled")),
                    ("stormlog.window_seconds", (window_end - started) / 1e9),
                    ("stormlog.drain_seconds", (ended - window_end) / 1e9),
                    ("stormlog.abandoned_requests", fields.get("abandoned")),
                ]
            ),
        )

    def _trace_window_span(self, fields: Mapping[str, Value]) -> Span:
        requested = _int(fields.get("requested_ns"))
        started = _int(fields.get("started_ns"), requested)
        ids = derived_ids(
            self.identity.session_id,
            TRACE_WINDOW_SPAN,
            str(fields.get("case")),
            str(fields.get("phase")),
        )
        return Span(
            name=TRACE_WINDOW_SPAN,
            trace_id=self.capture.trace_id,
            span_id=ids.span_id,
            parent_span_id=self.capture.span_id,
            kind=KIND_INTERNAL,
            start_ns=started,
            end_ns=max(_int(fields.get("stopped_ns"), started), started),
            attributes=self._clean(
                [
                    *self._common(),
                    ("stormlog.case_id", fields.get("case")),
                    ("stormlog.phase", fields.get("phase")),
                    ("stormlog.trace_window.started", fields.get("started")),
                    ("stormlog.trace_window.stop_reason", fields.get("stop_reason")),
                    ("stormlog.trace_window.start_status", fields.get("start_status")),
                    ("stormlog.trace_window.stop_status", fields.get("stop_status")),
                ]
            ),
            status_message=self._message(fields),
        )

    def _capture_span(self, fields: Mapping[str, Value]) -> Span:
        identity = self.identity
        outcome = str(fields.get("outcome"))
        started = _int(fields.get("started_ns"))
        return Span(
            name=CAPTURE_SPAN,
            trace_id=self.capture.trace_id,
            span_id=self.capture.span_id,
            kind=KIND_INTERNAL,
            start_ns=started,
            end_ns=max(_int(fields.get("ended_ns"), started), started),
            attributes=self._clean(
                [
                    *self._common(),
                    ("gen_ai.request.model", identity.model),
                    ("server.address", identity.server_address),
                    ("server.port", identity.server_port),
                    ("stormlog.capture.outcome", outcome),
                    ("error.type", _identifier(fields.get("error_type"))),
                ]
            ),
            status=STATUS_ERROR if outcome == "failed" else STATUS_UNSET,
        )

    # ------------------------------------------------------------- policy
    def _clean(self, attributes: list[tuple[str, Value]]) -> Attributes:
        """Leave out absent values; redact known secrets from every string."""
        cleaned: list[tuple[str, Any]] = []
        for key, value in attributes:
            if value is None:
                continue
            if isinstance(value, str):
                value = self.secrets.redact(value)
            elif isinstance(value, tuple):
                continue  # no span attribute here is an array
            cleaned.append((key, value))
        return tuple(cleaned)

    def _free_text(self, value: Value) -> str | None:
        if not isinstance(value, str):
            return None
        return scrub_text(value, max_bytes=MAX_CONTENT_BYTES, secrets=self.secrets)

    def _message(self, fields: Mapping[str, Value]) -> str | None:
        # Only present when the operator consented to error text.
        return self._free_text(fields.get("error_message"))


def _request_fields(
    record: dict[str, Any], extras: Mapping[str, Any], trace_id: str, span_id: str
) -> list[tuple[str, Value]]:
    diagnostics = extras.get("error_diagnostics")
    api_type, api_code = diagnostics if isinstance(diagnostics, tuple) else (None, None)
    first_chunk = record.get("first_chunk_latency_ms")
    gaps = record.get("chunk_interarrival_ms")
    chunks = 0 if first_chunk is None else 1 + (len(gaps) if gaps else 0)
    return [
        ("trace_id", trace_id),
        ("span_id", span_id),
        ("request_id", record.get("request_id")),
        ("x_request_id", record.get("x_request_id")),
        ("case", record.get("case_id")),
        ("phase", record.get("phase")),
        ("status", record.get("status")),
        ("started_ns", record.get("started_at_ns")),
        ("ended_ns", record.get("ended_at_ns")),
        ("arrival", record.get("arrival_mode")),
        ("index", record.get("request_index")),
        ("lag_ms", record.get("dispatch_lag_ms")),
        ("ttft_ms", record.get("ttft_ms")),
        ("first_chunk_ms", first_chunk),
        ("prompt_tokens", record.get("prompt_tokens")),
        ("prompt_source", record.get("prompt_token_source")),
        ("output_tokens", record.get("output_tokens")),
        ("output_source", record.get("output_token_source")),
        ("prompt_id", record.get("prompt_id")),
        ("prefix_group", record.get("prefix_group")),
        ("shared_prefix", record.get("shared_prefix_tokens")),
        ("chunk_count", chunks),
        ("max_tokens", record.get("target_output_tokens")),
        ("http_status", record.get("http_status")),
        ("error_type", record.get("error_type")),
        ("api_type", api_type),
        ("api_code", api_code),
    ]


def _phase_envelope(record: dict[str, Any], _extras: Mapping | None) -> Envelope:
    abandoned = record.get("abandoned_requests")
    running = abandoned.get("running_at_start") if isinstance(abandoned, dict) else None
    return make_envelope(
        "phase",
        [
            ("case", record.get("case_id")),
            ("phase", record.get("phase")),
            ("arrival", record.get("arrival_mode")),
            ("started_ns", record.get("started_at_ns")),
            ("window_ended_ns", record.get("window_ended_at_ns")),
            ("drained_ns", record.get("drained_at_ns")),
            ("scheduled", record.get("scheduled_arrivals")),
            ("abandoned", running),
        ],
        SPAN_LIMITS,
    )


def _error_type(status: str, fields: Mapping[str, Value]) -> str:
    if status != "error":
        return status
    http_status = fields.get("http_status")
    if isinstance(http_status, int):
        return str(http_status)
    return _identifier(fields.get("error_type")) or "_OTHER"


def _identifier(value: Value) -> str | None:
    """An exception class name, or None: never free text."""
    return value if isinstance(value, str) and _IDENTIFIER.match(value) else None


def _seconds(fields: Mapping[str, Value], key: str) -> float | None:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value / 1e3


def _int(value: Value, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value
