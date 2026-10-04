"""Which profile records become spans, and what those spans may hold."""

from typing import Any

import pytest

from stormlog._export.spans import KIND_CLIENT, STATUS_ERROR, STATUS_UNSET, Span
from stormlog.infer.export_spans import (
    CAPTURE_SPAN,
    DIGESTS,
    ERRORS,
    FIRST_TOKEN_EVENT,
    OUTPUTS,
    PHASE_SPAN,
    PROMPTS,
    REQUEST_SPAN,
    TRACE_WINDOW_SPAN,
    ProfileSpans,
    SpanIdentity,
    error_diagnostics,
)
from stormlog.scrub import REDACTED, KnownSecrets

API_KEY = "sk-live-canary-0123456789abcdef"
TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN = "00f067aa0ba902b7"


def _identity(**changes: Any) -> SpanIdentity:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "session_id": "session-1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "endpoint": "http://10.0.0.5:8000/v1/chat/completions",
    }
    values.update(changes)
    return SpanIdentity(**values)


def _spans(**changes: Any) -> ProfileSpans:
    return ProfileSpans(_identity(**changes), KnownSecrets([API_KEY]))


def _request(**changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_type": "infer.request",
        "request_id": "c1_measured_3",
        "x_request_id": "stormlog-run-1-c1_measured_3",
        "trace_id": TRACE,
        "span_id": SPAN,
        "case_id": "c1",
        "phase": "measured",
        "status": "ok",
        "started_at_ns": 1_000_000_000,
        "ended_at_ns": 1_250_000_000,
        "arrival_mode": "closed",
        "request_index": 3,
        "dispatch_lag_ms": 0.5,
        "ttft_ms": 40.0,
        "first_chunk_latency_ms": 38.0,
        "chunk_interarrival_ms": [1.0, 2.0],
        "prompt_tokens": 128,
        "prompt_token_source": "server_usage",
        "output_tokens": 16,
        "output_token_source": "server_usage",
        "prompt_id": "p3",
        "prefix_group": None,
        "shared_prefix_tokens": None,
        "target_output_tokens": 16,
        "http_status": 200,
        "endpoint": f"http://user:{API_KEY}@10.0.0.5:8000/v1/chat/completions",
        "error_message": None,
        "prompt_digest": "sha256:abc",
    }
    record.update(changes)
    return record


def _span(spans: ProfileSpans, record: dict[str, Any], **extras: Any) -> Span:
    envelope = spans.envelope(record, extras or None)
    assert envelope is not None
    return spans.to_span(envelope)


def test_a_request_becomes_a_client_span_with_its_sent_ids() -> None:
    spans = _spans()
    span = _span(spans, _request())
    assert (span.name, span.kind, span.trace_id, span.span_id) == (
        REQUEST_SPAN,
        KIND_CLIENT,
        TRACE,
        SPAN,
    )
    assert span.parent_span_id is None and span.status == STATUS_UNSET
    attributes = dict(span.attributes)
    assert attributes == {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "Qwen/Qwen2.5-7B-Instruct",
        "gen_ai.request.max_tokens": 16,
        "server.address": "10.0.0.5",
        "server.port": 8000,
        "http.request.method": "POST",
        "url.path": "/v1/chat/completions",
        "http.response.status_code": 200,
        "gen_ai.response.time_to_first_chunk": 0.038,
        "stormlog.run_id": "run-1",
        "stormlog.session_id": "session-1",
        "stormlog.request_id": "c1_measured_3",
        "stormlog.x_request_id": "stormlog-run-1-c1_measured_3",
        "stormlog.case_id": "c1",
        "stormlog.phase": "measured",
        "stormlog.request.status": "ok",
        "stormlog.arrival_mode": "closed",
        "stormlog.request_index": 3,
        "stormlog.dispatch_lag_seconds": 0.0005,
        "stormlog.time_to_first_token_seconds": 0.04,
        "stormlog.prompt_tokens": 128,
        "stormlog.prompt_token_source": "server_usage",
        "stormlog.output_tokens": 16,
        "stormlog.output_token_source": "server_usage",
        "stormlog.prompt_id": "p3",
        "stormlog.chunk_count": 3,
    }
    (event,) = span.events
    assert (event.name, event.time_ns) == (FIRST_TOKEN_EVENT, 1_040_000_000)
    # Linked to its phase, which is a child of the capture.
    phase = _span(
        spans,
        {
            "event_type": "infer.phase_window",
            "case_id": "c1",
            "phase": "measured",
            "arrival_mode": "closed",
            "started_at_ns": 900_000_000,
            "window_ended_at_ns": 2_000_000_000,
            "drained_at_ns": 2_500_000_000,
            "scheduled_arrivals": 8,
            "abandoned_requests": {"running_at_start": 0},
        },
    )
    (link,) = span.links
    assert (link.trace_id, link.span_id) == (phase.trace_id, phase.span_id)
    assert phase.name == PHASE_SPAN
    assert phase.parent_span_id == spans.capture.span_id
    assert phase.trace_id == spans.capture.trace_id
    assert dict(phase.attributes)["stormlog.drain_seconds"] == 0.5
    capture = spans.to_span(
        spans.capture_envelope(
            started_ns=1, ended_ns=3_000_000_000, outcome="completed", error_type=None
        )
    )
    assert capture.name == CAPTURE_SPAN and capture.parent_span_id is None
    assert (capture.trace_id, capture.span_id) == (
        spans.capture.trace_id,
        spans.capture.span_id,
    )


def test_without_trace_context_the_ids_are_derived_and_stable() -> None:
    record = _request(trace_id=None, span_id=None)
    first = _span(_spans(), record)
    again = _span(_spans(), record)
    assert (first.trace_id, first.span_id) == (again.trace_id, again.span_id)
    other = _span(_spans(), _request(trace_id=None, span_id=None, request_id="x"))
    assert other.trace_id != first.trace_id


def test_a_request_never_sent_and_engine_records_are_not_spans() -> None:
    spans = _spans()
    assert spans.envelope(_request(x_request_id=None, status="dropped"), None) is None
    for event_type in (
        "infer.vllm_scrape",
        "infer.vllm_span",
        "infer.vllm_execution",
        "infer.server_sample",
        "infer.system_sample",
        "infer.session",
    ):
        assert spans.envelope({"event_type": event_type}, None) is None


def test_sampling_keeps_every_failure() -> None:
    # Requests sent without trace context: the ratio samples their spans.
    spans = _spans(sample_ratio=0.0)
    assert spans.envelope(_request(trace_id=None, span_id=None), None) is None
    assert spans.sampled_out == 1
    for status in ("timeout", "rejected", "error", "cancelled"):
        failed = _request(status=status, trace_id=None, span_id=None)
        assert spans.envelope(failed, None) is not None
    assert spans.sampled_out == 1


def test_a_request_span_that_carried_a_traceparent_is_always_exported() -> None:
    # The server may have recorded a child of it: left out, that child's
    # parent would never reach the backend.
    spans = _spans(sample_ratio=0.0)
    assert spans.envelope(_request(), None) is not None
    assert spans.sampled_out == 0


@pytest.mark.parametrize(
    ("status", "http_status", "error_type", "expected"),
    [
        ("timeout", None, "TimeoutError", ("timeout", STATUS_ERROR)),
        ("rejected", 429, "EndpointHTTPError", ("rejected", STATUS_ERROR)),
        ("error", 400, "EndpointHTTPError", ("400", STATUS_ERROR)),
        ("error", None, "ConnectionResetError", ("ConnectionResetError", STATUS_ERROR)),
        ("error", None, "not an identifier!", ("_OTHER", STATUS_ERROR)),
        ("cancelled", None, None, (None, STATUS_UNSET)),
    ],
)
def test_failures_carry_a_structured_error_type(
    status: str, http_status: int | None, error_type: str | None, expected: Any
) -> None:
    span = _span(
        _spans(),
        _request(status=status, http_status=http_status, error_type=error_type),
    )
    assert (dict(span.attributes).get("error.type"), span.status) == expected


ECHO = (
    'HTTP 400: {"error": {"message": "bad prompt: Bearer sk-abcdefghijklmnop '
    + API_KEY
    + '", "type": "invalid_request_error", "param": "secret-param", '
    '"code": "context_length_exceeded"}}'
)


def test_error_text_needs_consent_and_is_scrubbed_when_given() -> None:
    record = _request(status="error", http_status=400, error_message=ECHO)
    diagnostics = error_diagnostics(ECHO)
    assert diagnostics == ("invalid_request_error", "context_length_exceeded")
    without = _span(_spans(), record, error_diagnostics=diagnostics)
    assert without.status_message is None
    attributes = dict(without.attributes)
    assert attributes["stormlog.error.api_type"] == "invalid_request_error"
    assert attributes["stormlog.error.api_code"] == "context_length_exceeded"
    for value in attributes.values():
        assert "secret-param" not in str(value) and "bad prompt" not in str(value)
    consented = _span(
        _spans(content=frozenset({ERRORS})), record, error_diagnostics=diagnostics
    )
    message = consented.status_message
    assert message is not None and "bad prompt" in message
    assert API_KEY not in message and "sk-abcdefghijklmnop" not in message


def test_content_is_exported_only_as_consented() -> None:
    record = _request()
    extras = {"prompt": "the prompt", "output": "the output", "output_digest": "o1"}
    plain = dict(_span(_spans(), record, **extras).attributes)
    assert not {
        k for k in plain if k.startswith(("stormlog.prompt.", "stormlog.output."))
    }
    full = dict(
        _span(
            _spans(content=frozenset({DIGESTS, PROMPTS, OUTPUTS})), record, **extras
        ).attributes
    )
    assert full["stormlog.prompt.text"] == "the prompt"
    assert full["stormlog.output.text"] == "the output"
    assert full["stormlog.prompt.digest"] == "sha256:abc"
    assert full["stormlog.output.digest"] == "o1"


def test_a_megabyte_error_cannot_make_a_big_envelope() -> None:
    spans = _spans(content=frozenset({ERRORS}))
    record = _request(status="error", error_message="x" * 1_000_000)
    envelope = spans.envelope(record, None)
    assert envelope is not None and envelope.size <= 8192
    span = spans.to_span(envelope)
    assert span.status_message is not None and len(span.status_message) <= 1024
    # Every core field survived the content.
    assert dict(span.attributes)["stormlog.request_id"] == "c1_measured_3"


def test_paths_and_known_secrets_never_leave() -> None:
    spans = ProfileSpans(
        _identity(endpoint=f"https://gw.example/{API_KEY}/v1/chat?key={API_KEY}"),
        KnownSecrets([API_KEY]),
    )
    span = _span(spans, _request(case_id=f"case-{API_KEY}"))
    attributes = dict(span.attributes)
    assert attributes["url.path"] == REDACTED
    assert attributes["server.port"] == 443
    assert attributes["stormlog.case_id"] == f"case-{REDACTED}"
    assert all(API_KEY not in str(value) for value in attributes.values())


def test_trace_windows_are_children_of_the_capture() -> None:
    spans = _spans()
    span = _span(
        spans,
        {
            "event_type": "infer.trace_window",
            "case_id": "c1",
            "phase": "measured",
            "requested_at_ns": 10,
            "started_at_ns": 20,
            "stopped_at_ns": 90,
            "started": True,
            "stop_reason": "phase_end",
            "start_status": 200,
            "stop_status": 200,
            "start_error": "server said " + API_KEY,
        },
    )
    assert span.name == TRACE_WINDOW_SPAN
    assert span.parent_span_id == spans.capture.span_id
    assert (span.start_ns, span.end_ns) == (20, 90)
    assert span.status_message is None
    attributes = dict(span.attributes)
    assert attributes["stormlog.trace_window.stop_reason"] == "phase_end"


def _window(**changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_type": "infer.trace_window",
        "case_id": "c1",
        "phase": "measured",
        "requested_at_ns": 10,
        "started_at_ns": 20,
        "stopped_at_ns": 90,
        "started": True,
        "stop_reason": "phase_end",
        "start_status": 200,
        "stop_status": 200,
    }
    record.update(changes)
    return record


def test_a_failed_profiler_window_is_an_error_with_its_message_if_consented() -> None:
    failed = _window(started=False, start_status=500, start_error="HTTP 500: busy")
    plain = _span(_spans(), failed)
    assert plain.status == STATUS_ERROR and plain.status_message is None
    consented = _span(_spans(content=frozenset({ERRORS})), failed)
    assert consented.status == STATUS_ERROR
    assert consented.status_message == "HTTP 500: busy"
    fine = _span(_spans(content=frozenset({ERRORS})), _window())
    assert fine.status == STATUS_UNSET and fine.status_message is None


def test_a_span_that_did_not_fail_carries_no_status_message() -> None:
    # OpenTelemetry: a status description is only for the error status.
    cancelled = _request(status="cancelled", error_message="still in flight")
    span = _span(_spans(content=frozenset({ERRORS})), cancelled)
    assert span.status == STATUS_UNSET and span.status_message is None


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '{"error": {"type": "rate_limit_error", "code": null}}',
            ("rate_limit_error", "other"),
        ),
        (
            '{"object": "error", "message": "m", "type": "BadRequestError", "code": 400}',
            ("BadRequestError", "400"),
        ),
        ('{"error": {"type": "custom thing", "code": "x"}}', ("other", "other")),
        (
            'HTTP 503: {"type": "ServiceUnavailableError"}',
            ("ServiceUnavailableError", "other"),
        ),
        ("upstream connect error", None),
        ('{"detail": "no"}', None),
        ("[1, 2]", None),
        ('{"error": "not an object", "code": true}', ("other", "other")),
    ],
)
def test_error_diagnostics_map_to_closed_sets(
    body: str, expected: tuple[str, str] | None
) -> None:
    assert error_diagnostics(body) == expected
