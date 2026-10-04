"""OTLP export settings: headers, resource attributes, and what a request adds."""

from typing import Any

import pytest

from stormlog.infer.export_config import ExportConfig
from stormlog.infer.export_otlp import (
    OtlpExport,
    parse_pairs,
    resolve_headers,
    resolve_resource,
)
from stormlog.infer.export_spans import SpanIdentity
from stormlog.scrub import REDACTED, KnownSecrets

BASE = (("service.name", "stormlog"), ("process.pid", 1))


def test_pairs_are_percent_decoded_and_bad_members_skipped() -> None:
    assert parse_pairs(" a = 1 ,b=x%2Cy,,=nokey, novalue ,c=") == [
        ("a", "1"),
        ("b", "x,y"),
        ("c", ""),
    ]


def test_headers_come_from_the_variables_then_the_flags() -> None:
    environ = {
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20abc,X-A=1",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "x-a=2",
    }
    headers = resolve_headers(["X-B=3", "x-a=4"], environ)
    assert headers == {"authorization": "Basic abc", "x-a": "4", "x-b": "3"}


@pytest.mark.parametrize("flag", ["bad name=x", "x-a=line\r\nInjected: 1", "=v"])
def test_headers_that_would_break_the_request_are_refused(flag: str) -> None:
    with pytest.raises(ValueError):
        resolve_headers([flag], {})


def test_only_listed_resource_keys_pass_and_left_out_keys_are_named() -> None:
    secrets = KnownSecrets(["opaque-canary-123"])
    environ = {
        "OTEL_RESOURCE_ATTRIBUTES": (
            "deployment.environment.name=prod,db.password=hunter2,"
            "api.key=opaque-canary-123,service.api_key=opaque-canary-123,"
            "k8s.secret.x=1,host.name=gpu-7"
        ),
        "OTEL_SERVICE_NAME": "bench",
    }
    resource, dropped = resolve_resource(
        BASE,
        flags=["cloud.region=eu-west-1", "team=ml", "host.id=" + "x" * 200],
        allow=["team"],
        environ=environ,
        secrets=secrets,
    )
    assert dict(resource) == {
        "service.name": "bench",
        "process.pid": 1,
        "deployment.environment.name": "prod",
        "host.name": "gpu-7",
        "cloud.region": "eu-west-1",
        "team": "ml",
    }
    assert dropped == (
        "api.key",
        "db.password",
        "host.id",
        "k8s.secret.x",
        "service.api_key",
    )
    assert "hunter2" not in repr(resource) and "opaque" not in repr(resource)


def test_a_listed_value_holding_a_known_secret_is_redacted() -> None:
    secrets = KnownSecrets(["sk-live-canary-9999"])
    resource, _ = resolve_resource(
        BASE,
        flags=["service.namespace=team-sk-live-canary-9999"],
        allow=[],
        environ={},
        secrets=secrets,
    )
    assert dict(resource)["service.namespace"] == f"team-{REDACTED}"


def test_the_resource_is_capped() -> None:
    allow = [f"extra.{index}" for index in range(40)]
    resource, dropped = resolve_resource(
        BASE,
        flags=[f"{key}=v" for key in allow],
        allow=allow,
        environ={},
        secrets=KnownSecrets(),
    )
    assert len(resource) == 32 and len(dropped) == 10


def _otlp(content: frozenset[str] = frozenset(), **kw: Any) -> OtlpExport:
    config = ExportConfig(
        otlp_endpoint="http://127.0.0.1:4318", export_content=content, **kw
    )
    identity = SpanIdentity(
        run_id="r",
        session_id="s",
        model="m",
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        content=content,
    )
    return OtlpExport(
        config,
        identity,
        host="h",
        version="0",
        secrets=KnownSecrets(),
        environ={"OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=secret-header-value"},
    )


def test_header_values_join_the_known_secrets() -> None:
    otlp = _otlp()
    assert otlp.spans.secrets.found_in("leaked secret-header-value")
    assert otlp.headers == {"x-api-key": "secret-header-value"}


def test_a_request_adds_only_what_was_consented() -> None:
    error = RuntimeError('HTTP 400: {"error": {"type": "invalid_request_error"}}')
    plain = _otlp().request_extras("p" * 5000, "out", error)
    assert plain == {"error_diagnostics": ("invalid_request_error", "other")}
    full = _otlp(frozenset({"prompts", "outputs", "digests"})).request_extras(
        "é" * 5000, "the output", None
    )
    assert full["prompt"] == "é" * 512
    assert full["output"] == "the output"
    assert len(full["output_digest"]) == 16


def test_the_span_queue_holds_its_byte_bound_at_maximum_payloads() -> None:
    import tracemalloc

    from stormlog._export.span_export import SPAN_QUEUE_BYTES

    content = frozenset({"errors", "prompts", "outputs", "digests"})
    otlp = _otlp(content)
    tracemalloc.start()
    try:
        for index in range(3000):
            text = f"{index}:" + "é" * 32_768
            record = {
                "event_type": "infer.request",
                "x_request_id": f"x{index}",
                "request_id": f"r{index}" + "r" * 300,
                "case_id": "c" * 300,
                "status": "error",
                "error_message": text,
            }
            otlp.observe(record, otlp.request_extras(text, text, RuntimeError(text)))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    stats = otlp.exporter.queue.stats()
    assert stats.dropped_full > 0 and stats.depth_bytes <= SPAN_QUEUE_BYTES
    # The documented overhead factor for Python objects over the estimate.
    assert peak < 3 * SPAN_QUEUE_BYTES
    otlp.close(0.1)
