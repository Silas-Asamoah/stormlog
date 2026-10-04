"""OTLP trace requests in protobuf and JSON, span caps, and strict responses."""

import json
from dataclasses import replace
from typing import Any

import pytest

from stormlog._export import otlp_encoding
from stormlog._export.otlp_encoding import (
    JSON_MEDIA,
    PROTOBUF_MEDIA,
    ExportResult,
    JsonEncoding,
    NonconformantResponse,
    ProtobufEncoding,
    SpanEncoding,
    decode_response,
    encoding_for,
    status_message,
)
from stormlog._export.spans import (
    KIND_CLIENT,
    STATUS_ERROR,
    Scope,
    Span,
    SpanEvent,
    SpanLimits,
    SpanLink,
    capped,
)
from stormlog.infer.vllm_spans import decode_otlp_json, decode_otlp_protobuf

pb2 = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")

SCOPE = Scope("stormlog.infer", "1.2.3", "https://opentelemetry.io/schemas/1.44.0")
RESOURCE = (("service.name", "stormlog"), ("process.pid", 4242))


def _span(**changes: Any) -> Span:
    values: dict[str, Any] = {
        "name": "stormlog.infer.request",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "span_id": "b7ad6b7169203331",
        "parent_span_id": "00f067aa0ba902b7",
        "kind": KIND_CLIENT,
        "start_ns": 1_700_000_000_000_000_000,
        "end_ns": 1_700_000_000_250_000_000,
        "attributes": (
            ("stormlog.request_id", "c1_measured_0"),
            ("http.response.status_code", 503),
            ("stormlog.ok", False),
            ("stormlog.dispatch_lag_seconds", 0.0125),
            ("stormlog.tags", ("a", "b")),
        ),
        "events": (
            SpanEvent("stormlog.first_token", 1_700_000_000_050_000_000, (("n", 1),)),
        ),
        "links": (SpanLink("1" * 32, "2" * 16, (("stormlog.link", "phase"),)),),
        "status": STATUS_ERROR,
        "status_message": "timeout",
    }
    values.update(changes)
    return Span(**values)


def _both(span: Span) -> tuple[bytes, bytes]:
    encoded = []
    encodings: tuple[SpanEncoding, ...] = (ProtobufEncoding(), JsonEncoding())
    for encoding in encodings:
        unit, _size = encoding.unit(span)
        encoded.append(encoding.request(RESOURCE, SCOPE, [unit]))
    return encoded[0], encoded[1]


def test_protobuf_and_json_carry_the_same_span() -> None:
    protobuf, as_json = _both(_span())
    (from_protobuf,) = decode_otlp_protobuf(protobuf)
    (from_json,) = decode_otlp_json(json.loads(as_json))
    # The receiver reads protobuf's zero drop counts; JSON leaves them out.
    assert from_protobuf.dropped == {"attributes": 0, "events": 0, "links": 0}
    assert replace(from_protobuf, dropped={}) == from_json
    assert from_json.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert from_json.parent_span_id == "00f067aa0ba902b7"
    assert from_json.kind == "CLIENT"
    assert from_json.attributes == {
        "stormlog.request_id": "c1_measured_0",
        "http.response.status_code": 503,
        "stormlog.ok": False,
        "stormlog.dispatch_lag_seconds": 0.0125,
        "stormlog.tags": ["a", "b"],
    }
    assert from_json.resource == {"service.name": "stormlog", "process.pid": 4242}
    assert from_json.status == {"code": "ERROR", "message": "timeout"}
    message = pb2.ExportTraceServiceRequest.FromString(protobuf)
    scope_spans = message.resource_spans[0].scope_spans[0]
    assert scope_spans.schema_url == SCOPE.schema_url
    (span,) = scope_spans.spans
    assert span.events[0].name == "stormlog.first_token"
    assert bytes(span.links[0].span_id).hex() == "2" * 16


def test_json_follows_the_otlp_json_rules() -> None:
    _, as_json = _both(_span(attributes=(("big", 2**62), ("nan", float("nan")))))
    document = json.loads(as_json)
    raw = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert raw["kind"] == 3 and raw["status"]["code"] == 2
    assert raw["startTimeUnixNano"] == "1700000000000000000"
    assert raw["attributes"][0]["value"] == {"intValue": str(2**62)}
    assert raw["attributes"][1]["value"] == {"doubleValue": "NaN"}
    assert document["resourceSpans"][0]["scopeSpans"][0]["schemaUrl"] == (
        SCOPE.schema_url
    )


def test_a_root_span_has_no_parent_and_unset_status() -> None:
    span = _span(parent_span_id=None, status=0, status_message=None, events=())
    for request in _both(span):
        decoded = (
            decode_otlp_json(json.loads(request))
            if request.startswith(b"{")
            else decode_otlp_protobuf(request)
        )
        assert decoded[0].parent_span_id is None
        assert decoded[0].status is not None
        assert decoded[0].status["code"] == "UNSET"


@pytest.mark.parametrize("encoding", [ProtobufEncoding(), JsonEncoding()])
def test_unit_sizes_bound_the_request(encoding: Any) -> None:
    spans = [_span(span_id=f"{index + 1:016x}") for index in range(50)]
    units = [encoding.unit(span) for span in spans]
    body = encoding.request(RESOURCE, SCOPE, [unit for unit, _ in units])
    total = sum(size for _, size in units)
    # The request is the spans plus a resource and scope of a few hundred bytes.
    assert total <= len(body) + 50 * 6 and len(body) <= total + 512


def test_caps_cut_strings_arrays_and_counts_and_say_how_much() -> None:
    limits = SpanLimits(max_attributes=3, max_events=1, max_links=1, max_array=2)
    span = _span(
        attributes=tuple((f"k{index}", "é" * 300) for index in range(5))
        + (("arr", ("x",) * 10),),
        events=(SpanEvent("a", 1), SpanEvent("b", 2)),
        links=(SpanLink("1" * 32, "2" * 16), SpanLink("3" * 32, "4" * 16)),
        status_message="m" * 1000,
    )
    out = capped(span, limits)
    assert [key for key, _ in out.attributes] == ["k0", "k1", "k2"]
    assert all(len(str(v).encode()) <= 256 for _, v in out.attributes)
    assert out.dropped_attributes == 3
    assert (len(out.events), out.dropped_events) == (1, 1)
    assert (len(out.links), out.dropped_links) == (1, 1)
    assert out.status_message is not None and len(out.status_message) == 256
    arrays = capped(_span(attributes=(("arr", ("x",) * 10),)), limits)
    assert arrays.attributes == (("arr", ("x", "x")),)
    for request in _both(out):
        assert len(request) < 4096


def test_the_encoding_falls_back_to_json_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert encoding_for().name == "protobuf"
    assert encoding_for("json").name == "json"
    monkeypatch.setattr(otlp_encoding, "_pb2", lambda: None)
    assert encoding_for().name == "json"
    with pytest.raises(RuntimeError, match="infer-otlp"):
        ProtobufEncoding()


def _protobuf_response(rejected: int, message: str = "") -> bytes:
    response = pb2.ExportTraceServiceResponse()
    response.partial_success.rejected_spans = rejected
    response.partial_success.error_message = message
    return bytes(response.SerializeToString())


def test_responses_give_the_rejected_count() -> None:
    assert decode_response(b"", PROTOBUF_MEDIA, sent=10) == ExportResult()
    assert decode_response(b"", JSON_MEDIA, sent=10) == ExportResult()
    assert decode_response(
        _protobuf_response(4, "bad spans"), PROTOBUF_MEDIA, sent=10
    ) == ExportResult(4, "bad spans")
    assert decode_response(b"{}", JSON_MEDIA, sent=10) == ExportResult()
    for rejected in ('"3"', "3"):
        body = ('{"partialSuccess":{"rejectedSpans":%s}}' % rejected).encode()
        assert decode_response(body, JSON_MEDIA, sent=10).rejected == 3
    warning = b'{"partialSuccess":{"errorMessage":"slow down"}}'
    assert decode_response(warning, JSON_MEDIA, sent=10) == ExportResult(0, "slow down")
    # An empty protobuf message (a success) and unknown fields are both fine.
    assert decode_response(b"\x12\x00", PROTOBUF_MEDIA, sent=1) == ExportResult()


@pytest.mark.parametrize(
    ("body", "media"),
    [
        (_protobuf_response(11), PROTOBUF_MEDIA),
        (_protobuf_response(-1), PROTOBUF_MEDIA),
        (b"\x0a\x05\x08", PROTOBUF_MEDIA),
        (b"\xff\xff\xff", PROTOBUF_MEDIA),
        (b"<html>busy</html>", JSON_MEDIA),
        (b"[]", JSON_MEDIA),
        (b'{"partialSuccess":{"rejectedSpans":"lots"}}', JSON_MEDIA),
        (b'{"partialSuccess":{"rejectedSpans":true}}', JSON_MEDIA),
        (b'{"partialSuccess":{"rejectedSpans":2.5}}', JSON_MEDIA),
        (b'{"partialSuccess":[]}', JSON_MEDIA),
        (b'{"partialSuccess":{"rejectedSpans":"11"}}', JSON_MEDIA),
    ],
)
def test_responses_that_do_not_say_what_was_kept_are_refused(
    body: bytes, media: str
) -> None:
    with pytest.raises(NonconformantResponse):
        decode_response(body, media, sent=10)


def test_status_messages_are_read_from_error_bodies() -> None:
    # google.rpc.Status: code = 1 (varint), message = 2 (string).
    status = b"\x08\x03\x12\x05nope!"
    assert status_message(status, PROTOBUF_MEDIA) == "nope!"
    assert status_message(b'{"code":3,"message":"bad"}', JSON_MEDIA) == "bad"
    assert status_message(b"\xff", PROTOBUF_MEDIA) is None
    assert status_message(b"not json", JSON_MEDIA) is None
