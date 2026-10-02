"""vLLM span ingest: OTLP protobuf and JSON, the receiver, and the v2 mapping.

Protobuf payloads are built with the generated ``opentelemetry-proto``
classes, so the decoder is tested against the real wire format; those tests
skip when the ``infer-otlp`` extra is not installed. JSON bodies, the file
readers, the receiver's gate and the v2 mapping need no extra.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from stormlog.infer import vllm_spans
from stormlog.infer.config import ProfileConfig
from stormlog.infer.correlation_events import RequestEvent, StageEvent
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.vllm_spans import (
    OTLP_EXTRA_HINT,
    OtlpProtobufUnavailable,
    OtlpSpanReceiver,
    ProtobufDecodeError,
    RawSpan,
    decode_otlp_json,
    decode_otlp_protobuf,
    parse_listen_address,
    read_span_file,
    span_clock_domain,
    span_record,
    spans_to_correlation_events,
)
from stormlog.infer.vllm_telemetry import SPAN_SOURCE_JSONL, SPAN_SOURCE_OTLP_JSON

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "vllm"
SPAN_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "docs/schemas/inference_vllm_v1.schema.json").read_text())
)
V2_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "docs/schemas/inference_correlation_v2.schema.json").read_text())
)
METRICS_TEXT = (FIXTURES / "q05_c08_metrics_post.txt").read_text(encoding="utf-8")

REQUEST_ATTRIBUTES: dict[str, Any] = {
    "gen_ai.request.id": "chatcmpl-stormlog-run-1-c1_in8_out4_measured_0_1-0",
    "gen_ai.latency.time_in_queue": 0.001,
    "gen_ai.latency.time_in_model_prefill": 0.03,
    "gen_ai.latency.time_in_model_decode": 0.4,
    "gen_ai.latency.time_in_model_inference": 0.43,
    "gen_ai.latency.e2e": 0.5,
    "gen_ai.usage.prompt_tokens": 512,
    "gen_ai.usage.completion_tokens": 128,
    "gen_ai.request.n": 1,
    "gen_ai.request.top_p": 1.0,
    "custom.flag": True,
    "custom.list": ["a", 2],
    "custom.nested": {"k": "v"},
}
START_NS = 1_790_000_000_000_000_000
END_NS = START_NS + 500_000_000
TRACE_ID = bytes(range(16))
SPAN_ID = b"\x01\x02\x03\x04\x05\x06\x07\x08"


# ----------------------------------------------------------- generated classes
def _otlp() -> tuple[Any, Any, Any]:
    """The generated modules, or skip the test without the extra."""
    trace_service = pytest.importorskip(
        "opentelemetry.proto.collector.trace.v1.trace_service_pb2"
    )
    common = pytest.importorskip("opentelemetry.proto.common.v1.common_pb2")
    trace = pytest.importorskip("opentelemetry.proto.trace.v1.trace_pb2")
    return trace_service, common, trace


def _any_value(common: Any, value: Any) -> Any:
    any_value = common.AnyValue()
    if isinstance(value, bool):
        any_value.bool_value = value
    elif isinstance(value, int):
        any_value.int_value = value
    elif isinstance(value, float):
        any_value.double_value = value
    elif isinstance(value, str):
        any_value.string_value = value
    elif isinstance(value, bytes):
        any_value.bytes_value = value
    elif isinstance(value, list):
        any_value.array_value.values.extend(_any_value(common, item) for item in value)
    elif isinstance(value, dict):
        any_value.kvlist_value.values.extend(
            _key_value(common, key, item) for key, item in value.items()
        )
    else:
        raise TypeError(type(value))
    return any_value


def _key_value(common: Any, key: str, value: Any) -> Any:
    return common.KeyValue(key=key, value=_any_value(common, value))


def _span_message(
    name: str,
    *,
    attributes: dict[str, Any],
    start: int,
    end: int,
    kind: int = 2,
    status: int | None = 1,
) -> Any:
    _trace_service, common, trace = _otlp()
    span = trace.Span(
        name=name,
        trace_id=TRACE_ID,
        span_id=SPAN_ID,
        kind=kind,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
    )
    span.attributes.extend(_key_value(common, k, v) for k, v in attributes.items())
    if status is not None:
        span.status.code = status
    return span


def _export_request(spans: list[Any]) -> bytes:
    trace_service, common, _trace = _otlp()
    request = trace_service.ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    resource_spans.resource.attributes.extend(
        [
            _key_value(common, "service.name", "vllm"),
            _key_value(common, "host.name", "gpu-box"),
        ]
    )
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "vllm.llm_engine"
    scope_spans.scope.version = "0.30.0"
    scope_spans.spans.extend(spans)
    return bytes(request.SerializeToString())


def _request_payload() -> bytes:
    return _export_request(
        [
            _span_message(
                "llm_request", attributes=REQUEST_ATTRIBUTES, start=START_NS, end=END_NS
            ),
            _span_message(
                "Worker init", attributes={}, start=1, end=2, kind=1, status=None
            ),
        ]
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestProtobufDecoder:
    def test_decodes_every_value_kind(self) -> None:
        spans = decode_otlp_protobuf(_request_payload())
        assert [s.name for s in spans] == ["llm_request", "Worker init"]
        first = spans[0]
        assert first.attributes == REQUEST_ATTRIBUTES
        assert first.start_unix_ns == START_NS and first.end_unix_ns == END_NS
        assert first.trace_id == TRACE_ID.hex()
        assert first.span_id == SPAN_ID.hex()
        assert first.parent_span_id is None
        assert first.kind == "SERVER"
        assert first.status == {"code": "OK", "message": ""}
        assert first.resource == {"service.name": "vllm", "host.name": "gpu-box"}
        assert first.scope == {"name": "vllm.llm_engine", "version": "0.30.0"}
        assert first.dropped == {"attributes": 0, "events": 0, "links": 0}
        assert spans[1].kind == "INTERNAL"
        assert spans[1].status is None

    def test_negative_ints_and_bytes_values(self) -> None:
        payload = _export_request(
            [
                _span_message(
                    "x", attributes={"neg": -5, "raw": b"\x00\xff"}, start=1, end=2
                )
            ]
        )
        (decoded,) = decode_otlp_protobuf(payload)
        assert decoded.attributes == {"neg": -5, "raw": "00ff"}

    def test_malformed_bytes_raise(self) -> None:
        _otlp()
        with pytest.raises(ProtobufDecodeError):
            decode_otlp_protobuf(b"\x0a\xff\xff")

    def test_empty_request_has_no_spans(self) -> None:
        _otlp()
        assert decode_otlp_protobuf(b"") == []

    def test_without_the_extra_the_decoder_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vllm_spans, "_otlp_request_class", lambda: None)
        with pytest.raises(OtlpProtobufUnavailable, match="infer-otlp"):
            decode_otlp_protobuf(b"")
        assert not vllm_spans.otlp_protobuf_available()


class TestJsonReaders:
    def test_otlp_json_document(self) -> None:
        document = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "vllm"}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "vllm.llm_engine", "version": "0.30.0"},
                            "spans": [
                                {
                                    "traceId": "0af7",
                                    "spanId": "b7ad",
                                    "name": "llm_request",
                                    "kind": 2,
                                    "startTimeUnixNano": str(START_NS),
                                    "endTimeUnixNano": str(END_NS),
                                    "attributes": [
                                        {
                                            "key": "gen_ai.request.id",
                                            "value": {"stringValue": "cmpl-abc-0"},
                                        },
                                        {
                                            "key": "gen_ai.usage.prompt_tokens",
                                            "value": {"intValue": "512"},
                                        },
                                        {
                                            "key": "gen_ai.latency.e2e",
                                            "value": {"doubleValue": 0.5},
                                        },
                                        {
                                            "key": "custom.list",
                                            "value": {
                                                "arrayValue": {
                                                    "values": [{"stringValue": "a"}]
                                                }
                                            },
                                        },
                                    ],
                                    "droppedAttributesCount": 0,
                                    "status": {"code": 1},
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        (span,) = decode_otlp_json(document)
        assert span.attributes == {
            "gen_ai.request.id": "cmpl-abc-0",
            "gen_ai.usage.prompt_tokens": 512,
            "gen_ai.latency.e2e": 0.5,
            "custom.list": ["a"],
        }
        assert span.kind == "SERVER" and span.status == {"code": "OK"}
        assert span.start_unix_ns == START_NS
        assert span.resource == {"service.name": "vllm"}
        with pytest.raises(ValueError):
            decode_otlp_json({"nope": []})

    def test_read_span_file_detects_sink_jsonl_and_otlp_json(
        self, tmp_path: Path
    ) -> None:
        source, spans = read_span_file(FIXTURES / "q05_spans_sample.jsonl")
        assert source == SPAN_SOURCE_JSONL
        assert sum(1 for s in spans if s.name == "llm_request") == 12
        otlp = tmp_path / "export.json"
        otlp.write_text(
            json.dumps(
                {
                    "resourceSpans": [
                        {"scopeSpans": [{"spans": [{"name": "llm_request"}]}]}
                    ]
                }
            )
        )
        source, spans = read_span_file(otlp)
        assert source == SPAN_SOURCE_OTLP_JSON and [s.name for s in spans] == [
            "llm_request"
        ]
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"name": "ok"}\n{"attributes": {}}\n')
        with pytest.raises(ValueError, match="line 2"):
            read_span_file(bad)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_span_file(empty)


def _post(url: str, body: bytes, media: str) -> int:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": media}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code


JSON_EXPORT = json.dumps(
    {"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "Worker init"}]}]}]}
).encode()


class TestReceiver:
    def test_receives_protobuf_and_json_and_rejects_others(self) -> None:
        payload = _request_payload()
        receiver = OtlpSpanReceiver(
            listen="127.0.0.1:0", session_id="s", run_id="run-1"
        )
        assert receiver.protobuf_available
        receiver.start()
        try:
            url = f"http://{receiver.listen}/v1/traces"
            assert _post(url, payload, "application/x-protobuf") == 200
            assert _post(url, JSON_EXPORT, "application/json") == 200
            assert _post(url, b"x", "text/plain") == 415
            assert _post(url, b"\xff\xff", "application/x-protobuf") == 400
            assert (
                _post(f"http://{receiver.listen}/other", b"", "application/x-protobuf")
                == 404
            )
        finally:
            receiver.stop()
        records = receiver.drain()
        assert [r.name for r in records] == [
            "llm_request",
            "Worker init",
            "Worker init",
        ]
        first = records[0]
        assert first.request_id == "stormlog-run-1-c1_in8_out4_measured_0_1"
        assert first.clock_domain == "gpu-box/unix_epoch_ns"
        assert first.source == "otlp_http_receiver"
        assert first.received_at_ns is not None
        for record in records:
            SPAN_VALIDATOR.validate(record.to_record())
        metadata = receiver.capability_metadata()
        assert metadata["requests"] == 4
        assert metadata["spans"] == 3
        assert metadata["decode_failures"] == 1
        assert metadata["unsupported_media"] == 1
        assert metadata["protobuf_unavailable"] == 0
        assert "otlp_http_protobuf" not in metadata
        assert receiver.enabled == ["otlp_http_protobuf", "otlp_http_json"]
        assert receiver.drain() == []

    def test_without_the_extra_protobuf_is_refused_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vllm_spans, "_otlp_request_class", lambda: None)
        receiver = OtlpSpanReceiver(
            listen="127.0.0.1:0", session_id="s", run_id="run-1"
        )
        assert not receiver.protobuf_available
        assert receiver.enabled == ["otlp_http_json"]
        receiver.start()
        try:
            url = f"http://{receiver.listen}/v1/traces"
            assert _post(url, b"anything", "application/x-protobuf") == 415
            assert _post(url, JSON_EXPORT, "application/json") == 200
        finally:
            receiver.stop()
        metadata = receiver.capability_metadata()
        assert metadata["protobuf_unavailable"] == 1
        assert metadata["otlp_http_protobuf"] == OTLP_EXTRA_HINT
        assert [r.name for r in receiver.drain()] == ["Worker init"]
        assert receiver.config_record()["protobuf"] is False

    def test_listen_address_parsing(self) -> None:
        assert parse_listen_address("127.0.0.1:4318") == ("127.0.0.1", 4318)
        assert parse_listen_address(":0") == ("127.0.0.1", 0)
        assert parse_listen_address("[::1]:4318") == ("::1", 4318)
        for bad in ("4318", "host:port", "host:70000"):
            with pytest.raises(ValueError):
                parse_listen_address(bad)

    def test_clock_domain_prefers_the_resource_host(self) -> None:
        assert span_clock_domain({"host.name": "a/b"}, "1.2.3.4") == "a_b/unix_epoch_ns"
        assert span_clock_domain({}, "1.2.3.4") == "1.2.3.4/unix_epoch_ns"


@contextlib.contextmanager
def _receiver(listen: str = "127.0.0.1:0") -> Iterator[OtlpSpanReceiver]:
    receiver = OtlpSpanReceiver(listen=listen, session_id="s", run_id="run-1")
    receiver.start()
    try:
        yield receiver
    finally:
        receiver.stop()


def _raw_post(listen: str, raw: bytes) -> bytes:
    host, port = parse_listen_address(listen.replace("[", "").replace("]", ""))
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(raw)
        sock.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


class TestReceiverRobustness:
    def test_grpc_preface_is_counted_and_explained(self) -> None:
        with _receiver() as receiver:
            reply = _raw_post(receiver.listen, b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
            metadata = receiver.capability_metadata()
        assert reply == b""  # the connection is closed, no HTTP answer
        assert metadata["grpc_attempts"] == 1
        assert metadata["requests"] == 0
        assert "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf" in metadata["grpc"]

    def test_oversized_and_malformed_lengths_are_refused(self) -> None:
        with _receiver() as receiver:
            url = f"http://{receiver.listen}/v1/traces"
            huge = _raw_post(
                receiver.listen,
                b"POST /v1/traces HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                b"Content-Length: 40000000\r\n\r\n{}",
            )
            bad = _raw_post(
                receiver.listen,
                b"POST /v1/traces HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                b"Content-Length: abc\r\n\r\n{}",
            )
            assert _post(url, JSON_EXPORT, "application/json") == 200
            metadata = receiver.capability_metadata()
        assert huge.startswith(b"HTTP/1.1 413")
        assert bad.startswith(b"HTTP/1.1 400")
        assert metadata["oversized"] == 1
        assert metadata["bad_requests"] == 1
        assert metadata["spans"] == 1

    def test_gzip_bodies_are_decoded_and_other_encodings_refused(self) -> None:
        import gzip

        with _receiver() as receiver:
            url = f"http://{receiver.listen}/v1/traces"
            request = urllib.request.Request(
                url,
                data=gzip.compress(JSON_EXPORT),
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
            broken = urllib.request.Request(
                url,
                data=b"\x1f\x8bnot gzip",
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(broken, timeout=5)
            assert exc_info.value.code == 400
            brotli = urllib.request.Request(
                url,
                data=JSON_EXPORT,
                headers={"Content-Type": "application/json", "Content-Encoding": "br"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(brotli, timeout=5)
            assert exc_info.value.code == 415
            assert [r.name for r in receiver.drain()] == ["Worker init"]
            metadata = receiver.capability_metadata()
        assert metadata["decode_failures"] == 1
        assert metadata["unsupported_media"] == 1

    def test_a_handler_bug_is_counted_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(body: bytes, media: str) -> list[RawSpan]:
            raise RuntimeError("boom")

        monkeypatch.setattr(vllm_spans, "_decode_export", explode)
        with _receiver() as receiver:
            url = f"http://{receiver.listen}/v1/traces"
            assert _post(url, JSON_EXPORT, "application/json") == 400
            monkeypatch.undo()
            assert _post(url, JSON_EXPORT, "application/json") == 200
            metadata = receiver.capability_metadata()
        assert metadata["handler_errors"] == 1
        assert metadata["spans"] == 1

    def test_ipv6_listen_uses_an_ipv6_socket(self) -> None:
        try:
            probe = socket.socket(socket.AF_INET6)
            probe.bind(("::1", 0))
            probe.close()
        except OSError:
            pytest.skip("no IPv6 loopback")
        with _receiver("[::1]:0") as receiver:
            assert receiver.listen.startswith("[::1]:")
            assert (
                _post(
                    f"http://{receiver.listen}/v1/traces",
                    JSON_EXPORT,
                    "application/json",
                )
                == 200
            )
            assert receiver.drain()[0].clock_domain == "::1/unix_epoch_ns"


def _raw_request_span(attributes: dict[str, Any]) -> RawSpan:
    return RawSpan(
        name="llm_request",
        span_id=SPAN_ID.hex(),
        start_unix_ns=START_NS,
        end_unix_ns=END_NS,
        attributes=attributes,
    )


class TestCorrelationMapping:
    def test_request_and_stage_events_validate(self) -> None:
        record = span_record(
            _raw_request_span(REQUEST_ATTRIBUTES),
            session_id="s",
            run_id="run-1",
            source="otlp_json_file",
            clock_domain="h/unix_epoch_ns",
        )
        events = spans_to_correlation_events([record])
        for event in events:
            V2_VALIDATOR.validate(event.to_record())
        request = events[0]
        assert isinstance(request, RequestEvent)
        assert request.request_ref.id == "stormlog-run-1-c1_in8_out4_measured_0_1"
        assert request.backend_request_ref is not None
        assert request.backend_request_ref.id == REQUEST_ATTRIBUTES["gen_ai.request.id"]
        assert (request.start_ns, request.end_ns) == (START_NS, END_NS)
        assert (request.input_tokens, request.output_tokens) == (512, 128)
        assert request.context.provenance == "reported"
        stages = [e for e in events if isinstance(e, StageEvent)]
        assert [s.name for s in stages] == ["queue", "prefill", "decode", "inference"]
        assert all(s.context.provenance == "estimated" for s in stages)
        queue, prefill, decode, inference = stages
        assert (queue.start_ns, queue.end_ns) == (START_NS, START_NS + 1_000_000)
        assert prefill.start_ns == queue.end_ns
        assert decode.start_ns == prefill.end_ns
        assert inference.start_ns == queue.end_ns
        assert queue.end_ns is not None
        assert inference.end_ns == queue.end_ns + 430_000_000
        assert stages[0].metadata["native_attribute"] == "gen_ai.latency.time_in_queue"
        assert "not GPU time" in stages[0].metadata["meaning"]

    def test_non_request_spans_and_missing_ids_are_skipped(self) -> None:
        worker = span_record(
            RawSpan(name="Worker init", start_unix_ns=1, end_unix_ns=2),
            session_id="s",
            run_id="r",
            source="otlp_json_file",
            clock_domain="h/unix_epoch_ns",
        )
        anonymous = span_record(
            _raw_request_span({"x": 1}),
            session_id="s",
            run_id="r",
            source="otlp_json_file",
            clock_domain="h/unix_epoch_ns",
        )
        assert spans_to_correlation_events([worker, anonymous]) == []


# ----------------------------------------------------------------- profile run
class _ExportingVllmHandler(BaseHTTPRequestHandler):
    """A chat endpoint that exports a span for each request, like vLLM does."""

    protocol_version = "HTTP/1.1"
    otlp_url = ""
    # Like vLLM's batch processor, the export may leave after the answer.
    export_delay_seconds = 0.0

    def do_GET(self) -> None:  # noqa: N802
        body = METRICS_TEXT.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        request_id = self.headers.get("X-Request-Id") or "none"
        if type(self).otlp_url:
            delay = type(self).export_delay_seconds
            if delay:
                threading.Timer(delay, self._export, args=(request_id,)).start()
            else:
                self._export(request_id)
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _export(self, request_id: str) -> None:
        attributes = {
            **REQUEST_ATTRIBUTES,
            "gen_ai.request.id": f"chatcmpl-{request_id}-0",
        }
        payload = _export_request(
            [
                _span_message(
                    "llm_request", attributes=attributes, start=START_NS, end=END_NS
                )
            ]
        )
        request = urllib.request.Request(
            type(self).otlp_url,
            data=payload,
            headers={"Content-Type": "application/x-protobuf"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except OSError:
            # Like vLLM's exporter: a receiver that is gone loses the batch.
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _exporting_vllm(otlp_url: str) -> Iterator[str]:
    _ExportingVllmHandler.otlp_url = otlp_url
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ExportingVllmHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _run(
    tmp_path: Path, origin: str, listen: str | None, drain_seconds: float = 0.0
) -> list[dict[str, Any]]:
    output = tmp_path / "infer.jsonl"
    warnings: list[str] = []
    InferenceProfiler(
        ProfileConfig(
            endpoint=f"{origin}/v1/chat/completions",
            model="fake-model",
            concurrency=(1,),
            input_tokens=(8,),
            output_tokens=(4,),
            output_path=str(output),
            stream=False,
            request_count=2,
            tokenizer="none",
            system_sampler="none",
            run_id="run-1",
            vllm_spans_listen=listen,
            vllm_spans_drain_seconds=drain_seconds,
        ),
        on_warning=warnings.append,
    ).run()
    records = [
        json.loads(line) for line in output.read_text().splitlines() if line.strip()
    ]
    records.append({"event_type": "_warnings", "warnings": warnings})
    return records


def _of_type(records: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("event_type") == event_type]


def _span_capability(records: list[dict[str, Any]]) -> dict[str, Any]:
    return [
        r
        for r in _of_type(records, "infer.capabilities")
        if r["component"] == "vllm.spans"
    ][0]


class TestProfileReceiver:
    def test_spans_land_in_the_artifact_and_join_by_header(
        self, tmp_path: Path
    ) -> None:
        _otlp()
        port = _free_port()
        with _exporting_vllm(f"http://127.0.0.1:{port}/v1/traces") as origin:
            records = _run(tmp_path, origin, f"127.0.0.1:{port}")
        spans = _of_type(records, "infer.vllm_span")
        requests = _of_type(records, "infer.request")
        assert len(spans) == len(requests) == 2
        for span in spans:
            SPAN_VALIDATOR.validate(span)
        assert {s["request_id"] for s in spans} == {r["x_request_id"] for r in requests}
        session = _of_type(records, "infer.session")[0]
        assert session["config"]["vllm_spans"] == {
            "listen": f"127.0.0.1:{port}",
            "path": "/v1/traces",
            "protobuf": True,
            "drain_seconds": 0.0,
        }
        capability = _span_capability(records)
        assert capability["available"] is True
        assert capability["collected"] == ["otlp_http_protobuf"]
        assert capability["enabled"] == ["otlp_http_protobuf", "otlp_http_json"]
        assert capability["metadata"]["spans"] == 2
        assert records[-1]["warnings"] == []

    def test_drain_window_catches_a_batch_exported_after_the_last_request(
        self, tmp_path: Path
    ) -> None:
        """vLLM flushes spans on a schedule; the receiver waits for that batch."""
        _otlp()
        port = _free_port()
        _ExportingVllmHandler.export_delay_seconds = 0.4
        try:
            with _exporting_vllm(f"http://127.0.0.1:{port}/v1/traces") as origin:
                late = _run(tmp_path, origin, f"127.0.0.1:{port}", drain_seconds=0.0)
            with _exporting_vllm(f"http://127.0.0.1:{port}/v1/traces") as origin:
                caught = _run(tmp_path, origin, f"127.0.0.1:{port}", drain_seconds=1.5)
        finally:
            _ExportingVllmHandler.export_delay_seconds = 0.0
        assert len(_of_type(late, "infer.vllm_span")) < 2
        assert len(_of_type(caught, "infer.vllm_span")) == 2
        session = _of_type(caught, "infer.session")[0]
        assert session["config"]["vllm_spans"]["drain_seconds"] == 1.5

    def test_without_the_extra_the_run_warns_once_and_records_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vllm_spans, "_otlp_request_class", lambda: None)
        port = _free_port()
        with _exporting_vllm("") as origin:
            records = _run(tmp_path, origin, f"127.0.0.1:{port}")
        assert [r["status"] for r in _of_type(records, "infer.request")] == ["ok", "ok"]
        capability = _span_capability(records)
        assert capability["available"] is True
        assert "otlp_http_protobuf" in capability["supported"]
        assert capability["enabled"] == ["otlp_http_json"]
        assert capability["collected"] == []
        assert capability["metadata"]["otlp_http_protobuf"] == OTLP_EXTRA_HINT
        warnings = records[-1]["warnings"]
        assert len(warnings) == 1 and "infer-otlp" in warnings[0]

    def test_port_in_use_is_recorded_not_fatal(self, tmp_path: Path) -> None:
        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        try:
            with _exporting_vllm("") as origin:
                records = _run(tmp_path, origin, f"127.0.0.1:{port}")
        finally:
            holder.close()
        assert [r["status"] for r in _of_type(records, "infer.request")] == ["ok", "ok"]
        assert _of_type(records, "infer.vllm_span") == []
        session = _of_type(records, "infer.session")[0]
        assert session["config"]["vllm_spans"] is None
        capability = _span_capability(records)
        assert capability["available"] is False
        assert capability["supported"] == []
        assert capability["metadata"]["listen"] == f"127.0.0.1:{port}"
        assert "OSError" in capability["metadata"]["error"]
        warnings = records[-1]["warnings"]
        assert len(warnings) == 1 and "could not listen" in warnings[0]
