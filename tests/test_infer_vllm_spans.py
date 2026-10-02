"""vLLM span ingest: OTLP protobuf and JSON, the receiver, and the v2 mapping."""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from stormlog.infer.config import ProfileConfig
from stormlog.infer.correlation_events import RequestEvent, StageEvent
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.vllm_spans import (
    OtlpSpanReceiver,
    ProtobufDecodeError,
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


# ----------------------------------------------------------------- an encoder
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _ld(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _vi(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _f64(number: int, value: int) -> bytes:
    return _varint((number << 3) | 1) + struct.pack("<Q", value)


def _any(value: Any) -> bytes:
    if isinstance(value, bool):
        return _vi(2, int(value))
    if isinstance(value, int):
        return _vi(3, value & ((1 << 64) - 1))
    if isinstance(value, float):
        return _varint((4 << 3) | 1) + struct.pack("<d", value)
    if isinstance(value, str):
        return _ld(1, value.encode())
    if isinstance(value, list):
        return _ld(5, b"".join(_ld(1, _any(item)) for item in value))
    if isinstance(value, dict):
        return _ld(6, _kvs(value))
    raise TypeError(type(value))


def _kv(key: str, value: Any) -> bytes:
    """One KeyValue message body."""
    return _ld(1, key.encode()) + _ld(2, _any(value))


def _kvs(values: dict[str, Any]) -> bytes:
    """Repeated KeyValue under field 1: Resource.attributes, KeyValueList.values."""
    return b"".join(_ld(1, _kv(key, value)) for key, value in values.items())


def _span(
    name: str,
    *,
    attributes: dict[str, Any],
    start: int,
    end: int,
    trace_id: bytes = bytes(range(16)),
    span_id: bytes = b"\x01\x02\x03\x04\x05\x06\x07\x08",
    kind: int = 2,
    status: int = 1,
) -> bytes:
    return (
        _ld(1, trace_id)
        + _ld(2, span_id)
        + _ld(5, name.encode())
        + _vi(6, kind)
        + _f64(7, start)
        + _f64(8, end)
        # Span.attributes is repeated KeyValue under field 9, one entry each.
        + b"".join(_ld(9, _kv(key, value)) for key, value in attributes.items())
        + _vi(10, 0)
        + _ld(15, _ld(2, b"") + _vi(3, status))
    )


def _export_request(spans: list[bytes]) -> bytes:
    resource = _ld(1, _kvs({"service.name": "vllm", "host.name": "gpu-box"}))
    scope = _ld(1, _ld(1, b"vllm.llm_engine") + _ld(2, b"0.30.0"))
    scope_spans = scope + b"".join(_ld(2, span) for span in spans)
    return _ld(1, resource + _ld(2, scope_spans))


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestProtobufDecoder:
    def test_decodes_every_value_kind(self) -> None:
        payload = _export_request(
            [
                _span(
                    "llm_request",
                    attributes=REQUEST_ATTRIBUTES,
                    start=START_NS,
                    end=END_NS,
                ),
                _span("Worker init", attributes={}, start=1, end=2, kind=1, status=0),
            ]
        )
        spans = decode_otlp_protobuf(payload)
        assert [s.name for s in spans] == ["llm_request", "Worker init"]
        first = spans[0]
        assert first.attributes == REQUEST_ATTRIBUTES
        assert first.start_unix_ns == START_NS and first.end_unix_ns == END_NS
        assert first.trace_id == bytes(range(16)).hex()
        assert first.span_id == "0102030405060708"
        assert first.kind == "SERVER"
        assert first.status == {"message": "", "code": "OK"}
        assert first.resource == {"service.name": "vllm", "host.name": "gpu-box"}
        assert first.scope == {"name": "vllm.llm_engine", "version": "0.30.0"}
        assert first.dropped == {"attributes": 0}
        assert spans[1].kind == "INTERNAL" and spans[1].status == {
            "message": "",
            "code": "UNSET",
        }

    def test_negative_ints_and_unknown_fields_are_handled(self) -> None:
        span = (
            _span("x", attributes={"neg": -5}, start=1, end=2)
            + _vi(99, 7)
            + _ld(98, b"?")
        )
        (decoded,) = decode_otlp_protobuf(_export_request([span]))
        assert decoded.attributes == {"neg": -5}

    @pytest.mark.parametrize(
        "data", [b"\x0a\xff\xff", b"\x0b", b"\x0a\x03\x01\x02", bytes([0x08 | 3])]
    )
    def test_malformed_bytes_raise(self, data: bytes) -> None:
        with pytest.raises(ProtobufDecodeError):
            decode_otlp_protobuf(data)

    def test_empty_request_has_no_spans(self) -> None:
        assert decode_otlp_protobuf(b"") == []


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


class TestReceiver:
    def _post(self, url: str, body: bytes, media: str) -> int:
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": media}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_receives_protobuf_and_json_and_rejects_others(self) -> None:
        receiver = OtlpSpanReceiver(
            listen="127.0.0.1:0", session_id="s", run_id="run-1"
        )
        receiver.start()
        try:
            url = f"http://{receiver.listen}/v1/traces"
            payload = _export_request(
                [
                    _span(
                        "llm_request",
                        attributes=REQUEST_ATTRIBUTES,
                        start=START_NS,
                        end=END_NS,
                    )
                ]
            )
            assert self._post(url, payload, "application/x-protobuf") == 200
            json_doc = {
                "resourceSpans": [
                    {"scopeSpans": [{"spans": [{"name": "Worker init"}]}]}
                ]
            }
            assert (
                self._post(url, json.dumps(json_doc).encode(), "application/json")
                == 200
            )
            assert self._post(url, b"x", "text/plain") == 415
            assert self._post(url, b"\xff\xff", "application/x-protobuf") == 400
            assert (
                self._post(
                    f"http://{receiver.listen}/other", b"", "application/x-protobuf"
                )
                == 404
            )
        finally:
            receiver.stop()
        records = receiver.drain()
        assert [r.name for r in records] == ["llm_request", "Worker init"]
        first = records[0]
        assert first.request_id == "stormlog-run-1-c1_in8_out4_measured_0_1"
        assert first.clock_domain == "gpu-box/unix_epoch_ns"
        assert first.source == "otlp_http_receiver"
        assert first.received_at_ns is not None
        for record in records:
            SPAN_VALIDATOR.validate(record.to_record())
        metadata = receiver.capability_metadata()
        assert metadata["requests"] == 4
        assert metadata["spans"] == 2
        assert metadata["decode_failures"] == 1
        assert metadata["unsupported_media"] == 1
        assert receiver.drain() == []

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


class TestCorrelationMapping:
    def test_request_and_stage_events_validate(self) -> None:
        raw = decode_otlp_protobuf(
            _export_request(
                [
                    _span(
                        "llm_request",
                        attributes=REQUEST_ATTRIBUTES,
                        start=START_NS,
                        end=END_NS,
                    )
                ]
            )
        )[0]
        record = span_record(
            raw,
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
            decode_otlp_protobuf(
                _export_request([_span("Worker init", attributes={}, start=1, end=2)])
            )[0],
            session_id="s",
            run_id="r",
            source="otlp_json_file",
            clock_domain="h/unix_epoch_ns",
        )
        anonymous = span_record(
            decode_otlp_protobuf(
                _export_request(
                    [_span("llm_request", attributes={"x": 1}, start=1, end=2)]
                )
            )[0],
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
            [_span("llm_request", attributes=attributes, start=START_NS, end=END_NS)]
        )
        request = urllib.request.Request(
            type(self).otlp_url,
            data=payload,
            headers={"Content-Type": "application/x-protobuf"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5):
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


def _run(tmp_path: Path, origin: str, listen: str | None) -> list[dict[str, Any]]:
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
        ),
        on_warning=warnings.append,
    ).run()
    records = [
        json.loads(line) for line in output.read_text().splitlines() if line.strip()
    ]
    records.append({"event_type": "_warnings", "warnings": warnings})
    return records


class TestProfileReceiver:
    def test_spans_land_in_the_artifact_and_join_by_header(
        self, tmp_path: Path
    ) -> None:
        port = _free_port()
        with _exporting_vllm(f"http://127.0.0.1:{port}/v1/traces") as origin:
            records = _run(tmp_path, origin, f"127.0.0.1:{port}")
        spans = [r for r in records if r.get("event_type") == "infer.vllm_span"]
        requests = [r for r in records if r.get("event_type") == "infer.request"]
        assert len(spans) == len(requests) == 2
        for span in spans:
            SPAN_VALIDATOR.validate(span)
        assert {s["request_id"] for s in spans} == {r["x_request_id"] for r in requests}
        session = [r for r in records if r.get("event_type") == "infer.session"][0]
        assert session["config"]["vllm_spans"] == {
            "listen": f"127.0.0.1:{port}",
            "path": "/v1/traces",
        }
        capability = [
            r
            for r in records
            if r.get("event_type") == "infer.capabilities"
            and r["component"] == "vllm.spans"
        ][0]
        assert capability["available"] is True
        assert capability["collected"] == ["otlp_http_protobuf"]
        assert "otlp_http_json" in capability["enabled"]
        assert capability["metadata"]["spans"] == 2
        assert records[-1]["warnings"] == []

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
        requests = [r for r in records if r.get("event_type") == "infer.request"]
        assert [r["status"] for r in requests] == ["ok", "ok"]
        assert [r for r in records if r.get("event_type") == "infer.vllm_span"] == []
        session = [r for r in records if r.get("event_type") == "infer.session"][0]
        assert session["config"]["vllm_spans"] is None
        capability = [
            r
            for r in records
            if r.get("event_type") == "infer.capabilities"
            and r["component"] == "vllm.spans"
        ][0]
        assert capability["available"] is False
        assert capability["supported"] == []
        assert capability["metadata"]["listen"] == f"127.0.0.1:{port}"
        assert "OSError" in capability["metadata"]["error"]
        warnings = records[-1]["warnings"]
        assert len(warnings) == 1 and "could not listen" in warnings[0]
