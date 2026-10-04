"""Against a real OpenTelemetry Collector: opt-in, with STORMLOG_OTELCOL.

Set ``STORMLOG_OTELCOL`` to an ``otelcol-contrib`` binary to run these. They
check what tests with fakes cannot: that a real collector accepts
Stormlog's exports in both encodings and reads back the same spans, and
that the example's analysis filter, as the collector parses it, passes
vLLM's spans and nothing else.
"""

import json
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from examples.observability import fake_collector
from stormlog._export.otlp_encoding import JsonEncoding, ProtobufEncoding
from stormlog._export.otlp_http import CONFIRMED, Destination, OtlpHttpTransport
from stormlog._export.spans import KIND_CLIENT, Scope, Span
from stormlog.infer.vllm_spans import read_span_file

OTELCOL = os.environ.get("STORMLOG_OTELCOL")
pytestmark = pytest.mark.skipif(
    not OTELCOL, reason="set STORMLOG_OTELCOL to an otelcol-contrib binary"
)
pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
yaml = pytest.importorskip("yaml")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "observability"


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _otelcol(config: dict[str, Any], tmp_path: Path) -> Iterator[None]:
    path = tmp_path / "otelcol.yaml"
    path.write_text(yaml.safe_dump(config))
    assert OTELCOL is not None
    with (tmp_path / "otelcol.log").open("wb") as log:
        process = subprocess.Popen(
            [OTELCOL, f"--config={path}"], stdout=log, stderr=subprocess.STDOUT
        )
    try:
        port = int(
            config["receivers"]["otlp"]["protocols"]["http"]["endpoint"].split(":")[1]
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            assert process.poll() is None, (tmp_path / "otelcol.log").read_text()
            time.sleep(0.2)
        yield
    finally:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()


def _span(index: int, resource_marker: bool = False) -> Span:
    return Span(
        name="llm_request" if resource_marker else "stormlog.infer.request",
        trace_id=f"{index + 1:032x}",
        span_id=f"{index + 1:016x}",
        kind=KIND_CLIENT,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_000_000_100_000_000,
        attributes=(("stormlog.request_id", f"r{index}"), ("n", index)),
    )


def _send(port: int, encoding: Any, spans: list[Span], resource: Any) -> str:
    transport = OtlpHttpTransport(
        Destination.parse(f"http://127.0.0.1:{port}"),
        media_type=encoding.media_type,
    )
    body = encoding.request(
        resource, Scope("stormlog.infer", "0"), [encoding.unit(s)[0] for s in spans]
    )
    return transport.send(body, spans=len(spans)).kind


@pytest.mark.parametrize("encoding", [ProtobufEncoding(), JsonEncoding()])
def test_a_real_collector_takes_both_encodings(tmp_path: Path, encoding: Any) -> None:
    port = _port()
    out = tmp_path / "traces.jsonl"
    config = {
        "receivers": {
            "otlp": {"protocols": {"http": {"endpoint": f"127.0.0.1:{port}"}}}
        },
        "exporters": {"file": {"path": str(out)}},
        "service": {
            "pipelines": {"traces": {"receivers": ["otlp"], "exporters": ["file"]}},
            "telemetry": {"metrics": {"level": "none"}},
        },
    }
    with _otelcol(config, tmp_path):
        assert _send(port, encoding, [_span(i) for i in range(5)], ()) == CONFIRMED
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (out.exists() and out.stat().st_size):
            time.sleep(0.2)
    _source, spans = read_span_file(out)
    assert sorted(s.attributes["stormlog.request_id"] for s in spans) == [
        f"r{i}" for i in range(5)
    ]


def test_the_example_filter_passes_only_vllm_spans(tmp_path: Path) -> None:
    example = yaml.safe_load((EXAMPLES / "otelcol.yaml").read_text())
    port = _port()
    store = tmp_path / "forwarded.jsonl"
    receiver = fake_collector.FakeCollector("127.0.0.1:0", store)
    receiver.start()
    try:
        exporter = dict(example["exporters"]["otlphttp/stormlog"])
        exporter["endpoint"] = f"http://{receiver.address}"
        config = {
            "receivers": {
                "otlp": {"protocols": {"http": {"endpoint": f"127.0.0.1:{port}"}}}
            },
            "processors": {
                "memory_limiter": example["processors"]["memory_limiter"],
                "filter/vllm-only": example["processors"]["filter/vllm-only"],
            },
            "exporters": {"otlphttp/stormlog": exporter},
            "service": {
                "pipelines": {
                    "traces/stormlog-analysis": example["service"]["pipelines"][
                        "traces/stormlog-analysis"
                    ]
                },
                "telemetry": {"metrics": {"level": "none"}},
            },
        }
        with _otelcol(config, tmp_path):
            encoding = ProtobufEncoding()
            vllm = (("service.name", "vllm"), ("vllm.instrumenting_module_name", "m"))
            sent = _send(port, encoding, [_span(i, True) for i in range(3)], vllm)
            assert sent == CONFIRMED
            for name in ("stormlog", "vllm", "other-app"):
                spans = [_span(10 + i) for i in range(2)]
                resource = (("service.name", name),)
                assert _send(port, encoding, spans, resource) == CONFIRMED
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and receiver.counts()["raw_spans"] < 3:
                time.sleep(0.2)
            time.sleep(2)  # anything the filter let through has arrived too
    finally:
        receiver.stop()
    names = [json.loads(line)["name"] for line in store.read_text().splitlines()]
    assert names == ["llm_request"] * 3
