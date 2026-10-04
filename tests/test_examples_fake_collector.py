"""The fake collector for outage episodes: it stores first, then answers."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from examples.observability import fake_collector
from stormlog._export.delivery import RetryPolicy
from stormlog._export.otlp_encoding import ProtobufEncoding
from stormlog._export.otlp_http import Destination, OtlpHttpTransport
from stormlog._export.span_export import HttpSink, SpanExporter
from stormlog._export.spans import KIND_INTERNAL, Scope, Span

pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")


def _span(index: int) -> Span:
    return Span(
        name="t",
        trace_id=f"{index + 1:032x}",
        span_id=f"{index + 1:016x}",
        kind=KIND_INTERNAL,
        start_ns=1,
        end_ns=2,
    )


def _exporter(url: str) -> SpanExporter[int]:
    transport = OtlpHttpTransport(
        Destination.parse(url),
        media_type="application/x-protobuf",
        attempt_seconds=0.3,
    )
    return SpanExporter(
        HttpSink(transport),
        ProtobufEncoding(),
        resource=(),
        scope=Scope("t", "0"),
        to_span=_span,
        retry=RetryPolicy(initial_seconds=0.01, max_seconds=0.02, max_attempts=2),
        schedule_delay=0.0,
    )


def test_x3_a_slow_collector_leaves_spans_unknown_within_the_bounds(
    tmp_path: Path,
) -> None:
    store = tmp_path / "spans.jsonl"
    collector = fake_collector.FakeCollector("127.0.0.1:0", store, delay_seconds=1.0)
    collector.start()
    try:
        exporter = _exporter(f"http://{collector.address}/v1/traces")
        exporter.start()
        for index in range(4):
            exporter.offer(index, 64)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and exporter.accounting()["in_flight"]:
            time.sleep(0.05)
        exporter.close(1.0)
        counts = json.loads(
            urllib.request.urlopen(f"http://{collector.address}/counts").read()
        )
    finally:
        collector.stop()
    accounting = exporter.accounting()
    assert set(accounting["unknown"]) == {"timeout_after_send"}
    assert accounting["exported"] == 0
    unknown = sum(accounting["unknown"].values())
    unique, raw = counts["unique_spans"], counts["raw_spans"]
    # The fake stores before it answers: every span offered is stored, so
    # the bounds are met, not merely by storing nothing.
    assert unique == 4
    assert accounting["exported"] <= unique <= accounting["exported"] + unknown
    assert raw - unique <= accounting["max_extra_copies"]
    # The store was fsynced before each answer, so the file agrees.
    assert fake_collector.count_store(store) == {
        "raw_spans": raw,
        "unique_spans": unique,
    }


def test_a_refusing_collector_stores_nothing_it_refuses(tmp_path: Path) -> None:
    store = tmp_path / "spans.jsonl"
    collector = fake_collector.FakeCollector("127.0.0.1:0", store, refuse_first=1)
    collector.start()
    try:
        exporter = _exporter(f"http://{collector.address}/v1/traces")
        exporter.start()
        for index in range(3):
            exporter.offer(index, 64)
        exporter.close(3.0)
    finally:
        collector.stop()
    assert exporter.accounting()["exported"] == 3
    assert collector.counts() == {"requests": 2, "raw_spans": 3, "unique_spans": 3}


def test_counting_a_store_skips_a_broken_last_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "spans.jsonl"
    store.write_text(
        '{"trace_id": "a", "span_id": "1"}\n'
        '{"trace_id": "a", "span_id": "1"}\n'
        '{"trace_id": "a", "span_id": "2"}\n'
        '{"trace_id": "b", "spa'
    )
    assert fake_collector.main(["--count", str(store)]) == 0
    assert json.loads(capsys.readouterr().out) == {"raw_spans": 3, "unique_spans": 2}


def _post(
    collector: fake_collector.FakeCollector, path: str, body: bytes, media: str
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        f"http://{collector.address}{path}",
        data=body,
        headers={"Content-Type": media},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _json_export(count: int, start: int = 100) -> bytes:
    spans = [
        {"traceId": f"{i + 1:032x}", "spanId": f"{i + 1:016x}", "name": "t"}
        for i in range(start, start + count)
    ]
    return json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}).encode()


def test_it_answers_as_a_collector_does(tmp_path: Path) -> None:
    collector = fake_collector.FakeCollector("127.0.0.1:0", tmp_path / "s.jsonl")
    collector.start()
    try:
        body = _json_export(2)
        assert _post(collector, "/v1/metrics", body, "application/json")[0] == 404
        assert _post(collector, "/", body, "application/json")[0] == 404
        assert _post(collector, "/v1/traces", body, "text/plain")[0] == 415
        status, headers, answer = _post(
            collector, "/v1/traces", body, "application/json"
        )
    finally:
        collector.stop()
    # A JSON request is answered in JSON.
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(answer) == {}
    assert collector.counts()["raw_spans"] == 2


def test_it_can_reject_part_of_each_export(tmp_path: Path) -> None:
    store = tmp_path / "spans.jsonl"
    collector = fake_collector.FakeCollector("127.0.0.1:0", store, partial_rejected=1)
    collector.start()
    try:
        exporter = _exporter(f"http://{collector.address}/v1/traces")
        for index in range(3):
            exporter.offer(index, 64)
        exporter.start()
        exporter.close(3.0)
        status, _, answer = _post(
            collector, "/v1/traces", _json_export(2), "application/json"
        )
    finally:
        collector.stop()
    accounting = exporter.accounting()
    assert (accounting["exported"], accounting["rejected"]) == (2, 1)
    assert json.loads(answer)["partialSuccess"]["rejectedSpans"] == "1"
    assert fake_collector.count_store(store)["unique_spans"] == 3  # 2 + 1


def test_a_refusal_can_carry_retry_after(tmp_path: Path) -> None:
    collector = fake_collector.FakeCollector(
        "127.0.0.1:0", tmp_path / "s.jsonl", refuse_first=1, retry_after=2
    )
    collector.start()
    try:
        status, headers, _ = _post(
            collector, "/v1/traces", _json_export(1), "application/json"
        )
    finally:
        collector.stop()
    assert status == 503 and headers["Retry-After"] == "2"
