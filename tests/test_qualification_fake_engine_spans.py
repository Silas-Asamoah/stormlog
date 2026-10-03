"""The fake engine's OTLP request spans, received by Stormlog's own receiver."""

from __future__ import annotations

import contextlib
import socket
from pathlib import Path
from typing import Iterator

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from stormlog.infer.vllm_spans import JSON_MEDIA, PROTOBUF_MEDIA, OtlpSpanReceiver
from stormlog.infer.vllm_telemetry import VllmSpanRecord
from tests.qualification_fake_engine_helpers import (
    chat,
    run_profile,
    wait_until,
    words,
)

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT = "00f067aa0ba902b7"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _receiver() -> Iterator[OtlpSpanReceiver]:
    receiver = OtlpSpanReceiver(
        listen=f"127.0.0.1:{_free_port()}", session_id="s-1", run_id="run-1"
    )
    receiver.start()
    try:
        yield receiver
    finally:
        receiver.stop()


def _engine(
    receiver: OtlpSpanReceiver, *, span_encoding: str = "protobuf"
) -> FakeEngine:
    return FakeEngine(
        FakeEngineConfig(
            step_seconds=0.001,
            spans_endpoint=f"http://{receiver.listen}/v1/traces",
            span_export_seconds=0.05,
            span_encoding=span_encoding,
        )
    )


def _received(receiver: OtlpSpanReceiver, spans: list[VllmSpanRecord]) -> int:
    spans.extend(receiver.drain())
    return len(spans)


def test_spans_reach_stormlogs_receiver_and_name_their_request() -> None:
    spans: list[VllmSpanRecord] = []
    with _receiver() as receiver:
        with _engine(receiver) as engine:
            chat(engine, words(8, "a"), max_tokens=3, request_id="stormlog-run-1-a")
            chat(engine, words(8, "b"), max_tokens=3, request_id="stormlog-run-1-b")
        assert wait_until(lambda: _received(receiver, spans) >= 2)
    assert {span.name for span in spans} == {"llm_request"}
    assert {span.request_id for span in spans} == {
        "stormlog-run-1-a",
        "stormlog-run-1-b",
    }
    attributes = spans[0].attributes
    assert attributes["gen_ai.usage.completion_tokens"] == 3
    for name in ("time_in_queue", "time_to_first_token", "time_in_model_inference"):
        assert attributes[f"gen_ai.latency.{name}"] >= 0


@pytest.mark.parametrize(
    ("encoding", "media"),
    [("protobuf", PROTOBUF_MEDIA), ("json", JSON_MEDIA)],
)
def test_spans_leave_as_protobuf_like_vllms_exporter_unless_json_is_asked(
    encoding: str, media: str
) -> None:
    # vLLM 0.30 exports through the OpenTelemetry SDK's OTLP/HTTP exporter,
    # which sends protobuf only.
    spans: list[VllmSpanRecord] = []
    with _receiver() as receiver:
        with _engine(receiver, span_encoding=encoding) as engine:
            chat(engine, words(4, "a"), max_tokens=2, request_id="stormlog-run-1-a")
        assert wait_until(lambda: _received(receiver, spans) >= 1)
        by_media = dict(receiver.stats.by_media)
    assert by_media == {media: 1}
    assert spans[0].request_id == "stormlog-run-1-a"
    assert spans[0].attributes["gen_ai.usage.completion_tokens"] == 2


def test_a_traceparent_makes_the_span_its_child() -> None:
    spans: list[VllmSpanRecord] = []
    traceparent = f"00-{TRACE_ID}-{PARENT}-01"
    with _receiver() as receiver:
        with _engine(receiver) as engine:
            chat(engine, "x", headers={"traceparent": traceparent})
            chat(engine, "y")
        assert wait_until(lambda: _received(receiver, spans) >= 2)
    child = [span for span in spans if span.parent_span_id]
    assert [(span.trace_id, span.parent_span_id) for span in child] == [
        (TRACE_ID, PARENT)
    ]


def test_spans_can_arrive_late_and_twice() -> None:
    spans: list[VllmSpanRecord] = []
    with _receiver() as receiver:
        engine = _engine(receiver)
        engine.controls.span_delay_seconds = 0.4
        engine.controls.span_duplicates = True
        with engine:
            chat(engine, "x", max_tokens=1)
            early = _received(receiver, spans)
            assert wait_until(lambda: _received(receiver, spans) >= 2, timeout=5)
    assert early == 0
    assert len(spans) == 2
    assert spans[0].span_id == spans[1].span_id


def test_the_receiver_refuses_oversized_and_inflating_bodies() -> None:
    with _receiver() as receiver:
        with _engine(receiver) as engine:
            exporter = engine.spans
            assert exporter is not None
            statuses = [exporter.send_abusive(k) for k in ("oversized", "gzip_bomb")]
        oversized = receiver.stats.oversized
    assert statuses == [413, 413]
    assert oversized == 2


def test_infer_profile_joins_the_fake_spans(tmp_path: Path) -> None:
    port = _free_port()
    config = FakeEngineConfig(
        step_seconds=0.001,
        spans_endpoint=f"http://127.0.0.1:{port}/v1/traces",
        span_export_seconds=0.05,
    )
    with FakeEngine(config) as engine:
        report = run_profile(
            engine,
            tmp_path / "infer.jsonl",
            vllm_spans_listen=f"127.0.0.1:{port}",
            vllm_spans_drain_seconds=1.0,
        )
    (case,) = report["telemetry"]["vllm"]["cases"].values()
    assert case["spans"]["requests_with_span"] == 4
