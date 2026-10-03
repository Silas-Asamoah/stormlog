"""The fake engine's OTLP request spans, received by Stormlog's own receiver."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Iterator

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.fake_engine.config import Controls
from examples.qualification.fake_engine.engine import FakeRequest, prompt_tokens
from examples.qualification.fake_engine.spans import SpanExporter
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
    receiver: OtlpSpanReceiver,
    *,
    span_encoding: str = "protobuf",
    seed: int | None = None,
) -> FakeEngine:
    return FakeEngine(
        FakeEngineConfig(
            step_seconds=0.001,
            spans_endpoint=f"http://{receiver.listen}/v1/traces",
            span_export_seconds=0.05,
            span_encoding=span_encoding,
            seed=seed,
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


def test_a_span_carries_the_requests_sampling_parameters() -> None:
    # vLLM adds top_p, max_tokens, temperature and n when each is set and not
    # zero (OutputProcessor.do_tracing); unset ones take SamplingParams'
    # defaults of 1.0.
    spans: list[VllmSpanRecord] = []
    with _receiver() as receiver:
        with _engine(receiver) as engine:
            chat(
                engine,
                "x",
                max_tokens=3,
                request_id="stormlog-run-1-set",
                sampling={"top_p": 0.25, "temperature": 0.5},
            )
            chat(
                engine,
                "y",
                max_tokens=2,
                request_id="stormlog-run-1-greedy",
                sampling={"temperature": 0.0},
            )
        assert wait_until(lambda: _received(receiver, spans) >= 2)
    by_request = {span.request_id: span.attributes for span in spans}
    chosen, greedy = (
        by_request["stormlog-run-1-set"],
        by_request["stormlog-run-1-greedy"],
    )
    assert (chosen["gen_ai.request.top_p"], chosen["gen_ai.request.temperature"]) == (
        0.25,
        0.5,
    )
    assert (chosen["gen_ai.request.max_tokens"], chosen["gen_ai.request.n"]) == (3, 1)
    assert greedy["gen_ai.request.top_p"] == 1.0
    assert "gen_ai.request.temperature" not in greedy


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


class _ScriptedCollector(ThreadingHTTPServer):
    """Answers each export with the next scripted status, then 200."""

    daemon_threads = True

    def __init__(self, statuses: list[int]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)
        self.script = list(statuses)
        self.arrivals: list[float] = []


class _ScriptedHandler(BaseHTTPRequestHandler):
    server: _ScriptedCollector

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.arrivals.append(time.monotonic())
        status = self.server.script.pop(0) if self.server.script else 200
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return None


@contextlib.contextmanager
def _exporter(
    statuses: list[int], *, timeout: float = 10.0
) -> Iterator[tuple[SpanExporter, _ScriptedCollector]]:
    collector = _ScriptedCollector(statuses)
    thread = threading.Thread(target=collector.serve_forever, daemon=True)
    thread.start()
    host, port = collector.server_address[:2]
    exporter = SpanExporter(
        f"http://{host!s}:{port}/v1/traces", Controls(), interval=0.02, timeout=timeout
    )
    exporter.start()
    exporter.on_free(
        FakeRequest("r-0a1b2c3d", "r", prompt_tokens("x"), 1, time.time_ns())
    )
    try:
        yield exporter, collector
    finally:
        exporter.close()
        collector.shutdown()
        collector.server_close()


def test_a_failed_export_is_retried_with_the_sdks_backoff() -> None:
    # opentelemetry-exporter-otlp-proto-http 1.44.0 retries 408, 5xx and
    # connection errors, 2**n s apart with 20% jitter (trace_exporter:197-252).
    with _exporter([503, 408]) as (exporter, collector):
        assert wait_until(lambda: len(exporter.statuses) == 3, timeout=10)
        statuses, failed = list(exporter.statuses), exporter.failed
        arrivals = list(collector.arrivals)
    assert statuses == [503, 408, 200]
    assert failed == 0
    first, second = (later - earlier for earlier, later in pairwise(arrivals))
    assert 0.75 <= first <= 1.6
    assert 1.55 <= second <= 2.9


def test_a_429_or_other_client_error_is_final() -> None:
    # The SDK's _is_retryable admits 408 and 5xx only (_common:15-20).
    with _exporter([429]) as (exporter, _collector):
        assert wait_until(lambda: exporter.failed == 1)
        time.sleep(1.5)
        statuses = list(exporter.statuses)
    assert statuses == [429]


def test_retries_stop_at_the_export_timeout() -> None:
    # A retry is skipped when its backoff would pass the export's deadline,
    # OTEL_EXPORTER_OTLP_TRACES_TIMEOUT (10 s by default).
    with _exporter([503] * 10, timeout=1.5) as (exporter, _collector):
        assert wait_until(lambda: exporter.failed == 1, timeout=10)
        statuses = list(exporter.statuses)
    assert statuses == [503, 503]


def _identities(seed: int | None) -> tuple[str, str, str, str]:
    """A request's IDs and its span's, from a fresh engine and receiver."""
    spans: list[VllmSpanRecord] = []
    with _receiver() as receiver:
        with _engine(receiver, seed=seed) as engine:
            chat(engine, "x", max_tokens=1)
            (request,) = engine.engine.finished
        assert wait_until(lambda: _received(receiver, spans) >= 1)
    (span,) = spans
    assert span.trace_id is not None and span.span_id is not None
    return request.external_id, request.internal_id, span.trace_id, span.span_id


def test_a_seed_makes_every_generated_identity_repeat() -> None:
    # Request IDs without an X-Request-Id, vLLM's random suffixes, and span
    # and trace IDs all come from the seed, for the same arrival order.
    assert _identities(7) == _identities(7)
    first, second = _identities(None), _identities(None)
    assert all(a != b for a, b in zip(first, second))
