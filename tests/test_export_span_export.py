"""The span exporter: batching, retries, the breaker, close, and exact accounting."""

import errno
import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from stormlog._export import filesink
from stormlog._export.delivery import Breaker, RetryPolicy
from stormlog._export.otlp_encoding import JsonEncoding, ProtobufEncoding
from stormlog._export.otlp_http import Destination, OtlpHttpTransport, Transmission
from stormlog._export.span_export import FileSink, HttpSink, SpanExporter
from stormlog._export.spans import KIND_INTERNAL, Scope, Span
from stormlog.infer.vllm_spans import read_span_file
from stormlog.infer.vllm_telemetry import SPAN_SOURCE_OTLP_JSON
from tests.fake_otlp_collector import RESET, SILENT, FakeCollector, Reply, running

pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")

FAST = RetryPolicy(initial_seconds=0.01, max_seconds=0.05)


def _span(index: int) -> Span:
    return Span(
        name="stormlog.test",
        trace_id=f"{index + 1:032x}",
        span_id=f"{index + 1:016x}",
        kind=KIND_INTERNAL,
        start_ns=index,
        end_ns=index + 1,
        attributes=(("i", index),),
    )


def _exporter(url: str, **kw: Any) -> SpanExporter[int]:
    attempt_seconds = kw.pop("attempt_seconds", 2.0)
    transport = OtlpHttpTransport(
        Destination.parse(url),
        media_type="application/x-protobuf",
        attempt_seconds=attempt_seconds,
    )
    kw.setdefault("retry", FAST)
    kw.setdefault("schedule_delay", 0.05)
    return SpanExporter(
        HttpSink(transport),
        ProtobufEncoding(),
        resource=(("service.name", "t"),),
        scope=Scope("t", "0"),
        to_span=_span,
        **kw,
    )


def _offer(exporter: SpanExporter[int], count: int, start: int = 0) -> None:
    for index in range(start, start + count):
        exporter.offer(index, 64)


def _balanced(accounting: dict[str, Any]) -> bool:
    return bool(
        accounting["offered"]
        == accounting["exported"]
        + accounting["rejected"]
        + sum(accounting["refused"].values())
        + sum(accounting["dropped"].values())
        + sum(accounting["unknown"].values())
        + accounting["queued"]
        + accounting["in_flight"]
    )


def _wait(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return bool(predicate())


def test_spans_are_batched_delivered_and_counted() -> None:
    with running() as collector:
        exporter = _exporter(collector.url)
        exporter.start()
        _offer(exporter, 20)
        assert _wait(lambda: len(collector.stored_spans()) == 20)
        assert _balanced(exporter.accounting())
        exporter.close(2.0)
    accounting = exporter.accounting()
    assert accounting["exported"] == 20 and accounting["frozen"]
    assert accounting["queued"] == accounting["in_flight"] == 0
    assert len(collector.unique_stored()) == 20
    assert exporter.summary()["transmissions"] == {"confirmed": len(collector.received)}


def test_a_batch_closes_at_its_span_limit() -> None:
    with running() as collector:
        exporter = _exporter(collector.url, schedule_delay=30.0, max_batch_spans=8)
        _offer(exporter, 20)
        exporter.start()
        # Two full batches go at once; the last four wait for the delay or close.
        assert _wait(lambda: len(collector.received) == 2)
        assert [len(r.spans) for r in collector.received] == [8, 8]
        exporter.close(2.0)
    assert [len(r.spans) for r in collector.received] == [8, 8, 4]
    assert exporter.accounting()["exported"] == 20


def test_retryable_answers_are_retried_and_counted() -> None:
    replies = [Reply(503), Reply(502), Reply()]
    with running(replies) as collector:
        exporter = _exporter(collector.url)
        exporter.start()
        _offer(exporter, 3)
        assert _wait(lambda: len(collector.received) == 3)
        exporter.close(2.0)
    summary = exporter.summary()
    assert summary["retries"] == 2
    assert summary["transmissions"] == {"refused": 1, "ambiguous": 1, "confirmed": 1}
    accounting = summary["spans"]
    # The 502 may have stored them: confirmed after an ambiguous transmission.
    assert accounting["exported"] == 3 and accounting["max_extra_copies"] == 3
    assert summary["first_error"] == {
        "kind": "refused",
        "category": "throttled",
        "status": 503,
    }


def test_a_retry_after_beyond_the_budget_ends_the_batch() -> None:
    replies = [Reply(429, headers=(("Retry-After", "3600"),))]
    with running(replies) as collector:
        exporter = _exporter(collector.url)
        exporter.start()
        _offer(exporter, 2)
        assert _wait(lambda: exporter.accounting()["refused"] == {"throttled": 2})
        exporter.close(1.0)
    assert len(collector.received) == 1


@pytest.mark.parametrize(
    ("replies", "expected", "copies"),
    [
        # Stored, reset, then a 400 that stored nothing.
        (
            [Reply(action=RESET), Reply(400, store=False)],
            {"unknown": {"refused_after_ambiguous": 4}},
            0,
        ),
        # Four stores whose answers were lost, then a 200.
        ([Reply(action=RESET)] * 4 + [Reply()], {"exported": 4}, 16),
        # An unreadable 200 after storing.
        (
            [Reply(body=b"\xff\xfe")],
            {"unknown": {"nonconformant_response": 4}},
            0,
        ),
    ],
)
def test_lost_answers_settle_within_the_collectors_bounds(
    replies: list[Reply], expected: dict[str, Any], copies: int
) -> None:
    with running(replies) as collector:
        exporter = _exporter(collector.url, schedule_delay=0.2)
        _offer(exporter, 4)
        exporter.start()
        assert _wait(lambda: exporter.accounting()["in_flight"] == 0, 10)
        exporter.close(2.0)
    accounting = exporter.accounting()
    for key, value in expected.items():
        assert accounting[key] == value
    assert accounting["max_extra_copies"] == copies
    unique = len(collector.unique_stored())
    raw = len(collector.stored_spans())
    unknown = sum(accounting["unknown"].values())
    assert accounting["exported"] <= unique <= accounting["exported"] + unknown
    assert raw - unique <= accounting["max_extra_copies"]


def test_the_breaker_opens_probes_and_closes() -> None:
    def script(index: int) -> Reply:
        return Reply(400, store=False) if index < 3 else Reply()

    with running(script) as collector:
        exporter = _exporter(
            collector.url,
            breaker=Breaker(probe_interval=0.3),
            max_batch_spans=1,
            schedule_delay=0.0,
        )
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: exporter.accounting()["exported"] == 2, 10)
        exporter.close(1.0)
    assert len(collector.received) == 5
    # Once open, the next attempt waits one probe interval.
    gap = collector.received[3].at - collector.received[2].at
    assert gap >= 0.25
    destination = exporter.summary()["destination"]
    assert destination["up"] is True
    events = [t["event"] for t in destination["transitions"]]
    assert events == [
        "first_failure",
        "breaker_open",
        "first_success",
        "breaker_closed",
    ]
    assert exporter.accounting()["refused"] == {"http_4xx": 3}


def test_close_cuts_a_stuck_send_and_counts_it_unknown() -> None:
    with running([Reply(action=SILENT)]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        exporter.start()
        _offer(exporter, 5)
        assert _wait(lambda: bool(collector.received))
        _offer(exporter, 2, start=5)
        started = time.monotonic()
        exporter.close(0.5)
        elapsed = time.monotonic() - started
        assert elapsed < 1.5
        accounting = exporter.accounting()
        assert accounting["unknown"] == {"shutdown_in_flight": 5}
        assert accounting["dropped"] == {"shutdown": 2}
        assert accounting["max_extra_copies"] == 0
        assert _balanced(accounting) and accounting["in_flight"] == 0
        # The aborted attempt's answer comes back after the freeze: late.
        assert _wait(lambda: exporter.accounting()["late_results"] == {"ambiguous": 1})
    assert exporter.summary()["flush_seconds"] < 1.5


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_down_collector_drops_within_the_close_deadline() -> None:
    exporter = _exporter(f"http://127.0.0.1:{_closed_port()}")
    exporter.start()
    _offer(exporter, 6)
    time.sleep(0.2)
    started = time.monotonic()
    exporter.close(0.5)
    assert time.monotonic() - started < 1.5
    accounting = exporter.accounting()
    assert accounting["exported"] == 0 and _balanced(accounting)
    assert set(accounting["dropped"]) <= {"connect_refused", "shutdown"}
    assert sum(accounting["dropped"].values()) == 6


def test_a_full_queue_drops_and_counts() -> None:
    exporter = _exporter("http://127.0.0.1:9", queue_items=2)
    _offer(exporter, 5)
    accounting = exporter.accounting()
    assert accounting["dropped"] == {"queue_full": 3} and accounting["queued"] == 2
    assert _balanced(accounting)
    exporter.close(0.1)
    assert exporter.accounting()["dropped"] == {"queue_full": 3, "shutdown": 2}
    # Offers after close are refused and counted.
    assert not exporter.offer(9, 64)
    assert exporter.accounting()["dropped"]["closed"] == 1


def test_a_span_that_cannot_be_built_is_dropped_and_counted() -> None:
    def to_span(index: int) -> Span:
        if index == 1:
            raise ValueError("bad record")
        return _span(index)

    with running() as collector:
        exporter = _exporter(collector.url)
        exporter.to_span = to_span
        exporter.start()
        _offer(exporter, 3)
        assert _wait(lambda: len(collector.stored_spans()) == 2)
        exporter.close(1.0)
    accounting = exporter.accounting()
    assert accounting["dropped"] == {"encode_error": 1} and accounting["exported"] == 2
    assert exporter.summary()["internal_errors"] == {"encode": 1}


def test_a_sink_that_raises_leaves_its_batch_unknown() -> None:
    class Broken(HttpSink):
        def send(self, body: bytes, *, spans: int) -> Transmission:
            raise RuntimeError("bug")

    transport = OtlpHttpTransport(
        Destination.parse("http://127.0.0.1:9"), media_type="application/x-protobuf"
    )
    exporter: SpanExporter[int] = SpanExporter(
        Broken(transport),
        ProtobufEncoding(),
        resource=(),
        scope=Scope("t", "0"),
        to_span=_span,
        schedule_delay=0.0,
    )
    exporter.start()
    _offer(exporter, 2)
    assert _wait(lambda: exporter.accounting()["unknown"] == {"send_failed": 2})
    exporter.close(0.5)
    assert exporter.summary()["internal_errors"] == {"send": 1}


def _file_exporter(path: Path) -> SpanExporter[int]:
    return SpanExporter(
        FileSink(path),
        JsonEncoding(),
        resource=(("service.name", "t"),),
        scope=Scope("t", "0"),
        to_span=_span,
        schedule_delay=0.05,
        max_batch_spans=4,
    )


def test_the_file_sink_writes_otlp_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    exporter = _file_exporter(path)
    exporter.start()
    _offer(exporter, 10)
    exporter.close(2.0)
    assert exporter.accounting()["exported"] == 10
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(line)["resourceSpans"] for line in lines)
    source, spans = read_span_file(path)
    assert source == SPAN_SOURCE_OTLP_JSON and len(spans) == 10


def test_a_failed_file_write_drops_its_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_space(_fd: int, _data: Any) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(filesink, "_write", no_space)
    path = tmp_path / "spans.jsonl"
    exporter = _file_exporter(path)
    exporter.start()
    _offer(exporter, 4)
    exporter.close(2.0)
    accounting = exporter.accounting()
    assert accounting["dropped"] == {"file_error": 4} and _balanced(accounting)
    assert path.read_text() == ""


def test_offers_never_wait_for_the_worker() -> None:
    collector = FakeCollector()
    exporter = _exporter("http://127.0.0.1:9")
    started = time.perf_counter()
    for index in range(2048):
        exporter.offer(index, 64)
    assert time.perf_counter() - started < 1.0
    assert exporter.accounting()["queued"] == 2048
    assert not collector.received


def test_collector_text_is_kept_only_through_the_consent_scrubber() -> None:
    from stormlog.scrub import scrub_text

    status = b"\x08\x03\x12\x1bbad: Bearer sk-abcdefghijkl"
    warning = Reply(
        body=b'{"partialSuccess":{"errorMessage":"slow down"}}',
        content_type="application/json",
    )
    for keep, expected in ((None, None), (scrub_text, "bad: Bearer <redacted>")):
        with running([Reply(400, status, store=False), warning]) as collector:
            exporter = _exporter(collector.url, keep_message=keep, max_batch_spans=1)
            exporter.start()
            _offer(exporter, 2)
            assert _wait(lambda: len(collector.received) == 2)
            exporter.close(1.0)
        summary = exporter.summary()
        assert summary["collector_message"] == expected
        assert summary["warnings"] == 1


def test_a_write_stuck_past_the_stall_time_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from stormlog._export import span_export

    release = threading.Event()
    real = filesink._write

    def stuck(fd: int, data: Any) -> int:
        release.wait(10)
        return real(fd, data)

    monkeypatch.setattr(filesink, "_write", stuck)
    monkeypatch.setattr(span_export, "STALL_SECONDS", 0.1)
    exporter = _file_exporter(tmp_path / "spans.jsonl")
    exporter.start()
    _offer(exporter, 1)
    assert _wait(lambda: exporter.summary()["stalled"] == {"write": True})
    release.set()
    exporter.close(2.0)
    assert exporter.summary()["stalled"] == {"write": False}
    assert exporter.accounting()["exported"] == 1
    assert exporter.summary()["worker_cpu_seconds"] is not None
