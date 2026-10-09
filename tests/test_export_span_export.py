"""The span exporter: batching, retries, the breaker, close, and exact accounting."""

import errno
import json
import random
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from stormlog._export import filesink, otlp_http
from stormlog._export.delivery import Breaker, RetryPolicy
from stormlog._export.otlp_encoding import JsonEncoding, ProtobufEncoding
from stormlog._export.otlp_http import Destination, OtlpHttpTransport, Transmission
from stormlog._export.span_export import FileSink, HttpSink, SpanExporter
from stormlog._export.spans import KIND_INTERNAL, Scope, Span
from stormlog._export.watchdog import Watchdog
from stormlog.infer.vllm_spans import read_span_file
from stormlog.infer.vllm_telemetry import SPAN_SOURCE_OTLP_JSON
from tests.fake_otlp_collector import (
    ANSWER,
    RESET,
    SILENT,
    FakeCollector,
    Reply,
    running,
)

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
        watchdog=kw.pop("watchdog", None),
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
        _offer(exporter, 3)
        exporter.start()
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
        _offer(exporter, 2)
        exporter.start()
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


def test_spans_taken_but_not_yet_batched_are_counted_at_the_freeze() -> None:
    # The reviewer's case: one take returns all 10 spans, the worker sticks
    # in the first batch's send, and the close freezes: the 8 the worker
    # still held were counted nowhere.
    with running([Reply(action=SILENT)] * 50) as collector:
        exporter = _exporter(
            collector.url, attempt_seconds=30, max_batch_spans=2, schedule_delay=0.0
        )
        _offer(exporter, 10)
        exporter.start()
        time.sleep(0.3)
        exporter.close(0.2)
    accounting = exporter.accounting()
    assert accounting["offered"] == 10 and _balanced(accounting)


@pytest.mark.parametrize("seed", range(12))
def test_every_offered_span_is_accounted_for_whatever_happens(seed: int) -> None:
    rng = random.Random(seed)
    actions = [ANSWER, ANSWER, SILENT, RESET]
    replies = [
        Reply(status=rng.choice([200, 200, 500, 503]), action=rng.choice(actions))
        for _ in range(60)
    ]
    with running(replies) as collector:
        exporter = _exporter(
            collector.url,
            attempt_seconds=rng.uniform(0.05, 0.5),
            max_batch_spans=rng.randint(1, 6),
            schedule_delay=rng.uniform(0.0, 0.05),
        )
        _offer(exporter, rng.randint(1, 20))
        exporter.start()
        _offer(exporter, rng.randint(0, 20), start=100)
        time.sleep(rng.uniform(0.0, 0.3))
        exporter.close(rng.uniform(0.0, 0.3))
    assert _balanced(exporter.accounting())


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


def _settled_as_aborted(exporter: SpanExporter[int], spans: int) -> bool:
    """The batch whose send the close aborted, settled by the freeze alone.

    The aborted attempt's reset is the sender's own, so the worker leaves
    the batch to the freeze, which counts it unknown as shutdown_in_flight;
    nothing comes back late, and nothing claims an extra copy.
    """
    worker = exporter._worker
    assert worker is not None
    worker.join(5)
    accounting = exporter.accounting()
    return bool(
        accounting["unknown"] == {"shutdown_in_flight": spans}
        and accounting["late_results"] == {}
        and accounting["max_extra_copies"] == 0
    )


def test_close_cuts_a_stuck_send_and_counts_it_unknown() -> None:
    with running([Reply(action=SILENT)]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        # Offered before the worker starts, so all five go in one batch
        # however long this thread is held up between offers.
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: bool(collector.received))
        _offer(exporter, 2, start=5)
        started = time.monotonic()
        exporter.close(0.5)
        elapsed = time.monotonic() - started
        assert elapsed < 1.5
        accounting = exporter.accounting()
        assert _settled_as_aborted(exporter, 5)
        assert accounting["dropped"] == {"shutdown": 2}
        assert accounting["max_extra_copies"] == 0
        assert _balanced(accounting) and accounting["in_flight"] == 0
    assert exporter.summary()["flush_seconds"] < 1.5


def _interrupt_the_wait(
    exporter: SpanExporter[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl+C that lands while close() first waits for the worker."""
    worker = exporter._worker
    assert worker is not None
    real_join = worker.join
    presses = [KeyboardInterrupt()]

    def interrupted(timeout: float | None = None) -> None:
        if presses:
            raise presses.pop()
        real_join(timeout)

    monkeypatch.setattr(worker, "join", interrupted)


def test_an_interrupt_in_the_close_s_wait_still_finishes_the_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The interrupt ends the wait, not the close: the send in flight is
    # aborted, the queue drained, the ledger frozen and the sink closed.
    with running([Reply(action=SILENT)]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: bool(collector.received))
        _offer(exporter, 2, start=5)
        _interrupt_the_wait(exporter, monkeypatch)
        with pytest.raises(KeyboardInterrupt):
            exporter.close(5.0)
        accounting = exporter.accounting()
        assert accounting["frozen"] and _balanced(accounting)
        assert accounting["queued"] == accounting["in_flight"] == 0
        assert _settled_as_aborted(exporter, 5)
        assert accounting["dropped"] == {"shutdown": 2}
        exporter.close(5.0)  # already closed: nothing left to do


def test_a_retry_refused_after_the_abort_claims_no_extra_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gate's probe: the freeze held back until a retry after the abort
    # is inside send(). The transport refuses that retry before any byte
    # leaves, but the freeze read what abort() had returned, "a body had
    # left", and claimed 5 extra copies, where the collector got one.
    with running([Reply(action=SILENT)]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: bool(collector.received))
        transport = exporter.sink.transport  # type: ignore[attr-defined]
        retried = threading.Event()
        real_send = transport.send

        def send(body: bytes, *, spans: int) -> Any:
            if transport._aborted:
                retried.set()
                time.sleep(0.5)  # a worker preempted inside its retry
            return real_send(body, spans=spans)

        monkeypatch.setattr(transport, "send", send)
        real_freeze = exporter.ledger.freeze

        def freeze(**kwargs: Any) -> None:
            retried.wait(1.0)
            real_freeze(**kwargs)

        monkeypatch.setattr(exporter.ledger, "freeze", freeze)
        _interrupt_the_wait(exporter, monkeypatch)
        with pytest.raises(KeyboardInterrupt):
            exporter.close(5.0)
        assert not retried.is_set()  # the aborted attempt is not retried
        assert _settled_as_aborted(exporter, 5) and _balanced(exporter.accounting())
        assert len(collector.received) == 1


@pytest.mark.parametrize("first", ["reset", "503"])
def test_a_retry_not_yet_sending_is_not_taken_for_its_last_attempt(
    monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    # The gate's probe: the first attempt reaches the collector (stored and
    # reset, or a 503), and the retry has been marked as started but has
    # not entered the transport when the close freezes. Its body has not
    # left, but the freeze read the last attempt's "a body had left": 5
    # extra copies after the reset, or the 503's spans counted unknown.
    first_reply = Reply(action=RESET) if first == "reset" else Reply(503, store=False)
    with running([first_reply, Reply(action=SILENT)]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        _offer(exporter, 5)
        transport = exporter.sink.transport  # type: ignore[attr-defined]
        sends = 0
        in_window, frozen = threading.Event(), threading.Event()
        real_send = transport.send

        def send(body: bytes, *, spans: int) -> Any:
            nonlocal sends
            sends += 1
            if sends == 2:  # the retry, held before the transport's lock
                in_window.set()
                frozen.wait(3.0)
            return real_send(body, spans=spans)

        monkeypatch.setattr(transport, "send", send)
        real_freeze = exporter.ledger.freeze

        def freeze(**kwargs: Any) -> None:
            real_freeze(**kwargs)
            frozen.set()

        monkeypatch.setattr(exporter.ledger, "freeze", freeze)
        exporter.start()
        assert in_window.wait(5)
        _interrupt_the_wait(exporter, monkeypatch)
        with pytest.raises(KeyboardInterrupt):
            exporter.close(5.0)
        worker = exporter._worker
        assert worker is not None
        worker.join(5)
        accounting = exporter.accounting()
        assert _balanced(accounting) and len(collector.received) == 1
        if first == "reset":
            assert accounting["unknown"] == {"reset_after_send": 5}
            assert accounting["max_extra_copies"] == 0
        else:
            assert accounting["refused"] == {"throttled": 5}
            assert accounting["unknown"] == {}


def test_a_close_while_the_worker_connects_sends_nothing_and_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gate's probe: the worker is past send()'s abort check, connecting,
    # when the close finishes and stops the watchdog. Arming it then raised:
    # a late ambiguous result, an internal error, and the socket left open.
    with running([Reply()]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        transport = exporter.sink.transport  # type: ignore[attr-defined]
        real_connect = transport._connect
        inside, go = threading.Event(), threading.Event()

        def connect(deadline: float) -> Any:
            inside.set()
            go.wait(3.0)
            return real_connect(deadline)

        closed: list[socket.socket] = []
        real_close = otlp_http._close

        def noting_close(sock: socket.socket) -> None:
            closed.append(sock)
            real_close(sock)

        monkeypatch.setattr(transport, "_connect", connect)
        monkeypatch.setattr(otlp_http, "_close", noting_close)
        _offer(exporter, 5)
        exporter.start()
        assert inside.wait(5)
        _interrupt_the_wait(exporter, monkeypatch)
        with pytest.raises(KeyboardInterrupt):
            exporter.close(5.0)
        go.set()
        worker = exporter._worker
        assert worker is not None
        worker.join(5)
        accounting = exporter.accounting()
        assert len(collector.received) == 0
        assert accounting["dropped"] == {"shutdown": 5} and _balanced(accounting)
        assert accounting["late_results"] == {}
        assert exporter.summary()["internal_errors"] == {}
        assert closed and all(sock.fileno() == -1 for sock in closed)


def test_a_watchdog_thread_that_cannot_start_is_not_taken_for_the_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gate's probe: arming raised because the watchdog's thread could
    # not start, not because the close had stopped it, and the exporter took
    # it for the close: its worker left, and every later span was dropped at
    # shutdown with no error recorded. That attempt is an internal error, its
    # batch unknown, its socket closed, nothing of it left armed, and
    # delivery goes on.
    with running([Reply(), Reply()]) as collector:
        exporter = _exporter(collector.url)
        transport = exporter.sink.transport  # type: ignore[attr-defined]
        watchdog = transport.watchdog
        real_ensure_thread = watchdog._ensure_thread
        failures = [RuntimeError("can't start new thread")]

        def ensure_thread() -> None:
            if failures:
                raise failures.pop()
            real_ensure_thread()

        registered: list[socket.socket] = []
        real_register = transport._register

        def noting_register(sock: socket.socket, deadline: float) -> Any:
            registered.append(sock)
            return real_register(sock, deadline)

        monkeypatch.setattr(watchdog, "_ensure_thread", ensure_thread)
        monkeypatch.setattr(transport, "_register", noting_register)
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: exporter.accounting()["unknown"] == {"send_failed": 5})
        assert watchdog.stats.armed == 0 and not watchdog._sockets
        _offer(exporter, 3, start=5)
        assert _wait(lambda: exporter.accounting()["exported"] == 3)
        worker = exporter._worker
        assert worker is not None and worker.is_alive()
        assert exporter.summary()["internal_errors"] == {"send": 1}
        exporter.close(1.0)
        accounting = exporter.accounting()
        assert accounting["dropped"] == {} and _balanced(accounting)
    assert len(registered) == 2
    assert all(sock.fileno() == -1 for sock in registered)


def test_a_watchdog_stopped_by_another_owner_is_not_taken_for_the_close() -> None:
    # The gate's probe: a watchdog shared with another owner, which has
    # stopped it, though this exporter never closed. Each attempt is an
    # internal error and its batch unknown, as for any other failure to
    # arm; the worker does not take it for its own close and leave.
    watchdog = Watchdog()
    watchdog.stop()
    with running([Reply()]) as collector:
        exporter = _exporter(collector.url, watchdog=watchdog)
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: exporter.accounting()["unknown"] == {"send_failed": 5})
        worker = exporter._worker
        assert worker is not None and worker.is_alive()
        assert exporter.summary()["internal_errors"] == {"send": 1}
        exporter.close(1.0)
        assert _balanced(exporter.accounting())


def test_a_close_cut_short_is_finished_by_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A second Ctrl+C inside the close's own steps: the next close, the
    # run's fallback, finishes them instead of returning at once.
    with running([Reply(action=SILENT)]) as collector:
        exporter = _exporter(collector.url, attempt_seconds=30.0)
        _offer(exporter, 5)
        exporter.start()
        assert _wait(lambda: bool(collector.received))
        _interrupt_the_wait(exporter, monkeypatch)
        real_drain = exporter.queue.drain
        calls = 0

        def drain_interrupted_once() -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return real_drain()

        monkeypatch.setattr(exporter.queue, "drain", drain_interrupted_once)
        with pytest.raises(KeyboardInterrupt):
            exporter.close(5.0)
        assert not exporter.accounting()["frozen"]
        finished = exporter.closed
        assert not finished
        exporter.close(0.5)
        accounting = exporter.accounting()
        assert exporter.closed and accounting["frozen"] and _balanced(accounting)
        assert accounting["in_flight"] == 0


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


def _first_success_ns(exporter: SpanExporter[int]) -> int | None:
    for transition in exporter.summary()["destination"]["transitions"]:
        if transition["event"] == "first_success":
            return int(transition["at_ns"])
    return None


@dataclass(frozen=True)
class _UnluckyRetries(RetryPolicy):
    """The default policy, with every jitter drawn at its ceiling."""

    def delay(self, attempt: int, *, rng: Any = None) -> float:
        return super().delay(attempt, rng=lambda _low, high: high)


@pytest.mark.parametrize("outage", [6.0, 30.0])
def test_a_collector_back_from_an_outage_is_used_within_a_probe_interval(
    outage: float,
) -> None:
    # Contract X1: at --otlp-probe-interval 1, the first success comes within
    # 1 s and an attempt of the collector's return, whether the outage left
    # the breaker closed (6 s) or opened it (30 s). Retries inside a batch
    # once waited out exponential backoff, up to 8 s.
    port = _closed_port()
    transport = OtlpHttpTransport(
        Destination.parse(f"http://127.0.0.1:{port}/v1/traces"),
        media_type="application/x-protobuf",
    )
    exporter: SpanExporter[int] = SpanExporter(
        HttpSink(transport),
        ProtobufEncoding(),
        resource=(("service.name", "t"),),
        scope=Scope("t", "0"),
        to_span=_span,
        retry=_UnluckyRetries(),
        breaker=Breaker(probe_interval=1.0),
    )
    exporter.start()
    stop = threading.Event()

    def feed() -> None:
        index = 0
        while not stop.wait(0.1):
            exporter.offer(index, 64)
            index += 1

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    time.sleep(outage)
    try:
        with running(port=port):
            returned_ns = time.time_ns()
            assert _wait(lambda: _first_success_ns(exporter) is not None, 10)
    finally:
        stop.set()
        feeder.join(5)
        exporter.close(1.0)
    first = _first_success_ns(exporter)
    assert first is not None and (first - returned_ns) / 1e9 <= 1.5
    events = [t["event"] for t in exporter.summary()["destination"]["transitions"]]
    assert ("breaker_open" in events) is (outage > 10)


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
    _offer(exporter, 2)
    exporter.start()
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
    _offer(exporter, 10)
    exporter.start()
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


def test_a_line_left_half_written_is_unknown_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Part of the line reached the file and could not be cut back: a reader
    # may take those spans, or not.
    real_write = filesink._write

    def half_then_full(fd: int, data: Any) -> int:
        real_write(fd, bytes(data[: len(data) // 2]))
        raise OSError(errno.ENOSPC, "No space left on device")

    def no_truncate(_fd: int, _length: int) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(filesink, "_write", half_then_full)
    monkeypatch.setattr(filesink, "_truncate", no_truncate)
    exporter = _file_exporter(tmp_path / "spans.jsonl")
    _offer(exporter, 4)
    exporter.start()
    exporter.close(2.0)
    accounting = exporter.accounting()
    assert accounting["unknown"] == {"file_partial": 4} and _balanced(accounting)


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

    # A Bearer token has a digit, =, + or / since the scrub fixes.
    status = b"\x08\x03\x12\x1bbad: Bearer sk-abcdefgh1234"
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
