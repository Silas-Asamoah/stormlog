"""Batch spans and deliver them from one worker thread, settling every one.

The producer offers items to a bounded queue and moves on. The worker turns
each into a capped span, encodes it, and closes a batch at 512 spans, 4 MiB
or one second after its first span. It then sends the batch to its sink,
an OTLP/HTTP endpoint or an OTLP JSON lines file, retrying what may be
retried, and settles it in the ledger.

``close`` stops admission and lets the worker finish within the deadline,
cutting any wait that would outlast it. At the deadline the sink is
aborted and the ledger frozen: what is still queued or unbatched is
dropped at shutdown, and a batch mid-transmission is unknown.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from .delivery import (
    ENCODE_ERROR,
    BatchHistory,
    Breaker,
    DeliveryLedger,
    RetryPolicy,
)
from .filesink import (
    FILE_DISABLED,
    FILE_ERROR,
    FILE_FULL,
    FILE_PARTIAL,
    WRITTEN,
    LineFileSink,
)
from .otlp_encoding import ExportResult, SpanEncoding
from .otlp_http import (
    AMBIGUOUS,
    CONFIRMED,
    NOT_SENT,
    SEND_FAILED,
    OtlpHttpTransport,
    Transmission,
)
from .queue import BoundedQueue
from .spans import Attributes, Scope, Span, SpanLimits, capped

T = TypeVar("T")

SPAN_QUEUE_ITEMS = 2048
SPAN_QUEUE_BYTES = 8 * 1024 * 1024
MAX_BATCH_SPANS = 512
MAX_BATCH_BYTES = 4 * 1024 * 1024
SCHEDULE_DELAY_SECONDS = 1.0
MAX_FILE_BYTES = 256 * 1024 * 1024


# A write or resolution running this long is reported as stalled.
STALL_SECONDS = 5.0


class SpanSink(Protocol):
    """Where batches go. ``abort`` is final and says whether a body had left;
    ``sending`` says so for the attempt in progress at the moment of asking."""

    kind: str

    def start(self) -> None: ...

    def send(self, body: bytes, *, spans: int) -> Transmission: ...

    def abort(self) -> bool: ...

    def sending(self) -> bool: ...

    def close(self) -> None: ...

    def stalled(self) -> dict[str, bool]:
        """Each uncancellable stage, and whether it is stuck now."""
        ...


class HttpSink:
    kind = "endpoint"

    def __init__(self, transport: OtlpHttpTransport) -> None:
        self.transport = transport

    def start(self) -> None:
        self.transport.start()

    def send(self, body: bytes, *, spans: int) -> Transmission:
        return self.transport.send(body, spans=spans)

    def abort(self) -> bool:
        return self.transport.abort()

    def sending(self) -> bool:
        return self.transport.body_started

    def close(self) -> None:
        self.transport.watchdog.stop()

    def stalled(self) -> dict[str, bool]:
        return {"resolve": self.transport.resolver.stalled}


class FileSink:
    """OTLP JSON, one export request per line; a line is written whole or not at all."""

    kind = "file"
    _OUTCOMES = {
        FILE_FULL: Transmission(NOT_SENT, FILE_FULL),
        FILE_ERROR: Transmission(NOT_SENT, FILE_ERROR),
        FILE_DISABLED: Transmission(NOT_SENT, FILE_DISABLED),
        FILE_PARTIAL: Transmission(AMBIGUOUS, FILE_PARTIAL),
    }

    def __init__(
        self, path: Path, *, max_bytes: int = MAX_FILE_BYTES, fsync: bool = False
    ) -> None:
        self.lines = LineFileSink(path, max_bytes=max_bytes, fsync=fsync)
        self._lock = threading.Lock()
        self._aborted = False
        self._writing = False
        self._write_started = 0.0

    def start(self) -> None:
        self.lines.open()

    def send(self, body: bytes, *, spans: int) -> Transmission:
        with self._lock:
            if self._aborted:
                return Transmission(NOT_SENT, SEND_FAILED)
            self._writing = True
            self._write_started = time.monotonic()
        outcome = self.lines.write_line(body)
        with self._lock:
            self._writing = False
        if outcome == WRITTEN:
            return Transmission(CONFIRMED, result=ExportResult(), sent_bytes=len(body))
        return self._OUTCOMES[outcome]

    def abort(self) -> bool:
        with self._lock:
            self._aborted = True
            return self._writing

    def sending(self) -> bool:
        with self._lock:
            return self._writing

    def close(self) -> None:
        self.lines.close()

    def stalled(self) -> dict[str, bool]:
        with self._lock:
            stuck = self._writing and (
                time.monotonic() - self._write_started > STALL_SECONDS
            )
        return {"write": stuck}


@dataclass
class _Stats:
    transmissions: Counter[str] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)
    retries: int = 0
    sent_bytes: int = 0
    encoded_bytes: int = 0
    batches: int = 0
    # Confirmations that rejected nothing but said something.
    warnings: int = 0
    first_error: dict[str, Any] | None = None
    # The collector's own text, kept only by consent, scrubbed.
    collector_message: str | None = None
    errors: Counter[str] = field(default_factory=Counter)
    flush_seconds: float | None = None
    # CPU seconds the worker thread used.
    worker_cpu_seconds: float | None = None


@dataclass
class _Batch:
    units: list[Any] = field(default_factory=list)
    size: int = 0
    opened_at: float = 0.0

    def add(self, unit: Any, size: int) -> None:
        if not self.units:
            self.opened_at = time.monotonic()
        self.units.append(unit)
        self.size += size

    def clear(self) -> None:
        self.units = []
        self.size = 0


class SpanExporter(Generic[T]):
    """Items in, through ``to_span``; batches out to one sink."""

    def __init__(
        self,
        sink: SpanSink,
        encoding: SpanEncoding,
        *,
        resource: Attributes,
        scope: Scope,
        to_span: Callable[[T], Span],
        limits: SpanLimits = SpanLimits(),
        retry: RetryPolicy = RetryPolicy(),
        breaker: Breaker | None = None,
        keep_message: Callable[[str], str] | None = None,
        schedule_delay: float = SCHEDULE_DELAY_SECONDS,
        max_batch_spans: int = MAX_BATCH_SPANS,
        max_batch_bytes: int = MAX_BATCH_BYTES,
        queue_items: int = SPAN_QUEUE_ITEMS,
        queue_bytes: int = SPAN_QUEUE_BYTES,
    ) -> None:
        self.sink = sink
        self.encoding = encoding
        self.resource = resource
        self.scope = scope
        self.to_span = to_span
        self.limits = limits
        self.retry = retry
        self.breaker = breaker or Breaker()
        # With consent to keep a collector's text, how to scrub it first.
        self.keep_message = keep_message
        self.schedule_delay = schedule_delay
        self.max_batch_spans = max_batch_spans
        self.max_batch_bytes = max_batch_bytes
        self.queue: BoundedQueue[T] = BoundedQueue(
            max_items=queue_items, max_bytes=queue_bytes
        )
        self.ledger = DeliveryLedger()
        self.stats = _Stats()
        self._closing = threading.Event()
        # Every step of the close done: sink aborted, queue drained, ledger
        # frozen, sink closed. Set apart from _closing, which stops admission,
        # so a close an interrupt cut short can be finished by the next.
        self._closed = False
        self._until = float("inf")
        self._last_attempt_at: float | None = None
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._finishing = threading.Lock()

    def start(self) -> str | None:
        """Open the sink and start the worker; the sink's error, if it failed.

        The worker starts either way, so what is offered is still settled:
        a file that could not be opened drops every batch as disabled.
        """
        error = None
        try:
            self.sink.start()
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._worker = threading.Thread(
            target=self._run, name="stormlog-export-spans", daemon=True
        )
        self._worker.start()
        return error

    def offer(self, item: T, size: int) -> bool:
        """Queue ``item``; never blocks. False when full or closed (counted)."""
        return self.queue.offer(item, size)

    # ------------------------------------------------------------- the worker
    def _run(self) -> None:
        try:
            self._loop()
        except Exception:
            # The freeze settles whatever the worker left unfinished.
            self._error("worker")
        finally:
            self.stats.worker_cpu_seconds = round(time.thread_time(), 3)

    def _loop(self) -> None:
        batch = _Batch()
        while True:
            # Taken into the ledger as they leave the queue, so a freeze
            # counts every one, whether still queued or in the worker's hands.
            items = self.queue.take(
                timeout=self._take_timeout(batch), claim=self.ledger.take
            )
            for item in items:
                if not self._add(batch, item):
                    return
            if batch.units and self._due(batch) and not self._flush(batch):
                return
            if not items and not batch.units and self.queue.stats().closed:
                return

    def _take_timeout(self, batch: _Batch) -> float:
        if self._closing.is_set():
            return 0.0
        if not batch.units:
            return 0.25
        return max(0.0, batch.opened_at + self.schedule_delay - time.monotonic())

    def _add(self, batch: _Batch, item: T) -> bool:
        unit = self._unit(item)
        if unit is None:
            return not self.ledger.frozen
        encoded, size = unit
        full = len(batch.units) >= self.max_batch_spans
        if batch.units and (full or batch.size + size > self.max_batch_bytes):
            if not self._flush(batch):
                return False
        batch.add(encoded, size)
        return True

    def _flush(self, batch: _Batch) -> bool:
        delivered = self._deliver(batch.units)
        batch.clear()
        return delivered

    def _due(self, batch: _Batch) -> bool:
        if len(batch.units) >= self.max_batch_spans:
            return True
        if self._closing.is_set():
            return self.queue.stats().depth == 0
        return time.monotonic() >= batch.opened_at + self.schedule_delay

    def _unit(self, item: T) -> tuple[Any, int] | None:
        if self.ledger.frozen:
            return None
        try:
            unit, unit_size = self.encoding.unit(
                capped(self.to_span(item), self.limits)
            )
        except Exception:
            self._error("encode")
            self.ledger.drop_pending(ENCODE_ERROR, 1)
            return None
        if unit_size > self.max_batch_bytes:
            self.ledger.drop_pending(ENCODE_ERROR, 1)
            return None
        return unit, unit_size

    def _deliver(self, units: list[Any]) -> bool:
        """Send one batch until it settles; False once the ledger is frozen."""
        history = BatchHistory(len(units))
        if not self.ledger.begin(history):
            return False
        try:
            body = self.encoding.request(self.resource, self.scope, units)
        except Exception:
            self._error("encode")
            failed = Transmission(NOT_SENT, ENCODE_ERROR)
            return self.ledger.record(history, failed) and self._finish(history)
        self.stats.encoded_bytes += len(body)
        self.stats.batches += 1
        return self._transmit(history, body) and self._finish(history)

    def _transmit(self, history: BatchHistory, body: bytes) -> bool:
        """Attempt until a transmission is final or no retry fits; False once frozen."""
        while True:
            if not self._pause(self._probe_wait()) or not self.ledger.attempting():
                return False
            transmission = self._send(body, history.spans)
            self._last_attempt_at = time.monotonic()
            if not self.ledger.record(history, transmission):
                return False
            self._note(transmission, retry=history.attempts > 1)
            if transmission.kind == CONFIRMED or not transmission.retryable:
                return True
            delay = self._retry_delay(history, transmission)
            if delay is None:
                return True
            if not self._pause(delay):
                return False

    def _send(self, body: bytes, spans: int) -> Transmission:
        try:
            return self.sink.send(body, spans=spans)
        except Exception:
            # Whether the body left is unknown, so the batch may be stored.
            self._error("send")
            return Transmission(AMBIGUOUS, SEND_FAILED)

    def _finish(self, history: BatchHistory) -> bool:
        if self.ledger.finish(history) is None:
            return False
        self.breaker.settled(history)
        return True

    def _retry_delay(
        self, history: BatchHistory, transmission: Transmission
    ) -> float | None:
        """The wait before the next attempt; None when the budget cannot cover it.

        The backoff never exceeds the probe interval, so a destination that
        comes back is used again within one interval, whether the outage
        opened the breaker or not. A server's Retry-After is kept as given.
        """
        delay = (
            transmission.retry_after
            if transmission.retry_after is not None
            else min(self.retry.delay(history.attempts), self.breaker.probe_interval)
        )
        if delay > self.retry.remaining(history, time.monotonic()):
            return None
        return delay

    def _probe_wait(self) -> float:
        # While the destination is down, attempts are one probe interval apart.
        if self.breaker.up or self._last_attempt_at is None:
            return 0.0
        next_probe = self._last_attempt_at + self.breaker.probe_interval
        return max(0.0, next_probe - time.monotonic())

    def _pause(self, seconds: float) -> bool:
        """Wait ``seconds``; False when that would outlast the close deadline."""
        if seconds <= 0:
            return not (self._closing.is_set() and time.monotonic() > self._until)
        end = time.monotonic() + seconds
        if not self._closing.is_set():
            self._closing.wait(seconds)
        if self._closing.is_set():
            if end > self._until:
                return False
            time.sleep(max(0.0, end - time.monotonic()))
        return True

    def _note(self, transmission: Transmission, *, retry: bool) -> None:
        self.breaker.attempted(transmission)
        stats = self.stats
        stats.transmissions[transmission.kind] += 1
        if transmission.category is not None:
            stats.categories[transmission.category] += 1
            if stats.first_error is None:
                stats.first_error = {
                    "kind": transmission.kind,
                    "category": transmission.category,
                    "status": transmission.status,
                }
        stats.retries += retry
        stats.sent_bytes += transmission.sent_bytes
        result = transmission.result
        if result is not None and result.rejected == 0 and result.message:
            stats.warnings += 1
        if transmission.message and self.keep_message and not stats.collector_message:
            stats.collector_message = self.keep_message(transmission.message)

    def _error(self, entry: str) -> None:
        self.stats.errors[entry] += 1

    # ------------------------------------------------------------- the end
    @property
    def closed(self) -> bool:
        """Whether every step of the close is done."""
        return self._closed

    def close(self, deadline: float) -> None:
        """Stop admission, deliver within ``deadline`` seconds, then freeze.

        An interrupt, such as a Ctrl+C, in the wait ends only the wait: the
        sink is aborted, the queue drained, the ledger frozen and the sink
        closed before it propagates. A close cut short inside those steps is
        finished by the next call, which does not wait again.
        """
        started = time.monotonic()
        with self._lock:
            if self._closed:
                return
            if self._closing.is_set():
                self._until = min(self._until, started)
            else:
                self._until = started + max(0.0, deadline)
                self._closing.set()
        self.queue.close()
        worker = self._worker
        try:
            if worker is not None:
                worker.join(max(0.0, self._until - time.monotonic()))
        finally:
            self._finish_close(worker, started)

    def _finish_close(self, worker: threading.Thread | None, started: float) -> None:
        with self._finishing:
            if self._closed:
                return
            self.ledger.close_attempts()
            if worker is not None and worker.is_alive():
                self.sink.abort()
            drained = len(self.queue.drain())
            # Read at the freeze, under the ledger's lock: whether the
            # attempt in progress then had begun to send, not whether one
            # had when the sink was aborted.
            self.ledger.freeze(drained=drained, sending=self.sink.sending)
            try:
                self.sink.close()
            except Exception:
                self._error("close")
            self.stats.flush_seconds = round(time.monotonic() - started, 3)
            self._closed = True

    # ------------------------------------------------------------- reading
    def accounting(self) -> dict[str, Any]:
        """Every offered span's disposition; exact at every instant."""
        queue = self.queue.stats()
        ledger: dict[str, Any] = self.ledger.snapshot()
        dropped = Counter(ledger["dropped"])
        dropped.update(
            {"queue_full": queue.dropped_full, "closed": queue.dropped_closed}
        )
        return {
            "offered": queue.offered,
            "exported": ledger["exported"],
            "rejected": ledger["rejected"],
            "refused": ledger["refused"],
            "dropped": {k: v for k, v in dropped.items() if v},
            "unknown": ledger["unknown"],
            "queued": queue.depth,
            "in_flight": ledger["in_flight"],
            "max_extra_copies": ledger["max_extra_copies"],
            "late_results": ledger["late_results"],
            "frozen": ledger["frozen"],
            "loss_accounting": "exact_local",
        }

    def summary(self) -> dict[str, Any]:
        queue = self.queue.stats()
        stats = self.stats
        return {
            "spans": self.accounting(),
            "queue": {
                "bytes": queue.depth_bytes,
                "high_water": queue.high_water,
                "high_water_bytes": queue.high_water_bytes,
                "capacity_spans": queue.max_items,
                "capacity_bytes": queue.max_bytes,
            },
            "transmissions": dict(stats.transmissions),
            "categories": dict(stats.categories),
            "retries": stats.retries,
            "batches": stats.batches,
            "encoded_bytes": stats.encoded_bytes,
            "sent_bytes": stats.sent_bytes,
            "first_error": stats.first_error,
            "warnings": stats.warnings,
            "collector_message": stats.collector_message,
            "destination": self.breaker.snapshot(),
            "stalled": self._stalled(),
            "flush_seconds": stats.flush_seconds,
            "worker_cpu_seconds": stats.worker_cpu_seconds,
            "internal_errors": dict(stats.errors),
        }

    def _stalled(self) -> dict[str, bool]:
        try:
            return self.sink.stalled()
        except Exception:
            return {}
