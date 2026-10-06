"""I/O off the watcher's loop: the serial worker and the ledger (N4)."""

from __future__ import annotations

import errno
import functools
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from stormlog import telemetry_sink
from stormlog.infer.watch.io import SerialWorker
from stormlog.infer.watch.ledger import Ledger
from stormlog.infer.watch.records import WATCH_HEALTH, envelope
from stormlog.infer.watch.stats import WatchStats, counter_value
from tests.watch_test_helpers import read_ledger


def _record(index: int) -> dict[str, Any]:
    record = envelope(WATCH_HEALTH, session_id="s", run_id="r", timestamp_ns=index)
    record["index"] = index
    return record


def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


# ------------------------------------------------------------- SerialWorker


def test_worker_runs_operations_in_order_on_one_thread() -> None:
    worker = SerialWorker("test-order")
    seen: list[tuple[int, str]] = []

    def record(index: int) -> None:
        seen.append((index, threading.current_thread().name))

    for index in range(20):
        assert worker.submit(functools.partial(record, index))
    assert worker.close(5.0)
    assert [i for i, _ in seen] == list(range(20))
    assert {name for _, name in seen} == {"test-order"}
    assert worker.stats().completed == 20


def test_worker_counts_failures_and_keeps_going() -> None:
    worker = SerialWorker("test-fail")
    done: list[int] = []

    def fail() -> None:
        raise OSError(errno.EIO, "disk")

    worker.submit(fail)
    worker.submit(lambda: done.append(1))
    assert worker.close(5.0)
    stats = worker.stats()
    assert (stats.failed, stats.completed, done) == (1, 1, [1])
    assert stats.last_error is not None and "OSError" in stats.last_error


def test_worker_rejects_past_its_count_and_byte_bounds() -> None:
    gate = threading.Event()
    worker = SerialWorker("test-bounds", max_queued=2, max_queued_bytes=100)
    worker.submit(gate.wait)
    _wait_for(lambda: worker.stats().queued == 0)  # the gate is running
    assert worker.submit(lambda: None, nbytes=60)
    assert not worker.submit(lambda: None, nbytes=60)  # 120 bytes > 100
    assert worker.submit(lambda: None, nbytes=10)
    assert not worker.submit(lambda: None)  # 2 queued already
    assert worker.stats().rejected == 2
    gate.set()
    assert worker.close(5.0)


def test_a_stalled_operation_rejects_work_and_never_adds_threads() -> None:
    gate = threading.Event()
    worker = SerialWorker("test-stall", stall_seconds=0.05)
    worker.submit(gate.wait)
    time.sleep(0.1)
    threads = threading.active_count()
    started = time.perf_counter()
    taken = [worker.submit(lambda: None) for _ in range(1000)]
    elapsed = time.perf_counter() - started
    assert not any(taken)
    assert worker.stats().stalled
    assert worker.stats().rejected == 1000
    assert threading.active_count() == threads
    assert elapsed < 0.5
    gate.set()
    assert worker.close(5.0)
    assert not worker.stats().stalled


def test_close_runs_the_final_operation_even_with_a_full_queue() -> None:
    gate = threading.Event()
    worker = SerialWorker("test-final", max_queued=1)
    worker.submit(gate.wait)
    _wait_for(lambda: worker.stats().queued == 0)
    assert worker.submit(lambda: None)
    order: list[str] = []
    closer = threading.Thread(
        target=lambda: worker.close(5.0, final=lambda: order.append("final"))
    )
    closer.start()
    gate.set()
    closer.join(5.0)
    assert order == ["final"]
    assert not worker.submit(lambda: None)  # closed


def test_close_times_out_on_a_blocked_operation() -> None:
    gate = threading.Event()
    worker = SerialWorker("test-timeout")
    worker.submit(gate.wait)
    assert not worker.close(0.05)
    gate.set()


@pytest.mark.parametrize(
    "bounds",
    [{"max_queued": 0}, {"max_queued_bytes": 0}, {"stall_seconds": 0}],
)
def test_worker_bounds_must_be_positive(bounds: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SerialWorker("test-bad", **bounds)


# ------------------------------------------------------------------- Ledger


class _Observer:
    def __init__(self, *, fail: bool = False) -> None:
        self.records: list[Mapping[str, Any]] = []
        self.fail = fail

    def observe(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)
        if self.fail:
            raise RuntimeError("exporter down")


def test_ledger_appends_records_and_taps_them(tmp_path: Path) -> None:
    observer = _Observer()
    ledger = Ledger(tmp_path, observer=observer)
    for index in range(5):
        assert ledger.write(_record(index))
    assert ledger.close(5.0)
    assert [r["index"] for r in read_ledger(tmp_path)] == list(range(5))
    assert [r["index"] for r in observer.records] == list(range(5))
    assert ledger.stats()["ledger_dropped"] == 0


def test_ledger_refuses_an_invalid_record(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.write({"event_type": WATCH_HEALTH})
    assert ledger.close(5.0)


def test_ledger_writes_a_snapshot_of_the_record(tmp_path: Path) -> None:
    gate = threading.Event()
    ledger = Ledger(tmp_path)
    ledger._worker.submit(gate.wait)  # hold the worker so the write queues
    record = _record(1)
    record["open_incidents"] = ["a"]
    ledger.write(record)
    record["open_incidents"].append("b")  # the caller reuses its objects
    gate.set()
    assert ledger.close(5.0)
    assert read_ledger(tmp_path)[0]["open_incidents"] == ["a"]


def test_a_failing_observer_is_counted_never_raised(tmp_path: Path) -> None:
    stats = WatchStats()
    ledger = Ledger(tmp_path, observer=_Observer(fail=True), stats=stats)
    assert ledger.write(_record(1))
    assert ledger.close(5.0)
    assert ledger.stats()["export_failures"] == 1
    assert counter_value(stats.health(), "sink_dropped_total", ("export",)) == 1
    assert len(read_ledger(tmp_path)) == 1


def test_a_blocked_fsync_never_blocks_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    real_fsync = os.fsync

    def blocked_fsync(fd: int) -> None:
        gate.wait()
        real_fsync(fd)

    monkeypatch.setattr(telemetry_sink.os, "fsync", blocked_fsync)
    stats = WatchStats()
    ledger = Ledger(tmp_path, stats=stats, stall_seconds=0.05)
    threads = threading.active_count()
    slowest = 0.0
    for index in range(2000):
        started = time.perf_counter()
        ledger.write(_record(index))
        slowest = max(slowest, time.perf_counter() - started)
        if index == 60:
            time.sleep(0.1)  # the 50th record's flush is now stalled
    ledger_stats = ledger.stats()
    assert slowest < 0.25
    assert ledger_stats["io_stalled"]
    assert ledger_stats["ledger_dropped"] > 0
    assert counter_value(stats.health(), "sink_dropped_total", ("ledger",)) > 0
    assert threading.active_count() <= threads + 1  # at most the sink's flusher
    gate.set()
    assert ledger.close(5.0)


def test_sustained_write_failure_keeps_the_ledger_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_write(fd: int, data: bytes | memoryview) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    ledger = Ledger(tmp_path, max_buffer_bytes=64 * 1024)
    monkeypatch.setattr(telemetry_sink.os, "write", failing_write)
    for index in range(3000):
        ledger.write({**_record(index), "pad": "x" * 200})
    _wait_for(lambda: ledger.stats()["queued"] == 0)
    sink = ledger.stats()["sink"]
    assert sink["buffered_bytes"] <= 64 * 1024
    assert sink["dropped_records"] > 0
    assert ledger.stats()["ledger_dropped"] >= sink["dropped_records"]
    monkeypatch.undo()
    ledger.close(5.0)
