"""The watcher's ledger: every record, appended locally, then handed on.

Records go to an :class:`~stormlog.telemetry_sink.AppendOnlyTelemetrySink` in
its bounded mode, driven by its own :class:`~.io.SerialWorker`, so the
watcher's loop never waits on the disk: it only queues. Right after queueing,
the same record is handed to an observer (#220's export pipeline), whose
``observe`` must not block; a failing observer is counted, never raised. A
record the queue cannot take, or the sink cuts back, is dropped and counted;
the counts live in memory, so they stay readable when the disk does not.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ...telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig
from .io import SerialWorker
from .records import validate_record
from .stats import WatchStats


class RecordObserver(Protocol):
    """Receives every ledger record; must neither block nor raise."""

    def observe(self, record: Mapping[str, Any]) -> None: ...


class Ledger:
    """Append watcher records without blocking, and tap them for export."""

    def __init__(
        self,
        root: Path,
        *,
        observer: RecordObserver | None = None,
        stats: WatchStats | None = None,
        max_buffer_bytes: int = 8 * 1024 * 1024,
        max_queued: int = 4096,
        max_queued_bytes: int = 8 * 1024 * 1024,
        stall_seconds: float = 5.0,
    ) -> None:
        self.directory = Path(root) / "ledger"
        self._sink = AppendOnlyTelemetrySink(
            TelemetrySinkConfig(
                root_dir=self.directory,
                flush_every_events=50,
                flush_every_seconds=1.0,
                write_rollups=False,
                max_buffer_bytes=max_buffer_bytes,
            )
        )
        self._worker = SerialWorker(
            "stormlog-watch-ledger",
            max_queued=max_queued,
            max_queued_bytes=max_queued_bytes,
            stall_seconds=stall_seconds,
        )
        self.observer = observer
        self._stats = stats
        self._lock = threading.Lock()
        self._rejected = 0
        self._sink_dropped = 0
        self._export_failures = 0
        self._sink_health: dict[str, Any] = {}

    def write(self, record: Mapping[str, Any]) -> bool:
        """Queue one record and hand it to the observer; False if dropped."""
        validate_record(record)
        # A deep snapshot: the caller may change its objects once this returns.
        encoded = json.dumps(record, separators=(",", ":"), default=str)
        payload = json.loads(encoded)
        taken = self._worker.submit(lambda: self._append(payload), nbytes=len(encoded))
        if not taken:
            with self._lock:
                self._rejected += 1
            self._count_drop("ledger")
        if self.observer is not None:
            try:
                self.observer.observe(payload)
            except Exception:  # an exporter's failure is never the watcher's
                with self._lock:
                    self._export_failures += 1
                self._count_drop("export")
        return taken

    def close(self, timeout: float) -> bool:
        """Flush and close the sink within ``timeout``; True if it finished."""
        return self._worker.close(timeout, final=self._close_sink)

    def stats(self) -> dict[str, Any]:
        """Counters read from memory; the sink's own are as of its last write."""
        worker = self._worker.stats()
        with self._lock:
            return {
                # Refused by the queue, failed in the sink, or cut back by it.
                "ledger_dropped": self._rejected + worker.failed + self._sink_dropped,
                "export_failures": self._export_failures,
                "io_rejected": worker.rejected,
                "io_stalled": worker.stalled,
                "queued": worker.queued,
                "write_failures": worker.failed,
                "sink": dict(self._sink_health),
            }

    def _append(self, record: dict[str, Any]) -> None:
        try:
            self._sink.append(record)
        except Exception:
            self._count_drop("ledger")
            raise
        finally:
            self._refresh_sink_health()

    def _close_sink(self) -> None:
        try:
            self._sink.close()
        finally:
            # Closing counts what the sink still held unwritten as dropped.
            self._refresh_sink_health()

    def _refresh_sink_health(self) -> None:
        health = self._sink.failure_diagnostics()
        dropped = int(health.get("dropped_records") or 0)
        with self._lock:
            self._sink_health = health
            newly, self._sink_dropped = dropped - self._sink_dropped, dropped
        if newly > 0:
            self._count_drop("ledger", newly)

    def _count_drop(self, sink: str, amount: int = 1) -> None:
        if self._stats is not None:
            self._stats.add("sink_dropped_total", amount, (sink,))


__all__ = ["Ledger", "RecordObserver"]
