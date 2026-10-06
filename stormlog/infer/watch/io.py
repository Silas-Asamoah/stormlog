"""Blocking I/O off the watcher's loop, with nothing piling up behind it.

A :class:`SerialWorker` owns one thread and one mutable target, such as the
ledger's sink: its operations run one at a time, in order. Submissions go
into a queue bounded by count and by the bytes each submission declares, and
a full queue rejects new work instead of growing. A caller can reserve a
finite extra count once for its shutdown batch, then close the worker;
the byte and stall bounds still apply. An operation that runs past
``stall_seconds`` (an fsync on a failing disk, say) keeps the thread busy,
and while it does every submission is rejected and counted: the worker never
starts a second thread, so a blocked call cannot multiply into many.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

# What an operation returns is ignored.
Operation = Callable[[], object]


@dataclass(frozen=True)
class WorkerStats:
    """A snapshot of one worker's queue and outcomes."""

    queued: int
    queued_bytes: int
    completed: int
    failed: int
    rejected: int
    stalled: bool
    last_error: str | None


class SerialWorker:
    """Runs submitted operations on one thread, one at a time."""

    def __init__(
        self,
        name: str,
        *,
        max_queued: int = 1024,
        max_queued_bytes: int = 4 * 1024 * 1024,
        stall_seconds: float = 5.0,
    ) -> None:
        if max_queued <= 0 or max_queued_bytes <= 0 or stall_seconds <= 0:
            raise ValueError("worker bounds must be > 0")
        self.name = name
        self.max_queued = max_queued
        self.max_queued_bytes = max_queued_bytes
        self.stall_seconds = stall_seconds
        self._queue: deque[tuple[Operation, int]] = deque()
        self._queued_bytes = 0
        self._reserved_slots: int | None = None
        self._condition = threading.Condition()
        self._closing = False
        self._started_at: float | None = None
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, operation: Operation, *, nbytes: int = 0) -> bool:
        """Queue one operation; False, and counted, when it cannot be taken."""
        with self._condition:
            full = (
                len(self._queue) >= self.max_queued + (self._reserved_slots or 0)
                or self._queued_bytes + nbytes > self.max_queued_bytes
            )
            if self._closing or full or self._stalled_locked():
                self._rejected += 1
                return False
            self._queue.append((operation, nbytes))
            self._queued_bytes += nbytes
            self._condition.notify()
            return True

    def reserve(self, slots: int) -> None:
        """Reserve extra count capacity once for a finite shutdown batch.

        Submit the batch, then close the worker. Existing queued work keeps
        its slots; byte, stalled and closing rejection still apply.
        """
        if slots < 0:
            raise ValueError("reserved slots must be >= 0")
        with self._condition:
            if self._reserved_slots is not None:
                raise ValueError("shutdown capacity can only be reserved once")
            self._reserved_slots = slots

    def stats(self) -> WorkerStats:
        with self._condition:
            return WorkerStats(
                queued=len(self._queue),
                queued_bytes=self._queued_bytes,
                completed=self._completed,
                failed=self._failed,
                rejected=self._rejected,
                stalled=self._stalled_locked(),
                last_error=self._last_error,
            )

    def close(self, timeout: float, *, final: Operation | None = None) -> bool:
        """Stop taking work and wait for what is queued; True once it is done.

        ``final``, such as closing the sink, is queued last whatever the
        bounds. At the timeout the thread is left behind: it is a daemon, so
        it never holds the process open.
        """
        with self._condition:
            if final is not None and not self._closing:
                self._queue.append((final, 0))
            self._closing = True
            self._condition.notify()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _stalled_locked(self) -> bool:
        started = self._started_at
        return started is not None and time.monotonic() - started > self.stall_seconds

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closing:
                    self._condition.wait()
                if not self._queue:
                    return
                operation, nbytes = self._queue.popleft()
                self._queued_bytes -= nbytes
                self._started_at = time.monotonic()
            error: str | None = None
            try:
                operation()
            except Exception as exc:  # a failed write is counted, never raised
                error = f"{type(exc).__name__}: {exc}"
            with self._condition:
                self._started_at = None
                if error is None:
                    self._completed += 1
                else:
                    self._failed += 1
                    self._last_error = error


__all__ = ["SerialWorker", "WorkerStats"]
