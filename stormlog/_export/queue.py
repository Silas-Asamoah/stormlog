"""A FIFO bounded by item count and by estimated bytes, whose offers never block.

The producer is the code being measured: an inference client's event loop,
or a watcher's tick. It appends and moves on. When either bound is reached
the new item is dropped and counted, so queued items keep their order and
nothing already accepted is ever evicted. Consumers take short batches, so
a producer waiting for the lock waits for at most one of them.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

# The most items a consumer removes per lock acquisition.
TAKE_LIMIT = 64


@dataclass(frozen=True)
class QueueStats:
    """Counts since the queue was made, and its state now."""

    offered: int
    accepted: int
    dropped_full: int
    dropped_closed: int
    depth: int
    depth_bytes: int
    high_water: int
    high_water_bytes: int
    max_items: int
    max_bytes: int
    closed: bool


class BoundedQueue(Generic[T]):
    """Append without blocking; take in short batches from another thread."""

    def __init__(self, *, max_items: int, max_bytes: int) -> None:
        if max_items < 1 or max_bytes < 1:
            raise ValueError("a queue needs room for at least one item and byte")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items: deque[tuple[T, int]] = deque()
        self._bytes = 0
        self._cond = threading.Condition(threading.Lock())
        self._closed = False
        self._offered = 0
        self._accepted = 0
        self._dropped_full = 0
        self._dropped_closed = 0
        self._high_water = 0
        self._high_water_bytes = 0

    def offer(self, item: T, size: int) -> bool:
        """Queue ``item`` of estimated ``size`` bytes, or drop it and count why."""
        with self._cond:
            self._offered += 1
            if self._closed:
                self._dropped_closed += 1
                return False
            if (
                len(self._items) >= self.max_items
                or self._bytes + size > self.max_bytes
            ):
                self._dropped_full += 1
                return False
            self._items.append((item, size))
            self._bytes += size
            self._accepted += 1
            self._high_water = max(self._high_water, len(self._items))
            self._high_water_bytes = max(self._high_water_bytes, self._bytes)
            self._cond.notify()
            return True

    def take(self, limit: int = TAKE_LIMIT, timeout: float | None = None) -> list[T]:
        """Up to ``limit`` items, waiting at most ``timeout`` seconds for one.

        Returns an empty list on timeout, or at once when the queue is closed
        and empty. ``limit`` is capped at ``TAKE_LIMIT``.
        """
        limit = max(1, min(limit, TAKE_LIMIT))
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._items and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return []
                self._cond.wait(remaining)
            return self._pop_locked(limit)

    def drain(self) -> list[T]:
        """Remove and return everything still queued, in order."""
        with self._cond:
            return self._pop_locked(len(self._items))

    def close(self) -> None:
        """Refuse further offers and wake every waiting consumer."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def stats(self) -> QueueStats:
        with self._cond:
            return QueueStats(
                offered=self._offered,
                accepted=self._accepted,
                dropped_full=self._dropped_full,
                dropped_closed=self._dropped_closed,
                depth=len(self._items),
                depth_bytes=self._bytes,
                high_water=self._high_water,
                high_water_bytes=self._high_water_bytes,
                max_items=self.max_items,
                max_bytes=self.max_bytes,
                closed=self._closed,
            )

    def _pop_locked(self, limit: int) -> list[T]:
        taken: list[T] = []
        while self._items and len(taken) < limit:
            item, size = self._items.popleft()
            self._bytes -= size
            taken.append(item)
        return taken
