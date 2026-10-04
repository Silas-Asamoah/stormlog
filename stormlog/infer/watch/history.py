"""The watcher's bounded memory of the recent past.

Items are held serialized and compressed (zlib level 1), so the byte bound
counts exactly what is held: a vLLM 0.30.0 scrape is about 5.7 KB this way
against about 130 KB as parsed Python objects. Each item carries the
watcher's own stamps (:class:`Stamped`), so windows are cut on its monotonic
clock, which wall-clock steps cannot move. Only the last few scrapes, as many
as the widest trigger window needs, are also kept parsed.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from ..vllm_telemetry import VllmScrapeRecord

EVICT_AGE = "age"
EVICT_BYTES = "bytes"
EVICT_OVERSIZED = "oversized"


@dataclass(frozen=True)
class Stamped:
    """When the watcher saw an item, on its own clocks.

    ``mono_ns`` is when it began fetching or received the item and
    ``done_mono_ns`` when the fetch returned; the server's sample instant
    lies between them. ``wall_ns`` pairs ``mono_ns`` with the wall clock.
    """

    mono_ns: int
    done_mono_ns: int
    wall_ns: int

    def __post_init__(self) -> None:
        if self.done_mono_ns < self.mono_ns:
            raise ValueError("done_mono_ns precedes mono_ns")


class BoundedRing:
    """Serialized items held for ``max_seconds`` and in at most ``max_bytes``."""

    def __init__(self, *, max_seconds: float, max_bytes: int) -> None:
        if max_seconds <= 0 or max_bytes <= 0:
            raise ValueError("ring bounds must be > 0")
        self.max_ns = int(max_seconds * 1e9)
        self.max_bytes = max_bytes
        self._items: deque[tuple[Stamped, bytes]] = deque()
        self._bytes = 0
        self.evictions: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def bytes(self) -> int:
        return self._bytes

    def held_seconds(self) -> float:
        """The span between the oldest and newest items held."""
        if not self._items:
            return 0.0
        return (self._items[-1][0].mono_ns - self._items[0][0].mono_ns) / 1e9

    def append(self, stamp: Stamped, record: Mapping[str, Any]) -> bool:
        """Hold one item; False when it alone is over the byte bound."""
        blob = zlib.compress(
            json.dumps(record, separators=(",", ":"), sort_keys=True).encode(), 1
        )
        if len(blob) > self.max_bytes:
            self.evictions[EVICT_OVERSIZED] += 1
            return False
        self._items.append((stamp, blob))
        self._bytes += len(blob)
        while self._bytes > self.max_bytes:
            self._evict(EVICT_BYTES)
        self.expire(stamp.mono_ns)
        return True

    def expire(self, now_mono_ns: int) -> None:
        """Drop items older than ``max_seconds`` before ``now_mono_ns``."""
        cutoff = now_mono_ns - self.max_ns
        while self._items and self._items[0][0].mono_ns < cutoff:
            self._evict(EVICT_AGE)

    def items(
        self, start_mono_ns: int | None = None, end_mono_ns: int | None = None
    ) -> Iterator[tuple[Stamped, dict[str, Any]]]:
        """Held items whose ``mono_ns`` lies in ``[start, end]``, oldest first."""
        for stamp, blob in list(self._items):
            if start_mono_ns is not None and stamp.mono_ns < start_mono_ns:
                continue
            if end_mono_ns is not None and stamp.mono_ns > end_mono_ns:
                continue
            yield stamp, json.loads(zlib.decompress(blob))

    def _evict(self, cause: str) -> None:
        _stamp, blob = self._items.popleft()
        self._bytes -= len(blob)
        self.evictions[cause] += 1


class ScrapeHistory:
    """Every scrape, compressed, plus the last ``parsed_count`` kept parsed."""

    def __init__(self, *, max_seconds: float, max_bytes: int, parsed_count: int):
        if parsed_count < 2:
            raise ValueError("parsed_count must be >= 2")
        self.ring = BoundedRing(max_seconds=max_seconds, max_bytes=max_bytes)
        self._parsed: deque[tuple[Stamped, VllmScrapeRecord]] = deque(
            maxlen=parsed_count
        )

    def add(self, stamp: Stamped, record: VllmScrapeRecord) -> None:
        self.ring.append(stamp, record.to_record())
        self._parsed.append((stamp, record))

    def parsed(self) -> list[tuple[Stamped, VllmScrapeRecord]]:
        """The parsed tail, oldest first."""
        return list(self._parsed)

    def records(
        self, start_mono_ns: int | None = None, end_mono_ns: int | None = None
    ) -> list[tuple[Stamped, VllmScrapeRecord]]:
        """Scrapes held between two monotonic instants, parsed back."""
        return [
            (stamp, VllmScrapeRecord.from_record(record))
            for stamp, record in self.ring.items(start_mono_ns, end_mono_ns)
        ]


__all__ = [
    "EVICT_AGE",
    "EVICT_BYTES",
    "EVICT_OVERSIZED",
    "BoundedRing",
    "ScrapeHistory",
    "Stamped",
]
