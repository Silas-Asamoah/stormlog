"""Studio's document and selection caches (design §2.1, §8.1; component C4).

A summary over 100,000 rows costs about a second (§2.1), and the frontend
re-reads the same documents while a user pages, brushes and replays. So
the routes keep their last results here, keyed by what makes a result
valid: the artifact's path, its index ID, the index generation and the
canonical query parameters. A rebuilt index has a new index ID and a
refreshed one a new generation, so a stale entry is never hit; it ages out
of the LRU instead of being invalidated.

Both caches are bounded by count (``Limits.cached_documents`` and
``Limits.cached_selections``), locked, and make each value once: callers
that ask for a key while another thread makes it wait for that result
instead of computing it again. A ``make`` that raises caches nothing; every
caller waiting on it gets the same exception, and the next call tries
again.

Cached values are shared between callers and must be treated as
read-only. Routes encode a document as it is (``contracts.encode_json``
copies it), so none needs to change one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any, Generic, TypeVar

from stormlog.studio.contracts import Limits, Selection

__all__ = ["Documents", "cache_key"]

T = TypeVar("T")


def cache_key(
    name: str, path: str, index_id: str, generation: int, params: str = ""
) -> tuple[str, ...]:
    """The key of one document or selection: what it is and what it read.

    ``name`` is the document's name (``"summary"``, ``"export_preview"``)
    or ``"selection"``; ``params`` the canonical query string, empty for a
    document without one.
    """
    if not name:
        raise ValueError("a cache key needs a document name")
    return (name, path, index_id, str(generation), params)


class _Flight:
    """One value being made, for the threads waiting on it."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._value: Any = None
        self._error: BaseException | None = None

    def finish(self, value: Any) -> None:
        self._value = value
        self._done.set()

    def fail(self, error: BaseException) -> None:
        self._error = error
        self._done.set()

    def result(self) -> Any:
        self._done.wait()
        if self._error is not None:
            raise self._error
        return self._value


class _Lru(Generic[T]):
    """A locked LRU of at most ``capacity`` values, each made once."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a cache holds at least one value")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._values: OrderedDict[Hashable, T] = OrderedDict()
        self._flights: dict[Hashable, _Flight] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def get_or_make(self, key: Hashable, make: Callable[[], T]) -> T:
        with self._lock:
            if key in self._values:
                self._values.move_to_end(key)
                self.hits += 1
                return self._values[key]
            flight = self._flights.get(key)
            owner = flight is None
            if flight is None:
                flight = self._flights[key] = _Flight()
                self.misses += 1
        if not owner:
            result: T = flight.result()
            return result
        return self._make(key, make, flight)

    def _make(self, key: Hashable, make: Callable[[], T], flight: _Flight) -> T:
        try:
            value = make()
        except BaseException as error:
            with self._lock:
                del self._flights[key]
            flight.fail(error)
            raise
        with self._lock:
            del self._flights[key]
            self._values[key] = value
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)
        flight.finish(value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


class Documents:
    """C4's :class:`~stormlog.studio.contracts.DocumentCache`.

    Two LRUs: documents (64 by default) and selections (16 by default),
    from :class:`~stormlog.studio.contracts.Limits`. Keys are tuples of
    strings, normally from :func:`cache_key`.
    """

    def __init__(self, limits: Limits | None = None) -> None:
        limits = limits or Limits()
        self._documents: _Lru[dict[str, Any]] = _Lru(limits.cached_documents)
        self._selections: _Lru[Selection] = _Lru(limits.cached_selections)

    def selection(
        self, key: tuple[str, ...], make: Callable[[], Selection]
    ) -> Selection:
        """The cached selection for ``key``, made once."""
        return self._selections.get_or_make(_checked(key), make)

    def document(
        self, key: tuple[str, ...], make: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """The cached document for ``key``, made once."""
        return self._documents.get_or_make(_checked(key), make)

    def clear(self) -> None:
        """Drop every cached value (values being made still finish)."""
        self._documents.clear()
        self._selections.clear()

    def stats(self) -> dict[str, Any]:
        """Sizes, capacities and hit counts, for the server document."""
        return {
            name: {
                "size": len(lru),
                "capacity": lru.capacity,
                "hits": lru.hits,
                "misses": lru.misses,
            }
            for name, lru in (
                ("documents", self._documents),
                ("selections", self._selections),
            )
        }


def _checked(key: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(key, tuple) or not key:
        raise TypeError("a cache key is a non-empty tuple of strings")
    if not all(isinstance(part, str) for part in key):
        raise TypeError("a cache key is a non-empty tuple of strings")
    return key
