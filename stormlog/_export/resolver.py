"""Resolve an export destination to a few addresses, without blocking the caller.

``getaddrinfo`` cannot be cancelled, so it runs in one resolver thread and a
caller waits for it only so long. A resolution still stuck is reported, and
never joined by a second one. Up to four addresses are kept, in the order
the system returned them, with the last one that worked tried first: a host
such as ``localhost`` often lists ``::1`` before ``127.0.0.1``, and a
collector listening on one family must still be reached through the other.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_CANDIDATES = 4


@dataclass(frozen=True)
class Candidate:
    """One address to connect to, as ``getaddrinfo`` gave it."""

    family: int
    type: int
    proto: int
    address: tuple[Any, ...]


@dataclass
class ResolverStats:
    resolutions: int = 0
    failures: int = 0
    last_error: str | None = None


class Resolver:
    """Resolve ``host`` and ``port`` in the background; hand out candidates."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_candidates: int = MAX_CANDIDATES,
        stall_seconds: float = 2.0,
        getaddrinfo: Callable[..., list[Any]] = socket.getaddrinfo,
    ) -> None:
        self.host = host
        self.port = port
        self.max_candidates = max_candidates
        self.stall_seconds = stall_seconds
        self._getaddrinfo = getaddrinfo
        self._lock = threading.Lock()
        self._candidates: list[Candidate] = []
        self._good: Candidate | None = None
        self._done: threading.Event | None = None
        self._started_at: float | None = None
        self.stats = ResolverStats()

    def resolve(self, wait: float) -> bool:
        """Resolve, waiting at most ``wait`` seconds; whether any address is known.

        A resolution already running is waited for, not repeated.
        """
        with self._lock:
            done = self._done
            if done is None:
                done = self._done = threading.Event()
                self._started_at = time.monotonic()
                threading.Thread(
                    target=self._run,
                    args=(done,),
                    name="stormlog-resolver",
                    daemon=True,
                ).start()
        done.wait(wait)
        with self._lock:
            return bool(self._candidates)

    @property
    def stalled(self) -> bool:
        """Whether a resolution has run longer than ``stall_seconds``."""
        with self._lock:
            started = self._started_at
            return started is not None and (
                time.monotonic() - started > self.stall_seconds
            )

    def candidates(self) -> list[Candidate]:
        """The known addresses, the last one that worked first."""
        with self._lock:
            ordered = list(self._candidates)
            good = self._good
        if good in ordered:
            ordered.remove(good)
            ordered.insert(0, good)
        return ordered

    def mark_good(self, candidate: Candidate) -> None:
        with self._lock:
            self._good = candidate

    def _run(self, done: threading.Event) -> None:
        try:
            found = self._getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        except OSError as exc:
            with self._lock:
                self.stats.failures += 1
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
            found = None
        with self._lock:
            if found is not None:
                self._candidates = _first_unique(found, self.max_candidates)
                self.stats.resolutions += 1
            self._done = None
            self._started_at = None
        done.set()


def _first_unique(found: list[Any], limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for family, kind, proto, _canonical, address in found:
        candidate = Candidate(family, kind, proto, tuple(address))
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) == limit:
            break
    return candidates
