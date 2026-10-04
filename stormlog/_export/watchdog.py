"""Shut sockets down at their deadlines, from one daemon thread.

A per-operation socket timeout cannot bound a peer that sends one byte just
before each timeout, and it does not reach a TLS handshake in progress.
Shutting the socket down from another thread ends a blocked receive, a
dribbled HTTP read and a handshake on a registered TLS socket alike. Each
arming gets its own token, so a deadline that passes after its attempt has
ended never touches a later attempt's socket.
"""

from __future__ import annotations

import heapq
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class WatchdogStats:
    armed: int = 0
    fired: int = 0


class Watchdog:
    """Arm a socket with a deadline; disarm it when the work is done."""

    def __init__(
        self,
        *,
        name: str = "stormlog-watchdog",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._clock = clock
        self._cond = threading.Condition(threading.Lock())
        self._heap: list[tuple[float, int]] = []
        self._sockets: dict[int, socket.socket] = {}
        self._fired: set[int] = set()
        self._next_token = 0
        self._thread: threading.Thread | None = None
        self._stopped = False
        self.stats = WatchdogStats()

    def arm(self, sock: socket.socket, deadline: float) -> int:
        """Shut ``sock`` down when the clock passes ``deadline``; returns a token."""
        with self._cond:
            self._next_token += 1
            token = self._next_token
            self._sockets[token] = sock
            heapq.heappush(self._heap, (deadline, token))
            self.stats.armed += 1
            self._ensure_thread()
            self._cond.notify()
            return token

    def disarm(self, token: int) -> bool:
        """Forget ``token``; True when its deadline had not yet passed."""
        with self._cond:
            self._sockets.pop(token, None)
            if token in self._fired:
                self._fired.discard(token)
                return False
            return True

    def fired(self, token: int) -> bool:
        with self._cond:
            return token in self._fired

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name=self._name, daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        with self._cond:
            while not self._stopped:
                if not self._heap:
                    self._cond.wait()
                    continue
                deadline, token = self._heap[0]
                remaining = deadline - self._clock()
                if remaining > 0:
                    self._cond.wait(remaining)
                    continue
                heapq.heappop(self._heap)
                self._fire(token)

    def _fire(self, token: int) -> None:
        # Called with the lock held. A disarmed token has no socket left.
        sock = self._sockets.pop(token, None)
        if sock is None:
            return
        self._fired.add(token)
        self.stats.fired += 1
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
