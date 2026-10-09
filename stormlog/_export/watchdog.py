"""Shut sockets down at their deadlines, from one daemon thread.

A per-operation socket timeout cannot bound a peer that sends one byte just
before each timeout, and it does not reach a TLS handshake in progress.
Shutting the socket down from another thread ends a blocked receive, a
dribbled HTTP read and a handshake on a registered TLS socket alike. Each
arming gets its own token, so a deadline that passes after its attempt has
ended never touches a later attempt's socket. Disarmed deadlines are
dropped once they outnumber the armed ones, so the pending deadlines stay
within about twice the sockets armed at once.
"""

from __future__ import annotations

import heapq
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# Below this many pending deadlines, disarmed ones are left to expire.
_COMPACT_AT = 64


@dataclass
class WatchdogStats:
    armed: int = 0
    # Deadlines that passed and shut their socket down.
    fired: int = 0
    # Deadlines that passed but whose shutdown failed: nothing was cut.
    failed: int = 0


class WatchdogStopped(RuntimeError):
    """Raised by ``arm`` once the watchdog has stopped, and only then."""


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
        """Shut ``sock`` down when the clock passes ``deadline``; returns a token.

        ``ValueError`` for a closed or detached socket, which no shutdown
        could reach; ``WatchdogStopped`` once the watchdog has stopped; and
        ``RuntimeError`` when its thread cannot start, with nothing armed.
        """
        if sock.fileno() == -1:
            raise ValueError("a closed or detached socket cannot be armed")
        with self._cond:
            if self._stopped:
                raise WatchdogStopped("the watchdog has stopped")
            self._ensure_thread()
            self._next_token += 1
            token = self._next_token
            self._sockets[token] = sock
            heapq.heappush(self._heap, (deadline, token))
            self.stats.armed += 1
            self._cond.notify()
            return token

    def disarm(self, token: int) -> bool:
        """Forget ``token``; True when its deadline had not yet passed."""
        with self._cond:
            self._sockets.pop(token, None)
            self._compact()
            if token in self._fired:
                self._fired.discard(token)
                return False
            return True

    def fired(self, token: int) -> bool:
        """Whether ``token``'s deadline passed and shut its socket down."""
        with self._cond:
            return token in self._fired

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def _compact(self) -> None:
        # Called with the lock held: drop disarmed deadlines once they are
        # more than half of those pending.
        heap = self._heap
        if len(heap) > _COMPACT_AT and len(heap) > 2 * len(self._sockets):
            self._heap = [entry for entry in heap if entry[1] in self._sockets]
            heapq.heapify(self._heap)

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
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            self.stats.failed += 1
            return
        self._fired.add(token)
        self.stats.fired += 1
