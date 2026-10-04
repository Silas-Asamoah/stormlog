"""A gate that pauses hold shut, for the step loop and the front end."""

from __future__ import annotations

import math
import threading
import time


class Hold:
    """Open until paused. An untimed pause holds until ``resume()``; a timed
    one until its own deadline. Pauses stack: the gate opens when the last
    of them ends, so an earlier, shorter pause's timer never releases it."""

    def __init__(self) -> None:
        self._open = threading.Event()
        self._open.set()
        self._lock = threading.Lock()
        # On the monotonic clock; infinite while an untimed pause holds.
        self._until = 0.0
        # The timers of timed pauses, cancelled when resume() ends them all.
        self._timers: list[threading.Timer] = []

    @property
    def held(self) -> bool:
        return not self._open.is_set()

    def pause(self, seconds: float | None = None) -> None:
        until = math.inf if seconds is None else time.monotonic() + seconds
        with self._lock:
            self._until = max(self._until, until) if self.held else until
            self._open.clear()
        if seconds is not None:
            self._arm(seconds)

    def resume(self) -> None:
        with self._lock:
            self._until = 0.0
            self._open.set()
            timers, self._timers = self._timers, []
        for timer in timers:
            timer.cancel()

    def wait(self, timeout: float | None = None) -> bool:
        return self._open.wait(timeout)

    def _arm(self, seconds: float) -> None:
        timer = threading.Timer(seconds, self._expire)
        timer.daemon = True
        with self._lock:
            self._timers = [t for t in self._timers if t.is_alive()] + [timer]
        timer.start()

    def _expire(self) -> None:
        with self._lock:
            remaining = self._until - time.monotonic()
            if remaining <= 0:
                self._open.set()
                return
        if not math.isinf(remaining):
            # A timer that fired a little early, or a shorter pause's: wait
            # out the rest of the latest deadline.
            self._arm(remaining)


__all__ = ["Hold"]
