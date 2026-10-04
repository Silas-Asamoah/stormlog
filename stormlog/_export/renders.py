"""Rendered expositions shared by every reader, at most three alive at once.

A render is built at most once per interval and handed to every scrape and
to the textfile writer. A slow reader keeps its generation alive while it
sends it, so readers that start on successive renders could otherwise keep
one generation each. The publication limit bounds that: a new render starts
only while at most two published generations are alive, so the newest, one
older one still being read, and one being built are the most that exist.
When the limit holds a render back, readers get the newest existing one,
which is then up to one reader's deadline old.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

MAX_GENERATIONS = 3


@dataclass(eq=False)
class Generation:
    """One rendered exposition and the readers still holding it."""

    body: bytes
    number: int
    created_at: float
    readers: int = 0


@dataclass
class RenderStats:
    builds: int = 0
    reused: int = 0
    # Renders that were due but held back by the publication limit.
    deferred: int = 0
    alive_high_water: int = 0
    failures: int = 0


@dataclass
class _State:
    newest: Generation | None = None
    held: set[Generation] = field(default_factory=set)
    building: bool = False


class RenderCache:
    """Hand out the newest render, building a fresh one when it is due and allowed."""

    def __init__(
        self,
        render: Callable[[], bytes],
        *,
        min_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._render = render
        self.min_interval = min_interval
        self._clock = clock
        self._cond = threading.Condition(threading.Lock())
        self._state = _State()
        self._numbers = 0
        self.stats = RenderStats()

    def acquire(self) -> Generation:
        """The generation to send; release it when the send ends."""
        with self._cond:
            while not self._should_build():
                if self._state.newest is not None:
                    return self._reuse()
                # The first render is being built by another reader.
                self._cond.wait()
            self._state.building = True
            self._note_alive()
        try:
            body = self._render()
        except BaseException:
            with self._cond:
                self._state.building = False
                self.stats.failures += 1
                self._cond.notify_all()
            raise
        return self._publish(body)

    def release(self, generation: Generation) -> None:
        with self._cond:
            generation.readers -= 1
            if generation is not self._state.newest and generation.readers <= 0:
                self._state.held.discard(generation)

    def alive(self) -> int:
        """Generations alive now, including one being built."""
        with self._cond:
            return self._alive_locked()

    # ------------------------------------------------------------- internals
    def _published_alive(self) -> int:
        state = self._state
        return (1 if state.newest is not None else 0) + len(state.held)

    def _alive_locked(self) -> int:
        return self._published_alive() + (1 if self._state.building else 0)

    def _should_build(self) -> bool:
        state = self._state
        due = (
            state.newest is None
            or self._clock() - state.newest.created_at >= self.min_interval
        )
        if not due or state.building:
            return False
        if self._published_alive() >= MAX_GENERATIONS:
            self.stats.deferred += 1
            return False
        return True

    def _reuse(self) -> Generation:
        newest = self._state.newest
        assert newest is not None
        newest.readers += 1
        self.stats.reused += 1
        return newest

    def _note_alive(self) -> None:
        self.stats.alive_high_water = max(
            self.stats.alive_high_water, self._alive_locked()
        )

    def _publish(self, body: bytes) -> Generation:
        with self._cond:
            self._numbers += 1
            generation = Generation(body, self._numbers, self._clock(), readers=1)
            previous = self._state.newest
            if previous is not None and previous.readers > 0:
                self._state.held.add(previous)
            self._state.newest = generation
            self._state.building = False
            self.stats.builds += 1
            self._note_alive()
            self._cond.notify_all()
            return generation
