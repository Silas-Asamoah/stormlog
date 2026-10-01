"""Send requests at scheduled times, bounded by an in-flight limit."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

Overflow = Literal["wait", "drop"]


@dataclass(frozen=True)
class Arrival:
    """One scheduled request and how it got onto the wire."""

    index: int
    mode: str
    intended_at_ns: int
    # Every in-flight slot was busy when the dispatcher reached this arrival.
    held_for_slot: bool = False
    in_flight_at_dispatch: int | None = None


class InFlightLimiter:
    """Count outstanding requests and hold new ones at the limit."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("in-flight limit must be >= 1")
        self.limit = limit
        self.active = 0
        self._slots = asyncio.Semaphore(limit)

    @property
    def full(self) -> bool:
        return self.active >= self.limit

    async def acquire(self) -> int:
        """Wait for a free slot; return the in-flight count including this one."""
        await self._slots.acquire()
        self.active += 1
        return self.active

    def release(self) -> None:
        self.active -= 1
        self._slots.release()


@dataclass(frozen=True)
class Dispatch:
    """The requests a schedule sent, and when the schedule started."""

    started_at_ns: int
    tasks: list[asyncio.Task[None]]


async def dispatch_schedule(
    offsets: Sequence[float],
    *,
    mode: str,
    limiter: InFlightLimiter,
    overflow: Overflow,
    send: Callable[[Arrival], Awaitable[None]],
    drop: Callable[[Arrival], None],
) -> Dispatch:
    """Start each request at its offset from now, in seconds.

    With ``overflow="wait"`` an arrival that finds every slot busy is held
    until one frees up; while it waits, later arrivals fall behind schedule
    too, which their dispatch lag shows. With ``"drop"`` it is handed to
    ``drop`` and never sent. Returns once every arrival has been sent or
    dropped; the requests themselves may still be running.
    """
    started = time.perf_counter()
    started_ns = time.time_ns()
    tasks: list[asyncio.Task[None]] = []
    for index, offset in enumerate(offsets):
        delay = started + offset - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        arrival = Arrival(
            index, mode, started_ns + round(offset * 1e9), held_for_slot=limiter.full
        )
        if overflow == "drop" and arrival.held_for_slot:
            drop(arrival)
            continue
        in_flight = await limiter.acquire()
        arrival = replace(arrival, in_flight_at_dispatch=in_flight)
        tasks.append(asyncio.create_task(_release_after(send(arrival), limiter)))
    return Dispatch(started_at_ns=started_ns, tasks=tasks)


async def _release_after(request: Awaitable[None], limiter: InFlightLimiter) -> None:
    try:
        await request
    finally:
        limiter.release()
