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
    """The requests a schedule sent, and when the schedule started.

    ``tasks`` holds the requests still running and any that failed, whose
    errors are raised by whoever drains them; finished requests are dropped
    so a long schedule holds no more than its in-flight limit.
    """

    started_at_ns: int
    tasks: list[asyncio.Task[None]]


async def dispatch_schedule(
    offsets: Sequence[float],
    *,
    mode: str,
    limiter: InFlightLimiter,
    overflow: Overflow,
    send: Callable[[Arrival], Awaitable[None]],
    drop: Callable[[Arrival, str], None],
    deadline: float | None = None,
) -> Dispatch:
    """Start each request at its offset from now, in seconds.

    With ``overflow="wait"`` an arrival that finds every slot busy is held
    until one frees up; while it waits, later arrivals fall behind schedule
    too, which their dispatch lag shows. With ``"drop"`` it is handed to
    ``drop`` and never sent, with the reason. ``deadline`` is when the
    phase's drain ends, in seconds from now: an arrival still unsent then is
    dropped rather than held any longer. Returns once every arrival has been
    sent or dropped; the requests themselves may still be running.
    """
    started = time.perf_counter()
    started_ns = time.time_ns()
    deadline_at = None if deadline is None else started + deadline
    # In creation order, so the first failure is the one raised.
    tasks: dict[asyncio.Task[None], None] = {}

    def forget(task: asyncio.Task[None]) -> None:
        if not _still_matters(task):
            tasks.pop(task, None)

    try:
        for index, offset in enumerate(offsets):
            delay = started + offset - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            arrival = Arrival(
                index,
                mode,
                started_ns + round(offset * 1e9),
                held_for_slot=limiter.full,
            )
            if overflow == "drop" and arrival.held_for_slot:
                drop(arrival, f"in-flight limit of {limiter.limit} reached")
                continue
            in_flight = await _acquire_by(limiter, deadline_at)
            if in_flight is None:
                drop(arrival, "still waiting to be sent when the drain deadline passed")
                continue
            arrival = replace(arrival, in_flight_at_dispatch=in_flight)
            task = asyncio.create_task(_release_after(send(arrival), limiter))
            tasks[task] = None
            task.add_done_callback(forget)
    except asyncio.CancelledError:
        # Stopped part-way: the requests already sent finish cancelling first.
        await cancel_all(list(tasks))
        raise
    # Callbacks for the last requests to finish may not have run yet.
    return Dispatch(
        started_at_ns=started_ns, tasks=[task for task in tasks if _still_matters(task)]
    )


def _still_matters(task: asyncio.Task[None]) -> bool:
    """Still running, or failed with an error the drain has to raise."""
    if not task.done():
        return True
    return not task.cancelled() and task.exception() is not None


async def _acquire_by(limiter: InFlightLimiter, deadline: float | None) -> int | None:
    """Take a slot, or return None once ``deadline`` (a perf_counter time) passes."""
    if deadline is None:
        return await limiter.acquire()
    timeout = deadline - time.perf_counter()
    if timeout <= 0:
        return None
    try:
        return await asyncio.wait_for(limiter.acquire(), timeout)
    except asyncio.TimeoutError:
        return None


async def cancel_all(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel tasks and wait until each has finished cancelling."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _release_after(request: Awaitable[None], limiter: InFlightLimiter) -> None:
    try:
        await request
    finally:
        limiter.release()
