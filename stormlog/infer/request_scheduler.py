"""Task-neutral arrival, drain and blocking-call accounting.

Text and audio own their payloads/events. These callbacks retain the existing
worker indices and open-loop admission semantics without inventing token data.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal

from .arrivals import CLOSED, ArrivalSpec, arrival_offsets, scheduled_endpoint
from .open_loop import Arrival, Dispatch, InFlightLimiter, cancel_all, dispatch_schedule


@dataclass(frozen=True)
class PhaseWindow:
    started_at_ns: int
    window_ended_at_ns: int
    drained_at_ns: int
    scheduled_arrivals: int | None = None
    scheduled_endpoint_offset_ns: int | None = None


@dataclass(frozen=True)
class AbandonedWait:
    running_at_start: int = 0
    waited_seconds: float = 0.0
    still_running: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "running_at_start": self.running_at_start,
            "waited_seconds": self.waited_seconds,
            "still_running": self.still_running,
        }


class LateCalls:
    """Blocking requests outlive cancellation of their asyncio waiter."""

    def __init__(self) -> None:
        self._calls: set[Future[Any]] = set()
        self._lock = threading.Lock()

    def track(self, call: Future[Any]) -> None:
        with self._lock:
            self._calls.add(call)
        call.add_done_callback(self.untrack)

    def untrack(self, call: Future[Any]) -> None:
        with self._lock:
            self._calls.discard(call)

    def running(self) -> list[Future[Any]]:
        with self._lock:
            return [call for call in self._calls if not call.done()]

    async def wait(self, timeout_seconds: float) -> AbandonedWait:
        calls = self.running()
        if not calls:
            return AbandonedWait()
        started = time.perf_counter()
        await asyncio.wait(
            [asyncio.wrap_future(call) for call in calls], timeout=timeout_seconds + 1.0
        )
        return AbandonedWait(
            len(calls),
            time.perf_counter() - started,
            sum(not call.done() for call in calls),
        )


class RequestScheduler:
    def __init__(
        self,
        *,
        case_id: str,
        phase: str,
        arrival: ArrivalSpec,
        concurrency: int,
        seed: int,
        overflow: Literal["wait", "drop"],
        drain_timeout: float,
        send: Callable[[str, Arrival], Awaitable[None]],
        drop: Callable[[str, Arrival, str], None],
        until: asyncio.Event | None = None,
    ) -> None:
        self.case_id, self.phase, self.arrival = case_id, phase, arrival
        self.concurrency, self.seed, self.overflow = concurrency, seed, overflow
        self.drain_timeout, self.send, self.drop = drain_timeout, send, drop
        self.until = until

    async def closed(
        self,
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> "PhaseWindow":
        if total_requests is None and duration_seconds is None:
            total_requests = 1
        started_at_ns = time.time_ns()
        counter = RequestCounter(limit=total_requests)
        limiter = InFlightLimiter(self.concurrency)
        end_time = (
            time.monotonic() + duration_seconds
            if duration_seconds is not None
            else None
        )
        tasks = [
            asyncio.create_task(
                self._worker(
                    worker_id=index,
                    counter=counter,
                    limiter=limiter,
                    end_time=end_time,
                )
            )
            for index in range(self.concurrency)
        ]
        if duration_seconds is None:
            # Every request counts, so each one finishes or times out.
            await drain(tasks, timeout=None)
            window_ended_at_ns = counter.last_issued_at_ns or time.time_ns()
        else:
            await drain(tasks, timeout=duration_seconds + self.drain_timeout)
            window_ended_at_ns = started_at_ns + round(duration_seconds * 1e9)
        return PhaseWindow(started_at_ns, window_ended_at_ns, time.time_ns())

    async def _worker(
        self,
        *,
        worker_id: int,
        counter: "RequestCounter",
        limiter: InFlightLimiter,
        end_time: float | None,
    ) -> None:
        while True:
            if end_time is not None and time.monotonic() >= end_time:
                return
            if self.until is not None and self.until.is_set():
                return
            request_index = await counter.next()
            if request_index is None:
                return
            # In a closed loop a request is due as soon as its worker is free.
            in_flight = await limiter.acquire()
            arrival = Arrival(
                index=request_index,
                mode=CLOSED,
                intended_at_ns=time.time_ns(),
                in_flight_at_dispatch=in_flight,
            )
            try:
                await self.send(
                    f"{self.case_id}_{self.phase}_{worker_id}_{request_index}",
                    arrival,
                )
            finally:
                limiter.release()

    async def open(
        self,
        total_requests: int | None,
        duration_seconds: float | None,
    ) -> "PhaseWindow":
        offsets = arrival_offsets(
            self.arrival,
            count=total_requests,
            duration_seconds=duration_seconds,
            seed=self.seed,
        )

        async def send(arrival: Arrival) -> None:
            await self.send(f"{self.case_id}_{self.phase}_{arrival.index}", arrival)

        def drop(arrival: Arrival, reason: str) -> None:
            self.drop(f"{self.case_id}_{self.phase}_{arrival.index}", arrival, reason)

        drain_timeout = self.drain_timeout
        dispatch = await dispatch_schedule(
            offsets,
            mode=self.arrival.mode,
            limiter=InFlightLimiter(self.concurrency),
            overflow=self.overflow,
            send=send,
            drop=drop,
            deadline=(
                duration_seconds + drain_timeout
                if duration_seconds is not None
                else None
            ),
            until=self.until,
        )
        if dispatch.ended_at_ns is not None:
            return await self._stopped_early(dispatch, drain_timeout)
        if duration_seconds is None:
            # Counted arrivals are all sent, held ones included, so the
            # window closes with the last send and the whole drain follows.
            window_ended_at_ns = time.time_ns()
        else:
            # The drain is measured from the window end: arrivals held past
            # it have used up part of it, or all of it.
            window_ended_at_ns = dispatch.started_at_ns + round(duration_seconds * 1e9)
            drain_timeout += (window_ended_at_ns - time.time_ns()) / 1e9
        await drain(dispatch.tasks, timeout=max(drain_timeout, 0.0))
        endpoint = scheduled_endpoint(
            self.arrival,
            count=total_requests,
            duration_seconds=duration_seconds,
            seed=self.seed,
        )
        return PhaseWindow(
            dispatch.started_at_ns,
            window_ended_at_ns,
            time.time_ns(),
            scheduled_arrivals=len(offsets),
            scheduled_endpoint_offset_ns=(
                None if endpoint is None else round(endpoint * 1e9)
            ),
        )

    async def _stopped_early(
        self, dispatch: Dispatch, drain_timeout: float
    ) -> PhaseWindow:
        """Close an interrupted open-loop schedule before its normal endpoint."""
        assert dispatch.ended_at_ns is not None
        await drain(dispatch.tasks, timeout=drain_timeout)
        return PhaseWindow(
            dispatch.started_at_ns,
            dispatch.ended_at_ns,
            time.time_ns(),
            scheduled_arrivals=dispatch.scheduled,
            scheduled_endpoint_offset_ns=dispatch.ended_at_ns - dispatch.started_at_ns,
        )


async def drain(tasks: list[asyncio.Task[None]], *, timeout: float | None) -> None:
    """Wait for requests to finish, cancelling any still running at the timeout.

    A cancelled request records itself as cancelled. Other failures inside a
    task are bugs and propagate.
    """
    if not tasks:
        return
    try:
        _done, pending = await asyncio.wait(tasks, timeout=timeout)
    except asyncio.CancelledError:
        # Ctrl+C: requests record themselves while the artifact is still open.
        await cancel_all(tasks)
        raise
    for task in pending:
        task.cancel()
    for result in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(result, Exception):
            raise result


class RequestCounter:
    def __init__(self, *, limit: int | None) -> None:
        self.limit = limit
        self.value = 0
        self.last_issued_at_ns: int | None = None
        self.lock = asyncio.Lock()

    async def next(self) -> int | None:
        async with self.lock:
            if self.limit is not None and self.value >= self.limit:
                return None
            current = self.value
            self.value += 1
            if self.value == self.limit:
                self.last_issued_at_ns = time.time_ns()
            return current
