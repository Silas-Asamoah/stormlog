"""Split a request's TTFT and end-to-end latency into disjoint segments.

TTFT, from the client's send to its first content, is tiled by:

- ``send_to_ingress``: the send to the engine's admission (its alias), which
  covers HTTP, the API server and IPC; client clock to engine clock;
- ``engine_ingress``: admission to entering the scheduler's queue;
- ``scheduler_wait``: entering the queue to the ``schedule()`` call that first
  ran the request;
- ``prefill``: that call to the completion of the first step that kept an
  output token for it;
- ``first_token_delivery``: that completion to the client's first content.

End-to-end shares the first four, then ``decode`` to the completion of the
step it finished in, and ``final_delivery`` to the client's end. On a hook
log without ``enqueued`` records the second and third are one segment,
``engine_ingress_to_schedule``, which is never called a queue wait. Segments
on the engine's monotonic clock are exact; the deliveries and
``send_to_ingress`` cross clocks and are intervals, or unknown. The residual,
what the segments leave of the client's interval, exists only when every
segment does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .correlation_events import MembershipEvent
from .diagnosis_clocks import EngineClock, Interval, Placed
from .diagnosis_join import ClientRequest, Execution, RunView

TTFT = "ttft"
E2E = "e2e"
TTFT_SEGMENTS = (
    "send_to_ingress",
    "engine_ingress",
    "scheduler_wait",
    "prefill",
    "first_token_delivery",
)
E2E_SEGMENTS = (
    "send_to_ingress",
    "engine_ingress",
    "scheduler_wait",
    "prefill",
    "decode",
    "final_delivery",
)
MERGED_INGRESS = "engine_ingress_to_schedule"

NO_EXECUTION = "no_engine_execution"
SEVERAL_EXECUTIONS = "several_engine_executions"
NO_RETAINED_TOKEN = "no_retained_token"
NO_FINISH = "no_finish_step"
NOT_FINISHED = "client_not_finished"


@dataclass(frozen=True)
class Part:
    """One segment: a duration interval in ns, or why there is none."""

    name: str
    interval: Interval | None
    unknown: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval_ns": self.interval,
            "unknown": self.unknown,
        }


@dataclass(frozen=True)
class Decomposition:
    """One request's TTFT or end-to-end, segment by segment."""

    request_id: str
    kind: str
    total_ns: int | None
    parts: tuple[Part, ...] = ()
    unavailable: str | None = None
    lines: tuple[int, ...] = field(default_factory=tuple)

    def part(self, name: str) -> Part | None:
        return next((part for part in self.parts if part.name == name), None)

    @property
    def residual(self) -> Interval | None:
        """What the segments leave of the client's interval, bounded."""
        if self.total_ns is None or not self.parts:
            return None
        if any(part.interval is None for part in self.parts):
            return None
        low = sum(part.interval[0] for part in self.parts if part.interval)
        high = sum(part.interval[1] for part in self.parts if part.interval)
        return self.total_ns - high, self.total_ns - low


@dataclass(frozen=True)
class _Points:
    """An execution's engine times, monotonic, with the wall stamps the
    cross-clock segments need."""

    admitted: int | None
    admitted_wall: Placed
    enqueued: int | None
    first_scheduled: int | None
    first_retained: int | None
    first_retained_wall: Placed
    finished: int | None
    finished_wall: Placed


def decompose(
    view: RunView, request: ClientRequest, clocks: dict[str, EngineClock]
) -> tuple[Decomposition, Decomposition]:
    """The request's TTFT and end-to-end decompositions."""
    executions = view.executions_of(request.request_id)
    if len(executions) != 1:
        reason = NO_EXECUTION if not executions else SEVERAL_EXECUTIONS
        return (
            Decomposition(request.request_id, TTFT, _ttft(request), unavailable=reason),
            Decomposition(request.request_id, E2E, _e2e(request), unavailable=reason),
        )
    (execution,) = executions
    clock = clocks.get(execution.producer) or EngineClock(view, execution.producer)
    points = _points(view, execution, clock)
    lines = _lines(request, execution)
    return (
        _ttft_parts(request, points, clock, lines),
        _e2e_parts(request, points, clock, lines),
    )


def _ttft_parts(
    request: ClientRequest, points: _Points, clock: EngineClock, lines: tuple[int, ...]
) -> Decomposition:
    sent, first = request.sent_at_ns, request.first_content_at_ns
    if points.first_retained is None:
        return Decomposition(
            request.request_id,
            TTFT,
            _ttft(request),
            unavailable=NO_RETAINED_TOKEN,
            lines=lines,
        )
    parts = [
        _cross(clock, "send_to_ingress", sent, points.admitted_wall, outbound=True),
        *_ingress(points),
        _span("prefill", points.first_scheduled, points.first_retained),
        _cross(
            clock,
            "first_token_delivery",
            first,
            points.first_retained_wall,
            outbound=False,
        ),
    ]
    return Decomposition(
        request.request_id, TTFT, _ttft(request), tuple(parts), lines=lines
    )


def _e2e_parts(
    request: ClientRequest, points: _Points, clock: EngineClock, lines: tuple[int, ...]
) -> Decomposition:
    sent, ended = request.sent_at_ns, request.ended_at_ns
    if ended is None:
        return Decomposition(
            request.request_id, E2E, None, unavailable=NOT_FINISHED, lines=lines
        )
    if points.first_retained is None or points.finished is None:
        reason = NO_RETAINED_TOKEN if points.first_retained is None else NO_FINISH
        return Decomposition(
            request.request_id, E2E, _e2e(request), unavailable=reason, lines=lines
        )
    parts = [
        _cross(clock, "send_to_ingress", sent, points.admitted_wall, outbound=True),
        *_ingress(points),
        _span("prefill", points.first_scheduled, points.first_retained),
        _span("decode", points.first_retained, points.finished),
        _cross(clock, "final_delivery", ended, points.finished_wall, outbound=False),
    ]
    return Decomposition(
        request.request_id, E2E, _e2e(request), tuple(parts), lines=lines
    )


def _ingress(points: _Points) -> list[Part]:
    if points.enqueued is None:
        return [_span(MERGED_INGRESS, points.admitted, points.first_scheduled)]
    return [
        _span("engine_ingress", points.admitted, points.enqueued),
        _span("scheduler_wait", points.enqueued, points.first_scheduled),
    ]


def _span(name: str, start: int | None, end: int | None) -> Part:
    if start is None or end is None:
        return Part(name, None, "no_stamp")
    return Part(name, (end - start, end - start))


def _cross(
    clock: EngineClock,
    name: str,
    client_ns: int | None,
    engine: Placed,
    *,
    outbound: bool,
) -> Part:
    """A segment between a client read and an engine read."""
    if client_ns is None:
        return Part(name, None, "no_client_stamp")
    reason = clock.comparable(client_ns, engine)
    if reason is not None or engine.interval is None:
        return Part(name, None, reason)
    low, high = engine.interval
    if outbound:  # client first, then the engine
        return Part(name, (low - client_ns, high - client_ns))
    return Part(name, (client_ns - high, client_ns - low))


def _points(view: RunView, execution: Execution, clock: EngineClock) -> _Points:
    metadata = execution.metadata
    first_scheduled = first_retained = finished = None
    first_wall = finished_wall = Placed.unknown_because("no_stamp")
    for _, membership in execution.memberships:
        iteration = view.iterations.get(membership.iteration_ref)
        if iteration is None:
            continue
        step = iteration[1]
        if first_scheduled is None:
            first_scheduled = step.start_ns
        if first_retained is None and _retained(membership):
            first_retained = step.end_ns
            first_wall = _completion_wall(clock, step)
        finish = membership.metadata.get("finish")
        if isinstance(finish, dict) and finished is None:
            finished, finished_wall = _finish_point(clock, step, finish)
    return _Points(
        admitted=execution.event.start_ns,
        admitted_wall=clock.place(
            metadata.get("admitted_wall_ns"),
            metadata.get("admitted_wall_after_ns"),
            execution.event.start_ns,
        ),
        enqueued=_integer(metadata.get("enqueued_mono_ns")),
        first_scheduled=first_scheduled,
        first_retained=first_retained,
        first_retained_wall=first_wall,
        finished=finished,
        finished_wall=finished_wall,
    )


def _finish_point(
    clock: EngineClock, step: Any, finish: dict[str, Any]
) -> tuple[int | None, Placed]:
    """Where a request finished: the completion of the step it was freed in,
    or, freed between steps, the free itself."""
    if finish.get("in_step"):
        return step.end_ns, _completion_wall(clock, step)
    return _integer(finish.get("mono_ns")), clock.place(
        finish.get("wall_ns"), finish.get("wall_after_ns"), finish.get("mono_ns")
    )


def _completion_wall(clock: EngineClock, step: Any) -> Placed:
    return clock.place(
        step.metadata.get("completed_wall_ns"),
        step.metadata.get("completed_wall_after_ns"),
        step.end_ns,
    )


def _retained(membership: MembershipEvent) -> bool:
    """A step that kept an output token for the request: not stale, not a
    finished request's leftover."""
    return (
        membership.metadata.get("outcome") == "kept"
        and (membership.output_tokens or 0) > 0
    )


def _ttft(request: ClientRequest) -> int | None:
    sent, first = request.sent_at_ns, request.first_content_at_ns
    return None if sent is None or first is None else first - sent


def _e2e(request: ClientRequest) -> int | None:
    sent, ended = request.sent_at_ns, request.ended_at_ns
    return None if sent is None or ended is None else ended - sent


def _lines(request: ClientRequest, execution: Execution) -> tuple[int, ...]:
    numbers = [line.number for line in request.lines()]
    numbers.append(execution.line.number)
    numbers.extend(line.number for line, _ in execution.memberships)
    return tuple(sorted(set(numbers)))


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "E2E",
    "E2E_SEGMENTS",
    "MERGED_INGRESS",
    "TTFT",
    "TTFT_SEGMENTS",
    "Decomposition",
    "Part",
    "decompose",
]
