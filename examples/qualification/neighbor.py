"""Neighbor traffic: another tenant's load, injected with its actuation checked.

F1, F2, F3 and their workload twins load the server with a neighbor's
requests while the victim runs. A neighbor is an ``infer profile`` run of its
own, with its own run ID (so its requests are told apart from the victim's
by their ``X-Request-Id``), a high in-flight limit, and its own artifact under
``truth/``. Open-loop neighbors arrive at a fixed rate, so the plan is a
schedule, not a distribution.

Actuation is checked from that artifact (#221 A.3), on what was actually
sent, not on how many records there are:
- an open-loop neighbor reaches its planned rate within 5% overall and in
  every 5 s window of actual send times, holds no arrival for a slot, and
  dispatches with a p95 lag under 0.25 s;
- a closed-loop neighbor keeps its workers busy: on average at least 90% of
  them have a request in flight;
- the server's own token counts match the dose: the median prompt within a
  factor of 2 of it (the client counts words), the median output at least
  90% of it (the neighbor ignores EOS).

Any failed request is reported.
"""

from __future__ import annotations

import bisect
import json
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from stormlog.infer.arrivals import CLOSED, FIXED_RATE
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler

RATE_TOLERANCE = 0.05
MAX_IN_FLIGHT = 512
WINDOW_SECONDS = 5.0
MAX_DISPATCH_LAG_SECONDS = 0.25
PROMPT_RANGE = (0.5, 2.0)
OUTPUT_FLOOR = 0.9
BUSY_SHARE = 0.9
SECOND = 1_000_000_000


@dataclass(frozen=True)
class NeighborShape:
    """What a neighbor sends: a fixed rate (open loop) or a number of
    concurrent workers (closed loop), of one token shape."""

    input_tokens: int
    output_tokens: int
    rate_per_second: float | None = None
    concurrency: int | None = None
    prompt_mode: str = "unique"
    shared_prefix_ratio: float | None = None
    prefix_groups: int | None = None

    def __post_init__(self) -> None:
        if (self.rate_per_second is None) == (self.concurrency is None):
            raise ValueError("a neighbor has a rate or a concurrency, not both")

    @property
    def open_loop(self) -> bool:
        return self.rate_per_second is not None


@dataclass(frozen=True)
class Actuation:
    """Whether the neighbor's load happened as planned."""

    ok: bool
    sent: int
    failed: int
    held: int
    first_send_ns: int | None
    achieved_rate: float | None
    planned_rate: float | None
    problems: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sent": self.sent,
            "failed": self.failed,
            "held": self.held,
            "first_send_ns": self.first_send_ns,
            "achieved_rate": self.achieved_rate,
            "planned_rate": self.planned_rate,
            "problems": list(self.problems),
        }


@dataclass
class Neighbor:
    """One neighbor, run on a thread of the harness."""

    name: str
    shape: NeighborShape
    endpoint: str
    model: str
    duration_seconds: float
    output: Path
    seed: int = 0
    _thread: threading.Thread | None = None
    _error: list[BaseException] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return f"neighbor-{self.name}"

    @property
    def external_prefix(self) -> str:
        """The prefix vLLM's external request IDs carry for this neighbor."""
        return f"chatcmpl-stormlog-{self.run_id}-"

    def config(self) -> ProfileConfig:
        shape = self.shape
        return ProfileConfig(
            endpoint=self.endpoint,
            model=self.model,
            concurrency=(shape.concurrency or 1,),
            input_tokens=(shape.input_tokens,),
            output_tokens=(shape.output_tokens,),
            output_path=str(self.output),
            duration_seconds=self.duration_seconds,
            request_count=None,
            seed=self.seed,
            tokenizer="none",
            system_sampler="none",
            run_id=self.run_id,
            arrival_mode=FIXED_RATE if shape.open_loop else CLOSED,
            rates=(shape.rate_per_second,) if shape.rate_per_second else (),
            max_in_flight=MAX_IN_FLIGHT,
            prompt_mode=shape.prompt_mode,
            shared_prefix_ratio=shape.shared_prefix_ratio,
            prefix_groups=shape.prefix_groups,
            extra_body={"ignore_eos": True},
        )

    def start(self) -> None:
        """Start sending, on a thread."""
        profiler = InferenceProfiler(self.config())

        def run() -> None:
            try:
                profiler.run()
            except BaseException as error:  # reported by actuation()
                self._error.append(error)

        self._thread = threading.Thread(target=run, name=self.run_id, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the neighbor to finish; True once it has."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def actuation(self) -> Actuation:
        """Judge the run from its artifact."""
        requests = _requests(self.output) if self.output.exists() else []
        return judge(requests, self.shape, self.duration_seconds, self._error)


def _requests(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [
        record
        for record in records
        if record.get("event_type") == "infer.request"
        and record.get("phase") == "measured"
    ]


def judge(
    requests: list[dict[str, Any]],
    shape: NeighborShape,
    duration_seconds: float,
    errors: Sequence[BaseException] = (),
) -> Actuation:
    """A.3's actuation rule for a neighbor, from its request records."""
    tally = _Tally.of(requests)
    planned = shape.rate_per_second
    achieved = tally.sent / duration_seconds if duration_seconds > 0 else None
    sent = [r for r in requests if r.get("status") != "dropped"]
    problems = [f"neighbor raised {error!r}" for error in errors]
    if shape.open_loop:
        problems += _open_loop_problems(achieved, planned, tally.held)
        problems += _window_problems(sent, planned or 0.0, duration_seconds)
        problems += _lag_problems(sent)
    else:
        problems += _closed_loop_problems(sent, shape.concurrency or 0)
    problems += _dose_problems(sent, shape)
    if tally.failed:
        problems.append(f"{tally.failed} requests failed")
    return Actuation(
        ok=not problems,
        sent=tally.sent,
        failed=tally.failed,
        held=tally.held,
        first_send_ns=tally.first_send_ns,
        achieved_rate=achieved,
        planned_rate=planned,
        problems=tuple(problems),
    )


@dataclass(frozen=True)
class _Tally:
    sent: int
    held: int
    failed: int
    first_send_ns: int | None

    @classmethod
    def of(cls, requests: list[dict[str, Any]]) -> _Tally:
        sent = [r for r in requests if r.get("status") != "dropped"]
        return cls(
            sent=len(sent),
            held=sum(1 for r in requests if r.get("held_for_slot")),
            failed=sum(1 for r in sent if r.get("status") not in ("ok", "cancelled")),
            first_send_ns=min((int(r["started_at_ns"]) for r in sent), default=None),
        )


def _open_loop_problems(
    achieved: float | None, planned: float | None, held: int
) -> list[str]:
    problems = []
    if achieved is None or planned is None:
        return ["no rate to compare"]
    if abs(achieved - planned) > RATE_TOLERANCE * planned:
        problems.append(f"rate {achieved:.3g}/s against {planned:.3g}/s planned")
    if held:
        problems.append(f"{held} arrivals held for a slot")
    return problems


def _window_problems(
    sent: list[dict[str, Any]], planned: float, duration_seconds: float
) -> list[str]:
    """Sends in each full 5 s window of actual send times, from the first,
    within 5% (or one request) of the plan: a schedule sent late, in a
    burst, still has the right count overall."""
    if not sent or duration_seconds < 2 * WINDOW_SECONDS:
        return []
    starts = sorted(int(r["started_at_ns"]) for r in sent)
    window_ns = int(WINDOW_SECONDS * SECOND)
    expected = planned * WINDOW_SECONDS
    allowed = max(1.0, RATE_TOLERANCE * expected)
    for index in range(int(duration_seconds // WINDOW_SECONDS)):
        low = starts[0] + index * window_ns
        count = bisect.bisect_left(starts, low + window_ns) - bisect.bisect_left(
            starts, low
        )
        if abs(count - expected) > allowed:
            return [
                f"{count} sends in 5 s window {index} against {expected:.3g} planned"
            ]
    return []


def _lag_problems(sent: list[dict[str, Any]]) -> list[str]:
    lags = sorted(
        float(r["dispatch_lag_ms"])
        for r in sent
        if r.get("dispatch_lag_ms") is not None
    )
    if not lags:
        return []
    p95 = lags[min(len(lags) - 1, int(0.95 * len(lags)))] / 1000
    if p95 > MAX_DISPATCH_LAG_SECONDS:
        return [f"dispatch lag p95 {p95:.3g} s, above {MAX_DISPATCH_LAG_SECONDS} s"]
    return []


def _closed_loop_problems(sent: list[dict[str, Any]], workers: int) -> list[str]:
    """Every worker busy: the time-weighted number in flight over the run."""
    if len(sent) < workers:
        return ["fewer requests than workers"]
    spans = [
        (int(r["started_at_ns"]), int(r["ended_at_ns"]))
        for r in sent
        if r.get("ended_at_ns") is not None
    ]
    if not spans:
        return ["no request ended"]
    first, last = min(s for s, _e in spans), max(e for _s, e in spans)
    busy = sum(end - start for start, end in spans) / max(1, last - first)
    if busy < BUSY_SHARE * workers:
        return [f"{busy:.2g} of {workers} workers busy on average"]
    return []


def _dose_problems(sent: list[dict[str, Any]], shape: NeighborShape) -> list[str]:
    """The server's own counts against the dose, where the server gave them."""
    problems = []
    prompt = _server_median(sent, "prompt")
    if prompt is not None:
        ratio = prompt / shape.input_tokens
        if not PROMPT_RANGE[0] <= ratio <= PROMPT_RANGE[1]:
            problems.append(f"prompt tokens {ratio:.2g}x the dose")
    output = _server_median(sent, "output")
    if output is not None and output / shape.output_tokens < OUTPUT_FLOOR:
        problems.append(f"output tokens {output / shape.output_tokens:.2g}x the dose")
    return problems


def _server_median(sent: list[dict[str, Any]], which: str) -> float | None:
    counts = [
        float(r[f"{which}_tokens"])
        for r in sent
        if r.get(f"{which}_token_source") == "server_usage"
        and r.get(f"{which}_tokens") is not None
    ]
    return statistics.median(counts) if counts else None


__all__ = [
    "MAX_IN_FLIGHT",
    "RATE_TOLERANCE",
    "Actuation",
    "Neighbor",
    "NeighborShape",
    "judge",
]
