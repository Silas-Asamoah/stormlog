"""Neighbor traffic and its actuation check, against the fake vLLM engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.neighbor import Neighbor, NeighborShape, judge

FAST = FakeEngineConfig(step_seconds=0.001, decode_token_seconds=0.0001)


def _neighbor(engine: FakeEngine, tmp_path: Path, shape: NeighborShape) -> Neighbor:
    return Neighbor(
        name="1",
        shape=shape,
        endpoint=engine.endpoint,
        model=engine.config.model,
        duration_seconds=1.0,
        output=tmp_path / "neighbor-1.jsonl",
    )


def test_an_open_loop_neighbor_reaches_its_planned_rate(tmp_path: Path) -> None:
    with FakeEngine(FAST) as engine:
        neighbor = _neighbor(
            engine, tmp_path, NeighborShape(8, 4, rate_per_second=40.0)
        )
        neighbor.start()
        assert neighbor.join(timeout=30)
        externals = {request.external_id for request in engine.engine.finished}
    actuation = neighbor.actuation()
    assert actuation.ok, actuation.problems
    assert actuation.sent == 40
    assert actuation.first_send_ns is not None
    assert all(e.startswith(neighbor.external_prefix) for e in externals)


def test_a_closed_loop_neighbor_keeps_its_workers_busy(tmp_path: Path) -> None:
    with FakeEngine(FAST) as engine:
        neighbor = _neighbor(engine, tmp_path, NeighborShape(8, 4, concurrency=3))
        neighbor.start()
        assert neighbor.join(timeout=30)
    actuation = neighbor.actuation()
    assert actuation.ok, actuation.problems
    assert actuation.sent >= 3


S = 1_000_000_000


def _request(
    status: str = "ok",
    held: bool = False,
    at: int = 0,
    *,
    lag_ms: float = 1.0,
    prompt: int = 12,
    output: int = 4,
    ended: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "held_for_slot": held,
        "started_at_ns": at,
        "ended_at_ns": at + S // 20 if ended is None else ended,
        "dispatch_lag_ms": lag_ms,
        "prompt_tokens": prompt,
        "prompt_token_source": "server_usage",
        "output_tokens": output,
        "output_token_source": "server_usage",
    }


def test_actuation_names_every_shortfall() -> None:
    shape = NeighborShape(8, 4, rate_per_second=10.0)
    on_plan = [_request(at=index) for index in range(10)]
    assert judge(on_plan, shape, 1.0).ok
    slow = judge(on_plan[:9], shape, 1.0)
    assert slow.problems == ("rate 9/s against 10/s planned",)
    held = judge(on_plan[:9] + [_request(held=True)], shape, 1.0)
    assert held.problems == ("1 arrivals held for a slot",)
    failed = judge(on_plan[:9] + [_request("timeout")], shape, 1.0)
    assert failed.problems == ("1 requests failed",)
    broken = judge(on_plan, shape, 1.0, [RuntimeError("boom")])
    assert broken.problems == ("neighbor raised RuntimeError('boom')",)
    assert judge(on_plan, shape, 1.0).first_send_ns == 0


def test_a_shape_is_open_or_closed() -> None:
    with pytest.raises(ValueError):
        NeighborShape(8, 4)
    with pytest.raises(ValueError):
        NeighborShape(8, 4, rate_per_second=1.0, concurrency=2)


def test_an_open_loop_neighbor_sent_late_or_short_does_not_actuate() -> None:
    # 450 arrivals at 10/s over 45 s, 400 of them dispatched in the last 5 s
    # (the harness stalled), and every prompt a tenth of its dose: counting
    # the records alone called it actuated.
    shape = NeighborShape(input_tokens=2048, output_tokens=16, rate_per_second=10.0)
    requests = []
    for index in range(450):
        intended = int(index * 0.1 * S)
        sent = int((40 + index * 0.0125) * S) if index < 400 else intended
        requests.append(
            _request(at=sent, lag_ms=(sent - intended) / 1e6, prompt=205, output=16)
        )
    actuation = judge(requests, shape, 45.0)
    assert not actuation.ok
    assert any("dispatch lag" in problem for problem in actuation.problems)
    assert any("5 s window" in problem for problem in actuation.problems)
    assert any("prompt tokens" in problem for problem in actuation.problems)


def test_a_closed_loop_neighbor_with_idle_workers_does_not_actuate() -> None:
    # Four workers planned, but only one request in flight at a time.
    shape = NeighborShape(8, 4, concurrency=4)
    serial = [_request(at=i * S // 10, ended=(i + 1) * S // 10) for i in range(10)]
    actuation = judge(serial, shape, 1.0)
    assert actuation.problems == ("1 of 4 workers busy on average",)
    busy = [
        _request(at=i * S // 10, ended=(i + 1) * S // 10)
        for i in range(10)
        for _worker in range(4)
    ]
    assert judge(busy, shape, 1.0).ok


def test_a_short_output_does_not_actuate() -> None:
    shape = NeighborShape(8, 64, rate_per_second=10.0)
    short = [_request(at=i * S // 10, output=6) for i in range(10)]
    assert judge(short, shape, 1.0).problems == ("output tokens 0.094x the dose",)


def test_a_closed_loops_busy_share_ignores_the_tail_after_its_window() -> None:
    # Four workers busy for the whole 1 s window, then finishing one by one
    # over 0.5 s more: the tail is not idle time in the window.
    shape = NeighborShape(8, 4, concurrency=4)
    busy = [
        _request(at=i * S // 10, ended=(i + 1) * S // 10)
        for i in range(10)
        for _worker in range(4)
    ]
    tail = [_request(at=S, ended=S + worker * S // 8) for worker in range(4)]
    assert judge(busy + tail, shape, 1.0).ok
