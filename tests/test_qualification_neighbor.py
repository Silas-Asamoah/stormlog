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


def _request(status: str = "ok", held: bool = False, at: int = 0) -> dict[str, Any]:
    return {"status": status, "held_for_slot": held, "started_at_ns": at}


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
