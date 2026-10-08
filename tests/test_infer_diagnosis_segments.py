"""Placing engine times on the client's clock, and the two decompositions."""

from __future__ import annotations

from pathlib import Path

import pytest

from stormlog.infer.diagnosis_clocks import (
    UNKNOWN_DISCONTINUITY,
    UNKNOWN_LEGACY,
    UNKNOWN_OTHER_HOST,
    EngineClock,
    continuity_segments,
)
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import RunView, join
from stormlog.infer.diagnosis_segments import (
    E2E_SEGMENTS,
    MERGED_INGRESS,
    TTFT_SEGMENTS,
    Decomposition,
    decompose,
)
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run, poisson_free
from tests.vllm_execution_helpers import HOST, SECOND


def _view(
    tmp_path: Path,
    requests: list[SimRequest],
    engine: Engine,
    client_host: str = HOST,
) -> RunView:
    return join(
        read_input(build_run(tmp_path, requests, engine, client_host=client_host))
    )


def _decomposed(view: RunView, request_id: str) -> tuple[Decomposition, Decomposition]:
    producer = next(iter(view.executions.values())).producer
    return decompose(
        view, view.client[request_id], {producer: EngineClock(view, producer)}
    )


def test_segments_tile_the_client_s_interval_to_within_the_brackets(
    tmp_path: Path,
) -> None:
    requests = poisson_free(6, 10 * SECOND, 2 * MS, output=3)
    view = _view(tmp_path, requests, Engine(max_num_seqs=2, step_ns=10 * MS))

    ttft, e2e = _decomposed(view, "r5")

    assert [part.name for part in ttft.parts] == list(TTFT_SEGMENTS)
    assert [part.name for part in e2e.parts] == list(E2E_SEGMENTS)
    wait = ttft.part("scheduler_wait")
    assert wait is not None and wait.interval is not None and wait.interval[0] > 0
    assert ttft.part("engine_ingress").interval == (100_000, 100_000)  # type: ignore[union-attr]
    send = ttft.part("send_to_ingress")
    assert send is not None and send.interval == (1 * MS, 1 * MS + 800)
    for decomposition in (ttft, e2e):
        residual = decomposition.residual
        assert residual is not None and residual[0] <= 0 <= residual[1]
        assert residual[1] - residual[0] == 1_600  # two brackets of 800 ns


def test_a_hook_without_second_reads_leaves_cross_clock_segments_unknown(
    tmp_path: Path,
) -> None:
    view = _view(tmp_path, poisson_free(2, 10 * SECOND, MS), Engine(bracketed=False))

    ttft, _ = _decomposed(view, "r1")

    assert ttft.part("send_to_ingress").unknown == UNKNOWN_LEGACY  # type: ignore[union-attr]
    assert ttft.part("first_token_delivery").unknown == UNKNOWN_LEGACY  # type: ignore[union-attr]
    assert ttft.part("prefill").interval is not None  # type: ignore[union-attr]
    assert ttft.residual is None


def test_an_engine_on_another_host_has_no_comparable_clock(tmp_path: Path) -> None:
    view = _view(
        tmp_path, poisson_free(2, 10 * SECOND, MS), Engine(), client_host="client-1"
    )

    ttft, _ = _decomposed(view, "r1")

    assert ttft.part("send_to_ingress").unknown == UNKNOWN_OTHER_HOST  # type: ignore[union-attr]


def test_a_wall_jump_between_two_reads_withholds_the_pair(tmp_path: Path) -> None:
    # The host's clock steps by 5 ms 0.5 ms after r1's send, before the
    # engine admits it.
    requests = [
        SimRequest("r0", 10 * SECOND, output=2),
        SimRequest("r1", 10 * SECOND + 200 * MS, output=2),
        SimRequest("r2", 10 * SECOND + 400 * MS, output=2),
    ]
    engine = Engine(wall_jump=(10 * SECOND + 200 * MS + 500_000, 5 * MS))
    view = _view(tmp_path, requests, engine)
    producer = next(iter(view.executions.values())).producer
    clock = EngineClock(view, producer)

    assert len(clock.segments) == 2
    across = _decomposed(view, "r1")[0].part("send_to_ingress")
    assert across is not None and across.unknown == UNKNOWN_DISCONTINUITY
    for request_id in ("r0", "r2"):  # each pair on one side of the jump
        part = _decomposed(view, request_id)[0].part("send_to_ingress")
        assert part is not None and part.interval == (1 * MS, 1 * MS + 800)


def test_continuity_allows_slew_and_splits_at_a_jump() -> None:
    # 1 s apart, an offset drift of 50 us is slew (100 ppm allows 100 us).
    slewing = [(0, 1000), (SECOND, SECOND + 1000 + 50_000)]
    assert len(continuity_segments(slewing)) == 1
    jumped = [*slewing, (2 * SECOND, 2 * SECOND + 1000 + 50_000 + 5 * MS)]
    segments = continuity_segments(jumped)
    assert [s.samples for s in segments] == [2, 1]
    assert segments[0].max_gap_ns == SECOND


def test_an_older_log_merges_ingress_and_the_queue_wait(tmp_path: Path) -> None:
    requests = poisson_free(2, 10 * SECOND, MS)
    path = build_run(tmp_path, requests, Engine())
    text = path.read_text()
    # A hook from before enqueued records: the import left these null.
    path.write_text(
        text.replace('"enqueued_mono_ns": ', '"enqueued_mono_ns": null, "x": ')
    )
    view = join(read_input(path))

    ttft, _ = _decomposed(view, "r1")

    names = [part.name for part in ttft.parts]
    assert MERGED_INGRESS in names and "scheduler_wait" not in names


@pytest.mark.parametrize("output", [1, 3])
def test_a_request_with_no_execution_is_not_decomposed(
    tmp_path: Path, output: int
) -> None:
    view = _view(tmp_path, poisson_free(1, 10 * SECOND, MS, output=output), Engine())
    view.executions.clear()

    ttft, e2e = _decomposed_without_engine(view, "r0")

    assert ttft.unavailable == e2e.unavailable == "no_engine_execution"
    assert ttft.total_ns is not None


def _decomposed_without_engine(
    view: RunView, request_id: str
) -> tuple[Decomposition, Decomposition]:
    return decompose(view, view.client[request_id], {})
