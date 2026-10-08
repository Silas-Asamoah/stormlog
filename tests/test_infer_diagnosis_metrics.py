"""What vLLM's metrics can say: window observations, and a witness only
from an exporter asserted to be the engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_kv import assess_kv
from stormlog.infer.diagnosis_queue import assess_queue
from stormlog.infer.diagnosis_selection import SelectionOptions, select
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET
from tests.vllm_scrape_helpers import START, exposition, scrape

AT = 90 * SECOND


def _with_scrapes(
    path: Path,
    restart_at: int | None = None,
    *,
    failed: tuple[int, ...] = (),
    labelled_until: int = 12,
    **gauges: Any,
) -> None:
    """Scrapes every second across the window, on the client's clock; from
    ``restart_at`` seconds on, from a restarted exporter. The scrapes at
    ``failed`` seconds fail, and from ``labelled_until`` on the labelled
    families are gone."""
    counters = gauges.pop("counters", None)
    records = []
    for second in range(12):
        restarted = restart_at is not None and second >= restart_at
        text = exposition(
            gauges=gauges.get("gauges"),
            labelled=gauges.get("labelled") if second < labelled_until else None,
            counters={k: v * second for k, v in (counters or {}).items()},
            start=START + (1000.0 if restarted else 0.0),
        )
        record = scrape(
            None if second in failed else text, AT / SECOND + second
        ).to_record()
        records.append(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)


def _client_only(path: Path) -> None:
    kept = [
        line
        for line in path.read_text().splitlines()
        if '"schema_version": 2' not in line or '"infer.artifact"' in line
    ]
    path.write_text("\n".join(kept) + "\n")


def _context(path: Path, *, asserted: bool = False) -> Context:
    view = join(read_input(path))
    start = AT + WALL_OFFSET
    selection = select(view, SelectionOptions(windows=((start, start + 10 * SECOND),)))
    return Context(view, selection, metrics_from_engine=asserted)


def test_without_the_hook_a_queue_is_a_window_observation(tmp_path: Path) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    path = build_run(
        tmp_path, calm + poisson_free(20, AT, 100 * MS, prefix="b"), Engine()
    )
    _client_only(path)
    _with_scrapes(path, gauges={"vllm:num_requests_waiting": 12.0})
    context = _context(path)

    assessment = assess_queue(context, context.subjects()[0])

    (finding,) = assessment.findings
    assert (assessment.status, assessment.reasons) == (
        "partial",
        ["aggregate_only", "no_server_queue_signal"],
    )
    assert finding.claim == "observation" and finding.observations[0].value == 12.0
    assert finding.location["exporter_binding"] == "exporter_scoped"


def test_without_the_hook_preemptions_are_a_window_observation(tmp_path: Path) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    path = build_run(
        tmp_path, calm + poisson_free(20, AT, 100 * MS, prefix="b"), Engine()
    )
    _client_only(path)
    _with_scrapes(path, counters={"vllm:num_preemptions_total": 3.0})
    context = _context(path, asserted=True)

    assessment = assess_kv(context, context.subjects()[0])

    (finding,) = assessment.findings
    assert assessment.reasons == ["aggregate_only", "no_hook_preemption_data"]
    assert finding.location["exporter_binding"] == "asserted"
    assert finding.observations[0].value == 30.0  # 3 a second over the 10 s window


def test_capacity_waits_witness_a_queue_only_from_an_asserted_exporter(
    tmp_path: Path,
) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, AT, 5 * MS, prefix="b")
    # A hello that does not give max_num_seqs or the token budget.
    engine = Engine(
        max_num_seqs=4, config={"max_num_seqs": None, "max_num_batched_tokens": None}
    )
    path = build_run(tmp_path, calm + burst, engine)
    _with_scrapes(
        path,
        gauges={"vllm:num_requests_waiting": 200.0},
        labelled={
            "vllm:num_requests_waiting_by_reason": {
                "reason=capacity": 200.0,
                "reason=deferred": 0.0,
            }
        },
    )

    gates = []
    for asserted in (False, True):
        context = _context(path, asserted=asserted)
        (finding,) = assess_queue(context, context.subjects()[0]).findings
        gates.append(
            (
                finding.gates["capacity_witness"],
                finding.detail["capacity_witness_source"],
            )
        )

    assert gates == [(False, None), (True, "exporter")]


def _capacity_queue(tmp_path: Path, restart_at: int | None, **scrapes: Any) -> Any:
    """The burst, a hello without capacity (so the hook measures nothing),
    and an asserted exporter saying 200 requests wait for capacity."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, AT, 5 * MS, prefix="b")
    engine = Engine(
        max_num_seqs=4, config={"max_num_seqs": None, "max_num_batched_tokens": None}
    )
    path = build_run(tmp_path, calm + burst, engine)
    _with_scrapes(
        path,
        restart_at,
        **scrapes,
        gauges={"vllm:num_requests_waiting": 200.0},
        labelled={
            "vllm:num_requests_waiting_by_reason": {
                "reason=capacity": 200.0,
                "reason=deferred": 0.0,
            }
        },
    )
    context = _context(path, asserted=True)
    (finding,) = assess_queue(context, context.subjects()[0]).findings
    return finding


def test_an_exporter_window_across_a_restart_witnesses_nothing(
    tmp_path: Path,
) -> None:
    """Two exporters' samples are no one gauge: a window the queue signal
    cannot decide on witnesses no capacity either."""
    finding = _capacity_queue(tmp_path, restart_at=4)

    assert not finding.gates["capacity_witness"]
    assert finding.detail["capacity_witness_source"] is None


def test_a_window_the_signal_cannot_decide_witnesses_nothing(tmp_path: Path) -> None:
    """The window's first scrape failed, so the queue signal decides
    nothing, though the capacity gauge's other samples are whole."""
    finding = _capacity_queue(tmp_path, restart_at=None, failed=(0,))

    assert finding.detail["capacity_witness_source"] is None


def test_a_capacity_series_missing_from_some_scrapes_witnesses_nothing(
    tmp_path: Path,
) -> None:
    """The waiting count is whole, so the signal decides, but the capacity
    reason is gone from the later scrapes: no one series to take a median
    of."""
    finding = _capacity_queue(tmp_path, restart_at=None, labelled_until=6)

    assert finding.detail["capacity_witness_source"] is None


def test_requests_that_waited_through_no_step_were_measured(tmp_path: Path) -> None:
    """Bursts of 6 into 8 slots: every request waited only for the next step
    to begin, which the hook measured, so an asserted exporter's capacity
    count does not stand in for it."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    bursts = [
        SimRequest(f"b{b * 6 + j}", AT + b * 300 * MS + j * MS)
        for b in range(100)
        for j in range(6)
    ]
    path = build_run(tmp_path, calm + bursts, Engine(max_num_seqs=8, wake_ns=20_000))
    _with_scrapes(
        path,
        gauges={"vllm:num_requests_waiting": 3.0},
        labelled={"vllm:num_requests_waiting_by_reason": {"reason=capacity": 3.0}},
    )
    context = _context(path, asserted=True)

    (finding,) = assess_queue(context, context.subjects()[0]).findings

    assert finding.metrics["requests_waiting_at_capacity_share"] == 0.0
    assert finding.detail["capacity_witness_source"] is None


def test_an_exporter_never_overrules_the_hook_s_measured_witness(
    tmp_path: Path,
) -> None:
    """Bursts of 10 into 8 slots: the hook measured that most requests
    waited only for the next step, and an asserted exporter's capacity
    count (which also counts KV-bound waits) does not replace that."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    bursts = [
        SimRequest(f"b{b * 10 + j}", AT + b * 300 * MS + j * MS)
        for b in range(100)
        for j in range(10)
    ]
    path = build_run(tmp_path, calm + bursts, Engine(max_num_seqs=8, wake_ns=20_000))
    _with_scrapes(
        path,
        gauges={"vllm:num_requests_waiting": 2.0},
        labelled={"vllm:num_requests_waiting_by_reason": {"reason=capacity": 2.0}},
    )
    context = _context(path, asserted=True)

    (finding,) = assess_queue(context, context.subjects()[0]).findings

    assert finding.metrics["requests_waiting_at_capacity_share"] is not None
    assert not finding.gates["capacity_witness"]
    assert finding.detail["capacity_witness_source"] is None
