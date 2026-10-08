"""Choosing what a diagnosis explains: windows, references and incidents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_inputs import Line, read_input
from stormlog.infer.diagnosis_join import ClientRequest, RunView, join
from stormlog.infer.diagnosis_selection import (
    E2E,
    FAILED,
    INSUFFICIENT_REFERENCE,
    LEGACY_NO_DISPATCH,
    TTFT,
    SelectionOptions,
    _Censoring,
    fisher_one_sided,
    nearest_rank,
    select,
)
from tests.diagnosis_scenarios import MS, Engine, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET


def _burst_run(tmp_path: Path, *, burst: int = 600) -> RunView:
    """Two minutes at 2 requests/s, then 3 s at 200 requests/s."""
    calm = poisson_free(240, 10 * SECOND, 500 * MS, prefix="a")
    heavy = poisson_free(burst, 130 * SECOND, 5 * MS, prefix="b")
    return join(read_input(build_run(tmp_path, calm + heavy, Engine(max_num_seqs=4))))


@pytest.fixture(scope="module")
def burst_view(tmp_path_factory: pytest.TempPathFactory) -> RunView:
    return _burst_run(tmp_path_factory.mktemp("burst"))


def test_fisher_s_exact_test_matches_known_tables() -> None:
    # Fable's cases: 7 of 20 against 5 of 50 is not enough at 0.01; 8 is.
    assert fisher_one_sided(7, 13, 5, 45) == pytest.approx(0.0184967, rel=1e-4)
    assert fisher_one_sided(8, 12, 5, 45) == pytest.approx(0.0065187, rel=1e-4)
    assert fisher_one_sided(3, 17, 0, 114) == pytest.approx(0.0029075, rel=1e-4)
    assert fisher_one_sided(0, 20, 5, 45) == 1.0


def test_the_p90_is_a_nearest_rank_order_statistic() -> None:
    assert nearest_rank(list(range(1, 11)), 0.9) == 9
    assert nearest_rank(list(range(1, 115)), 0.9) == 103
    assert nearest_rank([], 0.9) == 0


def test_a_sustained_burst_is_one_incident_detected_on_its_second_window(
    burst_view: RunView,
) -> None:
    selection = select(burst_view)

    calm = [
        w
        for w in selection.windows
        if w.start_ns < 130 * SECOND + 1_790_000_000_000_000_000
    ]
    assert all(len(w.requests) == 20 for w in calm)
    assert {w.status for w in calm[:5]} == {INSUFFICIENT_REFERENCE}  # under 114
    assert not any(w.flagged for w in calm)
    (incident,) = selection.subjects
    assert incident.kind == "window" and incident.declared_by is None
    assert len(incident.requests) == 600 and len(incident.reference) == 240
    # 3 s at 200 requests/s: windows of 20, each ending at the next arrival.
    assert len(incident.windows) == 30 and all(w.flagged for w in incident.windows)
    assert incident.first_detectable_ns == incident.windows[1].evaluated_at_ns
    test = incident.windows[0].tests[TTFT]
    assert test.p_value < 0.01 and test.window_above >= 3


def test_an_incident_is_placed_where_it_was_first_seen(tmp_path: Path) -> None:
    """Calm requests 500 ms apart, then a burst 10 s after the last: the
    burst's first window was joined forward from the calm, so it begins 10 s
    before the burst. The finding's window starts at the onset instead, the
    first moment a flagged request had run past the reference's p90, with
    that window's span as its resolution."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, 90 * SECOND, 5 * MS, prefix="b")
    view = join(read_input(build_run(tmp_path, calm + burst, Engine(max_num_seqs=4))))
    selection = select(view)
    (incident,) = selection.subjects
    onset = 90 * SECOND + WALL_OFFSET

    assert incident.start_ns == onset - 10 * SECOND
    assert incident.onset_ns is not None
    assert onset <= incident.onset_ns < onset + 100 * MS
    window = Context(view, selection).window(incident)
    assert window is not None and window["start_ns"] == incident.onset_ns
    first = incident.windows[0]
    assert window["resolution_ns"] == first.end_ns - first.start_ns
    # #221's match: start >= effect onset - (resolution + uncertainty).
    assert window["start_ns"] >= onset - window["resolution_ns"]


def test_one_bad_window_alone_is_not_an_incident(tmp_path: Path) -> None:
    # 30 requests: a window of 20, then 10, too few to be tested.
    view = _burst_run(tmp_path, burst=30)

    selection = select(view)

    assert sum(w.flagged for w in selection.windows) == 1
    assert selection.subjects == []


def test_a_burst_inside_one_base_window_is_still_an_incident(tmp_path: Path) -> None:
    """300 requests 1 ms apart, as a batch job submits them: they arrive in
    one base window, but every 20 make a window ending at the next arrival,
    so the sustain rule sees the burst."""
    calm = poisson_free(240, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, 130 * SECOND, MS, prefix="b")
    view = join(read_input(build_run(tmp_path, calm + burst, Engine(max_num_seqs=4))))

    (incident,) = select(view).subjects

    assert len(incident.requests) == 300 and len(incident.windows) == 15
    assert incident.first_detectable_ns == incident.windows[1].evaluated_at_ns


def test_without_dispatch_records_no_detection_time_is_given(tmp_path: Path) -> None:
    path = build_run(
        tmp_path,
        poisson_free(240, 10 * SECOND, 500 * MS, prefix="a")
        + poisson_free(600, 130 * SECOND, 5 * MS, prefix="b"),
        Engine(max_num_seqs=4),
    )
    lines = path.read_text().splitlines()
    kept = [
        line
        for line in lines
        if '"infer.dispatch"' not in line and '"infer.first_content"' not in line
    ]
    path.write_text("\n".join(kept) + "\n")

    (incident,) = select(join(read_input(path))).subjects

    assert incident.first_detectable_ns is None
    assert incident.detection_unavailable == LEGACY_NO_DISPATCH


def test_a_declared_window_is_compared_with_earlier_unflagged_requests(
    burst_view: RunView,
) -> None:
    start = 130 * SECOND + 1_790_000_000_000_000_000

    selection = select(burst_view, SelectionOptions(windows=((start, start + SECOND),)))

    (subject,) = selection.subjects
    assert subject.declared_by == "caller" and subject.incident
    assert len(subject.requests) == 200 and len(subject.reference) == 240


# ------------------------------------------------------------ censoring


def _line(number: int, **raw: Any) -> Line:
    return Line(number, "0" * 64, {"schema_version": 1, **raw}, None, None)


def _request(*, first_content: bool, ended: bool = True) -> ClientRequest:
    """Astra's case: sent at 0 s, first content at 0.05 s, done at 10 s."""
    common = {"request_id": "r", "case_id": "c", "phase": "measured"}
    request = ClientRequest(
        "r", dispatch=_line(0, event_type="infer.dispatch", started_at_ns=0, **common)
    )
    if first_content:
        request.first_content = _line(
            1, event_type="infer.first_content", first_content_at_ns=50 * MS, **common
        )
    if ended:
        request.terminal = _line(
            2,
            event_type="infer.request",
            started_at_ns=0,
            ended_at_ns=10 * SECOND,
            status="ok",
            ttft_ms=50.0,
            **common,
        )
    return request


def _censoring(*requests: ClientRequest) -> _Censoring:
    view = RunView(source=None)  # type: ignore[arg-type]
    view.client = {f"r{i}": r for i, r in enumerate(requests)}
    return _Censoring(view)


def test_ttft_is_known_from_its_first_content_record_before_the_end() -> None:
    request = _request(first_content=True)

    assert _censoring(request).value(request, TTFT, 2 * SECOND) == (50 * MS, True)
    # The end-to-end is still running at 2 s: censored at its elapsed time.
    assert _censoring(request).value(request, E2E, 2 * SECOND) == (2 * SECOND, False)


def test_without_first_content_records_ttft_waits_for_the_end() -> None:
    request = _request(first_content=False)

    assert _censoring(request).value(request, TTFT, 2 * SECOND) is None
    assert _censoring(request).value(request, TTFT, 11 * SECOND) == (50 * MS, True)


def test_a_failed_request_is_beyond_every_threshold() -> None:
    request = _request(first_content=True)
    assert request.terminal is not None and request.terminal.raw is not None
    request.terminal.raw["status"] = "timeout"

    assert _censoring(request).value(request, E2E, 11 * SECOND) == (FAILED, True)
    assert json.dumps(FAILED)  # an integer: reports can carry it
