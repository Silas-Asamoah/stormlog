"""Per-case cohorts, populations, intervals and segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.analysis import analyze_inference_events, format_analysis_text
from stormlog.infer.populations import (
    MeasuredInterval,
    Segment,
    case_populations,
    count_population,
    goodput,
    rate,
)
from stormlog.infer.slo import parse_slo_flags
from stormlog.infer.vllm_analysis import JoinedSpans

SECOND = 1_000_000_000
START = 100 * SECOND


def _request(index: int, status: str = "ok", **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_type": "infer.request",
        "phase": "measured",
        "session_id": "s1",
        "case_id": "c1",
        "request_id": f"c1_measured_{index}",
        "x_request_id": f"stormlog-run-c1_measured_{index}",
        "request_index": index,
        "status": status,
        "intended_at_ns": START + index * SECOND // 10,
        "started_at_ns": START + index * SECOND // 10,
        "ended_at_ns": START + index * SECOND // 10 + SECOND // 2,
    }
    record.update(overrides)
    return record


def _window(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_type": "infer.phase_window",
        "phase": "measured",
        "session_id": "s1",
        "case_id": "c1",
        "arrival_mode": "fixed-rate",
        "started_at_ns": START,
        "window_ended_at_ns": START + SECOND,
        "drained_at_ns": START + 2 * SECOND,
        "scheduled_arrivals": 10,
        "scheduled_endpoint_offset_ns": SECOND,
        "abandoned_requests": {"still_running": 0},
    }
    record.update(overrides)
    return record


def _workload(
    arrival: dict[str, Any] | None = None, **measurement: Any
) -> dict[str, Any]:
    return {
        "event_type": "infer.workload",
        "seed": 0,
        "cases": [
            {
                "case_id": "c1",
                "arrival": arrival or {"mode": "fixed-rate", "rate_per_second": 10.0},
            }
        ],
        "measurement": {"request_count": 10, "duration_seconds": None, **measurement},
    }


def _case(records: list[dict[str, Any]], **options: Any) -> Any:
    return case_populations(records, **options)["c1"]


def test_every_status_lands_in_one_population_and_the_identities_hold() -> None:
    statuses = [
        "ok",
        "ok",
        "ok",
        "error",
        "timeout",
        "cancelled",
        "rejected",
        "unreachable",
        "delivery_unknown",
        "dropped",
        "mystery",
    ]
    records = [_request(i, status) for i, status in enumerate(statuses)]
    population = _case([*records, _window(scheduled_arrivals=11)]).population

    assert (population.offered, population.scheduled) == (11, 11)
    assert (population.dropped, population.sent) == (1, 10)
    assert (population.unreachable, population.delivery_unknown) == (1, 1)
    assert (population.rejected, population.accepted) == (1, 7)
    assert (population.successful, population.failed) == (3, 1)
    assert (population.timed_out, population.cancelled) == (1, 1)
    assert population.censored == 2
    assert population.other == {"mystery": 1}
    assert population.offered == population.dropped + population.sent
    assert population.sent == (
        population.unreachable
        + population.delivery_unknown
        + population.rejected
        + population.accepted
    )
    assert population.accepted == (
        population.successful
        + population.failed
        + population.timed_out
        + population.cancelled
        + sum(population.other.values())
    )
    assert population.cohort_valid


def test_a_failed_only_case_has_a_population_and_no_successes() -> None:
    records = [_request(i, "error") for i in range(10)]
    population = _case([*records, _window()]).population
    assert (population.offered, population.accepted, population.successful) == (
        10,
        10,
        0,
    )


def test_a_duplicate_cannot_stand_in_for_a_missing_arrival() -> None:
    records = [_request(i) for i in range(9)] + [_request(3, request_id="again")]
    population = _case([*records, _window()]).population

    assert population.offered == 10  # the count alone looks complete
    assert not population.cohort_valid
    assert "request_index_repeated" in population.issues
    assert "records_missing: 1 of 10" in population.issues


@pytest.mark.parametrize(
    ("change", "issue"),
    [
        ({"request_id": "c1_measured_0"}, "duplicate_request_id: 1"),
        ({"x_request_id": "stormlog-run-c1_measured_0"}, "duplicate_x_request_id: 1"),
        ({"request_index": 12}, "request_index_out_of_range"),
        ({"session_id": "other"}, "session_mismatch"),
        ({"started_at_ns": START - SECOND}, "outside_window: 1"),
        ({"ended_at_ns": START + 9 * SECOND}, "outside_window: 1"),
        ({"started_at_ns": None}, "outside_window: 1"),
    ],
)
def test_cohort_problems_invalidate_the_cohort(
    change: dict[str, Any], issue: str
) -> None:
    records = [_request(i) for i in range(9)] + [_request(9, **change)]
    population = _case([*records, _window()]).population
    assert issue in population.issues
    assert not population.cohort_valid


def test_a_request_may_end_within_a_second_of_the_recorded_drain() -> None:
    # The window record is written after the drain returns, so a request
    # can end just after drained_at_ns; a second later it is outside.
    drained = START + 2 * SECOND
    late = [_request(i) for i in range(9)] + [
        _request(9, ended_at_ns=drained + SECOND // 2)
    ]
    assert _case([*late, _window()]).population.cohort_valid
    outside = [_request(i) for i in range(9)] + [
        _request(9, ended_at_ns=drained + 3 * SECOND // 2)
    ]
    assert "outside_window: 1" in _case([*outside, _window()]).population.issues


def test_stragglers_and_old_records_are_noted_without_invalidating() -> None:
    records = [_request(i) for i in range(10)]
    stragglers = _window(abandoned_requests={"still_running": 2})
    assert _case([*records, stragglers]).population.issues == (
        "abandoned_requests_at_start: 2",
    )
    legacy = [{**r, "request_index": None} for r in records]
    population = _case([*legacy, _window()]).population
    assert population.cohort_valid
    assert population.issues == ("request_index_unrecorded",)


def test_an_open_loop_rate_divides_by_the_schedules_own_window() -> None:
    records = [_request(i) for i in range(10)]
    intervals = _case([*records, _window(), _workload()]).intervals

    assert intervals.rate is not None
    assert intervals.rate.kind == "scheduled_window"
    assert intervals.rate.numerator_cohort == "arrival_cohort"
    assert intervals.rate.seconds == pytest.approx(1.0)
    assert intervals.realized_offered_rate_per_second == pytest.approx(10.0)
    assert intervals.configured_rate_per_second == 10.0
    assert intervals.measured_span is not None
    assert intervals.measured_span.seconds == pytest.approx(2.0)
    assert intervals.drain is not None and intervals.drain.seconds == pytest.approx(1.0)
    assert intervals.dispatch_window is not None
    assert intervals.dispatch_window.seconds == pytest.approx(0.9)


def test_an_old_open_loop_artifact_recomputes_its_endpoint_from_the_workload() -> None:
    records = [_request(i) for i in range(10)]
    window = _window(scheduled_endpoint_offset_ns=None)
    intervals = _case([*records, window, _workload()]).intervals
    assert intervals.rate is not None and intervals.rate.kind == "scheduled_window"
    assert intervals.rate.seconds == pytest.approx(1.0)


def test_a_replay_without_an_endpoint_has_no_rate_interval(tmp_path: Path) -> None:
    # The span to the drain's end would make an open loop's rate depend on
    # when its requests finished, so no rate is given at all.
    records = [_request(i, e2e_latency_ms=500.0) for i in range(10)]
    window = _window(arrival_mode="replay", scheduled_endpoint_offset_ns=None)
    replay = _workload({"mode": "replay", "trace": {"arrivals": 10}})
    intervals = _case([*records, window, replay]).intervals
    assert intervals.rate is None
    assert intervals.rate_reason == "endpoint_undeclared"
    assert intervals.scheduled_window is None
    assert intervals.measured_span is not None

    report = _analyze(tmp_path, [*records, window, replay])
    throughput = report["cases"]["c1"]["throughput"]
    assert throughput["interval_kind"] is None
    assert throughput["requests_per_second"] is None
    assert "no rate interval (endpoint_undeclared)" in format_analysis_text(report)

    evaluation = goodput(records, parse_slo_flags(["e2e:1000"]), intervals.rate)
    assert evaluation.status == "evaluated" and evaluation.met == 10
    assert evaluation.goodput_lower_rps is None


def test_a_closed_loop_rate_divides_by_the_phase_start_to_drain_end() -> None:
    records = [_request(i) for i in range(10)]
    window = _window(arrival_mode="closed", scheduled_arrivals=None)
    intervals = _case([*records, window]).intervals
    assert intervals.rate is not None
    assert (intervals.rate.kind, intervals.rate.numerator_cohort) == (
        "measured_span",
        "all_measured",
    )
    assert intervals.rate.seconds == pytest.approx(2.0)
    assert intervals.scheduled_window is None


def test_without_a_phase_window_the_span_covers_failed_requests_too() -> None:
    records = [_request(i) for i in range(5)] + [
        _request(5, "timeout", ended_at_ns=START + 9 * SECOND)
    ]
    intervals = _case(records).intervals
    assert intervals.rate is not None and intervals.rate.kind == "request_span"
    # A last request that timed out still bounds the interval.
    assert intervals.rate.ended_at_ns == START + 9 * SECOND
    assert intervals.rate_reason == "no_phase_window"


def test_an_interrupted_open_loop_is_not_a_complete_case() -> None:
    # The run stopped after 6 of 10 scheduled arrivals, before its phase
    # window was written. The workload record shows it is no old artifact.
    records = [_request(i) for i in range(5)] + [_request(5, "cancelled")]
    case = _case([*records, _workload()])
    population = case.population
    assert population.scheduled == 10
    assert not population.cohort_valid
    assert "phase_window_missing" in population.issues
    assert "records_missing: 4 of 10" in population.issues
    assert case.intervals.rate is None
    assert case.intervals.rate_reason == "phase_window_missing"


def test_an_interrupted_closed_loop_is_not_a_complete_case() -> None:
    records = [_request(i) for i in range(6)]
    case = _case([*records, _workload({"mode": "closed"})])
    assert case.population.scheduled is None
    assert case.population.issues == ("phase_window_missing",)
    assert not case.population.cohort_valid
    assert case.intervals.rate is None
    assert case.intervals.rate_reason == "phase_window_missing"


@pytest.mark.parametrize("missing", ["started_at_ns", "drained_at_ns"])
def test_a_phase_window_without_its_bounds_gives_no_rate(missing: str) -> None:
    # A missing start used to read as 0 ns: an interval of decades.
    records = [_request(i) for i in range(10)]
    case = _case([*records, _window(**{missing: None}), _workload()])
    assert "phase_window_incomplete" in case.population.issues
    assert not case.population.cohort_valid
    assert case.intervals.rate is None
    assert case.intervals.rate_reason == "phase_window_incomplete"


def test_a_known_schedule_needs_one_record_per_arrival_without_indexes() -> None:
    records = [{**_request(i), "request_index": None} for i in range(11)]
    population = _case([*records, _window()]).population
    assert "offered_differs_from_scheduled: 11 of 10" in population.issues
    assert not population.cohort_valid


def test_segments_count_by_arrival_or_by_overlap() -> None:
    records = [_request(i) for i in range(10)]
    first_half = Segment("first", 0, SECOND // 2)
    arrival = _case([*records, _window()], segments=[first_half])
    overlap = _case([*records, _window()], segments=[first_half], membership="overlap")

    by_arrival = arrival.segments["first"]
    assert by_arrival.population.offered == 5  # arrivals at 0.0 .. 0.4 s
    assert by_arrival.interval.seconds == pytest.approx(0.5)
    assert by_arrival.interval.numerator_cohort == "arrival_cohort"
    # Every request ran for 0.5 s, so each one before 0.5 s overlaps the slice.
    assert overlap.segments["first"].population.offered == 5
    late = _case([*records, _window()], segments=[Segment("late", SECOND, 2 * SECOND)])
    assert late.segments["late"].population.offered == 0
    tail = _case(
        [*records, _window()],
        segments=[Segment("tail", SECOND, 2 * SECOND)],
        membership="overlap",
    )
    assert tail.segments["tail"].population.offered == 5


def test_a_segment_is_clipped_to_the_case_and_uses_send_time_without_intent() -> None:
    records = [_request(i, intended_at_ns=None) for i in range(10)]
    case = _case([*records, _window()], segments=[Segment("all", 0, 60 * SECOND)])
    segment = case.segments["all"]
    assert segment.population.offered == 10
    assert segment.interval.seconds == pytest.approx(2.0)  # clipped to the drain end


@pytest.mark.parametrize(
    ("name", "start", "end"),
    [("Bad Name", 0, 1), ("ok", 5, 5), ("ok", -1, 5)],
)
def test_invalid_segments_are_refused(name: str, start: int, end: int) -> None:
    with pytest.raises(ValueError):
        Segment(name, start, end)


def test_segment_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="more than once"):
        case_populations([], segments=[Segment("a", 0, 1), Segment("a", 1, 2)])


def test_server_admission_is_separate_from_the_clients_view() -> None:
    records = [_request(i) for i in range(4)] + [_request(4, "delivery_unknown")]
    seen = {"stormlog-run-c1_measured_0", "stormlog-run-c1_measured_4"}

    without = _case([*records, _window(scheduled_arrivals=5)]).population
    assert (without.server_admitted, without.server_evidence_coverage) == (None, None)

    population = _case(
        [*records, _window(scheduled_arrivals=5)], server_admitted_ids=seen
    ).population
    assert population.accepted == 4
    # The server saw the request whose delivery the client could not confirm.
    assert population.server_admitted == 2
    # Coverage is over the accepted requests: one of the four was seen.
    assert population.server_evidence_coverage == pytest.approx(0.25)


def test_a_dropped_request_is_never_admitted_even_with_a_known_id() -> None:
    records = [_request(0), _request(1, "dropped")]
    seen = {"stormlog-run-c1_measured_0", "stormlog-run-c1_measured_1"}
    population = _case(
        [*records, _window(scheduled_arrivals=2)], server_admitted_ids=seen
    ).population
    assert population.server_admitted == 1


def test_evidence_coverage_never_exceeds_the_accepted_requests() -> None:
    records = [_request(0)] + [_request(i, "delivery_unknown") for i in range(1, 5)]
    seen = {f"stormlog-run-c1_measured_{i}" for i in range(5)}
    population = _case(
        [*records, _window(scheduled_arrivals=5)], server_admitted_ids=seen
    ).population
    assert (population.accepted, population.server_admitted) == (1, 5)
    assert population.server_evidence_coverage == 1.0


def test_rates_need_an_interval_with_length() -> None:
    assert rate(5, MeasuredInterval("segment", 0, SECOND, "arrival_cohort")) == 5.0
    assert (
        rate(5, MeasuredInterval("segment", SECOND, SECOND, "arrival_cohort")) is None
    )
    assert rate(5, None) is None


def test_count_population_takes_any_iterable() -> None:
    population = count_population(r for r in [_request(0), _request(1, "dropped")])
    assert (population.offered, population.sent, population.successful) == (2, 1, 1)


# --- goodput -------------------------------------------------------------------


def _ok(index: int, ttft: float | None, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "ttft_ms": ttft,
        "e2e_latency_ms": 500.0,
        "output_tokens": 10,
        "output_token_source": "server_usage",
    }
    fields.update(overrides)
    return _request(index, **fields)


ONE_SECOND = MeasuredInterval("scheduled_window", 0, SECOND, "arrival_cohort")


def test_goodput_counts_good_requests_per_second_of_the_interval() -> None:
    requests = [_ok(0, 100.0), _ok(1, 300.0), _ok(2, 150.0), _request(3, "timeout")]
    evaluation = goodput(requests, parse_slo_flags(["ttft:200"]), ONE_SECOND)

    assert evaluation.status == "evaluated"
    assert (evaluation.offered, evaluation.met, evaluation.missed) == (4, 2, 2)
    assert evaluation.attainment_lower == evaluation.attainment_upper == 0.5
    assert evaluation.goodput_lower_rps == 2.0
    assert evaluation.goodput_lower_output_tps == 20.0
    assert evaluation.evidence_coverage == 1.0
    counts = evaluation.per_criterion["client.ttft"]
    assert (counts.passed, counts.failed) == (2, 1)
    record = evaluation.to_record()
    assert (record["format"], record["version"]) == ("stormlog.infer.slo_evaluation", 1)
    assert record["interval"]["kind"] == "scheduled_window"


def test_a_failed_only_case_is_measurable_and_its_goodput_is_zero() -> None:
    requests = [_request(i, "error") for i in range(5)]
    evaluation = goodput(requests, parse_slo_flags(["ttft:200"]), ONE_SECOND)
    assert evaluation.status == "evaluated"
    assert (evaluation.met, evaluation.missed) == (0, 5)
    assert evaluation.goodput_lower_rps == 0.0
    assert evaluation.attainment_lower == 0.0
    assert evaluation.evidence_coverage is None  # nothing succeeded to judge


def test_unknown_outcomes_widen_the_bounds_instead_of_moving_one_figure() -> None:
    requests = [_ok(0, 100.0), _ok(1, 100.0), _ok(2, 100.0), _ok(3, 100.0)]
    spec = parse_slo_flags(["ttft:200", "server.ttft:200"])
    spans = JoinedSpans(
        by_request={
            "stormlog-run-c1_measured_0": {"gen_ai.latency.time_to_first_token": 0.1},
            "stormlog-run-c1_measured_1": {"gen_ai.latency.time_to_first_token": 0.1},
            "stormlog-run-c1_measured_2": {"gen_ai.latency.time_to_first_token": 0.1},
        },
        quarantined={"stormlog-run-c1_measured_3": "conflicting_spans"},
    )
    evaluation = goodput(requests, spec, ONE_SECOND, spans=spans)

    assert (evaluation.met, evaluation.unknown) == (3, 1)
    assert evaluation.attainment_lower == 0.75
    assert evaluation.attainment_upper == 1.0
    assert (evaluation.goodput_lower_rps, evaluation.goodput_upper_rps) == (3.0, 4.0)
    assert evaluation.evidence_coverage == 0.75
    assert evaluation.per_criterion["server.ttft"].unknown == 1


def test_a_criterion_no_successful_request_can_be_judged_is_unmeasurable() -> None:
    not_streamed = [_ok(i, None) for i in range(3)]
    evaluation = goodput(not_streamed, parse_slo_flags(["ttft:200"]), ONE_SECOND)
    assert evaluation.status == "unmeasurable"
    assert evaluation.reason == "client.ttft: no_client_ttft"
    assert evaluation.attainment_lower is None
    assert evaluation.goodput_lower_rps is None
    # Nor does the criterion get marginal bounds of [0, 1]: none, not 0.
    counts = evaluation.per_criterion["client.ttft"]
    assert (counts.attainment_lower, counts.attainment_upper) == (None, None)
    assert counts.unknown == 3


def test_an_aggregate_only_criterion_is_unmeasurable_per_request() -> None:
    evaluation = goodput(
        [_ok(0, 100.0)], parse_slo_flags(["server.itl:50"]), ONE_SECOND
    )
    assert evaluation.status == "unmeasurable"
    assert evaluation.reason == "server.itl: aggregate_only"
    counts = evaluation.per_criterion["server.itl"]
    assert (counts.attainment_lower, counts.attainment_upper) == (None, None)
    # With no successful request nothing is judged at all, and the bounds
    # must still be none, not [0, 0].
    failed = goodput(
        [_ok(0, None, status="error")], parse_slo_flags(["server.itl:50"]), ONE_SECOND
    )
    counts = failed.per_criterion["server.itl"]
    assert (counts.attainment_lower, counts.attainment_upper) == (None, None)


def test_an_evaluation_carries_whether_its_cohort_is_valid() -> None:
    # Ten copies of one record: the figures are over what was recorded, and
    # a consumer reading only the evaluation must see that it is not sound.
    requests = [_ok(0, 100.0) for _ in range(10)]
    population = count_population(
        requests, issues=["duplicate_request_id: 9"], cohort_valid=False
    )
    spec = parse_slo_flags(["ttft:200"])
    evaluation = goodput(requests, spec, ONE_SECOND, cohort=population)
    assert evaluation.cohort_valid is False
    assert evaluation.cohort_issues == ("duplicate_request_id: 9",)
    record = evaluation.to_record()
    assert record["cohort_valid"] is False
    assert record["cohort_issues"] == ["duplicate_request_id: 9"]
    assert goodput(requests, spec, ONE_SECOND).to_record()["cohort_valid"] is None


def test_marginal_attainment_is_judged_per_criterion() -> None:
    requests = [_ok(0, 100.0), _ok(1, 300.0), _ok(2, 100.0, e2e_latency_ms=900.0)]
    spec = parse_slo_flags(["ttft:200", "e2e:800"])
    evaluation = goodput(requests, spec, ONE_SECOND)
    assert evaluation.met == 1  # joint: both criteria
    assert evaluation.per_criterion["client.ttft"].attainment_lower == pytest.approx(
        2 / 3
    )
    assert evaluation.per_criterion["client.e2e"].attainment_lower == pytest.approx(
        2 / 3
    )


def test_a_tpot_with_one_output_token_counts_as_good_and_as_judged() -> None:
    requests = [
        _ok(i, 100.0, output_tokens=1, output_token_source="server_usage")
        for i in range(4)
    ]
    evaluation = goodput(requests, parse_slo_flags(["tpot:1"]), ONE_SECOND)
    counts = evaluation.per_criterion["client.tpot"]
    assert counts.not_applicable == 4
    assert counts.attainment_lower == counts.attainment_upper == 1.0
    assert evaluation.evidence_coverage == 1.0
    assert evaluation.met == 4


def test_goodput_without_an_interval_has_no_rate() -> None:
    evaluation = goodput([_ok(0, 100.0)], parse_slo_flags(["ttft:200"]), None)
    assert evaluation.status == "evaluated"
    assert evaluation.goodput_lower_rps is None
    assert evaluation.attainment_lower == 1.0


# --- in the analysis report -----------------------------------------------------


def _analyze(
    tmp_path: Path, records: list[dict[str, Any]], status: str = "running"
) -> dict[str, Any]:
    path = tmp_path / "infer.jsonl"
    session = {"event_type": "infer.session", "session_id": "s1", "status": status}
    path.write_text(
        "\n".join(json.dumps(record) for record in [session, *records]) + "\n",
        encoding="utf-8",
    )
    return analyze_inference_events(path)


def test_the_report_is_version_2_with_populations_and_intervals(
    tmp_path: Path,
) -> None:
    records = [_request(i, total_tokens=12, output_tokens=4) for i in range(10)]
    report = _analyze(tmp_path, [*records, _window(), _workload()])
    case = report["cases"]["c1"]

    assert report["analysis_version"] == 2
    assert case["population"]["offered"] == 10
    assert case["population"]["cohort_valid"] is True
    assert case["intervals"]["rate"]["kind"] == "scheduled_window"
    throughput = case["throughput"]
    assert "duration_seconds" not in throughput
    assert throughput["interval_kind"] == "scheduled_window"
    assert throughput["numerator_cohort"] == "arrival_cohort"
    assert throughput["interval_seconds"] == pytest.approx(1.0)
    # 10 successful requests per second of the scheduled window, not per
    # second of the span their sends and responses happened to cover.
    assert throughput["requests_per_second"] == pytest.approx(10.0)
    assert throughput["output_tokens_per_second"] == pytest.approx(40.0)
    text = format_analysis_text(report)
    assert "rates per scheduled window of 1.00 s (arrival cohort)" in text


def test_late_timeouts_no_longer_shorten_a_closed_loops_interval(
    tmp_path: Path,
) -> None:
    # The successful requests span 0.9 s, but the case ran until its last
    # request timed out at 9 s. The old denominator was the 0.9 s.
    records = [_request(i) for i in range(9)] + [
        _request(9, "timeout", ended_at_ns=START + 9 * SECOND)
    ]
    window = _window(
        arrival_mode="closed",
        scheduled_arrivals=None,
        drained_at_ns=START + 9 * SECOND,
        window_ended_at_ns=START + SECOND,
    )
    case = _analyze(tmp_path, [*records, window])["cases"]["c1"]
    assert case["throughput"]["interval_kind"] == "measured_span"
    assert case["throughput"]["interval_seconds"] == pytest.approx(9.0)
    assert case["throughput"]["requests_per_second"] == pytest.approx(1.0)


def test_an_invalid_cohort_is_shown_in_the_text_report(tmp_path: Path) -> None:
    records = [_request(i) for i in range(9)] + [_request(3, request_id="again")]
    report = _analyze(tmp_path, [*records, _window()])
    text = format_analysis_text(report)
    assert "cohort invalid:" in text
    assert "request_index_repeated" in text
    assert "records_missing: 1 of 10" in text


def test_a_cases_slo_says_whether_its_cohort_is_valid(tmp_path: Path) -> None:
    records = [_request(i, e2e_latency_ms=100.0) for i in range(9)]
    records.append(_request(3, request_id="again", e2e_latency_ms=100.0))
    path = tmp_path / "infer.jsonl"
    session = {"event_type": "infer.session", "session_id": "s1"}
    path.write_text(
        "\n".join(json.dumps(r) for r in [session, *records, _window()]) + "\n",
        encoding="utf-8",
    )
    report = analyze_inference_events(path, slo=parse_slo_flags(["e2e:200"]))
    slo = report["cases"]["c1"]["slo"]
    assert slo["cohort_valid"] is False
    assert "records_missing: 1 of 10" in slo["cohort_issues"]


def test_the_report_says_how_the_session_ended(tmp_path: Path) -> None:
    records = [_request(i) for i in range(6)]
    ended = {"event_type": "infer.session", "session_id": "s1", "status": "interrupted"}
    report = _analyze(tmp_path, [*records, _workload(), ended])
    assert report["summary"]["session_status"] == "interrupted"
    case = report["cases"]["c1"]
    assert case["population"]["cohort_valid"] is False
    text = format_analysis_text(report)
    assert "Session status: interrupted" in text
    assert "phase_window_missing" in text

    finished = _analyze(
        tmp_path, [*records, _window(scheduled_arrivals=6)], "completed"
    )
    assert finished["summary"]["session_status"] == "completed"
    assert "Session status" not in format_analysis_text(finished)


def test_each_segment_is_reported_on_its_own(tmp_path: Path) -> None:
    records = [_request(i, total_tokens=12, output_tokens=4) for i in range(10)]
    path = tmp_path / "infer.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in [*records, _window(), _workload()])
    )
    report = analyze_inference_events(
        path,
        slo=parse_slo_flags(["e2e:1000"]),
        segments=[Segment("first_half", 0, SECOND // 2)],
    )
    segment = report["cases"]["c1"]["segments"]["first_half"]
    assert segment["population"]["offered"] == 5
    assert segment["intervals"]["rate"]["seconds"] == pytest.approx(0.5)
    assert segment["throughput"]["requests_per_second"] == pytest.approx(10.0)
    assert segment["slo"]["offered"] == 5
    assert "client.e2e" in segment["latency"]["metrics"]
