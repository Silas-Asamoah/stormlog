"""Per-case cohorts, populations, intervals and segments."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.populations import (
    MeasuredInterval,
    Segment,
    case_populations,
    count_population,
    rate,
)

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


def test_a_replay_without_an_endpoint_falls_back_to_the_measured_span() -> None:
    records = [_request(i) for i in range(10)]
    window = _window(arrival_mode="replay", scheduled_endpoint_offset_ns=None)
    replay = _workload({"mode": "replay", "trace": {"arrivals": 10}})
    intervals = _case([*records, window, replay]).intervals
    assert intervals.rate is not None and intervals.rate.kind == "measured_span"
    assert intervals.rate_reason == "scheduled_endpoint_unknown"
    assert intervals.scheduled_window is None


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
    assert population.server_evidence_coverage == pytest.approx(0.5)


def test_rates_need_an_interval_with_length() -> None:
    assert rate(5, MeasuredInterval("segment", 0, SECOND, "arrival_cohort")) == 5.0
    assert (
        rate(5, MeasuredInterval("segment", SECOND, SECOND, "arrival_cohort")) is None
    )
    assert rate(5, None) is None


def test_count_population_takes_any_iterable() -> None:
    population = count_population(r for r in [_request(0), _request(1, "dropped")])
    assert (population.offered, population.sent, population.successful) == (2, 1, 1)
