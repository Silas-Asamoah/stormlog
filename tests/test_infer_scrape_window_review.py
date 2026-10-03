"""Window edge cases found in review: label changes, ordering, failures,
duplicate stamps, inconsistent histograms, an invalid reference and a window
over several engines."""

from __future__ import annotations

import pytest

from stormlog.infer.diagnosis_signals import SignalConfig, evaluate_signal
from stormlog.infer.scrape_window import (
    REASON_COUNTER_RESET,
    REASON_DUPLICATE_TIME,
    REASON_ENGINE_REQUIRED,
    REASON_HISTOGRAM_INCONSISTENT,
    REASON_OUT_OF_ORDER,
    REASON_SCRAPE_FAILED,
    REASON_SERIES_LABELS_CHANGED,
    check_window,
    counter_window,
    gauge_window,
    histogram_quantile_bounds,
    histogram_share_above,
)
from stormlog.infer.vllm_telemetry import VllmScrapeRecord
from tests.vllm_scrape_helpers import LABELS, START, exposition, scrape, series

PREEMPTIONS = "vllm:num_preemptions_total"
HITS = "vllm:prefix_cache_hits_total"
QUERIES = "vllm:prefix_cache_queries_total"
WAITING = "vllm:num_requests_waiting"
E2E = "vllm:e2e_request_latency_seconds"


def _relabelled(text: str) -> str:
    return text.replace('model_name="m"', 'model_name="other"')


def test_a_counter_whose_labels_change_between_scrapes_is_not_differenced() -> None:
    window = series(
        [
            exposition(counters={PREEMPTIONS: 5.0}),
            _relabelled(exposition(counters={PREEMPTIONS: 7.0})),
        ]
    )
    counter = counter_window(window, PREEMPTIONS)
    assert counter.delta is None
    assert counter.reasons == (REASON_SERIES_LABELS_CHANGED,)
    signal = evaluate_signal("kv_preemption_pressure", window)
    assert signal.exceeds is None


def test_a_gauge_whose_labels_change_is_flagged() -> None:
    window = series(
        [exposition(gauges={WAITING: 2}), _relabelled(exposition(gauges={WAITING: 9}))]
    )
    assert REASON_SERIES_LABELS_CHANGED in gauge_window(window, WAITING).reasons


def test_out_of_order_scrapes_are_named_as_such_by_counter_window() -> None:
    window = [
        scrape(exposition(counters={PREEMPTIONS: 10.0}), 2.0),
        scrape(exposition(counters={PREEMPTIONS: 5.0}), 1.0),
    ]
    assert counter_window(window, PREEMPTIONS).reasons == (REASON_OUT_OF_ORDER,)


def test_two_scrapes_at_one_instant_give_no_window() -> None:
    window = [
        scrape(exposition(counters={PREEMPTIONS: 1.0}), 1.0),
        scrape(exposition(counters={PREEMPTIONS: 3.0}), 1.0),
    ]
    check = check_window(window)
    assert not check.sufficient
    assert REASON_DUPLICATE_TIME in check.reasons
    assert counter_window(window, PREEMPTIONS).reasons == (REASON_DUPLICATE_TIME,)
    assert evaluate_signal("kv_preemption_pressure", window).exceeds is None


def test_a_quick_scrape_inside_a_slow_one_s_interval_is_out_of_order() -> None:
    # The first sampled somewhere in [0 s, 20 s], the second in [1 s, 1.004 s]:
    # the midpoints run backwards, which sampled first is unknown, and the
    # window would last -9 s.
    window = [
        scrape(exposition(counters={PREEMPTIONS: 5.0}), 0.0, duration_ms=20_000),
        scrape(exposition(counters={PREEMPTIONS: 9.0}), 1.0),
    ]
    check = check_window(window)
    assert check.reasons == (REASON_OUT_OF_ORDER,) and not check.sufficient
    assert counter_window(window, PREEMPTIONS).reasons == (REASON_OUT_OF_ORDER,)
    assert evaluate_signal("kv_preemption_pressure", window).exceeds is None


def test_overlapping_scrapes_whose_midpoints_advance_are_in_order() -> None:
    window = [
        scrape(exposition(counters={PREEMPTIONS: 5.0}), 0.0, duration_ms=1_500),
        scrape(exposition(counters={PREEMPTIONS: 9.0}), 1.0),
    ]
    check = check_window(window)
    assert check.sufficient and check.seconds is not None and check.seconds > 0
    assert counter_window(window, PREEMPTIONS).delta == 4.0


def test_an_interior_failed_scrape_does_not_blank_the_window() -> None:
    texts: list[str | None] = [exposition(gauges={WAITING: 9})] * 2
    window = series([*texts, None, *texts])
    check = check_window(window)
    assert check.sufficient and check.failed == 1
    signal = evaluate_signal("queue_saturation", window)
    assert (signal.sufficient, signal.exceeds) == (True, True)
    assert (signal.detail["scrapes"], signal.detail["failed_scrapes"]) == (4, 1)


def test_a_failed_boundary_scrape_still_blanks_the_window() -> None:
    texts: list[str | None] = [exposition(gauges={WAITING: 9})] * 3
    for window in (series([None, *texts]), series([*texts, None])):
        check = check_window(window)
        assert REASON_SCRAPE_FAILED in check.reasons and not check.sufficient


def test_a_counter_delta_across_an_interior_failure_is_valid() -> None:
    window = series(
        [
            exposition(counters={PREEMPTIONS: 1.0}),
            None,
            exposition(counters={PREEMPTIONS: 4.0}),
        ]
    )
    counter = counter_window(window, PREEMPTIONS)
    assert (counter.delta, counter.reasons) == (3.0, ())


def test_a_histogram_whose_count_disagrees_with_its_inf_bucket_is_refused() -> None:
    start = exposition(histograms={E2E: ((("0.1", 1), ("+Inf", 2)), 1.0)})
    end = exposition(histograms={E2E: ((("0.1", 3), ("+Inf", 11)), 9.0)}).replace(
        f"{E2E}_count{{{LABELS}}} 11", f"{E2E}_count{{{LABELS}}} 20"
    )
    share = histogram_share_above(series([start, end]), E2E, 0.1)
    assert share.reasons == (REASON_HISTOGRAM_INCONSISTENT,)
    assert share.lo is None and share.hi is None


def test_a_histogram_whose_labels_change_between_scrapes_is_not_differenced() -> None:
    start = exposition(histograms={E2E: ((("0.1", 1), ("+Inf", 2)), 1.0)})
    end = _relabelled(exposition(histograms={E2E: ((("0.1", 2), ("+Inf", 7)), 5.0)}))
    share = histogram_share_above(series([start, end]), E2E, 0.1)
    assert share.reasons == (REASON_SERIES_LABELS_CHANGED,)
    assert share.count_delta is None and share.lo is None


def test_a_step_that_is_not_a_histogram_is_refused() -> None:
    # Each scrape is a valid histogram and every part grew, but the step's
    # buckets [8, 1, 1] cannot hold one new observation: shares would be -7.
    start = exposition(histograms={E2E: ((("0.1", 0), ("0.5", 10), ("+Inf", 10)), 2.0)})
    end = exposition(histograms={E2E: ((("0.1", 8), ("0.5", 11), ("+Inf", 11)), 2.3)})
    window = series([start, end])
    for share in (
        histogram_share_above(window, E2E, 0.1),
        histogram_share_above(window, E2E, 0.3),
    ):
        assert share.reasons == (REASON_HISTOGRAM_INCONSISTENT,)
        assert share.lo is None and share.hi is None
    p90 = histogram_quantile_bounds(window, E2E, 0.9)
    assert p90.reasons == (REASON_HISTOGRAM_INCONSISTENT,) and p90.hi is None


def test_a_step_whose_cumulative_counts_fall_is_refused() -> None:
    # Every part grew and no bucket exceeds the step's count of 5, but the
    # step's cumulative buckets [3, 2, 5] fall: the share above 0.3 would be
    # bounded by lo 0.6 > hi 0.4.
    start = exposition(histograms={E2E: ((("0.1", 0), ("0.5", 5), ("+Inf", 5)), 1.0)})
    end = exposition(histograms={E2E: ((("0.1", 3), ("0.5", 7), ("+Inf", 10)), 2.0)})
    share = histogram_share_above(series([start, end]), E2E, 0.3)
    assert share.reasons == (REASON_HISTOGRAM_INCONSISTENT,)


def test_a_step_with_a_bucket_above_its_count_is_refused() -> None:
    # No +Inf bucket to compare with: the finite bucket alone exceeds the count.
    start = exposition(histograms={E2E: ((("0.1", 0), ("0.5", 0)), 0.0)})
    end = exposition(histograms={E2E: ((("0.1", 5), ("0.5", 5)), 1.0)}).replace(
        f"{E2E}_count{{{LABELS}}} 5", f"{E2E}_count{{{LABELS}}} 2"
    )
    share = histogram_share_above(series([start, end]), E2E, 0.1)
    assert share.reasons == (REASON_HISTOGRAM_INCONSISTENT,)


def test_a_consistent_step_still_resolves_and_a_falling_bucket_is_a_reset() -> None:
    start = exposition(histograms={E2E: ((("0.1", 2), ("0.5", 3), ("+Inf", 4)), 1.0)})
    grown = exposition(histograms={E2E: ((("0.1", 3), ("0.5", 6), ("+Inf", 8)), 3.0)})
    share = histogram_share_above(series([start, grown]), E2E, 0.5)
    # The step's buckets are [1, 3, 4]: one of its four observations is above.
    assert (share.count_delta, share.lo, share.hi, share.reasons) == (
        4.0,
        0.25,
        0.25,
        (),
    )
    fell = exposition(histograms={E2E: ((("0.1", 1), ("0.5", 6), ("+Inf", 8)), 3.0)})
    reset = histogram_share_above(series([start, fell]), E2E, 0.5)
    assert reset.reasons == (REASON_COUNTER_RESET,)


@pytest.mark.parametrize(
    "thresholds, message",
    [
        ({"queue_saturation.median_waiting_requests": float("nan")}, "finite"),
        ({"kv_preemption_pressure.preemptions": float("inf")}, "finite"),
        ({"queue_saturation.median_waiting_requestz": 3.0}, "unknown"),
    ],
)
def test_a_threshold_override_that_could_never_decide_is_refused(
    thresholds: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SignalConfig(thresholds=thresholds)


def test_a_known_finite_override_is_used_and_said() -> None:
    window = series([exposition(gauges={WAITING: 5})] * 2)
    config = SignalConfig(thresholds={"queue_saturation.median_waiting_requests": 6})
    signal = evaluate_signal("queue_saturation", window, config)
    assert (signal.exceeds, signal.threshold, signal.threshold_overridden) == (
        False,
        6.0,
        True,
    )


@pytest.mark.parametrize("reference", [-0.1, 1.5, float("nan")])
def test_a_prefix_reference_outside_zero_to_one_is_refused(reference: float) -> None:
    with pytest.raises(ValueError, match="reference"):
        SignalConfig(reference=reference)


def _split_engines(hits: float, queries: float) -> str:
    """Engine 0 exports the hits and engine 1 the queries."""
    text = exposition(counters={HITS: hits, QUERIES: queries})
    for family in (QUERIES, "vllm:prefix_cache_queries_created"):
        text = text.replace(f'{family}{{engine="0"', f'{family}{{engine="1"')
    return text


def test_a_signal_never_combines_two_engines_families() -> None:
    window = series([_split_engines(0.0, 0.0), _split_engines(50.0, 100.0)])
    check = check_window(window)
    assert check.reasons == (REASON_ENGINE_REQUIRED,) and not check.sufficient
    mixed = evaluate_signal("prefix_cache_loss", window, SignalConfig(reference=0.8))
    assert (mixed.sufficient, mixed.reason, mixed.exceeds) == (
        False,
        REASON_ENGINE_REQUIRED,
        None,
    )
    for engine in ("0", "1"):
        named = SignalConfig(reference=0.8, engine=engine)
        assert check_window(window, engine=engine).sufficient
        assert evaluate_signal("prefix_cache_loss", window, named).exceeds is None


def test_one_engine_needs_no_name() -> None:
    window = series([exposition(gauges={WAITING: 9})] * 2)
    assert check_window(window).sufficient


def _refused_windows() -> dict[str, list[VllmScrapeRecord]]:
    """Windows the window layer refuses, each with every family the three
    metric signals read, on one engine unless the case is about engines."""

    def text(i: float, start: float = START) -> str:
        return exposition(
            gauges={WAITING: 5.0},
            counters={PREEMPTIONS: i, HITS: i, QUERIES: 10 * i},
            start=start,
        )

    return {
        "failed_first": series([None, text(0), text(2)]),
        "failed_last": series([text(0), text(2), None]),
        "out_of_order": [scrape(text(2), 2.0), scrape(text(0), 1.0)],
        "one_instant": [scrape(text(0), 1.0), scrape(text(2), 1.0)],
        "midpoints_inverted": [
            scrape(text(0), 0.0, duration_ms=20_000),
            scrape(text(2), 1.0),
        ],
        "restarted": series([text(0), text(2, start=1_790_000_900.0)]),
        "two_engines": series([_split_engines(0.0, 0.0), _split_engines(5.0, 10.0)]),
    }


@pytest.mark.parametrize("window", sorted(_refused_windows()))
@pytest.mark.parametrize(
    "kind", ["queue_saturation", "kv_preemption_pressure", "prefix_cache_loss"]
)
def test_every_signal_abstains_over_a_refused_window(kind: str, window: str) -> None:
    # The composed path a trigger takes: evaluate_signal runs check_window
    # first, so a window it refuses never reaches a verdict.
    signal = evaluate_signal(
        kind, _refused_windows()[window], SignalConfig(reference=0.8)
    )
    assert (signal.sufficient, signal.exceeds) == (False, None)
    assert signal.reason is not None
