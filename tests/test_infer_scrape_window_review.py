"""Window edge cases found in review: label changes, ordering, failures,
duplicate stamps, inconsistent histograms and an invalid reference."""

from __future__ import annotations

from stormlog.infer.diagnosis_signals import evaluate_signal
from stormlog.infer.scrape_window import (
    REASON_DUPLICATE_TIME,
    REASON_OUT_OF_ORDER,
    REASON_SERIES_LABELS_CHANGED,
    check_window,
    counter_window,
    gauge_window,
)
from tests.vllm_scrape_helpers import exposition, scrape, series

PREEMPTIONS = "vllm:num_preemptions_total"
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
