"""Windows over vLLM scrapes: gauges, consecutive counter deltas, histogram bounds."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from stormlog.infer.scrape_window import (
    PLACEMENT_APPROXIMATE,
    REASON_AMBIGUOUS_SERIES,
    REASON_BOUNDARIES_CHANGED,
    REASON_COUNTER_RECREATED,
    REASON_COUNTER_RESET,
    REASON_ENGINE_REQUIRED,
    REASON_ENGINE_RESTART,
    REASON_IDENTITY_UNKNOWN,
    REASON_NO_OBSERVATIONS,
    REASON_NON_FINITE,
    REASON_OUT_OF_ORDER,
    REASON_OVERFLOW_BUCKET,
    REASON_SCRAPE_FAILED,
    REASON_SERIES_MISSING,
    REASON_TOO_FEW_SCRAPES,
    check_window,
    counter_window,
    gauge_median,
    gauge_window,
    histogram_quantile_bounds,
    histogram_share_above,
    sample_interval,
    share_at_least,
)
from stormlog.infer.vllm_metrics import compact_scrape, discover, parse_prometheus_text
from stormlog.infer.vllm_telemetry import (
    MARKER_INTERVAL,
    SCRAPE_ERROR,
    SCRAPE_OK,
    VllmScrapeRecord,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "vllm"
T0 = 1_790_000_000_000_000_000
SECOND = 1_000_000_000
WAITING = "vllm:num_requests_waiting"
BY_REASON = "vllm:num_requests_waiting_by_reason"
PREEMPTIONS = "vllm:num_preemptions_total"
E2E = "vllm:e2e_request_latency_seconds"
LABELS = 'engine="0",model_name="m"'
CLOCK = "client/boot/unix_epoch_ns"


def _text(
    *,
    waiting: float | str = 0.0,
    preemptions: float = 0.0,
    created: float = 1_790_000_000.0,
    start: float = 1_790_000_000.0,
    buckets: tuple[tuple[str, float], ...] = (("0.1", 0), ("0.5", 0), ("+Inf", 0)),
    e2e_sum: float = 0.0,
    capacity: float = 0.0,
    second_engine: bool = False,
) -> str:
    lines = [
        "# TYPE process_start_time_seconds gauge",
        f"process_start_time_seconds {start}",
        f"# TYPE {WAITING} gauge",
        f"{WAITING}{{{LABELS}}} {waiting}",
        f"# TYPE {BY_REASON} gauge",
        f'{BY_REASON}{{{LABELS},reason="capacity"}} {capacity}',
        f'{BY_REASON}{{{LABELS},reason="deferred"}} 0.0',
        f"# TYPE {PREEMPTIONS} counter",
        f"{PREEMPTIONS}{{{LABELS}}} {preemptions}",
        "# TYPE vllm:num_preemptions_created gauge",
        f"vllm:num_preemptions_created{{{LABELS}}} {created}",
        f"# TYPE {E2E} histogram",
    ]
    lines += [f'{E2E}_bucket{{{LABELS},le="{le}"}} {count}' for le, count in buckets]
    lines += [
        f"{E2E}_sum{{{LABELS}}} {e2e_sum}",
        f"{E2E}_count{{{LABELS}}} {buckets[-1][1]}",
    ]
    if second_engine:
        lines.append(f'{WAITING}{{engine="1",model_name="m"}} 7.0')
    return "\n".join(lines) + "\n"


def _scrape(
    text: str | None, at_s: float, *, duration_ms: float = 4.0
) -> VllmScrapeRecord:
    observed_at_ns = T0 + round(at_s * SECOND)
    if text is None:
        return VllmScrapeRecord(
            session_id="session",
            run_id="run-1",
            observed_at_ns=observed_at_ns,
            source_url="http://127.0.0.1:8000/metrics",
            marker=MARKER_INTERVAL,
            interval_ms=1000,
            status=SCRAPE_ERROR,
            clock_domain=CLOCK,
            duration_ms=duration_ms,
            error="HTTP 503",
        )
    compact = compact_scrape(parse_prometheus_text(text))
    return VllmScrapeRecord(
        session_id="session",
        run_id="run-1",
        observed_at_ns=observed_at_ns,
        source_url="http://127.0.0.1:8000/metrics",
        marker=MARKER_INTERVAL,
        interval_ms=1000,
        status=SCRAPE_OK,
        clock_domain=CLOCK,
        duration_ms=duration_ms,
        scrape=compact,
        discovery=discover(compact),
    )


# ------------------------------------------------------------------- window
def test_window_seconds_run_between_sample_midpoints_with_bounds() -> None:
    scrapes = [_scrape(_text(), 0.0), _scrape(_text(), 1.0), _scrape(_text(), 2.0)]
    check = check_window(scrapes)
    assert check.sufficient and check.reasons == ()
    assert check.scrapes == 3
    assert check.seconds == pytest.approx(2.0)
    # Each scrape sampled somewhere in its 4 ms response interval.
    assert check.seconds_bounds == pytest.approx((1.996, 2.004))
    # Records without completed_at_ns are bounded by their duration only.
    assert check.placement == PLACEMENT_APPROXIMATE


def test_sample_interval_prefers_the_completion_stamp() -> None:
    record = SimpleNamespace(
        observed_at_ns=T0, duration_ms=4.0, completed_at_ns=T0 + 9_000_000
    )
    assert sample_interval(cast(VllmScrapeRecord, record)) == (T0, T0 + 9_000_000)


def test_window_reasons_failed_short_out_of_order_and_restarted() -> None:
    failed = check_window([_scrape(_text(), 0.0), _scrape(None, 1.0)])
    assert failed.reasons == (REASON_SCRAPE_FAILED, REASON_TOO_FEW_SCRAPES)
    assert not failed.sufficient
    swapped = check_window([_scrape(_text(), 1.0), _scrape(_text(), 0.0)])
    assert REASON_OUT_OF_ORDER in swapped.reasons
    restarted = check_window(
        [_scrape(_text(), 0.0), _scrape(_text(start=1_790_000_500.0), 1.0)]
    )
    assert restarted.reasons == (REASON_ENGINE_RESTART,)
    anonymous = (
        _text()
        .replace("process_start_time_seconds 1790000000.0\n", "")
        .replace("# TYPE process_start_time_seconds gauge\n", "")
    )
    unknown = check_window([_scrape(anonymous, 0.0), _scrape(anonymous, 1.0)])
    assert unknown.reasons == (REASON_IDENTITY_UNKNOWN,)


def test_window_with_an_engine_the_scrapes_lack() -> None:
    scrapes = [_scrape(_text(), 0.0), _scrape(_text(), 1.0)]
    assert check_window(scrapes, engine="0").sufficient
    assert check_window(scrapes, engine="3").reasons == (REASON_SERIES_MISSING,)


# ------------------------------------------------------------------- gauges
def test_gauge_statistics_share_and_median() -> None:
    scrapes = [
        _scrape(_text(waiting=value), float(i)) for i, value in enumerate([0, 2, 5])
    ]
    gauge = gauge_window(scrapes, WAITING)
    assert gauge.reasons == ()
    assert (gauge.n, gauge.min, gauge.max, gauge.last) == (3, 0.0, 5.0, 5.0)
    assert gauge.mean == pytest.approx(7 / 3)
    assert gauge_median(gauge) == 2.0
    assert share_at_least(gauge, 1.0) == pytest.approx(2 / 3)
    # "Every sample >= t" is the minimum over the finite samples.
    assert gauge.min is not None and gauge.min < 1.0


def test_gauge_non_finite_samples_are_counted_not_averaged() -> None:
    scrapes = [_scrape(_text(waiting=3.0), 0.0), _scrape(_text(waiting="NaN"), 1.0)]
    gauge = gauge_window(scrapes, WAITING)
    assert (gauge.n, gauge.non_finite, gauge.mean) == (1, 1, 3.0)
    assert gauge.reasons == (REASON_NON_FINITE,)


def test_gauge_missing_series_and_failed_scrapes() -> None:
    without = _text().replace(f"{WAITING}{{{LABELS}}} 0.0\n", "")
    scrapes = [
        _scrape(_text(waiting=1.0), 0.0),
        _scrape(without, 1.0),
        _scrape(None, 2),
    ]
    gauge = gauge_window(scrapes, WAITING)
    # The failed scrape is check_window's to report; the missing series is ours.
    assert gauge.n == 1
    assert gauge.reasons == (REASON_SERIES_MISSING,)
    assert share_at_least(gauge_window([], WAITING), 1.0) is None


def test_two_engines_need_an_engine_and_label_sets_are_never_summed() -> None:
    scrapes = [_scrape(_text(second_engine=True), 0.0)]
    assert gauge_window(scrapes, WAITING).reasons == (REASON_ENGINE_REQUIRED,)
    assert gauge_window(scrapes, WAITING, engine="1").values == (7.0,)
    single = [_scrape(_text(capacity=4.0), 0.0)]
    assert gauge_window(single, BY_REASON).reasons == (REASON_AMBIGUOUS_SERIES,)
    capacity = gauge_window(single, BY_REASON, labels={"reason": "capacity"})
    assert capacity.values == (4.0,)


# ----------------------------------------------------------------- counters
def test_counter_sums_consecutive_deltas_with_rate_bounds() -> None:
    scrapes = [
        _scrape(_text(preemptions=value), float(i))
        for i, value in enumerate([3.0, 5.0, 9.0])
    ]
    counter = counter_window(scrapes, PREEMPTIONS)
    assert counter.reasons == ()
    assert counter.delta == 6.0
    assert counter.rate_per_s == pytest.approx(3.0)
    slow, fast = counter.rate_bounds or (None, None)
    assert slow == pytest.approx(6.0 / 2.004)
    assert fast == pytest.approx(6.0 / 1.996)


def test_an_interior_reset_is_caught_even_when_the_ends_agree() -> None:
    # First to last looks like +2; the counter went backwards in between.
    scrapes = [
        _scrape(_text(preemptions=value), float(i))
        for i, value in enumerate([10.0, 2.0, 12.0])
    ]
    counter = counter_window(scrapes, PREEMPTIONS)
    assert counter.delta is None and counter.rate_per_s is None
    assert counter.reasons == (REASON_COUNTER_RESET,)


def test_counter_recreated_restarted_or_too_short_is_unresolved() -> None:
    recreated = [
        _scrape(_text(preemptions=1.0), 0.0),
        _scrape(_text(preemptions=4.0, created=1_790_000_100.0), 1.0),
    ]
    assert counter_window(recreated, PREEMPTIONS).reasons == (REASON_COUNTER_RECREATED,)
    restarted = [
        _scrape(_text(preemptions=1.0), 0.0),
        _scrape(_text(preemptions=4.0, start=1_790_000_100.0), 1.0),
    ]
    assert counter_window(restarted, PREEMPTIONS).reasons == (REASON_ENGINE_RESTART,)
    single = counter_window([_scrape(_text(), 0.0)], PREEMPTIONS)
    assert single.reasons == (REASON_TOO_FEW_SCRAPES,)


def test_counter_matches_the_case_delta_on_real_vllm_scrapes() -> None:
    pre = (FIXTURES / "q05_c08_metrics_pre.txt").read_text(encoding="utf-8")
    post = (FIXTURES / "q05_c08_metrics_post.txt").read_text(encoding="utf-8")
    scrapes = [_scrape(pre, 0.0), _scrape(post, 10.0)]
    family = "vllm:generation_tokens_total"
    counter = counter_window(scrapes, family, engine="0")
    first = scrapes[0].scrape
    last = scrapes[1].scrape
    assert first is not None and last is not None
    before = next(iter(first.series(family).values()))
    after = next(iter(last.series(family).values()))
    assert isinstance(before, float) and isinstance(after, float)
    assert counter.delta == after - before
    assert counter.reasons == ()


# --------------------------------------------------------------- histograms
def _histogram_scrapes(
    before: tuple[tuple[str, float], ...], after: tuple[tuple[str, float], ...]
) -> list[VllmScrapeRecord]:
    return [
        _scrape(_text(buckets=before, e2e_sum=1.0), 0.0),
        _scrape(_text(buckets=after, e2e_sum=9.0), 1.0),
    ]


START = (("0.1", 1.0), ("0.5", 2.0), ("1.0", 2.0), ("+Inf", 2.0))
# In the window: 2 observations <= 0.1, 5 in (0.1, 0.5], 1 in (0.5, 1], 1 above 1.
END = (("0.1", 3.0), ("0.5", 9.0), ("1.0", 10.0), ("+Inf", 11.0))


def test_share_above_a_boundary_is_exact_and_between_is_bounded() -> None:
    scrapes = _histogram_scrapes(START, END)
    at_boundary = histogram_share_above(scrapes, E2E, 0.5)
    assert at_boundary.count_delta == 9.0
    assert at_boundary.lo == at_boundary.hi == pytest.approx(2 / 9)
    between = histogram_share_above(scrapes, E2E, 0.3)
    # Above 0.5 for certain (2 of 9); possibly every one of the 7 above 0.1.
    assert between.lo == pytest.approx(2 / 9)
    assert between.hi == pytest.approx(7 / 9)
    below_all = histogram_share_above(scrapes, E2E, 0.05)
    assert below_all.lo == pytest.approx(7 / 9)
    assert below_all.hi == 1.0


def test_quantile_bounds_and_the_overflow_bucket() -> None:
    scrapes = _histogram_scrapes(START, END)
    median_bounds = histogram_quantile_bounds(scrapes, E2E, 0.5)
    assert (median_bounds.lo, median_bounds.hi) == (0.1, 0.5)
    first = histogram_quantile_bounds(scrapes, E2E, 0.1)
    assert (first.lo, first.hi) == (None, 0.1)
    tail = histogram_quantile_bounds(scrapes, E2E, 0.99)
    assert (tail.lo, tail.hi) == (1.0, None)
    assert tail.reasons == (REASON_OVERFLOW_BUCKET,)
    with pytest.raises(ValueError):
        histogram_quantile_bounds(scrapes, E2E, 1.5)


def test_a_quantile_whose_rank_is_a_bucket_s_count_is_in_that_bucket() -> None:
    # 2 of the 9 observations are at or below 0.1, so the 2/9 quantile is.
    tie = histogram_quantile_bounds(_histogram_scrapes(START, END), E2E, 2 / 9)
    assert (tie.lo, tie.hi, tie.reasons) == (None, 0.1, ())


def test_a_share_above_nan_is_refused() -> None:
    # As a quantile outside 0 to 1 is: NaN would give bounds of 0 to 1.
    with pytest.raises(ValueError, match="NaN"):
        histogram_share_above(_histogram_scrapes(START, END), E2E, float("nan"))


def test_histogram_without_observations_or_with_changed_boundaries() -> None:
    quiet = _histogram_scrapes(START, START)
    assert histogram_share_above(quiet, E2E, 0.5).reasons == (REASON_NO_OBSERVATIONS,)
    reshaped = (("0.2", 3.0), ("0.5", 9.0), ("1.0", 10.0), ("+Inf", 11.0))
    changed = histogram_quantile_bounds(_histogram_scrapes(START, reshaped), E2E, 0.5)
    assert changed.reasons == (REASON_BOUNDARIES_CHANGED,)
    assert changed.lo is None and changed.hi is None


def test_histogram_bucket_that_went_backwards_is_a_reset() -> None:
    middle = (("0.1", 0.0), ("0.5", 2.0), ("1.0", 2.0), ("+Inf", 2.0))
    scrapes = [
        _scrape(_text(buckets=START, e2e_sum=1.0), 0.0),
        _scrape(_text(buckets=middle, e2e_sum=1.0), 1.0),
        _scrape(_text(buckets=END, e2e_sum=9.0), 2.0),
    ]
    share = histogram_share_above(scrapes, E2E, 0.5)
    assert share.reasons == (REASON_COUNTER_RESET,)
