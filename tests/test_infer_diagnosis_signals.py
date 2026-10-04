"""Online diagnosis signals: the vocabulary, the threshold table, evaluate_signal."""

from __future__ import annotations

import pytest

from stormlog.infer import diagnosis_vocabulary as kinds
from stormlog.infer.diagnosis_signals import (
    REASON_REQUIRES_CLIENT,
    REASON_REQUIRES_HOOK,
    REASON_REQUIRES_REFERENCE,
    REASON_REQUIRES_TRACE,
    REASON_TOO_FEW_QUERIES,
    SignalConfig,
    evaluate_signal,
)
from stormlog.infer.diagnosis_thresholds import (
    DEFAULT_THRESHOLDS,
    KV_PREEMPTIONS,
    PREFIX_MIN_QUERIED,
    QUEUE_MEDIAN_WAITING,
    THRESHOLDS_VERSION,
    resolve_threshold,
)
from stormlog.infer.scrape_window import (
    REASON_COUNTER_RESET,
    REASON_ENGINE_RESTART,
    REASON_NO_OBSERVATIONS,
    REASON_SCRAPE_FAILED,
    REASON_TOO_FEW_SCRAPES,
)
from stormlog.infer.vllm_telemetry import VllmScrapeRecord
from tests.vllm_scrape_helpers import exposition, scrape, series

WAITING = "vllm:num_requests_waiting"
BY_REASON = "vllm:num_requests_waiting_by_reason"
QUEUE_TIME = "vllm:request_queue_time_seconds"
PREEMPTIONS = "vllm:num_preemptions_total"
KV_USAGE = "vllm:kv_cache_usage_perc"
HITS = "vllm:prefix_cache_hits_total"
QUERIES = "vllm:prefix_cache_queries_total"


# --------------------------------------------------------------- vocabulary
def test_the_vocabulary_is_closed_and_every_kind_is_located() -> None:
    assert len(kinds.KINDS) == 13
    assert kinds.MECHANISM_KINDS == {
        kinds.QUEUE_SATURATION,
        kinds.KV_PREEMPTION_PRESSURE,
        kinds.PREFIX_CACHE_LOSS,
        kinds.MIXED_PREFILL_INTERFERENCE,
        kinds.HOST_STALL,
        kinds.RANK_DELAY,
        kinds.TRANSFER_DEGRADATION,
    }
    assert kinds.KIND_COMPONENTS[kinds.HOST_STALL] == {
        kinds.COMPONENT_ENGINE_CORE,
        kinds.COMPONENT_WORKER,
        kinds.COMPONENT_API_SERVER,
    }
    assert all(kinds.KIND_COMPONENTS[kind] for kind in kinds.KINDS)
    assert kinds.check_kind("queue_saturation") == "queue_saturation"
    with pytest.raises(ValueError, match="unknown diagnosis kind"):
        kinds.check_kind("host_launch_gap")


def test_thresholds_resolve_from_the_table_or_an_override() -> None:
    assert resolve_threshold(QUEUE_MEDIAN_WAITING) == (1.0, False)
    assert resolve_threshold(QUEUE_MEDIAN_WAITING, {QUEUE_MEDIAN_WAITING: 4}) == (
        4.0,
        True,
    )
    assert set(DEFAULT_THRESHOLDS) >= {QUEUE_MEDIAN_WAITING, KV_PREEMPTIONS}
    with pytest.raises(KeyError):
        resolve_threshold("queue_saturation.unknown")


# -------------------------------------------------------------------- queue
def _queue_window(*waiting: float) -> list[VllmScrapeRecord]:
    texts = [
        exposition(
            gauges={WAITING: value},
            labelled={BY_REASON: {"reason=capacity": value, "reason=deferred": 0.0}},
            histograms={
                QUEUE_TIME: (
                    (("0.1", 10 * i), ("1.0", 12 * i), ("+Inf", 12 * i)),
                    float(i),
                )
            },
        )
        for i, value in enumerate(waiting)
    ]
    return series(texts)


def test_queue_signal_is_the_median_waiting_count() -> None:
    signal = evaluate_signal("queue_saturation", _queue_window(0, 3, 4))
    assert signal.sufficient and signal.reason is None
    assert signal.value == 3.0
    assert signal.exceeds is True
    assert (signal.threshold, signal.threshold_overridden) == (1.0, False)
    assert signal.thresholds_version == THRESHOLDS_VERSION
    assert signal.detail["scope"] == "engine_global"
    assert signal.detail["max"] == 4.0
    assert signal.detail["waiting_by_reason"] == {"capacity": 3.0, "deferred": 0.0}
    # 24 queue-time observations in the window, 20 of them within 0.1 s.
    assert signal.detail["queue_time_p90_s"] == [0.1, 1.0]


def test_queue_below_threshold_and_with_an_override() -> None:
    quiet = evaluate_signal("queue_saturation", _queue_window(0, 0, 1))
    assert (quiet.value, quiet.exceeds) == (0.0, False)
    strict = evaluate_signal(
        "queue_saturation",
        _queue_window(0, 3, 4),
        SignalConfig(thresholds={QUEUE_MEDIAN_WAITING: 5.0}),
    )
    assert (strict.exceeds, strict.threshold, strict.threshold_overridden) == (
        False,
        5.0,
        True,
    )


def test_queue_window_too_short_failed_or_restarted_gives_no_verdict() -> None:
    short = evaluate_signal("queue_saturation", _queue_window(5))
    assert (short.sufficient, short.exceeds, short.reason) == (
        False,
        None,
        REASON_TOO_FEW_SCRAPES,
    )
    with_failure = _queue_window(5, 5)
    with_failure.append(scrape(None, 2.0))
    failed = evaluate_signal("queue_saturation", with_failure)
    assert failed.reason == REASON_SCRAPE_FAILED and failed.exceeds is None
    restarted = series(
        [
            exposition(gauges={WAITING: 5.0}),
            exposition(gauges={WAITING: 5.0}, start=1_790_000_900.0),
        ]
    )
    signal = evaluate_signal("queue_saturation", restarted)
    assert signal.reason == REASON_ENGINE_RESTART
    assert signal.detail["reasons"] == [REASON_ENGINE_RESTART]


# ----------------------------------------------------------------------- KV
def test_kv_signal_is_the_preemption_delta() -> None:
    texts = [
        exposition(counters={PREEMPTIONS: value}, gauges={KV_USAGE: usage})
        for value, usage in ((0.0, 0.5), (0.0, 0.9), (2.0, 1.0))
    ]
    signal = evaluate_signal("kv_preemption_pressure", series(texts))
    assert (signal.value, signal.exceeds) == (2.0, True)
    assert signal.detail["kv_cache_usage_max"] == 1.0
    steady = series([exposition(counters={PREEMPTIONS: 5.0})] * 2)
    assert evaluate_signal("kv_preemption_pressure", steady).exceeds is False


def test_kv_signal_refuses_a_window_with_an_interior_reset() -> None:
    texts = [exposition(counters={PREEMPTIONS: value}) for value in (7.0, 1.0, 9.0)]
    signal = evaluate_signal("kv_preemption_pressure", series(texts))
    assert (signal.value, signal.exceeds, signal.reason) == (
        None,
        None,
        REASON_COUNTER_RESET,
    )


# ------------------------------------------------------------- prefix cache
def _prefix_window(hits: float, queries: float) -> list[VllmScrapeRecord]:
    return series(
        [
            exposition(counters={HITS: 0.0, QUERIES: 0.0}),
            exposition(counters={HITS: hits, QUERIES: queries}),
        ]
    )


def test_prefix_signal_needs_the_callers_reference() -> None:
    signal = evaluate_signal("prefix_cache_loss", _prefix_window(5000.0, 10000.0))
    # The ratio is measured; with no reference there is no verdict.
    assert signal.value == 0.5
    assert (signal.sufficient, signal.exceeds) == (False, None)
    assert signal.reason == REASON_REQUIRES_REFERENCE
    assert signal.detail["scope"] == "engine_global"


def test_prefix_signal_compares_the_ratio_with_the_reference() -> None:
    fell = evaluate_signal(
        "prefix_cache_loss",
        _prefix_window(5000.0, 10000.0),
        SignalConfig(reference=0.8),
    )
    assert (fell.value, fell.exceeds) == (0.5, True)
    assert fell.detail["drop"] == pytest.approx(0.3)
    held = evaluate_signal(
        "prefix_cache_loss",
        _prefix_window(7500.0, 10000.0),
        SignalConfig(reference=0.8),
    )
    assert held.exceeds is False
    idle = evaluate_signal(
        "prefix_cache_loss", _prefix_window(0.0, 0.0), SignalConfig(reference=0.8)
    )
    assert (idle.value, idle.reason) == (None, REASON_NO_OBSERVATIONS)


def test_a_few_queried_tokens_decide_no_prefix_signal() -> None:
    """vLLM counts every prompt token of a new request as a query: one
    short, unseen prompt alone has a ratio of 0, which says nothing."""
    one_prompt = evaluate_signal(
        "prefix_cache_loss", _prefix_window(0.0, 16.0), SignalConfig(reference=0.5)
    )
    assert one_prompt.value == 0.0
    assert (one_prompt.sufficient, one_prompt.exceeds) == (False, None)
    assert one_prompt.reason == REASON_TOO_FEW_QUERIES
    enough = evaluate_signal(
        "prefix_cache_loss",
        _prefix_window(0.0, 16.0),
        SignalConfig(reference=0.5, thresholds={PREFIX_MIN_QUERIED: 16.0}),
    )
    assert enough.exceeds is True


def test_a_ratio_above_the_reference_is_no_loss() -> None:
    risen = evaluate_signal(
        "prefix_cache_loss",
        _prefix_window(10000.0, 10000.0),
        SignalConfig(reference=0.3),
    )
    assert (risen.value, risen.exceeds) == (1.0, False)
    assert risen.detail["drop"] == pytest.approx(-0.7)


# ------------------------------------------------------- at the threshold
def test_a_value_equal_to_its_threshold_exceeds_it() -> None:
    """Every threshold is reached at its value: one preemption is evidence."""
    queue = evaluate_signal("queue_saturation", _queue_window(0, 1, 1))
    assert (queue.value, queue.exceeds) == (1.0, True)
    preempted = series([exposition(counters={PREEMPTIONS: v}) for v in (0.0, 1.0)])
    kv = evaluate_signal("kv_preemption_pressure", preempted)
    assert (kv.value, kv.exceeds) == (1.0, True)
    # 0.5 - 0.3 is exactly 0.2 in floating point (0.8 - 0.6 is not).
    prefix = evaluate_signal(
        "prefix_cache_loss",
        _prefix_window(1500.0, 5000.0),
        SignalConfig(reference=0.5),
    )
    assert (prefix.detail["drop"], prefix.exceeds) == (0.2, True)


# ---------------------------------------------------------- other kinds
@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("mixed_prefill_interference", REASON_REQUIRES_HOOK),
        ("host_stall", REASON_REQUIRES_HOOK),
        ("capture_pause", REASON_REQUIRES_HOOK),
        ("rank_delay", REASON_REQUIRES_TRACE),
        ("transfer_degradation", REASON_REQUIRES_TRACE),
        ("client_admission", REASON_REQUIRES_CLIENT),
        ("load_increase", REASON_REQUIRES_CLIENT),
    ],
)
def test_kinds_metrics_cannot_decide_say_what_they_need(kind: str, reason: str) -> None:
    signal = evaluate_signal(kind, _queue_window(5, 5))
    assert (signal.sufficient, signal.exceeds, signal.reason) == (False, None, reason)


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError):
        evaluate_signal("frontend_delay", _queue_window(5, 5))
