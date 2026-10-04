"""Window selection, the predicates, and the per-tick trigger engine."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from stormlog.infer.diagnosis_signals import SignalConfig
from stormlog.infer.diagnosis_vocabulary import QUEUE_SATURATION
from stormlog.infer.scrape_window import REASON_ENGINE_REQUIRED
from stormlog.infer.vllm_telemetry import VllmScrapeRecord
from stormlog.infer.watch.evaluate import (
    ACTION_DEEP_CAPTURE,
    KIND_HEALTH,
    KIND_METRIC,
    KIND_SIGNAL,
    TriggerEngine,
    TriggerSpec,
)
from stormlog.infer.watch.history import Stamped
from stormlog.infer.watch.predicates import (
    REASON_END_FAILED,
    REASON_END_STALE,
    REASON_START_MISSING,
    REASON_TOO_FEW_SAMPLES,
    CounterRateAtLeast,
    FrozenExporter,
    GaugeAtLeast,
    HistogramShareAbove,
    ScrapeFailures,
    ScrapeFailureShare,
    SignalExceeds,
    WindowPredicate,
    exporter_restarted,
    overlaps,
    select_window,
)
from stormlog.infer.watch.triggers import (
    CLEAR,
    DATA_GAP,
    EVENT_FIRED,
    EVENT_RESET,
    MASKED,
    VIOLATING,
    Sustain,
)
from tests.vllm_scrape_helpers import exposition, scrape

S = 1_000_000_000
WAITING = "vllm:num_requests_waiting"
RUNNING = "vllm:num_requests_running"
TTFT = "vllm:time_to_first_token_seconds"
PREEMPTIONS = "vllm:num_preemptions_total"


def _entries(
    texts: Sequence[str | None], *, start_s: float = 0.0, step_s: float = 1.0
) -> list[tuple[Stamped, VllmScrapeRecord]]:
    """Scrapes one ``step_s`` apart; each finishes 4 ms after it starts."""
    entries = []
    for index, text in enumerate(texts):
        at = start_s + index * step_s
        mono = round(at * S)
        entries.append((Stamped(mono, mono + 4_000_000, mono), scrape(text, at)))
    return entries


def _waiting(value: float, *, running: float = 1.0, tokens: float = 0.0) -> str:
    return exposition(
        gauges={WAITING: value, RUNNING: running},
        counters={
            "vllm:generation_tokens_total": tokens,
            "vllm:prompt_tokens_total": tokens,
        },
    )


# ------------------------------------------------------------------ selection


def test_a_window_spans_its_start_and_end_scrapes() -> None:
    history = _entries([_waiting(1)] * 40)
    selection = select_window(history, at_ns=35 * S, window_ns=30 * S, tick_ns=S)
    assert selection.reason is None
    assert len(selection.scrapes) == 30  # the scrapes finished at 5.004 .. 34.004
    assert selection.start_ns == 5 * S + 4_000_000
    # The latest start scrape within a tick of t - W: at one scrape a tick,
    # the samples span W - Δ, as the docs say.
    assert selection.end_ns is not None
    assert selection.end_ns - selection.start_ns == 29 * S


def test_a_stale_or_failed_end_or_a_missing_start_is_a_data_gap() -> None:
    history = _entries([_waiting(1)] * 10)
    stale = select_window(history, at_ns=20 * S, window_ns=5 * S, tick_ns=S)
    assert stale.reason == REASON_END_STALE
    failed = _entries([_waiting(1)] * 9 + [None])
    end = select_window(failed, at_ns=9 * S + 5_000_000, window_ns=5 * S, tick_ns=S)
    assert end.reason == REASON_END_FAILED
    outage = _entries([None] * 8 + [_waiting(1)] * 3)
    start = select_window(outage, at_ns=10 * S + 5_000_000, window_ns=5 * S, tick_ns=S)
    assert start.reason == REASON_START_MISSING


def test_a_tick_during_a_slow_scrape_judges_the_one_before() -> None:
    """Scrapes start every second and take 50 ms or 300 ms. A tick 0.1 s into
    a slow one finds the newest finished scrape 1.05 s old, past a tick:
    fetch jitter under load must not turn into data gaps."""
    history = []
    for second in range(40):
        mono, took = second * S, (300 if second % 2 else 50) * 1_000_000
        record = scrape(_waiting(9), second, duration_ms=took / 1e6)
        history.append((Stamped(mono, mono + took, mono), record))
    at = 35 * S + 100_000_000
    done = [entry for entry in history if entry[0].done_mono_ns <= at]
    strict = select_window(done, at_ns=at, window_ns=30 * S, tick_ns=S)
    assert strict.reason == REASON_END_STALE
    judged = select_window(
        done, at_ns=at, window_ns=30 * S, tick_ns=S, scrape_timeout_ns=S // 2
    )
    assert judged.reason is None and judged.end_ns == 34 * S + 50_000_000
    engine = TriggerEngine(
        [_queue_trigger()], tick_seconds=1, scrape_timeout_seconds=0.5
    )
    (result,) = engine.tick(at, done)
    assert result.evaluation.classification == VIOLATING


# ----------------------------------------------------------------- predicates


def _scrapes(texts: Sequence[str | None]) -> list[VllmScrapeRecord]:
    return [record for _stamp, record in _entries(texts)]


def test_gauge_at_least_in_every_sample_or_in_a_share() -> None:
    every = GaugeAtLeast(WAITING, threshold=8)
    assert every.evaluate(_scrapes([_waiting(9)] * 5)).classification == VIOLATING
    dipped = _scrapes([_waiting(9), _waiting(2), _waiting(9)])
    result = every.evaluate(dipped)
    assert result.classification == CLEAR and result.observed == 2
    share = GaugeAtLeast(WAITING, threshold=8, share=0.6)
    assert share.evaluate(dipped).classification == VIOLATING
    too_few = GaugeAtLeast(WAITING, threshold=8, min_samples=4)
    assert too_few.evaluate(_scrapes([_waiting(9)] * 3)).reasons == (
        REASON_TOO_FEW_SAMPLES,
    )


def test_counter_rate_fires_on_its_lower_bound_and_not_across_a_reset() -> None:
    def preempted(total: float) -> str:
        return exposition(counters={PREEMPTIONS: total})

    rate = CounterRateAtLeast(PREEMPTIONS, rate_per_s=1.0)
    rising = rate.evaluate(_scrapes([preempted(v) for v in (0, 2, 4, 6)]))
    assert rising.classification == VIOLATING
    assert rising.observed_bounds is not None
    assert rising.observed == rising.observed_bounds[0] <= 2.0
    # 100 -> 0 -> 150: the endpoints look consistent, the interior reset is not.
    reset = rate.evaluate(_scrapes([preempted(v) for v in (100, 0, 150)]))
    assert reset.classification == DATA_GAP
    assert "counter_reset" in reset.reasons


@pytest.mark.parametrize("wall_step_s", [0.0, -0.9, 30.0])
def test_a_rate_is_timed_on_the_monotonic_clock(wall_step_s: float) -> None:
    """An NTP step inside a window moved the records' wall stamps, and a
    true rate of 1/s judged against 1.2/s read 1.29/s and fired."""
    history = []
    for second in range(20):
        wall = second + (wall_step_s if second >= 13 else 0.0)
        record = scrape(exposition(counters={PREEMPTIONS: float(second)}), wall)
        mono = second * S
        history.append((Stamped(mono, mono + 4_000_000, mono), record))
    spec = TriggerSpec(
        "preemptions",
        KIND_METRIC,
        Sustain.with_defaults(window=5, hold=5, clear=None, tick=1),
        CounterRateAtLeast(PREEMPTIONS, rate_per_s=1.2),
    )
    (result,) = TriggerEngine([spec], tick_seconds=1).tick(15 * S + 5_000_000, history)
    assert result.evaluation.classification == CLEAR
    assert result.evaluation.observed_bounds is not None
    lower, upper = result.evaluation.observed_bounds
    assert upper is not None and lower <= 1.0 <= upper


def _ttft(cumulative: Sequence[tuple[str, float]]) -> str:
    return exposition(histograms={TTFT: (cumulative, 10.0)})


def test_a_histogram_share_fires_only_on_its_lower_bucket_bound() -> None:
    before = _ttft([("0.1", 0), ("0.25", 0), ("+Inf", 0)])
    # 40 new requests: 20 under 0.1 s, 10 between 0.1 and 0.25, 10 over.
    after = _ttft([("0.1", 20), ("0.25", 30), ("+Inf", 40)])
    over_200ms = HistogramShareAbove(TTFT, value=0.2, share=0.2)
    result = over_200ms.evaluate(_scrapes([before, after]))
    assert result.observed_bounds == (0.25, 0.5)  # the threshold is between bounds
    assert result.classification == VIOLATING  # lo = 0.25 > 0.2
    cautious = HistogramShareAbove(TTFT, value=0.2, share=0.3)
    assert cautious.evaluate(_scrapes([before, after])).classification == CLEAR
    few = HistogramShareAbove(TTFT, value=0.2, share=0.2, min_samples=50)
    assert few.evaluate(_scrapes([before, after])).classification == DATA_GAP


def test_a_signal_predicate_uses_218s_threshold_and_says_suspected() -> None:
    queue = SignalExceeds(QUEUE_SATURATION, SignalConfig(engine="0"))
    saturated = queue.evaluate(_scrapes([_waiting(500)] * 5))
    assert saturated.classification == VIOLATING
    assert saturated.detail["status"] == "suspected"
    assert saturated.detail["thresholds_version"]
    idle = queue.evaluate(_scrapes([_waiting(0)] * 5))
    assert idle.classification == CLEAR
    one = queue.evaluate(_scrapes([_waiting(500)]))
    assert one.classification == DATA_GAP


def _two_engines(waiting: float) -> str:
    """vLLM with data parallelism: one series per engine."""
    base = _waiting(waiting)
    extra = f'{WAITING}{{engine="1",model_name="m"}} {waiting}'
    return base.replace(f"# TYPE {RUNNING} gauge", extra + f"\n# TYPE {RUNNING} gauge")


@pytest.mark.parametrize(
    ("unnamed", "named"),
    [
        (GaugeAtLeast(WAITING, threshold=8), GaugeAtLeast(WAITING, 8, engine="1")),
        (
            SignalExceeds(QUEUE_SATURATION),
            SignalExceeds(QUEUE_SATURATION, SignalConfig(engine="1")),
        ),
    ],
    ids=["gauge", "signal"],
)
def test_several_engines_need_one_named(
    unnamed: WindowPredicate, named: WindowPredicate
) -> None:
    """Without an engine, a figure could mix two engines' series."""
    scrapes = _scrapes([_two_engines(500)] * 5)
    refused = unnamed.evaluate(scrapes)
    assert refused.classification == DATA_GAP
    assert refused.reasons == (REASON_ENGINE_REQUIRED,)
    assert named.evaluate(scrapes).classification == VIOLATING


def test_a_series_whose_labels_change_is_a_data_gap() -> None:
    def preempted(total: float, model: str) -> str:
        return exposition(counters={PREEMPTIONS: total}).replace(
            'model_name="m"', f'model_name="{model}"'
        )

    rate = CounterRateAtLeast(PREEMPTIONS, rate_per_s=1.0)
    swapped = rate.evaluate(_scrapes([preempted(0, "a"), preempted(9, "b")]))
    assert swapped.classification == DATA_GAP
    assert "series_labels_changed" in swapped.reasons


def test_a_gauge_window_across_an_exporter_restart_or_out_of_order_is_a_gap() -> None:
    """A gauge is never differenced, so nothing else would notice that its
    samples came from two exporters or out of order."""
    gauge = GaugeAtLeast(WAITING, threshold=8)
    before = [
        scrape(exposition(gauges={WAITING: 9}, start=1000.0), at) for at in (0, 1)
    ]
    after = [scrape(exposition(gauges={WAITING: 9}, start=2000.0), at) for at in (2, 3)]
    restarted = gauge.evaluate([*before, *after])
    assert restarted.classification == DATA_GAP
    assert restarted.reasons == ("engine_restart",)
    first, second, third = _scrapes([_waiting(9)] * 3)
    shuffled = gauge.evaluate([first, third, second])
    assert shuffled.classification == DATA_GAP
    assert shuffled.reasons == ("scrapes_out_of_order",)
    assert gauge.evaluate([first, second, third]).classification == VIOLATING


def test_a_failed_scrape_inside_the_window_leaves_it_judged() -> None:
    """Only the window's ends must have succeeded; inside, a failure only
    leaves fewer samples, and a counter is differenced across it."""
    gauge = GaugeAtLeast(WAITING, threshold=8)
    judged = gauge.evaluate(_scrapes([_waiting(9), None, _waiting(9), _waiting(9)]))
    assert judged.classification == VIOLATING
    assert judged.samples == 3


def test_an_evaluation_records_the_failed_scrapes_it_judged_around() -> None:
    """A window judged across failed scrapes says so, so an incident shows
    on how much it was judged."""
    engine = TriggerEngine([_queue_trigger()], tick_seconds=1)
    texts = [_waiting(9)] * 20 + [None, None] + [_waiting(9)] * 15
    (around,) = engine.tick(36 * S + 5_000_000, _entries(texts))
    assert around.evaluation.classification == VIOLATING
    assert around.evaluation.detail["failed_scrapes"] == 2
    (clean,) = engine.tick(37 * S + 5_000_000, _entries([_waiting(9)] * 38))
    assert clean.evaluation.detail["failed_scrapes"] == 0


def test_scrapes_at_one_instant_are_a_data_gap() -> None:
    first, second = _scrapes([_waiting(9), _waiting(9)])
    rate = CounterRateAtLeast("vllm:generation_tokens_total", rate_per_s=0.0)
    duplicate = rate.evaluate([first, first, second])
    assert duplicate.classification == DATA_GAP
    assert "duplicate_scrape_time" in duplicate.reasons


def test_scrape_failures_count_failed_scrapes_as_evidence() -> None:
    failures = ScrapeFailures(consecutive=3)
    assert (
        failures.evaluate_history(_entries([_waiting(1)] + [None] * 3)).classification
        == VIOLATING
    )
    assert (
        failures.evaluate_history(_entries([None, _waiting(1), None])).classification
        == CLEAR
    )


def test_a_frozen_exporter_is_busy_without_progress() -> None:
    frozen = FrozenExporter(ticks=3)
    stuck = [_waiting(4, running=8, tokens=100)] * 4
    assert frozen.evaluate_history(_entries(stuck)).classification == VIOLATING
    moving = [_waiting(4, running=8, tokens=100 + i) for i in range(4)]
    assert frozen.evaluate_history(_entries(moving)).classification == CLEAR
    idle = [_waiting(0, running=0, tokens=100)] * 4
    assert frozen.evaluate_history(_entries(idle)).classification == CLEAR


def test_an_exporter_restart_is_a_change_of_process_start() -> None:
    first = scrape(exposition(gauges={WAITING: 0}, start=1000.0), 0)
    same = scrape(exposition(gauges={WAITING: 0}, start=1000.0), 1)
    again = scrape(exposition(gauges={WAITING: 0}, start=2000.0), 2)
    assert not exporter_restarted(first, same)
    assert exporter_restarted(same, again)


def test_intervals_overlap_when_closed_ranges_meet() -> None:
    assert overlaps(10, 20, [(20, 30)])
    assert not overlaps(10, 19, [(20, 30)])
    assert overlaps(10, 20, [(0, 5), (15, 16)])


# --------------------------------------------------------------------- engine


def _queue_trigger(**overrides: object) -> TriggerSpec:
    values: dict[str, object] = {
        "trigger_id": "queue",
        "kind": KIND_METRIC,
        "sustain": Sustain.with_defaults(window=30, hold=60, clear=None, tick=1),
        "predicate": GaugeAtLeast(WAITING, threshold=8),
    }
    values.update(overrides)
    return TriggerSpec(**values)  # type: ignore[arg-type]


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="only record"):
        TriggerSpec(
            "h",
            KIND_HEALTH,
            Sustain.with_defaults(window=5, hold=5, clear=None, tick=1),
            ScrapeFailures(),
            action=ACTION_DEEP_CAPTURE,
        )
    with pytest.raises(ValueError, match="not evaluated here"):
        _queue_trigger(kind="test")
    with pytest.raises(ValueError, match="unique"):
        TriggerEngine([_queue_trigger(), _queue_trigger()], tick_seconds=1)


def test_specs_follow_the_capture_and_exit_policy() -> None:
    """Decision 13: health triggers never count toward exit 3; signal
    triggers record without a trace; SLO triggers capture only when the
    cause is unexplained, unless told otherwise."""
    sustain = Sustain.with_defaults(window=5, hold=5, clear=None, tick=1)
    health = TriggerSpec("h", KIND_HEALTH, sustain, ScrapeFailures())
    assert health.counts_toward_exit is False
    with pytest.raises(ValueError, match="never count"):
        TriggerSpec(
            "h", KIND_HEALTH, sustain, ScrapeFailures(), counts_toward_exit=True
        )
    with pytest.raises(ValueError, match="record without a trace"):
        TriggerSpec(
            "s",
            KIND_SIGNAL,
            sustain,
            SignalExceeds(QUEUE_SATURATION),
            action=ACTION_DEEP_CAPTURE,
        )
    slo = TriggerSpec("slo", "slo", sustain, GaugeAtLeast(WAITING, 8))
    assert slo.deep_capture_when == "unexplained"
    assert slo.counts_toward_exit is True
    metric = TriggerSpec("m", KIND_METRIC, sustain, GaugeAtLeast(WAITING, 8))
    assert metric.deep_capture_when == "always"


def test_the_engine_resets_on_an_outage_and_fires_after_it_from_scrapes() -> None:
    """Saturated from 0 s; scrapes fail over 89-121 s; W=30, F=60, G=30.

    The worked example in the docs, on real window selection: a start scrape
    may finish up to one tick after ``t - W``, so the first full window is in
    at 29 s, one tick earlier than the docs' idealized 30 s.
    """
    texts: list[str | None] = []
    for second in range(262):
        texts.append(None if 89 <= second <= 121 else _waiting(9))
    history = _entries(texts)
    engine = TriggerEngine([_queue_trigger()], tick_seconds=1)
    events = []
    for second in range(260):
        at = second * S + 5_000_000  # just after the scrape of that second
        done = [entry for entry in history if entry[0].done_mono_ns <= at]
        for result in engine.tick(at, done):
            if result.transition is not None:
                events.append((second, result.transition.event))
    # 59 s accumulated by 88, so the outage at 89 stops it short of F; 31 s
    # of data gap reset it at 119; windows are informative again once their
    # start scrape (122) follows the outage, at 151; fired 60 s later.
    assert events == [
        (29, "pending"),
        (119, EVENT_RESET),
        (151, "pending"),
        (211, EVENT_FIRED),
    ]


def test_a_window_overlapping_a_perturbation_is_masked() -> None:
    history = _entries([_waiting(9)] * 100)
    engine = TriggerEngine([_queue_trigger()], tick_seconds=1)
    pause = [(60 * S, 70 * S)]
    at = 90 * S + 5_000_000
    result = engine.tick(at, history, perturbations=pause)[0]
    assert result.evaluation.classification == MASKED
    assert result.evaluation.observed == 9  # the value is still recorded
    assert "perturbation" in result.evaluation.reasons
    later = engine.tick(101 * S, history, perturbations=pause)[0]
    assert later.evaluation.classification != MASKED  # [71, 101] misses it


def test_a_window_that_reaches_back_past_t_minus_w_is_masked_there_too() -> None:
    """The scrape after a pause failed, so the window starts at the one
    before it, earlier than t - W: the pause is inside the window, and its
    counter jump made the trigger violating, unmasked."""
    history = []
    for second in range(100):
        total = 0 if second <= 60 else 50
        text = None if second == 61 else exposition(counters={PREEMPTIONS: total})
        mono = second * S
        history.append((Stamped(mono, mono + 4_000_000, mono), scrape(text, second)))
    pause = [(round(60.2 * S), round(60.8 * S))]
    spec = TriggerSpec(
        "preemptions",
        KIND_METRIC,
        Sustain.with_defaults(window=30, hold=60, clear=None, tick=1),
        CounterRateAtLeast(PREEMPTIONS, rate_per_s=1.0),
    )
    at = round(91.003 * S)
    done = [entry for entry in history if entry[0].done_mono_ns <= at]
    (result,) = TriggerEngine([spec], tick_seconds=1).tick(
        at, done, perturbations=pause
    )
    assert result.evaluation.classification == MASKED


def test_a_health_predicate_is_masked_over_the_tail_it_reads() -> None:
    """FrozenExporter reads its last ticks + 1 scrapes, more than its window."""
    spec = TriggerSpec(
        "frozen",
        KIND_HEALTH,
        Sustain.with_defaults(window=2, hold=2, clear=None, tick=1),
        FrozenExporter(ticks=5),
    )
    history = _entries([_waiting(3, running=2, tokens=7)] * 10)
    at = 9 * S + 5_000_000
    pause = [(round(4.5 * S), round(4.6 * S))]  # inside the tail, before t - W
    (result,) = TriggerEngine([spec], tick_seconds=1).tick(
        at, history, perturbations=pause
    )
    assert result.evaluation.classification == MASKED


def test_a_completion_recorded_trigger_is_masked_for_the_horizon_too() -> None:
    history = _entries([_waiting(9)] * 200)
    spec = _queue_trigger(completion_recorded=True)
    engine = TriggerEngine([spec], tick_seconds=1)
    pause = [(60 * S, 70 * S)]
    at = 150 * S
    plain = engine.tick(at, history, perturbations=pause)[0]
    assert plain.evaluation.classification != MASKED
    engine = TriggerEngine([spec], tick_seconds=1)
    widened = engine.tick(
        at, history, perturbations=pause, completion_horizon_ns=60 * S
    )[0]
    assert widened.evaluation.classification == MASKED


def test_health_predicates_are_asked_about_the_history_tail() -> None:
    spec = TriggerSpec(
        "scrapes",
        KIND_HEALTH,
        Sustain.with_defaults(window=3, hold=3, clear=None, tick=1),
        ScrapeFailures(consecutive=3),
    )
    history = _entries([_waiting(1)] * 2 + [None] * 10)
    engine = TriggerEngine([spec], tick_seconds=1)
    events = []
    for second in range(4, 12):
        done = [e for e in history if e[0].done_mono_ns <= second * S + 5_000_000]
        for result in engine.tick(second * S + 5_000_000, done):
            if result.transition is not None:
                events.append((second, result.transition.event))
    assert events == [(4, "pending"), (7, EVENT_FIRED)]


# ------------------------------------------------------- sparse scrape failures


def test_isolated_failed_scrapes_neither_blind_a_signal_nor_go_unseen() -> None:
    """One failed scrape every 20 s: no three in a row, so ScrapeFailures
    never fires. The window triggers still fire on the remaining scrapes
    (#218 0a judges a window across an interior failure), and the share of
    failed scrapes is a health incident of its own."""
    sustain = Sustain.with_defaults(window=30, hold=60, clear=None, tick=1)
    specs = [
        TriggerSpec("queue", KIND_SIGNAL, sustain, SignalExceeds(QUEUE_SATURATION)),
        TriggerSpec(
            "failures",
            KIND_HEALTH,
            Sustain.with_defaults(window=3, hold=3, clear=None, tick=1),
            ScrapeFailures(consecutive=3),
        ),
        TriggerSpec(
            "failure_share",
            KIND_HEALTH,
            Sustain.with_defaults(window=60, hold=60, clear=None, tick=1),
            ScrapeFailureShare(share=0.04, scrapes=60),
        ),
    ]
    engine = TriggerEngine(specs, tick_seconds=1)
    texts = [
        None if second and second % 20 == 0 else _waiting(500) for second in range(300)
    ]
    history = _entries(texts)
    fired: dict[str, int] = {}
    for second in range(300):
        at = second * S + 5_000_000
        done = [entry for entry in history if entry[0].done_mono_ns <= at][-90:]
        for result in engine.tick(at, done):
            transition = result.transition
            if transition is not None and transition.event == EVENT_FIRED:
                fired.setdefault(result.spec.trigger_id, second)
    assert "failures" not in fired
    assert fired["queue"] <= 92
    assert fired["failure_share"] <= 125  # 3 of the last 60, held for 60 s


def test_failure_share_needs_its_scrapes() -> None:
    share = ScrapeFailureShare(share=0.5, scrapes=4)
    assert share.evaluate_history(_entries([None, _waiting(1)])).reasons == (
        REASON_TOO_FEW_SAMPLES,
    )
    half = share.evaluate_history(_entries([None, _waiting(1), None, _waiting(1)]))
    assert half.classification == VIOLATING and half.observed == 0.5
    with pytest.raises(ValueError, match="share"):
        ScrapeFailureShare(share=0.0)
    with pytest.raises(ValueError, match="scrapes"):
        ScrapeFailureShare(scrapes=0)


def test_a_health_trigger_never_judges_a_stale_tail() -> None:
    """With no scrape finishing for minutes (a wedged scraper), the scrape
    health triggers said clear, and FrozenExporter kept firing on the same
    old scrapes."""
    sustain = Sustain.with_defaults(window=3, hold=3, clear=None, tick=1)
    specs = [
        TriggerSpec("failures", KIND_HEALTH, sustain, ScrapeFailures(consecutive=3)),
        TriggerSpec("share", KIND_HEALTH, sustain, ScrapeFailureShare(0.5, 3)),
        TriggerSpec("frozen", KIND_HEALTH, sustain, FrozenExporter(ticks=3)),
    ]
    engine = TriggerEngine(specs, tick_seconds=1, scrape_timeout_seconds=2)
    history = _entries([_waiting(3, running=2, tokens=7)] * 6)  # last at 5 s
    fresh = {r.spec.trigger_id: r.evaluation for r in engine.tick(6 * S, history)}
    assert fresh["failures"].classification == CLEAR
    assert fresh["frozen"].classification == VIOLATING
    # Past a tick plus the scrape timeout with nothing new: stale.
    stale = {r.spec.trigger_id: r.evaluation for r in engine.tick(600 * S, history)}
    assert stale["failures"].classification == VIOLATING
    assert stale["share"].classification == VIOLATING
    assert stale["frozen"].classification == DATA_GAP
    assert all(e.reasons == ("no_recent_scrape",) for e in stale.values())
