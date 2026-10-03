"""The fake engine's /metrics: vLLM 0.30.0's catalog, and each mechanism's counter."""

from __future__ import annotations

from pathlib import Path

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.vllm_metrics import (
    CompactScrape,
    compact_scrape,
    discover,
    parse_prometheus_text,
    process_start_ns,
)
from tests.qualification_fake_engine_helpers import (
    chat,
    chats_in_background,
    get,
    join_all,
    wait_until,
    words,
)

FAST = FakeEngineConfig(step_seconds=0.001, decode_token_seconds=0.0001)


def _scrape(engine: FakeEngine) -> CompactScrape:
    status, body = get(engine.metrics_url)
    assert status == 200
    return compact_scrape(parse_prometheus_text(body.decode()))


def _value(scrape: CompactScrape, name: str) -> float:
    (value,) = scrape.series(name).values()
    assert isinstance(value, float)
    return value


def test_every_series_is_in_stormlogs_vllm_030_catalog() -> None:
    with FakeEngine(FAST) as engine:
        chat(engine, words(10, "a"), max_tokens=3)
        scrape = _scrape(engine)
    discovery = discover(scrape)
    assert discovery.unknown == ()
    assert discovery.deprecated_present == discovery.removed_present == ()
    assert discovery.engines == ("0",)
    assert process_start_ns(scrape) is not None
    for name in (
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_hits_total",
        "vllm:cache_config_info",
        "vllm:time_to_first_token_seconds",
        "vllm:request_queue_time_seconds",
    ):
        assert name in discovery.present
    (set_id,) = scrape.series("vllm:cache_config_info")
    assert scrape.labels(set_id)["num_gpu_blocks"] == "256"
    assert _value(scrape, "vllm:generation_tokens_total") == 3.0


def test_a_saturated_queue_shows_in_the_waiting_gauge() -> None:
    config = FakeEngineConfig(
        step_seconds=0.001, decode_token_seconds=0.01, max_num_seqs=1
    )
    with FakeEngine(config) as engine:
        threads = chats_in_background(
            engine, [words(4, tag) for tag in "abcd"], max_tokens=6
        )
        seen = wait_until(
            lambda: _value(_scrape(engine), "vllm:num_requests_waiting") >= 2
        )
        join_all(threads)
    assert seen


def test_kv_pressure_shows_in_the_preemption_counter() -> None:
    config = FakeEngineConfig(
        step_seconds=0.001, num_gpu_blocks=6, block_size=4, max_num_seqs=4
    )
    with FakeEngine(config) as engine:
        engine.pause_engine()
        threads = chats_in_background(
            engine, [words(4, tag) for tag in "abc"], max_tokens=8
        )
        assert wait_until(lambda: len(engine.engine.waiting) == 3)
        engine.resume_engine()
        join_all(threads)
        scrape = _scrape(engine)
    assert _value(scrape, "vllm:num_preemptions_total") > 0


def test_a_shared_prefix_shows_in_the_hit_counter() -> None:
    shared = words(40, "shared")
    with FakeEngine(FAST) as engine:
        chat(engine, shared + " one", max_tokens=1)
        before = _value(_scrape(engine), "vllm:prefix_cache_hits_total")
        chat(engine, shared + " two", max_tokens=1)
        after = _scrape(engine)
    assert _value(after, "vllm:prefix_cache_hits_total") - before >= 32
    assert _value(after, "vllm:prompt_tokens_cached_total") >= 32


def test_metrics_stay_at_the_last_step_while_the_loop_is_paused() -> None:
    with FakeEngine(FAST) as engine:
        chat(engine, words(4, "a"), max_tokens=1)
        engine.pause_engine()
        before = _scrape(engine)
        threads = chats_in_background(engine, [words(4, "b"), words(4, "c")])
        assert wait_until(lambda: len(engine.engine.waiting) == 2)
        during = _scrape(engine)
        engine.resume_engine()
        join_all(threads)
    assert _value(during, "vllm:num_requests_waiting") == 0.0
    assert during.series("vllm:generation_tokens_total") == before.series(
        "vllm:generation_tokens_total"
    )


def test_metrics_can_fail_or_answer_slowly() -> None:
    with FakeEngine(FAST) as engine:
        engine.controls.metrics_mode = "fail"
        failed, _ = get(engine.metrics_url)
        engine.controls.metrics_mode = "slow"
        engine.controls.metrics_delay_seconds = 0.05
        slow, _ = get(engine.metrics_url)
    assert (failed, slow) == (500, 200)


def test_a_profile_scrapes_and_resolves_the_fake_metrics(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    with FakeEngine(FAST) as engine:
        config = ProfileConfig(
            endpoint=engine.endpoint,
            model=engine.config.model,
            concurrency=(2,),
            input_tokens=(32,),
            output_tokens=(4,),
            output_path=str(output),
            request_count=6,
            tokenizer="none",
            system_sampler="none",
            run_id="run-1",
            prompt_mode="unique",
            vllm_metrics_url=engine.metrics_url,
            vllm_metrics_interval_seconds=0.1,
        )
        report = InferenceProfiler(config).run()
    vllm = report["telemetry"]["vllm"]
    (case,) = vllm["cases"].values()
    counters = case["engines"]["0"]["counters"]
    assert vllm["status"] == "collected"
    assert case["state"] == "resolved"
    assert counters["generation_tokens"]["delta"] == 24.0
