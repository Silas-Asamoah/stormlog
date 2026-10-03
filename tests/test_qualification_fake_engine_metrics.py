"""The fake engine's /metrics: vLLM 0.30.0's catalog, and each mechanism's counter."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.vllm_metrics import (
    CompactScrape,
    HistogramValue,
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
# A real vLLM 0.30.0 page (Qwen2.5-0.5B on one A30) after a run.
REAL_PAGE = Path(__file__).parent / "fixtures" / "vllm" / "q05_c08_metrics_post.txt"


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
        prompts = sum(request.prompt_len for request in engine.engine.finished)
    assert _value(scrape, "vllm:num_preemptions_total") > 0
    # A resumed request's lookups are not exported, so each request is queried
    # once, at its first admission.
    assert _value(scrape, "vllm:prefix_cache_queries_total") == prompts


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


def _shape(text: str) -> tuple[dict[str, str], dict[str, set[tuple[str, ...]]]]:
    """vLLM's families on a page: each one's type, and each sample name's
    label-name sets with every ``le`` bucket spelled out."""
    types: dict[str, str] = {}
    labels: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for line in text.splitlines():
        if line.startswith("# TYPE vllm:"):
            _hash, _type, name, kind = line.split()
            types[name] = kind
        elif line.startswith("vllm:"):
            name, _brace, rest = line.partition("{")
            pairs = re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', rest.rpartition("}")[0])
            names = sorted(key for key, _value in pairs)
            buckets = [f"le={value}" for key, value in pairs if key == "le"]
            labels[name].add(tuple(names + buckets))
    return types, dict(labels)


def test_the_page_has_the_real_pages_families_labels_and_buckets() -> None:
    with FakeEngine(FAST) as engine:
        chat(engine, words(10, "a"), max_tokens=3)
        status, body = get(engine.metrics_url)
    assert status == 200
    fake_types, fake_labels = _shape(body.decode())
    real_types, real_labels = _shape(REAL_PAGE.read_text())
    assert fake_types == real_types
    assert fake_labels == real_labels


def _by(scrape: CompactScrape, name: str, label: str) -> dict[str, float]:
    return {
        scrape.labels(set_id)[label]: value
        for set_id, value in scrape.series(name).items()
        if isinstance(value, float)
    }


def _histogram(scrape: CompactScrape, name: str) -> tuple[float, float]:
    (value,) = scrape.series(name).values()
    assert isinstance(value, HistogramValue)
    assert value.count is not None and value.sum is not None
    return value.count, value.sum


def test_prompt_sources_and_request_parameters_count_as_in_vllm() -> None:
    # vLLM 0.30's loggers.py: prompt tokens split by source at each first
    # token; per finished request, its prompt less its cached tokens, its n,
    # its max_tokens and the tokens it generated.
    shared = words(40, "shared")
    with FakeEngine(FAST) as engine:
        chat(engine, shared + " one", max_tokens=2)
        chat(engine, shared + " two", max_tokens=3)
        scrape = _scrape(engine)
        finished = list(engine.engine.finished)
    prompt = sum(request.prompt_len for request in finished)
    cached = sum(request.cached_at_admission or 0 for request in finished)
    assert cached >= 32
    assert _by(scrape, "vllm:prompt_tokens_by_source_total", "source") == {
        "local_compute": prompt - cached,
        "local_cache_hit": cached,
        "external_kv_transfer": 0.0,
    }
    assert _histogram(scrape, "vllm:request_prefill_kv_computed_tokens") == (
        2,
        prompt - cached,
    )
    assert _histogram(scrape, "vllm:request_params_n") == (2, 2)
    assert _histogram(scrape, "vllm:request_params_max_tokens") == (2, 5)
    assert _histogram(scrape, "vllm:request_max_num_generation_tokens") == (2, 5)
    assert _by(scrape, "vllm:engine_sleep_state", "sleep_state") == {
        "awake": 1.0,
        "weights_offloaded": 0.0,
        "discard_all": 0.0,
    }
