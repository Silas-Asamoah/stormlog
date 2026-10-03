"""The fake vLLM engine: scheduling, KV blocks, the prefix cache and its routes."""

from __future__ import annotations

import json
import re
import threading
import time
from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.fake_engine.blocks import BlockPool
from examples.qualification.fake_engine.engine import (
    TEMPLATE_TOKENS,
    Engine,
    EngineObserver,
    FakeRequest,
    Step,
    prompt_tokens,
)
from examples.qualification.fake_engine.hook_log import HookLog
from tests.qualification_fake_engine_helpers import (
    chat,
    chats_in_background,
    get,
    in_threads,
    join_all,
    post,
    wait_until,
    words,
)

FAST = FakeEngineConfig(step_seconds=0.001, decode_token_seconds=0.0001)


def test_a_streamed_completion_carries_its_tokens_and_usage() -> None:
    with FakeEngine(FAST) as engine:
        chunks = chat(engine, words(20, "a"), max_tokens=5, request_id="stormlog-r-0")
    content = [
        choice["delta"].get("content")
        for chunk in chunks
        for choice in chunk["choices"]
        if choice["delta"].get("content")
    ]
    assert content == [" tok"] * 5
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 24,
        "completion_tokens": 5,
        "total_tokens": 29,
    }
    reasons = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert reasons[-1] == "length"


def test_a_whole_completion_answers_once() -> None:
    with FakeEngine(FAST) as engine:
        (body,) = chat(engine, words(8, "b"), max_tokens=3, stream=False)
    assert body["choices"][0]["message"]["content"] == " tok" * 3
    assert body["usage"]["completion_tokens"] == 3


def test_request_ids_follow_vllm_030() -> None:
    with FakeEngine(FAST) as engine:
        chat(engine, "x", request_id="stormlog-run-1-c1_0")
        (request,) = engine.engine.finished
    assert request.external_id == "chatcmpl-stormlog-run-1-c1_0"
    assert re.fullmatch(
        r"chatcmpl-stormlog-run-1-c1_0-[0-9a-f]{8}", request.internal_id
    )

    config = FakeEngineConfig(step_seconds=0.001, request_id_randomization=False)
    with FakeEngine(config) as engine:
        chat(engine, "x", request_id="stormlog-run-1-c1_0")
        (request,) = engine.engine.finished
    assert request.internal_id == request.external_id


def test_requests_beyond_max_num_seqs_wait_in_arrival_order() -> None:
    config = FakeEngineConfig(step_seconds=0.001, max_num_seqs=1)
    with FakeEngine(config) as engine:
        engine.pause_engine()
        threads = chats_in_background(
            engine, [words(4, tag) for tag in "pqr"], max_tokens=3
        )
        assert wait_until(lambda: len(engine.engine.waiting) == 3)
        engine.resume_engine()
        join_all(threads)
        steps = list(engine.engine.steps)
        finished = list(engine.engine.finished)
    assert len(finished) == 3
    assert max(len(step.members) for step in steps) == 1
    by_arrival = sorted(finished, key=lambda request: request.arrival_ns)
    scheduled = sorted(finished, key=lambda request: request.first_scheduled_ns or 0)
    assert [r.internal_id for r in by_arrival] == [r.internal_id for r in scheduled]


def test_kv_exhaustion_preempts_the_newest_running_request() -> None:
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
        log = list(engine.engine.preemption_log)
        finished = list(engine.engine.finished)
        preempted_in_steps = [i for step in engine.engine.steps for i in step.preempted]
        preemptions = engine.engine.stats.preemptions
    assert log, "the KV budget never ran out"
    assert all(victim == order[-1] for victim, order in log)
    assert preemptions == len(log) == len(preempted_in_steps)
    assert all(request.output_tokens == 8 for request in finished)
    assert any(request.preemptions for request in finished)


def test_a_shared_prefix_is_reused_until_unique_traffic_evicts_it() -> None:
    config = FakeEngineConfig(step_seconds=0.001, num_gpu_blocks=24, block_size=4)
    shared = words(32, "shared")
    with FakeEngine(config) as engine:
        chat(engine, shared + " one", max_tokens=1)
        chat(engine, shared + " two", max_tokens=1)
        first, second = engine.engine.finished
        assert (first.cached_at_admission or 0) <= len(TEMPLATE_TOKENS)
        assert (second.cached_at_admission or 0) >= 32
        hits_before = engine.engine.stats.prefix_hits
        for index in range(4):
            chat(engine, words(60, f"u{index}x"), max_tokens=1)
        chat(engine, shared + " three", max_tokens=1)
        third = engine.engine.finished[-1]
        evictions = engine.engine.pool.evictions
    assert hits_before >= 32
    assert evictions > 0
    # Only the chat template's block, which every request shares, survives.
    assert (third.cached_at_admission or 0) <= len(TEMPLATE_TOKENS)


def test_a_cached_prompt_still_computes_its_last_token() -> None:
    pool = BlockPool(8, 4, caching=True)
    hashes = pool.block_hashes([str(i) for i in range(8)])
    taken = pool.allocate(2)
    assert taken is not None
    for block_id, block_hash in zip(taken, hashes):
        pool.cache(block_id, block_hash)
    pool.free(taken)
    assert pool.lookup(hashes, 8) == taken[:1]
    assert pool.lookup(hashes, 9) == taken


def test_a_hash_cached_twice_keeps_its_hit_while_either_copy_stays() -> None:
    # vLLM's BlockHashToBlockMap keeps every block cached under a hash, and an
    # eviction removes only the block it reuses (core/block_pool.py:33-120).
    pool = BlockPool(2, 4, caching=True)
    taken = pool.allocate(2)
    assert taken is not None
    first, second = taken
    (block_hash,) = pool.block_hashes(["x"] * 4)
    pool.cache(first, block_hash)
    pool.cache(second, block_hash)
    pool.free([first])
    assert pool.allocate(1) == [first]
    assert pool.lookup([block_hash], 5) == [second]


def test_descriptive_routes_answer_like_vllm() -> None:
    with FakeEngine(FAST) as engine:
        version = json.loads(get(f"{engine.base_url}/version")[1])
        models = json.loads(get(f"{engine.base_url}/v1/models")[1])
        info = json.loads(get(f"{engine.base_url}/server_info?config_format=json")[1])
        text_info = json.loads(get(f"{engine.base_url}/server_info")[1])
        health, _ = get(f"{engine.base_url}/health")
        missing, _ = get(f"{engine.base_url}/nope")
    assert version == {"version": "0.30.0"}
    assert models["data"][0]["id"] == FAST.model
    assert info["vllm_config"]["cache_config"]["num_gpu_blocks"] == 256
    assert info["vllm_config"]["scheduler_config"]["max_num_seqs"] == 16
    assert isinstance(text_info["vllm_config"], str)
    assert (health, missing) == (200, 404)


@pytest.mark.parametrize("tokens", [1, 3])
def test_every_request_finishes_with_its_cap(tokens: int) -> None:
    with FakeEngine(FAST) as engine:
        in_threads(
            [partial(chat, engine, words(6, tag), max_tokens=tokens) for tag in "mnop"]
        )
        finished = list(engine.engine.finished)
    assert len(finished) == 4
    assert {request.output_tokens for request in finished} == {tokens}
    assert {request.finish_reason for request in finished} == {"length"}


def _reset(engine: FakeEngine, query: str = "") -> object:
    status, body = post(f"{engine.base_url}/reset_prefix_cache{query}")
    assert status == 200
    return json.loads(body)["success"]


def test_an_idle_reset_forgets_every_cached_prefix() -> None:
    shared = words(40, "shared")
    with FakeEngine(FAST) as engine:
        chat(engine, shared + " one", max_tokens=1)
        assert _reset(engine) is True
        chat(engine, shared + " two", max_tokens=1)
        second = engine.engine.finished[-1]
    assert (second.cached_at_admission or 0) == 0


def test_a_reset_is_refused_while_blocks_are_held_unless_it_preempts() -> None:
    config = FakeEngineConfig(step_seconds=0.001, decode_token_seconds=0.002)
    with FakeEngine(config) as engine:
        (thread,) = chats_in_background(engine, [words(8, "a")], max_tokens=200)
        assert wait_until(lambda: bool(engine.engine.running))
        refused = _reset(engine)
        preempting = _reset(engine, "?reset_running_requests=true")
        join_all([thread])
        (request,) = engine.engine.finished
    assert (refused, preempting) == (False, True)
    assert request.preemptions == 1
    assert request.output_tokens == 200


def test_resets_run_between_steps_and_never_drop_a_steps_output() -> None:
    # vLLM runs a reset as a utility call on the engine loop, between steps, so
    # with synchronous scheduling no step's output is ever dropped as stale.
    config = FakeEngineConfig(step_seconds=0.02, decode_token_seconds=0.001)
    with FakeEngine(config) as engine:
        threads = chats_in_background(
            engine, [words(6, tag) for tag in "ab"], max_tokens=40
        )
        assert wait_until(lambda: len(engine.engine.running) == 2)
        for _ in range(20):
            assert _reset(engine, "?reset_running_requests=true") is True
            time.sleep(0.007)
        join_all(threads)
        outcomes = {
            member.outcome for step in engine.engine.steps for member in step.members
        }
        finished = list(engine.engine.finished)
    assert outcomes == {"kept"}
    assert {request.output_tokens for request in finished} == {40}


def test_a_resets_victims_are_listed_in_the_next_steps_preempted() -> None:
    # vLLM keeps them in reset_preempted_req_ids, which the next
    # SchedulerOutput carries (scheduler.py:1416, 1518), as the hook records.
    config = FakeEngineConfig(step_seconds=0.005, decode_token_seconds=0.001)
    with FakeEngine(config) as engine:
        threads = chats_in_background(
            engine, [words(6, tag) for tag in "ab"], max_tokens=30
        )
        assert wait_until(lambda: len(engine.engine.running) == 2)
        steps_before = len(engine.engine.steps)
        victims = sorted(request.internal_id for request in engine.engine.running)
        assert _reset(engine, "?reset_running_requests=true") is True
        join_all(threads)
        later = engine.engine.steps[steps_before:]
    listed = [step for step in later if step.preempted]
    assert listed, "no step listed the reset's victims"
    assert sorted(listed[0].preempted) == victims


def _stepped_engine(
    *, max_num_batched_tokens: int = 256, enable_prefix_caching: bool = True
) -> Engine:
    """An engine whose steps a test drives by hand: its loop never starts."""
    config = FakeEngineConfig(
        block_size=4,
        num_gpu_blocks=64,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    return Engine(config)


def _request(engine: Engine, name: str, prompt: str, max_tokens: int) -> FakeRequest:
    return engine.submit(
        FakeRequest(
            internal_id=f"{name}-0a1b2c3d",
            external_id=name,
            prompt=prompt_tokens(prompt),
            max_tokens=max_tokens,
            arrival_ns=time.time_ns(),
        )
    )


def _step(engine: Engine) -> Step:
    step = engine._schedule()
    assert step is not None
    engine._complete(step)
    return step


def _preempt(engine: Engine, request: FakeRequest) -> None:
    with engine._lock:
        engine.running.remove(request)
        engine._preempt(request, [])


def test_a_resumed_request_reuses_its_own_generated_blocks() -> None:
    # vLLM hashes and caches every full block, generated tokens included, so a
    # request resumed after preemption hits blocks past its prompt.
    engine = _stepped_engine()
    request = _request(engine, "r", words(6, "p"), max_tokens=20)
    while request.output_tokens < 9:
        _step(engine)
    _preempt(engine, request)
    step = engine._schedule()
    assert step is not None
    (member,) = step.members
    assert request.prompt_len == 10
    assert member.computed_before == 16


def test_the_first_step_after_a_resume_is_context() -> None:
    # vLLM puts a resumed request in the output's new requests, so the hook's
    # phase is context even when only one token is left to compute: all 23
    # recompute memberships in the real preempt5 log are context.
    engine = _stepped_engine()
    request = _request(engine, "r", words(6, "p"), max_tokens=20)
    while request.output_tokens < 7:
        _step(engine)
    _preempt(engine, request)
    resumed = _step(engine)
    following = _step(engine)
    (first,) = resumed.members
    (second,) = following.members
    assert (first.computed_before, request.prompt_len + 6) == (16, 16)
    assert first.tokens == 1
    assert (first.context, second.context) == (True, False)


@pytest.mark.parametrize(("budget", "sampled"), [(256, 1), (4, 0)])
def test_a_member_aborted_mid_step_keeps_the_samplers_count(
    budget: int, sampled: int
) -> None:
    # vLLM 0.30 applies aborts that arrive during execution before the update
    # (core.py:611-613), so the member is discarded_finished, but the hook
    # still counts what the sampler produced: one token unless the step was a
    # partial prefill.
    engine = _stepped_engine(max_num_batched_tokens=budget)
    request = _request(engine, "r", words(6, "p"), max_tokens=20)
    step = engine._schedule()
    assert step is not None
    engine.abort(request)
    engine._complete(step)
    (member,) = step.members
    assert (member.outcome, member.sampled) == ("discarded_finished", sampled)


def test_a_finish_leaves_one_empty_step_as_in_vllm() -> None:
    # vLLM's has_requests() counts requests finished since the last schedule
    # (scheduler.py:2659-2680), so step() schedules once more after the last
    # finish (core.py:598-600), and the hook records a step with no members.
    engine = _stepped_engine()
    _request(engine, "r", words(6, "p"), max_tokens=1)
    _step(engine)
    iterations = engine.stats.histograms["vllm:iteration_tokens_total"]
    observed = iterations.count
    empty = _step(engine)
    assert (empty.members, empty.preempted, empty.total_tokens) == ([], [], 0)
    # It has no request outputs, so the front end records no iteration for it
    # (async_llm.py:792-793).
    assert iterations.count == observed
    assert engine._schedule() is None


def test_the_loop_runs_the_empty_step_after_the_last_finish() -> None:
    with FakeEngine(FAST) as engine:
        chat(engine, words(6, "a"), max_tokens=2)
        assert wait_until(lambda: not engine.engine.steps[-1].members)
        time.sleep(0.05)
        steps = list(engine.engine.steps)
    assert [len(step.members) for step in steps] == [1, 1, 0]


def test_iteration_tokens_count_outputs_as_vllms_front_end_does() -> None:
    # vLLM 0.30 records an iteration only for an output that carries tokens
    # (async_llm.py:792-793) and observes the prompt tokens computed for the
    # requests whose first token it carries, plus the tokens generated
    # (metrics/loggers.py, PrometheusStatLogger.record).
    engine = _stepped_engine(max_num_batched_tokens=4)
    request = _request(engine, "r", words(6, "p"), max_tokens=2)
    iterations = engine.stats.histograms["vllm:iteration_tokens_total"]
    chunks = [_step(engine), _step(engine)]
    assert iterations.count == 0
    _step(engine)
    _step(engine)
    assert [member.tokens for step in chunks for member in step.members] == [4, 4]
    assert request.prompt_len == 10
    assert (iterations.count, iterations.total) == (2, 10 + 1 + 1)


class _AdmissionWitness(EngineObserver):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.schedulable_at_admit: list[bool] = []

    def on_admit(self, request: FakeRequest) -> None:
        with self.engine._lock:
            self.schedulable_at_admit.append(request in self.engine.waiting)


def test_a_request_is_admitted_before_the_loop_can_schedule_it() -> None:
    # The hook writes alias before vLLM's preprocess_add_request hands the
    # request to the scheduler, so no scheduled record can precede it; an
    # importer reads a use before the alias as an earlier execution.
    engine = _stepped_engine()
    witness = _AdmissionWitness(engine)
    engine.observers.append(witness)
    _request(engine, "r", words(6, "p"), max_tokens=1)
    assert witness.schedulable_at_admit == [False]


def _calls(actions: list[Callable[[], object]]) -> list[str]:
    """Run every action at once; the errors any of them raised."""
    errors: list[str] = []

    def run(action: Callable[[], object]) -> None:
        try:
            action()
        except Exception as error:  # every failure is the finding
            errors.append(repr(error))

    threads = [threading.Thread(target=run, args=(action,)) for action in actions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    return errors


def test_a_burst_of_connections_is_never_reset() -> None:
    # The stdlib server listens with a backlog of 5, so a burst of connects
    # overflowed it and the client saw "Connection reset by peer"; vLLM's
    # uvicorn listens with 2048.
    with FakeEngine(FAST) as engine:
        health = partial(get, f"{engine.base_url}/health")
        errors = [error for _ in range(3) for error in _calls([health] * 100)]
        server_errors = list(engine.server_errors)
    assert (errors, server_errors) == ([], [])


def test_resets_while_requests_run_break_no_connection() -> None:
    config = FakeEngineConfig(
        step_seconds=0.0005, decode_token_seconds=0.0001, max_num_seqs=64
    )
    with FakeEngine(config) as engine:
        reset = partial(
            post, f"{engine.base_url}/reset_prefix_cache?reset_running_requests=true"
        )
        chats = [
            partial(chat, engine, words(6, f"r{index}"), max_tokens=200)
            for index in range(8)
        ]
        errors = _calls([*chats, *[reset] * 40])
        finished = list(engine.engine.finished)
        server_errors = list(engine.server_errors)
    assert (errors, server_errors) == ([], [])
    assert {request.output_tokens for request in finished} == {200}


def test_a_reset_by_hand_lists_its_victims_in_the_next_scheduled_record(
    tmp_path: Path,
) -> None:
    # An engine driven by hand has no loop to run the reset on, so it runs at
    # once; the victims still reach the hook's next scheduled record.
    engine = _stepped_engine()
    hook = HookLog(tmp_path / "hook", engine.config)
    engine.observers.append(hook)
    try:
        request = _request(engine, "r", words(6, "p"), max_tokens=20)
        step = _step(engine)
        hook.on_scheduled(step)
        assert engine.reset_prefix_cache(True)
        following = _step(engine)
        hook.on_scheduled(following)
    finally:
        hook.close()
    records = [
        json.loads(line)
        for path in (tmp_path / "hook").rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    preempted = [
        internal
        for record in records
        if record["kind"] == "scheduled"
        for internal in record["preempted"]
    ]
    assert preempted == [request.internal_id]


def test_a_resumed_requests_lookup_stays_out_of_the_exported_prefix_counters() -> None:
    # vLLM records a lookup at admission, a preempted request's under
    # preempted_queries/preempted_hits, which /metrics does not export
    # (kv_cache_manager.record_prefix_cache_stats, PrefixCacheStats.record).
    engine = _stepped_engine()
    request = _request(engine, "r", words(6, "p"), max_tokens=20)
    while request.output_tokens < 9:
        _step(engine)
    exported = (engine.stats.prefix_queries, engine.stats.prefix_hits)
    _preempt(engine, request)
    _step(engine)
    assert (engine.stats.prefix_queries, engine.stats.prefix_hits) == exported
    assert (
        engine.stats.preempted_prefix_queries,
        engine.stats.preempted_prefix_hits,
    ) == (
        request.prompt_len + 9,
        16,
    )


def test_no_prefix_lookup_is_counted_with_caching_off() -> None:
    # vLLM skips the lookup, and its record, when prefix caching is disabled.
    engine = _stepped_engine(enable_prefix_caching=False)
    _request(engine, "r", words(6, "p"), max_tokens=1)
    _step(engine)
    assert (engine.stats.prefix_queries, engine.stats.prefix_hits) == (0, 0)


def test_the_queue_time_is_observed_when_the_request_finishes() -> None:
    # vLLM observes every per-request histogram, queue time included, from its
    # finished requests (metrics/loggers.py, PrometheusStatLogger.record).
    engine = _stepped_engine()
    request = _request(engine, "r", words(6, "p"), max_tokens=3)
    queue = engine.stats.histograms["vllm:request_queue_time_seconds"]
    _step(engine)
    assert (request.output_tokens, queue.count) == (1, 0)
    while not request.finished:
        _step(engine)
    assert queue.count == 1


def test_a_client_abort_records_no_request_statistics() -> None:
    # A client that goes away aborts its request in vLLM's output processor,
    # which drops the request's state without updating finished-request stats
    # (OutputProcessor.abort_requests); the engine core then frees it with no
    # output to report.
    engine = _stepped_engine()
    request = _request(engine, "r", words(6, "p"), max_tokens=20)
    _step(engine)
    engine.abort(request)
    _step(engine)
    histograms = engine.stats.histograms
    assert request.finish_reason == "abort"
    assert sum(engine.stats.success.values()) == 0
    assert histograms["vllm:e2e_request_latency_seconds"].count == 0
    assert histograms["vllm:request_generation_tokens"].count == 0


def test_a_one_token_completion_adds_a_zero_time_per_output_token() -> None:
    # vLLM's mean time per output token is 0 for a single token, and it is
    # observed for every finished request (metrics/stats.py, loggers.py).
    engine = _stepped_engine()
    request = _request(engine, "r", words(6, "p"), max_tokens=1)
    _step(engine)
    tpot = engine.stats.histograms["vllm:request_time_per_output_token_seconds"]
    assert request.finished
    assert (tpot.count, tpot.total) == (1, 0.0)
