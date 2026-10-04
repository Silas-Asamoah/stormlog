"""Shared renders: built at most once per interval, at most three alive."""

import threading
import time
import tracemalloc

import pytest

from stormlog._export.renders import MAX_GENERATIONS, Generation, RenderCache


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _cache(size: int = 16, clock: _Clock | None = None) -> RenderCache:
    return RenderCache(lambda: bytes(size), min_interval=1.0, clock=clock or _Clock())


def test_readers_within_one_interval_share_one_render() -> None:
    clock = _Clock()
    cache = _cache(clock=clock)
    first = cache.acquire()
    second = cache.acquire()
    assert first is second and cache.stats.builds == 1
    clock.now = 1.0
    third = cache.acquire()
    assert third is not first and cache.stats.builds == 2


def test_staggered_slow_readers_never_keep_more_than_three_renders() -> None:
    # Astra's history: four scrapes, each starting on the next render tick and
    # holding its generation to the deadline, plus a textfile writer stuck in
    # a write, plus a render in progress.
    size = 1024 * 1024
    clock = _Clock()
    cache = RenderCache(lambda: bytes(size), min_interval=1.0, clock=clock)
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    stuck_writer = cache.acquire()
    held: list[Generation] = []
    for tick in range(1, 5):
        clock.now = float(tick)
        held.append(cache.acquire())
        assert cache.alive() <= MAX_GENERATIONS
    peak = tracemalloc.get_traced_memory()[1] - baseline
    tracemalloc.stop()
    assert peak <= MAX_GENERATIONS * size + 256 * 1024
    assert cache.stats.alive_high_water <= MAX_GENERATIONS
    assert cache.stats.deferred >= 1
    newest = held[-1]
    assert all(g is newest for g in held[2:])  # later readers reuse the newest
    # The scrapes reach their deadline; renders resume while the writer is stuck.
    for generation in held:
        cache.release(generation)
    clock.now = 10.0
    fresh = cache.acquire()
    assert fresh.number > newest.number and cache.alive() <= MAX_GENERATIONS
    cache.release(fresh)
    cache.release(stuck_writer)


def test_a_reader_that_waits_for_the_first_render_gets_it() -> None:
    started = threading.Event()

    def slow_render() -> bytes:
        started.set()
        time.sleep(0.1)
        return b"body"

    cache = RenderCache(slow_render)
    results: list[Generation] = []
    builder = threading.Thread(target=lambda: results.append(cache.acquire()))
    builder.start()
    started.wait(5)
    waiter = cache.acquire()
    builder.join(5)
    assert waiter is results[0] and cache.stats.builds == 1


def test_a_failed_render_reaches_its_reader_and_the_next_reader_retries() -> None:
    attempts = []

    def flaky() -> bytes:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("render failed")
        return b"ok"

    cache = RenderCache(flaky)
    with pytest.raises(RuntimeError):
        cache.acquire()
    assert cache.acquire().body == b"ok"
    assert cache.stats.failures == 1 and cache.stats.builds == 1


def test_an_unread_older_generation_is_dropped_at_once() -> None:
    clock = _Clock()
    cache = _cache(clock=clock)
    first = cache.acquire()
    cache.release(first)
    clock.now = 1.0
    second = cache.acquire()
    assert cache.alive() == 1 and second is not first


def test_an_invalidated_render_is_rebuilt_at_once() -> None:
    clock = _Clock()
    values = iter([b"before", b"after"])
    cache = RenderCache(lambda: next(values), min_interval=60.0, clock=clock)
    first = cache.acquire()
    cache.release(first)
    cache.invalidate()
    assert not cache.is_fresh(first)
    second = cache.acquire()
    assert second.body == b"after" and cache.is_fresh(second)


def test_the_limit_still_holds_after_an_invalidation() -> None:
    clock = _Clock()
    cache = _cache(clock=clock)
    held = [cache.acquire()]
    for tick in (1.0, 2.0):
        clock.now = tick
        held.append(cache.acquire())
    cache.invalidate()
    stale = cache.acquire()
    assert stale is held[-1] and not cache.is_fresh(stale)
    assert cache.alive() <= MAX_GENERATIONS


def _blocked_build(cache: RenderCache, results: list[object]) -> threading.Thread:
    def read() -> None:
        try:
            results.append(cache.acquire())
        except Exception as exc:
            results.append(exc)

    builder = threading.Thread(target=read)
    builder.start()
    return builder


def test_a_render_built_across_an_invalidation_is_stale_and_rebuilt() -> None:
    # Fable's race: the build took its values before they changed for good,
    # and published after the invalidation.
    values = {"total": 0}
    started, release = threading.Event(), threading.Event()

    def render() -> bytes:
        body = f"total {values['total']}".encode()
        started.set()
        release.wait(5)
        return body

    cache = RenderCache(render, min_interval=60.0, clock=_Clock())
    results: list[object] = []
    builder = _blocked_build(cache, results)
    assert started.wait(5)
    values["total"] = 7
    cache.invalidate()
    release.set()
    builder.join(5)
    (built,) = results
    assert isinstance(built, Generation) and built.body == b"total 0"
    assert not cache.is_fresh(built)
    cache.release(built)
    rebuilt = cache.acquire()
    assert rebuilt.body == b"total 7" and cache.is_fresh(rebuilt)


def test_a_failed_build_across_an_invalidation_leaves_the_next_build_fresh() -> None:
    started, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def render() -> bytes:
        calls.append(1)
        if len(calls) == 1:
            started.set()
            release.wait(5)
            raise RuntimeError("render failed")
        return b"after"

    cache = RenderCache(render, min_interval=60.0, clock=_Clock())
    results: list[object] = []
    builder = _blocked_build(cache, results)
    assert started.wait(5)
    cache.invalidate()
    release.set()
    builder.join(5)
    assert isinstance(results[0], RuntimeError)
    after = cache.acquire()
    assert after.body == b"after" and cache.is_fresh(after)


def test_a_held_back_render_is_counted_once_however_many_readers_ask() -> None:
    clock = _Clock()
    cache = _cache(clock=clock)
    held = [cache.acquire()]
    for tick in (1.0, 2.0):
        clock.now = tick
        held.append(cache.acquire())
    clock.now = 3.0  # due, but three generations are alive
    readers = [cache.acquire() for _ in range(5)]
    assert cache.stats.deferred == 1
    for generation in held + readers:
        cache.release(generation)
    clock.now = 4.0
    cache.acquire()
    assert cache.stats.deferred == 1 and cache.stats.builds == 4
