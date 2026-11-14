"""C4's document and selection caches: bounds, keys, once-only, threads."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from stormlog.studio.cache import Documents, cache_key
from stormlog.studio.contracts import (
    DocumentCache,
    Limits,
    RateScope,
    RequestFilter,
    Selection,
)

INDEX_ID = "9a3e5b7c1d2f4a6b8c0e2d4f6a8b0c1d"


def selection(generation: int = 0) -> Selection:
    return Selection(
        filter=RequestFilter(),
        params="phase=measured",
        fingerprint="sha256:" + "0" * 64,
        path="runs/042.jsonl",
        index_id=INDEX_ID,
        generation=generation,
        rows=(0, 1, 2),
        cases=("c1",),
        whole_case=None,
        rate=RateScope(eligible=False, rule=None, interval=None, reason="x"),
        searched_fields=(),
    )


def key(name: str, generation: int = 0, params: str = "") -> tuple[str, ...]:
    return cache_key(name, "runs/042.jsonl", INDEX_ID, generation, params)


def test_documents_implements_the_document_cache_protocol() -> None:
    cache: DocumentCache = Documents()
    assert cache.document(key("summary"), lambda: {"n": 1}) == {"n": 1}


def test_cache_key_names_path_index_generation_and_params() -> None:
    assert cache_key("summary", "a.jsonl", INDEX_ID, 7, "case=c1") == (
        "summary",
        "a.jsonl",
        INDEX_ID,
        "7",
        "case=c1",
    )
    with pytest.raises(ValueError):
        cache_key("", "a.jsonl", INDEX_ID, 7)


def test_a_document_is_made_once_and_then_served() -> None:
    cache = Documents()
    calls: list[int] = []

    def make() -> dict[str, Any]:
        calls.append(1)
        return {"value": len(calls)}

    first = cache.document(key("summary"), make)
    second = cache.document(key("summary"), make)
    assert first is second
    assert calls == [1]
    assert cache.stats()["documents"]["hits"] == 1


def test_another_generation_or_params_is_another_entry() -> None:
    cache = Documents()
    one = cache.document(key("summary", 1), lambda: {"g": 1})
    two = cache.document(key("summary", 2), lambda: {"g": 2})
    other = cache.document(key("summary", 2, "case=c1"), lambda: {"g": 3})
    assert (one, two, other) == ({"g": 1}, {"g": 2}, {"g": 3})


def test_documents_are_bounded_least_recently_used_first() -> None:
    cache = Documents(Limits(cached_documents=2))
    cache.document(key("a"), lambda: {"v": "a"})
    cache.document(key("b"), lambda: {"v": "b"})
    cache.document(key("a"), lambda: {"v": "stale"})  # a is now the newest
    cache.document(key("c"), lambda: {"v": "c"})  # evicts b
    assert cache.document(key("a"), lambda: {"v": "new"}) == {"v": "a"}
    assert cache.document(key("b"), lambda: {"v": "remade"}) == {"v": "remade"}
    assert cache.stats()["documents"]["size"] == 2


def test_selections_have_their_own_bound() -> None:
    cache = Documents(Limits(cached_selections=1, cached_documents=4))
    first = cache.selection(key("selection", 1), lambda: selection(1))
    cache.selection(key("selection", 2), lambda: selection(2))
    again = cache.selection(key("selection", 1), lambda: selection(1))
    assert again is not first
    assert cache.stats()["selections"] == {
        "size": 1,
        "capacity": 1,
        "hits": 0,
        "misses": 3,
    }


def test_a_failed_make_caches_nothing() -> None:
    cache = Documents()

    def boom() -> dict[str, Any]:
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        cache.document(key("summary"), boom)
    assert cache.document(key("summary"), lambda: {"ok": True}) == {"ok": True}


@pytest.mark.parametrize("bad", [(), ["a"], ("a", 1)])
def test_keys_are_non_empty_tuples_of_strings(bad: Any) -> None:
    with pytest.raises(TypeError):
        Documents().document(bad, dict)


def test_zero_capacity_is_refused() -> None:
    with pytest.raises(ValueError):
        Documents(Limits(cached_documents=0))


def test_concurrent_callers_share_one_make() -> None:
    cache = Documents()
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow() -> dict[str, Any]:
        calls.append(1)
        started.set()
        assert release.wait(10)
        return {"made": True}

    results: list[dict[str, Any]] = []

    def call() -> None:
        results.append(cache.document(key("summary"), slow))

    threads = [threading.Thread(target=call) for _ in range(8)]
    threads[0].start()
    assert started.wait(10)
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(10)
    assert calls == [1]
    assert len(results) == 8
    assert all(result is results[0] for result in results)


def test_waiters_get_the_makers_exception() -> None:
    cache = Documents()
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def failing() -> dict[str, Any]:
        started.set()
        assert release.wait(10)
        raise LookupError("gone")

    def call() -> None:
        try:
            cache.document(key("summary"), failing)
        except LookupError as error:
            errors.append(error)

    first = threading.Thread(target=call)
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=call)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(10)
    second.join(10)
    assert len(errors) == 2


def test_other_keys_do_not_wait_for_a_slow_make() -> None:
    cache = Documents()
    started = threading.Event()
    release = threading.Event()

    def slow() -> dict[str, Any]:
        started.set()
        assert release.wait(10)
        return {}

    thread = threading.Thread(target=cache.document, args=(key("slow"), slow))
    thread.start()
    assert started.wait(10)
    assert cache.document(key("fast"), lambda: {"fast": 1}) == {"fast": 1}
    release.set()
    thread.join(10)


def test_clear_drops_every_value() -> None:
    cache = Documents()
    cache.document(key("a"), lambda: {"v": 1})
    cache.selection(key("selection"), selection)
    cache.clear()
    assert cache.stats()["documents"]["size"] == 0
    assert cache.stats()["selections"]["size"] == 0
