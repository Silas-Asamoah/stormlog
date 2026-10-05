"""The watcher's bounded history: byte and age bounds, and scrape round trips."""

from __future__ import annotations

import gc
import json
import tracemalloc
import zlib
from pathlib import Path

import pytest

from stormlog.infer.vllm_telemetry import VllmScrapeRecord
from stormlog.infer.watch import history as history_module
from stormlog.infer.watch.history import (
    EVICT_AGE,
    EVICT_BYTES,
    EVICT_OVERSIZED,
    BoundedRing,
    ScrapeHistory,
    Stamped,
)
from tests.vllm_scrape_helpers import exposition, scrape

S = 1_000_000_000
FIXTURES = Path(__file__).parent / "fixtures" / "vllm"


def _stamp(second: float) -> Stamped:
    mono = round(second * S)
    return Stamped(mono, mono + 1, mono)


def test_stamps_must_finish_after_they_start() -> None:
    with pytest.raises(ValueError):
        Stamped(10, 5, 10)


def test_the_byte_bound_counts_the_compressed_bytes_and_overhead_held() -> None:
    ring = BoundedRing(max_seconds=3600, max_bytes=2000)
    for second in range(100):
        ring.append(_stamp(second), {"seq": second, "pad": f"{second:08d}" * 40})
    assert ring.bytes <= 2000
    assert ring.evictions[EVICT_BYTES] == 100 - len(ring)
    held = [record["seq"] for _stamp, record in ring.items()]
    assert held == list(range(100 - len(ring), 100))  # the newest are kept


def test_items_older_than_the_age_bound_are_evicted() -> None:
    ring = BoundedRing(max_seconds=10, max_bytes=1 << 20)
    for second in range(30):
        ring.append(_stamp(second), {"seq": second})
    assert [r["seq"] for _s, r in ring.items()] == list(range(19, 30))
    assert ring.evictions[EVICT_AGE] == 19
    assert ring.held_seconds() == 10.0
    ring.expire(100 * S)
    assert len(ring) == 0 and ring.bytes == 0


def test_an_item_over_the_byte_bound_alone_is_refused() -> None:
    ring = BoundedRing(max_seconds=10, max_bytes=64)
    assert not ring.append(_stamp(0), {"pad": "x" * 4096 + str(range(500))})
    assert ring.evictions[EVICT_OVERSIZED] == 1 and len(ring) == 0


def test_items_are_selected_by_their_monotonic_stamp() -> None:
    ring = BoundedRing(max_seconds=3600, max_bytes=1 << 20)
    for second in range(10):
        ring.append(_stamp(second), {"seq": second})
    window = [r["seq"] for _s, r in ring.items(3 * S, 6 * S)]
    assert window == [3, 4, 5, 6]


def test_scrape_history_keeps_a_parsed_tail_and_round_trips_the_rest() -> None:
    history = ScrapeHistory(max_seconds=600, max_bytes=1 << 20, parsed_count=3)
    text = exposition(gauges={"vllm:num_requests_waiting": 4})
    for second in range(10):
        history.add(_stamp(second), scrape(text, second))
    assert [s.mono_ns for s, _r in history.parsed()] == [7 * S, 8 * S, 9 * S]
    records = list(history.records(2 * S, 4 * S))
    assert [r.observed_at_ns for _s, r in records] == [
        scrape(text, second).observed_at_ns for second in (2, 3, 4)
    ]
    assert records[0][1].scrape == scrape(text, 2).scrape
    with pytest.raises(ValueError):
        ScrapeHistory(max_seconds=1, max_bytes=1, parsed_count=1)


@pytest.mark.parametrize(
    ("max_seconds", "max_bytes", "cause"),
    [(600, 1400, EVICT_BYTES), (2, 1 << 20, EVICT_AGE)],
)
def test_ring_evictions_remove_the_same_scrapes_from_the_parsed_tail(
    max_seconds: float, max_bytes: int, cause: str
) -> None:
    history = ScrapeHistory(
        max_seconds=max_seconds, max_bytes=max_bytes, parsed_count=5
    )
    text = exposition(gauges={"vllm:num_requests_waiting": 4})
    for second in range(10):
        assert history.add(_stamp(second), scrape(text, second))
        retained = list(history.records())
        assert history.parsed() == retained[-5:]
    assert history.ring.evictions[cause] > 0


def test_explicit_ring_expiry_removes_the_expired_parsed_scrapes() -> None:
    history = ScrapeHistory(max_seconds=2, max_bytes=1 << 20, parsed_count=5)
    text = exposition(gauges={"vllm:num_requests_waiting": 4})
    for second in range(3):
        history.add(_stamp(second), scrape(text, second))
    history.ring.expire(3 * S)
    assert history.parsed() == list(history.records())
    assert [stamp.mono_ns for stamp, _record in history.parsed()] == [S, 2 * S]
    history.ring.expire(5 * S)
    assert history.parsed() == []
    assert list(history.records()) == []


def _real_scrape() -> VllmScrapeRecord:
    text = (FIXTURES / "q05_c08_scrape_record_eb5ad3f.json").read_text()
    return VllmScrapeRecord.from_record(json.loads(text))


def test_the_byte_bound_bounds_what_the_ring_retains() -> None:
    # A tiny item compresses to about 20 bytes, while the objects holding it
    # take hundreds more: counting only the compressed bytes let a 256 KiB
    # ring retain 6 MiB.
    bound = 256 * 1024
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        ring = BoundedRing(max_seconds=1e6, max_bytes=bound)
        for second in range(20_000):
            mono = 10**15 + second * S
            ring.append(Stamped(mono, mono + 1, 1_790_000_000 * S + mono), {"ok": 1})
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert ring.evictions[EVICT_BYTES] > 0
    assert ring.bytes <= bound
    assert retained <= ring.bytes  # what it holds, not only what it compressed
    overhead = history_module.ITEM_OVERHEAD_BYTES
    assert ring.bytes == len(ring) * (overhead + len(_blob({"ok": 1})))


def _blob(record: dict[str, int]) -> bytes:
    return zlib.compress(json.dumps(record, separators=(",", ":")).encode(), 1)


def test_a_scrape_the_ring_refuses_is_not_kept_parsed_either() -> None:
    # A trigger judges the parsed tail and an incident's pre-window comes from
    # the ring, so a scrape only one of them holds would make a firing the
    # bundle cannot show.
    record = _real_scrape()
    history = ScrapeHistory(max_seconds=600, max_bytes=4096, parsed_count=3)
    small = scrape(exposition(gauges={"vllm:num_requests_waiting": 4}), 0)
    assert history.add(_stamp(0), small)
    assert not history.add(_stamp(1), record)
    assert history.ring.evictions[EVICT_OVERSIZED] == 1
    assert [s.mono_ns for s, _r in history.parsed()] == [0]
    assert [s.mono_ns for s, _r in history.records()] == [0]


def _peak_consuming(history: ScrapeHistory, end_mono_ns: int) -> tuple[int, int]:
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        seen = 0
        for _stamp_, parsed in history.records(None, end_mono_ns):
            seen += parsed.scrape is not None
        return seen, tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def test_records_parses_one_scrape_at_a_time() -> None:
    # A parsed vLLM scrape is about 20 times its compressed size: building
    # them all at once took 75 MiB for a 3.6 MiB ring of 600 scrapes. Read
    # one at a time, the peak does not grow with the number read.
    record = _real_scrape()
    history = ScrapeHistory(max_seconds=600, max_bytes=64 << 20, parsed_count=2)
    for second in range(60):
        history.add(_stamp(second), record)
    seen_few, peak_few = _peak_consuming(history, 9 * S)
    seen_all, peak_all = _peak_consuming(history, 59 * S)
    assert (seen_few, seen_all) == (10, 60)
    assert peak_all < 2 * peak_few
