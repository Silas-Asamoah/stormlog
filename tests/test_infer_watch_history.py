"""The watcher's bounded history: byte and age bounds, and scrape round trips."""

from __future__ import annotations

import pytest

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


def _stamp(second: float) -> Stamped:
    mono = round(second * S)
    return Stamped(mono, mono + 1, mono)


def test_stamps_must_finish_after_they_start() -> None:
    with pytest.raises(ValueError):
        Stamped(10, 5, 10)


def test_the_byte_bound_counts_the_compressed_bytes_held() -> None:
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
    records = history.records(2 * S, 4 * S)
    assert [r.observed_at_ns for _s, r in records] == [
        scrape(text, second).observed_at_ns for second in (2, 3, 4)
    ]
    assert records[0][1].scrape == scrape(text, 2).scrape
    with pytest.raises(ValueError):
        ScrapeHistory(max_seconds=1, max_bytes=1, parsed_count=1)
