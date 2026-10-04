"""The exporters' bounded queue: never blocks, never evicts, counts drops."""

import threading
import time

import pytest

from stormlog._export.queue import ENTRY_BYTES, TAKE_LIMIT, BoundedQueue

# Bytes for a few small items on any Python: ENTRY_BYTES is 92 on 3.10-3.13
# and 100 on 3.14, so a fixed 100 held none there.
ROOM = 10 * (ENTRY_BYTES + 8)


def test_offers_past_the_item_bound_are_dropped_and_counted() -> None:
    queue: BoundedQueue[int] = BoundedQueue(max_items=2, max_bytes=1000)
    assert queue.offer(1, 10) and queue.offer(2, 10)
    assert not queue.offer(3, 10)
    stats = queue.stats()
    assert (stats.offered, stats.accepted, stats.dropped_full) == (3, 2, 1)
    assert queue.drain() == [1, 2]  # the queued items keep their order


def test_offers_past_the_byte_bound_are_dropped_and_counted() -> None:
    # Each item is charged its size and the queue's own entry for it.
    bound = 25 + 3 * ENTRY_BYTES
    queue: BoundedQueue[str] = BoundedQueue(max_items=100, max_bytes=bound)
    assert queue.offer("a", 10) and queue.offer("b", 10)
    assert not queue.offer("big", 10)
    assert queue.offer("small", 5)
    stats = queue.stats()
    assert stats.dropped_full == 1
    assert (stats.depth, stats.depth_bytes) == (3, bound)
    assert (stats.high_water, stats.high_water_bytes) == (3, bound)


def test_take_returns_at_most_the_limit_per_call() -> None:
    queue: BoundedQueue[int] = BoundedQueue(max_items=500, max_bytes=10**6)
    for item in range(200):
        queue.offer(item, 1)
    first = queue.take(limit=1000)
    assert first == list(range(TAKE_LIMIT))
    assert queue.stats().depth == 200 - TAKE_LIMIT


def test_take_times_out_empty() -> None:
    queue: BoundedQueue[int] = BoundedQueue(max_items=1, max_bytes=1)
    started = time.monotonic()
    assert queue.take(timeout=0.05) == []
    assert time.monotonic() - started >= 0.05


def test_an_offer_wakes_a_waiting_consumer() -> None:
    queue: BoundedQueue[int] = BoundedQueue(max_items=10, max_bytes=ROOM)
    got: list[list[int]] = []
    consumer = threading.Thread(target=lambda: got.append(queue.take(timeout=5)))
    consumer.start()
    time.sleep(0.05)
    queue.offer(7, 1)
    consumer.join(5)
    assert got == [[7]]


def test_close_wakes_consumers_and_refuses_offers() -> None:
    queue: BoundedQueue[int] = BoundedQueue(max_items=10, max_bytes=ROOM)
    got: list[list[int]] = []
    consumer = threading.Thread(target=lambda: got.append(queue.take(timeout=None)))
    consumer.start()
    time.sleep(0.05)
    queue.close()
    consumer.join(5)
    assert not consumer.is_alive() and got == [[]]
    assert not queue.offer(1, 1)
    stats = queue.stats()
    assert stats.closed and stats.dropped_closed == 1


def test_items_queued_before_close_can_still_be_taken() -> None:
    queue: BoundedQueue[int] = BoundedQueue(max_items=10, max_bytes=ROOM)
    queue.offer(1, 1)
    queue.close()
    assert queue.take(timeout=0) == [1]
    assert queue.take(timeout=0) == []


@pytest.mark.parametrize(("items", "size"), [(0, 1), (1, 0)])
def test_a_queue_needs_room(items: int, size: int) -> None:
    with pytest.raises(ValueError):
        BoundedQueue(max_items=items, max_bytes=size)
