"""What the export core really retains, measured with tracemalloc on the real parts."""

import os
import random
import threading
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from stormlog._export import textfile
from stormlog._export.envelope import Envelope, EnvelopeLimits, Value, make_envelope
from stormlog._export.queue import BoundedQueue
from stormlog._export.registry import FamilySpec, Registry, render
from stormlog._export.renders import MAX_GENERATIONS, Generation, RenderCache
from stormlog._export.textfile import TextfileWriter


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _registry(series: int) -> Registry:
    registry = Registry(const_labels={"stormlog_producer": "alpha"}, headroom=0)
    registry.add(
        FamilySpec("stormlog_x_total", "counter", "h", labels=("case",)),
        known=[{"case": f"case-{index:06d}-" + "x" * 50} for index in range(series)],
    )
    return registry


def test_renders_with_a_stuck_textfile_writer_stay_within_three_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Astra's staggered history with the real parts: the registry's render,
    # the render cache, a textfile writer stuck inside its file write, and
    # four scrapes that start on successive ticks and hold their renders.
    registry = _registry(20_000)
    size = len(render(registry.snapshot()))
    clock = _Clock()
    cache = RenderCache(lambda: render(registry.snapshot()), clock=clock)
    in_write = threading.Event()
    release = threading.Event()
    real_open = open

    class Stuck:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> "Stuck":
            return self

        def __exit__(self, *exc: object) -> None:
            self.handle.close()

        def write(self, data: bytes) -> int:
            in_write.set()
            release.wait(30)
            return int(self.handle.write(data))

        def flush(self) -> None:
            self.handle.flush()

    def stuck_open(path: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(path, mode, *args, **kwargs)
        return Stuck(handle) if str(path).endswith(".tmp") else handle

    monkeypatch.setattr(textfile, "open", stuck_open, raising=False)
    writer = TextfileWriter(
        tmp_path, "alpha", cache, const_labels=registry.const_labels, interval=3600
    )
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        writer.start()
        assert in_write.wait(30)
        held: list[Generation] = []
        for tick in range(1, 5):
            clock.now = float(tick)
            held.append(cache.acquire())
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        release.set()
    for generation in held:
        cache.release(generation)
    writer.close()
    assert cache.alive() <= MAX_GENERATIONS
    # Three renders of exactly their size; at the peak, the snapshot of the
    # one being built too, which holds references to the rendered prefixes
    # and one float per value: well under a fifth of a render.
    assert (current - baseline) <= MAX_GENERATIONS * size * 1.03
    assert (peak - baseline) <= MAX_GENERATIONS * size + 0.25 * size


_RANDOM = random.Random(1)


def _distinct_floats() -> Value:
    return _RANDOM.random() * 1e6


def _ascii_strings() -> Value:
    return os.urandom(128).hex()  # 256 characters, the default cap


def _wide_strings() -> Value:
    return "\u65e5" + os.urandom(60).hex()  # two bytes a character in memory


def _ints_past_int64() -> Value:
    return int("9" * 4000)  # JSON allows these


def _float_tuples() -> Value:
    return tuple(_RANDOM.random() for _ in range(32))


@pytest.mark.parametrize(
    ("fields", "value"),
    [
        (1, _distinct_floats),
        (4, _distinct_floats),
        (32, _distinct_floats),
        (15, _ascii_strings),
        (8, _wide_strings),
        (2, _float_tuples),
        (8, lambda: None),
    ],
)
def test_a_queue_of_envelopes_holds_what_it_charges(
    fields: int, value: Callable[[], Value]
) -> None:
    retained, charged = _fill_queue(fields, value)
    assert retained <= charged * 1.1
    assert charged <= retained * 1.5


def test_ints_past_64_bits_are_clamped_so_they_hold_what_they_charge() -> None:
    # A 4000-digit int was charged 8 bytes and held 1.8 KB.
    retained, charged = _fill_queue(32, _ints_past_int64)
    assert retained <= charged * 1.1
    envelope = make_envelope(
        "request", [("a", 2**70), ("b", -(2**70)), ("c", 5)], EnvelopeLimits()
    )
    assert envelope.fields == (("a", 2**63 - 1), ("b", -(2**63)), ("c", 5))
    assert envelope.truncated == 2


def _fill_queue(fields: int, value: Callable[[], Value]) -> tuple[int, int]:
    """The bytes a queue filled to its byte bound holds, and the bytes it charged."""
    # The review's measurement: fill a queue to a bound, then compare what
    # tracemalloc says it holds with the bytes the queue charged.
    queue: BoundedQueue[Envelope] = BoundedQueue(
        max_items=1_000_000, max_bytes=2 * 1024 * 1024
    )
    names = [f"field_{index:02d}" for index in range(fields)]
    limits = EnvelopeLimits()
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        while True:
            envelope = make_envelope(
                "request", ((name, value()) for name in names), limits
            )
            if not queue.offer(envelope, envelope.size):
                break
        del envelope
        retained = tracemalloc.get_traced_memory()[0] - baseline
    finally:
        tracemalloc.stop()
    return retained, queue.stats().depth_bytes


def test_a_registry_holds_its_render_and_a_few_hundred_bytes_a_series() -> None:
    # The capacity row: M for the rendered prefixes, and per series its key,
    # its record and its values. Without __slots__ on the series it was
    # about 400 bytes a series.
    known = [{"case": f"case-{index:06d}-" + "x" * 50} for index in range(10_000)]
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        registry = Registry(const_labels={"stormlog_producer": "alpha"}, headroom=0)
        registry.add(
            FamilySpec("stormlog_x_total", "counter", "h", labels=("case",)),
            known=known,
        )
        retained = tracemalloc.get_traced_memory()[0] - baseline
    finally:
        tracemalloc.stop()
    size = len(render(registry.snapshot()))
    assert retained <= size + 350 * registry.budget().samples
