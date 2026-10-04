"""What the export core really retains, measured with tracemalloc on the real parts."""

import threading
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from stormlog._export import textfile
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
    writer = TextfileWriter(tmp_path, "alpha", cache, interval=3600)
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
