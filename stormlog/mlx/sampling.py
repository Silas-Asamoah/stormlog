"""Shared interruptible sampler and bounded history for MLX instrumentation."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

from .models import MemorySnapshot


def positive_interval(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("sampling_interval must be positive and finite")
    return value


class SampleHistory:
    """Retain a bounded window while aggregating every valid observation."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_history must be a positive integer")
        self.samples: deque[MemorySnapshot] = deque(maxlen=limit)
        self.total = 0
        self.valid = 0
        self.sum_active = 0
        self.peak: int | None = None
        self.minimum: int | None = None

    def append(self, snapshot: MemorySnapshot) -> None:
        self.samples.append(snapshot)
        self.total += 1
        value = snapshot.active_bytes
        if value is None:
            return
        self.valid += 1
        self.sum_active += value
        self.peak = value if self.peak is None else max(value, self.peak)
        self.minimum = value if self.minimum is None else min(value, self.minimum)


class Sampler:
    """One monotonic sampling lifecycle. Callbacks may request retry backoff."""

    def __init__(self, callback: Callable[[], float | None], interval: float) -> None:
        self.interval = positive_interval(interval)
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run, name="stormlog-mlx-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._thread = None

    def _run(self) -> None:
        deadline = time.monotonic() + self.interval
        try:
            while not self._stop.wait(max(0, deadline - time.monotonic())):
                delay = self.callback()
                next_interval = (
                    self.interval if delay is None else max(self.interval, delay)
                )
                deadline += next_interval
                now = time.monotonic()
                if deadline <= now:
                    deadline = now + next_interval
        except BaseException as exc:
            self.error = exc
            self._stop.set()
