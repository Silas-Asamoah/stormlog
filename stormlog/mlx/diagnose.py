"""Bounded diagnostics of this process's MLX runtime and allocator."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

from .artifacts import CaptureBundle, CaptureOptions, Outcome, capture_failure
from .models import MemorySnapshot
from .runtime import Runtime
from .tracker import MemoryTracker


def capture_process(
    *,
    output: str | Path | None,
    duration: float = 0,
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
    command: str = "diagnose",
    wait: Callable[[float], None] = time.sleep,
    on_sample: Callable[[MemorySnapshot], None] | None = None,
) -> tuple[Path, Outcome]:
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    bundle = CaptureBundle(output, command, options or CaptureOptions())
    outcome = Outcome()
    try:
        bundle.start(runtime)
        assert bundle.tracker is not None
        _wait_capture(bundle.tracker, duration, wait, on_sample)
    except BaseException as exc:
        outcome = capture_failure(exc)
        bundle.environment["runtime_error"] = outcome.error
    finally:
        bundle.finish(outcome)
    return bundle.directory, outcome


def _wait_capture(
    tracker: MemoryTracker,
    duration: float,
    wait: Callable[[float], None],
    on_sample: Callable[[MemorySnapshot], None] | None,
) -> None:
    if on_sample is None:
        wait(duration)
        return
    end = time.monotonic() + duration
    while True:
        samples = tracker.get_results().samples
        if samples:
            on_sample(samples[-1])
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        wait(min(tracker.sampling_interval, remaining))


def run_diagnose(
    *,
    output: str | Path | None,
    duration: float = 0,
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
) -> tuple[Path, Outcome]:
    return capture_process(
        output=output, duration=duration, options=options, runtime=runtime
    )
