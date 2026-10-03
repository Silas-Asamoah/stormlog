"""I1: pause the engine with one profiler window, and stamp its calls.

vLLM runs ``/start_profile`` and ``/stop_profile`` between engine steps, and
the stop writes the trace while the step loop waits, so a capture pauses
serving (docs/inference.md, "Profiler traces"). I1 opens one window and
records when each call was requested and when it returned; the stop's
interval, plus #219's drain, is the episode's effect.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Call:
    requested_ns: int
    returned_ns: int
    status: int | None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "requested_ns": self.requested_ns,
            "returned_ns": self.returned_ns,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class CaptureWindow:
    start: Call
    stop: Call | None

    @property
    def ok(self) -> bool:
        return (
            self.start.status == 200
            and self.stop is not None
            and self.stop.status == 200
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "start": self.start.to_record(),
            "stop": None if self.stop is None else self.stop.to_record(),
        }


def post(url: str, *, timeout_seconds: float) -> Call:
    """POST with no body, timed from the request to its answer."""
    requested = time.time_ns()
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as answer:
            status: int | None = int(answer.status)
        error = None
    except urllib.error.HTTPError as failure:
        status, error = int(failure.code), None
    except (OSError, urllib.error.URLError) as failure:
        status, error = None, repr(failure)
    return Call(requested, time.time_ns(), status, error)


def capture_window(
    base_url: str, seconds: float, *, timeout_seconds: float = 120.0
) -> CaptureWindow:
    """Open a profiler window for ``seconds``, then close it. A start the
    server refused is never followed by a stop; an ambiguous one (a timeout
    or a reset, so the server may have started) is, and so is a window cut
    short by an interrupt, which is then re-raised."""
    start = post(f"{base_url}/start_profile", timeout_seconds=timeout_seconds)
    if start.status is not None and start.status != 200:
        return CaptureWindow(start, None)
    try:
        _wait(seconds)
    finally:
        stop = post(f"{base_url}/stop_profile", timeout_seconds=timeout_seconds)
    return CaptureWindow(start, stop)


_wait = time.sleep


__all__ = ["Call", "CaptureWindow", "capture_window", "post"]
