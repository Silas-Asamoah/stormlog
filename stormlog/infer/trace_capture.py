"""Bounded vLLM profiler windows during ``stormlog infer profile``.

vLLM's torch profiler is configured when the server starts
(``--profiler-config.profiler=torch`` and ``torch_profiler_dir``). A client can
then start and stop it over HTTP (``/start_profile`` and ``/stop_profile``).
This module opens one window per profiled phase, closes it at the phase's end,
at a time bound, or on cancellation, and finds the worker traces vLLM wrote.
It stops the profiler only after its own start succeeded. vLLM 0.30.0 answers
200 to a second ``/start_profile`` and to ``/stop_profile`` with nothing
running, so a client cannot tell from HTTP whether another profile was already
active; do not run two profilers against one server. The traces are imported
after the run.

The profiler adds no synchronization per request; the server writes the trace
while handling ``/stop_profile``, so that call can take tens of seconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..session import SessionSummary
from .cache_state import redact_url
from .correlation_capture import (
    CaptureCapabilities,
    TraceCapture,
    append_inference_capture,
)
from .trace_import import DeviceUuids, KinetoTraceCollector
from .trace_kineto import SUPPORTED, Detail

VLLM_TORCH = "vllm-torch"
TRACE_MODES = (VLLM_TORCH,)
TRACE_PHASES = ("measured", "warmup")
# vLLM worker traces; the API server's own trace is named "*.async_llm.*".
WORKER_TRACE_GLOB = "rank*.pt.trace.json*"


@dataclass(frozen=True)
class TraceCaptureConfig:
    """What to capture, where vLLM writes it, and the bounds on the capture."""

    control_url: str
    trace_dir: Path | None = None
    mode: str = VLLM_TORCH
    phase: str = "measured"
    max_seconds: float | None = None
    max_bytes: int | None = None
    device_uuids: DeviceUuids = field(default_factory=DeviceUuids)
    detail: Detail = "launch"
    control_timeout_seconds: float = 600.0
    flush_timeout_seconds: float = 60.0
    # vLLM writes the trace inside /stop_profile, so a file that has not
    # appeared shortly after the stop is not coming (allow for shared storage).
    missing_grace_seconds: float = 5.0
    settle_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in TRACE_MODES:
            raise ValueError(f"--trace must be one of {', '.join(TRACE_MODES)}")
        if self.phase not in TRACE_PHASES:
            raise ValueError(f"--trace-phase must be one of {', '.join(TRACE_PHASES)}")
        for name in ("max_seconds", "max_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"trace {name} must be > 0")


def server_root(endpoint: str) -> str:
    """The scheme and host of an endpoint URL, where vLLM serves its controls."""
    parts = urllib.parse.urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True)
class ControlResult:
    """What a profiler control call returned; it never raises."""

    status: int | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


class ProfilerControl(Protocol):
    def post(self, route: str) -> ControlResult: ...


class HttpProfilerControl:
    """POST to vLLM's profiler routes with the profile's API key."""

    def __init__(self, root: str, *, api_key: str | None, timeout: float) -> None:
        self.root = root.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def post(self, route: str) -> ControlResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request = urllib.request.Request(
            f"{self.root}{route}", data=b"", headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return ControlResult(int(response.status))
        except urllib.error.HTTPError as exc:
            return ControlResult(exc.code, f"HTTP {exc.code}")
        except (OSError, http.client.HTTPException) as exc:
            # HTTPException covers a malformed reply (BadStatusLine and the like).
            return ControlResult(None, f"{type(exc).__name__}: {exc}")


@dataclass
class TraceWindow:
    """One profiler window, recorded as an ``infer.trace_window`` event."""

    case_id: str
    phase: str
    control_url: str
    requested_at_ns: int
    started: bool = False
    started_at_ns: int | None = None
    start: ControlResult | None = None
    stop: ControlResult | None = None
    stop_reason: str | None = None
    stopped_at_ns: int | None = None
    files: list[Path] = field(default_factory=list)
    note: str | None = None
    before: dict[str, int] = field(default_factory=dict, repr=False)

    def to_record(self, *, session_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "infer.trace_window",
            "session_id": session_id,
            "case_id": self.case_id,
            "phase": self.phase,
            "control_url": redact_url(self.control_url),
            "requested_at_ns": self.requested_at_ns,
            "started": self.started,
            "started_at_ns": self.started_at_ns,
            "start_status": self.start.status if self.start else None,
            "start_error": self.start.error if self.start else None,
            "stop_status": self.stop.status if self.stop else None,
            "stop_error": self.stop.error if self.stop else None,
            "stop_reason": self.stop_reason,
            "stopped_at_ns": self.stopped_at_ns,
            "trace_files": [path.name for path in self.files],
            "note": self.note,
        }


class TraceWindows:
    """Open, bound, and close profiler windows; import their traces afterwards."""

    def __init__(
        self,
        config: TraceCaptureConfig,
        *,
        api_key: str | None = None,
        control: ProfilerControl | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.control = control or HttpProfilerControl(
            config.control_url,
            api_key=api_key,
            timeout=config.control_timeout_seconds,
        )
        self.on_warning = on_warning
        self.windows: list[TraceWindow] = []

    @asynccontextmanager
    async def window(
        self, case_id: str, phase: str
    ) -> AsyncIterator[TraceWindow | None]:
        """Profile the body if ``phase`` is the configured one."""
        if phase != self.config.phase:
            yield None
            return
        window = TraceWindow(
            case_id, phase, self.config.control_url, requested_at_ns=time.time_ns()
        )
        before = await asyncio.to_thread(self._snapshot)
        window.before = before
        window.start = await asyncio.to_thread(self.control.post, "/start_profile")
        window.started = window.start.ok
        window.started_at_ns = time.time_ns() if window.started else None
        if not window.started:
            window.note = "the profiler did not start; see start_status and start_error"
            self._warn(window, "could not start the profiler")
        timer = self._bound(window)
        reason = "phase_end"
        try:
            yield window
        except BaseException:
            reason = "cancelled"
            raise
        finally:
            await _finish_timer(timer, window)
            await self._close(window, reason, before)

    def _bound(self, window: TraceWindow) -> asyncio.Task[None] | None:
        if not window.started or self.config.max_seconds is None:
            return None
        return asyncio.create_task(self._stop_after(window, self.config.max_seconds))

    async def _stop_after(self, window: TraceWindow, seconds: float) -> None:
        await asyncio.sleep(seconds)
        await self._stop(window, "time_bound")

    async def _stop(self, window: TraceWindow, reason: str) -> None:
        if not window.started or window.stop_reason is not None:
            return
        window.stop_reason = reason
        if await asyncio.to_thread(self._wrote_before_stop, window.before):
            # A profile with max_iterations stops itself and writes its trace.
            window.stop_reason = "stopped_by_server"
        window.stop = await asyncio.to_thread(self.control.post, "/stop_profile")
        window.stopped_at_ns = time.time_ns()
        if not window.stop.ok:
            self._warn(window, "the profiler did not confirm the stop")

    async def _close(
        self, window: TraceWindow, reason: str, before: dict[str, int]
    ) -> None:
        await self._stop(window, reason)
        if window.started:
            window.files = await asyncio.to_thread(self._new_files, before)
            window.note = self._files_note(window)
        self.windows.append(window)

    def _snapshot(self) -> dict[str, int]:
        directory = self.config.trace_dir
        if directory is None or not directory.is_dir():
            return {}
        return {path.name: path.stat().st_size for path in _worker_traces(directory)}

    def _wrote_before_stop(self, before: dict[str, int]) -> bool:
        directory = self.config.trace_dir
        if directory is None or not directory.is_dir():
            return False
        return any(path.name not in before for path in _worker_traces(directory))

    def _new_files(self, before: dict[str, int]) -> list[Path]:
        """Worker traces written since ``before``, once their sizes settle."""
        directory = self.config.trace_dir
        if directory is None or not directory.is_dir():
            return []
        started = time.monotonic()
        previous: dict[str, int] | None = None
        while not _flush_over(started, previous, self.config):
            current = {
                path.name: path.stat().st_size
                for path in _worker_traces(directory)
                if path.name not in before
            }
            if current and current == previous:
                break
            previous = current
            time.sleep(self.config.settle_seconds)
        return sorted(directory / name for name in previous or {})

    def _files_note(self, window: TraceWindow) -> str | None:
        if self.config.trace_dir is None:
            return "trace left on the server; no --trace-dir to read it from"
        if not window.files:
            return f"no new worker trace appeared in {self.config.trace_dir}"
        return None

    def import_into(
        self, artifact: Path, *, run_id: str, session: SessionSummary
    ) -> None:
        """Append every window's traces to the artifact; warn instead of failing.

        With no trace to import, the trace collector is still recorded, with
        nothing collected and each window's reason in its summary.
        """
        files = [path for window in self.windows for path in window.files]
        collector: Any = (
            KinetoTraceCollector(
                files,
                device_uuids=self.config.device_uuids,
                detail=self.config.detail,
                max_bytes=self.config.max_bytes,
            )
            if files
            else _NothingCollected(self.windows)
        )
        try:
            append_inference_capture(
                artifact, run_id=run_id, session=session, trace_collector=collector
            )
        except Exception as exc:  # a finished run must not fail on its import
            if self.on_warning is not None:
                self.on_warning(
                    f"traces were captured but not imported ({exc}); run "
                    "`stormlog infer import-trace` on them"
                )

    def _warn(self, window: TraceWindow, message: str) -> None:
        if self.on_warning is None:
            return
        result = window.stop if window.stop_reason else window.start
        detail = result.error if result and result.error else "no response"
        self.on_warning(f"{window.case_id} {window.phase}: {message} ({detail})")


class _NothingCollected:
    """The trace collector of a run whose windows produced no trace to import."""

    def __init__(self, windows: list[TraceWindow]) -> None:
        self.windows = windows

    def collect(self, *, run_id: str, session_id: str) -> TraceCapture:
        return TraceCapture(
            capabilities=CaptureCapabilities(SUPPORTED, SUPPORTED, ()),
            summary={
                "traces": [],
                "windows": [
                    {
                        "case_id": window.case_id,
                        "started": window.started,
                        "note": window.note,
                    }
                    for window in self.windows
                ],
            },
        )


def _flush_over(
    started: float, seen: dict[str, int] | None, config: TraceCaptureConfig
) -> bool:
    """Stop waiting at the flush timeout, or at the grace with nothing seen."""
    waited = time.monotonic() - started
    if waited >= config.flush_timeout_seconds:
        return True
    return seen == {} and waited >= config.missing_grace_seconds


async def _finish_timer(timer: asyncio.Task[None] | None, window: TraceWindow) -> None:
    """Cancel a time bound that has not fired; let one that is stopping finish."""
    if timer is None:
        return
    if window.stop_reason is None:
        timer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await timer


def _worker_traces(directory: Path) -> list[Path]:
    return [path for path in directory.glob(WORKER_TRACE_GLOB) if path.is_file()]


__all__ = [
    "TRACE_MODES",
    "TRACE_PHASES",
    "VLLM_TORCH",
    "ControlResult",
    "HttpProfilerControl",
    "ProfilerControl",
    "TraceCaptureConfig",
    "TraceWindow",
    "TraceWindows",
    "server_root",
]
