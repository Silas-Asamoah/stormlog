"""Bounded vLLM profiler windows during ``stormlog infer profile``.

vLLM's torch profiler is configured when the server starts
(``--profiler-config.profiler=torch`` and ``torch_profiler_dir``). A client can
then start and stop it over HTTP (``/start_profile`` and ``/stop_profile``).
This module opens one window per profiled phase, closes it at the phase's end,
at a time bound, or on cancellation, and finds the worker traces vLLM wrote.
A start the server may have received, whether it answered 2xx or not at all,
is always followed by one stop: the engine runs ``/start_profile`` before the
HTTP reply goes out, so a lost or failed reply can leave it profiling. A stop
that fails is not retried; the window's record says the profiler may still be
running. Only a 4xx, which never reaches the engine, is not stopped. vLLM
0.30.0 answers 200 to a second ``/start_profile`` and to ``/stop_profile``
with nothing running, so a client cannot tell from HTTP whether another
profile was already active; do not run two profilers against one server. The
traces are imported after the run.

The profiler adds no synchronization per request; the server writes the trace
while handling ``/stop_profile``, so that call can take tens of seconds.
"""

from __future__ import annotations

import asyncio
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
from .trace_import import DeviceUuids, TraceFileCollector
from .trace_kineto import SUPPORTED, Detail
from .vllm_execution_devices import WorkerIndex

VLLM_TORCH = "vllm-torch"
TRACE_MODES = (VLLM_TORCH,)
TRACE_PHASES = ("measured", "warmup")
# vLLM names worker traces rank<N>.*, or dp<D>_pp<P>_tp<T>_dcp<C>_ep<E>_rank<N>.*
# when every parallel group exists (MoE models); the API server's own trace is
# named *.async_llm.* and is not GPU work.
TRACE_GLOB = "*.pt.trace.json*"
# What a /start_profile call established about the server's profiler.
START_ACKNOWLEDGED = "acknowledged"
START_REJECTED = "rejected"
START_UNKNOWN = "unknown"


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


def start_outcome(result: ControlResult) -> str:
    """``acknowledged`` (2xx), ``rejected`` (4xx), else ``unknown``.

    A 4xx means the route refused the call before the engine saw it. A 5xx, a
    timeout, a reset or a malformed reply comes after the call may have
    started the profiler, so the profiler's state is unknown.
    """
    if result.ok:
        return START_ACKNOWLEDGED
    if result.status is not None and 400 <= result.status < 500:
        return START_REJECTED
    return START_UNKNOWN


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
    start_outcome: str | None = None
    # Wall times taken immediately around the /start_profile request itself.
    start_requested_at_ns: int | None = None
    start_returned_at_ns: int | None = None
    stop: ControlResult | None = None
    stop_reason: str | None = None
    stopped_at_ns: int | None = None
    files: list[Path] = field(default_factory=list)
    note: str | None = None
    # Worker traces present before the start; None when they could not be read.
    before: dict[str, int] | None = field(default=None, repr=False)
    # The stop in flight, shared by the time bound and the window's close.
    stopping: asyncio.Future[None] | None = field(
        default=None, repr=False, compare=False
    )

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
            "start_outcome": self.start_outcome,
            "start_requested_at_ns": self.start_requested_at_ns,
            "start_returned_at_ns": self.start_returned_at_ns,
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
        self._unwritten: list[TraceWindow] = []

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
        before = window.before = await asyncio.to_thread(self._snapshot, window)
        # A bare executor future, not a task: on Ctrl+C, asyncio.run cancels
        # every remaining task (Python 3.10 included), and the start's answer
        # is needed to know whether to stop the profiler.
        start = asyncio.get_running_loop().run_in_executor(
            None, self._send_start, window
        )
        try:
            window.start = await asyncio.shield(start)
        except asyncio.CancelledError:
            # The server may still start profiling: wait for its answer, then
            # stop and record the window before letting the cancellation through.
            await self._abandon(window, start, before)
            raise
        _mark_started(window)
        timer: asyncio.Task[None] | None = None
        reason = "phase_end"
        try:
            # Inside the cleanup: a warning that fails (a closed stderr, a
            # Ctrl+C while it prints) must not skip the stop or the record.
            self._warn_start(window)
            timer = self._bound(window)
            if window.start_outcome == START_UNKNOWN:
                # The server may be profiling with nobody to stop it: stop it
                # now, and let the phase run unprofiled.
                await self._stop(window, "start_unknown")
            yield window
        except BaseException:
            reason = "cancelled"
            raise
        finally:
            try:
                await _finish_timer(timer, window)
            finally:
                await self._close(window, reason, before)

    def _send_start(self, window: TraceWindow) -> ControlResult:
        window.start_requested_at_ns = time.time_ns()
        try:
            return self.control.post("/start_profile")
        finally:
            window.start_returned_at_ns = time.time_ns()

    def _warn_start(self, window: TraceWindow) -> None:
        if window.start_outcome == START_REJECTED:
            self._warn(window, "could not start the profiler")
        elif window.start_outcome == START_UNKNOWN:
            self._warn(window, "the profiler may have started; stopping it")

    async def _abandon(
        self,
        window: TraceWindow,
        start: asyncio.Future[ControlResult],
        before: dict[str, int] | None,
    ) -> None:
        # Bounded by the control timeout of the start request itself.
        window.start = await start
        _mark_started(window)
        try:
            self._warn_start(window)
        finally:
            await self._close(window, "cancelled", before)

    def take_records(self, *, session_id: str) -> list[dict[str, Any]]:
        """``infer.trace_window`` records for windows closed since the last call."""
        records = [
            window.to_record(session_id=session_id) for window in self._unwritten
        ]
        self._unwritten.clear()
        return records

    def _bound(self, window: TraceWindow) -> asyncio.Task[None] | None:
        if not window.started or self.config.max_seconds is None:
            return None
        return asyncio.create_task(self._stop_after(window, self.config.max_seconds))

    async def _stop_after(self, window: TraceWindow, seconds: float) -> None:
        await asyncio.sleep(seconds)
        await self._stop(window, "time_bound")

    async def _stop(self, window: TraceWindow, reason: str) -> None:
        """Send the stop once; a cancellation waits for it before going through.

        Every start the server may have received is stopped: acknowledged or
        unknown. Only a rejected start, which never reached the engine, is not.
        """
        if not _may_be_profiling(window):
            return
        first = window.stopping is None
        if window.stopping is None:
            window.stop_reason = reason
            window.stopping = asyncio.get_running_loop().run_in_executor(
                None, self._send_stop, window
            )
        try:
            await asyncio.shield(window.stopping)
        except asyncio.CancelledError:
            # The server keeps profiling until it is told to stop: wait for the
            # stop (bounded by the control timeout), then let Ctrl+C through.
            await window.stopping
            raise
        if first and window.stop is not None and not window.stop.ok:
            self._warn(window, "the profiler did not confirm the stop")

    def _send_stop(self, window: TraceWindow) -> None:
        try:
            if self._wrote_before_stop(window.before):
                # A profile with max_iterations stops itself and writes its trace.
                window.stop_reason = "stopped_by_server"
        except OSError as exc:
            window.note = (
                f"could not read {self.config.trace_dir}: {exc.strerror or exc}"
            )
        window.stop = self.control.post("/stop_profile")
        window.stopped_at_ns = time.time_ns()

    async def _close(
        self, window: TraceWindow, reason: str, before: dict[str, int] | None
    ) -> None:
        """Stop, look for the traces, and record the window whatever happens."""
        try:
            await self._stop(window, reason)
            if _may_be_profiling(window) and before is not None:
                window.files = await asyncio.to_thread(self._find_files, window, before)
        finally:
            # Written from what the stop returned, never ahead of it.
            window.note = _joined(
                window.note or self._result_note(window), _stop_note(window)
            )
            self.windows.append(window)
            self._unwritten.append(window)

    def _find_files(self, window: TraceWindow, before: dict[str, int]) -> list[Path]:
        try:
            return self._new_files(before)
        except OSError as exc:
            window.note = (
                f"could not read {self.config.trace_dir}: {exc.strerror or exc}"
            )
            return []

    def _snapshot(self, window: TraceWindow) -> dict[str, int] | None:
        directory = self.config.trace_dir
        if directory is None or not directory.is_dir():
            return {}
        try:
            return {
                path.name: path.stat().st_size for path in _worker_traces(directory)
            }
        except OSError as exc:
            # Without the files already there, new ones cannot be told apart.
            window.note = (
                f"could not read {directory} before the start: {exc.strerror or exc}"
            )
            return None

    def _wrote_before_stop(self, before: dict[str, int] | None) -> bool:
        directory = self.config.trace_dir
        if before is None or directory is None or not directory.is_dir():
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

    def _result_note(self, window: TraceWindow) -> str | None:
        if window.start_outcome != START_UNKNOWN:
            return self._files_note(window) if window.started else None
        unknown = "the start's answer does not say whether the profiler started"
        if window.stop is None or not window.stop.ok:
            return unknown  # the stop's own note says what is left
        if window.stop_reason == "start_unknown":
            return f"{unknown}; it was stopped before the phase, which ran unprofiled"
        return f"{unknown}; it was stopped"

    def _files_note(self, window: TraceWindow) -> str | None:
        if self.config.trace_dir is None:
            return "trace left on the server; no --trace-dir to read it from"
        if not window.files:
            return f"no new worker trace appeared in {self.config.trace_dir}"
        return None

    def import_into(
        self,
        artifact: Path,
        *,
        run_id: str,
        session: SessionSummary,
        worker_index: WorkerIndex | None = None,
    ) -> None:
        """Append every window's traces to the artifact; warn instead of failing.

        With no trace to import, the trace collector is still recorded, with
        nothing collected and each window's reason in its summary. The
        ``worker_index`` is the vLLM execution hook's, which names each traced
        process's GPU where ``--trace-device-uuid`` does not.
        """
        files = [path for window in self.windows for path in window.files]
        try:
            # Inside the guard: the collector checks the files exist, and one
            # removed since its window closed must not fail the finished run.
            collector: Any = (
                TraceFileCollector(
                    files,
                    device_uuids=self.config.device_uuids,
                    detail=self.config.detail,
                    max_bytes=self.config.max_bytes,
                    worker_index=worker_index,
                )
                if files
                else _NothingCollected(self.windows)
            )
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


def _mark_started(window: TraceWindow) -> None:
    """Classify the start's answer; no callback runs here."""
    assert window.start is not None
    window.start_outcome = start_outcome(window.start)
    window.started = window.start_outcome == START_ACKNOWLEDGED
    window.started_at_ns = window.start_returned_at_ns if window.started else None
    if window.start_outcome == START_REJECTED:
        window.note = "the profiler did not start; see start_status and start_error"


def _stop_note(window: TraceWindow) -> str | None:
    """Said when a profile may have started and no stop was confirmed."""
    if not _may_be_profiling(window) or (window.stop is not None and window.stop.ok):
        return None
    if window.stop is None:
        return "no stop could be sent; the profiler may still be running"
    return (
        "the stop was sent once and not confirmed (see stop_status and "
        "stop_error); the profiler may still be running"
    )


def _joined(*notes: str | None) -> str | None:
    return "; ".join(note for note in notes if note) or None


def _may_be_profiling(window: TraceWindow) -> bool:
    """The server may have started profiling for this window."""
    return window.start_outcome in (START_ACKNOWLEDGED, START_UNKNOWN)


def _flush_over(
    started: float, seen: dict[str, int] | None, config: TraceCaptureConfig
) -> bool:
    """Stop waiting at the flush timeout, or at the grace with nothing seen."""
    waited = time.monotonic() - started
    if waited >= config.flush_timeout_seconds:
        return True
    return seen == {} and waited >= config.missing_grace_seconds


async def _finish_timer(timer: asyncio.Task[None] | None, window: TraceWindow) -> None:
    """Cancel a time bound that has not fired; let one that is stopping finish.

    ``asyncio.wait`` does not raise the timer's own cancellation, while a
    cancellation of the caller still goes through.
    """
    if timer is None:
        return
    if window.stop_reason is None:
        timer.cancel()
    await asyncio.wait({timer})


def _worker_traces(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.glob(TRACE_GLOB)
        if path.is_file() and _is_worker_trace(path.name)
    ]


def _is_worker_trace(name: str) -> bool:
    return ".async_llm." not in name and (
        name.startswith("rank") or "_rank" in name.split(".", 1)[0]
    )


__all__ = [
    "START_ACKNOWLEDGED",
    "START_REJECTED",
    "START_UNKNOWN",
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
    "start_outcome",
]
