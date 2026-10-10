"""Explicit lazy-root evaluation and synchronized host timing for MLX.

Wrappers must be placed outside compile/value_and_grad transforms. Memory is
process allocator memory; synchronization covers only declared streams.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from stormlog.session import (
    SessionSummary,
    create_session_summary,
    update_session_summary,
)

from .collector import MLXCollector
from .models import MemorySnapshot, ProfileResult
from .runtime import MLXRuntime, Runtime
from .sampling import SampleHistory, Sampler, positive_interval

R = TypeVar("R")
_PEAK_LEASE = threading.Lock()


class ProfileRegion:
    """Evaluate explicitly provided roots without retaining them."""

    def __init__(self, runtime: Runtime, *, name: str, already_evaluated: bool) -> None:
        self.name = name
        self._runtime = runtime
        self.completion_verified = already_evaluated
        self.evaluation_count = 0
        self.closed = False

    def evaluate(self, *roots: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot evaluate roots in a closed profile region")
        self._runtime.evaluate(*roots)
        self.evaluation_count += 1
        self.completion_verified = True


class MLXMemoryProfiler:
    """Bounded sampling with optional exclusive process-local peak reset."""

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        device_id: int = 0,
        sampling_interval: float = 0.01,
        max_history: int = 1000,
        peak_mode: str = "sampled",
    ) -> None:
        self.sampling_interval = positive_interval(sampling_interval)
        SampleHistory(max_history)
        self._validate_peak_mode(peak_mode)
        if type(device_id) is not int or device_id != 0:
            raise ValueError("MLX supports only device_id=0")
        self.runtime = runtime if runtime is not None else MLXRuntime(device_id)
        self.collector = MLXCollector(self.runtime)
        self.max_history = max_history
        self.peak_mode = peak_mode
        self._results: list[ProfileResult] = []
        self._lock = threading.Lock()

    @staticmethod
    def _validate_peak_mode(mode: str) -> None:
        if mode not in {"sampled", "reset"}:
            raise ValueError("peak_mode must be 'sampled' or 'reset'")

    def capture_snapshot(self, name: str = "snapshot") -> MemorySnapshot:
        return self.collector.capture_snapshot(name)

    def profile_function(
        self,
        function: Callable[..., R],
        *args: Any,
        name: str | None = None,
        state_getter: Callable[[], Any] | None = None,
        root_extractor: Callable[[R], Any] | None = None,
        streams: tuple[Any, ...] | None = None,
        peak_mode: str | None = None,
        **kwargs: Any,
    ) -> R:
        """Execute once, evaluate output/state trees, and return output unchanged."""
        with self.profile_context(
            name or str(getattr(function, "__name__", "function")),
            streams=streams,
            peak_mode=peak_mode,
        ) as region:
            output = function(*args, **kwargs)
            region.evaluate(
                output if root_extractor is None else root_extractor(output)
            )
            if state_getter is not None:
                region.evaluate(state_getter())
        return output

    @contextmanager
    def profile_context(
        self,
        name: str = "context",
        *,
        streams: tuple[Any, ...] | None = None,
        peak_mode: str | None = None,
        already_evaluated: bool = False,
    ) -> Iterator[ProfileRegion]:
        mode = self.peak_mode if peak_mode is None else peak_mode
        self._validate_peak_mode(mode)
        selected = self._resolve_streams(streams)
        leased = self._acquire_peak_lease(mode)
        try:
            self.runtime.synchronize(selected)
            baseline = self.capture_snapshot("baseline")
            if mode == "reset":
                self.runtime.reset_peak()
            with self._measure(
                name, baseline, selected, mode, already_evaluated
            ) as region:
                yield region
        finally:
            if leased:
                _PEAK_LEASE.release()

    def _resolve_streams(self, streams: tuple[Any, ...] | None) -> tuple[Any, ...]:
        if streams is None:
            return (self.runtime.default_stream(),)
        if not streams:
            raise ValueError("Declare at least one stream, or use the default stream")
        return tuple(streams)

    @staticmethod
    def _acquire_peak_lease(mode: str) -> bool:
        if mode != "reset":
            return False
        if not _PEAK_LEASE.acquire(blocking=False):
            raise RuntimeError("Nested/overlapping MLX peak-reset scopes are forbidden")
        return True

    @contextmanager
    def _measure(
        self,
        name: str,
        baseline: MemorySnapshot,
        streams: tuple[Any, ...],
        mode: str,
        already_evaluated: bool,
    ) -> Iterator[ProfileRegion]:
        session = create_session_summary(source="stormlog.mlx.profiler")
        history = SampleHistory(self.max_history)
        history.append(baseline)
        history_lock = threading.Lock()

        def sample() -> None:
            snapshot = self.capture_snapshot("sample")
            with history_lock:
                history.append(snapshot)

        sampler = Sampler(sample, self.sampling_interval)
        region = ProfileRegion(
            self.runtime, name=name, already_evaluated=already_evaluated
        )
        sampler.start()
        start_ns = time.perf_counter_ns()
        original: BaseException | None = None
        cleanup_errors: list[str] = []
        try:
            yield region
        except BaseException as exc:
            original = exc
            raise
        finally:
            self._finish_measurement(
                session,
                region,
                sampler,
                history,
                baseline,
                streams,
                mode,
                start_ns,
                original,
                cleanup_errors,
            )

    def _finish_measurement(
        self,
        session: SessionSummary,
        region: ProfileRegion,
        sampler: Sampler,
        history: SampleHistory,
        baseline: MemorySnapshot,
        streams: tuple[Any, ...],
        mode: str,
        start_ns: int,
        original: BaseException | None,
        errors: list[str],
    ) -> None:
        failure = self._complete_streams(streams, region, errors)
        elapsed = time.perf_counter_ns() - start_ns
        sampler.stop()
        region.closed = True
        if sampler.error is not None:
            errors.append(f"sampler: {type(sampler.error).__name__}: {sampler.error}")
        final, final_failure = self._final_snapshot(baseline, errors)
        if failure is None:
            failure = final_failure
        history.append(final)
        failed = original is not None or failure is not None
        status = _measurement_status(original, failure, errors, history)
        ended = time.time_ns()
        result = ProfileResult(
            name=region.name,
            session_summary=update_session_summary(
                session, status=status, ended_at_ns=ended
            ),
            started_at_ns=session.started_at_ns,
            ended_at_ns=ended,
            elapsed_ns=elapsed,
            completion_verified=region.completion_verified and not failed,
            status=status,
            baseline=baseline,
            final=final,
            sampled_peak_bytes=history.peak,
            valid_sample_count=history.valid,
            peak_mode=mode,
            metadata={
                "framework": "mlx",
                "backend": "metal",
                "memory_scope": "process_allocator",
                "memory_model": "unified",
                "timing_scope": "synchronized_host_elapsed",
                "stream_scope": [str(stream) for stream in streams],
                "evaluation_count": region.evaluation_count,
                "peak_reset_performed": mode == "reset",
                "reset_peak_including_baseline_bytes": _reset_peak(
                    mode, baseline, final
                ),
                "reset_isolation": (
                    "process_local_lease_only_unrelated_threads_not_isolated"
                ),
                "sample_count": history.total,
                "retained_sample_count": len(history.samples),
                "unavailable_sample_count": history.total - history.valid,
                "cleanup_errors": errors,
                "exception_type": None if original is None else type(original).__name__,
            },
        )
        with self._lock:
            self._results.append(result)
        if failure is not None and original is None:
            raise failure

    def _complete_streams(
        self,
        streams: tuple[Any, ...],
        region: ProfileRegion,
        errors: list[str],
    ) -> BaseException | None:
        try:
            self.runtime.synchronize(streams)
        except BaseException as exc:
            errors.append(f"synchronize: {type(exc).__name__}: {exc}")
            region.completion_verified = False
            return exc
        return None

    def _final_snapshot(
        self,
        baseline: MemorySnapshot,
        errors: list[str],
    ) -> tuple[MemorySnapshot, BaseException | None]:
        try:
            return self.capture_snapshot("final"), None
        except BaseException as exc:
            errors.append(f"final snapshot: {type(exc).__name__}: {exc}")
            snapshot = MemorySnapshot(
                time.time_ns(),
                0,
                "final",
                None,
                None,
                None,
                None,
                baseline.metadata,
                {"final_snapshot": str(exc)},
            )
            return snapshot, exc

    def get_results(self) -> list[ProfileResult]:
        with self._lock:
            return list(self._results)

    def clear_results(self) -> None:
        with self._lock:
            self._results.clear()

    def export(self, path: str) -> None:
        from .profile_artifact import write_profiles

        write_profiles(path, self.get_results())


def _measurement_status(
    original: BaseException | None,
    failure: BaseException | None,
    errors: list[str],
    history: SampleHistory,
) -> str:
    if isinstance(original, KeyboardInterrupt) or isinstance(
        failure, KeyboardInterrupt
    ):
        return "interrupted"
    if (
        original is not None
        or failure is not None
        or errors
        or history.valid != history.total
    ):
        return "incomplete"
    return "completed"


def _reset_peak(
    mode: str, baseline: MemorySnapshot, final: MemorySnapshot
) -> int | None:
    if mode != "reset" or final.runtime_peak_bytes is None:
        return None
    return max(baseline.active_bytes or 0, final.runtime_peak_bytes)
