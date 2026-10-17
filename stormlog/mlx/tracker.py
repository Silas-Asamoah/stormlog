"""Passive process-allocator tracking with sessions, health, phases and sinks."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from stormlog.collector_health import (
    CollectorHealthState,
    collector_retry_delay_seconds,
)
from stormlog.phases import PhaseHandle, PhaseRecorder, PhaseToken
from stormlog.session import (
    SessionSummary,
    create_session_summary,
    update_session_summary,
)
from stormlog.telemetry import resolve_distributed_identity
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig

from .collector import MLXCollector
from .models import MemorySnapshot, TrackingResult
from .runtime import MLXRuntime, Runtime
from .sampling import SampleHistory, Sampler, positive_interval

if TYPE_CHECKING:
    from stormlog.oom_flight_recorder import OOMFlightRecorder


class MemoryTracker:
    """Observe MLX without evaluating, synchronizing or changing allocator state."""

    def __init__(
        self,
        sampling_interval: float = 0.1,
        *,
        runtime: Runtime | None = None,
        device_id: int = 0,
        max_history: int = 10_000,
        alert_threshold_mb: float | None = None,
        job_id: str | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        telemetry_sink_config: TelemetrySinkConfig | None = None,
        enable_oom_flight_recorder: bool = False,
        oom_dump_dir: str = "oom_dumps",
        oom_buffer_size: int | None = None,
        oom_max_dumps: int = 5,
        oom_max_total_mb: int = 256,
    ) -> None:
        self.sampling_interval = positive_interval(sampling_interval)
        self._history = SampleHistory(max_history)
        if type(device_id) is not int or device_id != 0:
            raise ValueError("MLX supports only device_id=0")
        if alert_threshold_mb is not None:
            positive_interval(alert_threshold_mb)
        self.runtime = runtime if runtime is not None else MLXRuntime(device_id)
        self.collector = MLXCollector(self.runtime)
        self.max_history = max_history
        self.alert_threshold_mb = alert_threshold_mb
        self._identity = resolve_distributed_identity(
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            env=os.environ,
        )
        self._sink_config = telemetry_sink_config
        self._sink: AppendOnlyTelemetrySink | None = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._session: SessionSummary | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._alerts: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._total_events = 0
        self._total_alerts = 0
        self._previous: int | None = None
        self._health = CollectorHealthState()
        self._sink_error: str | None = None
        self._phase_recorder = PhaseRecorder()
        self._sampler = Sampler(self._sample, sampling_interval)
        self._oom_recorder: OOMFlightRecorder | None = None
        if enable_oom_flight_recorder:
            self._configure_oom(
                oom_dump_dir,
                max_history if oom_buffer_size is None else oom_buffer_size,
                oom_max_dumps,
                oom_max_total_mb,
            )

    def _configure_oom(self, path: str, limit: int, dumps: int, size_mb: int) -> None:
        from stormlog.oom_flight_recorder import (
            OOMFlightRecorder,
            OOMFlightRecorderConfig,
        )
        from stormlog.system_info import get_system_info

        if min(limit, dumps, size_mb) <= 0:
            raise ValueError("OOM retention limits must be positive")
        self._oom_recorder = OOMFlightRecorder(
            OOMFlightRecorderConfig(True, path, limit, dumps, size_mb),
            system_info_provider=get_system_info,
        )

    @property
    def is_tracking(self) -> bool:
        return self._sampler.is_running

    @property
    def session_summary(self) -> SessionSummary | None:
        with self._lock:
            return self._session

    def start_tracking(self) -> None:
        with self._lifecycle_lock:
            if self.is_tracking:
                return
            if self._session is not None and self._session.status == "running":
                self.stop_tracking(status="incomplete")
            try:
                self._start_session()
                self._sample("start")
                self._sampler.start()
            except BaseException:
                self._stop_best_effort("incomplete")
                raise

    def _start_session(self) -> None:
        with self._lock:
            self._history = SampleHistory(self.max_history)
            self._events.clear()
            self._alerts.clear()
            self._total_events = self._total_alerts = 0
            self._previous = None
            self._health = CollectorHealthState()
            self._sink_error = None
            self._phase_recorder.reset()
            if self._oom_recorder is not None:
                self._oom_recorder.clear()
            self._session = create_session_summary(
                source="stormlog.mlx.memory_tracker",
                **self._identity,
            )
            self._sink = None
            if self._sink_config is not None:
                self._sink = AppendOnlyTelemetrySink(self._sink_config)
                self._sink.start_session(self._session)

    def stop_tracking(self, *, status: str | None = None) -> TrackingResult | None:
        with self._lifecycle_lock:
            self._sampler.stop()
            with self._lock:
                session = self._session
                if session is None:
                    return None
                if session.status != "running":
                    return self.get_results()
                self._sample("stop")
                self._flush_sink()
                final_status = status or self._terminal_status()
                self._session = update_session_summary(
                    session,
                    status=final_status,
                    ended_at_ns=time.time_ns(),
                )
                self._close_sink(final_status)
                return self.get_results()

    def _terminal_status(self) -> str:
        if self._sampler.error is not None or self._sink_error:
            return "incomplete"
        if self._health.status == "unhealthy":
            return "incomplete"
        if self._sink is not None:
            diagnostics = self._sink.failure_diagnostics()
            if (
                diagnostics["consecutive_flush_failures"]
                or diagnostics["dropped_records"]
            ):
                return "incomplete"
        return "completed"

    def _flush_sink(self) -> None:
        if self._sink is None:
            return
        try:
            self._sink.flush(force=True)
        except Exception as exc:
            self._sink_error = f"{type(exc).__name__}: {exc}"

    def _close_sink(self, status: str) -> None:
        if self._sink is None:
            return
        try:
            self._sink.close(session_status=status)
        except Exception as exc:
            self._sink_error = f"{type(exc).__name__}: {exc}"
            if self._session is not None:
                self._session = update_session_summary(
                    self._session, status="incomplete"
                )

    def _sample(
        self,
        event_type: str = "sample",
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float | None:
        with self._lock:
            if self._session is None or self._session.status != "running":
                raise RuntimeError("Start tracking before collecting events")
            snapshot = self.collector.capture_snapshot(event_type)
            previous_health = self._health
            delay = self._update_health(snapshot)
            self._history.append(snapshot)
            record = self.collector.telemetry_record(
                snapshot,
                self._session,
                event_type=event_type,
                sampling_interval_ms=max(1, round(self.sampling_interval * 1000)),
                previous_active_bytes=self._previous,
                context=context,
                metadata={**self._health.to_dict(), **(metadata or {})},
            )
            self._previous = snapshot.active_bytes
            self._emit(record)
            self._emit_health_transition(record, previous_health)
            self._check_threshold(snapshot)
            return delay

    def _emit_health_transition(
        self,
        record: dict[str, Any],
        previous: CollectorHealthState,
    ) -> None:
        current = self._health
        if previous.status == current.status:
            return
        transition = "recovered" if current.status == "healthy" else "degraded"
        self._emit(
            {
                **record,
                "event_type": f"collector_{transition}",
                "allocator_change_bytes": None,
                "metadata": {
                    **record["metadata"],
                    "collector_transition": transition,
                    "collector_previous_error": previous.last_error,
                    "collector_previous_status": previous.status,
                },
            }
        )

    def _update_health(self, snapshot: MemorySnapshot) -> float | None:
        old = self._health
        fields = tuple(snapshot.unavailable)
        if snapshot.active_bytes is None:
            failures = old.consecutive_failures + 1
            delay = collector_retry_delay_seconds(
                failures,
                initial_delay_s=self.sampling_interval,
                factor=2,
                max_delay_s=max(30, self.sampling_interval),
            )
            self._health = CollectorHealthState(
                "unhealthy",
                True,
                fields,
                snapshot.unavailable.get("active_memory", "active memory unavailable"),
                failures,
                time.time() + delay,
            )
            return delay
        self._health = CollectorHealthState(
            "degraded" if fields else "healthy",
            bool(fields),
            fields,
            "; ".join(snapshot.unavailable.values()) or None,
        )
        return None

    def _emit(self, record: dict[str, Any]) -> None:
        self._events.append(record)
        self._total_events += 1
        if self._oom_recorder is not None:
            self._oom_recorder.record_event(record)
        if self._sink is not None:
            try:
                self._sink.append(record)
            except Exception as exc:
                self._sink_error = f"{type(exc).__name__}: {exc}"

    def _check_threshold(self, snapshot: MemorySnapshot) -> None:
        value = snapshot.active_bytes
        if value is None or self.alert_threshold_mb is None:
            return
        if value >= self.alert_threshold_mb * 1024**2:
            self._alerts.append(
                {
                    "timestamp_ns": snapshot.timestamp_ns,
                    "active_bytes": value,
                    "threshold_mb": self.alert_threshold_mb,
                    "action": "alert_only",
                }
            )
            self._total_alerts += 1

    def get_results(self) -> TrackingResult:
        with self._lock:
            if self._session is None:
                raise RuntimeError("No tracking session has started")
            history = self._history
            diagnostics: dict[str, Any] = {"last_error": self._sink_error}
            if self._sink is not None:
                diagnostics.update(self._sink.get_diagnostics())
                diagnostics.update(self._sink.failure_diagnostics())
            health = self._health.to_dict()
            health["sampler_error"] = (
                None if self._sampler.error is None else str(self._sampler.error)
            )
            return TrackingResult(
                self._session,
                history.total,
                history.valid,
                history.peak,
                None if not history.valid else history.sum_active / history.valid,
                history.minimum,
                list(history.samples),
                list(self._events),
                list(self._alerts),
                self._total_events,
                self._total_alerts,
                self.max_history,
                health,
                diagnostics,
            )

    def export(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.get_results().to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")

    def enter_phase(
        self, name: str, attrs: Mapping[str, Any] | None = None
    ) -> PhaseHandle:
        if not self.is_tracking or self._session is None:
            raise RuntimeError("Start tracking before entering a phase")
        token, boundary = self._phase_recorder.enter(
            session_id=self._session.session_id,
            rank=self._session.rank,
            name=name,
            attrs=attrs,
        )
        self._sample(boundary.event_type, boundary.context, boundary.metadata)
        return PhaseHandle(
            scope_id=token.scope_id,
            name=name,
            path=boundary.path,
            close_callback=lambda: self.exit_phase(token),
        )

    def exit_phase(self, token: PhaseToken) -> None:
        if self._session is None or self._session.status != "running":
            raise RuntimeError("Cannot close a phase after tracking stops")
        if token.session_id != self._session.session_id:
            raise RuntimeError("Phase belongs to a previous tracking session")
        boundary = self._phase_recorder.exit(token)
        self._sample(boundary.event_type, boundary.context, boundary.metadata)

    def phase(self, name: str, attrs: Mapping[str, Any] | None = None) -> PhaseHandle:
        return self.enter_phase(name, attrs)

    @contextmanager
    def tracking(self) -> Iterator[MemoryTracker]:
        self.start_tracking()
        try:
            yield self
        except BaseException as exc:
            with suppress(BaseException):
                self.record_exception(exc)
            self._stop_best_effort(
                ("interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete")
            )
            raise
        else:
            self.stop_tracking()

    def _stop_best_effort(self, status: str) -> None:
        try:
            self.stop_tracking(status=status)
        except BaseException as exc:
            self._sink_error = f"cleanup: {type(exc).__name__}: {exc}"
            with suppress(BaseException):
                self._sampler.stop()
            if self._session is not None:
                self._session = update_session_summary(
                    self._session,
                    status=status,
                    ended_at_ns=time.time_ns(),
                )
            with suppress(BaseException):
                self._close_sink(status)

    def record_exception(
        self, exc: BaseException, context: str | None = None
    ) -> str | None:
        from .oom import classify_oom_exception

        classification = classify_oom_exception(exc)
        if not classification.is_oom or self._oom_recorder is None:
            return None
        try:
            return self._oom_recorder.dump(
                reason=classification.reason or "mlx_oom",
                exception=exc,
                context=context,
                backend="metal",
                metadata={"framework": "mlx"},
                session_summary=self._session,
            )
        except Exception:
            return None  # Diagnostic failure must not replace the user exception.
