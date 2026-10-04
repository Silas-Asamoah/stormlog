"""The export pipeline: records in, Prometheus out, never in the run's way.

The pipeline is told about each record the run writes (``observe``) and
copies the fields it needs into a bounded queue; a worker thread applies them
to the registry, a health thread reads the health sources once a second,
and the ``/metrics`` server and the textfile writer render the registry.
None of them can block the producer or change the run's exit code:
every entry point catches its own failures and counts them.

``close`` ends it in a fixed order: stop taking records, let the worker
finish within the deadline, then freeze the registry, with any record still
unapplied counted as dropped at shutdown, so the counts written to the
artifact are final. The ``/metrics`` endpoint keeps serving the frozen values
until ``stop_serving``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Union

from .._export.envelope import Envelope
from .._export.http_server import MetricsServer
from .._export.queue import BoundedQueue, QueueStats
from .._export.registry import Family, FamilySpec, Registry, render
from .._export.renders import RenderCache
from .._export.textfile import PRODUCER_LABEL, SlotInUse, TextfileWriter
from .correlation_events import CapabilityEvent, CorrelationContext
from .export_config import Command, ExportConfig
from .export_metrics import ProfileLabels, ProfileMetrics

COMPONENT_PROMETHEUS = "export.prometheus"
METRIC_QUEUE_ITEMS = 65_536
METRIC_QUEUE_BYTES = 8 * 1024 * 1024
HEALTH_INTERVAL_SECONDS = 1.0
DROP_REASONS = ("queue_full", "closed", "shutdown", "error")
# The textfile's final write gets at least this long, whatever is left of
# the close's deadline: it renders once and writes one file.
TEXTFILE_CLOSE_FLOOR_SECONDS = 0.5
# The steps of a close, in order; each runs once, across calls.
_CLOSE_STEPS = (
    "_close_queue",
    "_join_worker",
    "_stop_poller",
    "_freeze",
    "_invalidate",
    "_close_textfile",
)
# Drops within this window make the pipeline report itself degraded.
HEALTH_WINDOW_SECONDS = 60

HealthScalar = Union[int, float, str, None]
HealthValue = Union[HealthScalar, Mapping[tuple[str, ...], HealthScalar]]


@dataclass(frozen=True)
class HealthMetric:
    """How one key of a health snapshot becomes a metric family.

    A labelled family's value maps label-value tuples to scalars. A ``state``
    family's value is the current state, exported one-hot over ``states``.
    ``enums`` optionally lists the closed values of some labels. A ``None``
    value leaves its series out; it is never exported as 0.
    """

    key: str
    name: str
    kind: Literal["gauge", "counter", "state"]
    unit: str
    help: str
    labels: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    enums: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class HealthSource(Protocol):
    """A component whose own health becomes metrics; ``health`` is a pure read."""

    def health(self) -> Mapping[str, HealthValue]: ...

    def health_metrics(self) -> Sequence[HealthMetric]: ...


class ExportUsageError(ValueError):
    """An export setting the run cannot use: refused before anything is sent."""


@dataclass
class _Counts:
    applied: int = 0
    dropped_shutdown: int = 0
    # Records whose update raised and was undone: an exporter bug.
    dropped_error: int = 0
    internal_errors: dict[str, int] = field(
        default_factory=lambda: {"observe": 0, "apply": 0, "health": 0, "close": 0}
    )
    last_health_at: float | None = None


class ExportPipeline:
    """Records in, metrics out; bounded, isolated, closed in a fixed order."""

    def __init__(
        self,
        config: ExportConfig,
        labels: ProfileLabels,
        *,
        command: Command = "profile",
        health: Sequence[tuple[str, Sequence[HealthMetric]]] = (),
        forbidden_paths: Sequence[Path] = (),
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.on_warning = on_warning
        self.registry = Registry(
            const_labels={PRODUCER_LABEL: config.prometheus_slot},
            max_samples=config.prometheus_max_series,
            max_bytes=config.prometheus_max_bytes,
            headroom=config.headroom(command),
        )
        self.metrics = ProfileMetrics(self.registry, labels)
        self._health: dict[str, tuple[list[tuple[HealthMetric, Family]], Any]] = {}
        for name, metrics in health:
            self.declare_health(name, metrics)
        self._own = _OwnMetrics(
            self.registry, textfile=bool(config.prometheus_textfile_dir)
        )
        self.budget = self.registry.check_budget()
        self.queue: BoundedQueue[Envelope] = BoundedQueue(
            max_items=METRIC_QUEUE_ITEMS, max_bytes=METRIC_QUEUE_BYTES
        )
        self.renders = RenderCache(lambda: render(self.registry.snapshot()))
        self.server: MetricsServer | None = None
        self.server_error: str | None = None
        self.textfile = (
            TextfileWriter(
                config.prometheus_textfile_dir,
                config.prometheus_slot,
                self.renders,
                const_labels=self.registry.const_labels,
                interval=config.prometheus_textfile_interval_seconds,
                remove_on_exit=config.prometheus_textfile_remove_on_exit,
                forbidden=forbidden_paths,
            )
            if config.prometheus_textfile_dir is not None
            else None
        )
        self._counts = _Counts()
        self._drops = _DropWindow()
        self._stop_health = threading.Event()
        self._stop_applying = threading.Event()
        self._worker: threading.Thread | None = None
        self._poller: threading.Thread | None = None
        self._closed = False
        self._close_done: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------- set-up
    def declare_health(self, name: str, metrics: Sequence[HealthMetric]) -> None:
        """Declare a health source's families; before the budget is checked."""
        families = [
            (metric, _health_family(self.registry, metric)) for metric in metrics
        ]
        self._health[name] = (families, None)

    def attach_health(self, name: str, source: HealthSource) -> None:
        """Start reading a declared health source."""
        with self._lock:
            families, _ = self._health[name]
            self._health[name] = (families, source)

    def prepare(self) -> None:
        """Take the textfile's lock; a clash is a usage error, before the run."""
        if self.textfile is None:
            return
        try:
            self.textfile.acquire()
        except (SlotInUse, ValueError) as exc:
            raise ExportUsageError(str(exc)) from exc

    def start(self, started_at: float) -> None:
        """Start the threads and outputs. A port that cannot be bound warns."""
        self.prepare()
        self.registry.apply(lambda: self.metrics.set_run_info(started_at))
        self._worker = threading.Thread(
            target=self._apply_loop, name="stormlog-export-metrics", daemon=True
        )
        self._worker.start()
        self._poller = threading.Thread(
            target=self._health_loop, name="stormlog-export-health", daemon=True
        )
        self._poller.start()
        self._start_server()
        if self.textfile is not None:
            self.textfile.start()

    def _start_server(self) -> None:
        listen = self.config.prometheus_listen
        if listen is None:
            return
        server = MetricsServer(listen, self.renders)
        try:
            server.start()
        except OSError as exc:
            self.server_error = f"{type(exc).__name__}: {exc}"
            self._warn(
                f"Prometheus endpoint could not listen on {listen}: "
                f"{self.server_error}; metrics are not served"
            )
            return
        self.server = server
        if not server.loopback:
            self._warn(
                f"Prometheus endpoint {server.address} is not on loopback and "
                "has no authentication; anyone who can reach it can read the "
                "run's metrics"
            )

    # ------------------------------------------------------------- the run
    def observe(
        self, record: dict[str, Any], extras: Mapping[str, Any] | None = None
    ) -> None:
        """Queue what the metrics need from ``record``; never raises or blocks.

        Records after ``close`` (the run's capability records, written once
        the counts are final) are not taken.
        """
        if self._closed:
            return
        try:
            envelope = self.metrics.envelope(record, extras)
            if envelope is not None:
                self.queue.offer(envelope, envelope.size)
        except Exception:
            self._error("observe")

    def _apply_loop(self) -> None:
        while True:
            batch = self.queue.take(timeout=0.25)
            if not batch:
                if self.queue.stats().closed:
                    return
                continue
            for envelope in batch:
                # Past the close's wait the freeze must not queue behind
                # the backlog: what is left is counted as a shutdown drop.
                if self._stop_applying.is_set() or not self._apply(envelope):
                    return

    def _apply(self, envelope: Envelope) -> bool:
        def update() -> None:
            self.metrics.apply(envelope)
            self._counts.applied += 1

        try:
            return self.registry.apply(update)
        except Exception:
            self._error("apply")
            # Counted under the registry's lock, so the final counts see it
            # either as this error or, if the freeze came first, as a
            # shutdown drop, never as both.
            self.registry.apply(self._note_apply_error)
            return True

    def _note_apply_error(self) -> None:
        self._counts.dropped_error += 1

    def _health_loop(self) -> None:
        while not self._stop_health.wait(HEALTH_INTERVAL_SECONDS):
            self._poll_health()

    def _poll_health(self) -> None:
        self._poll_sources()
        self._counts.last_health_at = time.monotonic()
        self._drops.note(self._dropped_total())
        self.registry.apply(self._update_own)

    def _poll_sources(self) -> None:
        with self._lock:
            sources = list(self._health.values())
        for families, source in sources:
            if source is None:
                continue
            try:
                snapshot = source.health()
                self.registry.apply(lambda: _apply_health(families, snapshot))
            except Exception:
                self._error("health")

    def _update_own(self) -> None:
        queue = self.queue.stats()
        self._own.update(
            applied=self._counts.applied,
            dropped=self._drop_counts(queue),
            families=self.registry.families,
            server=self.server,
            textfile=self.textfile,
            health_age=self._health_age(),
            errors=self._counts.internal_errors,
        )

    # ------------------------------------------------------------- the end
    def close(self, deadline: float) -> None:
        """Stop, drain within ``deadline`` seconds, freeze; never raises an error.

        The steps run in order, each once across calls. An interrupt, such as
        a Ctrl+C, inside one ends that step; the steps after it then run
        without waiting, and the interrupt propagates once they are done. A
        later close finishes any step an interrupt kept from starting.
        """
        with self._lock:
            self._closed = True
            pending = [step for step in _CLOSE_STEPS if step not in self._close_done]
        until = time.monotonic() + max(0.0, deadline)
        interrupt: BaseException | None = None
        for step in pending:
            try:
                getattr(self, step)(until if interrupt is None else 0.0)
            except Exception:
                self._error("close")
            except BaseException as exc:  # KeyboardInterrupt, SystemExit
                interrupt = interrupt or exc
            finally:
                self._close_done.add(step)
        if interrupt is not None:
            raise interrupt

    def _close_queue(self, until: float) -> None:
        self.queue.close()

    def _join_worker(self, until: float) -> None:
        try:
            if self._worker is not None:
                self._worker.join(max(0.0, until - time.monotonic()))
        finally:
            self._stop_applying.set()

    def _stop_poller(self, until: float) -> None:
        self._stop_health.set()
        poller = self._poller
        if poller is not None:
            poller.join(max(0.0, min(1.0, until - time.monotonic())))
        # A last read of the sources, unless one is stuck in the poller.
        if poller is None or not poller.is_alive():
            self._poll_sources()

    def _freeze(self, until: float) -> None:
        self.registry.freeze(final=self._final_counts)

    def _invalidate(self, until: float) -> None:
        # Renders from before the freeze no longer show the final values.
        self.renders.invalidate()

    def _close_textfile(self, until: float) -> None:
        if self.textfile is not None:
            left = until - time.monotonic()
            self.textfile.close(max(left, TEXTFILE_CLOSE_FLOOR_SECONDS))

    def _final_counts(self) -> None:
        # Under the registry's lock: no record can be applied meanwhile, so
        # what the queue accepted and the worker did not apply is exact.
        self.queue.drain()
        self._counts.dropped_shutdown = (
            self.queue.stats().accepted
            - self._counts.applied
            - self._counts.dropped_error
        )
        self._update_own()

    def stop_serving(self) -> None:
        if self.server is not None:
            try:
                self.server.stop()
            except Exception:
                self._error("close")

    # ------------------------------------------------------------- reading
    def health(self) -> dict[str, Any]:
        """A pure read of the pipeline's own state."""
        queue = self.queue.stats()
        dropped = self._dropped_total()
        status = "healthy"
        if self._drops.recent(dropped):
            status = "degraded"
        if self.server_error is not None and self.textfile is None:
            status = "unhealthy"
        return {
            "status": status,
            "applied": self._counts.applied,
            "queued": queue.depth,
            "dropped": self._drop_counts(queue),
            "frozen": self.registry.frozen,
        }

    def summary(self) -> dict[str, Any]:
        """The figures the capability record keeps; final once closed."""
        queue = self.queue.stats()
        families = self.registry.families
        textfile = self.textfile
        server = self.server
        return {
            "records": {
                "offered": queue.offered,
                "applied": self._counts.applied,
                "dropped": self._drop_counts(queue),
                # Every record offered was applied, and none was lost before
                # it could be offered.
                "exact": self._dropped_total() == 0
                and self._counts.internal_errors["observe"] == 0,
                "queue_high_water": queue.high_water,
            },
            "budget": {
                "samples": self.budget.samples,
                "max_bytes_per_scrape": self.budget.size,
                "max_series": self.config.prometheus_max_series,
                "max_bytes": self.config.prometheus_max_bytes,
            },
            "series": {f.spec.name: f.stats.series for f in families},
            "series_overflow": {
                f.spec.name: f.stats.overflow_redirects
                for f in families
                if f.stats.overflow_redirects
            },
            "series_rejected": {
                f.spec.name: f.stats.rejected for f in families if f.stats.rejected
            },
            "scrapes": _server_counts(server),
            "endpoint": server.address if server is not None else None,
            "endpoint_error": self.server_error,
            "textfile": _textfile_summary(textfile),
            "slot": self.config.prometheus_slot,
            "internal_errors": dict(self._counts.internal_errors),
        }

    def capability_events(self, context: CorrelationContext) -> list[CapabilityEvent]:
        """The ``export.prometheus`` record: what was asked for and what worked."""
        enabled = []
        collected = []
        if self.config.prometheus_listen is not None:
            enabled.append("endpoint")
            if self.server is not None:
                collected.append("endpoint")
        if self.textfile is not None:
            enabled.append("textfile")
            if self.textfile.stats.writes_ok:
                collected.append("textfile")
        available = bool(collected) or self.server is not None
        return [
            CapabilityEvent(
                context=context,
                event_id=f"capability:{COMPONENT_PROMETHEUS}",
                component=COMPONENT_PROMETHEUS,
                available=available,
                supported=["endpoint", "textfile"] if available else [],
                enabled=enabled if available else [],
                collected=collected if available else [],
                metadata={"summary": self.summary()},
            )
        ]

    # ------------------------------------------------------------- helpers
    def _drop_counts(self, queue: QueueStats) -> dict[str, int]:
        return {
            "queue_full": queue.dropped_full,
            "closed": queue.dropped_closed,
            "shutdown": self._counts.dropped_shutdown,
            "error": self._counts.dropped_error,
        }

    def _dropped_total(self) -> int:
        """Records offered but not applied; 0 while the totals are exact."""
        drops = self._drop_counts(self.queue.stats())
        return drops["queue_full"] + drops["shutdown"] + drops["error"]

    def _health_age(self) -> float | None:
        last = self._counts.last_health_at
        return None if last is None else time.monotonic() - last

    def _error(self, entry: str) -> None:
        errors = self._counts.internal_errors
        errors[entry] = errors.get(entry, 0) + 1

    def _warn(self, message: str) -> None:
        if self.on_warning is not None:
            try:
                self.on_warning(message)
            except Exception:
                pass


# ------------------------------------------------------------------ health
def _health_family(registry: Registry, metric: HealthMetric) -> Family:
    labels = metric.labels
    enums = dict(metric.enums)
    kind: Literal["gauge", "counter"] = (
        "counter" if metric.kind == "counter" else "gauge"
    )
    if metric.kind == "state":
        labels = labels + ("state",)
        enums["state"] = metric.states
    known: list[dict[str, str]] = [{}] if not set(labels) - set(enums) else []
    return registry.add(
        FamilySpec(
            metric.name,
            kind,
            metric.help,
            unit=metric.unit,
            labels=labels,
            enums=enums,
        ),
        known,
        precreate=False,
    )


def _apply_health(
    families: Sequence[tuple[HealthMetric, Family]],
    snapshot: Mapping[str, HealthValue],
) -> None:
    for metric, family in families:
        value = snapshot.get(metric.key)
        if value is None:
            continue
        items = value.items() if isinstance(value, Mapping) else [((), value)]
        for labels, item in items:
            if item is None:
                continue
            if metric.kind == "state":
                for state in metric.states:
                    family.set(tuple(labels) + (state,), 1.0 if item == state else 0.0)
            elif isinstance(item, (int, float)):
                family.set(tuple(labels), float(item))


@dataclass
class _DropWindow:
    """Drop totals per second, to tell recent drops from old ones."""

    # Starts with no drops at the pipeline's start.
    samples: deque[tuple[float, int]] = field(
        default_factory=lambda: deque([(time.monotonic(), 0)], maxlen=128)
    )

    def note(self, total: int) -> None:
        self.samples.append((time.monotonic(), total))

    def recent(self, total: int) -> bool:
        cutoff = time.monotonic() - HEALTH_WINDOW_SECONDS
        older = [count for at, count in self.samples if at <= cutoff]
        baseline = older[-1] if older else (self.samples[0][1] if self.samples else 0)
        return total > baseline


class _OwnMetrics:
    """The pipeline's own health, as metrics beside what it exports."""

    def __init__(self, registry: Registry, *, textfile: bool) -> None:
        self.applied = registry.add(
            FamilySpec(
                "stormlog_metrics_records_applied_total",
                "counter",
                "Records applied to these metrics.",
            )
        )
        self.dropped = registry.add(
            FamilySpec(
                "stormlog_metrics_records_dropped_total",
                "counter",
                "Records these metrics never saw, by why. While all are 0, the "
                "totals are exact.",
                labels=("reason",),
                enums={"reason": DROP_REASONS},
            )
        )
        self.overflow = registry.add(
            FamilySpec(
                "stormlog_metrics_series_overflow_total",
                "counter",
                "Updates folded into an __overflow__ series past a family's cap.",
            )
        )
        self.rejected = registry.add(
            FamilySpec(
                "stormlog_metrics_series_rejected_total",
                "counter",
                "Updates rejected: a gauge past its cap, a value outside a "
                "closed set, or a non-finite observation.",
            )
        )
        self.scrapes = registry.add(
            FamilySpec(
                "stormlog_metrics_scrapes_total",
                "counter",
                "Requests to the /metrics endpoint by outcome.",
                labels=("outcome",),
                enums={"outcome": SCRAPE_OUTCOMES},
            )
        )
        self.textfile = (
            registry.add(
                FamilySpec(
                    "stormlog_metrics_textfile_writes_total",
                    "counter",
                    "Textfile writes by outcome.",
                    labels=("outcome",),
                    enums={"outcome": ("ok", "failed")},
                )
            )
            if textfile
            else None
        )
        self.health_age = registry.add(
            FamilySpec(
                "stormlog_health_snapshot_age_seconds",
                "gauge",
                "Seconds since the health sources were last read.",
                unit="seconds",
            ),
            precreate=False,
        )
        self.errors = registry.add(
            FamilySpec(
                "stormlog_exporter_internal_errors_total",
                "counter",
                "Exporter failures caught before they could reach the run.",
                labels=("entry",),
                enums={"entry": ("observe", "apply", "health", "close")},
            )
        )

    def update(
        self,
        *,
        applied: int,
        dropped: Mapping[str, int],
        families: Sequence[Family],
        server: MetricsServer | None,
        textfile: TextfileWriter | None,
        health_age: float | None,
        errors: Mapping[str, int],
    ) -> None:
        self.applied.set((), float(applied))
        for reason, count in dropped.items():
            self.dropped.set((reason,), float(count))
        self.overflow.set((), float(sum(f.stats.overflow_redirects for f in families)))
        self.rejected.set((), float(sum(f.stats.rejected for f in families)))
        for outcome, count in _server_counts(server).items():
            self.scrapes.set((outcome,), float(count))
        if self.textfile is not None and textfile is not None:
            self.textfile.set(("ok",), float(textfile.stats.writes_ok))
            self.textfile.set(("failed",), float(textfile.stats.writes_failed))
        if health_age is not None:
            self.health_age.set((), health_age)
        for entry, count in errors.items():
            self.errors.set((entry,), float(count))


SCRAPE_OUTCOMES = (
    "ok",
    "rejected_busy",
    "timeout",
    "not_found",
    "bad_request",
    "error",
)


def _server_counts(server: MetricsServer | None) -> dict[str, int]:
    if server is None:
        return {outcome: 0 for outcome in SCRAPE_OUTCOMES}
    stats = server.stats
    return {
        "ok": stats.ok,
        "rejected_busy": stats.rejected_busy,
        "timeout": stats.timeout,
        "not_found": stats.not_found,
        "bad_request": stats.bad_request,
        "error": stats.errors,
    }


def _textfile_summary(textfile: TextfileWriter | None) -> dict[str, Any] | None:
    if textfile is None:
        return None
    stats = textfile.stats
    return {
        "path": textfile.path.name,
        "writes_ok": stats.writes_ok,
        "writes_failed": stats.writes_failed,
        "abandoned": stats.abandoned,
        # The final write had only a render from before the values froze.
        "final_stale": stats.final_stale,
    }


# ------------------------------------------------------------------ receiver
RECEIVER_OUTCOMES = {
    "decode_failures": "decode_failure",
    "unsupported_media": "unsupported_media",
    "protobuf_unavailable": "protobuf_unavailable",
    "grpc_attempts": "grpc_attempt",
    "oversized": "oversized",
    "bad_requests": "bad_request",
    "handler_errors": "handler_error",
    "after_stop": "after_stop",
}
RECEIVER_HEALTH = "vllm_span_receiver"


class ReceiverHealth:
    """The span receiver's own counts, read through its capability metadata."""

    def __init__(self, receiver: Any) -> None:
        self.receiver = receiver

    @staticmethod
    def health_metrics() -> Sequence[HealthMetric]:
        return (
            HealthMetric(
                key="requests",
                name="stormlog_engine_span_receiver_requests_total",
                kind="counter",
                unit="",
                help="Span exports the receiver refused or could not read, by "
                "why. These count HTTP requests, not spans.",
                labels=("outcome",),
                enums={"outcome": tuple(RECEIVER_OUTCOMES.values())},
            ),
            HealthMetric(
                key="spans",
                name="stormlog_engine_span_receiver_spans_total",
                kind="counter",
                unit="",
                help="Spans the receiver decoded and kept. Never forwarded.",
            ),
        )

    def health(self) -> Mapping[str, HealthValue]:
        metadata = self.receiver.capability_metadata()
        return {
            "requests": {
                (outcome,): metadata.get(key)
                for key, outcome in RECEIVER_OUTCOMES.items()
            },
            "spans": metadata.get("spans"),
        }
