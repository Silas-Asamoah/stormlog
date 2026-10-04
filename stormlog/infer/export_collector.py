"""Prometheus health for ``infer collect-server``, labelled with what it observes.

The collector's own health is exported, never its measurements: memory
values stay in its JSONL, where the analysis joins them to the run, and a
DCGM or node exporter already reports device and process memory to
Prometheus. ``stormlog_collector_info`` carries the identity the collector
confirmed (host, boot, process, GPU or MIG instance, replica, group, rank)
as labels on one series, so a dashboard can tell collectors apart and see
which GPU each one watched. An identity part that is not known is empty,
never guessed.

Every label value is known once the collector has found its process and
GPU, so the series and the budget are fixed then, before anything is
collected; a run over the budget, or a slot another live run holds, is a
usage error. The collector closes the export in its stop path, with the
reason it stopped, and the final textfile says so.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .._export.http_server import MetricsServer
from .._export.registry import BudgetExceeded, Family, FamilySpec, Registry, render
from .._export.renders import RenderCache
from .._export.textfile import PRODUCER_LABEL, SlotInUse, TextfileWriter
from .export import ExportUsageError
from .export_config import ExportConfig
from .server_collector import (
    STOP_DURATION_ELAPSED,
    STOP_GPU_IDENTITY_CHANGED,
    STOP_REQUESTED,
    STOP_SERVER_PROCESS_ENDED,
)
from .telemetry import SAMPLE_STATES, ServerIdentity, TelemetrySample

STOP_REASONS = (
    STOP_DURATION_ELAPSED,
    STOP_REQUESTED,
    STOP_SERVER_PROCESS_ENDED,
    STOP_GPU_IDENTITY_CHANGED,
    "error",
)
SAMPLE_METRICS = (
    "process_rss_bytes",
    "device_memory_used_bytes",
    "device_memory_reserved_bytes",
    "instance_memory_used_bytes",
    "instance_memory_reserved_bytes",
)
IDENTITY_LABELS = (
    "run_id",
    "host",
    "boot_id",
    "pid",
    "process_start_ns",
    "device_uuid",
    "gpu_instance_id",
    "replica_id",
    "group_id",
    "rank",
    "world_size",
    "version",
)
CLOSE_SECONDS = 5.0


class CollectorExport:
    """The collector's health at ``/metrics`` or in a textfile."""

    def __init__(
        self,
        config: ExportConfig,
        *,
        run_id: str,
        version: str,
        forbidden_paths: Sequence[Path] = (),
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.version = version
        self.on_warning = on_warning
        self.registry = Registry(
            const_labels={PRODUCER_LABEL: config.prometheus_slot},
            max_samples=config.prometheus_max_series,
            max_bytes=config.prometheus_max_bytes,
            headroom=config.headroom("collect-server"),
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
        self._families: dict[str, Family] = {}
        self._identity_values: tuple[str, ...] = ()
        self.polls = 0

    def prepare(self) -> None:
        """Take the textfile's slot; a clash is a usage error, before collecting."""
        if self.textfile is None:
            return
        try:
            self.textfile.acquire()
        except (SlotInUse, ValueError) as exc:
            raise ExportUsageError(str(exc)) from exc

    # ------------------------------------------------------------- the run
    def identify(self, identity: ServerIdentity) -> None:
        """Declare the series for this identity, check the budget, start serving."""
        self._identity_values = _identity_values(self.run_id, self.version, identity)
        self._declare()
        try:
            self.registry.check_budget()
        except BudgetExceeded as exc:
            raise ExportUsageError(str(exc)) from exc
        self.prepare()
        self.registry.apply(self._started)
        self._start_server()
        if self.textfile is not None:
            self.textfile.start()

    def poll(self, samples: Sequence[TelemetrySample]) -> None:
        """Count one poll and the state of each sample in it. Never raises."""
        try:
            self.registry.apply(lambda: self._count(samples))
        except Exception:
            pass

    def close(self, stop_reason: str) -> None:
        """Freeze with the reason the collector stopped; write the final textfile."""
        reason = stop_reason if stop_reason in STOP_REASONS else "error"

        def final() -> None:
            self._families["running"].set((), 0.0)
            self._families["stops"].inc((reason,))

        try:
            if self._families:
                self.registry.freeze(final=final)
            self.renders.invalidate()
            if self.textfile is not None:
                self.textfile.close(CLOSE_SECONDS)
        except Exception:
            pass

    def stop_serving(self) -> None:
        if self.server is not None:
            self.server.stop()

    # ------------------------------------------------------------- helpers
    def _declare(self) -> None:
        if self._families:
            return
        add = self.registry.add
        self._families = {
            "info": add(
                FamilySpec(
                    "stormlog_collector_info",
                    "gauge",
                    "1 for the collector, labelled with the identity it confirmed. "
                    "An unknown part is empty.",
                    labels=IDENTITY_LABELS,
                ),
                [dict(zip(IDENTITY_LABELS, self._identity_values))],
            ),
            "running": add(
                FamilySpec(
                    "stormlog_collector_running",
                    "gauge",
                    "1 while the collector polls; 0 once it has stopped.",
                )
            ),
            "start": add(
                FamilySpec(
                    "stormlog_collector_start_time_seconds",
                    "gauge",
                    "When the collector started, in Unix seconds.",
                    unit="seconds",
                )
            ),
            "polls": add(
                FamilySpec(
                    "stormlog_collector_polls_total",
                    "counter",
                    "Polls written to the collector's output.",
                )
            ),
            "last_poll": add(
                FamilySpec(
                    "stormlog_collector_last_poll_timestamp_seconds",
                    "gauge",
                    "When the last poll was taken, in Unix seconds.",
                    unit="seconds",
                ),
                precreate=False,
            ),
            "samples": add(
                FamilySpec(
                    "stormlog_collector_samples_total",
                    "counter",
                    "Samples by what was read and whether it could be: valid, "
                    "missing, stale or invalid. The values are in the output, "
                    "not here.",
                    labels=("metric", "state"),
                    enums={
                        "metric": SAMPLE_METRICS,
                        "state": tuple(sorted(SAMPLE_STATES)),
                    },
                )
            ),
            "stops": add(
                FamilySpec(
                    "stormlog_collector_stops_total",
                    "counter",
                    "Why the collector stopped; 1 for the reason, in the final "
                    "values.",
                    labels=("reason",),
                    enums={"reason": STOP_REASONS},
                )
            ),
        }

    def _started(self) -> None:
        families = self._families
        families["info"].set(self._identity_values, 1.0)
        families["running"].set((), 1.0)
        families["start"].set((), time.time())

    def _count(self, samples: Sequence[TelemetrySample]) -> None:
        families = self._families
        self.polls += 1
        families["polls"].inc(())
        if samples:
            families["last_poll"].set((), samples[0].observed_at_ns / 1e9)
        for sample in samples:
            families["samples"].inc((sample.metric, sample.state))

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
                f"{self.server_error}; collector health is not served"
            )
            return
        self.server = server
        if not server.loopback:
            self._warn(
                f"Prometheus endpoint {server.address} is not on loopback and "
                "has no authentication"
            )

    def _warn(self, message: str) -> None:
        if self.on_warning is not None:
            self.on_warning(message)


def _identity_values(
    run_id: str, version: str, identity: ServerIdentity
) -> tuple[str, ...]:
    def text(value: object) -> str:
        return "" if value is None else str(value)

    return (
        run_id,
        identity.host,
        text(identity.boot_id),
        str(identity.pid),
        str(identity.process_start_ns),
        text(identity.device_uuid),
        text(identity.gpu_instance_id),
        text(identity.replica_id),
        text(identity.group_id),
        text(identity.rank),
        text(identity.world_size),
        version,
    )
