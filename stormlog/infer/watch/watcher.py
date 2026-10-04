"""The ``stormlog infer watch`` loop.

Once per tick the watcher scrapes the server's ``/metrics`` (one scrape in
flight; a tick that comes due during a slow scrape is skipped and counted),
keeps the scrape in its bounded history, evaluates every trigger, and opens,
joins or seals incidents. Between ticks it polls its control files every
0.1 s. Blocking I/O runs on serial workers: the ledger on its own, bundle
writes on the store's. Nothing it does can block vLLM, which only sees
``GET /metrics``.

The watch ends when its duration elapses or on SIGINT or SIGTERM, the
documented way to stop it. It seals open incidents as interrupted, waits for
its writers within one shutdown deadline, writes ``report.json`` and returns
the exit code: 1 when the watch could not judge or keep what it saw (no
successful scrape, a failing ledger, every incident write failing) or could
not write its report; else 3 when a counting incident was detected; else 0.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ... import __version__
from ...exit_codes import ExitCode
from ...report import (
    REPORT_FILENAME,
    Artifact,
    Evidence,
    Finding,
    build_report,
    write_report,
)
from ..scrape_window import REASON_ENGINE_REQUIRED
from ..trace_capture import server_root
from ..vllm_scraper import VllmMetricsScraper
from ..vllm_telemetry import MARKER_INTERVAL, SCRAPE_OK, VllmScrapeRecord
from .config import DEFAULTS_VERSION, WatchConfig
from .evaluate import TickResult, TriggerEngine
from .history import ScrapeHistory, Stamped
from .incidents import Identity, IncidentManager, WatchClock
from .io import SerialWorker
from .ledger import Ledger, RecordObserver
from .predicates import FrozenExporter, exporter_restarted
from .records import (
    INCIDENT_PRUNED,
    TRIGGER_STATE,
    WATCH_HEALTH,
    WATCH_SESSION,
    envelope,
    finite,
)
from .stats import WatchStats, counter_value
from .store import IncidentStore, PrunedBundle
from .triggers import EVENT_FIRED, VIOLATING

REPORT_KIND = "inference_watch"
TEST_TRIGGER_FILE = "test-trigger"
CONTROL_POLL_SECONDS = 0.1
PRUNE_EVERY_TICKS = 60
# The scraper's error for a body or series count over its cap.
OVERSIZED_PREFIX = "oversized:"
_NS = 1_000_000_000


@dataclass(frozen=True)
class WatchOptions:
    """What the command line adds to the config."""

    duration_seconds: float | None = None
    ready_file: Path | None = None
    test_trigger_every: float | None = None
    test_trigger_file: bool = False
    shutdown_deadline_seconds: float = 30.0
    api_key: str | None = None
    argv: tuple[str, ...] | None = None


@dataclass
class WatchOutcome:
    """How the watch ended."""

    exit_code: int
    report_path: Path | None
    unsound: list[str] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)


class Watcher:
    """Watch one vLLM server and record its incidents."""

    def __init__(
        self,
        config: WatchConfig,
        root: Path,
        *,
        options: WatchOptions | None = None,
        observer: RecordObserver | None = None,
        scraper: VllmMetricsScraper | None = None,
        clock: WatchClock | None = None,
    ) -> None:
        self.config = config
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.options = options or WatchOptions()
        self.clock = clock or WatchClock()
        self.identity = Identity(
            session_id=str(uuid.uuid4()),
            run_id=f"watch-{uuid.uuid4().hex[:12]}",
            owner=f"{os.uname().nodename}:{os.getpid()}:{self.clock.wall_origin_ns}",
        )
        self.stats = WatchStats()
        self.ledger = Ledger(self.root, observer=observer, stats=self.stats)
        self.store = IncidentStore(self.root, config.store)
        self.history = ScrapeHistory(
            max_seconds=config.history_seconds,
            max_bytes=config.history_bytes,
            parsed_count=self._parsed_count(),
        )
        self.engine = TriggerEngine(config.triggers, tick_seconds=config.tick_seconds)
        self.scraper = scraper or self._scraper()
        self._store_worker = SerialWorker("stormlog-watch-store", max_queued=8)
        self._loop: asyncio.AbstractEventLoop | None = None
        self.incidents = IncidentManager(
            store=self.store,
            history=self.history,
            limits=config.incident,
            identity=self.identity,
            clock=self.clock,
            stats=self.stats,
            emit=self.ledger.write,
            submit=self._store_worker.submit,
            post=self._post,
            loss=self._loss,
            tick_seconds=config.tick_seconds,
        )
        self._ok_scrapes = 0
        self._failed_scrapes = 0
        self._oversized_scrapes = 0
        self._watches_frozen = any(
            isinstance(spec.predicate, FrozenExporter) for spec in config.triggers
        )
        self._previous_ok: VllmScrapeRecord | None = None
        self._ticks = 0
        # Triggers refused a window because the server runs several engines
        # and they name none; such a watch could not judge them.
        self._engine_required: set[str] = set()
        self._lag_max = 0.0
        self._next_test: int | None = None

    # ----------------------------------------------------------------- run

    async def run(self, stop: asyncio.Event) -> WatchOutcome:
        """Watch until the duration elapses or ``stop`` is set."""
        self._loop = asyncio.get_running_loop()
        recovery = self.store.recover()
        self._session("started", {"recovery": recovery.__dict__})
        self._prune()
        start = self.clock.mono_ns()
        tick_ns = int(self.config.tick_seconds * _NS)
        deadline = (
            start + int(self.options.duration_seconds * _NS)
            if self.options.duration_seconds is not None
            else None
        )
        if self.options.test_trigger_every is not None:
            self._next_test = start + int(self.options.test_trigger_every * _NS)
        next_tick = start
        while not stop.is_set():
            now = self.clock.mono_ns()
            if deadline is not None and now >= deadline:
                break
            if now >= next_tick:
                await self._tick((now - next_tick) / _NS)
                scheduled = next_tick + tick_ns
                next_tick = _next_poll(next_tick, tick_ns, self.clock.mono_ns())
                missed = (next_tick - scheduled) // tick_ns
                if missed:
                    self.stats.add("ticks_missed_total", missed)
            self._poll_controls()
            await _sleep_until(
                stop, min(CONTROL_POLL_SECONDS, _seconds(next_tick, self.clock))
            )
        return await self._shutdown()

    async def _tick(self, lag_seconds: float) -> None:
        self._ticks += 1
        started = self.clock.mono_ns()
        wall = self.clock.wall_ns()
        record = await self.scraper.scrape_async(marker=MARKER_INTERVAL)
        done = max(started, self.clock.mono_ns())
        stamp = Stamped(started, done, wall)
        self._account_scrape(record)
        self.history.add(stamp, record)
        self._check_exporter(record, done)
        results = list(self.engine.tick(done, self.history.parsed()))
        if any(_frozen(result) for result in results):
            self.stats.add("frozen_ticks_total")
        for result in results:
            self._on_result(result)
        self.incidents.on_tick(done)
        self._maybe_test(done)
        if self._ticks % PRUNE_EVERY_TICKS == 0:
            self._prune()
        self._health(record, stamp, lag_seconds)

    def _account_scrape(self, record: VllmScrapeRecord) -> None:
        if record.status == SCRAPE_OK:
            self._ok_scrapes += 1
            self.stats.add("scrapes_total", labels=("ok",))
            if self._ok_scrapes == 1 and self.options.ready_file is not None:
                _write_ready(self.options.ready_file, self.identity.session_id)
        elif (record.error or "").startswith(OVERSIZED_PREFIX):
            self._oversized_scrapes += 1
            self.stats.add("scrapes_total", labels=("oversized",))
        else:
            self._failed_scrapes += 1
            self.stats.add("scrapes_total", labels=("failed",))

    def _check_exporter(self, record: VllmScrapeRecord, at_mono: int) -> None:
        if record.status != SCRAPE_OK:
            return
        previous, self._previous_ok = self._previous_ok, record
        if previous is not None and exporter_restarted(previous, record):
            self.incidents.on_event(
                "exporter_restart", "the metrics exporter restarted", at_mono
            )

    def _on_result(self, result: TickResult) -> None:
        self.stats.set_trigger_state(result.spec.trigger_id, result.state)
        if REASON_ENGINE_REQUIRED in result.evaluation.reasons:
            self._engine_required.add(result.spec.trigger_id)
        transition = result.transition
        if transition is None:
            return
        incident_id = (
            self.incidents.on_fired(result) if transition.event == EVENT_FIRED else None
        )
        record = envelope(
            TRIGGER_STATE,
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            timestamp_ns=self.clock.to_wall(result.at_ns),
        )
        evaluation = result.evaluation
        record.update(
            trigger_id=result.spec.trigger_id,
            kind=result.spec.kind,
            event=transition.event,
            state=result.state,
            reason=transition.reason,
            classification=evaluation.classification,
            observed=finite(evaluation.observed),
            threshold=finite(evaluation.threshold),
            accumulated_ns=transition.accumulated_ns,
            pending_since_ns=(
                self.clock.to_wall(transition.pending_since_ns)
                if transition.pending_since_ns is not None
                else None
            ),
            joined_incident_id=incident_id,
        )
        self.ledger.write(record)

    def _poll_controls(self) -> None:
        if not self.options.test_trigger_file:
            return
        path = self.root / TEST_TRIGGER_FILE
        try:
            requested = path.stat().st_mtime_ns
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return
        self.incidents.on_test(self.clock.mono_ns(), requested_wall_ns=requested)

    def _maybe_test(self, at_mono: int) -> None:
        every = self.options.test_trigger_every
        if every is None or self._next_test is None or at_mono < self._next_test:
            return
        self.incidents.on_test(at_mono, requested_wall_ns=None)
        self._next_test = at_mono + int(every * _NS)

    def _prune(self) -> None:
        """Apply retention on the store's worker, then record what it removed."""
        protected = frozenset(self.incidents.open)

        def prune() -> None:
            pruned = self.store.prune(protected=protected)
            self.store.reclaim_deferred()
            held = len(self.store.bundles())
            self._post(lambda: self._pruned(pruned, held))

        self._store_worker.submit(prune)

    def _pruned(self, pruned: list[PrunedBundle], held: int) -> None:
        self.stats.set("retention_incidents", held)
        for bundle in pruned:
            self.stats.add("pruned_total")
            self.stats.add("pruned_bytes_total", bundle.bytes)
            record = envelope(
                INCIDENT_PRUNED,
                session_id=self.identity.session_id,
                run_id=self.identity.run_id,
                timestamp_ns=self.clock.wall_ns(),
            )
            record.update(
                incident_id=bundle.incident_id, reason=bundle.reason, bytes=bundle.bytes
            )
            self.ledger.write(record)

    # -------------------------------------------------------------- health

    def _health(self, record: VllmScrapeRecord, stamp: Stamped, lag: float) -> None:
        ring = self.history.ring
        self.stats.set("history_bytes", ring.bytes)
        self.stats.set("history_capacity_bytes", ring.max_bytes)
        self.stats.set("history_seconds", ring.held_seconds())
        self.stats.set("history_capacity_seconds", self.config.history_seconds)
        self.stats.set("retention_bytes", self.store.budget.used_bytes)
        self._lag_max = max(self._lag_max, lag)
        self.stats.set("loop_lag_seconds_max", self._lag_max)
        health = envelope(
            WATCH_HEALTH,
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            timestamp_ns=stamp.wall_ns,
        )
        health.update(
            scrape={
                "status": record.status,
                "duration_ms": finite(record.duration_ms),
                "error": record.error,
            },
            loop_lag_seconds=lag,
            history={
                "bytes": ring.bytes,
                "seconds": ring.held_seconds(),
                "evictions": dict(ring.evictions),
            },
            open_incidents=sorted(self.incidents.open),
        )
        self.ledger.write(health)

    def _loss(self) -> dict[str, int]:
        """Cumulative loss by key; a source that is not running is left out."""
        snapshot = self.stats.health()
        ring = self.history.ring
        ledger = self.ledger.stats()
        loss = {
            "scrapes_failed": self._failed_scrapes,
            "scrapes_oversized": self._oversized_scrapes,
            "scrape_ticks_missed": int(counter_value(snapshot, "ticks_missed_total")),
            "history_evicted_age": ring.evictions["age"],
            "history_evicted_bytes": ring.evictions["bytes"],
            # The ledger's own refusals are in ledger_dropped.
            "io_rejected": self._store_worker.stats().rejected,
            "ledger_dropped": int(ledger["ledger_dropped"]),
            "export_failures": int(ledger["export_failures"]),
        }
        if self._watches_frozen:
            loss["scrape_frozen_ticks"] = int(
                counter_value(snapshot, "frozen_ticks_total")
            )
        return loss

    # ------------------------------------------------------------ shutdown

    async def _shutdown(self) -> WatchOutcome:
        loop = asyncio.get_running_loop()
        budget = self.options.shutdown_deadline_seconds
        ends = loop.time() + budget
        self.incidents.close(self.clock.mono_ns())
        drained = await asyncio.to_thread(
            self._store_worker.close, max(0.0, ends - loop.time() - 5.0)
        )
        await asyncio.sleep(0)  # run the seals' bookkeeping posted to the loop
        unsound = self._unsound(drained)
        exit_code = self._exit_code(unsound)
        self._session("ended", {"exit_code": exit_code, "unsound": list(unsound)})
        closed = await asyncio.to_thread(
            self.ledger.close, max(0.5, ends - loop.time() - 0.5)
        )
        # Known only once the ledger is closed, so not in its "ended" record.
        late = self._ledger_unsound(closed)
        if late:
            unsound.extend(late)
            exit_code = int(ExitCode.ERROR)
        try:
            report = self._report(exit_code, unsound)
            path = self.root / REPORT_FILENAME
            write_report(path, report)
        except (OSError, ValueError):
            return WatchOutcome(
                int(ExitCode.ERROR), None, unsound, self.incidents.sealed
            )
        return WatchOutcome(exit_code, path, unsound, self.incidents.sealed)

    def _unsound(self, drained: bool) -> list[str]:
        reasons = []
        if self._ok_scrapes == 0:
            reasons.append("no_successful_scrape")
        if self._engine_required:
            reasons.append("engine_required")
        if self.incidents.persist_failures and not self.incidents.persisted:
            reasons.append("incident_writes_failing")
        if not drained:
            reasons.append("store_writer_timeout")
        return reasons

    def _ledger_unsound(self, closed: bool) -> list[str]:
        if not closed:
            return ["ledger_close_timeout"]
        ledger = self.ledger.stats()
        sink = ledger.get("sink") or {}
        if ledger["ledger_dropped"] or sink.get("consecutive_flush_failures"):
            return ["ledger_failing"]
        return []

    def _exit_code(self, unsound: list[str]) -> int:
        if unsound:
            return int(ExitCode.ERROR)
        if self.incidents.detected_counting:
            return int(ExitCode.FINDINGS)
        return int(ExitCode.OK)

    def _report(self, exit_code: int, unsound: list[str]) -> dict[str, Any]:
        findings = [
            _finding(index, incident)
            for index, incident in enumerate(self.incidents.sealed, start=1)
        ]
        summary = (
            f"watch unsound: {', '.join(unsound)}"
            if unsound
            else f"{self.incidents.detected} incident(s) detected, "
            f"{self.incidents.detected_counting} counting toward the exit code"
        )
        return build_report(
            report_kind=REPORT_KIND,
            tool_name="stormlog",
            command="infer watch",
            exit_code=exit_code,
            summary=summary,
            findings=findings,
            metrics={
                "incidents_detected": self.incidents.detected,
                "incidents_counting": self.incidents.detected_counting,
                "incidents_persisted": self.incidents.persisted,
                "incident_write_failures": self.incidents.persist_failures,
                "scrapes_ok": self._ok_scrapes,
                "scrapes_failed": self._failed_scrapes,
                "scrapes_oversized": self._oversized_scrapes,
            },
            artifacts=[
                Artifact(kind="watch_ledger", path="ledger"),
                Artifact(kind="incident_bundles", path="incidents"),
            ],
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            payload={
                "format": "stormlog.infer.watch_report",
                "version": 1,
                "config_digest": self.config.digest(),
                "defaults": DEFAULTS_VERSION,
                "unsound": unsound,
                "incidents": self.incidents.sealed,
                "stats": _jsonable(self.stats.health()),
            },
            argv=self.options.argv,
            tool_version=__version__,
        )

    # ------------------------------------------------------------- helpers

    def _session(self, phase: str, extra: Mapping[str, Any]) -> None:
        record = envelope(
            WATCH_SESSION,
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            timestamp_ns=self.clock.wall_ns(),
        )
        record.update(
            phase=phase,
            owner=self.identity.owner,
            config=self.config.resolved(),
            config_digest=self.config.digest(),
            **extra,
        )
        self.ledger.write(record)

    def _post(self, task: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            task()
            return
        loop.call_soon_threadsafe(task)

    def _parsed_count(self) -> int:
        widest = max((s.sustain.window for s in self.config.triggers), default=1.0)
        return int(widest / self.config.tick_seconds) + 8

    def _scraper(self) -> VllmMetricsScraper:
        configured = self.config.metrics_url
        url = (
            server_root(self.config.base_url) + "/metrics"
            if configured in (None, "auto")
            else str(configured)
        )
        return VllmMetricsScraper(
            url=url,
            interval_seconds=self.config.tick_seconds,
            timeout_seconds=self.config.scrape_timeout_seconds,
            session_id=self.identity.session_id,
            run_id=self.identity.run_id,
            clock_domain=self.identity.clock_domain,
            api_key=self.options.api_key,
        )


def _finding(index: int, incident: Mapping[str, Any]) -> Finding:
    counting = bool(incident["counts_toward_exit"])
    return Finding(
        id=f"watch.incident.{index}",
        kind=f"incident_{incident['kind']}",
        severity="warning" if counting else "info",
        title=f"Incident from trigger {incident['trigger_id']}",
        message=None if incident["persisted"] else "the bundle could not be written",
        evidence=[
            Evidence(
                kind="incident_bundle",
                path=incident.get("bundle"),
                record_id=str(incident["incident_id"]),
            )
        ],
    )


def _frozen(result: TickResult) -> bool:
    return (
        isinstance(result.spec.predicate, FrozenExporter)
        and result.evaluation.classification == VIOLATING
    )


def _jsonable(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Labelled families keyed by tuples, as JSON objects keyed by joined labels."""
    result: dict[str, Any] = {}
    for key, value in snapshot.items():
        if isinstance(value, Mapping):
            result[key] = {"|".join(labels): v for labels, v in value.items()}
        else:
            result[key] = value
    return result


def _next_poll(previous: int, interval: int, now: int) -> int:
    """Keep a steady cadence, but skip missed ticks instead of bursting."""
    scheduled = previous + interval
    if scheduled > now:
        return scheduled
    return now + interval - (now - previous) % interval


def _seconds(target_ns: int, clock: WatchClock) -> float:
    return max(0.0, (target_ns - clock.mono_ns()) / _NS)


async def _sleep_until(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=max(seconds, 0.001))
    except asyncio.TimeoutError:
        pass


def _write_ready(path: Path, session_id: str) -> None:
    try:
        path.write_text(session_id + "\n", encoding="utf-8")
    except OSError:
        pass


__all__ = ["REPORT_KIND", "WatchOptions", "WatchOutcome", "Watcher"]
