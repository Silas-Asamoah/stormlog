"""The watch loop end to end, against a fake vLLM ``/metrics`` endpoint."""

from __future__ import annotations

import asyncio
import errno
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from email.message import Message
from functools import partial
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request
from urllib.response import addinfourl

import pytest

from stormlog import telemetry_sink
from stormlog.infer import vllm_scraper
from stormlog.infer.errors import InferUsageError
from stormlog.infer.watch.config import resolve_watch_config
from stormlog.infer.watch.history import Stamped
from stormlog.infer.watch.incidents import WatchClock
from stormlog.infer.watch.records import (
    INCIDENT,
    INCIDENT_EVENT,
    INCIDENT_PRUNED,
    TRIGGER_STATE,
    WATCH_HEALTH,
    WATCH_SESSION,
    validate_record,
)
from stormlog.infer.watch.store import IncidentStore, open_incident_bundle
from stormlog.infer.watch.watcher import (
    LOCK_FILENAME,
    REPORT_KIND,
    Watcher,
    WatchOptions,
    WatchOutcome,
)
from tests import watch_fixture_helpers as fixtures
from tests.vllm_scrape_helpers import exposition, scrape
from tests.watch_test_helpers import (
    FakeMetrics,
    of_type,
    read_ledger,
    serve_metrics,
    watch_config,
)


class _Observer:
    def __init__(
        self, stop_when: Callable[[Mapping[str, Any]], bool] | None = None
    ) -> None:
        self.records: list[Mapping[str, Any]] = []
        self.stop_when = stop_when
        self.stop: Callable[[], object] = lambda: None  # set by _watch

    def observe(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)
        if self.stop_when is not None and self.stop_when(record):
            self.stop_when = None  # once
            self.stop()


def _watch(
    root: Path,
    payload: dict[str, Any],
    *,
    options: WatchOptions | None = None,
    observer: _Observer | None = None,
    stop_after: float | None = None,
) -> WatchOutcome:
    watcher = Watcher(
        resolve_watch_config(payload),
        root,
        options=options or WatchOptions(duration_seconds=2.0),
        observer=observer,
    )

    async def main() -> WatchOutcome:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        if stop_after is not None:
            loop.call_later(stop_after, stop.set)
        if observer is not None:
            observer.stop = lambda: loop.call_soon_threadsafe(stop.set)
        return await watcher.run(stop)

    return asyncio.run(main())


def _report(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((root / "report.json").read_text("utf-8"))
    return report


class Run:
    def __init__(self, root: Path, outcome: WatchOutcome, observer: _Observer) -> None:
        self.root = root
        self.outcome = outcome
        self.observer = observer
        self.records = read_ledger(root)


@pytest.fixture(scope="module")
def violation(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """One watch of a server whose queue stays over the trigger's threshold,
    stopped once an incident is sealed and ten scrapes are recorded: a 2 s
    watch at a load of 20 sometimes ended before its incident opened. The
    30 s duration only bounds a failure."""
    root = tmp_path_factory.mktemp("violation")
    metrics = FakeMetrics()
    metrics.waiting = 20
    observer = _Observer()

    def enough(_record: Mapping[str, Any]) -> bool:
        types = [r["event_type"] for r in observer.records]
        return INCIDENT in types and types.count(WATCH_HEALTH) >= 10

    observer.stop_when = enough
    with serve_metrics(metrics) as base_url:
        outcome = _watch(
            root,
            watch_config(base_url),
            options=WatchOptions(duration_seconds=30.0, ready_file=root / "ready"),
            observer=observer,
        )
    yield Run(root, outcome, observer)


def test_a_sustained_violation_exits_three_with_a_report(violation: Run) -> None:
    assert violation.outcome.exit_code == 3
    assert violation.outcome.unsound == []
    report = _report(violation.root)
    assert report["report_kind"] == REPORT_KIND
    assert report["verdict"]["exit_code"] == 3
    assert report["findings"][0]["id"] == "watch.incident.1"
    assert report["findings"][0]["severity"] == "warning"
    assert report["metrics"]["incidents_persisted"] >= 1
    assert report["metrics"]["incident_write_failures"] == 0
    assert report["payload"]["config_digest"]
    assert report["payload"]["stats"]["scrapes_total"]["ok"] >= 10


def test_the_incident_is_sealed_into_a_bundle(violation: Run) -> None:
    incidents = of_type(violation.records, INCIDENT)
    assert incidents
    first = incidents[0]
    assert first["trigger"]["trigger_id"] == "queue"
    assert first["counts_toward_exit"] is True
    bundle = violation.root / first["bundle"]
    with open_incident_bundle(bundle) as view:
        lines = view.file("incident.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["incident_id"] == first["incident_id"]


def test_the_ledger_records_the_whole_watch(violation: Run) -> None:
    records = violation.records
    for record in records:
        validate_record(record)
    sessions = of_type(records, WATCH_SESSION)
    assert [s["phase"] for s in sessions] == ["started", "ended"]
    assert sessions[1]["exit_code"] == 3
    fired = [r for r in of_type(records, TRIGGER_STATE) if r["event"] == "fired"]
    assert fired and fired[0]["joined_incident_id"] is not None
    opened = of_type(records, INCIDENT_EVENT)
    assert opened[0]["incident_id"] == fired[0]["joined_incident_id"]
    assert len(of_type(records, WATCH_HEALTH)) >= 10
    assert {r["session_id"] for r in records} == {sessions[0]["session_id"]}


def test_every_ledger_record_reaches_the_observer(violation: Run) -> None:
    assert len(violation.observer.records) == len(violation.records)
    assert [r["event_type"] for r in violation.observer.records] == [
        r["event_type"] for r in violation.records
    ]


def test_the_ready_file_names_the_session(violation: Run) -> None:
    session_id = of_type(violation.records, WATCH_SESSION)[0]["session_id"]
    assert (violation.root / "ready").read_text().strip() == session_id


def _keys(record: Mapping[str, Any], *path: str) -> set[str]:
    node: Any = record
    for key in path:
        node = node[key]
    return set(node)


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("trigger",),
        ("capture",),
        ("pre_window",),
        ("post_window",),
        ("pre_window", "fidelity_detail", "scrapes"),
        ("loss",),
    ],
)
def test_real_incidents_have_the_fixtures_shape(
    violation: Run, path: tuple[str, ...]
) -> None:
    (expected,) = [r for r in fixtures.records() if r["event_type"] == INCIDENT]
    for record in of_type(violation.records, INCIDENT):
        assert _keys(record, *path) == _keys(expected, *path)


@pytest.mark.parametrize(
    ("event_type", "phase"),
    [
        (WATCH_SESSION, "started"),
        (WATCH_SESSION, "ended"),
        (TRIGGER_STATE, None),
        (INCIDENT_EVENT, None),
        (WATCH_HEALTH, None),
    ],
)
def test_real_records_have_the_fixtures_shape(
    violation: Run, event_type: str, phase: str | None
) -> None:
    def matching(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            r
            for r in records
            if r["event_type"] == event_type and (phase is None or r["phase"] == phase)
        ]

    expected = matching(fixtures.records())[0]
    real = matching(violation.records)
    assert real
    for record in real:
        assert set(record) == set(expected)
    if event_type == WATCH_HEALTH:
        assert _keys(real[0], "scrape") == _keys(expected, "scrape")
        assert _keys(real[0], "history") == _keys(expected, "history")


def test_a_quiet_server_exits_zero(tmp_path: Path) -> None:
    with serve_metrics(FakeMetrics()) as base_url:
        outcome = _watch(
            tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=1.0)
        )
    assert (outcome.exit_code, outcome.unsound, outcome.incidents) == (0, [], [])
    report = _report(tmp_path)
    assert report["verdict"]["exit_code"] == 0
    assert report["findings"] == []
    assert of_type(read_ledger(tmp_path), INCIDENT) == []


@pytest.mark.parametrize(
    ("metrics_setting", "expected_authorization", "expected_warnings"),
    [
        (None, "Bearer server-secret", 0),
        ("auto", "Bearer server-secret", 0),
        ("same", "Bearer server-secret", 0),
        ("other", None, 1),
    ],
)
def test_bearer_token_stays_on_the_servers_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics_setting: str | None,
    expected_authorization: str | None,
    expected_warnings: int,
) -> None:
    authorizations: list[str | None] = []
    send_response = BaseHTTPRequestHandler.send_response

    def record_authorization(
        handler: BaseHTTPRequestHandler, code: int, message: str | None = None
    ) -> None:
        authorizations.append(handler.headers.get("Authorization"))
        send_response(handler, code, message)

    monkeypatch.setattr(BaseHTTPRequestHandler, "send_response", record_authorization)
    with serve_metrics(FakeMetrics()) as origin:
        # The watcher only contacts the metrics URL; another port is another origin.
        base_url = (
            "http://127.0.0.1:1/v1" if metrics_setting == "other" else origin + "/v1"
        )
        metrics_url = (
            metrics_setting
            if metrics_setting in (None, "auto")
            else origin + "/metrics?key=metrics-secret"
        )
        outcome = _watch(
            tmp_path,
            watch_config(
                base_url, server={"base_url": base_url, "metrics_url": metrics_url}
            ),
            options=WatchOptions(duration_seconds=0.3, api_key="server-secret"),
        )

    assert outcome.exit_code == 0
    assert set(authorizations) == {expected_authorization}
    started = of_type(read_ledger(tmp_path), WATCH_SESSION)[0]
    warnings = [warning for warning in started["warnings"] if "not sent" in warning]
    assert len(warnings) == expected_warnings
    assert "server-secret" not in json.dumps(started)
    assert "metrics-secret" not in json.dumps(started)


def test_same_origin_default_port_preserves_bearer_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorizations: list[str | None] = []

    def open_metrics(request: Request, **_kwargs: Any) -> addinfourl:
        authorizations.append(request.get_header("Authorization"))
        return addinfourl(
            BytesIO(exposition().encode()), Message(), request.full_url, 200
        )

    monkeypatch.setattr(vllm_scraper._OPENER, "open", open_metrics)
    outcome = _watch(
        tmp_path,
        watch_config(
            "https://metrics.example/v1",
            server={
                "base_url": "https://metrics.example/v1",
                "metrics_url": "https://metrics.example:443/metrics",
            },
        ),
        options=WatchOptions(duration_seconds=0.3, api_key="server-secret"),
    )
    assert outcome.exit_code == 0
    assert authorizations and set(authorizations) == {"Bearer server-secret"}
    started = of_type(read_ledger(tmp_path), WATCH_SESSION)[0]
    assert not any("not sent" in warning for warning in started["warnings"])


def test_session_urls_are_redacted_in_the_ledger_and_export(tmp_path: Path) -> None:
    observer = _Observer()
    with serve_metrics(FakeMetrics()) as base_url:
        configured_base_url = (
            base_url.replace("://", "://base-user:base-password@")
            + "/v1?key=base-token"
        )
        metrics_url = base_url + "/metrics?key=metrics-token"
        config = resolve_watch_config(
            watch_config(
                base_url,
                server={"base_url": configured_base_url, "metrics_url": metrics_url},
            )
        )
        watcher = Watcher(
            config,
            tmp_path,
            options=WatchOptions(duration_seconds=0.3),
            observer=observer,
        )
        assert watcher.config.base_url == configured_base_url
        assert watcher.config.metrics_url == watcher.scraper.url == metrics_url
        outcome = asyncio.run(watcher.run(asyncio.Event()))
    assert outcome.exit_code == 0
    sessions = of_type(read_ledger(tmp_path), WATCH_SESSION)
    exported = [r for r in observer.records if r["event_type"] == WATCH_SESSION]
    assert [s["phase"] for s in sessions] == ["started", "ended"]
    assert sessions == exported
    for session in sessions:
        assert session["config"] == config.resolved()
        assert session["config_digest"] == config.digest()
    recorded = json.dumps(sessions)
    for secret in ("base-user", "base-password", "base-token", "metrics-token"):
        assert secret not in recorded


def test_no_successful_scrape_exits_one(tmp_path: Path) -> None:
    with serve_metrics(FakeMetrics()) as base_url:
        pass  # the port is closed again
    outcome = _watch(
        tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=0.6)
    )
    assert outcome.exit_code == 1
    assert outcome.unsound == ["no_successful_scrape"]
    report = _report(tmp_path)
    assert report["verdict"]["exit_code"] == 1
    assert report["metrics"]["scrapes_ok"] == 0
    ended = of_type(read_ledger(tmp_path), WATCH_SESSION)[-1]
    assert ended["unsound"] == ["no_successful_scrape"]


def test_every_incident_write_failing_exits_one(tmp_path: Path) -> None:
    metrics = FakeMetrics()
    metrics.waiting = 20
    payload = watch_config(
        "", store={"max_total_bytes": 1024, "max_incident_bytes": 1024}
    )
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        outcome = _watch(tmp_path, payload, options=WatchOptions(duration_seconds=1.5))
    assert outcome.exit_code == 1
    assert outcome.unsound == ["incident_writes_failing"]
    assert outcome.incidents and not any(i["persisted"] for i in outcome.incidents)
    report = _report(tmp_path)
    assert report["findings"][0]["message"].startswith(
        "the bundle could not be written: the store cannot hold"
    )


def test_a_failing_ledger_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_write(fd: int, data: bytes | memoryview) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(telemetry_sink.os, "write", failing_write)
    with serve_metrics(FakeMetrics()) as base_url:
        outcome = _watch(
            tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=0.8)
        )
    monkeypatch.undo()
    assert outcome.exit_code == 1
    assert outcome.unsound == ["ledger_failing"]
    assert _report(tmp_path)["payload"]["unsound"] == ["ledger_failing"]


def test_an_unwritable_report_exits_one(tmp_path: Path) -> None:
    (tmp_path / "report.json").mkdir()
    with serve_metrics(FakeMetrics()) as base_url:
        outcome = _watch(
            tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=0.5)
        )
    assert (outcome.exit_code, outcome.report_path) == (1, None)


def test_the_stop_event_seals_open_incidents_as_interrupted(tmp_path: Path) -> None:
    """Stopped 1.2 s in, the watch at a load of 20 sometimes had not yet
    opened the incident; it now stops once the incident opens. The 30 s
    duration only bounds a failure: the post-window would end at 20 s and
    seal the incident completed."""
    metrics = FakeMetrics()
    metrics.waiting = 20
    payload = watch_config("", incident={"pre_seconds": 5, "post_seconds": 20})
    opened = _Observer(
        stop_when=lambda r: r["event_type"] == INCIDENT_EVENT and r["event"] == "opened"
    )
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        outcome = _watch(
            tmp_path,
            payload,
            options=WatchOptions(duration_seconds=30.0),
            observer=opened,
        )
    assert outcome.exit_code == 3
    (incident,) = of_type(read_ledger(tmp_path), INCIDENT)
    with open_incident_bundle(tmp_path / incident["bundle"]) as view:
        assert view.manifest.status == "interrupted"


def test_shutdown_keeps_a_full_backlog_and_every_open_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident_count = 10  # configurable admission can exceed the eight store queue slots
    gate, entered = threading.Event(), threading.Event()
    accepted: list[bool] = []
    backlog: list[int] = []
    observer = _Observer()
    stop = asyncio.Event()
    observer.stop = stop.set
    with serve_metrics(FakeMetrics()) as base_url:
        watcher = Watcher(
            resolve_watch_config(
                watch_config(
                    base_url,
                    incident={
                        "pre_seconds": 5,
                        "post_seconds": 20,
                        "max_open_incidents": incident_count,
                        "max_incidents_per_hour": incident_count,
                    },
                )
            ),
            tmp_path,
            options=WatchOptions(duration_seconds=30.0),
            observer=observer,
        )

        def hold_writer() -> None:
            entered.set()
            gate.wait()

        def stop_with_open_incidents(record: Mapping[str, Any]) -> bool:
            if record["event_type"] != WATCH_HEALTH:
                return False
            accepted.append(watcher._store_worker.submit(hold_writer))
            accepted.append(entered.wait(5.0))
            accepted.extend(
                watcher._store_worker.submit(partial(backlog.append, index))
                for index in range(8)
            )
            for _ in range(incident_count):
                watcher.incidents.on_test(
                    watcher.clock.mono_ns(), requested_wall_ns=None
                )
            return True

        close_incidents = watcher.incidents.close

        def seal_then_release(at_mono: int) -> None:
            try:
                close_incidents(at_mono)
            finally:
                gate.set()

        observer.stop_when = stop_with_open_incidents
        monkeypatch.setattr(watcher.incidents, "close", seal_then_release)
        try:
            outcome = asyncio.run(watcher.run(stop))
        finally:
            gate.set()

    assert all(accepted)
    assert backlog == list(range(8))
    assert outcome.exit_code == 0
    assert len(outcome.incidents) == incident_count
    assert all(incident["persisted"] for incident in outcome.incidents)
    assert watcher._store_worker.stats().rejected == 0
    for incident in outcome.incidents:
        with open_incident_bundle(tmp_path / incident["bundle"]) as view:
            assert view.manifest.status == "interrupted"


def test_a_test_trigger_file_records_an_incident_that_does_not_count(
    tmp_path: Path,
) -> None:
    with serve_metrics(FakeMetrics()) as base_url:
        threading.Timer(0.4, (tmp_path / "test-trigger").touch).start()
        outcome = _watch(
            tmp_path,
            watch_config(base_url),
            options=WatchOptions(duration_seconds=1.5, test_trigger_file=True),
        )
    assert outcome.exit_code == 0
    (incident,) = of_type(read_ledger(tmp_path), INCIDENT)
    assert incident["trigger"]["kind"] == "test"
    assert incident["trigger"]["requested_at_ns"] is not None
    assert not (tmp_path / "test-trigger").exists()


def test_a_periodic_test_trigger(tmp_path: Path) -> None:
    with serve_metrics(FakeMetrics()) as base_url:
        outcome = _watch(
            tmp_path,
            watch_config(base_url),
            options=WatchOptions(duration_seconds=2.0, test_trigger_every=0.7),
        )
    assert outcome.exit_code == 0
    incidents = of_type(read_ledger(tmp_path), INCIDENT)
    assert len(incidents) >= 2
    assert {i["trigger"]["kind"] for i in incidents} == {"test"}


def test_an_exporter_restart_is_a_health_incident(tmp_path: Path) -> None:
    metrics = FakeMetrics()

    def restart() -> None:
        metrics.start += 100

    with serve_metrics(metrics) as base_url:
        threading.Timer(0.5, restart).start()
        outcome = _watch(
            tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=1.5)
        )
    assert outcome.exit_code == 0
    (incident,) = of_type(read_ledger(tmp_path), INCIDENT)
    assert incident["trigger"]["trigger_id"] == "exporter_restart"
    assert incident["capture"]["status"] == "health_only"


def test_a_frozen_exporter_is_counted_and_recorded(tmp_path: Path) -> None:
    metrics = FakeMetrics()
    metrics.advance = False
    payload = watch_config("")
    payload["triggers"].append(
        {
            "id": "frozen",
            "kind": "health",
            "window_seconds": 0.3,
            "hold_seconds": 0.3,
            "frozen_exporter": {"ticks": 3},
        }
    )
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        outcome = _watch(tmp_path, payload, options=WatchOptions(duration_seconds=1.5))
    assert outcome.exit_code == 0  # health incidents do not count
    incidents = of_type(read_ledger(tmp_path), INCIDENT)
    assert [i["trigger"]["trigger_id"] for i in incidents] == ["frozen"]
    assert incidents[0]["loss"]["scrape_frozen_ticks"] is not None
    assert _report(tmp_path)["payload"]["stats"]["frozen_ticks_total"] > 0


def test_a_slow_scrape_skips_ticks_and_counts_them(tmp_path: Path) -> None:
    metrics = FakeMetrics()
    metrics.delay = 0.25  # the tick is 0.1 s
    with serve_metrics(metrics) as base_url:
        threading.Timer(0.3, (tmp_path / "test-trigger").touch).start()
        outcome = _watch(
            tmp_path,
            watch_config(base_url, scrape_timeout_seconds=1.0),
            options=WatchOptions(duration_seconds=1.5, test_trigger_file=True),
        )
    assert outcome.exit_code == 0
    stats = _report(tmp_path)["payload"]["stats"]
    # About 1.5 / 0.25 scrapes ran, never two at once; the rest were skipped.
    assert 3 <= metrics.scrapes <= 8
    assert stats["ticks_missed_total"] >= 5
    assert stats["loop_lag_seconds_max"] >= 0.0
    (incident,) = of_type(read_ledger(tmp_path), INCIDENT)
    assert incident["loss"]["scrape_ticks_missed"] > 0


def test_several_engines_with_none_named_cannot_be_judged(tmp_path: Path) -> None:
    metrics = FakeMetrics()
    metrics.engines = 2
    metrics.waiting = 20
    with serve_metrics(metrics) as base_url:
        outcome = _watch(
            tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=1.0)
        )
    assert outcome.exit_code == 1
    assert outcome.unsound == ["engine_required"]
    assert of_type(read_ledger(tmp_path), INCIDENT) == []


def test_a_named_engine_is_judged_on_a_server_with_several(tmp_path: Path) -> None:
    metrics = FakeMetrics()
    metrics.engines = 2
    metrics.waiting = 20
    payload = watch_config("")
    with serve_metrics(metrics) as base_url:
        payload["server"] = {"base_url": base_url, "engine": "1"}
        outcome = _watch(tmp_path, payload, options=WatchOptions(duration_seconds=2.0))
    assert outcome.exit_code == 3
    assert outcome.unsound == []


def test_retention_prunes_old_bundles_and_says_so(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path)
    old_id = store.new_incident_id(1_000_000_000_000_000_000)
    writer = store.new_bundle(old_id, 4096)
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(b"{}\n")
    writer.publish(
        status="completed", complete=True, sealed_at_ns=1_000_000_000_000_000_000
    )
    store.close()  # one process owns a store root at a time
    with serve_metrics(FakeMetrics()) as base_url:
        outcome = _watch(
            tmp_path, watch_config(base_url), options=WatchOptions(duration_seconds=0.6)
        )
    assert outcome.exit_code == 0
    (pruned,) = of_type(read_ledger(tmp_path), INCIDENT_PRUNED)
    assert (pruned["incident_id"], pruned["reason"]) == (old_id, "max_age_hours")
    assert pruned["bytes"] > 0
    stats = _report(tmp_path)["payload"]["stats"]
    assert stats["pruned_total"] == {"max_age_hours": 1}
    assert stats["pruned_bytes_total"] == {"max_age_hours": pruned["bytes"]}
    assert stats["retention_incidents"] == 0
    assert not (tmp_path / "incidents" / old_id).exists()


def test_one_watcher_owns_a_root_and_a_second_is_refused(tmp_path: Path) -> None:
    """Two watchers on one root would interleave their ledgers and delete
    each other's bundles; the second is refused before it writes."""
    config = resolve_watch_config(watch_config("http://127.0.0.1:9"))
    first = Watcher(config, tmp_path)
    before = sorted(tmp_path.rglob("*"))
    with pytest.raises(InferUsageError, match="another watcher"):
        Watcher(config, tmp_path)
    assert sorted(tmp_path.rglob("*")) == before
    first.close()
    Watcher(config, tmp_path).close()  # the root is free once the first ends
    assert (tmp_path / LOCK_FILENAME).exists()


def test_a_store_another_process_owns_is_refused(tmp_path: Path) -> None:
    other = IncidentStore(tmp_path)
    config = resolve_watch_config(watch_config("http://127.0.0.1:9"))
    with pytest.raises(InferUsageError, match="incident store"):
        Watcher(config, tmp_path)
    other.close()
    Watcher(config, tmp_path).close()  # the refusal let the root go


def test_a_finished_watch_lets_the_next_one_own_the_root(tmp_path: Path) -> None:
    with serve_metrics(FakeMetrics()) as base_url:
        payload = watch_config(base_url)
        first = _watch(tmp_path, payload, options=WatchOptions(duration_seconds=0.3))
        second = _watch(tmp_path, payload, options=WatchOptions(duration_seconds=0.3))
    assert (first.exit_code, second.exit_code) == (0, 0)


def test_triggers_allow_their_scrapes_the_scrape_timeout(tmp_path: Path) -> None:
    """A tick runs when its scrape returns, so the newest scrape can be a
    tick plus the scrape timeout old: windows and health tails allow it."""
    payload = watch_config("http://127.0.0.1:9", scrape_timeout_seconds=0.25)
    watcher = Watcher(resolve_watch_config(payload), tmp_path)
    try:
        assert watcher.engine.scrape_timeout_seconds == 0.25
    finally:
        watcher.close()


def test_the_parsed_tail_holds_what_every_health_trigger_reads(
    tmp_path: Path,
) -> None:
    """A failed-scrape share over 200 scrapes with a 10 s window needs 200
    parsed scrapes, not the window's 10: with fewer it is never judged."""
    trigger = {
        "id": "share",
        "kind": "health",
        "window_seconds": 10,
        "hold_seconds": 10,
        "scrape_failure_share": {"scrapes": 200},
    }
    payload = watch_config(
        "http://127.0.0.1:9",
        tick_seconds=1,
        history={"seconds": 200},  # the most a 200-scrape tail may have
        triggers=[trigger],
    )
    watcher = Watcher(resolve_watch_config(payload), tmp_path)
    try:
        record = scrape(exposition(gauges={"vllm:num_requests_waiting": 0}), 0)
        for second in range(250):
            mono = second * 1_000_000_000
            watcher.history.add(Stamped(mono, mono + 1, mono), record)
        assert len(watcher.history.parsed()) >= 200
    finally:
        watcher.close()


def test_bundles_removed_to_make_room_for_a_seal_are_recorded(tmp_path: Path) -> None:
    """Three recent bundles of 100 KiB fill a 400 KiB store as far as
    retention goes; the watch's own seal then needs the oldest gone."""
    store = IncidentStore(tmp_path)
    kept = []
    for index in range(3):
        incident_id = store.new_incident_id()
        writer = store.new_bundle(incident_id, 200 * 1024)
        assert writer is not None
        with writer.file("incident.jsonl") as out:
            out.write(b"x" * 100 * 1024)
        sealed = time.time_ns() - (3 - index) * 1_000_000_000
        writer.publish(status="completed", complete=True, sealed_at_ns=sealed)
        kept.append(incident_id)
    store.close()
    metrics = FakeMetrics()
    metrics.waiting = 20
    limits = {"max_total_bytes": 400 * 1024, "max_incident_bytes": 380 * 1024}
    with serve_metrics(metrics) as base_url:
        outcome = _watch(
            tmp_path,
            watch_config(base_url, store=limits),
            options=WatchOptions(duration_seconds=1.5),
        )
    assert outcome.exit_code == 3
    records = read_ledger(tmp_path)
    pruned = of_type(records, INCIDENT_PRUNED)
    assert pruned and pruned[0]["incident_id"] == kept[0]
    assert {record["reason"] for record in pruned} == {"max_total_bytes"}
    (incident,) = of_type(records, INCIDENT)
    assert incident["bundle"] is not None
    assert records.index(pruned[-1]) < records.index(incident)
    stats = _report(tmp_path)["payload"]["stats"]
    assert sum(stats["pruned_total"].values()) == len(pruned)


def test_a_scrape_larger_than_the_whole_history_is_counted_oversized(
    tmp_path: Path,
) -> None:
    """A history of 400 bytes holds no scrape: each is refused, counted as
    oversized rather than ok, and the watch could judge nothing."""
    with serve_metrics(FakeMetrics()) as base_url:
        payload = watch_config(base_url, history={"seconds": 60, "bytes": 400})
        outcome = _watch(tmp_path, payload, options=WatchOptions(duration_seconds=0.5))
    assert outcome.exit_code == 1
    assert "no_successful_scrape" in outcome.unsound
    metrics = _report(tmp_path)["metrics"]
    assert metrics["scrapes_ok"] == 0 and metrics["scrapes_oversized"] > 0


def test_a_scrape_cut_short_by_the_stop_is_not_a_failed_scrape(
    tmp_path: Path,
) -> None:
    """The stop cuts a scrape in flight short; recording it as failed (an
    abandoned scrape) was never told apart, though no scrape failed."""
    metrics = FakeMetrics()
    metrics.dribble = 0.2
    payload = watch_config("", scrape_timeout_seconds=5.0)
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        outcome = _watch(
            tmp_path,
            payload,
            options=WatchOptions(shutdown_deadline_seconds=10.0),
            stop_after=0.8,
        )
    assert outcome.unsound == ["no_successful_scrape"]
    health = of_type(read_ledger(tmp_path), WATCH_HEALTH)
    assert "error" not in [record["scrape"]["status"] for record in health]
    assert _report(tmp_path)["metrics"]["scrapes_failed"] == 0


@pytest.mark.parametrize("ends_by", ["stop", "duration"])
def test_a_trickling_scrape_is_given_up_and_the_watch_still_ends_on_time(
    tmp_path: Path, ends_by: str
) -> None:
    """A /metrics answering a byte every 0.2 s never times out per read: it
    held the watch for minutes, deaf to SIGTERM and to --duration. Now each
    scrape has its timeout in all, the fetch it gives up on is the only one
    in flight, and the ticks meanwhile still run."""
    metrics = FakeMetrics()
    metrics.dribble = 0.2
    payload = watch_config("", scrape_timeout_seconds=0.5)
    options = WatchOptions(
        duration_seconds=1.2 if ends_by == "duration" else None,
        shutdown_deadline_seconds=10.0,
    )
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        started = time.monotonic()
        outcome = _watch(
            tmp_path,
            payload,
            options=options,
            stop_after=1.2 if ends_by == "stop" else None,
        )
        elapsed = time.monotonic() - started
    assert elapsed < 4.0
    assert outcome.unsound == ["no_successful_scrape"]
    assert metrics.max_in_flight == 1
    records = read_ledger(tmp_path)
    health = of_type(records, WATCH_HEALTH)
    statuses = [record["scrape"]["status"] for record in health]
    assert "error" in statuses and "skipped" in statuses
    given_up, *_rest = [r["scrape"] for r in health if r["scrape"]["status"] == "error"]
    assert given_up["error"] == (
        "abandoned: no whole response within the 0.5 s scrape timeout"
    )
    stats = _report(tmp_path)["payload"]["stats"]
    assert stats["ticks_missed_total"] >= 1


def test_an_abandoned_scrape_ends_on_the_watcher_s_clock(tmp_path: Path) -> None:
    """The watcher stamps a scrape's start on its own clock. The one it gave
    up on ended on the real clock, so with the watcher's clock behind it the
    abandoned record lasted 11 days instead of its 0.3 s timeout."""
    metrics = FakeMetrics()
    metrics.dribble = 0.2
    behind = 10**15  # about 11.6 days
    clock = WatchClock(wall_ns=lambda: time.time_ns() - behind)
    payload = watch_config("", scrape_timeout_seconds=0.3)
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        watcher = Watcher(resolve_watch_config(payload), tmp_path, clock=clock)

        async def scrape() -> Any:
            return await watcher._scrape(clock.wall_ns(), asyncio.Event())

        try:
            record = asyncio.run(scrape())
        finally:
            watcher.close()
    assert record is not None and record.status == "error"
    assert record.completed_at_ns is not None
    took = record.completed_at_ns - record.observed_at_ns
    assert 0.25 * 1e9 <= took < 5 * 1e9


def test_an_incident_whose_bundle_cannot_be_created_is_reported(
    tmp_path: Path,
) -> None:
    """incidents/ made read-only mid-watch, a real EACCES: the watch exited 3
    with an empty report and no incident record."""
    import os
    import stat

    metrics = FakeMetrics()
    metrics.waiting = 20
    incidents = tmp_path / "incidents"

    def lock_incidents() -> None:
        os.chmod(incidents, stat.S_IRUSR | stat.S_IXUSR)

    try:
        with serve_metrics(metrics) as base_url:
            threading.Timer(0.3, lock_incidents).start()
            outcome = _watch(
                tmp_path,
                watch_config(base_url),
                options=WatchOptions(duration_seconds=2.0),
            )
    finally:
        os.chmod(incidents, stat.S_IRWXU)
    (incident,) = of_type(read_ledger(tmp_path), INCIDENT)
    assert incident["bundle"] is None
    assert incident["bundle_error"].startswith("PermissionError")
    # Every incident write failed: the watch could not keep what it saw.
    assert outcome.exit_code == 1
    assert outcome.unsound == ["incident_writes_failing"]
    report = _report(tmp_path)
    (finding,) = report["findings"]
    assert finding["message"].startswith("the bundle could not be written: Permission")


def test_history_evictions_are_counted_by_cause(tmp_path: Path) -> None:
    """stormlog_watch_history_evictions_total never moved, while the ring
    evicted by age every tick; and a health record's evictions lacked the
    causes that had not happened yet."""
    with serve_metrics(FakeMetrics()) as base_url:
        payload = watch_config(
            base_url,
            history={"seconds": 0.5},
            incident={"pre_seconds": 0.2, "post_seconds": 0.3},
            default_health_triggers=False,
        )
        _watch(tmp_path, payload, options=WatchOptions(duration_seconds=1.5))
    stats = _report(tmp_path)["payload"]["stats"]
    assert stats["history_evictions_total"]["age"] > 0
    health = of_type(read_ledger(tmp_path), WATCH_HEALTH)
    assert set(health[0]["history"]["evictions"]) == {"age", "bytes", "oversized"}
    assert health[-1]["history"]["evictions"]["age"] == (
        stats["history_evictions_total"]["age"]
    )


def test_due_incidents_are_sealed_before_a_tick_s_firings_are_admitted(
    tmp_path: Path,
) -> None:
    """Firings were admitted first, so an incident whose post-window had just
    ended still held its open slot: with max_open_incidents 1, a firing at
    the first tick after it was turned away."""
    metrics = FakeMetrics()
    metrics.waiting = 20
    calls: list[tuple[str, int]] = []
    with serve_metrics(metrics) as base_url:
        watcher = Watcher(
            resolve_watch_config(watch_config(base_url)),
            tmp_path,
            options=WatchOptions(duration_seconds=1.5),
        )
        manager = watcher.incidents
        seal, admit = manager.on_tick, manager.on_fired

        def on_tick(at_mono: int) -> list[str]:
            calls.append(("seal", at_mono))
            return seal(at_mono)

        def on_fired(result: Any) -> str | None:
            calls.append(("fire", result.at_ns))
            return admit(result)

        manager.on_tick = on_tick  # type: ignore[method-assign]
        manager.on_fired = on_fired  # type: ignore[method-assign]

        async def main() -> WatchOutcome:
            return await watcher.run(asyncio.Event())

        asyncio.run(main())
    fired = [at for kind, at in calls if kind == "fire"]
    assert fired
    for at in fired:
        assert calls.index(("seal", at)) < calls.index(("fire", at))


def test_a_short_shutdown_deadline_still_drains_a_quiet_watch(tmp_path: Path) -> None:
    """The store's writer got the deadline less 5 s: with 5 s or less it got
    nothing, and a quiet watch ended unsound."""
    with serve_metrics(FakeMetrics()) as base_url:
        outcome = _watch(
            tmp_path,
            watch_config(base_url),
            options=WatchOptions(duration_seconds=0.5, shutdown_deadline_seconds=2.0),
        )
    assert (outcome.exit_code, outcome.unsound) == (0, [])


def test_a_second_signal_cuts_the_shutdown_short_and_keeps_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second Ctrl+C was ignored: a stalled store writer held the watch
    for the whole 30 s deadline. After a hurry it waits FAST_EXIT_SECONDS
    at most; the root stays locked, since the writer left behind may still
    write there."""
    from stormlog.infer.watch import watcher as watcher_module

    monkeypatch.setattr(watcher_module, "FAST_EXIT_SECONDS", 0.3)
    with serve_metrics(FakeMetrics()) as base_url:
        watcher = Watcher(
            resolve_watch_config(watch_config(base_url)),
            tmp_path,
            options=WatchOptions(shutdown_deadline_seconds=30.0),
        )
        stall = threading.Event()

        def stalled_close(timeout: float) -> bool:
            stall.wait(timeout)
            return False

        watcher._store_worker.close = stalled_close  # type: ignore[method-assign,assignment]

        async def main() -> WatchOutcome:
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            loop.call_later(0.5, stop.set)
            loop.call_later(0.8, watcher.hurry)
            return await watcher.run(stop)

        started = time.monotonic()
        outcome = asyncio.run(main())
        elapsed = time.monotonic() - started
        stall.set()
    assert elapsed < 3.0
    assert "store_writer_timeout" in outcome.unsound
    with pytest.raises(InferUsageError, match="another watcher"):
        Watcher(resolve_watch_config(watch_config(base_url)), tmp_path)
    watcher.close()


def _stall_store(watcher: Watcher) -> threading.Event:
    stall = threading.Event()

    def stalled_close(timeout: float) -> bool:
        stall.wait(timeout)
        return False

    watcher._store_worker.close = stalled_close  # type: ignore[method-assign,assignment]
    return stall


def _slow_ledger(watcher: Watcher, seconds: float) -> None:
    real_close = watcher.ledger.close

    def slow_close(timeout: float) -> bool:
        time.sleep(min(seconds, timeout))
        return seconds <= timeout and real_close(timeout)

    watcher.ledger.close = slow_close  # type: ignore[method-assign]


@pytest.mark.parametrize("hurried", [False, True])
def test_a_stalled_store_leaves_the_ledger_its_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hurried: bool
) -> None:
    """The store's writer gets two thirds of the shutdown, and of a hurry,
    and the ledger the rest. After a hurry a stalled store took all of it,
    so the ledger, closing in a few hundred ms, was cut off and reported
    as ledger_close_timeout; the even split was never tested."""
    from stormlog.infer.watch import watcher as watcher_module

    monkeypatch.setattr(watcher_module, "FAST_EXIT_SECONDS", 1.5)
    with serve_metrics(FakeMetrics()) as base_url:
        watcher = Watcher(
            resolve_watch_config(watch_config(base_url)),
            tmp_path,
            # 1.5 s for the ledger without a hurry, 0.5 s after one.
            options=WatchOptions(shutdown_deadline_seconds=6.0 if not hurried else 30),
        )
        stall = _stall_store(watcher)
        _slow_ledger(watcher, 0.3 if hurried else 0.75)

        async def main() -> WatchOutcome:
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            loop.call_later(0.3, stop.set)
            if hurried:
                loop.call_later(0.5, watcher.hurry)
            return await watcher.run(stop)

        outcome = asyncio.run(main())
        stall.set()
    assert outcome.unsound == ["store_writer_timeout"]
    watcher.close()


def test_a_store_operation_that_raises_is_in_the_report(tmp_path: Path) -> None:
    """A retention pass that raised on the store's worker was swallowed
    there: with store.max_age_hours at 1e300, no retention ever ran."""
    with serve_metrics(FakeMetrics()) as base_url:
        watcher = Watcher(
            resolve_watch_config(watch_config(base_url)),
            tmp_path,
            options=WatchOptions(duration_seconds=0.5),
        )

        def broken(**_kwargs: Any) -> list[Any]:
            raise OverflowError("cannot convert float infinity to integer")

        watcher.store.prune = broken  # type: ignore[method-assign]
        outcome = asyncio.run(watcher.run(asyncio.Event()))
    assert outcome.exit_code == 0
    report = _report(tmp_path)
    assert report["metrics"]["store_operations_failed"] >= 1
    assert report["payload"]["store_last_error"] == (
        "OverflowError: cannot convert float infinity to integer"
    )
    assert _quiet_report_metrics(tmp_path)["store_operations_failed"] == 0


def _quiet_report_metrics(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "quiet"
    with serve_metrics(FakeMetrics()) as base_url:
        _watch(root, watch_config(base_url), options=WatchOptions(duration_seconds=0.5))
    metrics: dict[str, Any] = _report(root)["metrics"]
    assert _report(root)["payload"]["store_last_error"] is None
    return metrics


def test_a_ledger_left_behind_keeps_the_root(tmp_path: Path) -> None:
    """Only the store's writer kept the root locked: a ledger writer left
    behind still appending under it let a second watcher take the root."""
    with serve_metrics(FakeMetrics()) as base_url:
        watcher = Watcher(
            resolve_watch_config(watch_config(base_url)),
            tmp_path,
            options=WatchOptions(duration_seconds=0.5),
        )
        real_close = watcher.ledger.close

        def left_behind(timeout: float) -> bool:
            real_close(timeout)
            return False  # as if its writer were still running

        watcher.ledger.close = left_behind  # type: ignore[method-assign]
        outcome = asyncio.run(watcher.run(asyncio.Event()))
    assert outcome.unsound == ["ledger_close_timeout"]
    with pytest.raises(InferUsageError, match="another watcher"):
        Watcher(resolve_watch_config(watch_config(base_url)), tmp_path)
    watcher.close()


@pytest.mark.parametrize("exporting", [False, True])
def test_export_failures_are_null_without_an_exporter(
    tmp_path: Path, exporting: bool
) -> None:
    """A source that is not running is null, never 0: export_failures was 0
    with no observer."""
    metrics = FakeMetrics()
    metrics.waiting = 20
    with serve_metrics(metrics) as base_url:
        _watch(
            tmp_path,
            watch_config(base_url),
            options=WatchOptions(duration_seconds=1.5),
            observer=_Observer() if exporting else None,
        )
    incident = of_type(read_ledger(tmp_path), INCIDENT)[0]
    assert incident["loss"]["export_failures"] == (0 if exporting else None)


def test_a_trigger_resolved_before_the_seal_is_recorded_so(tmp_path: Path) -> None:
    metrics = FakeMetrics()
    metrics.waiting = 20
    payload = watch_config("", incident={"pre_seconds": 5, "post_seconds": 2.5})
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url

        def quiet() -> None:
            metrics.waiting = 0

        threading.Timer(1.0, quiet).start()
        _watch(tmp_path, payload, options=WatchOptions(duration_seconds=4.5))
    records = read_ledger(tmp_path)
    (incident,) = of_type(records, INCIDENT)
    (resolved,) = [
        r for r in of_type(records, TRIGGER_STATE) if r["event"] == "resolved"
    ]
    assert incident["trigger"]["resolved_at_ns"] == resolved["timestamp_ns"]


def test_a_store_budget_larger_than_the_disk_is_warned_of(tmp_path: Path) -> None:
    payload = watch_config(
        "http://127.0.0.1:9",
        store={"max_total_bytes": 1 << 62, "max_incident_bytes": 1 << 20},
    )
    watcher = Watcher(resolve_watch_config(payload), tmp_path)
    try:
        (warning,) = watcher.warnings
        assert "less than store.max_total_bytes" in warning
    finally:
        watcher.close()


def test_a_ledger_that_misses_the_shutdown_deadline_is_unsound(
    tmp_path: Path,
) -> None:
    """Treating ledger_close_timeout as sound survived every test."""
    with serve_metrics(FakeMetrics()) as base_url:
        watcher = Watcher(
            resolve_watch_config(watch_config(base_url)),
            tmp_path,
            options=WatchOptions(duration_seconds=0.5),
        )
        real_close = watcher.ledger.close

        def late_close(timeout: float) -> bool:
            real_close(timeout)
            return False  # as if the sink had not finished in time

        watcher.ledger.close = late_close  # type: ignore[method-assign]

        async def main() -> WatchOutcome:
            return await watcher.run(asyncio.Event())

        outcome = asyncio.run(main())
    assert outcome.exit_code == 1
    assert outcome.unsound == ["ledger_close_timeout"]
