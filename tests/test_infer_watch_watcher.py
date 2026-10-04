"""The watch loop end to end, against a fake vLLM ``/metrics`` endpoint."""

from __future__ import annotations

import asyncio
import errno
import json
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from stormlog import telemetry_sink
from stormlog.infer.errors import InferUsageError
from stormlog.infer.watch.config import resolve_watch_config
from stormlog.infer.watch.history import Stamped
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
    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []

    def observe(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)


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
        if stop_after is not None:
            asyncio.get_running_loop().call_later(stop_after, stop.set)
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
    """One watch of a server whose queue stays over the trigger's threshold."""
    root = tmp_path_factory.mktemp("violation")
    metrics = FakeMetrics()
    metrics.waiting = 20
    observer = _Observer()
    with serve_metrics(metrics) as base_url:
        outcome = _watch(
            root,
            watch_config(base_url),
            options=WatchOptions(duration_seconds=2.0, ready_file=root / "ready"),
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
    assert report["findings"][0]["message"] == "the bundle could not be written"


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
    metrics = FakeMetrics()
    metrics.waiting = 20
    payload = watch_config("", incident={"pre_seconds": 5, "post_seconds": 30})
    with serve_metrics(metrics) as base_url:
        payload["server"]["base_url"] = base_url
        outcome = _watch(
            tmp_path,
            payload,
            options=WatchOptions(duration_seconds=None),
            stop_after=1.2,
        )
    assert outcome.exit_code == 3
    (incident,) = of_type(read_ledger(tmp_path), INCIDENT)
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
    assert stats["pruned_total"] == 1
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
    payload = watch_config("http://127.0.0.1:9", tick_seconds=1, triggers=[trigger])
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
    assert stats["pruned_total"] == len(pruned)
