"""Incidents: opened by firings, joined, bounded, and sealed into bundles."""

from __future__ import annotations

import gc
import json
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.correlation_events import load_inference_artifact
from stormlog.infer.watch import incidents as incidents_module
from stormlog.infer.watch import store as store_module
from stormlog.infer.watch.config import IncidentLimits
from stormlog.infer.watch.disk import StoreLimits
from stormlog.infer.watch.evaluate import TickResult, TriggerSpec
from stormlog.infer.watch.history import ScrapeHistory, Stamped
from stormlog.infer.watch.incidents import Identity, IncidentManager, WatchClock
from stormlog.infer.watch.predicates import Evaluation, GaugeAtLeast
from stormlog.infer.watch.records import (
    INCIDENT,
    INCIDENT_EVENT,
    LOSS_KEYS,
    validate_record,
)
from stormlog.infer.watch.stats import WatchStats, counter_value
from stormlog.infer.watch.store import (
    STATUS_COMPLETED,
    STATUS_INTERRUPTED,
    IncidentStore,
    PrunedBundle,
    open_incident_bundle,
)
from stormlog.infer.watch.triggers import Sustain, Transition
from tests.vllm_scrape_helpers import exposition, scrape

S = 1_000_000_000
T0 = 1_790_000_000_000_000_000
BUSY = exposition(gauges={"vllm:num_requests_waiting": 20})


class Harness:
    """An ``IncidentManager`` on a hand-driven clock, writing inline."""

    def __init__(
        self,
        root: Path,
        *,
        limits: IncidentLimits | None = None,
        store_limits: StoreLimits | None = None,
        accept: bool = True,
    ) -> None:
        self.now = 0
        self.records: list[dict[str, Any]] = []
        self.loss: dict[str, int] = {"scrapes_failed": 0}
        self.accept = accept
        self.stats = WatchStats()
        self.history = ScrapeHistory(
            max_seconds=600, max_bytes=32 * 1024 * 1024, parsed_count=64
        )
        self.store = IncidentStore(root, store_limits)
        self.pruned: list[PrunedBundle] = []
        self.manager = IncidentManager(
            store=self.store,
            history=self.history,
            limits=limits or IncidentLimits(),
            identity=Identity(
                session_id="session",
                run_id="run-1",
                owner="host:1:0",
                host="host",
                boot_id="boot",
            ),
            clock=WatchClock(
                mono_ns=lambda: self.now,
                wall_ns=lambda: T0 + self.now,
                mono_origin_ns=0,
                wall_origin_ns=T0,
            ),
            stats=self.stats,
            emit=self.records.append,
            submit=self._submit,
            post=lambda task: task(),
            loss=lambda: dict(self.loss),
            tick_seconds=1.0,
            on_pruned=self._pruned,
        )

    def _pruned(self, bundles: list[PrunedBundle]) -> None:
        self.pruned.extend(bundles)
        self.records.append({"event_type": "pruned"})

    def _submit(self, task: Callable[[], None]) -> bool:
        if self.accept:
            task()
        return self.accept

    def scrapes(
        self, start_s: int, end_s: int, *, failed: frozenset[int] = frozenset()
    ) -> None:
        for second in range(start_s, end_s + 1):
            text = None if second in failed else BUSY
            stamp = Stamped(second * S, second * S + 4_000_000, T0 + second * S)
            self.history.add(stamp, scrape(text, second))

    def fire(
        self,
        at_s: float,
        *,
        trigger_id: str = "queue",
        pending_since_s: float | None = None,
        kind: str = "metric",
        counts: bool = True,
    ) -> str | None:
        at = int(at_s * S)
        pending = int(
            (pending_since_s if pending_since_s is not None else at_s - 60) * S
        )
        spec = TriggerSpec(
            trigger_id=trigger_id,
            kind=kind,
            sustain=Sustain.with_defaults(window=30, hold=60, clear=None, tick=1.0),
            predicate=GaugeAtLeast(family="vllm:num_requests_waiting", threshold=8),
            counts_toward_exit=counts,
        )
        result = TickResult(
            spec=spec,
            at_ns=at,
            evaluation=Evaluation(
                "violating",
                observed=20.0,
                observed_bounds=(20.0, None),
                threshold=8.0,
                samples=30.0,
            ),
            transition=Transition(
                event="fired",
                at_ns=at,
                state="firing",
                pending_since_ns=pending,
                fired_at_ns=at,
                accumulated_ns=at - pending,
            ),
            state="firing",
        )
        return self.manager.on_fired(result)

    def tick(self, at_s: float) -> list[str]:
        self.now = int(at_s * S)
        return self.manager.on_tick(self.now)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["event_type"] == event_type]


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_a_firing_opens_an_incident_reaching_back_to_its_first_window(
    harness: Harness,
) -> None:
    incident_id = harness.fire(200, pending_since_s=140)
    assert incident_id is not None
    incident = harness.manager.open[incident_id]
    # The first violating window began W = 30 s before pending_since.
    assert incident.pre_start_mono == 110 * S
    assert incident.post_end_mono == 260 * S
    (opened,) = harness.of_type(INCIDENT_EVENT)
    validate_record(opened)
    assert (opened["event"], opened["trigger_id"], opened["rearm_basis"]) == (
        "opened",
        "queue",
        None,
    )
    assert opened["timestamp_ns"] == T0 + 200 * S


def test_the_pre_window_is_capped_at_pre_seconds(harness: Harness) -> None:
    incident_id = harness.fire(200, pending_since_s=20)
    assert incident_id is not None
    assert harness.manager.open[incident_id].pre_start_mono == 80 * S


def test_firings_within_the_post_window_join_the_incident(harness: Harness) -> None:
    first = harness.fire(200, counts=False)
    second = harness.fire(230, trigger_id="kv", counts=True)
    assert first == second and first is not None
    incident = harness.manager.open[first]
    assert [t["trigger_id"] for t in incident.joined] == ["kv"]
    assert incident.counts_toward_exit
    assert len(harness.of_type(INCIDENT_EVENT)) == 1
    assert harness.manager.firings == 2


def test_at_most_sixteen_triggers_join_one_incident(harness: Harness) -> None:
    incident_id = harness.fire(200)
    assert incident_id is not None
    for index in range(16):
        assert harness.fire(201 + index, trigger_id=f"t{index}") == incident_id
    assert harness.fire(230, trigger_id="t16") is None
    incident = harness.manager.open[incident_id]
    assert len(incident.joined) == 16
    # Its own reason, not open_limit: the incident is full, not the store.
    assert incident.suppressed == {"join_limit": 1}
    snapshot = harness.stats.health()
    assert counter_value(snapshot, "suppressed_total", ("join_limit",)) == 1


def test_a_firing_after_the_seal_opens_a_new_incident(harness: Harness) -> None:
    harness.scrapes(80, 262)
    first = harness.fire(200)
    assert harness.tick(260) == [first]
    second = harness.fire(270)
    assert second is not None and second != first
    (sealed,) = harness.of_type(INCIDENT)
    assert sealed["joined_triggers"] == []  # membership froze at the seal


def test_open_incidents_are_limited(tmp_path: Path) -> None:
    one = Harness(tmp_path / "one", limits=IncidentLimits(max_open_incidents=1))
    assert one.fire(0) is not None
    # Past the first post-window but before the tick seals it.
    assert one.fire(61) is None
    assert counter_value(one.stats.health(), "suppressed_total", ("open_limit",)) == 1

    two = Harness(tmp_path / "two", limits=IncidentLimits(max_open_incidents=2))
    first, second = two.fire(0), two.fire(61)
    assert first is not None and second is not None
    assert set(two.manager.open) == {first, second}


def test_incidents_per_hour_are_limited(harness: Harness) -> None:
    harness.manager.limits = IncidentLimits(max_incidents_per_hour=2)
    assert harness.fire(0) is not None
    harness.tick(60)
    assert harness.fire(100) is not None
    harness.tick(160)
    assert harness.fire(200) is None
    snapshot = harness.stats.health()
    assert counter_value(snapshot, "suppressed_total", ("rate_limit",)) == 1
    assert harness.fire(3601) is not None  # the first has left the hour


def test_a_sealed_incident_is_published_as_a_bundle(harness: Harness) -> None:
    harness.scrapes(80, 262)
    incident_id = harness.fire(200, pending_since_s=140)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    validate_record(record)
    assert record["bundle"] == f"incidents/{incident_id}"
    assert record["capture"]["status"] == "disabled"
    assert record["pre_window"]["start_ns"] == T0 + 110 * S
    assert record["pre_window"]["end_ns"] == T0 + 200 * S
    assert record["post_window"]["end_ns"] == T0 + 260 * S
    assert record["pre_window"]["fidelity"] == "complete"
    assert record["post_window"]["fidelity"] == "complete"
    assert record["pre_window"]["fidelity_detail"]["scrapes"]["ok"] == 91
    assert record["pre_window"]["clock_domain"].startswith("host/boot/")
    assert record["trigger"]["observed_bounds"] == [20.0, None]
    assert record["attachment_ids"] == [f"incident:{incident_id}"]
    assert harness.manager.persisted == 1
    snapshot = harness.stats.health()
    assert counter_value(snapshot, "incidents_total", ("metric", "disabled")) == 1
    windows = ("pre", "complete", "metrics")
    assert counter_value(snapshot, "incident_windows_total", windows) == 1

    with open_incident_bundle(harness.store.root / str(incident_id)) as view:
        assert view.manifest.status == STATUS_COMPLETED
        path = view.file("incident.jsonl")
        loaded = load_inference_artifact(path)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(loaded) == len(lines)
    kinds = [line["event_type"] for line in lines]
    assert kinds[0] == "infer.session" and kinds[-1] == INCIDENT
    windows_written = [
        line["window"]
        for line in lines
        if line["event_type"] == "infer.incident_window"
    ]
    assert windows_written == ["pre", "post"]
    assert kinds.count("infer.vllm_scrape") == 151  # 110 s through 260 s


@pytest.mark.parametrize(
    ("scrapes", "failed", "fidelity"),
    [
        ((201, 262), set(), "missing"),  # nothing before the firing
        ((80, 262), {150}, "partial"),  # a failed scrape inside
        ((180, 262), set(), "partial"),  # history began late
        ((80, 262), set(), "complete"),
    ],
)
def test_window_fidelity(
    harness: Harness, scrapes: tuple[int, int], failed: set[int], fidelity: str
) -> None:
    harness.scrapes(*scrapes, failed=frozenset(failed))
    harness.fire(200, pending_since_s=140)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["pre_window"]["fidelity"] == fidelity
    detail = record["pre_window"]["detail_collected"]
    assert detail == (None if fidelity == "missing" else "metrics")


def test_a_bundle_over_budget_is_a_persist_failure(tmp_path: Path) -> None:
    harness = Harness(
        tmp_path,
        store_limits=StoreLimits(max_total_bytes=1024, max_incident_bytes=1024),
    )
    harness.scrapes(80, 262)
    harness.fire(200)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["bundle"] is None
    assert (harness.manager.persisted, harness.manager.persist_failures) == (0, 1)
    assert harness.manager.sealed[0]["persisted"] is False
    bundles = [p for p in harness.store.root.iterdir() if not p.name.startswith(".")]
    assert bundles == []


def test_a_rejected_write_is_a_persist_failure(tmp_path: Path) -> None:
    harness = Harness(tmp_path, accept=False)
    harness.scrapes(80, 262)
    harness.fire(200)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["bundle"] is None
    assert harness.manager.persist_failures == 1


def test_close_seals_open_incidents_as_interrupted(harness: Harness) -> None:
    harness.scrapes(80, 230)
    incident_id = harness.fire(200)
    harness.now = 230 * S
    harness.manager.close(harness.now)
    assert harness.manager.open == {}
    (record,) = harness.of_type(INCIDENT)
    assert record["post_window"]["end_ns"] == T0 + 230 * S
    with open_incident_bundle(harness.store.root / str(incident_id)) as view:
        assert view.manifest.status == STATUS_INTERRUPTED


def test_loss_is_counted_from_the_incidents_open(harness: Harness) -> None:
    harness.loss = {"scrapes_failed": 2, "io_rejected": 1}
    harness.fire(200)
    harness.loss = {"scrapes_failed": 5, "io_rejected": 1}
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert set(record["loss"]) == set(LOSS_KEYS)
    assert record["loss"]["scrapes_failed"] == 3
    assert record["loss"]["io_rejected"] == 0
    # A source that is not running is null, never 0.
    assert record["loss"]["hook_dropped"] is None


def test_a_health_event_records_without_counting(harness: Harness) -> None:
    harness.scrapes(80, 262)
    harness.now = 200 * S
    incident_id = harness.manager.on_event("exporter_restart", "restarted", harness.now)
    assert incident_id is not None
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["trigger"]["kind"] == "health"
    assert record["capture"]["status"] == "health_only"
    assert record["counts_toward_exit"] is False
    assert harness.manager.recorded_counting == 0


def test_a_test_trigger_records_when_it_was_requested(harness: Harness) -> None:
    harness.scrapes(80, 262)
    harness.now = 200 * S
    harness.manager.on_test(harness.now, requested_wall_ns=T0 + 199 * S)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    validate_record(record)
    assert record["trigger"]["kind"] == "test"
    assert record["trigger"]["requested_at_ns"] == T0 + 199 * S
    assert record["counts_toward_exit"] is False


REAL_SCRAPE = (
    Path(__file__).parent / "fixtures" / "vllm" / "q05_c08_scrape_record_eb5ad3f.json"
)


def test_a_seal_writes_the_history_s_own_scrapes_without_parsing_them(
    tmp_path: Path,
) -> None:
    """Parsed together, the 151 real scrapes of this incident take 19 MiB;
    the seal holds the history's compressed copies and expands one at a time
    as it writes."""
    from stormlog.infer.vllm_telemetry import VllmScrapeRecord

    harness = Harness(tmp_path)
    real = VllmScrapeRecord.from_record(json.loads(REAL_SCRAPE.read_text()))
    for second in range(80, 262):
        stamp = Stamped(second * S, second * S + 4_000_000, T0 + second * S)
        harness.history.add(stamp, real)
    incident_id = harness.fire(200, pending_since_s=140)
    gc.collect()
    tracemalloc.start()
    try:
        harness.tick(261)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024
    held = harness.history.ring.compressed(110 * S, 260 * S)
    with open_incident_bundle(harness.store.root / str(incident_id)) as view:
        lines = view.file("incident.jsonl").read_bytes().splitlines()
    scrapes = [line for line in lines if b'"event_type":"infer.vllm_scrape"' in line]
    assert len(scrapes) == len(held) == 151
    assert VllmScrapeRecord.from_record(json.loads(scrapes[0])) == real


def test_bundles_removed_to_make_room_are_reported_before_the_seal(
    tmp_path: Path,
) -> None:
    # Each seal reserves about 575 KB and keeps about 320 KB of a 1 MiB
    # store: the third and fourth seals each need the oldest bundle gone.
    limits = StoreLimits(max_total_bytes=1 << 20, max_incident_bytes=1 << 20)
    harness = Harness(tmp_path, store_limits=limits)
    sealed = []
    for start in (100, 300, 500, 700):
        harness.scrapes(start - 60, start + 61)
        incident_id = harness.fire(start, pending_since_s=start - 30)
        harness.tick(start + 61)
        sealed.append(incident_id)
    assert harness.manager.persisted == 4
    assert [bundle.incident_id for bundle in harness.pruned] == sealed[:2]
    assert {bundle.reason for bundle in harness.pruned} == {"max_total_bytes"}
    kinds = [
        r["event_type"] for r in harness.records if r["event_type"] != INCIDENT_EVENT
    ]
    assert kinds == [INCIDENT, INCIDENT, "pruned", INCIDENT, "pruned", INCIDENT]


@pytest.mark.parametrize(
    "failure",
    [OSError(28, "No space left on device"), RuntimeError("store bug")],
    ids=["enospc", "unexpected"],
)
def test_a_bundle_that_cannot_be_created_still_records_its_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """new_bundle raised outside the write's handler: the store's worker
    swallowed it, and the incident was never recorded, counted or reported."""
    harness = Harness(tmp_path)
    harness.scrapes(80, 262)

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(harness.store, "new_bundle", broken)
    harness.fire(200)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    validate_record(record)
    assert record["bundle"] is None
    assert record["bundle_error"] == f"{type(failure).__name__}: {failure}"
    assert (harness.manager.persisted, harness.manager.persist_failures) == (0, 1)
    (sealed,) = harness.manager.sealed
    assert sealed["bundle_error"] == record["bundle_error"]


def test_an_unexpected_error_while_writing_a_bundle_lets_its_generation_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a budget or disk error abandoned the generation: any other left
    its reservation held and its bundle pinned against deletion for good."""
    harness = Harness(tmp_path)
    harness.scrapes(80, 262)

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("store bug")

    monkeypatch.setattr(store_module.GenerationWriter, "publish", broken)
    incident_id = harness.fire(200)
    assert incident_id is not None
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["bundle_error"] == "RuntimeError: store bug"
    assert harness.store.budget.reserved_bytes == 0
    assert not (tmp_path / "incidents" / incident_id).exists()


def test_health_incidents_never_use_up_the_counting_budget(tmp_path: Path) -> None:
    """A flapping exporter's restarts used up the hour's incidents, and a
    real violation after them was turned away."""
    harness = Harness(tmp_path, limits=IncidentLimits(max_incidents_per_hour=2))
    harness.scrapes(0, 900)
    for second in (100, 200, 300):
        harness.now = second * S
        harness.manager.on_event("exporter_restart", "restarted", harness.now)
        harness.tick(second + 61)
    queue = harness.fire(500)
    assert queue is not None
    harness.tick(561)
    kinds = [r["trigger"]["kind"] for r in harness.of_type(INCIDENT)]
    assert kinds == ["health", "health", "metric"]  # the third restart: refused
    assert (harness.manager.recorded, harness.manager.recorded_counting) == (3, 1)


def test_a_counting_firing_turned_away_is_not_counted_as_recorded(
    tmp_path: Path,
) -> None:
    """A refused counting firing made the watch exit 3 with no counting
    incident to show for it."""
    harness = Harness(tmp_path, limits=IncidentLimits(max_incidents_per_hour=1))
    harness.scrapes(0, 900)
    assert harness.fire(100, counts=False) is not None
    harness.tick(161)
    assert harness.fire(300, trigger_id="kv", counts=True) is None  # rate_limit
    assert harness.manager.firings == 2
    assert (harness.manager.recorded, harness.manager.recorded_counting) == (1, 0)
    snapshot = harness.stats.health()
    assert counter_value(snapshot, "suppressed_total", ("rate_limit",)) == 1


def test_events_and_tests_reach_back_the_whole_pre_window(harness: Harness) -> None:
    """An exporter restart's incident kept no scrape from before it: the
    evidence of what led to the restart."""
    harness.scrapes(0, 262)
    harness.now = 200 * S
    event = harness.manager.on_event("exporter_restart", "restarted", harness.now)
    test = harness.manager.on_test(harness.now, requested_wall_ns=None)
    assert event is not None and test is not None and event != test
    for incident_id in (event, test):
        assert harness.manager.open[incident_id].pre_start_mono == 80 * S
    harness.tick(261)
    for record in harness.of_type(INCIDENT):
        assert record["pre_window"]["fidelity"] == "complete"
        assert record["pre_window"]["fidelity_detail"]["scrapes"]["ok"] == 121


def test_a_trigger_that_joins_widens_the_pre_window_to_its_own(
    harness: Harness,
) -> None:
    first = harness.fire(200, pending_since_s=190)  # reaches back to 160 s
    second = harness.fire(230, trigger_id="kv", pending_since_s=150)  # to 120 s
    assert first == second and first is not None
    assert harness.manager.open[first].pre_start_mono == 120 * S


def test_test_triggers_neither_join_nor_are_joined(harness: Harness) -> None:
    """Test incidents absorbed real firings within their post-window, and
    each other, using up the 16-trigger cap."""
    harness.now = 200 * S
    first = harness.manager.on_test(harness.now, requested_wall_ns=None)
    harness.now = 210 * S
    second = harness.manager.on_test(harness.now, requested_wall_ns=None)
    real = harness.fire(220)
    assert len({first, second, real}) == 3
    assert all(not incident.joined for incident in harness.manager.open.values())


def test_a_firing_at_the_post_window_s_last_instant_joins(harness: Harness) -> None:
    first = harness.fire(200)  # post-window ends at 260 s
    assert harness.fire(260, trigger_id="kv") == first


def test_a_post_window_cut_short_is_partial_and_says_why(harness: Harness) -> None:
    """Cut at 230 s by the watch's end, a post-window meant to run to 260 s
    was recorded complete over the 30 s it got."""
    harness.scrapes(80, 230)
    harness.fire(200)
    harness.now = 230 * S
    harness.manager.close(harness.now)
    (record,) = harness.of_type(INCIDENT)
    validate_record(record)
    assert record["status"] == "interrupted"
    post = record["post_window"]
    assert post["fidelity"] == "partial"
    assert post["fidelity_detail"]["scrapes"]["requested_seconds"] == 60.0
    assert record["pre_window"]["fidelity"] == "complete"


def test_a_window_with_missed_ticks_is_partial(harness: Harness) -> None:
    """Ten ticks with no scrape left a post-window of 51 scrapes in 60 s
    recorded complete."""
    harness.scrapes(80, 209)
    harness.scrapes(220, 262)  # ticks 210-219 never scraped
    harness.fire(200, pending_since_s=140)
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["status"] == "completed"
    post = record["post_window"]
    assert post["fidelity_detail"]["scrapes"]["attempted"] == 51
    assert post["fidelity_detail"]["scrapes"]["expected"] == 60
    assert post["fidelity"] == "partial"
    assert record["pre_window"]["fidelity"] == "complete"


def test_a_trigger_that_resolves_within_the_post_window_says_when(
    harness: Harness,
) -> None:
    """resolved_at_ns was always null, even for a trigger that resolved
    before its incident was sealed."""
    harness.scrapes(80, 262)
    harness.fire(200)
    harness.fire(210, trigger_id="kv")
    harness.manager.on_resolved("kv", 230 * S)
    harness.manager.on_resolved("other", 231 * S)  # in no incident
    harness.tick(261)
    (record,) = harness.of_type(INCIDENT)
    assert record["trigger"]["resolved_at_ns"] is None  # still firing
    assert record["joined_triggers"][0]["resolved_at_ns"] == T0 + 230 * S


def test_a_seal_expands_no_scrape_on_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seal expanded and parsed every scrape in its windows on the loop
    to read its status and size: 1.8 s for 180 scrapes of a 135 KiB
    /metrics, every tick of it late. The history keeps both beside each
    scrape; only the store's worker expands them, to write them."""
    expanded: list[int] = []
    real_expand = incidents_module.expand

    def counting(blob: bytes) -> bytes:
        expanded.append(len(blob))
        return real_expand(blob)

    monkeypatch.setattr(incidents_module, "expand", counting)
    queued: list[Callable[[], None]] = []

    def defer(task: Callable[[], None]) -> bool:
        queued.append(task)
        return True

    harness = Harness(tmp_path)
    harness.manager._submit = defer
    harness.scrapes(80, 262, failed=frozenset({150}))
    harness.fire(200, pending_since_s=140)
    harness.tick(261)  # the seal, on the loop
    assert expanded == [] and len(queued) == 1
    queued[0]()  # the write, on the store's worker
    assert len(expanded) == 151
    (record,) = harness.of_type(INCIDENT)
    assert record["pre_window"]["fidelity_detail"]["scrapes"]["failed"] == 1


@pytest.mark.parametrize("free", ["none", "a_bundle_short"])
def test_a_disk_something_else_fills_keeps_the_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free: str
) -> None:
    """On a disk another process filled, one seal removed every bundle and
    still failed. Now nothing goes when removal could not make room, and
    room is made once: if the write still fails, the space is going
    elsewhere."""
    import errno

    harness = Harness(tmp_path)
    harness.scrapes(80, 900)
    first = harness.fire(200)
    harness.tick(261)
    middle = harness.fire(400)
    harness.tick(461)

    def full(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(harness.store, "new_bundle", full)
    first_bytes = store_module._payload_bytes(
        tmp_path / "incidents" / str(first), seen=set()
    )
    disk = {"free": 0}
    make_room = harness.store.make_room_on_disk

    def short_of(need: int, protected: frozenset[str]) -> bool:
        if free == "a_bundle_short":  # removing the oldest bundle makes room
            disk["free"] = need - first_bytes
        return make_room(need, protected)

    monkeypatch.setattr(harness.store, "make_room_on_disk", short_of)
    monkeypatch.setattr(store_module, "_free_bytes", lambda path: disk["free"])
    last = harness.fire(600)
    harness.tick(661)
    removed = [bundle.incident_id for bundle in harness.pruned]
    assert removed == ([first] if free == "a_bundle_short" else [])
    kept = [m.incident_id for _p, m in harness.store.bundles()]
    assert kept == ([middle] if free == "a_bundle_short" else [first, middle])
    record = harness.of_type(INCIDENT)[-1]
    assert record["incident_id"] == last and record["bundle"] is None
    assert "No space left" in record["bundle_error"]


def test_a_seal_the_disk_refuses_makes_room_and_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a disk with less room than the budget, the newest incidents failed
    for space while the oldest bundles were kept."""
    import errno

    harness = Harness(tmp_path)
    harness.scrapes(80, 600)
    first = harness.fire(200)
    harness.tick(261)
    real_new_bundle = harness.store.new_bundle
    refusals = [OSError(errno.ENOSPC, "No space left on device")]

    def full_once(*args: Any, **kwargs: Any) -> Any:
        if refusals:
            raise refusals.pop()
        return real_new_bundle(*args, **kwargs)

    monkeypatch.setattr(harness.store, "new_bundle", full_once)
    # The disk lacks the second bundle's lines, which the first one's free.
    monkeypatch.setattr(
        store_module, "_free_bytes", lambda path: incidents_module._RESERVE_BASE
    )
    second = harness.fire(400)
    harness.tick(461)
    assert [bundle.incident_id for bundle in harness.pruned] == [first]
    assert harness.pruned[0].reason == "disk_full"
    records = [r for r in harness.records if r["event_type"] in (INCIDENT, "pruned")]
    assert [r["event_type"] for r in records] == [INCIDENT, "pruned", INCIDENT]
    assert records[-1]["incident_id"] == second
    assert records[-1]["bundle"] == f"incidents/{second}"
