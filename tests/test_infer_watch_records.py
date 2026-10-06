"""The frozen ``stormlog.infer.watch/1`` records and their shared fixtures."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

import pytest

from stormlog.infer.watch.records import (
    CAPTURE_TIMING_KEYS,
    EVENT_TYPES,
    INCIDENT,
    INCIDENT_EVENT,
    INCIDENT_EVENTS,
    LOSS_KEYS,
    MAX_ENGINE_SPAN_LINKS,
    MAX_JOINED_TRIGGERS,
    MAX_REQUEST_REFS,
    SUPPRESSION_REASONS,
    WATCH_FORMAT,
    capture_fields,
    empty_loss,
    envelope,
    finite,
    trigger_fields,
    validate_record,
)
from stormlog.infer.watch.stats import DESCRIPTORS
from tests import watch_fixture_helpers as fixtures


def _fixture_records() -> list[dict[str, Any]]:
    lines = fixtures.RECORDS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _incident() -> dict[str, Any]:
    return copy.deepcopy(fixtures.maximal_incident())


def test_records_fixture_is_what_the_builders_write() -> None:
    assert (
        fixtures.RECORDS.read_text(encoding="utf-8") == fixtures.render_records()
    ), "records_v1.jsonl is stale: run python -m tests.watch_fixture_helpers"


def test_stats_fixture_is_what_watch_stats_reports() -> None:
    assert (
        fixtures.STATS.read_text(encoding="utf-8") == fixtures.render_stats()
    ), "watch_stats_v1.json is stale: run python -m tests.watch_fixture_helpers"


def test_fixture_holds_every_record_type_and_incident_event() -> None:
    records = _fixture_records()
    for record in records:
        validate_record(record)
    assert {r["event_type"] for r in records} == set(EVENT_TYPES)
    events = Counter(r["event"] for r in records if r["event_type"] == INCIDENT_EVENT)
    assert set(events) == set(INCIDENT_EVENTS)


def test_fixture_incident_is_maximal() -> None:
    (incident,) = [r for r in _fixture_records() if r["event_type"] == INCIDENT]
    assert len(incident["joined_triggers"]) == MAX_JOINED_TRIGGERS
    assert len(incident["engine_span_links"]) == MAX_ENGINE_SPAN_LINKS
    assert len(incident["request_refs"]) == MAX_REQUEST_REFS
    assert all(
        incident[w] is not None for w in ("pre_window", "post_window", "deep_window")
    )
    assert list(incident["loss"]) == sorted(LOSS_KEYS)
    assert all(isinstance(v, int) for v in incident["loss"].values())
    assert set(incident["suppressed"]) == set(SUPPRESSION_REASONS)
    assert set(CAPTURE_TIMING_KEYS) <= set(incident["capture"])
    assert all(incident["capture"][key] is not None for key in CAPTURE_TIMING_KEYS)


def test_fixture_has_two_incidents_open_with_interleaved_capture_events() -> None:
    events = [
        (r["incident_id"], r["event"])
        for r in _fixture_records()
        if r["event_type"] == INCIDENT_EVENT
    ]
    first, second = fixtures.FIRST, fixtures.SECOND
    assert events == [
        (first, "opened"),
        (first, "capture_started"),
        (second, "opened"),
        (first, "capture_stopped"),
        (second, "capture_started"),
        (second, "capture_stopped"),
    ]
    health = [r for r in _fixture_records() if r["event_type"] == "infer.watch_health"]
    assert health[0]["open_incidents"] == [first, second]


def test_fixture_ledger_is_in_time_order() -> None:
    stamps = [r["timestamp_ns"] for r in _fixture_records()]
    assert stamps == sorted(stamps)


def test_stats_fixture_describes_every_snapshot_key() -> None:
    payload = json.loads(fixtures.STATS.read_text(encoding="utf-8"))
    described = {d["key"] for d in payload["descriptors"]}
    assert set(payload["snapshot"]) <= described
    assert [d["key"] for d in payload["descriptors"]] == [d.key for d in DESCRIPTORS]
    assert all(d["name"].startswith("stormlog_watch_") for d in payload["descriptors"])


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("trigger", "kind"), "slo_typo", "trigger kind"),
        (("capture", "status"), "maybe", "capture status"),
        (("capture", "start_outcome"), "sent", "start_outcome"),
        (("pre_window", "fidelity"), "most", "fidelity"),
        (("deep_window", "detail_collected"), "everything", "detail_collected"),
        (("rearm_basis",), "vibes", "rearm_basis"),
        (("self_induced_reason",), "because", "self_induced_reason"),
    ],
)
def test_incident_values_come_from_closed_vocabularies(
    path: tuple[str, ...], value: str, message: str
) -> None:
    record = _incident()
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_record(record)


def test_incident_rejects_an_unknown_suppression_reason() -> None:
    record = _incident()
    record["suppressed"]["merged"] = 1
    with pytest.raises(ValueError, match="suppression reason"):
        validate_record(record)


@pytest.mark.parametrize(
    ("name", "limit"),
    [
        ("joined_triggers", MAX_JOINED_TRIGGERS),
        ("engine_span_links", MAX_ENGINE_SPAN_LINKS),
        ("request_refs", MAX_REQUEST_REFS),
    ],
)
def test_incident_lists_are_bounded(name: str, limit: int) -> None:
    record = _incident()
    record[name] = record[name] + [record[name][0]]
    assert len(record[name]) == limit + 1
    with pytest.raises(ValueError, match=f"at most {limit}"):
        validate_record(record)


def test_loss_holds_exactly_the_closed_keys_as_integers_or_null() -> None:
    record = _incident()
    record["loss"] = empty_loss()
    validate_record(record)
    record["loss"]["scrapes_failed"] = 2
    validate_record(record)
    missing = _incident()
    del missing["loss"]["hook_capped"]
    with pytest.raises(ValueError, match="closed set"):
        validate_record(missing)
    extra = _incident()
    extra["loss"]["gpu_melted"] = 1
    with pytest.raises(ValueError, match="closed set"):
        validate_record(extra)
    for bad in (1.5, True, "3"):
        wrong = _incident()
        wrong["loss"]["ledger_dropped"] = bad
        with pytest.raises(ValueError, match="integers or null"):
            validate_record(wrong)


@pytest.mark.parametrize(
    ("event_type", "field", "value"),
    [
        ("infer.trigger_state", "state", "exploded"),
        ("infer.trigger_state", "event", "fired_twice"),
        ("infer.incident_event", "event", "closed"),
        ("infer.incident_event", "rearm_basis", "whenever"),
        ("infer.incident_finalized", "outcome", "maybe"),
        ("infer.incident_pruned", "reason", "felt_like_it"),
        ("infer.watch_session", "phase", "middle"),
    ],
)
def test_every_record_type_checks_its_enums(
    event_type: str, field: str, value: str
) -> None:
    (record,) = [
        r for r in _fixture_records() if r["event_type"] == event_type and field in r
    ][:1]
    record[field] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_session_unsound_reasons_are_closed() -> None:
    (ended,) = [r for r in _fixture_records() if r.get("phase") == "ended"]
    ended["unsound"] = ["no_successful_scrape"]
    validate_record(ended)
    ended["unsound"] = ["bad_vibes"]
    with pytest.raises(ValueError, match="unsound reason"):
        validate_record(ended)


def test_envelope_and_identity_are_required() -> None:
    record = envelope("infer.watch_health", session_id="s", run_id="r", timestamp_ns=1)
    assert record["format"] == WATCH_FORMAT
    validate_record(record)
    with pytest.raises(ValueError, match="unknown watch record type"):
        validate_record({**record, "event_type": "infer.other"})
    with pytest.raises(ValueError, match="not a stormlog.infer.watch/1"):
        validate_record({**record, "format": "stormlog.infer.watch/2"})
    with pytest.raises(ValueError, match="run_id"):
        validate_record({**record, "run_id": ""})


def test_trigger_fields_always_carry_every_key() -> None:
    minimal = trigger_fields(
        trigger_id="t", kind="test", reason="r", fired_at_ns=1, counts_toward_exit=False
    )
    full = fixtures.maximal_incident()["trigger"]
    assert set(minimal) == set(full)
    assert minimal["observed_bounds"] is None and minimal["detail"] == {}
    nan = trigger_fields(
        trigger_id="t",
        kind="metric",
        reason="r",
        fired_at_ns=1,
        counts_toward_exit=True,
        observed=float("nan"),
        observed_bounds=(1.0, float("inf")),
    )
    assert nan["observed"] is None
    assert nan["observed_bounds"] == [1.0, None]


def test_capture_fields_hold_every_timing_key_and_refuse_others() -> None:
    capture = capture_fields("disabled", owner="o")
    assert set(capture) == {"status", "stop_reason", "owner", *CAPTURE_TIMING_KEYS}
    assert all(capture[key] is None for key in CAPTURE_TIMING_KEYS)
    with pytest.raises(ValueError, match="unknown capture fields"):
        capture_fields("captured", owner="o", stopped_at_ns=1)


def test_finite_drops_nan_and_infinities() -> None:
    assert finite(None) is None
    assert finite(float("nan")) is None
    assert finite(float("-inf")) is None
    assert finite(3) == 3.0
