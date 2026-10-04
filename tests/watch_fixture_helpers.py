"""Build the shared ``stormlog.infer.watch/1`` fixtures from the real code.

``tests/fixtures/watch/records_v1.jsonl`` is one watch's ledger in order: a
record of every type the watcher writes, including #220's two cases, a
maximal ``infer.incident`` (16 joined triggers, all three windows, 32 engine
span links, every loss key and every suppression reason) and two incidents
open at once with their capture events interleaved. ``watch_stats_v1.json``
is one ``WatchStats.health()`` snapshot with its descriptors.

The records are built with the builders the watcher itself uses.
``test_infer_watch_records`` checks that the files equal what this module
builds, and the watcher's end-to-end tests that a real watch writes records
of the same shape. To regenerate after a deliberate schema change::

    python -m tests.watch_fixture_helpers
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stormlog.infer.watch.config import resolve_watch_config
from stormlog.infer.watch.records import (
    INCIDENT,
    INCIDENT_ASSOCIATION,
    INCIDENT_EVENT,
    INCIDENT_FINALIZED,
    INCIDENT_PRUNED,
    LOSS_KEYS,
    MAX_ENGINE_SPAN_LINKS,
    MAX_JOINED_TRIGGERS,
    MAX_REQUEST_REFS,
    SUPPRESSION_REASONS,
    TRIGGER_STATE,
    WATCH_HEALTH,
    WATCH_SESSION,
    capture_fields,
    envelope,
    trigger_fields,
)
from stormlog.infer.watch.stats import WatchStats
from stormlog.infer.watch.store import RecoveryReport

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "watch"
RECORDS = FIXTURES / "records_v1.jsonl"
STATS = FIXTURES / "watch_stats_v1.json"
SESSION = "6f1e2d3c-4b5a-4968-8776-655443322110"
RUN = "watch-0123456789ab"
OWNER = "gpu-box:4242:1790000000000000000"
T0 = 1_790_000_000_000_000_000
S = 1_000_000_000
CLOCK = "gpu-box/0d6da50f-6268-43f3-9966-ff5b177b4da9/unix_epoch_ns"
FIRST = "inc-20260922T123320Z-0001-0a1b2c3d"
SECOND = "inc-20260922T123425Z-0002-4e5f6a7b"
OLDEST = "inc-20260919T090000Z-0001-8c9d0e1f"


def _at(seconds: float) -> int:
    return T0 + int(seconds * S)


def _envelope(event_type: str, at_s: float) -> dict[str, Any]:
    return envelope(event_type, session_id=SESSION, run_id=RUN, timestamp_ns=_at(at_s))


def _trigger(index: int, fired_s: float, kind: str = "signal") -> dict[str, Any]:
    return trigger_fields(
        trigger_id=f"trigger_{index}",
        kind=kind,
        reason=f"trigger_{index}: 12 against 8 for 60 s",
        fired_at_ns=_at(fired_s),
        counts_toward_exit=True,
        threshold=8.0,
        observed=12.0,
        observed_bounds=(12.0, 15.5),
        samples=31.0,
        pending_since_ns=_at(fired_s - 60),
        sustained_ns=60 * S,
        window_seconds=30.0,
        hold_seconds=60.0,
        clear_seconds=60.0,
        detail={"signal": "queue_saturation", "status": "suspected"},
    )


def _window(start_s: float, end_s: float, detail: str) -> dict[str, Any]:
    return {
        "start_ns": _at(start_s),
        "end_ns": _at(end_s),
        "clock_domain": CLOCK,
        "fidelity": "complete",
        "detail_requested": detail,
        "detail_collected": detail,
        "fidelity_detail": {
            "scrapes": {
                "attempted": int(end_s - start_s),
                "expected": int(end_s - start_s),
                "ok": int(end_s - start_s),
                "failed": 0,
                "requested_seconds": end_s - start_s,
                "held_seconds": end_s - start_s,
            }
        },
    }


def maximal_incident() -> dict[str, Any]:
    """Every list at its bound, every optional field set, every key present."""
    record = _envelope(INCIDENT, 176)
    record.update(
        incident_id=FIRST,
        detected_at_ns=_at(100),
        owner=OWNER,
        status="completed",
        trigger=_trigger(0, 100),
        joined_triggers=[
            _trigger(i, 100 + i, "metric") for i in range(1, MAX_JOINED_TRIGGERS + 1)
        ],
        counts_toward_exit=True,
        last_informative_before_capture={"at_ns": _at(100), "state": "firing"},
        first_informative_after_mask={"at_ns": _at(150), "state": "firing"},
        pre_window=_window(40, 100, "metrics"),
        post_window=_window(100, 160, "metrics"),
        deep_window=_window(101, 131, "kernel_trace"),
        capture=capture_fields(
            "captured",
            owner=OWNER,
            stop_reason="time_bound",
            start_requested_at_ns=_at(101),
            start_returned_at_ns=_at(101.25),
            start_outcome="acknowledged",
            stop_requested_at_ns=_at(131),
            stop_returned_at_ns=_at(175),
            start_call_ns=250_000_000,
            stop_call_ns=44 * S,
        ),
        possibly_self_induced=True,
        self_induced_reason="outstanding_cohort",
        outstanding_requests=3,
        perturbation_id=OLDEST,
        rearm_basis="cap",
        suppressed={reason: 1 for reason in SUPPRESSION_REASONS},
        bundle=f"incidents/{FIRST}",
        bundle_error=None,
        traces=[
            {
                "name": "rank0.1790000101.pt.trace.json.gz",
                "bytes": 52_428_800,
                "owned": True,
            }
        ],
        attachment_ids=[f"incident:{FIRST}", f"trace:{FIRST}:rank0"],
        request_refs=[
            {"request_pseudonym": f"{i:016x}", "join_key": f"{i + 1000:016x}"}
            for i in range(1, MAX_REQUEST_REFS + 1)
        ],
        request_ref_total=57,
        engine_span_links=[
            {"trace_id": f"{i:032x}", "span_id": f"{i:016x}"}
            for i in range(1, MAX_ENGINE_SPAN_LINKS + 1)
        ],
        loss={key: index for index, key in enumerate(LOSS_KEYS)},
        trace_loss="unknown",
    )
    return record


def _session(phase: str, at_s: float, **extra: Any) -> dict[str, Any]:
    config = resolve_watch_config(
        {
            "format": "stormlog.infer.watch_config",
            "version": 1,
            "server": {"base_url": "http://127.0.0.1:8000"},
        }
    )
    record = _envelope(WATCH_SESSION, at_s)
    record.update(
        phase=phase,
        owner=OWNER,
        config=config.resolved(),
        config_digest=config.digest(),
        **extra,
    )
    return record


def _trigger_state(
    at_s: float, event: str, state: str, joined: str | None
) -> dict[str, Any]:
    record = _envelope(TRIGGER_STATE, at_s)
    record.update(
        trigger_id="trigger_0",
        kind="signal",
        event=event,
        state=state,
        reason=None,
        classification="violating",
        observed=12.0,
        threshold=8.0,
        accumulated_ns=int((at_s - 40) * S),
        pending_since_ns=_at(40),
        joined_incident_id=joined,
    )
    return record


def _event(at_s: float, incident_id: str, event: str, trigger: str) -> dict[str, Any]:
    record = _envelope(INCIDENT_EVENT, at_s)
    record.update(
        incident_id=incident_id,
        event=event,
        trigger_id=trigger,
        kind="signal",
        rearm_basis=None if event == "opened" else "cap",
    )
    return record


def _health(at_s: float, open_incidents: list[str]) -> dict[str, Any]:
    record = _envelope(WATCH_HEALTH, at_s)
    record.update(
        scrape={"status": "ok", "duration_ms": 4.25, "error": None},
        loop_lag_seconds=0.001,
        history={
            "bytes": 3_400_000,
            "seconds": 600.0,
            "evictions": {"age": 12, "bytes": 0, "oversized": 0},
        },
        open_incidents=open_incidents,
    )
    return record


def records() -> list[dict[str, Any]]:
    """A ledger in order: the first incident's capture is still stopping when
    the second opens, and the second captures once the first has stopped."""
    association = _envelope(INCIDENT_ASSOCIATION, 200)
    association.update(
        incident_id=FIRST, trigger=_trigger(17, 200), after_seal_ns=24 * S
    )
    finalized = _envelope(INCIDENT_FINALIZED, 300)
    finalized.update(
        incident_id=FIRST,
        outcome="ok",
        finding_kinds=["queue_saturation", "host_stall"],
        finding_count=2,
        report=f"incidents/{FIRST}/gen-1/report.json",
        diagnosis=f"incidents/{FIRST}/gen-1/diagnosis.json",
        finalized_at_ns=_at(300),
    )
    pruned = _envelope(INCIDENT_PRUNED, 3600)
    pruned.update(incident_id=OLDEST, reason="max_age_hours", bytes=41_783)
    return [
        _session("started", 0, recovery=asdict(RecoveryReport())),
        _trigger_state(40, "pending", "pending", None),
        _trigger_state(100, "fired", "firing", FIRST),
        _event(100, FIRST, "opened", "trigger_0"),
        _event(101, FIRST, "capture_started", "trigger_0"),
        _event(165, SECOND, "opened", "trigger_18"),
        _health(166, [FIRST, SECOND]),
        _event(175, FIRST, "capture_stopped", "trigger_0"),
        maximal_incident(),
        _event(176, SECOND, "capture_started", "trigger_18"),
        _event(186, SECOND, "capture_stopped", "trigger_18"),
        association,
        finalized,
        pruned,
        _session("ended", 3700, exit_code=3, unsound=[]),
    ]


def stats_snapshot() -> dict[str, Any]:
    """One ``WatchStats.health()`` snapshot as JSON, with its descriptors.

    A labelled family, a mapping from label tuples in memory, is written as
    a list of ``[*labels, value]`` rows.
    """
    stats = WatchStats()
    stats.set("history_bytes", 3_400_000)
    stats.set("history_capacity_bytes", 33_554_432)
    stats.set("history_seconds", 600.0)
    stats.set("history_capacity_seconds", 600.0)
    stats.set("retention_bytes", 41_783)
    stats.set("retention_incidents", 1)
    stats.set("loop_lag_seconds_max", 0.004)
    stats.add("history_evictions_total", 12, ("age",))
    stats.add("scrapes_total", 590, ("ok",))
    stats.add("scrapes_total", 9, ("failed",))
    stats.add("scrapes_total", 1, ("oversized",))
    stats.add("ticks_missed_total", 2)
    stats.add("incidents_total", labels=("signal", "disabled"))
    stats.add("incident_windows_total", labels=("pre", "complete", "metrics"))
    stats.add("incident_windows_total", labels=("post", "partial", "metrics"))
    stats.add("suppressed_total", 2, ("rate_limit",))
    stats.add("pruned_total")
    stats.add("pruned_bytes_total", 41_783)
    stats.add("sink_dropped_total", 1, ("export",))
    stats.set_trigger_state("queue_saturation", "firing")
    stats.set_trigger_state("kv_preemption", "inactive")
    return {
        "snapshot": snapshot_json(stats.health()),
        "descriptors": [asdict(d) for d in stats.health_metrics()],
    }


def snapshot_json(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            [[*labels, value] for labels, value in sorted(value.items())]
            if isinstance(value, dict)
            else value
        )
        for key, value in sorted(snapshot.items())
    }


def render_records() -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records())


def render_stats() -> str:
    return json.dumps(stats_snapshot(), indent=2, sort_keys=True) + "\n"


def write() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text(render_records(), encoding="utf-8")
    STATS.write_text(render_stats(), encoding="utf-8")


if __name__ == "__main__":
    write()
