"""The metric catalog and each artifact's panel states (design §5.5; C2)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.latency_report import CLIENT_METRICS
from stormlog.infer.slo import CRITERIA
from stormlog.studio.contracts import AnalysisState, DictTable, envelope, to_wire
from stormlog.studio.metric_catalog import (
    PANEL_STATES,
    PANELS,
    metric_catalog,
    panels,
)
from tests.studio.conftest import StandardArtifacts
from tests.studio.test_selection import (
    closed_case,
    session_record,
    snapshot_from_records,
    snapshot_of,
)
from tests.studio.test_summaries import schema_errors

READY = AnalysisState("ready", None, 0, {"observers": {"observers": {}}}, 100)
PENDING = AnalysisState("pending", None, None, None, None)


def _entries() -> dict[str, dict[str, Any]]:
    return {entry["key"]: entry for entry in metric_catalog()["metrics"]}


def test_every_entry_states_unit_scope_source_window_and_status() -> None:
    entries = _entries()
    assert len(entries) == len(metric_catalog()["metrics"]), "keys are unique"
    for entry in entries.values():
        assert entry["unit"], entry["key"]
        assert entry["scope"] in ("client", "server", "client_local_host")
        assert entry["sources"], entry["key"]
        assert entry["aggregation_window"], entry["key"]
        assert entry["status"] in ("indexed", "not_indexed")
        assert entry["computed_by"], entry["key"]


def test_latency_entries_come_from_the_slo_criteria() -> None:
    entries = _entries()
    for key in CLIENT_METRICS:
        assert entries[key]["label"] == CRITERIA[key].definition
        assert entries[key]["status"] == "indexed"
    for key in CRITERIA:
        if key.startswith("server."):
            assert entries[key]["status"] == "not_indexed"
            assert entries[key]["reason"] == "server_evidence"


def test_server_counts_are_not_indexed_with_their_reason() -> None:
    entries = _entries()
    for key in ("counts.server_admitted", "counts.server_evidence_coverage"):
        assert entries[key]["status"] == "not_indexed"
        assert entries[key]["reason"] == "server_evidence"
        assert entries[key]["collector"]


def test_every_summary_number_has_an_entry() -> None:
    entries = _entries()
    for key in (
        "failure.share",
        "counts.offered",
        "counts.censored",
        "throughput.requests_per_second",
        "streaming.chunk_interarrival_ms",
        "tokens.output.known",
        "tokens.prompt.known_sum",
        "client_memory.peak_process_rss_bytes",
        "request.elapsed_ms",
        "request.gaps",
    ):
        assert key in entries, key
    assert "chunk-level" in entries["streaming.chunk_interarrival_ms"]["notes"][0]


def _states(document: dict[str, Any]) -> dict[str, str]:
    return {name: panel["state"] for name, panel in document["panels"].items()}


def test_every_panel_has_a_known_state(studio_artifacts: StandardArtifacts) -> None:
    for name in (
        "mixed",
        "all_failed",
        "non_streaming",
        "execution_import",
        "growing",
        "empty",
    ):
        snapshot = snapshot_of(studio_artifacts[name])
        document = panels(snapshot, READY)
        assert set(document["panels"]) == set(PANELS)
        assert set(_states(document).values()) <= set(PANEL_STATES), name
        wire = to_wire(envelope("artifact_capabilities", document, snapshot))
        assert schema_errors("artifact_capabilities", wire) == [], name


def test_the_catalog_validates_against_its_schema() -> None:
    wire = to_wire(envelope("metric_catalog", metric_catalog()))
    assert schema_errors("metric_catalog", wire) == []


def test_server_evidence_in_the_artifact_is_not_indexed(
    studio_artifacts: StandardArtifacts,
) -> None:
    imported = panels(snapshot_of(studio_artifacts["execution_import"]), READY)
    server = imported["panels"]["server_evidence"]
    assert server["state"] == "not_indexed" and server["reason"] == "server_evidence"
    assert server["detail"]["records"]
    assert imported["panels"]["execution"]["state"] == "in_artifact_panel_later"
    plain = panels(snapshot_of(studio_artifacts["mixed"]), READY)["panels"]
    assert plain["server_evidence"]["state"] == "unavailable"
    assert plain["server_evidence"]["collector"]


def test_an_unsupported_client_line_is_no_server_evidence(tmp_path: Path) -> None:
    # Fable's F-9: a client infer.request at version 2 (no context) is an
    # unsupported_version problem, never a row; it was counted as a
    # server-view request, so the panel claimed server evidence.
    from tests.studio.test_index import Harness

    records = [session_record(), *closed_case()]
    records.append(dict(records[-2], schema_version=2))
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    snapshot = Harness(tmp_path).build(path.name)
    assert snapshot.problem_counts == {"unsupported_version": 1}
    server = panels(snapshot, READY)["panels"]["server_evidence"]
    assert (server["state"], server["reason"]) == ("unavailable", "no_server_evidence")


def test_server_view_requests_are_counted_apart_from_unsupported_client_lines() -> None:
    snapshot = replace(
        snapshot_from_records(closed_case()),
        inventory={"infer.request": {1: 10, 2: 3}},
        problem_counts={"unsupported_version": 1},
    )
    server = panels(snapshot, READY)["panels"]["server_evidence"]
    assert server["state"] == "not_indexed"
    assert server["detail"]["records"] == {"infer.request (server view)": 2}


def test_states_follow_what_the_artifact_holds(
    studio_artifacts: StandardArtifacts,
) -> None:
    failed = _states(panels(snapshot_of(studio_artifacts["all_failed"]), READY))
    assert failed["latency"] == "empty" and failed["failures"] == "available"
    plain = _states(panels(snapshot_of(studio_artifacts["non_streaming"]), READY))
    assert plain["streaming"] == "empty"
    growing = panels(snapshot_of(studio_artifacts["growing"]), READY)["panels"]
    assert growing["requests"] == {
        "state": "partial",
        "reason": "growing",
        "collector": None,
        "detail": growing["requests"]["detail"],
    }
    assert growing["rates"]["state"] == "unavailable"
    assert growing["rates"]["reason"] == "phase_window_missing"


def test_an_index_that_is_building_makes_every_panel_pending() -> None:
    snapshot = replace(
        snapshot_from_records([]), status="building", requests=(), table=DictTable(())
    )
    document = panels(snapshot, PENDING)
    assert set(_states(document).values()) == {"pending"}
    assert document["panels"]["requests"]["reason"] == "index_building"
    assert document["observers"] is None


@pytest.mark.parametrize(
    ("failure", "state", "reason"),
    [
        ("audio", "unsupported", "audio_artifact"),
        ("unreadable", "unavailable", "unreadable"),
    ],
)
def test_a_failed_index_says_why_on_every_panel(
    failure: Any, state: str, reason: str
) -> None:
    snapshot = replace(
        snapshot_from_records(closed_case()), status="failed", failure=failure
    )
    document = panels(snapshot, READY)
    for name, panel in document["panels"].items():
        if name != "analysis":
            assert (panel["state"], panel["reason"]) == (state, reason), name


def test_gaps_that_are_not_resident_make_streaming_unavailable() -> None:
    snapshot = replace(snapshot_from_records(closed_case()), gaps_resident=False)
    assert (
        panels(snapshot, READY)["panels"]["streaming"]["reason"] == "gaps_not_resident"
    )


def test_no_samples_name_the_collector() -> None:
    memory = panels(snapshot_from_records(closed_case()), READY)["panels"][
        "client_memory"
    ]
    assert memory["state"] == "unavailable" and memory["reason"] == "no_system_samples"
    assert "--system-sampler" in memory["collector"]


def test_a_kept_record_over_the_limit_makes_rates_unavailable() -> None:
    records = closed_case()
    records[1]["padding"] = "x" * 400
    snapshot = snapshot_from_records(records, kept_record_bytes=350)
    rates = panels(snapshot, READY)["panels"]["rates"]
    assert rates["reason"] == "kept_record_over_limit"
    assert rates["detail"] == {"record_types": ["infer.workload"]}


def test_the_analyzers_observers_are_passed_through_once_ready() -> None:
    snapshot = snapshot_from_records(closed_case())
    ready = panels(snapshot, READY)
    assert ready["observers"] == {"observers": {}}
    assert ready["observers_source"].startswith("infer analyze")
    failed = AnalysisState("failed", "timeout", None, None, None)
    analysis = panels(snapshot, failed)["panels"]["analysis"]
    assert (analysis["state"], analysis["reason"]) == ("unavailable", "timeout")
    assert (
        panels(snapshot, PENDING)["panels"]["analysis"]["reason"] == "analysis_pending"
    )
