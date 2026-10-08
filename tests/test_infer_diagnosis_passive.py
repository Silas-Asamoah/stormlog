"""A server-only artifact, as an incident watcher captures it: the engine's
steps and no client requests, diagnosed over a declared window."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from tests.diagnosis_scenarios import MS, SESSION, Engine, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET

AT = 90 * SECOND


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, AT, 5 * MS, prefix="b")
    # A phase begun and never ended keeps every foreign-only step.
    begun = {
        "schema_version": 1,
        "event_type": "infer.phase_start",
        "session_id": SESSION,
        "case_id": "watch",
        "phase": "measured",
        "arrival_mode": "passive",
        "started_at_ns": WALL_OFFSET,
        "timestamp_ns": WALL_OFFSET,
    }
    artifact = build_run(
        tmp_path_factory.mktemp("p"),
        calm + burst,
        Engine(max_num_seqs=4),
        windows=[begun],
        client=False,
    )
    start = AT + WALL_OFFSET
    return diagnose_artifact(
        artifact,
        windows=[(start, start + 2 * SECOND)],
        options=DiagnoseOptions(generated_at_ns=1),
    )


def test_a_window_of_engine_executions_is_the_subject(report: dict[str, Any]) -> None:
    (subject,) = report["payload"]["selection"]["subjects"]

    assert (subject["basis"], subject["requests"]) == ("engine", 0)
    assert subject["executions"] == 300 and subject["reference_executions"] == 140


def test_the_queue_is_found_from_the_engine_alone(report: dict[str, Any]) -> None:
    payload = report["payload"]
    (queue,) = [f for f in report["findings"] if f["kind"] == "queue_saturation"]
    detail = payload["findings_detail"][queue["id"]]

    assert detail["status"] == "partial"
    assert detail["partial_reasons"] == ["no_client_latency"]
    assert detail["segments"]["decomposition"] == "engine_ttft"
    assert detail["eligibility"]["eligible"] is True
    statuses = {a["kind"]: a["status"] for a in detail["alternatives"]}
    assert statuses["client_admission"] == "untestable"
    assert detail["confidence"]["level"] == "medium"  # partial caps it


def test_client_classes_say_there_are_no_client_requests(
    report: dict[str, Any]
) -> None:
    coverage = report["payload"]["coverage"]

    for kind in ("client_admission", "load_increase", "prefix_cache_loss"):
        assert coverage[kind]["status"] == "unsupported", kind
    assert "no_client_requests" in coverage["client_admission"]["reasons"]


def _incident_window(name: str, start: int, end: int) -> dict[str, Any]:
    """An incident watcher's window record, exactly as #219's bundle writes
    it (``stormlog.infer.watch.incidents``): no ``timestamp_ns``."""
    return {
        "schema_version": 1,
        "event_type": "infer.incident_window",
        "session_id": SESSION,
        "run_id": "run-1",
        "incident_id": "incident-1",
        "window": name,
        "start_ns": start,
        "end_ns": end,
    }


def test_a_watcher_s_bundle_is_diagnosed_after_its_detection(tmp_path: Any) -> None:
    """The bundle's own windows place the other clients' steps on import,
    and its post window is the subject, against the pre window before it."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, AT, 5 * MS, prefix="b")
    # The post window runs until the queue the burst built has drained.
    windows = [
        _incident_window("pre", 10 * SECOND + WALL_OFFSET, AT + WALL_OFFSET),
        _incident_window("post", AT + WALL_OFFSET, AT + WALL_OFFSET + 15 * SECOND),
    ]
    artifact = build_run(
        tmp_path, calm + burst, Engine(max_num_seqs=4), windows=windows, client=False
    )

    report = diagnose_artifact(artifact, options=DiagnoseOptions(generated_at_ns=1))

    (subject,) = report["payload"]["selection"]["subjects"]
    assert subject["declared_by"] == "incident_window"
    assert subject["executions"] == 300 and subject["reference_executions"] == 140
    assert "queue_saturation" in {f["kind"] for f in report["findings"]}


def test_an_earlier_incident_is_no_part_of_a_later_one_s_reference(
    tmp_path: Any,
) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    first = poisson_free(100, AT - 20 * SECOND, 5 * MS, prefix="f")
    burst = poisson_free(300, AT, 5 * MS, prefix="b")
    windows = [
        _incident_window("pre", 10 * SECOND + WALL_OFFSET, AT + WALL_OFFSET),
        _incident_window(
            "post", AT - 20 * SECOND + WALL_OFFSET, AT - 18 * SECOND + WALL_OFFSET
        ),
        _incident_window("post", AT + WALL_OFFSET, AT + WALL_OFFSET + 15 * SECOND),
    ]
    artifact = build_run(
        tmp_path,
        calm + first + burst,
        Engine(max_num_seqs=4),
        windows=windows,
        client=False,
    )

    report = diagnose_artifact(artifact, options=DiagnoseOptions(generated_at_ns=1))

    earlier, later = report["payload"]["selection"]["subjects"]
    assert earlier["executions"] == 104  # its 100, and 4 calm ones
    # The earlier incident's are no part of the later one's reference.
    assert later["executions"] == 300 and later["reference_executions"] == 136
