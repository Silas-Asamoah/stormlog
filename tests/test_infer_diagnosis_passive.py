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
