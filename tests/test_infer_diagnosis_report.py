"""diagnose_artifact: the report it returns, and what it reads."""

from __future__ import annotations

import builtins
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from tests.diagnosis_scenarios import MS, Engine, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET

SCHEMAS = Path(__file__).resolve().parents[1] / "docs" / "schemas"
GENERATED = 1_800_000_000_000_000_000


def _schema(name: str) -> dict[str, Any]:
    schema = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    assert isinstance(schema, dict)
    return schema


def _validate(report: dict[str, Any]) -> None:
    Draft202012Validator(_schema("stormlog_report_v1.schema.json")).validate(report)
    Draft202012Validator(_schema("inference_diagnosis_v1.schema.json")).validate(
        report["payload"]
    )


@pytest.fixture(scope="module")
def burst(tmp_path_factory: pytest.TempPathFactory) -> Path:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    heavy = poisson_free(300, 90 * SECOND, 5 * MS, prefix="b")
    return build_run(tmp_path_factory.mktemp("d"), calm + heavy, Engine(max_num_seqs=4))


def _options(**changes: Any) -> DiagnoseOptions:
    return DiagnoseOptions(generated_at_ns=GENERATED, **changes)


def test_a_queue_incident_is_a_warning_report(burst: Path) -> None:
    report = diagnose_artifact(burst, options=_options())

    _validate(report)
    assert report["report_kind"] == "inference_diagnosis"
    assert report["verdict"]["exit_code"] == 3
    assert report["verdict"]["status"] == "findings"
    kinds = [f["kind"] for f in report["findings"]]
    assert kinds[0] == "queue_saturation" and "load_increase" in kinds
    finding = report["findings"][0]
    payload = report["payload"]
    detail = payload["findings_detail"][finding["id"]]
    assert (finding["kind"], finding["severity"]) == ("queue_saturation", "warning")
    # The edge table the roles were settled against, as the scorer reads it.
    assert payload["edges"]["version"] == "diagnosis_edges_v1"
    assert len(payload["edges"]["table"]) == 6
    assert (detail["claim"], detail["cause"], detail["rank"]) == ("fault", "fault", 1)
    assert payload["outcome"] == "findings"
    assert payload["inputs"][0]["sha256"] and payload["inputs"][0]["lines"] > 0
    assert payload["coverage"]["queue_saturation"]["status"] == "assessed"
    assert payload["coverage"]["rank_delay"] == {
        "status": "unsupported",
        "reasons": ["not_assessed_by_this_version"],
        "by_subject": {},
    }
    assert payload["coverage"]["host_stall"]["status"] == "partial"
    assert "worker_not_assessed_by_this_version" in (
        payload["coverage"]["host_stall"]["reasons"]
    )
    # Per component: the API server and the engine loop assessed, each by its
    # own class; the workers not by this version.
    assert payload["coverage"]["host_stall"]["components"] == {
        "api_server": "assessed",
        "engine_core": "assessed",
        "worker": "unsupported",
    }
    assert len(finding["evidence"]) <= 8
    first = finding["evidence"][0]
    assert first["pointer"].startswith("/") and first["record_id"]
    assert report["generated_at_utc"] == "2027-01-15T08:00:00Z"
    assert report["run_id"] == "run-1"


def test_the_same_artifact_gives_the_same_report(burst: Path) -> None:
    assert diagnose_artifact(burst, options=_options()) == diagnose_artifact(
        burst, options=_options()
    )


def test_options_change_the_digest_and_are_recorded(burst: Path) -> None:
    plain = diagnose_artifact(burst, options=_options())
    strict = diagnose_artifact(
        burst, options=_options(thresholds={"selection.alpha": 0.001})
    )

    digests = {r["payload"]["diagnoser"]["config_digest"] for r in (plain, strict)}
    assert len(digests) == 2
    assert strict["payload"]["thresholds"]["overridden"] == ["selection.alpha"]


def test_a_calm_run_has_no_findings(tmp_path: Path) -> None:
    calm = build_run(tmp_path, poisson_free(160, 10 * SECOND, 500 * MS), Engine())

    report = diagnose_artifact(calm, options=_options())

    _validate(report)
    assert report["verdict"]["exit_code"] == 0
    assert report["payload"]["outcome"] == "no_findings"
    assert report["findings"] == []


def test_a_run_too_short_to_test_is_inconclusive_not_clean(tmp_path: Path) -> None:
    # 40 requests never reach the 114 a reference needs: selection abstains
    # on every window, so it ruled nothing out.
    short = build_run(tmp_path, poisson_free(40, 10 * SECOND, 500 * MS), Engine())

    report = diagnose_artifact(short, options=_options())

    _validate(report)
    windows = report["payload"]["selection"]["windows"]
    assert windows and all(w["status"] is not None for w in windows)
    assert report["payload"]["outcome"] == "inconclusive"
    assert report["verdict"]["exit_code"] == 0  # an abstention is not a warning
    assert f"{len(windows)} windows untested" in report["verdict"]["summary"]


def test_a_calm_run_counts_the_windows_it_could_not_test(tmp_path: Path) -> None:
    calm = build_run(tmp_path, poisson_free(160, 10 * SECOND, 500 * MS), Engine())

    report = diagnose_artifact(calm, options=_options())

    windows = report["payload"]["selection"]["windows"]
    untested = sum(1 for w in windows if w["status"] is not None)
    assert 0 < untested < len(windows)
    assert report["payload"]["outcome"] == "no_findings"
    assert report["verdict"]["summary"].endswith(f"{untested} windows untested")


def test_a_finding_says_what_made_its_incident_detectable(burst: Path) -> None:
    report = diagnose_artifact(burst, options=_options())

    detail = report["payload"]["findings_detail"][report["findings"][0]["id"]]
    evidence = detail["detection_evidence"]
    detected = detail["first_detectable_ns"]
    assert evidence["basis"] == "selection_sustained/1"
    assert evidence["client_records_through_ns"] == detected
    first, second = evidence["windows"]
    assert second["evaluated_at_ns"] == detected
    assert first["flagged_on"] and second["flagged_on"]
    test = first["tests"][first["flagged_on"][0]]
    assert test["above"] >= 3 and test["p"] < 0.01


def test_a_declared_window_is_the_subject(burst: Path) -> None:
    start = 90 * SECOND + WALL_OFFSET

    report = diagnose_artifact(
        burst, windows=[(start, start + 2 * SECOND)], options=_options()
    )

    (subject,) = report["payload"]["selection"]["subjects"]
    assert subject["declared_by"] == "caller" and subject["requests"] == 300
    assert report["payload"]["selection"]["automatic"] is False
    assert subject["detection_unavailable"] == "declared"
    details = report["payload"]["findings_detail"].values()
    assert all(d["detection_evidence"] is None for d in details)


def test_a_declared_window_over_healthy_traffic_has_no_findings(
    tmp_path: Path,
) -> None:
    calm = build_run(tmp_path, poisson_free(160, 10 * SECOND, 500 * MS), Engine())
    start = 70 * SECOND + WALL_OFFSET

    report = diagnose_artifact(
        calm, windows=[(start, start + 10 * SECOND)], options=_options()
    )

    _validate(report)
    assert report["payload"]["outcome"] == "no_findings"
    assert "0 incidents unexplained" in report["verdict"]["summary"]
    assert report["verdict"]["summary"].endswith("no excess in the declared window")


def test_an_incident_without_an_eligible_explanation_is_inconclusive(
    tmp_path: Path,
) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    heavy = poisson_free(300, 90 * SECOND, 5 * MS, prefix="b")
    engine = Engine(max_num_seqs=4, enqueued_records=False, observes=None)
    artifact = build_run(tmp_path, calm + heavy, engine)

    report = diagnose_artifact(artifact, options=_options())

    _validate(report)
    assert report["payload"]["outcome"] == "inconclusive"
    assert report["verdict"]["exit_code"] == 0  # an observation is not a warning
    assert {f["severity"] for f in report["findings"]} == {"info"}


@pytest.mark.parametrize(
    "windows, options, message",
    [
        (None, {"thresholds": {"no.such.key": 1.0}}, "unknown threshold keys"),
        (None, {"thresholds": {"selection.window_seconds": math.nan}}, "finite"),
        (None, {"thresholds": {"selection.window_seconds": math.inf}}, "finite"),
        ([(5, 5)], {}, "is empty"),
    ],
)
def test_bad_options_are_refused(
    burst: Path, windows: Any, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        diagnose_artifact(burst, windows=windows, options=_options(**options))


def test_diagnosis_reads_only_the_artifact(
    burst: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    listed: list[str] = []
    real_open, real_path_open = builtins.open, Path.open
    real_scandir, real_listdir = os.scandir, os.listdir

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    def spy_scandir(path: Any = ".") -> Any:
        listed.append(str(path))
        return real_scandir(path)

    def spy_listdir(path: Any = ".") -> Any:
        listed.append(str(path))
        return real_listdir(path)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path_open)
    monkeypatch.setattr(os, "scandir", spy_scandir)
    monkeypatch.setattr(os, "listdir", spy_listdir)

    diagnose_artifact(burst, options=_options())

    assert set(opened) == {str(burst)} and listed == []
