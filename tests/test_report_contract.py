"""Contract tests for the stormlog.report v1 envelope.

Every malformed case is checked against both the published JSON Schema and
``validate_report`` so the two cannot drift apart.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from stormlog.exit_codes import ExitCode, verdict_status
from stormlog.report import (
    REPORT_FORMAT,
    REPORT_SCHEMA_VERSION,
    Artifact,
    Evidence,
    Finding,
    build_report,
    load_report,
    validate_report,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "stormlog_report_v1.schema.json"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "reports"


def _schema() -> dict[str, object]:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _validator() -> Any:
    return jsonschema.Draft202012Validator(_schema())


def _fixture(name: str) -> dict[str, Any]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_rejected(report: dict[str, Any], message: str) -> None:
    assert list(_validator().iter_errors(report)), "schema accepted the report"
    with pytest.raises(ValueError, match=message):
        validate_report(report)


def _assert_accepted(report: dict[str, Any]) -> None:
    _validator().validate(report)
    validate_report(report)


def test_schema_file_is_a_valid_draft_2020_12_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_fixture_report_with_findings_is_accepted() -> None:
    _assert_accepted(_fixture("diagnose_findings.json"))


def test_build_report_output_is_accepted() -> None:
    report = build_report(
        report_kind="diagnose",
        tool_name="gpumemprof",
        command="diagnose",
        exit_code=ExitCode.FINDINGS,
        summary="Memory risk detected: high_utilization",
        findings=[
            Finding(
                id="diagnose.high_utilization",
                kind="high_utilization",
                severity="warning",
                title="Device memory utilization is above the risk threshold",
                metrics={"utilization_ratio": 0.9, "threshold": 0.85},
                evidence=[
                    Evidence(
                        kind="diagnose_summary",
                        path="diagnostic_summary.json",
                        pointer="/risk_flags/high_utilization",
                    ),
                    Evidence(kind="session", session_id="session-1"),
                ],
            )
        ],
        metrics={"utilization_ratio": 0.9, "total_bytes": None},
        artifacts=[
            Artifact(kind="diagnose_manifest", path="manifest.json", schema_version=2),
            Artifact(
                kind="telemetry_sink",
                path="sink/manifest.json",
                format="stormlog.append_only_telemetry_sink",
                schema_version=2,
            ),
        ],
        recommendations=["Reduce batch size."],
        session_id="session-1",
        payload={"risk_flags": {"high_utilization": True}},
        argv=["gpumemprof", "diagnose"],
        tool_version="0.4.0",
        generated_at_utc="2026-10-01T12:00:00Z",
    )

    _assert_accepted(report)
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["format"] == REPORT_FORMAT
    assert report["verdict"] == {
        "status": "findings",
        "exit_code": 3,
        "summary": "Memory risk detected: high_utilization",
    }
    assert report["findings"][0]["evidence"][1] == {
        "kind": "session",
        "session_id": "session-1",
    }
    assert "message" not in report["findings"][0]


def test_build_report_defaults_fill_version_and_timestamp() -> None:
    report = build_report(
        report_kind="diagnose",
        tool_name="gpumemprof",
        command="diagnose",
        exit_code=ExitCode.OK,
        summary="No memory risk detected",
    )

    _assert_accepted(report)
    assert report["tool"]["version"]
    assert report["generated_at_utc"].endswith("Z")
    assert "argv" not in report["tool"]
    assert "payload" not in report
    assert report["findings"] == []


def test_build_report_rejects_exit_code_outside_the_contract() -> None:
    with pytest.raises(ValueError, match="not in the contract"):
        build_report(
            report_kind="diagnose",
            tool_name="gpumemprof",
            command="diagnose",
            exit_code=42,
            summary="?",
        )


@pytest.mark.parametrize("exit_code", list(ExitCode))
def test_every_contract_exit_code_pairs_with_its_status_in_the_schema(
    exit_code: ExitCode,
) -> None:
    report = build_report(
        report_kind="diagnose",
        tool_name="gpumemprof",
        command="diagnose",
        exit_code=exit_code,
        summary=f"exit {int(exit_code)}",
    )

    _assert_accepted(report)
    assert report["verdict"]["status"] == verdict_status(exit_code)

    mismatched = copy.deepcopy(report)
    mismatched["verdict"]["exit_code"] = 0 if exit_code != ExitCode.OK else 1
    _assert_rejected(mismatched, "does not pair")


def test_minimal_report_is_accepted() -> None:
    _assert_accepted(
        {
            "schema_version": 1,
            "format": "stormlog.report",
            "report_kind": "diagnose",
            "generated_at_utc": "2026-10-01T12:00:00Z",
            "tool": {"name": "gpumemprof", "command": "diagnose"},
            "verdict": {"status": "pass", "exit_code": 0, "summary": "ok"},
            "findings": [],
        }
    )


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    def mutate(report: dict[str, Any]) -> None:
        del report[key]

    return mutate


def _set(path: tuple[str | int, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(report: dict[str, Any]) -> None:
        target: Any = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop("verdict"), "missing required fields: verdict"),
        (_drop("findings"), "missing required fields: findings"),
        (_set(("schema_version",), 2), "schema_version must be 1"),
        (_set(("format",), "stormlog.run_envelope"), "format must be"),
        (_set(("report_kind",), "Diagnose Bundle"), "report_kind"),
        (_set(("generated_at_utc",), ""), "generated_at_utc"),
        (_set(("extra",), True), "unknown fields: extra"),
        (_set(("tool", "name"), ""), "tool.name"),
        (_set(("tool", "argv"), ["ok", 1]), "tool.argv"),
        (_set(("verdict", "status"), "risk"), "not a verdict status"),
        (_set(("verdict", "exit_code"), "3"), "exit_code must be an integer"),
        (_set(("verdict", "exit_code"), 4), "does not pair"),
        (_set(("verdict", "summary"), ""), "verdict.summary"),
        (_set(("findings", 0, "severity"), "fatal"), "severity"),
        (_set(("findings", 0, "id"), "Diagnose OOM"), "findings\\[0\\].id"),
        (_set(("findings", 0, "evidence"), None), "evidence must be a list"),
        (_set(("findings", 0, "evidence", 0, "pointer"), "risk_flags"), "pointer"),
        (_set(("findings", 0, "evidence", 0, "start_ns"), -1), "start_ns"),
        (_set(("findings", 0, "evidence", 0, "url"), "x"), "unknown fields: url"),
        (_set(("findings", 0, "metrics", "num_ooms"), "1"), "number or null"),
        (_set(("metrics", "total_bytes"), "big"), "number or null"),
        (_set(("artifacts", 0, "path"), ""), "artifacts\\[0\\].path"),
        (_set(("artifacts", 0, "schema_version"), 0), "schema_version"),
        (_set(("recommendations",), [""]), "recommendations"),
        (_set(("session_id",), ""), "session_id"),
        (_set(("payload",), []), "payload must be an object"),
    ],
)
def test_malformed_reports_are_rejected_by_schema_and_validator(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    report = _fixture("diagnose_findings.json")
    mutate(report)
    _assert_rejected(report, message)


def test_write_and_load_report_round_trip(tmp_path: Path) -> None:
    report = _fixture("diagnose_findings.json")
    path = tmp_path / "report.json"

    write_report(path, report)

    assert load_report(path) == report
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_write_report_refuses_invalid_report(tmp_path: Path) -> None:
    report = _fixture("diagnose_findings.json")
    report["verdict"]["exit_code"] = 0

    with pytest.raises(ValueError, match="does not pair"):
        write_report(tmp_path / "report.json", report)
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "Cannot read report"),
        ("[]", "is not a JSON object"),
        ('{"schema_version": 1}', "missing required fields"),
    ],
)
def test_load_report_rejects_unreadable_files(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "report.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_report(path)


def test_load_report_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot read report"):
        load_report(tmp_path / "missing.json")
