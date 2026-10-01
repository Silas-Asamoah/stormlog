"""Tests for the diagnose-bundle report builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from stormlog.diagnose_report import (
    build_diagnose_report,
    write_incomplete_bundle,
    write_verdict_report,
)
from stormlog.exit_codes import ExitCode
from stormlog.report import load_report, validate_report

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "schemas"
    / "stormlog_report_v1.schema.json"
)


def _summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "backend": "cuda",
        "allocated_bytes": 900,
        "reserved_bytes": 950,
        "peak_bytes": 900,
        "total_bytes": 1000,
        "allocator_gap_bytes": 50,
        "utilization_ratio": 0.9,
        "fragmentation_ratio": 0.35,
        "num_ooms": 1,
        "risk_flags": {
            "oom_occurred": True,
            "high_utilization": True,
            "fragmentation_warning": True,
        },
        "suggestions": ["Reduce batch size."],
    }
    summary.update(overrides)
    return summary


def _assert_valid(report: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(report)
    validate_report(report)


def test_findings_report_maps_every_raised_flag() -> None:
    report = build_diagnose_report(
        tool_name="gpumemprof",
        summary=_summary(),
        exit_code=ExitCode.FINDINGS,
        session_id="session-1",
        files=["environment.json", "diagnostic_summary.json", "manifest.json"],
        thresholds={"high_utilization": 0.85, "fragmentation_warning": 0.3},
        tool_version="0.4.0",
    )

    _assert_valid(report)
    assert report["report_kind"] == "diagnose"
    assert report["tool"] == {
        "name": "gpumemprof",
        "command": "diagnose",
        "version": "0.4.0",
    }
    assert report["verdict"] == {
        "status": "findings",
        "exit_code": 3,
        "summary": "Memory risk detected: oom, high_utilization, fragmentation",
    }
    assert [finding["id"] for finding in report["findings"]] == [
        "diagnose.oom_occurred",
        "diagnose.high_utilization",
        "diagnose.fragmentation_warning",
    ]
    oom, utilization, fragmentation = report["findings"]
    assert oom["severity"] == "critical"
    assert oom["metrics"] == {"num_ooms": 1}
    assert oom["evidence"] == [
        {
            "kind": "diagnose_summary",
            "path": "diagnostic_summary.json",
            "pointer": "/risk_flags/oom_occurred",
        },
        {"kind": "session", "session_id": "session-1"},
    ]
    assert utilization["metrics"] == {"utilization_ratio": 0.9, "threshold": 0.85}
    assert fragmentation["metrics"] == {"fragmentation_ratio": 0.35, "threshold": 0.3}
    assert report["metrics"]["allocator_gap_bytes"] == 50
    assert report["recommendations"] == ["Reduce batch size."]
    assert report["session_id"] == "session-1"
    assert report["payload"]["risk_flags"]["oom_occurred"] is True
    assert report["payload"]["backend"] == "cuda"
    assert report["artifacts"] == [
        {"kind": "environment", "path": "environment.json"},
        {"kind": "diagnose_summary", "path": "diagnostic_summary.json"},
        {"kind": "diagnose_manifest", "path": "manifest.json", "schema_version": 2},
    ]


def test_pass_report_has_no_findings() -> None:
    summary = _summary(
        num_ooms=0,
        utilization_ratio=0.2,
        fragmentation_ratio=0.0,
        risk_flags={
            "oom_occurred": False,
            "high_utilization": False,
            "fragmentation_warning": False,
        },
        suggestions=[],
    )

    report = build_diagnose_report(
        tool_name="tfmemprof",
        summary=summary,
        exit_code=ExitCode.OK,
        session_id="session-2",
        files=["manifest.json"],
    )

    _assert_valid(report)
    assert report["verdict"] == {
        "status": "pass",
        "exit_code": 0,
        "summary": "No memory risk detected",
    }
    assert report["findings"] == []
    assert report["recommendations"] == []


def test_report_tolerates_partial_summaries() -> None:
    report = build_diagnose_report(
        tool_name="jaxmemprof",
        summary={"risk_flags": {"custom_flag": True}, "total_bytes": None},
        exit_code=ExitCode.FINDINGS,
        session_id="session-3",
        files=["cuda_allocator_snapshot.pickle", "report.json", "manifest.json"],
    )

    _assert_valid(report)
    (finding,) = report["findings"]
    assert finding["id"] == "diagnose.custom_flag"
    assert finding["severity"] == "warning"
    assert finding["metrics"] == {}
    assert report["metrics"] == {"total_bytes": None}
    assert report["artifacts"][:2] == [
        {"kind": "file", "path": "cuda_allocator_snapshot.pickle"},
        {
            "kind": "report",
            "path": "report.json",
            "format": "stormlog.report",
            "schema_version": 1,
        },
    ]
    assert report["payload"]["backend"] is None


def test_incomplete_report_claims_no_findings() -> None:
    report = build_diagnose_report(
        tool_name="gpumemprof",
        summary=_summary(),
        exit_code=ExitCode.ERROR,
        session_id="session-5",
        files=["environment.json", "report.json", "manifest.json"],
        error="disk full",
    )

    _assert_valid(report)
    assert report["verdict"] == {
        "status": "error",
        "exit_code": 1,
        "summary": "Bundle incomplete: disk full",
    }
    assert report["findings"] == []
    assert report["payload"]["risk_flags"]["oom_occurred"] is True
    assert report["metrics"]["num_ooms"] == 1


def test_write_incomplete_bundle_rewrites_report_then_manifest(
    tmp_path: Path,
) -> None:
    write_verdict_report(
        tmp_path,
        tool_name="gpumemprof",
        summary=_summary(),
        exit_code=ExitCode.FINDINGS,
        session_id="session-6",
        files=["report.json", "manifest.json"],
    )
    manifests: list[list[str]] = []

    write_incomplete_bundle(
        tmp_path,
        tool_name="gpumemprof",
        summary=_summary(),
        session_id="session-6",
        files_written=["environment.json", "report.json"],
        error="disk full",
        thresholds=None,
        write_manifest=manifests.append,
    )

    report = load_report(tmp_path / "report.json")
    assert report["verdict"]["exit_code"] == 1
    assert report["verdict"]["summary"] == "Bundle incomplete: disk full"
    assert [item["path"] for item in report["artifacts"]] == [
        "environment.json",
        "report.json",
        "manifest.json",
    ]
    assert manifests == [["environment.json", "report.json", "manifest.json"]]


def test_write_incomplete_bundle_removes_a_stale_report_it_cannot_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "report.json").write_text('{"verdict": "stale"}', encoding="utf-8")
    manifests: list[list[str]] = []

    def _fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full again")

    monkeypatch.setattr("stormlog.diagnose_report.write_verdict_report", _fail)

    write_incomplete_bundle(
        tmp_path,
        tool_name="gpumemprof",
        summary={},
        session_id="session-7",
        files_written=["environment.json", "report.json"],
        error="disk full",
        thresholds=None,
        write_manifest=manifests.append,
    )

    assert not (tmp_path / "report.json").exists()
    assert manifests == [["environment.json", "manifest.json"]]


def test_write_incomplete_bundle_swallows_a_failed_manifest_write(
    tmp_path: Path,
) -> None:
    def _fail(files: list[str]) -> None:
        raise OSError("still full")

    write_incomplete_bundle(
        tmp_path,
        tool_name="gpumemprof",
        summary={},
        session_id="session-8",
        files_written=[],
        error="disk full",
        thresholds=None,
        write_manifest=_fail,
    )

    assert load_report(tmp_path / "report.json")["verdict"]["exit_code"] == 1


def test_report_refuses_a_verdict_that_contradicts_the_summary() -> None:
    with pytest.raises(ValueError, match="not in the contract"):
        build_diagnose_report(
            tool_name="gpumemprof",
            summary=_summary(),
            exit_code=99,
            session_id="session-4",
            files=[],
        )
