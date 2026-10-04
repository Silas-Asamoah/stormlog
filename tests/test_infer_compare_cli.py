"""`stormlog infer compare`: the envelope, its pointers and its exit codes."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.report import load_report, validate_report
from tests.infer_workload_helpers import run_profile_with_fake_client


def _runs(tmp_path: Path, arm: str, latency: float, blocks: int = 3) -> list[str]:
    paths = []
    for block in range(blocks):
        directory = tmp_path / f"{arm}{block}"
        directory.mkdir()
        run_profile_with_fake_client(
            directory,
            latency_seconds=latency,
            request_count=6,
            labels={"experiment": "e", "arm": arm, "block": str(block)},
        )
        paths.append(str(directory / "infer.jsonl"))
    return paths


@pytest.fixture(scope="module")
def arms(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    root = tmp_path_factory.mktemp("arms")
    return {
        "baseline": _runs(root, "baseline", 0.01),
        "slower": _runs(root, "slower", 0.05),
    }


def _compare(*argv: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = infer_main(["compare", *argv])
    return code, stdout.getvalue(), stderr.getvalue()


def _resolve(document: Any, pointer: str) -> Any:
    """RFC 6901, as a reader of a pathless pointer resolves it in the report."""
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        document = document[int(key)] if isinstance(document, list) else document[key]
    return document


def test_a_comparison_without_gates_reports_and_exits_0(
    arms: dict[str, list[str]]
) -> None:
    code, out, _err = _compare(
        "--baseline", *arms["baseline"], "--candidate", *arms["baseline"]
    )
    assert code == ExitCode.OK
    assert "Design: paired_blocks" in out
    assert "client.e2e.p95" in out


def test_the_json_envelope_is_a_valid_report_with_the_payload(
    arms: dict[str, list[str]],
) -> None:
    code, out, _err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--gate",
        "client.e2e.p95=non-inferiority:0.05",
        "--allow-not-evaluable",
        "--format",
        "json",
    )
    report = json.loads(out)
    validate_report(report)
    assert report["report_kind"] == "inference_comparison"
    payload = report["payload"]
    assert (payload["format"], payload["version"]) == ("stormlog.infer.comparison", 1)
    assert report["verdict"]["exit_code"] == code


def test_a_gate_that_cannot_be_evaluated_exits_4_and_points_at_its_metric(
    arms: dict[str, list[str]],
) -> None:
    # These runs carry no server description, so their comparability is unverified.
    code, out, _err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--gate",
        "client.e2e.p95=significant:0.05",
        "--format",
        "json",
    )
    report = json.loads(out)
    assert code == ExitCode.GATE_FAILED == report["verdict"]["exit_code"]
    finding = next(f for f in report["findings"] if f["kind"] == "not_evaluable")
    assert finding["message"] == "unverified"
    (evidence,) = finding["evidence"]
    # On stdout the pointer has no path: it resolves inside this report.
    assert "path" not in evidence
    metric = _resolve(report, evidence["pointer"])
    assert metric["name"] == "client.e2e.p95"
    assert metric["gate"]["status"] == "not_evaluable"


SLO_MISSED = ("--slo", "e2e:30")


def test_a_candidate_that_meets_no_slo_is_a_regression_finding(
    arms: dict[str, list[str]], tmp_path: Path
) -> None:
    path = tmp_path / "comparison.json"
    code, out, _err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        *SLO_MISSED,
        "--gate",
        "goodput_rps=non-inferiority:0.05",
        "--format",
        "json",
        "--report",
        str(path),
    )
    printed = json.loads(out)
    regression = next(f for f in printed["findings"] if f["kind"] == "regression")
    assert code == ExitCode.GATE_FAILED
    assert regression["message"] == "candidate_zero"
    (evidence,) = regression["evidence"]
    assert "path" not in evidence
    assert _resolve(printed, evidence["pointer"])["reason"] == "candidate_zero"

    # A written report names its own file, and the pointer resolves there.
    written = load_report(path)
    on_file = next(f for f in written["findings"] if f["kind"] == "regression")
    (evidence,) = on_file["evidence"]
    assert evidence["path"] == "comparison.json"
    target = json.loads((tmp_path / evidence["path"]).read_text())
    assert _resolve(target, evidence["pointer"])["name"] == "goodput_rps"


def test_runs_that_cannot_be_compared_exit_5_and_still_leave_a_report(
    arms: dict[str, list[str]], tmp_path: Path
) -> None:
    path = tmp_path / "comparison.json"
    code, _out, err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        str(tmp_path / "missing.jsonl"),
        "--report",
        str(path),
    )
    assert code == ExitCode.INVALID_INPUT
    assert "not found" in err
    report = load_report(path)
    assert report["verdict"]["exit_code"] == ExitCode.INVALID_INPUT
    assert report["findings"][0]["id"] == "inference.comparison.invalid_input"


def test_an_unreadable_slo_file_exits_5_and_still_leaves_a_report(
    arms: dict[str, list[str]], tmp_path: Path
) -> None:
    path = tmp_path / "comparison.json"
    code, _out, err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--slo-file",
        str(tmp_path / "missing-slo.json"),
        "--report",
        str(path),
    )
    assert code == ExitCode.INVALID_INPUT
    assert "SLO policy" in err
    assert load_report(path)["verdict"]["exit_code"] == ExitCode.INVALID_INPUT


def test_allowed_gates_that_could_not_be_evaluated_are_not_called_passed(
    arms: dict[str, list[str]]
) -> None:
    code, out, _err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--gate",
        "client.e2e.p95=non-inferiority:0.05",
        "--allow-not-evaluable",
        "--format",
        "json",
    )
    summary = json.loads(out)["verdict"]["summary"]
    assert code == ExitCode.OK
    assert not summary.startswith("every gate passed")
    assert "could not be evaluated (allowed)" in summary


def test_a_fallback_that_names_no_gate_is_a_usage_error(
    arms: dict[str, list[str]]
) -> None:
    # Fallbacks belong to a gate by its METRIC text; one under another
    # name would be ignored without a word.
    code, _out, err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--gate",
        "goodput*=non-inferiority:0.05",
        "--fallback",
        "goodput_rps=0.5:requests_per_second",
    )
    assert code == ExitCode.USAGE
    assert "names no --gate" in err


def test_two_runs_of_one_arm_in_a_block_are_invalid_input(
    arms: dict[str, list[str]],
) -> None:
    code, _out, err = _compare(
        "--baseline",
        arms["baseline"][0],
        arms["baseline"][0],
        "--candidate",
        *arms["slower"],
    )
    assert code == ExitCode.INVALID_INPUT
    assert "two baseline runs" in err


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ("client.e2e.p95", "METRIC=RULE:BUDGET"),
        ("nothing.here=non-inferiority:0.05", "names no metric"),
        ("client.e2e.p95=sometimes:0.05", "gate rule must be one of"),
        ("client.e2e.p95=significant:lots", "is not a number"),
        # Budgets that can never fail: a fraction above 1, and a fall of
        # 100% or more in a rate, which cannot fall below zero.
        ("attainment=non-inferiority:1.5", "can never fail"),
        ("goodput_rps=non-inferiority:1", "can never fail"),
        ("throughput_rps=significant:5", "can never fail"),
    ],
)
def test_a_gate_it_cannot_read_is_a_usage_error(
    arms: dict[str, list[str]], gate: str, message: str
) -> None:
    code, _out, err = _compare(
        "--baseline", *arms["baseline"], "--candidate", *arms["slower"], "--gate", gate
    )
    assert code == ExitCode.USAGE
    assert message in err


@pytest.mark.parametrize(
    "flags",
    [
        ["--min-attainment", "0.99", "--min-run-pass", "0"],
        ["--min-attainment", "0.99", "--min-run-pass", "1.5"],
        ["--min-attainment", "0"],
        ["--min-attainment", "1.5"],
        ["--evidence-floor", "-1"],
        ["--evidence-floor", "2"],
    ],
)
def test_shares_outside_their_range_are_usage_errors(
    arms: dict[str, list[str]], flags: list[str]
) -> None:
    code, _out, err = _compare(
        "--baseline", *arms["baseline"], "--candidate", *arms["slower"], *flags
    )
    assert code == ExitCode.USAGE
    assert "is not valid" in err


def test_a_gate_on_a_metric_the_runs_lack_points_at_its_absence(
    arms: dict[str, list[str]]
) -> None:
    code, out, _err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--gate",
        "server.e2e.p95=non-inferiority:0.05",
        "--format",
        "json",
    )
    report = json.loads(out)
    assert code == ExitCode.GATE_FAILED
    absent = [f for f in report["findings"] if f["kind"] == "not_evaluable"]
    pointers = [e["pointer"] for f in absent for e in f["evidence"]]
    assert any("/absent_gates/server.e2e.p95" in pointer for pointer in pointers)
    for pointer in pointers:
        _resolve(report, pointer)


def test_segments_are_compared_as_cases_of_their_own(
    arms: dict[str, list[str]],
) -> None:
    code, out, _err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--segment",
        "early=0:0.05",
        "--segment",
        "whole=0:60",
        "--format",
        "json",
    )
    assert code == ExitCode.OK
    cases = json.loads(out)["payload"]["cases"]
    (case_id,) = [c for c in cases if "/" not in c]
    assert {f"{case_id}/early", f"{case_id}/whole"} <= set(cases)
    whole = cases[f"{case_id}/whole"]["metrics"]
    assert whole["throughput_rps"]["n_pairs"] == 3
    assert "client.e2e.p95" in whole


@pytest.mark.parametrize(
    ("segment", "message"),
    [("early", "NAME=START:END"), ("early=5:1", "offsets must satisfy")],
)
def test_a_segment_it_cannot_read_is_a_usage_error(
    arms: dict[str, list[str]], segment: str, message: str
) -> None:
    code, _out, err = _compare(
        "--baseline",
        *arms["baseline"],
        "--candidate",
        *arms["slower"],
        "--segment",
        segment,
    )
    assert code == ExitCode.USAGE
    assert message in err
