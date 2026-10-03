"""SLO policies through `infer profile` and `infer analyze`."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.analysis import format_analysis_text
from stormlog.infer.cli import main as infer_main
from stormlog.infer.slo import parse_slo_flags
from tests.infer_workload_helpers import run_profile_with_fake_client

SECOND = 1_000_000_000


def _request(index: int, e2e_ms: float, status: str = "ok") -> dict[str, Any]:
    return {
        "event_type": "infer.request",
        "phase": "measured",
        "session_id": "s1",
        "case_id": "c1",
        "request_id": f"r{index}",
        "x_request_id": f"x{index}",
        "request_index": index,
        "status": status,
        "started_at_ns": index * SECOND,
        "ended_at_ns": index * SECOND + int(e2e_ms * 1e6),
        "e2e_latency_ms": e2e_ms,
    }


def _artifact(tmp_path: Path, *extra: dict[str, Any]) -> Path:
    path = tmp_path / "infer.jsonl"
    records = [
        {"event_type": "infer.session", "session_id": "s1"},
        _request(0, 100.0),
        _request(1, 300.0),
        _request(2, 150.0),
        _request(3, 50.0, status="error"),
        {
            "event_type": "infer.phase_window",
            "phase": "measured",
            "session_id": "s1",
            "case_id": "c1",
            "arrival_mode": "closed",
            "started_at_ns": 0,
            "window_ended_at_ns": 4 * SECOND,
            "drained_at_ns": 4 * SECOND,
        },
        *extra,
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _analyze(*argv: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = infer_main(["analyze", *argv])
    return code, stdout.getvalue(), stderr.getvalue()


def test_analyze_judges_cases_against_slo_flags(tmp_path: Path) -> None:
    code, out, _err = _analyze(
        str(_artifact(tmp_path)), "--format", "json", "--slo", "e2e:200"
    )
    report = json.loads(out)

    assert code == ExitCode.OK
    assert report["slo"]["source"] == "flags"
    assert report["slo"]["policy"]["criteria"] == [
        {"metric": "e2e", "boundary": "client", "max_ms": 200.0}
    ]
    slo = report["cases"]["c1"]["slo"]
    assert (slo["offered"], slo["met"], slo["missed"]) == (4, 2, 2)
    assert slo["attainment_lower"] == 0.5
    assert slo["goodput_lower_rps"] == pytest.approx(0.5)  # 2 per 4 s
    assert "SLO cli: attainment 50.0% of 4 offered" in format_analysis_text(report)


def test_analyze_reads_a_policy_file(tmp_path: Path) -> None:
    policy = tmp_path / "slo.json"
    policy.write_text(
        json.dumps(parse_slo_flags(["e2e:120"], name="tight").to_record())
    )
    code, out, _err = _analyze(
        str(_artifact(tmp_path)), "--format", "json", "--slo-file", str(policy)
    )
    report = json.loads(out)
    assert code == ExitCode.OK
    assert (report["slo"]["name"], report["slo"]["source"]) == ("tight", "file")
    assert report["cases"]["c1"]["slo"]["met"] == 1


def test_analyze_uses_the_artifacts_policy_unless_told_otherwise(
    tmp_path: Path,
) -> None:
    spec = parse_slo_flags(["e2e:120"], name="recorded")
    recorded = {
        "event_type": "infer.slo",
        "session_id": "s1",
        "source": "file",
        "digest": spec.digest(),
        "slo": spec.to_record(),
    }
    path = _artifact(tmp_path, recorded)

    _code, out, _err = _analyze(str(path), "--format", "json")
    report = json.loads(out)
    assert (report["slo"]["name"], report["slo"]["source"]) == ("recorded", "artifact")
    assert report["cases"]["c1"]["slo"]["met"] == 1

    _code, out, _err = _analyze(str(path), "--format", "json", "--slo", "e2e:400")
    overridden = json.loads(out)
    assert overridden["slo"]["source"] == "flags"
    assert overridden["cases"]["c1"]["slo"]["met"] == 3


def test_without_a_policy_there_is_no_slo_block(tmp_path: Path) -> None:
    _code, out, _err = _analyze(str(_artifact(tmp_path)), "--format", "json")
    report = json.loads(out)
    assert report["slo"] is None
    assert "slo" not in report["cases"]["c1"]


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (["--slo", "itl:50"], ExitCode.USAGE, "chunks are not tokens"),
        (["--slo", "e2e:fast"], ExitCode.USAGE, "is not a number"),
        (["--slo", "e2e:100", "--slo-file", "slo.json"], ExitCode.USAGE, "not both"),
    ],
)
def test_bad_slo_flags_are_usage_errors(
    tmp_path: Path, argv: list[str], code: int, message: str
) -> None:
    status, _out, err = _analyze(str(_artifact(tmp_path)), *argv)
    assert status == code
    assert message in err


def test_an_unusable_policy_file_is_invalid_input(tmp_path: Path) -> None:
    bad = tmp_path / "slo.json"
    bad.write_text('{"format": "stormlog.infer.slo", "version": 9}')
    status, _out, err = _analyze(str(_artifact(tmp_path)), "--slo-file", str(bad))
    assert status == ExitCode.INVALID_INPUT
    assert "SLO policy" in err


def test_profile_records_its_policy_and_judges_the_run_by_it(tmp_path: Path) -> None:
    spec = parse_slo_flags(["e2e:60000"], name="loose")
    _requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.0, request_count=3, slo=spec, slo_source="flags"
    )
    records = [
        json.loads(line) for line in (tmp_path / "infer.jsonl").read_text().splitlines()
    ]
    (recorded,) = [r for r in records if r.get("event_type") == "infer.slo"]

    assert recorded["digest"] == spec.digest()
    assert recorded["source"] == "flags"
    assert report["slo"]["source"] == "artifact"
    (case,) = report["cases"].values()
    assert (case["slo"]["offered"], case["slo"]["met"]) == (3, 3)


@pytest.mark.parametrize(
    ("flags", "code"),
    [
        (["--slo", "client.itl:50"], ExitCode.USAGE),
        (["--slo", "e2e:100", "--slo-file", "slo.json"], ExitCode.USAGE),
        (["--slo-file", "missing-slo.json"], ExitCode.INVALID_INPUT),
    ],
)
def test_profile_refuses_a_policy_it_cannot_use_before_sending(
    tmp_path: Path, flags: list[str], code: int
) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        status = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "fake-model",
                "--system-sampler",
                "none",
                "--tokenizer",
                "none",
                "--output",
                str(tmp_path / "infer.jsonl"),
                *flags,
            ]
        )
    assert status == code
    assert not (tmp_path / "infer.jsonl").exists()


def test_an_invalid_recorded_policy_is_invalid_input_unless_overridden(
    tmp_path: Path,
) -> None:
    broken = {"event_type": "infer.slo", "session_id": "s1", "slo": {"version": 9}}
    path = _artifact(tmp_path, broken)

    status, _out, err = _analyze(str(path))
    assert status == ExitCode.INVALID_INPUT
    assert "infer.slo" in err
    status, _out, _err = _analyze(str(path), "--slo", "e2e:200")
    assert status == ExitCode.OK
