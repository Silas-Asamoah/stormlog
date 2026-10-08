"""``stormlog infer diagnose``: the report, the text view and --inspect."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer import diagnosis_model
from stormlog.infer.cli import main
from tests.diagnosis_scenarios import MS, Engine, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND


@pytest.fixture(scope="module")
def burst(tmp_path_factory: pytest.TempPathFactory) -> Path:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    heavy = poisson_free(300, 90 * SECOND, 5 * MS, prefix="b")
    return build_run(
        tmp_path_factory.mktemp("cli"), calm + heavy, Engine(max_num_seqs=4)
    )


def _diagnosed(
    tmp_path: Path, artifact: Path, capsys: pytest.CaptureFixture[str], *flags: str
) -> tuple[int, dict[str, Any], str]:
    copy = tmp_path / "infer.jsonl"
    shutil.copy(artifact, copy)
    report = tmp_path / "report.json"
    code = main(["diagnose", str(copy), "--output", str(report), *flags])
    return code, json.loads(report.read_text()), capsys.readouterr().out


def test_diagnose_writes_the_report_and_prints_the_findings(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, report, text = _diagnosed(tmp_path, burst, capsys)

    assert code == 3 and report["verdict"]["exit_code"] == 3
    assert "WARNING queue_saturation at scheduler" in text
    assert "competitor engine_stall (indispensable): ruled_out" in text
    assert "queue_saturation             assessed" in text
    # Evidence paths are relative to the report's directory.
    assert report["findings"][0]["evidence"][0]["path"] == "infer.jsonl"


def test_json_format_prints_the_same_report(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, report, out = _diagnosed(tmp_path, burst, capsys, "--format", "json")

    assert json.loads(out)["findings"] == report["findings"]


def test_inspect_prints_a_finding_s_records_by_line(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, report, _ = _diagnosed(tmp_path, burst, capsys)
    finding = report["findings"][0]["id"]
    total = report["payload"]["findings_detail"][finding]["evidence_total"]

    assert main(["diagnose", "--inspect", str(tmp_path / "report.json"), finding]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert (
        main(["diagnose", "--inspect", str(tmp_path / "report.json"), finding, "--all"])
        == 0
    )
    everything = capsys.readouterr().out.splitlines()

    assert len(shown) == 8 and len(everything) == total
    line, _, rest = shown[0].partition(" ")
    assert line.rsplit(":", 1)[1].isdigit()
    assert json.loads(rest.partition(" ")[2])["event_type"] in (
        "infer.request",
        "infer.iteration",
    )


def test_inspect_resolves_a_report_moved_with_its_artifact(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, report, _ = _diagnosed(tmp_path, burst, capsys)
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("report.json", "infer.jsonl"):
        shutil.move(tmp_path / name, moved / name)

    code = main(
        [
            "diagnose",
            "--inspect",
            str(moved / "report.json"),
            report["findings"][0]["id"],
        ]
    )

    assert code == 0 and str(moved / "infer.jsonl") in capsys.readouterr().out


def test_inspect_without_the_source_is_invalid_input(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, report, _ = _diagnosed(tmp_path, burst, capsys)
    (tmp_path / "infer.jsonl").unlink()

    code = main(
        [
            "diagnose",
            "--inspect",
            str(tmp_path / "report.json"),
            report["findings"][0]["id"],
        ]
    )

    assert code == 5 and "not found" in capsys.readouterr().err


def test_inspect_finds_records_again_after_an_append(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, report, _ = _diagnosed(tmp_path, burst, capsys)
    artifact = tmp_path / "infer.jsonl"
    lines = artifact.read_text().splitlines()
    # A record inserted near the top moves every line: IDs still find them.
    artifact.write_text("\n".join([lines[0], "", *lines[1:]]) + "\n")
    finding = report["findings"][0]["id"]

    code = main(
        ["diagnose", "--inspect", str(tmp_path / "report.json"), finding, "--all"]
    )

    out = capsys.readouterr().out
    assert code == 0 and out.startswith("warning:")
    assert ":1169 infer.request/b299" in out  # one line further down


def test_a_changed_supporting_record_is_not_resolved(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, report, _ = _diagnosed(tmp_path, burst, capsys)
    artifact = tmp_path / "infer.jsonl"
    artifact.write_text(
        artifact.read_text().replace(
            '"request_id": "b299"', '"request_id": "b299", "x": 1'
        )
    )

    code = main(
        [
            "diagnose",
            "--inspect",
            str(tmp_path / "report.json"),
            report["findings"][0]["id"],
        ]
    )

    assert code == 5 and "infer.request/b299" in capsys.readouterr().err


def test_support_kept_as_ranges_cannot_follow_a_changed_source(
    tmp_path: Path,
    burst: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnosis_model, "SUPPORT_LIMIT", 10)
    _, report, _ = _diagnosed(tmp_path, burst, capsys)
    artifact = tmp_path / "infer.jsonl"
    artifact.write_text(artifact.read_text() + "\n")
    finding = report["findings"][0]["id"]

    code = main(
        ["diagnose", "--inspect", str(tmp_path / "report.json"), finding, "--all"]
    )

    assert code == 5
    assert "support_unresolvable_after_modification" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, code",
    [
        (["diagnose"], 2),
        (["diagnose", "x.jsonl", "--window", "5"], 2),
        (["diagnose", "{artifact}", "--window", "9,5"], 2),
        (["diagnose", "missing.jsonl"], 5),
    ],
)
def test_bad_invocations_have_their_codes(
    burst: Path, argv: list[str], code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([a.format(artifact=burst) for a in argv]) == code


def test_a_nan_threshold_is_a_usage_error_not_a_silent_pass(
    tmp_path: Path, burst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # json.loads reads NaN; never exceeded, it would hide the queue warning.
    overrides = tmp_path / "thresholds.json"
    overrides.write_text('{"queue_saturation.witness_step_share": NaN}')

    code = main(["diagnose", str(burst), "--thresholds", str(overrides)])

    assert code == 2 and "finite" in capsys.readouterr().err
