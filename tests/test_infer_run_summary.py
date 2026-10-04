"""One run summarized for comparison: labels, fields and protocol failures."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.run_summary import summarize_run
from tests.infer_workload_helpers import run_profile_with_fake_client

LABELS = {
    "experiment": "e1",
    "arm": "baseline",
    "block": "3",
    "position": 1,
    "attempt": 1,
}


def _run(tmp_path: Path, **changes: Any) -> Path:
    run_profile_with_fake_client(tmp_path, latency_seconds=0.0, **changes)
    return tmp_path / "infer.jsonl"


def test_a_completed_run_is_summarized_with_its_labels(tmp_path: Path) -> None:
    summary = summarize_run(_run(tmp_path, labels=LABELS))

    assert summary.session_status == "completed"
    assert summary.labels == LABELS
    assert summary.label("block") == "3"
    assert summary.protocol_failures == ()
    assert summary.run_id and summary.session_id
    assert summary.sha256 is not None and len(summary.sha256) == 64
    (case_id,) = summary.cases
    assert summary.failures_for(case_id) == ()
    assert summary.fields["workload.spec_digest"].known


def test_an_unfinished_run_is_a_protocol_failure(tmp_path: Path) -> None:
    path = _run(tmp_path)
    lines = path.read_text().splitlines()
    # Drop the terminal session record, as a crash before it would.
    path.write_text("\n".join(lines[:-1]) + "\n")
    summary = summarize_run(path)
    assert summary.session_status == "running"
    assert summary.protocol_failures == ("session_running",)


def test_a_cold_case_whose_reset_failed_cannot_stand_for_its_case(
    tmp_path: Path,
) -> None:
    summary = summarize_run(_run(tmp_path, cache_state="cold"))
    (case_id,) = summary.cases
    assert summary.failures_for(case_id) == ("cache_reset_not_acknowledged",)


def test_labels_are_given_on_the_command_line(tmp_path: Path) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "m",
                "--system-sampler",
                "none",
                "--tokenizer",
                "none",
                "--server-probe",
                "none",
                "--block",
                "3",
                "--output",
                str(tmp_path / "infer.jsonl"),
            ]
        )
    assert code == ExitCode.USAGE
    assert "--block needs --experiment and --arm" in stderr.getvalue()


def test_a_labelled_profile_records_its_labels(tmp_path: Path) -> None:
    path = _run(tmp_path, labels=LABELS)
    session = json.loads(path.read_text().splitlines()[0])
    assert session["config"]["labels"] == LABELS
