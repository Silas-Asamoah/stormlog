"""One run summarized for comparison: labels, fields and protocol failures."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.run_summary import summarize_run, summary_from_records
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


def test_an_unfinished_run_is_an_outcome_kept_as_data(tmp_path: Path) -> None:
    # A crash before the terminal record may be the treatment's doing: it is
    # compared, never set aside, unless an external cause is recorded.
    path = _run(tmp_path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    summary = summarize_run(path)
    assert summary.session_status == "running"
    assert summary.protocol_failures == ()
    assert summary.outcome_failures == ("session_running",)


def test_an_external_cause_the_runner_recorded_sets_the_run_aside(
    tmp_path: Path,
) -> None:
    path = _run(tmp_path)
    lines = path.read_text().splitlines()
    state = {
        "event_type": "infer.run_state",
        "state": "protocol_failure",
        "reasons": ["server_never_healthy"],
        "before_treatment": ["server_never_healthy"],
    }
    path.write_text("\n".join([*lines[:-1], json.dumps(state)]) + "\n")
    summary = summarize_run(path)
    assert summary.protocol_failures == ("external:server_never_healthy",)
    assert summary.outcome_failures == ("session_running",)


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


def _summary(report: dict[str, Any], *extra: dict[str, Any]) -> Any:
    session = {"event_type": "infer.session", "session_id": "s", "status": "completed"}
    return summary_from_records([session, *extra], report)


def test_a_cohort_cut_short_by_an_unfinished_run_is_an_outcome() -> None:
    interrupted = {
        "event_type": "infer.session",
        "session_id": "s",
        "status": "interrupted",
    }
    cut: dict[str, Any] = {
        "c1": {
            "population": {
                "cohort_valid": False,
                "issues": ["phase_window_missing", "records_missing: 24 of 50"],
            }
        }
    }
    summary = summary_from_records([interrupted], {"cases": cut})
    assert summary.failures_for("c1") == ()
    assert summary.outcome_failures == ("session_interrupted",)
    # A duplicate is a fault of the harness, whatever ended the run.
    cut["c1"]["population"]["issues"].append("duplicate_request_id: 1")
    assert summary_from_records([interrupted], {"cases": cut}).failures_for("c1") == (
        "cohort_invalid",
    )


def test_each_protocol_failure_sets_its_run_or_case_aside() -> None:
    cases = {"c1": {"population": {"cohort_valid": False}}}
    assert _summary({"cases": cases}).failures_for("c1") == ("cohort_invalid",)
    changed = _summary({"manifest": {"protocol_failure": "identity_changed"}})
    assert changed.protocol_failures == ("identity_changed",)
    probe = {"event_type": "infer.server_probe", "phase": "before", "incomplete": True}
    assert _summary({}, probe).protocol_failures == ("probe_incomplete",)


def test_an_unfinished_run_keeps_its_run_faults_as_outcomes() -> None:
    # Outcome beats protocol: a treatment that kills the server also cuts
    # the probe short and may restart it with a new identity.
    interrupted = {
        "event_type": "infer.session",
        "session_id": "s",
        "status": "interrupted",
    }
    probe = {"event_type": "infer.server_probe", "phase": "after", "incomplete": True}
    report = {"manifest": {"protocol_failure": "identity_changed"}}
    summary = summary_from_records([interrupted, probe], report)
    assert summary.protocol_failures == ()
    assert summary.outcome_failures == (
        "session_interrupted",
        "identity_changed",
        "probe_incomplete",
    )
    state = {
        "event_type": "infer.run_state",
        "state": "protocol_failure",
        "reasons": ["preempted"],
    }
    external = summary_from_records([interrupted, probe, state], report)
    assert external.protocol_failures == (
        "identity_changed",
        "probe_incomplete",
        "external:preempted",
    )


def test_an_outcome_the_runner_recorded_is_kept_and_outranks_protocol() -> None:
    # The server died during the treatment and came back with a new identity:
    # the runner's precedence already made it an outcome.
    probe = {"event_type": "infer.server_probe", "phase": "after", "incomplete": True}
    state = {
        "event_type": "infer.run_state",
        "state": "outcome_failure",
        "reasons": ["server_exited:-9"],
        "before_treatment": [],
    }
    report = {"manifest": {"protocol_failure": "identity_changed"}}
    summary = _summary(report, probe, state)
    assert summary.protocol_failures == ()
    assert summary.outcome_failures == (
        "runner:server_exited:-9",
        "identity_changed",
        "probe_incomplete",
    )
    completed = {**state, "state": "completed", "reasons": []}
    assert _summary(report, probe, completed).protocol_failures == (
        "identity_changed",
        "probe_incomplete",
    )


def test_an_external_cause_keeps_its_evidence() -> None:
    interrupted = {
        "event_type": "infer.session",
        "session_id": "s",
        "status": "interrupted",
    }
    state = {
        "event_type": "infer.run_state",
        "state": "protocol_failure",
        "reasons": ["spot_preemption"],
        "external_cause": {
            "reason": "spot_preemption",
            "evidence": "box paused without a release",
        },
    }
    summary = summary_from_records([interrupted, state], {})
    assert summary.protocol_failures == ("external:spot_preemption",)
    assert summary.external_evidence == {
        "external:spot_preemption": "box paused without a release"
    }
