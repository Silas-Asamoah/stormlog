"""The experiment runner, end to end against a stand-in vLLM server."""

from __future__ import annotations

import json
import platform
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.compare import ComparisonSpec, compare_runs
from stormlog.infer.comparison_stats import GateRule
from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.experiment import Environment, ExternalCause, run_plan
from stormlog.infer.experiment_plan import plan_from_document
from stormlog.infer.run_summary import summarize_run

FAKE = str(Path(__file__).with_name("fake_vllm_server.py"))
# The same server, with the command line of `vllm serve` (see its docstring).
VLLM = str(Path(__file__).with_name("fake_vllm") / "vllm")
PROFILE = [
    "{python}",
    "-m",
    "stormlog",
    "infer",
    "profile",
    "--base-url",
    "{base_url}",
    "--model",
    "m",
    "--requests",
    "3",
    "--server-probe",
    "basic",
    "--system-sampler",
    "none",
    "--tokenizer",
    "none",
    "--seed",
    "{block_seed}",
    "--experiment",
    "{experiment_id}",
    "--arm",
    "{arm}",
    "--block",
    "{block}",
    "--output",
    "{run_dir}/c1.jsonl",
]
READY = "import pathlib, sys, time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"


@pytest.fixture(autouse=True)
def _set_aside_macos_services() -> Iterator[None]:
    """On macOS, launchd keeps starting its own services (Spotlight workers,
    Metal compilers) as the user, with environments psutil cannot read and
    init as their parent. The runner rightly cannot clear them, so a run
    beside one stops at random. These end-to-end tests set aside system
    executables; the rule itself is tested unchanged in
    test_infer_experiment_process, and on Linux these tests run it as is.
    Its own MonkeyPatch, so a test's monkeypatch.undo() keeps it."""
    if platform.system() == "Linux":
        yield
        return
    import psutil

    from stormlog.infer import experiment_process

    real = experiment_process._may_be_launched

    def may_be(pid: int, since: Any, proc: Path, method: str, **kwargs: Any) -> bool:
        if not real(pid, since, proc, method, **kwargs):
            return False
        try:
            exe = psutil.Process(pid).exe()
        except (psutil.Error, OSError):
            return True
        return not exe.startswith(("/System/", "/usr/libexec/", "/usr/sbin/"))

    patch = pytest.MonkeyPatch()
    patch.setattr(experiment_process, "_may_be_launched", may_be)
    yield
    patch.undo()


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _plan(port: int, **changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": "stormlog.infer.experiment_plan",
        "version": 1,
        "experiment_id": "t213",
        "seed": 7,
        "blocks": 2,
        "order": {"kind": "williams"},
        "server": {
            "command": ["{python}", VLLM, "serve", "m", "--port", str(port)],
            "base_url": f"http://127.0.0.1:{port}/v1",
            "start_timeout_s": 20,
            "stop_timeout_s": 5,
        },
        "arms": {
            "off": {
                "workload": [
                    {
                        "name": "c1",
                        "command": PROFILE,
                        "timeout_s": 120,
                        "expect_exit": [0, 3],
                        "artifacts": ["{run_dir}/c1.jsonl"],
                    }
                ]
            },
            "watch": {
                "workload": "same_as:off",
                "treatments": [
                    {
                        "name": "watcher",
                        "command": ["{python}", "-c", READY, "{run_dir}/watch.ready"],
                        "ready_file": "{run_dir}/watch.ready",
                        "stop_signal": "SIGTERM",
                        "expect_exit": [0, -15],
                    }
                ],
            },
        },
    }
    document.update(changes)
    return document


def _run(
    tmp_path: Path, document: dict[str, Any], **options: Any
) -> list[dict[str, Any]]:
    plan = plan_from_document(document)
    return run_plan(
        plan,
        tmp_path / "exp",
        environment=Environment(python=sys.executable),
        **options,
    )


def test_the_stand_in_server_passes_the_role_check_as_vllms_api_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fable-213's lens a: launched as `python fake_vllm_server.py`, the
    # stand-in was `other` to the role check, so on Linux every end-to-end
    # run ended unexpected_server_children (off Linux the check does not
    # run). Its tree, as Linux /proc shows it, holds only vLLM's roles.
    from stormlog.infer import experiment_process
    from tests.infer_proc_helpers import fake_process

    command = tuple(
        sys.executable if part == "{python}" else part
        for part in _plan(8000)["server"]["command"]
    )
    proc = tmp_path / "proc"
    fake_process(proc, 100, comm=Path(sys.executable).name[:15], cmdline=command)
    monkeypatch.setattr(experiment_process, "_linux", lambda: True)
    assert experiment_process.unexpected_roles(100, proc=proc) == []


def test_a_plan_runs_every_arm_of_every_block_into_comparable_runs(
    tmp_path: Path,
) -> None:
    records = _run(tmp_path, _plan(_port()))
    exp = tmp_path / "exp"

    assert [(r["block"], r["arm"]) for r in records] == [
        (0, "off"),
        (0, "watch"),
        (1, "watch"),
        (1, "off"),
    ]
    assert {r["state"] for r in records} == {"completed"}, records
    index = [
        json.loads(line) for line in (exp / "index.jsonl").read_text().splitlines()
    ]
    assert len(index) == 4
    order = json.loads((exp / "order.json").read_text())
    assert order["balanced"] is True
    run_dir = Path(records[0]["run_dir"])
    assert (run_dir / "SHA256SUMS").exists() and (run_dir / "commands.sh").exists()
    assert not list((exp / "runs").glob("*.partial"))
    assert records[0]["cleanup"]["verified"] is True

    def arm(name: str) -> list[Any]:
        return [
            summarize_run(Path(r["run_dir"]) / "c1.jsonl")
            for r in records
            if r["arm"] == name
        ]

    comparison = compare_runs(
        arm("off"), arm("watch"), ComparisonSpec(allow_not_evaluable=True)
    )
    assert comparison.design == "paired_blocks"
    # The workload probes only the basic routes; the configuration comes
    # from the probe the runner took before measuring, on every run.
    for summary in [*arm("off"), *arm("watch")]:
        assert summary.fields["scope.vllm_config"].value is True
    unverified = {item.name for item in comparison.comparability.unverified}
    assert "vllm_config" not in unverified


def test_a_candidate_whose_server_crashes_fails_its_gate_end_to_end(
    tmp_path: Path,
) -> None:
    # Runner, summarizer and gate: a crashed run is an outcome the
    # comparison counts against the candidate, never a run it sets aside.
    document = _plan(_port(), blocks=3, order={"kind": "random"})
    document["arms"] = {
        "off": document["arms"]["off"],
        "crash": {"server": {"args": ["--die-after", "1"]}, "workload": "same_as:off"},
    }
    records = _run(tmp_path, document)
    crashed = [r for r in records if r["arm"] == "crash"]
    assert {r["state"] for r in crashed} == {"outcome_failure"}, crashed

    def arm(name: str) -> list[Any]:
        return [
            summarize_run(Path(r["run_dir"]) / "c1.jsonl")
            for r in records
            if r["arm"] == name
        ]

    candidate = arm("crash")
    for run in candidate:
        assert run.protocol_failures == ()
        assert "runner:server_exited:1" in run.outcome_failures
    gate = GateRule("non-inferiority", 0.05, "relative")
    comparison = compare_runs(
        arm("off"), candidate, ComparisonSpec(gates=(("client.e2e.p95", gate),))
    )
    assert comparison.excluded == []
    (case,) = comparison.cases.values()
    outcome = case["metrics"]["client.e2e.p95"].gate
    assert outcome is not None and outcome.status == "fail", outcome
    assert comparison.exit_code == 4


def test_a_resumed_experiment_skips_finished_runs_and_refuses_a_changed_plan(
    tmp_path: Path,
) -> None:
    port = _port()
    document = _plan(port, blocks=1)
    _run(tmp_path, document)
    assert _run(tmp_path, document, resume=True) == []
    with pytest.raises(InferInputError, match="already holds an experiment"):
        _run(tmp_path, document)
    with pytest.raises(InferInputError, match="plan changed"):
        _run(tmp_path, _plan(port, blocks=1, seed=8), resume=True)


def _interrupt(exp: Path, label: str) -> None:
    """Leave a finished attempt as a killed runner would: still ``.partial``,
    with no state, no digests and no line in the index."""
    run_dir = exp / "runs" / label
    for name in ("run.json", "SHA256SUMS"):
        (run_dir / name).unlink()
    artifact = run_dir / "c1.jsonl"
    lines = [
        line
        for line in artifact.read_text().splitlines()
        if json.loads(line).get("event_type") != "infer.run_state"
    ]
    artifact.write_text("\n".join(lines) + "\n")
    run_dir.rename(run_dir.with_name(label + ".partial"))
    index = exp / "index.jsonl"
    kept = [
        line
        for line in index.read_text().splitlines()
        if json.loads(line)["label"] != label
    ]
    index.write_text("\n".join(kept) + "\n")


PREEMPTED = ExternalCause("spot_preemption", "box paused without a release at 03:12")


def test_a_preempted_attempt_is_set_aside_with_its_evidence_and_retried(
    tmp_path: Path,
) -> None:
    document = _plan(_port(), blocks=1)
    records = _run(tmp_path, document)
    exp = tmp_path / "exp"
    label = next(r["label"] for r in records if r["arm"] == "watch")
    _interrupt(exp, label)
    first, second = _run(
        tmp_path,
        document,
        resume=True,
        retry_incomplete=True,
        external_causes={label: PREEMPTED},
    )
    assert (first["label"], first["state"], first["reasons"]) == (
        label,
        "protocol_failure",
        ["spot_preemption"],
    )
    assert first["interrupted"] is True
    assert first["external_cause"] == {
        "reason": "spot_preemption",
        "evidence": "box paused without a release at 03:12",
    }
    assert (second["attempt"], second["state"]) == (2, "completed")
    # The interrupted attempt keeps the slot it ran in; the retry is the
    # block's third start.
    planned = next(r["position_planned"] for r in records if r["label"] == label)
    assert first["position_actual"] == planned
    assert (second["position_planned"], second["position_actual"]) == (planned, 2)
    index = [
        json.loads(line) for line in (exp / "index.jsonl").read_text().splitlines()
    ]
    assert [r["label"] for r in index].count(label) == 1

    preempted = summarize_run(Path(first["run_dir"]) / "c1.jsonl")
    assert preempted.protocol_failures == ("external:spot_preemption",)
    retried = summarize_run(Path(second["run_dir"]) / "c1.jsonl")
    off = [
        summarize_run(Path(r["run_dir"]) / "c1.jsonl")
        for r in records
        if r["arm"] == "off"
    ]
    comparison = compare_runs(
        off, [preempted, retried], ComparisonSpec(allow_not_evaluable=True)
    )
    assert [
        (item["run"], item["reasons"], item["evidence"], item["attempt_kept"])
        for item in comparison.excluded
    ] == [
        (
            preempted.name,
            ["superseded", "external:spot_preemption"],
            {"external:spot_preemption": "box paused without a release at 03:12"},
            retried.name,
        )
    ]


def test_a_resume_refuses_an_interrupted_attempt_no_one_has_explained(
    tmp_path: Path,
) -> None:
    # Turning an interruption into a permanent outcome failure, or setting
    # it aside, is a decision: a forgotten flag after a preemption must not
    # make it.
    document = _plan(_port(), blocks=1)
    records = _run(tmp_path, document)
    exp = tmp_path / "exp"
    label = next(r["label"] for r in records if r["arm"] == "watch")
    _interrupt(exp, label)
    with pytest.raises(InferUsageError, match=f"{label} was interrupted"):
        _run(tmp_path, document, resume=True, retry_incomplete=True)
    with pytest.raises(InferUsageError, match="both an external cause and"):
        _run(
            tmp_path,
            document,
            resume=True,
            external_causes={label: PREEMPTED},
            interrupted_as_outcome={label},
        )
    # Refused before anything changed: the attempt is still as it was left.
    assert (exp / "runs" / f"{label}.partial").is_dir()
    assert len((exp / "index.jsonl").read_text().splitlines()) == 1


def test_an_interrupted_attempt_marked_as_an_outcome_is_kept(tmp_path: Path) -> None:
    # The lead's C6: an interruption is an outcome unless its cause is
    # recorded, and a retry never replaces an outcome.
    document = _plan(_port(), blocks=1)
    records = _run(tmp_path, document)
    label = next(r["label"] for r in records if r["arm"] == "watch")
    _interrupt(tmp_path / "exp", label)
    (only,) = _run(
        tmp_path,
        document,
        resume=True,
        retry_incomplete=True,
        interrupted_as_outcome={label},
    )
    assert (only["label"], only["state"], only["reasons"], only["interrupted"]) == (
        label,
        "outcome_failure",
        ["runner_interrupted"],
        True,
    )
    summary = summarize_run(Path(only["run_dir"]) / "c1.jsonl")
    assert summary.protocol_failures == ()
    assert "runner:runner_interrupted" in summary.outcome_failures


def _unrenamed(exp: Path, label: str, *, run_json: bool) -> None:
    """Leave an attempt as a runner killed inside finish() would: its state
    appended to its artifacts (and run.json written, or not), still
    ``.partial``, with no line in the index."""
    run_dir = exp / "runs" / label
    if not run_json:
        for name in ("run.json", "SHA256SUMS"):
            (run_dir / name).unlink()
    run_dir.rename(run_dir.with_name(label + ".partial"))
    index = exp / "index.jsonl"
    kept = [
        line
        for line in index.read_text().splitlines()
        if json.loads(line)["label"] != label
    ]
    index.write_text("\n".join(kept) + "\n")


@pytest.mark.parametrize("run_json", [True, False])
def test_an_attempt_that_recorded_its_state_keeps_it_on_resume(
    tmp_path: Path, run_json: bool
) -> None:
    # rev-213-a's E2: an outcome recorded before the rename could be named
    # an external cause on resume, and so set aside.
    from stormlog.infer.experiment import _sums_verify

    document = _plan(_port(), blocks=1)
    records = _run(tmp_path, document)
    exp = tmp_path / "exp"
    label = next(r["label"] for r in records if r["arm"] == "watch")
    _unrenamed(exp, label, run_json=run_json)
    with pytest.raises(InferUsageError, match=f"{label} is not an interrupted"):
        _run(tmp_path, document, resume=True, external_causes={label: PREEMPTED})
    with pytest.raises(InferUsageError, match=f"{label} is not an interrupted"):
        _run(tmp_path, document, resume=True, interrupted_as_outcome={label})
    (only,) = _run(tmp_path, document, resume=True, retry_incomplete=True)
    assert (only["label"], only["state"], only["interrupted"]) == (
        label,
        "completed",
        False,
    )
    run_dir = exp / "runs" / label
    assert _sums_verify(run_dir)
    states = [
        line
        for line in (run_dir / "c1.jsonl").read_text().splitlines()
        if json.loads(line).get("event_type") == "infer.run_state"
    ]
    assert len(states) == 1
    index = (exp / "index.jsonl").read_text()
    assert index.count(f'"{label}"') == 1


def test_a_cause_or_an_outcome_must_name_an_interrupted_attempt(
    tmp_path: Path,
) -> None:
    document = _plan(_port(), blocks=1)
    records = _run(tmp_path, document)
    finished = records[0]["label"]
    with pytest.raises(InferUsageError, match="not an interrupted attempt"):
        _run(tmp_path, document, resume=True, external_causes={finished: PREEMPTED})
    with pytest.raises(InferUsageError, match="not an interrupted attempt"):
        _run(tmp_path, document, resume=True, interrupted_as_outcome={finished})
    with pytest.raises(InferUsageError, match="only on resume"):
        _run(tmp_path / "fresh", document, external_causes={finished: PREEMPTED})
    with pytest.raises(InferUsageError, match="only on resume"):
        _run(tmp_path / "fresh", document, interrupted_as_outcome={finished})


@pytest.mark.parametrize(
    ("reason", "evidence", "message"),
    [
        ("oom", "dmesg", "spot_preemption, operator_abort or infra_fault"),
        ("spot_preemption", "  ", "needs evidence"),
    ],
)
def test_an_external_cause_is_one_of_three_with_evidence(
    reason: str, evidence: str, message: str
) -> None:
    with pytest.raises(InferUsageError, match=message):
        ExternalCause(reason, evidence)


def test_a_failed_step_is_an_outcome_and_kept(tmp_path: Path) -> None:
    document = _plan(_port(), blocks=1)
    document["arms"]["off"]["workload"][0]["command"] = [
        "{python}",
        "-c",
        "raise SystemExit(5)",
    ]
    records = _run(tmp_path, document)
    off = next(r for r in records if r["arm"] == "off")
    assert off["state"] == "outcome_failure"
    assert "step_failed:c1:5" in off["reasons"]
    assert "artifact_missing:c1.jsonl" in off["reasons"]


def test_a_treatment_that_stops_early_is_unhealthy(tmp_path: Path) -> None:
    document = _plan(_port(), blocks=1)
    leaves = "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()"
    document["arms"]["watch"]["treatments"][0]["command"] = [
        "{python}",
        "-c",
        leaves,
        "{run_dir}/watch.ready",
    ]
    records = _run(tmp_path, document)
    watch = next(r for r in records if r["arm"] == "watch")
    assert watch["state"] == "outcome_failure"
    assert "treatment_unhealthy:watcher" in watch["reasons"]


def test_a_server_that_never_becomes_healthy_is_a_protocol_failure(
    tmp_path: Path,
) -> None:
    document = _plan(_port(), blocks=1)
    document["server"]["command"] = ["{python}", "-c", "raise SystemExit(1)"]
    records = _run(tmp_path, document)
    assert {r["state"] for r in records} == {"protocol_failure"}
    assert all("server_never_healthy" in r["reasons"] for r in records)


def test_an_artifact_labelled_for_another_run_is_refused(tmp_path: Path) -> None:
    document = _plan(_port(), blocks=1)
    command = list(PROFILE)
    command[command.index("{arm}")] = "someone-else"
    document["arms"]["off"]["workload"][0]["command"] = command
    records = _run(tmp_path, document)
    off = next(r for r in records if r["arm"] == "off")
    assert off["state"] == "protocol_failure"
    assert "label_mismatch:c1.jsonl" in off["reasons"]


def test_overlapping_cpus_are_refused_when_declared_disjoint(tmp_path: Path) -> None:
    document = _plan(_port(), blocks=1, affinity_disjoint=True)
    document["server"]["cpu_affinity"] = "0-1"
    document["arms"]["off"]["workload"][0]["cpu_affinity"] = "1"
    records = _run(tmp_path, document)
    off = next(r for r in records if r["arm"] == "off")
    assert off["state"] == "protocol_failure"
    assert "affinity_overlap:1" in off["reasons"]


def test_secrets_reach_the_commands_but_never_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "hf_plantedSecret0123456789"
    monkeypatch.setenv("PLANT_TOKEN", secret)
    document = _plan(_port(), blocks=1, secret_env=["PLANT_TOKEN"])
    writes = "import os, sys; open(sys.argv[1], 'w').write(str(len(os.environ['PLANT_TOKEN'])))"
    document["arms"]["off"]["workload"][0]["command"] = [
        "{python}",
        "-c",
        writes,
        "{run_dir}/length.txt",
    ]
    document["arms"]["off"]["workload"][0]["artifacts"] = []
    plan = plan_from_document(document)
    records = run_plan(plan, tmp_path / "exp")
    off = next(r for r in records if r["arm"] == "off")
    run_dir = Path(off["run_dir"])
    assert (run_dir / "length.txt").read_text() == str(len(secret))
    commands = (run_dir / "commands.sh").read_text()
    assert "PLANT_TOKEN=${PLANT_TOKEN}" in commands and secret not in commands


def test_a_staged_model_is_fixed_before_launch_and_recorded(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    document = _plan(_port(), blocks=1)
    document["arms"] = {"off": document["arms"]["off"]}
    document["order"] = {"kind": "explicit", "blocks": [["off"]]}
    document["server"]["command"] += ["--model", "{model}"]
    document["server"]["model"] = {
        "route": "staged",
        "source": str(source),
        "store": str(tmp_path / "store"),
    }
    (record,) = _run(tmp_path, document)
    assert record["state"] == "completed", record
    run_dir = Path(record["run_dir"])
    identity = json.loads((run_dir / "model_identity.json").read_text())
    assert identity["identity_evidence"] == "staged_snapshot_verified"
    assert identity["directory"] in (run_dir / "commands.sh").read_text()


def test_the_runner_binds_the_weights_it_verified_to_the_server_it_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Linux the runner reads the server's start ticks, the boot and the
    # description from /proc as it is. Elsewhere it cannot: stand them in.
    # fable-213's lens a: standing them in on Linux too bound the record to
    # a server the real description did not show.
    from stormlog.infer import experiment
    from stormlog.infer.host_clock import host_boot_id

    linux = platform.system() == "Linux"
    if not linux:
        monkeypatch.setattr(experiment, "process_key", lambda pid: (pid, 4242))
        monkeypatch.setattr(experiment, "host_boot_id", lambda: "boot-test")

    def describe(options: Any, **kwargs: Any) -> dict[str, Any]:
        # Off Linux describe-server cannot read /proc: a stand-in of the
        # launched server, as Linux would describe it.
        from stormlog.infer.describe_server import description_digest

        document = {
            "format": "stormlog.infer.server_description",
            "version": 1,
            "observed_at_ns": time.time_ns(),
            "run_id": options.run_id,
            "host": {"hostname": "h", "boot_id": "boot-test"},
            "server": {"pid": options.pid, "start_ticks": 4242, "environ": {}},
            "model": dict(options.model_identity or {}),
        }
        document["sha256"] = description_digest(document)
        return document

    if not linux:
        monkeypatch.setattr(experiment, "describe_server", describe)
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    document = _plan(_port(), blocks=1)
    document["arms"] = {"off": document["arms"]["off"]}
    document["order"] = {"kind": "explicit", "blocks": [["off"]]}
    document["server"]["command"] += ["--model", "{model}"]
    document["server"]["model"] = {
        "route": "staged",
        "source": str(source),
        "store": str(tmp_path / "store"),
    }
    (record,) = _run(tmp_path, document)
    assert record["state"] == "completed", record
    artifact = next(Path(record["run_dir"]).glob("*.jsonl"))
    lines = [json.loads(line) for line in artifact.read_text().splitlines()]
    (bound,) = [r for r in lines if r.get("event_type") == "infer.model_identity"]
    server = next(p for p in record["processes"] if p["name"] == "server")
    # The record names the server its run's before description shows.
    before = json.loads((Path(record["run_dir"]) / "describe-before.json").read_text())
    shown = before["server"]
    assert bound["server"] == {
        "pid": server["pid"],
        "start_ticks": shown["start_ticks"],
    }
    assert bound["boot_id"] == before["host"]["boot_id"]
    if linux:
        # Read from /proc: this boot, and the stand-in as vLLM's API server.
        assert bound["boot_id"] == host_boot_id()
        assert [p["role"] for p in shown["processes"]] == ["api_server"]
    else:
        assert (shown["start_ticks"], bound["boot_id"]) == (4242, "boot-test")
    assert bound["model"]["identity_evidence"] == "staged_snapshot_verified"
    assert bound["run_id"] == record["label"]
    # The runner attached its before description, so the comparison binds
    # the verified weights to the server that description shows.
    fields = summarize_run(artifact).fields
    assert fields["model.weights_digest"].known
    assert fields["model.weights_digest"].source.startswith("experiment runner")


def test_a_probe_that_does_not_finish_is_retried_once_on_a_fresh_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stormlog.infer import experiment
    from stormlog.infer.server_probe import SERVER_INFO, ProbeAnswer, ServerProbe

    calls: list[str] = []

    def probe(endpoint: str, **_: Any) -> ServerProbe:
        calls.append(endpoint)
        status = "timeout" if len(calls) == 1 else "ok"
        return ServerProbe(
            "before",
            "auto",
            endpoint,
            0,
            {SERVER_INFO: ProbeAnswer(SERVER_INFO, status)},
        )

    monkeypatch.setattr(experiment, "probe_server", probe)
    document = _plan(_port(), blocks=1)
    document["arms"] = {"off": document["arms"]["off"]}
    document["order"] = {"kind": "explicit", "blocks": [["off"]]}
    first, second = _run(tmp_path, document)
    assert (first["state"], first["reasons"]) == (
        "protocol_failure",
        ["probe_incomplete"],
    )
    assert first["cleanup"]["verified"] is True
    assert second["state"] == "completed" and second["attempt"] == 2
    assert second["order_broken"] is True
    # The retry ran in the block's second slot, after the first attempt.
    assert (first["position_actual"], second["position_actual"]) == (0, 1)
    assert second["position_planned"] == 0


def _survivor_on_calls(monkeypatch: pytest.MonkeyPatch, *calls: int) -> None:
    """verify_cleanup finds a survivor on these calls (1-based), in the order
    the runner checks: within a run, its treatments, then its server."""
    from stormlog.infer import experiment
    from stormlog.infer.experiment_process import Cleanup

    real = experiment.verify_cleanup
    seen: list[int] = []

    def verify(*args: Any, **kwargs: Any) -> Cleanup:
        seen.append(1)
        result = real(*args, **kwargs)
        if len(seen) in calls:
            return Cleanup(False, result.method, ({"pid": 1},))
        return result

    monkeypatch.setattr(experiment, "verify_cleanup", verify)


def test_a_probe_retry_never_starts_beside_an_unverified_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fable-213's repro: the probe times out, and the server's group leaves
    # a survivor; a second server must not start beside it.
    from stormlog.infer import experiment
    from stormlog.infer.server_probe import SERVER_INFO, ProbeAnswer, ServerProbe

    def probe(endpoint: str, **_: Any) -> ServerProbe:
        answer = ProbeAnswer(SERVER_INFO, "timeout")
        return ServerProbe("before", "auto", endpoint, 0, {SERVER_INFO: answer})

    monkeypatch.setattr(experiment, "probe_server", probe)
    _survivor_on_calls(monkeypatch, 1)
    document = _plan(_port(), blocks=1)
    document["order"] = {"kind": "explicit", "blocks": [["off", "watch"]]}
    only, skipped = _run(tmp_path, document)
    assert only["reasons"] == ["probe_incomplete", "collector_cleanup_unverified"]
    assert (skipped["arm"], skipped["state"]) == ("watch", "not_run")


def _index(tmp_path: Path) -> list[tuple[str, int, str, str]]:
    lines = (tmp_path / "exp" / "index.jsonl").read_text().splitlines()
    return [
        (r["type"], r["block"], r["arm"], r["state"]) for r in map(json.loads, lines)
    ]


TWO_BLOCKS = {"kind": "explicit", "blocks": [["off", "watch"], ["watch", "off"]]}


def test_a_server_left_running_stops_the_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rev-213-a's E1: the next block's servers started beside the survivor.
    _survivor_on_calls(monkeypatch, 1)
    document = _plan(_port(), order=TWO_BLOCKS)
    records = _run(tmp_path, document)
    first, *rest = records
    assert first["reasons"] == ["collector_cleanup_unverified"]
    # The lead's ruling: each run left is not_run: cleanup_unverified.
    assert [(r["block"], r["arm"], r["reasons"]) for r in rest] == [
        (0, "watch", ["cleanup_unverified"]),
        (1, "watch", ["cleanup_unverified"]),
        (1, "off", ["cleanup_unverified"]),
    ]
    assert {r["stopped_after"] for r in rest} == {first["label"]}
    assert {r["state"] for r in rest} == {"not_run"}
    assert _index(tmp_path) == [
        ("run", 0, "off", "protocol_failure"),
        ("not_run", 0, "watch", "not_run"),
        ("not_run", 1, "watch", "not_run"),
        ("not_run", 1, "off", "not_run"),
    ]
    # No server started after it.
    assert len(list((tmp_path / "exp" / "runs").iterdir())) == 1

    # Once the host is clean, a resume runs what the stop left.
    monkeypatch.undo()
    resumed = _run(tmp_path, document, resume=True)
    assert [(r["block"], r["arm"], r["state"]) for r in resumed] == [
        (0, "watch", "completed"),
        (1, "watch", "completed"),
        (1, "off", "completed"),
    ]


def test_a_resume_waits_until_what_a_cleanup_left_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from stormlog.infer import experiment
    from stormlog.infer.experiment_process import Cleanup, identify

    left = subprocess.Popen(["/bin/sleep", "60"])
    try:
        real = experiment.verify_cleanup

        def verify(*args: Any, **kwargs: Any) -> Cleanup:
            result = real(*args, **kwargs)
            return Cleanup(False, result.method, (identify(left.pid),))

        monkeypatch.setattr(experiment, "verify_cleanup", verify)
        document = _plan(_port(), order=TWO_BLOCKS)
        _run(tmp_path, document)
        monkeypatch.undo()
        # A usage error (exit 2) until the survivor is gone.
        with pytest.raises(InferUsageError, match=f"left {left.pid} running"):
            _run(tmp_path, document, resume=True)
    finally:
        left.kill()
        left.wait()
    resumed = _run(tmp_path, document, resume=True)
    assert {r["state"] for r in resumed} == {"completed"}


def test_a_resume_waits_until_a_process_the_cleanup_could_not_judge_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rev-213-a's N2: a stop caused only by a blind process did not hold the
    # resume, and three servers started beside it.
    import subprocess

    from stormlog.infer import experiment
    from stormlog.infer.experiment_process import Cleanup, identify

    blind = subprocess.Popen(["/bin/sleep", "60"])
    try:
        real = experiment.verify_cleanup

        def verify(*args: Any, **kwargs: Any) -> Cleanup:
            result = real(*args, **kwargs)
            return Cleanup(False, result.method, (), (), 1, (identify(blind.pid),))

        monkeypatch.setattr(experiment, "verify_cleanup", verify)
        document = _plan(_port(), order=TWO_BLOCKS)
        _run(tmp_path, document)
        monkeypatch.undo()
        with pytest.raises(InferUsageError, match=f"left {blind.pid} running"):
            _run(tmp_path, document, resume=True)
    finally:
        blind.kill()
        blind.wait()
    resumed = _run(tmp_path, document, resume=True)
    assert {r["state"] for r in resumed} == {"completed"}


def test_a_resume_stops_the_server_a_killed_runner_left(tmp_path: Path) -> None:
    # rev-213-a's N1: SIGKILL runs no finally, so the runner's server lived
    # on in its own session; the resumed servers could not bind its port,
    # and the workload measured it for both arms.
    import os
    import signal
    import subprocess

    from stormlog.infer.experiment_process import journaled, still_there, stop_journaled

    document = _plan(_port(), blocks=1)
    document["order"] = {"kind": "explicit", "blocks": [["off", "watch"]]}
    hold = {"name": "hold", "command": ["{python}", "-c", "import time; time.sleep(5)"]}
    document["arms"]["off"]["workload"].append(hold)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(document))
    exp = tmp_path / "exp"
    runner = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "examples.cli.infer_repeated_baseline",
            "--plan",
            str(plan_file),
            "--output",
            str(exp),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    journal = None
    try:
        deadline = time.monotonic() + 90
        while journal is None and time.monotonic() < deadline:
            for found in exp.glob("runs/*.partial/launches.ndjson"):
                if "step:hold" in found.read_text():
                    journal = found
            time.sleep(0.05)
        assert journal is not None
    finally:
        os.kill(runner.pid, signal.SIGKILL)
        runner.wait()
    left = {entry["name"]: entry for entry in journaled(journal)}
    try:
        assert still_there(left["server"]["identity"])
        label = journal.parent.name[: -len(".partial")]
        cause = ExternalCause("operator_abort", "the test killed it")
        records = run_plan(
            plan_from_document(document),
            exp,
            resume=True,
            retry_incomplete=True,
            external_causes={label: cause},
            environment=Environment(python=sys.executable),
        )
        assert not any(still_there(entry["identity"]) for entry in left.values())
    finally:
        for entry in left.values():
            stop_journaled(entry, timeout_s=5)
    assert [(r["arm"], r.get("attempt"), r["state"]) for r in records] == [
        ("off", 1, "protocol_failure"),
        ("off", 2, "completed"),
        ("watch", 1, "completed"),
    ]


def test_a_port_already_taken_stops_the_experiment_before_a_launch(
    tmp_path: Path,
) -> None:
    port = _port()
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", port))
        taken.listen()
        first, *rest = _run(tmp_path, _plan(port, blocks=1))
    assert first["reasons"] == ["server_port_in_use"]
    assert first["processes"] == []
    assert {(r["state"], tuple(r["reasons"])) for r in rest} == {
        ("not_run", ("server_port_in_use",))
    }


def test_a_health_answer_counts_only_from_this_launchs_server(tmp_path: Path) -> None:
    from stormlog.infer import experiment
    from stormlog.infer.experiment_process import launch, stop

    port = _port()
    url = f"http://127.0.0.1:{port}/v1"
    other = launch("other", [sys.executable, FAKE, "--port", str(port)])
    idle = launch("server", [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert experiment._wait_healthy(url, other, 20)
        # Another process answers on the port; this launch listens on none.
        assert not experiment._wait_healthy(url, idle, 2)
    finally:
        stop(other, timeout_s=5)
        stop(idle, timeout_s=5)


def test_what_a_step_leaves_behind_is_stopped_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psutil

    # rev-213-a's N7: a step that exits 0 but leaves a process in its group
    # (a load generator, a monitor) let it run into the next run's server.
    document = _plan(_port(), blocks=1)
    document["arms"] = {"off": document["arms"]["off"]}
    document["order"] = {"kind": "explicit", "blocks": [["off"]]}
    leave = {
        "name": "leave",
        "command": [
            "/bin/sh",
            "-c",
            # Python, not /bin/sleep: macOS hides a platform binary's
            # environment, which would make it blind to concurrent tests.
            "{python} -c 'import time; time.sleep(30)' & "
            "echo $! > {run_dir}/left.pid; exit 0",
        ],
    }
    document["arms"]["off"]["workload"].append(leave)
    (record,) = _run(tmp_path, document)
    assert record["state"] == "completed"
    step = next(p for p in record["processes"] if p["name"] == "step:leave")
    assert step["cleanup"]["verified"] is True
    left = int((Path(record["run_dir"]) / "left.pid").read_text())
    assert not psutil.pid_exists(left) or (
        psutil.Process(left).status() == psutil.STATUS_ZOMBIE
    )

    # One that outlives its kill stops the experiment, as a server's would.
    from stormlog.infer import experiment_process
    from stormlog.infer.experiment_process import Cleanup

    monkeypatch.setattr(
        experiment_process,
        "verify_cleanup",
        lambda *args, **kwargs: Cleanup(False, "test", ({"pid": 1},)),
    )
    (record,) = _run(tmp_path / "again", document)
    assert "step_cleanup_unverified:c1" in record["reasons"]


def test_a_prelude_server_loads_the_verified_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A warm-up must warm the snapshot the runs measure, offline.
    from stormlog.infer import experiment
    from stormlog.infer.model_identity import VerifiedModel

    launched: list[tuple[list[str], dict[str, str]]] = []

    class _Stop(Exception):
        pass

    def launch(_name: str, command: list[str], *, env: dict[str, str], **_: Any) -> Any:
        launched.append((command, env))
        raise _Stop

    monkeypatch.setattr(experiment, "launch", launch)
    document = _plan(_port(), blocks=1)
    document["block_prelude"] = [
        {"name": "warm", "server_arm": "off", "command": ["{python}", "-c", "pass"]}
    ]
    plan = plan_from_document(document)
    commit = "c" * 40
    model = VerifiedModel(
        route="pinned_hub",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        server_args=("--revision", commit, "--tokenizer-revision", commit),
        env={"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": "/hub"},
        directory=tmp_path,
        files=(),
    )
    env = Environment(python=sys.executable, model=model)
    with pytest.raises(_Stop):
        experiment._prelude(plan, plan.preludes[0], 0, tmp_path / "exp", env)
    ((command, server_env),) = launched
    assert command[-4:] == ["--revision", commit, "--tokenizer-revision", commit]
    assert (server_env["HF_HUB_OFFLINE"], server_env["HF_HUB_CACHE"]) == ("1", "/hub")


def _unhealthy_on(monkeypatch: pytest.MonkeyPatch, *calls: int) -> None:
    """The server never becomes healthy on these runs (1-based, in order)."""
    from stormlog.infer import experiment

    real = experiment._wait_healthy
    seen: list[int] = []

    def wait(base_url: str, server: Any, timeout_s: float) -> bool:
        seen.append(1)
        return False if len(seen) in calls else real(base_url, server, timeout_s)

    monkeypatch.setattr(experiment, "_wait_healthy", wait)


def _against_control(
    order: list[str], *, args: bool, env: bool = False
) -> dict[str, Any]:
    document = _plan(_port(), blocks=1, control_arm="off")
    document["order"] = {"kind": "explicit", "blocks": [order]}
    if args:
        # The treatment is the arm's own launch.
        document["arms"]["watch"]["server"] = {"args": ["--latency", "0"]}
    if env:
        document["arms"]["watch"]["server"] = {"env": {"VLLM_PLUGINS": "stormlog"}}
    return document


def test_a_launch_that_differs_only_in_its_environment_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rev-213-a's N9 (mutant e7_env_ignored): a hook plugin is enabled by
    # environment alone.
    _unhealthy_on(monkeypatch, 2)
    records = _run(tmp_path, _against_control(["off", "watch"], args=False, env=True))
    watch = records[-1]
    assert (watch["state"], watch["decided_by"]) == (
        "outcome_failure",
        "arm_launch_differs",
    )


def test_a_decision_the_runner_died_before_is_made_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rev-213-a's N3: the runner died after the control ran but before the
    # block ended, so the never-healthy attempt stayed a protocol failure,
    # was never indexed, and --retry-incomplete retried it away.
    from stormlog.infer import experiment

    _unhealthy_on(monkeypatch, 1)

    def die(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the runner died")

    monkeypatch.setattr(experiment, "_decide_unhealthy", die)
    document = _against_control(["watch", "off"], args=True)
    with pytest.raises(RuntimeError):
        _run(tmp_path, document)
    exp = tmp_path / "exp"
    (watch_dir,) = exp.glob("runs/*-watch-a1")
    assert json.loads((watch_dir / "run.json").read_text())["decided_by"] == "pending"
    monkeypatch.undo()
    resumed = _run(tmp_path, document, resume=True, retry_incomplete=True)
    assert [(r["label"], r["state"], r["decided_by"]) for r in resumed] == [
        (watch_dir.name, "outcome_failure", "arm_launch_differs")
    ]
    index = [json.loads(line) for line in (exp / "index.jsonl").open()]
    assert [r["label"] for r in index].count(watch_dir.name) == 1


def test_a_control_seen_before_a_stop_decides_nothing_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rev-213-a's N5: the control came up, then its cleanup did not verify
    # and the experiment stopped; after the resume the arm's server never
    # came up, and the old evidence made it the arm's outcome.
    _survivor_on_calls(monkeypatch, 1)
    document = _against_control(["off", "watch"], args=True)
    _run(tmp_path, document)
    monkeypatch.undo()
    _unhealthy_on(monkeypatch, 1)
    (watch,) = _run(tmp_path, document, resume=True)
    assert (watch["state"], watch["decided_by"]) == (
        "protocol_failure",
        "control_not_launched",
    )


@pytest.mark.parametrize("order", [["off", "watch"], ["watch", "off"]])
def test_a_server_the_arms_own_launch_kept_from_starting_is_an_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: list[str]
) -> None:
    # The lead's ruling on rev-213-a's E7: a Stormlog arm that stops vLLM
    # from starting must not read as a harness problem and be set aside,
    # whether the control ran before it in the block or after.
    _unhealthy_on(monkeypatch, order.index("watch") + 1)
    records = _run(tmp_path, _against_control(order, args=True))
    watch = next(r for r in records if r["arm"] == "watch")
    off = next(r for r in records if r["arm"] == "off")
    assert (watch["state"], watch["reasons"], watch["decided_by"]) == (
        "outcome_failure",
        ["server_never_healthy"],
        "arm_launch_differs",
    )
    assert off["state"] == "completed"
    index = [json.loads(line) for line in (tmp_path / "exp" / "index.jsonl").open()]
    assert [r["state"] for r in index if r["arm"] == "watch"] == ["outcome_failure"]
    recorded = json.loads((Path(watch["run_dir"]) / "run.json").read_text())
    assert recorded["state"] == "outcome_failure"


def test_a_server_that_fails_with_the_controls_own_launch_is_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _unhealthy_on(monkeypatch, 2)
    records = _run(tmp_path, _against_control(["off", "watch"], args=False))
    watch = records[-1]
    assert (watch["state"], watch["decided_by"]) == (
        "protocol_failure",
        "identical_launch",
    )


def test_a_server_that_fails_when_the_controls_did_too_is_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _unhealthy_on(monkeypatch, 1, 2)
    records = _run(tmp_path, _against_control(["off", "watch"], args=True))
    by_arm = {r["arm"]: r for r in records}
    assert (by_arm["watch"]["state"], by_arm["watch"]["decided_by"]) == (
        "protocol_failure",
        "control_also_unhealthy",
    )
    assert by_arm["off"]["decided_by"] == "identical_launch"


def test_a_treatment_left_running_stops_the_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fable-213: treatment_cleanup_unverified did not even stop the block.
    _survivor_on_calls(monkeypatch, 1)
    document = _plan(_port(), blocks=1)
    document["order"] = {"kind": "explicit", "blocks": [["watch", "off"]]}
    watch, off = _run(tmp_path, document)
    assert watch["reasons"] == ["treatment_cleanup_unverified:watcher"]
    assert (off["arm"], off["state"]) == ("off", "not_run")


def test_a_prelude_server_left_running_stops_the_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _survivor_on_calls(monkeypatch, 1)
    document = _plan(_port(), order=TWO_BLOCKS)
    document["block_prelude"] = [
        {"name": "warm", "server_arm": "off", "command": ["{python}", "-c", "pass"]}
    ]
    first, *rest = _run(tmp_path, document)
    assert first["reasons"] == ["prelude_failed:warm:cleanup_unverified"]
    assert first["processes"] == []
    assert len(rest) == 3 and {r["state"] for r in rest} == {"not_run"}
    cleanup = tmp_path / "exp" / "preludes" / "b00-warm" / "cleanup.json"
    assert json.loads(cleanup.read_text())["verified"] is False


def test_the_bundle_is_scanned_for_the_plans_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "hf_plantedSecret0123456789"
    monkeypatch.setenv("PLANT_TOKEN", secret)
    document = _plan(_port(), blocks=1, secret_env=["PLANT_TOKEN"])
    leaks = "import os, sys; open(sys.argv[1], 'w').write(os.environ['PLANT_TOKEN'])"
    document["arms"]["off"]["workload"][0]["command"] = [
        "{python}",
        "-c",
        leaks,
        "{run_dir}/leak.txt",
    ]
    document["arms"]["off"]["workload"][0]["artifacts"] = []
    run_plan(plan_from_document(document), tmp_path / "exp")
    report = json.loads((tmp_path / "exp" / "sanitizer.json").read_text())
    assert report["publishable"] is False
    assert any(hit["file"].endswith("leak.txt") for hit in report["hits"])


def test_the_example_cli_runs_a_plan_and_says_how_it_ended(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from examples.cli.infer_repeated_baseline import main

    document = _plan(_port(), blocks=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(document))
    assert main(["--plan", str(plan_path), "--output", str(tmp_path / "exp")]) == 0
    printed = capsys.readouterr().out
    assert "t213-b00-p0-off-a1: completed" in printed
    original = plan_path.read_text()
    document["server"]["command"] = ["{python}", "-c", "raise SystemExit(1)"]
    plan_path.write_text(json.dumps(document))
    code = main(["--plan", str(plan_path), "--output", str(tmp_path / "broken")])
    assert code == 3
    assert (
        main(
            ["--plan", str(tmp_path / "nothing.json"), "--output", str(tmp_path / "x")]
        )
        == 5
    )
    cause = ["--external-cause", "t213-b00-p0-off-a1=oom:dmesg"]
    assert main(["--plan", str(plan_path), "--output", "y", "--resume", *cause]) == 2
    _interrupt(tmp_path / "exp", "t213-b00-p1-watch-a1")
    plan_path.write_text(original)
    resume = ["--plan", str(plan_path), "--output", str(tmp_path / "exp"), "--resume"]
    assert main(resume) == 2
    assert "t213-b00-p1-watch-a1 was interrupted" in capsys.readouterr().err
    marked = ["--interrupted-as-outcome", "t213-b00-p1-watch-a1"]
    assert main([*resume, *marked]) == 0
    assert "t213-b00-p1-watch-a1: outcome_failure (runner_interrupted)" in (
        capsys.readouterr().out
    )


def test_treatments_are_observers_a_comparison_can_see(tmp_path: Path) -> None:
    records = _run(tmp_path, _plan(_port(), blocks=2))
    by_arm: dict[str, list[Any]] = {"off": [], "watch": []}
    for record in records:
        by_arm[record["arm"]].append(
            summarize_run(Path(record["run_dir"]) / "c1.jsonl")
        )
    watcher = by_arm["watch"][0].observers["treatment:watcher"]
    assert (watcher["requested"], watcher["active"], watcher["healthy"]) == (
        True,
        True,
        True,
    )
    assert "treatment:watcher" not in by_arm["off"][0].observers
    # A comparison of off against watch has to declare the watcher.
    with pytest.raises(InferInputError, match="--added-observers: treatment:watcher"):
        compare_runs(by_arm["off"], by_arm["watch"], ComparisonSpec(mode="incremental"))
    declared = compare_runs(
        by_arm["off"],
        by_arm["watch"],
        ComparisonSpec(mode="incremental", added_observers=("treatment:watcher",)),
    )
    assert declared.observer_issues == []
