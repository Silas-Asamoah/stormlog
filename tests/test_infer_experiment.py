"""The experiment runner, end to end against a stand-in vLLM server."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.compare import ComparisonSpec, compare_runs
from stormlog.infer.errors import InferInputError
from stormlog.infer.experiment import Environment, run_plan
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
