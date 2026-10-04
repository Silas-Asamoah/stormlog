"""Experiment plans: validation, order, block seeds and placeholders."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.errors import InferInputError
from stormlog.infer.experiment_plan import (
    block_seed,
    expand,
    load_plan,
    plan_from_document,
    plan_order,
    williams,
)


def _plan(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": "stormlog.infer.experiment_plan",
        "version": 1,
        "experiment_id": "q213",
        "seed": 213,
        "blocks": 4,
        "order": {"kind": "williams"},
        "server": {
            "command": ["vllm", "serve", "{model}", "--port", "8000"],
            "base_url": "http://127.0.0.1:8000/v1",
            "cpu_affinity": "0-3",
        },
        "arms": {
            "off": {
                "workload": [
                    {
                        "name": "c1",
                        "command": [
                            "{python}",
                            "-m",
                            "stormlog",
                            "infer",
                            "profile",
                            "--seed",
                            "{block_seed}",
                            "--output",
                            "{run_dir}/c1.jsonl",
                        ],
                        "artifacts": ["{run_dir}/c1.jsonl"],
                        "expect_exit": [0, 3],
                    }
                ]
            },
            "watch": {
                "workload": "same_as:off",
                "treatments": [
                    {
                        "name": "watcher",
                        "command": [
                            "{python}",
                            "-m",
                            "watcher",
                            "--ready",
                            "{run_dir}/ready",
                        ],
                        "ready_file": "{run_dir}/ready",
                        "stop_signal": "SIGINT",
                    }
                ],
            },
        },
    }
    document.update(changes)
    return document


def test_a_plan_loads_with_matched_workloads(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan()))
    plan = load_plan(path)

    assert plan.experiment_id == "q213" and plan.blocks == 4
    assert plan.arms["watch"].workload == plan.arms["off"].workload
    (treatment,) = plan.arms["watch"].treatments
    assert treatment.stop_signal == 2  # SIGINT
    assert plan.arms["off"].workload[0].expect_exit == (0, 3)
    assert len(plan.digest) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": "other"}, "not stormlog.infer.experiment_plan"),
        ({"version": 2}, "version 2"),
        ({"blocks": 0}, "blocks must be"),
        ({"experiment_id": "has space"}, "short identifier"),
        ({"order": {"kind": "alphabetical"}}, "order.kind"),
        ({"arms": {}}, "arms must be a non-empty object"),
    ],
)
def test_a_plan_it_cannot_run_is_invalid_input(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan(**changes)))
    with pytest.raises(InferInputError, match=message):
        load_plan(path)


def test_an_unknown_placeholder_is_refused_when_the_plan_loads() -> None:
    document = _plan()
    document["arms"]["off"]["workload"][0]["command"].append("{output_dir}")
    with pytest.raises(ValueError, match="unknown placeholders: output_dir"):
        plan_from_document(document)


def test_same_as_must_name_an_arm_with_its_own_steps() -> None:
    document = _plan()
    document["arms"]["watch"]["workload"] = "same_as:nowhere"
    with pytest.raises(ValueError, match="names no arm"):
        plan_from_document(document)


@pytest.mark.parametrize("arms", [2, 3, 4, 5, 8])
def test_a_williams_design_balances_positions_and_carryover(arms: int) -> None:
    square = williams(arms)
    assert len(square) == (arms if arms % 2 == 0 else 2 * arms)
    for position in range(arms):
        counts = [sum(1 for row in square if row[position] == a) for a in range(arms)]
        assert len(set(counts)) == 1
    pairs = [(row[i], row[i + 1]) for row in square for i in range(arms - 1)]
    expected = 1 if arms % 2 == 0 else 2
    for a, b in itertools.permutations(range(arms), 2):
        assert pairs.count((a, b)) == expected


def test_a_williams_order_that_does_not_fill_its_square_says_so() -> None:
    document = _plan(blocks=3)
    order = plan_order(plan_from_document(document))
    assert order.kind == "williams" and order.balanced is False
    assert order.position_counts["off"] == [2, 1]
    full = plan_order(plan_from_document(_plan(blocks=4)))
    assert full.balanced is True


def test_an_explicit_order_must_hold_every_arm_in_every_block() -> None:
    good = _plan(
        blocks=2,
        order={"kind": "explicit", "blocks": [["off", "watch"], ["watch", "off"]]},
    )
    assert plan_order(plan_from_document(good)).blocks == [
        ["off", "watch"],
        ["watch", "off"],
    ]
    bad = copy.deepcopy(good)
    bad["order"]["blocks"][1] = ["watch", "watch"]
    with pytest.raises(ValueError, match="every arm once"):
        plan_from_document(bad)


def test_a_random_order_is_seeded() -> None:
    first = plan_order(plan_from_document(_plan(order={"kind": "random"})))
    again = plan_order(plan_from_document(_plan(order={"kind": "random"})))
    assert first.blocks == again.blocks


def test_every_arm_of_a_block_shares_one_seed() -> None:
    plan = plan_from_document(_plan())
    seeds = [block_seed(plan, block) for block in range(4)]
    assert len(set(seeds)) == 4
    assert all(0 <= seed < 2**31 for seed in seeds)
    assert block_seed(plan, 2) == block_seed(plan_from_document(_plan()), 2)


def test_templates_are_filled_and_a_missing_value_is_an_error() -> None:
    assert expand("{run_dir}/c1.jsonl", {"run_dir": "/x"}) == "/x/c1.jsonl"
    with pytest.raises(KeyError, match="server_pid"):
        expand("--pid {server_pid}", {})


def test_a_preregistration_is_part_of_the_plan_and_has_its_own_digest() -> None:
    plan = plan_from_document(_plan(prereg={"budgets": {"client.e2e.p95": 0.05}}))
    other = plan_from_document(_plan(prereg={"budgets": {"client.e2e.p95": 0.10}}))
    assert plan.prereg_digest is not None and plan.prereg_digest != other.prereg_digest
    assert plan.digest != other.digest


def test_the_control_arm_is_named_or_the_one_with_the_plans_own_launch() -> None:
    plan = plan_from_document(_plan(control_arm="off"))
    assert plan.control_arm == "off"
    with pytest.raises(ValueError, match="control_arm 'nope' is not an arm"):
        plan_from_document(_plan(control_arm="nope"))
    # Both arms launch the plan's server as it is: no control can be told.
    assert plan_from_document(_plan()).control_arm is None
    document = _plan()
    document["arms"]["watch"]["server"] = {"args": ["--enforce-eager"]}
    assert plan_from_document(document).control_arm == "off"
