"""The comparison's method validation, examples/analysis/simulation_study.py."""

from __future__ import annotations

import json
from pathlib import Path

from examples.analysis.simulation_study import run

RESULTS = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "analysis"
    / "simulation_results.json"
)


def test_the_published_study_meets_every_acceptance_criterion() -> None:
    results = json.loads(RESULTS.read_text())
    assert (results["seed"], results["reps"]) == (213, 20_000)
    assert results["verdict"] == {
        "paired_false_safe": True,
        "missing_outcomes": True,
        "higher_is_better_and_independent": True,
        "independent_coverage": True,
        "run_gate": True,
        "skew_screen": True,
        "module_agrees": True,
        "fraction_floor_independent": True,
    }


def test_the_study_runs_and_its_rules_are_the_modules() -> None:
    results = run(reps=300, module_samples=20)
    assert results["module_disagreements"] == 0
    assert set(results["verdict"]) == {
        "paired_false_safe",
        "missing_outcomes",
        "higher_is_better_and_independent",
        "independent_coverage",
        "run_gate",
        "skew_screen",
        "module_agrees",
        "fraction_floor_independent",
    }
    assert results["fraction_module_disagreements"] == 0


def test_a_run_gate_cell_that_cannot_pass_says_so() -> None:
    # A cell where the gate can never pass measures nothing: it is listed as
    # one, with the fewest runs that could, never as a pass rate of 0.
    results = json.loads(RESULTS.read_text())
    rows = results["run_gate_false_pass"]
    for row in rows:
        reachable = 0.025 ** (1 / row["runs"]) >= row["q"]
        assert row["can_pass"] is reachable
        assert (row["false_pass"] is None) is not reachable
        assert (row["runs"] >= row["min_runs_to_pass"]) is reachable
        assert 0.025 ** (1 / (row["min_runs_to_pass"] - 1)) < row["q"]
        assert row["p_run"] == row["q"]  # the supremum of a false claim
    assert not all(row["can_pass"] for row in rows)
    for runs in results["clustered_runs"]:
        assert 0.025 ** (1 / int(runs)) >= 0.8


def test_clustered_failures_are_published_as_a_limit_of_fraction_intervals() -> None:
    # Under correlation within a run no interval on a fraction holds 2.5%,
    # which is why the gate is the claim about runs.
    results = json.loads(RESULTS.read_text())
    rows = results["fraction_clustered_limit"]
    assert {row["rho"] for row in rows} == {0.02, 0.05}
    assert all(row["floored_t_false_safe"] > 0.1 for row in rows)
    assert all(row["runs_within_budget"] > 0.5 for row in rows)
