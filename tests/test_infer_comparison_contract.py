"""The units contract with #221: comparison_contract_v1.json, case by case.

The fixture's expected numbers come from the textbook formulas
(examples/analysis/comparison_contract.py), not from comparison_stats, and
#221's gate wrapper runs the same cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.comparison_stats import GateRule, compare_values

FIXTURE = Path(__file__).parent / "fixtures" / "infer" / "comparison_contract_v1.json"
CONTRACT = json.loads(FIXTURE.read_text())
TOLERANCE = CONTRACT["tolerance"]


def _value(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _run(case: dict[str, Any]) -> Any:
    given = case["input"]
    blocks = given["blocks"]
    return compare_values(
        case["id"],
        [_value(v) for v in given["baseline"]],
        [_value(v) for v in given["candidate"]],
        direction=given["direction"],
        scale=given["scale"],
        unit=given["unit"],
        value_unit=given["value_unit"],
        blocks=None if blocks is None else (blocks[0], blocks[1]),
        confidence=CONTRACT["confidence"],
        gate=GateRule(**given["gate"]) if given["gate"] else None,
        unavailable=given["unavailable"],
    )


def _close(actual: float | None, expected: float | None) -> bool:
    if expected is None or actual is None:
        return actual is expected
    return actual == pytest.approx(
        expected, rel=TOLERANCE["relative"], abs=TOLERANCE["absolute"]
    )


def test_the_contract_is_version_1_of_the_comparison_payload() -> None:
    assert (CONTRACT["format"], CONTRACT["version"]) == (
        "stormlog.infer.comparison_contract",
        1,
    )
    assert CONTRACT["payload"] == "stormlog.infer.comparison v1"


@pytest.mark.parametrize("case", CONTRACT["cases"], ids=lambda case: case["id"])
def test_every_contract_case_holds(case: dict[str, Any]) -> None:
    result = _run(case)
    expect = case["expect"]
    worst = result.worst
    if "lower" in expect:
        assert worst is not None
        assert _close(worst.effect, expect["effect"])
        assert _close(worst.lower, expect["lower"])
        assert _close(worst.upper, expect["upper"])
    elif "effect" in expect:
        assert worst is None or worst.effect is None
    if "best" in expect:
        assert result.best is not None
        assert _close(result.best.lower, expect["best"]["lower"])
        assert _close(result.best.upper, expect["best"]["upper"])
    if "difference" in expect:
        assert result.difference is not None
        assert _close(result.difference.effect, expect["difference"]["effect"])
        assert _close(result.difference.lower, expect["difference"]["lower"])
    if "df" in expect:
        assert worst is not None and worst.df == expect["df"]
    if "n_pairs" in expect:
        assert result.n_pairs == expect["n_pairs"]
    if "run_departs_upper" in expect:
        assert result.degenerate is not None
        bound = result.degenerate["run_departs_upper"]["candidate"]
        assert _close(bound, expect["run_departs_upper"])
    if "gate" in expect:
        assert result.gate is not None
        assert result.gate.status == expect["gate"]
    if "reason" in expect:
        assert result.gate is not None and result.gate.reason == expect["reason"]
    if "case" in expect:
        assert result.gate is not None and result.gate.case == expect["case"]


def test_the_fixture_is_what_its_generator_writes() -> None:
    from examples.analysis.comparison_contract import build

    assert json.loads(json.dumps(build())) == CONTRACT["cases"]
