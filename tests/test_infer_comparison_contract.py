"""The units contract with #221: comparison_contract_v1.json, case by case.

The fixture's expected numbers come from the textbook formulas
(examples/analysis/comparison_contract.py), not from comparison_stats, and
#221's gate wrapper runs the same cases.
"""

from __future__ import annotations

import copy
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


def _differences(actual: Any, expected: Any, where: str = "cases") -> list[str]:
    """Where two contract documents differ by the fixture's tolerance_rule:
    a float within the tolerance of the other, anything else exactly."""
    if isinstance(actual, float) or isinstance(expected, float):
        numbers = all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in (actual, expected)
        )
        if numbers and _close(actual, expected):
            return []
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() == expected.keys():
            return [
                difference
                for key in expected
                for difference in _differences(
                    actual[key], expected[key], f"{where}.{key}"
                )
            ]
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) == len(expected):
            return [
                difference
                for index, pair in enumerate(zip(actual, expected))
                for difference in _differences(*pair, f"{where}[{index}]")
            ]
    elif type(actual) is type(expected) and actual == expected:
        return []
    return [f"{where}: {actual!r} != {expected!r}"]


def test_the_fixture_is_what_its_generator_writes() -> None:
    from examples.analysis.comparison_contract import build

    actual = json.loads(json.dumps(build()))
    assert _differences(actual, CONTRACT["cases"]) == []


def test_the_fixtures_floats_hold_to_its_tolerance_and_no_further() -> None:
    # fable-213's lens a: on Linux (numpy 2.2.6, scipy 1.14.1) the generator
    # wrote these two floats with other last bits, and the exact check failed.
    rule = CONTRACT["tolerance_rule"]
    assert "max(relative * |expected|, absolute)" in rule
    linux = copy.deepcopy(CONTRACT["cases"])
    by_id = {case["id"]: case for case in linux}
    by_id["latency_regression_fails_non_inferiority"]["expect"][
        "lower"
    ] = 0.3966848882246326
    by_id["latency_boundary_just_above"]["input"]["gate"][
        "budget"
    ] = 0.050429563144216594
    assert _differences(linux, CONTRACT["cases"]) == []
    beyond = by_id["latency_regression_fails_non_inferiority"]["expect"]
    beyond["lower"] *= 1 + 1e-6
    (difference,) = _differences(linux, CONTRACT["cases"])
    assert difference.startswith("cases[0].expect.lower: ")
    # Anything but a float is exact: a status, a reason, a missing value.
    beyond["lower"] = CONTRACT["cases"][0]["expect"]["lower"]
    for key, value in (("gate", "pass"), ("reason", "within"), ("lower", None)):
        changed = copy.deepcopy(linux)
        changed[0]["expect"][key] = value
        assert _differences(changed, CONTRACT["cases"]), key
