"""#221's episode catalog and injection plans."""

from __future__ import annotations

from typing import Any

import pytest

from examples.qualification.catalog import CATALOG, episode_type
from examples.qualification.plan import FORMAT, PlanError, parse_plan
from stormlog.infer.qualify.ground_truth import Injection, Times


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_every_catalog_label_is_valid_ground_truth(name: str) -> None:
    row = CATALOG[name]
    injection = Injection(
        episode_id=f"q221-{name}",
        episode_type=row.id,
        cause_class=row.cause_class,
        injected={"method": row.method},
        expects=row.expects,
        secondary=row.secondary,
        allows=row.allows,
        times=Times(),
        clock_domain=None,
        status="valid",
    )
    assert injection.problems() == []
    assert (row.cause_class == "fault") == bool(
        row.expects and row.expects[0].cause == "fault"
    )


def test_types_this_harness_does_not_run_are_named() -> None:
    with pytest.raises(ValueError, match="not run by this harness yet"):
        episode_type("F5")
    with pytest.raises(ValueError, match="not run by this harness yet"):
        episode_type("S-F1")
    with pytest.raises(ValueError, match="unknown episode type"):
        episode_type("F9")


def _plan(*episodes: dict[str, Any], **changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "format": FORMAT,
        "profile": "dx-off",
        "seed": 7,
        "victim": {"rate_per_second": 3.0, "input_tokens": 64, "output_tokens": 8},
        "timeline": {"priming": 5, "baseline": 5, "episode": 5},
        "episodes": list(episodes),
    }
    record.update(changes)
    return record


def test_a_plan_fills_each_dose_from_the_catalog() -> None:
    plan = parse_plan(
        _plan(
            {"type": "F2"},
            {
                "type": "F1",
                "dose": {
                    "rate_per_second": 20,
                    "input_tokens": 128,
                    "output_tokens": 16,
                },
            },
            {"type": "N"},
            {"type": "F4a", "dose": {"pulse_ms": 60}},
        )
    )
    assert [episode.type for episode in plan.episodes] == ["F2", "F1", "N", "F4a"]
    f2 = plan.episodes[0].neighbor_shape()
    assert (f2.concurrency, f2.input_tokens, f2.output_tokens) == (8, 2048, 1024)
    assert plan.episodes[3].dose == {"pulse_ms": 60, "period_ms": 2000}
    assert plan.victim.shared_prefix_tokens == 48
    assert parse_plan(plan.to_record()) == plan
    # 5 + 5 + 4 x (5 + 150) + 60
    assert plan.victim_duration_seconds() == 690


def test_a_bad_plan_lists_every_problem() -> None:
    with pytest.raises(PlanError) as error:
        parse_plan(
            _plan(
                {"type": "F1"},  # no rate or shape
                {"type": "F4a", "dose": {"pulse_ms": 2500}},
                {"type": "H0", "dose": {"pulse_ms": 1500, "period_ms": 2000}},
                {"type": "I1", "dose": {"seconds": 0}},
                binding="sglang",
            )
        )
    problems = error.value.problems
    assert len(problems) == 5
    assert problems[-1] == "no binding 'sglang'"
    assert all(problem.startswith("episodes[") for problem in problems[:4])
    with pytest.raises(PlanError, match="unknown episode type"):
        parse_plan(_plan({"type": "F9"}))
    with pytest.raises(PlanError, match="format is not"):
        parse_plan({"format": "other"})
