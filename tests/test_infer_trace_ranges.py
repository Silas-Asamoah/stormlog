"""The iteration range naming convention shared by engines and trace importers."""

from __future__ import annotations

import pytest

from stormlog.infer.correlation_events import EntityRef
from stormlog.infer.trace_ranges import (
    iteration_range,
    iteration_range_name,
    parse_iteration_range,
)


def test_name_and_parse_round_trip() -> None:
    name = iteration_range_name("vllm-engine-0", "step/42")

    assert name == "stormlog.iteration/vllm-engine-0/step/42"
    assert parse_iteration_range(name) == EntityRef("vllm-engine-0", "step/42")


@pytest.mark.parametrize(
    "name",
    [
        "execute_context_1(512)_generation_0(0)",
        "stormlog.iteration/",
        "stormlog.iteration/producer-only",
        "stormlog.iteration//iteration-only",
    ],
)
def test_other_or_incomplete_names_are_not_iterations(name: str) -> None:
    assert parse_iteration_range(name) is None


@pytest.mark.parametrize(
    ("producer_id", "iteration_id"),
    [("", "1"), ("a/b", "1"), ("engine", ""), ("engine", "1\n2")],
)
def test_names_reject_ambiguous_identities(producer_id: str, iteration_id: str) -> None:
    with pytest.raises(ValueError):
        iteration_range_name(producer_id, iteration_id)


def test_iteration_range_runs_its_body_without_a_profiler() -> None:
    calls = []
    with iteration_range("engine", "1"):
        calls.append("ran")
    with iteration_range("engine", "2", nvtx=True):
        calls.append("ran")

    assert calls == ["ran", "ran"]
