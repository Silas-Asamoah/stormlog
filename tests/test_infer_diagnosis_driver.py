"""What drove a finding: load, capacity, or neither shown."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_driver import driver_of
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_selection import SelectionOptions, select
from tests.diagnosis_scenarios import MS, Engine, SimRequest, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET

AT = 90 * SECOND


def _subject(
    tmp_path: Path, requests: list[SimRequest], engine: Engine, window: tuple[int, int]
) -> tuple[Context, Any]:
    view = join(read_input(build_run(tmp_path, requests, engine)))
    declared = ((window[0] + WALL_OFFSET, window[1] + WALL_OFFSET),)
    context = Context(view, select(view, SelectionOptions(windows=declared)))
    (subject,) = context.subjects()
    return context, subject


def test_more_arrivals_at_the_same_pace_are_load(tmp_path: Path) -> None:
    """Calm traffic, then a burst arriving 25 times as fast to an engine
    with room for it. The batches differ, so the capacity comparison has no
    support: load stands, and the rubric says what it lacked."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(120, AT, 20 * MS, prefix="b")
    window = (AT, AT + 3 * SECOND)

    context, subject = _subject(
        tmp_path, calm + burst, Engine(max_num_seqs=256), window
    )
    driver = driver_of(context, subject)

    assert driver.driver == "load"
    assert driver.evidence["demand"]["arrival_rate_ratio"]["passes"]
    assert driver.confidence.met == ("compatible_reference",)
    assert driver.confidence.unmet == ("matched_common_support",)


def _steady(start: int, end: int, gap: int, prefix: str) -> list[SimRequest]:
    count = (end - start) // gap
    return poisson_free(count, start, gap, prefix=prefix, prompt=8, output=8)


def test_a_slower_engine_at_the_same_work_is_capacity(tmp_path: Path) -> None:
    """A stream the engine is always full with; from 20 s every step takes
    half again as long. The subject's steps are compared with the
    reference's at the same batch and work, and are 1.5 times slower. (Had
    the engine had room before, the slower one would run fuller batches
    than any of the reference's, and nothing would match: capacity is
    claimed only at matched work.)"""
    requests = _steady(16 * SECOND, 24 * SECOND, 8 * MS, "s")
    engine = Engine(max_num_seqs=8, slower=(20 * SECOND, 1.5))

    context, subject = _subject(tmp_path, requests, engine, (20 * SECOND, 22 * SECOND))
    driver = driver_of(context, subject)

    capacity = driver.evidence["capacity"]
    assert driver.driver == "capacity", capacity
    assert capacity["ratio"]["estimate"] == pytest.approx(1.5, abs=0.05)
    assert capacity["common_support"] >= 0.8
    assert driver.confidence.unmet == ()


def test_neither_shown_is_undetermined(tmp_path: Path) -> None:
    """The same full engine at the same pace throughout."""
    requests = _steady(16 * SECOND, 24 * SECOND, 8 * MS, "s")

    context, subject = _subject(
        tmp_path, requests, Engine(max_num_seqs=8), (20 * SECOND, 22 * SECOND)
    )
    driver = driver_of(context, subject)

    assert driver.driver == "undetermined"
    assert driver.evidence["capacity"]["screened"] == "not_above_floor"


def test_another_client_s_load_is_load(tmp_path: Path) -> None:
    """The run's own arrivals are unchanged; another client's requests
    join the engine's steps."""
    mine = _steady(10 * SECOND, 30 * SECOND, 100 * MS, "s")
    theirs = poisson_free(
        60, 22 * SECOND, 50 * MS, prefix="f", run="other", prompt=8, output=40
    )

    context, subject = _subject(
        tmp_path, mine + theirs, Engine(max_num_seqs=64), (22 * SECOND, 26 * SECOND)
    )
    driver = driver_of(context, subject)

    foreign = driver.evidence["demand"]["foreign_share"]
    assert foreign["passes"] and foreign["subject"] > foreign["reference"]
    assert driver.driver == "load"


def test_the_report_says_what_drove_each_finding(tmp_path: Path) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(120, AT, 20 * MS, prefix="b")
    artifact = build_run(tmp_path, calm + burst, Engine(max_num_seqs=8))
    window = (AT + WALL_OFFSET, AT + 3 * SECOND + WALL_OFFSET)

    report = diagnose_artifact(
        artifact, windows=[window], options=DiagnoseOptions(generated_at_ns=1)
    )

    details = report["payload"]["findings_detail"].values()
    queue = next(d for d in details if d["kind"] == "queue_saturation")
    assert queue["driver"] == "load" and queue["driver_evidence"]["compatible"]
    assert queue["driver_evidence"]["demand_passed"] == ["arrival_rate_ratio"]
    assert queue["confidence"]["driver"] == {
        "level": "medium",
        "met": ["compatible_reference"],
        "unmet": ["matched_common_support"],
        "coverage": "observed",
    }
