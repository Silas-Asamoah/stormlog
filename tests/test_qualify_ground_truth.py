"""The qualification's ground-truth format, its validity layers and status."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.qualify import vocabulary
from stormlog.infer.qualify.ground_truth import (
    IMPACT,
    IMPACT_PARTIAL,
    INCOMPARABLE,
    INVALID_ALIGNMENT,
    NO_IMPACT,
    NOT_ACTUATED,
    NOT_REALIZED,
    PROTOCOL_FAILURE,
    RECOVERY_INCOMPLETE,
    VALID,
    GroundTruthError,
    Interval,
    OutcomeCounts,
    PhaseWindow,
    RunRecord,
    Times,
    assess_impact,
    decide_status,
    is_aligned,
    load_injections,
    load_run,
    parse_injection,
    parse_run,
    write_injections,
    write_run,
)

S = 1_000_000_000


def f2_record() -> dict[str, Any]:
    """#221 A.5's KV-pressure episode, with v3.2's additions."""
    return {
        "format": "stormlog.qualify.injection/1",
        "episode_id": "q221-0f3a9c1b2d4e5f60",
        "run_id": "q221-dxoff-b03-r07",
        "episode_type": "F2",
        "cause_class": "fault",
        "injected": {
            "method": "neighbor_traffic",
            "target": {"role": None, "pid": None, "start_ns": None},
            "dose": {"concurrency": 8, "input_tokens": 2048, "output_tokens": 1024},
        },
        "expects": [
            {
                "kind": "kv_preemption_pressure",
                "component": "kv_cache",
                "rank": None,
                "engine": None,
                "role": "primary",
                "cause": "fault",
                "min_severity": "warning",
            }
        ],
        "secondary": [
            {
                "kind": "queue_saturation",
                "component": "scheduler",
                "rank": None,
                "engine": None,
                "edge": "kv_preemption_pressure->queue_saturation",
            },
            {
                "kind": "mixed_prefill_interference",
                "component": "scheduler",
                "rank": None,
                "engine": None,
                "edge": "kv_preemption_pressure->mixed_prefill_interference",
            },
        ],
        "allows": [
            {"kind": kind, "component": "workload", "rank": None, "engine": None}
            for kind in ("load_increase", "longer_inputs", "longer_outputs")
        ],
        "times": {
            "action_onset_ns": 100 * S,
            "action_end_ns": 145 * S,
            "effect_onset_ns": 103 * S,
            "effect_end_ns": 150 * S,
            "effect_basis": "reference_hook_preempted_victim",
            "first_observation_ns": 104 * S,
            "recovery_held_at_ns": 172 * S,
            "predicate_duration_ns": None,
            "priming_check": {"cached_fraction_median": 0.94, "passed": True},
        },
        "clock_domain": "node-a/boot-1/unix_epoch_ns",
        "actions": [
            {
                "kind": "neighbor_phase",
                "at_wall_ns": 100 * S,
                "at_mono_ns": 7,
                "result": "ok",
            }
        ],
        "validity": {
            "actuation": "ok",
            "realization": "realized",
            "observation": "complete",
            "impact": {
                "status": "impact",
                "reason": None,
                "p_value": 0.001,
                "effect": {"violations": 9, "met": 31, "unknown": 0},
                "baseline": {"violations": 4, "met": 131, "unknown": 0},
            },
            "realized_mechanisms": ["kv_preemption_pressure", "queue_saturation"],
            "checks": [
                {"layer": "realization", "name": "victim_preempted", "passed": True}
            ],
        },
        "status": "valid",
    }


def test_a_record_round_trips() -> None:
    record = f2_record()
    injection = parse_injection(record)
    assert injection.to_record() == record
    assert injection.expects[0].kind == "kv_preemption_pressure"
    assert injection.validity.impact is not None
    assert injection.validity.impact.status == IMPACT


def test_records_round_trip_through_a_file(tmp_path: Path) -> None:
    first = parse_injection(f2_record())
    record = f2_record()
    record["episode_id"] = "q221-1111111111111111"
    record["status"] = "not_realized"
    second = parse_injection(record)
    path = tmp_path / "injections.jsonl"
    write_injections(path, [first, second])
    assert load_injections(path) == [first, second]


def _broken(change: Any) -> list[str]:
    record = copy.deepcopy(f2_record())
    change(record)
    with pytest.raises(GroundTruthError) as error:
        parse_injection(record)
    return error.value.problems


def test_labels_use_218s_vocabulary_and_edges() -> None:
    def unknown_kind(record: dict[str, Any]) -> None:
        record["expects"][0]["kind"] = "kv_pressure"

    def wrong_place(record: dict[str, Any]) -> None:
        record["expects"][0]["component"] = "scheduler"

    def no_edge(record: dict[str, Any]) -> None:
        del record["secondary"][0]["edge"]

    def unknown_edge(record: dict[str, Any]) -> None:
        record["secondary"][0]["edge"] = "queue_saturation->kv_preemption_pressure"

    def edge_elsewhere(record: dict[str, Any]) -> None:
        record["secondary"][0][
            "edge"
        ] = "kv_preemption_pressure->mixed_prefill_interference"

    def secondary_expectation(record: dict[str, Any]) -> None:
        record["expects"][0]["role"] = "secondary"

    assert _broken(unknown_kind) == ["expects[0]: unknown kind 'kv_pressure'"]
    assert _broken(wrong_place) == [
        "expects[0]: kv_preemption_pressure is never at 'scheduler'"
    ]
    assert _broken(no_edge) == ["secondary[0]: a secondary names its edge"]
    assert _broken(unknown_edge) == [
        "secondary[0]: unknown edge 'queue_saturation->kv_preemption_pressure'"
    ]
    assert _broken(edge_elsewhere) == [
        "secondary[0]: edge kv_preemption_pressure->mixed_prefill_interference"
        " does not lead to queue_saturation"
    ]
    assert _broken(secondary_expectation) == [
        "expects[0]: an expectation is always primary"
    ]


def test_a_malformed_record_is_refused_with_its_line(tmp_path: Path) -> None:
    def extra_time(record: dict[str, Any]) -> None:
        record["times"]["when"] = 1

    def bad_status(record: dict[str, Any]) -> None:
        record["status"] = "maybe"

    assert _broken(extra_time)[0].startswith("malformed record")
    assert _broken(bad_status) == ["unknown status 'maybe'"]
    path = tmp_path / "injections.jsonl"
    path.write_text('{"format": "other"}\n', encoding="utf-8")
    with pytest.raises(GroundTruthError, match="line 1: format is not"):
        load_injections(path)


def test_the_status_is_the_first_failure_in_order() -> None:
    assert decide_status() == VALID
    assert decide_status(recovered=False) == RECOVERY_INCOMPLETE
    assert decide_status(realized=False, recovered=False) == NOT_REALIZED
    assert decide_status(aligned=False, realized=False) == INVALID_ALIGNMENT
    assert decide_status(actuated=False, aligned=False) == NOT_ACTUATED
    assert decide_status(same_clock=False, actuated=False) == INCOMPARABLE
    assert decide_status(protocol_failure=True, same_clock=False) == PROTOCOL_FAILURE


def test_an_episode_needs_its_whole_effect_and_a_clean_baseline_in_the_window() -> None:
    window = PhaseWindow(start_ns=0, end_ns=400 * S)
    inside = Times(
        action_onset_ns=100 * S, action_end_ns=145 * S, effect_end_ns=150 * S
    )
    assert is_aligned(inside, window)
    # The effect runs past the measured window.
    late = Times(action_onset_ns=100 * S, effect_end_ns=401 * S)
    assert not is_aligned(late, window)
    # Starts before the window.
    early = Times(action_onset_ns=-1, effect_end_ns=10 * S)
    assert not is_aligned(early, window)
    # Only 20 s after the previous episode recovered.
    assert not is_aligned(inside, window, clean_since_ns=80 * S)
    assert is_aligned(inside, window, clean_since_ns=70 * S)
    # No action time recorded.
    assert not is_aligned(Times(effect_end_ns=150 * S), window)


def test_impact_needs_more_violations_than_the_baseline_and_at_least_three() -> None:
    baseline = OutcomeCounts(violations=6, met=129, unknown=0)
    hurt = assess_impact(OutcomeCounts(violations=9, met=31), baseline)
    assert (hurt.status, hurt.reason) == (IMPACT, None)
    assert hurt.p_value is not None and hurt.p_value < 0.05
    # Two violations out of two is a striking rate, but fewer than three.
    few = assess_impact(OutcomeCounts(violations=2, met=0), OutcomeCounts(0, 135))
    assert few.status == NO_IMPACT
    same = assess_impact(OutcomeCounts(violations=2, met=43), baseline)
    assert same.status == NO_IMPACT


def test_impact_is_partial_when_slo_evidence_is_thin() -> None:
    # Unknown outcomes are not violations, but they leave the window's
    # evidence covering less than 0.9 of its requests.
    thin = assess_impact(
        OutcomeCounts(violations=9, met=31, unknown=5), OutcomeCounts(4, 131)
    )
    assert (thin.status, thin.reason) == (IMPACT_PARTIAL, "slo_evidence_coverage")
    empty = assess_impact(OutcomeCounts(), OutcomeCounts(4, 131))
    assert empty.status == IMPACT_PARTIAL


def test_the_vocabulary_matches_218s_when_present() -> None:
    theirs = pytest.importorskip("stormlog.infer.diagnosis_vocabulary")
    assert dict(vocabulary.KIND_COMPONENTS) == dict(theirs.KIND_COMPONENTS)
    assert vocabulary.CAUSES == theirs.CAUSES
    assert vocabulary.WORKLOAD_KINDS == theirs.WORKLOAD_KINDS


def test_the_component_exclusions_match_218s_when_present() -> None:
    theirs = pytest.importorskip("stormlog.infer.diagnosis")
    assert set(vocabulary.NOT_ASSESSED_COMPONENTS) == set(theirs.NOT_YET.values())


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (
            lambda r: r["validity"].update(actuation="failed"),
            "status valid, but actuation is 'failed'",
        ),
        (
            lambda r: r["validity"].update(realization="not_realized"),
            "status valid, but realization is 'not_realized'",
        ),
        (lambda r: r.update(expects=[]), "a fault episode expects exactly one finding"),
        (
            lambda r: r["expects"][0].update(min_severity="info"),
            "expects[0]: a fault is claimed at warning",
        ),
        (
            lambda r: r["expects"][0].update(cause="undetermined"),
            "expects[0]: a fault episode expects cause 'fault'",
        ),
        (
            lambda r: r.update(cause_class="none"),
            "a none episode expects no fault",
        ),
        (
            lambda r: r["times"].update(effect_onset_ns=160 * S),
            "the effect ends before it begins",
        ),
        (
            lambda r: r["times"].update(effect_onset_ns=None),
            "a valid episode needs its effect onset and end",
        ),
        (
            lambda r: r["times"].update(effect_end_ns="150"),
            "times.effect_end_ns must be an integer or null",
        ),
        (
            lambda r: r["expects"][0].update(
                kind="load_increase", component="workload"
            ),
            "expects[0]: load_increase is a workload kind, claimed as workload_change"
            " at info",
        ),
    ],
)
def test_ground_truth_that_would_misscore_is_refused(change: Any, problem: str) -> None:
    assert problem in _broken(change)


def test_a_negative_may_expect_its_workload_kind() -> None:
    record = f2_record()
    record.update(
        episode_type="T1",
        cause_class="workload_change",
        secondary=[],
        expects=[
            {
                "kind": "load_increase",
                "component": "workload",
                "rank": None,
                "engine": None,
                "role": "primary",
                "cause": "workload_change",
                "min_severity": "info",
            }
        ],
    )
    assert parse_injection(record).expects[0].kind == "load_increase"


def test_an_episode_written_twice_is_refused(tmp_path: Path) -> None:
    injection = parse_injection(f2_record())
    path = tmp_path / "injections.jsonl"
    write_injections(path, [injection, injection])
    with pytest.raises(GroundTruthError, match="line 2: episode q221-0f3a9c1b2d4e5f60"):
        load_injections(path)


def test_a_run_record_round_trips(tmp_path: Path) -> None:
    run = RunRecord(
        run_id="q221-dxoff-b03-r07",
        clock_domain="node-a/boot-1/unix_epoch_ns",
        measured=Interval(0, 420 * S),
        priming=Interval(0, 30 * S),
        baseline=Interval(30 * S, 75 * S),
        final_recovery=Interval(360 * S, 420 * S),
    )
    path = tmp_path / "run.json"
    write_run(path, run)
    assert load_run(path) == run
    record = run.to_record()
    record["priming"] = {"start_ns": 30 * S, "end_ns": 0}
    with pytest.raises(GroundTruthError, match="priming ends before it begins"):
        parse_run(record)
    with pytest.raises(GroundTruthError, match="format"):
        parse_run({"format": "other"})


def test_impact_without_baseline_outcomes_is_partial() -> None:
    # With nothing known in the baseline there is nothing to compare with:
    # Fisher's p of 1.0 would read as no impact.
    alone = assess_impact(
        OutcomeCounts(violations=9, met=31), OutcomeCounts(unknown=12)
    )
    assert (alone.status, alone.reason) == (IMPACT_PARTIAL, "no_baseline_outcomes")


def test_the_capture_pause_edge_is_218s_without_a_narrower_location() -> None:
    # #218 v2.1 §4.6 puts no location on capture_pause -> host_stall; the
    # copy must not narrow it. The I1 label's allows say where #221 expects
    # the stall.
    edge = vocabulary.EDGES["capture_pause->host_stall"]
    assert edge.downstream_components == vocabulary.KIND_COMPONENTS["host_stall"]
