"""The rules that grade and rank a diagnosis finding."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer import diagnosis_model as model
from stormlog.infer.diagnosis import link_edges
from stormlog.infer.diagnosis_inputs import Line
from stormlog.infer.diagnosis_model import (
    CONTRIBUTING,
    NOT_RULED_OUT,
    PARTIAL,
    RULED_OUT,
    SECONDARY,
    UNTESTABLE,
    UPSTREAM,
    Alternative,
    Criteria,
    Finding,
    rank_findings,
    support_block,
)
from stormlog.infer.diagnosis_stats import block_length, median_difference


def _finding(**changes: Any) -> Finding:
    values: dict[str, Any] = {
        "kind": "queue_saturation",
        "component": "scheduler",
        "subject": {"key": "window:c1:0"},
        "title": "t",
        "message": "m",
        "gates": {"capacity_witness": True},
        "alternatives": [Alternative("engine_stall", RULED_OUT, "r", True)],
        "condition": Criteria(met=("direct_evidence",)),
        "contribution": Criteria(
            met=("excess_ci_excludes_zero", "explains_ttft_excess")
        ),
        "incident": True,
        "explains": "explains_ttft_excess",
    }
    values.update(changes)
    return Finding(**values)


def test_a_claim_s_level_counts_what_it_missed() -> None:
    assert Criteria(met=("a", "b")).level == "high"
    assert Criteria(met=("a",), unmet=("b",)).level == "medium"
    assert Criteria(met=("a",), coverage_unknown=True).level == "medium"
    assert Criteria(unmet=("a", "b")).level == "low"
    assert Criteria(met=("a", "b"), known_loss=True).level == "low"


def test_an_eligible_strong_incident_finding_is_a_fault_warning() -> None:
    finding = _finding()

    assert finding.eligible and finding.failed_gates == []
    assert (finding.severity, finding.cause, finding.claim) == (
        "warning",
        "fault",
        "fault",
    )
    assert finding.confidence_level == "high"


@pytest.mark.parametrize(
    "changes, failed",
    [
        ({"gates": {"capacity_witness": False}}, ["capacity_witness"]),
        (
            {"alternatives": [Alternative("blocked_waiting", UNTESTABLE, "r", True)]},
            ["competitor:blocked_waiting:untestable"],
        ),
        (
            {"alternatives": [Alternative("engine_stall", NOT_RULED_OUT, "r", True)]},
            ["competitor:engine_stall:not_ruled_out"],
        ),
    ],
)
def test_a_failed_gate_makes_an_observation_at_info(
    changes: dict[str, Any], failed: list[str]
) -> None:
    # Astra's case: a positive excess and enough samples are not enough when
    # an indispensable competitor could not be tested.
    finding = _finding(**changes)

    assert not finding.eligible and finding.failed_gates == failed
    assert (finding.severity, finding.cause, finding.claim) == (
        "info",
        "undetermined",
        "observation",
    )


def test_a_competitor_that_is_not_indispensable_only_costs_confidence() -> None:
    alternatives = [
        Alternative("engine_stall", RULED_OUT, "r", True),
        Alternative("client_admission", UNTESTABLE, "r"),
    ]
    assert _finding(alternatives=alternatives).eligible


def test_severity_needs_contribution_and_an_incident() -> None:
    weak = Criteria(unmet=("excess_ci_excludes_zero", "witness"))
    assert _finding(contribution=weak).severity == "info"
    assert _finding(incident=False).severity == "info"
    # Eligible but info: a condition, with the driver undetermined.
    finding = _finding(incident=False)
    assert (finding.cause, finding.claim) == ("undetermined", "condition")


@pytest.mark.parametrize(
    "changes",
    [
        # The excess is real, but the mechanism explains too little of it.
        {
            "contribution": Criteria(
                met=("excess_ci_excludes_zero",), unmet=("explains_ttft_excess",)
            )
        },
        # A kind that names no criterion for explaining the incident.
        {"explains": None},
        # The mechanism itself is barely shown.
        {"condition": Criteria(unmet=("direct_evidence", "sufficient_samples"))},
    ],
)
def test_a_warning_needs_the_mechanism_shown_and_explaining_the_incident(
    changes: dict[str, Any],
) -> None:
    finding = _finding(**changes)

    assert finding.eligible
    assert (finding.severity, finding.cause, finding.claim) == (
        "info",
        "undetermined",
        "condition",
    )


def test_a_contributing_competitor_leaves_the_claim_a_condition() -> None:
    finding = _finding(
        alternatives=[Alternative("engine_stall", CONTRIBUTING, "r", True)]
    )

    assert finding.eligible and finding.failed_gates == []
    assert finding.contested == ["competitor:engine_stall:contributing"]
    assert (finding.severity, finding.cause, finding.claim) == (
        "info",
        "undetermined",
        "condition",
    )


@pytest.mark.parametrize("kv_severity_held", [True, False])
def test_a_queue_is_secondary_only_to_a_kv_fault_it_names_upstream(
    kv_severity_held: bool,
) -> None:
    kv = _finding(
        kind="kv_preemption_pressure",
        component="kv_cache",
        explains="explains_e2e_excess" if kv_severity_held else None,
        contribution=Criteria(met=("excess_ci_excludes_zero", "explains_e2e_excess")),
    )
    queue = _finding(
        alternatives=[
            Alternative("engine_stall", RULED_OUT, "r", True),
            Alternative("kv_preemption_pressure", UPSTREAM, "r"),
        ]
    )

    link_edges([kv, queue], "run-1")

    if kv_severity_held:
        assert (queue.role, queue.secondary_to) == (SECONDARY, [kv.identity("run-1")])
        assert queue.claim == "condition"
    else:
        assert (queue.role, queue.secondary_to) == ("primary", [])


def test_workload_and_instrumentation_kinds_keep_their_causes() -> None:
    workload = _finding(kind="load_increase", component="workload")
    assert (workload.severity, workload.cause) == ("info", "workload_change")
    capture = _finding(kind="capture_pause", component="profiler")
    assert (capture.severity, capture.cause) == ("warning", "instrumentation")
    assert capture.claim == "condition"


def test_partial_assessments_never_reach_high_confidence() -> None:
    assert _finding(status=PARTIAL).confidence_level == "medium"


def test_identity_is_stable_and_names_its_subject() -> None:
    first, again = _finding(), _finding()
    other = _finding(subject={"key": "window:c1:9"})

    assert first.identity("run-1") == again.identity("run-1")
    assert first.identity("run-1") != other.identity("run-1")
    assert first.identity("run-1").startswith("diagnosis.queue_saturation.")


def test_findings_rank_in_one_total_order() -> None:
    eligible = _finding()
    secondary = _finding(role=SECONDARY, subject={"key": "b"})
    observation = _finding(gates={"capacity_witness": False}, subject={"key": "c"})
    weaker = _finding(contribution_lower=1.0, subject={"key": "d"})
    stronger = _finding(contribution_lower=5.0, subject={"key": "e"})

    ranked = [
        f
        for _, f in rank_findings(
            [secondary, observation, weaker, eligible, stronger], "r"
        )
    ]

    assert ranked[0] is stronger and ranked[1] is weaker
    assert ranked.index(eligible) < ranked.index(observation) < ranked.index(secondary)
    # Equal on every criterion: the id decides, so the order never depends
    # on the input order.
    twins = [_finding(subject={"key": k}) for k in ("x", "y")]
    assert rank_findings(twins, "r") == rank_findings(list(reversed(twins)), "r")


def _line(number: int) -> Line:
    return Line(
        number,
        f"{number:064x}",
        {"event_type": "infer.request"},
        None,
        f"infer.request/r{number}",
    )


def test_support_is_identity_triples_then_ranges_above_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [_line(5), _line(3), _line(4), _line(3)]

    block = support_block(lines)

    assert block == {
        "support_identity": "triples",
        "lines": [[n, f"infer.request/r{n}", f"{n:064x}"] for n in (3, 4, 5)],
    }
    monkeypatch.setattr(model, "SUPPORT_LIMIT", 2)
    capped = support_block([*lines, _line(9)])
    assert capped["support_identity"] == "ranges_only"
    assert capped["ranges"] == [[3, 5], [9, 9]] and capped["count"] == 4


def test_median_differences_are_seeded_and_need_twenty_per_arm() -> None:
    subject = [float(v) for v in range(100, 140)]
    reference = [float(v) for v in range(0, 40)]

    first = median_difference(subject, reference)
    assert first is not None and first == median_difference(subject, reference)
    assert first.estimate == 100.0 and first.low <= 100.0 <= first.high
    assert first.excludes_zero
    assert median_difference(subject[:19], reference) is None


def test_a_queue_s_ramp_keeps_its_dependence_in_the_interval() -> None:
    """Waits in a burst rise one after another; resampling single waits
    treats them as independent and gives a narrower interval than runs of
    consecutive ones (7 for 300 values) do."""
    ramp = [float(wait) for wait in range(300)]
    reference = [float(index % 7) for index in range(200)]

    blocked = median_difference(ramp, reference)
    single = median_difference(ramp, reference, blocks=False)

    assert block_length(300) == 7 and block_length(20) == 3
    assert blocked is not None and single is not None
    assert blocked.estimate == single.estimate
    assert blocked.high - blocked.low > 2 * (single.high - single.low)
