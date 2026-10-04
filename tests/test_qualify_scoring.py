"""score_v1: diagnoses scored against injected ground truth (#221 C.2)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import pytest

from stormlog.infer.qualify.ground_truth import (
    Expectation,
    Impact,
    Injection,
    Neutral,
    OutcomeCounts,
    Times,
    Validity,
)
from stormlog.infer.qualify.scoring import (
    MISS_COVERAGE_GAP,
    MISS_INELIGIBLE,
    MISS_MISMATCH,
    MISS_NO_FINDING,
    MISS_OUTRANKED,
    MISS_SECONDARY_ONLY,
    TOP1,
    TOP3,
    EpisodeScore,
    ScoreConfig,
    score_episode,
    summarize,
)
from stormlog.infer.qualify.vocabulary import KIND_COMPONENTS

S = 1_000_000_000
CONFIG = ScoreConfig(default_grace_ns=20 * S, supported_types=frozenset({"F2"}))
KV = "kv_preemption_pressure"
QUEUE = "queue_saturation"


def finding(
    name: str,
    kind: str,
    rank: int,
    *,
    window: tuple[float, float] | None = (102, 140),
    role: str = "primary",
    severity: str = "warning",
    cause: str = "fault",
    claim: str | None = None,
    component: str | None = None,
    location_rank: int | None = None,
    resolution: float = 1,
    secondary_to: Sequence[str] = (),
    failed: Sequence[str] = (),
) -> dict[str, Any]:
    """A ``findings_detail`` entry with the fields the scorer reads."""
    if claim is None:
        claim = "fault" if role == "primary" and cause == "fault" else "condition"
    return {
        "id": f"diagnosis.{kind}.{name:0>12}",
        "kind": kind,
        "role": role,
        "secondary_to": [f"diagnosis.{target}" for target in secondary_to],
        "rank": rank,
        "severity": severity,
        "cause": cause,
        "claim": claim,
        "location": {
            "component": component or sorted(KIND_COMPONENTS[kind])[0],
            "rank": location_rank,
            "engine": "0",
        },
        "window": (
            None
            if window is None
            else {
                "start_ns": int(window[0] * S),
                "end_ns": int(window[1] * S),
                "resolution_ns": int(resolution * S),
                "uncertainty_ns": 0,
            }
        ),
        "eligibility": {"eligible": claim != "observation", "failed": list(failed)},
    }


def diagnosis(*findings: dict[str, Any], coverage: str = "assessed") -> dict[str, Any]:
    kinds = set(KIND_COMPONENTS)
    return {
        "payload": {
            "findings_detail": {entry["id"]: entry for entry in findings},
            "coverage": {kind: {"status": coverage} for kind in kinds},
        }
    }


def episode(
    episode_type: str = "F2",
    *,
    expects: Sequence[Expectation] = (Expectation(KV, "kv_cache"),),
    cause_class: str = "fault",
    status: str = "valid",
    impact: str | None = "impact",
) -> Injection:
    """F2 from #221's catalog: KV pressure, with its two declared secondaries
    and the workload kinds allowed."""
    return Injection(
        episode_id=f"q221-{episode_type}",
        episode_type=episode_type,
        cause_class=cause_class,
        injected={"method": "neighbor_traffic"},
        expects=tuple(expects),
        secondary=(
            Neutral(QUEUE, "scheduler", edge=f"{KV}->{QUEUE}"),
            Neutral(
                "mixed_prefill_interference",
                "scheduler",
                edge=f"{KV}->mixed_prefill_interference",
            ),
        ),
        allows=(Neutral("load_increase", "workload"),),
        times=Times(
            action_onset_ns=100 * S, effect_onset_ns=100 * S, effect_end_ns=150 * S
        ),
        clock_domain="node/boot/unix_epoch_ns",
        status=status,
        validity=Validity(
            impact=(
                None
                if impact is None
                else Impact(impact, OutcomeCounts(9, 31), OutcomeCounts(4, 131))
            )
        ),
    )


def negative(episode_type: str = "N") -> Injection:
    return replace(
        episode(episode_type, expects=(), cause_class="none", impact=None),
        secondary=(),
        allows=(),
    )


def kv_id(name: str = "a") -> str:
    return f"diagnosis.{KV}.{name:0>12}"


# ------------------------------------------------------------------ one episode


def test_the_correct_primary_ranked_first_is_correct_at_top1() -> None:
    score = score_episode(episode(), diagnosis(finding("a", KV, 1)), CONFIG)
    assert score.correct(TOP1, 2) and score.correct(TOP3, 2)
    assert (score.miss, score.false_claims, score.duplicates) == (None, (), 0)
    assert score.localized


def test_an_unrelated_claim_ranked_above_the_correct_one_fails_top1() -> None:
    host = finding("h", "host_stall", 1, component="engine_core")
    score = score_episode(episode(), diagnosis(host, finding("a", KV, 2)), CONFIG)
    assert not score.correct(TOP1, 2)
    assert score.correct(TOP3, 2)
    assert score.match_rank[2] == 2
    # A fault claim the label neither expects nor declares: spurious.
    assert score.false_claims == (host["id"],)


def test_a_false_secondary_through_an_invalid_edge_is_a_false_claim() -> None:
    # #218 has no edge from KV pressure to a host stall, so the secondary is
    # scored as if it were primary.
    stall = finding(
        "h",
        "host_stall",
        2,
        role="secondary",
        component="engine_core",
        secondary_to=[f"{KV}.{'a':0>12}"],
    )
    score = score_episode(episode(), diagnosis(finding("a", KV, 1), stall), CONFIG)
    assert score.correct(TOP1, 2)
    assert stall["id"] in score.candidates
    assert score.false_claims == (stall["id"],)
    assert score.secondary_errors == 1


def test_a_secondary_through_a_valid_edge_is_neutral() -> None:
    queue = finding(
        "q",
        QUEUE,
        1,
        role="secondary",
        window=(110, 130),
        secondary_to=[f"{KV}.{'a':0>12}"],
    )
    score = score_episode(episode(), diagnosis(queue, finding("a", KV, 2)), CONFIG)
    assert score.neutral == (queue["id"],)
    assert score.candidates == (kv_id(),)
    assert score.correct(TOP1, 2)
    assert (score.false_claims, score.secondary_errors) == ((), 0)


def test_a_secondary_outside_its_upstreams_window_is_not_neutral() -> None:
    # The edge holds, but the queue window runs past KV's end plus grace.
    queue = finding(
        "q",
        QUEUE,
        1,
        role="secondary",
        window=(110, 145),
        secondary_to=[f"{KV}.{'a':0>12}"],
    )
    kv = finding("a", KV, 2, window=(102, 120))
    score = score_episode(
        episode(), diagnosis(queue, kv), ScoreConfig(default_grace_ns=20 * S)
    )
    assert score.neutral == ()
    assert score.candidates == (queue["id"], kv["id"])
    # Declared in the label's secondary list, so not a false claim.
    assert score.false_claims == ()
    assert score.secondary_errors == 1


def test_a_run_wide_finding_is_never_a_candidate() -> None:
    run_wide = finding("w", KV, 1, window=(0, 1000))
    score = score_episode(episode(), diagnosis(run_wide), CONFIG)
    assert score.candidates == ()
    assert score.miss == MISS_NO_FINDING


def test_duplicates_are_counted() -> None:
    first, second = finding("a", KV, 1), finding("b", KV, 2, window=(104, 140))
    score = score_episode(episode(), diagnosis(first, second), CONFIG)
    assert score.matched == (first["id"], second["id"])
    assert score.duplicates == 1


@pytest.mark.parametrize(
    "wrong",
    [
        {"cause": "undetermined", "severity": "info"},
        {"severity": "info"},
        {"role": "secondary"},
    ],
    ids=["cause", "severity", "role"],
)
def test_a_mismatched_finding_of_the_right_kind_does_not_match(
    wrong: dict[str, Any]
) -> None:
    score = score_episode(episode(), diagnosis(finding("a", KV, 1, **wrong)), CONFIG)
    assert not score.correct(TOP3, 1)
    expected = MISS_SECONDARY_ONLY if "role" in wrong else MISS_MISMATCH
    assert score.miss == expected
    # The label's own kind is never a false claim, whatever its role.
    assert score.false_claims == ()


def test_a_rank_delay_on_the_wrong_rank_matches_only_at_l1() -> None:
    label = episode("F5", expects=(Expectation("rank_delay", "worker", rank=1),))
    delayed = finding("r", "rank_delay", 1, location_rank=0)
    score = score_episode(label, diagnosis(delayed), CONFIG)
    assert score.correct(TOP1, 1)
    assert not score.correct(TOP1, 2)
    assert score.miss == MISS_MISMATCH


def test_an_ineligible_finding_is_labelled_ineligible() -> None:
    observed = finding(
        "a",
        KV,
        1,
        severity="info",
        cause="undetermined",
        claim="observation",
        failed=["allocation_cause"],
    )
    score = score_episode(episode(), diagnosis(observed), CONFIG)
    assert score.miss == MISS_INELIGIBLE
    assert not score.localized


def test_no_finding_is_a_coverage_gap_when_the_kind_was_not_assessed() -> None:
    assessed = score_episode(episode(), diagnosis(), CONFIG)
    unsupported = score_episode(episode(), diagnosis(coverage="unsupported"), CONFIG)
    assert (assessed.miss, unsupported.miss) == (MISS_NO_FINDING, MISS_COVERAGE_GAP)


@pytest.mark.parametrize(
    ("window", "resolution", "qualifies"),
    [
        ((98, 140), 5, True),  # starts within pre_grace of the onset
        ((94, 140), 5, False),  # starts before it
        ((140, 200), 1, True),  # half inside effect end + grace (170 s)
        ((145, 215), 1, False),  # less than half inside
        (None, 1, False),  # no window to place
    ],
)
def test_the_temporal_rule(
    window: tuple[float, float] | None, resolution: float, qualifies: bool
) -> None:
    candidate = finding("a", KV, 1, window=window, resolution=resolution)
    score = score_episode(episode(), diagnosis(candidate), CONFIG)
    assert (score.candidates == (candidate["id"],)) is qualifies


def test_a_fault_claim_in_a_negative_run_is_a_false_positive() -> None:
    queue = finding("q", QUEUE, 1)
    workload = finding(
        "w", "load_increase", 2, severity="info", cause="workload_change"
    )
    score = score_episode(negative(), diagnosis(queue, workload), CONFIG)
    assert score.false_claims == (queue["id"],)
    assert score.miss is None


# ------------------------------------------------------------------ a campaign


def _scores(correct: int, total: int, episode_type: str = "F2") -> list[EpisodeScore]:
    right = diagnosis(finding("a", KV, 1))
    wrong = diagnosis(finding("h", "host_stall", 1, component="engine_core"))
    label = episode(episode_type)
    return [
        score_episode(label, right if index < correct else wrong, CONFIG)
        for index in range(total)
    ]


def test_a_stratum_passes_at_15_of_15_and_fails_at_14() -> None:
    passing = summarize(_scores(15, 15), CONFIG)
    both = replace(CONFIG, supported_types=frozenset({"F1", "F2"}))
    failing = summarize(_scores(14, 15) + _scores(15, 15, "F1"), both)
    assert passing.accuracy_passes
    assert passing.strata[0].lower_bound == pytest.approx(0.819, abs=5e-4)
    assert not failing.accuracy_passes
    assert [stratum.passes for stratum in failing.strata] == [True, False]
    assert failing.misses == {MISS_NO_FINDING: 1}


def test_only_valid_supported_fault_episodes_count_for_accuracy() -> None:
    invalid = score_episode(
        episode(status="not_realized"), diagnosis(finding("h", "host_stall", 1)), CONFIG
    )
    unsupported = _scores(0, 3, "F6")
    config = ScoreConfig(default_grace_ns=20 * S, supported_types=frozenset({"F2"}))
    summary = summarize(_scores(15, 15) + [invalid] + unsupported, config)
    assert [(s.episode_type, s.episodes) for s in summary.strata] == [("F2", 15)]


def test_the_fpr_is_bounded_over_negative_runs() -> None:
    clean = score_episode(negative(), diagnosis(), CONFIG)
    flagged = score_episode(negative(), diagnosis(finding("q", QUEUE, 1)), CONFIG)
    none_flagged = summarize([clean] * 60, CONFIG, negative_hours=4.9)
    one_flagged = summarize([clean] * 59 + [flagged], CONFIG)
    assert none_flagged.fpr_upper_bound == pytest.approx(0.0487, abs=5e-5)
    assert none_flagged.fpr_passes
    assert none_flagged.false_claims_per_hour_upper == pytest.approx(0.611, abs=5e-4)
    assert (one_flagged.false_positive_runs, one_flagged.fpr_passes) == (1, False)


def test_a_declared_stratum_without_valid_episodes_fails_the_gate() -> None:
    # Every F2 episode failed realization: the stratum has no evidence, so
    # the gate must not pass on F1 alone.
    both = replace(CONFIG, supported_types=frozenset({"F1", "F2"}))
    f1 = episode("F1", expects=(Expectation(QUEUE, "scheduler"),))
    right = [score_episode(f1, diagnosis(finding("q", QUEUE, 1)), both)] * 15
    unrealized = [score_episode(episode(status="not_realized"), diagnosis(), both)] * 15
    summary = summarize(right + unrealized, both)
    f1_stratum, f2_stratum = summary.strata
    assert f1_stratum.passes
    assert (f2_stratum.episodes, f2_stratum.lower_bound) == (0, None)
    assert f2_stratum.excluded == {"not_realized": 15}
    assert not f2_stratum.passes and not summary.accuracy_passes


def test_a_gated_summary_needs_the_support_matrix() -> None:
    with pytest.raises(ValueError, match="supported_types"):
        summarize(_scores(15, 15), ScoreConfig(default_grace_ns=20 * S))


def test_incident_attribution_counts_only_episodes_with_impact() -> None:
    hurt = score_episode(episode(), diagnosis(finding("a", KV, 1)), CONFIG)
    unhurt = score_episode(
        episode(impact="no_impact"), diagnosis(finding("a", KV, 1)), CONFIG
    )
    outranked = score_episode(
        episode(),
        diagnosis(
            finding("h", "host_stall", 1, component="engine_core"), finding("a", KV, 2)
        ),
        CONFIG,
    )
    summary = summarize([hurt, unhurt, outranked], CONFIG)
    assert summary.attributed == (1, 2)
    assert summary.localized == (3, 3)
    assert summary.misses == {MISS_OUTRANKED: 1}
    assert summary.spurious == 1


def test_the_summary_records_what_was_frozen() -> None:
    record = summarize(_scores(15, 15), CONFIG).to_record(CONFIG)
    assert record["score"] == "score_v1"
    assert record["edge_table"] == "diagnosis_edges_v1"
    assert (record["gated_metric"], record["gated_level"]) == ("top1", "L2")
    assert (record["accuracy_floor"], record["fpr_ceiling"]) == (0.78, 0.05)
    episode_record = _scores(1, 1)[0].to_record()
    assert episode_record["match_rank"] == {"L1": 1, "L2": 1}
