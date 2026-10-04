"""score_v1: diagnoses scored against injected ground truth (#221 C.2)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest

from stormlog.infer.qualify.ground_truth import (
    Expectation,
    Impact,
    Injection,
    Interval,
    Neutral,
    OutcomeCounts,
    RunRecord,
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
    RunScore,
    ScoreConfig,
    negative_exposure,
    score_episode,
    score_run,
    summarize,
)
from stormlog.infer.qualify.vocabulary import KIND_COMPONENTS

S = 1_000_000_000
CONFIG = ScoreConfig(default_grace_ns=20 * S, supported_types=frozenset({"F2"}))
KV = "kv_preemption_pressure"
QUEUE = "queue_saturation"
PRODUCER = "vllm:node-7:boot-aaaa:2600:1790000000000000000"
# A real #218 payload (PR 1b): a queue saturation finding and its coverage.
REAL_218 = (
    Path(__file__).parent / "fixtures" / "qualify" / "diagnosis_218_queue_burst.json"
)


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
            "engine_producer": PRODUCER,
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
        run_id="q221-run",
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
    # A.4: a declared secondary is neutral only through its valid edge, in
    # time and place; otherwise it is scored as a primary, so a false claim.
    assert score.false_claims == (queue["id"],)
    assert score.secondary_errors == 1


def test_a_declared_secondary_kind_claimed_as_a_primary_fault_is_false() -> None:
    # T2 declares mixed-prefill interference as a secondary; a primary fault
    # claim of it, with no upstream at all, is a false positive.
    twin = replace(
        negative("T2"),
        cause_class="workload_change",
        expects=(
            Expectation(
                "longer_inputs",
                "workload",
                cause="workload_change",
                min_severity="info",
            ),
        ),
        secondary=(
            Neutral(
                "mixed_prefill_interference",
                "scheduler",
                edge=f"{KV}->mixed_prefill_interference",
            ),
        ),
        allows=(Neutral("load_increase", "workload"),),
    )
    claim = finding("m", "mixed_prefill_interference", 1, component="scheduler")
    score = score_episode(twin, diagnosis(claim), CONFIG)
    assert score.false_claims == (claim["id"],)


def test_an_allowed_finding_is_never_a_false_claim() -> None:
    # allows entries are neutral whatever their role: here a fault claim of
    # a kind the label allows (a contrived one, to isolate the rule).
    allowing = replace(episode(), allows=(Neutral("host_stall", "engine_core"),))
    stall = finding("h", "host_stall", 1, component="engine_core")
    score = score_episode(allowing, diagnosis(stall), CONFIG)
    assert score.false_claims == ()


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


def test_l2_reads_218s_own_location_fields() -> None:
    # #218 names the engine as location.engine_producer; a label naming that
    # engine matches at L2, and one naming another engine only at L1.
    real = json.loads(REAL_218.read_text(encoding="utf-8"))
    (queue,) = [
        detail
        for detail in real["payload"]["findings_detail"].values()
        if detail["kind"] == QUEUE
    ]
    window = queue["window"]
    label = replace(
        episode("F1", expects=(Expectation(QUEUE, "scheduler", engine=PRODUCER),)),
        secondary=(),
        clock_domain=window["clock_domain"],
        times=Times(
            action_onset_ns=window["start_ns"],
            effect_onset_ns=window["start_ns"],
            effect_end_ns=window["end_ns"],
        ),
    )
    assert queue["location"]["engine_producer"] == PRODUCER
    assert score_episode(label, real, CONFIG).correct(TOP1, 2)
    elsewhere = replace(
        label, expects=(Expectation(QUEUE, "scheduler", engine="vllm:other:1:1:1"),)
    )
    other = score_episode(elsewhere, real, CONFIG)
    assert other.correct(TOP1, 1) and not other.correct(TOP1, 2)


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


def test_a_mismatched_primary_outweighs_a_secondary_in_the_miss_label() -> None:
    # A primary of the label's kind at the wrong severity, plus the same kind
    # as a secondary: the label is mismatch, since the episode doesn't have
    # the kind only as a secondary.
    weak = finding("a", KV, 1, severity="info")
    secondary = finding("b", KV, 2, role="secondary")
    score = score_episode(episode(), diagnosis(weak, secondary), CONFIG)
    assert score.miss == MISS_MISMATCH


def _real_coverage(label: Expectation) -> str | None:
    real = json.loads(REAL_218.read_text(encoding="utf-8"))
    nothing = {
        "payload": {"findings_detail": {}, "coverage": real["payload"]["coverage"]}
    }
    return score_episode(replace(episode(), expects=(label,)), nothing, CONFIG).miss


def test_a_kind_assessed_at_the_labels_component_is_no_coverage_gap() -> None:
    # #218 PR 1b reports host_stall as partial: engine_core and worker aren't
    # assessed by this version, but api_server is. An F4b miss is then
    # no_finding; an F4a miss is a coverage gap.
    assert _real_coverage(Expectation("host_stall", "api_server")) == MISS_NO_FINDING
    assert _real_coverage(Expectation("host_stall", "engine_core")) == MISS_COVERAGE_GAP


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


def test_a_finding_outside_218s_vocabulary_is_refused_by_name() -> None:
    # #218's severities and roles are closed: an unknown one is a schema
    # mismatch, refused with the finding named, not a crash deep in a rule.
    odd = finding("a", KV, 1, severity="critical")
    with pytest.raises(
        ValueError,
        match="diagnosis.kv_preemption_pressure.00000000000a: unknown severity 'critical'",
    ):
        score_episode(episode(), diagnosis(odd), CONFIG)
    sideways = finding("a", KV, 1, role="tertiary")
    with pytest.raises(ValueError, match="unknown role 'tertiary'"):
        score_episode(episode(), diagnosis(sideways), CONFIG)


def test_a_window_that_ends_before_it_starts_never_qualifies() -> None:
    backwards = finding("a", KV, 1, window=(140, 102))
    assert score_episode(episode(), diagnosis(backwards), CONFIG).candidates == ()
    stall = finding("h", "host_stall", 1, component="engine_core", window=(280, 250))
    assert run_of([null_run()], diagnosis(stall)).false_claims == ()


def test_a_window_on_another_clock_never_qualifies() -> None:
    # The scoring window is on the victim's clock; a finding placed on
    # another clock domain can't be compared with it.
    elsewhere = finding("a", KV, 1)
    elsewhere["window"]["clock_domain"] = "other-node/boot/unix_epoch_ns"
    assert score_episode(episode(), diagnosis(elsewhere), CONFIG).candidates == ()
    same = finding("a", KV, 1)
    same["window"]["clock_domain"] = "node/boot/unix_epoch_ns"
    assert score_episode(episode(), diagnosis(same), CONFIG).correct(TOP1, 2)
    claim = finding("h", "host_stall", 1, component="engine_core", window=(250, 280))
    claim["window"]["clock_domain"] = "other-node/boot/unix_epoch_ns"
    assert run_of([null_run()], diagnosis(claim)).false_claims == ()


def test_a_fault_claim_in_a_negative_run_is_a_false_positive() -> None:
    queue = finding("q", QUEUE, 1)
    workload = finding(
        "w", "load_increase", 2, severity="info", cause="workload_change"
    )
    score = score_episode(negative(), diagnosis(queue, workload), CONFIG)
    assert score.false_claims == (queue["id"],)
    assert score.miss is None


# ------------------------------------------------------------------ runs


def run_record(run_id: str) -> RunRecord:
    """324 s measured, the first 30 s priming: 294 s of negative time when
    nothing else is injected, C.5's figure per run."""
    return RunRecord(
        run_id=run_id,
        clock_domain="node/boot/unix_epoch_ns",
        measured=Interval(0, 324 * S),
        priming=Interval(0, 30 * S),
    )


def run_of(
    injections: Sequence[Injection],
    diag: dict[str, Any],
    config: ScoreConfig = CONFIG,
    *,
    run_id: str = "q221-run",
) -> RunScore:
    """The episodes, renamed into the run ``run_id``."""
    members = [
        replace(injection, run_id=run_id, episode_id=f"{run_id}-{index}")
        for index, injection in enumerate(injections)
    ]
    return score_run(run_record(run_id), members, diag, config)


def null_run() -> Injection:
    """N: nothing injected; its slot is 100–150 s."""
    return negative("N")


def test_a_run_wide_false_claim_in_a_null_run_is_a_false_positive() -> None:
    # N has no effect of its own: its fault claims count over the run's
    # whole negative exposure, so an N run is never blind.
    claim = finding("h", "host_stall", 1, component="engine_core", window=(250, 280))
    score = run_of([null_run()], diagnosis(claim))
    assert score.false_claims == (claim["id"],)
    assert score.exposure == (Interval(30 * S, 324 * S),)


def test_a_false_claim_in_a_negative_runs_baseline_counts() -> None:
    # A T1 run's own window is 200–230 s; a claim at 40–60 s, in its
    # baseline, is still in the run's negative time.
    twin = replace(
        negative("T1"),
        cause_class="workload_change",
        times=Times(
            action_onset_ns=200 * S, effect_onset_ns=200 * S, effect_end_ns=230 * S
        ),
    )
    early = finding("q", QUEUE, 1, window=(40, 60))
    final = finding("k", KV, 2, window=(300, 320))
    score = run_of([twin], diagnosis(early, final))
    assert score.false_claims == (early["id"], final["id"])


def test_a_fault_episodes_span_is_not_negative_time() -> None:
    # A run with F2 (100–150 s) and its one negative: F2's own correct
    # finding is no false positive, and its span leaves the exposure.
    late_null = replace(
        null_run(),
        times=Times(
            action_onset_ns=240 * S, effect_onset_ns=240 * S, effect_end_ns=280 * S
        ),
    )
    kv = finding("a", KV, 1, window=(101, 140))
    # This one starts before F2's action, within its pre-grace, so most of it
    # lies in negative time; it is F2's, not a false positive.
    early = finding("b", KV, 2, window=(98, 101), resolution=5)
    score = run_of([episode(), late_null], diagnosis(kv, early))
    assert score.false_claims == ()
    assert score.episodes[0].correct(TOP1, 2)
    assert score.episodes[0].matched == (kv["id"], early["id"])
    assert score.exposure == (
        Interval(30 * S, 100 * S),
        Interval(150 * S + CONFIG.grace(KV), 324 * S),
    )
    # The hours are the exposure's: 70 s and 154 s.
    assert summarize([score], CONFIG).negative_hours == pytest.approx(224 / 3600)


def test_an_episode_whose_effect_never_ended_takes_the_rest_of_the_run() -> None:
    unended = replace(
        episode(status="recovery_incomplete"),
        times=Times(action_onset_ns=100 * S, effect_onset_ns=103 * S),
    )
    run = run_record("r")
    members = [replace(unended, run_id="r")]
    assert negative_exposure(run, members, CONFIG) == (Interval(30 * S, 100 * S),)


def test_one_finding_is_credited_to_one_episode() -> None:
    # Two F2 episodes 60 s apart; one late KV finding qualifies for both
    # (its uncertainty is 40 s). It goes to the one whose effect began
    # latest before it: the first.
    first = replace(
        episode(),
        times=Times(
            action_onset_ns=100 * S, effect_onset_ns=100 * S, effect_end_ns=130 * S
        ),
    )
    second = replace(
        episode(),
        times=Times(
            action_onset_ns=190 * S, effect_onset_ns=190 * S, effect_end_ns=220 * S
        ),
    )
    late = finding("a", KV, 1, window=(150, 165), resolution=5)
    late["window"]["uncertainty_ns"] = 40 * S
    config = replace(CONFIG, default_grace_ns=30 * S)
    score = run_of([first, second], diagnosis(late), config)
    assert [e.correct(TOP1, 2) for e in score.episodes] == [True, False]


def test_instrumentation_and_second_negatives_are_no_fpr_unit() -> None:
    capture = replace(negative("I1"), cause_class="instrumentation")
    assert run_of([capture], diagnosis()).negative_episode is None
    two = run_of(
        [null_run(), replace(null_run(), episode_type="P", cause_class="placebo")],
        diagnosis(),
    )
    assert two.negative_episode is None
    assert two.problems == ("run q221-run: more than one negative episode",)


def test_an_episode_of_another_run_is_refused() -> None:
    with pytest.raises(ValueError, match="another run"):
        score_run(
            run_record("r1"), [replace(null_run(), run_id="r2")], diagnosis(), CONFIG
        )


# ------------------------------------------------------------------ a campaign


def _scores(correct: int, total: int, episode_type: str = "F2") -> list[RunScore]:
    right = diagnosis(finding("a", KV, 1))
    wrong = diagnosis(finding("h", "host_stall", 1, component="engine_core"))
    label = episode(episode_type)
    return [
        run_of(
            [label],
            right if index < correct else wrong,
            run_id=f"{episode_type}-{index}",
        )
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
    invalid = run_of(
        [episode(status="not_realized")],
        diagnosis(finding("h", "host_stall", 1)),
        run_id="invalid",
    )
    unsupported = _scores(0, 3, "F6")
    summary = summarize(_scores(15, 15) + [invalid] + unsupported, CONFIG)
    assert [(s.episode_type, s.episodes) for s in summary.strata] == [("F2", 15)]
    assert summary.strata[0].excluded == {"not_realized": 1}


def test_the_fpr_is_bounded_over_negative_runs() -> None:
    clean = [run_of([null_run()], diagnosis(), run_id=f"n{i}") for i in range(60)]
    flagged = run_of([null_run()], diagnosis(finding("q", QUEUE, 1)), run_id="flagged")
    none_flagged = summarize(clean, CONFIG)
    one_flagged = summarize(clean[:59] + [flagged], CONFIG)
    assert none_flagged.fpr_upper_bound == pytest.approx(0.0487, abs=5e-5)
    assert none_flagged.fpr_passes
    # 60 runs of 294 s: 4.9 h, so no claim bounds the rate at 0.61 per hour.
    assert none_flagged.negative_hours == pytest.approx(4.9)
    assert none_flagged.false_claims_per_hour_upper == pytest.approx(0.611, abs=5e-4)
    assert (one_flagged.false_positive_runs, one_flagged.fpr_passes) == (1, False)


def test_the_hourly_rate_counts_claims_over_the_hours_it_divides_by() -> None:
    # Three claims per negative run, all outside the negative episode's own
    # window: each counts, over the same exposure the hours measure.
    claims = (
        finding("h", "host_stall", 1, component="engine_core", window=(40, 70)),
        finding("q", QUEUE, 2, window=(300, 320)),
        finding("k", KV, 3, window=(198, 240)),
    )
    runs = [run_of([null_run()], diagnosis(*claims), run_id=f"n{i}") for i in range(60)]
    summary = summarize(runs, CONFIG)
    assert summary.false_positive_runs == 60
    assert summary.false_claims_per_hour_upper is not None
    assert summary.false_claims_per_hour_upper > 180 / 4.9


def test_a_declared_stratum_without_valid_episodes_fails_the_gate() -> None:
    # Every F2 episode failed realization: the stratum has no evidence, so
    # the gate must not pass on F1 alone.
    both = replace(CONFIG, supported_types=frozenset({"F1", "F2"}))
    f1 = episode("F1", expects=(Expectation(QUEUE, "scheduler"),))
    right = [
        run_of([f1], diagnosis(finding("q", QUEUE, 1)), both, run_id=f"f1-{i}")
        for i in range(15)
    ]
    unrealized = [
        run_of([episode(status="not_realized")], diagnosis(), both, run_id=f"f2-{i}")
        for i in range(15)
    ]
    summary = summarize(right + unrealized, both)
    f1_stratum, f2_stratum = summary.strata
    assert f1_stratum.passes
    assert (f2_stratum.episodes, f2_stratum.lower_bound) == (0, None)
    assert f2_stratum.excluded == {"not_realized": 15}
    assert not f2_stratum.passes and not summary.accuracy_passes


def test_a_gated_summary_needs_the_support_matrix() -> None:
    with pytest.raises(ValueError, match="supported_types"):
        summarize(_scores(15, 15), ScoreConfig(default_grace_ns=20 * S))


def test_an_episode_scored_twice_is_refused() -> None:
    run = _scores(1, 1)[0]
    with pytest.raises(ValueError, match="twice"):
        summarize([run, run], CONFIG)


def test_incident_attribution_counts_only_episodes_with_impact() -> None:
    hurt = run_of([episode()], diagnosis(finding("a", KV, 1)), run_id="hurt")
    unhurt = run_of(
        [episode(impact="no_impact")], diagnosis(finding("a", KV, 1)), run_id="unhurt"
    )
    outranked = run_of(
        [episode()],
        diagnosis(
            finding("h", "host_stall", 1, component="engine_core"), finding("a", KV, 2)
        ),
        run_id="outranked",
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
    episode_record = _scores(1, 1)[0].episodes[0].to_record()
    assert episode_record["match_rank"] == {"L1": 1, "L2": 1}
    # Everything the score froze: grace per kind (the default shown for a
    # kind without its own), the support matrix, the confidence and the
    # negative types.
    graced = replace(CONFIG, grace_ns={KV: 12 * S})
    frozen = summarize(_scores(15, 15), graced).to_record(graced)
    assert frozen["grace_ns"][KV] == 12 * S
    assert frozen["grace_ns"][QUEUE] == 20 * S
    assert frozen["supported_types"] == ["F2"]
    assert frozen["confidence"] == 0.95
    assert frozen["negative_types"] == sorted(CONFIG.negative_types)


def test_the_secondary_error_rate_is_reported() -> None:
    # C.2 reports how often a secondary failed to be neutral, over the fault
    # episodes' secondaries.
    kv = finding("a", KV, 1)
    neutral = finding(
        "q",
        QUEUE,
        2,
        role="secondary",
        window=(110, 130),
        secondary_to=[f"{KV}.{'a':0>12}"],
    )
    stray = finding(
        "h",
        "host_stall",
        3,
        role="secondary",
        component="engine_core",
        secondary_to=[f"{KV}.{'a':0>12}"],
    )
    run = run_of([episode()], diagnosis(kv, neutral, stray))
    summary = summarize([run], CONFIG)
    assert summary.secondary_errors == (1, 2)
    assert summary.to_record(CONFIG)["secondary_errors"] == [1, 2]
