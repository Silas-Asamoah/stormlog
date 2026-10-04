"""score_v1's C.2 rules, one at a time: each case differs from a passing
fixture in the one rule it pins, so removing that rule fails it."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from stormlog.infer.qualify.ground_truth import (
    Expectation,
    GroundTruthError,
    Interval,
    Neutral,
    Times,
    parse_run,
)
from stormlog.infer.qualify.scoring import TOP1, score_episode, score_run, summarize
from tests.test_qualify_scoring import (
    CONFIG,
    KV,
    QUEUE,
    S,
    diagnosis,
    episode,
    finding,
    kv_id,
    run_of,
)

# ------------------------------------------------------------------ temporal


def test_pre_grace_includes_the_findings_uncertainty() -> None:
    # Onset 100 s, resolution 1 s, uncertainty 5 s: a start at 97 s is
    # inside S_e only with the uncertainty.
    early = finding("a", KV, 1, window=(97, 140), resolution=1)
    early["window"]["uncertainty_ns"] = 5 * S
    assert score_episode(episode(), diagnosis(early), CONFIG).correct(TOP1, 2)


def test_grace_is_the_findings_kinds() -> None:
    # Effect end 150 s; KV's own grace is 40 s, the default 20 s. A KV
    # finding starting at 175 s qualifies only by its kind's grace.
    graced = replace(CONFIG, grace_ns={KV: 40 * S})
    late = finding("a", KV, 1, window=(175, 185))
    assert score_episode(episode(), diagnosis(late), graced).candidates == (kv_id(),)


# ------------------------------------------------------------------ neutral secondaries


def secondary_queue(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "role": "secondary",
        "window": (110, 130),
        "secondary_to": [f"{KV}.{'a':0>12}"],
    }
    base.update(changes)
    return finding("q", QUEUE, 2, **base)


def test_only_a_secondary_can_be_neutral() -> None:
    primary = secondary_queue(role="primary")
    score = score_episode(episode(), diagnosis(finding("a", KV, 1), primary), CONFIG)
    assert score.neutral == ()


def test_a_neutral_secondarys_upstream_must_be_eligible() -> None:
    observed = finding("a", KV, 1, claim="observation", cause="undetermined")
    score = score_episode(episode(), diagnosis(observed, secondary_queue()), CONFIG)
    assert score.neutral == ()


def test_an_edges_upstream_must_be_where_the_edge_allows() -> None:
    # host_stall -> queue_saturation needs its stall at the engine core; an
    # API-server stall can't make the queue neutral, labelled or not.
    stall_label = episode("F4b", expects=(Expectation("host_stall", "api_server"),))
    stall = finding("s", "host_stall", 1, component="api_server")
    queue = secondary_queue(secondary_to=[f"host_stall.{'s':0>12}"])
    score = score_episode(stall_label, diagnosis(stall, queue), CONFIG)
    assert score.neutral == ()
    at_core = replace(stall_label, expects=(Expectation("host_stall", "engine_core"),))
    core = finding("s", "host_stall", 1, component="engine_core")
    assert score_episode(at_core, diagnosis(core, queue), CONFIG).neutral == (
        queue["id"],
    )


def test_an_edges_downstream_must_be_where_the_edge_allows() -> None:
    # A queue finding #218 placed off the scheduler (a schema slip) is not
    # the edge's downstream.
    misplaced = secondary_queue(component="workload")
    score = score_episode(episode(), diagnosis(finding("a", KV, 1), misplaced), CONFIG)
    assert score.neutral == ()


def test_nesting_allows_the_downstream_kinds_grace() -> None:
    # KV ends at 140 s; the queue secondary runs to 155 s, within 20 s.
    trailing = secondary_queue(window=(110, 155))
    score = score_episode(episode(), diagnosis(finding("a", KV, 1), trailing), CONFIG)
    assert score.neutral == (trailing["id"],)


def test_a_neutral_secondarys_upstream_must_be_labelled() -> None:
    # host_stall@engine_core -> queue is an edge, but this F2 label names no
    # host stall: the queue isn't neutral through it.
    stall = finding("s", "host_stall", 1, component="engine_core")
    queue = secondary_queue(secondary_to=[f"host_stall.{'s':0>12}"])
    score = score_episode(episode(), diagnosis(stall, queue), CONFIG)
    assert score.neutral == ()


# ------------------------------------------------------------------ match


def test_an_ineligible_claim_never_matches() -> None:
    # Right kind, role, cause, severity and place; only the claim differs.
    observed = finding("a", KV, 1, claim="observation")
    assert not score_episode(episode(), diagnosis(observed), CONFIG).correct(TOP1, 1)


def test_a_claim_of_another_cause_never_matches() -> None:
    other = finding("a", KV, 1, cause="undetermined", claim="condition")
    assert not score_episode(episode(), diagnosis(other), CONFIG).correct(TOP1, 1)


def test_l1_needs_the_component() -> None:
    elsewhere = finding("a", KV, 1, component="scheduler")
    assert not score_episode(episode(), diagnosis(elsewhere), CONFIG).correct(TOP1, 1)


# ------------------------------------------------------------------ false claims


def test_only_eligible_fault_claims_at_warning_are_false() -> None:
    def stall(**changes: Any) -> dict[str, Any]:
        return finding("s", "host_stall", 2, component="engine_core", **changes)

    right = finding("a", KV, 1)
    for harmless in (
        stall(claim="observation"),
        stall(cause="undetermined", claim="condition"),
        stall(severity="info"),
    ):
        score = score_episode(episode(), diagnosis(right, harmless), CONFIG)
        assert score.false_claims == (), harmless
    assert score_episode(episode(), diagnosis(right, stall()), CONFIG).false_claims


def test_an_all_primary_payloads_downstream_info_finding_is_no_false_claim() -> None:
    # Until #218's PR 2 adds its edge table, every finding is primary. A
    # queue finding with KV pressure upstream can't claim fault: #218 leaves
    # it at info ("an upstream cause is present"). Primary, it isn't neutral;
    # at info, it isn't a false claim either. At warning it would be one.
    def downstream(severity: str) -> dict[str, Any]:
        return finding("q", QUEUE, 2, severity=severity, claim="condition")

    right = finding("a", KV, 1)
    score = score_episode(episode(), diagnosis(right, downstream("info")), CONFIG)
    assert score.correct(TOP1, 1)
    assert score.neutral == ()
    assert score.false_claims == ()
    warned = score_episode(episode(), diagnosis(right, downstream("warning")), CONFIG)
    assert warned.false_claims == (downstream("warning")["id"],)


def test_a_negative_runs_false_claim_must_be_eligible_and_at_warning() -> None:
    from tests.test_qualify_scoring import null_run

    for harmless in (
        finding("s", "host_stall", 1, component="engine_core", claim="observation"),
        finding("s", "host_stall", 1, component="engine_core", severity="info"),
    ):
        assert run_of([null_run()], diagnosis(harmless)).false_claims == ()


# ------------------------------------------------------------------ the summary


def test_only_valid_supported_fault_episodes_feed_the_descriptive_counts() -> None:
    right = diagnosis(finding("a", KV, 1))
    wrong = diagnosis(finding("h", "host_stall", 1, component="engine_core"))
    runs = [run_of([episode()], right, run_id=f"ok{i}") for i in range(15)]
    runs.append(run_of([episode(status="not_realized")], right, run_id="invalid"))
    runs += [run_of([episode("F6")], wrong, run_id=f"f6-{i}") for i in range(3)]
    summary = summarize(runs, CONFIG)
    assert summary.localized == (15, 15)
    assert summary.misses == {}
    assert summary.spurious == 0


def test_an_allowed_kind_in_a_negative_run_is_no_false_positive() -> None:
    from tests.test_qualify_scoring import null_run

    allowing = replace(null_run(), allows=(Neutral("host_stall", "engine_core"),))
    stall = finding("s", "host_stall", 1, component="engine_core")
    assert run_of([allowing], diagnosis(stall)).false_claims == ()


def test_an_invalid_negative_episode_makes_no_fpr_unit() -> None:
    from tests.test_qualify_scoring import null_run

    unrealized = replace(null_run(), episode_type="T3b", status="not_realized")
    assert run_of([unrealized], diagnosis()).negative_episode is None


def test_a_negative_runs_claim_is_placed_by_its_part_in_the_run() -> None:
    # A window is clipped to the run's measured window (0-324 s, priming to
    # 30 s), then counted when at least half of what is left lies in the
    # exposure, wherever it starts (rev-220-b's delta D5).
    from tests.test_qualify_scoring import null_run

    def claims(window: tuple[float, float]) -> tuple[str, ...]:
        stall = finding("s", "host_stall", 1, component="engine_core", window=window)
        return run_of([null_run()], diagnosis(stall)).false_claims

    # A claim over the whole run, from its start or from inside the priming:
    # the plainest false positive. Both used to be dropped for their start.
    assert claims((0, 400)) and claims((25, 400))
    # Mostly in the priming: not placed.
    assert claims((10, 35)) == ()
    # Starting near the window's end: what lies inside the run is exposure.
    assert claims((320, 400))


def test_void_negative_runs_are_reported_by_why() -> None:
    # A void negative shrinks the FPR's denominator: 0 of 52 bounds the
    # rate at 0.056, over the 0.05 ceiling. So the summary says how many
    # negative runs were left out, and why, instead of dropping them.
    from tests.test_qualify_scoring import negative, null_run, run_record

    unrealized = replace(negative("T3b"), status="not_realized")
    runs = [run_of([null_run()], diagnosis(), run_id=f"n{i}") for i in range(3)]
    runs += [run_of([unrealized], diagnosis(), run_id=f"t{i}") for i in range(2)]
    failed = replace(run_record("p0"), protocol_failure="priming_check_failed")
    runs.append(
        score_run(failed, [replace(null_run(), run_id="p0")], diagnosis(), CONFIG)
    )
    summary = summarize(runs, CONFIG)
    assert summary.negative_runs == 3
    assert summary.excluded_negatives == {"not_realized": 2, "protocol_failure": 1}
    assert summary.to_record(CONFIG)["excluded_negative_runs"] == {
        "not_realized": 2,
        "protocol_failure": 1,
    }
    assert runs[3].excluded_negative == "not_realized"


# ------------------------------------------------------------------ run bookkeeping


def test_a_negative_run_counts_only_with_its_episode_in_enough_exposure() -> None:
    # rev-220-b's delta D6. A run that is nearly all priming used to be a
    # clean unit with no exposure; one whose N slot an unended fault had
    # cut out was a unit that could never be flagged.
    from tests.test_qualify_scoring import null_run, run_record

    mostly_priming = replace(
        run_record("p"), measured=Interval(0, 200 * S), priming=Interval(0, 190 * S)
    )
    score = score_run(
        mostly_priming, [replace(null_run(), run_id="p")], diagnosis(), CONFIG
    )
    assert score.negative_episode is None
    assert score.excluded_negative == "exposure_below_minimum"
    unended = replace(
        episode(status="recovery_incomplete"),
        run_id="u",
        episode_id="u-0",
        times=Times(action_onset_ns=120 * S, effect_onset_ns=120 * S),
    )
    late_null = replace(
        null_run(),
        run_id="u",
        episode_id="u-1",
        times=Times(
            action_onset_ns=200 * S, effect_onset_ns=200 * S, effect_end_ns=250 * S
        ),
    )
    score = score_run(run_record("u"), [unended, late_null], diagnosis(), CONFIG)
    assert score.excluded_negative == "negative_outside_exposure"


def test_an_episode_outside_its_runs_window_is_refused() -> None:
    from tests.test_qualify_scoring import null_run, run_record

    short = replace(run_record("s"), measured=Interval(0, 120 * S), priming=None)
    with pytest.raises(ValueError, match="outside its window"):
        score_run(short, [replace(null_run(), run_id="s")], diagnosis(), CONFIG)


def test_a_run_record_needs_an_id_and_a_measured_length() -> None:
    from tests.test_qualify_scoring import run_record

    nameless = replace(run_record("x"), run_id="")
    empty = replace(run_record("x"), measured=Interval(0, 0), priming=None)
    assert "a run needs its run_id" in nameless.problems()
    assert "measured has no length" in empty.problems()


@pytest.mark.parametrize("clock", [None, ""])
def test_a_run_record_needs_its_clock(clock: str | None) -> None:
    # rev-220-b's delta 2, E8: a run with no clock domain was accepted, and
    # every finding with a clock then looked off the run's clock.
    from tests.test_qualify_scoring import run_record

    record = replace(run_record("x"), clock_domain=clock)
    assert "a run needs its clock_domain" in record.problems()
    with pytest.raises(GroundTruthError, match="clock_domain"):
        parse_run(record.to_record())


def test_a_run_scored_twice_is_refused() -> None:
    from tests.test_qualify_scoring import null_run

    run = run_of([null_run()], diagnosis(), run_id="twice")
    other = run_of(
        [replace(null_run(), episode_id="other")], diagnosis(), run_id="twice"
    )
    with pytest.raises(ValueError, match="a run is scored twice"):
        summarize([run, other], CONFIG)


def test_findings_on_another_clock_make_no_clean_negative_run() -> None:
    # rev-220-b's delta D4: a window clock spelled differently from the
    # run's hid every claim, and 60 N runs passed the FPR gate, 0 flagged,
    # with no problem. Now each such run says so and is no unit.
    from tests.test_qualify_scoring import null_run

    claim = finding("s", "host_stall", 1, component="engine_core", window=(110, 130))
    claim["window"]["clock_domain"] = "node/boot/UNIX_EPOCH_NS"
    runs = [run_of([null_run()], diagnosis(claim), run_id=f"n{i}") for i in range(60)]
    assert runs[0].problems == ("run n0: 1 findings on another clock",)
    summary = summarize(runs, CONFIG)
    assert summary.negative_runs == 0 and not summary.fpr_passes
    assert summary.excluded_negatives == {"findings_off_clock": 60}


def test_a_fault_claim_without_a_window_in_a_negative_run() -> None:
    # A claim the scorer can't place in time counts as false in a negative
    # run that injected nothing it could be about; in one that also held a
    # fault or capture, it is reported as unplaced instead.
    from tests.test_qualify_scoring import null_run

    windowless = finding("s", "host_stall", 1, component="engine_core", window=None)
    alone = run_of([null_run()], diagnosis(windowless))
    assert alone.false_claims == (windowless["id"],)
    late_null = replace(
        null_run(),
        times=Times(
            action_onset_ns=220 * S, effect_onset_ns=220 * S, effect_end_ns=260 * S
        ),
    )
    mixed = run_of([episode(), late_null], diagnosis(windowless))
    assert mixed.false_claims == ()
    assert mixed.problems == ("run q221-run: 1 fault claims without a window",)


# ------------------------------------------------------------------ pinned by mutation


def test_a_finding_goes_to_the_latest_episode_begun_before_it() -> None:
    # Two KV episodes; a finding at 165-175 s qualifies for both: the first
    # by its grace, the second by its pre-grace. It goes to the second,
    # whose effect began latest before it, not to the earliest.
    first = replace(episode(), episode_id="e1")
    second = replace(
        episode(),
        episode_id="e2",
        times=Times(
            action_onset_ns=160 * S, effect_onset_ns=160 * S, effect_end_ns=200 * S
        ),
    )
    late = finding("a", KV, 1, window=(165, 175))
    score = run_of([first, second], diagnosis(late))
    assert [e.correct(TOP1, 2) for e in score.episodes] == [False, True]


def test_a_neutral_secondary_in_a_negative_run_is_no_false_claim() -> None:
    # A KV twin whose label declares the queue a secondary through KV -> queue:
    # a queue fault claim secondary to the twin's KV finding is neutral.
    from tests.test_qualify_scoring import negative

    twin = replace(
        negative("T2"),
        cause_class="workload_change",
        expects=(Expectation(KV, "kv_cache", cause="workload_change"),),
        secondary=(Neutral(QUEUE, "scheduler", edge=f"{KV}->{QUEUE}"),),
    )
    kv = finding("a", KV, 1, cause="workload_change")
    queue = finding(
        "q",
        QUEUE,
        2,
        role="secondary",
        window=(110, 130),
        secondary_to=[f"{KV}.{'a':0>12}"],
    )
    assert run_of([twin], diagnosis(kv, queue)).false_claims == ()


def test_an_episode_given_twice_is_refused() -> None:
    from tests.test_qualify_scoring import null_run, run_record

    null = replace(null_run(), run_id="r")
    with pytest.raises(ValueError, match="given twice"):
        score_run(run_record("r"), [null, null], diagnosis(), CONFIG)


def test_a_capture_beside_a_null_run_leaves_the_null_its_unit() -> None:
    # The FPR population is C.5's negative types, not "anything but a
    # fault": an I1 capture in the same run is no second negative.
    from tests.test_qualify_scoring import negative, null_run

    capture = replace(
        negative("I1"),
        cause_class="instrumentation",
        episode_id="cap",
        times=Times(
            action_onset_ns=200 * S, effect_onset_ns=200 * S, effect_end_ns=220 * S
        ),
    )
    score = run_of([null_run(), capture], diagnosis())
    assert score.negative_episode is not None
    assert score.excluded_negative is None


def test_an_episode_scored_in_two_runs_is_refused() -> None:
    # Two runs, each with its own id, both holding the same episode.
    from tests.test_qualify_scoring import null_run

    first = run_of([null_run()], diagnosis(), run_id="a")
    second = run_of([null_run()], diagnosis(), run_id="b")
    twice = replace(second, episodes=first.episodes)
    with pytest.raises(ValueError, match="an episode is scored twice"):
        summarize([first, twice], CONFIG)


def test_a_claim_window_from_before_the_run_is_clipped_to_it() -> None:
    # A window from 400 s before the run to 100 s into it: clipped to the
    # run, 70 of its 100 s lie in the exposure, so it counts.
    from tests.test_qualify_scoring import null_run

    early = finding("s", "host_stall", 1, component="engine_core", window=(-400, 100))
    assert run_of([null_run()], diagnosis(early)).false_claims == (early["id"],)
