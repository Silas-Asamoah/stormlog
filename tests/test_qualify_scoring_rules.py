"""score_v1's C.2 rules, one at a time: each case differs from a passing
fixture in the one rule it pins, so removing that rule fails it."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from stormlog.infer.qualify.ground_truth import Expectation, Neutral
from stormlog.infer.qualify.scoring import TOP1, score_episode, summarize
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


def test_a_negative_runs_claim_must_lie_mostly_in_its_exposure() -> None:
    # The run's measured window ends at 324 s: a claim from 320 s to 400 s
    # starts in the exposure but lies mostly outside it.
    from tests.test_qualify_scoring import null_run

    trailing = finding("s", "host_stall", 1, component="engine_core", window=(320, 400))
    assert run_of([null_run()], diagnosis(trailing)).false_claims == ()
    inside = finding("s", "host_stall", 1, component="engine_core", window=(300, 340))
    assert run_of([null_run()], diagnosis(inside)).false_claims == (inside["id"],)
