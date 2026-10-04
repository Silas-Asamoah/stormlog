"""Order-statistic sufficiency and the two latency estimands."""

from __future__ import annotations

import pytest
from scipy import stats

from stormlog.infer.quantiles import (
    SufficiencyRule,
    penalized_quantile,
    quantile,
    quantile_interval,
    quantile_minimum_n,
    successful_quantile,
)
from stormlog.infer.report_stats import percentile


@pytest.mark.parametrize(
    ("p", "margin", "expected"),
    [
        (0.5, 5, 20),
        (0.75, 5, 44),
        (0.9, 5, 114),
        (0.95, 5, 230),
        (0.99, 5, 1164),
        (0.999, 5, 11665),
        (0.5, 0, 6),
        (0.75, 0, 13),
        (0.9, 0, 36),
        (0.95, 0, 72),
        (0.99, 0, 368),
        (0.999, 0, 3688),
    ],
)
def test_the_symmetric_rule_matches_the_shared_table(
    p: float, margin: int, expected: int
) -> None:
    assert quantile_minimum_n(p, margin=margin) == expected


@pytest.mark.parametrize(
    ("p", "expected"), [(0.5, 6), (0.95, 59), (0.99, 299), (0.999, 2995)]
)
def test_the_narrowest_rule_matches_le_boudecs_tables(p: float, expected: int) -> None:
    assert quantile_minimum_n(p, margin=0, tails="narrowest") == expected


def test_le_boudecs_example_interval_for_the_95th_percentile() -> None:
    # Table A.3: n = 59 gives [X(50), X(59)] at level 0.951.
    values = [float(v) for v in range(1, 60)]
    interval = quantile_interval(
        values, 0.95, SufficiencyRule(margin=0, tails="narrowest")
    )
    assert interval is not None
    assert (interval.lower_rank, interval.upper_rank) == (50, 59)
    assert interval.coverage == pytest.approx(0.9509, abs=1e-4)


@pytest.mark.parametrize(
    ("n", "p", "expected"),
    [
        # Three intervals of width 8 reach 95%; the middle one is centred on
        # rank (n + 1) p = 7, as Le Boudec's symmetric median intervals are.
        (13, 0.5, (3, 11)),
        (21, 0.5, (6, 16)),
        (28, 0.5, (9, 20)),
    ],
)
def test_narrowest_ties_go_to_the_more_central_interval(
    n: int, p: float, expected: tuple[int, int]
) -> None:
    values = [float(v) for v in range(1, n + 1)]
    interval = quantile_interval(
        values, p, SufficiencyRule(margin=0, tails="narrowest")
    )
    assert interval is not None
    assert (interval.lower_rank, interval.upper_rank) == expected


def _narrowest(n: int, p: float) -> tuple[int, int] | None:
    values = [float(v) for v in range(1, n + 1)]
    interval = quantile_interval(
        values, p, SufficiencyRule(margin=0, tails="narrowest")
    )
    return None if interval is None else (interval.lower_rank, interval.upper_rank)


@pytest.mark.parametrize(
    ("n", "p", "expected"),
    [(94, 0.9, (79, 91)), (109, 0.9, (92, 105)), (164, 0.9, (140, 156))],
)
def test_equally_central_ties_go_to_the_higher_coverage(
    n: int, p: float, expected: tuple[int, int]
) -> None:
    # (80, 92) and (79, 91) at n = 94 sit 0.5 either side of rank 85.5,
    # covering 0.9503 and 0.9636. One ulp in (n + 1) p used to decide.
    assert _narrowest(n, p) == expected


def test_mirror_image_ties_at_the_median_go_to_the_higher_ranks() -> None:
    # Equal width, equally central, equal coverage: (1, 7) and (2, 8) at
    # n = 8. The higher ranks are the pessimistic side for a latency, and
    # float noise in the coverage no longer picks one.
    for n in range(6, 400):
        found = _narrowest(n, 0.5)
        assert found is not None
        lower, upper = found
        mirror = (n + 1 - upper, n + 1 - lower)
        if (
            mirror != (lower, upper)
            and _covers(n, mirror)
            and _width(mirror) == upper - lower
        ):
            assert lower > mirror[0], n


def _width(ranks: tuple[int, int]) -> int:
    return ranks[1] - ranks[0]


def _covers(n: int, ranks: tuple[int, int]) -> bool:
    lower, upper = ranks
    level = stats.binom.cdf(upper - 1, n, 0.5) - stats.binom.cdf(lower - 1, n, 0.5)
    return bool(level >= 0.95)


@pytest.mark.parametrize(("p", "below", "at"), [(0.95, 229, 230), (0.99, 1163, 1164)])
def test_sufficiency_turns_on_exactly_at_the_minimum(
    p: float, below: int, at: int
) -> None:
    assert not successful_quantile([1.0] * below, p).sufficient
    assert successful_quantile([1.0] * at, p).sufficient


def test_the_symmetric_interval_keeps_each_tail_within_alpha() -> None:
    n, p = 300, 0.95
    interval = quantile_interval([float(v) for v in range(n)], p)
    assert interval is not None
    assert stats.binom.cdf(interval.lower_rank - 1, n, p) <= 0.025
    assert stats.binom.sf(interval.upper_rank - 1, n, p) <= 0.025
    assert interval.upper_rank <= n - 5
    assert interval.coverage >= 0.95


def test_the_ranks_come_from_n_and_p_never_from_the_values() -> None:
    rule = SufficiencyRule()
    ordered = [float(v) for v in range(500)]
    shuffled = list(reversed(ordered))
    first = quantile_interval(ordered, 0.9, rule)
    second = quantile_interval(shuffled, 0.9, rule)
    assert first is not None and second is not None
    assert (first.lower_rank, first.upper_rank) == (
        second.lower_rank,
        second.upper_rank,
    )
    spread = [v * v for v in ordered]
    third = quantile_interval(spread, 0.9, rule)
    assert third is not None
    assert (third.lower_rank, third.upper_rank) == (first.lower_rank, first.upper_rank)


def test_too_few_observations_have_no_interval() -> None:
    assert quantile_interval([1.0] * 100, 0.99) is None
    assert successful_quantile([1.0] * 100, 0.99).interval is None


@pytest.mark.parametrize("percent", [0, 5, 50, 90, 95, 99, 100])
def test_quantile_agrees_with_the_existing_percentile_helper(percent: int) -> None:
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8]
    assert quantile(values, percent / 100) == pytest.approx(percentile(values, percent))


def test_a_successful_quantile_reports_its_rule_and_counts() -> None:
    estimate = successful_quantile([float(v) for v in range(1, 301)], 0.95)
    assert estimate.estimand == "successful"
    assert estimate.value_ms == pytest.approx(285.05)
    assert (estimate.n, estimate.n_min, estimate.n_min_exists) == (300, 230, 59)
    assert estimate.sufficient
    assert estimate.interval is not None


def test_a_penalized_quantile_is_the_rescaled_successful_quantile() -> None:
    successful = [float(v) for v in range(1, 91)]
    estimate = penalized_quantile(successful, 10, 0.5)
    assert estimate.estimand == "failure_penalized"
    assert estimate.n == 100
    assert not estimate.penalized
    assert estimate.value_ms == pytest.approx(quantile(successful, 0.5 / 0.9))


def test_ninety_five_successes_and_five_failures_keep_a_finite_p95() -> None:
    # At level 0.95 / 0.95 = 1 the quantile is the largest success; a rule
    # that interpolated across the failure mass would make it infinite.
    successful = [float(v) for v in range(1, 96)]
    estimate = penalized_quantile(successful, 5, 0.95)
    assert not estimate.penalized
    assert estimate.value_ms == 95.0


def test_a_quantile_in_the_failure_mass_has_no_value() -> None:
    estimate = penalized_quantile([float(v) for v in range(1, 91)], 10, 0.95)
    assert estimate.penalized
    assert estimate.value_ms is None
    assert estimate.observed_lower_bound_ms is None


def test_only_real_timeouts_give_an_observed_lower_bound() -> None:
    successful = [float(v) for v in range(1, 91)]
    timeouts = [60_000.0] * 9 + [59_990.0]
    estimate = penalized_quantile(successful, 10, 0.95, timeout_elapsed_ms=timeouts)
    assert estimate.penalized
    assert estimate.observed_lower_bound_ms == 60_000.0
    # A count that does not cover every failure is not evidence of a bound.
    partial = penalized_quantile(successful, 10, 0.95, timeout_elapsed_ms=[60_000.0])
    assert partial.observed_lower_bound_ms is None


def test_the_observed_bound_holds_for_the_interpolated_quantile() -> None:
    # 227 successes and 12 timeouts: level 0.95 * 239 / 227 is just above 1,
    # yet interpolation at rank 0.95 * 238 still weighs the largest success.
    # Had each timeout ended when it was abandoned, p95 would be 6,204.3 ms,
    # far below the shortest timeout; a later end can only raise it.
    successful = [float(v) for v in range(1, 228)]
    timeouts = [60_000.0] * 12
    estimate = penalized_quantile(successful, 12, 0.95, timeout_elapsed_ms=timeouts)
    assert estimate.penalized
    assert estimate.observed_lower_bound_ms == pytest.approx(0.9 * 227 + 6_000)
    assert estimate.observed_lower_bound_ms == quantile(successful + timeouts, 0.95)


def test_a_missing_successful_value_leaves_the_penalized_quantile_undefined() -> None:
    # The share that failed is over every offered request; a success whose
    # value is missing cannot be ranked, so no level can be computed.
    estimate = penalized_quantile(
        [float(v) for v in range(1, 91)], 10, 0.5, successful_missing=5
    )
    assert estimate.value_ms is None
    assert estimate.n == 105
    assert not estimate.penalized
    assert estimate.reason == "successful_values_missing"
    assert estimate.interval is None
    assert estimate.observed_lower_bound_ms is None


def test_without_failures_nothing_is_penalized() -> None:
    estimate = penalized_quantile([], 0, 0.5)
    assert not estimate.penalized
    assert estimate.value_ms is None
    assert estimate.reason == "no_values"
    assert estimate.n == 0
    missing = penalized_quantile([], 0, 0.5, successful_missing=3)
    assert not missing.penalized
    assert missing.reason == "successful_values_missing"


def test_a_case_with_no_successes_is_entirely_penalized() -> None:
    estimate = penalized_quantile([], 20, 0.5)
    assert estimate.penalized and estimate.value_ms is None
    assert estimate.n == 20


def test_penalized_failures_push_the_upper_rank_out_of_the_values() -> None:
    successful = [float(v) for v in range(1, 1001)]
    interval = quantile_interval(successful, 0.95, penalized=60)
    assert interval is not None
    assert interval.upper_ms is None
    assert interval.width_ms is None


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_quantile_levels_must_be_strictly_inside_zero_and_one(p: float) -> None:
    with pytest.raises(ValueError):
        quantile_minimum_n(p)


@pytest.mark.parametrize(("confidence", "margin"), [(0.0, 5), (1.0, 5), (0.95, -1)])
def test_invalid_rules_are_refused(confidence: float, margin: int) -> None:
    with pytest.raises(ValueError):
        SufficiencyRule(confidence=confidence, margin=margin)
