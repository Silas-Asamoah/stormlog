"""``quantile_sorted``: ``quantile``'s interpolation over values already sorted."""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from stormlog.infer import quantiles
from stormlog.infer.quantiles import (
    QuantileEstimate,
    SufficiencyRule,
    penalized_quantile,
    quantile,
    quantile_interval,
    quantile_sorted,
    successful_quantile,
)

LEVELS = (0.0, 0.01, 0.25, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def _bits(value: float | None) -> bytes | None:
    return None if value is None else struct.pack("<d", value)


def _samples() -> list[list[float]]:
    rng = random.Random(227)
    samples: list[list[float]] = [[3.0], [2.0, 1.0], [0.0, 0.0, 5e-324]]
    for size in (3, 10, 231, 1_000):
        samples.append([rng.uniform(0, 5_000) for _ in range(size)])
        samples.append([rng.lognormvariate(4, 1.5) for _ in range(size)])
    samples.append([1e300, 1.7976931348623157e308, 0.1, 1e-300])
    return samples


@pytest.mark.parametrize("p", LEVELS)
def test_a_sorted_list_gives_quantiles_floats_bit_for_bit(p: float) -> None:
    for values in _samples():
        expected = quantile(values, p)
        got = quantile_sorted(sorted(values), p)
        assert _bits(got) == _bits(expected)


@pytest.mark.parametrize("p", LEVELS)
def test_a_sorted_float64_array_gives_the_same_python_float(p: float) -> None:
    for values in _samples():
        got = quantile_sorted(np.sort(np.asarray(values, dtype=np.float64)), p)
        assert type(got) is float
        assert _bits(got) == _bits(quantile(values, p))


def test_integers_interpolate_to_a_float_as_quantile_does() -> None:
    assert quantile_sorted([1, 2, 4], 0.75) == quantile([4, 1, 2], 0.75) == 3.0
    assert type(quantile_sorted([1, 2, 4], 0.5)) is float


def test_the_values_are_read_in_the_order_given() -> None:
    # Not sorted: the caller promised order, so the positions are read as-is.
    assert quantile_sorted([9.0, 1.0, 5.0], 0.0) == 9.0
    assert quantile_sorted([9.0, 1.0, 5.0], 1.0) == 5.0


@pytest.mark.parametrize("p", (-0.01, 1.01))
def test_a_level_outside_zero_and_one_has_no_value(p: float) -> None:
    assert quantile_sorted([1.0, 2.0], p) is None
    assert quantile_sorted(np.array([1.0, 2.0]), p) is None


def test_no_values_have_no_quantile() -> None:
    assert quantile_sorted([], 0.5) is None
    assert quantile_sorted(np.array([], dtype=np.float64), 0.5) is None


# --- presorted: the estimates read a sorted sample without sorting it again ---

ESTIMATE_LEVELS = (0.5, 0.9, 0.95, 0.99)
RULES = (SufficiencyRule(), SufficiencyRule(margin=0, tails="narrowest"))


def _estimate_bits(estimate: QuantileEstimate) -> str:
    """Every field, floats by their bits, so ±0.0 and the last ulp count."""
    return repr(
        {
            name: _bits(value) if isinstance(value, float) else value
            for name, value in vars(estimate).items()
        }
    )


def _estimate_samples() -> list[list[float]]:
    samples = _samples()
    samples.append([0.0, -0.0, 0.0, -0.0] * 60)
    samples.append([7.0] * 300)
    return samples


@pytest.mark.parametrize("p", ESTIMATE_LEVELS)
@pytest.mark.parametrize("rule", RULES)
def test_presorted_successful_estimates_are_the_same_bit_for_bit(
    p: float, rule: SufficiencyRule
) -> None:
    for values in _estimate_samples():
        ordered = sorted(values)
        expected = successful_quantile(values, p, rule)
        got = successful_quantile(ordered, p, rule, presorted=True)
        assert _estimate_bits(got) == _estimate_bits(expected)


@pytest.mark.parametrize("p", ESTIMATE_LEVELS)
@pytest.mark.parametrize("failures", (0, 1, 7, 400))
def test_presorted_penalized_estimates_are_the_same_bit_for_bit(
    p: float, failures: int
) -> None:
    for values in _estimate_samples():
        ordered = sorted(values)
        timeouts = [float(5_000 + i) for i in range(failures)]
        for missing in (0, 2):
            expected = penalized_quantile(
                values,
                failures,
                p,
                timeout_elapsed_ms=timeouts,
                successful_missing=missing,
            )
            got = penalized_quantile(
                ordered,
                failures,
                p,
                timeout_elapsed_ms=timeouts,
                successful_missing=missing,
                presorted=True,
            )
            assert _estimate_bits(got) == _estimate_bits(expected)


def test_a_presorted_interval_reads_the_values_in_the_order_given() -> None:
    values = [float(v) for v in range(300)]
    interval = quantile_interval(values[::-1], 0.5, presorted=True)
    assert interval is not None
    assert interval.lower_ms == values[::-1][interval.lower_rank - 1]


def test_presorted_estimates_never_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def counting(values: list[float]) -> list[float]:
        calls.append(len(values))
        return list(values)

    monkeypatch.setattr(quantiles, "sorted", counting, raising=False)
    ordered = [float(v) for v in range(500)]
    for p in ESTIMATE_LEVELS:
        successful_quantile(ordered, p, presorted=True)
        penalized_quantile(ordered, 3, p, presorted=True)
    assert calls == []
    successful_quantile(ordered, 0.5)
    assert calls == [500, 500]  # quantile and quantile_interval each sort
