"""``quantile_sorted``: ``quantile``'s interpolation over values already sorted."""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from stormlog.infer.quantiles import quantile, quantile_sorted

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
