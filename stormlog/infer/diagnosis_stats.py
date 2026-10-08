"""Small, seeded statistics for diagnosis: medians and their differences.

A difference of medians between a subject and its reference comes with a
percentile bootstrap interval: each arm, in arrival order, is resampled
apart from the other by a moving-block bootstrap, B = 2000 times, from a
fixed seed, so a rerun on the same artifact gives the same interval. Waits
within a burst are a ramp, not independent draws; resampling runs of
consecutive requests keeps that dependence, where resampling single
requests would give an interval narrower than its 95%. Below the per-arm
floor nothing is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

import numpy as np

BOOTSTRAP_REPLICATES = 2000
SEED = 20261003
MIN_PER_ARM = 20
INSUFFICIENT_SAMPLES = "insufficient_samples"


@dataclass(frozen=True)
class Difference:
    """A subject's median minus its reference's, with a 95% interval."""

    estimate: float
    low: float
    high: float
    n: int
    n_ref: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def median_difference(
    subject: Sequence[float],
    reference: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
    blocks: bool = True,
) -> Difference | None:
    """The difference of medians with a percentile bootstrap 95% interval,
    each arm in arrival order; None when either arm is under the floor.
    ``blocks=False`` resamples single values, for comparison."""
    if len(subject) < MIN_PER_ARM or len(reference) < MIN_PER_ARM:
        return None
    rng = np.random.default_rng(seed)
    draws = np.concatenate(
        [
            _medians(rng, np.asarray(subject, dtype=float), size, blocks)
            - _medians(rng, np.asarray(reference, dtype=float), size, blocks)
            for size in _blocks(replicates)
        ]
    )
    draws.sort()
    return Difference(
        estimate=float(median(subject) - median(reference)),
        low=float(draws[int(0.025 * (len(draws) - 1))]),
        high=float(draws[int(0.975 * (len(draws) - 1))]),
        n=len(subject),
        n_ref=len(reference),
    )


def _medians(
    rng: np.random.Generator, values: np.ndarray, size: int, blocks: bool = True
) -> np.ndarray:
    """Medians of ``size`` resamples of ``values``, with replacement: runs
    of ``block_length`` consecutive values, from starts drawn uniformly,
    joined and cut to the arm's length."""
    count = len(values)
    length = block_length(count) if blocks else 1
    runs = -(-count // length)
    starts = rng.integers(0, count - length + 1, size=(size, runs))
    picks = (starts[..., None] + np.arange(length)).reshape(size, -1)[:, :count]
    return np.asarray(np.median(values[picks], axis=1), dtype=float)


def block_length(count: int) -> int:
    """The moving block's length: the cube root of the arm's size, the
    usual rate for a variance of a smooth statistic."""
    return max(1, int(round(count ** (1 / 3))))


def _blocks(replicates: int, block: int = 250) -> list[int]:
    """Replicates in blocks, so a large arm never needs one huge array."""
    sizes = [block] * (replicates // block)
    if replicates % block:
        sizes.append(replicates % block)
    return sizes


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "INSUFFICIENT_SAMPLES",
    "MIN_PER_ARM",
    "SEED",
    "Difference",
    "block_length",
    "median",
    "median_difference",
]
