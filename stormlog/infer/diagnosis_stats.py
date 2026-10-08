"""Small, seeded statistics for diagnosis: medians and their differences.

A difference of medians between a subject and its reference comes with a
percentile bootstrap interval: both arms are resampled independently,
B = 2000 times, from a fixed seed, so a rerun on the same artifact gives the
same interval. Below the per-arm floor nothing is estimated.
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
) -> Difference | None:
    """The difference of medians with a percentile bootstrap 95% interval;
    None when either arm is under the floor."""
    if len(subject) < MIN_PER_ARM or len(reference) < MIN_PER_ARM:
        return None
    rng = np.random.default_rng(seed)
    draws = np.concatenate(
        [
            _medians(rng, np.asarray(subject, dtype=float), size)
            - _medians(rng, np.asarray(reference, dtype=float), size)
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


def _medians(rng: np.random.Generator, values: np.ndarray, size: int) -> np.ndarray:
    """Medians of ``size`` resamples of ``values``, with replacement."""
    picks = rng.integers(0, len(values), size=(size, len(values)))
    return np.asarray(np.median(values[picks], axis=1), dtype=float)


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
    "median",
    "median_difference",
]
