"""The skew screen's table: the 95% quantile of |G1| under normal noise.

``stormlog.infer.comparison_stats`` flags a metric whose block log ratios
look skewed, comparing their adjusted sample skewness G1 (scipy's
``skew(..., bias=False)``) with what normal noise produces at the same n.
This script draws that reference: for each n from 3 to 30, one million
normal samples with a fixed seed, and prints the table pinned in the module.
Beyond 30 the module uses 1.96 times the standard error of G1.

    python -m examples.analysis.skew_quantiles
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

SEED = 213
DRAWS = 1_000_000
CHUNK = 100_000
SIZES = range(3, 31)


def q95_abs_g1(n: int, rng: np.random.Generator) -> float:
    values = np.concatenate(
        [
            np.abs(skew(rng.standard_normal((CHUNK, n)), axis=1, bias=False))
            for _ in range(DRAWS // CHUNK)
        ]
    )
    return float(np.quantile(values, 0.95))


def main() -> None:
    rng = np.random.default_rng(SEED)
    table = {n: round(q95_abs_g1(n, rng), 3) for n in SIZES}
    print("SKEW_Q95_NORMAL = {")
    for n, value in table.items():
        print(f"    {n}: {value},")
    print("}")


if __name__ == "__main__":
    main()
