"""Small statistics helpers shared by the inference reports."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, TypeGuard


def number_values(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if is_number(value):
            values.append(float(value))
    return values


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percent / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


#: The smallest int that ``float()`` cannot convert: halfway from the
#: largest float to 2**1024, which rounds up and overflows.
FLOAT_INT_LIMIT = 2**1024 - 2**970


def int_value(value: Any) -> int:
    """``value`` as an int, or 0 for what ``is_number`` refuses."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if -FLOAT_INT_LIMIT < value < FLOAT_INT_LIMIT else 0
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return 0


def is_number(value: Any) -> TypeGuard[int | float]:
    """A finite real number; NaN and infinities in an artifact are no value.

    JSON integers have no size limit, and one past a float's range (a
    400-digit gap) is no value either: ``float()`` would raise on it.
    """
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return -FLOAT_INT_LIMIT < value < FLOAT_INT_LIMIT
    return False
