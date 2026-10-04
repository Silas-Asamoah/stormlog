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


def int_value(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return 0


def is_number(value: Any) -> TypeGuard[int | float]:
    """A finite real number; NaN and infinities in an artifact are no value."""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, int) and not isinstance(value, bool)
