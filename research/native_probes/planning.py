"""Counterbalanced trial planning without executing profiler commands."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

from .models import ExperimentMode


def counterbalanced_order(
    modes: Iterable[ExperimentMode], *, repetitions: int, seed: int
) -> list[list[ExperimentMode]]:
    """Return reproducible rotated and reversed mode orders."""
    unique = _unique_modes(modes)
    if len(unique) < 2:
        raise ValueError("at least two modes are required")
    if repetitions < 5:
        raise ValueError("at least five repetitions are required")
    generator = random.Random(seed)
    generator.shuffle(unique)
    orders: list[list[ExperimentMode]] = []
    for repetition in range(repetitions):
        rotation = repetition % len(unique)
        order = unique[rotation:] + unique[:rotation]
        if (repetition // len(unique)) % 2:
            order = list(reversed(order))
        orders.append(order)
    return orders


def trial_id(
    configuration_id: str,
    workload_id: str,
    mode: ExperimentMode,
    repetition: int,
) -> str:
    """Create a stable, filesystem-safe identity for one trial."""
    if repetition < 0:
        raise ValueError("repetition must be nonnegative")
    for field, value in (
        ("configuration_id", configuration_id),
        ("workload_id", workload_id),
    ):
        if not value or not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"{field} must contain only letters, numbers, '-' or '_'")
    digest = hashlib.sha256(
        f"{configuration_id}\0{workload_id}\0{mode.value}\0{repetition}".encode()
    ).hexdigest()[:12]
    return f"{configuration_id}-{workload_id}-{mode.value}-r{repetition:02d}-{digest}"


def _unique_modes(modes: Iterable[ExperimentMode]) -> list[ExperimentMode]:
    result: list[ExperimentMode] = []
    seen: set[ExperimentMode] = set()
    for mode in modes:
        if mode in seen:
            raise ValueError(f"duplicate mode: {mode.value}")
        seen.add(mode)
        result.append(mode)
    return result
