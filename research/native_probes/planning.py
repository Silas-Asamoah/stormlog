"""Counterbalanced trial planning without executing profiler commands."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from .mode_commands import (
    Workload,
    expected_artifacts,
    microbenchmark_command,
    process_roles,
)
from .models import ExperimentMode, WorkloadId


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


def build_plan(
    *,
    configuration_id: str,
    vendor: str,
    workloads: Sequence[WorkloadId],
    modes: Sequence[ExperimentMode],
    repetitions: int,
    seed: int,
    environment_artifact: str,
    artifact_root: Path,
    cupti_library: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic, serializable execution plan without running it."""
    if vendor not in {"nvidia", "amd"}:
        raise ValueError("vendor must be 'nvidia' or 'amd'")
    orders = counterbalanced_order(modes, repetitions=repetitions, seed=seed)
    trials = []
    for workload_id in workloads:
        workload = Workload(workload_id, seed=seed)
        for repetition, order in enumerate(orders):
            for mode in order:
                identity = trial_id(
                    configuration_id, workload_id.value, mode, repetition
                )
                directory = artifact_root / configuration_id / "trials" / identity
                command = microbenchmark_command(
                    mode,
                    workload,
                    directory,
                    cupti_library=cupti_library,
                    vendor=vendor,
                )
                trials.append(
                    {
                        "trial_id": identity,
                        "configuration_id": configuration_id,
                        "workload_id": workload_id.value,
                        "mode": mode.value,
                        "repetition": repetition,
                        "command": {
                            "argv": list(command.argv),
                            "environment": dict(command.environment),
                            "timeout_seconds": command.timeout_seconds,
                        },
                        "expected_artifacts": [
                            row.__dict__ for row in expected_artifacts(mode, vendor)
                        ],
                        "process_roles": [
                            {
                                "role": row.role.value,
                                "discovery": row.discovery,
                                "argv_contains": row.argv_contains,
                            }
                            for row in process_roles(mode)
                        ],
                        "measurement_range_id": workload.measurement_range_id,
                    }
                )
    return {
        "schema_version": 1,
        "artifact_kind": "native_probe_plan",
        "configuration_id": configuration_id,
        "vendor": vendor,
        "seed": seed,
        "repetitions": repetitions,
        "environment_artifact": environment_artifact,
        "artifact_root": str(artifact_root),
        "trials": trials,
    }
