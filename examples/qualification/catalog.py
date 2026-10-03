"""#221's episode catalog (design A.4): each type's label and how it is injected.

A type's label is what a correct diagnosis claims (``expects``), which other
findings are neutral (``secondary`` through one of #218's edges, and
``allows``), and its cause class. Its method is how the harness injects it:
neighbor traffic, pulses to a role's process, a profiler window, or nothing.

This harness covers the single-GPU offline types. The short twins (S-x),
the TP=2 types (F5, R0, F6) and the outages (X1–X3) need #219's predicates, a
second GPU and #220's tools; plans that name them are refused for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from stormlog.infer.qualify.ground_truth import Expectation, Neutral
from stormlog.infer.qualify.vocabulary import (
    CAPTURE_PAUSE,
    HOST_STALL,
    KV_PREEMPTION_PRESSURE,
    LOAD_INCREASE,
    LONGER_INPUTS,
    LONGER_OUTPUTS,
    MIXED_PREFILL_INTERFERENCE,
    PREFIX_CACHE_LOSS,
    QUEUE_SATURATION,
)

NEIGHBOR = "neighbor"
PULSE = "pulse"
CAPTURE = "capture"
NOTHING = "none"

# Reserved for campaigns this harness doesn't run yet.
NOT_YET = ("S-", "F5", "R0", "F6", "X1", "X2", "X3")


@dataclass(frozen=True)
class EpisodeType:
    """One row of A.4's catalog."""

    id: str
    cause_class: str
    method: str
    expects: tuple[Expectation, ...] = ()
    secondary: tuple[Neutral, ...] = ()
    allows: tuple[Neutral, ...] = ()
    pulse_role: str | None = None
    default_dose: Mapping[str, Any] | None = None


def _fault(kind: str, component: str) -> tuple[Expectation, ...]:
    return (Expectation(kind, component),)


def _workload(kind: str) -> tuple[Expectation, ...]:
    return (
        Expectation(kind, "workload", cause="workload_change", min_severity="info"),
    )


def _via(upstream: str, kind: str, component: str) -> Neutral:
    return Neutral(kind, component, edge=f"{upstream}->{kind}")


def _allowed(*kinds: str) -> tuple[Neutral, ...]:
    return tuple(Neutral(kind, "workload") for kind in kinds)


_LONG = {"input_tokens": 2048, "output_tokens": 1024}
_PULSE = {"pulse_ms": 100, "period_ms": 2000}

CATALOG: dict[str, EpisodeType] = {
    episode.id: episode
    for episode in (
        EpisodeType(
            "F1",
            "fault",
            NEIGHBOR,
            _fault(QUEUE_SATURATION, "scheduler"),
            (_via(QUEUE_SATURATION, MIXED_PREFILL_INTERFERENCE, "scheduler"),),
            _allowed(LOAD_INCREASE),
        ),
        EpisodeType("T1", "workload_change", NEIGHBOR, _workload(LOAD_INCREASE)),
        EpisodeType(
            "F2",
            "fault",
            NEIGHBOR,
            _fault(KV_PREEMPTION_PRESSURE, "kv_cache"),
            (
                _via(KV_PREEMPTION_PRESSURE, QUEUE_SATURATION, "scheduler"),
                _via(KV_PREEMPTION_PRESSURE, MIXED_PREFILL_INTERFERENCE, "scheduler"),
            ),
            _allowed(LOAD_INCREASE, LONGER_INPUTS, LONGER_OUTPUTS),
            default_dose={"concurrency": 8, **_LONG},
        ),
        # No #218 edge leads from a workload change to mixed prefill, so T2
        # allows it rather than declaring it secondary.
        EpisodeType(
            "T2",
            "workload_change",
            NEIGHBOR,
            _workload(LONGER_INPUTS),
            allows=(Neutral(MIXED_PREFILL_INTERFERENCE, "scheduler"),)
            + _allowed(LOAD_INCREASE),
            default_dose={"concurrency": 2, **_LONG},
        ),
        EpisodeType(
            "F3",
            "fault",
            NEIGHBOR,
            _fault(PREFIX_CACHE_LOSS, "prefix_cache"),
            (_via(PREFIX_CACHE_LOSS, MIXED_PREFILL_INTERFERENCE, "scheduler"),),
            _allowed(LOAD_INCREASE),
        ),
        EpisodeType("T3", "workload_change", NEIGHBOR, _workload(LOAD_INCREASE)),
        EpisodeType("T3b", "workload_change", NEIGHBOR, _workload(LOAD_INCREASE)),
        EpisodeType(
            "F4a",
            "fault",
            PULSE,
            _fault(HOST_STALL, "engine_core"),
            (_via(HOST_STALL, QUEUE_SATURATION, "scheduler"),),
            pulse_role="engine_core",
            default_dose=_PULSE,
        ),
        EpisodeType(
            "F4b",
            "fault",
            PULSE,
            _fault(HOST_STALL, "api_server"),
            pulse_role="api_server",
            default_dose=_PULSE,
        ),
        EpisodeType(
            "H0",
            "workload_change",
            PULSE,
            pulse_role="engine_core",
            default_dose={"pulse_ms": 5, "period_ms": 2000},
        ),
        EpisodeType(
            "W1",
            "workload_change",
            NEIGHBOR,
            _workload(LONGER_OUTPUTS),
            allows=_allowed(LOAD_INCREASE),
        ),
        EpisodeType(
            "I1",
            "instrumentation",
            CAPTURE,
            allows=(
                Neutral(CAPTURE_PAUSE, "profiler"),
                _via(CAPTURE_PAUSE, HOST_STALL, "engine_core"),
            ),
            default_dose={"seconds": 10},
        ),
        EpisodeType("P", "placebo", PULSE, pulse_role="sidecar", default_dose=_PULSE),
        EpisodeType("N", "none", NOTHING),
    )
}


def episode_type(name: str) -> EpisodeType:
    """A catalog row.

    Raises:
        ValueError: for a type the catalog doesn't have, or one this
            harness doesn't run yet.
    """
    if name in CATALOG:
        return CATALOG[name]
    if name.startswith(NOT_YET):
        raise ValueError(f"episode type {name} is not run by this harness yet")
    raise ValueError(f"unknown episode type {name!r}")


__all__ = [
    "CAPTURE",
    "CATALOG",
    "NEIGHBOR",
    "NOTHING",
    "PULSE",
    "EpisodeType",
    "episode_type",
]
