"""An injection plan: ``stormlog.qualify.plan/1``.

A plan names the victim's workload, the run's timeline (#221 design A.4:
priming, baseline, episodes each after the previous one recovered, final
recovery), its episodes in order with their doses, and a seed. ``load_plan``
checks it before anything is sent: every episode type is in the catalog and
run by this harness, every dose has what its method needs, and pulses keep
the pulser's caps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .catalog import CAPTURE, NEIGHBOR, PULSE, EpisodeType, episode_type
from .neighbor import NeighborShape
from .pulser import PulseRefused, check_schedule

FORMAT = "stormlog.qualify.plan/1"


class PlanError(ValueError):
    """A plan that cannot be run; ``problems`` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Timeline:
    """A.4's run timeline, in seconds."""

    priming: float = 30.0
    baseline: float = 45.0
    episode: float = 45.0
    min_recovery: float = 60.0
    recovery_timeout: float = 150.0
    final_recovery: float = 60.0


@dataclass(frozen=True)
class Victim:
    """The victim's workload, identical in every episode of a profile: shared
    prompt prefixes in a few groups, at a steady open-loop rate, streamed,
    with its SLO."""

    rate_per_second: float = 3.0
    input_tokens: int = 512
    output_tokens: int = 64
    prefix_groups: int = 4
    shared_prefix_ratio: float = 0.75
    slo_ttft_ms: float | None = None
    slo_e2e_ms: float | None = None

    @property
    def shared_prefix_tokens(self) -> int:
        return int(self.input_tokens * self.shared_prefix_ratio)


@dataclass(frozen=True)
class EpisodePlan:
    type: str
    dose: Mapping[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> EpisodeType:
        return episode_type(self.type)

    def neighbor_shape(self) -> NeighborShape:
        dose = dict(self.dose)
        return NeighborShape(
            input_tokens=int(dose.pop("input_tokens")),
            output_tokens=int(dose.pop("output_tokens")),
            **dose,
        )


@dataclass(frozen=True)
class Plan:
    profile: str
    seed: int
    victim: Victim
    timeline: Timeline
    episodes: tuple[EpisodePlan, ...]
    binding: str = "vllm-0.30"

    def victim_duration_seconds(self) -> float:
        """Long enough for every episode to time out its recovery."""
        t = self.timeline
        per_episode = t.episode + max(t.recovery_timeout, t.min_recovery)
        return (
            t.priming + t.baseline + len(self.episodes) * per_episode + t.final_recovery
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "profile": self.profile,
            "seed": self.seed,
            "binding": self.binding,
            "victim": self.victim.__dict__,
            "timeline": self.timeline.__dict__,
            "episodes": [
                {"type": episode.type, "dose": dict(episode.dose)}
                for episode in self.episodes
            ],
        }


def parse_plan(record: Mapping[str, Any]) -> Plan:
    """A plan record checked and read.

    Raises:
        PlanError: listing every problem with it.
    """
    if record.get("format") != FORMAT:
        raise PlanError([f"format is not {FORMAT}"])
    try:
        plan = Plan(
            profile=str(record["profile"]),
            seed=int(record.get("seed", 0)),
            binding=str(record.get("binding", "vllm-0.30")),
            victim=Victim(**record.get("victim", {})),
            timeline=Timeline(**record.get("timeline", {})),
            episodes=tuple(_episode(entry) for entry in record["episodes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PlanError([f"malformed plan: {error}"]) from error
    problems = [
        problem
        for index, episode in enumerate(plan.episodes)
        for problem in _dose_problems(index, episode)
    ]
    if plan.binding != "vllm-0.30":
        problems.append(f"no binding {plan.binding!r}")
    if problems:
        raise PlanError(problems)
    return plan


def load_plan(path: Path) -> Plan:
    return parse_plan(json.loads(path.read_text(encoding="utf-8")))


def _episode(entry: Mapping[str, Any]) -> EpisodePlan:
    row = episode_type(str(entry["type"]))
    dose = {**(row.default_dose or {}), **dict(entry.get("dose") or {})}
    return EpisodePlan(row.id, dose)


def _dose_problems(index: int, episode: EpisodePlan) -> list[str]:
    where = f"episodes[{index}] ({episode.type})"
    method = episode.row.method
    try:
        if method == NEIGHBOR:
            episode.neighbor_shape()
        elif method == PULSE:
            check_schedule(
                episode.dose["pulse_ms"] / 1000, episode.dose["period_ms"] / 1000
            )
        elif method == CAPTURE and not 0 < float(episode.dose["seconds"]) <= 60:
            return [f"{where}: a capture lasts more than 0 and at most 60 s"]
    except (KeyError, TypeError, ValueError, PulseRefused) as error:
        return [f"{where}: bad dose: {error}"]
    return []


__all__ = [
    "FORMAT",
    "EpisodePlan",
    "Plan",
    "PlanError",
    "Timeline",
    "Victim",
    "load_plan",
    "parse_plan",
]
