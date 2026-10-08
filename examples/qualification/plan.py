"""An injection plan: ``stormlog.qualify.plan/1``.

A plan names the victim's workload, the run's timeline (#221 design A.4:
priming, baseline, episodes each after the previous one recovered, final
recovery), its episodes in order with their doses, and a seed. ``load_plan``
checks it before anything is sent, and lists every problem: every episode
type is in the catalog and run by this harness, every dose has what its
method needs, pulses keep the pulser's caps, the timeline's and the
victim's values are numbers in range, every threshold override is a
known one, and a plan with queue episodes has a baseline and a hold long
enough for the samples queue recovery needs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from stormlog.infer.qualify.recovery import SECOND, Thresholds, base_type

from .catalog import CAPTURE, NEIGHBOR, PULSE, EpisodeType, episode_type
from .neighbor import NeighborShape
from .pulser import PulseRefused, check_schedule

FORMAT = "stormlog.qualify.plan/1"
# inject scrapes the engine's gauges once a poll, and polls once a second.
POLL_SECONDS = 1.0


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
    # The clean time an episode needs before its action (alignment).
    min_clean: float = 30.0


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
    # Overrides of recovery's frozen thresholds, in seconds (window, hold,
    # cadence_hold, priming_window), sample counts (min_wait_samples,
    # min_gauge_samples) or fractions (the rest).
    thresholds: Mapping[str, float] = field(default_factory=dict)

    def recovery_thresholds(self) -> Thresholds:
        """Recovery's thresholds: the design's, the plan's overrides, and the
        timeline's recovery minimum and timeout."""
        values: dict[str, Any] = {
            "min_recovery_ns": int(self.timeline.min_recovery * SECOND),
            "recovery_timeout_ns": int(self.timeline.recovery_timeout * SECOND),
        }
        for name, value in self.thresholds.items():
            if name in _SECONDS:
                values[f"{name}_ns"] = int(value * SECOND)
            elif name in _COUNTS:
                values[name] = int(value)
            else:
                values[name] = float(value)
        return Thresholds(**values)

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
            "thresholds": dict(self.thresholds),
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
    entries = record.get("episodes")
    if not isinstance(entries, list):
        raise PlanError(["episodes must be a list"])
    episodes, problems = _episodes(entries)
    sections = [
        *_section_problems("timeline", record.get("timeline", {}), _TIMELINE),
        *_section_problems("victim", record.get("victim", {}), _VICTIM),
        *_threshold_problems(record.get("thresholds") or {}),
    ]
    problems += sections
    problems += _order_problems(record.get("timeline", {}))
    if record.get("binding", "vllm-0.30") != "vllm-0.30":
        problems.append(f"no binding {record.get('binding')!r}")
    if not sections:
        # With the windows and rates readable, their samples are checked in
        # the same round: every problem is listed at once.
        problems += _sample_problems(_plan_of(record, episodes, "", 0))
    if problems:
        raise PlanError(problems)
    try:
        return _plan_of(
            record, episodes, str(record["profile"]), int(record.get("seed", 0))
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PlanError([f"malformed plan: {error}"]) from error


def _plan_of(
    record: Mapping[str, Any], episodes: list[EpisodePlan], profile: str, seed: int
) -> Plan:
    return Plan(
        profile=profile,
        seed=seed,
        victim=Victim(**record.get("victim", {})),
        timeline=Timeline(**record.get("timeline", {})),
        episodes=tuple(episodes),
        thresholds=dict(record.get("thresholds") or {}),
    )


def _episodes(entries: list[Any]) -> tuple[list[EpisodePlan], list[str]]:
    episodes: list[EpisodePlan] = []
    problems = [] if entries else ["a plan has at least one episode"]
    for index, entry in enumerate(entries):
        try:
            episode = _episode(entry)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"episodes[{index}]: {error}")
            continue
        episodes.append(episode)
        problems += _dose_problems(index, episode)
    return episodes, problems


# Each section's fields: their kind, and whether 0 is allowed.
_POSITIVE, _NON_NEGATIVE, _COUNT, _RATIO, _OPTIONAL = (
    "positive", "non_negative", "count", "ratio", "optional",
)  # fmt: skip
_TIMELINE = {
    "priming": _POSITIVE,
    "baseline": _POSITIVE,
    "episode": _POSITIVE,
    "min_recovery": _NON_NEGATIVE,
    "recovery_timeout": _POSITIVE,
    "final_recovery": _NON_NEGATIVE,
    "min_clean": _NON_NEGATIVE,
}
_VICTIM = {
    "rate_per_second": _POSITIVE,
    "input_tokens": _COUNT,
    "output_tokens": _COUNT,
    "prefix_groups": _COUNT,
    "shared_prefix_ratio": _RATIO,
    "slo_ttft_ms": _OPTIONAL,
    "slo_e2e_ms": _OPTIONAL,
}


def _section_problems(name: str, values: Any, kinds: Mapping[str, str]) -> list[str]:
    if not isinstance(values, Mapping):
        return [f"{name} must be an object"]
    problems = [f"{name}.{key} is not a field" for key in values if key not in kinds]
    for key, value in values.items():
        kind = kinds.get(key)
        if kind is not None and not _fits(value, kind):
            problems.append(f"{name}.{key} must be {_DESCRIBED[kind]}")
    return problems


_DESCRIBED = {
    _POSITIVE: "a positive number",
    _NON_NEGATIVE: "a number, 0 or more",
    _COUNT: "a positive integer",
    _RATIO: "a number in (0, 1]",
    _OPTIONAL: "null or a positive number",
}


def _fits(value: Any, kind: str) -> bool:
    if kind == _OPTIONAL and value is None:
        return True
    if kind == _COUNT:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if not _is_number(value):
        return False
    if kind == _NON_NEGATIVE:
        return bool(value >= 0)
    if kind == _RATIO:
        return bool(0 < value <= 1)
    return bool(value > 0)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _order_problems(timeline: Any) -> list[str]:
    if not isinstance(timeline, Mapping):
        return []
    defaults = {f.name: f.default for f in fields(Timeline)}
    least = timeline.get("min_recovery", defaults["min_recovery"])
    most = timeline.get("recovery_timeout", defaults["recovery_timeout"])
    if _is_number(least) and _is_number(most) and least > most:
        return ["timeline.min_recovery is longer than timeline.recovery_timeout"]
    return []


def _threshold_problems(thresholds: Any) -> list[str]:
    if not isinstance(thresholds, Mapping):
        return ["thresholds must be an object"]
    known = _SECONDS | _COUNTS | _FRACTIONS
    problems = []
    for name, value in thresholds.items():
        if name not in known:
            problems.append(f"thresholds.{name} is not a threshold")
        elif name in _COUNTS and not _fits(value, _COUNT):
            problems.append(f"thresholds.{name} must be {_DESCRIBED[_COUNT]}")
        elif not _is_number(value):
            problems.append(f"thresholds.{name} must be a number")
    return problems


# The types whose recovery is judged on the victim's waits and the waiting
# gauge (recovery's queue criteria), and their short twins.
_QUEUE_TYPES = frozenset({"F1", "T1", "W1"})


def _sample_problems(plan: Plan, poll_seconds: float = POLL_SECONDS) -> list[str]:
    """A baseline or hold too short for the samples queue recovery needs.

    A queue criterion holds over a hold with at least ``min_gauge_samples``
    waiting counts and ``min_wait_samples`` waits, and never holds at all
    when the baseline has fewer: such a plan's F1, T1 and W1 would always
    time out. Each window needs one scrape more than the minimum, since a
    poll takes a little longer than its period, and the victim's expected
    requests in it.
    """
    if not any(base_type(e.type) in _QUEUE_TYPES for e in plan.episodes):
        return []
    thresholds = plan.recovery_thresholds()
    windows = {
        "timeline.baseline": plan.timeline.baseline,
        "thresholds.hold": thresholds.hold_ns / SECOND,
    }
    scrapes = (thresholds.min_gauge_samples + 1) * poll_seconds
    rate = plan.victim.rate_per_second
    problems = []
    for name, seconds in windows.items():
        if seconds < scrapes:
            problems.append(
                f"{name} is {seconds:g} s: queue recovery needs "
                f"{thresholds.min_gauge_samples} waiting counts, at least "
                f"{scrapes:g} s at one scrape per {poll_seconds:g} s"
            )
        if seconds * rate < thresholds.min_wait_samples:
            problems.append(
                f"{name} is {seconds:g} s: queue recovery needs "
                f"{thresholds.min_wait_samples} victim waits, at least "
                f"{thresholds.min_wait_samples / rate:g} s at {rate:g} requests/s"
            )
    return problems


_SECONDS = frozenset({"window", "hold", "cadence_hold", "priming_window"})
_COUNTS = frozenset({"min_wait_samples", "min_gauge_samples"})
_FRACTIONS = frozenset(
    {
        "cached_loss_below",
        "cached_recovered_at",
        "kv_margin",
        "priming_cached_at_least",
        "rate_tolerance",
        "long_gap_factor",
        "exceedance_share",
        "exceedance_quantile",
        "hit_ratio_drop",
    }
)


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
    "POLL_SECONDS",
    "EpisodePlan",
    "Plan",
    "PlanError",
    "Timeline",
    "Victim",
    "load_plan",
    "parse_plan",
]
