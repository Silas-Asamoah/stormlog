"""Ground truth for injected episodes: ``stormlog.qualify.injection/1``, and
for the run they belong to: ``stormlog.qualify.run/1``.

One record per attempted episode, written by the injection harness into a
run's ``truth/injections.jsonl`` and read only by the scorer, beside one
run record in ``truth/run.json``. An episode record says
what was injected, what a correct diagnosis is (``expects``), which other
findings are neutral (``secondary`` through one of #218's edges, and
``allows``), when the action and its effect happened on the victim's clock,
and whether the episode counts: its validity in four layers and a status.

Every attempted episode is published with its status; accuracy is computed
over the ``valid`` ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bounds import fisher_greater
from .vocabulary import (
    CAUSE_FAULT,
    CAUSE_WORKLOAD_CHANGE,
    CAUSES,
    EDGES,
    KIND_COMPONENTS,
    PRIMARY,
    SEVERITIES,
    WORKLOAD_KINDS,
)

FORMAT = "stormlog.qualify.injection/1"
RUN_FORMAT = "stormlog.qualify.run/1"

CAUSE_CLASSES = ("fault", "workload_change", "instrumentation", "placebo", "none")

VALID = "valid"
INVALID_ALIGNMENT = "invalid_alignment"
NOT_ACTUATED = "not_actuated"
NOT_REALIZED = "not_realized"
INCOMPARABLE = "incomparable"
RECOVERY_INCOMPLETE = "recovery_incomplete"
PROTOCOL_FAILURE = "protocol_failure"
STATUSES = (
    VALID,
    INVALID_ALIGNMENT,
    NOT_ACTUATED,
    NOT_REALIZED,
    INCOMPARABLE,
    RECOVERY_INCOMPLETE,
    PROTOCOL_FAILURE,
)

IMPACT = "impact"
NO_IMPACT = "no_impact"
IMPACT_PARTIAL = "partial"

# A.6: the impact test, and the baseline an episode needs before it.
IMPACT_ALPHA = 0.05
IMPACT_MIN_VIOLATIONS = 3
IMPACT_MIN_COVERAGE = 0.9
MIN_BASELINE_NS = 30_000_000_000


class GroundTruthError(ValueError):
    """A record that is not a valid ``stormlog.qualify.injection/1`` record;
    ``problems`` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


# ------------------------------------------------------------------ labels


@dataclass(frozen=True)
class Location:
    """A kind at a location; a null field means "not specified"."""

    kind: str
    component: str
    rank: int | None = None
    engine: str | None = None

    def problems(self, where: str) -> list[str]:
        components = KIND_COMPONENTS.get(self.kind)
        if components is None:
            return [f"{where}: unknown kind {self.kind!r}"]
        if self.component not in components:
            return [f"{where}: {self.kind} is never at {self.component!r}"]
        return []

    def location_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "component": self.component,
            "rank": self.rank,
            "engine": self.engine,
        }


@dataclass(frozen=True)
class Expectation(Location):
    """What a correct diagnosis claims: the fault, by default, as a primary
    finding at ``warning`` or above."""

    role: str = PRIMARY
    cause: str = "fault"
    min_severity: str = "warning"

    def problems(self, where: str) -> list[str]:
        found = super().problems(where)
        if self.role != PRIMARY:
            found.append(f"{where}: an expectation is always primary")
        if self.cause not in CAUSES:
            found.append(f"{where}: unknown cause {self.cause!r}")
        if self.min_severity not in SEVERITIES:
            found.append(f"{where}: unknown severity {self.min_severity!r}")
        workload = (CAUSE_WORKLOAD_CHANGE, "info")
        if self.kind in WORKLOAD_KINDS and (self.cause, self.min_severity) != workload:
            found.append(
                f"{where}: {self.kind} is a workload kind, claimed as"
                " workload_change at info"
            )
        return found

    def to_record(self) -> dict[str, Any]:
        return {
            **self.location_record(),
            "role": self.role,
            "cause": self.cause,
            "min_severity": self.min_severity,
        }


@dataclass(frozen=True)
class Neutral(Location):
    """A finding that is neither credited nor counted as false: a declared
    secondary (which must name #218's edge to it) or an allowed finding."""

    edge: str | None = None

    def problems(self, where: str) -> list[str]:
        found = super().problems(where)
        if self.edge is None:
            return found
        edge = EDGES.get(self.edge)
        if edge is None:
            found.append(f"{where}: unknown edge {self.edge!r}")
        elif edge.downstream != self.kind:
            found.append(f"{where}: edge {self.edge} does not lead to {self.kind}")
        return found

    def to_record(self) -> dict[str, Any]:
        record = self.location_record()
        if self.edge is not None:
            record["edge"] = self.edge
        return record


# ------------------------------------------------------------------ times and validity


@dataclass(frozen=True)
class Times:
    """When the action and its effect happened, in ns on the victim's
    clock. Effect onset and end come from the reference channel at event
    time, independent of any capture's latency."""

    action_onset_ns: int | None = None
    action_end_ns: int | None = None
    effect_onset_ns: int | None = None
    effect_end_ns: int | None = None
    effect_basis: str | None = None
    first_observation_ns: int | None = None
    recovery_held_at_ns: int | None = None
    predicate_duration_ns: int | None = None
    priming_check: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _TIME_FIELDS}

    def problems(self) -> list[str]:
        found = [
            f"times.{name} must be an integer or null"
            for name in _TIME_FIELDS[:4] + _TIME_FIELDS[5:8]
            if not _is_time(getattr(self, name))
        ]
        onset, end = self.effect_onset_ns, self.effect_end_ns
        if not found and onset is not None and end is not None and end < onset:
            found.append("the effect ends before it begins")
        return found


def _is_time(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


_TIME_FIELDS = (
    "action_onset_ns",
    "action_end_ns",
    "effect_onset_ns",
    "effect_end_ns",
    "effect_basis",
    "first_observation_ns",
    "recovery_held_at_ns",
    "predicate_duration_ns",
    "priming_check",
)


@dataclass(frozen=True)
class OutcomeCounts:
    """Victim requests' SLO outcomes in one interval: a missed SLO, an
    unreachable request included, is a violation; unknown is not."""

    violations: int = 0
    met: int = 0
    unknown: int = 0

    @property
    def coverage(self) -> float | None:
        total = self.violations + self.met + self.unknown
        return None if total == 0 else (self.violations + self.met) / total

    def to_record(self) -> dict[str, int]:
        return {"violations": self.violations, "met": self.met, "unknown": self.unknown}


@dataclass(frozen=True)
class Impact:
    """Layer 4: did the episode hurt the victim's SLO?"""

    status: str
    effect: OutcomeCounts
    baseline: OutcomeCounts
    p_value: float | None = None
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "p_value": self.p_value,
            "effect": self.effect.to_record(),
            "baseline": self.baseline.to_record(),
        }


def assess_impact(effect: OutcomeCounts, baseline: OutcomeCounts) -> Impact:
    """A.6 layer 4: impact when the effect window's violations are more
    likely than the baseline's (one-sided Fisher exact, α = 0.05) and at
    least 3; partial when the effect window's SLO evidence covers less than
    0.9 of its requests."""
    coverage = effect.coverage
    if coverage is None or coverage < IMPACT_MIN_COVERAGE:
        return Impact(IMPACT_PARTIAL, effect, baseline, reason="slo_evidence_coverage")
    p_value = fisher_greater(
        violations=effect.violations,
        met=effect.met,
        baseline_violations=baseline.violations,
        baseline_met=baseline.met,
    )
    hurt = effect.violations >= IMPACT_MIN_VIOLATIONS and p_value < IMPACT_ALPHA
    return Impact(IMPACT if hurt else NO_IMPACT, effect, baseline, p_value)


@dataclass(frozen=True)
class Validity:
    """The four layers (A.6): whether the action happened, whether the
    mechanism occurred, whether the diagnosed configuration's capture was
    complete (reported, never used to drop an episode), and victim impact."""

    actuation: str = "ok"
    realization: str = "realized"
    observation: str = "complete"
    impact: Impact | None = None
    realized_mechanisms: tuple[str, ...] = ()
    checks: tuple[dict[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "actuation": self.actuation,
            "realization": self.realization,
            "observation": self.observation,
            "impact": None if self.impact is None else self.impact.to_record(),
            "realized_mechanisms": list(self.realized_mechanisms),
            "checks": [dict(check) for check in self.checks],
        }


# ------------------------------------------------------------------ the record


@dataclass(frozen=True)
class Injection:
    """One attempted episode's ground truth, in the run ``run_id``."""

    episode_id: str
    run_id: str
    episode_type: str
    cause_class: str
    injected: dict[str, Any]
    expects: tuple[Expectation, ...]
    times: Times
    clock_domain: str | None
    status: str
    secondary: tuple[Neutral, ...] = ()
    allows: tuple[Neutral, ...] = ()
    actions: tuple[dict[str, Any], ...] = ()
    validity: Validity = field(default_factory=Validity)

    def problems(self) -> list[str]:
        found: list[str] = []
        if self.cause_class not in CAUSE_CLASSES:
            found.append(f"unknown cause_class {self.cause_class!r}")
        if self.status not in STATUSES:
            found.append(f"unknown status {self.status!r}")
        for index, expectation in enumerate(self.expects):
            found += expectation.problems(f"expects[{index}]")
        for index, entry in enumerate(self.secondary):
            if entry.edge is None:
                found.append(f"secondary[{index}]: a secondary names its edge")
            found += entry.problems(f"secondary[{index}]")
        for index, entry in enumerate(self.allows):
            found += entry.problems(f"allows[{index}]")
        return (
            found
            + self._label_problems()
            + self.times.problems()
            + self._valid_problems()
        )

    def _label_problems(self) -> list[str]:
        """What a correct diagnosis can be, for this cause class: a fault
        episode expects exactly one fault, claimed at warning; any other
        episode expects no fault."""
        faults = [e for e in self.expects if e.cause == CAUSE_FAULT]
        if self.cause_class != "fault":
            return [f"a {self.cause_class} episode expects no fault"] if faults else []
        if len(self.expects) != 1:
            return ["a fault episode expects exactly one finding"]
        expectation = self.expects[0]
        found = []
        if expectation.cause != CAUSE_FAULT:
            found.append("expects[0]: a fault episode expects cause 'fault'")
        if expectation.min_severity != "warning":
            found.append("expects[0]: a fault is claimed at warning")
        return found

    def _valid_problems(self) -> list[str]:
        """A valid episode passed every layer and has its effect window."""
        if self.status != VALID:
            return []
        found = [
            f"status valid, but {layer} is {value!r}"
            for layer, value, ok in (
                ("actuation", self.validity.actuation, "ok"),
                ("realization", self.validity.realization, "realized"),
            )
            if value != ok
        ]
        if self.times.effect_onset_ns is None or self.times.effect_end_ns is None:
            found.append("a valid episode needs its effect onset and end")
        return found

    def to_record(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "episode_type": self.episode_type,
            "cause_class": self.cause_class,
            "injected": self.injected,
            "expects": [entry.to_record() for entry in self.expects],
            "secondary": [entry.to_record() for entry in self.secondary],
            "allows": [entry.to_record() for entry in self.allows],
            "times": self.times.to_record(),
            "clock_domain": self.clock_domain,
            "actions": [dict(action) for action in self.actions],
            "validity": self.validity.to_record(),
            "status": self.status,
        }


def parse_injection(record: Mapping[str, Any]) -> Injection:
    """A record read back into an ``Injection``.

    Raises:
        GroundTruthError: listing every problem with the record.
    """
    if record.get("format") != FORMAT:
        raise GroundTruthError([f"format is not {FORMAT}"])
    try:
        injection = _build(record)
    except (KeyError, TypeError, ValueError) as error:
        raise GroundTruthError([f"malformed record: {error!r}"]) from error
    problems = injection.problems()
    if problems:
        raise GroundTruthError(problems)
    return injection


def _build(record: Mapping[str, Any]) -> Injection:
    expects, secondary, allows = _labels(record)
    return Injection(
        episode_id=str(record["episode_id"]),
        run_id=str(record["run_id"]),
        episode_type=str(record["episode_type"]),
        cause_class=str(record["cause_class"]),
        injected=dict(record.get("injected") or {}),
        expects=expects,
        secondary=secondary,
        allows=allows,
        times=Times(**(record.get("times") or {})),
        clock_domain=record.get("clock_domain"),
        actions=tuple(dict(action) for action in record.get("actions") or ()),
        validity=_validity(record.get("validity") or {}),
        status=str(record["status"]),
    )


def _labels(
    record: Mapping[str, Any],
) -> tuple[tuple[Expectation, ...], tuple[Neutral, ...], tuple[Neutral, ...]]:
    expects = tuple(Expectation(**entry) for entry in record.get("expects") or ())
    secondary = tuple(Neutral(**entry) for entry in record.get("secondary") or ())
    allows = tuple(Neutral(**entry) for entry in record.get("allows") or ())
    return expects, secondary, allows


def _validity(record: Mapping[str, Any]) -> Validity:
    impact = record.get("impact")
    return Validity(
        actuation=record.get("actuation", "ok"),
        realization=record.get("realization", "realized"),
        observation=record.get("observation", "complete"),
        impact=None if impact is None else _impact(impact),
        realized_mechanisms=tuple(record.get("realized_mechanisms") or ()),
        checks=tuple(dict(check) for check in record.get("checks") or ()),
    )


def _impact(record: Mapping[str, Any]) -> Impact:
    return Impact(
        status=record["status"],
        effect=OutcomeCounts(**record["effect"]),
        baseline=OutcomeCounts(**record["baseline"]),
        p_value=record.get("p_value"),
        reason=record.get("reason"),
    )


def write_injections(path: Path, injections: Iterable[Injection]) -> None:
    """Write records as JSON lines, one per attempted episode."""
    lines = [
        json.dumps(injection.to_record(), sort_keys=True) for injection in injections
    ]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def load_injections(path: Path) -> list[Injection]:
    """Every record in ``truth/injections.jsonl``.

    Raises:
        GroundTruthError: naming the line of the first bad record.
    """
    injections: list[Injection] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            injection = parse_injection(json.loads(line))
        except GroundTruthError as error:
            raise GroundTruthError(
                [f"line {number}: {problem}" for problem in error.problems]
            ) from error
        if injection.episode_id in seen:
            raise GroundTruthError(
                [f"line {number}: episode {injection.episode_id} is written twice"]
            )
        seen.add(injection.episode_id)
        injections.append(injection)
    return injections


# ------------------------------------------------------------------ the run


@dataclass(frozen=True)
class Interval:
    """[start_ns, end_ns] on the victim's clock."""

    start_ns: int
    end_ns: int

    @property
    def length_ns(self) -> int:
        return self.end_ns - self.start_ns

    def to_record(self) -> dict[str, int]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns}


@dataclass(frozen=True)
class RunRecord:
    """One run's truth beside its episodes: the windows the harness
    measured, on the victim's clock, and whether the run itself failed its
    protocol (a failed priming check, say). The scorer derives the run's
    negative exposure from these and the run's episodes."""

    run_id: str
    clock_domain: str | None
    measured: Interval
    priming: Interval | None = None
    baseline: Interval | None = None
    final_recovery: Interval | None = None
    protocol_failure: str | None = None
    actions: tuple[dict[str, Any], ...] = ()

    def problems(self) -> list[str]:
        found = []
        for name in ("measured", "priming", "baseline", "final_recovery"):
            interval = getattr(self, name)
            if interval is None:
                continue
            if not (_is_time(interval.start_ns) and _is_time(interval.end_ns)):
                found.append(f"{name}: times must be integers")
            elif interval.end_ns < interval.start_ns:
                found.append(f"{name} ends before it begins")
        return found

    def to_record(self) -> dict[str, Any]:
        return {
            "format": RUN_FORMAT,
            "run_id": self.run_id,
            "clock_domain": self.clock_domain,
            "measured": self.measured.to_record(),
            "priming": _interval_record(self.priming),
            "baseline": _interval_record(self.baseline),
            "final_recovery": _interval_record(self.final_recovery),
            "protocol_failure": self.protocol_failure,
            "actions": [dict(action) for action in self.actions],
        }


def _interval_record(interval: Interval | None) -> dict[str, int] | None:
    return None if interval is None else interval.to_record()


def parse_run(record: Mapping[str, Any]) -> RunRecord:
    """A ``truth/run.json`` record read back.

    Raises:
        GroundTruthError: listing every problem with the record.
    """
    if record.get("format") != RUN_FORMAT:
        raise GroundTruthError([f"format is not {RUN_FORMAT}"])
    try:
        run = RunRecord(
            run_id=str(record["run_id"]),
            clock_domain=record.get("clock_domain"),
            measured=Interval(**record["measured"]),
            priming=_interval(record.get("priming")),
            baseline=_interval(record.get("baseline")),
            final_recovery=_interval(record.get("final_recovery")),
            protocol_failure=record.get("protocol_failure"),
            actions=tuple(dict(action) for action in record.get("actions") or ()),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GroundTruthError([f"malformed run record: {error!r}"]) from error
    problems = run.problems()
    if problems:
        raise GroundTruthError(problems)
    return run


def _interval(record: Mapping[str, Any] | None) -> Interval | None:
    return None if record is None else Interval(**record)


def write_run(path: Path, run: RunRecord) -> None:
    path.write_text(
        json.dumps(run.to_record(), sort_keys=True) + "\n", encoding="utf-8"
    )


def load_run(path: Path) -> RunRecord:
    return parse_run(json.loads(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ status


@dataclass(frozen=True)
class PhaseWindow:
    """The victim's measured window, on its own clock."""

    start_ns: int
    end_ns: int


def is_aligned(
    times: Times,
    window: PhaseWindow,
    *,
    clean_since_ns: int | None = None,
    min_baseline_ns: int = MIN_BASELINE_NS,
) -> bool:
    """Whether an episode lies wholly inside the victim's measured window,
    with at least ``min_baseline_ns`` of clean time before its action:
    since the window opened, or since the previous episode recovered."""
    onset = times.action_onset_ns
    end = (
        times.effect_end_ns if times.effect_end_ns is not None else times.action_end_ns
    )
    if onset is None or end is None:
        return False
    if onset < window.start_ns or end > window.end_ns:
        return False
    clean_from = max(window.start_ns, clean_since_ns or window.start_ns)
    return onset - clean_from >= min_baseline_ns


def decide_status(
    *,
    protocol_failure: bool = False,
    same_clock: bool = True,
    actuated: bool = True,
    aligned: bool = True,
    realized: bool = True,
    recovered: bool = True,
) -> str:
    """An episode's status: the first failure, in this order, or ``valid``.
    A protocol failure (a failed priming check, a shortfall the plan did not
    allow) voids the episode before any other judgement."""
    for failed, status in (
        (protocol_failure, PROTOCOL_FAILURE),
        (not same_clock, INCOMPARABLE),
        (not actuated, NOT_ACTUATED),
        (not aligned, INVALID_ALIGNMENT),
        (not realized, NOT_REALIZED),
        (not recovered, RECOVERY_INCOMPLETE),
    ):
        if failed:
            return status
    return VALID


__all__ = [
    "CAUSE_CLASSES",
    "FORMAT",
    "IMPACT",
    "IMPACT_PARTIAL",
    "INCOMPARABLE",
    "INVALID_ALIGNMENT",
    "MIN_BASELINE_NS",
    "NOT_ACTUATED",
    "NOT_REALIZED",
    "NO_IMPACT",
    "PROTOCOL_FAILURE",
    "RECOVERY_INCOMPLETE",
    "RUN_FORMAT",
    "STATUSES",
    "VALID",
    "Expectation",
    "GroundTruthError",
    "Impact",
    "Injection",
    "Interval",
    "Location",
    "Neutral",
    "OutcomeCounts",
    "PhaseWindow",
    "RunRecord",
    "Times",
    "Validity",
    "assess_impact",
    "decide_status",
    "is_aligned",
    "load_injections",
    "load_run",
    "parse_injection",
    "parse_run",
    "write_injections",
    "write_run",
]
