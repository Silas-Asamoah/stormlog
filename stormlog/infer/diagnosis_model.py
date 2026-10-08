"""What a diagnosis finding is, and the rules that grade it.

A finding names a kind at a location for one subject, with the observations
that support it, the competing mechanisms and whether each was ruled out,
and an experiment that would confirm it. Three rules keep a finding honest:

- **Eligibility.** A kind may claim a fault only when its gates hold and
  every competitor indispensable to it was ruled out; an untestable one
  fails the gate. An ineligible finding is ``claim: observation``, cause
  ``undetermined``, at ``info``, and lists what failed. A competitor shown
  to explain a material but minor share is ``contributing``: the finding
  stays eligible, but it is contested and makes no fault claim, as it is
  with a cause ``upstream`` of it. Every finding is primary in this
  version: the edge table that makes one secondary to another comes later.
- **Confidence** is ordinal and per claim: whether the mechanism occurred
  (condition), and whether it explains the incident (contribution). Its
  level is the lower of the two.
- **Severity.** ``warning`` needs an eligible claim, an incident subject,
  a condition and a contribution of at least medium, and the contribution
  criterion that says the mechanism explains the incident (``explains``)
  met: one unmet criterion may lower confidence, never that one. Only then
  is the cause ``fault``.

Findings are ranked in one total order, so a scorer's top three is never a
tie broken by chance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

from .diagnosis_inputs import Line
from .diagnosis_vocabulary import (
    CAUSE_FAULT,
    CAUSE_INSTRUMENTATION,
    CAUSE_UNDETERMINED,
    CAUSE_WORKLOAD_CHANGE,
    INSTRUMENTATION_KINDS,
    WORKLOAD_KINDS,
    check_kind,
)

ASSESSED = "assessed"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
NOT_OBSERVED = "not_observed"

PRIMARY = "primary"
SECONDARY = "secondary"

CLAIM_FAULT = "fault"
CLAIM_CONDITION = "condition"
CLAIM_OBSERVATION = "observation"

RULED_OUT = "ruled_out"
CONTRIBUTING = "contributing"
UNTESTABLE = "untestable"
NOT_RULED_OUT = "not_ruled_out"
UPSTREAM = "upstream"

HIGH, MEDIUM, LOW = "high", "medium", "low"
_LEVELS = (LOW, MEDIUM, HIGH)
DRIVER_UNDETERMINED = "undetermined"
SUPPORT_LIMIT = 10_000
DISPLAY_LIMIT = 8


@dataclass(frozen=True)
class Criteria:
    """One claim's rubric: what was met and what was not."""

    met: tuple[str, ...] = ()
    unmet: tuple[str, ...] = ()
    known_loss: bool = False
    coverage_unknown: bool = False
    assessed: bool = True  # False: a claim this version does not judge

    @property
    def level(self) -> str:
        """High when everything was met; medium with one miss, or with
        coverage unknown; low otherwise, or with known loss."""
        if self.known_loss:
            return LOW
        misses = len(self.unmet) + (1 if self.coverage_unknown else 0)
        return HIGH if misses == 0 else MEDIUM if misses == 1 else LOW

    def as_dict(self) -> dict[str, Any]:
        if not self.assessed:
            return {"level": None, "met": [], "unmet": list(self.unmet)}
        coverage = (
            "known_loss"
            if self.known_loss
            else "unknown" if self.coverage_unknown else "observed"
        )
        return {
            "level": self.level,
            "met": list(self.met),
            "unmet": list(self.unmet),
            "coverage": coverage,
        }


@dataclass(frozen=True)
class Alternative:
    """A competing mechanism and what became of it."""

    kind: str
    status: str  # ruled_out, contributing, untestable, not_ruled_out, upstream
    reason: str
    indispensable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "indispensable": self.indispensable,
        }


@dataclass(frozen=True)
class Observation:
    """A measured statement, with its numbers."""

    id: str
    statement: str
    metric: str
    value: float | int | None
    ci: tuple[float, float] | None = None
    n: int | None = None
    n_ref: int | None = None
    provenance: str = "observed"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "statement": self.statement,
            "metric": self.metric,
            "value": self.value,
            "provenance": self.provenance,
        }
        if self.ci is not None:
            out["ci"] = list(self.ci)
        if self.n is not None:
            out["n"] = self.n
        if self.n_ref is not None:
            out["n_ref"] = self.n_ref
        return out


@dataclass
class Finding:
    """One finding, before it is graded and ranked."""

    kind: str
    component: str
    subject: dict[str, Any]
    title: str
    message: str
    status: str = ASSESSED
    gates: dict[str, bool] = field(default_factory=dict)
    alternatives: list[Alternative] = field(default_factory=list)
    condition: Criteria = field(default_factory=Criteria)
    contribution: Criteria = field(default_factory=Criteria)
    # The latency excess it explains, its lower bound in ms: compared across
    # kinds by rank, so every kind gives it in the same unit.
    contribution_lower: float | None = None
    observations: list[Observation] = field(default_factory=list)
    location: dict[str, Any] = field(default_factory=dict)
    window: dict[str, Any] | None = None
    segments: dict[str, Any] | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] | None = None
    support: list[Line] = field(default_factory=list)
    display: list[Line] = field(default_factory=list)
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    first_detectable_ns: int | None = None
    partial_reasons: list[str] = field(default_factory=list)
    role: str = PRIMARY
    secondary_to: list[str] = field(default_factory=list)
    incident: bool = False  # whether the subject's impact is supported
    # The contribution criterion that says the mechanism explains the
    # incident; a kind without one never warns.
    explains: str | None = None

    def __post_init__(self) -> None:
        check_kind(self.kind)

    # ------------------------------------------------------------ grading
    @property
    def failed_gates(self) -> list[str]:
        failed = [name for name, held in sorted(self.gates.items()) if not held]
        failed += [
            f"competitor:{alternative.kind}:{alternative.status}"
            for alternative in self.alternatives
            if alternative.indispensable
            and alternative.status not in (RULED_OUT, CONTRIBUTING)
        ]
        return failed

    @property
    def contested(self) -> list[str]:
        """Competitors that leave the mechanism standing but not as the
        fault: an indispensable one that explains a minor share itself, or
        a cause upstream of it, of which it may be the consequence."""
        return [
            f"competitor:{alternative.kind}:{alternative.status}"
            for alternative in self.alternatives
            if alternative.status == UPSTREAM
            or (alternative.indispensable and alternative.status == CONTRIBUTING)
        ]

    @property
    def eligible(self) -> bool:
        return self.status != UNSUPPORTED and not self.failed_gates

    @property
    def confidence_level(self) -> str:
        """The lower of the condition's and the contribution's levels; the
        condition's alone when the contribution is not assessed."""
        claims = [self.condition, self.contribution]
        assessed = [claim.level for claim in claims if claim.assessed]
        level = min(assessed, key=_LEVELS.index)
        if self.status == PARTIAL and level == HIGH:
            return MEDIUM  # a partial assessment never reaches high
        return level

    @property
    def severity(self) -> str:
        if self.kind in WORKLOAD_KINDS or not self.eligible:
            return "info"
        explained = self.explains is not None and self.explains in self.contribution.met
        strong = _at_least(self.condition.level, MEDIUM) and _at_least(
            self.contribution.level, MEDIUM
        )
        clear = not self.contested
        return "warning" if explained and strong and clear and self.incident else "info"

    @property
    def cause(self) -> str:
        """Workload changes are always that; an ineligible finding's cause
        is undetermined; a warning is a fault; the driver, which would say
        whether load drove an info finding, is not yet determined."""
        if self.kind in WORKLOAD_KINDS:
            return CAUSE_WORKLOAD_CHANGE
        if not self.eligible:
            return CAUSE_UNDETERMINED
        if self.kind in INSTRUMENTATION_KINDS:
            return CAUSE_INSTRUMENTATION
        if self.severity == "warning":
            return CAUSE_FAULT
        return CAUSE_UNDETERMINED

    @property
    def claim(self) -> str:
        if not self.eligible:
            return CLAIM_OBSERVATION
        if self.cause == CAUSE_FAULT and self.role == PRIMARY:
            return CLAIM_FAULT
        return CLAIM_CONDITION

    # ------------------------------------------------------------ identity
    def identity(self, run_id: str | None) -> str:
        """``diagnosis.<kind>.<hex12>`` from the run, subject, kind,
        location and window start: stable across reruns of one artifact."""
        window_start = (self.window or {}).get("start_ns")
        parts = [
            run_id or "",
            str(self.subject.get("key", "")),
            self.kind,
            self.component,
            str(self.location.get("engine_producer", "")),
            str(window_start if window_start is not None else ""),
        ]
        digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]
        return f"diagnosis.{self.kind}.{digest}"

    def rank_key(self, finding_id: str) -> tuple[Any, ...]:
        """Primary first, eligible first, then confidence, the
        contribution's lower bound, kind, location, window start and id."""
        lower = self.contribution_lower
        return (
            0 if self.role == PRIMARY else 1,
            0 if self.eligible else 1,
            -_LEVELS.index(self.confidence_level),
            -(lower if lower is not None else float("-inf")),
            self.kind,
            self.component,
            str(self.location.get("engine_producer", "")),
            (self.window or {}).get("start_ns") or 0,
            finding_id,
        )


def _at_least(level: str, floor: str) -> bool:
    return _LEVELS.index(level) >= _LEVELS.index(floor)


# A contribution no class of this version judges, as a workload change's:
# what share of the incident the demand explains is the driver's question.
NOT_DETERMINED = Criteria(unmet=("not_determined",), assessed=False)


def met(**criteria: bool) -> Criteria:
    """A claim's rubric from named criteria, each met or not."""
    return Criteria(
        met=tuple(name for name, held in criteria.items() if held),
        unmet=tuple(name for name, held in criteria.items() if not held),
    )


def support_block(lines: Sequence[Line]) -> dict[str, Any]:
    """Every line a finding used, by identity: (line, record_id, sha256)
    triples, or above the limit ranges and a digest of their hashes."""
    unique = {line.number: line for line in lines}
    ordered = [unique[number] for number in sorted(unique)]
    if len(ordered) <= SUPPORT_LIMIT:
        return {
            "support_identity": "triples",
            "lines": [[line.number, line.record_id, line.sha256] for line in ordered],
        }
    digest = hashlib.sha256("".join(line.sha256 for line in ordered).encode())
    return {
        "support_identity": "ranges_only",
        "ranges": _ranges([line.number for line in ordered]),
        "count": len(ordered),
        "digest": digest.hexdigest(),
    }


def _ranges(numbers: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for number in numbers:
        if ranges and ranges[-1][1] == number - 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ranges


def rank_findings(
    findings: Sequence[Finding], run_id: str | None
) -> list[tuple[str, Finding]]:
    """(id, finding) in rank order."""
    identified = [(finding.identity(run_id), finding) for finding in findings]
    return sorted(identified, key=lambda pair: pair[1].rank_key(pair[0]))


__all__ = [
    "ASSESSED",
    "CLAIM_CONDITION",
    "CLAIM_FAULT",
    "CLAIM_OBSERVATION",
    "CONTRIBUTING",
    "DRIVER_UNDETERMINED",
    "HIGH",
    "LOW",
    "MEDIUM",
    "NOT_DETERMINED",
    "NOT_OBSERVED",
    "NOT_RULED_OUT",
    "PARTIAL",
    "PRIMARY",
    "RULED_OUT",
    "SECONDARY",
    "UNSUPPORTED",
    "UNTESTABLE",
    "UPSTREAM",
    "Alternative",
    "Criteria",
    "Finding",
    "Observation",
    "met",
    "rank_findings",
    "support_block",
]
