"""``score_v1``: a diagnosis scored against injected ground truth.

Frozen before any evaluation data is drawn (#221 design C.2). For each
episode, the **candidate set** is every finding of the diagnosis, of any
kind, subject or role, that passes the temporal rule for the episode, minus
the *neutral secondaries*; it is ranked by #218's total ``rank``. An episode
is diagnosed correctly at top-k when a finding that *matches* the label is
among the first k candidates.

- **Temporal rule.** With ``S_e = [effect_onset − pre_grace, effect_end +
  grace(kind)]``, where ``pre_grace`` is the finding's ``resolution_ns`` plus
  ``uncertainty_ns`` and ``grace`` is frozen per finding kind, a finding
  qualifies when it starts no earlier than ``S_e`` and at least half of its
  window lies inside ``S_e``. A finding without a window never does.
- **Neutral secondary.** A ``secondary`` finding whose ``secondary_to``
  names an eligible candidate that matches one of the label's ``expects``,
  ``secondary`` or ``allows`` entries, through an edge in #218's table, with
  each end where the edge allows, and its window inside that finding's
  window ± grace. Any other secondary is scored as if it were primary.
- **Match.** The label's kind, ``role: primary``, the label's cause, at least
  its severity, and an eligible claim, at the same location: the component
  (L1), and the rank and engine where the label names them (L2), read from
  #218's ``location.rank`` and ``location.engine_producer``. A
  secondary of the right kind never matches: the diagnoser said the
  mechanism followed from something else.
- **Runs.** Findings and false positives are counted per run
  (``score_run``). Each finding goes to at most one episode: of those whose
  window it qualifies for, the one whose effect began latest at or before
  its start. A run with exactly one valid negative episode of the eight
  negative types is one false-positive unit, and its false claims are
  counted over its whole negative exposure, not only that episode's
  window.
- **False claims.** A scored fault claim (eligible, cause ``fault``, at
  ``warning``) among the candidates that is not neutral, is not of the
  label's kind at its component, and is not in ``allows``. A ``secondary``
  entry exempts a finding only through its valid edge (it is then neutral,
  so not a candidate); in any other role or place it counts, as A.4 says.
  In a negative episode it is a false positive; in a fault episode it is
  counted as spurious. A secondary of the label's kind is
  neither a match nor a false claim: it names the true mechanism, in the
  wrong role.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .bounds import (
    CONFIDENCE,
    clopper_pearson_lower,
    clopper_pearson_upper,
    poisson_rate_upper,
)
from .ground_truth import VALID, Expectation, Injection, Interval, Location, RunRecord
from .vocabulary import (
    CAUSE_FAULT,
    CLAIM_OBSERVATION,
    EDGE_TABLE_VERSION,
    EDGES,
    NOT_ASSESSED_COMPONENTS,
    PRIMARY,
    SECONDARY,
    Edge,
    severity_at_least,
)

SCORE_VERSION = "score_v1"

# The user's approved targets (#221 decision D8): per-stratum accuracy at
# least 0.78 and a false-positive rate of at most 0.05, both one-sided 95%.
ACCURACY_FLOOR = 0.78
FPR_CEILING = 0.05

TOP1 = "top1"
TOP3 = "top3"

# C.5's eight negative types: each negative run holds exactly one.
NEGATIVE_TYPES = frozenset({"T1", "T2", "T3", "T3b", "H0", "W1", "P", "N"})
# Cause classes whose episodes' windows are not negative time.
INJECTED_CLASSES = frozenset({"fault", "instrumentation"})
HOUR_NS = 3_600_000_000_000

# Why an episode was missed; each counts against accuracy.
MISS_INELIGIBLE = "ineligible"
MISS_SECONDARY_ONLY = "secondary_only"
MISS_MISMATCH = "mismatch"
MISS_COVERAGE_GAP = "coverage_gap"
MISS_NO_FINDING = "no_finding"
# A match exists, but below the gated top-k.
MISS_OUTRANKED = "outranked"


@dataclass(frozen=True)
class ScoreConfig:
    """What ``score_v1`` freezes before evaluation: grace per finding kind
    (from ``dev_v1``), the edge table, the support matrix and the targets."""

    grace_ns: Mapping[str, int] = field(default_factory=dict)
    default_grace_ns: int = 30_000_000_000
    edges: Mapping[str, Edge] = field(default_factory=lambda: dict(EDGES))
    edge_table_version: str = EDGE_TABLE_VERSION
    supported_types: frozenset[str] | None = None
    accuracy_floor: float = ACCURACY_FLOOR
    fpr_ceiling: float = FPR_CEILING
    confidence: float = CONFIDENCE
    gated_metric: str = TOP1
    gated_level: int = 2
    negative_types: frozenset[str] = NEGATIVE_TYPES

    def grace(self, kind: str) -> int:
        return self.grace_ns.get(kind, self.default_grace_ns)


# ------------------------------------------------------------------ findings


@dataclass(frozen=True)
class Window:
    start_ns: int
    end_ns: int
    resolution_ns: int = 0
    uncertainty_ns: int = 0

    @property
    def pre_grace_ns(self) -> int:
        return self.resolution_ns + self.uncertainty_ns


@dataclass(frozen=True)
class FindingView:
    """The fields of one ``payload.findings_detail`` entry the scorer reads."""

    id: str
    kind: str
    role: str
    rank: int
    severity: str
    cause: str
    claim: str
    component: str
    location_rank: int | None = None
    engine: str | None = None
    window: Window | None = None
    secondary_to: tuple[str, ...] = ()
    eligibility_failed: tuple[str, ...] = ()

    @classmethod
    def from_detail(cls, detail: Mapping[str, Any]) -> FindingView:
        location = detail.get("location") or {}
        window = detail.get("window")
        eligibility = detail.get("eligibility") or {}
        return cls(
            id=str(detail["id"]),
            kind=str(detail["kind"]),
            role=str(detail["role"]),
            rank=int(detail["rank"]),
            severity=str(detail["severity"]),
            cause=str(detail["cause"]),
            claim=str(detail["claim"]),
            component=str(location["component"]),
            # #218's location names the engine by its hook producer and
            # carries the TP rank where it knows it.
            location_rank=location.get("rank"),
            engine=location.get("engine_producer"),
            window=None if window is None else _window(window),
            secondary_to=tuple(detail.get("secondary_to") or ()),
            eligibility_failed=tuple(eligibility.get("failed") or ()),
        )

    @property
    def eligible(self) -> bool:
        return self.claim != CLAIM_OBSERVATION

    @property
    def scored_fault_claim(self) -> bool:
        """A fault claim, a secondary's included: a secondary that is not
        neutral is scored as if it were primary."""
        return (
            self.eligible
            and self.cause == CAUSE_FAULT
            and severity_at_least(self.severity, "warning")
        )


def _window(record: Mapping[str, Any]) -> Window:
    return Window(
        start_ns=int(record["start_ns"]),
        end_ns=int(record["end_ns"]),
        resolution_ns=int(record.get("resolution_ns") or 0),
        uncertainty_ns=int(record.get("uncertainty_ns") or 0),
    )


def findings_of(diagnosis: Mapping[str, Any]) -> list[FindingView]:
    """Every finding of a diagnosis: a ``stormlog.report`` or its payload."""
    payload = diagnosis.get("payload", diagnosis)
    details = payload.get("findings_detail") or {}
    return [FindingView.from_detail(detail) for detail in details.values()]


def coverage_of(diagnosis: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = diagnosis.get("payload", diagnosis)
    coverage: Mapping[str, Any] = payload.get("coverage") or {}
    return coverage


# ------------------------------------------------------------------ the rules


def in_scoring_window(
    finding: FindingView, injection: Injection, config: ScoreConfig
) -> bool:
    """The temporal rule: start within ``S_e``, and at least half inside."""
    window = finding.window
    onset, end = injection.times.effect_onset_ns, injection.times.effect_end_ns
    if window is None or onset is None or end is None:
        return False
    first = onset - window.pre_grace_ns
    last = end + config.grace(finding.kind)
    if window.start_ns < first or window.start_ns > last:
        return False
    length = window.end_ns - window.start_ns
    if length <= 0:
        return True
    inside = min(window.end_ns, last) - window.start_ns
    return 2 * inside >= length


def location_matches(finding: FindingView, entry: Location, level: int) -> bool:
    """L1: the component. L2: also the rank and engine the label names."""
    if finding.kind != entry.kind or finding.component != entry.component:
        return False
    if level < 2:
        return True
    if entry.rank is not None and finding.location_rank != entry.rank:
        return False
    return entry.engine is None or finding.engine == entry.engine


def matches(finding: FindingView, expectation: Expectation, level: int) -> bool:
    """The label's kind as a primary, eligible claim of its cause, at least
    its severity, at its location."""
    return (
        finding.role == PRIMARY
        and finding.eligible
        and finding.cause == expectation.cause
        and severity_at_least(finding.severity, expectation.min_severity)
        and location_matches(finding, expectation, level)
    )


def _labelled(finding: FindingView, injection: Injection) -> bool:
    entries: tuple[Location, ...] = (
        *injection.expects,
        *injection.secondary,
        *injection.allows,
    )
    return any(location_matches(finding, entry, 2) for entry in entries)


def _allowed(finding: FindingView, injection: Injection) -> bool:
    return any(location_matches(finding, entry, 2) for entry in injection.allows)


def is_neutral(
    finding: FindingView,
    qualifying: Mapping[str, FindingView],
    injection: Injection,
    config: ScoreConfig,
) -> bool:
    """A secondary whose upstream is an eligible, labelled candidate joined
    to it by one of #218's edges."""
    if finding.role != SECONDARY:
        return False
    for upstream_id in finding.secondary_to:
        upstream = qualifying.get(upstream_id)
        if upstream is None or not upstream.eligible:
            continue
        if _edge_holds(upstream, finding, config) and _labelled(upstream, injection):
            return True
    return False


def _edge_holds(
    upstream: FindingView, downstream: FindingView, config: ScoreConfig
) -> bool:
    edge = config.edges.get(f"{upstream.kind}->{downstream.kind}")
    if edge is None:
        return False
    if upstream.component not in edge.upstream_components:
        return False
    if downstream.component not in edge.downstream_components:
        return False
    return _nested(downstream.window, upstream.window, config.grace(downstream.kind))


def _nested(inner: Window | None, outer: Window | None, grace_ns: int) -> bool:
    if inner is None or outer is None:
        return False
    return (
        inner.start_ns >= outer.start_ns - grace_ns
        and inner.end_ns <= outer.end_ns + grace_ns
    )


# ------------------------------------------------------------------ an episode


@dataclass(frozen=True)
class EpisodeScore:
    """One episode scored against its run's diagnosis."""

    episode_id: str
    episode_type: str
    cause_class: str
    status: str
    impact: str | None
    candidates: tuple[str, ...]
    neutral: tuple[str, ...]
    match_rank: Mapping[int, int | None]
    matched: tuple[str, ...]
    false_claims: tuple[str, ...]
    localized: bool
    miss: str | None
    secondaries: int
    secondary_errors: int

    def correct(self, metric: str, level: int) -> bool:
        rank = self.match_rank.get(level)
        limit = 1 if metric == TOP1 else 3
        return rank is not None and rank <= limit

    @property
    def duplicates(self) -> int:
        return max(0, len(self.matched) - 1)

    @property
    def valid(self) -> bool:
        return self.status == VALID

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_type": self.episode_type,
            "cause_class": self.cause_class,
            "status": self.status,
            "impact": self.impact,
            "candidates": list(self.candidates),
            "neutral": list(self.neutral),
            "match_rank": {
                f"L{level}": rank for level, rank in self.match_rank.items()
            },
            "matched": list(self.matched),
            "duplicates": self.duplicates,
            "false_claims": list(self.false_claims),
            "localized": self.localized,
            "miss": self.miss,
            "secondaries": self.secondaries,
            "secondary_errors": self.secondary_errors,
        }


def score_episode(
    injection: Injection,
    diagnosis: Mapping[str, Any],
    config: ScoreConfig,
    *,
    findings: Sequence[FindingView] | None = None,
) -> EpisodeScore:
    """Score one episode against the diagnosis of its run; ``findings``
    narrows it to the findings ``score_run`` assigned to the episode."""
    if findings is None:
        findings = findings_of(diagnosis)
    qualifying, ranked, neutral = _candidates(injection, findings, config)
    expectation = injection.expects[0] if injection.expects else None
    match_rank = {level: _first_match(ranked, expectation, level) for level in (1, 2)}
    impact = injection.validity.impact
    secondaries = [f for f in qualifying if f.role == SECONDARY]
    return EpisodeScore(
        episode_id=injection.episode_id,
        episode_type=injection.episode_type,
        cause_class=injection.cause_class,
        status=injection.status,
        impact=None if impact is None else impact.status,
        candidates=tuple(finding.id for finding in ranked),
        neutral=neutral,
        match_rank=match_rank,
        matched=_matched(ranked, expectation),
        false_claims=_false_claims(ranked, injection),
        localized=_localized(ranked, expectation),
        miss=_miss(ranked, expectation, match_rank[2], coverage_of(diagnosis)),
        secondaries=len(secondaries),
        secondary_errors=len(secondaries) - len(neutral),
    )


def _candidates(
    injection: Injection, findings: Sequence[FindingView], config: ScoreConfig
) -> tuple[list[FindingView], list[FindingView], tuple[str, ...]]:
    """The findings that pass the temporal rule, the ranked candidate set
    (those less the neutral secondaries), and the neutral ones' IDs."""
    qualifying = [
        finding for finding in findings if in_scoring_window(finding, injection, config)
    ]
    by_id = {finding.id: finding for finding in qualifying}
    neutral = {
        finding.id
        for finding in qualifying
        if is_neutral(finding, by_id, injection, config)
    }
    ranked = sorted(
        (finding for finding in qualifying if finding.id not in neutral),
        key=lambda finding: (finding.rank, finding.id),
    )
    return qualifying, ranked, tuple(sorted(neutral))


def _first_match(
    ranked: Sequence[FindingView], expectation: Expectation | None, level: int
) -> int | None:
    if expectation is None:
        return None
    for position, finding in enumerate(ranked, 1):
        if matches(finding, expectation, level):
            return position
    return None


def _matched(
    ranked: Sequence[FindingView], expectation: Expectation | None
) -> tuple[str, ...]:
    if expectation is None:
        return ()
    return tuple(f.id for f in ranked if matches(f, expectation, 2))


def _false_claims(
    ranked: Sequence[FindingView], injection: Injection
) -> tuple[str, ...]:
    expectation = injection.expects[0] if injection.expects else None
    return tuple(
        finding.id
        for finding in ranked
        if finding.scored_fault_claim
        and not _named(finding, expectation)
        and not _allowed(finding, injection)
    )


def _named(finding: FindingView, expectation: Expectation | None) -> bool:
    """The label's own kind at its component, in any role."""
    return expectation is not None and location_matches(finding, expectation, 1)


def _localized(ranked: Sequence[FindingView], expectation: Expectation | None) -> bool:
    """Condition localization: an eligible finding of the label's kind at
    its location, whatever its role, cause or severity."""
    if expectation is None:
        return False
    return any(f.eligible and location_matches(f, expectation, 2) for f in ranked)


def _miss(
    ranked: Sequence[FindingView],
    expectation: Expectation | None,
    match_rank: int | None,
    coverage: Mapping[str, Any],
) -> str | None:
    if expectation is None or match_rank is not None:
        return None
    same = [f for f in ranked if f.kind == expectation.kind]
    if same:
        return _wrong_finding(same)
    entry = coverage.get(expectation.kind) or {}
    covered = assessed_at(entry, expectation.component)
    return MISS_NO_FINDING if covered else MISS_COVERAGE_GAP


def assessed_at(entry: Mapping[str, Any], component: str) -> bool:
    """Whether #218 assessed a kind at ``component``: fully, or partly only
    for reasons that exclude other components, with every subject assessed."""
    status = entry.get("status")
    if status == "assessed":
        return True
    if status != "partial" or not _subjects_assessed(entry):
        return False
    return _excluded_elsewhere(entry.get("reasons") or (), component)


def _subjects_assessed(entry: Mapping[str, Any]) -> bool:
    subjects = (entry.get("by_subject") or {}).values()
    return all(subject.get("status") == "assessed" for subject in subjects)


def _excluded_elsewhere(reasons: Sequence[str], component: str) -> bool:
    """Some reason excludes components, and none excludes ``component``."""
    known = [
        NOT_ASSESSED_COMPONENTS[r] for r in reasons if r in NOT_ASSESSED_COMPONENTS
    ]
    return bool(known) and not any(component in excluded for excluded in known)


def _wrong_finding(same: Sequence[FindingView]) -> str:
    """Why findings of the label's kind did not match: an eligible primary
    with the wrong cause, severity or location; else only secondaries; else
    #218 made them ineligible."""
    if any(f.role == PRIMARY and f.eligible for f in same):
        return MISS_MISMATCH
    if all(f.role == SECONDARY for f in same):
        return MISS_SECONDARY_ONLY
    return MISS_INELIGIBLE


# ------------------------------------------------------------------ a run


@dataclass(frozen=True)
class RunScore:
    """One run's episodes and, when the run is a false-positive unit, its
    negative episode, negative exposure and false claims over it."""

    run_id: str
    episodes: tuple[EpisodeScore, ...]
    negative_episode: str | None = None
    exposure: tuple[Interval, ...] = ()
    false_claims: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def exposure_ns(self) -> int:
        return sum(interval.length_ns for interval in self.exposure)

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episodes": [episode.to_record() for episode in self.episodes],
            "negative_episode": self.negative_episode,
            "exposure": [interval.to_record() for interval in self.exposure],
            "false_claims": list(self.false_claims),
            "problems": list(self.problems),
        }


def score_run(
    run: RunRecord,
    injections: Sequence[Injection],
    diagnosis: Mapping[str, Any],
    config: ScoreConfig,
) -> RunScore:
    """Score a run's episodes against its diagnosis, and, when the run holds
    exactly one valid negative episode, count its false claims over the
    run's whole negative exposure (C.2, C.5).

    Raises:
        ValueError: for an episode of another run, or one given twice.
    """
    _check_run(run, injections)
    findings = findings_of(diagnosis)
    assigned = assign_findings(injections, findings, config)
    episodes = tuple(
        score_episode(
            injection, diagnosis, config, findings=assigned[injection.episode_id]
        )
        for injection in injections
    )
    unit, problems = _negative_unit(run, injections, config)
    if unit is None:
        return RunScore(run.run_id, episodes, problems=problems)
    exposure = negative_exposure(run, injections, config)
    elsewhere = {
        finding.id
        for injection in injections
        if injection.cause_class in INJECTED_CLASSES
        for finding in assigned[injection.episode_id]
    }
    claims = _negative_claims(findings, elsewhere, exposure, unit, config)
    return RunScore(run.run_id, episodes, unit.episode_id, exposure, claims, problems)


def _check_run(run: RunRecord, injections: Sequence[Injection]) -> None:
    ids = [injection.episode_id for injection in injections]
    if len(set(ids)) != len(ids):
        raise ValueError(f"run {run.run_id}: an episode is given twice")
    strays = sorted(i.episode_id for i in injections if i.run_id != run.run_id)
    if strays:
        raise ValueError(f"run {run.run_id}: episodes of another run: {strays}")


def assign_findings(
    injections: Sequence[Injection],
    findings: Sequence[FindingView],
    config: ScoreConfig,
) -> dict[str, list[FindingView]]:
    """Each finding goes to at most one episode: of those whose window it
    qualifies for, the one whose effect began latest at or before the
    finding's start, else the earliest. So one finding is never credited
    to two episodes."""
    assigned: dict[str, list[FindingView]] = {i.episode_id: [] for i in injections}
    for finding in findings:
        qualifying = [i for i in injections if in_scoring_window(finding, i, config)]
        if qualifying:
            assigned[_closest(finding, qualifying).episode_id].append(finding)
    return assigned


def _closest(finding: FindingView, injections: Sequence[Injection]) -> Injection:
    start = finding.window.start_ns if finding.window else 0
    onsets = [(_onset(injection), injection) for injection in injections]
    before = [pair for pair in onsets if pair[0] <= start]
    if before:
        return max(before, key=lambda pair: pair[0])[1]
    return min(onsets, key=lambda pair: pair[0])[1]


def _onset(injection: Injection) -> int:
    onset = injection.times.effect_onset_ns
    return onset if onset is not None else 0


def _negative_unit(
    run: RunRecord, injections: Sequence[Injection], config: ScoreConfig
) -> tuple[Injection | None, tuple[str, ...]]:
    """The run's one valid negative episode, if the run is an FPR unit."""
    if run.protocol_failure:
        return None, (f"run {run.run_id}: protocol failure {run.protocol_failure}",)
    negatives = [i for i in injections if i.episode_type in config.negative_types]
    if len(negatives) > 1:
        return None, (f"run {run.run_id}: more than one negative episode",)
    if negatives and negatives[0].status == VALID:
        return negatives[0], ()
    return None, ()


def negative_exposure(
    run: RunRecord, injections: Sequence[Injection], config: ScoreConfig
) -> tuple[Interval, ...]:
    """The run's negative time (C.5): its measured window less the priming
    and the span of every attempted fault or instrumentation episode, from
    its action or effect onset to its effect end plus grace (to the window's
    end when its effect never ended)."""
    removed = [run.priming] if run.priming is not None else []
    for injection in injections:
        if injection.cause_class in INJECTED_CLASSES:
            span = _episode_span(injection, run, config)
            if span is not None:
                removed.append(span)
    return _subtract(run.measured, removed)


def _episode_span(
    injection: Injection, run: RunRecord, config: ScoreConfig
) -> Interval | None:
    times = injection.times
    starts = [
        t for t in (times.action_onset_ns, times.effect_onset_ns) if t is not None
    ]
    if not starts:
        return None
    if times.effect_end_ns is None:
        return Interval(min(starts), run.measured.end_ns)
    kind = injection.expects[0].kind if injection.expects else ""
    return Interval(min(starts), times.effect_end_ns + config.grace(kind))


def _subtract(whole: Interval, removed: Sequence[Interval]) -> tuple[Interval, ...]:
    pieces = [whole]
    for cut in removed:
        pieces = [rest for piece in pieces for rest in _cut(piece, cut)]
    return tuple(sorted(pieces, key=lambda piece: piece.start_ns))


def _cut(piece: Interval, cut: Interval) -> list[Interval]:
    if cut.end_ns <= piece.start_ns or cut.start_ns >= piece.end_ns:
        return [piece]
    rest = []
    if cut.start_ns > piece.start_ns:
        rest.append(Interval(piece.start_ns, cut.start_ns))
    if cut.end_ns < piece.end_ns:
        rest.append(Interval(cut.end_ns, piece.end_ns))
    return rest


def _negative_claims(
    findings: Sequence[FindingView],
    elsewhere: set[str],
    exposure: Sequence[Interval],
    unit: Injection,
    config: ScoreConfig,
) -> tuple[str, ...]:
    """Scored fault claims placed in the negative exposure, less those an
    injected episode took and those the negative episode makes neutral or
    allows."""
    qualifying = {f.id: f for f in findings if in_scoring_window(f, unit, config)}
    return tuple(
        finding.id
        for finding in findings
        if finding.scored_fault_claim
        and finding.id not in elsewhere
        and _placed_in(finding.window, exposure)
        and not _allowed(finding, unit)
        and not (
            finding.id in qualifying and is_neutral(finding, qualifying, unit, config)
        )
    )


def _placed_in(window: Window | None, exposure: Sequence[Interval]) -> bool:
    """The temporal rule over the exposure: the window starts in it, and at
    least half of the window lies in it."""
    if window is None or not exposure:
        return False
    if not any(i.start_ns <= window.start_ns <= i.end_ns for i in exposure):
        return False
    length = window.end_ns - window.start_ns
    if length <= 0:
        return True
    inside = sum(
        max(0, min(window.end_ns, i.end_ns) - max(window.start_ns, i.start_ns))
        for i in exposure
    )
    return 2 * inside >= length


# ------------------------------------------------------------------ a campaign


@dataclass(frozen=True)
class Stratum:
    """One declared fault episode type's accuracy and its gate, with its
    attempted episodes that didn't count, by status."""

    episode_type: str
    episodes: int
    correct: int
    lower_bound: float | None
    passes: bool
    excluded: Mapping[str, int] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_type": self.episode_type,
            "episodes": self.episodes,
            "correct": self.correct,
            "lower_bound": self.lower_bound,
            "passes": self.passes,
            "excluded": dict(self.excluded),
        }


@dataclass(frozen=True)
class Summary:
    """A campaign's claims: per-stratum accuracy, the false-positive rate
    over negative runs (one negative episode per run), and the descriptive
    counts."""

    strata: tuple[Stratum, ...]
    accuracy_passes: bool
    negative_runs: int
    false_positive_runs: int
    fpr_upper_bound: float | None
    fpr_passes: bool
    false_claims_per_hour_upper: float | None
    spurious: int
    duplicates: int
    misses: Mapping[str, int]
    attributed: tuple[int, int]
    localized: tuple[int, int]
    negative_hours: float = 0.0
    problems: tuple[str, ...] = ()

    def to_record(self, config: ScoreConfig) -> dict[str, Any]:
        return {
            "score": SCORE_VERSION,
            "edge_table": config.edge_table_version,
            "gated_metric": config.gated_metric,
            "gated_level": f"L{config.gated_level}",
            "accuracy_floor": config.accuracy_floor,
            "fpr_ceiling": config.fpr_ceiling,
            "strata": [stratum.to_record() for stratum in self.strata],
            "accuracy_passes": self.accuracy_passes,
            "negative_runs": self.negative_runs,
            "false_positive_runs": self.false_positive_runs,
            "fpr_upper_bound": self.fpr_upper_bound,
            "fpr_passes": self.fpr_passes,
            "negative_hours": self.negative_hours,
            "false_claims_per_hour_upper": self.false_claims_per_hour_upper,
            "spurious": self.spurious,
            "duplicates": self.duplicates,
            "misses": dict(self.misses),
            "incident_attribution": list(self.attributed),
            "condition_localization": list(self.localized),
            "problems": list(self.problems),
        }


def summarize(runs: Sequence[RunScore], config: ScoreConfig) -> Summary:
    """Gate per-stratum accuracy over valid fault episodes, a stratum for
    every type the support matrix declares; and the false-positive rate
    over the negative runs, with the per-hour rate over their exposure.

    Raises:
        ValueError: without ``config.supported_types`` (a declared stratum
            with no valid episode must fail, not vanish), or for an episode
            scored twice.
    """
    scores = _episodes(runs)
    faults = [s for s in scores if _counts_for_accuracy(s, config)]
    units = [run for run in runs if run.negative_episode is not None]
    strata = _strata(scores, config)
    flagged, upper = _fpr(units, config)
    hours = sum(run.exposure_ns for run in units) / HOUR_NS
    return Summary(
        strata=strata,
        accuracy_passes=all(stratum.passes for stratum in strata),
        negative_runs=len(units),
        false_positive_runs=flagged,
        fpr_upper_bound=upper,
        fpr_passes=_within(upper, config.fpr_ceiling),
        false_claims_per_hour_upper=_hourly(units, hours, config),
        spurious=_spurious(faults),
        duplicates=sum(score.duplicates for score in faults),
        misses=_miss_counts(faults, config),
        attributed=_attributed(faults, config),
        localized=_localized_count(faults),
        negative_hours=hours,
        problems=tuple(problem for run in runs for problem in run.problems),
    )


def _episodes(runs: Sequence[RunScore]) -> list[EpisodeScore]:
    scores = [episode for run in runs for episode in run.episodes]
    if len({score.episode_id for score in scores}) != len(scores):
        raise ValueError("an episode is scored twice")
    return scores


def _spurious(faults: Sequence[EpisodeScore]) -> int:
    return sum(len(score.false_claims) for score in faults)


def _within(upper: float | None, ceiling: float) -> bool:
    return upper is not None and upper <= ceiling


def _localized_count(faults: Sequence[EpisodeScore]) -> tuple[int, int]:
    """Condition localization over the engaged fault episodes."""
    return sum(1 for score in faults if score.localized), len(faults)


def _strata(scores: Sequence[EpisodeScore], config: ScoreConfig) -> tuple[Stratum, ...]:
    if not config.supported_types:
        raise ValueError("a gated summary needs the support matrix: supported_types")
    return tuple(
        _stratum(episode_type, scores, config)
        for episode_type in sorted(config.supported_types)
    )


def _counts_for_accuracy(score: EpisodeScore, config: ScoreConfig) -> bool:
    """A valid fault episode of a type the support matrix declares."""
    supported = config.supported_types
    return (
        score.valid
        and score.cause_class == "fault"
        and (supported is None or score.episode_type in supported)
    )


def _fpr(units: Sequence[RunScore], config: ScoreConfig) -> tuple[int, float | None]:
    """Negative runs with any false claim over their exposure, and the
    rate's one-sided upper bound."""
    flagged = sum(1 for run in units if run.false_claims)
    if not units:
        return flagged, None
    return flagged, clopper_pearson_upper(flagged, len(units), config.confidence)


def _stratum(
    episode_type: str, scores: Sequence[EpisodeScore], config: ScoreConfig
) -> Stratum:
    """A stratum with no valid episode has no bound, and fails."""
    attempted = _attempts(scores, episode_type)
    members = [s for s in attempted if s.valid]
    correct = sum(_gated_correct(s, config) for s in members)
    lower = _accuracy_bound(correct, len(members), config)
    return Stratum(
        episode_type=episode_type,
        episodes=len(members),
        correct=correct,
        lower_bound=lower,
        passes=lower is not None and lower >= config.accuracy_floor,
        excluded=_excluded(attempted),
    )


def _attempts(scores: Sequence[EpisodeScore], episode_type: str) -> list[EpisodeScore]:
    return [
        s for s in scores if s.episode_type == episode_type and s.cause_class == "fault"
    ]


def _excluded(attempted: Sequence[EpisodeScore]) -> dict[str, int]:
    counts = Counter(s.status for s in attempted if not s.valid)
    return dict(sorted(counts.items()))


def _gated_correct(score: EpisodeScore, config: ScoreConfig) -> bool:
    return score.correct(config.gated_metric, config.gated_level)


def _accuracy_bound(correct: int, episodes: int, config: ScoreConfig) -> float | None:
    if not episodes:
        return None
    return clopper_pearson_lower(correct, episodes, config.confidence)


def _hourly(
    units: Sequence[RunScore], hours: float, config: ScoreConfig
) -> float | None:
    """False claims per negative hour: counted over the same exposure the
    hours measure."""
    if not hours:
        return None
    claims = sum(len(run.false_claims) for run in units)
    return poisson_rate_upper(claims, hours, config.confidence)


def _miss_counts(faults: Sequence[EpisodeScore], config: ScoreConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in faults:
        if not score.correct(config.gated_metric, config.gated_level):
            label = score.miss or MISS_OUTRANKED
            counts[label] = counts.get(label, 0) + 1
    return counts


def _attributed(faults: Sequence[EpisodeScore], config: ScoreConfig) -> tuple[int, int]:
    """Incident attribution: a matching fault claim over the episodes with
    victim impact."""
    hurt = [s for s in faults if s.impact == "impact"]
    hits = sum(1 for s in hurt if s.correct(config.gated_metric, config.gated_level))
    return hits, len(hurt)


__all__ = [
    "ACCURACY_FLOOR",
    "FPR_CEILING",
    "MISS_COVERAGE_GAP",
    "MISS_INELIGIBLE",
    "MISS_MISMATCH",
    "MISS_NO_FINDING",
    "MISS_OUTRANKED",
    "MISS_SECONDARY_ONLY",
    "NEGATIVE_TYPES",
    "SCORE_VERSION",
    "TOP1",
    "TOP3",
    "EpisodeScore",
    "FindingView",
    "RunScore",
    "ScoreConfig",
    "Stratum",
    "Summary",
    "Window",
    "assessed_at",
    "assign_findings",
    "coverage_of",
    "findings_of",
    "in_scoring_window",
    "is_neutral",
    "location_matches",
    "matches",
    "negative_exposure",
    "score_episode",
    "score_run",
    "summarize",
]
