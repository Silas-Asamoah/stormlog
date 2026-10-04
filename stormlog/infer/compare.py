"""Comparing a baseline arm of runs with a candidate arm, case by case.

``compare_runs`` takes run summaries (``run_summary``) and:

1. sets aside each run that cannot stand for a case, with its protocol
   failure, and lists it; outcome failures are never set aside;
2. checks the runs are comparable (``compatibility``): within each arm,
   and across arms in the comparison's mode. ``incompatible`` is an input
   error; ``unverified`` leaves every gate ``not_evaluable``;
3. checks the observers against the mode's contract: an ``overhead``
   baseline has none, an ``incremental`` candidate's added observers are
   declared, and the observers that matter are active and healthy;
4. pairs runs by block when every run is labelled, and otherwise treats
   the arms as independent;
5. compares each metric (``comparison_stats``) and applies its gate;
6. for ``--min-attainment``, judges the share of candidate runs that meet
   the target, a claim about runs, not requests.

The result is ``stormlog.infer.comparison`` v1.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from scipy import stats

from .compare_metrics import MetricSpec, default_metrics, evidence_coverage
from .comparison_stats import (
    FAIL,
    INDEPENDENT,
    NOT_EVALUABLE,
    PAIRED,
    PASS,
    SIGNIFICANT,
    GateOutcome,
    GateRule,
    MetricComparison,
    clopper_pearson,
    compare_values,
    holm,
    run_pass_gate,
)
from .compatibility import (
    CONFIG,
    INCOMPATIBLE,
    INCREMENTAL,
    MODES,
    OVERHEAD,
    UNVERIFIED,
    Compatibility,
    compatible,
)
from .errors import InferInputError, InferUsageError
from .run_summary import RunSummary

COMPARISON_FORMAT = "stormlog.infer.comparison"
COMPARISON_VERSION = 1
BASELINE = "baseline"
CANDIDATE = "candidate"
ALL_BUDGETS = "all_budgets"
ANY_REGRESSION = "any_regression"
FAMILIES = (ALL_BUDGETS, ANY_REGRESSION)
EXCLUDE = "exclude"
FAIL_INCOMPLETE = "fail"


@dataclass(frozen=True)
class ComparisonSpec:
    """How to compare: design, mode, gates and what may differ."""

    confidence: float = 0.95
    design: str = "auto"
    mode: str = CONFIG
    allow: tuple[str, ...] = ()
    on_incomplete: str = EXCLUDE
    gates: tuple[tuple[str, GateRule], ...] = ()
    allow_not_evaluable: bool = False
    min_attainment: float | None = None
    min_run_pass: float = 0.5
    attainment_model: str = "runs"
    family: str = ALL_BUDGETS
    added_observers: tuple[str, ...] = ()
    evidence_floor: float = 1.0
    cases: tuple[str, ...] | None = None
    bootstrap_min_n: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        checks = (
            (self.design in ("auto", PAIRED, INDEPENDENT), "design"),
            (self.mode in MODES, "mode"),
            (self.on_incomplete in (EXCLUDE, FAIL_INCOMPLETE), "on_incomplete"),
            (self.attainment_model in ("runs", "bernoulli"), "attainment_model"),
            (self.family in FAMILIES, "family"),
            (0 < self.confidence < 1, "confidence"),
            # Shares that cannot be met or cannot fail mean nothing.
            (_share(self.min_attainment, low_open=True), "min_attainment"),
            (_share(self.min_run_pass, low_open=True), "min_run_pass"),
            (_share(self.evidence_floor, low_open=False), "evidence_floor"),
        )
        for ok, name in checks:
            if not ok:
                raise InferUsageError(
                    f"comparison {name} {getattr(self, name)!r} is not valid"
                )
        if self.family == ANY_REGRESSION and any(
            rule.rule != SIGNIFICANT for _, rule in self.gates
        ):
            raise InferUsageError(
                "--family any_regression gates with the significant rule"
            )

    def gate_for(self, metric: str) -> GateRule | None:
        """The gate on a metric: the last pattern that matches its name."""
        found = None
        for pattern, rule in self.gates:
            if fnmatch.fnmatchcase(metric, pattern):
                found = rule
        return found

    def to_record(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "design": self.design,
            "mode": self.mode,
            "allow": list(self.allow),
            "on_incomplete": self.on_incomplete,
            "gates": {pattern: rule.to_record() for pattern, rule in self.gates},
            "allow_not_evaluable": self.allow_not_evaluable,
            "min_attainment": self.min_attainment,
            "min_run_pass": self.min_run_pass,
            "attainment_model": self.attainment_model,
            "family": self.family,
            "added_observers": list(self.added_observers),
            "evidence_floor": self.evidence_floor,
            "cases": None if self.cases is None else list(self.cases),
            "seed": self.seed,
        }


def _share(value: float | None, *, low_open: bool) -> bool:
    """None, or a share in (0, 1] (``low_open``) or [0, 1]."""
    if value is None:
        return True
    if not math.isfinite(value) or value > 1:
        return False
    return value > 0 if low_open else value >= 0


@dataclass(frozen=True)
class Comparison:
    """Everything a comparison found, and the exit code it implies."""

    spec: ComparisonSpec
    design: str
    runs: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    comparability: Compatibility
    observer_issues: list[str]
    cases: dict[str, dict[str, Any]]
    family: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def gates(self) -> list[tuple[str, str, GateOutcome]]:
        """Every gate outcome, a requested gate that matched no metric included."""
        found = []
        for case_id, case in self.cases.items():
            for name, metric in case["metrics"].items():
                if metric.gate is not None:
                    found.append((case_id, name, metric.gate))
            for pattern, outcome in case.get("absent_gates", {}).items():
                found.append((case_id, pattern, outcome))
        return found

    @property
    def failed(self) -> list[tuple[str, str]]:
        failed = [
            (case, name) for case, name, gate in self.gates if gate.status == FAIL
        ]
        failed += [
            (case_id, "min_attainment")
            for case_id, case in self.cases.items()
            if (case.get("attainment_gate") or {}).get("status") == FAIL
        ]
        return failed

    @property
    def not_evaluable(self) -> list[tuple[str, str, str | None]]:
        found = [
            (case, name, gate.reason)
            for case, name, gate in self.gates
            if gate.status == NOT_EVALUABLE
        ]
        found += [
            (case_id, "min_attainment", gate.get("reason"))
            for case_id, case in self.cases.items()
            if (gate := case.get("attainment_gate") or {}).get("status")
            == NOT_EVALUABLE
        ]
        return found

    @property
    def exit_code(self) -> int:
        """4 when a gate failed, or could not be evaluated unless that is allowed."""
        if self.failed:
            return 4
        if self.not_evaluable and not self.spec.allow_not_evaluable:
            return 4
        return 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": COMPARISON_FORMAT,
            "version": COMPARISON_VERSION,
            "spec": self.spec.to_record(),
            "design": self.design,
            "runs": self.runs,
            "excluded": self.excluded,
            "compatibility": self.comparability.to_record(),
            "observer_issues": self.observer_issues,
            "cases": {
                case_id: {
                    "metrics": {
                        name: metric.to_record()
                        for name, metric in case["metrics"].items()
                    },
                    "absent_gates": {
                        pattern: outcome.to_record()
                        for pattern, outcome in case.get("absent_gates", {}).items()
                    },
                    "attainment_gate": case.get("attainment_gate"),
                    "attainment_mean": case.get("attainment_mean"),
                    "attrition": case.get("attrition", []),
                }
                for case_id, case in self.cases.items()
            },
            "family": self.family,
            "verdict": {
                "exit_code": self.exit_code,
                "failed": [list(item) for item in self.failed],
                "not_evaluable": [list(item) for item in self.not_evaluable],
                "allow_not_evaluable": self.spec.allow_not_evaluable,
            },
            "diagnostics": self.diagnostics,
            "limits": LIMITS,
        }


LIMITS = [
    "Paired t is nominal for roughly symmetric block log ratios; under strongly "
    "skewed noise it covers 86-88% instead of 95%.",
    "Guards flag; they do not certify coverage.",
    "The independent design's intervals are conservative by construction.",
    "Latency quantile screens assume independent requests, which queueing violates.",
]


# ------------------------------------------------------------------ entry


def compare_runs(
    baseline: Sequence[RunSummary],
    candidate: Sequence[RunSummary],
    spec: ComparisonSpec,
) -> Comparison:
    """Compare two arms of runs; InferInputError when they cannot be compared."""
    if not baseline or not candidate:
        raise InferInputError("each arm needs at least one run")
    arms = {BASELINE: list(baseline), CANDIDATE: list(candidate)}
    _check_cases_exist(arms, spec)
    excluded = _excluded(arms, spec)
    usable = {arm: _usable(runs) for arm, runs in arms.items()}
    if not usable[BASELINE] or not usable[CANDIDATE]:
        raise InferInputError("an arm has no usable run: every run was excluded")
    compatibility, unverified_pairs = _compatibility(usable, spec)
    observer_issues = _observer_contract(usable, spec)
    design = _design(arms, spec)
    cases = {
        case_id: _case(
            case_id, arms, excluded, design, spec, compatibility, observer_issues
        )
        for case_id in _case_ids(arms, spec)
    }
    comparison = Comparison(
        spec=spec,
        design=design,
        runs=[_run_record(arm, run) for arm, runs in arms.items() for run in runs],
        excluded=excluded,
        comparability=compatibility,
        observer_issues=observer_issues,
        cases=cases,
        diagnostics={
            **_diagnostics(arms, compatibility),
            "unverified_pairs": unverified_pairs,
        },
    )
    return _with_family(comparison)


# ---------------------------------------------------------- set-asides


def _excluded(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> list[dict[str, Any]]:
    """Every (run, case) a protocol failure sets aside, with its reasons."""
    excluded = []
    for arm, runs in arms.items():
        for run in runs:
            for case_id in _case_ids(arms, spec):
                reasons = list(run.failures_for(case_id))
                if case_id not in run.comparable_cases:
                    reasons.append("case_missing")
                if reasons:
                    excluded.append(
                        {
                            "arm": arm,
                            "run": run.name,
                            "case": case_id,
                            "reasons": reasons,
                        }
                    )
    return excluded


def _usable(runs: list[RunSummary]) -> list[RunSummary]:
    """Runs with no run-level protocol failure, for the comparability checks."""
    return [run for run in runs if not run.protocol_failures]


def _set_aside(excluded: list[dict[str, Any]], run: RunSummary, case_id: str) -> bool:
    return any(item["run"] == run.name and item["case"] == case_id for item in excluded)


# --------------------------------------------------------- comparability


def _compatibility(
    usable: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> tuple[Compatibility, list[list[Any]]]:
    """Every run within its arm (config mode), and every run across arms.

    Comparability is not transitive once a value is unknown, so each run is
    checked, not only each arm's first. The reported result is the first
    runs' across the arms, made unverified, with every unverified field,
    when any pair was; the unverified pairs are returned with their fields.
    """
    within: list[tuple[str, str, Compatibility]] = []
    for arm, runs in usable.items():
        for run in runs[1:]:
            result = compatible(runs[0].fields, run.fields)
            if result.status == INCOMPATIBLE:
                raise InferInputError(_incompatible(f"{arm} runs", result))
            within.append((runs[0].name, run.name, result))
    across: list[tuple[str, str, Compatibility]] = []
    for first, second in _cross_pairs(usable):
        result = compatible(
            first.fields, second.fields, allowed=spec.allow, mode=spec.mode
        )
        if result.status == INCOMPATIBLE:
            raise InferInputError(_incompatible("the arms", result))
        across.append((first.name, second.name, result))
    results = [*within, *across]
    pairs = [
        [a, b, sorted(item.name for item in result.unverified)]
        for a, b, result in results
        if result.status == UNVERIFIED
    ]
    return _worst(across[0][2], [result for _a, _b, result in results]), pairs


def _cross_pairs(
    usable: Mapping[str, list[RunSummary]]
) -> list[tuple[RunSummary, RunSummary]]:
    """The first runs, then each other run against the other arm's first."""
    baseline, candidate = usable[BASELINE], usable[CANDIDATE]
    return [
        (baseline[0], candidate[0]),
        *((baseline[0], run) for run in candidate[1:]),
        *((run, candidate[0]) for run in baseline[1:]),
    ]


def _worst(reference: Compatibility, results: list[Compatibility]) -> Compatibility:
    """``reference``, made unverified with every unverified field if any was."""
    unverified = [result for result in results if result.status == UNVERIFIED]
    if not unverified:
        return reference
    names: dict[str, Any] = {}
    for result in unverified:
        for item in result.unverified:
            names.setdefault(item.name, item)
    return replace(reference, status=UNVERIFIED, unverified=tuple(names.values()))


def _incompatible(what: str, result: Compatibility) -> str:
    fields = ", ".join(
        f"{item.name} ({item.a!r} vs {item.b!r})" for item in result.blocking[:5]
    )
    more = f" and {len(result.blocking) - 5} more" if len(result.blocking) > 5 else ""
    return f"{what} are incompatible: {fields}{more}; allow a difference with --allow"


def _observer_contract(
    usable: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> list[str]:
    """Problems with the observers under the mode's contract.

    An observer present where the contract forbids it, or an undeclared one
    added, is an input error; one that did not hold up makes the gates
    ``not_evaluable``, and is returned.
    """
    if spec.mode == OVERHEAD:
        present = _requested(usable[BASELINE])
        if present:
            raise InferInputError(
                "an overhead comparison needs a baseline without observers; it ran "
                + ", ".join(sorted(present))
            )
        return _unhealthy(usable[CANDIDATE], _requested(usable[CANDIDATE]))
    if spec.mode == INCREMENTAL:
        added = _requested(usable[CANDIDATE]) - _requested(usable[BASELINE])
        undeclared = added - set(spec.added_observers)
        if undeclared:
            raise InferInputError(
                "observers added in the candidate must be declared with "
                "--added-observers: " + ", ".join(sorted(undeclared))
            )
        shared = _requested(usable[BASELINE]) & _requested(usable[CANDIDATE])
        return _unhealthy(usable[BASELINE], shared) + _unhealthy(
            usable[CANDIDATE], shared | added
        )
    return []


def _requested(runs: list[RunSummary]) -> set[str]:
    return {
        name
        for run in runs
        for name, state in run.observers.items()
        if state.get("requested")
    }


def _unhealthy(runs: list[RunSummary], names: set[str]) -> list[str]:
    issues = []
    for run in runs:
        for name in sorted(names):
            state = run.observers.get(name) or {}
            if state.get("healthy") is True:
                continue
            word = (
                "unjudged"
                if state.get("active") and state.get("healthy") is None
                else "not active and healthy"
            )
            issues.append(f"{name} {word} in {run.name}")
    return issues


# ---------------------------------------------------------------- design


def _design(arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec) -> str:
    """Paired by block when every run is labelled with one, else independent."""
    runs = [run for group in arms.values() for run in group]
    labelled = all(run.label("block") is not None for run in runs)
    if spec.design == PAIRED and not labelled:
        raise InferInputError("a paired design needs every run labelled with --block")
    if spec.design == INDEPENDENT or not labelled:
        return INDEPENDENT
    _check_one_run_per_block(arms)
    return PAIRED


def _check_one_run_per_block(arms: Mapping[str, list[RunSummary]]) -> None:
    """A block holds one run of each arm: a second one is an input error."""
    seen: set[tuple[Any, Any, str]] = set()
    for arm, group in arms.items():
        for run in group:
            key = (run.label("experiment"), run.label("block"), arm)
            if key in seen:
                raise InferInputError(
                    f"block {key[1]!r} of experiment {key[0]!r} has two {arm} runs"
                )
            seen.add(key)


def _block_key(run: RunSummary) -> str:
    return f"{run.label('experiment')}/{run.label('block')}"


# ----------------------------------------------------------------- cases


def _case_ids(arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec) -> list[str]:
    if spec.cases is not None:
        return list(spec.cases)
    ids: set[str] = set()
    for runs in arms.values():
        for run in runs:
            ids.update(run.comparable_cases)
    return sorted(ids)


def _case(
    case_id: str,
    arms: Mapping[str, list[RunSummary]],
    excluded: list[dict[str, Any]],
    design: str,
    spec: ComparisonSpec,
    compatibility: Compatibility,
    observer_issues: list[str],
) -> dict[str, Any]:
    kept = {
        arm: [run for run in runs if not _set_aside(excluded, run, case_id)]
        for arm, runs in arms.items()
    }
    attrition = [item for item in excluded if item["case"] == case_id]
    blocked = _case_blocker(compatibility, observer_issues, attrition, spec)
    metrics = _case_metrics(case_id, kept, design, spec, blocked)
    case: dict[str, Any] = {
        "metrics": metrics,
        "absent_gates": _absent_gates(metrics, spec),
        "attrition": attrition,
    }
    if spec.min_attainment is not None:
        slo_blocked = blocked or _slo_blocker(case_id, kept)
        case.update(_attainment_gate(case_id, kept[CANDIDATE], spec, slo_blocked))
    return case


def _check_cases_exist(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> None:
    """Every case asked for is in some run: a misspelled one compares nothing."""
    present = {
        case for runs in arms.values() for run in runs for case in run.comparable_cases
    }
    missing = [case for case in spec.cases or () if case not in present]
    if missing:
        raise InferInputError(
            f"case {', '.join(missing)} is in no run; the runs have "
            + (", ".join(sorted(present)) or "no case")
        )


def _absent_gates(
    metrics: Mapping[str, MetricComparison], spec: ComparisonSpec
) -> dict[str, GateOutcome]:
    """Each requested gate that matched no metric this case has.

    A gate on what the runs never measured (server latency without spans)
    gated nothing; reading that as a pass would hide the change it asked
    about.
    """
    return {
        pattern: GateOutcome(NOT_EVALUABLE, "metric_absent", rule)
        for pattern, rule in spec.gates
        if not any(fnmatch.fnmatchcase(name, pattern) for name in metrics)
    }


def _case_metrics(
    case_id: str,
    kept: Mapping[str, list[RunSummary]],
    design: str,
    spec: ComparisonSpec,
    blocked: str | None,
) -> dict[str, MetricComparison]:
    reference = next(
        (run.comparable_cases[case_id] for runs in kept.values() for run in runs), None
    )
    if reference is None:
        return {}
    return {
        metric.name: _metric(metric, case_id, kept, design, spec, blocked)
        for metric in default_metrics(reference)
    }


def _case_blocker(
    compatibility: Compatibility,
    observer_issues: list[str],
    attrition: list[dict[str, Any]],
    spec: ComparisonSpec,
) -> str | None:
    """Why no gate of this case can be decided, before any metric is read."""
    if compatibility.status == UNVERIFIED:
        return "unverified"
    if observer_issues:
        return "observer_not_active"
    if attrition and spec.on_incomplete == FAIL_INCOMPLETE:
        return "protocol_failure"
    return None


def _metric(
    metric: MetricSpec,
    case_id: str,
    kept: Mapping[str, list[RunSummary]],
    design: str,
    spec: ComparisonSpec,
    blocked: str | None,
) -> MetricComparison:
    first, second, reasons = _readings(metric, case_id, kept)
    gate = spec.gate_for(metric.name)
    # With --on-incomplete fail, a set-aside run fails the contrast outright.
    protocol_gate = gate if blocked == "protocol_failure" else None
    compared = compare_values(
        metric.name,
        first,
        second,
        direction=metric.direction,  # type: ignore[arg-type]
        scale=metric.scale,  # type: ignore[arg-type]
        unit=metric.unit,
        value_unit=metric.value_unit,
        blocks=_blocks(kept, design),
        confidence=spec.confidence,
        gate=None if protocol_gate else gate,
        bootstrap_min_n=spec.bootstrap_min_n,
        seed=spec.seed,
        unavailable=blocked or _metric_blocker(metric, case_id, kept, reasons, spec),
    )
    if protocol_gate is None:
        return compared
    return replace(compared, gate=GateOutcome(FAIL, "protocol_failure", protocol_gate))


def _readings(
    metric: MetricSpec, case_id: str, kept: Mapping[str, list[RunSummary]]
) -> tuple[list[Any], list[Any], list[str]]:
    """Each arm's values for a metric, and every reason a run gave for none."""
    values: dict[str, list[Any]] = {}
    reasons: list[str] = []
    for arm, runs in kept.items():
        values[arm] = []
        for run in runs:
            value, reason = metric.read(run.comparable_cases[case_id])
            values[arm].append(value)
            if reason:
                reasons.append(reason)
    return values[BASELINE], values[CANDIDATE], reasons


def _metric_blocker(
    metric: MetricSpec,
    case_id: str,
    kept: Mapping[str, list[RunSummary]],
    reasons: list[str],
    spec: ComparisonSpec,
) -> str | None:
    """Why this metric's gate cannot be decided, from the runs' readings."""
    if reasons:
        return sorted(set(reasons))[0]
    if metric.slo:
        differs = _slo_blocker(case_id, kept)
        if differs is not None:
            return differs
        coverages = [
            evidence_coverage(run.comparable_cases[case_id])
            for runs in kept.values()
            for run in runs
        ]
        if any(c is None or c < spec.evidence_floor for c in coverages):
            return "evidence_coverage_below_floor"
    return None


def _slo_blocker(case_id: str, kept: Mapping[str, list[RunSummary]]) -> str | None:
    """SLO metrics are comparable only when one policy judged every run.

    Without ``--slo``, each run is judged by the policy it recorded; a
    candidate judged by a looser one would meet it however slow it was.
    """
    digests = {
        (run.comparable_cases[case_id].get("slo") or {}).get("slo_digest")
        for runs in kept.values()
        for run in runs
    }
    return "slo_policy_differs" if len(digests - {None}) > 1 else None


def _blocks(
    kept: Mapping[str, list[RunSummary]], design: str
) -> tuple[list[str], list[str]] | None:
    if design != PAIRED:
        return None
    return (
        [_block_key(run) for run in kept[BASELINE]],
        [_block_key(run) for run in kept[CANDIDATE]],
    )


# ------------------------------------------------------------ attainment


def _attainment_gate(
    case_id: str,
    candidate: list[RunSummary],
    spec: ComparisonSpec,
    blocked: str | None,
) -> dict[str, Any]:
    """Whether a share of candidate runs meets the target, and the mean, described.

    It obeys the case's blockers like any gate. A candidate run whose SLO
    could not be judged counts as not meeting the target: leaving it out
    would keep only the runs that met it.
    """
    target = spec.min_attainment
    assert target is not None
    if blocked is not None:
        status = FAIL if blocked == "protocol_failure" else NOT_EVALUABLE
        return {"attainment_gate": {"status": status, "reason": blocked}}
    lowers = _attainment_lowers(case_id, candidate)
    known = [x for x in lowers if x is not None]
    if not known:
        return {
            "attainment_gate": {"status": NOT_EVALUABLE, "reason": "no_slo_evaluation"}
        }
    if spec.attainment_model == "bernoulli":
        gate = _pooled_attainment(case_id, candidate, target, spec)
    else:
        meeting = sum(1 for value in known if value >= target)
        gate = run_pass_gate(meeting, len(lowers), spec.min_run_pass, spec.confidence)
    gate["target"] = target
    gate["runs_unmeasurable"] = len(lowers) - len(known)
    return {"attainment_gate": gate, "attainment_mean": _mean_interval(known, spec)}


def _attainment_lowers(case_id: str, runs: list[RunSummary]) -> list[float | None]:
    """Each run's attainment lower bound, or None where its SLO was not judged."""
    lowers: list[float | None] = []
    for run in runs:
        value = (run.comparable_cases[case_id].get("slo") or {}).get("attainment_lower")
        lowers.append(float(value) if isinstance(value, (int, float)) else None)
    return lowers


def _pooled_attainment(
    case_id: str, candidate: list[RunSummary], target: float, spec: ComparisonSpec
) -> dict[str, Any]:
    """Requests as independent trials: model-based, and only when asked for."""
    met = sum(
        int((r.comparable_cases[case_id].get("slo") or {}).get("met") or 0)
        for r in candidate
    )
    offered = sum(
        int((r.comparable_cases[case_id].get("slo") or {}).get("offered") or 0)
        for r in candidate
    )
    bound = clopper_pearson(
        met, max(offered, 1), spec.confidence, model="independent_requests"
    )
    return {
        **bound.to_record(),
        "status": PASS if bound.lower >= target else FAIL,
    }


def _mean_interval(values: list[float], spec: ComparisonSpec) -> dict[str, Any]:
    """The runs' mean attainment with a t interval: descriptive, model-based."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "lower": None, "upper": None, "model": "normal_run_means"}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    half = (
        float(stats.t.ppf(0.5 + spec.confidence / 2, len(values) - 1))
        * sd
        / math.sqrt(len(values))
    )
    return {
        "mean": mean,
        "lower": mean - half,
        "upper": mean + half,
        "model": "normal_run_means",
    }


# ---------------------------------------------------------------- family


def _with_family(comparison: Comparison) -> Comparison:
    """Under any_regression, Holm adjusts the regression tests across the family."""
    if comparison.spec.family != ANY_REGRESSION:
        return comparison
    p_values = _family_p_values(comparison)
    # One-sided tests, as each regression rule reads one side of its interval.
    rejected = holm(p_values, alpha=(1 - comparison.spec.confidence) / 2)
    cases = {
        case_id: {
            **case,
            "metrics": {
                name: _holm_gate(metric, rejected.get(f"{case_id}/{name}"))
                for name, metric in case["metrics"].items()
            },
        }
        for case_id, case in comparison.cases.items()
    }
    family = {
        "name": ANY_REGRESSION,
        "method": "holm",
        "tests": len(p_values),
        "rejected": sorted(key for key, value in rejected.items() if value),
    }
    return replace(comparison, cases=cases, family=family)


def _family_p_values(comparison: Comparison) -> dict[str, float | None]:
    """Each gated metric's one-sided p-value for a regression, best case.

    Gates decided otherwise (not evaluable, a zero, a degenerate metric) are
    left out of the family and keep their decision.
    """
    found: dict[str, float | None] = {}
    for case_id, case in comparison.cases.items():
        for name, metric in case["metrics"].items():
            gate = metric.gate
            if gate is None or gate.reason not in (None, "exceeds_budget"):
                continue
            if metric.best is not None and metric.best.p_worse is not None:
                found[f"{case_id}/{name}"] = metric.best.p_worse
    return found


def _holm_gate(metric: MetricComparison, rejected: bool | None) -> MetricComparison:
    """A significant-rule gate, with Holm deciding whether the change is real."""
    gate = metric.gate
    if gate is None or rejected is None or gate.status == NOT_EVALUABLE:
        return metric
    best = metric.best
    assert best is not None and best.effect is not None
    bad = best.effect if metric.direction == "lower_is_better" else -best.effect
    failed = rejected and bad > gate.rule.budget
    status = FAIL if failed else PASS
    reason = "exceeds_budget_holm" if failed else None
    return replace(metric, gate=GateOutcome(status, reason, gate.rule, "best_case"))


# ------------------------------------------------------------ reporting


def _run_record(arm: str, run: RunSummary) -> dict[str, Any]:
    return {
        "arm": arm,
        "path": None if run.path is None else str(run.path),
        "sha256": run.sha256,
        "run_id": run.run_id,
        "session_id": run.session_id,
        "labels": dict(run.labels),
        "session_status": run.session_status,
        "protocol_failures": list(run.protocol_failures),
        "started_at_ns": run.started_at_ns,
    }


def _diagnostics(
    arms: Mapping[str, list[RunSummary]], compatibility: Compatibility
) -> dict[str, Any]:
    """Covariates, and whether the arms were run interleaved."""
    starts = {
        arm: sorted(run.started_at_ns for run in runs if run.started_at_ns is not None)
        for arm, runs in arms.items()
    }
    warnings = []
    if starts[BASELINE] and starts[CANDIDATE]:
        if max(starts[BASELINE]) < min(starts[CANDIDATE]):
            warnings.append(
                "every baseline run came before every candidate run: drift over "
                "time is confounded with the change"
            )
    return {
        "started_at_ns": starts,
        "covariates": [item.to_record() for item in compatibility.covariates],
        "warnings": warnings,
    }


__all__ = [
    "ALL_BUDGETS",
    "ANY_REGRESSION",
    "BASELINE",
    "CANDIDATE",
    "COMPARISON_FORMAT",
    "COMPARISON_VERSION",
    "Comparison",
    "ComparisonSpec",
    "compare_runs",
]
