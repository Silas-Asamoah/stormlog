"""Comparing a baseline arm of runs with a candidate arm, case by case.

``compare_runs`` takes run summaries (``run_summary``) and:

1. sets aside each run that cannot stand for a case, with its protocol
   failure, and lists it. In a paired design the whole block goes, both
   arms, and more than one block set aside leaves the case's gates
   ``not_evaluable``. Outcome failures are never set aside: a run that did
   not finish, or a case a run lacks, is compared, and a value it lost
   fails the candidate's gate (``outcome_unrecoverable``). A retried block
   keeps its last attempt and lists the others; a retry never replaces an
   outcome failure;
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
# More blocks (or runs) set aside than this leave a case's gates unjudged.
SET_ASIDE_LIMIT = 1
OUTCOME_UNRECOVERABLE = "outcome_unrecoverable"
BASELINE_OUTCOME_UNRECOVERABLE = "baseline_outcome_unrecoverable"
PROTOCOL_FAILURE = "protocol_failure"
# Reasons that fail a gate outright rather than leave it unjudged.
_FAILING = (PROTOCOL_FAILURE, OUTCOME_UNRECOVERABLE)
# A lost outcome is why a metric has too few values, whatever the
# statistics say.
_LOST = (OUTCOME_UNRECOVERABLE, BASELINE_OUTCOME_UNRECOVERABLE)


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
    # Pre-registered fallback budgets: (metric name or pattern, budget, unit).
    fallbacks: tuple[tuple[str, float, str], ...] = ()

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
        """The gate on a metric: the last pattern that matches its name.

        A fallback budget whose name or pattern matches the metric applies
        to it, whichever gate pattern named the metric.
        """
        found = None
        for pattern, rule in self.gates:
            if fnmatch.fnmatchcase(metric, pattern):
                found = rule
        for pattern, budget, unit in self.fallbacks:
            if found is not None and fnmatch.fnmatchcase(metric, pattern):
                found = replace(found, fallback_budget=budget, fallback_unit=unit)
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
            "fallbacks": [list(item) for item in self.fallbacks],
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
    _check_each_run_once(arms)
    _check_cases_exist(arms, spec)
    _check_cases_hold_requests(arms, spec)
    _check_one_slo_policy(arms, spec)
    design = _design(arms, spec)
    excluded, standing = _standing(arms, spec, design)
    usable = {arm: _usable(runs) for arm, runs in standing.items()}
    if not usable[BASELINE] or not usable[CANDIDATE]:
        raise InferInputError("an arm has no usable run: every run was excluded")
    compatibility, unverified_pairs = _compatibility(usable, spec)
    observer_issues = _observer_contract(usable, spec)
    if design == PAIRED:
        excluded += _unpaired_blocks(standing, spec, excluded)
        excluded += _partners(standing, spec, excluded)
    cases = {
        case_id: _case(
            case_id, standing, excluded, design, spec, compatibility, observer_issues
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


def _check_each_run_once(arms: Mapping[str, list[RunSummary]]) -> None:
    """One artifact given twice in an arm would count a single run as two."""
    for runs in arms.values():
        seen: set[str] = set()
        for run in runs:
            keys = {run.name} | ({run.sha256} if run.sha256 else set())
            if keys & seen:
                raise InferInputError(f"run {run.name} is given twice")
            seen |= keys


def _item(arm: str, run: RunSummary, case_id: str, reasons: list[str]) -> dict:
    return {"arm": arm, "run": run.name, "case": case_id, "reasons": reasons}


def _standing(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec, design: str
) -> tuple[list[dict[str, Any]], dict[str, list[RunSummary]]]:
    """Each arm's runs once superseded attempts are listed, and every
    (run, case) set aside so far."""
    excluded = _superseded(arms, spec, design)
    replaced = {item["run"] for item in excluded}
    standing = {
        arm: [run for run in runs if run.name not in replaced]
        for arm, runs in arms.items()
    }
    return excluded + _excluded(standing, spec), standing


def _excluded(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> list[dict[str, Any]]:
    """Every (run, case) a protocol failure sets aside, with its reasons.

    A case the run lacks is an outcome, not a set-aside: the treatment may
    have stopped the run before it.
    """
    return [
        _with_evidence(_item(arm, run, case_id, list(run.failures_for(case_id))), run)
        for arm, runs in arms.items()
        for run in runs
        for case_id in _case_ids(arms, spec)
        if run.failures_for(case_id)
    ]


def _with_evidence(item: dict[str, Any], run: RunSummary) -> dict[str, Any]:
    """An external cause's set-aside lists the evidence given for it."""
    evidence = {
        reason: text
        for reason, text in run.external_evidence.items()
        if reason in item["reasons"]
    }
    return {**item, "evidence": evidence} if evidence else item


def _superseded(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec, design: str
) -> list[dict[str, Any]]:
    """Every attempt at a block that another attempt of its arm replaces.

    A block's arm keeps its last attempt, except that a retry never
    replaces an outcome failure: the first attempt that failed as an
    outcome stands, and each later one is listed as
    ``retry_of_outcome_failure``. Every attempt is listed, never dropped.
    """
    if design != PAIRED:
        return []
    found = []
    for arm, attempts in _repeated_blocks(arms):
        kept = _kept_attempt(attempts)
        for index, run in enumerate(attempts):
            if run is kept:
                continue
            after = index > attempts.index(kept)
            reason = "retry_of_outcome_failure" if after else "superseded"
            for case_id in _case_ids(arms, spec):
                item = _item(arm, run, case_id, [reason, *run.failures_for(case_id)])
                item = _with_evidence(item, run)
                item["attempt_kept"] = kept.name
                found.append(item)
    return found


def _repeated_blocks(
    arms: Mapping[str, list[RunSummary]],
) -> list[tuple[str, list[RunSummary]]]:
    """Each arm's attempts at a block it ran more than once, in start order."""
    found: dict[tuple[str, str], list[RunSummary]] = {}
    for arm, runs in arms.items():
        for run in runs:
            found.setdefault((arm, _block_key(run)), []).append(run)
    return [
        (arm, _in_start_order(attempts))
        for (arm, _), attempts in found.items()
        if len(attempts) > 1
    ]


def _in_start_order(attempts: list[RunSummary]) -> list[RunSummary]:
    """By start time, or as given when any start is unknown."""
    if any(run.started_at_ns is None for run in attempts):
        return attempts
    return sorted(attempts, key=lambda run: run.started_at_ns or 0)


def _kept_attempt(attempts: list[RunSummary]) -> RunSummary:
    outcome = next(
        (run for run in attempts if run.outcome_failures and not run.protocol_failures),
        None,
    )
    return outcome or attempts[-1]


def _partners(
    arms: Mapping[str, list[RunSummary]],
    spec: ComparisonSpec,
    excluded: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The other run of each block set aside: the block goes, both arms."""
    found = []
    for case_id in _case_ids(arms, spec):
        lost = _blocks_set_aside(arms, excluded, case_id)
        found += [
            _item(arm, run, case_id, ["block_set_aside"])
            for arm, runs in arms.items()
            for run in runs
            if _block_key(run) in lost and not _set_aside(excluded, run, case_id)
        ]
    return found


def _blocks_set_aside(
    arms: Mapping[str, list[RunSummary]], excluded: list[dict[str, Any]], case_id: str
) -> set[str]:
    return {
        _block_key(run)
        for runs in arms.values()
        for run in runs
        if _set_aside(excluded, run, case_id)
    }


def _lost(
    case_id: str,
    arms: Mapping[str, list[RunSummary]],
    excluded: list[dict[str, Any]],
    design: str,
) -> dict[str, Any]:
    """The blocks (or runs) this case lost: set aside, or missing an arm's run.

    A block given for one arm only lost its other run without a recorded
    cause, so it counts like one set aside.
    """
    if design != PAIRED:
        items = {
            run.name
            for runs in arms.values()
            for run in runs
            if _set_aside(excluded, run, case_id)
        }
        return {"unit": "run", "items": sorted(items), "limit": SET_ASIDE_LIMIT}
    blocks = [{_block_key(run) for run in arms[arm]} for arm in (BASELINE, CANDIDATE)]
    items = _blocks_set_aside(arms, excluded, case_id) | (blocks[0] ^ blocks[1])
    return {"unit": "block", "items": sorted(items), "limit": SET_ASIDE_LIMIT}


def _unpaired_blocks(
    arms: Mapping[str, list[RunSummary]],
    spec: ComparisonSpec,
    excluded: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Blocks whose two runs sent different workload realizations.

    A pair stands for one block's conditions: the same realization (the
    block's seed) in both arms. Runs already set aside are left alone.
    """
    blocks: dict[str, dict[str, RunSummary]] = {}
    for arm, runs in arms.items():
        for run in _usable(runs):
            blocks.setdefault(_block_key(run), {})[arm] = run
    found = []
    for pair in blocks.values():
        if len(pair) != 2 or _same_realization(pair[BASELINE], pair[CANDIDATE]):
            continue
        for arm, run in pair.items():
            for case_id in _case_ids(arms, spec):
                if not _set_aside(excluded, run, case_id):
                    found.append(
                        {
                            "arm": arm,
                            "run": run.name,
                            "case": case_id,
                            "reasons": ["block_realization_differs"],
                        }
                    )
    return found


def _same_realization(first: RunSummary, second: RunSummary) -> bool:
    """Unknown on either side is not shown different."""
    a = first.fields.get("workload.realization_digest")
    b = second.fields.get("workload.realization_digest")
    if a is None or b is None or not (a.known and b.known):
        return True
    return bool(a.value == b.value)


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
    return PAIRED


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
    lost = _lost(case_id, arms, excluded, design)
    blocked = _case_blocker(compatibility, observer_issues, lost, spec)
    metrics = _case_metrics(case_id, kept, design, spec, blocked)
    case: dict[str, Any] = {
        "metrics": metrics,
        "absent_gates": _absent_gates(metrics, spec),
        "attrition": attrition,
        "set_aside": lost,
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
        raise InferUsageError(
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
        (
            run.comparable_cases[case_id]
            for runs in kept.values()
            for run in runs
            if case_id in run.comparable_cases
        ),
        None,
    )
    if reference is None:
        return {}
    return {
        metric.name: _metric(metric, case_id, kept, design, spec, blocked)
        for metric in default_metrics(reference)
    }


def _check_cases_hold_requests(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> None:
    """Refuse a case no run offered a request: a segment outside every phase."""
    for case_id in _case_ids(arms, spec):
        offered = [
            (run.comparable_cases[case_id].get("population") or {}).get("offered")
            for runs in arms.values()
            for run in runs
            if case_id in run.comparable_cases
        ]
        if offered and all(value == 0 for value in offered):
            raise InferInputError(
                f"{case_id} holds no request in any run: a segment outside every "
                "run's measured phase compares nothing"
            )


def _case_blocker(
    compatibility: Compatibility,
    observer_issues: list[str],
    lost: Mapping[str, Any],
    spec: ComparisonSpec,
) -> str | None:
    """Why no gate of this case can be decided, before any metric is read."""
    if compatibility.status == UNVERIFIED:
        return "unverified"
    if observer_issues:
        return "observer_not_active"
    if lost["items"] and spec.on_incomplete == FAIL_INCOMPLETE:
        return PROTOCOL_FAILURE
    if len(lost["items"]) > lost["limit"]:
        return f"{lost['unit']}s_set_aside"
    return None


def _metric(
    metric: MetricSpec,
    case_id: str,
    kept: Mapping[str, list[RunSummary]],
    design: str,
    spec: ComparisonSpec,
    blocked: str | None,
) -> MetricComparison:
    first, second, reasons, lost = _readings(metric, case_id, kept)
    gate = spec.gate_for(metric.name)
    unavailable = _lost_outcome(lost, blocked) or _metric_blocker(
        metric, case_id, kept, reasons, spec
    )
    failing = unavailable in _FAILING
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
        gate=None if failing else gate,
        bootstrap_min_n=spec.bootstrap_min_n,
        seed=spec.seed,
        unavailable=unavailable,
    )
    if unavailable is not None and (compared.reason is None or unavailable in _LOST):
        # Why it was not compared, gated or not.
        compared = replace(compared, reason=unavailable)
    if not failing or gate is None:
        return compared
    return replace(compared, gate=GateOutcome(FAIL, str(unavailable), gate))


def _lost_outcome(lost: set[str], blocked: str | None) -> str | None:
    """A candidate outcome lost, or a set-aside run under --on-incomplete
    fail, fails the contrast outright; a lost baseline one leaves it
    unjudged."""
    if CANDIDATE in lost:
        return OUTCOME_UNRECOVERABLE
    if blocked is not None:
        return blocked
    return BASELINE_OUTCOME_UNRECOVERABLE if BASELINE in lost else None


def _readings(
    metric: MetricSpec, case_id: str, kept: Mapping[str, list[RunSummary]]
) -> tuple[list[Any], list[Any], list[str], set[str]]:
    """Each arm's values for a metric, every reason a run gave for none, and
    the arms that lost an outcome: a value missing from a run that did not
    finish, or from a run that lacks the case."""
    values: dict[str, list[Any]] = {}
    reasons: list[str] = []
    lost: set[str] = set()
    for arm, runs in kept.items():
        values[arm] = []
        for run in runs:
            case = run.comparable_cases.get(case_id)
            value, reason = (
                (None, "case_missing") if case is None else metric.read(case)
            )
            values[arm].append(value)
            if reason:
                reasons.append(reason)
            if value is None and (case is None or run.outcome_failures):
                lost.add(arm)
    return values[BASELINE], values[CANDIDATE], reasons, lost


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
            if case_id in run.comparable_cases
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
        _slo(run, case_id).get("slo_digest") for runs in kept.values() for run in runs
    }
    return "slo_policy_differs" if len(digests - {None}) > 1 else None


def _slo(run: RunSummary, case_id: str) -> Mapping[str, Any]:
    """A case's SLO evaluation; empty when the run lacks the case."""
    return (run.comparable_cases.get(case_id) or {}).get("slo") or {}


def _check_one_slo_policy(
    arms: Mapping[str, list[RunSummary]], spec: ComparisonSpec
) -> None:
    """An SLO gate needs one policy to have judged every run.

    Without ``--slo``, each run is judged by the policy it recorded; a
    candidate judged by a looser one would meet it however slow it was.
    """
    gated = spec.min_attainment is not None or any(
        spec.gate_for(name) is not None for name in SLO_METRICS
    )
    if not gated:
        return
    digests = {
        (case.get("slo") or {}).get("slo_digest")
        for runs in arms.values()
        for run in runs
        for case in run.comparable_cases.values()
    } - {None}
    if len(digests) > 1:
        raise InferInputError(
            "the runs were judged by different SLO policies ("
            + ", ".join(sorted(str(d)[:12] for d in digests))
            + "); give --slo or --slo-file to judge them all by one"
        )


SLO_METRICS = ("goodput_rps", "attainment")


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
    would keep only the runs that met it. Pooling requests would leave a
    lost run's requests out of both counts, so a lost one fails a
    ``bernoulli`` gate.
    """
    target = spec.min_attainment
    assert target is not None
    lowers = _attainment_lowers(case_id, candidate)
    known = [x for x in lowers if x is not None]
    undecided = _attainment_undecided(case_id, candidate, spec, blocked, known)
    if undecided is not None:
        return {"attainment_gate": undecided}
    if spec.attainment_model == "bernoulli":
        gate = _pooled_attainment(case_id, candidate, target, spec)
    else:
        meeting = sum(1 for value in known if value >= target)
        gate = run_pass_gate(meeting, len(lowers), spec.min_run_pass, spec.confidence)
    gate["target"] = target
    gate["runs_unmeasurable"] = len(lowers) - len(known)
    return {"attainment_gate": gate, "attainment_mean": _mean_interval(known, spec)}


def _attainment_undecided(
    case_id: str,
    candidate: list[RunSummary],
    spec: ComparisonSpec,
    blocked: str | None,
    known: list[float],
) -> dict[str, Any] | None:
    """The attainment gate when the runs cannot decide it."""
    if blocked is not None:
        status = FAIL if blocked == PROTOCOL_FAILURE else NOT_EVALUABLE
        return {"status": status, "reason": blocked}
    if spec.attainment_model == "bernoulli" and _lost_slo(case_id, candidate):
        return {"status": FAIL, "reason": OUTCOME_UNRECOVERABLE}
    if not known:
        return {"status": NOT_EVALUABLE, "reason": "no_slo_evaluation"}
    return None


def _attainment_lowers(case_id: str, runs: list[RunSummary]) -> list[float | None]:
    """Each run's attainment lower bound, or None where its SLO was not judged."""
    lowers: list[float | None] = []
    for run in runs:
        value = _slo(run, case_id).get("attainment_lower")
        lowers.append(float(value) if isinstance(value, (int, float)) else None)
    return lowers


def _lost_slo(case_id: str, runs: list[RunSummary]) -> bool:
    """A run lacks the case, or did not finish and has no SLO evaluation."""
    return any(
        case_id not in run.comparable_cases
        or (run.outcome_failures and not _slo(run, case_id))
        for run in runs
    )


def _pooled_attainment(
    case_id: str, candidate: list[RunSummary], target: float, spec: ComparisonSpec
) -> dict[str, Any]:
    """Requests as independent trials: model-based, and only when asked for."""
    met = sum(int(_slo(r, case_id).get("met") or 0) for r in candidate)
    offered = sum(int(_slo(r, case_id).get("offered") or 0) for r in candidate)
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
        "outcome_failures": list(run.outcome_failures),
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
