"""Coverage of an imported vLLM execution log, as separate dimensions.

The report answers how much of the measured GPU time the execution log
explains, and what it does not, without mixing the answers:

1. linkage: busy time with an iteration link against without, by reason;
2. membership: linked time whose iteration has complete, incomplete or no
   membership;
3. ownership: linked time in steps that ran only this run's requests, only
   other clients', both, or unresolved IDs;
4. measurement: measured activity (a device UUID and a device clock)
   against unmeasured, which is counted but never added to a device's time;
5. capture loss: what the hook dropped, what still waits, and the calls that
   ran without a range.

Every GPU figure is a union of busy intervals per device and clock scope,
never a sum of per-iteration unions. A case's figure covers every step one
of its requests shared, so case figures that share a batch are labelled
non-additive. Nothing here estimates per-request GPU cost.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .correlation_accounting import resolve_inference_events
from .correlation_events import (
    ActivityReferenceEvent,
    CapabilityEvent,
    CorrelationEvent,
    EntityRef,
    IterationEvent,
    LegacyInferenceRecord,
    MembershipEvent,
    RequestEvent,
    activity_busy_intervals,
    parse_inference_record,
)
from .trace_kineto import merge_intervals
from .vllm_execution import FOREIGN, OWN, STATE_INCOMPLETE, UNRESOLVED
from .vllm_execution_import import SOURCE

NON_ADDITIVE = "steps shared between cases or with other clients count in each"
# Latest-value counters per epoch: what the writer or the last read says now.
# Step loss is counted from the canonical records instead, and unattached
# finishes from their record sequences, so a later import that reads the same
# log again cannot make the report look better than the data is.
_LOSS_COUNTERS = ("gaps", "iterations_pending", "range_misses", "startup_unranged")


@dataclass
class _Scope:
    """Busy intervals of one device and clock, bucketed per dimension."""

    linked: list[tuple[int, int]] = field(default_factory=list)
    unlinked: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    membership: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    ownership: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    cases: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def summary(self) -> dict[str, Any]:
        measured = self.linked + [i for items in self.unlinked.values() for i in items]
        return {
            "measured_busy_ns": _union_ns(measured),
            "linkage": {
                "linked_ns": _union_ns(self.linked),
                "unlinked_ns": _union_ns(
                    [i for items in self.unlinked.values() for i in items]
                ),
                "unlinked_by_reason_ns": _unions(self.unlinked),
            },
            "membership_ns": _unions(self.membership),
            "ownership_ns": _unions(self.ownership),
        }


@dataclass
class _Steps:
    """What the imported steps and memberships say about each iteration."""

    iterations: dict[EntityRef, IterationEvent]
    members: dict[EntityRef, list[MembershipEvent]]
    cases_by_request: dict[str, str]

    def membership_state(self, ref: EntityRef) -> str:
        iteration = self.iterations.get(ref)
        if iteration is None or not self.members.get(ref):
            return "none"
        if iteration.metadata.get("state") == STATE_INCOMPLETE:
            return STATE_INCOMPLETE
        return "complete"

    def ownership(self, ref: EntityRef) -> str:
        """Established ownership only: ``mixed`` needs both a run member and
        another client's; an unresolved member leaves the split unknown."""
        run, foreign, unresolved = self._member_counts(ref)
        if run and foreign:
            return "mixed"
        if unresolved:
            return UNRESOLVED
        if run:
            return OWN
        return FOREIGN if foreign else "none"

    def _member_counts(self, ref: EntityRef) -> tuple[int, int, int]:
        """Run, foreign and unresolved members: from the step's own counts,
        which include members withheld for privacy, else its memberships."""
        iteration = self.iterations.get(ref)
        metadata = iteration.metadata if iteration is not None else {}
        run = _count(metadata.get("run_members"))
        foreign = _count(metadata.get("foreign_members"))
        unresolved = _count(metadata.get("unresolved_members"))
        if run is not None and foreign is not None and unresolved is not None:
            return run, foreign, unresolved
        owners = Counter(_ownership(m) for m in self.members.get(ref, []))
        return owners[OWN], owners[FOREIGN], owners[UNRESOLVED]

    def cases(self, ref: EntityRef) -> set[str]:
        return {
            self.cases_by_request[m.request_ref.id]
            for m in self.members.get(ref, [])
            if _ownership(m) == OWN and m.request_ref.id in self.cases_by_request
        }

    def shared(self, ref: EntityRef) -> bool:
        """Shared with another case, or with anyone not established as this run."""
        _run, foreign, unresolved = self._member_counts(ref)
        return len(self.cases(ref)) > 1 or foreign > 0 or unresolved > 0


def execution_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The coverage block for an artifact's raw records."""
    events, skipped = _correlation_events(records)
    imports = _imports(events)
    try:
        graph = resolve_inference_events(events)
    except ValueError as exc:
        return {"available": bool(imports), "error": str(exc), "imports": imports}
    iterations = {
        ref: it for ref, it in graph.iterations.items() if it.context.source == SOURCE
    }
    if not imports and not iterations:
        return {"available": False}
    steps = _Steps(
        iterations, _members_by_iteration(graph.memberships), _cases(records)
    )
    scopes, unmeasured = _gpu_dimensions(graph.activities.values(), steps)
    report: dict[str, Any] = {
        "available": True,
        "imports": imports,
        "iterations": _iteration_counts(steps),
        # Only the memberships of the steps counted above: another engine
        # adapter's memberships are not this import's.
        "memberships": dict(
            Counter(
                _ownership(m)
                for m in graph.memberships
                if m.iteration_ref in iterations
            )
        ),
        "requests": _request_counts(graph.requests.values(), records),
        "gpu": {key: scope.summary() for key, scope in sorted(scopes.items())},
        "unmeasured": unmeasured,
        "capture_loss": _capture_loss(imports, steps),
        "cases": _case_coverage(steps, scopes),
        "non_additive": NON_ADDITIVE,
    }
    if skipped:
        report["skipped_records"] = skipped
    return report


def _correlation_events(
    records: list[dict[str, Any]],
) -> tuple[list[CorrelationEvent], int]:
    events: list[CorrelationEvent] = []
    skipped = 0
    for record in records:
        if record.get("schema_version", 1) == 1:
            continue
        try:
            parsed = parse_inference_record(record)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not isinstance(parsed, LegacyInferenceRecord):
            events.append(parsed)
    return events, skipped


def _imports(events: list[CorrelationEvent]) -> list[dict[str, Any]]:
    """Each execution import's summary, in artifact order."""
    imports = []
    for event in events:
        if (
            not isinstance(event, CapabilityEvent)
            or event.component != "engine_adapter"
        ):
            continue
        summary = event.metadata.get("summary") or {}
        execution = summary.get("execution") if isinstance(summary, dict) else None
        if isinstance(execution, dict):
            imports.append({**execution, "collected": list(event.collected)})
    return imports


def _members_by_iteration(
    memberships: tuple[MembershipEvent, ...],
) -> dict[EntityRef, list[MembershipEvent]]:
    members: dict[EntityRef, list[MembershipEvent]] = defaultdict(list)
    for membership in memberships:
        members[membership.iteration_ref].append(membership)
    return members


def _cases(records: list[dict[str, Any]]) -> dict[str, str]:
    """The client's request IDs to their cases."""
    cases = {}
    for record in records:
        if record.get("event_type") == "infer.request" and record.get("x_request_id"):
            cases[str(record.get("request_id"))] = str(record.get("case_id"))
    return cases


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ownership(membership: MembershipEvent) -> str:
    owner = membership.metadata.get("ownership")
    if isinstance(owner, str) and owner:
        return owner
    return OWN if membership.request_ref.producer_id == "stormlog" else "unknown"


def _gpu_dimensions(
    activities: Any, steps: _Steps
) -> tuple[dict[str, _Scope], dict[str, Any]]:
    scopes: dict[str, _Scope] = defaultdict(_Scope)
    unmeasured = {"activity_records": 0, "summed_activity_ns": 0, "linked_records": 0}
    for activity in activities:
        if activity.activity_domain != "gpu":
            continue
        key, busy = _measured(activity)
        if key is None or busy is None:
            unmeasured["activity_records"] += 1
            unmeasured["linked_records"] += int(activity.iteration_ref is not None)
            unmeasured["summed_activity_ns"] += _summed_ns(activity)
            continue
        _bucket(scopes[key], activity, busy, steps)
    return dict(scopes), unmeasured


def _bucket(
    scope: _Scope,
    activity: ActivityReferenceEvent,
    busy: list[tuple[int, int]],
    steps: _Steps,
) -> None:
    ref = activity.iteration_ref
    if ref is None:
        reason = str(activity.metadata.get("unresolved_reason") or "unknown")
        scope.unlinked[reason].extend(busy)
        return
    scope.linked.extend(busy)
    scope.membership[steps.membership_state(ref)].extend(busy)
    scope.ownership[steps.ownership(ref)].extend(busy)
    for case_id in steps.cases(ref):
        scope.cases[case_id].extend(busy)


def _measured(
    activity: ActivityReferenceEvent,
) -> tuple[str | None, list[tuple[int, int]] | None]:
    """The device and clock scope of a measurable activity, with its busy
    intervals; a UUID-less or wall-clock activity is not measured."""
    context = activity.context
    if context.device_uuid is None or context.clock_kind not in {"monotonic", "device"}:
        return None, None
    return f"{context.device_uuid}@{context.clock_domain}", activity_busy_intervals(
        activity
    )


def _summed_ns(activity: ActivityReferenceEvent) -> int:
    summed = activity.metadata.get("summed_duration_ns")
    if isinstance(summed, int) and not isinstance(summed, bool):
        return summed
    if activity.start_ns is not None and activity.end_ns is not None:
        return activity.end_ns - activity.start_ns
    return 0


def _iteration_counts(steps: _Steps) -> dict[str, Any]:
    states = Counter(
        str(it.metadata.get("state") or "complete") for it in steps.iterations.values()
    )
    owners = Counter(steps.ownership(ref) for ref in steps.iterations)
    return {
        "total": len(steps.iterations),
        "complete": states.get("complete", 0),
        "incomplete": states.get(STATE_INCOMPLETE, 0),
        "ownership": dict(owners),
    }


def _request_counts(requests: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    owners: Counter[str] = Counter()
    bound: set[str] = set()
    for request in requests:
        if not isinstance(request, RequestEvent) or request.context.source != SOURCE:
            continue
        owner = str(request.metadata.get("ownership") or "unknown")
        owners[owner] += 1
        if owner == OWN:
            bound.add(request.request_ref.id)
    return {
        "executions": dict(owners),
        "run_requests_bound": len(bound),
        "run_requests_total": len(_cases(records)),
    }


def _capture_loss(imports: list[dict[str, Any]], steps: _Steps) -> dict[str, Any]:
    """Per epoch the latest import's counters, summed across epochs; step
    loss from the records the artifact holds; unattached finishes by their
    record sequence across every import."""
    latest: dict[str, dict[str, Any]] = {}
    unattached: set[tuple[str, int]] = set()
    for item in imports:
        for name, epoch in (item.get("epochs") or {}).items():
            if isinstance(epoch, dict):
                latest[str(name)] = epoch
                for seq in epoch.get("finish_unattached_seqs") or []:
                    unattached.add((str(name), int(seq)))
    dropped: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for epoch in latest.values():
        _add_epoch_loss(epoch, dropped, totals)
    iterations = steps.iterations.values()
    return {
        "epochs": len(latest),
        "dropped": dict(dropped),
        "missing_sequences": totals["gaps"],
        "pending_iterations": totals["iterations_pending"],
        "incomplete_iterations": sum(
            it.metadata.get("state") == STATE_INCOMPLETE for it in iterations
        ),
        # Steps whose update_from_output raised: every member's outcome unknown.
        "update_failed_iterations": sum(
            bool(it.metadata.get("update_failed")) for it in iterations
        ),
        "range_misses": totals["range_misses"],
        "finish_unattached": len(unattached),
        # Not loss: the calls before the first serving step never have a range.
        "startup_unranged": totals["startup_unranged"],
        "truncated_epochs": totals["truncated"],
        "read_errors": totals["errors"],
    }


def _add_epoch_loss(
    epoch: dict[str, Any], dropped: Counter[str], totals: Counter[str]
) -> None:
    dropped.update({k: int(v) for k, v in (epoch.get("dropped") or {}).items()})
    for name in _LOSS_COUNTERS:
        totals[name] += int(epoch.get(name) or 0)
    totals["truncated"] += int(bool(epoch.get("truncated")))
    totals["errors"] += len(epoch.get("errors") or [])


def _case_coverage(steps: _Steps, scopes: dict[str, _Scope]) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for ref in steps.iterations:
        for case_id in steps.cases(ref):
            entry = cases.setdefault(
                case_id, {"iterations": 0, "shared_iterations": 0, "gpu": {}}
            )
            entry["iterations"] += 1
            entry["shared_iterations"] += int(steps.shared(ref))
    for case_id, entry in cases.items():
        entry["non_additive"] = entry["shared_iterations"] > 0
        entry["gpu"] = {
            key: {"linked_ns": _union_ns(scope.cases[case_id])}
            for key, scope in sorted(scopes.items())
            if case_id in scope.cases
        }
    return dict(sorted(cases.items()))


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def _unions(buckets: dict[str, list[tuple[int, int]]]) -> dict[str, int]:
    return {name: _union_ns(items) for name, items in sorted(buckets.items())}


# ------------------------------------------------------------------- text


def execution_lines(block: Any) -> list[str]:
    """Text lines for the coverage block; nothing when it was not imported."""
    if not isinstance(block, dict) or not block.get("available"):
        return []
    if block.get("error"):
        return [f"vLLM execution: unresolved ({block['error']})"]
    failed = [item["failed"] for item in block.get("imports", []) if item.get("failed")]
    if failed and not block.get("iterations", {}).get("total"):
        return [f"vLLM execution: import failed ({failed[-1]})"]
    lines = [_headline(block)]
    for key, scope in block.get("gpu", {}).items():
        lines.append(f"  {key}: {_scope_text(scope)}")
    lines.extend(_unmeasured_lines(block.get("unmeasured", {})))
    lines.append(f"  capture loss: {_loss_text(block.get('capture_loss', {}))}")
    for case_id, case in block.get("cases", {}).items():
        lines.append(f"  {case_id}: {_case_text(case)}")
    return lines


def _headline(block: dict[str, Any]) -> str:
    iterations = block.get("iterations", {})
    memberships = block.get("memberships", {})
    requests = block.get("requests", {})
    members = ", ".join(f"{k} {v}" for k, v in sorted(memberships.items()))
    return (
        f"vLLM execution: {iterations.get('total', 0)} steps imported "
        f"({iterations.get('complete', 0)} complete, "
        f"{iterations.get('incomplete', 0)} incomplete), memberships "
        f"{members or 'none'}, {requests.get('run_requests_bound', 0)} of "
        f"{requests.get('run_requests_total', 0)} run requests bound"
    )


def _scope_text(scope: dict[str, Any]) -> str:
    linkage = scope.get("linkage", {})
    reasons = ", ".join(
        f"{reason} {_ms(ns)}"
        for reason, ns in linkage.get("unlinked_by_reason_ns", {}).items()
    )
    return (
        f"linked {_ms(linkage.get('linked_ns', 0))} of "
        f"{_ms(scope.get('measured_busy_ns', 0))} measured busy"
        + (f" (unlinked: {reasons})" if reasons else "")
        + f"; membership {_split(scope.get('membership_ns', {}))}"
        + f"; ownership {_split(scope.get('ownership_ns', {}))}"
    )


def _unmeasured_lines(unmeasured: dict[str, Any]) -> list[str]:
    count = unmeasured.get("activity_records", 0)
    if not count:
        return []
    return [
        f"  unmeasured: {count} GPU activity records without a device UUID or "
        f"device clock, {_ms(unmeasured.get('summed_activity_ns', 0))} summed "
        "(not comparable with the unions above)"
    ]


def _loss_text(loss: dict[str, Any]) -> str:
    dropped = sum(int(v) for v in (loss.get("dropped") or {}).values())
    return (
        f"{dropped} records dropped by the hook, "
        f"{loss.get('missing_sequences', 0)} missing, "
        f"{loss.get('pending_iterations', 0)} steps pending, "
        f"{loss.get('incomplete_iterations', 0)} incomplete, "
        f"{loss.get('update_failed_iterations', 0)} update failures, "
        f"{loss.get('range_misses', 0)} range misses, "
        f"{loss.get('finish_unattached', 0)} finishes unattached; "
        f"{loss.get('startup_unranged', 0)} start-up calls unranged (not loss)"
    )


def _case_text(case: dict[str, Any]) -> str:
    gpu = ", ".join(
        f"{key} linked {_ms(v.get('linked_ns', 0))}"
        for key, v in case.get("gpu", {}).items()
    )
    text = f"{case.get('iterations', 0)} steps"
    if case.get("shared_iterations"):
        text += f", {case['shared_iterations']} shared (non-additive)"
    return text + (f"; {gpu}" if gpu else "")


def _split(values: dict[str, int]) -> str:
    return ", ".join(f"{name} {_ms(ns)}" for name, ns in values.items()) or "none"


def _ms(ns: Any) -> str:
    return f"{int(ns) / 1e6:.3f} ms"


__all__ = ["NON_ADDITIVE", "execution_lines", "execution_report"]
