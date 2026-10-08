"""An engine's steps as the matched design reads them: one unit per step
whose completion cadence is defined, with what it ran.

A unit is one complete step of one engine epoch whose every member is known
(none withheld, unresolved, or of unknown role) and whose decode set
continues from the step completed before it, so its ``completion_cadence``,
from that step's completion to its own, is defined. Under async scheduling
that is the pace at which the engine completed steps, not one step's
execution time.

A unit carries what matching compares: its running requests (decode and
prefill members), its drafts, whether it and the step before it in schedule
order ran short of a refill (``Step.refill``), the mean ``computed_before``
per decode member (its decode context), and its prefill members. A prefill
member's tokens are those it scheduled in its context phase (its prompt, and
the output a resumed request computes again); the part that is recompute
after a preemption is the membership's ``recompute`` flag's, or, from a hook
without it, the positions below the request's high-water ``computed_after``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .diagnosis_context import Context
from .diagnosis_inputs import Line
from .diagnosis_join import RunView

DECODE_ROLES = frozenset({"decode", "spec_decode"})
PREFILL_ROLE = "prefill"


@dataclass(frozen=True)
class Prefill:
    """One prefill member of a step."""

    attempt: str
    tokens: int
    cached: int  # computed_before: the prefix computed or found cached
    recompute: int  # of its tokens, those computed again after a preemption


@dataclass(frozen=True)
class Unit:
    """One step with a defined completion cadence."""

    iteration: str
    completed_ns: int
    cadence_ns: int
    running: int  # members: decode and prefill
    drafts: int
    refill: bool  # it ran short of a slot the step before freed
    after_refill: bool  # the step before it in schedule order did
    context: float  # mean computed_before per decode member
    decoders: tuple[str, ...]  # the attempts it decoded for
    prefills: tuple[Prefill, ...] = ()
    line: Line | None = None

    @property
    def dose(self) -> int:
        """Prefill tokens scheduled: above 0, the unit is treated."""
        return sum(p.tokens for p in self.prefills)

    @property
    def treated(self) -> bool:
        return self.dose > 0

    @property
    def cached(self) -> int:
        """The prefill members' summed prefix already computed or cached."""
        return sum(p.cached for p in self.prefills)

    @property
    def longest(self) -> int:
        return max((p.tokens for p in self.prefills), default=0)


@dataclass
class EpochUnits:
    """One engine epoch's units in completion order, and when its steps
    whose members are not all known completed."""

    producer: str
    units: list[Unit] = field(default_factory=list)
    unknown_ns: list[int] = field(default_factory=list)

    def between(self, span: tuple[int, int]) -> list[Unit]:
        return [u for u in self.units if span[0] <= u.completed_ns <= span[1]]

    def unknown_share(self, span: tuple[int, int]) -> float:
        """The share of the steps completed in ``span`` whose members are
        not all known."""
        unknown = sum(1 for at in self.unknown_ns if span[0] <= at <= span[1])
        known = sum(1 for u in self.units if span[0] <= u.completed_ns <= span[1])
        return unknown / (unknown + known) if unknown + known else 0.0


def units_of(context: Context, producer: str) -> EpochUnits:
    """The engine's units, computed once per diagnosis."""
    key = ("units", producer)
    if key not in context.cache:
        context.cache[key] = epoch_units(context, producer)
    found: EpochUnits = context.cache[key]
    return found


def epoch_units(context: Context, producer: str) -> EpochUnits:
    rows = _rows(context.view, producer)
    steps = context.steps(producer).steps
    refill = {step.iteration: step.refill > 0 for step in steps}
    after = {b.iteration: a.refill > 0 for a, b in zip(steps, steps[1:])}
    found = EpochUnits(producer)
    last: tuple[int, frozenset[str]] | None = None  # completion, decoders
    for completed, iteration, line, metadata in _completed(context.view, producer):
        members = rows.get(iteration, [])
        decoders = frozenset(r.attempt for r in members if r.role in DECODE_ROLES)
        if not _known(metadata, members):
            found.unknown_ns.append(completed)
        elif last is not None and decoders & last[1]:
            flags = (refill.get(iteration, False), after.get(iteration, False))
            found.units.append(
                _unit(iteration, completed, completed - last[0], members, flags, line)
            )
        last = (completed, decoders)
    return found


@dataclass(frozen=True)
class _Row:
    attempt: str
    role: str
    context: int
    drafts: int
    prefill: int
    recompute: int


def _rows(view: RunView, producer: str) -> dict[str, list[_Row]]:
    """Each step's members, by iteration."""
    rows: dict[str, list[_Row]] = {}
    for key, execution in view.executions.items():
        if execution.producer != producer:
            continue
        attempt = execution.attempt.id if execution.attempt else key.id
        high = 0
        for _, membership in execution.memberships:
            data = membership.metadata
            role = str(membership.role)
            before = _int(data.get("computed_before"))
            tokens = _prefill(role, data)
            rows.setdefault(membership.iteration_ref.id, []).append(
                _Row(
                    attempt,
                    role,
                    before,
                    _int(data.get("drafts_scheduled")),
                    tokens,
                    _recompute(data.get("recompute"), before, tokens, high),
                )
            )
            after = data.get("computed_after")
            if data.get("outcome") == "kept" and isinstance(after, int):
                high = max(high, after)
    return rows


def _prefill(role: str, data: Mapping[str, Any]) -> int:
    """The context-phase tokens: the prompt's, and a resumed request's
    output computed again."""
    if role != PREFILL_ROLE:
        return 0
    return _int(data.get("prefill_scheduled")) + _int(data.get("past_prompt_scheduled"))


def _recompute(flag: Any, before: int, tokens: int, high: int) -> int:
    if flag is True:
        return tokens
    if flag is False:
        return 0
    return max(0, min(before + tokens, high) - before)


def _completed(
    view: RunView, producer: str
) -> list[tuple[int, str, Line, Mapping[str, Any]]]:
    return sorted(
        (
            (iteration.end_ns, ref.id, line, iteration.metadata)
            for ref, (line, iteration) in view.iterations.items()
            if ref.producer_id == producer and iteration.end_ns is not None
        ),
        key=lambda item: item[0],
    )


def _known(metadata: Mapping[str, Any], members: list[_Row]) -> bool:
    """Complete, with every member written and of a known role."""
    return (
        metadata.get("state") == "complete"
        and not metadata.get("update_failed")
        and not _int(metadata.get("withheld_members"))
        and not _int(metadata.get("unresolved_members"))
        and all(r.role in DECODE_ROLES or r.role == PREFILL_ROLE for r in members)
    )


def _unit(
    iteration: str,
    completed: int,
    cadence: int,
    members: list[_Row],
    refill: tuple[bool, bool],
    line: Line,
) -> Unit:
    decode = [r for r in members if r.role in DECODE_ROLES]
    return Unit(
        iteration=iteration,
        completed_ns=completed,
        cadence_ns=cadence,
        running=len(members),
        drafts=sum(r.drafts for r in decode),
        refill=refill[0],
        after_refill=refill[1],
        context=sum(r.context for r in decode) / len(decode),
        decoders=tuple(sorted(r.attempt for r in decode)),
        prefills=tuple(
            Prefill(r.attempt, r.prefill, r.context, r.recompute)
            for r in members
            if r.role == PREFILL_ROLE and r.prefill > 0
        ),
        line=line,
    )


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "DECODE_ROLES",
    "EpochUnits",
    "Prefill",
    "Unit",
    "epoch_units",
    "units_of",
]
