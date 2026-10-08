"""What drove a finding: more demand than the reference (``load``), less
capacity at the same work (``capacity``), or neither shown
(``undetermined``).

- **Demand**, the subject against its reference: the arrival rate ratio,
  passing when its exact interval's lower end reaches
  ``load_increase.arrival_rate_ratio``; longer prompts or outputs, as the
  workload kinds judge them; and, from the hook, the share of the steps'
  members that no client request of the run accounts for, risen by
  ``driver.foreign_share_rise``: another client's load on the same engine.
- **Capacity**: the subject's steps against the reference's at the same
  work (``diagnosis_matched``), decode-only steps with decode-only ones at
  the same batch and treated steps with treated ones at the same prefill,
  as the median ratio of their cadences.

The verdict follows the frozen rubric. ``capacity`` when the ratio's
interval lies above ``driver.capacity_ratio`` with matched support of at
least ``driver.min_common_support``: the engine slowed at the same work.
``load`` when a demand ratio passes against a compatible reference (the
same engine epoch, or one of the same config) and no capacity loss is
shown; without matched support it still stands, and ``confidence.driver``
lists ``matched_common_support`` unmet. ``undetermined`` otherwise.

The driver explains; it never changes a severity. At ``info`` an eligible
primary's cause is ``workload_change`` when the driver is ``load``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .diagnosis_context import Context
from .diagnosis_matched import Arms, Columns, Comparison, Design, Ratio, compare
from .diagnosis_model import Criteria, met
from .diagnosis_selection import Subject
from .diagnosis_thresholds import (
    DRIVER_CAPACITY_RATIO,
    DRIVER_FOREIGN_RISE,
    DRIVER_MIN_SUPPORT,
    WORKLOAD_RATE_RATIO,
    resolve_threshold,
)
from .diagnosis_units import Span, arm_spans, foreign_members, units_of
from .diagnosis_workload import (
    RateRatio,
    arrival_ratio,
    length_change,
    rate_ratio_interval,
)

CAPACITY = "capacity"
LOAD = "load"
UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class Driver:
    """The verdict, the rubric it met, and what it rests on."""

    driver: str
    confidence: Criteria
    evidence: dict[str, Any]


def driver_of(context: Context, subject: Subject) -> Driver:
    """The subject's driver, computed once per diagnosis."""
    key = ("driver", subject.key)
    if key not in context.cache:
        context.cache[key] = _driver(context, subject)
    found: Driver = context.cache[key]
    return found


def _driver(context: Context, subject: Subject) -> Driver:
    floor = resolve_threshold(DRIVER_CAPACITY_RATIO, context.thresholds)[0]
    demand = _demand(context, subject)
    compatible = _compatible(context, subject)
    capacity = _capacity(context, subject, floor)
    supported = _supported(context, capacity)
    shown = supported and capacity is not None and _above(capacity, floor)
    passing = [name for name, row in demand.items() if row and row.get("passes")]
    verdict = CAPACITY if shown else LOAD if compatible and passing else UNDETERMINED
    return Driver(
        verdict,
        met(compatible_reference=compatible, matched_common_support=supported),
        {
            "reference": {
                "requests": len(subject.reference),
                "executions": len(context.subject_executions(subject, reference=True)),
            },
            "compatible": compatible,
            "demand": demand,
            "demand_passed": passing,
            "capacity": None if capacity is None else capacity.as_dict(),
        },
    )


def _supported(context: Context, capacity: Ratio | None) -> bool:
    needed = resolve_threshold(DRIVER_MIN_SUPPORT, context.thresholds)[0]
    support = capacity.support if capacity is not None else None
    return support is not None and support >= needed


def _above(capacity: Ratio, floor: float) -> bool:
    return capacity.ratio is not None and capacity.ratio.above(floor)


# ------------------------------------------------------------------ demand
def _demand(context: Context, subject: Subject) -> dict[str, dict[str, Any] | None]:
    rates = (
        arrival_ratio(context, subject)
        if subject.requests
        else _engine_arrivals(context, subject)
    )
    return {
        "arrival_rate_ratio": None if rates is None else _rate_row(rates),
        "prompt_tokens": _length_row(context, subject, "prompt_tokens"),
        "output_tokens": _length_row(context, subject, "output_tokens"),
        "foreign_share": _foreign_row(context, subject),
    }


def _rate_row(rates: RateRatio) -> dict[str, Any]:
    return {
        "estimate": round(rates.ratio, 4),
        "ci": [round(rates.low, 4), round(rates.high, 4)],
        "passes": rates.passes,
    }


def _length_row(
    context: Context, subject: Subject, field: str
) -> dict[str, Any] | None:
    change = length_change(context, subject, field) if subject.requests else None
    if change is None:
        return None
    excess = change.excess
    return {
        "excess": round(excess.estimate, 3),
        "ci": [round(excess.low, 3), round(excess.high, 3)],
        "reference_median": change.reference,
        "passes": change.passes,
    }


def _engine_arrivals(context: Context, subject: Subject) -> RateRatio | None:
    """A server-only subject's executions admitted per second, against
    those admitted before it."""
    if subject.start_ns is None or subject.end_ns is None:
        return None
    seconds = (subject.end_ns - subject.start_ns) / 1e9
    admitted = sorted(
        e.event.start_ns
        for e in context.subject_executions(subject, reference=True)
        if e.event.start_ns is not None
    )
    n, n_ref = len(subject.executions), len(admitted)
    if n_ref < 2 or seconds <= 0 or admitted[-1] <= admitted[0]:
        return None
    seconds_ref = (admitted[-1] - admitted[0]) / 1e9
    low, high = rate_ratio_interval(n, seconds, n_ref, seconds_ref)
    needed = resolve_threshold(WORKLOAD_RATE_RATIO, context.thresholds)[0]
    return RateRatio(n, seconds, n_ref, seconds_ref, low, high, low >= needed)


def _foreign_row(context: Context, subject: Subject) -> dict[str, Any] | None:
    """The share of the steps' members that are not the run's requests, in
    the subject's span and in the reference's."""
    totals = [[0, 0], [0, 0]]
    for producer in _producers(context, subject):
        spans = arm_spans(context, subject, producer)
        for total, span in zip(totals, spans or ()):
            foreign, members = foreign_members(context, producer, span)
            total[0] += foreign
            total[1] += members
    if not totals[0][1] or not totals[1][1]:
        return None
    mine, theirs = (foreign / members for foreign, members in totals)
    rise = resolve_threshold(DRIVER_FOREIGN_RISE, context.thresholds)[0]
    return {
        "subject": round(mine, 4),
        "reference": round(theirs, 4),
        "passes": mine - theirs >= rise,
    }


def _producers(context: Context, subject: Subject) -> list[str]:
    return sorted({e.producer for e in context.subject_executions(subject)})


# ---------------------------------------------------------------- capacity
def _compatible(context: Context, subject: Subject) -> bool:
    """Whether the reference ran on the subject's engine epochs, or on
    epochs of the same config: without engine evidence, unknown."""
    mine = {e.producer for e in context.subject_executions(subject)}
    theirs = {e.producer for e in context.subject_executions(subject, reference=True)}
    if not mine or not theirs:
        return False
    configs = {_config(context, producer) for producer in mine | theirs}
    return len(configs) == 1 and None not in configs


def _config(context: Context, producer: str) -> str | None:
    epoch = context.epoch_of(producer)
    if epoch is None:
        return None
    return json.dumps(epoch.config, sort_keys=True, default=str)


def _capacity(context: Context, subject: Subject, floor: float) -> Ratio | None:
    """The subject's steps against the reference's on each of its engines;
    None without any."""
    design = Design.from_thresholds(context.thresholds)
    arms = [
        _arms(context, producer, spans)
        for producer in _producers(context, subject)
        if (spans := arm_spans(context, subject, producer)) is not None
    ]
    if not any(len(arm.subject) for arm in arms):
        return None
    needed = resolve_threshold(DRIVER_MIN_SUPPORT, context.thresholds)[0]
    both = ((False, design.controls), (True, design.treated))
    return compare(arms, Comparison(both, floor, needed), design)


def _arms(context: Context, producer: str, spans: tuple[Span, Span]) -> Arms:
    units = units_of(context, producer)
    return Arms(
        Columns.of(units.between(spans[0])), Columns.of(units.between(spans[1]))
    )


__all__ = ["CAPACITY", "LOAD", "UNDETERMINED", "Driver", "driver_of"]
