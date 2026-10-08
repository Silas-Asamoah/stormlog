"""What every diagnosis class reads: the joined run, its subjects, clocks,
decompositions and steps, each computed once and shared."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diagnosis_clocks import EngineClock
from .diagnosis_join import EngineEpoch, RunView
from .diagnosis_segments import Decomposition, decompose
from .diagnosis_selection import Selection, Subject
from .diagnosis_stats import Difference, median_difference
from .diagnosis_steps import Steps, steps_of

UNSUPPORTED = "unsupported"
PARTIAL = "partial"
ASSESSED = "assessed"


@dataclass
class Assessment:
    """One class's verdict on one subject: assessed (with a finding or
    not_observed), partial, or unsupported, with reasons."""

    kind: str
    subject_key: str
    status: str
    reasons: list[str] = field(default_factory=list)
    findings: list[Any] = field(default_factory=list)  # diagnosis_model.Finding

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "findings": len(self.findings),
        }


@dataclass
class Context:
    """Shared, lazily computed views of one run."""

    view: RunView
    selection: Selection
    thresholds: dict[str, float] | None = None
    metrics_from_engine: bool = False
    _clocks: dict[str, EngineClock] = field(default_factory=dict)
    _decompositions: dict[str, tuple[Decomposition, Decomposition]] = field(
        default_factory=dict
    )
    _steps: dict[str, Steps] = field(default_factory=dict)

    def clock(self, producer: str) -> EngineClock:
        if producer not in self._clocks:
            self._clocks[producer] = EngineClock(self.view, producer)
        return self._clocks[producer]

    def decomposition(self, request_id: str) -> tuple[Decomposition, Decomposition]:
        """The request's TTFT and end-to-end decompositions."""
        if request_id not in self._decompositions:
            request = self.view.client[request_id]
            producers = {e.producer for e in self.view.executions_of(request_id)}
            clocks = {producer: self.clock(producer) for producer in producers}
            self._decompositions[request_id] = decompose(self.view, request, clocks)
        return self._decompositions[request_id]

    def segment_values(self, request_ids: list[str], name: str) -> list[float]:
        """A segment's duration for each request that has it placed: the
        middle of its interval (exact on the engine clock, within its
        bracket across clocks)."""
        values = []
        for request_id in request_ids:
            ttft, e2e = self.decomposition(request_id)
            part = ttft.part(name) or e2e.part(name)
            if part is not None and part.interval is not None:
                values.append(sum(part.interval) / 2)
        return values

    def segment_excess(self, subject: Subject, name: str) -> Difference | None:
        """The subject's median segment minus the reference's."""
        return median_difference(
            self.segment_values(subject.requests, name),
            self.segment_values(subject.reference, name),
        )

    def total_excess(self, subject: Subject, kind: str) -> Difference | None:
        """The subject's median TTFT (``kind`` ttft) or end-to-end latency
        minus the reference's."""
        index = 0 if kind == "ttft" else 1
        arms = []
        for request_ids in (subject.requests, subject.reference):
            totals = (self.decomposition(r)[index].total_ns for r in request_ids)
            arms.append([float(t) for t in totals if t is not None])
        return median_difference(arms[0], arms[1])

    def producer_of(self, request_ids: list[str]) -> str | None:
        """The one engine that served these requests; None for none or
        several."""
        producers = {
            execution.producer
            for request_id in request_ids
            for execution in self.view.executions_of(request_id)
        }
        return next(iter(producers)) if len(producers) == 1 else None

    def steps(self, producer: str) -> Steps:
        if producer not in self._steps:
            self._steps[producer] = steps_of(self.view, producer)
        return self._steps[producer]

    def epoch_of(self, producer: str) -> EngineEpoch | None:
        return next(
            (e for e in self.view.engines.values() if e.producer == producer), None
        )

    def subjects(self) -> list[Subject]:
        return self.selection.subjects

    def window(self, subject: Subject) -> dict[str, Any] | None:
        """A finding's window: its subject's, on the artifact's clock."""
        if subject.start_ns is None or subject.end_ns is None:
            return None
        return {
            "start_ns": subject.start_ns,
            "end_ns": subject.end_ns,
            "clock_domain": self.view.clock_domain,
            "uncertainty_ns": 0,
            "resolution_ns": 1_000_000_000,
            "placement": "client_clock",
        }


__all__ = ["ASSESSED", "PARTIAL", "UNSUPPORTED", "Assessment", "Context"]
