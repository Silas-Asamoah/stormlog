"""What every diagnosis class reads: the joined run, its subjects, clocks,
decompositions and steps, each computed once and shared."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diagnosis_clocks import EngineClock
from .diagnosis_join import EngineEpoch, RunView
from .diagnosis_segments import Decomposition, decompose
from .diagnosis_selection import Selection, Subject
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


__all__ = ["ASSESSED", "PARTIAL", "UNSUPPORTED", "Assessment", "Context"]
