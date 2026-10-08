"""What every diagnosis class reads: the joined run, its subjects, clocks,
decompositions and steps, each computed once and shared."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diagnosis_clocks import EngineClock
from .diagnosis_join import EngineEpoch, Execution, RunView
from .diagnosis_model import NOT_RULED_OUT, RULED_OUT, UNTESTABLE, Alternative
from .diagnosis_segments import Decomposition, decompose
from .diagnosis_selection import Selection, Subject
from .diagnosis_stats import Difference, median_difference
from .diagnosis_steps import Steps, steps_of
from .vllm_telemetry import VllmScrapeRecord

UNSUPPORTED = "unsupported"
PARTIAL = "partial"
ASSESSED = "assessed"
RESET_STAGES = ("engine.cache_reset", "engine.preempted_by_reset")


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
    _scrapes: list[VllmScrapeRecord] | None = None

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

    def subject_executions(
        self, subject: Subject, *, reference: bool = False
    ) -> list[Execution]:
        """The engine executions a subject (or its reference) stands for:
        the run's executions of its client requests, or, for a server-only
        subject, the executions themselves."""
        if subject.basis == "engine":
            refs = subject.reference_executions if reference else subject.executions
            return [self.view.executions[ref] for ref in refs]
        request_ids = subject.reference if reference else subject.requests
        return [e for r in request_ids for e in self.view.executions_of(r)]

    def engine_segments(self, execution: Execution) -> dict[str, int]:
        """The segments the engine's own clock measures exactly for one
        execution: ``engine_ingress`` and ``scheduler_wait`` (or, on a log
        without ``enqueued`` records, ``engine_ingress_to_schedule``),
        ``prefill``, and ``engine_ttft`` from admission to the first step
        that kept an output token."""
        admitted = execution.event.start_ns
        enqueued = execution.metadata.get("enqueued_mono_ns")
        first, retained = self._first_steps(execution)
        segments: dict[str, int] = {}
        if admitted is not None and first is not None:
            if isinstance(enqueued, int):
                segments["engine_ingress"] = enqueued - admitted
                segments["scheduler_wait"] = first - enqueued
            else:
                segments["engine_ingress_to_schedule"] = first - admitted
        if first is not None and retained is not None:
            segments["prefill"] = retained - first
        if admitted is not None and retained is not None:
            segments["engine_ttft"] = retained - admitted
        return segments

    def _first_steps(self, execution: Execution) -> tuple[int | None, int | None]:
        """The execution's first schedule() entry, and the completion of its
        first step that kept an output token."""
        first = retained = None
        for _, membership in execution.memberships:
            step = self.view.iterations.get(membership.iteration_ref)
            if step is None:
                continue
            if first is None:
                first = step[1].start_ns
            kept = membership.metadata.get("outcome") == "kept"
            if retained is None and kept and (membership.output_tokens or 0) > 0:
                retained = step[1].end_ns
        return first, retained

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

    def observes(self, producer: str, kind: str) -> bool:
        """Whether the engine's hook records ``kind`` at all."""
        epoch = self.epoch_of(producer)
        observes = epoch.observes if epoch is not None else None
        return observes is not None and kind in observes

    def lossless(self, producer: str, span: tuple[int, int]) -> bool:
        """Whether the hook's loss coverage spans [start, end] on the
        engine's monotonic clock: nothing it records can be missing there."""
        epoch = self.epoch_of(producer)
        spans = (epoch.coverage or {}).get("spans") or [] if epoch is not None else []
        return any(
            s.get("start_mono_ns", 0) <= span[0] and span[1] <= s.get("end_mono_ns", -1)
            for s in spans
        )

    def reset_absent(self, producer: str, span: tuple[int, int] | None) -> Alternative:
        """Whether a prefix-cache reset is ruled out over ``span``, as the
        indispensable competitor ``prefix_cache_reset``: only where the hook
        records resets, none happened, and nothing was lost."""
        kind = "prefix_cache_reset"
        if self._resets(producer, span):
            return Alternative(
                kind, NOT_RULED_OUT, "the prefix cache was reset over the subject", True
            )
        if not self.observes(producer, "cache_reset"):
            return Alternative(
                kind, UNTESTABLE, "the hook does not record cache resets", True
            )
        if span is None or not self.lossless(producer, span):
            return Alternative(
                kind, UNTESTABLE, "records may have been lost over the subject", True
            )
        return Alternative(
            kind,
            RULED_OUT,
            "no reset, and the hook records them with nothing lost",
            True,
        )

    def _resets(self, producer: str, span: tuple[int, int] | None) -> bool:
        """Whether a reset overlaps ``span``: a stage, or a dated fact the
        import kept for a reset before any step was written."""
        intervals = [
            (stage.start_ns or 0, stage.end_ns or stage.start_ns or 0)
            for _, stage in self.view.stages
            if stage.name in RESET_STAGES and stage.stage_ref.producer_id == producer
        ] + self._unanchored_resets(producer)
        return any(
            span is None or (start <= span[1] and span[0] <= end)
            for start, end in intervals
        )

    def _unanchored_resets(self, producer: str) -> list[tuple[int, int]]:
        intervals = []
        for epoch in self.view.engines.values():
            facts = epoch.unanchored if epoch.producer == producer else []
            for fact in facts:
                start, end = fact.get("start_mono_ns"), fact.get("end_mono_ns")
                if fact.get("name") in RESET_STAGES and isinstance(start, int):
                    intervals.append((start, end if isinstance(end, int) else start))
        return intervals

    def subjects(self) -> list[Subject]:
        return self.selection.subjects

    def scrapes(
        self, start_ns: int | None = None, end_ns: int | None = None
    ) -> list[VllmScrapeRecord]:
        """The run's vLLM metric scrapes in time order, those observed in
        [start, end] when given; a record that does not parse is skipped."""
        if self._scrapes is None:
            parsed = []
            for line in self.view.scrapes:
                try:
                    parsed.append(VllmScrapeRecord.from_record(dict(line.raw or {})))
                except (KeyError, TypeError, ValueError):
                    continue
            self._scrapes = sorted(parsed, key=lambda r: r.observed_at_ns)
        return [
            s
            for s in self._scrapes
            if (start_ns is None or s.observed_at_ns >= start_ns)
            and (end_ns is None or s.observed_at_ns <= end_ns)
        ]

    def window(self, subject: Subject) -> dict[str, Any] | None:
        """A finding's window: its subject's, on the artifact's clock, from
        an incident's onset; its resolution is how finely selection placed
        that onset (the first flagged window's span), else one base window."""
        if subject.start_ns is None or subject.end_ns is None:
            return None
        onset = subject.onset_ns
        return {
            "start_ns": subject.start_ns if onset is None else onset,
            "end_ns": subject.end_ns,
            "clock_domain": self.view.clock_domain,
            "uncertainty_ns": 0,
            "resolution_ns": subject.resolution_ns or 1_000_000_000,
            "placement": "client_clock",
        }


__all__ = ["ASSESSED", "PARTIAL", "UNSUPPORTED", "Assessment", "Context"]
