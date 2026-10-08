"""What vLLM's own metrics can say about a subject, and how little.

Scrapes describe the whole engine between two instants, never a request, so
on their own they give a window-level observation: ``partial`` with the
reason ``aggregate_only``, a finding that can never be a fault claim. A
scraped exporter is bound to the engine whose hook log was imported only
when the operator asserts it (``--metrics-from-engine``); otherwise it is
``exporter_scoped``, and its metrics never stand in for hook evidence.
"""

from __future__ import annotations

from .diagnosis_context import PARTIAL, Assessment, Context
from .diagnosis_model import Finding, Observation, met
from .diagnosis_selection import Subject
from .diagnosis_signals import SignalConfig, SignalValue, evaluate_signal
from .diagnosis_vocabulary import KIND_COMPONENTS

AGGREGATE_ONLY = "aggregate_only"
ASSERTED = "asserted"
EXPORTER_SCOPED = "exporter_scoped"


def binding(context: Context) -> str:
    return ASSERTED if context.metrics_from_engine else EXPORTER_SCOPED


def subject_signal(context: Context, subject: Subject, kind: str) -> SignalValue | None:
    """A kind's online signal over the scrapes inside the subject's window."""
    if subject.start_ns is None or subject.end_ns is None:
        return None
    scrapes = context.scrapes(subject.start_ns, subject.end_ns)
    if len(scrapes) < 2:
        return None
    return evaluate_signal(
        kind, scrapes, SignalConfig(thresholds=dict(context.thresholds or {}))
    )


def aggregate_assessment(
    context: Context, subject: Subject, kind: str, statement: str, reason: str
) -> Assessment | None:
    """A window-level observation from the scrapes, or None when they say
    nothing: insufficient, or not over the threshold."""
    signal = subject_signal(context, subject, kind)
    if signal is None or not signal.sufficient or not signal.exceeds:
        return None
    component = sorted(KIND_COMPONENTS[kind])[0]
    finding = Finding(
        kind=kind,
        component=component,
        subject=subject.as_dict(),
        title=f"vLLM's metrics show {kind.replace('_', ' ')} over the window",
        message=statement.format(value=signal.value),
        status=PARTIAL,
        gates={"hook_evidence": False},
        condition=met(direct_evidence=False, sufficient_samples=True),
        contribution=met(excess_ci_excludes_zero=False),
        observations=[
            Observation(
                "o1",
                statement.format(value=signal.value),
                f"{kind}_signal",
                signal.value,
                provenance="reported",
            )
        ],
        location={"component": component, "exporter_binding": binding(context)},
        window=context.window(subject),
        incident=subject.incident,
        detail={"signal": dict(signal.detail), "threshold": signal.threshold},
        partial_reasons=[AGGREGATE_ONLY, reason],
    )
    lines = [
        line
        for line in context.view.scrapes
        if subject.start_ns is not None
        and subject.end_ns is not None
        and subject.start_ns
        <= (line.raw or {}).get("observed_at_ns", 0)
        <= subject.end_ns
    ]
    finding.support, finding.display = lines, lines[:8]
    return Assessment(kind, subject.key, PARTIAL, [AGGREGATE_ONLY, reason], [finding])


__all__ = ["AGGREGATE_ONLY", "aggregate_assessment", "binding", "subject_signal"]
