"""Roles: whether one finding stands behind another.

A class marks a competitor ``upstream`` when its evidence says the
mechanism may be another one's consequence. That is a claim about another
finding, so it is settled here, after every class has assessed every
subject. A cause is upstream only when the subject's finding of that kind
is eligible (an observation establishes nothing); otherwise the competitor
is merely not ruled out. When it is, the edge forms:

- the downstream finding becomes ``secondary``, lists the upstream's ID in
  ``secondary_to``, and keeps the evidence in ``detail.role_evidence``
  instead of the competitor (an upstream cause is never a competitor);
- the upstream finding claims what its consequence explains: KV pressure
  that held the queue explains the TTFT excess the queue does
  (``explains_ttft_excess_through_queue``), when the requests' time held
  behind preempted ones is itself that share of it.

This version has one edge, KV preemption pressure upstream of the queue;
the others come with the engine-loop class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .diagnosis_model import (
    NOT_RULED_OUT,
    RULED_OUT,
    SECONDARY,
    UPSTREAM,
    Alternative,
    Finding,
)
from .diagnosis_vocabulary import KV_PREEMPTION_PRESSURE, QUEUE_SATURATION

NOT_ESTABLISHED = "but no eligible {kind} finding establishes it"
THROUGH_QUEUE = "explains_ttft_excess_through_queue"
EXCLUDED = "competitors_excluded"
EDGES = frozenset({(KV_PREEMPTION_PRESSURE, QUEUE_SATURATION)})


def link_roles(findings: Sequence[Finding], run_id: str | None) -> None:
    """Settle every ``upstream`` competitor against the subject's findings."""
    by_subject: dict[str, dict[str, Finding]] = {}
    for finding in findings:
        key = str(finding.subject.get("key"))
        by_subject.setdefault(key, {}).setdefault(finding.kind, finding)
    for kinds in by_subject.values():
        for finding in kinds.values():
            _settle(finding, kinds, run_id)


def _settle(finding: Finding, kinds: dict[str, Finding], run_id: str | None) -> None:
    for alternative in [a for a in finding.alternatives if a.status == UPSTREAM]:
        upstream = kinds.get(alternative.kind)
        index = finding.alternatives.index(alternative)
        if upstream is None or not upstream.eligible:
            reason = f"{alternative.reason}, " + NOT_ESTABLISHED.format(
                kind=alternative.kind
            )
            finding.alternatives[index] = replace(
                alternative, status=NOT_RULED_OUT, reason=reason
            )
        elif (upstream.kind, finding.kind) in EDGES:
            del finding.alternatives[index]
            _link(upstream, finding, alternative, run_id)
            _recount_excluded(finding)


def _recount_excluded(finding: Finding) -> None:
    """An upstream cause is no competitor: with it gone, the competitors
    may all be ruled out after all."""
    contribution = finding.contribution
    if EXCLUDED in contribution.unmet and all(
        a.status == RULED_OUT for a in finding.alternatives
    ):
        finding.contribution = replace(
            contribution,
            met=(*contribution.met, EXCLUDED),
            unmet=tuple(c for c in contribution.unmet if c != EXCLUDED),
        )


def _link(
    upstream: Finding, finding: Finding, evidence: Alternative, run_id: str | None
) -> None:
    upstream_id = upstream.identity(run_id)
    finding.role = SECONDARY
    finding.secondary_to.append(upstream_id)
    finding.detail.setdefault("role_evidence", []).append(
        {
            "edge": f"{upstream.kind}->{finding.kind}",
            "upstream": upstream_id,
            "evidence": evidence.reason,
        }
    )
    hold = finding.detail.get("kv_hold") or {}
    if not hold.get("explains_ttft_excess"):
        return
    contribution = upstream.contribution
    upstream.contribution = replace(
        contribution, met=(*contribution.met, THROUGH_QUEUE)
    )
    if upstream.explains not in contribution.met:
        upstream.explains = THROUGH_QUEUE
    upstream.detail.setdefault("claims", []).append(
        {
            "kind": finding.kind,
            "id": finding.identity(run_id),
            "held_p50_ms": hold.get("held_p50_ms"),
        }
    )


__all__ = ["EDGES", "THROUGH_QUEUE", "link_roles"]
