"""Roles: whether one finding stands behind another.

A class marks a competitor ``upstream`` when its evidence says the
mechanism may be another one's consequence, naming the cause as ``kind``
or ``kind@component``. That is a claim about another finding, so it is
settled here, after every class has assessed every subject, one downstream
kind at a time in the edge table's order (``diagnosis_edges``), so an
upstream's eligibility is final before anything reads it. A cause is
upstream only when the subject's finding of that kind is eligible (an
observation establishes nothing); otherwise the competitor is merely not
ruled out, and stays indispensable if it was. When an edge of the table
joins the two, it forms:

- the downstream finding becomes ``secondary``, lists the upstream's ID in
  ``secondary_to``, and keeps the evidence in ``detail.role_evidence``
  instead of the competitor (an upstream cause is never a competitor). Its
  severity is capped at the upstream's and its cause is the upstream's,
  through any chain, so an edge never raises the exit code above what the
  cause says, nor calls a capture's consequence a fault;
- the upstream finding claims what its consequence explains, when the
  downstream class recorded that it does (``detail.edge_claims``): KV
  pressure that held the queue explains the TTFT excess the queue does
  (``explains_ttft_excess_through_queue``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .diagnosis_edges import edge_between, settle_order
from .diagnosis_model import (
    NOT_RULED_OUT,
    RULED_OUT,
    SECONDARY,
    UPSTREAM,
    Alternative,
    Finding,
)

NOT_ESTABLISHED = "but no eligible {kind} finding establishes it"
EXCLUDED = "competitors_excluded"

Located = dict[tuple[str, str], Finding]


def link_roles(findings: Sequence[Finding], run_id: str | None) -> None:
    """Settle every ``upstream`` competitor against the subject's findings."""
    by_subject: dict[str, Located] = {}
    for finding in findings:
        key = str(finding.subject.get("key"))
        by_subject.setdefault(key, {}).setdefault(_location(finding), finding)
    order = settle_order()
    for located in by_subject.values():
        ordered = sorted(
            located.values(),
            key=lambda f: order.index(f.kind) if f.kind in order else len(order),
        )
        for finding in ordered:
            _settle(finding, located, run_id)


def _location(finding: Finding) -> tuple[str, str]:
    return finding.kind, finding.component


def _settle(finding: Finding, located: Located, run_id: str | None) -> None:
    for alternative in [a for a in finding.alternatives if a.status == UPSTREAM]:
        upstream = _upstream(alternative.kind, finding, located)
        index = finding.alternatives.index(alternative)
        if upstream is None or not upstream.eligible:
            reason = f"{alternative.reason}, " + NOT_ESTABLISHED.format(
                kind=alternative.kind
            )
            finding.alternatives[index] = replace(
                alternative, status=NOT_RULED_OUT, reason=reason
            )
            continue
        edge = edge_between(_location(upstream), _location(finding))
        if edge is not None:
            del finding.alternatives[index]
            _link(upstream, finding, alternative, edge.name, run_id)
            _recount_excluded(finding)


def _upstream(named: str, finding: Finding, located: Located) -> Finding | None:
    """The subject's finding a competitor names, as ``kind`` or
    ``kind@component``; for a bare kind, the one an edge joins to
    ``finding``."""
    kind, _, component = named.partition("@")
    if component:
        return located.get((kind, component))
    candidates = [f for (k, _), f in sorted(located.items()) if k == kind]
    joined = [f for f in candidates if edge_between(_location(f), _location(finding))]
    found = joined or candidates
    return found[0] if found else None


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
    upstream: Finding,
    finding: Finding,
    evidence: Alternative,
    edge: str,
    run_id: str | None,
) -> None:
    upstream_id = upstream.identity(run_id)
    finding.role = SECONDARY
    finding.secondary_to.append(upstream_id)
    finding.upstreams.append(upstream)
    finding.detail.setdefault("role_evidence", []).append(
        {"edge": edge, "upstream": upstream_id, "evidence": evidence.reason}
    )
    claim: dict[str, Any] = (finding.detail.get("edge_claims") or {}).get(edge) or {}
    if claim.get("met") and claim.get("criterion"):
        _claim(upstream, str(claim["criterion"]), finding, claim, run_id)


def _claim(
    upstream: Finding,
    criterion: str,
    finding: Finding,
    claim: dict[str, Any],
    run_id: str | None,
) -> None:
    """The upstream explains, through its consequence, what that explains."""
    contribution = upstream.contribution
    if criterion not in contribution.met:
        upstream.contribution = replace(
            contribution, met=(*contribution.met, criterion)
        )
    if upstream.explains not in contribution.met:
        upstream.explains = criterion
    extra = {k: v for k, v in claim.items() if k not in ("criterion", "met")}
    upstream.detail.setdefault("claims", []).append(
        {"kind": finding.kind, "id": finding.identity(run_id), **extra}
    )


__all__ = ["link_roles"]
