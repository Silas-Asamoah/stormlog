"""Roles: whether one finding stands behind another.

A class marks a competitor ``upstream`` when its evidence says the
mechanism may be another one's consequence. That is a claim about another
finding, so it is settled here, after every class has assessed every
subject: a cause is upstream only when the subject's finding of that kind
is eligible (an observation establishes nothing), otherwise the competitor
is merely not ruled out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .diagnosis_model import NOT_RULED_OUT, UPSTREAM, Finding

NOT_ESTABLISHED = "but no eligible {kind} finding establishes it"


def link_roles(findings: Sequence[Finding]) -> None:
    """Settle every ``upstream`` competitor against the subject's findings."""
    by_subject: dict[str, dict[str, Finding]] = {}
    for finding in findings:
        key = str(finding.subject.get("key"))
        by_subject.setdefault(key, {}).setdefault(finding.kind, finding)
    for kinds in by_subject.values():
        for finding in kinds.values():
            _settle(finding, kinds)


def _settle(finding: Finding, kinds: dict[str, Finding]) -> None:
    for index, alternative in enumerate(finding.alternatives):
        if alternative.status != UPSTREAM:
            continue
        upstream = kinds.get(alternative.kind)
        if upstream is None or not upstream.eligible:
            reason = f"{alternative.reason}, " + NOT_ESTABLISHED.format(
                kind=alternative.kind
            )
            finding.alternatives[index] = replace(
                alternative, status=NOT_RULED_OUT, reason=reason
            )


__all__ = ["link_roles"]
