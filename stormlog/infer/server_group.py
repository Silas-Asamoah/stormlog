"""Declared groups of server processes, such as tensor-parallel GPU workers.

Each collector in a group passes the same ``--group-id`` and ``--world-size``
and its own ``--rank``. Membership is declared, never inferred: an identity
without the group ID, a second identity for one rank (a restarted worker or a
changed GPU), or a missing rank keeps the join from happening.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .telemetry import ServerIdentity, TelemetrySample


def membership_issue(samples: Iterable[TelemetrySample]) -> str | None:
    """Return why the samples are neither one server nor one complete group."""
    identities = {sample.identity for sample in samples}
    group_ids = {identity.group_id for identity in identities}
    if group_ids == {None}:
        return None if len(identities) == 1 else "multiple_server_identities"
    if None in group_ids:
        return "undeclared_server_identity"
    if len(group_ids) > 1:
        return "multiple_server_groups"
    return _rank_issue(identities)


def _rank_issue(identities: set[ServerIdentity]) -> str | None:
    world_sizes = {identity.world_size for identity in identities}
    if len(world_sizes) != 1:
        return "inconsistent_group_size"
    ranks = Counter(identity.rank for identity in identities)
    if any(count > 1 for count in ranks.values()):
        # One rank seen under two identities: a restart or a different GPU.
        return "group_member_changed"
    # ServerIdentity keeps every rank below world_size, so only gaps remain.
    if len(ranks) < (world_sizes.pop() or 0):
        return "group_member_missing"
    return None


def members(
    samples: Iterable[TelemetrySample],
) -> dict[ServerIdentity, list[TelemetrySample]]:
    """Group samples by identity, ordered by rank for a declared group."""
    grouped: dict[ServerIdentity, list[TelemetrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.identity, []).append(sample)
    ordered = sorted(grouped, key=lambda identity: identity.rank or 0)
    return {identity: grouped[identity] for identity in ordered}
