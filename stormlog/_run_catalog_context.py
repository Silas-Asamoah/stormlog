"""Internal run synthesis, identity policy, and catalog filtering."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from ._run_catalog_models import (
    CatalogRunEnvelope,
    RunAttachmentFilter,
    RunAttachmentRow,
    RunContext,
    RunFilter,
    RunIdentityConflict,
    RunIdentityIndex,
    RunRow,
    SessionRowLike,
)


def build_run_contexts(
    sessions: Sequence[SessionRowLike],
    envelopes: Sequence[CatalogRunEnvelope],
) -> dict[str, RunContext]:
    """Build explicit contexts plus implicit contexts for uncovered sessions."""
    contexts = _explicit_run_contexts(sessions, envelopes)
    covered_session_ids = {
        session.session_id
        for context in contexts.values()
        for session in context.sessions
    }
    uncovered_sessions = [
        session for session in sessions if session.session_id not in covered_session_ids
    ]
    contexts.update(
        _implicit_run_contexts(
            uncovered_sessions,
            existing_run_ids=set(contexts),
        )
    )
    return contexts


def build_identity_index(
    contexts: Mapping[str, RunContext],
) -> RunIdentityIndex:
    """Build unambiguous identity maps for attachment projection."""
    session_candidates: dict[str, set[str]] = defaultdict(set)
    job_candidates: dict[str, set[str]] = defaultdict(set)
    source_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)

    for context in contexts.values():
        if context.job_id is not None:
            job_candidates[context.job_id].add(context.run_id)
        if context.source_namespace is not None and context.source_ref is not None:
            source_candidates[(context.source_namespace, context.source_ref)].add(
                context.run_id
            )
        for session in context.sessions:
            session_candidates[session.session_id].add(context.run_id)
            if session.job_id is not None:
                job_candidates[session.job_id].add(context.run_id)

    conflicts: list[RunIdentityConflict] = []
    session_to_run = _unique_identity_map("session_id", session_candidates, conflicts)
    job_to_run = _unique_identity_map("job_id", job_candidates, conflicts)
    source_ref_to_run = _unique_source_ref_map(source_candidates, conflicts)
    return RunIdentityIndex(
        session_to_run=session_to_run,
        job_to_run=job_to_run,
        source_ref_to_run=source_ref_to_run,
        conflicts=tuple(conflicts),
    )


def run_id_for_identity(
    *,
    run_id: str | None,
    session_id: str | None,
    job_id: str | None,
    source_namespace: str | None,
    source_ref: str | None,
    identity_index: RunIdentityIndex,
) -> str | None:
    """Resolve an attachment run id without using ambiguous identities."""
    if run_id is not None:
        return run_id
    if source_namespace is not None and source_ref is not None:
        resolved = identity_index.source_ref_to_run.get((source_namespace, source_ref))
        if resolved is not None:
            return resolved
    if session_id is not None:
        resolved = identity_index.session_to_run.get(session_id)
        if resolved is not None:
            return resolved
    if job_id is not None:
        return identity_index.job_to_run.get(job_id)
    return None


def run_matches(row: RunRow, filters: RunFilter) -> bool:
    """Return whether a run row satisfies filters."""
    if filters.run_id is not None and row.run_id != filters.run_id:
        return False
    if filters.session_id is not None and filters.session_id not in row.sessions:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    if filters.rank is not None and filters.rank not in row.ranks:
        return False
    return _source_matches(row, filters)


def _source_matches(
    row: RunRow | RunAttachmentRow,
    filters: RunFilter | RunAttachmentFilter,
) -> bool:
    if (
        filters.source_namespace is not None
        and row.source_namespace != filters.source_namespace
    ):
        return False
    if filters.source_ref is not None and row.source_ref != filters.source_ref:
        return False
    return True


def run_attachment_matches(
    row: RunAttachmentRow,
    filters: RunAttachmentFilter,
) -> bool:
    """Return whether an attachment row satisfies filters."""
    if filters.run_id is not None and row.run_id != filters.run_id:
        return False
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    if filters.rank is not None and row.rank != filters.rank:
        return False
    return _attachment_source_matches(row, filters)


def _attachment_source_matches(
    row: RunAttachmentRow, filters: RunAttachmentFilter
) -> bool:
    if filters.kind is not None and row.kind != filters.kind:
        return False
    return _source_matches(row, filters)


def _explicit_run_contexts(
    sessions: Sequence[SessionRowLike],
    envelopes: Sequence[CatalogRunEnvelope],
) -> dict[str, RunContext]:
    session_by_id = _first_session_by_id(sessions)
    contexts: dict[str, RunContext] = {}
    for envelope in envelopes:
        members = _envelope_members(envelope, sessions, session_by_id)
        contexts[envelope.run_id] = RunContext(
            run_id=envelope.run_id,
            explicit=True,
            title=envelope.title,
            description=envelope.description,
            job_id=envelope.job_id or _common_job_id(members),
            started_at_ns=(
                envelope.started_at_ns
                if envelope.started_at_ns is not None
                else _min_started_at_ns(members)
            ),
            ended_at_ns=(
                envelope.ended_at_ns
                if envelope.ended_at_ns is not None
                else _max_ended_at_ns(members)
            ),
            source_path=str(envelope.path),
            source_kind="run_envelope",
            source_namespace=envelope.source_namespace,
            source_ref=envelope.source_ref,
            sessions=tuple(members),
            tags=envelope.tags,
            metadata=envelope.metadata,
        )
    return contexts


def _envelope_members(
    envelope: CatalogRunEnvelope,
    sessions: Sequence[SessionRowLike],
    session_by_id: Mapping[str, SessionRowLike],
) -> list[SessionRowLike]:
    member_ids = {session.session_id for session in envelope.sessions}
    members = [
        session_by_id[session_id]
        for session_id in member_ids
        if session_id in session_by_id
    ]
    if not members and envelope.job_id is not None:
        members = [session for session in sessions if session.job_id == envelope.job_id]
    members.sort(key=lambda session: (session.started_at_ns, session.session_id))
    return members


def _implicit_run_contexts(
    sessions: Sequence[SessionRowLike],
    *,
    existing_run_ids: set[str],
) -> dict[str, RunContext]:
    grouped: dict[str, list[SessionRowLike]] = defaultdict(list)
    for session in sessions:
        if session.job_id is not None:
            grouped[f"job:{session.job_id}"].append(session)
        else:
            grouped[f"session:{session.session_id}"].append(session)

    contexts: dict[str, RunContext] = {}
    used_run_ids = set(existing_run_ids)
    for base_run_id, members in grouped.items():
        run_id = _unique_run_id(base_run_id, used_run_ids)
        used_run_ids.add(run_id)
        members.sort(key=lambda session: (session.started_at_ns, session.session_id))
        job_id = _common_job_id(members)
        contexts[run_id] = RunContext(
            run_id=run_id,
            explicit=False,
            title=(
                f"Distributed job {job_id}"
                if job_id is not None
                else f"Session {members[0].session_id}"
            ),
            description=None,
            job_id=job_id,
            started_at_ns=_min_started_at_ns(members),
            ended_at_ns=_max_ended_at_ns(members),
            source_path=members[0].source_path if members else "",
            source_kind="implicit_run",
            source_namespace=None,
            source_ref=None,
            sessions=tuple(members),
        )
    return contexts


def _unique_identity_map(
    identity_kind: str,
    candidates: Mapping[str, set[str]],
    conflicts: list[RunIdentityConflict],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for identity_value, run_ids in candidates.items():
        if len(run_ids) == 1:
            resolved[identity_value] = next(iter(run_ids))
            continue
        conflicts.append(
            RunIdentityConflict(
                identity_kind=identity_kind,
                identity_value=identity_value,
                run_ids=tuple(sorted(run_ids)),
            )
        )
    return resolved


def _unique_source_ref_map(
    candidates: Mapping[tuple[str, str], set[str]],
    conflicts: list[RunIdentityConflict],
) -> dict[tuple[str, str], str]:
    resolved: dict[tuple[str, str], str] = {}
    for source_ref, run_ids in candidates.items():
        if len(run_ids) == 1:
            resolved[source_ref] = next(iter(run_ids))
            continue
        conflicts.append(
            RunIdentityConflict(
                identity_kind="source_ref",
                identity_value=f"{source_ref[0]}:{source_ref[1]}",
                run_ids=tuple(sorted(run_ids)),
            )
        )
    return resolved


def _unique_run_id(base_run_id: str, used_run_ids: set[str]) -> str:
    if base_run_id not in used_run_ids:
        return base_run_id
    candidate = f"implicit:{base_run_id}"
    if candidate not in used_run_ids:
        return candidate
    suffix = 2
    while f"{candidate}:{suffix}" in used_run_ids:
        suffix += 1
    return f"{candidate}:{suffix}"


def _first_session_by_id(
    sessions: Sequence[SessionRowLike],
) -> dict[str, SessionRowLike]:
    rows: dict[str, SessionRowLike] = {}
    for session in sessions:
        rows.setdefault(session.session_id, session)
    return rows


def _common_job_id(sessions: Sequence[SessionRowLike]) -> str | None:
    job_ids = {session.job_id for session in sessions if session.job_id is not None}
    return next(iter(job_ids)) if len(job_ids) == 1 else None


def _min_started_at_ns(sessions: Sequence[SessionRowLike]) -> int | None:
    if not sessions:
        return None
    return min(session.started_at_ns for session in sessions)


def _max_ended_at_ns(sessions: Sequence[SessionRowLike]) -> int | None:
    if not sessions or any(session.ended_at_ns is None for session in sessions):
        return None
    return max(cast(int, session.ended_at_ns) for session in sessions)


__all__ = [
    "build_identity_index",
    "build_run_contexts",
    "run_attachment_matches",
    "run_id_for_identity",
    "run_matches",
]
