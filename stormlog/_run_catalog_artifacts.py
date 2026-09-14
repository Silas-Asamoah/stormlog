"""Internal projection of artifacts into run attachment query rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._datetime import datetime_to_ns as _datetime_to_ns
from ._datetime import parse_datetime as _parse_datetime
from ._run_catalog_context import build_identity_index, run_id_for_identity
from ._run_catalog_models import (
    CatalogRunEnvelope,
    CatalogSourceLike,
    OOMBundleLike,
    RunAttachmentRow,
    RunContext,
    RunIdentityIndex,
    SessionRowLike,
)
from ._run_catalog_parser import attachment_storage_or_default
from .correlation import ExternalAttachment
from .session import SessionSummary
from .telemetry_rollups import ROLLUP_FILENAME
from .telemetry_sink import TelemetrySinkManifest, TelemetrySinkSegment


def envelope_attachment_rows(
    envelopes: Sequence[CatalogRunEnvelope],
    contexts: Mapping[str, RunContext],
) -> list[RunAttachmentRow]:
    """Project explicit envelope attachments into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    for envelope in envelopes:
        if envelope.run_id not in contexts:
            continue
        for attachment in envelope.attachments:
            run_id = attachment.run_id or envelope.run_id
            rows.append(
                RunAttachmentRow(
                    run_id=run_id,
                    title=attachment.title,
                    kind=attachment.kind,
                    storage=attachment.storage,
                    attachment_id=attachment.attachment_id,
                    url=attachment.url,
                    path=attachment.path,
                    session_id=attachment.session_id,
                    job_id=attachment.job_id or envelope.job_id,
                    rank=attachment.rank,
                    local_rank=attachment.local_rank,
                    world_size=attachment.world_size,
                    start_ns=attachment.start_ns,
                    end_ns=attachment.end_ns,
                    source_path=attachment.url or attachment.path or str(envelope.path),
                    source_kind="run_envelope_attachment",
                    source_namespace=attachment.source_namespace
                    or envelope.source_namespace,
                    source_ref=attachment.source_ref or envelope.source_ref,
                    metadata=attachment.metadata,
                )
            )
    return rows


def sidecar_attachment_rows(
    attachments: Sequence[ExternalAttachment],
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project external sidecar attachments into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    for attachment in attachments:
        run_id = run_id_for_identity(
            run_id=attachment.run_id,
            session_id=attachment.session_id,
            job_id=attachment.job_id,
            source_namespace=attachment.source_namespace,
            source_ref=attachment.source_ref,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            RunAttachmentRow(
                run_id=run_id,
                title=attachment.title,
                kind=attachment.kind,
                storage=attachment_storage_or_default(attachment.storage),
                attachment_id=attachment.attachment_id,
                url=attachment.url,
                path=attachment.path,
                session_id=attachment.session_id,
                job_id=attachment.job_id,
                rank=attachment.rank,
                local_rank=None,
                world_size=None,
                start_ns=attachment.start_ns,
                end_ns=attachment.end_ns,
                source_path=attachment.url
                or attachment.path
                or attachment.sidecar_path,
                source_kind="attachment_sidecar",
                source_namespace=attachment.source_namespace,
                source_ref=attachment.source_ref,
                metadata=attachment.metadata,
            )
        )
    return rows


def sink_attachment_rows(
    source: CatalogSourceLike,
    manifest: TelemetrySinkManifest,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project sink directory and segment rows for all mapped sessions."""
    rows: list[RunAttachmentRow] = []
    for summary in manifest.sessions:
        run_id = run_id_for_identity(
            run_id=None,
            session_id=summary.session_id,
            job_id=summary.job_id,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title="Telemetry sink",
                kind="telemetry_sink",
                path=str(source.path),
                session_id=summary.session_id,
                job_id=summary.job_id,
                rank=summary.rank,
                local_rank=summary.local_rank,
                world_size=summary.world_size,
                start_ns=summary.started_at_ns,
                end_ns=summary.ended_at_ns,
                source_kind="sink",
                metadata={
                    "manifest_path": (
                        str(source.manifest_path)
                        if source.manifest_path is not None
                        else None
                    ),
                },
            )
        )
    rows.extend(sink_segment_attachment_rows(source, manifest, identity_index))
    if (source.path / ROLLUP_FILENAME).exists():
        rows.extend(rollup_attachment_rows(source, manifest, identity_index))
    return rows


def sink_segment_attachment_rows(
    source: CatalogSourceLike,
    manifest: TelemetrySinkManifest,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project sink segment files into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    session_by_id = {summary.session_id: summary for summary in manifest.sessions}
    for segment in manifest.segments:
        if segment.session_id is None:
            continue
        summary = session_by_id.get(segment.session_id)
        run_id = run_id_for_identity(
            run_id=None,
            session_id=segment.session_id,
            job_id=summary.job_id if summary is not None else None,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(_sink_segment_attachment_row(source, segment, summary, run_id))
    return rows


def _sink_segment_attachment_row(
    source: CatalogSourceLike,
    segment: TelemetrySinkSegment,
    summary: SessionSummary | None,
    run_id: str,
) -> RunAttachmentRow:
    return local_attachment_row(
        run_id=run_id,
        title=segment.filename,
        kind="telemetry_sink_segment",
        path=str(source.path / segment.filename),
        session_id=segment.session_id,
        job_id=summary.job_id if summary is not None else None,
        rank=summary.rank if summary is not None else None,
        local_rank=summary.local_rank if summary is not None else None,
        world_size=summary.world_size if summary is not None else None,
        start_ns=summary.started_at_ns if summary is not None else None,
        end_ns=summary.ended_at_ns if summary is not None else None,
        source_kind="sink_segment",
        metadata={
            "event_count": segment.event_count,
            "size_bytes": segment.size_bytes,
            "closed": segment.closed,
        },
    )


def rollup_attachment_rows(
    source: CatalogSourceLike,
    manifest: TelemetrySinkManifest,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project sink rollup sidecars into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    rollup_path = source.path / ROLLUP_FILENAME
    for summary in manifest.sessions:
        run_id = run_id_for_identity(
            run_id=None,
            session_id=summary.session_id,
            job_id=summary.job_id,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title="Telemetry rollups",
                kind="telemetry_rollup",
                path=str(rollup_path),
                session_id=summary.session_id,
                job_id=summary.job_id,
                rank=summary.rank,
                local_rank=summary.local_rank,
                world_size=summary.world_size,
                start_ns=summary.started_at_ns,
                end_ns=summary.ended_at_ns,
                source_kind="rollup",
                metadata={
                    "manifest_session_count": len(manifest.sessions),
                    "manifest_segment_count": len(manifest.segments),
                },
            )
        )
    return rows


def diagnose_attachment_rows(
    source: CatalogSourceLike,
    summary: SessionSummary | None,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project a diagnose bundle into a run attachment row."""
    if summary is None:
        return []
    run_id = run_id_for_identity(
        run_id=None,
        session_id=summary.session_id,
        job_id=summary.job_id,
        source_namespace=None,
        source_ref=None,
        identity_index=identity_index,
    )
    if run_id is None:
        return []
    return [
        local_attachment_row(
            run_id=run_id,
            title="Diagnose bundle",
            kind="diagnose_bundle",
            path=str(source.path),
            session_id=summary.session_id,
            job_id=summary.job_id,
            rank=summary.rank,
            local_rank=summary.local_rank,
            world_size=summary.world_size,
            start_ns=summary.started_at_ns,
            end_ns=summary.ended_at_ns,
            source_kind="diagnose_bundle",
            metadata={
                "manifest_path": (
                    str(source.manifest_path)
                    if source.manifest_path is not None
                    else None
                ),
                "session_status": summary.status,
            },
        )
    ]


def flat_telemetry_attachment_rows(
    source: CatalogSourceLike,
    contexts: Mapping[str, RunContext],
) -> list[RunAttachmentRow]:
    """Project flat telemetry files into run attachment rows."""
    identity_index = build_identity_index(contexts)
    rows: list[RunAttachmentRow] = []
    for context in contexts.values():
        for session in context.sessions:
            if session.source_path != str(source.path):
                continue
            run_id = run_id_for_identity(
                run_id=None,
                session_id=session.session_id,
                job_id=session.job_id,
                source_namespace=None,
                source_ref=None,
                identity_index=identity_index,
            )
            if run_id is None:
                continue
            rows.append(
                local_attachment_row(
                    run_id=run_id,
                    title=Path(session.source_path).name or "Telemetry file",
                    kind="telemetry_file",
                    path=session.source_path,
                    session_id=session.session_id,
                    job_id=session.job_id,
                    rank=session.rank,
                    local_rank=session.local_rank,
                    world_size=session.world_size,
                    start_ns=session.started_at_ns,
                    end_ns=session.ended_at_ns,
                    source_kind=session.source_kind,
                    metadata={"event_count": session.event_count},
                )
            )
    return rows


def oom_attachment_rows(
    bundles: Sequence[OOMBundleLike],
    identity_index: RunIdentityIndex,
    session_by_id: Mapping[str, SessionRowLike],
) -> list[RunAttachmentRow]:
    """Project OOM bundles into run attachment rows with session metadata."""
    rows: list[RunAttachmentRow] = []
    for bundle in bundles:
        if bundle.session_id is None:
            continue
        session = session_by_id.get(bundle.session_id)
        run_id = run_id_for_identity(
            run_id=None,
            session_id=bundle.session_id,
            job_id=session.job_id if session is not None else None,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        seen_ns = _datetime_to_ns(_parse_datetime(bundle.created_at_utc))
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title="OOM bundle",
                kind="oom_bundle",
                path=str(bundle.bundle_path),
                session_id=bundle.session_id,
                job_id=session.job_id if session is not None else None,
                rank=session.rank if session is not None else None,
                local_rank=session.local_rank if session is not None else None,
                world_size=session.world_size if session is not None else None,
                start_ns=seen_ns,
                end_ns=seen_ns,
                source_kind="oom_bundle",
                metadata=bundle.as_dict(),
            )
        )
    return rows


def local_attachment_row(
    *,
    run_id: str,
    title: str,
    kind: str,
    path: str,
    session_id: str | None,
    job_id: str | None,
    rank: int | None,
    local_rank: int | None,
    world_size: int | None,
    start_ns: int | None,
    end_ns: int | None,
    source_kind: str,
    metadata: Mapping[str, Any],
) -> RunAttachmentRow:
    """Build a copied local artifact attachment row."""
    return RunAttachmentRow(
        run_id=run_id,
        title=title,
        kind=kind,
        storage="copy",
        attachment_id=None,
        url=None,
        path=path,
        session_id=session_id,
        job_id=job_id,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        start_ns=start_ns,
        end_ns=end_ns,
        source_path=path,
        source_kind=source_kind,
        source_namespace="stormlog",
        source_ref=None,
        metadata=metadata,
    )


__all__ = [
    "diagnose_attachment_rows",
    "envelope_attachment_rows",
    "flat_telemetry_attachment_rows",
    "local_attachment_row",
    "oom_attachment_rows",
    "rollup_attachment_rows",
    "sidecar_attachment_rows",
    "sink_attachment_rows",
    "sink_segment_attachment_rows",
]
