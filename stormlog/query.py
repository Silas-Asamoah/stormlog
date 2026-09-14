"""Local query API for Stormlog artifact directories and telemetry files."""

from __future__ import annotations

import builtins
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from ._datetime import datetime_to_ns as _datetime_to_ns
from ._datetime import parse_datetime as _parse_datetime
from .correlation import (
    ATTACHMENTS_FILENAME,
    ATTACHMENTS_FORMAT,
    ATTACHMENTS_SCHEMA_VERSION,
    CLOCK_DOMAIN_UNIX_EPOCH_NS,
    CLOCK_NORMALIZATION_PRODUCER_EPOCH_NS,
    AttachmentFilter,
    CorrelationEvidence,
    CorrelationFilter,
    CorrelationResult,
    ExternalAttachment,
)
from .gap_analysis import analyze_hidden_memory_gaps
from .issues import (
    ISSUE_STATE_OPEN,
    IssueEvidenceLink,
    IssueFingerprint,
    IssueKind,
    IssueState,
    StormlogIssue,
    categorize_alert_context,
    normalize_issue_state,
    normalize_text_dimension,
    normalized_error_stem,
)
from .phases import (
    PhaseReplayIndex,
    phase_attribution_to_payload,
    summarize_phase_attribution,
)
from .run_catalog import (
    RUN_ENVELOPE_FILENAME,
    AttachmentStorage,
    CatalogRunEnvelope,
    RunAttachmentFilter,
    RunAttachmentRow,
    RunContext,
    RunFilter,
    RunIdentityIndex,
    RunRow,
    attachment_storage_or_default,
    build_identity_index,
    build_run_contexts,
    diagnose_attachment_rows,
    envelope_attachment_rows,
    flat_telemetry_attachment_rows,
    oom_attachment_rows,
    run_attachment_matches,
    run_envelope_from_payload,
    run_matches,
    sidecar_attachment_rows,
    sink_attachment_rows,
)
from .session import (
    SESSION_STATUS_INTERRUPTED,
    SessionSummary,
    infer_session_summary_from_events,
    session_summary_from_dict,
    session_summary_to_dict,
    stable_legacy_session_id,
)
from .telemetry import (
    LoadedTelemetrySession,
    TelemetryEvent,
    TelemetryEventV2,
    load_telemetry_sessions,
    project_telemetry_event,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from .telemetry_classification import (
    COLLECTOR_TRANSITION_TYPES,
    event_backend,
    event_severity,
    is_alert_event,
    is_collector_degradation_event,
    is_oom_event,
)
from .telemetry_rollups import (
    RankRollup,
    SessionRollup,
    TelemetryRollupFile,
    read_telemetry_rollups,
)
from .telemetry_sink import (
    MANIFEST_FILENAME,
    SEGMENT_SUFFIX,
    TelemetrySinkManifest,
    read_telemetry_sink_manifest,
    resolve_telemetry_sink_segment_paths,
)
from .timeline_markers import (
    derive_session_timeline_markers,
    timeline_marker_to_dict,
)
from .utils import format_bytes

SourceKind = Literal[
    "sink",
    "telemetry_json",
    "telemetry_jsonl",
    "telemetry_csv",
    "diagnose_bundle",
    "oom_bundle",
]
SummaryMetric = Literal[
    "session_count_by_status",
    "peak_allocator_allocated_bytes",
    "peak_allocator_reserved_bytes",
    "peak_device_used_bytes",
    "alert_count",
    "collector_degradation_transitions",
    "interrupted_sessions_with_oom_bundles",
    "hidden_memory_gap_growth",
]
SummaryGroupBy = Literal["session", "session-rank", "rank", "status"]

_TELEMETRY_FILE_NAME_PARTS = ("event", "events", "track", "telemetry")
_GAP_ANALYSIS_THRESHOLDS = {
    "gap_ratio_threshold": 0.05,
    "gap_spike_zscore": 2.0,
    "gap_drift_r_squared": 0.6,
    "gap_fragmentation_ratio": 0.3,
}
_GAP_REMEDIATION_BY_CLASSIFICATION: Mapping[str, list[str]] = {}
_SEVERITY_RANK = {"critical": 0, "error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class CatalogWarning:
    """Non-fatal discovery or loading warning."""

    path: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True)
class CatalogSource:
    """One discovered source of queryable artifact data."""

    path: Path
    source_kind: SourceKind
    event_paths: tuple[Path, ...] = ()
    manifest_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_kind": self.source_kind,
            "event_paths": [str(path) for path in self.event_paths],
            "manifest_path": (
                str(self.manifest_path) if self.manifest_path is not None else None
            ),
        }


@dataclass(frozen=True)
class CatalogOOMBundle:
    """Manifest-backed OOM dump bundle discovered during cataloging."""

    bundle_path: Path
    manifest_path: Path
    created_at_utc: str | None
    backend: str | None
    reason: str | None
    event_count: int | None
    session_id: str | None
    session_status: str | None
    exception_type: str | None = None
    exception_module: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path),
            "manifest_path": str(self.manifest_path),
            "created_at_utc": self.created_at_utc,
            "backend": self.backend,
            "reason": self.reason,
            "event_count": self.event_count,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "exception_type": self.exception_type,
            "exception_module": self.exception_module,
        }


@dataclass(frozen=True)
class SessionFilter:
    """Filters for catalog/session rows."""

    session_id: str | None = None
    status: str | None = None
    job_id: str | None = None
    rank: int | None = None
    world_size: int | None = None
    has_oom_bundle: bool | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class EventFilter:
    """Filters for canonical telemetry event rows."""

    session_id: str | None = None
    event_type: str | None = None
    rank: int | None = None
    collector: str | None = None
    status: str | None = None
    time_start_ns: int | None = None
    time_end_ns: int | None = None
    has_alert: bool | None = None
    collector_health_status: str | None = None
    backend: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class OOMBundleFilter:
    """Filters for OOM bundle rows."""

    session_id: str | None = None
    backend: str | None = None
    reason: str | None = None
    created_after: str | None = None
    created_before: str | None = None


@dataclass(frozen=True)
class IssueFilter:
    """Filters for grouped issue rows."""

    fingerprint_id: str | None = None
    kind: IssueKind | None = None
    state: IssueState | None = None
    severity: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class SessionRow:
    """Query row describing one loaded or manifest-backed session."""

    session_id: str
    status: str
    started_at_ns: int
    ended_at_ns: int | None
    host: str
    pid: int
    job_id: str | None
    rank: int
    local_rank: int
    world_size: int
    source: str
    source_path: str
    source_kind: str
    source_count: int
    warning_count: int
    event_count: int | None
    oom_bundle_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "host": self.host,
            "pid": self.pid,
            "job_id": self.job_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "source": self.source,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_count": self.source_count,
            "warning_count": self.warning_count,
            "event_count": self.event_count,
            "oom_bundle_count": self.oom_bundle_count,
        }


@dataclass(frozen=True)
class EventRow:
    """Query row wrapping a canonical telemetry event and provenance."""

    event: TelemetryEvent
    source_path: str
    source_kind: str
    session_status: str

    def as_dict(self) -> dict[str, Any]:
        row = telemetry_event_to_dict(self.event)
        row["source_path"] = self.source_path
        row["source_kind"] = self.source_kind
        row["session_status"] = self.session_status
        return row


@dataclass(frozen=True)
class OOMBundleRow:
    """Query row for one OOM bundle manifest."""

    bundle_path: str
    created_at_utc: str | None
    backend: str | None
    reason: str | None
    event_count: int | None
    session_id: str | None
    session_status: str | None
    exception_type: str | None
    exception_module: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": self.bundle_path,
            "created_at_utc": self.created_at_utc,
            "backend": self.backend,
            "reason": self.reason,
            "event_count": self.event_count,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "exception_type": self.exception_type,
            "exception_module": self.exception_module,
        }


@dataclass(frozen=True)
class SummaryRow:
    """Built-in summary result row."""

    metric: str
    group_by: str
    value: int | float | str | None
    session_id: str | None = None
    rank: int | None = None
    status: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row = {
            "metric": self.metric,
            "group_by": self.group_by,
            "session_id": self.session_id,
            "rank": self.rank,
            "status": self.status,
            "value": self.value,
        }
        row.update(dict(self.details))
        return row


class ArtifactCatalog:
    """Manifest-first catalog of local Stormlog artifacts."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.sources: list[CatalogSource] = []
        self.oom_bundles: list[CatalogOOMBundle] = []
        self.attachments: list[ExternalAttachment] = []
        self.run_envelopes: list[CatalogRunEnvelope] = []
        self.warnings: list[CatalogWarning] = []
        self._source_keys: set[tuple[Path, SourceKind]] = set()
        self._oom_bundle_paths: set[Path] = set()
        self._attachment_sidecar_paths: set[Path] = set()
        self._run_envelope_paths: set[Path] = set()
        self._run_envelope_by_id: dict[str, CatalogRunEnvelope] = {}
        self._duplicate_run_ids: set[str] = set()
        self._covered_event_paths: set[Path] = set()
        self._discover()

    def as_dict(self) -> dict[str, Any]:
        return {
            "paths": [str(path) for path in self.paths],
            "sources": [source.as_dict() for source in self.sources],
            "oom_bundles": [bundle.as_dict() for bundle in self.oom_bundles],
            "attachments": [attachment.as_dict() for attachment in self.attachments],
            "run_envelopes": [envelope.as_dict() for envelope in self.run_envelopes],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }

    def _discover(self) -> None:
        for path in self.paths:
            self._discover_path(path)

    def _discover_path(self, path: Path) -> None:
        if not path.exists():
            self._warn(path, "path does not exist")
            return
        if path.is_file():
            self._discover_file(path)
            return
        if not path.is_dir():
            self._warn(path, "path is neither a file nor a directory")
            return
        self._discover_directory(path)

    def _discover_directory(self, directory: Path) -> None:
        self._discover_run_envelope(directory / RUN_ENVELOPE_FILENAME)
        self._discover_attachment_sidecar(directory / ATTACHMENTS_FILENAME)
        manifest_path = directory / MANIFEST_FILENAME
        if self._discover_directory_manifest(manifest_path):
            return

        for nested_manifest in sorted(directory.rglob(MANIFEST_FILENAME)):
            if nested_manifest == manifest_path:
                continue
            self._discover_manifest(nested_manifest)

        for attachment_sidecar in sorted(directory.rglob(ATTACHMENTS_FILENAME)):
            self._discover_attachment_sidecar(attachment_sidecar)

        for run_envelope in sorted(directory.rglob(RUN_ENVELOPE_FILENAME)):
            self._discover_run_envelope(run_envelope)

        for candidate in self._discover_candidate_files(directory):
            if candidate in self._covered_event_paths:
                continue
            self._discover_file(candidate)

    def _discover_directory_manifest(self, path: Path) -> bool:
        """Register the root manifest, returning whether it is an opaque OOM bundle."""
        payload = _read_json_object(path)
        if payload is None:
            return False
        if _is_oom_manifest(payload):
            self._add_oom_bundle(path.parent, path, payload)
            return True
        if _is_diagnose_manifest(payload):
            self._add_diagnose_source(path)
        if _is_sink_manifest(payload):
            self._add_sink_source(path)
        return False

    def _discover_manifest(self, path: Path, *, warn: bool = False) -> None:
        """Resolve one manifest using OOM, diagnose, then sink precedence."""
        payload = _read_json_object(path)
        if payload is None:
            if warn:
                self._warn(path, "manifest is not a JSON object")
            return
        if _is_oom_manifest(payload):
            self._add_oom_bundle(path.parent, path, payload)
        elif _is_diagnose_manifest(payload):
            self._add_diagnose_source(path)
        elif _is_sink_manifest(payload):
            self._add_sink_source(path)
        elif warn:
            self._warn(path, "unrecognized manifest shape")

    def _add_diagnose_source(self, manifest_path: Path) -> None:
        self._add_source(
            CatalogSource(
                path=manifest_path.parent,
                source_kind="diagnose_bundle",
                manifest_path=manifest_path,
            )
        )

    def _add_sink_source(self, manifest_path: Path) -> None:
        directory = manifest_path.parent
        segment_paths = tuple(resolve_telemetry_sink_segment_paths(directory))
        self._covered_event_paths.update(segment_paths)
        self._add_source(
            CatalogSource(
                path=directory,
                source_kind="sink",
                event_paths=segment_paths,
                manifest_path=manifest_path,
            )
        )

    def _discover_file(self, path: Path) -> None:
        if path.name == RUN_ENVELOPE_FILENAME:
            self._discover_run_envelope(path)
            return
        if path.name == ATTACHMENTS_FILENAME:
            self._discover_attachment_sidecar(path)
            return
        if path.name == MANIFEST_FILENAME:
            self._discover_manifest(path, warn=True)
            return

        suffix = path.suffix.lower()
        if suffix == SEGMENT_SUFFIX:
            self._add_source(
                CatalogSource(
                    path=path,
                    source_kind="telemetry_jsonl",
                    event_paths=(path,),
                )
            )
        elif suffix == ".json":
            self._add_source(
                CatalogSource(
                    path=path,
                    source_kind="telemetry_json",
                    event_paths=(path,),
                )
            )
        elif suffix == ".csv":
            self._add_source(
                CatalogSource(
                    path=path,
                    source_kind="telemetry_csv",
                    event_paths=(path,),
                )
            )

    def _discover_candidate_files(self, directory: Path) -> list[Path]:
        candidates: set[Path] = set()
        for suffix in ("*.json", "*.jsonl", "*.csv"):
            for path in directory.rglob(suffix):
                if not path.is_file() or path.name == MANIFEST_FILENAME:
                    continue
                if self._is_inside_discovered_bundle(path):
                    continue
                if path.suffix.lower() == SEGMENT_SUFFIX:
                    candidates.add(path)
                    continue
                lowered = path.name.lower()
                if any(part in lowered for part in _TELEMETRY_FILE_NAME_PARTS):
                    candidates.add(path)
        return sorted(candidates)

    def _is_inside_discovered_bundle(self, path: Path) -> bool:
        return any(
            bundle_path in path.parents for bundle_path in self._oom_bundle_paths
        )

    def _add_source(self, source: CatalogSource) -> None:
        key = (source.path.resolve(), source.source_kind)
        if key in self._source_keys:
            return
        self._source_keys.add(key)
        self.sources.append(source)

    def _add_oom_bundle(
        self,
        bundle_path: Path,
        manifest_path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        resolved = bundle_path.resolve()
        if resolved in self._oom_bundle_paths:
            return
        self._oom_bundle_paths.add(resolved)
        metadata = _read_json_object(bundle_path / "metadata.json") or {}
        self.oom_bundles.append(
            CatalogOOMBundle(
                bundle_path=bundle_path,
                manifest_path=manifest_path,
                created_at_utc=_string_or_none(payload.get("created_at_utc")),
                backend=_string_or_none(payload.get("backend")),
                reason=_string_or_none(payload.get("reason")),
                event_count=_int_or_none(payload.get("event_count")),
                session_id=_string_or_none(payload.get("session_id")),
                session_status=_string_or_none(payload.get("session_status")),
                exception_type=_string_or_none(metadata.get("exception_type")),
                exception_module=_string_or_none(metadata.get("exception_module")),
            )
        )

    def _discover_attachment_sidecar(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        resolved = path.resolve()
        if resolved in self._attachment_sidecar_paths:
            return
        self._attachment_sidecar_paths.add(resolved)
        payload = _read_json_object(path)
        if payload is None:
            self._warn(path, "attachment sidecar is not a JSON object")
            return
        if not _is_attachment_sidecar(payload):
            self._warn(path, "unrecognized attachment sidecar shape")
            return
        attachments = payload.get("attachments")
        if not isinstance(attachments, Sequence) or isinstance(attachments, str):
            self._warn(path, "attachment sidecar attachments must be a list")
            return
        self._add_sidecar_attachments(path, attachments)

    def _add_sidecar_attachments(self, path: Path, attachments: Sequence[Any]) -> None:
        for index, item in enumerate(attachments):
            if not isinstance(item, Mapping):
                self._warn(path, f"attachment {index} is not an object")
                continue
            attachment = _attachment_from_payload(item, path)
            if attachment is None:
                self._warn(path, f"attachment {index} is invalid")
                continue
            self.attachments.append(attachment)

    def _discover_run_envelope(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        resolved = path.resolve()
        if resolved in self._run_envelope_paths:
            return
        self._run_envelope_paths.add(resolved)
        payload = _read_json_object(path)
        if payload is None:
            self._warn(path, "run envelope is not a JSON object")
            return
        envelope = run_envelope_from_payload(payload, path)
        if envelope is None:
            self._warn(path, "unrecognized run envelope shape")
            return
        if envelope.run_id in self._duplicate_run_ids:
            self._warn_duplicate_run_id(path, envelope.run_id)
            return
        existing = self._run_envelope_by_id.pop(envelope.run_id, None)
        if existing is not None:
            self.run_envelopes.remove(existing)
            self._duplicate_run_ids.add(envelope.run_id)
            self._warn_duplicate_run_id(existing.path, envelope.run_id)
            self._warn_duplicate_run_id(path, envelope.run_id)
            return
        self._run_envelope_by_id[envelope.run_id] = envelope
        self.run_envelopes.append(envelope)

    def _warn_duplicate_run_id(self, path: Path, run_id: str) -> None:
        self._warn(path, f"duplicate run envelope run_id {run_id!r}")

    def _warn(self, path: Path, message: str) -> None:
        self.warnings.append(CatalogWarning(path=str(path), message=message))


class QueryStore:
    """Reusable local query surface over Stormlog artifacts."""

    def __init__(self, catalog: ArtifactCatalog) -> None:
        self.catalog = catalog
        self._loaded_sessions_by_source: dict[
            tuple[Path, SourceKind], list[LoadedTelemetrySession]
        ] = {}
        self._run_identity_warning_keys: set[str] = set()

    def list_runs(self, filters: RunFilter | None = None) -> list[RunRow]:
        """Return explicit run envelopes or synthesized local run rows."""
        filters = filters or RunFilter()
        contexts = self._run_contexts()
        attachments = self._run_attachment_rows_for_contexts(contexts)
        attachment_counts: dict[str, int] = defaultdict(int)
        for attachment in attachments:
            attachment_counts[attachment.run_id] += 1
        rows = [
            context.to_row(attachment_counts.get(context.run_id, 0))
            for context in contexts.values()
        ]
        rows = [row for row in rows if run_matches(row, filters)]
        rows.sort(
            key=lambda row: (
                row.started_at_ns if row.started_at_ns is not None else -1,
                row.run_id,
            ),
            reverse=True,
        )
        return rows

    def list_run_attachments(
        self,
        filters: RunAttachmentFilter | None = None,
    ) -> list[RunAttachmentRow]:
        """Return local, distributed, and external attachments indexed by run."""
        filters = filters or RunAttachmentFilter()
        rows = self._run_attachment_rows_for_contexts(self._run_contexts())
        rows = [row for row in rows if run_attachment_matches(row, filters)]
        rows.sort(
            key=lambda row: (
                row.run_id,
                row.session_id or "",
                row.rank if row.rank is not None else -1,
                row.kind,
                row.title,
                row.source_path,
            )
        )
        return rows

    def list_sessions(self, filters: SessionFilter | None = None) -> list[SessionRow]:
        """Return session rows from manifest metadata or loaded flat files."""
        filters = filters or SessionFilter()
        oom_counts = _count_oom_bundles_by_session(self.catalog.oom_bundles)
        rows: list[SessionRow] = []
        seen: set[tuple[str, str, str]] = set()

        for source in self.catalog.sources:
            for row in self._session_rows_for_source(source, oom_counts):
                key = (row.session_id, row.source_path, row.source_kind)
                if key in seen:
                    continue
                seen.add(key)
                if _session_matches(row, filters):
                    rows.append(row)

        rows.sort(key=lambda row: (row.started_at_ns, row.session_id), reverse=True)
        return rows

    def query_events(self, filters: EventFilter | None = None) -> list[EventRow]:
        """Return filtered canonical telemetry event rows."""
        filters = filters or EventFilter()
        rows: list[EventRow] = []

        for source in self.catalog.sources:
            if source.source_kind in {"diagnose_bundle", "oom_bundle"}:
                continue
            for loaded in self._load_sessions_for_source(source):
                rows.extend(_query_session_events(loaded, source, filters))

        rows.sort(key=lambda row: (row.event.timestamp_ns, row.event.session_id))
        if filters.limit is not None:
            return rows[: filters.limit]
        return rows

    def list_oom_bundles(
        self,
        filters: OOMBundleFilter | None = None,
    ) -> list[OOMBundleRow]:
        """Return filtered OOM bundle rows."""
        filters = filters or OOMBundleFilter()
        session_status_by_id = self._manifest_session_status_by_id()
        rows = [
            OOMBundleRow(
                bundle_path=str(bundle.bundle_path),
                created_at_utc=bundle.created_at_utc,
                backend=bundle.backend,
                reason=bundle.reason,
                event_count=bundle.event_count,
                session_id=bundle.session_id,
                session_status=(
                    bundle.session_status
                    or (
                        session_status_by_id.get(bundle.session_id)
                        if bundle.session_id is not None
                        else None
                    )
                ),
                exception_type=bundle.exception_type,
                exception_module=bundle.exception_module,
            )
            for bundle in self.catalog.oom_bundles
        ]
        rows = [row for row in rows if _oom_matches(row, filters)]
        rows.sort(key=lambda row: (row.created_at_utc or "", row.bundle_path))
        return rows

    def list_attachments(
        self,
        filters: AttachmentFilter | None = None,
    ) -> list[ExternalAttachment]:
        """Return filtered external attachment sidecar rows."""
        filters = filters or AttachmentFilter()
        rows = [
            attachment
            for attachment in self.catalog.attachments
            if _attachment_matches(attachment, filters)
        ]
        rows.sort(
            key=lambda row: (
                row.start_ns if row.start_ns is not None else -1,
                row.title,
                row.sidecar_path,
            )
        )
        return rows

    def correlate(self, filters: CorrelationFilter) -> CorrelationResult:
        """Return evidence related to a timestamp or telemetry record anchor."""
        anchor = self._correlation_anchor(filters)
        evidence = self._correlation_evidence(anchor, filters)
        return CorrelationResult(
            anchor=anchor,
            evidence=evidence,
            warnings=[warning.message for warning in self.catalog.warnings],
        )

    def list_issues(
        self,
        filters: IssueFilter | None = None,
        *,
        state_overrides: Mapping[str, str] | None = None,
    ) -> list[StormlogIssue]:
        """Return grouped issues derived from discovered artifacts."""
        filters = filters or IssueFilter()
        state_by_fingerprint = {
            fingerprint_id: normalize_issue_state(state)
            for fingerprint_id, state in (state_overrides or {}).items()
        }
        accumulator: dict[str, _IssueAccumulator] = {}

        for oom_row in self.list_oom_bundles():
            _accumulate_oom_bundle_issue(accumulator, oom_row)

        event_rows = self.query_events(EventFilter())
        _accumulate_event_issues(accumulator, event_rows)

        for source in self.catalog.sources:
            for loaded in self._load_sessions_for_source(source):
                _accumulate_hidden_memory_issues(accumulator, loaded, source)

        issues = [
            item.to_issue(
                state=state_by_fingerprint.get(
                    item.fingerprint.fingerprint_id,
                    ISSUE_STATE_OPEN,
                )
            )
            for item in accumulator.values()
        ]
        issues = [issue for issue in issues if _issue_matches(issue, filters)]
        issues.sort(key=_issue_sort_key)
        return issues

    def _manifest_session_status_by_id(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for source in self.catalog.sources:
            if source.source_kind == "sink":
                manifest = read_telemetry_sink_manifest(source.path)
                if manifest is None:
                    continue
                for summary in manifest.sessions:
                    statuses.setdefault(summary.session_id, summary.status)
            elif source.source_kind == "diagnose_bundle":
                diagnose_summary = _diagnose_session_summary(source.manifest_path)
                if diagnose_summary is not None:
                    statuses.setdefault(
                        diagnose_summary.session_id,
                        diagnose_summary.status,
                    )
        return statuses

    def summarize(
        self,
        metric: SummaryMetric,
        *,
        group_by: SummaryGroupBy | None = None,
    ) -> list[SummaryRow]:
        """Run one built-in summary query."""
        if metric == "session_count_by_status":
            return self._summarize_session_count_by_status()
        if metric == "interrupted_sessions_with_oom_bundles":
            return self._summarize_interrupted_sessions_with_oom_bundles()

        resolved_group_by: SummaryGroupBy = group_by or "session"
        rollup_rows = self._summarize_from_rollups(metric, resolved_group_by)
        if rollup_rows is not None:
            return rollup_rows

        events = self.query_events(EventFilter())
        return _summarize_events(events, metric, resolved_group_by)

    def _run_contexts(self) -> dict[str, RunContext]:
        return build_run_contexts(
            self.list_sessions(SessionFilter()),
            self.catalog.run_envelopes,
        )

    def _run_attachment_rows_for_contexts(
        self,
        contexts: Mapping[str, RunContext],
    ) -> list[RunAttachmentRow]:
        identity_index = build_identity_index(contexts)
        self._record_run_identity_warnings(identity_index)
        session_by_id = {
            session.session_id: session
            for context in contexts.values()
            for session in context.sessions
        }
        rows: list[RunAttachmentRow] = []

        rows.extend(envelope_attachment_rows(self.catalog.run_envelopes, contexts))
        rows.extend(sidecar_attachment_rows(self.catalog.attachments, identity_index))
        rows.extend(self._local_source_attachment_rows(contexts, identity_index))
        rows.extend(
            oom_attachment_rows(
                self.catalog.oom_bundles,
                identity_index,
                session_by_id,
            )
        )
        return rows

    def _record_run_identity_warnings(
        self,
        identity_index: RunIdentityIndex,
    ) -> None:
        for conflict in identity_index.conflicts:
            key = (
                f"{conflict.identity_kind}:"
                f"{conflict.identity_value}:"
                f"{','.join(conflict.run_ids)}"
            )
            if key in self._run_identity_warning_keys:
                continue
            self._run_identity_warning_keys.add(key)
            self.catalog.warnings.append(
                CatalogWarning(path="<run-catalog>", message=conflict.message)
            )

    def _local_source_attachment_rows(
        self,
        contexts: Mapping[str, RunContext],
        identity_index: RunIdentityIndex,
    ) -> list[RunAttachmentRow]:
        rows: list[RunAttachmentRow] = []
        for source in self.catalog.sources:
            if source.source_kind == "sink":
                manifest = read_telemetry_sink_manifest(source.path)
                if manifest is not None:
                    rows.extend(sink_attachment_rows(source, manifest, identity_index))
            elif source.source_kind == "diagnose_bundle":
                rows.extend(
                    diagnose_attachment_rows(
                        source,
                        _diagnose_session_summary(source.manifest_path),
                        identity_index,
                    )
                )
            elif source.source_kind in {
                "telemetry_json",
                "telemetry_jsonl",
                "telemetry_csv",
            }:
                rows.extend(flat_telemetry_attachment_rows(source, contexts))
        return rows

    def _session_rows_for_source(
        self,
        source: CatalogSource,
        oom_counts: Mapping[str, int],
    ) -> list[SessionRow]:
        if source.source_kind == "sink":
            manifest = read_telemetry_sink_manifest(source.path)
            if manifest is not None and manifest.sessions:
                return [
                    _session_row_from_manifest(
                        summary=summary,
                        manifest=manifest,
                        source=source,
                        oom_count=oom_counts.get(summary.session_id, 0),
                    )
                    for summary in manifest.sessions
                ]

        if source.source_kind == "diagnose_bundle":
            summary = _diagnose_session_summary(source.manifest_path)
            if summary is None:
                return []
            return [
                _session_row_from_summary(
                    summary=summary,
                    source=source,
                    source_count=1,
                    warning_count=0,
                    event_count=None,
                    oom_count=oom_counts.get(summary.session_id, 0),
                )
            ]

        rows: list[SessionRow] = []
        for loaded in self._load_sessions_for_source(source):
            rows.append(
                _session_row_from_summary(
                    summary=loaded.summary,
                    source=source,
                    source_count=len(loaded.sources_loaded) or 1,
                    warning_count=len(loaded.warnings),
                    event_count=len(loaded.events),
                    oom_count=oom_counts.get(loaded.summary.session_id, 0),
                )
            )
        return rows

    def _summarize_from_rollups(
        self,
        metric: SummaryMetric,
        group_by: SummaryGroupBy,
    ) -> list[SummaryRow] | None:
        rollups = self._fresh_sink_rollups()
        if rollups is None:
            return None
        if metric in {
            "peak_allocator_allocated_bytes",
            "peak_allocator_reserved_bytes",
            "peak_device_used_bytes",
        }:
            return _summarize_rollup_peaks(rollups, metric, group_by)
        if metric == "alert_count" and group_by in {"session", "status"}:
            return _summarize_rollup_session_counts(
                rollups,
                metric,
                group_by,
                lambda session: session.alerts.total_count,
            )
        if metric == "collector_degradation_transitions" and group_by in {
            "session",
            "status",
        }:
            return _summarize_rollup_session_counts(
                rollups,
                metric,
                group_by,
                lambda session: session.collector_health.transition_count,
            )
        if metric == "hidden_memory_gap_growth" and group_by in {
            "rank",
            "session-rank",
        }:
            return _summarize_rollup_hidden_gap_growth(rollups, group_by)
        return None

    def _fresh_sink_rollups(self) -> list[TelemetryRollupFile] | None:
        if not self.catalog.sources:
            return None
        rollups: list[TelemetryRollupFile] = []
        for source in self.catalog.sources:
            if source.source_kind != "sink":
                return None
            rollup = read_telemetry_rollups(source.path)
            if rollup is None:
                return None
            rollups.append(rollup)
        return rollups

    def _load_sessions_for_source(
        self,
        source: CatalogSource,
    ) -> list[LoadedTelemetrySession]:
        key = (source.path.resolve(), source.source_kind)
        cached = self._loaded_sessions_by_source.get(key)
        if cached is not None:
            return cached

        try:
            if source.source_kind == "telemetry_csv":
                loaded = _load_csv_sessions(source.path)
            elif source.source_kind in {"sink", "telemetry_json", "telemetry_jsonl"}:
                loaded = load_telemetry_sessions(source.path, permissive_legacy=True)
            else:
                loaded = []
        except Exception as exc:
            self.catalog.warnings.append(
                CatalogWarning(path=str(source.path), message=f"load failed: {exc}")
            )
            loaded = []

        self._loaded_sessions_by_source[key] = loaded
        return loaded

    def _correlation_anchor(self, filters: CorrelationFilter) -> dict[str, Any]:
        if filters.record_id is None and filters.at_ns is None:
            raise ValueError("correlation requires --at-ns or --record-id")
        if filters.record_id is not None:
            for row in self.query_events(EventFilter()):
                projected = project_telemetry_event(row.event)
                if projected.record_id != filters.record_id:
                    continue
                if (
                    filters.at_ns is not None
                    and filters.at_ns != projected.timestamp_ns
                ):
                    raise ValueError("record_id timestamp does not match at_ns")
                return {
                    "at_ns": projected.timestamp_ns,
                    "record_id": projected.record_id,
                    "session_id": row.event.session_id,
                    "job_id": row.event.job_id,
                    "rank": row.event.rank,
                    "world_size": row.event.world_size,
                    "source_path": row.source_path,
                    "source_kind": row.source_kind,
                    "event_type": row.event.event_type,
                    "scope": filters.scope,
                    "window_ns": filters.window_ns,
                    "clock_domain": CLOCK_DOMAIN_UNIX_EPOCH_NS,
                    "clock_normalization": CLOCK_NORMALIZATION_PRODUCER_EPOCH_NS,
                    "clock_note": "correlation does not rewrite cross-host clocks",
                }
            raise ValueError(f"record_id not found: {filters.record_id}")
        return {
            "at_ns": filters.at_ns,
            "record_id": None,
            "session_id": filters.session_id,
            "job_id": filters.job_id,
            "rank": filters.rank,
            "world_size": None,
            "source_path": None,
            "source_kind": None,
            "event_type": None,
            "scope": filters.scope,
            "window_ns": filters.window_ns,
            "clock_domain": CLOCK_DOMAIN_UNIX_EPOCH_NS,
            "clock_normalization": CLOCK_NORMALIZATION_PRODUCER_EPOCH_NS,
            "clock_note": "correlation does not rewrite cross-host clocks",
        }

    def _correlation_evidence(
        self,
        anchor: Mapping[str, Any],
        filters: CorrelationFilter,
    ) -> list[CorrelationEvidence]:
        rows: list[CorrelationEvidence] = []
        for candidate in self._candidate_correlation_evidence():
            matched = _match_correlation_evidence(candidate, anchor, filters)
            if matched is not None:
                rows.append(matched)
        rows.sort(key=lambda row: _correlation_sort_key(row, anchor))
        if filters.limit is not None:
            return rows[: filters.limit]
        return rows

    def _candidate_correlation_evidence(self) -> list[CorrelationEvidence]:
        evidence: list[CorrelationEvidence] = []
        event_rows = self.query_events(EventFilter())
        evidence.extend(_event_correlation_evidence(event_rows))
        evidence.extend(_alert_correlation_evidence(event_rows))
        evidence.extend(self._marker_correlation_evidence())
        evidence.extend(self._oom_correlation_evidence())
        evidence.extend(self._diagnose_correlation_evidence())
        evidence.extend(self._rollup_correlation_evidence())
        evidence.extend(_attachment_correlation_evidence(self.catalog.attachments))
        return evidence

    def _marker_correlation_evidence(self) -> list[CorrelationEvidence]:
        evidence: list[CorrelationEvidence] = []
        for source in self.catalog.sources:
            for loaded in self._load_sessions_for_source(source):
                source_path = _event_source_path(loaded, source)
                for marker in derive_session_timeline_markers(loaded):
                    evidence.append(
                        CorrelationEvidence(
                            evidence_id=_evidence_id(
                                "marker",
                                marker.session_id,
                                marker.kind,
                                marker.start_ns,
                                marker.rank,
                                marker.label,
                            ),
                            kind="timeline_marker",
                            title=marker.label,
                            session_id=marker.session_id,
                            job_id=loaded.summary.job_id,
                            rank=marker.rank,
                            world_size=marker.world_size,
                            start_ns=marker.start_ns,
                            end_ns=marker.end_ns,
                            source_path=source_path,
                            source_kind=source.source_kind,
                            metadata=timeline_marker_to_dict(marker),
                        )
                    )
        return evidence

    def _oom_correlation_evidence(self) -> list[CorrelationEvidence]:
        session_by_id = {
            session.session_id: session for session in self.list_sessions()
        }
        evidence: list[CorrelationEvidence] = []
        for row in self.list_oom_bundles():
            summary = (
                session_by_id.get(row.session_id)
                if row.session_id is not None
                else None
            )
            seen_ns = _datetime_to_ns(_parse_datetime(row.created_at_utc))
            evidence.append(
                CorrelationEvidence(
                    evidence_id=_evidence_id("oom", row.bundle_path, row.session_id),
                    kind="oom_bundle",
                    title="OOM bundle",
                    session_id=row.session_id,
                    job_id=summary.job_id if summary is not None else None,
                    rank=summary.rank if summary is not None else None,
                    world_size=summary.world_size if summary is not None else None,
                    start_ns=seen_ns,
                    end_ns=seen_ns,
                    source_path=row.bundle_path,
                    source_kind="oom_bundle",
                    metadata=row.as_dict(),
                )
            )
        return evidence

    def _diagnose_correlation_evidence(self) -> list[CorrelationEvidence]:
        evidence: list[CorrelationEvidence] = []
        for source in self.catalog.sources:
            if source.source_kind != "diagnose_bundle":
                continue
            summary = _diagnose_session_summary(source.manifest_path)
            if summary is None:
                continue
            evidence.append(
                CorrelationEvidence(
                    evidence_id=_evidence_id(
                        "diagnose",
                        str(source.manifest_path),
                        summary.session_id,
                    ),
                    kind="diagnose_bundle",
                    title="Diagnose bundle",
                    session_id=summary.session_id,
                    job_id=summary.job_id,
                    rank=summary.rank,
                    world_size=summary.world_size,
                    start_ns=summary.started_at_ns,
                    end_ns=summary.ended_at_ns,
                    source_path=str(source.path),
                    source_kind=source.source_kind,
                    metadata={
                        "manifest_path": (
                            str(source.manifest_path)
                            if source.manifest_path is not None
                            else None
                        ),
                        "session_status": summary.status,
                    },
                )
            )
        return evidence

    def _rollup_correlation_evidence(self) -> list[CorrelationEvidence]:
        evidence: list[CorrelationEvidence] = []
        for source in self.catalog.sources:
            if source.source_kind != "sink":
                continue
            rollup = read_telemetry_rollups(source.path)
            if rollup is None:
                continue
            for session in rollup.sessions:
                for window in session.windows:
                    evidence.append(
                        CorrelationEvidence(
                            evidence_id=_evidence_id(
                                "rollup",
                                session.session.session_id,
                                window.index,
                                window.start_ns,
                            ),
                            kind="rollup_window",
                            title=f"Rollup window {window.index}",
                            session_id=session.session.session_id,
                            job_id=session.session.job_id,
                            rank=None,
                            world_size=session.session.world_size,
                            start_ns=window.start_ns,
                            end_ns=window.end_ns,
                            source_path=str(source.path),
                            source_kind="rollup",
                            metadata={
                                "event_count": window.event_count,
                                "sample_count": window.sample_count,
                                "rank_count": window.rank_count,
                                "alert_count": window.alert_count,
                                "collector_transition_count": (
                                    window.collector_transition_count
                                ),
                                "oom_count": window.oom_count,
                                "window_duration_ns": rollup.window_duration_ns,
                            },
                        )
                    )
        return evidence

    def _summarize_session_count_by_status(self) -> list[SummaryRow]:
        counts: dict[str, int] = defaultdict(int)
        for row in self.list_sessions(SessionFilter()):
            counts[row.status] += 1
        return [
            SummaryRow(
                metric="session_count_by_status",
                group_by="status",
                status=status,
                value=count,
            )
            for status, count in sorted(counts.items())
        ]

    def _summarize_interrupted_sessions_with_oom_bundles(self) -> list[SummaryRow]:
        oom_counts = _count_oom_bundles_by_session(self.catalog.oom_bundles)
        rows: list[SummaryRow] = []
        for session in self.list_sessions(
            SessionFilter(
                status=SESSION_STATUS_INTERRUPTED,
                has_oom_bundle=True,
            )
        ):
            rows.append(
                SummaryRow(
                    metric="interrupted_sessions_with_oom_bundles",
                    group_by="session",
                    session_id=session.session_id,
                    status=session.status,
                    value=oom_counts.get(session.session_id, 0),
                )
            )
        return rows


def open(paths: Sequence[str | Path]) -> QueryStore:
    """Open one or more local artifact paths for in-process querying."""
    return QueryStore(ArtifactCatalog(paths))


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _is_sink_manifest(payload: Mapping[str, Any]) -> bool:
    fmt = payload.get("format")
    return fmt == "stormlog.append_only_telemetry_sink" or "segments" in payload


def _is_oom_manifest(payload: Mapping[str, Any]) -> bool:
    return (
        "bundle_name" in payload
        and "reason" in payload
        and "backend" in payload
        and "event_count" in payload
    )


def _is_diagnose_manifest(payload: Mapping[str, Any]) -> bool:
    return (
        "command_line" in payload
        and "files" in payload
        and "risk_detected" in payload
        and "session_id" in payload
    )


def _is_attachment_sidecar(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema_version") == ATTACHMENTS_SCHEMA_VERSION
        and payload.get("format") == ATTACHMENTS_FORMAT
    )


def _attachment_from_payload(
    payload: Mapping[str, Any],
    sidecar_path: Path,
) -> ExternalAttachment | None:
    title = _string_or_none(payload.get("title"))
    kind = _string_or_none(payload.get("kind"))
    if title is None or kind is None:
        return None
    url = _string_or_none(payload.get("url"))
    raw_path = _string_or_none(payload.get("path"))
    if url is None and raw_path is None:
        return None
    resolved_path = _resolve_sidecar_attachment_path(raw_path, sidecar_path)
    start_ns = _int_or_none(payload.get("start_ns"))
    end_ns = _int_or_none(payload.get("end_ns"))
    if start_ns is not None and end_ns is not None and end_ns < start_ns:
        return None
    metadata = payload.get("metadata")
    return ExternalAttachment(
        title=title,
        kind=kind,
        attachment_id=_string_or_none(payload.get("attachment_id")),
        url=url,
        path=resolved_path,
        session_id=_string_or_none(payload.get("session_id")),
        job_id=_string_or_none(payload.get("job_id")),
        rank=_int_or_none(payload.get("rank")),
        start_ns=start_ns,
        end_ns=end_ns,
        created_at_utc=_string_or_none(payload.get("created_at_utc")),
        updated_at_utc=_string_or_none(payload.get("updated_at_utc")),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        sidecar_path=str(sidecar_path),
        run_id=_string_or_none(payload.get("run_id")),
        storage=attachment_storage_or_default(payload.get("storage")),
        source_namespace=_string_or_none(payload.get("source_namespace")),
        source_ref=_string_or_none(payload.get("source_ref")),
    )


def _resolve_sidecar_attachment_path(
    raw_path: str | None, sidecar_path: Path
) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = sidecar_path.parent / path
    return str(path.resolve())


def _diagnose_session_summary(manifest_path: Path | None) -> SessionSummary | None:
    if manifest_path is None:
        return None
    payload = _read_json_object(manifest_path)
    if payload is None:
        return None
    session_payload = payload.get("session")
    if isinstance(session_payload, Mapping):
        try:
            return session_summary_from_dict(session_payload)
        except Exception:
            pass
    session_id = _string_or_none(payload.get("session_id"))
    if session_id is None:
        return None
    try:
        return session_summary_from_dict(
            {
                "session_id": session_id,
                "status": payload.get("session_status", "incomplete"),
                "started_at_ns": 0,
                "ended_at_ns": None,
                "host": "unknown",
                "pid": -1,
                "job_id": None,
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "source": "stormlog.diagnose",
            }
        )
    except Exception:
        return None


def _session_row_from_manifest(
    *,
    summary: SessionSummary,
    manifest: TelemetrySinkManifest,
    source: CatalogSource,
    oom_count: int,
) -> SessionRow:
    session_segments = [
        segment
        for segment in manifest.segments
        if segment.session_id == summary.session_id
    ]
    event_count = sum(segment.event_count for segment in session_segments)
    return _session_row_from_summary(
        summary=summary,
        source=source,
        source_count=len(session_segments),
        warning_count=0,
        event_count=event_count,
        oom_count=oom_count,
    )


def _session_row_from_summary(
    *,
    summary: SessionSummary,
    source: CatalogSource,
    source_count: int,
    warning_count: int,
    event_count: int | None,
    oom_count: int,
) -> SessionRow:
    payload = session_summary_to_dict(summary)
    return SessionRow(
        session_id=str(payload["session_id"]),
        status=str(payload["status"]),
        started_at_ns=int(payload["started_at_ns"]),
        ended_at_ns=(
            int(payload["ended_at_ns"])
            if payload.get("ended_at_ns") is not None
            else None
        ),
        host=str(payload["host"]),
        pid=int(payload["pid"]),
        job_id=_string_or_none(payload.get("job_id")),
        rank=int(payload["rank"]),
        local_rank=int(payload["local_rank"]),
        world_size=int(payload["world_size"]),
        source=str(payload["source"]),
        source_path=str(source.path),
        source_kind=source.source_kind,
        source_count=source_count,
        warning_count=warning_count,
        event_count=event_count,
        oom_bundle_count=oom_count,
    )


def _load_csv_sessions(path: Path) -> list[LoadedTelemetrySession]:
    warnings: list[str] = []
    events: list[TelemetryEvent] = []
    default_session_id = stable_legacy_session_id(str(path.resolve()), "csv")

    with builtins.open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            try:
                event = telemetry_event_from_record(
                    _normalize_csv_record(row),
                    permissive_legacy=True,
                    default_collector="legacy.csv",
                    default_sampling_interval_ms=0,
                    default_session_id=default_session_id,
                )
                events.append(event)
            except Exception as exc:
                warnings.append(f"CSV parse error {path}:{line_number}: {exc}")

    grouped: dict[str, list[TelemetryEvent]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)

    sessions: list[LoadedTelemetrySession] = []
    for session_id, session_events in grouped.items():
        session_events.sort(key=lambda event: event.timestamp_ns)
        summary = infer_session_summary_from_events(
            session_id=session_id,
            events=session_events,
            source=f"artifact:{path.name or path.resolve()}",
        )
        sessions.append(
            LoadedTelemetrySession(
                summary=summary,
                events=session_events,
                sources_loaded=[str(path.resolve())],
                warnings=warnings,
            )
        )
    return sessions


def _normalize_csv_record(row: Mapping[str, str]) -> dict[str, Any]:
    int_fields = {
        "schema_version",
        "timestamp_ns",
        "sampling_interval_ms",
        "pid",
        "rank",
        "local_rank",
        "world_size",
        "device_id",
        "allocator_allocated_bytes",
        "allocator_reserved_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
        "allocator_change_bytes",
        "device_used_bytes",
        "device_free_bytes",
        "device_total_bytes",
        "memory_allocated",
        "memory_reserved",
        "memory_change",
        "total_memory",
    }
    float_fields = {"timestamp"}
    normalized: dict[str, Any] = {}
    for key, raw_value in row.items():
        value: Any = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if value == "":
            normalized[key] = None
        elif key == "metadata" and isinstance(value, str):
            normalized[key] = _parse_csv_metadata(value)
        elif key in int_fields:
            text_value = str(value).strip()
            try:
                normalized[key] = int(text_value)
            except ValueError:
                normalized[key] = int(float(text_value))
        elif key in float_fields:
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def _parse_csv_metadata(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _count_oom_bundles_by_session(
    bundles: Iterable[CatalogOOMBundle],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for bundle in bundles:
        if bundle.session_id is not None:
            counts[bundle.session_id] += 1
    return counts


def _session_matches(row: SessionRow, filters: SessionFilter) -> bool:
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.status is not None and row.status != filters.status:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    return _session_topology_and_artifacts_match(row, filters)


def _session_topology_and_artifacts_match(
    row: SessionRow, filters: SessionFilter
) -> bool:
    if filters.rank is not None and row.rank != filters.rank:
        return False
    if filters.world_size is not None and row.world_size != filters.world_size:
        return False
    if filters.has_oom_bundle is not None and (
        (row.oom_bundle_count > 0) != filters.has_oom_bundle
    ):
        return False
    if filters.source_kind is not None and row.source_kind != filters.source_kind:
        return False
    return True


def _event_matches(row: EventRow, filters: EventFilter) -> bool:
    event = row.event
    if filters.event_type is not None and event.event_type != filters.event_type:
        return False
    if filters.rank is not None and event.rank != filters.rank:
        return False
    if filters.collector is not None and event.collector != filters.collector:
        return False
    if not _event_time_and_alert_match(event, filters):
        return False
    return _event_collector_metadata_matches(event, filters)


def _event_collector_metadata_matches(
    event: TelemetryEvent, filters: EventFilter
) -> bool:
    metadata = event.metadata
    if filters.collector_health_status is not None and (
        metadata.get("collector_health_status") != filters.collector_health_status
    ):
        return False
    if filters.backend is not None and metadata.get("backend") != filters.backend:
        return False
    return True


def _event_time_and_alert_match(event: TelemetryEvent, filters: EventFilter) -> bool:
    if filters.time_start_ns is not None and event.timestamp_ns < filters.time_start_ns:
        return False
    if filters.time_end_ns is not None and event.timestamp_ns > filters.time_end_ns:
        return False
    if filters.has_alert is not None and (is_alert_event(event) != filters.has_alert):
        return False
    return True


def _query_session_events(
    loaded: LoadedTelemetrySession,
    source: CatalogSource,
    filters: EventFilter,
) -> Iterator[EventRow]:
    summary = loaded.summary
    if filters.session_id is not None and summary.session_id != filters.session_id:
        return
    if filters.status is not None and summary.status != filters.status:
        return
    for event in loaded.events:
        row = EventRow(
            event=event,
            source_path=_event_source_path(loaded, source),
            source_kind=source.source_kind,
            session_status=summary.status,
        )
        if _event_matches(row, filters):
            yield row


def _oom_matches(row: OOMBundleRow, filters: OOMBundleFilter) -> bool:
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.backend is not None and row.backend != filters.backend:
        return False
    if filters.reason is not None and row.reason != filters.reason:
        return False
    return _oom_creation_time_matches(row, filters)


def _oom_creation_time_matches(row: OOMBundleRow, filters: OOMBundleFilter) -> bool:
    created = _parse_datetime(row.created_at_utc)
    after = _parse_datetime(filters.created_after)
    before = _parse_datetime(filters.created_before)
    if after is not None and (created is None or created < after):
        return False
    if before is not None and (created is None or created > before):
        return False
    return True


def _attachment_matches(
    row: ExternalAttachment,
    filters: AttachmentFilter,
) -> bool:
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    if filters.rank is not None and row.rank != filters.rank:
        return False
    if filters.kind is not None and row.kind != filters.kind:
        return False
    return True


def _event_correlation_evidence(
    rows: Sequence[EventRow],
) -> list[CorrelationEvidence]:
    evidence: list[CorrelationEvidence] = []
    for row in rows:
        event = row.event
        projected = project_telemetry_event(event)
        metadata = {
            "record_id": projected.record_id,
            "observed_timestamp_ns": projected.observed_timestamp_ns,
            "event_type": event.event_type,
            "collector": event.collector,
            "context": event.context,
            "session_status": row.session_status,
            "metadata": event.metadata,
        }
        evidence.append(
            CorrelationEvidence(
                evidence_id=f"event:{projected.record_id}",
                kind="telemetry_event",
                title=f"{event.event_type} telemetry event",
                session_id=event.session_id,
                job_id=event.job_id,
                rank=event.rank,
                world_size=event.world_size,
                start_ns=event.timestamp_ns,
                end_ns=None,
                source_path=row.source_path,
                source_kind=row.source_kind,
                metadata=metadata,
            )
        )
    return evidence


def _alert_correlation_evidence(
    rows: Sequence[EventRow],
) -> list[CorrelationEvidence]:
    evidence: list[CorrelationEvidence] = []
    for row in rows:
        event = row.event
        if not is_alert_event(event):
            continue
        evidence.append(
            CorrelationEvidence(
                evidence_id=_evidence_id(
                    "alert",
                    event.session_id,
                    event.timestamp_ns,
                    event.rank,
                    event.event_type,
                ),
                kind="alert",
                title=f"Alert: {event.event_type}",
                session_id=event.session_id,
                job_id=event.job_id,
                rank=event.rank,
                world_size=event.world_size,
                start_ns=event.timestamp_ns,
                end_ns=None,
                source_path=row.source_path,
                source_kind=row.source_kind,
                metadata={
                    "event_type": event.event_type,
                    "severity": event_severity(event),
                    "collector": event.collector,
                    "context": event.context,
                    "metadata": event.metadata,
                },
            )
        )
    return evidence


def _attachment_correlation_evidence(
    attachments: Sequence[ExternalAttachment],
) -> list[CorrelationEvidence]:
    evidence: list[CorrelationEvidence] = []
    for attachment in attachments:
        created_ns = _datetime_to_ns(_parse_datetime(attachment.created_at_utc))
        start_ns = (
            attachment.start_ns if attachment.start_ns is not None else created_ns
        )
        end_ns = attachment.end_ns if attachment.end_ns is not None else start_ns
        evidence.append(
            CorrelationEvidence(
                evidence_id=_evidence_id(
                    "attachment",
                    attachment.attachment_id,
                    attachment.sidecar_path,
                    attachment.title,
                    attachment.url,
                    attachment.path,
                ),
                kind="attachment",
                title=attachment.title,
                session_id=attachment.session_id,
                job_id=attachment.job_id,
                rank=attachment.rank,
                world_size=None,
                start_ns=start_ns,
                end_ns=end_ns,
                source_path=attachment.url
                or attachment.path
                or attachment.sidecar_path,
                source_kind="attachment",
                metadata=attachment.as_dict(),
            )
        )
    return evidence


def _match_correlation_evidence(
    evidence: CorrelationEvidence,
    anchor: Mapping[str, Any],
    filters: CorrelationFilter,
) -> CorrelationEvidence | None:
    if filters.kinds and evidence.kind not in filters.kinds:
        return None
    if not _explicit_correlation_filters_match(evidence, filters):
        return None

    anchor_at = cast(int | None, anchor.get("at_ns"))
    if anchor_at is None:
        return None

    has_time = evidence.start_ns is not None or evidence.end_ns is not None
    time_matches = (
        _evidence_overlaps_anchor_window(evidence, anchor_at, filters.window_ns)
        if has_time
        else False
    )
    identity_match = _identity_match(evidence, anchor, filters.scope)
    if not _correlation_context_matches(
        evidence, anchor, filters.scope, identity_match, has_time, time_matches
    ):
        return None

    confidence, reasons = _correlation_confidence(
        evidence,
        anchor,
        identity_match,
        has_time,
        time_matches,
    )
    return replace(evidence, confidence=confidence, reasons=tuple(reasons))


def _correlation_context_matches(
    evidence: CorrelationEvidence,
    anchor: Mapping[str, Any],
    scope: str,
    identity_match: str | None,
    has_time: bool,
    time_matches: bool,
) -> bool:
    if has_time and not time_matches:
        return False
    if identity_match is None and not time_matches:
        return False
    return not _has_identity_conflict(evidence, anchor, scope)


def _explicit_correlation_filters_match(
    evidence: CorrelationEvidence,
    filters: CorrelationFilter,
) -> bool:
    if filters.session_id is not None and evidence.session_id != filters.session_id:
        return False
    if filters.job_id is not None and evidence.job_id != filters.job_id:
        return False
    if (
        filters.rank is not None
        and evidence.rank is not None
        and evidence.rank != filters.rank
    ):
        return False
    return True


def _evidence_overlaps_anchor_window(
    evidence: CorrelationEvidence,
    at_ns: int,
    window_ns: int,
) -> bool:
    evidence_start = (
        evidence.start_ns if evidence.start_ns is not None else evidence.end_ns
    )
    evidence_end = evidence.end_ns if evidence.end_ns is not None else evidence.start_ns
    if evidence_start is None or evidence_end is None:
        return False
    anchor_start = at_ns - window_ns
    anchor_end = at_ns + window_ns
    return evidence_start <= anchor_end and evidence_end >= anchor_start


def _identity_match(
    evidence: CorrelationEvidence,
    anchor: Mapping[str, Any],
    scope: str,
) -> str | None:
    anchor_session = _string_or_none(anchor.get("session_id"))
    anchor_job = _string_or_none(anchor.get("job_id"))
    if (
        anchor_session is not None
        and evidence.session_id is not None
        and evidence.session_id == anchor_session
    ):
        return "session"
    if (
        scope == "distributed"
        and anchor_job is not None
        and evidence.job_id is not None
        and evidence.job_id == anchor_job
    ):
        return "job"
    return None


def _has_identity_conflict(
    evidence: CorrelationEvidence,
    anchor: Mapping[str, Any],
    scope: str,
) -> bool:
    anchor_session = _string_or_none(anchor.get("session_id"))
    anchor_job = _string_or_none(anchor.get("job_id"))
    same_job = (
        anchor_job is not None
        and evidence.job_id is not None
        and evidence.job_id == anchor_job
    )
    if _known_identifiers_differ(anchor_session, evidence.session_id) and not (
        scope == "distributed" and same_job
    ):
        return True
    if (
        _known_identifiers_differ(anchor_job, evidence.job_id)
        and evidence.session_id != anchor_session
    ):
        return True
    return False


def _known_identifiers_differ(first: str | None, second: str | None) -> bool:
    return first is not None and second is not None and first != second


def _correlation_confidence(
    evidence: CorrelationEvidence,
    anchor: Mapping[str, Any],
    identity_match: str | None,
    has_time: bool,
    time_matches: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if time_matches:
        reasons.append("overlaps_anchor_window")
    elif not has_time:
        reasons.append("identifier_only_no_time")

    anchor_rank = _int_or_none(anchor.get("rank"))
    same_rank = (
        anchor_rank is None or evidence.rank is None or evidence.rank == anchor_rank
    )
    if identity_match == "session":
        reasons.append("same_session")
        if same_rank:
            reasons.append("same_rank_or_rank_agnostic")
            return "high", reasons
        reasons.append("cross_rank_same_session")
        return "medium", reasons
    if identity_match == "job":
        reasons.append("same_job_distributed")
        if same_rank:
            reasons.append("same_rank_or_rank_agnostic")
            return "high", reasons
        reasons.append("cross_rank_same_job")
        return "medium", reasons

    reasons.append("time_only_missing_shared_identifier")
    return "low", reasons


def _correlation_sort_key(
    evidence: CorrelationEvidence,
    anchor: Mapping[str, Any],
) -> tuple[int, int, str, str]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(
        evidence.confidence,
        9,
    )
    anchor_at = cast(int | None, anchor.get("at_ns"))
    distance = _evidence_distance_ns(evidence, anchor_at)
    return confidence_rank, distance, evidence.kind, evidence.title


def _evidence_distance_ns(
    evidence: CorrelationEvidence,
    at_ns: int | None,
) -> int:
    if at_ns is None:
        return 0
    bounds = [
        value for value in (evidence.start_ns, evidence.end_ns) if value is not None
    ]
    if not bounds:
        return 0
    start_ns = min(bounds)
    end_ns = max(bounds)
    if start_ns <= at_ns <= end_ns:
        return 0
    return min(abs(start_ns - at_ns), abs(end_ns - at_ns))


def _evidence_id(*parts: object) -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"evidence-{digest}"


def _event_source_path(loaded: LoadedTelemetrySession, source: CatalogSource) -> str:
    if loaded.sources_loaded:
        return loaded.sources_loaded[0]
    return str(source.path)


def _summary_group_key(
    row: EventRow,
    group_by: SummaryGroupBy,
) -> tuple[str | None, int | None, str | None]:
    if group_by == "rank":
        return None, row.event.rank, None
    if group_by == "session-rank":
        return row.event.session_id, row.event.rank, None
    if group_by == "status":
        return None, None, row.session_status
    return row.event.session_id, None, None


def _summarize_rollup_peaks(
    rollups: Sequence[TelemetryRollupFile],
    metric: str,
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    field_name = {
        "peak_allocator_allocated_bytes": "allocator_allocated_bytes",
        "peak_allocator_reserved_bytes": "allocator_reserved_bytes",
        "peak_device_used_bytes": "device_used_bytes",
    }[metric]
    best: dict[tuple[str | None, int | None, str | None], tuple[int, int | None]] = {}

    for session in _rollup_sessions(rollups):
        if group_by in {"rank", "session-rank"}:
            for rank_rollup in session.ranks:
                peak = getattr(rank_rollup.counters, field_name)
                key = _rollup_rank_group_key(session, rank_rollup, group_by)
                _observe_rollup_peak(best, key, peak.value, peak.timestamp_ns)
            continue

        peak = getattr(session.counters, field_name)
        key = _rollup_session_group_key(session, group_by)
        _observe_rollup_peak(best, key, peak.value, peak.timestamp_ns)

    output: list[SummaryRow] = []
    for key, (value, timestamp_ns) in sorted(
        best.items(), key=lambda item: str(item[0])
    ):
        session_id, row_rank, status = key
        output.append(
            SummaryRow(
                metric=metric,
                group_by=group_by,
                session_id=session_id,
                rank=row_rank,
                status=status,
                value=value,
                details={"timestamp_ns": timestamp_ns},
            )
        )
    return output


def _summarize_rollup_session_counts(
    rollups: Sequence[TelemetryRollupFile],
    metric: str,
    group_by: SummaryGroupBy,
    value_for_session: Callable[[SessionRollup], int],
) -> list[SummaryRow]:
    counts: dict[tuple[str | None, int | None, str | None], int] = defaultdict(int)
    for session in _rollup_sessions(rollups):
        value = value_for_session(session)
        if value <= 0:
            continue
        counts[_rollup_session_group_key(session, group_by)] += value

    output: list[SummaryRow] = []
    for session_id, rank, status in sorted(counts, key=str):
        output.append(
            SummaryRow(
                metric=metric,
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=counts[(session_id, rank, status)],
            )
        )
    return output


def _summarize_rollup_hidden_gap_growth(
    rollups: Sequence[TelemetryRollupFile],
    group_by: SummaryGroupBy,
) -> list[SummaryRow] | None:
    sessions = _rollup_sessions(rollups)
    if group_by == "rank" and len(sessions) != 1:
        return None

    output: list[SummaryRow] = []
    for session in sessions:
        for rank in session.ranks:
            if rank.hidden_gap_delta_bytes is None:
                continue
            session_id = (
                session.session.session_id if group_by == "session-rank" else None
            )
            output.append(
                SummaryRow(
                    metric="hidden_memory_gap_growth",
                    group_by=group_by,
                    session_id=session_id,
                    rank=rank.rank,
                    value=rank.hidden_gap_delta_bytes,
                    details={
                        "first_gap_bytes": rank.hidden_gap_first_bytes,
                        "latest_gap_bytes": rank.hidden_gap_latest_bytes,
                        "peak_gap_bytes": rank.hidden_gap_peak_bytes,
                        "sample_count": rank.sample_count,
                    },
                )
            )
    return sorted(output, key=lambda row: str((row.session_id, row.rank, row.status)))


def _rollup_sessions(
    rollups: Sequence[TelemetryRollupFile],
) -> list[SessionRollup]:
    return [session for rollup in rollups for session in rollup.sessions]


def _rollup_session_group_key(
    session: SessionRollup,
    group_by: SummaryGroupBy,
) -> tuple[str | None, int | None, str | None]:
    if group_by == "status":
        return None, None, session.session.status
    return session.session.session_id, None, None


def _rollup_rank_group_key(
    session: SessionRollup,
    rank: RankRollup,
    group_by: SummaryGroupBy,
) -> tuple[str | None, int | None, str | None]:
    if group_by == "rank":
        return None, rank.rank, None
    return session.session.session_id, rank.rank, None


def _observe_rollup_peak(
    best: dict[tuple[str | None, int | None, str | None], tuple[int, int | None]],
    key: tuple[str | None, int | None, str | None],
    value: int | None,
    timestamp_ns: int | None,
) -> None:
    if value is None:
        return
    existing = best.get(key)
    if existing is None or value > existing[0]:
        best[key] = (value, timestamp_ns)


@dataclass
class _IssueAccumulator:
    fingerprint: IssueFingerprint
    title: str
    severity: str
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[IssueEvidenceLink] = field(default_factory=list)
    affected_sessions: set[str] = field(default_factory=set)
    first_seen_ns: int | None = None
    last_seen_ns: int | None = None

    def add(
        self,
        *,
        evidence: IssueEvidenceLink,
        seen_ns: int | None,
        session_id: str | None,
        severity: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one evidence hit to this accumulator."""
        self.evidence.append(evidence)
        if session_id is not None:
            self.affected_sessions.add(session_id)
        if seen_ns is not None:
            if self.first_seen_ns is None or seen_ns < self.first_seen_ns:
                self.first_seen_ns = seen_ns
            if self.last_seen_ns is None or seen_ns > self.last_seen_ns:
                self.last_seen_ns = seen_ns
        if severity is not None:
            self.severity = _max_severity(self.severity, severity)
        if details:
            self.details.update(details)

    def to_issue(self, *, state: IssueState) -> StormlogIssue:
        """Build an immutable issue row from accumulated hits."""
        representative = self.evidence[0]
        return StormlogIssue(
            fingerprint=self.fingerprint,
            title=self.title,
            state=state,
            severity=self.severity,
            hit_count=len(self.evidence),
            first_seen_ns=self.first_seen_ns,
            last_seen_ns=self.last_seen_ns,
            affected_sessions=tuple(self.affected_sessions),
            representative_evidence=representative,
            evidence=tuple(self.evidence),
            details=dict(self.details),
        )


def _accumulate_issue(
    accumulator: dict[str, _IssueAccumulator],
    *,
    fingerprint: IssueFingerprint,
    title: str,
    severity: str,
    evidence: IssueEvidenceLink,
    seen_ns: int | None,
    session_id: str | None,
    details: Mapping[str, Any] | None = None,
) -> None:
    fingerprint_id = fingerprint.fingerprint_id
    issue = accumulator.get(fingerprint_id)
    if issue is None:
        issue = _IssueAccumulator(
            fingerprint=fingerprint,
            title=title,
            severity=severity,
            details=dict(details or {}),
        )
        accumulator[fingerprint_id] = issue
    issue.add(
        evidence=evidence,
        seen_ns=seen_ns,
        session_id=session_id,
        severity=severity,
        details=details,
    )


def _accumulate_oom_bundle_issue(
    accumulator: dict[str, _IssueAccumulator],
    row: OOMBundleRow,
) -> None:
    fingerprint = IssueFingerprint(
        kind="oom",
        dimensions={
            "backend": row.backend,
            "reason": row.reason,
        },
    )
    seen_ns = _datetime_to_ns(_parse_datetime(row.created_at_utc))
    evidence = IssueEvidenceLink(
        session_id=row.session_id,
        timestamp_ns=seen_ns,
        source_path=row.bundle_path,
        source_kind="oom_bundle",
        bundle_path=row.bundle_path,
        metadata={
            "created_at_utc": row.created_at_utc,
            "event_count": row.event_count,
            "session_status": row.session_status,
        },
    )
    _accumulate_issue(
        accumulator,
        fingerprint=fingerprint,
        title="OOM captured by flight recorder",
        severity="critical",
        evidence=evidence,
        seen_ns=seen_ns,
        session_id=row.session_id,
        details={
            "backend": row.backend,
            "reason": row.reason,
            "exception_type": row.exception_type,
            "exception_module": row.exception_module,
        },
    )


def _accumulate_event_issues(
    accumulator: dict[str, _IssueAccumulator], rows: Sequence[EventRow]
) -> None:
    for row in rows:
        if is_oom_event(row.event):
            _accumulate_oom_event_issue(accumulator, row)
        elif is_collector_degradation_event(row.event):
            _accumulate_collector_issue(accumulator, row)
        elif is_alert_event(row.event):
            _accumulate_alert_issue(accumulator, row)


def _accumulate_oom_event_issue(
    accumulator: dict[str, _IssueAccumulator],
    row: EventRow,
) -> None:
    event = row.event
    metadata = event.metadata
    reason = metadata.get("oom_reason")
    bundle_path = _string_or_none(metadata.get("oom_dump_path"))
    fingerprint = IssueFingerprint(
        kind="oom",
        dimensions={
            "backend": event_backend(event),
            "reason": reason,
        },
    )
    evidence = IssueEvidenceLink(
        session_id=event.session_id,
        timestamp_ns=event.timestamp_ns,
        rank=event.rank,
        source_path=row.source_path,
        source_kind=row.source_kind,
        event_type=event.event_type,
        bundle_path=bundle_path,
        metadata={"context": event.context, "session_status": row.session_status},
    )
    _accumulate_issue(
        accumulator,
        fingerprint=fingerprint,
        title="OOM telemetry event",
        severity="critical",
        evidence=evidence,
        seen_ns=event.timestamp_ns,
        session_id=event.session_id,
        details={
            "backend": event_backend(event),
            "reason": reason,
            "collector": event.collector,
            "device_id": event.device_id,
        },
    )


def _accumulate_collector_issue(
    accumulator: dict[str, _IssueAccumulator],
    row: EventRow,
) -> None:
    event = row.event
    metadata = event.metadata
    health_status = normalize_text_dimension(
        metadata.get("collector_health_status"),
        default="degraded",
    )
    partial_fields = metadata.get("collector_partial_fields")
    if not isinstance(partial_fields, Sequence) or isinstance(partial_fields, str):
        partial_fields = ()
    last_error = metadata.get("collector_last_error")
    fingerprint = IssueFingerprint(
        kind="collector_degradation",
        dimensions={
            "collector": event.collector,
            "backend": event_backend(event),
            "health_status": health_status,
            "partial_fields": list(partial_fields),
            "error_stem": normalized_error_stem(last_error),
        },
    )
    evidence = IssueEvidenceLink(
        session_id=event.session_id,
        timestamp_ns=event.timestamp_ns,
        rank=event.rank,
        source_path=row.source_path,
        source_kind=row.source_kind,
        event_type=event.event_type,
        metadata={
            "collector_consecutive_failures": metadata.get(
                "collector_consecutive_failures"
            ),
            "collector_next_retry_epoch_s": metadata.get(
                "collector_next_retry_epoch_s"
            ),
            "session_status": row.session_status,
        },
    )
    _accumulate_issue(
        accumulator,
        fingerprint=fingerprint,
        title="Collector degradation",
        severity="critical" if health_status == "unhealthy" else "warning",
        evidence=evidence,
        seen_ns=event.timestamp_ns,
        session_id=event.session_id,
        details={
            "collector": event.collector,
            "backend": event_backend(event),
            "health_status": health_status,
            "partial_fields": list(partial_fields),
            "error_stem": normalized_error_stem(last_error),
        },
    )


def _accumulate_alert_issue(
    accumulator: dict[str, _IssueAccumulator],
    row: EventRow,
) -> None:
    event = row.event
    severity = event_severity(event)
    category = categorize_alert_context(event.context)
    fingerprint = IssueFingerprint(
        kind="alert",
        dimensions={
            "event_type": event.event_type,
            "severity": severity,
            "collector": event.collector,
            "backend": event_backend(event),
            "category": category,
        },
    )
    evidence = IssueEvidenceLink(
        session_id=event.session_id,
        timestamp_ns=event.timestamp_ns,
        rank=event.rank,
        source_path=row.source_path,
        source_kind=row.source_kind,
        event_type=event.event_type,
        metadata={"context": event.context, "session_status": row.session_status},
    )
    _accumulate_issue(
        accumulator,
        fingerprint=fingerprint,
        title=f"Alert: {category.replace('_', ' ')}",
        severity=severity,
        evidence=evidence,
        seen_ns=event.timestamp_ns,
        session_id=event.session_id,
        details={
            "event_type": event.event_type,
            "collector": event.collector,
            "backend": event_backend(event),
            "category": category,
        },
    )


def _accumulate_hidden_memory_issues(
    accumulator: dict[str, _IssueAccumulator],
    loaded: LoadedTelemetrySession,
    source: CatalogSource,
) -> None:
    if len(loaded.events) < 3:
        return
    phase_resolver = PhaseReplayIndex.from_events(loaded.events)
    findings = analyze_hidden_memory_gaps(
        events=cast(Sequence[TelemetryEventV2], loaded.events),
        thresholds=_GAP_ANALYSIS_THRESHOLDS,
        format_memory=format_bytes,
        remediation_by_classification=_GAP_REMEDIATION_BY_CLASSIFICATION,
        phase_resolver=phase_resolver,
    )
    for finding in findings:
        evidence_event = _event_at_timestamp(
            loaded.events,
            finding.evidence_timestamp_ns,
        )
        phase_label = summarize_phase_attribution(finding.phase_attribution)
        fingerprint = IssueFingerprint(
            kind="hidden_memory_anomaly",
            dimensions={
                "classification": finding.classification,
                "severity": finding.severity,
                "phase": phase_label,
                "collector": (
                    evidence_event.collector
                    if evidence_event is not None
                    else "unknown"
                ),
                "backend": (
                    event_backend(evidence_event)
                    if evidence_event is not None
                    else "unknown"
                ),
            },
        )
        evidence = IssueEvidenceLink(
            session_id=loaded.summary.session_id,
            timestamp_ns=finding.evidence_timestamp_ns,
            rank=evidence_event.rank if evidence_event is not None else None,
            source_path=_event_source_path(loaded, source),
            source_kind=source.source_kind,
            event_type=(
                evidence_event.event_type if evidence_event is not None else "sample"
            ),
            metadata={
                "classification": finding.classification,
                "confidence": finding.confidence,
                "phase_attribution": phase_attribution_to_payload(
                    finding.phase_attribution
                ),
            },
        )
        _accumulate_issue(
            accumulator,
            fingerprint=fingerprint,
            title=f"Hidden-memory anomaly: {finding.classification}",
            severity=finding.severity,
            evidence=evidence,
            seen_ns=finding.evidence_timestamp_ns,
            session_id=loaded.summary.session_id,
            details={
                "classification": finding.classification,
                "description": finding.description,
                "confidence": finding.confidence,
                "evidence": dict(finding.evidence),
                "phase": phase_label,
            },
        )


def _event_at_timestamp(
    events: Sequence[TelemetryEvent],
    timestamp_ns: int | None,
) -> TelemetryEvent | None:
    if timestamp_ns is None:
        return None
    for event in events:
        if event.timestamp_ns == timestamp_ns:
            return event
    return None


def _max_severity(first: str, second: str) -> str:
    first_rank = _SEVERITY_RANK.get(first, 9)
    second_rank = _SEVERITY_RANK.get(second, 9)
    return second if second_rank < first_rank else first


def _issue_matches(issue: StormlogIssue, filters: IssueFilter) -> bool:
    if not _issue_classification_matches(issue, filters):
        return False
    if filters.session_id is not None and (
        filters.session_id not in issue.affected_sessions
    ):
        return False
    return True


def _issue_classification_matches(issue: StormlogIssue, filters: IssueFilter) -> bool:
    if filters.fingerprint_id is not None and (
        issue.fingerprint_id != filters.fingerprint_id
    ):
        return False
    if filters.kind is not None and issue.kind != filters.kind:
        return False
    if filters.state is not None and issue.state != filters.state:
        return False
    if filters.severity is not None and issue.severity != filters.severity:
        return False
    return True


def _issue_sort_key(issue: StormlogIssue) -> tuple[int, int, str]:
    last_seen = issue.last_seen_ns if issue.last_seen_ns is not None else -1
    return (
        _SEVERITY_RANK.get(issue.severity, 9),
        -last_seen,
        issue.fingerprint_id,
    )


def _summarize_events(
    events: Sequence[EventRow], metric: SummaryMetric, group_by: SummaryGroupBy
) -> list[SummaryRow]:
    peak_fields = {
        "peak_allocator_allocated_bytes": "allocator_allocated_bytes",
        "peak_allocator_reserved_bytes": "allocator_reserved_bytes",
        "peak_device_used_bytes": "device_used_bytes",
    }
    if metric in peak_fields:
        field_name = peak_fields[metric]
        return _summarize_peak(events, metric, field_name, group_by)
    if metric == "alert_count":
        alert_events = [row for row in events if is_alert_event(row.event)]
        return _summarize_count(alert_events, metric, group_by)
    if metric == "collector_degradation_transitions":
        transition_events = [
            row for row in events if row.event.event_type in COLLECTOR_TRANSITION_TYPES
        ]
        return _summarize_count(transition_events, metric, group_by)
    if metric == "hidden_memory_gap_growth":
        return _summarize_hidden_memory_gap_growth(events, group_by)
    raise ValueError(f"unsupported summary metric: {metric}")


def _summarize_peak(
    rows: Sequence[EventRow],
    metric: str,
    field_name: str,
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    best: dict[tuple[str | None, int | None, str | None], tuple[int, EventRow]] = {}
    for row in rows:
        value = int(getattr(row.event, field_name))
        key = _summary_group_key(row, group_by)
        existing = best.get(key)
        if existing is None or value > existing[0]:
            best[key] = (value, row)

    output: list[SummaryRow] = []
    for key, (value, event_row) in sorted(best.items(), key=lambda item: str(item[0])):
        session_id, rank, status = key
        output.append(
            SummaryRow(
                metric=metric,
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=value,
                details={"timestamp_ns": event_row.event.timestamp_ns},
            )
        )
    return output


def _summarize_count(
    rows: Sequence[EventRow],
    metric: str,
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    counts: dict[tuple[str | None, int | None, str | None], int] = defaultdict(int)
    for row in rows:
        counts[_summary_group_key(row, group_by)] += 1
    output: list[SummaryRow] = []
    for session_id, rank, status in sorted(counts, key=str):
        output.append(
            SummaryRow(
                metric=metric,
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=counts[(session_id, rank, status)],
            )
        )
    return output


def _summarize_hidden_memory_gap_growth(
    rows: Sequence[EventRow],
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    grouped: dict[tuple[str | None, int | None, str | None], list[EventRow]] = (
        defaultdict(list)
    )
    for row in rows:
        if row.event.event_type != "sample":
            continue
        grouped[_summary_group_key(row, group_by)].append(row)

    output: list[SummaryRow] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        group_rows.sort(key=lambda row: row.event.timestamp_ns)
        gaps = [
            row.event.device_used_bytes - row.event.allocator_reserved_bytes
            for row in group_rows
            if row.event.device_used_bytes is not None
            and row.event.allocator_reserved_bytes is not None
        ]
        if not gaps:
            continue
        session_id, rank, status = key
        output.append(
            SummaryRow(
                metric="hidden_memory_gap_growth",
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=gaps[-1] - gaps[0],
                details={
                    "first_gap_bytes": gaps[0],
                    "latest_gap_bytes": gaps[-1],
                    "peak_gap_bytes": max(gaps),
                    "sample_count": len(gaps),
                },
            )
        )
    return output


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "AttachmentStorage",
    "AttachmentFilter",
    "ArtifactCatalog",
    "CatalogRunEnvelope",
    "CatalogOOMBundle",
    "CatalogSource",
    "CatalogWarning",
    "CorrelationEvidence",
    "CorrelationFilter",
    "CorrelationResult",
    "EventFilter",
    "EventRow",
    "ExternalAttachment",
    "IssueFilter",
    "OOMBundleFilter",
    "OOMBundleRow",
    "QueryStore",
    "RunAttachmentFilter",
    "RunAttachmentRow",
    "RunFilter",
    "RunRow",
    "SessionFilter",
    "SessionRow",
    "SummaryRow",
    "open",
]
