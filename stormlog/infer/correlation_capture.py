"""Optional server evidence capture using existing inference and run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from .. import __version__
from ..run_catalog import (
    RUN_ENVELOPE_FILENAME,
    RUN_ENVELOPE_FORMAT,
    RUN_ENVELOPE_SCHEMA_VERSION,
    run_envelope_from_payload,
)
from ..session import SessionSummary
from .correlation_events import (
    ActivityReferenceEvent,
    CapabilityEvent,
    CorrelationContext,
    CorrelationEvent,
    EntityRef,
    load_inference_artifact,
)


@dataclass(frozen=True)
class CaptureCapabilities:
    """Features a component supports, enabled, and actually collected."""

    supported: tuple[str, ...] = ()
    enabled: tuple[str, ...] = ()
    collected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("supported", "enabled", "collected"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain non-empty names")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not set(self.collected) <= set(self.enabled) <= set(self.supported):
            raise ValueError("collected must be enabled and enabled must be supported")


@dataclass(frozen=True)
class TraceAttachment:
    """A raw trace to register through the existing run envelope catalog."""

    attachment_id: str
    title: str
    path: Path | None = None
    url: str | None = None
    storage: Literal["reference", "copy"] = "reference"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.attachment_id or not self.title:
            raise ValueError("trace attachment_id and title are required")
        if (self.path is None) == (self.url is None):
            raise ValueError("trace attachment needs exactly one path or URL")
        if self.url is not None and not self.url:
            raise ValueError("trace URL must be non-empty")
        if self.storage not in {"reference", "copy"}:
            raise ValueError("storage must be reference or copy")
        if self.storage == "copy" and self.path is None:
            raise ValueError("copied trace needs a local path")


@dataclass(frozen=True)
class EngineCapture:
    capabilities: CaptureCapabilities
    events: tuple[CorrelationEvent, ...] = ()


@dataclass(frozen=True)
class TraceCapture:
    capabilities: CaptureCapabilities
    events: tuple[CorrelationEvent, ...] = ()
    attachments: tuple[TraceAttachment, ...] = ()


class EngineAdapter(Protocol):
    """Optional source of server request, iteration, stage, and membership data."""

    def collect(self, *, run_id: str, session_id: str) -> EngineCapture: ...


class TraceCollector(Protocol):
    """Optional source of GPU activities and raw trace attachments."""

    def collect(self, *, run_id: str, session_id: str) -> TraceCapture: ...


def append_inference_capture(
    artifact_path: str | Path,
    *,
    run_id: str,
    session: SessionSummary,
    engine_adapter: EngineAdapter | None = None,
    trace_collector: TraceCollector | None = None,
    envelope_path: str | Path | None = None,
) -> None:
    """Append optional evidence and index its raw traces in a run envelope.

    Collection and validation finish before either artifact is changed. An
    absent component contributes an explicit unavailable capability record.
    """
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise ValueError("inference artifact must already exist")
    if not run_id:
        raise ValueError("run_id is required")
    _validate_existing_artifact(artifact, run_id, session.session_id)
    envelope = Path(envelope_path or artifact.parent / RUN_ENVELOPE_FILENAME)
    engine, trace = _collect_optional(
        run_id, session.session_id, engine_adapter, trace_collector
    )
    engine_events = engine.events if engine else ()
    trace_events = trace.events if trace else ()
    _validate_adapter_events(
        (*engine_events, *trace_events), run_id, session.session_id
    )
    linked_trace_events = link_trace_activities(engine_events, trace_events)
    capability_events = (
        _capability_event(run_id, session, "engine_adapter", engine),
        _capability_event(run_id, session, "trace_collector", trace),
    )
    events = (*engine_events, *linked_trace_events, *capability_events)
    payload = _prepare_run_envelope(
        envelope,
        run_id=run_id,
        session=session,
        artifact=artifact,
        attachments=trace.attachments if trace else (),
    )
    _validate_attachment_references(events, payload)
    _write_envelope(envelope, payload)
    _append_events(artifact, events)


def _validate_existing_artifact(artifact: Path, run_id: str, session_id: str) -> None:
    for record in load_inference_artifact(artifact):
        original = record.to_record()
        context = original.get("context")
        record_session = original.get("session_id")
        record_run = None
        if isinstance(context, dict):
            record_session = context.get("session_id")
            record_run = context.get("run_id")
        if record_session is not None and record_session != session_id:
            raise ValueError("artifact contains a different session_id")
        if record_run is not None and record_run != run_id:
            raise ValueError("artifact contains a different run_id")


def _collect_optional(
    run_id: str,
    session_id: str,
    engine_adapter: EngineAdapter | None,
    trace_collector: TraceCollector | None,
) -> tuple[EngineCapture | None, TraceCapture | None]:
    engine = (
        engine_adapter.collect(run_id=run_id, session_id=session_id)
        if engine_adapter is not None
        else None
    )
    trace = (
        trace_collector.collect(run_id=run_id, session_id=session_id)
        if trace_collector is not None
        else None
    )
    return engine, trace


def _validate_adapter_events(
    events: tuple[CorrelationEvent, ...], run_id: str, session_id: str
) -> None:
    for event in events:
        if not isinstance(event, CorrelationEvent):
            raise ValueError("adapters must return v2 correlation events")
        if event.context.run_id != run_id:
            raise ValueError("adapter event run_id does not match capture run_id")
        if event.context.session_id != session_id:
            raise ValueError("adapter event session_id does not match capture session")


def _validate_attachment_references(
    events: tuple[CorrelationEvent, ...], payload: dict[str, Any]
) -> None:
    attachment_ids = {
        item.get("attachment_id") for item in payload.get("attachments", [])
    }
    for event in events:
        if (
            isinstance(event, ActivityReferenceEvent)
            and event.trace_attachment_id is not None
            and event.trace_attachment_id not in attachment_ids
        ):
            raise ValueError("activity references an unregistered trace attachment")


def _append_events(artifact: Path, events: tuple[CorrelationEvent, ...]) -> None:
    with artifact.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), sort_keys=True) + "\n")


def link_trace_activities(
    engine_events: tuple[CorrelationEvent, ...],
    trace_events: tuple[CorrelationEvent, ...],
) -> tuple[CorrelationEvent, ...]:
    """Join only unambiguous scoped runtime/CUDA IDs, never timestamps alone."""
    links: dict[tuple[object, ...], set[EntityRef]] = {}
    for event in engine_events:
        if isinstance(event, ActivityReferenceEvent) and event.iteration_ref:
            for key in _correlation_keys(event):
                links.setdefault(key, set()).add(event.iteration_ref)
    result: list[CorrelationEvent] = []
    for event in trace_events:
        if not isinstance(event, ActivityReferenceEvent) or event.iteration_ref:
            result.append(event)
            continue
        candidates: set[EntityRef] = set()
        for key in _correlation_keys(event):
            candidates.update(links.get(key, set()))
        if len(candidates) == 1:
            event = replace(
                event,
                iteration_ref=next(iter(candidates)),
                attribution_status="linked",
            )
        result.append(event)
    return tuple(result)


def _correlation_keys(event: ActivityReferenceEvent) -> tuple[tuple[object, ...], ...]:
    context = event.context
    if event.correlation_scope is None or context.host is None or context.pid is None:
        return ()
    scope = (
        context.run_id,
        context.session_id,
        context.host,
        context.pid,
        context.device_uuid,
        event.correlation_scope,
    )
    keys = []
    if event.runtime_correlation_id is not None:
        keys.append((*scope, "runtime", event.runtime_correlation_id))
    if event.cuda_correlation_id is not None:
        keys.append((*scope, "cuda", event.cuda_correlation_id))
    return tuple(keys)


def _capability_event(
    run_id: str,
    session: SessionSummary,
    component: str,
    result: EngineCapture | TraceCapture | None,
) -> CapabilityEvent:
    capabilities = result.capabilities if result else CaptureCapabilities()
    return CapabilityEvent(
        context=CorrelationContext(
            run_id=run_id,
            session_id=session.session_id,
            producer_id="stormlog.infer.capture",
            source="stormlog.infer.capture",
            source_version=__version__,
            host=session.host,
            pid=session.pid,
            rank=session.rank,
            local_rank=session.local_rank,
            world_size=session.world_size,
            clock_domain=f"{session.host}/unix_epoch_ns",
            clock_kind="wall",
            collection_mode="active",
            provenance="observed",
        ),
        event_id=f"{component}:{uuid4()}",
        component=component,
        available=result is not None,
        supported=list(capabilities.supported),
        enabled=list(capabilities.enabled),
        collected=list(capabilities.collected),
    )


def _prepare_run_envelope(
    envelope: Path,
    *,
    run_id: str,
    session: SessionSummary,
    artifact: Path,
    attachments: tuple[TraceAttachment, ...],
) -> dict[str, Any]:
    payload = _load_or_create_envelope(envelope, run_id)
    _ensure_session(payload, session)
    catalog_attachments = payload.setdefault("attachments", [])
    _add_attachment(
        catalog_attachments,
        {
            "attachment_id": f"infer-jsonl:{session.session_id}:{artifact.name}",
            "title": "Inference JSONL",
            "kind": "inference_jsonl",
            "storage": "reference",
            "path": os.path.relpath(artifact, envelope.parent),
            "session_id": session.session_id,
            "metadata": {"format": "jsonl"},
        },
    )
    for attachment in attachments:
        _add_attachment(
            catalog_attachments,
            _trace_attachment_row(attachment, envelope, session.session_id),
        )
    if run_envelope_from_payload(payload, envelope) is None:
        raise ValueError("inference run envelope does not match the catalog schema")
    return payload


def _load_or_create_envelope(envelope: Path, run_id: str) -> dict[str, Any]:
    if envelope.exists():
        payload = json.loads(envelope.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or run_envelope_from_payload(payload, envelope) is None
        ):
            raise ValueError("existing run envelope is invalid")
        if payload["run_id"] != run_id:
            raise ValueError("existing run envelope has a different run_id")
    else:
        payload = {
            "schema_version": RUN_ENVELOPE_SCHEMA_VERSION,
            "format": RUN_ENVELOPE_FORMAT,
            "run_id": run_id,
            "sessions": [],
            "attachments": [],
            "metadata": {},
        }
    return payload


def _ensure_session(payload: dict[str, Any], session: SessionSummary) -> None:
    sessions = payload.setdefault("sessions", [])
    if not any(item["session_id"] == session.session_id for item in sessions):
        sessions.append(
            {
                "session_id": session.session_id,
                "job_id": session.job_id,
                "rank": session.rank,
                "local_rank": session.local_rank,
                "world_size": session.world_size,
                "role": "inference_capture",
                "metadata": {},
            }
        )


def _trace_attachment_row(
    attachment: TraceAttachment, envelope: Path, session_id: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "attachment_id": attachment.attachment_id,
        "title": attachment.title,
        "kind": "profiler_trace",
        "storage": attachment.storage,
        "session_id": session_id,
        "metadata": attachment.metadata or {},
    }
    if attachment.path is None:
        row["url"] = attachment.url
        return row
    trace_path = attachment.path
    if not trace_path.is_absolute():
        trace_path = envelope.parent / trace_path
    if attachment.storage == "copy" and not trace_path.is_file():
        raise ValueError("copied trace path must exist")
    row["path"] = os.path.relpath(trace_path, envelope.parent)
    return row


def _add_attachment(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    for row in rows:
        if row.get("attachment_id") == candidate["attachment_id"]:
            if row != candidate:
                raise ValueError("attachment_id already refers to different evidence")
            return
    rows.append(candidate)


def _write_envelope(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "CaptureCapabilities",
    "EngineAdapter",
    "EngineCapture",
    "TraceAttachment",
    "TraceCapture",
    "TraceCollector",
    "append_inference_capture",
    "link_trace_activities",
]
