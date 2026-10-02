"""Import profiler traces into an existing inference artifact.

The artifact's ``infer.artifact`` record supplies the run and session. Each
trace is registered in the run envelope and its GPU activity is appended as
``infer.activity_ref`` records through ``append_inference_capture``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..session import create_session_summary
from .correlation_capture import (
    CaptureCapabilities,
    TraceAttachment,
    TraceCapture,
    append_inference_capture,
)
from .correlation_events import (
    ArtifactIdentityEvent,
    CapabilityEvent,
    load_inference_artifact,
)
from .errors import InferInputError, InferUsageError
from .trace_kineto import SUPPORTED, Detail, import_kineto_trace


def artifact_run_identity(path: str | Path) -> tuple[str, str]:
    """Return the run and session IDs recorded by the artifact's identity record."""
    for record in load_inference_artifact(path):
        if isinstance(record, ArtifactIdentityEvent):
            return record.context.run_id, record.context.session_id
    raise InferInputError(
        f"{path}: no infer.artifact record; profile with a Stormlog version that "
        "writes run identity before importing traces"
    )


def parse_device_uuids(values: Sequence[str]) -> dict[int, str]:
    """Parse ``INDEX=UUID`` pairs; a bare ``UUID`` means device 0.

    The index is the CUDA device ordinal inside the traced process, after
    ``CUDA_VISIBLE_DEVICES``. It is not necessarily the host's NVML index.
    """
    mapping: dict[int, str] = {}
    for value in values:
        index_text, separator, uuid = value.partition("=")
        if not separator:
            index_text, uuid = "0", value
        if not index_text.isdigit() or not uuid:
            raise InferUsageError(f"--device-uuid {value!r}: expected INDEX=UUID")
        index = int(index_text)
        if index in mapping and mapping[index] != uuid:
            raise InferUsageError(f"--device-uuid: device {index} given twice")
        mapping[index] = uuid
    return mapping


def trace_attachment_id(path: Path, prefix: str = "kineto") -> str:
    """A stable ID per file: its name and a digest of its resolved path.

    The name alone would make ``run-a/rank0.pt.trace.json.gz`` and
    ``run-b/rank0.pt.trace.json.gz`` the same attachment.
    """
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{path.name}:{digest}"


class KinetoTraceCollector:
    """A ``TraceCollector`` over one or more Kineto trace files."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        device_uuids: dict[int, str] | None = None,
        detail: Detail = "launch",
    ) -> None:
        if not paths:
            raise InferUsageError("at least one trace file is required")
        self.paths = [Path(path) for path in paths]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise InferInputError(f"trace file not found: {', '.join(missing)}")
        self.device_uuids = dict(device_uuids or {})
        self.detail = detail

    def collect(self, *, run_id: str, session_id: str) -> TraceCapture:
        return combine_captures(
            [self._import(path, run_id, session_id) for path in self.paths]
        )

    def _import(self, path: Path, run_id: str, session_id: str) -> TraceCapture:
        attachment = TraceAttachment(
            attachment_id=trace_attachment_id(path),
            title=f"Kineto trace {path.name}",
            path=path.resolve(),
            storage="reference",
            metadata={"format": "kineto-chrome-trace"},
        )
        try:
            return import_kineto_trace(
                path,
                run_id=run_id,
                session_id=session_id,
                attachment=attachment,
                device_uuids=self.device_uuids,
                detail=self.detail,
            )
        except (ValueError, OSError) as exc:
            raise InferInputError(f"{path}: {exc}") from exc


def combine_captures(captures: Sequence[TraceCapture]) -> TraceCapture:
    """Merge per-trace captures; a capability is collected if any trace had it."""
    collected: set[str] = set()
    events: list[Any] = []
    attachments: list[TraceAttachment] = []
    summaries: list[dict[str, Any]] = []
    for capture in captures:
        collected.update(capture.capabilities.collected)
        events.extend(capture.events)
        attachments.extend(capture.attachments)
        if capture.summary is not None:
            summaries.append(capture.summary)
    return TraceCapture(
        capabilities=CaptureCapabilities(
            supported=SUPPORTED,
            enabled=SUPPORTED,
            collected=tuple(name for name in SUPPORTED if name in collected),
        ),
        events=tuple(events),
        attachments=tuple(attachments),
        summary={"traces": summaries},
    )


def import_traces_into_artifact(
    artifact: str | Path,
    traces: Sequence[str | Path],
    *,
    device_uuids: dict[int, str] | None = None,
    detail: Detail = "launch",
    envelope_path: str | Path | None = None,
) -> TraceCapture:
    """Append the traces' GPU activity to ``artifact`` and return what was added.

    A trace this artifact already imported from the same file is skipped and
    listed in the summary's ``already_imported``; importing it again would only
    duplicate its records. A trace that was only registered, for example over a
    size bound, can still be imported.
    """
    run_id, session_id = artifact_run_identity(artifact)
    imported = imported_trace_paths(artifact)
    skipped = [str(t) for t in traces if Path(t).resolve() in imported]
    pending = [t for t in traces if str(t) not in skipped]
    if not pending:
        return _nothing_imported(skipped)
    collector = KinetoTraceCollector(pending, device_uuids=device_uuids, detail=detail)
    captured: list[TraceCapture] = []

    class _Recording:
        def collect(self, *, run_id: str, session_id: str) -> TraceCapture:
            captured.append(collector.collect(run_id=run_id, session_id=session_id))
            return captured[-1]

    try:
        append_inference_capture(
            artifact,
            run_id=run_id,
            session=create_session_summary(
                source="stormlog.infer.import_trace", session_id=session_id
            ),
            trace_collector=_Recording(),
            envelope_path=envelope_path,
        )
    except (InferInputError, InferUsageError):
        raise
    except ValueError as exc:
        # Every remaining rejection is about the artifact or its envelope.
        raise InferInputError(f"cannot import into {artifact}: {exc}") from exc
    summary = dict(captured[0].summary or {}, already_imported=skipped)
    return replace(captured[0], summary=summary)


def imported_trace_paths(artifact: str | Path) -> set[Path]:
    """Resolved paths of the traces whose GPU activity the artifact already holds.

    Read from the trace collectors' capability summaries, which name each
    trace's file; a trace registered without being parsed does not count.
    """
    paths: set[Path] = set()
    for record in load_inference_artifact(artifact):
        if (
            isinstance(record, CapabilityEvent)
            and record.component == "trace_collector"
        ):
            summary = record.metadata.get("summary") or {}
            paths.update(
                Path(trace["path"])
                for trace in summary.get("traces", [])
                if isinstance(trace, dict)
                and trace.get("path")
                and not trace.get("skipped")
            )
    return paths


def _nothing_imported(skipped: list[str]) -> TraceCapture:
    return TraceCapture(
        capabilities=CaptureCapabilities(SUPPORTED, SUPPORTED, ()),
        summary={"traces": [], "already_imported": skipped},
    )


__all__ = [
    "KinetoTraceCollector",
    "artifact_run_identity",
    "combine_captures",
    "import_traces_into_artifact",
    "imported_trace_paths",
    "parse_device_uuids",
    "trace_attachment_id",
]
