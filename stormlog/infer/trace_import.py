"""Import profiler traces (Kineto, Nsight Systems) into an inference artifact.

The artifact's ``infer.artifact`` record supplies the run and session. Each
trace is registered in the run envelope and its GPU activity is appended as
``infer.activity_ref`` records through ``append_inference_capture``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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
from .trace_kineto import (
    SUPPORTED,
    Detail,
    KinetoTrace,
    capture_trace,
    load_kineto_trace,
)
from .trace_nsys import load_nsys_sqlite


def artifact_run_identity(path: str | Path) -> tuple[str, str]:
    """Return the run and session IDs recorded by the artifact's identity record."""
    for record in load_inference_artifact(path):
        if isinstance(record, ArtifactIdentityEvent):
            return record.context.run_id, record.context.session_id
    raise InferInputError(
        f"{path}: no infer.artifact record; profile with a Stormlog version that "
        "writes run identity before importing traces"
    )


@dataclass(frozen=True)
class DeviceUuids:
    """GPU UUIDs by CUDA ordinal: shared by every trace, or for one trace.

    An ordinal is local to the traced process (after ``CUDA_VISIBLE_DEVICES``),
    so a shared mapping is only safe when each ordinal belongs to one process.
    ``per_trace`` is keyed by the selector as given until ``bind`` matches each
    selector to one trace; from then on it is keyed by the trace's resolved path.
    """

    shared: dict[int, str] = field(default_factory=dict)
    per_trace: dict[str, dict[int, str]] = field(default_factory=dict)
    bound: bool = False

    def bind(self, paths: Sequence[str | Path]) -> DeviceUuids:
        """Match every selector to exactly one of ``paths``, or refuse.

        A selector is a trace's path, as given or resolved, or its file name
        when only one of ``paths`` has that name. A selector that names no
        trace, or a name that several traces share, is a usage error.
        """
        if self.bound:
            return self
        candidates = [Path(path) for path in paths]
        per_trace: dict[str, dict[int, str]] = {}
        for selector, mapping in self.per_trace.items():
            key = str(_select_trace(selector, candidates).resolve())
            _add_entries(per_trace.setdefault(key, {}), mapping, selector)
        return DeviceUuids(dict(self.shared), per_trace, bound=True)

    def scoped(self, path: str | Path) -> dict[int, str]:
        """The entries given for this trace alone."""
        if not self.bound:
            raise ValueError("bind the device UUIDs to the trace paths first")
        return self.per_trace.get(str(Path(path).resolve()), {})

    def for_trace(self, path: str | Path) -> dict[int, str]:
        """Shared entries, overridden by the ones given for this trace."""
        return {**self.shared, **self.scoped(path)}


def parse_device_uuids(values: Sequence[str]) -> DeviceUuids:
    """Parse ``[TRACE_FILE:]INDEX=UUID`` entries; a bare ``UUID`` means device 0.

    The index is the CUDA device ordinal inside the traced process, after
    ``CUDA_VISIBLE_DEVICES``. It is not necessarily the host's NVML index. A
    ``TRACE_FILE`` prefix limits the entry to one trace: the trace's path as
    passed to the command, or its file name when only one trace has it.
    ``DeviceUuids.bind`` checks the prefixes against the traces.
    """
    shared: dict[int, str] = {}
    per_trace: dict[str, dict[int, str]] = {}
    for value in values:
        target, index, uuid = _device_uuid_entry(value)
        table = shared if target is None else per_trace.setdefault(target, {})
        _add_entries(table, {index: uuid}, target)
    return DeviceUuids(shared, per_trace)


def _add_entries(
    table: dict[int, str], entries: dict[int, str], selector: str | None
) -> None:
    for index, uuid in entries.items():
        if table.get(index, uuid) != uuid:
            scope = f" of {selector}" if selector else ""
            raise InferUsageError(f"--device-uuid: device {index}{scope} given twice")
        table[index] = uuid


def _select_trace(selector: str, paths: Sequence[Path]) -> Path:
    """The one trace among ``paths`` that a ``--device-uuid`` prefix names."""
    wanted = Path(selector)
    by_path = {
        str(path.resolve()): path
        for path in paths
        if path == wanted or path.resolve() == wanted.resolve()
    }
    if len(by_path) == 1:
        return next(iter(by_path.values()))
    by_name = {str(path.resolve()): path for path in paths if path.name == selector}
    if len(by_name) == 1:
        return next(iter(by_name.values()))
    if by_name:
        raise InferUsageError(
            f"--device-uuid {selector}:...: {len(by_name)} traces in this import "
            f"are named {selector}; give the trace's path instead"
        )
    raise InferUsageError(
        f"--device-uuid {selector}:...: no trace in this import is {selector}"
    )


def _as_device_uuids(value: DeviceUuids | dict[int, str] | None) -> DeviceUuids:
    if isinstance(value, DeviceUuids):
        return value
    return DeviceUuids(dict(value or {}))


def _device_uuid_entry(value: str) -> tuple[str | None, int, str]:
    left, separator, uuid = value.partition("=")
    if not separator:
        left, uuid = "0", value
    target, colon, index_text = left.rpartition(":")
    if not index_text.isdigit() or not uuid or (colon and not target):
        raise InferUsageError(
            f"--device-uuid {value!r}: expected [TRACE_FILE:]INDEX=UUID"
        )
    return (target if colon else None), int(index_text), uuid


def trace_attachment_id(path: Path, prefix: str = "kineto") -> str:
    """A stable ID per file: its name and a digest of its resolved path.

    The name alone would make ``run-a/rank0.pt.trace.json.gz`` and
    ``run-b/rank0.pt.trace.json.gz`` the same attachment.
    """
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{path.name}:{digest}"


FORMATS = {
    ".sqlite": ("nsys", "Nsight Systems SQLite export", "nsys-sqlite"),
    ".nsys-rep": ("nsys", "Nsight Systems report", "nsys-rep"),
}
KINETO = ("kineto", "Kineto trace", "kineto-chrome-trace")


class TraceFileCollector:
    """A ``TraceCollector`` over profiler trace files, chosen by file type.

    ``.sqlite`` files are read as Nsight Systems exports; ``.nsys-rep`` reports
    are registered but not read (export them to SQLite first); anything else
    is read as a Kineto Chrome trace.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        device_uuids: DeviceUuids | dict[int, str] | None = None,
        detail: Detail = "launch",
        max_bytes: int | None = None,
    ) -> None:
        if not paths:
            raise InferUsageError("at least one trace file is required")
        self.paths = [Path(path) for path in paths]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise InferInputError(f"trace file not found: {', '.join(missing)}")
        # Refuses a per-trace selector that fits none or several of the paths.
        self.device_uuids = _as_device_uuids(device_uuids).bind(self.paths)
        self.detail = detail
        self.max_bytes = max_bytes

    def collect(self, *, run_id: str, session_id: str) -> TraceCapture:
        captures = [self._import(path, run_id, session_id) for path in self.paths]
        _check_shared_devices(self.paths, captures, self.device_uuids)
        return combine_captures(captures)

    def _import(self, path: Path, run_id: str, session_id: str) -> TraceCapture:
        prefix, title, file_format = FORMATS.get(path.suffix, KINETO)
        attachment = TraceAttachment(
            attachment_id=trace_attachment_id(path, prefix),
            title=f"{title} {path.name}",
            path=path.resolve(),
            storage="reference",
            metadata={"format": file_format},
        )
        if file_format == "nsys-rep":
            return _registered_only(attachment, path, "not_exported")
        if self.max_bytes is not None and path.stat().st_size > self.max_bytes:
            # Registered so `import-trace` can import it later; not parsed now.
            return _registered_only(attachment, path, "max_bytes")
        try:
            return capture_trace(
                _load(path, file_format),
                path,
                run_id=run_id,
                session_id=session_id,
                attachment=attachment,
                device_uuids=self.device_uuids.for_trace(path),
                detail=self.detail,
            )
        except InferUsageError:
            raise
        except (ValueError, OSError) as exc:
            raise InferInputError(f"{path}: {exc}") from exc


def _check_shared_devices(
    paths: Sequence[Path], captures: Sequence[TraceCapture], uuids: DeviceUuids
) -> None:
    """Refuse a ``--device-uuid`` ordinal that several processes used.

    A process is its host name, rank, and pid (for formats that hold several,
    such as Nsight Systems reports) or launching pids. Only UUIDs taken from
    the option count; a UUID the trace names is the process's own. Containers
    that share a host name and pid namespace layout can still look alike,
    which is why the docs recommend the per-trace form for traces from
    separate containers.
    """
    users: dict[tuple[str | None, int], set[tuple[Any, ...]]] = {}
    for path, capture in zip(paths, captures):
        summary = capture.summary or {}
        own = uuids.scoped(path)
        for key, device in summary.get("devices", {}).items():
            if device.get("device_uuid_source") != "option":
                continue
            pid, index = _split_device_key(str(key))
            scope = str(path) if index in own else None
            process = (
                summary.get("host"),
                summary.get("rank"),
                pid if pid is not None else tuple(summary.get("processes", ())),
            )
            users.setdefault((scope, index), set()).add(process)
    for (scope, index), processes in sorted(users.items(), key=str):
        if len(processes) > 1:
            raise InferUsageError(_shared_device_message(scope, index))


def _split_device_key(key: str) -> tuple[int | None, int]:
    """(pid, ordinal) from a summary device key: ``"0"`` or ``"<pid>/0"``."""
    pid, _, ordinal = key.rpartition("/")
    return (int(pid) if pid else None), int(ordinal)


def _shared_device_message(scope: str | None, index: int) -> str:
    given = f"{scope}:{index}" if scope else str(index)
    return (
        f"--device-uuid {given}=...: several processes use device {index}, which "
        "each may map to a different GPU; give a UUID per trace as "
        f"TRACE_FILE:{index}=UUID, or for an Nsight Systems report re-export it "
        "with nsys 2025.1 or later so it names each process's GPUs"
    )


def _load(path: Path, file_format: str) -> KinetoTrace:
    if file_format == "nsys-sqlite":
        return load_nsys_sqlite(path)
    return load_kineto_trace(path)


def _registered_only(
    attachment: TraceAttachment, path: Path, reason: str
) -> TraceCapture:
    return TraceCapture(
        capabilities=CaptureCapabilities(SUPPORTED, SUPPORTED, ()),
        attachments=(attachment,),
        summary={
            "file": path.name,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "skipped": reason,
        },
    )


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
    device_uuids: DeviceUuids | dict[int, str] | None = None,
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
    # Bound to every requested trace, so a selector for one that is skipped
    # below as already imported still names a trace.
    uuids = _as_device_uuids(device_uuids).bind(list(traces))
    imported = imported_trace_paths(artifact)
    skipped = [str(t) for t in traces if Path(t).resolve() in imported]
    pending = [t for t in traces if str(t) not in skipped]
    if not pending:
        return _nothing_imported(skipped)
    collector = TraceFileCollector(pending, device_uuids=uuids, detail=detail)
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
    "DeviceUuids",
    "TraceFileCollector",
    "artifact_run_identity",
    "combine_captures",
    "import_traces_into_artifact",
    "imported_trace_paths",
    "parse_device_uuids",
    "trace_attachment_id",
]
