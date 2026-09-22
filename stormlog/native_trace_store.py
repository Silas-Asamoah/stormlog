"""Bounded local artifact storage for optional native trace helpers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Mapping, Sequence

from .collector_health import (
    COLLECTOR_HEALTH_DEGRADED,
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    CollectorHealthState,
)
from .correlation import (
    ATTACHMENTS_FILENAME,
    ATTACHMENTS_FORMAT,
    ATTACHMENTS_SCHEMA_VERSION,
)
from .native_trace import (
    NATIVE_HELPER_PROTOCOL_VERSION,
    NATIVE_TRACE_FORMAT,
    NATIVE_TRACE_SCHEMA_VERSION,
    NativeBackend,
    NativeTraceRecord,
)

NATIVE_TRACE_MANIFEST_FILENAME = "native_trace_manifest.json"
_HEALTH_STATUSES = {
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_DEGRADED,
    COLLECTOR_HEALTH_UNHEALTHY,
}
_FLUSH_OUTCOMES = {"complete", "partial", "failed", "not_attempted"}
_MAX_ATTACHMENT_SIDECAR_BYTES = 4 * 1024 * 1024


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("capture directory must not be a symlink")
    created = not path.exists()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if created:
        path.chmod(0o700)
    stat_result = path.stat()
    if not stat.S_ISDIR(stat_result.st_mode):
        raise ValueError("capture directory must be a directory")
    if stat_result.st_mode & 0o077:
        raise ValueError("capture directory must be owner-only")
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        raise ValueError("capture directory must be owned by the current user")


def _safe_relative_path(value: str | Path) -> Path:
    raw = str(value)
    if not raw or "\\" in raw:
        raise ValueError("artifact path must be a non-empty POSIX relative path")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise ValueError("artifact path must stay within the capture directory")
    if any(part in {"", "."} for part in posix.parts):
        raise ValueError("artifact path must not contain empty or dot components")
    return Path(*posix.parts)


def _capture_path(root: Path, relative_path: str | Path) -> tuple[Path, Path]:
    relative = _safe_relative_path(relative_path)
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve(strict=False)
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise ValueError("artifact path escapes the capture directory")
    return relative, candidate


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_relative_regular_file(
    root: Path, relative: Path
) -> tuple[BinaryIO, os.stat_result]:
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(root, directory_flags)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(relative.name, file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    handle = os.fdopen(descriptor, "rb")
    stat_result = os.fstat(handle.fileno())
    if not stat.S_ISREG(stat_result.st_mode):
        handle.close()
        raise ValueError("native trace artifact must be a regular file")
    return handle, stat_result


def native_trace_artifact_from_file(
    root: str | Path,
    relative_path: str | Path,
    *,
    kind: str = "native_trace",
    content_type: str = "application/octet-stream",
    sensitive_fields: Sequence[str] = (),
) -> NativeTraceArtifact:
    """Describe an existing owner-only regular file produced by a native helper."""
    root_path = Path(root).resolve()
    relative = _safe_relative_path(relative_path)
    _capture_path(root_path, relative)
    try:
        handle, stat_result = _open_relative_regular_file(root_path, relative)
    except OSError as exc:
        raise ValueError(
            "native trace artifact path must not contain symlinks"
        ) from exc
    with handle:
        if stat_result.st_mode & 0o077:
            raise ValueError("native trace artifact must be owner-only")
        if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
            raise ValueError("native trace artifact must be owned by the current user")
        if stat_result.st_nlink != 1:
            raise ValueError("native trace artifact must have exactly one hard link")
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return NativeTraceArtifact(
        kind=kind,
        path=relative.as_posix(),
        content_type=content_type,
        size_bytes=stat_result.st_size,
        sha256=digest.hexdigest(),
        sensitive_fields=tuple(dict.fromkeys(sensitive_fields)),
    )


def _validate_required_manifest_strings(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        if not value:
            raise ValueError(f"{name} is required")


def _validate_capture_configuration(
    *,
    started_ns: int,
    ended_ns: int | None,
    max_bytes: int,
    clock_domains: Sequence[str],
    requested_activities: Sequence[str],
    enabled_activities: Sequence[str],
) -> None:
    if started_ns < 0 or (ended_ns is not None and ended_ns < started_ns):
        raise ValueError("capture time bounds are invalid")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if not clock_domains or any(not value for value in clock_domains):
        raise ValueError("at least one non-empty clock domain is required")
    if len(clock_domains) != len(set(clock_domains)):
        raise ValueError("clock_domains must not contain duplicates")
    if not set(enabled_activities) <= set(requested_activities):
        raise ValueError("enabled activities must be requested")


def _validate_health_and_loss(
    health: CollectorHealthState, loss: "NativeTraceLoss"
) -> None:
    if health.status not in _HEALTH_STATUSES:
        raise ValueError(f"unsupported health status: {health.status}")
    healthy_has_failure = any(
        (
            health.telemetry_partial,
            health.last_error is not None,
            bool(health.consecutive_failures),
            loss.truncated,
        )
    )
    if health.status == COLLECTOR_HEALTH_HEALTHY and healthy_has_failure:
        raise ValueError("a partial, failed, or truncated trace cannot be healthy")
    if health.telemetry_partial and not health.partial_fields:
        raise ValueError("partial trace health requires partial_fields")
    if loss.truncated and not health.telemetry_partial:
        raise ValueError("truncated trace loss requires partial health")


@dataclass(frozen=True)
class NativeTraceIdentity:
    """Identity used to attach a trace without guessing missing relationships."""

    session_id: str
    pid: int
    run_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    device_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.pid <= 0:
            raise ValueError("pid must be > 0")
        if self.rank is not None and self.rank < 0:
            raise ValueError("rank must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "pid": self.pid,
            "device_id": self.device_id,
        }


@dataclass(frozen=True)
class NativeTraceArtifact:
    """Integrity and sensitivity metadata for one local trace artifact."""

    kind: str
    path: str
    content_type: str
    size_bytes: int
    sha256: str
    sensitive_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind or not self.content_type:
            raise ValueError("artifact kind and content_type are required")
        _safe_relative_path(self.path)
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be >= 0")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("artifact sha256 must be a lowercase hexadecimal digest")
        if any(not field_name for field_name in self.sensitive_fields):
            raise ValueError("sensitive_fields must contain non-empty values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "sensitive_fields": list(self.sensitive_fields),
        }


@dataclass(frozen=True)
class NativeTraceLoss:
    """Explicit producer and local-bound loss accounting."""

    delivered_records: int = 0
    dropped_records: int = 0
    bytes_written: int = 0
    bytes_dropped: int = 0
    truncated: bool = False
    flush_outcome: str = "not_attempted"

    def __post_init__(self) -> None:
        values = (
            self.delivered_records,
            self.dropped_records,
            self.bytes_written,
            self.bytes_dropped,
        )
        if any(value < 0 for value in values):
            raise ValueError("trace loss counters must be >= 0")
        if self.flush_outcome not in _FLUSH_OUTCOMES:
            raise ValueError(f"unsupported flush outcome: {self.flush_outcome}")
        if (self.dropped_records or self.bytes_dropped) and not self.truncated:
            raise ValueError("dropped data requires truncated=True")

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered_records": self.delivered_records,
            "dropped_records": self.dropped_records,
            "bytes_written": self.bytes_written,
            "bytes_dropped": self.bytes_dropped,
            "truncated": self.truncated,
            "flush_outcome": self.flush_outcome,
        }


@dataclass(frozen=True)
class NativeTraceManifest:
    """Capture metadata for a bounded trace sidecar."""

    capture_id: str
    backend: NativeBackend
    identity: NativeTraceIdentity
    helper_executable: str
    helper_version: str
    started_ns: int
    ended_ns: int | None
    clock_domains: tuple[str, ...]
    requested_activities: tuple[str, ...]
    enabled_activities: tuple[str, ...]
    max_bytes: int
    privilege: str
    target_selector: str
    health: CollectorHealthState
    loss: NativeTraceLoss
    artifacts: tuple[NativeTraceArtifact, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_required_manifest_strings(
            {
                "capture_id": self.capture_id,
                "helper_executable": self.helper_executable,
                "helper_version": self.helper_version,
                "privilege": self.privilege,
                "target_selector": self.target_selector,
            }
        )
        if self.backend not in {"cupti_activity", "rocprofiler"}:
            raise ValueError(f"unsupported native backend: {self.backend}")
        _validate_capture_configuration(
            started_ns=self.started_ns,
            ended_ns=self.ended_ns,
            max_bytes=self.max_bytes,
            clock_domains=self.clock_domains,
            requested_activities=self.requested_activities,
            enabled_activities=self.enabled_activities,
        )
        _validate_health_and_loss(self.health, self.loss)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_TRACE_SCHEMA_VERSION,
            "format": NATIVE_TRACE_FORMAT,
            "capture_id": self.capture_id,
            "backend": self.backend,
            "identity": self.identity.to_dict(),
            "helper": {
                "protocol_version": NATIVE_HELPER_PROTOCOL_VERSION,
                "executable": self.helper_executable,
                "version": self.helper_version,
            },
            "capture": {
                "started_ns": self.started_ns,
                "ended_ns": self.ended_ns,
                "clock_domains": list(self.clock_domains),
                "requested_activities": list(self.requested_activities),
                "enabled_activities": list(self.enabled_activities),
                "max_bytes": self.max_bytes,
                "privilege": self.privilege,
                "target_selector": self.target_selector,
            },
            "health": {
                "status": self.health.status,
                "partial": self.health.telemetry_partial,
                "partial_fields": list(self.health.partial_fields),
                "last_error": self.health.last_error,
                "consecutive_failures": self.health.consecutive_failures,
            },
            "loss": self.loss.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }


class BoundedTraceWriter:
    """Write complete opaque records without exceeding a local byte budget."""

    def __init__(
        self,
        root: str | Path,
        relative_path: str | Path,
        *,
        max_bytes: int,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.root = Path(root)
        _ensure_private_directory(self.root)
        self.relative_path, self.path = _capture_path(self.root, relative_path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.partial_path = self.path.with_name(f"{self.path.name}.partial")
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.bytes_dropped = 0
        self.delivered_records = 0
        self.dropped_records = 0
        self._closed = False
        self._handle = self._open_partial()

    def _open_partial(self) -> BinaryIO:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.partial_path, flags, 0o600)
        return os.fdopen(descriptor, "wb")

    def write(self, record: bytes) -> bool:
        """Write one complete record, or account for dropping all of it."""
        if self._closed:
            raise RuntimeError("trace writer is closed")
        if not isinstance(record, bytes) or not record:
            raise ValueError("record must be non-empty bytes")
        if self.bytes_written + len(record) > self.max_bytes:
            self.bytes_dropped += len(record)
            self.dropped_records += 1
            return False
        self._handle.write(record)
        self.bytes_written += len(record)
        self.delivered_records += 1
        return True

    def write_normalized(self, record: NativeTraceRecord) -> bool:
        """Write one normalized JSON record with an unambiguous delimiter."""
        payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        return self.write(f"{payload}\n".encode("utf-8"))

    def finalize(
        self,
        *,
        kind: str = "native_trace",
        content_type: str = "application/octet-stream",
        sensitive_fields: Sequence[str] = (),
    ) -> NativeTraceArtifact:
        """Flush and atomically publish the trace under its final name."""
        self._close_handle()
        if self.path.exists():
            raise FileExistsError(f"trace artifact already exists: {self.path}")
        os.replace(self.partial_path, self.path)
        self.path.chmod(0o600)
        return self._artifact(
            self.path,
            kind=kind,
            content_type=content_type,
            sensitive_fields=sensitive_fields,
        )

    def preserve_partial(
        self,
        *,
        kind: str = "native_trace_partial",
        content_type: str = "application/octet-stream",
        sensitive_fields: Sequence[str] = (),
    ) -> NativeTraceArtifact:
        """Flush a failed capture and keep its explicit `.partial` artifact."""
        self._close_handle()
        self.partial_path.chmod(0o600)
        return self._artifact(
            self.partial_path,
            kind=kind,
            content_type=content_type,
            sensitive_fields=sensitive_fields,
        )

    def loss(self, *, flush_outcome: str) -> NativeTraceLoss:
        """Return loss metadata including local byte-bound drops."""
        return NativeTraceLoss(
            delivered_records=self.delivered_records,
            dropped_records=self.dropped_records,
            bytes_written=self.bytes_written,
            bytes_dropped=self.bytes_dropped,
            truncated=self.dropped_records > 0,
            flush_outcome=flush_outcome,
        )

    def _close_handle(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def _artifact(
        self,
        path: Path,
        *,
        kind: str,
        content_type: str,
        sensitive_fields: Sequence[str],
    ) -> NativeTraceArtifact:
        return NativeTraceArtifact(
            kind=kind,
            path=path.relative_to(self.root.resolve()).as_posix(),
            content_type=content_type,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
            sensitive_fields=tuple(dict.fromkeys(sensitive_fields)),
        )

    def __enter__(self) -> "BoundedTraceWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._closed:
            self._close_handle()


def write_native_trace_manifest(
    root: str | Path,
    manifest: NativeTraceManifest,
    *,
    relative_path: str | Path = NATIVE_TRACE_MANIFEST_FILENAME,
) -> Path:
    """Atomically write an owner-only manifest inside a capture directory."""
    root_path = Path(root)
    _ensure_private_directory(root_path)
    _, path = _capture_path(root_path, relative_path)
    _atomic_json_write(path, manifest.to_dict())
    return path


def register_native_trace_attachment(
    root: str | Path,
    manifest_path: str | Path,
    manifest: NativeTraceManifest,
) -> Path:
    """Register a manifest through the existing local attachment sidecar."""
    root_path = Path(root).resolve()
    manifest_file = Path(manifest_path).resolve()
    if root_path != manifest_file and root_path not in manifest_file.parents:
        raise ValueError("manifest must be inside the attachment sidecar directory")
    relative_manifest = manifest_file.relative_to(root_path).as_posix()
    sidecar = root_path / ATTACHMENTS_FILENAME
    payload = _load_attachment_sidecar(sidecar)
    candidate = {
        "attachment_id": f"native-trace:{manifest.capture_id}",
        "title": f"Native trace {manifest.capture_id}",
        "kind": "native_trace_manifest",
        "path": relative_manifest,
        "run_id": manifest.identity.run_id,
        "session_id": manifest.identity.session_id,
        "job_id": manifest.identity.job_id,
        "rank": manifest.identity.rank,
        "start_ns": manifest.started_ns,
        "end_ns": manifest.ended_ns,
        "storage": "reference",
        "source_namespace": "stormlog.native_trace",
        "source_ref": manifest.capture_id,
        "metadata": {
            "format": NATIVE_TRACE_FORMAT,
            "schema_version": NATIVE_TRACE_SCHEMA_VERSION,
            "backend": manifest.backend,
            "health": manifest.health.status,
            "truncated": manifest.loss.truncated,
            "dropped_records": manifest.loss.dropped_records,
        },
    }
    rows = payload["attachments"]
    existing = next(
        (row for row in rows if row.get("attachment_id") == candidate["attachment_id"]),
        None,
    )
    if existing is not None and existing != candidate:
        raise ValueError("attachment_id already refers to different evidence")
    if existing is None:
        rows.append(candidate)
    _atomic_json_write(sidecar, payload)
    return sidecar


def _load_attachment_sidecar(path: Path) -> dict[str, Any]:
    content = _read_attachment_sidecar(path)
    if content is None:
        return _empty_attachment_sidecar()
    payload = _decode_attachment_sidecar(content)
    if payload.get("schema_version") != ATTACHMENTS_SCHEMA_VERSION:
        raise ValueError("unsupported attachment sidecar schema version")
    if payload.get("format") != ATTACHMENTS_FORMAT:
        raise ValueError("unrecognized attachment sidecar format")
    rows = payload.get("attachments")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("attachment sidecar attachments must be objects")
    return payload


def _empty_attachment_sidecar() -> dict[str, Any]:
    return {
        "schema_version": ATTACHMENTS_SCHEMA_VERSION,
        "format": ATTACHMENTS_FORMAT,
        "attachments": [],
    }


def _read_attachment_sidecar(path: Path) -> bytes | None:
    try:
        handle, stat_result = _open_relative_regular_file(path.parent, Path(path.name))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("attachment sidecar path must not be a symlink") from exc
    with handle:
        _validate_attachment_sidecar_file(stat_result)
        content = handle.read(_MAX_ATTACHMENT_SIDECAR_BYTES + 1)
        if len(content) > _MAX_ATTACHMENT_SIDECAR_BYTES:
            raise ValueError("attachment sidecar exceeds the maximum supported size")
    return content


def _validate_attachment_sidecar_file(stat_result: os.stat_result) -> None:
    if stat_result.st_mode & 0o077:
        raise ValueError("attachment sidecar must be owner-only")
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        raise ValueError("attachment sidecar must be owned by the current user")
    if stat_result.st_nlink != 1:
        raise ValueError("attachment sidecar must have exactly one hard link")
    if stat_result.st_size > _MAX_ATTACHMENT_SIDECAR_BYTES:
        raise ValueError("attachment sidecar exceeds the maximum supported size")


def _decode_attachment_sidecar(content: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("attachment sidecar must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("attachment sidecar must contain an object")
    return payload


__all__ = [
    "NATIVE_TRACE_MANIFEST_FILENAME",
    "BoundedTraceWriter",
    "NativeTraceArtifact",
    "NativeTraceIdentity",
    "NativeTraceLoss",
    "NativeTraceManifest",
    "native_trace_artifact_from_file",
    "register_native_trace_attachment",
    "write_native_trace_manifest",
]
