"""Append-only telemetry sink with rollover and retention bounds."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, TextIO, cast

from .session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INTERRUPTED,
    SESSION_STATUS_RUNNING,
    SessionSummary,
    create_session_summary,
    now_ns,
    session_summary_from_dict,
    session_summary_to_dict,
    update_session_summary,
)

if TYPE_CHECKING:
    from .telemetry_rollups import RollupCoverage

MANIFEST_FILENAME = "manifest.json"
SEGMENT_PREFIX = "segment-"
SEGMENT_SUFFIX = ".jsonl"
SINK_SCHEMA_VERSION = 2
_LOGGER = logging.getLogger(__name__)


@dataclass
class TelemetrySinkConfig:
    """Runtime policy for append-only telemetry persistence."""

    root_dir: Path
    flush_every_events: int = 50
    flush_every_seconds: float = 2.0
    rollover_max_bytes: int = 64 * 1024 * 1024
    rollover_max_events: int = 10000
    retention_max_files: int = 8
    retention_max_total_bytes: int = 512 * 1024 * 1024
    write_rollups: bool = True
    rollup_window_seconds: int = 60

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if self.flush_every_events <= 0:
            raise ValueError("flush_every_events must be >= 1")
        if self.flush_every_seconds <= 0:
            raise ValueError("flush_every_seconds must be > 0")
        if self.rollover_max_bytes <= 0:
            raise ValueError("rollover_max_bytes must be >= 1")
        if self.rollover_max_events <= 0:
            raise ValueError("rollover_max_events must be >= 1")
        if self.retention_max_files <= 0:
            raise ValueError("retention_max_files must be >= 1")
        if self.retention_max_total_bytes <= 0:
            raise ValueError("retention_max_total_bytes must be >= 1")
        if self.retention_max_total_bytes < self.rollover_max_bytes:
            raise ValueError("retention_max_total_bytes must be >= rollover_max_bytes")
        if self.rollup_window_seconds <= 0:
            raise ValueError("rollup_window_seconds must be >= 1")


@dataclass
class TelemetrySinkSegment:
    """One JSONL segment tracked by the append-only sink manifest."""

    filename: str
    event_count: int
    size_bytes: int
    closed: bool
    session_id: str | None = None


@dataclass
class TelemetrySinkManifest:
    """Parsed append-only sink manifest with session ledger and segments."""

    schema_version: int
    format: str
    sessions: list[SessionSummary] = field(default_factory=list)
    segments: list[TelemetrySinkSegment] = field(default_factory=list)


class AppendOnlyTelemetrySink:
    """Write telemetry records to newline-delimited JSON segments."""

    def __init__(self, config: TelemetrySinkConfig) -> None:
        self.config = config
        self.root_dir = config.root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root_dir / MANIFEST_FILENAME
        self._segments: list[TelemetrySinkSegment] = []
        self._sessions: dict[str, SessionSummary] = {}
        self._active_session_id: str | None = None
        self._next_segment_index = 1
        self._buffer: list[str] = []
        self._buffered_event_count = 0
        self._handle: TextIO | None = None
        self._lock = threading.Lock()
        self._flush_stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._last_flush_monotonic = time.monotonic()
        self._closed = False
        self._rollover_count = 0
        self._pruned_segment_count = 0
        self._pruned_bytes = 0
        self._load_existing_state()

    def start_session(self, summary: SessionSummary | None = None) -> SessionSummary:
        """Register the active session for subsequent records."""
        with self._lock:
            if self._active_session_id is not None:
                active = self._sessions.get(self._active_session_id)
                if active is not None:
                    return active

            resolved = summary or create_session_summary(
                source="stormlog.telemetry_sink",
                status=SESSION_STATUS_RUNNING,
            )
            resolved = update_session_summary(
                resolved,
                status=SESSION_STATUS_RUNNING,
                ended_at_ns=None,
            )
            self._sessions[resolved.session_id] = resolved
            self._active_session_id = resolved.session_id
            self._closed = False
            self._write_manifest_locked()
            return resolved

    def current_session(self) -> SessionSummary | None:
        """Return the active session summary, if any."""
        with self._lock:
            if self._active_session_id is None:
                return None
            return self._sessions.get(self._active_session_id)

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._ensure_flush_thread_locked()
            self._closed = False
            self._ensure_active_session_locked(record)
            self._buffer.append(json.dumps(dict(record), sort_keys=True) + "\n")
            self._buffered_event_count += 1
            self._flush_locked(force=False)

    def flush(self, force: bool = False) -> None:
        with self._lock:
            self._flush_locked(force=force)

    def close(self, session_status: str = SESSION_STATUS_COMPLETED) -> None:
        rollup_inputs: tuple[TelemetrySinkManifest, RollupCoverage] | None = None
        try:
            with self._lock:
                self._flush_locked(force=True)
                if self._handle is not None:
                    self._handle.close()
                    self._handle = None
                current = self._current_segment()
                if current is not None and not current.closed:
                    current.closed = True
                if self._active_session_id is not None:
                    active = self._sessions.get(self._active_session_id)
                    if active is not None:
                        self._sessions[self._active_session_id] = (
                            update_session_summary(
                                active,
                                status=session_status,
                                ended_at_ns=now_ns(),
                            )
                        )
                self._active_session_id = None
                self._write_manifest_locked()
                rollup_inputs = self._rollup_inputs_locked()
                self._closed = True
            if rollup_inputs is not None:
                self._write_rollups(*rollup_inputs)
        finally:
            self._stop_flush_thread()

    def get_diagnostics(self) -> dict[str, int]:
        """Return runtime retention and rollover diagnostics."""
        with self._lock:
            return self._diagnostics_locked()

    def _ensure_active_session_locked(self, record: Mapping[str, Any]) -> None:
        record_session_id = record.get("session_id")
        if record_session_id is not None and not isinstance(record_session_id, str):
            raise ValueError("telemetry sink record session_id must be a string")
        if self._active_session_id is None:
            resolved = create_session_summary(
                source="stormlog.telemetry_sink",
                status=SESSION_STATUS_RUNNING,
                session_id=(
                    record_session_id if isinstance(record_session_id, str) else None
                ),
            )
            self._sessions[resolved.session_id] = resolved
            self._active_session_id = resolved.session_id
            self._write_manifest_locked()
            return
        if (
            isinstance(record_session_id, str)
            and record_session_id != self._active_session_id
        ):
            raise ValueError(
                "telemetry sink record session_id does not match the active session"
            )

    def _flush_locked(self, force: bool) -> None:
        if not self._buffer:
            return

        now = time.monotonic()
        if not force:
            if self._buffered_event_count < self.config.flush_every_events and (
                now - self._last_flush_monotonic < self.config.flush_every_seconds
            ):
                return

        current = self._ensure_current_segment_locked()
        payload = "".join(self._buffer)
        payload_bytes = payload.encode("utf-8")
        handle = self._ensure_handle_locked(current)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

        current.event_count += self._buffered_event_count
        current.size_bytes += len(payload_bytes)
        self._buffer.clear()
        self._buffered_event_count = 0
        self._last_flush_monotonic = now

        self._rollover_locked(current)
        self._prune_retention_locked()
        self._write_manifest_locked()

    def _rollover_locked(self, current: TelemetrySinkSegment) -> None:
        if (
            current.event_count < self.config.rollover_max_events
            and current.size_bytes < self.config.rollover_max_bytes
        ):
            return
        current.closed = True
        self._rollover_count += 1
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _prune_retention_locked(self) -> None:
        while True:
            total_bytes = sum(segment.size_bytes for segment in self._segments)
            over_file_limit = len(self._segments) > self.config.retention_max_files
            over_size_limit = total_bytes > self.config.retention_max_total_bytes
            if not over_file_limit and not over_size_limit:
                return

            removable = next(
                (segment for segment in self._segments if segment.closed),
                None,
            )
            if removable is None:
                return

            path = self.root_dir / removable.filename
            if path.exists():
                path.unlink()
            self._segments.remove(removable)
            self._pruned_segment_count += 1
            self._pruned_bytes += removable.size_bytes

    def _current_segment(self) -> TelemetrySinkSegment | None:
        if not self._segments:
            return None
        current = self._segments[-1]
        if current.closed:
            return None
        if (
            self._active_session_id is not None
            and current.session_id != self._active_session_id
        ):
            return None
        return current

    def _ensure_flush_thread_locked(self) -> None:
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._flush_stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._run_flush_loop,
            name="stormlog-telemetry-sink-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def _stop_flush_thread(self) -> None:
        thread = self._flush_thread
        if thread is None:
            return
        self._flush_stop_event.set()
        thread.join(timeout=self.config.flush_every_seconds + 1.0)
        self._flush_thread = None

    def _run_flush_loop(self) -> None:
        while not self._flush_stop_event.wait(timeout=self.config.flush_every_seconds):
            with self._lock:
                self._flush_locked(force=False)

    def _ensure_current_segment_locked(self) -> TelemetrySinkSegment:
        current = self._current_segment()
        if current is not None:
            return current

        segment = TelemetrySinkSegment(
            filename=f"{SEGMENT_PREFIX}{self._next_segment_index:06d}{SEGMENT_SUFFIX}",
            event_count=0,
            size_bytes=0,
            closed=False,
            session_id=self._active_session_id,
        )
        self._next_segment_index += 1
        self._segments.append(segment)
        return segment

    def _ensure_handle_locked(self, current: TelemetrySinkSegment) -> TextIO:
        if self._handle is None:
            segment_path = self.root_dir / current.filename
            self._recover_segment_tail_locked(segment_path, current)
            self._handle = segment_path.open("a", encoding="utf-8")
        return self._handle

    def _load_existing_state(self) -> None:
        discovered = _discover_segment_paths(self.root_dir)
        manifest = read_telemetry_sink_manifest(self.root_dir)

        if manifest is not None:
            self._segments = self._merge_segment_state(manifest.segments, discovered)
            self._sessions = {
                summary.session_id: summary for summary in manifest.sessions
            }
            manifest_needs_rewrite = self._interrupt_running_sessions()
            self._next_segment_index = self._compute_next_segment_index()
            if manifest.schema_version != SINK_SCHEMA_VERSION:
                manifest_needs_rewrite = True
            if manifest_needs_rewrite:
                self._write_manifest_locked()
            return

        if not discovered:
            return

        for path in discovered:
            self._segments.append(
                TelemetrySinkSegment(
                    filename=path.name,
                    event_count=self._count_records(path),
                    size_bytes=path.stat().st_size,
                    closed=True,
                    session_id=None,
                )
            )
        self._next_segment_index = self._compute_next_segment_index()
        self._write_manifest_locked()

    def _interrupt_running_sessions(self) -> bool:
        manifest_needs_rewrite = False
        for session_id, summary in list(self._sessions.items()):
            if summary.status == SESSION_STATUS_RUNNING:
                self._sessions[session_id] = update_session_summary(
                    summary,
                    status=SESSION_STATUS_INTERRUPTED,
                    ended_at_ns=now_ns(),
                )
                manifest_needs_rewrite = True
        for segment in self._segments:
            if segment.session_id in self._sessions and (
                self._sessions[segment.session_id].status == SESSION_STATUS_INTERRUPTED
            ):
                segment.closed = True
        return manifest_needs_rewrite

    def _compute_next_segment_index(self) -> int:
        max_index = 0
        for segment in self._segments:
            stem = Path(segment.filename).stem
            if not stem.startswith(SEGMENT_PREFIX):
                continue
            suffix = stem[len(SEGMENT_PREFIX) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return max_index + 1

    def _write_manifest_locked(self) -> None:
        payload = {
            "schema_version": SINK_SCHEMA_VERSION,
            "format": "stormlog.append_only_telemetry_sink",
            "sessions": [
                session_summary_to_dict(summary)
                for summary in sorted(
                    self._sessions.values(),
                    key=lambda session: (session.started_at_ns, session.session_id),
                )
            ],
            "segments": [asdict(segment) for segment in self._segments],
        }
        temp_path = self._manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self._manifest_path)

    def _diagnostics_locked(self) -> dict[str, int]:
        retained_bytes = sum(segment.size_bytes for segment in self._segments)
        return {
            "rollover_count": self._rollover_count,
            "pruned_segment_count": self._pruned_segment_count,
            "pruned_bytes": self._pruned_bytes,
            "final_retained_files": len(self._segments),
            "final_retained_bytes": retained_bytes,
        }

    def _rollup_inputs_locked(
        self,
    ) -> tuple[TelemetrySinkManifest, RollupCoverage] | None:
        if not self.config.write_rollups:
            return None
        from .telemetry_rollups import rollup_coverage_from_manifest

        manifest = TelemetrySinkManifest(
            schema_version=SINK_SCHEMA_VERSION,
            format="stormlog.append_only_telemetry_sink",
            sessions=list(self._sessions.values()),
            segments=list(self._segments),
        )
        coverage = rollup_coverage_from_manifest(
            manifest,
            pruned_segment_count=self._pruned_segment_count,
            pruned_bytes=self._pruned_bytes,
        )
        return manifest, coverage

    def _write_rollups(
        self,
        manifest: TelemetrySinkManifest,
        coverage: RollupCoverage,
    ) -> None:
        try:
            from .telemetry import load_telemetry_sessions
            from .telemetry_rollups import (
                build_telemetry_rollups,
                write_telemetry_rollups,
            )

            sessions = load_telemetry_sessions(self.root_dir)
            rollups = build_telemetry_rollups(
                sessions,
                manifest,
                window_duration_ns=(self.config.rollup_window_seconds * 1_000_000_000),
                coverage=coverage,
            )
            write_telemetry_rollups(self.root_dir, rollups)
        except Exception as exc:
            _LOGGER.warning("telemetry rollup write failed: %s", exc, exc_info=True)

    @staticmethod
    def _count_records(path: Path) -> int:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _recover_segment_tail_locked(
        self,
        segment_path: Path,
        current: TelemetrySinkSegment,
    ) -> None:
        if not segment_path.exists():
            current.event_count = 0
            current.size_bytes = 0
            return

        payload = segment_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            last_newline = payload.rfind(b"\n")
            payload = payload[: last_newline + 1] if last_newline >= 0 else b""
            segment_path.write_bytes(payload)

        current.size_bytes = len(payload)
        current.event_count = self._count_records(segment_path)

    def _merge_segment_state(
        self,
        manifest_segments: list[TelemetrySinkSegment],
        discovered_segments: list[Path],
    ) -> list[TelemetrySinkSegment]:
        manifest_by_name = {
            segment.filename: TelemetrySinkSegment(
                filename=segment.filename,
                event_count=segment.event_count,
                size_bytes=segment.size_bytes,
                closed=segment.closed,
                session_id=segment.session_id,
            )
            for segment in manifest_segments
        }
        discovered_by_name = {path.name: path for path in discovered_segments}
        merged_names = sorted(set(manifest_by_name) | set(discovered_by_name))
        merged: list[TelemetrySinkSegment] = []

        for name in merged_names:
            manifest_segment = manifest_by_name.get(name)
            if manifest_segment is not None:
                merged.append(manifest_segment)
                continue

            path = discovered_by_name[name]
            merged.append(
                TelemetrySinkSegment(
                    filename=name,
                    event_count=self._count_records(path),
                    size_bytes=path.stat().st_size,
                    closed=True,
                    session_id=None,
                )
            )

        return merged


def resolve_telemetry_sink_segment_paths(path: str | Path) -> list[Path]:
    """Resolve append-only sink inputs to ordered JSONL segment paths."""
    resolved_path = Path(path)
    if resolved_path.is_file():
        if resolved_path.suffix == SEGMENT_SUFFIX:
            return [resolved_path]
        if resolved_path.name == MANIFEST_FILENAME:
            manifest = read_telemetry_sink_manifest(resolved_path)
            manifest_segments = _manifest_segment_paths(resolved_path.parent, manifest)
            return _merge_segment_paths(
                manifest_segments,
                _discover_segment_paths(resolved_path.parent),
            )
        return []

    if not resolved_path.is_dir():
        return []

    manifest = read_telemetry_sink_manifest(resolved_path)
    if manifest is not None:
        manifest_segments = _manifest_segment_paths(resolved_path, manifest)
        return _merge_segment_paths(
            manifest_segments,
            _discover_segment_paths(resolved_path),
        )
    return _discover_segment_paths(resolved_path)


def resolve_telemetry_sink_manifest_path(path: str | Path) -> Path | None:
    """Resolve a file or directory path to a sink manifest path, if present."""
    resolved = Path(path)
    if resolved.is_file():
        if resolved.name == MANIFEST_FILENAME:
            return resolved
        if resolved.suffix == SEGMENT_SUFFIX:
            candidate = resolved.parent / MANIFEST_FILENAME
            return candidate if candidate.exists() else None
        return None
    if resolved.is_dir():
        candidate = resolved / MANIFEST_FILENAME
        return candidate if candidate.exists() else None
    return None


def _coerce_manifest_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    try:
        coerced = int(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if minimum is not None and coerced < minimum:
        return default
    return int(coerced)


def _coerce_manifest_entries(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def read_telemetry_sink_manifest(path: str | Path) -> TelemetrySinkManifest | None:
    """Read a sink manifest from a sink directory, manifest file, or segment file."""
    manifest_path = resolve_telemetry_sink_manifest_path(path)
    if manifest_path is None or not manifest_path.exists():
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, Mapping):
        return None

    schema_version = _coerce_manifest_int(
        payload.get("schema_version", 1),
        default=1,
        minimum=1,
    )
    fmt_value = payload.get("format", "stormlog.append_only_telemetry_sink")
    fmt = (
        fmt_value
        if isinstance(fmt_value, str) and fmt_value
        else "stormlog.append_only_telemetry_sink"
    )
    sessions = _manifest_sessions(payload, schema_version)
    segments = _manifest_segments(payload)

    return TelemetrySinkManifest(
        schema_version=schema_version,
        format=fmt,
        sessions=sessions,
        segments=segments,
    )


def _discover_segment_paths(root_dir: Path) -> list[Path]:
    return sorted(root_dir.glob(f"{SEGMENT_PREFIX}*{SEGMENT_SUFFIX}"))


def _merge_segment_paths(
    manifest_segments: list[Path],
    discovered_segments: list[Path],
) -> list[Path]:
    merged_by_name = {path.name: path for path in discovered_segments}
    for path in manifest_segments:
        merged_by_name[path.name] = path
    return [merged_by_name[name] for name in sorted(merged_by_name)]


def _manifest_sessions(
    payload: Mapping[str, Any], schema_version: int
) -> list[SessionSummary]:
    sessions: list[SessionSummary] = []
    if schema_version >= 2:
        for raw_session in _coerce_manifest_entries(payload.get("sessions", [])):
            if not isinstance(raw_session, Mapping):
                continue
            try:
                sessions.append(session_summary_from_dict(raw_session))
            except Exception:
                continue

    return sessions


def _manifest_segments(payload: Mapping[str, Any]) -> list[TelemetrySinkSegment]:
    segments: list[TelemetrySinkSegment] = []
    for raw_segment in _coerce_manifest_entries(payload.get("segments", [])):
        if not isinstance(raw_segment, Mapping):
            continue
        filename = raw_segment.get("filename")
        if not isinstance(filename, str):
            continue
        try:
            segments.append(
                TelemetrySinkSegment(
                    filename=filename,
                    event_count=_coerce_manifest_int(
                        raw_segment.get("event_count", 0),
                        default=0,
                        minimum=0,
                    ),
                    size_bytes=_coerce_manifest_int(
                        raw_segment.get("size_bytes", 0),
                        default=0,
                        minimum=0,
                    ),
                    closed=bool(raw_segment.get("closed", False)),
                    session_id=(
                        raw_segment.get("session_id")
                        if isinstance(raw_segment.get("session_id"), str)
                        else None
                    ),
                )
            )
        except Exception:
            continue

    return segments


def _manifest_segment_paths(
    root: Path, manifest: TelemetrySinkManifest | None
) -> list[Path]:
    return [
        root / segment.filename
        for segment in (manifest.segments if manifest is not None else [])
        if (root / segment.filename).exists()
    ]


__all__ = [
    "AppendOnlyTelemetrySink",
    "MANIFEST_FILENAME",
    "SEGMENT_PREFIX",
    "SEGMENT_SUFFIX",
    "SINK_SCHEMA_VERSION",
    "TelemetrySinkConfig",
    "TelemetrySinkManifest",
    "TelemetrySinkSegment",
    "read_telemetry_sink_manifest",
    "resolve_telemetry_sink_manifest_path",
    "resolve_telemetry_sink_segment_paths",
]
