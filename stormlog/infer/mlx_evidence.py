"""Versioned MLX server evidence: manifest, bounded loaders and identity rules.

An evidence directory holds one same-process capture of an explicitly launched
MLX server. Native allocator samples stay in framework telemetry v4 inside the
capture's own sink directory; they are never appended to inference JSONL,
where the correlation decoder would read schema version 4 as context
transport. Reading needs neither MLX nor a server package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .errors import InferInputError
from .host_clock import is_boot_qualified, wall_clock_domain
from .omlx_stats import validate_stats_record

if TYPE_CHECKING:
    from stormlog.session import SessionSummary
    from stormlog.telemetry import TelemetryEvent

FORMAT = "stormlog.infer.mlx_evidence"
SCHEMA_VERSION = 1
MANIFEST_NAME = "mlx_evidence.json"
STATS_NAME = "stats.jsonl"
TELEMETRY_DIR = "telemetry"
ENGINES = ("omlx", "mlx-lm", "mlx-audio")
MODES = ("mlx-serve", "embedded")
CAPTURE_STATUSES = ("complete", "interrupted", "incomplete")
VERSION_PROVENANCE = (
    "installed_distribution_metadata",
    "user_declared",
    "unknown",
)
FILE_ROLES = ("native_telemetry", "omlx_stats")
ENDPOINT_DECLARATIONS = ("launch_arguments", "operator", "unknown")

MAX_MANIFEST_BYTES = 1024**2
MAX_FILE_BYTES = 256 * 1024**2
MAX_TOTAL_BYTES = 1024**3
MAX_FILES = 512
MAX_STATS_RECORDS = 200_000
MAX_NATIVE_RECORDS = 100_000
MAX_RECORD_BYTES = 1024**2
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_LIMITATIONS = (
    "Native counters describe this server process's MLX allocator only; they "
    "are not per-request memory or total device capacity.",
    "Sampled maxima are sampled process peaks, not exact request or window peaks.",
    "A matching run ID does not prove that a client's requests reached this "
    "process.",
)

_TOP_KEYS = frozenset(
    {
        "format",
        "schema_version",
        "run_id",
        "producer",
        "engine",
        "process",
        "clock",
        "capture",
        "server",
        "endpoint",
        "files",
        "limitations",
    }
)


@dataclass(frozen=True)
class LoadedEvidence:
    """One verified evidence directory."""

    directory: Path
    manifest: dict[str, Any]
    events: tuple["TelemetryEvent", ...] = ()
    session: "SessionSummary | None" = None
    stats_records: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def engine(self) -> str:
        return str(self.manifest["engine"]["name"])

    @property
    def clock_domain(self) -> str:
        return str(self.manifest["clock"]["domain"])

    @property
    def capture_status(self) -> str:
        return str(self.manifest["capture"]["status"])

    @property
    def identity(self) -> tuple[Any, ...]:
        process = self.manifest["process"]
        return (
            process["host"],
            process["boot_id"],
            process["pid"],
            process["start_time_ns"],
            process["session_id"],
        )

    def label(self) -> str:
        process = self.manifest["process"]
        return f"{self.engine} pid {process['pid']} on {process['host']}"


# --- manifest construction -------------------------------------------------


def build_manifest(
    *,
    run_id: str,
    engine: str,
    mode: str,
    producer_version: str,
    server_version: str | None,
    version_provenance: str,
    entrypoint: str | None,
    source_revision: str | None,
    host: str,
    boot_id: str | None,
    pid: int,
    start_time_ns: int | None,
    start_time_source: str | None,
    session_id: str | None,
    capture_status: str,
    started_at_ns: int,
    ended_at_ns: int | None,
    interval_seconds: float,
    exit_code: int | None,
    argv: Sequence[str],
    config: Mapping[str, str],
    endpoint_host: str | None,
    endpoint_port: int | None,
    endpoint_declared_by: str,
    files: Sequence[Mapping[str, Any]],
    stats_sampling: bool,
    limitations: Sequence[str] = DEFAULT_LIMITATIONS,
    endpoint_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": {
            "name": "stormlog.infer.mlx_server",
            "version": producer_version,
            "mode": mode,
        },
        "engine": {
            "name": engine,
            "version": server_version,
            "version_provenance": version_provenance,
            "entrypoint": entrypoint,
            "source_revision": source_revision,
        },
        "process": {
            "host": host,
            "boot_id": boot_id,
            "pid": pid,
            "start_time_ns": start_time_ns,
            "start_time_source": start_time_source,
            "session_id": session_id,
        },
        "clock": {
            "domain": wall_clock_domain(host, boot_id),
            "kind": "wall",
            "boot_qualified": boot_id is not None,
        },
        "capture": {
            "status": capture_status,
            "started_at_ns": started_at_ns,
            "ended_at_ns": ended_at_ns,
            "interval_seconds": interval_seconds,
            "sampling": "passive",
            "memory_scope": "process_allocator",
            "exit_code": exit_code,
            "stats_sampling": stats_sampling,
        },
        "server": {"argv": list(argv), "config": dict(config)},
        "endpoint": {
            "host": endpoint_host,
            "port": endpoint_port,
            "declared_by": endpoint_declared_by,
            "routing_proof": (
                dict(endpoint_proof) if endpoint_proof is not None else None
            ),
        },
        "files": [dict(item) for item in files],
        "limitations": list(limitations),
    }
    validate_manifest(manifest)
    return manifest


def file_record(root: Path, relative: str, role: str) -> dict[str, Any]:
    """Digest one file under ``root``; call only after its writer has closed."""
    path = root / relative
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            size += len(chunk)
            digest.update(chunk)
    return {
        "role": role,
        "path": relative,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def collect_files(root: Path, *, stats: bool) -> list[dict[str, Any]]:
    """List the native sink files (and optional stats) with digests."""
    records = []
    telemetry = root / TELEMETRY_DIR
    if telemetry.is_dir():
        for path in sorted(telemetry.rglob("*")):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(root).as_posix()
                records.append(file_record(root, relative, "native_telemetry"))
    if stats and (root / STATS_NAME).is_file():
        records.append(file_record(root, STATS_NAME, "omlx_stats"))
    return records


def write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Validate, then replace the manifest atomically."""
    validate_manifest(manifest)
    target = root / MANIFEST_NAME
    temporary = root / (MANIFEST_NAME + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


# --- manifest validation ---------------------------------------------------


def _fail(message: str) -> InferInputError:
    return InferInputError(f"MLX evidence manifest: {message}")


def _mapping(parent: Mapping[str, Any], key: str, keys: frozenset[str]) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise _fail(f"{key} must be an object")
    unknown = set(value) - keys
    missing = keys - set(value)
    if unknown or missing:
        raise _fail(
            f"{key} has unsupported keys {sorted(unknown)} or lacks {sorted(missing)}"
        )
    return value


def _string(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise _fail(f"{name} must be a nonempty string")


def _integer(
    value: Any, name: str, *, minimum: int = 0, nullable: bool = False
) -> None:
    if value is None and nullable:
        return
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an integer >= {minimum}")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Strictly validate a manifest; unknown versions and keys are errors."""
    if not isinstance(manifest, Mapping) or manifest.get("format") != FORMAT:
        raise _fail("not an MLX server evidence manifest")
    version = manifest.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise _fail(f"unsupported schema_version {version!r}")
    unknown = set(manifest) - _TOP_KEYS
    missing = _TOP_KEYS - set(manifest)
    if unknown or missing:
        raise _fail(f"unsupported keys {sorted(unknown)} or missing {sorted(missing)}")
    _string(manifest["run_id"], "run_id")
    _validate_identity(manifest)
    _validate_capture(manifest)
    _validate_server(manifest)
    _validate_files(manifest)
    limitations = manifest["limitations"]
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise _fail("limitations must be a list of strings")


def _validate_identity(manifest: Mapping[str, Any]) -> None:
    producer = _mapping(manifest, "producer", frozenset({"name", "version", "mode"}))
    _string(producer["name"], "producer.name")
    _string(producer["version"], "producer.version")
    if producer["mode"] not in MODES:
        raise _fail("producer.mode is unsupported")
    engine = _mapping(
        manifest,
        "engine",
        frozenset(
            {"name", "version", "version_provenance", "entrypoint", "source_revision"}
        ),
    )
    if engine["name"] not in ENGINES:
        raise _fail("engine.name is unsupported")
    _string(engine["version"], "engine.version", nullable=True)
    if engine["version_provenance"] not in VERSION_PROVENANCE:
        raise _fail("engine.version_provenance is unsupported")
    if (engine["version"] is None) != (engine["version_provenance"] == "unknown"):
        raise _fail("an unknown version must say so, and a known one needs provenance")
    _string(engine["entrypoint"], "engine.entrypoint", nullable=True)
    _string(engine["source_revision"], "engine.source_revision", nullable=True)
    process = _mapping(
        manifest,
        "process",
        frozenset(
            {
                "host",
                "boot_id",
                "pid",
                "start_time_ns",
                "start_time_source",
                "session_id",
            }
        ),
    )
    _string(process["host"], "process.host")
    _string(process["boot_id"], "process.boot_id", nullable=True)
    _integer(process["pid"], "process.pid", minimum=1)
    _integer(
        process["start_time_ns"], "process.start_time_ns", minimum=1, nullable=True
    )
    _string(process["start_time_source"], "process.start_time_source", nullable=True)
    _string(process["session_id"], "process.session_id", nullable=True)
    clock = _mapping(manifest, "clock", frozenset({"domain", "kind", "boot_qualified"}))
    expected = wall_clock_domain(process["host"], process["boot_id"])
    if (
        clock["domain"] != expected
        or clock["kind"] != "wall"
        or type(clock["boot_qualified"]) is not bool
        or clock["boot_qualified"] != is_boot_qualified(expected)
    ):
        raise _fail("clock does not match the process host and boot")


def _validate_capture(manifest: Mapping[str, Any]) -> None:
    capture = _mapping(
        manifest,
        "capture",
        frozenset(
            {
                "status",
                "started_at_ns",
                "ended_at_ns",
                "interval_seconds",
                "sampling",
                "memory_scope",
                "exit_code",
                "stats_sampling",
            }
        ),
    )
    if capture["status"] not in CAPTURE_STATUSES:
        raise _fail("capture.status is unsupported")
    _integer(capture["started_at_ns"], "capture.started_at_ns")
    _integer(capture["ended_at_ns"], "capture.ended_at_ns", nullable=True)
    ended = capture["ended_at_ns"]
    if ended is not None and ended < capture["started_at_ns"]:
        raise _fail("capture ends before it starts")
    _validate_capture_options(capture)
    if ended is None:
        raise _fail("a finalized capture must record its end")


def _validate_capture_options(capture: Mapping[str, Any]) -> None:
    interval = capture["interval_seconds"]
    if (
        type(interval) not in (int, float)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise _fail("capture.interval_seconds must be positive and finite")
    if capture["sampling"] != "passive" or (
        capture["memory_scope"] != "process_allocator"
    ):
        raise _fail("capture must be passive process-allocator sampling")
    if capture["exit_code"] is not None and type(capture["exit_code"]) is not int:
        raise _fail("capture.exit_code must be an integer or null")
    if type(capture["stats_sampling"]) is not bool:
        raise _fail("capture.stats_sampling must be a boolean")


def _validate_server(manifest: Mapping[str, Any]) -> None:
    server = _mapping(manifest, "server", frozenset({"argv", "config"}))
    if not isinstance(server["argv"], list) or not all(
        isinstance(item, str) for item in server["argv"]
    ):
        raise _fail("server.argv must be a list of strings")
    _validate_server_config(server["config"])
    _validate_endpoint(manifest)


def _validate_server_config(config: Any) -> None:
    if not isinstance(config, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in config.items()
    ):
        raise _fail("server.config must map strings to strings")


def _validate_endpoint(manifest: Mapping[str, Any]) -> None:
    endpoint = _mapping(
        manifest,
        "endpoint",
        frozenset({"host", "port", "declared_by", "routing_proof"}),
    )
    _string(endpoint["host"], "endpoint.host", nullable=True)
    _integer(endpoint["port"], "endpoint.port", minimum=1, nullable=True)
    if endpoint["declared_by"] not in ENDPOINT_DECLARATIONS:
        raise _fail("endpoint.declared_by is unsupported")
    if endpoint["port"] is not None and endpoint["port"] > 65535:
        raise _fail("endpoint port exceeds 65535")
    _validate_routing_proof(manifest, endpoint)


def _validate_files(manifest: Mapping[str, Any]) -> None:
    files = manifest["files"]
    if not isinstance(files, list) or len(files) > MAX_FILES:
        raise _fail("files must be a bounded list")
    seen: set[str] = set()
    total = 0
    for entry in files:
        _validate_file_entry(entry)
        if entry["path"] in seen:
            raise _fail("duplicate file path")
        seen.add(entry["path"])
        total += entry["bytes"]
    if total > MAX_TOTAL_BYTES:
        raise _fail("files exceed the total bound")
    stats = [entry for entry in files if entry["role"] == "omlx_stats"]
    if len(stats) > 1 or bool(stats) != manifest["capture"]["stats_sampling"]:
        raise _fail("stats_sampling must match exactly one stats sidecar")


def _validate_file_entry(entry: Any) -> None:
    if not isinstance(entry, dict) or set(entry) != {"role", "path", "bytes", "sha256"}:
        raise _fail("each file needs exactly role, path, bytes and sha256")
    if entry["role"] not in FILE_ROLES:
        raise _fail("file role is unsupported")
    parts = _relative_parts(entry["path"])
    if entry["role"] == "native_telemetry" and parts[0] != TELEMETRY_DIR:
        raise _fail("native telemetry must be under the telemetry directory")
    if entry["role"] == "omlx_stats" and entry["path"] != STATS_NAME:
        raise _fail("statistics sidecar must be stats.jsonl")
    _validate_file_integrity(entry)


def _validate_file_integrity(entry: Mapping[str, Any]) -> None:
    _integer(entry["bytes"], "file bytes")
    if entry["bytes"] > MAX_FILE_BYTES:
        raise _fail("file exceeds the per-file bound")
    if not isinstance(entry["sha256"], str) or not _HEX64.match(entry["sha256"]):
        raise _fail("file sha256 must be 64 lowercase hex digits")


def _relative_parts(value: Any) -> list[str]:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
    ):
        raise _fail("file paths must be relative POSIX paths")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise _fail("file paths must stay inside the evidence directory")
    return parts


# --- loading ---------------------------------------------------------------


def load_evidence(path: str | Path) -> LoadedEvidence:
    """Load and verify one evidence directory; any inconsistency is input error."""
    root = Path(path)
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise InferInputError(f"--mlx-server-evidence {path}: {_reason(exc)}") from exc
    if not root.is_dir():
        raise InferInputError(f"--mlx-server-evidence {path}: not a directory")
    manifest = _read_manifest(root)
    _verify_files(root, manifest)
    events, session, warnings = _load_native(root, manifest)
    stats = _load_stats(root, manifest)
    return LoadedEvidence(root, manifest, events, session, stats, tuple(warnings))


def load_all_evidence(paths: Sequence[str | Path]) -> list[LoadedEvidence]:
    """Load every directory, merging identical copies and rejecting conflicts."""
    loaded: list[LoadedEvidence] = []
    for path in paths:
        item = load_evidence(path)
        match = next(
            (
                other
                for other in loaded
                if other.identity == item.identity and other.run_id == item.run_id
            ),
            None,
        )
        if match is None:
            _check_conflicts(loaded, item)
            loaded.append(item)
        elif match.manifest != item.manifest:
            raise InferInputError(
                f"MLX evidence {item.directory} conflicts with {match.directory}: "
                "the same process and session carry different contents"
            )
        # An identical copy of one capture is one source, not two processes.
    return loaded


def _check_conflicts(loaded: Sequence[LoadedEvidence], item: LoadedEvidence) -> None:
    for other in loaded:
        a, b = other.manifest["process"], item.manifest["process"]
        same_process = (a["host"], a["boot_id"], a["pid"], a["start_time_ns"]) == (
            b["host"],
            b["boot_id"],
            b["pid"],
            b["start_time_ns"],
        )
        if same_process and a["start_time_ns"] is not None:
            raise InferInputError(
                f"MLX evidence {item.directory} and {other.directory} describe "
                "the same process with different session identities"
            )
    ids = [
        entry["process"]["session_id"]
        for entry in (other.manifest for other in loaded)
        if entry["process"]["session_id"] is not None
    ]
    session = item.manifest["process"]["session_id"]
    if session is not None and session in ids:
        raise InferInputError(
            f"MLX evidence {item.directory} reuses a native session ID of "
            "another evidence directory"
        )


def _digests(evidence: LoadedEvidence) -> list[tuple[str, str]]:
    return sorted((f["path"], f["sha256"]) for f in evidence.manifest["files"])


def _reason(exc: Exception) -> str:
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return str(exc)


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    try:
        if path.is_symlink():
            raise ValueError("the manifest must not be a symbolic link")
        size = path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ValueError("the manifest exceeds the size bound")
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
    except FileNotFoundError as exc:
        raise InferInputError(
            f"{root}: not an MLX server evidence directory (no {MANIFEST_NAME})"
        ) from exc
    except (OSError, ValueError) as exc:
        raise InferInputError(f"{path}: {_reason(exc)}") from exc
    validate_manifest(payload)
    return dict(payload)


def _resolved_file(root: Path, relative: str) -> Path:
    parts = _relative_parts(relative)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise _fail(f"{relative} traverses a symbolic link")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise _fail(f"{relative} is missing or outside the evidence directory") from exc
    if not resolved.is_file():
        raise _fail(f"{relative} is not a regular file")
    return resolved


def _verify_files(root: Path, manifest: Mapping[str, Any]) -> None:
    listed = {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        _verify_file(root, entry)
    _check_unlisted_native(root, listed)


def _verify_file(root: Path, entry: Mapping[str, Any]) -> None:
    resolved = _resolved_file(root, entry["path"])
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            size += len(chunk)
            if size > entry["bytes"]:
                break
            digest.update(chunk)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise InferInputError(
            f"MLX evidence file {entry['path']} does not match its manifest "
            "size and digest; it was modified or truncated"
        )


def _check_unlisted_native(root: Path, listed: set[str]) -> None:
    telemetry = root / TELEMETRY_DIR
    if telemetry.is_dir():
        for found in telemetry.rglob("*"):
            relative = found.relative_to(root).as_posix()
            if found.is_symlink():
                raise _fail(f"{relative} is a symbolic link")
            if found.is_file() and relative not in listed:
                raise InferInputError(
                    f"MLX evidence file {relative} is not listed in the manifest"
                )


def _load_native(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[tuple["TelemetryEvent", ...], "SessionSummary | None", list[str]]:
    native = [f for f in manifest["files"] if f["role"] == "native_telemetry"]
    warnings: list[str] = []
    if manifest["capture"]["status"] != "complete":
        warnings.append(
            f"capture status is {manifest['capture']['status']}: coverage is "
            "incomplete and later case windows may have no samples"
        )
    if not native:
        warnings.append("the evidence lists no native telemetry files")
        return (), None, warnings
    _validate_native_files(root, manifest)
    from stormlog.telemetry import load_telemetry_sessions

    try:
        sessions = load_telemetry_sessions(
            root / TELEMETRY_DIR, permissive_legacy=False
        )
    except (OSError, ValueError) as exc:
        raise InferInputError(
            f"MLX evidence native telemetry is invalid: {_reason(exc)}"
        ) from exc
    session = _select_native_session(sessions, manifest["process"]["session_id"])
    warnings.extend(session.warnings)
    _check_process(manifest, session.events)
    events = tuple(sorted(session.events, key=lambda event: event.timestamp_ns))
    return events, session.summary, warnings


def _select_native_session(sessions: Sequence[Any], expected: str | None) -> Any:
    chosen = [
        s for s in sessions if expected is None or s.summary.session_id == expected
    ]
    if len(chosen) != 1 or len(sessions) != 1:
        raise InferInputError(
            "MLX evidence must contain exactly the declared native session"
        )
    return chosen[0]


def _check_process(manifest: Mapping[str, Any], events: Sequence[Any]) -> None:
    process = manifest["process"]
    for event in events:
        if (
            event.pid != process["pid"]
            or event.host != process["host"]
            or event.metadata.get("framework") != "mlx"
        ):
            raise InferInputError(
                "MLX evidence samples were not produced by the manifest's "
                f"process (pid {event.pid} on {event.host}, expected pid "
                f"{process['pid']} on {process['host']})"
            )


def _load_stats(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    entries = [f for f in manifest["files"] if f["role"] == "omlx_stats"]
    if not entries:
        return ()
    if manifest["engine"]["name"] != "omlx":
        raise _fail("oMLX statistics belong to an omlx capture")
    records: list[dict[str, Any]] = []
    path = _resolved_file(root, entries[0]["path"])
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(
            iter(lambda: handle.readline(MAX_RECORD_BYTES + 1), ""), 1
        ):
            if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                raise InferInputError("MLX statistics record exceeded byte limit")
            if number > MAX_STATS_RECORDS:
                raise InferInputError("MLX evidence statistics exceed the record bound")
            if not line.strip():
                continue
            records.append(_stats_record(line, number, manifest["run_id"]))

    return tuple(records)


def _stats_record(line: str, number: int, run_id: str) -> dict[str, Any]:
    try:
        record = json.loads(
            line, parse_constant=_reject_nonfinite, object_pairs_hook=_unique_object
        )
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        validate_stats_record(record)
        if record.get("run_id") != run_id:
            raise ValueError("statistics belong to another run")
    except ValueError as exc:
        raise InferInputError(
            f"MLX evidence {STATS_NAME} line {number}: {exc}"
        ) from exc
    return record


def _validate_routing_proof(
    manifest: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> None:
    proof = endpoint["routing_proof"]
    if proof is None:
        return
    keys = {"method", "pid", "start_time_ns", "address", "port", "observed_at_ns"}
    if not isinstance(proof, dict) or set(proof) != keys:
        raise _fail("invalid endpoint routing proof fields")
    if proof["method"] != "psutil_process_listening_socket":
        raise _fail("unsupported endpoint routing proof")
    process = manifest["process"]
    if (
        proof["pid"] != process["pid"]
        or proof["start_time_ns"] != process["start_time_ns"]
    ):
        raise _fail("routing proof belongs to a different process epoch")
    _integer(proof["pid"], "routing proof PID", minimum=1)
    _integer(proof["start_time_ns"], "routing proof process epoch", minimum=1)
    _integer(proof["port"], "routing proof port", minimum=1)
    if proof["port"] != endpoint["port"] or endpoint["host"] not in {
        "127.0.0.1",
        "::1",
    }:
        raise _fail("routing proof requires matching literal loopback endpoint")
    if proof["address"] != endpoint["host"]:
        raise _fail("routing proof address differs from endpoint")
    _validate_proof_time(proof, manifest["capture"])


def _validate_proof_time(proof: Mapping[str, Any], capture: Mapping[str, Any]) -> None:
    _integer(proof["observed_at_ns"], "routing proof timestamp")
    ended = capture["ended_at_ns"]
    if (
        ended is None
        or not capture["started_at_ns"] <= proof["observed_at_ns"] <= ended
    ):
        raise _fail("routing proof is outside capture window")


def _validate_native_files(root: Path, manifest: Mapping[str, Any]) -> None:
    count = 0
    for entry in manifest["files"]:
        name = Path(entry["path"]).name
        if entry["role"] != "native_telemetry" or not (
            name.startswith("segment-") and name.endswith(".jsonl")
        ):
            continue
        with _resolved_file(root, entry["path"]).open("rb") as handle:
            for line in iter(lambda: handle.readline(MAX_RECORD_BYTES + 1), b""):
                count += 1
                if count > MAX_NATIVE_RECORDS or len(line) > MAX_RECORD_BYTES:
                    raise InferInputError("MLX native telemetry exceeds record bounds")
                _validate_native_record(line, manifest)


def _validate_native_record(line: bytes, manifest: Mapping[str, Any]) -> None:
    from stormlog.telemetry import validate_telemetry_record

    try:
        record = json.loads(
            line, parse_constant=_reject_nonfinite, object_pairs_hook=_unique_object
        )
        if (
            not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 4
        ):
            raise ValueError("Native MLX telemetry requires framework schema v4")
        validate_telemetry_record(record)
        _native_identity(record, manifest)
    except (ValueError, UnicodeError) as exc:
        raise InferInputError(f"Invalid native MLX telemetry: {exc}") from exc


def _native_identity(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    process = manifest["process"]
    if record["pid"] != process["pid"] or record["host"] != process["host"]:
        raise ValueError("Native sample belongs to a different process")
    if (
        record.get("job_id") != manifest["run_id"]
        or record["session_id"] != process["session_id"]
    ):
        raise ValueError("Native sample belongs to a different run/session")
    if record["metadata"].get("framework") != "mlx":
        raise ValueError("Native sample is not MLX allocator evidence")
    capture = manifest["capture"]
    if not capture["started_at_ns"] <= record["timestamp_ns"] <= capture["ended_at_ns"]:
        raise ValueError("Native sample is outside capture window")
    start = process["start_time_ns"]
    if start is not None and record["timestamp_ns"] < start:
        raise ValueError("Native sample predates the process epoch")


def _reject_nonfinite(_value: str) -> Any:
    raise ValueError("Nonfinite JSON values are invalid")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key}")
        result[key] = value
    return result
