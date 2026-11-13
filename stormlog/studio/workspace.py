"""Workspace discovery: what lives under Studio's root (design §4, §5.4, §7.4).

Studio lists four kinds of file under its one workspace root:

- ``inference_artifact``: a ``.jsonl`` file holding an ``infer.session`` or
  ``infer.request`` record (``analysis.py:740-755``);
- ``audio_artifact``: a ``.jsonl`` file holding ``infer.audio.*`` records,
  listed and marked unsupported in slice 1;
- ``report``: a ``.json`` file whose ``format`` is ``stormlog.report`` at
  schema version 1, with its ``report_kind`` and verdict;
- ``run_envelope``: a ``.json`` file whose ``format`` is
  ``stormlog.run_envelope`` at version 1 (``stormlog_run.json``).

Everything else is not listed. :func:`classify` reads at most 64 KiB of a
file: for a JSONL file the first 32 KiB and the last 32 KiB (the whole file
when it is smaller), so it finds the first session record and, usually,
the terminal one; for a JSON file the first 64 KiB, of which it decodes the
leading top-level members only.

Confinement (§7.4). :class:`ArtifactScanner` walks the root with directory
file descriptors: every directory and file is opened relative to its
parent's descriptor with ``O_NOFOLLOW``, so a symlink, or a directory
swapped for one during the walk, is never followed and nothing outside the
root is read or listed. Symlinks are skipped, files and directories alike:
a listed path must open through ``WorkspaceRoot.open``, and a link to a
file inside the root lists that file at its own path. FIFOs and devices
are opened non-blocking and skipped. The walk skips ``.git``,
``node_modules``, ``__pycache__``, virtual environments (a directory named
like one, or holding ``pyvenv.cfg``) and import temp files.

Bounds (§8.1): depth 8 below the root, 50,000 directory entries and 10 s per
scan (``Limits.scan_*``); reaching one ends the scan as ``partial`` with
``limits_hit`` naming it. A scan runs on the builder pool; one at a time.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import hashlib
import json
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from stormlog._run_catalog_parser import is_run_envelope
from stormlog.report import REPORT_FORMAT, REPORT_SCHEMA_VERSION, validate_report
from stormlog.scrub import scrub_text
from stormlog.studio.contracts import (
    BusyError,
    ConflictError,
    InvalidInputError,
    Limits,
    UsageError,
    WorkspaceRoot,
    finite_copy,
)

__all__ = [
    "KINDS",
    "SNIFF_BYTES",
    "WorkspaceItem",
    "classify",
    "ArtifactScanner",
    "read_report",
]

INFERENCE_ARTIFACT = "inference_artifact"
AUDIO_ARTIFACT = "audio_artifact"
REPORT = "report"
RUN_ENVELOPE = "run_envelope"
KINDS: tuple[str, ...] = (INFERENCE_ARTIFACT, AUDIO_ARTIFACT, REPORT, RUN_ENVELOPE)

# classify reads at most this many bytes of a file.
SNIFF_BYTES = 64 * 2**10
_HALF = SNIFF_BYTES // 2

_JSONL_SUFFIXES = frozenset({".jsonl"})
_JSON_SUFFIXES = frozenset({".json"})
_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        "site-packages",
    }
)
# correlation_capture._replace_file stages an import as .<name>.<uuid>.tmp.
_IMPORT_TEMP = re.compile(r"\..+\.[0-9a-f]{32}\.tmp")
_WHITESPACE = re.compile(r"[ \t\n\r]*")
_DECODER = json.JSONDecoder()
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_CURSOR_VERSION = 1


@dataclass(frozen=True)
class WorkspaceItem:
    """One listed file. ``path`` is root-relative POSIX."""

    path: str
    kind: str  # one of KINDS
    size: int  # bytes, at the scan
    mtime_ns: int
    session_id: str | None  # artifacts: the first session record's, else a request's
    status: str | None  # artifacts: the last infer.session status read; else None
    report_kind: str | None  # reports only
    verdict: str | None  # reports only: verdict.status

    def to_record(self) -> dict[str, Any]:
        """The item as ``stormlog.studio.items`` lists it."""
        return asdict(self)


# --- classify ----------------------------------------------------------------


def classify(path: Path, *, rel: str | None = None) -> WorkspaceItem | None:
    """The item for one file, or None when Studio does not list it.

    Reads at most :data:`SNIFF_BYTES`. The file is opened without following
    a final symlink and without blocking; anything but a regular file is
    None. ``rel`` is the item's path; by default ``path`` as POSIX.
    """
    if not _candidate(path.name):
        return None
    try:
        fd = os.open(path, _FILE_FLAGS)
    except OSError:
        return None
    try:
        return _classify_fd(fd, path.name, rel or path.as_posix())
    finally:
        os.close(fd)


def _candidate(name: str) -> bool:
    if _IMPORT_TEMP.fullmatch(name):
        return False
    suffix = os.path.splitext(name)[1].lower()
    return suffix in _JSONL_SUFFIXES or suffix in _JSON_SUFFIXES


def _classify_fd(fd: int, name: str, rel: str) -> WorkspaceItem | None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        return None
    base = WorkspaceItem(
        path=rel,
        kind="",
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        session_id=None,
        status=None,
        report_kind=None,
        verdict=None,
    )
    if os.path.splitext(name)[1].lower() in _JSONL_SUFFIXES:
        return _jsonl_item(base, *_head_and_tail(fd, info.st_size))
    return _json_item(base, os.pread(fd, SNIFF_BYTES, 0), info.st_size)


def _head_and_tail(fd: int, size: int) -> tuple[bytes, bytes | None]:
    """The whole file when it fits, else its first and last 32 KiB."""
    if size <= SNIFF_BYTES:
        return os.pread(fd, SNIFF_BYTES, 0), None
    return os.pread(fd, _HALF, 0), os.pread(fd, _HALF, size - _HALF)


def _records(chunk: bytes, *, drop_first: bool) -> Iterator[Mapping[str, Any]]:
    """The JSON objects on the complete lines of ``chunk``."""
    lines = chunk.split(b"\n")
    lines.pop()  # after the last newline: empty, or a line cut by the read
    for line in lines[1:] if drop_first else lines:
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(record, Mapping):
            yield record


def _jsonl_item(
    base: WorkspaceItem, head: bytes, tail: bytes | None
) -> WorkspaceItem | None:
    head_records = list(_records(head, drop_first=False))
    tail_records = [] if tail is None else list(_records(tail, drop_first=True))
    event_types = {
        str(record.get("event_type"))
        for record in (*head_records, *tail_records)
        if isinstance(record.get("event_type"), str)
    }
    if any(name.startswith("infer.audio.") for name in event_types):
        kind = AUDIO_ARTIFACT
    elif event_types & {"infer.session", "infer.request"}:
        kind = INFERENCE_ARTIFACT
    else:
        return None
    # Without the whole file, only the tail can say how the run ended.
    status_records = head_records if tail is None else tail_records
    return replace(
        base,
        kind=kind,
        session_id=_session_id(head_records),
        status=_last_session_status(status_records),
    )


def _session_id(records: list[Mapping[str, Any]]) -> str | None:
    for event_type in ("infer.session", "infer.request"):
        for record in records:
            value = record.get("session_id")
            if record.get("event_type") == event_type and isinstance(value, str):
                return value
    return None


def _last_session_status(records: list[Mapping[str, Any]]) -> str | None:
    status = None
    for record in records:
        value = record.get("status")
        if record.get("event_type") == "infer.session" and isinstance(value, str):
            status = value
    return status


def _json_item(base: WorkspaceItem, chunk: bytes, size: int) -> WorkspaceItem | None:
    text = chunk.decode("utf-8", errors="ignore")
    members = _leading_members(text)
    version = members.get("schema_version")
    if members.get("format") == REPORT_FORMAT and _is_int(
        version, REPORT_SCHEMA_VERSION
    ):
        verdict = members.get("verdict")
        status = verdict.get("status") if isinstance(verdict, Mapping) else None
        return replace(
            base,
            kind=REPORT,
            session_id=_text_or_none(members.get("session_id")),
            report_kind=_text_or_none(members.get("report_kind")),
            verdict=_text_or_none(status),
        )
    if is_run_envelope(members):
        return replace(base, kind=RUN_ENVELOPE)
    return None


def _is_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _leading_members(text: str) -> dict[str, Any]:
    """The top-level members of a JSON object that ``text`` holds whole.

    ``text`` may be cut anywhere: decoding stops at the first member that
    is not followed by ``,`` or ``}``, which a cut value never is.
    """
    position = _skip(text, 0)
    if text[position : position + 1] != "{":
        return {}
    members: dict[str, Any] = {}
    position += 1
    while True:
        member = _member(text, position)
        if member is None:
            return members
        key, value, position = member
        members[key] = value
        if text[position - 1] == "}":
            return members


def _member(text: str, position: int) -> tuple[str, Any, int] | None:
    """One ``"key": value`` and the position after its ``,`` or ``}``."""
    try:
        key, position = _DECODER.raw_decode(text, _skip(text, position))
        position = _skip(text, position)
        if not isinstance(key, str) or text[position : position + 1] != ":":
            return None
        value, position = _DECODER.raw_decode(text, _skip(text, position + 1))
    except (ValueError, RecursionError):
        return None
    position = _skip(text, position)
    if text[position : position + 1] not in {",", "}"}:
        return None
    return key, value, position + 1


def _skip(text: str, position: int) -> int:
    match = _WHITESPACE.match(text, position)
    return position if match is None else match.end()


# --- the scan ----------------------------------------------------------------


@dataclass
class _Walk:
    """The state of one walk; it runs on one thread."""

    limits: Limits
    deadline: float
    clock: Callable[[], float]
    items: list[WorkspaceItem] = field(default_factory=list)
    entries: int = 0
    files_read: int = 0
    skipped: dict[str, int] = field(
        default_factory=lambda: {"symlinks": 0, "excluded_dirs": 0, "unreadable": 0}
    )
    limits_hit: set[str] = field(default_factory=set)

    def stopped(self) -> bool:
        return bool(self.limits_hit & {"entries", "time"})

    def directory(self, fd: int, prefix: str, depth: int) -> None:
        """List one directory, then descend into its subdirectories."""
        subdirs: list[str] = []
        # Read no more entries than the scan may still admit, plus one to
        # notice the cap; sort those, so a complete scan is in name order.
        room = max(self.limits.scan_entries - self.entries, 0) + 1
        with os.scandir(fd) as listing:
            entries = sorted(islice(listing, room), key=lambda item: item.name)
        for entry in entries:
            if not self._admit():
                break
            if entry.is_symlink():
                self.skipped["symlinks"] += 1
            elif entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.name)
            elif entry.is_file(follow_symlinks=False) and _candidate(entry.name):
                self._file(fd, entry.name, prefix + entry.name)
        for name in subdirs:
            if self.stopped():
                return
            self._subdirectory(fd, name, prefix, depth)

    def _admit(self) -> bool:
        if self.stopped():
            return False
        if self.entries >= self.limits.scan_entries:
            self.limits_hit.add("entries")
            return False
        if self.clock() >= self.deadline:
            self.limits_hit.add("time")
            return False
        self.entries += 1
        return True

    def _file(self, parent: int, name: str, rel: str) -> None:
        try:
            fd = os.open(name, _FILE_FLAGS, dir_fd=parent)
        except OSError:
            self.skipped["unreadable"] += 1
            return
        try:
            self.files_read += 1
            item = _classify_fd(fd, name, rel)
        except OSError:
            self.skipped["unreadable"] += 1
            return
        finally:
            os.close(fd)
        if item is not None:
            self.items.append(item)

    def _subdirectory(self, parent: int, name: str, prefix: str, depth: int) -> None:
        if depth >= self.limits.scan_depth:
            self.limits_hit.add("depth")
            return
        if name in _SKIPPED_DIRS:
            self.skipped["excluded_dirs"] += 1
            return
        try:
            fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
        except OSError:
            self.skipped["unreadable"] += 1
            return
        try:
            if _is_virtualenv(fd):
                self.skipped["excluded_dirs"] += 1
                return
            self.directory(fd, f"{prefix}{name}/", depth + 1)
        except OSError:
            self.skipped["unreadable"] += 1
        finally:
            os.close(fd)


def _is_virtualenv(fd: int) -> bool:
    try:
        os.stat("pyvenv.cfg", dir_fd=fd, follow_symlinks=False)
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class _ScanResult:
    status: str  # complete, partial, failed
    items: tuple[WorkspaceItem, ...]
    started_at_utc: str
    finished_at_utc: str
    duration_s: float
    entries: int
    files_read: int
    skipped: Mapping[str, int]
    limits_hit: tuple[str, ...]
    reason: str | None


class ArtifactScanner:
    """C4's :class:`~stormlog.studio.contracts.WorkspaceScanner` (§5.4).

    ``scan()`` queues one walk of ``root.path`` on ``pool`` and returns at
    once; a second ``scan()`` while one runs raises BusyError. The last
    finished scan's items stay listed while the next one runs, and are
    replaced in one step when it ends.
    """

    def __init__(
        self,
        root: WorkspaceRoot,
        pool: Executor,
        limits: Limits | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = root
        self._pool = pool
        self._limits = limits or Limits()
        self._clock = clock
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._running = False
        self._scans = 0
        self._result: _ScanResult | None = None
        self._paths: tuple[str, ...] = ()

    def scan(self) -> None:
        """Queue a scan on the builder pool; BusyError while one runs."""
        with self._lock:
            if self._running:
                raise BusyError("busy", "a workspace scan is already running")
            self._running = True
            self._idle.clear()
        try:
            self._pool.submit(self._run)
        except RuntimeError as error:  # the pool is shut down
            self._finish(None)
            raise BusyError("busy", "Studio is stopping") from error

    def wait(self, timeout_s: float) -> bool:
        """Whether no scan is running, waiting up to ``timeout_s``."""
        return self._idle.wait(timeout_s)

    def _run(self) -> None:
        """One walk; whatever happens, the scanner is idle again after it.

        An error other than OSError (a bug, MemoryError) fails the scan
        with a scrubbed reason instead of leaving it ``running``, which
        made every later ``scan()`` a 503 ``busy``.
        """
        started = self._clock()
        started_utc = _utc_now()
        walk = _Walk(self._limits, started + self._limits.scan_seconds, self._clock)
        result: _ScanResult | None = None
        try:
            try:
                reason = self._walk_root(walk)
            except Exception as error:
                reason = scrub_text(
                    f"the scan stopped on an unexpected error: "
                    f"{type(error).__name__}: {error}",
                    max_bytes=200,
                )
            result = _ScanResult(
                status=(
                    "failed"
                    if reason
                    else ("partial" if walk.limits_hit else "complete")
                ),
                items=tuple(sorted(walk.items, key=lambda item: item.path)),
                started_at_utc=started_utc,
                finished_at_utc=_utc_now(),
                duration_s=round(self._clock() - started, 6),
                entries=walk.entries,
                files_read=walk.files_read,
                skipped=dict(walk.skipped),
                limits_hit=tuple(sorted(walk.limits_hit)),
                reason=reason,
            )
        finally:
            self._finish(result)

    def _walk_root(self, walk: _Walk) -> str | None:
        """Walk the root into ``walk``; the failure reason, or None."""
        try:
            fd = os.open(self._root.path, _DIR_FLAGS)
            try:
                walk.directory(fd, "", 0)
            finally:
                os.close(fd)
        except OSError as error:
            return scrub_text(f"cannot read the root: {error.strerror}", max_bytes=200)
        return None

    def _finish(self, result: _ScanResult | None) -> None:
        with self._lock:
            if result is not None:
                self._result = result
                self._paths = tuple(item.path for item in result.items)
                self._scans += 1
            self._running = False
            self._idle.set()

    def state(self) -> dict[str, Any]:
        """The body of ``stormlog.studio.workspace``."""
        with self._lock:
            result, running, scans = self._result, self._running, self._scans
        counts = {kind: 0 for kind in KINDS}
        for item in () if result is None else result.items:
            counts[item.kind] += 1
        return {
            "scan": _scan_block(result, running, scans),
            "counts": {**counts, "total": sum(counts.values())},
            "limits": {
                "depth": self._limits.scan_depth,
                "entries": self._limits.scan_entries,
                "seconds": self._limits.scan_seconds,
                "sniff_bytes": SNIFF_BYTES,
            },
        }

    def items(
        self, *, kind: str | None, q: str | None, cursor: str | None, limit: int
    ) -> dict[str, Any]:
        """The body of ``stormlog.studio.items``: one page, sorted by path.

        UsageError for an unknown kind, a ``q`` over ``Limits.text_chars``,
        a limit outside 1 to ``Limits.page_rows`` or a malformed cursor;
        ConflictError ``cursor_mismatch`` for a cursor made with another
        filter. Paging is keyset on the path, so a cursor stays valid
        across rescans.
        """
        text = _check_filter(kind, q, limit, self._limits)
        after = _decode_cursor(cursor, kind, q) if cursor is not None else None
        with self._lock:
            result, paths, running, scans = (
                self._result,
                self._paths,
                self._running,
                self._scans,
            )
        listed = () if result is None else result.items
        start = 0 if after is None else bisect.bisect_right(paths, after)
        page, total, last = _page(listed, start, kind, text, limit)
        return {
            "filter": {"kind": kind, "q": q},
            "items": [item.to_record() for item in page],
            "total": total,
            "next_cursor": None if last is None else _encode_cursor(kind, q, last),
            "scan": _scan_block(result, running, scans),
        }


def _page(
    listed: tuple[WorkspaceItem, ...],
    start: int,
    kind: str | None,
    text: str | None,
    limit: int,
) -> tuple[list[WorkspaceItem], int, str | None]:
    """The page from ``start``, the matching total, and the cursor's path
    when more matches follow the page."""
    matches = [item for item in listed if _matches(item, kind, text)]
    page = [item for item in listed[start:] if _matches(item, kind, text)][:limit]
    more = bool(page) and matches[-1].path != page[-1].path
    return page, len(matches), page[-1].path if more else None


def _scan_block(
    result: _ScanResult | None, running: bool, scans: int
) -> dict[str, Any]:
    if result is None:
        status = "scanning" if running else "not_started"
        return {
            "status": status,
            "running": running,
            "scans": scans,
            "started_at_utc": None,
            "finished_at_utc": None,
            "duration_s": None,
            "entries": 0,
            "files_read": 0,
            "skipped": {"symlinks": 0, "excluded_dirs": 0, "unreadable": 0},
            "limits_hit": [],
            "reason": None,
        }
    return {
        "status": result.status,
        "running": running,
        "scans": scans,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "duration_s": result.duration_s,
        "entries": result.entries,
        "files_read": result.files_read,
        "skipped": dict(result.skipped),
        "limits_hit": list(result.limits_hit),
        "reason": result.reason,
    }


def _check_filter(
    kind: str | None, q: str | None, limit: int, limits: Limits
) -> str | None:
    if kind is not None and kind not in KINDS:
        raise UsageError(
            "bad_parameter",
            f"kind: unknown value {kind!r}",
            field="kind",
            details={"allowed": list(KINDS)},
        )
    if q is not None and len(q) > limits.text_chars:
        raise UsageError(
            "limit_exceeded",
            f"q: at most {limits.text_chars} characters",
            field="q",
            details={"max": limits.text_chars},
        )
    if isinstance(limit, bool) or not 1 <= limit <= limits.page_rows:
        raise UsageError(
            "limit_exceeded" if limit > limits.page_rows else "bad_parameter",
            f"limit: from 1 to {limits.page_rows}",
            field="limit",
            details={"max": limits.page_rows},
        )
    return None if q is None else q.casefold()


def _matches(item: WorkspaceItem, kind: str | None, text: str | None) -> bool:
    if kind is not None and item.kind != kind:
        return False
    return text is None or text in item.path.casefold()


def _encode_cursor(kind: str | None, q: str | None, after: str) -> str:
    payload = {"v": _CURSOR_VERSION, "kind": kind, "q": q, "after": after}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str, kind: str | None, q: str | None) -> str:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeError, RecursionError):
        # RecursionError: a cursor of nested brackets is a bad cursor too.
        payload = None
    if (
        not isinstance(payload, dict)
        or payload.get("v") != _CURSOR_VERSION
        or not isinstance(payload.get("after"), str)
    ):
        raise UsageError("bad_parameter", "cursor: not a cursor", field="cursor")
    if payload.get("kind") != kind or payload.get("q") != q:
        raise ConflictError(
            "cursor_mismatch",
            "the cursor was made with another filter",
            field="cursor",
            details={"reason": "other_filter"},
        )
    after: str = payload["after"]
    return after


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- report envelopes ----------------------------------------------------------


def read_report(root: WorkspaceRoot, rel: str, *, max_bytes: int) -> dict[str, Any]:
    """The body of ``stormlog.studio.report``: a validated report envelope.

    Opens ``rel`` through the root (NotFoundError, ForbiddenError), refuses
    a file over ``max_bytes`` before reading it (InvalidInputError
    ``report_too_large``), then parses and checks it with
    ``report.validate_report`` (InvalidInputError ``unreadable``, reason
    ``not_a_report``). NaN and infinities become null, listed in
    ``non_finite_fields``.
    """
    with root.open(rel) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > max_bytes:
            raise _too_large(size, max_bytes)
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _too_large(len(data), max_bytes)
    try:
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("a report is a JSON object")
        validate_report(payload)
    except (ValueError, RecursionError) as error:
        raise InvalidInputError(
            "unreadable",
            scrub_text(f"not a stormlog.report v1 envelope: {error}", max_bytes=512),
            details={"reason": "not_a_report"},
        ) from error
    report, non_finite = finite_copy(payload)
    return {
        "path": rel,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "report": report,
        "non_finite_fields": non_finite,
    }


def _too_large(size: int, max_bytes: int) -> InvalidInputError:
    return InvalidInputError(
        "report_too_large",
        f"the report is {size} bytes; Studio reads at most {max_bytes}",
        details={"size": size, "max_bytes": max_bytes},
    )
