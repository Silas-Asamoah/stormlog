"""The vLLM hook's raw log: one epoch directory per process and role.

Records go into a bounded in-memory queue and a daemon thread writes them, so a
patched vLLM call never waits on a disk. Each record's fields are serialized
once, as they are queued, and the queue bounds the number of records and the
exact size of that JSON until written; a record that does not fit, or is too
large on its own, is dropped and counted. The thread checks its deadlines after
every record, so heartbeats, flush requests and sealing stay on time under a
backlog. Each line is written whole or not at all. Segments are sealed by
renaming ``.part`` to ``.jsonl``; a failed seal is counted and retried. A status
file is rewritten every heartbeat, and keeps being rewritten after the disk cap
stops record writing, so loss stays visible. See ``docs/vllm_execution.md``.
"""

from __future__ import annotations

import atexit
import json
import multiprocessing.util
import os
import re
import shutil
import socket
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..host_clock import host_boot_id

FORMAT = "stormlog.vllm_hook/1"
EPOCH_NAME = re.compile(r"^(engine|worker)-\d+-\d+$")
# A queued record: its kind, and its fields as one JSON object. ``json`` escapes
# every character beyond ASCII, so the text's length is its size in bytes.
_Queued = tuple[str, str]
_dumps = json.JSONEncoder(separators=(",", ":")).encode


@dataclass(frozen=True)
class WriterLimits:
    """Bounds on memory, disk and timing for one epoch."""

    max_bytes: int = 256 * 1024 * 1024
    segment_bytes: int = 8 * 1024 * 1024
    seal_seconds: float = 60.0
    heartbeat_seconds: float = 1.0
    queue_records: int = 20_000
    queue_bytes: int = 32 * 1024 * 1024
    record_bytes: int = 4 * 1024 * 1024
    batch_records: int = 256
    close_seconds: float = 2.0


@dataclass
class _Counters:
    dropped: Counter[str] = field(default_factory=Counter)
    errors: int = 0
    bytes: int = 0
    capped: bool = False
    last_seq: int = -1


class EpochWriter:
    """Write one process role's raw log; every public method is thread-safe."""

    def __init__(
        self,
        root: Path,
        role: str,
        *,
        limits: WriterLimits | None = None,
        status_fields: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.limits = limits or WriterLimits()
        self.role = role
        self.pid = os.getpid()
        self.start_ns = time.time_ns()
        self.host = socket.gethostname()
        self.boot_id = host_boot_id()
        self.epoch = f"{role}-{self.pid}-{self.start_ns}"
        self.directory = root / f"{self.host}-{self.boot_id or 'noboot'}" / self.epoch
        self.directory.mkdir(parents=True, mode=0o700)
        self.key = _write_key(self.directory / "key")
        self._status_fields = status_fields or dict
        self._counters = _Counters()
        self._queue: deque[_Queued] = deque()
        self._queued_bytes = 0
        self._condition = threading.Condition()
        self._closing = False
        # Set once goodbye is written; no heartbeat record may follow it.
        self._ended = False
        self._segment = _Segment(self.directory)
        self._thread = threading.Thread(
            target=self._run, name=f"stormlog-vllm-{role}", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)
        # A forked multiprocessing child leaves through os._exit, which skips
        # atexit; multiprocessing's own finalizers still run before it.
        multiprocessing.util.Finalize(None, self.close, exitpriority=100)

    def emit(self, kind: str, fields: dict[str, Any]) -> None:
        """Queue one record; drop and count it when it does not fit.

        A record the queue cannot take at any size is dropped unserialized, so
        a backlog costs the caller no encoding. The rest are serialized now.
        ``fields`` must not reuse the common fields' names, which the writer
        thread adds with the record's sequence number.
        """
        if self._refused(kind):
            return
        body = self._body(fields)
        if body is None:
            return
        size = len(body)
        with self._condition:
            if size > self.limits.record_bytes:
                self._counters.dropped[f"{kind}_oversized"] += 1
                return
            full = (
                len(self._queue) >= self.limits.queue_records
                or self._queued_bytes + size > self.limits.queue_bytes
            )
            if full or self._closing:
                self._counters.dropped[kind] += 1
                return
            self._queue.append((kind, body))
            self._queued_bytes += size
            self._condition.notify()

    def count_error(self) -> None:
        with self._condition:
            self._counters.errors += 1

    def close(self, goodbye: bool = True) -> None:
        """Write the last records and seal; wait at most ``close_seconds``."""
        if os.getpid() != self.pid:
            return
        with self._condition:
            if self._closing:
                return
            if goodbye:
                self._queue.append(("goodbye", "{}"))
                self._queued_bytes += 2
            self._closing = True
            self._condition.notify()
        self._thread.join(self.limits.close_seconds)

    # ------------------------------------------------------------ writer thread

    def _run(self) -> None:
        # The first heartbeat waits a full interval, so the hello that follows
        # the writer's creation is the epoch's first record.
        next_beat = time.monotonic() + self.limits.heartbeat_seconds
        while True:
            batch, closing = self._take(next_beat)
            for kind, body in batch:
                self._write(kind, body)
                with self._condition:
                    self._queued_bytes -= len(body)
                if self._segment.size >= self.limits.segment_bytes:
                    self._seal()
                # Checked per record, so a slow disk delays these by one write.
                if time.monotonic() >= next_beat:
                    next_beat = self._on_time(next_beat, closing=False)
            next_beat = self._on_time(next_beat, closing)
            if closing:
                return

    def _on_time(self, next_beat: float, closing: bool) -> float:
        """Heartbeat, flush and seal when due; return the next heartbeat time."""
        # No heartbeat record on close: goodbye is the epoch's last record, and
        # sealing rewrites status.json.
        if time.monotonic() >= next_beat and not closing:
            self._heartbeat()
            next_beat = time.monotonic() + self.limits.heartbeat_seconds
        self._maybe_seal(closing)
        return next_beat

    def _take(self, next_beat: float) -> tuple[list[_Queued], bool]:
        """Up to ``batch_records`` records; the rest wait their turn."""
        with self._condition:
            if not self._queue and not self._closing:
                self._condition.wait(max(0.0, next_beat - time.monotonic()))
            batch: list[_Queued] = []
            while self._queue and len(batch) < self.limits.batch_records:
                batch.append(self._queue.popleft())
            closing = self._closing and not self._queue
            return batch, closing

    def _refused(self, kind: str) -> bool:
        """Drop and count a record that no size would let into the queue."""
        with self._condition:
            refused = (
                self._closing
                or self._counters.capped
                or len(self._queue) >= self.limits.queue_records
            )
            if refused:
                self._counters.dropped[kind] += 1
            return refused

    def _body(self, fields: object) -> str | None:
        """The fields as one JSON object, or None and an error counted."""
        if isinstance(fields, dict):  # _join splices objects only
            try:
                return _dumps(fields)
            except (TypeError, ValueError, RecursionError):
                pass
        self.count_error()
        return None

    def _write(self, kind: str, body: str) -> None:
        with self._condition:
            if self._counters.capped:
                self._counters.dropped[kind] += 1
                return
            seq = self._counters.last_seq + 1
        if kind == "goodbye":
            body = _dumps({**_stamp(), "last_seq": seq})
        common = {"format": FORMAT, "kind": kind, "epoch": self.epoch, "seq": seq}
        line = _join(_dumps(common), body)
        with self._condition:
            if self._counters.bytes + len(line) > self.limits.max_bytes:
                self._counters.capped = True
                self._counters.dropped[kind] += 1
                return
        written = self._segment.write(line)
        with self._condition:
            if written:
                self._counters.last_seq = seq
                self._counters.bytes += len(line)
                self._ended = self._ended or kind == "goodbye"
            else:
                self._counters.errors += 1
                self._counters.dropped[kind] += 1

    def _heartbeat(self) -> None:
        status = self._status()
        if not self._counters.capped and not self._ended:
            body = self._body(status)
            if body is not None:
                self._write("heartbeat", body)
        self._write_status()

    def _write_status(self) -> None:
        if not _replace_json(self.directory / "status.json", self._status()):
            self.count_error()

    def _status(self) -> dict[str, Any]:
        with self._condition:
            counters = self._counters
            status: dict[str, Any] = {
                **_stamp(),
                "last_seq": counters.last_seq,
                "dropped": dict(counters.dropped),
                "errors": counters.errors,
                "bytes": counters.bytes,
                "capped": counters.capped,
                "queued": len(self._queue),
            }
        try:
            status.update(self._status_fields())
        except Exception:  # a status provider must not stop the writer
            self.count_error()
        return status

    def _maybe_seal(self, closing: bool) -> None:
        flush = self.directory / "flush"
        asked = flush.exists()
        due = (
            closing
            or asked
            or self._segment.size >= self.limits.segment_bytes
            or self._segment.age() >= self.limits.seal_seconds
        )
        if not due:
            return
        if self._seal() and asked:
            # Only a seal that succeeded answers a flush request.
            _unlink(flush)
        if closing:
            self._write_status()

    def _seal(self) -> bool:
        if self._segment.seal():
            return True
        self.count_error()
        return False


class _Segment:
    """The open ``NNNNNN.jsonl.part`` file of an epoch."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index = 0
        self.size = 0
        self.opened_at: float | None = None
        self._fd: int | None = None

    def write(self, line: bytes) -> bool:
        """Append the whole line, or leave the file as it was."""
        try:
            if self._fd is None:
                self._fd = os.open(
                    self._path(".jsonl.part"),
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                self.size = os.fstat(self._fd).st_size
                if self.opened_at is None:
                    self.opened_at = time.monotonic()
        except OSError:
            return False
        before = self.size
        if _write_all(self._fd, line):
            self.size += len(line)
            return True
        try:
            # Cut off a partial line, so the next record starts a valid line.
            os.ftruncate(self._fd, before)
        except OSError:
            pass
        return False

    def age(self) -> float:
        return 0.0 if self.opened_at is None else time.monotonic() - self.opened_at

    def seal(self) -> bool:
        """Close and rename the segment; on failure keep it to retry."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self.opened_at is None:
            return True  # nothing written since the last seal
        try:
            os.replace(self._path(".jsonl.part"), self._path(".jsonl"))
        except OSError:
            # Still unsealed: the next write appends to it, and the next seal
            # retries the rename.
            return False
        self.index += 1
        self.size = 0
        self.opened_at = None
        return True

    def _path(self, suffix: str) -> Path:
        return self.directory / f"{self.index:06d}{suffix}"


def remove_old_epochs(
    root: Path, retain_hours: float, *, now: float | None = None
) -> int:
    """Delete epoch directories untouched for longer than ``retain_hours``."""
    cutoff = (time.time() if now is None else now) - retain_hours * 3600
    removed = 0
    for epoch in root.glob("*/*"):
        try:
            if EPOCH_NAME.match(epoch.name) and epoch.stat().st_mtime < cutoff:
                shutil.rmtree(epoch)
                removed += 1
        except OSError:
            continue
    return removed


def _join(first: str, second: str) -> bytes:
    """Two JSON objects' text as one object's line, with ``first``'s fields first."""
    if second == "{}":
        return (first + "\n").encode()
    return (first[:-1] + "," + second[1:] + "\n").encode()


def _write_all(fd: int, data: bytes) -> bool:
    view = memoryview(data)
    try:
        while view:
            written = os.write(fd, view)
            if written <= 0:
                return False
            view = view[written:]
    except OSError:
        return False
    return True


def _stamp() -> dict[str, int]:
    return {"wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}


def _write_key(path: Path) -> bytes:
    key = os.urandom(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if not _write_all(fd, key):
            raise OSError(f"could not write {path}")
    finally:
        os.close(fd)
    return key


def _replace_json(path: Path, payload: dict[str, Any]) -> bool:
    temporary = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            data = json.dumps(payload, separators=(",", ":")).encode()
            if not _write_all(fd, data):
                return False
        finally:
            os.close(fd)
        os.replace(temporary, path)
    except OSError:
        return False
    return True


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


__all__ = ["FORMAT", "EpochWriter", "WriterLimits", "remove_old_epochs"]
