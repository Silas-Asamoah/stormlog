"""The vLLM hook's raw log: one epoch directory per process and role.

Records go into a bounded in-memory queue and a daemon thread writes them, so a
patched vLLM call never waits on a disk. The queue bounds both the number of
records and their estimated size; a record that does not fit is dropped and
counted. Segments are sealed by renaming ``.part`` to ``.jsonl``. A status file
is rewritten every heartbeat, and keeps being rewritten after the disk cap
stops record writing, so loss stays visible. See ``docs/vllm_execution.md``.
"""

from __future__ import annotations

import atexit
import json
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


@dataclass(frozen=True)
class WriterLimits:
    """Bounds on memory, disk and timing for one epoch."""

    max_bytes: int = 256 * 1024 * 1024
    segment_bytes: int = 8 * 1024 * 1024
    seal_seconds: float = 60.0
    heartbeat_seconds: float = 1.0
    queue_records: int = 20_000
    queue_bytes: int = 32 * 1024 * 1024
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
        self._queue: deque[tuple[str, dict[str, Any], int]] = deque()
        self._queued_bytes = 0
        self._condition = threading.Condition()
        self._closing = False
        self._segment = _Segment(self.directory)
        self._thread = threading.Thread(
            target=self._run, name=f"stormlog-vllm-{role}", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)

    def emit(self, kind: str, fields: dict[str, Any], size_hint: int = 256) -> None:
        """Queue one record; drop and count it when the queue is full."""
        with self._condition:
            full = (
                len(self._queue) >= self.limits.queue_records
                or self._queued_bytes + size_hint > self.limits.queue_bytes
            )
            if full or self._closing:
                self._counters.dropped[kind] += 1
                return
            self._queue.append((kind, fields, size_hint))
            self._queued_bytes += size_hint
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
                self._queue.append(("goodbye", {}, 64))
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
            for kind, fields, _ in batch:
                self._write(kind, fields)
            # No heartbeat record on close: goodbye is the epoch's last record,
            # and sealing rewrites status.json.
            if time.monotonic() >= next_beat and not closing:
                self._heartbeat()
                next_beat = time.monotonic() + self.limits.heartbeat_seconds
            self._maybe_seal(closing)
            if closing:
                return

    def _take(
        self, next_beat: float
    ) -> tuple[list[tuple[str, dict[str, Any], int]], bool]:
        with self._condition:
            if not self._queue and not self._closing:
                self._condition.wait(max(0.0, next_beat - time.monotonic()))
            batch = list(self._queue)
            self._queue.clear()
            self._queued_bytes = 0
            return batch, self._closing

    def _write(self, kind: str, fields: dict[str, Any]) -> None:
        with self._condition:
            capped = self._counters.capped
            seq = self._counters.last_seq + 1
            if capped:
                self._counters.dropped[kind] += 1
                return
        record = {"format": FORMAT, "kind": kind, "epoch": self.epoch, "seq": seq}
        record.update(fields)
        if kind == "goodbye":
            record.update(_stamp(), last_seq=seq)
        try:
            line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        except (TypeError, ValueError):
            self.count_error()
            return
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
            else:
                self._counters.errors += 1

    def _heartbeat(self) -> None:
        status = self._status()
        if not self._counters.capped:
            self._write("heartbeat", status)
        _replace_json(self.directory / "status.json", self._status())

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
            }
        try:
            status.update(self._status_fields())
        except Exception:  # a status provider must not stop the writer
            self.count_error()
        return status

    def _maybe_seal(self, closing: bool) -> None:
        flush = self.directory / "flush"
        asked = flush.exists()
        if (
            closing
            or asked
            or self._segment.size >= self.limits.segment_bytes
            or self._segment.age() >= self.limits.seal_seconds
        ):
            self._segment.seal()
            if asked:
                _unlink(flush)
            if closing:
                _replace_json(self.directory / "status.json", self._status())


class _Segment:
    """The open ``NNNNNN.jsonl.part`` file of an epoch."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index = 0
        self.size = 0
        self.opened_at: float | None = None
        self._fd: int | None = None

    def write(self, line: bytes) -> bool:
        try:
            if self._fd is None:
                self._fd = os.open(
                    self._path(".jsonl.part"),
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                self.opened_at = time.monotonic()
            os.write(self._fd, line)
        except OSError:
            return False
        self.size += len(line)
        return True

    def age(self) -> float:
        return 0.0 if self.opened_at is None else time.monotonic() - self.opened_at

    def seal(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
            os.replace(self._path(".jsonl.part"), self._path(".jsonl"))
        except OSError:
            pass
        self._fd = None
        self.index += 1
        self.size = 0
        self.opened_at = None

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


def _stamp() -> dict[str, int]:
    return {"wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}


def _write_key(path: Path) -> bytes:
    key = os.urandom(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(payload, separators=(",", ":")).encode())
        finally:
            os.close(fd)
        os.replace(temporary, path)
    except OSError:
        pass


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


__all__ = ["FORMAT", "EpochWriter", "WriterLimits", "remove_old_epochs"]
