"""Append whole lines to a file, so an export file never holds half a line.

A line counts as written only when the whole of it has been handed to the
operating system (and synced, when asked). On an error the file is cut back
to where the line began, so the line is absent; if even that fails, the
file may end in a partial line, the outcome says so, and the sink stops
writing rather than append after a broken line. Repeated errors also stop
it. The caller settles each line from the outcome.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WRITTEN = "written"
FILE_FULL = "file_full"
FILE_ERROR = "file_error"
FILE_PARTIAL = "file_partial"
FILE_DISABLED = "file_disabled"

# The system calls, by name, so a test can fail one without touching the
# process's other writes.
_write = os.write
_sync = os.fsync
_truncate = os.ftruncate


@dataclass
class FileSinkStats:
    lines: int = 0
    bytes: int = 0
    errors: int = 0
    full: int = 0
    disabled: bool = False
    last_error: str | None = None


class LineFileSink:
    """One line per call, whole or not at all, up to ``max_bytes`` in total."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int,
        fsync: bool = False,
        max_consecutive_errors: int = 3,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.fsync = fsync
        self.max_consecutive_errors = max_consecutive_errors
        self.stats = FileSinkStats()
        self._fd: int | None = None
        self._size = 0
        self._consecutive = 0

    def open(self) -> None:
        """Open for appending; an ``OSError`` reaches the caller."""
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._size = os.fstat(self._fd).st_size

    def write_line(self, data: bytes) -> str:
        """Append ``data`` as one line and say how that went."""
        if self.stats.disabled or self._fd is None:
            return FILE_DISABLED
        line = data if data.endswith(b"\n") else data + b"\n"
        if self._size + len(line) > self.max_bytes:
            self.stats.full += 1
            return FILE_FULL
        start = self._size
        try:
            _write_all(self._fd, line)
            if self.fsync:
                _sync(self._fd)
        except OSError as exc:
            return self._failed(start, exc)
        self._size += len(line)
        self._consecutive = 0
        self.stats.lines += 1
        self.stats.bytes += len(line)
        return WRITTEN

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _failed(self, start: int, exc: OSError) -> str:
        assert self._fd is not None
        self.stats.errors += 1
        self.stats.last_error = f"{type(exc).__name__}: {exc}"
        self._consecutive += 1
        try:
            _truncate(self._fd, start)
        except OSError:
            self.stats.disabled = True
            return FILE_PARTIAL
        if self._consecutive >= self.max_consecutive_errors:
            self.stats.disabled = True
        return FILE_ERROR


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = _write(fd, view)
        if written <= 0:
            raise OSError("write made no progress")
        view = view[written:]
