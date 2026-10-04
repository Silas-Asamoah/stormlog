"""Byte budgets for the incident store, charged before anything is written.

Every byte the store keeps is charged to a :class:`DiskBudget` first: a file
adopted from elsewhere by its actual size, and a file being written through a
:class:`CappedWriter`, chunk by chunk, before the chunk reaches the disk. A
write that would go over its allowance raises :class:`BudgetExceeded`, so the
caller can abandon the operation before anything is published. Files that
share an inode (hard links between bundle generations) are counted once.
"""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


class BudgetExceeded(OSError):
    """A write or an adoption would go over the bytes it was allowed."""


@dataclass(frozen=True)
class StoreLimits:
    """Hard limits on what the incident store retains."""

    max_total_bytes: int = 4 * 1024**3
    max_incident_bytes: int = 1024**3
    max_incidents: int = 50
    max_age_hours: float = 72.0

    def __post_init__(self) -> None:
        if self.max_total_bytes <= 0 or self.max_incident_bytes <= 0:
            raise ValueError("store byte limits must be > 0")
        if self.max_incident_bytes > self.max_total_bytes:
            raise ValueError("max_incident_bytes must be <= max_total_bytes")
        if self.max_incidents <= 0 or self.max_age_hours <= 0:
            raise ValueError("max_incidents and max_age_hours must be > 0")


class Allowance:
    """Bytes one operation may write; charged before each write.

    An allowance belongs to a :class:`DiskBudget` or stands alone, as in a
    child process given a fixed number of bytes by its parent.
    """

    def __init__(self, limit: int, *, budget: DiskBudget | None = None) -> None:
        if limit < 0:
            raise ValueError("allowance limit must be >= 0")
        self.limit = limit
        self.used = 0
        self._cap = limit
        self._budget = budget
        self._lock = threading.Lock()
        self._released = False

    @property
    def remaining(self) -> int:
        return self._cap - self.used

    def charge(self, nbytes: int) -> None:
        """Take ``nbytes``, or raise :class:`BudgetExceeded` and take nothing."""
        with self._lock:
            if self._released:
                raise BudgetExceeded("the allowance was already released")
            if self.used + nbytes > self._cap:
                raise BudgetExceeded(
                    f"{nbytes} more bytes would exceed the {self._cap}-byte allowance"
                )
            self.used += nbytes

    def cap(self, limit: int) -> None:
        """Lower the limit (never raise it); :class:`BudgetExceeded` if more
        than ``limit`` is already used. The reservation stays as made."""
        with self._lock:
            if self.used > limit:
                raise BudgetExceeded(
                    f"{self.used} bytes already exceed the {limit}-byte cap"
                )
            self._cap = min(self._cap, limit)

    def release(self, *, keep: int | None = None) -> None:
        """Return what was not kept to the budget.

        ``keep`` is the bytes that stay on disk, normally what was used; an
        abandoned operation keeps 0 once its files are deleted.
        """
        with self._lock:
            if self._released:
                return
            self._released = True
            kept = self.used if keep is None else keep
        if self._budget is not None:
            self._budget._settle(self.limit, kept)


class DiskBudget:
    """The bytes the incident store holds, and the reservations against them."""

    def __init__(self, limits: StoreLimits, *, used_bytes: int = 0) -> None:
        self.limits = limits
        self._lock = threading.Lock()
        self._used = used_bytes
        self._reserved = 0

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved

    def free_bytes(self) -> int:
        with self._lock:
            return self.limits.max_total_bytes - self._used - self._reserved

    def reserve(self, nbytes: int) -> Allowance | None:
        """An allowance of ``nbytes``, or None when the store cannot hold them."""
        if nbytes < 0:
            raise ValueError("a reservation must be >= 0 bytes")
        if nbytes > self.limits.max_incident_bytes:
            return None
        with self._lock:
            if self._used + self._reserved + nbytes > self.limits.max_total_bytes:
                return None
            self._reserved += nbytes
        return Allowance(nbytes, budget=self)

    def forget(self, nbytes: int) -> None:
        """Bytes that were deleted, such as a pruned bundle."""
        with self._lock:
            self._used = max(0, self._used - nbytes)

    def _settle(self, reserved: int, kept: int) -> None:
        with self._lock:
            self._reserved = max(0, self._reserved - reserved)
            self._used += kept


class CappedWriter:
    """A binary file whose every write is charged to an allowance first.

    The file is created exclusively (it must not exist), written whole or
    not at all per call, and fsynced on a clean close. On
    :class:`BudgetExceeded`, or any other error, the caller removes it.
    """

    def __init__(self, path: Path, allowance: Allowance, *, mode: int = 0o600):
        self.path = path
        self.allowance = allowance
        self.written = 0
        self._fd: int | None = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)

    def write(self, data: bytes) -> int:
        if self._fd is None:
            raise ValueError(f"{self.path} is closed")
        self.allowance.charge(len(data))
        write_all(self._fd, data)
        self.written += len(data)
        return len(data)

    def close(self, *, sync: bool = True) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if sync:
                os.fsync(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> CappedWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close(sync=exc is None)


def write_all(fd: int, data: bytes) -> None:
    """Write every byte; a short write continues, an error raises."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"write to fd {fd} made no progress")
        view = view[written:]


def bytes_on_disk(paths: Iterable[Path], *, seen: set[tuple[int, int]]) -> int:
    """Bytes of the regular files under ``paths``, each inode counted once.

    ``seen`` carries the inodes already counted across calls, so a file
    hard-linked into two generations, or two bundles, is charged once.
    """
    total = 0
    for root in paths:
        for path in _walk(root):
            try:
                info = path.lstat()
            except OSError:
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen or not stat.S_ISREG(info.st_mode):
                continue
            seen.add(key)
            total += info.st_size
    return total


def _walk(root: Path) -> Iterable[Path]:
    if root.is_file() or root.is_symlink():
        yield root
        return
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            yield Path(dirpath) / name


__all__ = [
    "Allowance",
    "BudgetExceeded",
    "CappedWriter",
    "DiskBudget",
    "StoreLimits",
    "bytes_on_disk",
    "write_all",
]
