"""A Prometheus textfile per producer slot, for node_exporter or no server at all.

One slot names everything: the producer label in the file, the file
``DIR/stormlog-<slot>.prom`` and its lock ``DIR/stormlog-<slot>.lock``. Two
writers that would emit the same label in one directory therefore contend
for one lock, and the second is refused. A lock whose process is gone, or
whose process has a different start time, is stale and taken over.

Each write goes to a temporary file that replaces the real one, so a reader
never sees half a file; a failed write leaves the previous file in place,
and its freshness gauge shows its age. One thread writes, in order, so a
late write can only publish the newest content it was given.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import psutil

from .registry import escape_label_value
from .renders import RenderCache

SLOT = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
PRODUCER_LABEL = "stormlog_producer"
UNREADABLE_LOCK_SECONDS = 10.0


class SlotInUse(RuntimeError):
    """Another live writer holds this slot's lock in this directory."""


@dataclass
class TextfileStats:
    writes_ok: int = 0
    writes_failed: int = 0
    last_error: str | None = None
    last_write_at: float | None = None
    abandoned: bool = False
    # The final write could only use a render from before the last change.
    final_stale: bool = False


def validate_slot(slot: str) -> str:
    if not SLOT.match(slot):
        raise ValueError(f"a slot is 1 to 64 of A-Z a-z 0-9 _ . -, not {slot!r}")
    return slot


def slot_paths(directory: Path, slot: str) -> tuple[Path, Path]:
    """The file and the lock a slot names in ``directory``."""
    validate_slot(slot)
    return directory / f"stormlog-{slot}.prom", directory / f"stormlog-{slot}.lock"


class TextfileWriter:
    """Write the shared render to the slot's file now and every ``interval``."""

    def __init__(
        self,
        directory: Path,
        slot: str,
        renders: RenderCache,
        *,
        interval: float = 15.0,
        remove_on_exit: bool = False,
        forbidden: Iterable[Path] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = Path(directory)
        self.slot = validate_slot(slot)
        self.path, self.lock_path = slot_paths(self.directory, slot)
        self.renders = renders
        self.interval = interval
        self.remove_on_exit = remove_on_exit
        self.forbidden = [Path(path) for path in forbidden]
        self._clock = clock
        self.stats = TextfileStats()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._active = True
        self._thread: threading.Thread | None = None
        self._lock_owned = False

    def acquire(self) -> None:
        """Check the directory and take the slot's lock, before the run starts.

        ``ValueError`` for a directory that is missing or holds a forbidden
        file; ``SlotInUse`` when another live writer has the slot.
        """
        if self._lock_owned:
            return
        if not self.directory.is_dir():
            raise ValueError(f"textfile directory {self.directory} does not exist")
        directory = self.directory.resolve()
        for path in self.forbidden:
            if path.resolve().parent == directory:
                raise ValueError(
                    f"textfile directory {self.directory} holds {path.name}; "
                    "choose a directory of its own"
                )
        _take_lock(self.lock_path)
        self._lock_owned = True

    def start(self) -> None:
        """Take the lock if not yet taken, then write now and every interval."""
        self.acquire()
        self._thread = threading.Thread(
            target=self._run, name=f"stormlog-textfile-{self.slot}", daemon=True
        )
        self._thread.start()

    def close(self, deadline: float = 5.0) -> None:
        """Write the final file (``stormlog_run_active`` 0) within ``deadline``.

        A writer still stuck in I/O at the deadline is left to finish; its
        lock is kept, so no other writer can take the slot until this
        process has exited.
        """
        self._active = False
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(deadline)
            if self._thread.is_alive():
                self.stats.abandoned = True
                return
        if self.remove_on_exit:
            _unlink(self.path)
        if self._lock_owned:
            _release_lock(self.lock_path)
            self._lock_owned = False

    def _run(self) -> None:
        while not self._stop.is_set():
            self._write_once()
            self._wake.wait(self.interval)
            self._wake.clear()
        # The final write, after the run ended: stormlog_run_active is 0.
        self._write_once()

    def _write_once(self) -> None:
        if self.remove_on_exit and not self._active:
            return
        # The generation stays acquired for the whole write, so a writer
        # stuck in I/O is one of the readers the publication limit counts,
        # and the body is written as it is, never copied.
        generation = self.renders.acquire()
        try:
            if not self._active and not self.renders.is_fresh(generation):
                self.stats.final_stale = True
            self._write_file(generation.body)
        finally:
            self.renders.release(generation)

    def _write_file(self, body: bytes) -> None:
        temporary = self.directory / f".stormlog-{self.slot}.prom.{os.getpid()}.tmp"
        try:
            with open(temporary, "wb") as handle:
                handle.write(body)
                handle.write(self._own_lines())
                handle.flush()
            os.replace(temporary, self.path)
        except OSError as exc:
            _unlink(temporary)
            self.stats.writes_failed += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            return
        self.stats.writes_ok += 1
        self.stats.last_write_at = self._clock()

    def _own_lines(self) -> bytes:
        label = f'{{{PRODUCER_LABEL}="{escape_label_value(self.slot)}"}}'
        return (
            "# HELP stormlog_textfile_updated_timestamp_seconds "
            "When this file was last written, in Unix seconds.\n"
            "# TYPE stormlog_textfile_updated_timestamp_seconds gauge\n"
            f"stormlog_textfile_updated_timestamp_seconds{label} {self._clock():.3f}\n"
            "# HELP stormlog_run_active "
            "1 while the run writing this file is running, 0 after it ended.\n"
            "# TYPE stormlog_run_active gauge\n"
            f"stormlog_run_active{label} {1 if self._active else 0}\n"
        ).encode()


# ------------------------------------------------------------------ the lock
def _identity() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "started": psutil.Process().create_time(),
        "host": socket.gethostname(),
    }


def _take_lock(path: Path) -> None:
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _read_lock(path)
            if not _stale(path, holder):
                raise SlotInUse(
                    f"{path.name} is held by pid {holder.get('pid')} on "
                    f"{holder.get('host')}; choose another --prometheus-slot"
                ) from None
            _unlink(path)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_identity(), handle)
        return
    raise SlotInUse(f"{path.name} was taken by another writer while starting")


def _read_lock(path: Path) -> dict[str, object]:
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return holder if isinstance(holder, dict) else {}


def _stale(path: Path, holder: dict[str, object]) -> bool:
    """A lock whose writer is certainly gone. Another host's lock never is.

    An unreadable lock is stale once it is older than a writer needs to
    fill it in.
    """
    if not holder:
        try:
            return time.time() - path.stat().st_mtime > UNREADABLE_LOCK_SECONDS
        except OSError:
            return True
    pid = holder.get("pid")
    if holder.get("host") != socket.gethostname() or not isinstance(pid, int):
        return False
    try:
        started = psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except psutil.Error:
        return False
    return bool(started != holder.get("started"))


def _release_lock(path: Path) -> None:
    holder = _read_lock(path)
    if holder.get("pid") == os.getpid():
        _unlink(path)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
