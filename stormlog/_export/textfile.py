"""A Prometheus textfile per producer slot, for node_exporter or no server at all.

One slot names everything: the producer label on every series in the file,
the file ``DIR/stormlog-<slot>.prom`` and its lock
``DIR/stormlog-<slot>.lock``. The writer cannot label a render it is given,
so it is refused unless the render's constant labels name its slot. Two
writers that would emit the same label in one directory therefore contend
for one lock, and the second is refused.

The lock is an ``flock`` on the lock file, held for the whole run, so the
kernel drops it when its process ends, however it ends, and no lock on this
host is ever judged stale: the next writer simply takes it. The file names
its holder, for the refusal's message. A holder on another host is never
displaced, since an ``flock`` may not reach across hosts. Where the system
has no ``flock``, the lock is the file's existence, and one whose process
is gone, or has a different start time, is stale and taken over.

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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import psutil

from .registry import TEXTFILE_RUN_ACTIVE, TEXTFILE_UPDATED, escape_label_value
from .renders import RenderCache

_flock: Callable[[int, int], None] | None
try:
    import fcntl

    _flock = fcntl.flock
    _LOCK_NOW = fcntl.LOCK_EX | fcntl.LOCK_NB
except ImportError:  # no flock: the lock is the file's existence
    _flock = None
    _LOCK_NOW = 0

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
    """Write the shared render to the slot's file now and every ``interval``.

    ``const_labels`` are the labels the render puts on every series; they
    must include ``stormlog_producer`` with the slot as its value.
    """

    def __init__(
        self,
        directory: Path,
        slot: str,
        renders: RenderCache,
        *,
        const_labels: Mapping[str, str],
        interval: float = 15.0,
        remove_on_exit: bool = False,
        forbidden: Iterable[Path] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = Path(directory)
        self.slot = validate_slot(slot)
        if const_labels.get(PRODUCER_LABEL) != slot:
            raise ValueError(
                f"the render must label every series {PRODUCER_LABEL}={slot!r}, "
                f"not {const_labels.get(PRODUCER_LABEL)!r}"
            )
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
        # The descriptor holding the flock, while the lock is owned by one.
        self._lock_descriptor: int | None = None
        # close() decides, under this lock, whether the writer gave up its
        # slot in time; the slot is then freed on the writer's thread.
        self._closing = threading.Lock()
        self._closed = False
        self._freeing = False

    def acquire(self) -> None:
        """Check the directory and take the slot's lock, before the run starts.

        ``ValueError`` for a directory that is missing, holds a forbidden
        file, or cannot be written; ``SlotInUse`` when another live writer
        has the slot.
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
        self._lock_descriptor = _take_lock(self.lock_path)
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

        The slot is then freed: the file removed if asked, and the lock
        released. That I/O happens on the writer's thread, so ``close``
        returns by its deadline whatever the file system does. A writer
        still stuck in a write at the deadline keeps its lock, so no other
        writer can take the slot until this process has exited. A second
        ``close`` does nothing.
        """
        with self._closing:
            if self._closed:
                return
            self._closed = True
        self._active = False
        self._stop.set()
        self._wake.set()
        if self._thread is None:  # never started: only the lock to free
            self._thread = threading.Thread(
                target=self._free_slot,
                name=f"stormlog-textfile-{self.slot}",
                daemon=True,
            )
            self._thread.start()
        self._thread.join(deadline)
        with self._closing:
            if self._thread.is_alive() and not self._freeing:
                self.stats.abandoned = True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._write_once()
                self._wake.wait(self.interval)
                self._wake.clear()
        finally:
            # The final write, after the run ended: stormlog_run_active is 0.
            self._write_once()
            self._free_slot()

    def _free_slot(self) -> None:
        """Remove the file if asked and release the lock, unless abandoned."""
        with self._closing:
            if self.stats.abandoned or not self._lock_owned:
                return
            self._freeing = True
        if self.remove_on_exit:
            _unlink(self.path)
        _release_lock(self.lock_path, self._lock_descriptor)
        self._lock_owned = False
        self._lock_descriptor = None

    def _write_once(self) -> None:
        """One write; any failure is counted, never raised, so the writer lives."""
        try:
            self._write()
        except Exception as exc:  # a render that failed, or a bug
            self.stats.writes_failed += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"

    def _write(self) -> None:
        if self.remove_on_exit and not self._active:
            return
        # The generation stays acquired for the whole write, so a writer
        # stuck in I/O is one of the readers the publication limit counts,
        # and the body is written as it is, never copied.
        generation = self.renders.acquire()
        try:
            # Checked once, before the write: a change landing between the
            # check and the write would not be flagged. The run freezes its
            # values and invalidates the render before it closes the writer,
            # so none lands there. Each write after the run ended sets the
            # flag afresh, so the last one, the final write, decides it.
            if not self._active:
                self.stats.final_stale = not self.renders.is_fresh(generation)
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
            f"# HELP {TEXTFILE_UPDATED} "
            "When this file was last written, in Unix seconds.\n"
            f"# TYPE {TEXTFILE_UPDATED} gauge\n"
            f"{TEXTFILE_UPDATED}{label} {self._clock():.3f}\n"
            f"# HELP {TEXTFILE_RUN_ACTIVE} "
            "1 while the run writing this file is running, 0 after it ended.\n"
            f"# TYPE {TEXTFILE_RUN_ACTIVE} gauge\n"
            f"{TEXTFILE_RUN_ACTIVE}{label} {1 if self._active else 0}\n"
        ).encode()


# ------------------------------------------------------------------ the lock
def _identity() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "started": psutil.Process().create_time(),
        "host": socket.gethostname(),
    }


def _take_lock(path: Path) -> int | None:
    """Take the slot's lock: the descriptor holding its flock, if one does."""
    if _flock is None:
        _take_exclusive(path)
        return None
    for _ in range(3):
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except PermissionError as exc:
            if not path.exists():
                raise ValueError(
                    f"cannot write {path.name} in {path.parent}: {exc.strerror}"
                ) from None
            # Another user's lock, or a read-only one: some other writer's.
            raise _in_use(path, _read_lock(path)) from None
        try:
            if _flock_path(descriptor, path):
                return descriptor
        except BaseException:
            os.close(descriptor)
            raise
        os.close(descriptor)
    raise SlotInUse(f"{path.name} was taken by another writer while starting")


def _flock_path(descriptor: int, path: Path) -> bool:
    """Lock the file open at ``descriptor``; False if it is no longer ``path``."""
    assert _flock is not None
    try:
        _flock(descriptor, _LOCK_NOW)
    except BlockingIOError:
        raise _in_use(path, _read_lock(path)) from None
    try:
        if not os.path.samestat(os.fstat(descriptor), os.stat(path)):
            return False  # its last holder removed it after it was opened
    except FileNotFoundError:
        return False
    holder = _read_lock(path)
    if holder and holder.get("host") != socket.gethostname():
        raise _in_use(path, holder)
    os.ftruncate(descriptor, 0)
    os.pwrite(descriptor, json.dumps(_identity()).encode(), 0)
    return True


def _take_exclusive(path: Path) -> None:
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _read_lock(path)
            if not _stale(path, holder):
                raise _in_use(path, holder) from None
            _unlink(path)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_identity(), handle)
        return
    raise SlotInUse(f"{path.name} was taken by another writer while starting")


def _in_use(path: Path, holder: dict[str, object]) -> SlotInUse:
    return SlotInUse(
        f"{path.name} is held by pid {holder.get('pid')} on "
        f"{holder.get('host')}; choose another --prometheus-slot"
    )


def _read_lock(path: Path) -> dict[str, object]:
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return holder if isinstance(holder, dict) else {}


def _stale(path: Path, holder: dict[str, object]) -> bool:
    """Without flock: a lock whose writer is certainly gone.

    Another host's lock never is.

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


def _release_lock(path: Path, descriptor: int | None) -> None:
    if descriptor is None:
        if _read_lock(path).get("pid") == os.getpid():
            _unlink(path)
        return
    try:
        # Removed while still held, and only if it is still the file locked.
        if os.path.samestat(os.fstat(descriptor), os.stat(path)):
            _unlink(path)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
