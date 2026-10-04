"""Incident bundles: one directory per incident, published in generations.

A bundle looks like this::

    inc-<UTC>-<seq>-<hex>/
      .lock            readers hold it shared; reclamation takes it exclusively
      gen-0/           written at seal
      gen-1/           written by the finalizer, beside gen-0 until published
      manifest.json    names the current generation and its files

A generation is written whole, and its files and directories are fsynced.
Only then is the manifest replaced (temporary file, fsync, rename, fsync of
the bundle directory) to name it, so a reader sees one complete generation
or the one before it, never a mix. The previous generation is deleted only
afterwards, and only when no reader holds the bundle's lock; otherwise its
deletion waits for :meth:`IncidentStore.reclaim_deferred`. Readers use
:func:`open_incident_bundle`, which pins the bundle for the whole read, or
:func:`read_manifest_snapshot`, which re-reads the manifest when a file it
named has gone.

Every byte is charged to the store's :class:`~.disk.DiskBudget` before it
is written, and a write over its allowance abandons the generation before
anything is published. :meth:`IncidentStore.recover` runs before any
pruning: it seals bundles a crash left without a manifest, as
``interrupted``, and removes generations no manifest names.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .disk import (
    Allowance,
    BudgetExceeded,
    CappedWriter,
    DiskBudget,
    StoreLimits,
    bytes_on_disk,
    write_all,
)

BUNDLE_FORMAT = "stormlog.infer.incident_bundle"
BUNDLE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
LOCK_FILENAME = ".lock"
INCIDENTS_DIRNAME = "incidents"
BUNDLE_NAME = re.compile(r"^inc-\d{8}T\d{6}Z-\d{4}-[0-9a-f]{8}$")
GENERATION_NAME = re.compile(r"^gen-(\d+)$")
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUSES = (STATUS_COMPLETED, STATUS_INTERRUPTED)
# A bundle with no manifest and nothing recoverable is removed once it is
# this old; a younger one may belong to a seal still in progress.
JUNK_AGE_SECONDS = 3600.0
_COPY_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class BundleFile:
    """One file of the current generation, relative to the bundle."""

    path: str
    bytes: int
    sha256: str | None


@dataclass(frozen=True)
class BundleManifest:
    """What a bundle holds now: its current generation and that generation's files."""

    incident_id: str
    generation: int
    status: str
    sealed_at_ns: int
    finalized: bool
    complete: bool
    files: tuple[BundleFile, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current(self) -> str:
        return f"gen-{self.generation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "incident_id": self.incident_id,
            "current": self.current,
            "generation": self.generation,
            "status": self.status,
            "sealed_at_ns": self.sealed_at_ns,
            "finalized": self.finalized,
            "complete": self.complete,
            "membership_frozen": True,
            "files": [
                {"path": f.path, "bytes": f.bytes, "sha256": f.sha256}
                for f in self.files
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> BundleManifest:
        """Parse a manifest; anything malformed is a ``ValueError``."""
        if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
            raise ValueError("not an incident bundle manifest")
        if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported bundle schema_version {payload.get('schema_version')!r}"
            )
        try:
            files = tuple(
                BundleFile(str(f["path"]), int(f["bytes"]), f.get("sha256"))
                for f in payload["files"]
            )
            manifest = cls(
                incident_id=str(payload["incident_id"]),
                generation=int(payload["generation"]),
                status=str(payload["status"]),
                sealed_at_ns=int(payload["sealed_at_ns"]),
                finalized=bool(payload["finalized"]),
                complete=bool(payload["complete"]),
                files=files,
                metadata=dict(payload.get("metadata") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed bundle manifest: {exc}") from exc
        if manifest.status not in STATUSES:
            raise ValueError(f"unknown bundle status {manifest.status!r}")
        return manifest


@dataclass
class RecoveryReport:
    """What :meth:`IncidentStore.recover` found and did."""

    sealed_interrupted: list[str] = field(default_factory=list)
    junk_removed: list[str] = field(default_factory=list)
    generations_removed: int = 0
    temporaries_removed: int = 0
    unreadable: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PrunedBundle:
    """A bundle retention removed, and why."""

    incident_id: str
    reason: str
    bytes: int


class GenerationWriter:
    """Writes one generation of a bundle, then publishes it or abandons it."""

    def __init__(
        self,
        store: IncidentStore,
        bundle: Path,
        generation: int,
        allowance: Allowance,
    ) -> None:
        self.store = store
        self.bundle = bundle
        self.generation = generation
        self.allowance = allowance
        self.directory = bundle / f"gen-{generation}"
        self.directory.mkdir(mode=0o700)
        self._done = False

    @property
    def incident_id(self) -> str:
        return self.bundle.name

    def file(self, relpath: str) -> CappedWriter:
        """A new file in this generation, every byte charged before it is written."""
        target = self._target(relpath)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return CappedWriter(target, self.allowance)

    def adopt(self, source: Path, relpath: str) -> int:
        """Move a file in, charged by its actual size first; return its bytes.

        On the same filesystem the file is renamed in. Across filesystems it
        is copied through the allowance and the source is removed only once
        the copy is complete and synced.
        """
        target = self._target(relpath)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        size = source.stat().st_size
        if _same_filesystem(source, self.directory):
            self.allowance.charge(size)
            os.replace(source, target)
            return size
        with source.open("rb") as handle, CappedWriter(target, self.allowance) as out:
            while chunk := handle.read(_COPY_CHUNK):
                out.write(chunk)
        if target.stat().st_size != size:
            raise OSError(errno.EIO, f"copy of {source} is incomplete")
        source.unlink()
        return size

    def link_previous(self, relpath: str) -> None:
        """Hard-link a file of the current generation into this one, uncharged."""
        current = self.store.manifest(self.incident_id)
        if current is None:
            raise FileNotFoundError(f"{self.incident_id} has no published generation")
        source = self.bundle / current.current / relpath
        target = self._target(relpath)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.link(source, target)

    def publish(
        self,
        *,
        status: str = STATUS_COMPLETED,
        finalized: bool = False,
        complete: bool = True,
        sealed_at_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
        digests: bool = True,
    ) -> BundleManifest:
        """Sync this generation, then name it in the manifest; return the manifest."""
        if self._done:
            raise RuntimeError("generation already published or abandoned")
        previous = self.store.manifest(self.incident_id)
        files = tuple(_describe_files(self.bundle, self.directory, digests=digests))
        _sync_tree(self.directory)
        manifest = BundleManifest(
            incident_id=self.incident_id,
            generation=self.generation,
            status=status,
            sealed_at_ns=(
                sealed_at_ns
                if sealed_at_ns is not None
                else (previous.sealed_at_ns if previous else time.time_ns())
            ),
            finalized=finalized,
            complete=complete,
            files=files,
            metadata=dict(metadata or {}),
        )
        _replace_manifest(self.bundle, manifest)
        self._done = True
        self.allowance.release()
        self.store._reclaim_old_generations(self.bundle, keep=self.generation)
        return manifest

    def abandon(self) -> None:
        """Remove this generation; nothing it wrote stays charged."""
        if self._done:
            return
        self._done = True
        shutil.rmtree(self.directory, ignore_errors=True)
        self.allowance.release(keep=0)

    def _target(self, relpath: str) -> Path:
        target = (self.directory / relpath).resolve()
        if self.directory.resolve() not in target.parents:
            raise ValueError(f"{relpath!r} is outside the generation")
        return target


class IncidentStore:
    """The bundles under ``<root>/incidents``, their budget and their retention."""

    def __init__(self, root: Path, limits: StoreLimits | None = None) -> None:
        self.root = Path(root) / INCIDENTS_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.limits = limits or StoreLimits()
        self.budget = DiskBudget(self.limits, used_bytes=self._scan_bytes())
        self._sequence = 0
        # Deletions a reader held back: the path, and the bundle it is in.
        self._deferred: dict[Path, Path] = {}

    # -------------------------------------------------------------- creation

    def new_incident_id(self, now_ns: int | None = None) -> str:
        stamp = datetime.fromtimestamp(
            (now_ns if now_ns is not None else time.time_ns()) / 1e9, tz=timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        self._sequence += 1
        return f"inc-{stamp}-{self._sequence % 10_000:04d}-{secrets.token_hex(4)}"

    def new_bundle(
        self, incident_id: str, reserve_bytes: int
    ) -> GenerationWriter | None:
        """Generation 0 of a new bundle, or None when the budget cannot hold it."""
        if not BUNDLE_NAME.match(incident_id):
            raise ValueError(f"not an incident id: {incident_id!r}")
        allowance = self.budget.reserve(reserve_bytes)
        if allowance is None:
            return None
        bundle = self.root / incident_id
        try:
            bundle.mkdir(mode=0o700)
            (bundle / LOCK_FILENAME).touch(mode=0o600)
            return GenerationWriter(self, bundle, 0, allowance)
        except BaseException:
            allowance.release(keep=0)
            shutil.rmtree(bundle, ignore_errors=True)
            raise

    def next_generation(
        self, incident_id: str, reserve_bytes: int
    ) -> GenerationWriter | None:
        """The generation after the published one, or None over budget."""
        current = self.manifest(incident_id)
        if current is None:
            raise FileNotFoundError(f"{incident_id} has no published generation")
        allowance = self.budget.reserve(reserve_bytes)
        if allowance is None:
            return None
        bundle = self.root / incident_id
        generation = current.generation + 1
        stale = bundle / f"gen-{generation}"
        if stale.exists():  # an earlier attempt that never published
            shutil.rmtree(stale, ignore_errors=True)
        try:
            return GenerationWriter(self, bundle, generation, allowance)
        except BaseException:
            allowance.release(keep=0)
            raise

    # --------------------------------------------------------------- reading

    def manifest(self, incident_id: str) -> BundleManifest | None:
        try:
            return _read_manifest(self.root / incident_id)
        except FileNotFoundError:
            return None

    def bundles(self) -> list[tuple[Path, BundleManifest]]:
        """Published bundles, oldest seal first."""
        found = []
        for path in self._bundle_dirs():
            try:
                found.append((path, _read_manifest(path)))
            except (OSError, ValueError):
                continue
        return sorted(found, key=lambda item: (item[1].sealed_at_ns, item[0].name))

    # -------------------------------------------------------------- recovery

    def recover(self, *, now: float | None = None) -> RecoveryReport:
        """Repair what a crash left; run before any pruning."""
        report = RecoveryReport()
        clock = time.time() if now is None else now
        for bundle in self._bundle_dirs():
            report.temporaries_removed += _remove_temporaries(bundle)
            try:
                manifest = _read_manifest(bundle)
            except FileNotFoundError:
                self._recover_unpublished(bundle, clock, report)
                continue
            except (OSError, ValueError):
                report.unreadable.append(bundle.name)
                continue
            report.generations_removed += self._remove_unnamed(bundle, manifest)
        self._deferred.clear()  # the budget below is rescanned from disk
        self.budget = DiskBudget(self.limits, used_bytes=self._scan_bytes())
        return report

    def _recover_unpublished(
        self, bundle: Path, now: float, report: RecoveryReport
    ) -> None:
        generation = bundle / "gen-0"
        if generation.is_dir() and any(generation.iterdir()):
            # Sealed as interrupted from whatever gen-0 holds; never finalized.
            for stale in _generation_dirs(bundle):
                if stale != generation:
                    shutil.rmtree(stale, ignore_errors=True)
                    report.generations_removed += 1
            files = tuple(_describe_files(bundle, generation, digests=True))
            manifest = BundleManifest(
                incident_id=bundle.name,
                generation=0,
                status=STATUS_INTERRUPTED,
                sealed_at_ns=int(generation.stat().st_mtime * 1e9),
                finalized=False,
                complete=False,
                files=files,
            )
            (bundle / LOCK_FILENAME).touch(mode=0o600, exist_ok=True)
            _replace_manifest(bundle, manifest)
            report.sealed_interrupted.append(bundle.name)
            return
        if now - bundle.stat().st_mtime >= JUNK_AGE_SECONDS:
            shutil.rmtree(bundle, ignore_errors=True)
            report.junk_removed.append(bundle.name)

    def _remove_unnamed(self, bundle: Path, manifest: BundleManifest) -> int:
        removed = 0
        for stale in _generation_dirs(bundle):
            if stale.name != manifest.current and self._try_delete(stale, bundle):
                removed += 1
        return removed

    # ------------------------------------------------------------- retention

    def prune(
        self, *, now_ns: int | None = None, protected: frozenset[str] = frozenset()
    ) -> list[PrunedBundle]:
        """Remove the oldest sealed bundles until every limit holds.

        ``protected`` names bundles still open or being finalized; they are
        never removed. A bundle a reader holds is skipped and retried later.
        """
        clock = time.time_ns() if now_ns is None else now_ns
        cutoff = clock - int(self.limits.max_age_hours * 3600 * 1e9)
        candidates = [
            (path, manifest)
            for path, manifest in self.bundles()
            if manifest.incident_id not in protected
        ]
        count = len(self.bundles())
        pruned: list[PrunedBundle] = []
        for path, manifest in candidates:
            reason = self._prune_reason(manifest, count, cutoff)
            if reason is None:
                continue
            size = _payload_bytes(path, seen=set())
            count -= 1  # deferred or not, it is on its way out
            if self._try_delete(path, path):
                pruned.append(PrunedBundle(manifest.incident_id, reason, size))
        return pruned

    def _prune_reason(
        self, manifest: BundleManifest, count: int, cutoff_ns: int
    ) -> str | None:
        if manifest.sealed_at_ns < cutoff_ns:
            return "max_age_hours"
        if count > self.limits.max_incidents:
            return "max_incidents"
        # Bytes a deferred deletion will free are already on their way out.
        pending = sum(
            _freed_by(path, bundle) for path, bundle in self._deferred.items()
        )
        if self.budget.used_bytes - pending > self.limits.max_total_bytes:
            return "max_total_bytes"
        return None

    def reclaim_deferred(self) -> int:
        """Retry deletions a reader held back; return how many succeeded."""
        done = 0
        for path, bundle in sorted(self._deferred.items()):
            if path in self._deferred and self._try_delete(path, bundle):
                done += 1
        return done

    @property
    def deferred(self) -> int:
        return len(self._deferred)

    # --------------------------------------------------------------- helpers

    def _reclaim_old_generations(self, bundle: Path, *, keep: int) -> None:
        for stale in _generation_dirs(bundle):
            if stale.name != f"gen-{keep}":
                self._try_delete(stale, bundle)

    def _try_delete(self, path: Path, bundle: Path) -> bool:
        """Delete ``path`` (a generation, or the whole bundle) under the
        bundle's exclusive lock, and forget exactly the bytes that freed.

        While a reader holds the lock the deletion is deferred, and the bytes
        stay charged until :meth:`reclaim_deferred` succeeds. What was freed
        is measured under the lock, just before the deletion: a hard link
        another generation keeps frees nothing, and a path already gone
        frees nothing, so no byte is ever forgotten twice.
        """
        try:
            with _exclusive(bundle):
                freed = _freed_by(path, bundle)
                shutil.rmtree(path, ignore_errors=True)
        except BlockingIOError:
            self._deferred[path] = bundle
            return False
        self._deferred.pop(path, None)
        if path == bundle:
            # Anything deferred inside the bundle went with it.
            for inside in [p for p in self._deferred if bundle in p.parents]:
                del self._deferred[inside]
        self.budget.forget(freed)
        return True

    def _bundle_dirs(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and BUNDLE_NAME.match(path.name)
        )

    def _scan_bytes(self) -> int:
        seen: set[tuple[int, int]] = set()
        return sum(_payload_bytes(path, seen=seen) for path in self._bundle_dirs())


@dataclass(frozen=True)
class BundleView:
    """A pinned bundle: its manifest and the generation directory it names."""

    path: Path
    manifest: BundleManifest

    @property
    def generation_dir(self) -> Path:
        return self.path / self.manifest.current

    def file(self, relpath: str) -> Path:
        return self.generation_dir / relpath


@contextlib.contextmanager
def open_incident_bundle(path: str | Path) -> Iterator[BundleView]:
    """Pin a bundle for reading: no generation it names is deleted meanwhile."""
    bundle = Path(path)
    fd = os.open(bundle / LOCK_FILENAME, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield BundleView(bundle, _read_manifest(bundle))
    finally:
        os.close(fd)


def read_manifest_snapshot(path: str | Path, *, attempts: int = 3) -> BundleView:
    """A manifest whose files all exist, for readers that do not take the lock.

    When a file the manifest names has gone, a newer generation was published
    in between: the manifest is read again, up to ``attempts`` times.
    """
    bundle = Path(path)
    for _ in range(max(1, attempts)):
        manifest = _read_manifest(bundle)
        if all((bundle / f.path).exists() for f in manifest.files):
            return BundleView(bundle, manifest)
    raise FileNotFoundError(
        f"{bundle}: files changed under {attempts} manifest reads; "
        "use open_incident_bundle to pin the bundle"
    )


# ------------------------------------------------------------------ internals


def _read_manifest(bundle: Path) -> BundleManifest:
    payload = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return BundleManifest.from_dict(payload)


def _replace_manifest(bundle: Path, manifest: BundleManifest) -> None:
    temporary = bundle / (MANIFEST_FILENAME + ".tmp")
    data = (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, bundle / MANIFEST_FILENAME)
    _fsync_dir(bundle)


def _describe_files(
    bundle: Path, generation: Path, *, digests: bool
) -> Iterator[BundleFile]:
    for dirpath, _dirs, names in sorted(os.walk(generation)):
        for name in sorted(names):
            path = Path(dirpath) / name
            yield BundleFile(
                path=str(path.relative_to(bundle)),
                bytes=path.stat().st_size,
                sha256=_sha256(path) if digests else None,
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_tree(directory: Path) -> None:
    for dirpath, _dirs, names in os.walk(directory):
        for name in names:
            fd = os.open(Path(dirpath) / name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(dirpath))
    _fsync_dir(directory.parent)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _generation_dirs(bundle: Path) -> list[Path]:
    return sorted(
        path
        for path in bundle.iterdir()
        if path.is_dir() and GENERATION_NAME.match(path.name)
    )


def _remove_temporaries(bundle: Path) -> int:
    removed = 0
    for path in bundle.glob("*.tmp"):
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
    return removed


def _payload_bytes(bundle: Path, *, seen: set[tuple[int, int]]) -> int:
    """The bytes the budget charges for a bundle: its generations' files.

    The manifest and lock file are not charged; they are a few KiB per
    bundle, and ``max_incidents`` bounds how many exist.
    """
    return bytes_on_disk(_generation_dirs(bundle), seen=seen)


def _freed_by(path: Path, bundle: Path) -> int:
    """The bytes deleting ``path`` would free: its files no other generation
    of the bundle links to; nothing when it is already gone."""
    if not path.exists():
        return 0
    if path == bundle:
        return _payload_bytes(bundle, seen=set())
    seen: set[tuple[int, int]] = set()
    bytes_on_disk([g for g in _generation_dirs(bundle) if g != path], seen=seen)
    return bytes_on_disk([path], seen=seen)


def _same_filesystem(source: Path, directory: Path) -> bool:
    return source.stat().st_dev == directory.stat().st_dev


@contextlib.contextmanager
def _exclusive(bundle: Path) -> Iterator[None]:
    """Hold the bundle's lock exclusively, or raise ``BlockingIOError`` at once."""
    lock = bundle / LOCK_FILENAME
    try:
        fd = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        yield  # no lock file: nothing can be reading through the helper
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BudgetExceeded",
    "BundleFile",
    "BundleManifest",
    "BundleView",
    "GenerationWriter",
    "IncidentStore",
    "PrunedBundle",
    "RecoveryReport",
    "STATUS_COMPLETED",
    "STATUS_INTERRUPTED",
    "open_incident_bundle",
    "read_manifest_snapshot",
]
