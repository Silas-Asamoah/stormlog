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
# Held exclusively by the one store that owns ``<root>/incidents``.
STORE_LOCK_FILENAME = ".store.lock"
# A bundle being deleted is renamed to this prefix first.
TRASH_PREFIX = ".trash-"
# Each file costs up to a filesystem block beyond its charged bytes, and an
# entry in the manifest the budget does not charge: capping the count bounds
# both, to about 4 MiB of blocks and 200 KiB of manifest per generation.
MAX_GENERATION_FILES = 1024
INCIDENTS_DIRNAME = "incidents"
BUNDLE_NAME = re.compile(r"^inc-\d{8}T\d{6}Z-\d{4}-[0-9a-f]{8}$")
GENERATION_NAME = re.compile(r"^gen-(\d+)$")
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUSES = (STATUS_COMPLETED, STATUS_INTERRUPTED)
# A bundle with no manifest and nothing recoverable is removed once it is
# this old; a younger one may belong to a seal still in progress.
JUNK_AGE_SECONDS = 3600.0
# A hard link cannot be made: another device, or a filesystem without links.
_NO_LINK_ERRNOS = frozenset({errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP})
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


class StoreInUse(RuntimeError):
    """Another process already owns this store root."""


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
        # The bundle is pinned while this generation is written, so no
        # deletion (a recovery, a retention pass) can take it from under us.
        self._pin: int | None = _pin(bundle)
        # Every file this writer made, adopted or linked, checked at publish.
        self._written: list[Path] = []
        # Adopted files: their source, their name here, and what was charged.
        self._adopted: list[tuple[Path, Path, int]] = []
        # Bytes linked from the generation before, part of the bundle's total.
        self._linked = 0

    @property
    def incident_id(self) -> str:
        return self.bundle.name

    def file(self, relpath: str) -> CappedWriter:
        """A new file in this generation, every byte charged before it is written."""
        self._count_file()
        target = self._target(relpath)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        writer = CappedWriter(target, self.allowance)
        self._written.append(target)
        return writer

    def adopt(self, source: Path, relpath: str) -> int:
        """Bring a file in, charged by its actual size; return its bytes.

        The file is hard-linked in, or, where a link cannot be made (another
        device, including a bind mount of the same one), copied through the
        allowance. Its source is let go only once the generation is
        published, so an abandoned generation leaves it where it was. A file
        its producer is still writing is charged its growth at publication;
        adopt only a file its producer has finished.

        A link shares the file with its producer until the source's name is
        let go. Where it could never be (a directory this process cannot
        write), the file is copied instead, so a producer rewriting its own
        path cannot change a published bundle.
        """
        self._count_file()
        target = self._target(relpath)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        size = source.stat().st_size
        releasable = os.access(source.parent, os.W_OK)
        if not (releasable and self._link_in(source, target, size)):
            size = self._copy_in(source, target, size)
        self._adopted.append((source, target, size))
        self._written.append(target)
        return size

    def _link_in(self, source: Path, target: Path, size: int) -> bool:
        """Hard-link a file in and charge it; False where no link can be made."""
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno not in _NO_LINK_ERRNOS:
                raise
            return False
        try:
            self.allowance.charge(size)  # before anything else is written
        except BudgetExceeded:
            target.unlink()
            raise
        return True

    def _copy_in(self, source: Path, target: Path, size: int) -> int:
        try:
            with (
                source.open("rb") as handle,
                CappedWriter(target, self.allowance) as out,
            ):
                while chunk := handle.read(_COPY_CHUNK):
                    out.write(chunk)
            if target.stat().st_size != size:
                raise OSError(errno.EIO, f"copy of {source} is incomplete")
        except BaseException:
            # Not yet a file this writer made: a copy cut short would be
            # published, unchecked, if the caller went on.
            with contextlib.suppress(OSError):
                target.unlink()
            raise
        return size

    def _charge_growth(self) -> None:
        """Charge what an adopted file gained since it was adopted."""
        for index, (source, target, charged) in enumerate(self._adopted):
            grown = target.stat().st_size - charged
            if grown > 0:
                self.allowance.charge(grown)
                self._adopted[index] = (source, target, charged + grown)

    def link_previous(self, relpath: str) -> None:
        """Hard-link a file of the current generation into this one.

        The link is not charged again, but it counts toward
        ``max_incident_bytes``, which bounds the whole bundle: what this
        generation may still write shrinks by the linked file's size.
        """
        current = self.store.manifest(self.incident_id)
        if current is None:
            raise FileNotFoundError(f"{self.incident_id} has no published generation")
        source = self.bundle / current.current / relpath
        self._count_file()
        self._linked += source.stat().st_size
        self.allowance.cap(self.store.limits.max_incident_bytes - self._linked)
        target = self._target(relpath)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.link(source, target)
        self._written.append(target)

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
        self._check_written()
        self._charge_growth()
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
        staged = _stage_manifest(self.bundle, manifest)
        os.replace(staged, self.bundle / MANIFEST_FILENAME)
        # Published from here on: whatever fails next, abandon() must not
        # delete what the manifest now names.
        self._done = True
        # What this generation adds, exactly: a file linked from the one
        # before is already charged, and a copy may be a little smaller.
        self.allowance.release(keep=_freed_by(self.directory, self.bundle))
        self._release_sources()
        self._unpin()  # before reclaiming, which takes the lock exclusively
        _fsync_dir(self.bundle)
        self.store._reclaim_old_generations(self.bundle, keep=self.generation)
        return manifest

    def _release_sources(self) -> None:
        """The adopted files' original names, now that the bundle holds them."""
        for source, _target, _size in self._adopted:
            with contextlib.suppress(OSError):
                source.unlink()

    def abandon(self) -> None:
        """Remove this generation; nothing it wrote stays charged.

        A generation the manifest already names is published, whatever
        raised after the rename, and is never removed here.
        """
        if self._done:
            return
        self._done = True
        current = self.store.manifest(self.incident_id)
        if current is not None and current.generation == self.generation:
            self.allowance.release()
            self._unpin()
            return
        shutil.rmtree(self.directory, ignore_errors=True)
        self.allowance.release(keep=0)
        self._unpin()

    def _count_file(self) -> None:
        if len(self._written) >= MAX_GENERATION_FILES:
            raise BudgetExceeded(
                f"over the {MAX_GENERATION_FILES}-file cap of one generation"
            )

    def _check_written(self) -> None:
        """Every file this writer put here is still here, or nothing is named."""
        if not self.directory.is_dir():
            raise FileNotFoundError(f"{self.directory} vanished before publication")
        for path in self._written:
            if not path.exists():
                raise FileNotFoundError(f"{path} vanished before publication")

    def _unpin(self) -> None:
        fd, self._pin = self._pin, None
        if fd is not None:
            os.close(fd)

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
        self._owner: int | None = _own(self.root)
        self.limits = limits or StoreLimits()
        self.budget = DiskBudget(self.limits, used_bytes=self._scan_bytes())
        self._sequence = 0
        # Deletions a reader held back: the path, and the bundle it is in.
        self._deferred: dict[Path, Path] = {}
        # Bundles removed to make room for a reservation, until taken.
        self._room_pruned: list[PrunedBundle] = []

    def close(self) -> None:
        """Let another process own the root."""
        fd, self._owner = self._owner, None
        if fd is not None:
            os.close(fd)

    # -------------------------------------------------------------- creation

    def new_incident_id(self, now_ns: int | None = None) -> str:
        stamp = datetime.fromtimestamp(
            (now_ns if now_ns is not None else time.time_ns()) / 1e9, tz=timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        self._sequence += 1
        return f"inc-{stamp}-{self._sequence % 10_000:04d}-{secrets.token_hex(4)}"

    def new_bundle(
        self,
        incident_id: str,
        reserve_bytes: int,
        *,
        protected: frozenset[str] = frozenset(),
    ) -> GenerationWriter | None:
        """Generation 0 of a new bundle, or None when the budget cannot hold
        it even after making room (see :meth:`take_pruned`)."""
        if not BUNDLE_NAME.match(incident_id):
            raise ValueError(f"not an incident id: {incident_id!r}")
        allowance = self._reserve(reserve_bytes, protected)
        if allowance is None:
            return None
        bundle = self.root / incident_id
        try:
            # An existing bundle is never touched: only what this call made
            # is removed on the way out.
            bundle.mkdir(mode=0o700)
        except BaseException:
            allowance.release(keep=0)
            raise
        try:
            (bundle / LOCK_FILENAME).touch(mode=0o600)
            # The bundle's own entry must survive a power loss as well as
            # what is published inside it.
            _fsync_dir(self.root)
            return GenerationWriter(self, bundle, 0, allowance)
        except BaseException:
            allowance.release(keep=0)
            shutil.rmtree(bundle, ignore_errors=True)
            raise

    def next_generation(
        self,
        incident_id: str,
        reserve_bytes: int,
        *,
        protected: frozenset[str] = frozenset(),
    ) -> GenerationWriter | None:
        """The generation after the published one, or None over budget even
        after making room; the bundle itself is never pruned for it."""
        current = self.manifest(incident_id)
        if current is None:
            raise FileNotFoundError(f"{incident_id} has no published generation")
        allowance = self._reserve(reserve_bytes, protected | {incident_id})
        if allowance is None:
            return None
        bundle = self.root / incident_id
        generation = current.generation + 1
        stale = bundle / f"gen-{generation}"
        if stale.exists():  # an earlier attempt that never published
            # No manifest names it, so no reader can hold it; its bytes are
            # charged (a rescan counted them), and freed here.
            freed = _freed_by(stale, bundle)
            shutil.rmtree(stale, ignore_errors=True)
            self.budget.forget(freed)
        try:
            return GenerationWriter(self, bundle, generation, allowance)
        except BaseException:
            allowance.release(keep=0)
            raise

    def _reserve(self, nbytes: int, protected: frozenset[str]) -> Allowance | None:
        """An allowance, removing the oldest unprotected bundles until it fits.

        The store keeps the newest incidents. A reservation that does not fit
        is refused when removing every unprotected bundle no reader holds
        could not make room, and then nothing is removed. A bundle a reader
        takes meanwhile is skipped, not deferred: deferred, it would be
        removed later for room already made.
        """
        allowance = self.budget.reserve(nbytes)
        if allowance is not None or nbytes > self.limits.max_incident_bytes:
            return allowance
        removable = self._removable(protected)
        if nbytes > self.budget.free_bytes() + sum(freed for *_, freed in removable):
            return None
        for path, incident_id, freed in removable:  # oldest seal first
            if self._try_delete(path, path, defer=False):
                self._room_pruned.append(
                    PrunedBundle(incident_id, "max_total_bytes", freed)
                )
                allowance = self.budget.reserve(nbytes)
                if allowance is not None:
                    return allowance
        return None

    def _removable(self, protected: frozenset[str]) -> list[tuple[Path, str, int]]:
        """The unprotected bundles no reader holds, oldest seal first, each
        with the bytes removing it would free."""
        removable = []
        for path, manifest in self.bundles():
            if manifest.incident_id in protected:
                continue
            try:
                with _exclusive(path):
                    freed = _freed_by(path, path)
            except BlockingIOError:
                continue
            removable.append((path, manifest.incident_id, freed))
        return removable

    def take_pruned(self) -> list[PrunedBundle]:
        """Bundles removed to make room since the last call, for the ledger."""
        pruned, self._room_pruned = self._room_pruned, []
        return pruned

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
        # Cleared first: a deletion a reader defers during recovery stays.
        self._deferred.clear()
        self._empty_trash()
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
            if not (bundle / manifest.current).is_dir():
                # Nothing named survives: keep what is there, for a person.
                report.unreadable.append(bundle.name)
                continue
            report.generations_removed += self._remove_unnamed(bundle, manifest)
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
        pruned = self._prune_unreadable(cutoff, protected)
        candidates = [
            (path, manifest)
            for path, manifest in self.bundles()
            if manifest.incident_id not in protected
        ]
        count = len(self.bundles())
        for path, manifest in candidates:
            reason = self._prune_reason(manifest, count, cutoff)
            if reason is None:
                continue
            size = _payload_bytes(path, seen=set())
            count -= 1  # deferred or not, it is on its way out
            if self._try_delete(path, path):
                pruned.append(PrunedBundle(manifest.incident_id, reason, size))
        return pruned

    def _prune_unreadable(
        self, cutoff_ns: int, protected: frozenset[str]
    ) -> list[PrunedBundle]:
        """Bundles whose manifest exists but cannot be read (corrupt, or a
        future schema) are charged; they go by age, from their directory's
        modification time, so they never hold their bytes for good."""
        pruned = []
        for path in self._bundle_dirs():
            if path.name in protected or not _unreadable(path):
                continue
            if path.stat().st_mtime_ns >= cutoff_ns:
                continue
            size = _payload_bytes(path, seen=set())
            if self._try_delete(path, path):
                pruned.append(PrunedBundle(path.name, "max_age_hours", size))
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

    def _try_delete(self, path: Path, bundle: Path, *, defer: bool = True) -> bool:
        """Delete ``path`` (a generation, or the whole bundle) under the
        bundle's exclusive lock, and forget exactly the bytes that freed.

        While a reader holds the lock the deletion is deferred, unless
        ``defer`` is False, and the bytes stay charged until
        :meth:`reclaim_deferred` succeeds. What was freed
        is measured under the lock, just before the deletion: a hard link
        another generation keeps frees nothing, and a path already gone
        frees nothing, so no byte is ever forgotten twice.
        """
        try:
            with _exclusive(bundle):
                freed = _freed_by(path, bundle)
                self._remove(path, bundle)
        except BlockingIOError:
            if defer:
                self._deferred[path] = bundle
            return False
        self._deferred.pop(path, None)
        if path == bundle:
            # Anything deferred inside the bundle went with it.
            for inside in [p for p in self._deferred if bundle in p.parents]:
                del self._deferred[inside]
        self.budget.forget(freed)
        return True

    def _remove(self, path: Path, bundle: Path) -> None:
        """Delete a generation, or a whole bundle by first renaming it out of
        the store's namespace: a crash part-way through rmtree then leaves a
        trash directory recover() removes, never a bundle missing its
        manifest that it would seal again as interrupted."""
        if path != bundle or not path.exists():
            shutil.rmtree(path, ignore_errors=True)
            return
        trash = self.root / f"{TRASH_PREFIX}{bundle.name}"
        os.rename(bundle, trash)
        _fsync_dir(self.root)
        shutil.rmtree(trash, ignore_errors=True)

    def _empty_trash(self) -> None:
        for path in self.root.glob(f"{TRASH_PREFIX}*"):
            shutil.rmtree(path, ignore_errors=True)

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
    in between: the manifest is read again, up to ``attempts`` times. This
    guards against a stale manifest only: a file can still go before the
    caller reads it. :func:`read_bundle_file` reads one file safely without
    the lock, and :func:`open_incident_bundle` pins the whole bundle.
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


def read_bundle_file(path: str | Path, name: str, *, attempts: int = 3) -> bytes:
    """One file of the current generation, by its name inside it, read
    without the lock: a file that vanishes while it is read means a newer
    generation was published, so the manifest is read again."""
    for _ in range(max(1, attempts)):
        view = read_manifest_snapshot(path, attempts=attempts)
        try:
            return view.file(name).read_bytes()
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"{path}: {name} changed under {attempts} reads; "
        "use open_incident_bundle to pin the bundle"
    )


# ------------------------------------------------------------------ internals


def _read_manifest(bundle: Path) -> BundleManifest:
    payload = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return BundleManifest.from_dict(payload)


def _replace_manifest(bundle: Path, manifest: BundleManifest) -> None:
    os.replace(_stage_manifest(bundle, manifest), bundle / MANIFEST_FILENAME)
    _fsync_dir(bundle)


def _stage_manifest(bundle: Path, manifest: BundleManifest) -> Path:
    """The manifest written and fsynced beside the real one, not yet named."""
    temporary = bundle / (MANIFEST_FILENAME + ".tmp")
    data = (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return temporary


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

    The manifest and lock file are not charged, nor the filesystem blocks
    beyond each file's bytes; ``MAX_GENERATION_FILES`` bounds both per
    generation, and ``max_incidents`` how many bundles exist.
    """
    return bytes_on_disk(_generation_dirs(bundle), seen=seen)


def _unreadable(bundle: Path) -> bool:
    """A manifest is there, but cannot be read as one this store knows."""
    try:
        _read_manifest(bundle)
    except FileNotFoundError:
        return False  # unpublished: being written, or for recover()
    except (OSError, ValueError):
        return True
    return False


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


def _own(root: Path) -> int:
    """The store root's lock, held exclusively for the store's lifetime."""
    fd = os.open(root / STORE_LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise StoreInUse(f"another process holds the incident store {root}") from None
    return fd


def _pin(bundle: Path) -> int:
    """The bundle's lock, held shared, as a reader holds it."""
    fd = os.open(bundle / LOCK_FILENAME, os.O_RDONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
    except BaseException:
        os.close(fd)
        raise
    return fd


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
    "StoreInUse",
    "STATUS_INTERRUPTED",
    "open_incident_bundle",
    "read_bundle_file",
    "read_manifest_snapshot",
]
