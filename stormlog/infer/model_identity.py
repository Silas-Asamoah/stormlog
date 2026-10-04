"""Model weights bound to a launch: the only verified model identity.

A digest of a path taken after a server started says nothing about what it
loaded: the path may have changed in between. The runner therefore fixes
the weights before it launches, and points the server at exactly them:

- **A pinned hub snapshot.** The revision is resolved to a commit, and every
  file of that snapshot is hashed and checked against its blob's name
  (SHA-256 for a file stored in LFS, git's SHA-1 for the rest). The server
  gets ``--revision <commit> --tokenizer-revision <commit>`` and
  ``HF_HUB_OFFLINE=1``, so it cannot fetch anything else.
- **A staged local snapshot.** Every file of a local model directory is
  hashed, and the directory is copied (hard-linked where it can be) into a
  read-only, content-addressed directory, ``<store>/<weights_digest>/``.
  The server loads that directory, whose name is its content.

After each run the files are checked again, by size, modification time and
inode; a change makes the run a protocol failure. The evidence is
``pinned_commit_verified`` or ``staged_snapshot_verified``, and a server
description taken by the runner carries it, bound to the server's process.
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InferInputError
from .server_model import GIT_SHA1, SHA256, ModelFile
from .server_model import git_sha1_file as git_sha1
from .server_model import hub_snapshot
from .server_model import sha256_file as sha256
from .server_model import walk_files as walk
from .server_model import weights_digest

PINNED_HUB = "pinned_hub"
STAGED = "staged"
ROUTES = (PINNED_HUB, STAGED)
PINNED_COMMIT_VERIFIED = "pinned_commit_verified"
STAGED_SNAPSHOT_VERIFIED = "staged_snapshot_verified"


@dataclass(frozen=True)
class VerifiedModel:
    """Weights fixed before launch, and how to point the server at them."""

    route: str
    model: str
    server_args: tuple[str, ...]
    env: Mapping[str, str]
    directory: Path
    files: tuple[ModelFile, ...]
    commit: str | None = None
    stats: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)
    verified_at_ns: int = 0

    @property
    def evidence(self) -> str:
        return (
            PINNED_COMMIT_VERIFIED
            if self.route == PINNED_HUB
            else STAGED_SNAPSHOT_VERIFIED
        )

    def record(self) -> dict[str, Any]:
        """The model section of a server description, verified before launch."""
        return {
            "configured": self.model,
            "configured_revision": self.commit,
            "revision_immutable": self.commit is not None,
            "resolved_snapshot": self.commit,
            "directory": str(self.directory),
            "files": {item.path: item.to_record() for item in self.files},
            "weights_digest": weights_digest(self.files),
            "identity_evidence": self.evidence,
            "verified_at_ns": self.verified_at_ns,
        }


def prepare_model(spec: Mapping[str, Any]) -> VerifiedModel:
    """Fix the weights a plan names; InferInputError when they do not check out."""
    route = spec.get("route")
    if route == PINNED_HUB:
        return _pinned(spec)
    if route == STAGED:
        return _staged(spec)
    raise InferInputError(f"model route {route!r} is not one of {', '.join(ROUTES)}")


def changed_files(model: VerifiedModel) -> list[str]:
    """Files whose size, modification time or inode moved since verification."""
    changed = []
    for name, before in model.stats.items():
        try:
            now = _stat(model.directory / name)
        except OSError:
            changed.append(name)
            continue
        if now != before:
            changed.append(name)
    return changed


def _pinned(spec: Mapping[str, Any]) -> VerifiedModel:
    repo, cache = str(spec["repo"]), Path(str(spec["hub_cache"]))
    found = hub_snapshot(cache, repo, spec.get("revision"))
    if found is None:
        raise InferInputError(
            f"model {repo}@{spec.get('revision') or 'main'}: no snapshot in {cache}"
        )
    directory, commit = found
    files = tuple(_checked_blob(directory, path) for path in walk(directory))
    return VerifiedModel(
        route=PINNED_HUB,
        model=repo,
        server_args=("--revision", commit, "--tokenizer-revision", commit),
        env={"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(cache)},
        directory=directory,
        files=files,
        commit=commit,
        stats=_stats(directory),
        verified_at_ns=time.time_ns(),
    )


def _checked_blob(directory: Path, path: Path) -> ModelFile:
    """A snapshot file whose content matches the digest its blob is named by."""
    blob = path.resolve()
    expected = blob.name
    algorithm = SHA256 if len(expected) == 64 else GIT_SHA1
    actual = sha256(blob) if algorithm == SHA256 else git_sha1(blob)
    relative = path.relative_to(directory).as_posix()
    if actual != expected:
        raise InferInputError(
            f"model file {relative}: content {actual[:12]} does not match its "
            f"blob {expected[:12]}"
        )
    return ModelFile(relative, algorithm, actual, blob.stat().st_size, checked=True)


def _staged(spec: Mapping[str, Any]) -> VerifiedModel:
    source, store = Path(str(spec["source"])), Path(str(spec["store"]))
    if not source.is_dir():
        raise InferInputError(f"model source {source} is not a directory")
    files = tuple(
        ModelFile(
            path.relative_to(source).as_posix(),
            SHA256,
            sha256(path),
            path.stat().st_size,
            checked=True,
        )
        for path in walk(source)
    )
    digest = weights_digest(files)
    assert digest is not None
    target = store / digest
    if target.exists():
        _check_staged(target, files)
    else:
        _stage(source, target, files)
    return VerifiedModel(
        route=STAGED,
        model=str(target),
        server_args=(),
        env={"HF_HUB_OFFLINE": "1"},
        directory=target,
        files=files,
        stats=_stats(target),
        verified_at_ns=time.time_ns(),
    )


def _stage(source: Path, target: Path, files: Sequence[ModelFile]) -> None:
    """Copy (or hard-link) into a temporary directory, then rename and lock it."""
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    for item in files:
        destination = partial / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source / item.path, destination)
        except OSError:
            shutil.copy2(source / item.path, destination)
    _check_staged(partial, files)
    partial.rename(target)
    _read_only(target)


def _check_staged(directory: Path, files: Sequence[ModelFile]) -> None:
    present = {path.relative_to(directory).as_posix() for path in walk(directory)}
    expected = {item.path for item in files}
    if present != expected:
        raise InferInputError(f"staged model {directory}: files differ from its name")
    for item in files:
        if sha256(directory / item.path) != item.digest:
            raise InferInputError(f"staged model {directory}: {item.path} changed")


def _read_only(directory: Path) -> None:
    for root, dirs, names in os.walk(directory):
        for name in names:
            os.chmod(Path(root) / name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        for name in dirs:
            os.chmod(Path(root) / name, 0o555)
    os.chmod(directory, 0o555)


def _stats(directory: Path) -> dict[str, tuple[int, int, int]]:
    return {
        path.relative_to(directory).as_posix(): _stat(path) for path in walk(directory)
    }


def _stat(path: Path) -> tuple[int, int, int]:
    info = path.stat()  # follows the snapshot's links to their blobs
    return info.st_size, info.st_mtime_ns, info.st_ino


__all__ = [
    "PINNED_COMMIT_VERIFIED",
    "PINNED_HUB",
    "ROUTES",
    "STAGED",
    "STAGED_SNAPSHOT_VERIFIED",
    "VerifiedModel",
    "changed_files",
    "prepare_model",
]
