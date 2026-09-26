"""Pinned prototype sources used by the comparative experiment."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

CUPTI_REFERENCE_REVISION = "d4ea783b7ef98aa2f0e3bfbcffc169e4612fd3a9"
CUPTI_REFERENCE_PATH = "native/cupti"
PARCAGPU_REVISION = "1e7e8da62513fd188c121716fc5028b4bd8ac47c"
NEUTRINO_REVISION = "4a82cd22f474c31ac2fecfa174d381a19bb3f469"


def extract_cupti_reference(repository: Path, destination: Path) -> Path:
    """Extract PR #237's CUPTI helper without merging production integration."""
    if destination.exists():
        raise FileExistsError(destination)
    revision = _git(repository, "rev-parse", CUPTI_REFERENCE_REVISION)
    if revision != CUPTI_REFERENCE_REVISION:
        raise RuntimeError("the pinned CUPTI reference revision is unavailable")
    archive = subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            CUPTI_REFERENCE_REVISION,
            CUPTI_REFERENCE_PATH,
        ),
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True, mode=0o700)
    _safe_extract(archive, destination)
    return destination / CUPTI_REFERENCE_PATH


def _safe_extract(archive: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            _extract_member(source, member, destination)


def _extract_member(
    source: tarfile.TarFile, member: tarfile.TarInfo, destination: Path
) -> None:
    target = (destination / member.name).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"unsafe archive member: {member.name}")
    if member.issym() or member.islnk():
        raise ValueError(f"links are not permitted: {member.name}")
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        return
    if not member.isfile():
        raise ValueError(f"unsupported archive member: {member.name}")

    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    extracted = source.extractfile(member)
    if extracted is None:
        raise ValueError(f"archive file has no content: {member.name}")
    with extracted, target.open("xb") as output:
        shutil.copyfileobj(extracted, output)
    mode = 0o700 if member.mode & 0o100 else 0o600
    os.chmod(target, mode)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
