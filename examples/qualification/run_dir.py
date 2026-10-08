"""A qualification run's directory: opaque label, layout, atomic publication.

```text
<root>/<label>/       q221-<16 hex>, opaque: it says nothing about the episodes
  run/      victim.jsonl                 the only path handed to the diagnoser
  truth/    injections.jsonl, episodes.json, plan.json, neighbor-<n>.jsonl, reference/
  probes/   markers/, append-times.jsonl, client-idle.jsonl, hook-firstseen.jsonl, ...
  SHA256SUMS                             written last
```

A run is written under ``<root>/.<label>.partial`` and renamed into place
only once ``SHA256SUMS`` is written, so a reader never sees half a run.
``SHA256SUMS`` lists every file and ends with a line giving its count and a
digest of the lines above it; the digest of the whole ``SHA256SUMS`` is kept
beside the run, in ``<root>/<label>.sha256``. Every file, the partial
directory and the root are fsynced around the rename, so a power loss can't
leave a published run with empty sums.

A run claims its label: its partial directory appears, laid out and with
the run's own nonce in ``.owner``, by one rename, and only a partial
directory carrying the run's nonce is published. Two runs given one label
can't publish into each other's directory.
"""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path

LABEL_PREFIX = "q221-"
SUMS = "SHA256SUMS"
COUNT_LINE = "# files "
# In the partial directory: the nonce of the run that claimed it.
OWNER = ".owner"


def new_label() -> str:
    """An opaque run label: ``q221-`` and 16 random hex digits."""
    return f"{LABEL_PREFIX}{secrets.token_hex(8)}"


@dataclass(frozen=True)
class RunDirectory:
    """Where one run is written, then published. ``owner`` is this run's
    nonce, which the partial directory carries while it is this run's."""

    root: Path
    label: str
    owner: str = field(default_factory=lambda: secrets.token_hex(8), compare=False)

    @property
    def partial(self) -> Path:
        return self.root / f".{self.label}.partial"

    @property
    def final(self) -> Path:
        return self.root / self.label

    @property
    def run(self) -> Path:
        return self.partial / "run"

    @property
    def truth(self) -> Path:
        return self.partial / "truth"

    @property
    def probes(self) -> Path:
        return self.partial / "probes"

    @property
    def reference(self) -> Path:
        return self.truth / "reference"

    def create(self) -> RunDirectory:
        """Claim the label: lay out a directory of this run's own, with its
        nonce in ``.owner``, and rename it to the partial directory in one
        step. The partial directory is then this run's and whole, or not
        this run's at all: interrupted before the rename, nothing is left
        under the label, and a run that claimed the label first keeps it.

        Raises:
            FileExistsError: when the run, or a partial one, already exists.
        """
        if self.final.exists():
            raise FileExistsError(self.final)
        staging = self.root / f".{self.label}.{self.owner}.new"
        staging.mkdir(parents=True)
        try:
            for directory in ("run", "truth/reference", "probes"):
                (staging / directory).mkdir(parents=True)
            (staging / OWNER).write_text(self.owner)
            _claim(staging, self.partial)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if self.final.exists():
            # Published since the check above, which freed the partial
            # directory's name: the label is still that run's.
            shutil.rmtree(self.partial)
            raise FileExistsError(self.final)
        return self

    def owned(self) -> bool:
        """Whether the partial directory is this run's: it carries its
        nonce."""
        try:
            return (self.partial / OWNER).read_text() == self.owner
        except OSError:
            return False

    def publish(self) -> Path:
        """fsync every file; write ``SHA256SUMS`` over all of them, with its
        count line, and its digest beside the run; then rename the run into
        place in one step and fsync the root. Only this run's partial
        directory is published, never over a published run.

        Raises:
            FileExistsError: when the partial directory isn't this run's, or
                the run is already published.
        """
        if not self.owned() or self.final.exists():
            raise FileExistsError(self.final)
        (self.partial / OWNER).unlink()
        files = sorted(path for path in self.partial.rglob("*") if path.is_file())
        for path in files:
            _fsync(path)
        body = "".join(
            f"{_sha256(path)}  {path.relative_to(self.partial).as_posix()}\n"
            for path in files
        )
        sums = body + _count_line(len(files), body)
        _write_synced(self.partial / SUMS, sums)
        for directory in sorted({path.parent for path in files} | {self.partial}):
            _fsync(directory)
        digest = hashlib.sha256(sums.encode("utf-8")).hexdigest()
        _write_synced(_beside(self.final), f"{digest}  {self.label}/{SUMS}\n")
        os.replace(self.partial, self.final)
        _fsync(self.root)
        return self.final


def _claim(staging: Path, partial: Path) -> None:
    """Rename ``staging`` to ``partial``, which fails if another run's
    partial directory is there: it is never empty."""
    try:
        os.rename(staging, partial)
    except OSError as error:
        if error.errno in (errno.ENOTEMPTY, errno.EEXIST):
            raise FileExistsError(partial) from error
        raise


def verify(directory: Path) -> list[str]:
    """Everything that doesn't match the published run: ``SHA256SUMS``
    against the digest kept beside the run and its own count line, every
    listed file against its digest, and every file against the list."""
    path = directory / SUMS
    if not path.is_file():
        return [f"missing {SUMS}"]
    text = path.read_text(encoding="utf-8")
    problems = _check_sums_file(directory, text)
    listed = _listed(text)
    for name, digest in listed.items():
        file = directory / name
        if not file.is_file():
            problems.append(f"missing {name}")
        elif _sha256(file) != digest:
            problems.append(f"changed {name}")
    present = {
        file.relative_to(directory).as_posix()
        for file in directory.rglob("*")
        if file.is_file() and file.name != SUMS
    }
    problems += [f"unlisted {name}" for name in sorted(present - set(listed))]
    return problems


def _check_sums_file(directory: Path, text: str) -> list[str]:
    problems = []
    beside = _beside(directory)
    if not beside.is_file():
        problems.append(f"no digest of {SUMS} beside the run ({beside.name})")
    elif beside.read_text().split()[0] != hashlib.sha256(text.encode()).hexdigest():
        problems.append(f"{SUMS} differs from the digest beside the run")
    body, _, last = text.rstrip("\n").rpartition("\n")
    body = body + "\n" if body else ""
    if _count_line(len(_listed(text)), body).rstrip("\n") != last:
        problems.append(f"{SUMS} is truncated or edited: its count line doesn't match")
    return problems


def _listed(text: str) -> dict[str, str]:
    listed = {}
    for line in text.splitlines():
        if line and not line.startswith(COUNT_LINE):
            digest, _, name = line.partition("  ")
            listed[name] = digest
    return listed


def _count_line(count: int, body: str) -> str:
    return f"{COUNT_LINE}{count} sha256 {hashlib.sha256(body.encode()).hexdigest()}\n"


def _beside(run: Path) -> Path:
    return run.parent / f"{run.name}.sha256"


def _write_synced(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["LABEL_PREFIX", "SUMS", "RunDirectory", "new_label", "verify"]
