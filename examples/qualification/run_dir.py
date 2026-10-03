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
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

LABEL_PREFIX = "q221-"
SUMS = "SHA256SUMS"


def new_label() -> str:
    """An opaque run label: ``q221-`` and 16 random hex digits."""
    return f"{LABEL_PREFIX}{secrets.token_hex(8)}"


@dataclass(frozen=True)
class RunDirectory:
    """Where one run is written, then published."""

    root: Path
    label: str

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
        """Make the layout under the partial directory.

        Raises:
            FileExistsError: when the run, or a partial one, already exists.
        """
        if self.final.exists():
            raise FileExistsError(self.final)
        self.partial.mkdir(parents=True)
        for directory in (self.run, self.truth, self.probes, self.reference):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def publish(self) -> Path:
        """Write ``SHA256SUMS`` over every file, then rename the run into
        place in one step."""
        lines = [
            f"{_sha256(path)}  {path.relative_to(self.partial).as_posix()}\n"
            for path in sorted(self.partial.rglob("*"))
            if path.is_file()
        ]
        (self.partial / SUMS).write_text("".join(lines), encoding="utf-8")
        os.replace(self.partial, self.final)
        return self.final


def verify(directory: Path) -> list[str]:
    """Files whose digest differs from ``SHA256SUMS``, or that are missing."""
    problems = []
    for line in (directory / SUMS).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif _sha256(path) != digest:
            problems.append(f"changed {name}")
    return problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["LABEL_PREFIX", "SUMS", "RunDirectory", "new_label", "verify"]
