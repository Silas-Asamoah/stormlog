"""Whether an experiment's bundle can be published: no secret left in it.

Every file is scanned for the secrets the plan named (their values, as the
runner passed them to commands) and for the shapes credentials take: an
``Authorization: Bearer`` value, a Hugging Face ``hf_`` token, an ``sk-``
key. A hit names the file and the line, never the value. A bundle with any
hit is not publishable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TOKEN_SHAPES = {
    "bearer": re.compile(rb"Bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    "huggingface_token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}"),
    "sk_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
}
MIN_SECRET_LENGTH = 4


def sanitize_bundle(
    directory: Path, secret_values: Iterable[str] = ()
) -> dict[str, Any]:
    """Scan every file under ``directory``; the report says if it is publishable."""
    secrets = [s.encode() for s in secret_values if len(s) >= MIN_SECRET_LENGTH]
    hits: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        scanned += 1
        hits += _file_hits(path, directory, secrets)
    return {
        "publishable": not hits,
        "files_scanned": scanned,
        "secrets_checked": len(secrets),
        "hits": hits,
    }


def _file_hits(path: Path, root: Path, secrets: list[bytes]) -> list[dict[str, Any]]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        return [{"file": _name(path, root), "line": None, "kind": f"unreadable: {exc}"}]
    hits = []
    for number, line in enumerate(content.splitlines(), start=1):
        kinds = ["secret_value" for secret in secrets if secret in line]
        kinds += [name for name, shape in TOKEN_SHAPES.items() if shape.search(line)]
        hits += [
            {"file": _name(path, root), "line": number, "kind": kind} for kind in kinds
        ]
    return hits


def _name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = ["TOKEN_SHAPES", "sanitize_bundle"]
