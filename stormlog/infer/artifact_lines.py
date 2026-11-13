"""Where a line of an artifact is, why it was not used, and which file it is.

The reader (:mod:`stormlog.infer.artifact_reader`) produces these types and
Studio's contract (:mod:`stormlog.studio.contracts`) re-exports them under
the same names, so a core module never imports Studio. This module imports
the standard library only.

Line numbers are 0-based with blank lines counted, lines are split on
``b"\\n"`` only, and a line's SHA-256 covers its bytes without the newline
(``docs/report_contract.md``; :func:`diagnosis_inputs.line_digest`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["FileIdentity", "LineProblem", "LineRef", "ProblemCode"]


@dataclass(frozen=True)
class LineRef:
    """Where one line of an artifact is, and what its bytes hash to."""

    line: int  # 0-based
    offset: int  # byte offset of the line's first byte
    length: int  # bytes, without the b"\n"
    sha256: str  # hex, of those bytes (diagnosis_inputs.line_digest)


ProblemCode = Literal[
    "not_json",
    "not_object",
    "undecodable",
    "oversized",
    "unsupported_version",
    "truncated_final_line",
]


@dataclass(frozen=True)
class LineProblem:
    """A line the reader or the index could not use, and why."""

    ref: LineRef
    code: ProblemCode
    detail: str  # scrubbed, at most 200 characters


@dataclass(frozen=True)
class FileIdentity:
    """What ``os.fstat`` said about the artifact at the last read."""

    device: int
    inode: int
    size: int
    mtime_ns: int
