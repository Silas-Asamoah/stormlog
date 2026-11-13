"""Read a growing inference artifact in bounded batches (design §4.3).

The reader keeps one position in one JSONL artifact and, on each call,
opens the file through the opener it was given, reads from that position
and closes it again. It never holds more than one batch, or one line of at
most ``max_line_bytes``, in memory, so it can follow a file of any size
while ``stormlog infer profile`` is still writing it.

Lines are split on ``b"\\n"`` only and numbered from 0, blank lines
included, as the report contract and :mod:`diagnosis_inputs` count them. A
line's SHA-256 covers its bytes without the newline
(:func:`diagnosis_inputs.line_digest`). Only complete lines are consumed:
the bytes after the last newline are a pending tail, read again on the next
call, and reported as ``truncated_final_line`` once they have stayed the
same for ``tail_grace_s``.

Every record is decoded with one :class:`CorrelationRecordDecoder` kept
across calls, so a compact (v4) line finds the context line that defined
its ``context_id`` however far back it was. A line that is not JSON, not an
object or not decodable becomes a :class:`LineProblem`; nothing a line holds
can make the reader raise.

:meth:`ArtifactReader.identity_change` tells a caller whether the file is
the one already read (§4.3's table): ``same``, ``grown``,
``replaced_same_prefix`` (an import replaced it through ``os.replace``,
copying the bytes read so far) or ``changed`` (a rerun truncated and
rewrote it, or another file took its place), after which the caller calls
:meth:`ArtifactReader.reset` and reads from byte 0 again.

The line, problem and identity types are Studio's step-0 contract types,
so the reader's output needs no conversion in Studio's index.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal

from stormlog.infer.correlation_codec import CorrelationRecordDecoder
from stormlog.infer.diagnosis_inputs import line_digest
from stormlog.scrub import scrub_text
from stormlog.studio.contracts import FileIdentity, LineProblem, LineRef, ProblemCode

__all__ = [
    "BATCH_BYTES",
    "MAX_LINE_BYTES",
    "TAIL_GRACE_S",
    "ArtifactReader",
    "Change",
    "ItemKind",
    "ReadItem",
]

Change = Literal["same", "grown", "replaced_same_prefix", "changed"]
ItemKind = Literal["record", "blank", "context", "problem"]

# §8.1: a longer line is skipped unparsed, as "oversized".
MAX_LINE_BYTES = 64 * 2**20
# How much one read_batch reads, unless one line is longer.
BATCH_BYTES = 8 * 2**20
# §4.3: a pending tail unchanged this long is reported as truncated.
TAIL_GRACE_S = 5.0
# LineProblem.detail is at most this many characters.
DETAIL_CHARS = 200
# Bytes per read while hashing or skipping a long stretch of the file.
_BLOCK = 2**20


@dataclass(frozen=True)
class ReadItem:
    """One line the reader consumed, or the pending tail it reports.

    ``record`` is the decoded record (a compact line with its context
    restored), or None for a blank line, a context definition and a
    problem. ``kind`` says which of the four the line is.
    """

    ref: LineRef
    record: dict[str, Any] | None
    problem: LineProblem | None
    kind: ItemKind = "record"


@dataclass
class _Tail:
    """The pending tail: where it starts, how long it is, since when."""

    offset: int
    length: int
    since: float
    problem: LineProblem | None = None


@dataclass
class _Mark:
    """The reader's position at the start of the last batch, for rewind."""

    offset: int
    line: int
    hasher: Any  # hashlib's sha256 object, copied
    last: LineRef | None
    refs: list[LineRef]


def _identity(stat: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _same_file(a: FileIdentity, b: FileIdentity) -> bool:
    return a.device == b.device and a.inode == b.inode


def _hash_range(handle: BinaryIO, offset: int, length: int, hasher: Any) -> int:
    """Feed ``length`` bytes from ``offset`` into ``hasher``; bytes fed."""
    handle.seek(offset)
    left = length
    while left > 0:
        block = handle.read(min(_BLOCK, left))
        if not block:
            break
        hasher.update(block)
        left -= len(block)
    return length - left


def _detail(text: str) -> str:
    return scrub_text(text, max_bytes=4 * DETAIL_CHARS)[:DETAIL_CHARS]


def _problem(ref: LineRef, code: ProblemCode, detail: str) -> ReadItem:
    return ReadItem(ref, None, LineProblem(ref, code, _detail(detail)), "problem")


class ArtifactReader:
    """Consume one artifact's complete lines, batch by batch.

    ``opener`` returns a new binary handle on the artifact each call
    (Studio passes ``lambda: root.open(rel)``); the reader closes it. The
    public attributes describe what has been consumed: ``offset`` (bytes,
    complete lines only), ``next_line`` (the number the next line gets),
    ``pending_tail_bytes`` (bytes after the last newline at the last read
    that reached the end of the file), ``prefix_sha256`` (of bytes
    ``[0, offset)``; the file's SHA-256 when nothing is pending) and
    ``identity`` (the file as last statted; None before the first call).
    """

    def __init__(
        self,
        opener: Callable[[], BinaryIO],
        *,
        max_line_bytes: int = MAX_LINE_BYTES,
        tail_grace_s: float = TAIL_GRACE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_line_bytes < 1:
            raise ValueError("max_line_bytes must be positive")
        self._opener = opener
        self.max_line_bytes = max_line_bytes
        self.tail_grace_s = tail_grace_s
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Forget everything read: byte 0, line 0, a new decoder."""
        self.offset = 0
        self.next_line = 0
        self.pending_tail_bytes = 0
        self.identity: FileIdentity | None = None
        self._hasher = hashlib.sha256()
        self._decoder = CorrelationRecordDecoder()
        self._first: LineRef | None = None
        self._last: LineRef | None = None
        self._tail: _Tail | None = None
        self._mark = _Mark(0, 0, self._hasher.copy(), None, [])

    @property
    def prefix_sha256(self) -> str:
        """SHA-256 of the bytes consumed so far, ``[0, offset)``."""
        return str(self._hasher.copy().hexdigest())

    @property
    def tail_problem(self) -> LineProblem | None:
        """The pending tail's ``truncated_final_line`` problem, once due."""
        return None if self._tail is None else self._tail.problem

    # --- Change detection ---------------------------------------------

    def identity_change(self) -> Change:
        """How the file differs from the one read so far (§4.3).

        ``same``: the same device, inode, size and mtime. ``grown``: the
        same inode, at least as long as what was consumed, and line 0 and
        the last consumed line still hash as they did. ``changed``: an
        anchor line differs or the file is shorter than what was consumed
        (a rerun truncated and rewrote it), or a new inode does not start
        with the consumed bytes. ``replaced_same_prefix``: a new inode
        that starts with exactly the consumed bytes (an import).

        Before anything is read, a non-empty file is ``grown`` and an empty
        one ``same``. Updates ``identity`` unless the answer is
        ``changed``, after which the caller calls :meth:`reset`. Raises
        what the opener raises (OSError, or Studio's path errors).
        """
        with self._opener() as handle:
            current = _identity(os.fstat(handle.fileno()))
            change = self._classify(handle, current)
        if change != "changed":
            self.identity = current
        return change

    def _classify(self, handle: BinaryIO, current: FileIdentity) -> Change:
        previous = self.identity
        if previous is None:
            return "grown" if current.size > self.offset else "same"
        if not _same_file(current, previous):
            return self._replaced(handle, current)
        if current == previous:
            return "same"
        if current.size < self.offset or not self._anchors_match(handle):
            return "changed"
        return "grown"

    def _replaced(self, handle: BinaryIO, current: FileIdentity) -> Change:
        if current.size < self.offset:
            return "changed"
        hasher = hashlib.sha256()
        if _hash_range(handle, 0, self.offset, hasher) < self.offset:
            return "changed"
        if hasher.hexdigest() != self.prefix_sha256:
            return "changed"
        return "replaced_same_prefix"

    def _anchors_match(self, handle: BinaryIO) -> bool:
        return all(
            self._line_matches(handle, ref)
            for ref in (self._first, self._last)
            if ref is not None
        )

    @staticmethod
    def _line_matches(handle: BinaryIO, ref: LineRef) -> bool:
        hasher = hashlib.sha256()
        if _hash_range(handle, ref.offset, ref.length, hasher) < ref.length:
            return False
        return handle.read(1) == b"\n" and hasher.hexdigest() == ref.sha256

    # --- Reading ------------------------------------------------------

    def read_batch(self, *, max_bytes: int = BATCH_BYTES) -> list[ReadItem]:
        """The next complete lines, about ``max_bytes`` of them.

        Reads at least one whole line when one is complete, however long:
        a line over ``max_line_bytes`` is hashed and skipped in blocks, as
        an ``oversized`` problem. Returns [] when no complete line follows
        ``offset``. A pending tail that has stayed the same for
        ``tail_grace_s`` is returned once as a ``truncated_final_line``
        item, which consumes nothing; :attr:`tail_problem` keeps it until a
        newline completes the line or the tail changes.

        Call :meth:`identity_change` first: this reads from ``offset`` in
        whatever file the opener gives, and returns [] when that is not
        the file :attr:`identity` names.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        with self._opener() as handle:
            current = _identity(os.fstat(handle.fileno()))
            if self.identity is None:
                self.identity = current
            elif not _same_file(current, self.identity):
                return []
            self._mark = _Mark(
                self.offset, self.next_line, self._hasher.copy(), self._last, []
            )
            return self._scan(handle, current.size, max_bytes)

    def _scan(self, handle: BinaryIO, size: int, max_bytes: int) -> list[ReadItem]:
        handle.seek(self.offset)
        data = handle.read(max_bytes)
        end = data.rfind(b"\n") + 1
        items: list[ReadItem] = []
        if end:
            self._consume(data, end, items)
            if len(data) < max_bytes:  # the read reached the end of the file
                self._settle_tail(handle, len(data) - end, items)
            elif self.offset >= size:
                self._settle_tail(handle, 0, items)
            return items
        if len(data) < max_bytes:
            self._settle_tail(handle, len(data), items)
        else:
            self._long_line(handle, data, items)
        return items

    def _consume(self, data: bytes, end: int, items: list[ReadItem]) -> None:
        """Consume the complete lines in ``data[:end]``."""
        self._hasher.update(memoryview(data)[:end])
        offset = self.offset
        for chunk in data[: end - 1].split(b"\n"):
            items.append(self._item(chunk, offset, len(chunk), line_digest(chunk)))
            offset += len(chunk) + 1
        self.offset = offset

    def _long_line(self, handle: BinaryIO, head: bytes, items: list[ReadItem]) -> None:
        """A line longer than one batch: kept if it fits, else hashed only."""
        prefix = self._hasher.copy()
        line_hasher = hashlib.sha256()
        parts: list[bytes] | None = []
        length = 0
        piece, found = head, False
        while True:
            prefix.update(piece)
            line_hasher.update(piece)
            length += len(piece)
            parts = None if parts is None or length > self.max_line_bytes else parts
            if parts is not None:
                parts.append(piece)
            if found:
                break
            block = handle.read(_BLOCK)
            if not block:
                self._settle_tail(handle, length, items)
                return
            cut = block.find(b"\n")
            found = cut >= 0
            piece = block[:cut] if found else block
        prefix.update(b"\n")
        self._hasher = prefix
        chunk = b"".join(parts) if parts is not None else b""
        items.append(self._item(chunk, self.offset, length, line_hasher.hexdigest()))
        self.offset += length + 1

    def _item(self, chunk: bytes, offset: int, length: int, sha256: str) -> ReadItem:
        ref = LineRef(self.next_line, offset, length, sha256)
        self.next_line += 1
        if self._first is None:
            self._first = ref
        self._last = ref
        self._mark.refs.append(ref)
        self.pending_tail_bytes = 0
        self._tail = None
        if length > self.max_line_bytes:
            return _problem(ref, "oversized", f"{length} bytes")
        return self._decode(ref, chunk)

    def _decode(self, ref: LineRef, chunk: bytes) -> ReadItem:
        if not chunk or chunk.isspace():
            return ReadItem(ref, None, None, "blank")
        try:
            raw = json.loads(chunk)
        except (ValueError, RecursionError) as exc:
            return _problem(ref, "not_json", str(exc))
        if not isinstance(raw, dict):
            return _problem(ref, "not_object", type(raw).__name__)
        try:
            record = self._decoder.decode(raw)
        except Exception as exc:  # any decoder failure is this line's problem
            return _problem(ref, "undecodable", f"{type(exc).__name__}: {exc}")
        if record is None:
            return ReadItem(ref, None, None, "context")
        return ReadItem(ref, record, None)

    # --- The pending tail ---------------------------------------------

    def _settle_tail(
        self, handle: BinaryIO, length: int, items: list[ReadItem]
    ) -> None:
        """Note the bytes after the last newline at the end of the file."""
        self.pending_tail_bytes = length
        if length == 0:
            self._tail = None
            return
        now = self._clock()
        tail = self._tail
        if tail is None or (tail.offset, tail.length) != (self.offset, length):
            tail = self._tail = _Tail(self.offset, length, now)
        if tail.problem is not None or now - tail.since < self.tail_grace_s:
            return
        hasher = hashlib.sha256()
        _hash_range(handle, self.offset, length, hasher)
        ref = LineRef(self.next_line, self.offset, length, hasher.hexdigest())
        detail = f"{length} bytes after the last newline, unchanged"
        item = _problem(ref, "truncated_final_line", detail)
        tail.problem = item.problem
        items.append(item)

    def rewind(self, ref: LineRef) -> None:
        """Unconsume ``ref``'s line and every line after it.

        ``ref`` must be a line the last :meth:`read_batch` returned. Studio's
        index calls this when a budget stops it part way through a batch,
        so ``offset`` and ``prefix_sha256`` name exactly the lines indexed.
        The decoder keeps any context definitions from the unconsumed
        lines; reading them again defines nothing new.
        """
        mark = self._mark
        position = ref.line - mark.line
        if not 0 <= position < len(mark.refs) or mark.refs[position] != ref:
            raise ValueError("rewind takes a line of the last batch")
        hasher = mark.hasher.copy()
        with self._opener() as handle:
            span = ref.offset - mark.offset
            if _hash_range(handle, mark.offset, span, hasher) < span:
                raise OSError("the file is shorter than the lines it rewinds to")
        self._hasher = hasher
        self.offset = ref.offset
        self.next_line = ref.line
        self._last = mark.refs[position - 1] if position else mark.last
        if self._first == ref:
            self._first = None
        del mark.refs[position:]
        self.pending_tail_bytes = 0
        self._tail = None
