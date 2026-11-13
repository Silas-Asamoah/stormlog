"""Raw export of a selection's original lines (design §5.6, §7.8; C4).

Slice 1 has one export profile, ``original_bytes`` (decision 13): the
selection's client request lines, byte for byte and in file order, each
with its newline. Nothing is re-serialized, so integer timestamps stay
exact and the file is valid input to ``stormlog infer analyze``. While
#111 is open there is no ``share`` profile; instead the export is behind a
required preview that counts the fields that may carry something
sensitive, and says plainly that the export is not a redaction.

:func:`stream` reads the lines through ``IndexRegistry.read_lines``, so the
export is never held whole, and checks each line's length and SHA-256
against the index. A line that changed raises ConflictError
``stale_line`` in the middle of the stream: httpd then ends the response
without its final chunk, and the browser reports a failed download
instead of saving a mixed file.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from stormlog.studio.contracts import (
    ConflictError,
    IndexRegistry,
    IndexSnapshot,
    LineRef,
    NotResident,
    Selection,
)

__all__ = [
    "PROFILE",
    "CONTENT_TYPE",
    "CHUNK_BYTES",
    "preview",
    "download_headers",
    "stream",
    "looks_like_local_path",
]

PROFILE = "original_bytes"
CONTENT_TYPE = "application/x-ndjson"
# stream() yields about this many bytes at a time: whole lines, joined.
CHUNK_BYTES = 64 * 2**10

_LOCAL_PATH = re.compile(r"(?:/|~|\.{1,2}/|[A-Za-z]:[\\/]|\\\\|file:)")
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

_NOT_A_REDACTION = (
    "The export copies the selected lines' original bytes. It is not a "
    "redaction: error messages, endpoints and model names are exported as "
    "written. A redacting profile waits for #111."
)
_NO_PHASE_WINDOWS = (
    "Only the selection's infer.request lines are exported, without the "
    "run's phase windows and workload records. infer analyze of the export "
    "counts and summarizes them, but divides its rates by the requests' own "
    "span (rate_reason no_phase_window), not by the case's rate interval, so "
    "they can differ from the run's."
)


def _version() -> str:
    from stormlog import __version__

    return str(__version__)


def looks_like_local_path(value: Any) -> bool:
    """Whether a model name looks like a path on someone's disk.

    Absolute, home-relative, dot-relative, Windows drive and UNC paths and
    ``file:`` URLs; a hub name such as ``org/model`` does not.
    """
    return isinstance(value, str) and _LOCAL_PATH.match(value) is not None


def preview(
    snapshot: IndexSnapshot,
    sel: Selection,
    *,
    selection_block: Mapping[str, Any],
) -> dict[str, Any]:
    """The body of ``stormlog.studio.export_preview`` (§5.6).

    ``selection_block`` is C2's ``summaries.selection_block(sel, snapshot)``,
    so the preview carries the same §5.3 block as every selection route.
    The counts read index columns only, never the file.
    """
    _check_same_index(snapshot, sel)
    rows = sel.rows
    return {
        "selection": dict(selection_block),
        "profile": PROFILE,
        "count": len(rows),
        "bytes": sum(_ref(snapshot, row).length + 1 for row in rows),
        "source": _source(snapshot),
        "stormlog_version": _version(),
        "content": {
            "event_types": ["infer.request"],
            "order": "file",
            "line_ending": "\n",
        },
        "sensitive_fields": _sensitive_fields(snapshot, rows),
        "limitations": _limitations(snapshot),
        "download": {
            "filename": _filename(snapshot.path),
            "content_type": CONTENT_TYPE,
        },
    }


def download_headers(
    snapshot: IndexSnapshot, sel: Selection
) -> tuple[tuple[str, str], ...]:
    """The export response's headers (§5.6), besides httpd's own.

    ``X-Stormlog-Source-Sha256`` is the whole file's SHA-256; while the
    index has not read the whole file it is absent and
    ``X-Stormlog-Source-Prefix-Sha256`` and ``-Bytes`` name the prefix read.
    """
    _check_same_index(snapshot, sel)
    headers = [
        ("Content-Disposition", f'attachment; filename="{_filename(snapshot.path)}"'),
        ("X-Stormlog-Export-Profile", PROFILE),
        ("X-Stormlog-Export-Count", str(len(sel.rows))),
        ("X-Stormlog-Selection", _header_text(sel.params)),
    ]
    if snapshot.sha256 is not None:
        headers.append(("X-Stormlog-Source-Sha256", snapshot.sha256))
    else:
        headers.append(("X-Stormlog-Source-Prefix-Sha256", snapshot.prefix_sha256))
        headers.append(("X-Stormlog-Source-Prefix-Bytes", str(snapshot.bytes_read)))
    return tuple(headers)


def stream(
    indexes: IndexRegistry, snapshot: IndexSnapshot, sel: Selection
) -> Iterator[bytes]:
    """The selection's original lines, each with its newline, in file order.

    Lazy: it reads lines through ``indexes.read_lines`` as the caller
    iterates and yields whole lines joined into chunks of about
    :data:`CHUNK_BYTES`. Each line must have the length and SHA-256 the
    index recorded, else ConflictError ``stale_line``, raised before the
    chunk holding that line is yielded.

    Its ``details`` always carry ``line``. ``reason`` (one of ``length``,
    ``sha256``, ``newline``, ``missing``, ``unexpected``) is there when
    this module's own checks catch the line. Usually the index's
    ``read_lines`` catches it first and raises its own ``stale_line``,
    which passes through unchanged: with the index's ``reason`` when it
    sends one, and today C1's sends only ``line``.
    """
    _check_same_index(snapshot, sel)
    return _verified(indexes, snapshot, sel.rows)


def _verified(
    indexes: IndexRegistry, snapshot: IndexSnapshot, rows: Sequence[int]
) -> Iterator[bytes]:
    pending: deque[LineRef] = deque()
    buffer: list[bytes] = []
    size = 0
    for line in indexes.read_lines(snapshot.path, _refs(snapshot, rows, pending)):
        if not pending:
            raise ConflictError(
                "stale_line",
                "the export read more lines than the selection holds",
                details={"line": None, "reason": "unexpected"},
            )
        _check_line(pending.popleft(), line)
        buffer.append(line)
        size += len(line)
        if size >= CHUNK_BYTES:
            yield b"".join(buffer)
            buffer, size = [], 0
    if pending:
        raise _stale(pending[0], "missing")
    if buffer:
        yield b"".join(buffer)


def _refs(
    snapshot: IndexSnapshot, rows: Iterable[int], pending: deque[LineRef]
) -> Iterator[LineRef]:
    """Each row's line, noted in ``pending`` as read_lines asks for it."""
    for row in rows:
        ref = _ref(snapshot, row)
        pending.append(ref)
        yield ref


def _check_line(ref: LineRef, line: bytes) -> None:
    if len(line) != ref.length + 1:
        raise _stale(ref, "length")
    if not line.endswith(b"\n"):
        raise _stale(ref, "newline")
    if hashlib.sha256(line[:-1]).hexdigest() != ref.sha256:
        raise _stale(ref, "sha256")


def _stale(ref: LineRef, reason: str) -> ConflictError:
    return ConflictError(
        "stale_line",
        f"line {ref.line} changed since the index read it; reopen the artifact",
        details={"line": ref.line, "reason": reason},
    )


def _ref(snapshot: IndexSnapshot, row: int) -> LineRef:
    return snapshot.requests[row].ref


def _check_same_index(snapshot: IndexSnapshot, sel: Selection) -> None:
    """A selection indexes rows of its own index and generation or older."""
    if (sel.path, sel.index_id) != (snapshot.path, snapshot.index_id) or (
        sel.generation > snapshot.generation
    ):
        raise ValueError("the selection belongs to another index snapshot")


def _source(snapshot: IndexSnapshot) -> dict[str, Any]:
    identity = snapshot.identity
    return {
        "path": snapshot.path,
        "index_id": snapshot.index_id,
        "generation": snapshot.generation,
        "size": None if identity is None else identity.size,
        "bytes_read": snapshot.bytes_read,
        "sha256": snapshot.sha256,
        "prefix_sha256": snapshot.prefix_sha256,
        "records": sum(sum(counts.values()) for counts in snapshot.inventory.values()),
        "client_requests": len(snapshot.requests),
        "status": snapshot.status,
        "growing": snapshot.growing,
    }


def _sensitive_fields(
    snapshot: IndexSnapshot, rows: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        _error_messages(snapshot, rows),
        _field_count(
            snapshot,
            rows,
            "endpoint",
            "Endpoint URLs, as the writer recorded them through redact_url.",
            lambda value: value is not None,
        ),
        _field_count(
            snapshot,
            rows,
            "model",
            "Model names that look like a local path.",
            looks_like_local_path,
        ),
    ]


def _error_messages(snapshot: IndexSnapshot, rows: Sequence[int]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "field": "error_message",
        "description": (
            "Error messages, scrubbed by the writer; a server error body can "
            "still echo a prompt."
        ),
    }
    try:
        values = snapshot.column("error_message")
    except NotResident:
        return {**entry, "count": None, "distinct": None, "reason": "not_resident"}
    present = [values[row] for row in rows if values[row] is not None]
    return {
        **entry,
        "count": len(present),
        "distinct": len(set(map(str, present))),
        "reason": None,
    }


def _field_count(
    snapshot: IndexSnapshot,
    rows: Sequence[int],
    name: str,
    description: str,
    flagged: Any,
) -> dict[str, Any]:
    values = snapshot.column(name)
    hits = [values[row] for row in rows if flagged(values[row])]
    return {
        "field": name,
        "description": description,
        "count": len(hits),
        "distinct": len(set(map(str, hits))),
        "reason": None,
    }


def _limitations(snapshot: IndexSnapshot) -> list[dict[str, str]]:
    notes = [
        {"code": "not_a_redaction", "message": _NOT_A_REDACTION},
        {"code": "no_phase_windows", "message": _NO_PHASE_WINDOWS},
    ]
    if snapshot.sha256 is None:
        notes.append(
            {
                "code": "source_not_complete",
                "message": (
                    "The index has not read the whole file (it is growing or "
                    "its last line is incomplete); the export holds the "
                    "selection of the lines read so far."
                ),
            }
        )
    if snapshot.status == "partial":
        notes.append(
            {
                "code": "index_partial",
                "message": (
                    "The index stopped at a memory budget; requests past the "
                    "cut are not in the selection."
                ),
            }
        )
    return notes


def _filename(path: str) -> str:
    stem = path.rsplit("/", 1)[-1]
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    stem = _FILENAME_UNSAFE.sub("_", stem).strip("._") or "artifact"
    return f"{stem}.selection.jsonl"


def _header_text(text: str) -> str:
    """Header-safe text: ASCII printable only, the rest percent-encoded."""
    return "".join(
        char if " " <= char <= "~" else "".join(f"%{b:02X}" for b in char.encode())
        for char in text
    )
