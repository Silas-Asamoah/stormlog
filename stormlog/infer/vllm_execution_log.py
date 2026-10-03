"""Read the vLLM execution hook's raw log (``docs/vllm_execution.md``).

The hook writes one directory per process lifetime, an *epoch*, under
``<host>-<boot>/<role>-<pid>-<start ns>/``: sealed segments ``NNNNNN.jsonl``,
one open ``NNNNNN.jsonl.part``, a ``status.json`` the writer overwrites every
second, and a ``key`` for pseudonyms. Every record carries the epoch and a
sequence number, so a reader takes sealed segments whole and only the complete
lines of an open one, and tells records apart by ``(epoch, seq)`` rather than by
the file they came from. Nothing here interprets the records; the reducer in
``vllm_execution`` does.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "stormlog.vllm_hook/1"
RECORD_KINDS = frozenset(
    {"hello", "alias", "scheduled", "completed", "terminal", "heartbeat", "goodbye"}
)
# A process with no heartbeat for this long is taken as gone: its epoch has
# ended without a goodbye, and its pending iterations will never complete.
SILENCE_NS = 30 * 1_000_000_000
STATE_ENDED = "ended"
STATE_GONE = "gone"
STATE_ALIVE = "alive"

_EPOCH_NAME = re.compile(r"^(engine|worker)-(\d+)-(\d+)$")
_SEGMENT_NAME = re.compile(r"^(\d{6})\.jsonl(\.part)?$")


@dataclass(frozen=True)
class RawRecord:
    """One line of the raw log, as written."""

    epoch: str
    seq: int
    kind: str
    data: dict[str, Any]


@dataclass
class EpochRead:
    """What one epoch directory held when it was read."""

    directory: Path
    host_boot: str
    epoch: str
    role: str
    pid: int
    start_ns: int
    records: list[RawRecord] = field(default_factory=list)
    hello: dict[str, Any] | None = None
    goodbye: dict[str, Any] | None = None
    status: dict[str, Any] | None = None
    key: bytes | None = None
    last_seq: int | None = None
    high_water_before: int | None = None
    truncated: bool = False
    gaps: int = 0
    errors: list[str] = field(default_factory=list)
    last_seen_wall_ns: int | None = None
    state: str = STATE_ALIVE

    @property
    def host(self) -> str:
        """The hello's host; the directory name cannot be split reliably,
        since both a hostname and a boot ID may contain dashes."""
        return _hello_text(self.hello, "host") or self.host_boot

    @property
    def boot_id(self) -> str | None:
        return _hello_text(self.hello, "boot_id")

    def of_kind(self, kind: str) -> list[RawRecord]:
        return [record for record in self.records if record.kind == kind]

    def summary(self) -> dict[str, Any]:
        """The epoch's state for an import summary; no record content."""
        status = self.status or {}
        return {
            "host_boot": self.host_boot,
            "role": self.role,
            "pid": self.pid,
            "start_ns": self.start_ns,
            "state": self.state,
            "records": len(self.records),
            "last_seq": self.last_seq,
            "high_water_seq": (
                self.last_seq if self.last_seq is not None else self.high_water_before
            ),
            "truncated": self.truncated,
            "gaps": self.gaps,
            "dropped": dict(status.get("dropped") or {}),
            "capped": bool(status.get("capped", False)),
            "pending_iterations": status.get("pending_iterations"),
            "range_misses": status.get("range_misses"),
            "errors": list(self.errors),
        }


@dataclass
class LogRead:
    """Every epoch under a hook directory, read once."""

    directory: Path
    epochs: list[EpochRead]
    notes: list[str] = field(default_factory=list)

    def engines(self) -> list[EpochRead]:
        return [epoch for epoch in self.epochs if epoch.role == "engine"]

    def workers(self) -> list[EpochRead]:
        return [epoch for epoch in self.epochs if epoch.role == "worker"]


def read_execution_log(
    directory: str | Path,
    *,
    high_water: dict[str, int] | None = None,
    now_ns: int | None = None,
) -> LogRead:
    """Read every epoch under ``directory``.

    ``high_water`` maps an epoch name to the highest ``seq`` a previous import
    took; records at or below it are not returned again, though they still
    count towards the epoch's gaps and last sequence. ``now_ns`` is the wall
    time the silence rule is judged against (the current time by default).
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    marks = high_water or {}
    now = time.time_ns() if now_ns is None else now_ns
    read = LogRead(root, [])
    for host_boot, epoch_dir in _epoch_directories(root):
        epoch = _epoch_from_directory(host_boot, epoch_dir, marks)
        if epoch is None:
            read.notes.append(f"{epoch_dir}: not an epoch directory")
            continue
        _read_epoch(epoch, now)
        read.epochs.append(epoch)
    return read


def _epoch_directories(root: Path) -> list[tuple[Path, Path]]:
    """Every (host-boot, epoch) directory pair under the root, in name order."""
    pairs: list[tuple[Path, Path]] = []
    for host_boot in sorted(_subdirectories(root)):
        pairs.extend((host_boot, child) for child in sorted(_subdirectories(host_boot)))
    return pairs


def _subdirectories(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.is_dir()]


def _epoch_from_directory(
    host_boot: Path, epoch_dir: Path, marks: dict[str, int]
) -> EpochRead | None:
    """An unread epoch for a ``<role>-<pid>-<start ns>`` directory, else None."""
    match = _EPOCH_NAME.match(epoch_dir.name)
    if match is None:
        return None
    return EpochRead(
        directory=epoch_dir,
        host_boot=host_boot.name,
        epoch=epoch_dir.name,
        role=match.group(1),
        pid=int(match.group(2)),
        start_ns=int(match.group(3)),
        high_water_before=marks.get(epoch_dir.name),
    )


def _read_epoch(epoch: EpochRead, now_ns: int) -> None:
    seen: dict[int, RawRecord] = {}
    for segment in _segments(epoch.directory):
        for record in _read_segment(epoch, segment):
            # The first delivery of a sequence number is the one kept.
            seen.setdefault(record.seq, record)
    epoch.status = _read_json_file(epoch.directory / "status.json", epoch.errors)
    epoch.key = _read_key(epoch.directory / "key", epoch.errors)
    _settle(epoch, seen, now_ns)


def _segments(directory: Path) -> list[Path]:
    """Segments in sequence order; a sealed one before an open one of a number."""
    found = []
    for path in directory.iterdir():
        match = _SEGMENT_NAME.match(path.name)
        if match is not None and path.is_file():
            found.append((int(match.group(1)), match.group(2) is not None, path))
    return [path for _number, _open, path in sorted(found)]


def _read_segment(epoch: EpochRead, path: Path) -> list[RawRecord]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        epoch.errors.append(f"{path.name}: {exc}")
        return []
    lines = data.split(b"\n")
    tail = lines.pop()  # empty when the data ended with a newline
    if tail:
        if path.suffix == ".part":
            # A writer mid-line: the complete lines are taken, the rest waits.
            epoch.truncated = True
        else:
            epoch.errors.append(f"{path.name}: sealed segment ends mid-line")
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = _parse_line(epoch, line, f"{path.name}:{number}")
        if record is not None:
            records.append(record)
    return records


def _parse_line(epoch: EpochRead, line: bytes, where: str) -> RawRecord | None:
    try:
        data = json.loads(line)
    except ValueError as exc:
        epoch.errors.append(f"{where}: {exc}")
        return None
    problem = _check_record(epoch, data)
    if problem is not None:
        epoch.errors.append(f"{where}: {problem}")
        return None
    return RawRecord(epoch.epoch, int(data["seq"]), str(data["kind"]), data)


def _check_record(epoch: EpochRead, data: Any) -> str | None:
    if not isinstance(data, dict):
        return "not an object"
    if data.get("format") != FORMAT:
        return f"format {data.get('format')!r} is not {FORMAT}"
    if data.get("epoch") != epoch.epoch:
        return f"epoch {data.get('epoch')!r} is not this directory's"
    seq = data.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        return "seq must be a non-negative integer"
    if not isinstance(data.get("kind"), str) or not data["kind"]:
        return "kind must be a non-empty string"
    return None


def _read_json_file(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path.name}: not an object")
        return None
    return data


def _read_key(path: Path, errors: list[str]) -> bytes | None:
    if not path.is_file():
        return None
    try:
        key = path.read_bytes()
    except OSError as exc:
        errors.append(f"{path.name}: {exc}")
        return None
    if len(key) < 16:
        errors.append(f"{path.name}: too short to be a key")
        return None
    return key


def _settle(epoch: EpochRead, seen: dict[int, RawRecord], now_ns: int) -> None:
    """Fill the epoch's records, sequence facts and liveness from what was read."""
    _sequence_facts(epoch, seen)
    mark = epoch.high_water_before
    epoch.records = [
        record for seq, record in sorted(seen.items()) if mark is None or seq > mark
    ]
    for record in seen.values():
        if record.kind == "hello":
            epoch.hello = record.data
        elif record.kind == "goodbye":
            epoch.goodbye = record.data
    epoch.last_seen_wall_ns = _last_seen(epoch, seen)
    epoch.state = _state(epoch, now_ns)


def _sequence_facts(epoch: EpochRead, seen: dict[int, RawRecord]) -> None:
    """Gaps and the highest sequence, from the records and the status file."""
    if seen:
        lowest, highest = min(seen), max(seen)
        epoch.gaps = highest - lowest + 1 - len(seen)
        epoch.last_seq = highest
    status_seq = _integer((epoch.status or {}).get("last_seq"))
    if status_seq is not None:
        epoch.last_seq = max(status_seq, epoch.last_seq or -1)


def _last_seen(epoch: EpochRead, seen: dict[int, RawRecord]) -> int | None:
    """The latest wall time the process is known to have been alive."""
    stamps = [_wall_stamp(record) for record in seen.values()]
    stamps.append(_integer((epoch.status or {}).get("wall_ns")))
    known = [stamp for stamp in stamps if stamp is not None]
    return max(known) if known else None


def _wall_stamp(record: RawRecord) -> int | None:
    """When a liveness record was written; other kinds say nothing about it."""
    if record.kind == "hello":
        return _integer((record.data.get("clock") or {}).get("wall_ns"))
    if record.kind in {"heartbeat", "goodbye"}:
        return _integer(record.data.get("wall_ns"))
    return None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _hello_text(hello: dict[str, Any] | None, key: str) -> str | None:
    value = (hello or {}).get(key)
    return value if isinstance(value, str) and value else None


def _state(epoch: EpochRead, now_ns: int) -> str:
    if epoch.goodbye is not None:
        return STATE_ENDED
    seen = epoch.last_seen_wall_ns
    if seen is None or now_ns - seen >= SILENCE_NS:
        return STATE_GONE
    return STATE_ALIVE


__all__ = [
    "FORMAT",
    "RECORD_KINDS",
    "SILENCE_NS",
    "STATE_ALIVE",
    "STATE_ENDED",
    "STATE_GONE",
    "EpochRead",
    "LogRead",
    "RawRecord",
    "read_execution_log",
]
