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
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .host_clock import host_boot_id

FORMAT = "stormlog.vllm_hook/1"
RECORD_KINDS = frozenset(
    {"hello", "alias", "scheduled", "completed", "terminal", "heartbeat", "goodbye"}
)
# A process with no heartbeat for this long is taken as gone: its epoch has
# ended without a goodbye, and its pending iterations will never complete.
# Silence is judged on the server's own monotonic clock, which the reader
# shares only when it runs on the same host and boot.
SILENCE_NS = 30 * 1_000_000_000
STATE_ENDED = "ended"
STATE_GONE = "gone"
STATE_ALIVE = "alive"
# Not judged: the reader runs elsewhere, so an epoch without goodbye may be
# alive or gone, and its pending steps are left for a later import.
STATE_UNKNOWN = "unknown"
COVERAGE_BASIS = "heartbeat_counters/1"
LIVENESS_BASIS = "heartbeat_gaps/1"
# The hook's writer beats once a second, but under load its thread slips: on
# the real vLLM 0.30.0 runs from #217 the longest interval was 2.3 s, with
# nothing lost. A stretch over five beats is a gap it was not heard from in.
HEARTBEAT_GAP_NS = 5_000_000_000


@dataclass(frozen=True)
class Importer:
    """Where a read runs: its host, boot and monotonic clock, for judging an
    epoch's liveness on the clock that wrote it."""

    host: str
    boot_id: str | None
    monotonic_ns: int

    @classmethod
    def here(cls) -> Importer:
        return cls(socket.gethostname(), host_boot_id(), time.monotonic_ns())

    def shares_clock_with(self, epoch: EpochRead) -> bool:
        """Same host and boot means the same CLOCK_MONOTONIC."""
        return (
            self.boot_id is not None
            and epoch.boot_id == self.boot_id
            and epoch.host == self.host
        )


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
    # Every heartbeat read, including those an earlier import consumed.
    heartbeats: list[RawRecord] = field(default_factory=list)
    hello: dict[str, Any] | None = None
    goodbye: dict[str, Any] | None = None
    status: dict[str, Any] | None = None
    key: bytes | None = None
    # The highest sequence read, the highest one read with nothing missing
    # below it (what an import may safely consume), and the writer's own
    # count from status.json, which only says how far behind the read is.
    last_seq: int | None = None
    contiguous_seq: int | None = None
    writer_last_seq: int | None = None
    high_water_before: int | None = None
    truncated: bool = False
    gaps: int = 0
    errors: list[str] = field(default_factory=list)
    last_seen_wall_ns: int | None = None
    last_seen_mono_ns: int | None = None
    state: str = STATE_ALIVE
    state_reason: str | None = None

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

    @property
    def consumed_seq(self) -> int | None:
        """The mark an import may advance to from this read alone: the
        contiguous sequence, never the writer's count from status.json."""
        if self.contiguous_seq is None:
            return self.high_water_before
        if self.high_water_before is None:
            return self.contiguous_seq
        return max(self.contiguous_seq, self.high_water_before)

    def summary(self) -> dict[str, Any]:
        """The epoch's state for an import summary; no record content."""
        status = self.status or {}
        return {
            "host_boot": self.host_boot,
            "role": self.role,
            "pid": self.pid,
            "start_ns": self.start_ns,
            "state": self.state,
            "state_reason": self.state_reason,
            "records": len(self.records),
            "last_seq": self.last_seq,
            "contiguous_seq": self.contiguous_seq,
            "writer_last_seq": self.writer_last_seq,
            "high_water_seq": self.consumed_seq,
            "truncated": self.truncated,
            "gaps": self.gaps,
            "dropped": dict(status.get("dropped") or {}),
            "capped": bool(status.get("capped", False)),
            "queued": status.get("queued"),
            "pending_iterations": status.get("pending_iterations"),
            # Worker counters: serving calls without a range, and the warm-up
            # and graph-capture calls before the first serving step.
            "range_misses": status.get("range_misses"),
            "startup_unranged": status.get("startup_unranged"),
            "pending_samples": status.get("pending_samples"),
            "errors": list(self.errors),
            "coverage": self.coverage(),
            "liveness": self.liveness(),
        }

    def liveness(self) -> dict[str, Any]:
        """When the hook's writer was heard from: its first and last
        heartbeat, the longest interval between two, and every stretch
        between consecutive heartbeats longer than ``HEARTBEAT_GAP_NS``, on
        the server's monotonic and wall clocks.
        An interval between the first and the last that no gap overlaps had
        a writer beating throughout; it says nothing of what was lost."""
        beats = [
            beat
            for beat in self.heartbeats
            if _integer(beat.data.get("mono_ns")) is not None
        ]
        pairs = list(zip(beats, beats[1:]))
        intervals = [b.data["mono_ns"] - a.data["mono_ns"] for a, b in pairs]
        gaps = [
            {**_span_start(before), **_span_end(after)}
            for (before, after), interval in zip(pairs, intervals)
            if interval > HEARTBEAT_GAP_NS
        ]
        return {
            "basis": LIVENESS_BASIS,
            "gap_ns": HEARTBEAT_GAP_NS,
            "heartbeats": len(beats),
            "max_interval_ns": max(intervals) if intervals else None,
            "first": _heard(beats[0]) if beats else None,
            "last": _heard(beats[-1]) if beats else None,
            "gaps": gaps,
        }

    def coverage(self) -> dict[str, Any]:
        """Where the log is known to be whole: spans between two heartbeats
        whose drop counts and errors are unchanged, the writer not capped,
        with every record read that was accepted by the later one's stamp.
        A heartbeat says how many accepted records it was written ahead of
        (``pending``); they take the next sequences, so the span reaches
        past them, and the first heartbeat after them must show nothing lost
        while they were written. Only there does a kind the hello
        ``observes`` prove absent by having no record, matched by the
        record's last stamp, taken just before it is emitted."""
        observes = (self.hello or {}).get("observes")
        spans: list[dict[str, int]] = []
        beats = self.heartbeats
        joined: int | None = None
        for index, (before, after) in enumerate(zip(beats, beats[1:])):
            close = _close(after, beats[index + 2 :])
            if not self._whole_between(before, after, close, beats[index + 2 :]):
                continue
            if spans and joined == before.seq:
                spans[-1].update(_span_end(after), end_seq=close)
            else:
                spans.append(
                    {**_span_start(before), **_span_end(after), "end_seq": close}
                )
            joined = after.seq
        return {
            "basis": COVERAGE_BASIS,
            "observes": sorted(observes) if isinstance(observes, list) else None,
            "heartbeats": len(self.heartbeats),
            "spans": spans,
        }

    def _whole_between(
        self,
        before: RawRecord,
        after: RawRecord,
        close: int,
        later: Sequence[RawRecord],
    ) -> bool:
        if self.contiguous_seq is None or close > self.contiguous_seq:
            return False
        lost = _losses(before.data)
        if lost is None or lost != _losses(after.data):
            return False
        if close == after.seq:
            return True
        # A record lost while the pending ones were written counts later.
        witness = next((beat for beat in later if beat.seq > close), None)
        return witness is not None and _losses(witness.data) == lost


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
    importer: Importer | None = None,
    server_stopped: bool = False,
) -> LogRead:
    """Read every epoch under ``directory``.

    ``high_water`` maps an epoch name to the highest ``seq`` a previous import
    took; records at or below it are not returned again, though they still
    count towards the epoch's gaps and last sequence. ``importer`` is where
    this read runs (here, by default): an epoch's silence is judged only when
    the importer shares the epoch's host and boot, on the monotonic clock.
    ``server_stopped`` says the server that wrote the log is no longer
    running, so an epoch without ``goodbye`` is gone.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    marks = high_water or {}
    who = importer or Importer.here()
    read = LogRead(root, [])
    for host_boot, epoch_dir in _epoch_directories(root):
        epoch = _epoch_from_directory(host_boot, epoch_dir, marks)
        if epoch is None:
            read.notes.append(f"{epoch_dir}: not an epoch directory")
            continue
        _read_epoch(epoch, who, server_stopped)
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


_RELIST_ATTEMPTS = 3


class _Vanished(Exception):
    """A listed ``.part`` segment was sealed (renamed) before it was opened."""


def _read_epoch(epoch: EpochRead, importer: Importer, server_stopped: bool) -> None:
    seen: dict[int, RawRecord] = {}
    for attempt in range(_RELIST_ATTEMPTS):
        # The writer may seal a segment between the listing and the open; the
        # sealed file is then read under its new name on the next listing,
        # and a sequence already seen is not taken twice.
        if not _read_segments(epoch, seen, report=attempt == _RELIST_ATTEMPTS - 1):
            break
    epoch.status = _read_json_file(epoch.directory / "status.json", epoch.errors)
    epoch.key = _read_key(epoch.directory / "key", epoch.errors)
    _settle(epoch, seen, importer, server_stopped)


def _read_segments(
    epoch: EpochRead, seen: dict[int, RawRecord], *, report: bool
) -> bool:
    """Read every listed segment into ``seen``; True when one vanished."""
    vanished = False
    for segment in _segments(epoch.directory):
        try:
            records = _read_segment(epoch, segment)
        except _Vanished:
            vanished = True
            if report:
                epoch.errors.append(f"{segment.name}: sealed while being read")
            continue
        for record in records:
            # The first delivery of a sequence number is the one kept.
            seen.setdefault(record.seq, record)
    return vanished


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
    except FileNotFoundError as exc:
        if path.suffix == ".part":
            raise _Vanished(path.name) from exc
        epoch.errors.append(f"{path.name}: {exc}")
        return []
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


def _settle(
    epoch: EpochRead,
    seen: dict[int, RawRecord],
    importer: Importer,
    server_stopped: bool,
) -> None:
    """Fill the epoch's records, sequence facts and liveness from what was read."""
    _sequence_facts(epoch, seen)
    mark = epoch.high_water_before
    epoch.records = [
        record for seq, record in sorted(seen.items()) if mark is None or seq > mark
    ]
    epoch.heartbeats = [
        record for _seq, record in sorted(seen.items()) if record.kind == "heartbeat"
    ]
    for record in seen.values():
        if record.kind == "hello":
            epoch.hello = record.data
        elif record.kind == "goodbye":
            epoch.goodbye = record.data
    epoch.last_seen_wall_ns = _last_seen(epoch, seen, "wall_ns")
    epoch.last_seen_mono_ns = _last_seen(epoch, seen, "mono_ns")
    epoch.state, epoch.state_reason = _state(epoch, importer, server_stopped)


def _sequence_facts(epoch: EpochRead, seen: dict[int, RawRecord]) -> None:
    """Gaps and the sequences read; the writer's own count stays apart.

    The hook numbers records only as it writes them, so a hole below the
    highest sequence read is data not yet visible, not a dropped record:
    the contiguous sequence stops before it.
    """
    if seen:
        lowest, highest = min(seen), max(seen)
        epoch.gaps = highest - lowest + 1 - len(seen)
        epoch.last_seq = highest
        contiguous = lowest if lowest == 0 else None
        while contiguous is not None and contiguous + 1 in seen:
            contiguous += 1
        epoch.contiguous_seq = contiguous
    epoch.writer_last_seq = _integer((epoch.status or {}).get("last_seq"))


def _last_seen(epoch: EpochRead, seen: dict[int, RawRecord], clock: str) -> int | None:
    """The latest time, on the server's wall or monotonic clock, the process
    is known to have been alive."""
    stamps = [_stamp(record, clock) for record in seen.values()]
    stamps.append(_integer((epoch.status or {}).get(clock)))
    known = [stamp for stamp in stamps if stamp is not None]
    return max(known) if known else None


def _stamp(record: RawRecord, clock: str) -> int | None:
    """When a liveness record was written; other kinds say nothing about it."""
    if record.kind == "hello":
        return _integer((record.data.get("clock") or {}).get(clock))
    if record.kind in {"heartbeat", "goodbye"}:
        return _integer(record.data.get(clock))
    return None


def _losses(heartbeat: dict[str, Any]) -> tuple[Any, ...] | None:
    """What a heartbeat says was lost so far; None when it cannot say, or the
    writer is capped and stops writing records."""
    dropped, errors = heartbeat.get("dropped"), _integer(heartbeat.get("errors"))
    if not isinstance(dropped, dict) or errors is None or heartbeat.get("capped"):
        return None
    counts = {str(kind): count for kind, count in dropped.items() if count}
    return tuple(sorted(counts.items())), errors


def _close(heartbeat: RawRecord, later: Sequence[RawRecord]) -> int:
    """The sequence by which the records a heartbeat was written ahead of
    are all written. They take the next sequences but the writer thread's
    own heartbeats, which it may write before them, as when a record was
    reserved before its stamp and is emitted later."""
    close = heartbeat.seq + _pending(heartbeat.data)
    for beat in later:
        if beat.seq > close:
            break
        close += 1
    return close


def _pending(heartbeat: dict[str, Any]) -> int:
    """Records accepted before a heartbeat's stamp and written after it. A
    hook from before the count gives only its queue, which leaves out the
    batch being written: no more than a lower bound."""
    for key in ("pending", "queued"):
        value = _integer(heartbeat.get(key))
        if value is not None:
            return value
    return 0


def _heard(record: RawRecord) -> dict[str, int]:
    return {
        "seq": record.seq,
        "mono_ns": _integer(record.data.get("mono_ns")) or 0,
        "wall_ns": _integer(record.data.get("wall_ns")) or 0,
    }


def _span_start(record: RawRecord) -> dict[str, int]:
    return {
        "start_seq": record.seq,
        "start_mono_ns": _integer(record.data.get("mono_ns")) or 0,
        "start_wall_ns": _integer(record.data.get("wall_ns")) or 0,
    }


def _span_end(record: RawRecord) -> dict[str, int]:
    return {
        "end_seq": record.seq,
        "end_mono_ns": _integer(record.data.get("mono_ns")) or 0,
        "end_wall_ns": _integer(record.data.get("wall_ns")) or 0,
    }


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _hello_text(hello: dict[str, Any] | None, key: str) -> str | None:
    value = (hello or {}).get(key)
    return value if isinstance(value, str) and value else None


def _state(
    epoch: EpochRead, importer: Importer, server_stopped: bool
) -> tuple[str, str]:
    """The epoch's liveness and why: a goodbye ends it; a stopped server
    leaves it gone; otherwise silence counts only on the clock that wrote
    the stamps, which the importer shares on the same host and boot."""
    if epoch.goodbye is not None:
        return STATE_ENDED, "goodbye"
    if server_stopped:
        return STATE_GONE, "server_stopped"
    if not importer.shares_clock_with(epoch):
        return STATE_UNKNOWN, (
            "other_host" if importer.boot_id is not None else "importer_boot_unknown"
        )
    seen = epoch.last_seen_mono_ns
    if seen is None:
        return STATE_GONE, "no_heartbeat"
    if importer.monotonic_ns - seen >= SILENCE_NS:
        return STATE_GONE, "silence"
    return STATE_ALIVE, "heartbeat"


__all__ = [
    "COVERAGE_BASIS",
    "HEARTBEAT_GAP_NS",
    "LIVENESS_BASIS",
    "FORMAT",
    "RECORD_KINDS",
    "SILENCE_NS",
    "STATE_ALIVE",
    "STATE_ENDED",
    "STATE_GONE",
    "STATE_UNKNOWN",
    "EpochRead",
    "Importer",
    "LogRead",
    "RawRecord",
    "read_execution_log",
]
