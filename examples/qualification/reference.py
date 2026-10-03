"""The reference channel: what really happened, read beside any capture.

Every qualification run keeps a reference channel next to the configuration
being diagnosed: the execution hook writing into ``truth/reference/hook``, a
tailer that notes when each hook record first became readable, a poller
that notes when segments are sealed, and a scrape of ``/metrics`` every
second. The harness judges realization, effect timing and recovery from it
(``stormlog.infer.qualify.recovery``), never from the diagnosed capture.

This module is the vLLM 0.30 binding's reader. It turns the hook's engine
records, the scrapes and the victim's own client records into
``recovery.Signals`` for the victim: its requests are the ones whose
``X-Request-Id`` carries the victim's run prefix.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from stormlog.infer.qualify.recovery import Point, Signals
from stormlog.infer.vllm_metrics import compact_scrape, parse_prometheus_text

SEGMENT = re.compile(r"^(\d{6})\.jsonl(\.part)?$")
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"


# ------------------------------------------------------------------ the hook


@dataclass
class _Segment:
    offset: int = 0
    sealed: bool = False


class HookTailer:
    """Reads the hook's raw log as it is written, a complete line at a time.

    Each line read is noted with when it was first seen, in
    ``firstseen`` (one JSON line per record: epoch, seq, segment, first
    seen); each segment's seal, when the poller first sees the ``.part``
    renamed, in ``seals``. A sealed segment continues from where its
    ``.part`` was read, so no record is read twice.
    """

    def __init__(
        self,
        root: Path,
        *,
        firstseen: Path | None = None,
        seals: Path | None = None,
    ) -> None:
        self.root = root
        self.firstseen = firstseen
        self.seals = seals
        self._segments: dict[tuple[Path, int], _Segment] = {}

    def poll(self, now_ns: int | None = None) -> list[dict[str, Any]]:
        """Every record that became readable since the last poll."""
        now_ns = time.time_ns() if now_ns is None else now_ns
        records: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []
        seals: list[dict[str, Any]] = []
        for directory in self._epoch_directories():
            for number, path in _segments(directory):
                segment = self._segments.setdefault((directory, number), _Segment())
                fresh = self._read(path, segment)
                records += fresh
                notes += [_firstseen(record, path, now_ns) for record in fresh]
                if path.suffix == ".jsonl" and not segment.sealed:
                    segment.sealed = True
                    seals.append(
                        {
                            "epoch": directory.name,
                            "segment": path.name,
                            "seen_ns": now_ns,
                        }
                    )
        _append_lines(self.firstseen, notes)
        _append_lines(self.seals, seals)
        return records

    def _epoch_directories(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            epoch
            for host in self.root.iterdir()
            if host.is_dir()
            for epoch in host.iterdir()
            if epoch.is_dir()
        )

    @staticmethod
    def _read(path: Path, segment: _Segment) -> list[dict[str, Any]]:
        try:
            with path.open("rb") as handle:
                handle.seek(segment.offset)
                data = handle.read()
        except FileNotFoundError:
            return []  # sealed between listing and reading; next poll
        end = data.rfind(b"\n") + 1
        segment.offset += end
        records = []
        for line in data[:end].splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records


def _segments(directory: Path) -> list[tuple[int, Path]]:
    """Each segment number with its file, the sealed one when both exist."""
    found: dict[int, Path] = {}
    for path in directory.iterdir():
        match = SEGMENT.match(path.name)
        if match is None:
            continue
        number = int(match.group(1))
        if match.group(2) is None or number not in found:
            found[number] = path
    return sorted(found.items())


def _firstseen(record: dict[str, Any], path: Path, now_ns: int) -> dict[str, Any]:
    return {
        "epoch": record.get("epoch"),
        "seq": record.get("seq"),
        "segment": path.name.split(".")[0],
        "first_seen_ns": now_ns,
    }


def _append_lines(path: Path | None, lines: Iterable[dict[str, Any]]) -> None:
    lines = list(lines)
    if path is None or not lines:
        return
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


# ------------------------------------------------------------------ the victim


@dataclass
class VictimView:
    """The victim's series from the hook's engine records.

    An admission is the ``alias`` record that names a victim request; its
    wait runs to the start of the step that first schedules it; its cached
    fraction is the prefix-cache hit at that step over the victim's shared
    prefix, at most 1. A preemption is the victim's ID in a step's
    ``preempted``. Every engine step's start is kept for cadence.
    """

    victim_prefix: str
    shared_prefix_tokens: int
    waits: list[Point] = field(default_factory=list)
    cached_fraction: list[Point] = field(default_factory=list)
    preemptions: list[int] = field(default_factory=list)
    step_starts: list[int] = field(default_factory=list)
    _admitted: dict[str, int] = field(default_factory=dict)

    def add(self, record: dict[str, Any]) -> None:
        if not str(record.get("epoch", "")).startswith("engine-"):
            return
        kind = record.get("kind")
        if kind == "alias":
            self._alias(record)
        elif kind == "scheduled":
            self._scheduled(record)

    def _alias(self, record: dict[str, Any]) -> None:
        external = str(record.get("external") or "")
        if external.startswith(self.victim_prefix):
            self._admitted[str(record["internal"])] = int(record["wall_ns"])

    def _scheduled(self, record: dict[str, Any]) -> None:
        start = int(record["start_wall_ns"])
        self.step_starts.append(start)
        for member in record.get("members") or ():
            self._member(member, start)
        for internal in record.get("preempted") or ():
            if internal in self._admitted:
                self.preemptions.append(start)

    def _member(self, member: dict[str, Any], start: int) -> None:
        internal = str(member["internal"])
        admitted = self._admitted.get(internal)
        if admitted is None or member.get("sighting") != "first":
            return
        self.waits.append((start, (start - admitted) / 1e9))
        cached = member.get("cached_at_admission") or 0
        share = min(1.0, cached / self.shared_prefix_tokens)
        self.cached_fraction.append((start, share))


def chunk_gaps(records: Iterable[dict[str, Any]], victim_prefix: str) -> list[Point]:
    """The victim's gaps between streamed chunks, in seconds, at the later
    chunk, rebuilt from its client records' first-chunk latency and
    inter-arrival times."""
    gaps: list[Point] = []
    for record in records:
        if record.get("event_type") != "infer.request":
            continue
        if not f"chatcmpl-{record.get('x_request_id') or ''}".startswith(victim_prefix):
            continue
        first = record.get("first_chunk_latency_ms")
        if first is None:
            continue
        at = int(record["started_at_ns"]) + round(first * 1e6)
        for gap_ms in record.get("chunk_interarrival_ms") or ():
            at += round(gap_ms * 1e6)
            gaps.append((at, gap_ms / 1000.0))
    return sorted(gaps)


# ------------------------------------------------------------------ scrapes


@dataclass(frozen=True)
class Scrape:
    at_ns: int
    waiting: float | None
    kv_usage: float | None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "at_ns": self.at_ns,
            "waiting": self.waiting,
            "kv_usage": self.kv_usage,
            "error": self.error,
        }


def scrape(url: str, *, timeout_seconds: float = 5.0) -> Scrape:
    """One ``/metrics`` scrape: the waiting count summed over engines and the
    highest KV usage, at the time the answer arrived."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as answer:
            text = answer.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as error:
        return Scrape(time.time_ns(), None, None, repr(error))
    at_ns = time.time_ns()
    values = compact_scrape(parse_prometheus_text(text))
    waiting = _floats(values.series(WAITING).values())
    usage = _floats(values.series(KV_USAGE).values())
    return Scrape(
        at_ns,
        sum(waiting) if waiting else None,
        max(usage) if usage else None,
    )


def _floats(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


# ------------------------------------------------------------------ together


class ReferenceChannel:
    """The hook tailer, the victim's view, its client records and the
    scrapes, polled together into ``Signals``."""

    def __init__(
        self,
        *,
        hook_root: Path,
        metrics_url: str,
        victim_prefix: str,
        shared_prefix_tokens: int,
        reference_dir: Path,
        probes_dir: Path,
        victim_artifact: Path | None = None,
    ) -> None:
        reference_dir.mkdir(parents=True, exist_ok=True)
        probes_dir.mkdir(parents=True, exist_ok=True)
        self.tailer = HookTailer(
            hook_root,
            firstseen=probes_dir / "hook-firstseen.jsonl",
            seals=probes_dir / "seal-observations.jsonl",
        )
        self.view = VictimView(victim_prefix, shared_prefix_tokens)
        self.metrics_url = metrics_url
        self.scrape_log = reference_dir / "scrapes.jsonl"
        self.scrapes: list[Scrape] = []
        self.victim_artifact = victim_artifact
        self.victim_prefix = victim_prefix

    def poll(self) -> None:
        """Read new hook records and take one scrape."""
        for record in self.tailer.poll():
            self.view.add(record)
        taken = scrape(self.metrics_url)
        self.scrapes.append(taken)
        _append_lines(self.scrape_log, [taken.to_record()])

    def signals(self) -> Signals:
        ok = [taken for taken in self.scrapes if taken.error is None]
        return Signals(
            waits=sorted(self.view.waits),
            cached_fraction=sorted(self.view.cached_fraction),
            victim_preemptions=sorted(self.view.preemptions),
            waiting=[(s.at_ns, s.waiting) for s in ok if s.waiting is not None],
            kv_usage=[(s.at_ns, s.kv_usage) for s in ok if s.kv_usage is not None],
            step_starts=sorted(self.view.step_starts),
            chunk_gaps=self._chunk_gaps(),
        )

    def _chunk_gaps(self) -> list[Point]:
        path = self.victim_artifact
        if path is None or not path.exists():
            return []
        data = path.read_bytes()
        complete = data[: data.rfind(b"\n") + 1]  # a line being written waits
        records = [json.loads(line) for line in complete.splitlines() if line.strip()]
        return chunk_gaps(records, self.victim_prefix)


__all__ = [
    "HookTailer",
    "ReferenceChannel",
    "Scrape",
    "VictimView",
    "chunk_gaps",
    "scrape",
]
