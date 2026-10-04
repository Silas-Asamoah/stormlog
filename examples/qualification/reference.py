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
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from stormlog.infer.qualify.recovery import Point, Signals
from stormlog.infer.vllm_metrics import (
    CompactScrape,
    compact_scrape,
    parse_prometheus_text,
)

SEGMENT = re.compile(r"^(\d{6})\.jsonl(\.part)?$")
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
# vLLM 0.30's engine-wide prefix-cache counters, in tokens.
PREFIX_QUERIES = "vllm:prefix_cache_queries"
PREFIX_HITS = "vllm:prefix_cache_hits"


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
    ``.part`` was read, so no record is read twice. A damaged line (a torn
    write) is skipped and counted, as Stormlog's own hook-log reader does,
    and a segment found shorter than what was read is read again from its
    start; both are noted in ``problems``.
    """

    def __init__(
        self,
        root: Path,
        *,
        firstseen: Path | None = None,
        seals: Path | None = None,
        problems: Path | None = None,
    ) -> None:
        self.root = root
        self.firstseen = firstseen
        self.seals = seals
        self.problems = problems
        self.bad_lines = 0
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

    def copy_to(self, destination: Path) -> int:
        """Copy every epoch this tailer read, as it stands now, under
        ``destination`` with the same ``<host>/<epoch>`` layout; returns
        how many files were copied."""
        copied = 0
        for epoch in sorted({directory for directory, _number in self._segments}):
            target = destination / epoch.relative_to(self.root)
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted(epoch.iterdir()):
                if path.is_file():
                    shutil.copy2(path, target / path.name)
                    copied += 1
        return copied

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

    def _read(self, path: Path, segment: _Segment) -> list[dict[str, Any]]:
        try:
            with path.open("rb") as handle:
                if handle.seek(0, 2) < segment.offset:
                    self._note(path, segment.offset, "rewritten_shorter")
                    segment.offset = 0
                handle.seek(segment.offset)
                data = handle.read()
        except FileNotFoundError:
            return []  # sealed between listing and reading; next poll
        end = data.rfind(b"\n") + 1
        records = []
        position = segment.offset
        for line in data[:end].split(b"\n")[:-1]:
            record = _parse(line) if line.strip() else {}
            if record is None:
                self.bad_lines += 1
                self._note(path, position, "bad_line")
            elif record:
                records.append(record)
            position += len(line) + 1
        segment.offset += end
        return records

    def _note(self, path: Path, offset: int, kind: str) -> None:
        note = {"segment": path.name, "offset": offset, "kind": kind}
        _append_lines(self.problems, [{**note, "seen_ns": time.time_ns()}])


def _parse(line: bytes) -> dict[str, Any] | None:
    """A record, or None for a line that isn't one."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


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
    ``preempted``. Every engine step's start is kept for cadence, and each
    engine epoch's producer, from its hello, names the engine in the labels.
    """

    victim_prefix: str
    shared_prefix_tokens: int
    waits: list[Point] = field(default_factory=list)
    cached_fraction: list[Point] = field(default_factory=list)
    preemptions: list[int] = field(default_factory=list)
    step_starts: list[int] = field(default_factory=list)
    # Every request's admission, the victim's or not: (wall ns, external ID).
    admissions: list[tuple[int, str]] = field(default_factory=list)
    # Each step's schedule() call and its completion, by iteration.
    schedules: dict[str, tuple[int, int]] = field(default_factory=dict)
    completions: dict[str, int] = field(default_factory=dict)
    # Each engine epoch's producer, from its hello: (wall ns, producer).
    producers: list[tuple[int, str]] = field(default_factory=list)
    _admitted: dict[str, int] = field(default_factory=dict)

    def landing(self, at_ns: int) -> str:
        """Where ``at_ns`` fell in the step loop: ``in_schedule`` (inside a
        step's schedule() call), ``in_step`` (after it, before the step
        completed: execution or a GPU wait) or ``between_steps``."""
        for iteration, (start, end) in self.schedules.items():
            if start <= at_ns <= end:
                return "in_schedule"
            completed = self.completions.get(iteration)
            if completed is not None and end < at_ns <= completed:
                return "in_step"
        return "between_steps"

    def first_admission(self, external_prefix: str) -> int | None:
        """When the first request whose external ID has this prefix was
        admitted: a neighbor's onset, for T2."""
        return min(
            (
                at
                for at, external in self.admissions
                if external.startswith(external_prefix)
            ),
            default=None,
        )

    def add(self, record: dict[str, Any]) -> None:
        if not str(record.get("epoch", "")).startswith("engine-"):
            return
        kind = record.get("kind")
        if kind == "hello" and record.get("producer"):
            clock = record.get("clock") or {}
            self.producers.append((int(clock.get("wall_ns") or 0), record["producer"]))
        elif kind == "alias":
            self._alias(record)
        elif kind == "scheduled":
            self._scheduled(record)
        elif kind == "completed" and record.get("wall_ns") is not None:
            self.completions[str(record.get("iteration"))] = int(record["wall_ns"])

    def _alias(self, record: dict[str, Any]) -> None:
        external = str(record.get("external") or "")
        self.admissions.append((int(record["wall_ns"]), external))
        if external.startswith(self.victim_prefix):
            self._admitted[str(record["internal"])] = int(record["wall_ns"])

    def _scheduled(self, record: dict[str, Any]) -> None:
        start = int(record["start_wall_ns"])
        self.step_starts.append(start)
        end = record.get("end_wall_ns")
        if end is not None:
            self.schedules[str(record.get("iteration"))] = (start, int(end))
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


def _victim_requests(
    records: Iterable[dict[str, Any]], victim_prefix: str
) -> Iterator[dict[str, Any]]:
    for record in records:
        if record.get("event_type") != "infer.request":
            continue
        if f"chatcmpl-{record.get('x_request_id') or ''}".startswith(victim_prefix):
            yield record


def chunk_gaps(records: Iterable[dict[str, Any]], victim_prefix: str) -> list[Point]:
    """The victim's gaps between streamed chunks, in seconds, at the later
    chunk, rebuilt from its client records' first-chunk latency and
    inter-arrival times."""
    gaps: list[Point] = []
    for record in _victim_requests(records, victim_prefix):
        first = record.get("first_chunk_latency_ms")
        if first is None:
            continue
        at = int(record["started_at_ns"]) + round(first * 1e6)
        for gap_ms in record.get("chunk_interarrival_ms") or ():
            at += round(gap_ms * 1e6)
            gaps.append((at, gap_ms / 1000.0))
    return sorted(gaps)


def request_spans(
    records: Iterable[dict[str, Any]], victim_prefix: str
) -> list[tuple[int, int]]:
    """Each finished victim request's span, from its send to its end."""
    return [
        (int(record["started_at_ns"]), int(record["ended_at_ns"]))
        for record in _victim_requests(records, victim_prefix)
        if record.get("started_at_ns") is not None
        and record.get("ended_at_ns") is not None
    ]


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Spans as sorted, disjoint intervals: when at least one was open."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# ------------------------------------------------------------------ scrapes


@dataclass(frozen=True)
class Scrape:
    at_ns: int
    waiting: float | None
    kv_usage: float | None
    error: str | None = None
    prefix_queries: float | None = None
    prefix_hits: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "at_ns": self.at_ns,
            "waiting": self.waiting,
            "kv_usage": self.kv_usage,
            "prefix_queries": self.prefix_queries,
            "prefix_hits": self.prefix_hits,
            "error": self.error,
        }


def hit_ratios(scrapes: Sequence[Scrape]) -> list[Point]:
    """The engine-wide prefix-cache hit ratio between consecutive scrapes,
    at the later one: hits over queries added in between. An interval with
    no queries says nothing, and one where a counter fell (a restart) is
    skipped."""
    ratios: list[Point] = []
    for before, after in zip(scrapes, scrapes[1:]):
        first, last = _prefix_counters(before), _prefix_counters(after)
        if first is None or last is None:
            continue
        queries, hits = last[0] - first[0], last[1] - first[1]
        if queries > 0 and 0 <= hits <= queries:
            ratios.append((after.at_ns, hits / queries))
    return ratios


def _prefix_counters(scrape: Scrape) -> tuple[float, float] | None:
    if scrape.prefix_queries is None or scrape.prefix_hits is None:
        return None
    return scrape.prefix_queries, scrape.prefix_hits


def scrape_metrics(url: str, *, timeout_seconds: float = 5.0) -> Scrape:
    """One ``/metrics`` scrape: the waiting count summed over engines, the
    highest KV usage and the prefix-cache counters summed over engines, at
    the time the answer arrived."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as answer:
            text = answer.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as error:
        return Scrape(time.time_ns(), None, None, repr(error))
    at_ns = time.time_ns()
    values = compact_scrape(parse_prometheus_text(text))
    waiting = _floats(values.series(WAITING).values())
    usage = _floats(values.series(KV_USAGE).values())
    queries = _counter(values, PREFIX_QUERIES)
    hits = _counter(values, PREFIX_HITS)
    return Scrape(
        at_ns,
        sum(waiting) if waiting else None,
        max(usage) if usage else None,
        prefix_queries=sum(queries) if queries else None,
        prefix_hits=sum(hits) if hits else None,
    )


def _floats(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def _counter(values: CompactScrape, family: str) -> list[float]:
    """A counter's samples, whether its family is exposed with the
    ``_total`` sample name (as prometheus_client does) or as that name."""
    found = _floats(values.series(family).values())
    return found or _floats(values.series(f"{family}_total").values())


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
            problems=probes_dir / "hook-problems.jsonl",
        )
        self.view = VictimView(victim_prefix, shared_prefix_tokens)
        self.metrics_url = metrics_url
        self.scrape_log = reference_dir / "scrapes.jsonl"
        self.scrapes: list[Scrape] = []
        self.victim_artifact = victim_artifact
        self.victim_prefix = victim_prefix
        self._victim_offset = 0
        self._victim_gaps: list[Point] = []
        self._victim_spans: list[tuple[int, int]] = []

    def poll(self, *, scrape: bool = True) -> None:
        """Read new hook records, and take one scrape unless told not to."""
        for record in self.tailer.poll():
            self.view.add(record)
        if scrape:
            taken = scrape_metrics(self.metrics_url)
            self.scrapes.append(taken)
            _append_lines(self.scrape_log, [taken.to_record()])

    def signals(self) -> Signals:
        """The series so far. ``in_flight`` comes from the victim's finished
        requests, so a request still in flight counts only once it ends:
        until then its gaps aren't busy ones, which delays recovery, never
        hastens it."""
        ok = [taken for taken in self.scrapes if taken.error is None]
        self._read_victim()
        return Signals(
            waits=sorted(self.view.waits),
            cached_fraction=sorted(self.view.cached_fraction),
            victim_preemptions=sorted(self.view.preemptions),
            waiting=[(s.at_ns, s.waiting) for s in ok if s.waiting is not None],
            kv_usage=[(s.at_ns, s.kv_usage) for s in ok if s.kv_usage is not None],
            step_starts=sorted(self.view.step_starts),
            chunk_gaps=self._victim_gaps,
            engine_hit_ratio=hit_ratios(ok),
            in_flight=merge_spans(self._victim_spans),
        )

    def _read_victim(self) -> None:
        """The victim's chunk gaps and request spans, reading only what was
        appended since the last call: this runs every quarter second on the
        host under test."""
        path = self.victim_artifact
        if path is None or not path.exists():
            return
        with path.open("rb") as handle:
            handle.seek(self._victim_offset)
            data = handle.read()
        complete = data[: data.rfind(b"\n") + 1]  # a line being written waits
        self._victim_offset += len(complete)
        parsed = [_parse(line) for line in complete.splitlines() if line.strip()]
        records = [record for record in parsed if record is not None]
        fresh = chunk_gaps(records, self.victim_prefix)
        if fresh:
            self._victim_gaps = sorted(self._victim_gaps + fresh)
        self._victim_spans += request_spans(records, self.victim_prefix)


__all__ = [
    "HookTailer",
    "ReferenceChannel",
    "Scrape",
    "VictimView",
    "chunk_gaps",
    "hit_ratios",
    "merge_spans",
    "request_spans",
    "scrape_metrics",
]
