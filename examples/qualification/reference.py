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

from .victim import SENDS

SEGMENT = re.compile(r"^(\d{6})\.jsonl(\.part)?$")
# The hook's per-epoch HMAC key behind the import's foreign-ID pseudonyms.
PSEUDONYM_KEY = "key"
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
        how many files were copied. An epoch's pseudonym ``key`` stays
        behind: with it, anyone holding the copy could turn the import's
        pseudonyms back into other clients' request IDs."""
        copied = 0
        for epoch in sorted({directory for directory, _number in self._segments}):
            target = destination / epoch.relative_to(self.root)
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted(epoch.iterdir()):
                if path.is_file() and path.name != PSEUDONYM_KEY:
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
            self._hello(record)
        elif kind == "alias":
            self._alias(record)
        elif kind == "scheduled":
            self._scheduled(record)
        elif kind == "completed" and record.get("wall_ns") is not None:
            self.completions[str(record.get("iteration"))] = int(record["wall_ns"])

    # Each record is read whole before any of it is applied: one that lacks
    # a field raises with the view unchanged, and is skipped as bad.

    def _hello(self, record: dict[str, Any]) -> None:
        clock = record.get("clock") or {}
        if not isinstance(clock, dict):
            raise TypeError(f"a hello's clock is {type(clock).__name__}")
        self.producers.append((int(clock.get("wall_ns") or 0), record["producer"]))

    def _alias(self, record: dict[str, Any]) -> None:
        external = str(record.get("external") or "")
        at = int(record["wall_ns"])
        victim = external.startswith(self.victim_prefix)
        internal = str(record["internal"]) if victim else None
        self.admissions.append((at, external))
        if internal is not None:
            self._admitted[internal] = at

    def _scheduled(self, record: dict[str, Any]) -> None:
        start = int(record["start_wall_ns"])
        end = record.get("end_wall_ns")
        schedule = None if end is None else (start, int(end))
        firsts = [
            self._first_sighting(member) for member in record.get("members") or ()
        ]
        preempted = list(record.get("preempted") or ())
        self.step_starts.append(start)
        if schedule is not None:
            self.schedules[str(record.get("iteration"))] = schedule
        self._first_schedules(start, firsts)
        for internal in preempted:
            if internal in self._admitted:
                self.preemptions.append(start)

    def _first_schedules(
        self, start: int, firsts: list[tuple[str, float | None]]
    ) -> None:
        """Each victim request's wait and cached share at the step that
        first schedules it."""
        for internal, share in firsts:
            admitted = self._admitted.get(internal)
            if share is None or admitted is None:
                continue
            self.waits.append((start, (start - admitted) / 1e9))
            self.cached_fraction.append((start, share))

    def _first_sighting(self, member: dict[str, Any]) -> tuple[str, float | None]:
        """A member's request, and its cached share of the victim's prefix at
        its first sighting (None at a later one)."""
        internal = str(member["internal"])
        if member.get("sighting") != "first":
            return internal, None
        cached = member.get("cached_at_admission") or 0
        return internal, min(1.0, cached / self.shared_prefix_tokens)


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


def finished_requests(
    records: Iterable[dict[str, Any]], victim_prefix: str
) -> set[str]:
    """The external IDs, as the engine knows them, of finished victim
    requests."""
    return {
        f"chatcmpl-{record.get('x_request_id')}"
        for record in _victim_requests(records, victim_prefix)
    }


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
        self.sends = probes_dir / SENDS
        self._victim_offset = 0
        self._sends_offset = 0
        # Each victim request the victim sent, as the engine names it: when.
        self._victim_sent: dict[str, int] = {}
        self._victim_gaps: list[Point] = []
        self._victim_spans: list[tuple[int, int]] = []
        self._victim_finished: set[str] = set()
        self.bad_records = 0

    def poll(self, *, scrape: bool = True) -> None:
        """Read new hook records, and take one scrape unless told not to. A
        record that parses but lacks a field the view needs is skipped,
        counted and noted, like a damaged line: the records after it in the
        same poll are still read."""
        for record in self.tailer.poll():
            try:
                self.view.add(record)
            except (KeyError, TypeError, ValueError) as error:
                self.bad_records += 1
                note = {"kind": "bad_record", "epoch": record.get("epoch"),
                        "record_kind": record.get("kind"), "error": repr(error),
                        "seen_ns": time.time_ns()}  # fmt: skip
                _append_lines(self.tailer.problems, [note])
        if scrape:
            taken = scrape_metrics(self.metrics_url)
            self.scrapes.append(taken)
            _append_lines(self.scrape_log, [taken.to_record()])

    def signals(self) -> Signals:
        """The series so far. ``in_flight`` comes from the victim's finished
        requests, from send to end, and from each victim request sent (by
        the victim's send probe) or admitted (by the engine) that hasn't
        finished yet, from then to now: a request stuck in a stall is in
        flight, even one sent while the engine was hung and admitting
        nothing, so the stall still open at the latest poll is busy time,
        not idle."""
        ok = [taken for taken in self.scrapes if taken.error is None]
        self._read_victim()
        self._read_sends()
        return Signals(
            waits=sorted(self.view.waits),
            cached_fraction=sorted(self.view.cached_fraction),
            victim_preemptions=sorted(self.view.preemptions),
            waiting=[(s.at_ns, s.waiting) for s in ok if s.waiting is not None],
            kv_usage=[(s.at_ns, s.kv_usage) for s in ok if s.kv_usage is not None],
            step_starts=sorted(self.view.step_starts),
            chunk_gaps=self._victim_gaps,
            engine_hit_ratio=hit_ratios(ok),
            in_flight=self._in_flight(),
        )

    def _in_flight(self) -> list[tuple[int, int]]:
        now = time.time_ns()
        sent = [(at, external) for external, at in self._victim_sent.items()]
        open_requests = [
            (at, now)
            for at, external in self.view.admissions + sent
            if external.startswith(self.victim_prefix)
            and external not in self._victim_finished
        ]
        return merge_spans(self._victim_spans + open_requests)

    def _read_sends(self) -> None:
        """The victim's sends, reading only what was appended since."""
        records, self._sends_offset = _read_appended(self.sends, self._sends_offset)
        for record in records:
            external = f"chatcmpl-{record.get('x_request_id')}"
            sent = record.get("sent_ns")
            if isinstance(sent, int) and external.startswith(self.victim_prefix):
                self._victim_sent.setdefault(external, sent)

    def _read_victim(self) -> None:
        """The victim's chunk gaps and request spans, reading only what was
        appended since the last call: this runs every quarter second on the
        host under test."""
        if self.victim_artifact is None:
            return
        records, self._victim_offset = _read_appended(
            self.victim_artifact, self._victim_offset
        )
        fresh = chunk_gaps(records, self.victim_prefix)
        if fresh:
            self._victim_gaps = sorted(self._victim_gaps + fresh)
        self._victim_spans += request_spans(records, self.victim_prefix)
        self._victim_finished |= finished_requests(records, self.victim_prefix)


def _read_appended(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """The records of the whole lines appended to ``path`` since ``offset``,
    and the offset after them: a line still being written waits."""
    if not path.exists():
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    complete = data[: data.rfind(b"\n") + 1]
    parsed = [_parse(line) for line in complete.splitlines() if line.strip()]
    return [record for record in parsed if record is not None], offset + len(complete)


__all__ = [
    "HookTailer",
    "ReferenceChannel",
    "Scrape",
    "VictimView",
    "chunk_gaps",
    "hit_ratios",
    "finished_requests",
    "merge_spans",
    "request_spans",
    "scrape_metrics",
]
