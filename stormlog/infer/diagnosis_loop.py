"""Stalls of the vLLM engine loop, from the execution hook's raw records.

``engine_loop_gap`` reads one engine epoch's raw hook records
(``stormlog.vllm_hook/1``; see ``docs/vllm_execution.md``) and reports the
stretch in which the engine made no progress while it had work it could run
that is furthest over its own limit. The same rules serve an online trigger, which tails the raw log,
and the offline diagnoser, which feeds imported steps back through the same
adapter, so the two cannot disagree about what a stall is.

Where a stall sits decides what it can be blamed on:

- ``between_steps``: from a step's completion to the next ``schedule()``
  entry. Nothing ran on the host's behalf, so it is the host's.
- ``in_schedule``: inside ``schedule()``, which is host work.
- ``within_step``: a step that took far longer than the steps before it.
  The host or the GPU may have been slow; without a GPU trace it is
  ``host_or_gpu``.

Work is ready when a request ran in the step before the stretch and in the
step after it: it was running, not waiting for capacity or for streaming
input. A stretch covered by a scheduler pause (when the hook records pauses)
or by an interval the caller excludes, such as its own profiler stop, has no
ready work. A stall still going on at the evaluation time counts from the
last completion.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from statistics import median
from types import MappingProxyType
from typing import Any

from .diagnosis_signals import SignalValue
from .diagnosis_thresholds import (
    DEFAULT_THRESHOLDS,
    LOOP_BASELINE_WINDOW_NS,
    LOOP_HEARTBEAT_GRACE_NS,
    LOOP_MATCHED_BIN_MIN_STEPS,
    LOOP_MIN_BUSY_STEPS,
    LOOP_NO_BASELINE_FLOOR_NS,
    LOOP_STALL_FACTOR,
    LOOP_STALL_FLOOR_NS,
    THRESHOLDS_VERSION,
    resolve_threshold,
)

LOCUS_BETWEEN_STEPS = "between_steps"
LOCUS_IN_SCHEDULE = "in_schedule"
LOCUS_WITHIN_STEP = "within_step"
ATTRIBUTION_HOST = "host"
ATTRIBUTION_HOST_OR_GPU = "host_or_gpu"
BASELINE_MATCHED = "matched"
BASELINE_UNMATCHED = "unmatched"
BASELINE_FLOOR = "floor"

REASON_REQUIRES_HOOK = "requires_hook"
REASON_TOO_FEW_STEPS = "too_few_steps"
REASON_RECORDS_DROPPED = "hook_records_dropped"
REASON_CAPPED = "hook_capped"
REASON_WRITER_ERRORS = "hook_writer_errors"
REASON_EPOCH_CHANGED = "epoch_changed"
REASON_COVERAGE_UNKNOWN = "hook_coverage_unknown"
REASON_PAUSE_UNKNOWN = "pause_state_unknown"
OBSERVES_PAUSE = "pause"
# vLLM's pause states: only PAUSED_ALL stops steps; PAUSED_NEW stops admissions
# and keeps running requests stepping.
PAUSED_ALL = "PAUSED_ALL"

Interval = tuple[int, int]


@dataclass(frozen=True)
class LoopGapConfig:
    """How to evaluate one window of hook records.

    ``now_wall_ns`` is the evaluation time, for a stall still going on.
    ``exclude_wall`` lists wall-clock intervals with no ready work by the
    caller's knowledge, such as its own profiler stop. ``thresholds``
    overrides entries of the shared table by key. ``status`` is the epoch's
    ``status.json``, when the caller has it: a capped writer stops writing
    records and heartbeats alike, and only the status says so. ``hello`` is
    the epoch's hello for a window that starts later, as a tail does: it
    says whether the hook records pauses.
    """

    now_wall_ns: int | None = None
    exclude_wall: Sequence[Interval] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    status: Mapping[str, Any] | None = None
    hello: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Refuse overrides that could never decide: a key the table lacks
        (never read), a value that is not finite (never exceeded), or a
        loop threshold that is not positive (no limit at all)."""
        unknown = sorted(set(self.thresholds) - set(DEFAULT_THRESHOLDS))
        if unknown:
            raise ValueError(f"unknown threshold keys: {', '.join(unknown)}")
        if not all(math.isfinite(value) for value in self.thresholds.values()):
            raise ValueError("threshold overrides must be finite numbers")
        bad = sorted(k for k in _LOOP_KEYS if self.thresholds.get(k, 1.0) <= 0)
        if bad:
            raise ValueError(f"loop thresholds must be positive: {', '.join(bad)}")


@dataclass(frozen=True)
class Step:
    """One scheduler step, as the raw log or an import describes it."""

    iteration: str
    start_wall_ns: int
    start_mono_ns: int
    end_wall_ns: int
    end_mono_ns: int
    completed_wall_ns: int | None
    completed_mono_ns: int | None
    members: frozenset[str]
    total_tokens: int
    # Members whose request finished in this step (from its completion).
    finished: frozenset[str] = frozenset()
    # Streaming-input members: one whose next step starts a new turn (its
    # prompt grew) was waiting for its client's input in between.
    streaming: frozenset[str] = frozenset()
    # Each member's prompt length in this step, as (member, tokens).
    prompts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Stall:
    locus: str
    attribution: str
    duration_ns: int
    start_mono_ns: int
    start_wall_ns: int
    end_wall_ns: int
    # The work bucket of the step the stall belongs to (see work_bucket).
    bucket: int = 0
    ongoing: bool = False


def work_bucket(total_tokens: int) -> int:
    """Steps that schedule within a factor of two of each other's tokens
    share a bucket: a step running a long prefill is compared with steps of
    its own size, never with decode-only steps."""
    return max(total_tokens, 0).bit_length()


# ------------------------------------------------------------------ adapter
def steps_from_raw(records: Sequence[Mapping[str, Any]]) -> list[Step]:
    """Join each ``scheduled`` record to its ``completed`` by iteration."""
    done = {
        str(record.get("iteration")): record
        for record in records
        if record.get("kind") == "completed"
    }
    steps = [
        _step(record, done.get(str(record.get("iteration"))))
        for record in records
        if record.get("kind") == "scheduled"
    ]
    return sorted(steps, key=lambda step: step.start_mono_ns)


def _step(scheduled: Mapping[str, Any], completed: Mapping[str, Any] | None) -> Step:
    members = [m for m in scheduled.get("members") or [] if isinstance(m, Mapping)]
    return Step(
        iteration=str(scheduled.get("iteration")),
        start_wall_ns=int(scheduled["start_wall_ns"]),
        start_mono_ns=int(scheduled["start_mono_ns"]),
        end_wall_ns=int(scheduled["end_wall_ns"]),
        end_mono_ns=int(scheduled["end_mono_ns"]),
        completed_wall_ns=_int(completed, "wall_ns"),
        completed_mono_ns=_int(completed, "mono_ns"),
        members=frozenset(str(m.get("internal")) for m in members),
        total_tokens=int(scheduled.get("total_tokens") or 0),
        finished=_finished(completed),
        streaming=frozenset(
            str(m.get("internal")) for m in members if m.get("resumable") is True
        ),
        prompts=tuple(
            (str(m.get("internal")), int(m["prompt_tokens"]))
            for m in members
            if isinstance(m.get("prompt_tokens"), int)
        ),
    )


def ready(step: Step) -> frozenset[str]:
    """The members still ready to run after ``step``, with nothing scheduled
    since: not finished in it, and not a streaming-input request, which may
    be waiting for its client's next input."""
    return step.members - step.finished - step.streaming


def continuing(before: Step, after: Step) -> frozenset[str]:
    """The requests of ``before`` that ``after`` runs on: not finished in
    ``before``, and, for a streaming-input request, not starting a new turn
    in ``after``. Between turns it waited for its client's input, which is
    not the engine's to explain; within one it decodes like any other."""
    went_on = (before.members - before.finished) & after.members
    return went_on - _new_turns(before, after)


def _new_turns(before: Step, after: Step) -> frozenset[str]:
    """Streaming-input members whose prompt grew from ``before`` to
    ``after``, or whose prompt either step does not give."""
    old, new = dict(before.prompts), dict(after.prompts)
    return frozenset(
        member
        for member in before.streaming & after.members
        if member not in old or member not in new or new[member] > old[member]
    )


def _finished(completed: Mapping[str, Any] | None) -> frozenset[str]:
    """Members that finished in the step, by a finish reason or, under async
    scheduling, by being discarded after their end of sequence."""
    members = (completed or {}).get("members") or []
    return frozenset(
        str(m.get("internal"))
        for m in members
        if isinstance(m, Mapping)
        and (
            m.get("finish_reason") is not None
            or m.get("outcome") == "discarded_finished"
        )
    )


def _int(record: Mapping[str, Any] | None, key: str) -> int | None:
    value = record.get(key) if record is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --------------------------------------------------------------- evaluation
def engine_loop_gap(
    records: Sequence[Mapping[str, Any]], config: LoopGapConfig | None = None
) -> SignalValue:
    """The stall with ready work furthest over its limit in one epoch's
    records, in seq order: the longest when none is over."""
    config = config or LoopGapConfig()
    reasons = record_reasons(records)
    if (config.status or {}).get("capped"):
        reasons.append(REASON_CAPPED)
    steps = steps_from_raw(records)
    if not any(step.completed_mono_ns is not None for step in steps):
        reasons.append(REASON_TOO_FEW_STEPS if steps else REASON_REQUIRES_HOOK)
    blocked = [*pause_intervals(records), *config.exclude_wall]
    # An epoch that said goodbye has nothing going on.
    ended = any(record.get("kind") == "goodbye" for record in records)
    now = None if ended else config.now_wall_ns
    stalls = find_stalls(steps, blocked, now, _terminated(records))
    return _verdict(steps, stalls, reasons, records, config)


def record_reasons(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why the records cannot be trusted to show every step: another epoch,
    a sequence gap, records the writer dropped, or a capped writer."""
    reasons: list[str] = []
    if len({record.get("epoch") for record in records} - {None}) > 1:
        reasons.append(REASON_EPOCH_CHANGED)
    # A hello prepended to a later window is no gap in it.
    seqs = [
        record["seq"]
        for record in records
        if isinstance(record.get("seq"), int) and record.get("kind") != "hello"
    ]
    if any(b != a + 1 for a, b in zip(seqs, seqs[1:])):
        reasons.append(REASON_RECORDS_DROPPED)
    reasons.extend(_heartbeat_reasons(records))
    return list(dict.fromkeys(reasons))


@dataclass(frozen=True)
class Coverage:
    """Where the records are known to be whole, on the wall clock: spans
    between two heartbeats (the hello counting as one with nothing lost)
    whose drop counts and errors did not change, and when the writer was
    last heard from. Only inside a span can no record be missing."""

    spans: tuple[Interval, ...]
    last_beat_wall_ns: int | None

    @classmethod
    def of(cls, records: Sequence[Mapping[str, Any]]) -> Coverage:
        beats = _beats(records)
        spans: list[Interval] = []
        for (start, lost), (end, later) in zip(beats, beats[1:]):
            if lost is None or lost != later:
                continue
            if spans and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        last = beats[-1][0] if beats else None
        return cls(tuple(spans), last)

    def covers(self, stall: Stall, now_wall_ns: int | None, grace_ns: float) -> bool:
        """A stall lies inside one span. One still going on needs the writer
        heard from since it began, and within the grace before now."""
        end = stall.end_wall_ns
        if stall.ongoing:
            last = self.last_beat_wall_ns
            if last is None or now_wall_ns is None or now_wall_ns - last > grace_ns:
                return False
            end = last
        return end >= stall.start_wall_ns and any(
            low <= stall.start_wall_ns and end <= high for low, high in self.spans
        )


Beat = tuple[int, tuple[int, int] | None]


def _beats(records: Sequence[Mapping[str, Any]]) -> list[Beat]:
    """The window's heartbeats, and its hello when it bounds the window."""
    found = []
    for index, record in enumerate(records):
        beat = _beat(record) if _zero_point(records, index) else None
        if beat is not None:
            found.append(beat)
    return found


def _zero_point(records: Sequence[Mapping[str, Any]], index: int) -> bool:
    """Whether a record may bound coverage: any heartbeat, and a hello only
    when the window's next record follows it, so nothing between them is
    unread. A hello prepended to a tail window says nothing of the gap."""
    record = records[index]
    if record.get("kind") != "hello":
        return True
    following = records[index + 1] if index + 1 < len(records) else None
    seq = record.get("seq")
    return (
        following is None or not isinstance(seq, int) or following.get("seq") == seq + 1
    )


def _beat(record: Mapping[str, Any]) -> Beat | None:
    """A heartbeat's wall stamp and what it says was lost so far; the hello
    is the zero point, with nothing lost. A capped heartbeat says nothing."""
    if record.get("kind") == "hello":
        wall = (record.get("clock") or {}).get("wall_ns")
        return (wall, (0, 0)) if isinstance(wall, int) else None
    if record.get("kind") != "heartbeat" or not isinstance(record.get("wall_ns"), int):
        return None
    lost = (
        None
        if record.get("capped")
        else (_dropped(record), _count(record.get("errors")))
    )
    return int(record["wall_ns"]), lost


def _heartbeat_reasons(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Drop counts (every kind, oversized records included) or write errors
    that rose between heartbeats: either means a record may be missing that
    no sequence gap shows. Errors also count failed seals and status writes,
    which lose nothing, so this abstains more than it must. A capped writer
    writes no more heartbeats at all; only its ``status.json`` says so."""
    beats = [record for record in records if record.get("kind") == "heartbeat"]
    reasons = []
    if len({_dropped(beat) for beat in beats}) > 1:
        reasons.append(REASON_RECORDS_DROPPED)
    if len({_count(beat.get("errors")) for beat in beats}) > 1:
        reasons.append(REASON_WRITER_ERRORS)
    return reasons


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _dropped(heartbeat: Mapping[str, Any]) -> int:
    dropped = heartbeat.get("dropped") or {}
    return sum(int(v) for v in dropped.values() if isinstance(v, int))


def pause_intervals(
    records: Sequence[Mapping[str, Any]], state: str = PAUSED_ALL
) -> list[Interval]:
    """Wall-clock spans the scheduler spent in ``state``, from the hook's
    ``pause`` transitions; a pause not yet ended runs to the end of time."""
    spans: list[Interval] = []
    started: int | None = None
    for record in records:
        if record.get("kind") != "pause":
            continue
        entering = record.get("to") == state
        if entering and started is None:
            started = int(record["wall_ns"])
        elif not entering and started is not None:
            spans.append((started, int(record["wall_ns"])))
            started = None
    if started is not None:
        spans.append((started, 2**63 - 1))
    return spans


def observes_pauses(records: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the hook that wrote these records records scheduler pauses."""
    hello = next((r for r in records if r.get("kind") == "hello"), None)
    return hello is not None and OBSERVES_PAUSE in (hello.get("observes") or ())


def _observes_pauses(
    records: Sequence[Mapping[str, Any]], config: LoopGapConfig
) -> bool:
    if config.hello is not None:
        return observes_pauses([config.hello])
    return observes_pauses(records)


# ------------------------------------------------------------------- stalls
def find_stalls(
    steps: Sequence[Step],
    blocked: Sequence[Interval],
    now_wall_ns: int | None,
    terminated: frozenset[str] = frozenset(),
) -> list[Stall]:
    """Every candidate stall with ready work, in each locus. ``terminated``
    names requests the engine ended (a terminal record), which no longer
    hold up a stall going on."""
    completions = _Completions(steps)
    stalls: list[Stall] = []
    for before, after in zip(steps, steps[1:]):
        if continuing(before, after):
            stalls.append(_in_schedule(after))
        between = _between(after, completions)
        if between is not None:
            stalls.append(between)
    stalls.extend(_stalls_within(steps))
    ongoing = _ongoing(steps, now_wall_ns, terminated)
    if ongoing is not None:
        stalls.append(ongoing)
    return [piece for stall in stalls for piece in _uncovered(stall, blocked)]


class _Completions:
    """The completed steps that ran something, in completion order, for "the
    latest completion before" lookups in logarithmic time. A memberless step
    (vLLM's zero-token step sending finished IDs) runs nothing, so a gap is
    measured from the step before it."""

    def __init__(self, steps: Sequence[Step]) -> None:
        done = [
            step
            for step in steps
            if step.completed_mono_ns is not None and step.members
        ]
        self.steps = sorted(done, key=lambda step: step.completed_mono_ns or 0)
        self.ends = [step.completed_mono_ns or 0 for step in self.steps]

    def latest_before(self, mono_ns: int) -> Step | None:
        index = bisect_right(self.ends, mono_ns)
        return self.steps[index - 1] if index else None


def _in_schedule(after: Step) -> Stall:
    """``after``'s time inside schedule(), host work on a request that ran
    in the step before it."""
    return Stall(
        LOCUS_IN_SCHEDULE,
        ATTRIBUTION_HOST,
        after.end_mono_ns - after.start_mono_ns,
        after.start_mono_ns,
        after.start_wall_ns,
        after.end_wall_ns,
        work_bucket(after.total_tokens),
    )


def _between(after: Step, completions: _Completions) -> Stall | None:
    """The gap from the latest completion before ``after``'s schedule()
    entry to that entry, when a request the completed step ran, and did not
    finish, runs in ``after``: only then was work ready across it. Under
    async scheduling that completion can be of a step from before an idle
    stretch, since a request's second step is scheduled before its first
    completes."""
    last = completions.latest_before(after.start_mono_ns)
    if last is None or last.completed_mono_ns is None:
        return None
    if not continuing(last, after):
        return None
    return Stall(
        LOCUS_BETWEEN_STEPS,
        ATTRIBUTION_HOST,
        after.start_mono_ns - last.completed_mono_ns,
        last.completed_mono_ns,
        last.completed_wall_ns or after.start_wall_ns,
        after.start_wall_ns,
        work_bucket(after.total_tokens),
    )


def _stalls_within(steps: Sequence[Step]) -> list[Stall]:
    """Each completed step's own time: from its schedule() return, or from
    the previous completion when it was scheduled before that (async
    scheduling), to its completion. Host gaps and schedule() time are left
    to their own loci. A step runs its members, so its work is ready."""
    stalls = []
    completed = [step for step in steps if step.completed_mono_ns is not None]
    previous: Step | None = None
    for step in completed:
        if step.members:
            stalls.append(_within(step, previous))
        previous = step
    return stalls


def _within(step: Step, previous: Step | None) -> Stall:
    start = max(previous.completed_mono_ns or 0 if previous else 0, step.end_mono_ns)
    start_wall = max(
        previous.completed_wall_ns or 0 if previous else 0, step.end_wall_ns
    )
    return Stall(
        LOCUS_WITHIN_STEP,
        ATTRIBUTION_HOST_OR_GPU,
        (step.completed_mono_ns or 0) - start,
        start,
        start_wall,
        step.completed_wall_ns or step.end_wall_ns,
        work_bucket(step.total_tokens),
    )


def _ongoing(
    steps: Sequence[Step], now_wall_ns: int | None, terminated: frozenset[str]
) -> Stall | None:
    """A stall still going on: work continues and nothing completed since.

    A step scheduled and not yet completed may be waiting on the GPU, so it
    is ``host_or_gpu``; with nothing scheduled after the last completion,
    the host has not called schedule() and the stall is the host's."""
    completed = [step for step in steps if step.completed_wall_ns is not None]
    if now_wall_ns is None or not completed:
        return None
    last = completed[-1]
    pending = _pending_after(steps, last)
    if not _work_continues(last, pending, terminated):
        return None
    since = last.completed_wall_ns or 0
    host_only = not pending
    return Stall(
        LOCUS_BETWEEN_STEPS if host_only else LOCUS_WITHIN_STEP,
        ATTRIBUTION_HOST if host_only else ATTRIBUTION_HOST_OR_GPU,
        now_wall_ns - since,
        last.completed_mono_ns or 0,
        since,
        now_wall_ns,
        _ongoing_bucket(last, pending),
        ongoing=True,
    )


def _ongoing_bucket(last: Step, pending: Sequence[Step]) -> int:
    """The size of the step a stall is holding up: the one in flight, if any."""
    return work_bucket((pending[0] if pending else last).total_tokens)


def _pending_after(steps: Sequence[Step], last: Step) -> list[Step]:
    """Steps scheduled since ``last`` began that have not completed."""
    return [
        step
        for step in steps
        if step.completed_wall_ns is None and step.start_wall_ns >= last.start_wall_ns
    ]


def _work_continues(
    last: Step, pending: Sequence[Step], terminated: frozenset[str]
) -> bool:
    """Whether a request of the last completed step is still running: it is
    in a step scheduled since, or it neither finished in that step nor was
    ended by the engine (a cancel, say) since."""
    running = last.members - terminated
    if any(running & step.members for step in pending):
        return True
    return bool(ready(last) - terminated)


def _terminated(records: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(
        str(record.get("internal"))
        for record in records
        if record.get("kind") == "terminal"
    )


def _uncovered(stall: Stall, blocked: Sequence[Interval]) -> list[Stall]:
    """What remains of a stall outside the blocked intervals, each remaining
    stretch a stall of its own: work was ready there, whatever covered the
    rest."""
    pieces = [(stall.start_wall_ns, stall.end_wall_ns)]
    for start, end in sorted(blocked):
        pieces = [
            part
            for low, high in pieces
            for part in ((low, min(high, start)), (max(low, end), high))
            if part[1] > part[0]
        ]
    if pieces == [(stall.start_wall_ns, stall.end_wall_ns)]:
        return [stall]
    shift = stall.start_mono_ns - stall.start_wall_ns
    return [
        replace(
            stall,
            duration_ns=high - low,
            start_mono_ns=low + shift,
            start_wall_ns=low,
            end_wall_ns=high,
        )
        for low, high in pieces
    ]


# ----------------------------------------------------------------- baseline
Cadence = tuple[int, int, int]


def cadence_table(steps: Sequence[Step]) -> list[Cadence]:
    """Each busy step's completion cadence, as (completion, work bucket,
    cadence), in completion order: the time since the previous completion,
    for steps whose work continued from that step."""
    completed = [step for step in steps if step.completed_mono_ns is not None]
    table = []
    for before, after in zip(completed, completed[1:]):
        if continuing(before, after):
            end = after.completed_mono_ns or 0
            cadence = end - (before.completed_mono_ns or 0)
            table.append((end, work_bucket(after.total_tokens), cadence))
    return sorted(table)


def cadence_baseline(
    table: Sequence[Cadence],
    before_mono_ns: int,
    config: LoopGapConfig,
    bucket: int,
    ends: Sequence[int] | None = None,
) -> tuple[float | None, str]:
    """The median completion cadence of the busy steps that completed in the
    window before a stall: those of the stall's own work bucket when there
    are enough, else those of its bucket or larger when there are enough,
    else none (the floor applies). A step is never measured against smaller
    ones, so a long prefill is not judged by the decode cadence. Only
    earlier steps count, so the baseline is causal."""
    window, _ = resolve_threshold(LOOP_BASELINE_WINDOW_NS, config.thresholds)
    matched_min, _ = resolve_threshold(LOOP_MATCHED_BIN_MIN_STEPS, config.thresholds)
    busy_min, _ = resolve_threshold(LOOP_MIN_BUSY_STEPS, config.thresholds)
    if ends is None:
        ends = [end for end, _bucket, _value in table]
    lo = bisect_left(ends, before_mono_ns - int(window))
    hi = bisect_right(ends, before_mono_ns)
    recent = table[lo:hi]
    matched = [value for _end, size, value in recent if size == bucket]
    if len(matched) >= matched_min:
        return median(matched), BASELINE_MATCHED
    larger = [value for _end, size, value in recent if size >= bucket]
    if len(larger) >= busy_min:
        return median(larger), BASELINE_UNMATCHED
    return None, BASELINE_FLOOR


def _limit(
    stall: Stall,
    table: Sequence[Cadence],
    config: LoopGapConfig,
    ends: Sequence[int] | None = None,
) -> tuple[float, str, float | None]:
    """The duration a stall must reach, how it was set, and the baseline."""
    factor, _ = resolve_threshold(LOOP_STALL_FACTOR, config.thresholds)
    floor, _ = resolve_threshold(LOOP_STALL_FLOOR_NS, config.thresholds)
    no_baseline, _ = resolve_threshold(LOOP_NO_BASELINE_FLOOR_NS, config.thresholds)
    baseline, kind = cadence_baseline(
        table, stall.start_mono_ns, config, stall.bucket, ends
    )
    if baseline is None:
        return no_baseline, kind, None
    return max(factor * baseline, floor), kind, baseline


def _verdict(
    steps: Sequence[Step],
    stalls: Sequence[Stall],
    reasons: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    config: LoopGapConfig,
) -> SignalValue:
    overridden = any(key in config.thresholds for key in _LOOP_KEYS)
    worst, covered = _judged(stalls, steps, records, config)
    if worst is not None and worst[0].duration_ns >= worst[1]:
        reasons = [*reasons, *_doubts(covered, _observes_pauses(records, config))]
    detail: dict[str, Any] = {
        "steps": len(steps),
        "pause_capability": _observes_pauses(records, config),
        "reasons": list(reasons),
    }
    if worst is None:
        return _result(None, None, None, reasons, overridden, detail)
    stall, limit, kind, baseline = worst
    detail.update(
        locus=stall.locus,
        attribution=stall.attribution,
        ongoing=stall.ongoing,
        start_wall_ns=stall.start_wall_ns,
        end_wall_ns=stall.end_wall_ns,
        baseline_ns=baseline,
        baseline=kind,
        covered=covered,
    )
    exceeds = stall.duration_ns >= limit
    return _result(
        float(stall.duration_ns), limit, exceeds, reasons, overridden, detail
    )


def _doubts(covered: bool, pauses_observed: bool) -> list[str]:
    """Why a stall over its limit is still no verdict: a record may be
    missing around it, or the hook cannot say the scheduler was not paused."""
    doubts = [] if covered else [REASON_COVERAGE_UNKNOWN]
    return doubts if pauses_observed else [*doubts, REASON_PAUSE_UNKNOWN]


Judged = tuple[Stall, float, str, float | None]


def _judged(
    stalls: Sequence[Stall],
    steps: Sequence[Step],
    records: Sequence[Mapping[str, Any]],
    config: LoopGapConfig,
) -> tuple[Judged | None, bool]:
    """The stall the verdict rests on, and whether the records are whole
    around it: the worst covered stall when it is over its limit, else the
    worst of all, which over its limit can only mean no verdict."""
    coverage = Coverage.of(records)
    grace, _ = resolve_threshold(LOOP_HEARTBEAT_GRACE_NS, config.thresholds)
    covered = [s for s in stalls if coverage.covers(s, config.now_wall_ns, grace)]
    table = cadence_table(steps)
    best = _worst(covered, table, config)
    if best is not None and best[0].duration_ns >= best[1]:
        return best, True
    worst = _worst(stalls, table, config)
    if worst is None:
        return None, True
    return worst, worst[0] in covered


def _worst(
    stalls: Sequence[Stall], table: Sequence[Cadence], config: LoopGapConfig
) -> tuple[Stall, float, str, float | None] | None:
    """The stall furthest over its own limit, or the longest when none is.

    No stall shorter than the lowest floor can reach its limit, so only
    longer ones are ranked: a window of fast steps stays linear in its
    steps, while one where every step is that slow ranks each of them."""
    if not stalls:
        return None
    floor, _ = resolve_threshold(LOOP_STALL_FLOOR_NS, config.thresholds)
    no_baseline, _ = resolve_threshold(LOOP_NO_BASELINE_FLOOR_NS, config.thresholds)
    lowest = min(floor, no_baseline)
    candidates = [stall for stall in stalls if stall.duration_ns >= lowest]
    if not candidates:
        candidates = [max(stalls, key=lambda stall: stall.duration_ns)]
    ends = [end for end, _bucket, _value in table]
    ranked = []
    for stall in candidates:
        limit, kind, baseline = _limit(stall, table, config, ends)
        ranked.append((stall.duration_ns / limit, stall, limit, kind, baseline))
    _ratio, stall, limit, kind, baseline = max(ranked, key=lambda item: item[0])
    return stall, limit, kind, baseline


def _result(
    value: float | None,
    threshold: float | None,
    exceeds: bool | None,
    reasons: Sequence[str],
    overridden: bool,
    detail: dict[str, Any],
) -> SignalValue:
    sufficient = not reasons
    return SignalValue(
        value=value,
        sufficient=sufficient,
        reason=reasons[0] if reasons else None,
        exceeds=exceeds if sufficient else None,
        threshold=threshold,
        thresholds_version=THRESHOLDS_VERSION,
        threshold_overridden=overridden,
        detail=MappingProxyType(detail),
    )


_LOOP_KEYS = (
    LOOP_STALL_FACTOR,
    LOOP_STALL_FLOOR_NS,
    LOOP_NO_BASELINE_FLOOR_NS,
    LOOP_BASELINE_WINDOW_NS,
    LOOP_MIN_BUSY_STEPS,
    LOOP_MATCHED_BIN_MIN_STEPS,
    LOOP_HEARTBEAT_GRACE_NS,
)


__all__ = [
    "ATTRIBUTION_HOST",
    "ATTRIBUTION_HOST_OR_GPU",
    "LOCUS_BETWEEN_STEPS",
    "LOCUS_IN_SCHEDULE",
    "LOCUS_WITHIN_STEP",
    "REASON_CAPPED",
    "REASON_COVERAGE_UNKNOWN",
    "REASON_EPOCH_CHANGED",
    "REASON_PAUSE_UNKNOWN",
    "REASON_RECORDS_DROPPED",
    "REASON_REQUIRES_HOOK",
    "REASON_TOO_FEW_STEPS",
    "REASON_WRITER_ERRORS",
    "Coverage",
    "LoopGapConfig",
    "Stall",
    "Step",
    "cadence_baseline",
    "cadence_table",
    "ready",
    "work_bucket",
    "engine_loop_gap",
    "find_stalls",
    "pause_intervals",
    "record_reasons",
    "steps_from_raw",
]
