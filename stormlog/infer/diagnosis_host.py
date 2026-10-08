"""Host stalls in the engine core: the engine loop stalled with work ready.

The stalls are the online ``engine_loop_gap``'s own, found by the same rules
on the imported steps (``diagnosis_loop``, R11): a stretch with ready work at
least ``stall_factor`` times the matched completion cadence before it and at
least the floor. Two kinds of time are not the host's to blame and are cut
from them, exactly as the online trigger cuts what it is told to exclude:

- a scheduler paused for all requests (the hook's pause stages);
- the engine's own profiler calls (the hook's ``engine.profile_call``
  stages): a capture's start and stop run in the engine loop and hold it
  for their own bracket, and nothing else.

Without the hook's profile record only the client's trace windows say where
a capture was. They only bound the engine's calls, so a stall overlapping
one is neither cut nor counted: that time is *unresolved*, reported, and
judged by nothing.

Where a stall sits decides what it can be blamed on: between steps and
inside ``schedule()`` it is the host's (``host``); a step that took far
longer than its peers may be the GPU's too (``host_or_gpu``) until a trace
splits it. The claim rests on host stalls alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import Line
from .diagnosis_join import Execution
from .diagnosis_loop import (
    ATTRIBUTION_HOST,
    LOCUS_BETWEEN_STEPS,
    LoopGapConfig,
    Stall,
    cadence_baseline,
    cadence_table,
    stalls_over_limit,
)
from .diagnosis_model import (
    CONTRIBUTING,
    RULED_OUT,
    UNTESTABLE,
    UPSTREAM,
    Alternative,
    Finding,
    Observation,
    met,
)
from .diagnosis_selection import Subject
from .diagnosis_stats import Difference, median_difference
from .diagnosis_steps import loop_steps, merge_intervals
from .diagnosis_thresholds import (
    QUEUE_COMPETITOR_FLOOR,
    QUEUE_CONTRIBUTION,
    resolve_threshold,
)
from .diagnosis_vocabulary import CAPTURE_PAUSE, COMPONENT_ENGINE_CORE, HOST_STALL

NOT_OBSERVED = "not_observed"
NO_ENGINE_STEPS = "no_engine_steps"
SEVERAL_ENGINES = "several_engines"
COVERAGE_UNKNOWN = "hook_coverage_unknown"
PROFILE_CALL = "engine.profile_call"
PAUSED_ALL = "PAUSED_ALL"
FORM = "loop_stall"
# An edge needs its cause to hold this share of the finding's host time.
DOMINANT_SHARE = 0.5
# How far a capture's call or window is known on the engine's clock: a call's
# bracketed stamps, or the client's reads on the same host.
CAPTURE_UNCERTAINTY_NS = 1_000_000
END_OF_TIME = 2**63 - 1

Interval = tuple[int, int]


@dataclass(frozen=True)
class CountedStall:
    """A stall over its limit, after the cut, on the engine's monotonic
    clock, with the parts no client-only capture window overlaps."""

    start: int
    end: int
    locus: str
    attribution: str
    overlapped: bool
    limit_ns: float
    baseline_ns: float | None
    resolved: tuple[Interval, ...]

    @property
    def host(self) -> bool:
        return self.attribution == ATTRIBUTION_HOST

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def resolved_ns(self) -> int:
        return sum(end - start for start, end in self.resolved)

    def excess_share(self) -> float:
        """The share of the stall beyond the cadence it would have taken."""
        if self.baseline_ns is None or self.duration <= 0:
            return 1.0
        return max(0.0, 1.0 - self.baseline_ns / self.duration)


@dataclass
class EngineStalls:
    """One engine's stalls: every one over its limit with nothing cut (what
    holds the queue back), and those the host class counts."""

    every: list[Interval] = field(default_factory=list)
    counted: list[CountedStall] = field(default_factory=list)
    captures: list[Interval] = field(default_factory=list)  # mono
    engine_side: bool = False  # captures from the hook's own profile records
    unplaced_windows: bool = False  # client windows the clock could not place

    def host_intervals(self) -> list[Interval]:
        """The resolved host time the host class counts, merged."""
        return merge_intervals(
            [part for stall in self.counted if stall.host for part in stall.resolved]
        )


def engine_stalls(context: Context, producer: str) -> EngineStalls:
    """The engine's stalls, computed once per diagnosis."""
    key = ("engine_stalls", producer)
    if key not in context.cache:
        context.cache[key] = _engine_stalls(context, producer)
    found: EngineStalls = context.cache[key]
    return found


def _engine_stalls(context: Context, producer: str) -> EngineStalls:
    steps = loop_steps(context.view, producer)
    config = LoopGapConfig(thresholds=dict(context.thresholds or {}))
    every = [
        (stall.start_mono_ns, stall.start_mono_ns + stall.duration_ns)
        for stall, _ in stalls_over_limit(steps, (), config)
    ]
    calls = _profile_calls(context, producer)
    engine_side = context.observes(producer, "profile")
    windows = [] if engine_side else _client_windows(context, producer)
    blocked = [*_paused_all(context, producer), *(wall for wall, _ in calls)]
    table = cadence_table(steps)
    counted = [
        _counted(stall, limit, table, config, windows or [])
        for stall, limit in stalls_over_limit(steps, blocked, config)
    ]
    return EngineStalls(
        every=every,
        counted=counted,
        captures=[mono for _, mono in calls] if engine_side else _monos(windows),
        engine_side=engine_side,
        unplaced_windows=windows is None,
    )


def _counted(
    stall: Stall,
    limit: float,
    table: list[tuple[int, int, int]],
    config: LoopGapConfig,
    windows: list[tuple[Interval, int]],
) -> CountedStall:
    baseline, _ = cadence_baseline(table, stall.start_mono_ns, config, stall.bucket)
    shift = stall.start_mono_ns - stall.start_wall_ns
    outside = _outside(
        (stall.start_wall_ns, stall.end_wall_ns), [w for w, _ in windows]
    )
    return CountedStall(
        start=stall.start_mono_ns,
        end=stall.start_mono_ns + stall.duration_ns,
        locus=stall.locus,
        attribution=stall.attribution,
        overlapped=stall.overlapped,
        limit_ns=limit,
        baseline_ns=baseline,
        resolved=tuple((low + shift, high + shift) for low, high in outside),
    )


def _outside(span: Interval, windows: list[Interval]) -> list[Interval]:
    pieces = [span]
    for start, end in sorted(windows):
        pieces = [
            part
            for low, high in pieces
            for part in ((low, min(high, start)), (max(low, end), high))
            if part[1] > part[0]
        ]
    return pieces


def _paused_all(context: Context, producer: str) -> list[Interval]:
    """Wall spans the scheduler was paused for all requests."""
    spans = context.steps(producer).paused_intervals(END_OF_TIME, wall=True)
    return [(start, end) for start, end, state in spans if state == PAUSED_ALL]


def _profile_calls(context: Context, producer: str) -> list[tuple[Interval, Interval]]:
    """Each profiler call's bracket, on the engine's wall and monotonic
    clocks: from its start stamp's first wall read to its end's last."""
    facts = [
        dict(stage.metadata, start_mono_ns=stage.start_ns, end_mono_ns=stage.end_ns)
        for _, stage in context.view.stages
        if stage.name == PROFILE_CALL and stage.stage_ref.producer_id == producer
    ]
    for epoch in context.view.engines.values():
        if epoch.producer == producer:
            facts += [f for f in epoch.unanchored if f.get("name") == PROFILE_CALL]
    return [call for fact in facts if (call := _bracket(fact)) is not None]


def _bracket(fact: dict[str, Any]) -> tuple[Interval, Interval] | None:
    wall = fact.get("start_wall_ns")
    after = fact.get("end_wall_after_ns", fact.get("end_wall_ns"))
    start, end = fact.get("start_mono_ns"), fact.get("end_mono_ns")
    if not (
        isinstance(wall, int)
        and isinstance(after, int)
        and isinstance(start, int)
        and isinstance(end, int)
    ):
        return None
    return (wall, after), (start, end)


def _client_windows(
    context: Context, producer: str
) -> list[tuple[Interval, int]] | None:
    """The client's capture windows, a start's and a stop's, on the engine's
    wall clock with the shift to its monotonic one; None when they exist
    but cannot be placed (another host)."""
    spans = []
    for line in context.view.trace_windows:
        raw = line.raw or {}
        for low, high in (
            (raw.get("requested_at_ns"), raw.get("started_at_ns")),
            (raw.get("stop_requested_at_ns"), raw.get("stopped_at_ns")),
        ):
            if isinstance(low, int) and isinstance(high, int) and high >= low:
                spans.append((low, high))
    if not spans:
        return []
    shift = _wall_to_mono(context, producer)
    if shift is None:
        return None
    return [(span, shift) for span in spans]


def _wall_to_mono(context: Context, producer: str) -> int | None:
    """The offset from the engine's wall clock to its monotonic one, when
    the client shares its host and the engine's clock stayed continuous."""
    clock = context.clock(producer)
    if not clock.same_host or len(clock.segments) != 1:
        return None
    segment = clock.segments[0]
    return segment.mono_start - segment.wall_start


def _monos(windows: list[tuple[Interval, int]] | None) -> list[Interval]:
    return [(low + shift, high + shift) for (low, high), shift in windows or []]


# ------------------------------------------------------------------- class
def assess_engine_core(context: Context, subject: Subject) -> Assessment:
    """The engine-core host-stall class on one subject."""
    found = _subject_stalls(context, subject)
    if isinstance(found, Assessment):
        return found
    producer, stalls, mine, lifetimes = found
    finding = _finding(context, subject, producer, stalls, mine, lifetimes)
    covered = all(context.lossless(producer, (s.start, s.end)) for s in mine)
    finding.status = ASSESSED if covered else PARTIAL
    finding.partial_reasons = [] if covered else [COVERAGE_UNKNOWN]
    return Assessment(
        HOST_STALL, subject.key, finding.status, finding.partial_reasons, [finding]
    )


def _subject_stalls(
    context: Context, subject: Subject
) -> tuple[str, EngineStalls, list[CountedStall], list[Lifetime]] | Assessment:
    """The counted stalls the subject's requests lived through, or why
    there are none to judge."""
    executions = context.subject_executions(subject)
    producers = {execution.producer for execution in executions}
    if len(producers) != 1:
        reason = SEVERAL_ENGINES if producers else NO_ENGINE_STEPS
        return _verdict(subject, UNSUPPORTED, reason)
    (producer,) = producers
    lifetimes = [life for e in executions if (life := _lifetime(context, e))]
    if not lifetimes or not loop_steps(context.view, producer):
        return _verdict(subject, UNSUPPORTED, NO_ENGINE_STEPS)
    stalls = engine_stalls(context, producer)
    mine = _lived_through(stalls.counted, lifetimes)
    if not mine:
        return _verdict(subject, ASSESSED, NOT_OBSERVED)
    return producer, stalls, mine, lifetimes


def _lived_through(
    stalls: list[CountedStall], lifetimes: list[Lifetime]
) -> list[CountedStall]:
    return [s for s in stalls if any(_overlap(s, life) for life in lifetimes)]


def _verdict(subject: Subject, status: str, reason: str) -> Assessment:
    return Assessment(HOST_STALL, subject.key, status, [reason])


@dataclass(frozen=True)
class Lifetime:
    """One execution on the engine's monotonic clock: entering the queue
    (or admission), its first kept token, its last step's completion."""

    start: int
    first: int | None
    end: int


def _lifetime(context: Context, execution: Execution) -> Lifetime | None:
    enqueued = execution.metadata.get("enqueued_mono_ns")
    start = enqueued if isinstance(enqueued, int) else execution.event.start_ns
    ends, first = _completions(context, execution)
    if start is None or not ends:
        return None
    return Lifetime(start, first, max(ends))


def _completions(
    context: Context, execution: Execution
) -> tuple[list[int], int | None]:
    """Its steps' completions, and that of its first step that kept a token."""
    ends, first = [], None
    for _, membership in execution.memberships:
        step = context.view.iterations.get(membership.iteration_ref)
        end = step[1].end_ns if step is not None else None
        if end is None:
            continue
        ends.append(end)
        kept = membership.metadata.get("outcome") == "kept"
        if first is None and kept and (membership.output_tokens or 0) > 0:
            first = end
    return ends, first


def _overlap(stall: CountedStall, life: Lifetime) -> bool:
    return stall.start < life.end and life.start < stall.end


@dataclass(frozen=True)
class _Exposure:
    """Per request, the stall time beyond the cadence it lived through."""

    ttft: list[float]
    e2e: list[float]


def _exposure(stalls: list[CountedStall], lifetimes: list[Lifetime]) -> _Exposure:
    ttft, e2e = [], []
    for life in lifetimes:
        first = life.first if life.first is not None else life.end
        ttft.append(_held(stalls, (life.start, first)))
        e2e.append(_held(stalls, (life.start, life.end)))
    return _Exposure(ttft, e2e)


def _held(stalls: list[CountedStall], span: Interval) -> float:
    return sum(
        stall.excess_share() * _clip(part, span)
        for stall in stalls
        for part in stall.resolved
    )


def _clip(part: Interval, span: Interval) -> int:
    return max(0, min(part[1], span[1]) - max(part[0], span[0]))


@dataclass(frozen=True)
class _Claims:
    """What the host stalls explain, and against which excess."""

    own: _Exposure
    ttft: Difference | None
    e2e: Difference | None
    explains: dict[str, bool]
    only_with_gpu: bool  # they explain it only with host_or_gpu stalls too

    @property
    def excess(self) -> Difference | None:
        return self.ttft if self._by_ttft else self.e2e

    @property
    def _by_ttft(self) -> bool:
        return self.explains["explains_ttft_excess"] or self.e2e is None

    def lower(self) -> float | None:
        return _lower(self.own.ttft if self._by_ttft else self.own.e2e, self.excess)

    def criterion(self) -> str:
        held = [name for name, explained in self.explains.items() if explained]
        return held[0] if held else "explains_ttft_excess"


def _claims(
    context: Context,
    subject: Subject,
    host: list[CountedStall],
    mine: list[CountedStall],
    lifetimes: list[Lifetime],
) -> _Claims:
    ttft, e2e = _excesses(context, subject)
    own = _exposure(host, lifetimes)
    explains = _explained(context, own, ttft, e2e)
    with_gpu = _explained(context, _exposure(mine, lifetimes), ttft, e2e)
    only = not any(explains.values()) and any(with_gpu.values())
    return _Claims(own, ttft, e2e, explains, only)


def _finding(
    context: Context,
    subject: Subject,
    producer: str,
    stalls: EngineStalls,
    mine: list[CountedStall],
    lifetimes: list[Lifetime],
) -> Finding:
    host = [s for s in mine if s.host]
    claims = _claims(context, subject, host, mine, lifetimes)
    excess = claims.excess
    finding = Finding(
        kind=HOST_STALL,
        component=COMPONENT_ENGINE_CORE,
        subject=subject.as_dict(),
        title="The engine loop stalled with work ready",
        message=_message(host, mine),
        gates={"host_attribution": bool(host) and not claims.only_with_gpu},
        alternatives=[
            _scheduler_paused(context, producer, host),
            _capture(context, stalls, host),
        ],
        condition=met(
            direct_evidence=True,
            sufficient_samples=len(lifetimes) >= 3,
            robust_to_clock=True,
        ),
        contribution=met(
            excess_ci_excludes_zero=excess is not None and excess.low > 0,
            **claims.explains,
        ),
        contribution_lower=claims.lower(),
        location={"component": COMPONENT_ENGINE_CORE, "engine_producer": producer},
        window=context.window(subject),
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        explains=claims.criterion(),
        detail=_detail(stalls, mine),
        metrics=_metrics(host, mine),
        observations=_observations(host, mine, claims.own, claims.ttft, claims.e2e),
        experiment={
            "change": "rerun with the host's competing load removed (or the stalled process pinned to idle cores), same seed",
            "prediction": "no engine-loop stall over its limit, and latency falls by the stall time the requests lived through",
        },
    )
    finding.support, finding.display = _support(context, producer, mine)
    return finding


def _explained(
    context: Context, held: _Exposure, ttft: Difference | None, e2e: Difference | None
) -> dict[str, bool]:
    return {
        "explains_ttft_excess": _explains(held.ttft, ttft, context),
        "explains_e2e_excess": _explains(held.e2e, e2e, context),
    }


def _excesses(
    context: Context, subject: Subject
) -> tuple[Difference | None, Difference | None]:
    if subject.basis == "engine":
        arms = [
            [
                float(value)
                for e in context.subject_executions(subject, reference=reference)
                if (value := context.engine_segments(e).get("engine_ttft")) is not None
            ]
            for reference in (False, True)
        ]
        return median_difference(arms[0], arms[1]), None
    return context.total_excess(subject, "ttft"), context.total_excess(subject, "e2e")


def _explains(held: list[float], excess: Difference | None, context: Context) -> bool:
    """The time the requests were held adds up to the queue's own share of
    their excess."""
    share = resolve_threshold(QUEUE_CONTRIBUTION, context.thresholds)[0]
    if excess is None or excess.estimate <= 0 or not held:
        return False
    return sum(held) >= share * excess.estimate * len(held)


def _lower(held: list[float], excess: Difference | None) -> float | None:
    """The median request's held time, never above the excess's lower bound."""
    if excess is None or not held:
        return None
    return min(median(held), max(excess.low, 0.0)) / 1e6


def _scheduler_paused(
    context: Context, producer: str, host: list[CountedStall]
) -> Alternative:
    """A scheduler paused for all requests makes a gap between steps, or
    holds an overlapped step's output; it cannot lengthen a schedule() call
    or a step. Where the hook records pauses, they were cut already."""
    kind = "scheduler_paused"
    pausable = [s for s in host if s.locus == LOCUS_BETWEEN_STEPS or s.overlapped]
    if not pausable:
        reason = "no counted stall is one a pause could make"
        return Alternative(kind, RULED_OUT, reason, True)
    if context.observes(producer, "pause"):
        reason = "the hook records pauses, and paused time was cut from the stalls"
        return Alternative(kind, RULED_OUT, reason, True)
    reason = (
        f"{len(pausable)} stalls a pause could have made, and the hook does not "
        "record pauses"
    )
    return Alternative(kind, UNTESTABLE, reason, True)


def _capture(
    context: Context, stalls: EngineStalls, host: list[CountedStall]
) -> Alternative:
    """A capture's own calls were cut; what is left next to one may be its
    tail. Judged over the resolved host time only: client-only windows mark
    their overlap unresolved, never the finding."""
    kind = f"{CAPTURE_PAUSE}@profiler"
    if stalls.unplaced_windows:
        reason = "profiler windows exist but cannot be placed on the engine's clock"
        return Alternative(kind, UNTESTABLE, reason, True)
    total = sum(s.resolved_ns for s in host)
    if not stalls.captures or not total:
        return Alternative(kind, RULED_OUT, "no profiler call or window", True)
    near = sum(s.resolved_ns for s in host if _adjacent(s, stalls.captures))
    share = near / total
    floor = resolve_threshold(QUEUE_COMPETITOR_FLOOR, context.thresholds)[0]
    status = (
        RULED_OUT
        if share < floor
        else CONTRIBUTING if share < DOMINANT_SHARE else UPSTREAM
    )
    reason = (
        f"{share:.0%} of the {total / 1e6:.1f} ms of host stall time began next "
        "to a profiler call or window"
    )
    return Alternative(kind, status, reason, True)


def _adjacent(stall: CountedStall, captures: list[Interval]) -> bool:
    """The stall began inside a capture's call or window, or within the
    clock's uncertainty and one baseline cadence after it: a residue. One
    that began before it the capture cannot have caused."""
    radius = (stall.baseline_ns or 0.0) + CAPTURE_UNCERTAINTY_NS
    return any(
        start - CAPTURE_UNCERTAINTY_NS <= stall.start <= end + radius
        for start, end in captures
    )


def _detail(stalls: EngineStalls, mine: list[CountedStall]) -> dict[str, Any]:
    loci: dict[str, float] = {}
    for stall in mine:
        loci[stall.locus] = loci.get(stall.locus, 0.0) + stall.resolved_ns / 1e6
    dominant = max(loci, key=lambda locus: loci[locus]) if loci else None
    return {
        "form": FORM,
        "locus": dominant,
        "loci_ms": {k: round(v, 3) for k, v in sorted(loci.items())},
        "attribution": _attribution(mine),
        "unresolved_ms": round(sum(s.duration - s.resolved_ns for s in mine) / 1e6, 3),
        "captures": "engine" if stalls.engine_side else "client",
        "stalls": [_row(s) for s in sorted(mine, key=lambda s: -s.duration)[:20]],
    }


def _attribution(mine: list[CountedStall]) -> str:
    if all(s.host for s in mine):
        return ATTRIBUTION_HOST
    return "mixed" if any(s.host for s in mine) else "host_or_gpu"


def _row(stall: CountedStall) -> dict[str, Any]:
    baseline = stall.baseline_ns
    return {
        "start_mono_ns": stall.start,
        "duration_ms": round(stall.duration / 1e6, 3),
        "locus": stall.locus,
        "attribution": stall.attribution,
        "limit_ms": round(stall.limit_ns / 1e6, 3),
        "baseline_ms": None if baseline is None else round(baseline / 1e6, 3),
    }


def _metrics(host: list[CountedStall], mine: list[CountedStall]) -> dict[str, Any]:
    return {
        "loop_stalls": len(mine),
        "host_stall_ms": round(sum(s.resolved_ns for s in host) / 1e6, 3),
        "host_or_gpu_stall_ms": round(
            sum(s.resolved_ns for s in mine if not s.host) / 1e6, 3
        ),
        "longest_stall_ms": round(max(s.duration for s in mine) / 1e6, 3),
    }


def _message(host: list[CountedStall], mine: list[CountedStall]) -> str:
    total = sum(s.resolved_ns for s in host) / 1e6
    return (
        f"{len(mine)} engine-loop stalls over their limit while the subject's "
        f"requests ran; {total:.1f} ms of them the host's."
    )


def _observations(
    host: list[CountedStall],
    mine: list[CountedStall],
    own: _Exposure,
    ttft: Difference | None,
    e2e: Difference | None,
) -> list[Observation]:
    out = [
        Observation(
            "o1",
            _message(host, mine),
            "host_stall_ms",
            round(sum(s.resolved_ns for s in host) / 1e6, 3),
            n=len(mine),
        ),
        Observation(
            "o2",
            f"The median request lived through {median(own.e2e) / 1e6:.1f} ms of "
            "host stall time beyond the cadence.",
            "held_e2e_p50_ms",
            round(median(own.e2e) / 1e6, 3),
            n=len(own.e2e),
        ),
    ]
    for name, excess in (("ttft", ttft), ("e2e", e2e)):
        if excess is not None:
            out.append(
                Observation(
                    f"o{len(out) + 1}",
                    f"{name.upper()} rose by {excess.estimate / 1e6:.1f} ms "
                    f"(95% CI {excess.low / 1e6:.1f} to {excess.high / 1e6:.1f}).",
                    f"{name}_excess_ms",
                    round(excess.estimate / 1e6, 3),
                    (round(excess.low / 1e6, 3), round(excess.high / 1e6, 3)),
                    excess.n,
                    excess.n_ref,
                )
            )
    return out


def _support(
    context: Context, producer: str, mine: list[CountedStall]
) -> tuple[list[Line], list[Line]]:
    """The steps on either side of each stall."""
    steps = context.steps(producer).steps
    lines = [
        step.line
        for stall in mine
        for step in steps
        if stall.start - 1 <= (step.end_ns or step.start_ns)
        and step.start_ns <= stall.end + 1
    ]
    unique = list({line.number: line for line in lines}.values())
    return unique, unique[:8]


__all__ = ["EngineStalls", "assess_engine_core", "engine_stalls"]
