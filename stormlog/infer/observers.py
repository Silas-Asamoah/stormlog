"""Which observers ran during a run, and whether they held up where it counts.

An observer (the system sampler, the vLLM metrics scraper, the span
receiver, profiler traces, the execution hook) has four states:

- ``requested``: the run asked for it;
- ``configured``: the server was set up to feed it, where that can be seen;
- ``active``: it produced evidence during the compared phases;
- ``healthy``: its evidence covers every compared phase well enough.

The compared phases are every case's measured phase, from its start to the
end of its drain. Health is judged in each of them, not at one moment: a
scraper that answered once at the start of the run is not healthy over a
phase it missed. A state that the artifact cannot show is ``None`` with the
reason in ``unjudged``, never assumed; so is a phase too short to judge, such
as one shorter than a sample interval.

The execution hook is requested when the client imports its log, and also
when the ``before`` description shows it enabled in the server's
environment: it observes the server either way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .manifest import BEFORE, manifests
from .report_stats import is_number

if TYPE_CHECKING:
    from .vllm_analysis import JoinedSpans

SAMPLE_RATE_FLOOR = 0.9
SCRAPE_GAP_FACTOR = 2.0
SPAN_JOIN_FLOOR = 0.99
_NOT_DELIVERED = frozenset({"dropped", "unreachable", "delivery_unknown", "rejected"})
_SPAN_PROBLEMS = (
    "decode_failures",
    "protobuf_unavailable",
    "handler_errors",
    "bad_requests",
    "unsupported_media",
)


@dataclass(frozen=True)
class Phase:
    """One compared phase: a case's measured phase, start to drain end."""

    case_id: str
    started_at_ns: int
    ended_at_ns: int

    def holds(self, at_ns: Any) -> bool:
        return is_number(at_ns) and self.started_at_ns <= at_ns <= self.ended_at_ns


def compared_phases(records: Sequence[Mapping[str, Any]]) -> list[Phase]:
    phases = []
    for record in records:
        if record.get("event_type") != "infer.phase_window":
            continue
        if record.get("phase") != "measured":
            continue
        start, end = record.get("started_at_ns"), record.get("drained_at_ns")
        if is_number(start) and is_number(end):
            phases.append(Phase(str(record.get("case_id")), int(start), int(end)))
    return phases


def observer_states(
    records: Sequence[Mapping[str, Any]],
    *,
    spans: JoinedSpans | None = None,
    vllm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every observer's states over the run's compared phases.

    ``vllm`` is the report's vLLM block, whose cases say whether each
    phase's metrics window resolved.
    """
    config = _session_config(records)
    phases = compared_phases(records)
    windows = _section(vllm or {}, "cases")
    judges: dict[str, Callable[[], dict[str, Any]]] = {
        "system_sampler": lambda: _sampler(records, config, phases),
        "vllm_metrics": lambda: _scraper(records, config, phases, windows),
        "vllm_spans": lambda: _spans(records, config, phases, spans),
        "trace": lambda: _traces(records, config, phases),
        "execution": lambda: _execution(records, config, phases),
    }
    return {
        "compared_phases": [phase.case_id for phase in phases],
        "observers": {name: judge() for name, judge in judges.items()},
    }


def _state(
    requested: bool,
    *,
    configured: bool | None = None,
    phases: Mapping[str, Mapping[str, Any]] | None = None,
    settings: Any = None,
    unjudged: Sequence[str] = (),
    run_healthy: bool | None = True,
) -> dict[str, Any]:
    """An observer's record; active and healthy over every compared phase."""
    phases = phases or {}
    if not requested:
        return {
            "requested": False,
            "configured": configured,
            "active": None,
            "healthy": None,
            "settings": settings,
            "phases": {},
            "unjudged": list(unjudged),
        }
    active, healthy = _over_phases(phases, run_healthy)
    return {
        "requested": True,
        "configured": configured,
        "active": active,
        "healthy": healthy,
        "settings": settings,
        "phases": dict(phases),
        "unjudged": list(unjudged),
    }


def _over_phases(
    phases: Mapping[str, Mapping[str, Any]], run_healthy: bool | None
) -> tuple[bool | None, bool | None]:
    """Active and healthy over the phases that could be judged.

    A phase too short to judge has neither; when no phase could be judged,
    neither has the observer.
    """
    judged = _judged(phases)
    if phases and not judged:
        return None, None
    if not judged or not all(item["active"] for item in judged):
        return False, False
    if run_healthy is not True:
        return True, run_healthy
    return True, all(item["healthy"] for item in judged)


def _judged(phases: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [item for item in phases.values() if item["active"] is not None]


def _phase(active: bool, healthy: bool, reasons: list[str]) -> dict[str, Any]:
    return {"active": active, "healthy": active and healthy, "reasons": reasons}


def _unjudged_phase(reason: str) -> dict[str, Any]:
    return {"active": None, "healthy": None, "reasons": [reason]}


# ------------------------------------------------------------- observers


def _sampler(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], phases: list[Phase]
) -> dict[str, Any]:
    name = config.get("system_sampler")
    interval = config.get("sample_interval_seconds")
    settings = {"sampler": name, "interval_seconds": interval}
    if name in (None, "noop", "none"):
        return _state(False, settings=settings)
    times = _times(records, "infer.system_sample", "timestamp_ns")
    judged = {phase.case_id: _sampler_phase(phase, times, interval) for phase in phases}
    if is_number(interval):
        return _state(True, configured=True, phases=judged, settings=settings)
    return _state(
        True,
        configured=True,
        phases=judged,
        settings=settings,
        unjudged=["sample_interval_unrecorded"],
        run_healthy=None,
    )


def _sampler_phase(phase: Phase, times: list[int], interval: Any) -> dict[str, Any]:
    count = sum(1 for at in times if phase.holds(at))
    if not is_number(interval) or interval <= 0:
        return _phase(count > 0, False, ["interval_unrecorded"])
    expected = int((phase.ended_at_ns - phase.started_at_ns) / 1e9 / interval)
    if expected == 0:
        return _unjudged_phase("the phase is shorter than one sample interval")
    enough = count >= SAMPLE_RATE_FLOOR * expected
    reasons = [] if enough else [f"{count} samples of {expected} expected"]
    return _phase(count > 0, enough, reasons)


def _scraper(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    phases: list[Phase],
    windows: Mapping[str, Any],
) -> dict[str, Any]:
    settings = config.get("vllm_metrics")
    if not isinstance(settings, Mapping):
        return _state(False)
    interval = settings.get("interval_seconds")
    scrapes = [r for r in records if r.get("event_type") == "infer.vllm_scrape"]
    judged = {
        phase.case_id: _scraper_phase(
            phase, scrapes, interval, _section(windows, phase.case_id)
        )
        for phase in phases
    }
    return _state(True, configured=True, phases=judged, settings=dict(settings))


def _scraper_phase(
    phase: Phase,
    scrapes: list[Mapping[str, Any]],
    interval: Any,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    ok = [
        s
        for s in scrapes
        if s.get("case_id") == phase.case_id
        and s.get("phase") == "measured"
        and s.get("status") == "ok"
    ]
    reasons = [
        f"no ok {marker} scrape"
        for marker in ("phase_start", "phase_end")
        if not any(s.get("marker") == marker for s in ok)
    ]
    reasons.extend(_gap_reasons(ok, interval))
    reasons.extend(_window_reasons(window))
    return _phase(bool(ok), not reasons, reasons)


def _gap_reasons(ok: list[Mapping[str, Any]], interval: Any) -> list[str]:
    gap = _largest_gap_seconds(ok)
    if is_number(interval) and gap > SCRAPE_GAP_FACTOR * interval:
        return [f"a {gap:.1f} s gap between ok scrapes"]
    return []


def _window_reasons(window: Mapping[str, Any]) -> list[str]:
    """The phase's metrics window, when the vLLM analysis could not resolve it."""
    if window.get("state") != "unresolved":
        return []
    why = ", ".join(str(reason) for reason in window.get("reasons") or [])
    return [f"window unresolved: {why}"]


def _largest_gap_seconds(scrapes: list[Mapping[str, Any]]) -> float:
    times = sorted(
        int(s["observed_at_ns"]) for s in scrapes if is_number(s.get("observed_at_ns"))
    )
    return max((b - a for a, b in zip(times, times[1:])), default=0) / 1e9


def _spans(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    phases: list[Phase],
    spans: JoinedSpans | None,
) -> dict[str, Any]:
    settings = config.get("vllm_spans")
    configured = _otlp_configured(records)
    if not isinstance(settings, Mapping):
        return _state(False, configured=configured)
    problems = _capability_problems(records, "vllm.spans", _SPAN_PROBLEMS)
    joined = spans.by_request if spans is not None else {}
    judged = {
        phase.case_id: _span_phase(phase, records, joined, problems) for phase in phases
    }
    return _state(True, configured=configured, phases=judged, settings=dict(settings))


def _span_phase(
    phase: Phase,
    records: Sequence[Mapping[str, Any]],
    joined: Mapping[str, Any],
    problems: list[str],
) -> dict[str, Any]:
    accepted = [
        r
        for r in records
        if r.get("event_type") == "infer.request"
        and r.get("phase") == "measured"
        and r.get("case_id") == phase.case_id
        and r.get("status") not in _NOT_DELIVERED
    ]
    found = sum(1 for r in accepted if str(r.get("x_request_id")) in joined)
    share = found / len(accepted) if accepted else 0.0
    reasons = list(problems)
    if share < SPAN_JOIN_FLOOR:
        reasons.append(f"spans joined for {found} of {len(accepted)} accepted requests")
    return _phase(found > 0, not reasons, reasons)


def _traces(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], phases: list[Phase]
) -> dict[str, Any]:
    settings = config.get("trace")
    if not isinstance(settings, Mapping):
        unjudged = [] if "trace" in config else ["trace_settings_unrecorded"]
        return _state(False, unjudged=unjudged)
    windows = [r for r in records if r.get("event_type") == "infer.trace_window"]
    imported = _imported_traces(records)
    judged = {
        phase.case_id: _trace_phase(phase, windows, imported)
        for phase in phases
        if settings.get("phase") in (None, "measured")
    }
    return _state(True, configured=None, phases=judged, settings=dict(settings))


def _trace_phase(
    phase: Phase, windows: list[Mapping[str, Any]], imported: set[str]
) -> dict[str, Any]:
    window = next(
        (
            w
            for w in windows
            if w.get("case_id") == phase.case_id and w.get("phase") == "measured"
        ),
        None,
    )
    if window is None or not window.get("started"):
        return _phase(False, False, ["no trace started"])
    reasons = _trace_reasons(window, imported)
    return _phase(True, not reasons, reasons)


def _trace_reasons(window: Mapping[str, Any], imported: set[str]) -> list[str]:
    reasons = []
    if window.get("stop_error") or not is_number(window.get("stopped_at_ns")):
        reasons.append("the trace did not stop cleanly")
    files = [_base(name) for name in window.get("trace_files") or []]
    if not files:
        reasons.append("no trace file was written")
    elif not any(name in imported for name in files):
        reasons.append("its trace was not imported")
    return reasons


def _base(name: Any) -> str:
    return str(name).rsplit("/", 1)[-1]


def _execution(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], phases: list[Phase]
) -> dict[str, Any]:
    directory = config.get("vllm_execution_dir")
    server_hook = _server_hook_dir(records)
    if not directory and not server_hook:
        return _state(False)
    summary = _execution_summary(records)
    configured = summary is not None and "failed" not in summary
    problems = _epoch_problems(summary or {})
    judged = _execution_phases(records, phases, problems)
    # The artifact does not keep the hook's heartbeat times, so a run with
    # no other problem is not shown healthy, only not unhealthy.
    return _state(
        True,
        configured=configured,
        phases=judged,
        settings={"directory": directory, "server_hook_dir": server_hook},
        unjudged=["heartbeat_gaps"],
        run_healthy=False if problems else None,
    )


def _execution_phases(
    records: Sequence[Mapping[str, Any]], phases: list[Phase], problems: list[str]
) -> dict[str, dict[str, Any]]:
    starts = [
        (r.get("metadata") or {}).get("start_wall_ns")
        for r in records
        if r.get("event_type") == "infer.iteration"
    ]
    return {
        phase.case_id: _phase(
            any(phase.holds(at) for at in starts), not problems, list(problems)
        )
        for phase in phases
    }


# ---------------------------------------------------------------- helpers


def _server_hook_dir(records: Sequence[Mapping[str, Any]]) -> Any:
    """The hook's directory, when the before description shows it enabled."""
    before = manifests(records)[BEFORE]
    if not before:
        return None
    server = _section(_section(before[-1], "description"), "server")
    return _section(server, "environ").get(HOOK_DIR_VARIABLE) or None


# The variable that enables Stormlog's execution hook in a vLLM server.
HOOK_DIR_VARIABLE = "STORMLOG_VLLM_HOOK_DIR"


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    return value if isinstance(value, Mapping) else {}


def _session_config(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for record in records:
        config = record.get("config")
        if record.get("event_type") == "infer.session" and isinstance(config, Mapping):
            return config
    return {}


def _times(
    records: Sequence[Mapping[str, Any]], event_type: str, key: str
) -> list[int]:
    return [
        int(r[key])
        for r in records
        if r.get("event_type") == event_type and is_number(r.get(key))
    ]


def _capability(
    records: Sequence[Mapping[str, Any]], component: str
) -> Mapping[str, Any] | None:
    found = None
    for record in records:
        if (
            record.get("event_type") == "infer.capabilities"
            and record.get("component") == component
        ):
            found = record
    return found


def _capability_problems(
    records: Sequence[Mapping[str, Any]], component: str, counters: Sequence[str]
) -> list[str]:
    capability = _capability(records, component)
    if capability is None:
        return [f"no {component} capability record"]
    metadata = capability.get("metadata") or {}
    if "error" in metadata:
        return [f"{component}: {metadata['error']}"]
    return [
        f"{name}: {metadata[name]}"
        for name in counters
        if is_number(metadata.get(name)) and metadata[name] > 0
    ]


def _otlp_configured(records: Sequence[Mapping[str, Any]]) -> bool | None:
    """Whether the server exports spans, by its own configuration."""
    for record in records:
        if (
            record.get("event_type") != "infer.server_probe"
            or record.get("phase") != "before"
        ):
            continue
        body = (
            (record.get("answers") or {}).get("/server_info?config_format=json") or {}
        ).get("body")
        config = (body or {}).get("vllm_config") if isinstance(body, Mapping) else None
        if isinstance(config, Mapping):
            observability = config.get("observability_config") or {}
            return bool(observability.get("otlp_traces_endpoint"))
    return None


def _imported_traces(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """The file names of every trace an import read."""
    names: set[str] = set()
    for record in records:
        if (
            record.get("event_type") == "infer.capabilities"
            and record.get("component") == "trace_collector"
        ):
            summary = (record.get("metadata") or {}).get("summary") or {}
            names.update(_trace_names(summary.get("traces") or []))
    return names


def _trace_names(traces: list[Any]) -> set[str]:
    return {
        _base(trace[key])
        for trace in traces
        if isinstance(trace, Mapping) and not trace.get("skipped")
        for key in ("path", "file")
        if trace.get(key)
    }


def _execution_summary(
    records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    capability = _capability(records, "engine_adapter")
    if capability is None:
        return None
    summary = ((capability.get("metadata") or {}).get("summary") or {}).get("execution")
    return summary if isinstance(summary, Mapping) else None


def _epoch_problems(summary: Mapping[str, Any]) -> list[str]:
    if "failed" in summary:
        return [f"import failed: {summary['failed']}"]
    problems = []
    for epoch, state in sorted((summary.get("epochs") or {}).items()):
        dropped = state.get("dropped") or {}
        if any(is_number(count) and count > 0 for count in dropped.values()):
            problems.append(f"epoch {epoch} dropped records")
        if state.get("errors"):
            problems.append(f"epoch {epoch} recorded errors")
        if state.get("capped"):
            problems.append(f"epoch {epoch} reached its disk cap")
    return problems


def observer_lines(block: Any) -> list[str]:
    """The text report's line for the observers that were asked for."""
    if not isinstance(block, Mapping):
        return []
    parts = []
    for name, state in (block.get("observers") or {}).items():
        if not state.get("requested"):
            continue
        health = {True: "healthy", False: "unhealthy", None: "unjudged"}[
            state.get("healthy")
        ]
        parts.append(f"{name} {health}")
    return ["Observers: " + ", ".join(parts)] if parts else []


__all__ = [
    "SAMPLE_RATE_FLOOR",
    "SCRAPE_GAP_FACTOR",
    "SPAN_JOIN_FLOOR",
    "Phase",
    "compared_phases",
    "observer_lines",
    "observer_states",
]
