"""Classes about the path to the engine: the client, the API server, and
Stormlog's own profiler.

- ``client_admission``: Stormlog's open-loop client held requests back at
  its in-flight limit, or dropped them, so they were sent late.
- ``host_stall`` at ``api_server`` (form ``frontend``): requests took longer
  to reach the engine while the engine kept stepping, so the time went in
  HTTP, the API server or its IPC.
- ``capture_pause``: a profiler stop, which blocks the server while the
  trace is written, overlapped the stalled requests.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from .diagnosis_context import ASSESSED, PARTIAL, UNSUPPORTED, Assessment, Context
from .diagnosis_inputs import Line
from .diagnosis_model import (
    NOT_RULED_OUT,
    RULED_OUT,
    UNTESTABLE,
    Alternative,
    Finding,
    Observation,
    met,
)
from .diagnosis_selection import Subject
from .diagnosis_stats import INSUFFICIENT_SAMPLES, Difference, median_difference
from .diagnosis_steps import Steps
from .diagnosis_thresholds import QUEUE_CONTRIBUTION, resolve_threshold
from .diagnosis_vocabulary import (
    CAPTURE_PAUSE,
    CLIENT_ADMISSION,
    COMPONENT_API_SERVER,
    COMPONENT_CLIENT,
    COMPONENT_PROFILER,
    HOST_STALL,
)

NOT_OBSERVED = "not_observed"
NO_CLIENT_REQUESTS = "no_client_requests"
NO_INTENDED_ARRIVALS = "no_intended_arrivals"
NO_ENGINE_PROGRESS_EVIDENCE = "no_engine_progress_evidence"
CLOCK_ALIGNMENT_REQUIRED = "clock_alignment_required"
NO_TRACE_WINDOWS = "no_trace_windows"
NO_STOP_REQUEST_STAMP = "no_stop_request_stamp"
SEND = "send_to_ingress"


# ---------------------------------------------------------- client admission
def assess_client_admission(context: Context, subject: Subject) -> Assessment:
    """Requests the client sent late because its in-flight limit held them,
    or never sent because it dropped them."""
    if not subject.requests:
        return _verdict(CLIENT_ADMISSION, subject, UNSUPPORTED, NO_CLIENT_REQUESTS)
    lags = _lags(context, subject.requests)
    if not lags:
        return _verdict(CLIENT_ADMISSION, subject, UNSUPPORTED, NO_INTENDED_ARRIVALS)
    held = _matching(context, subject.requests, "held_for_slot", True)
    dropped = _matching(context, subject.requests, "status", "dropped")
    reference = _lags(context, subject.reference)
    excess = median_difference(list(lags.values()), list(reference.values()))
    late = excess is not None and excess.low > 0
    if not (held or dropped or late):
        return _verdict(CLIENT_ADMISSION, subject, ASSESSED, NOT_OBSERVED)
    finding = _admission_finding(context, subject, held, dropped, excess)
    finding.support = _client_lines(context, [*held, *dropped, *lags])
    finding.display = finding.support[:8]
    if excess is None:
        finding.status = PARTIAL
        return Assessment(
            CLIENT_ADMISSION, subject.key, PARTIAL, [INSUFFICIENT_SAMPLES], [finding]
        )
    return Assessment(CLIENT_ADMISSION, subject.key, ASSESSED, [], [finding])


def _admission_finding(
    context: Context,
    subject: Subject,
    held: list[str],
    dropped: list[str],
    excess: Difference | None,
) -> Finding:
    intended = _from_intended_excess(context, subject)
    lag_ms = None if excess is None else round(excess.estimate / 1e6, 3)
    return Finding(
        kind=CLIENT_ADMISSION,
        component=COMPONENT_CLIENT,
        subject=subject.as_dict(),
        title="The client held requests back at its in-flight limit",
        message=f"{len(held)} requests were held for a slot and {len(dropped)} dropped.",
        gates={"arrivals_recorded": True},
        condition=met(
            direct_evidence=bool(held or dropped),
            sufficient_samples=excess is not None,
        ),
        contribution=met(
            excess_ci_excludes_zero=excess is not None and excess.low > 0,
            explains_intended_latency_excess=_explains(excess, intended, context),
        ),
        contribution_lower=None if excess is None else excess.low / 1e6,
        observations=_admission_observations(held, dropped, excess),
        location={"component": COMPONENT_CLIENT},
        window=context.window(subject),
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        metrics={
            "held_for_slot": len(held),
            "dropped": len(dropped),
            "dispatch_lag_excess_ms": lag_ms,
        },
        experiment={
            "change": "rerun with a larger --max-in-flight, same seed",
            "prediction": "no request is held for a slot, and dispatch lag falls to the reference's",
        },
        explains="explains_intended_latency_excess",
    )


def _matching(
    context: Context, request_ids: list[str], field: str, value: Any
) -> list[str]:
    return [r for r in request_ids if _raw(context, r).get(field) == value]


def _verdict(kind: str, subject: Subject, status: str, reason: str) -> Assessment:
    return Assessment(kind, subject.key, status, [reason])


def _lags(context: Context, request_ids: list[str]) -> dict[str, float]:
    """Send minus intended arrival, for requests that had one."""
    lags = {}
    for request_id in request_ids:
        request = context.view.client[request_id]
        sent, intended = request.sent_at_ns, request.intended_at_ns
        if sent is not None and intended is not None:
            lags[request_id] = float(sent - intended)
    return lags


def _from_intended_excess(context: Context, subject: Subject) -> Difference | None:
    """The excess of first content measured from the intended arrival."""
    arms = []
    for request_ids in (subject.requests, subject.reference):
        values = []
        for request_id in request_ids:
            request = context.view.client[request_id]
            first, intended = request.first_content_at_ns, request.intended_at_ns
            if first is not None and intended is not None:
                values.append(float(first - intended))
        arms.append(values)
    return median_difference(arms[0], arms[1])


def _admission_observations(
    held: list[str], dropped: list[str], excess: Difference | None
) -> list[Observation]:
    out = [
        Observation(
            "o1",
            f"{len(held)} requests were held for an in-flight slot.",
            "held_for_slot",
            len(held),
        ),
        Observation(
            "o2",
            f"{len(dropped)} arrivals were dropped and never sent.",
            "dropped",
            len(dropped),
        ),
    ]
    if excess is not None:
        out.append(
            Observation(
                "o3",
                f"Dispatch lag rose by {excess.estimate / 1e6:.1f} ms (95% CI {excess.low / 1e6:.1f} to {excess.high / 1e6:.1f}).",
                "dispatch_lag_excess_ms",
                round(excess.estimate / 1e6, 3),
                (round(excess.low / 1e6, 3), round(excess.high / 1e6, 3)),
                excess.n,
                excess.n_ref,
            )
        )
    return out


# -------------------------------------------------------------- API server
def assess_api_server(context: Context, subject: Subject) -> Assessment:
    """Requests reached the engine late while the engine kept stepping."""
    if not subject.requests:
        return _verdict(HOST_STALL, subject, UNSUPPORTED, NO_CLIENT_REQUESTS)
    producer = context.producer_of(subject.requests)
    if producer is None:
        return Assessment(
            HOST_STALL, subject.key, UNSUPPORTED, [NO_ENGINE_PROGRESS_EVIDENCE]
        )
    if not context.segment_values(subject.requests, SEND):
        return Assessment(
            HOST_STALL, subject.key, UNSUPPORTED, [CLOCK_ALIGNMENT_REQUIRED]
        )
    excess = context.segment_excess(subject, SEND)
    if excess is None:
        return Assessment(HOST_STALL, subject.key, PARTIAL, [INSUFFICIENT_SAMPLES])
    if excess.low <= 0:
        return Assessment(HOST_STALL, subject.key, ASSESSED, [NOT_OBSERVED])
    stalls = _stall_intervals(context, subject)
    steps = context.steps(producer)
    progressed = _engine_progressed(steps, stalls)
    ttft = context.total_excess(subject, "ttft")
    alternatives = [
        _paused(context, producer, steps, stalls),
        _capture(context, stalls),
    ]
    finding = Finding(
        kind=HOST_STALL,
        component=COMPONENT_API_SERVER,
        subject=subject.as_dict(),
        title="Requests took longer to reach the engine while it kept stepping",
        message=f"Median send_to_ingress rose by {excess.estimate / 1e6:.1f} ms against the reference.",
        gates={"engine_progress": progressed, "bounded_placement": True},
        alternatives=alternatives,
        condition=met(
            direct_evidence=True, sufficient_samples=True, robust_to_clock=True
        ),
        contribution=met(
            excess_ci_excludes_zero=excess.excludes_zero,
            explains_ttft_excess=_explains(excess, ttft, context),
            competitors_excluded=all(a.status == RULED_OUT for a in alternatives),
        ),
        contribution_lower=excess.low / 1e6,
        observations=[
            Observation(
                "o1",
                f"send_to_ingress rose by {excess.estimate / 1e6:.1f} ms (95% CI {excess.low / 1e6:.1f} to {excess.high / 1e6:.1f}) against the reference.",
                "send_to_ingress_excess_ms",
                round(excess.estimate / 1e6, 3),
                (round(excess.low / 1e6, 3), round(excess.high / 1e6, 3)),
                excess.n,
                excess.n_ref,
            )
        ],
        location={"component": COMPONENT_API_SERVER, "engine_producer": producer},
        window=context.window(subject),
        detail={"form": "frontend", "attribution": "host"},
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        metrics={"send_to_ingress_excess_ms": round(excess.estimate / 1e6, 3)},
        experiment={
            "change": "rerun with the API server given more CPU, or with fewer API server workers sharing it",
            "prediction": f"median send_to_ingress falls by at least {excess.low / 1e6:.1f} ms",
        },
        explains="explains_ttft_excess",
    )
    finding.support = _client_lines(context, subject.requests)
    finding.display = finding.support[:8]
    return Assessment(HOST_STALL, subject.key, ASSESSED, [], [finding])


def _stall_intervals(context: Context, subject: Subject) -> list[tuple[int, int]]:
    """Each subject request's send to admission, on the client's clock."""
    intervals = []
    for request_id in subject.requests:
        request = context.view.client[request_id]
        for execution in context.view.executions_of(request_id):
            admitted = execution.metadata.get("admitted_wall_ns")
            sent = request.sent_at_ns
            if isinstance(admitted, int) and sent is not None and admitted > sent:
                intervals.append((sent, admitted))
    return intervals


def _engine_progressed(steps: Steps, stalls: list[tuple[int, int]]) -> bool:
    """Whether a step completed inside at least half of the stall
    intervals: the engine was alive while requests failed to reach it."""
    completions = sorted(
        step.completed_wall_ns for step in steps.steps if step.completed_wall_ns
    )
    if not stalls:
        return False
    alive = sum(1 for start, end in stalls if any(start < c < end for c in completions))
    return alive * 2 >= len(stalls)


def _paused(
    context: Context, producer: str, steps: Steps, stalls: list[tuple[int, int]]
) -> Alternative:
    """A scheduler paused for new requests holds them while running ones
    step, which looks the same from the client: a pause overlapping the
    stalls is not ruled out, and only pause records with nothing lost rule
    one out."""
    kind = "scheduler_paused"
    if not stalls:
        return Alternative(kind, UNTESTABLE, "no stall interval", True)
    paused = steps.paused_intervals(max(e for _, e in stalls), wall=True)
    if any(p[0] < e and s < p[1] for p in paused for s, e in stalls):
        return Alternative(kind, NOT_RULED_OUT, "a pause overlaps the stalls", True)
    if _records_pauses(context, producer):
        return Alternative(
            kind, RULED_OUT, "no pause transition, and the hook records them", True
        )
    return Alternative(kind, UNTESTABLE, "the hook does not record pauses", True)


def _records_pauses(context: Context, producer: str) -> bool:
    """Whether the engine's hook records pauses and reported no loss."""
    epoch = context.epoch_of(producer)
    if epoch is None or epoch.observes is None or "pause" not in epoch.observes:
        return False
    return bool((epoch.coverage or {}).get("spans"))


def _capture(context: Context, stalls: list[tuple[int, int]]) -> Alternative:
    """A profiler window that overlaps no stall cannot explain it."""
    windows = _trace_spans(context)
    if not windows:
        return Alternative(
            "capture_pause", RULED_OUT, "no profiler window in the run", True
        )
    overlapping = [
        w for w in windows if any(w[0] < end and start < w[1] for start, end in stalls)
    ]
    if overlapping:
        return Alternative(
            "capture_pause",
            NOT_RULED_OUT,
            f"{len(overlapping)} profiler windows overlap the stalls",
            True,
        )
    return Alternative(
        "capture_pause", RULED_OUT, "no profiler window overlaps the stalls", True
    )


# ----------------------------------------------------------- capture pause
def assess_capture_pause(context: Context, subject: Subject) -> Assessment:
    """A profiler stop overlapping the subject's stalled requests."""
    windows = context.view.trace_windows
    if not windows:
        return _verdict(CAPTURE_PAUSE, subject, ASSESSED, NO_TRACE_WINDOWS)
    if not subject.requests:
        return _verdict(CAPTURE_PAUSE, subject, UNSUPPORTED, NO_CLIENT_REQUESTS)
    stops = [stop for stop in (_stop(line) for line in windows) if stop is not None]
    if not stops:
        return _verdict(CAPTURE_PAUSE, subject, UNSUPPORTED, NO_STOP_REQUEST_STAMP)
    affected = {stop: _across(context, subject, stop) for stop in stops}
    stop, requests = max(affected.items(), key=lambda item: len(item[1]))
    if not requests:
        return _verdict(CAPTURE_PAUSE, subject, ASSESSED, NOT_OBSERVED)
    finding = _capture_finding(context, subject, stop, requests)
    finding.support = [*windows, *_client_lines(context, requests)]
    finding.display = finding.support[:8]
    return Assessment(CAPTURE_PAUSE, subject.key, ASSESSED, [], [finding])


def _across(context: Context, subject: Subject, stop: tuple[int, int]) -> list[str]:
    """Subject requests waiting for their first content across the stop."""
    found = []
    for request_id in subject.requests:
        request = context.view.client[request_id]
        sent, first = request.sent_at_ns, request.first_content_at_ns
        if (
            sent is not None
            and first is not None
            and sent < stop[1]
            and stop[0] < first
        ):
            found.append(request_id)
    return found


def _capture_finding(
    context: Context, subject: Subject, stop: tuple[int, int], requests: list[str]
) -> Finding:
    excess = context.total_excess(subject, "ttft")
    duration_ms = round((stop[1] - stop[0]) / 1e6, 3)
    held = _held_by_stop(context, subject, stop)
    explains = _explains_ns(held, excess, context)
    return Finding(
        kind=CAPTURE_PAUSE,
        component=COMPONENT_PROFILER,
        subject=subject.as_dict(),
        title="A profiler stop blocked the server while requests were in flight",
        message=f"The stop took {duration_ms:.1f} ms and {len(requests)} requests waited across it.",
        gates={"stop_request_stamp": True},
        condition=met(direct_evidence=True, sufficient_samples=len(requests) >= 3),
        contribution=met(
            excess_ci_excludes_zero=excess is not None and excess.low > 0,
            explains_ttft_excess=explains,
        ),
        # The stop can hold a request back by no more than it overlapped it.
        contribution_lower=None if excess is None else min(held, excess.low) / 1e6,
        observations=[
            Observation(
                "o1",
                f"A profiler stop lasted {duration_ms:.1f} ms.",
                "stop_duration_ms",
                duration_ms,
            ),
            Observation(
                "o2",
                f"{len(requests)} subject requests were waiting for their first content across it.",
                "requests_across_stop",
                len(requests),
            ),
            Observation(
                "o3",
                f"The median subject request spent {held / 1e6:.1f} ms of its wait for first content inside the stop.",
                "median_wait_inside_stop_ms",
                round(held / 1e6, 3),
                n=len(subject.requests),
            ),
        ],
        location={"component": COMPONENT_PROFILER},
        window={
            "start_ns": stop[0],
            "end_ns": stop[1],
            "clock_domain": context.view.clock_domain,
            "uncertainty_ns": 0,
            "resolution_ns": 1_000_000,
            "placement": "client_clock",
        },
        first_detectable_ns=subject.first_detectable_ns,
        incident=subject.incident,
        metrics={
            "stop_duration_ms": duration_ms,
            "requests_across_stop": len(requests),
        },
        experiment={
            "change": "rerun with torch_profiler_dump_cuda_time_total off and the trace stack off, or without the profiler window",
            "prediction": "the stall around the stop shrinks or disappears",
        },
        explains="explains_ttft_excess",
    )


def _held_by_stop(context: Context, subject: Subject, stop: tuple[int, int]) -> float:
    """The median, over the subject's requests, of how much of each one's
    wait for first content the stop overlapped: the most of the median TTFT
    excess the stop can explain."""
    overlaps = []
    for request_id in subject.requests:
        request = context.view.client[request_id]
        sent, first = request.sent_at_ns, request.first_content_at_ns
        if sent is not None and first is not None:
            overlaps.append(max(0, min(first, stop[1]) - max(sent, stop[0])))
    return float(median(overlaps)) if overlaps else 0.0


def _stop(line: Line) -> tuple[int, int] | None:
    raw = line.raw or {}
    requested, stopped = raw.get("stop_requested_at_ns"), raw.get("stopped_at_ns")
    if isinstance(requested, int) and isinstance(stopped, int) and stopped >= requested:
        return requested, stopped
    return None


def _trace_spans(context: Context) -> list[tuple[int, int]]:
    """Each profiler window from its request to its stop, on the client's
    clock; a window still open runs to the end of time."""
    spans = []
    for line in context.view.trace_windows:
        raw = line.raw or {}
        start = raw.get("requested_at_ns") or raw.get("started_at_ns")
        end = raw.get("stopped_at_ns")
        if isinstance(start, int):
            spans.append((start, end if isinstance(end, int) else 2**63 - 1))
    return spans


# ----------------------------------------------------------------- helpers
def _explains(
    excess: Difference | None, total: Difference | None, context: Context
) -> bool:
    return excess is not None and _explains_ns(excess.estimate, total, context)


def _explains_ns(part: float, total: Difference | None, context: Context) -> bool:
    """Whether ``part`` is at least the threshold share of the excess."""
    share = resolve_threshold(QUEUE_CONTRIBUTION, context.thresholds)[0]
    return total is not None and total.estimate > 0 and part / total.estimate >= share


def _raw(context: Context, request_id: str) -> dict[str, Any]:
    return context.view.client[request_id].terminal_raw


def _client_lines(context: Context, request_ids: Any) -> list[Line]:
    lines: list[Line] = []
    for request_id in dict.fromkeys(request_ids):
        lines.extend(context.view.client[request_id].lines())
    return lines


__all__ = ["assess_api_server", "assess_capture_pause", "assess_client_admission"]
