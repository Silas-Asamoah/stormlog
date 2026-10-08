"""Choose what to explain: declared subjects, or windows that got worse.

Requests belong to the window of their intended arrival (their send, when an
arrival had no intended time), so a window holds what arrived in it however
long it took. Base windows are coalesced forward until each holds enough
requests, up to a span cap. A window is compared with the earlier windows of
its case that were not flagged themselves, pooled to the reference floor; it
is flagged when a one-sided Fisher's exact test finds more of its requests
above the reference's p90 than the reference has, at the threshold table's
alpha, with at least three above. An incident is a run of at least two
consecutive flagged windows, so one bad window alone is not one.

Every decision is causal: a window is judged at its evaluation time, the end
of the window after it, from what the artifact held by then. A request still
in flight then is censored at its elapsed time, and counts as above only if
that already exceeds the threshold. Without dispatch records a request is
known only from its end, so no detection time can be given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Sequence

from .correlation_events import EntityRef
from .diagnosis_join import ClientRequest, RunView
from .diagnosis_thresholds import (
    SELECTION_ALPHA,
    SELECTION_MIN_ABOVE,
    SELECTION_MIN_REQUESTS,
    SELECTION_REFERENCE_MIN,
    SELECTION_SPAN_CAP_S,
    SELECTION_WINDOW_S,
    resolve_threshold,
)

TTFT = "ttft"
E2E = "e2e"
METRICS = (TTFT, E2E)
MEASURED = "measured"

DECLARED_BY_CALLER = "caller"
INSUFFICIENT_REFERENCE = "insufficient_reference"
TOO_FEW_REQUESTS = "too_few_requests"
LEGACY_NO_DISPATCH = "legacy_no_dispatch_records"


@dataclass(frozen=True)
class FisherTest:
    """One metric's comparison of a window with its reference."""

    metric: str
    threshold_ns: int
    window_above: int
    window_below: int
    reference_above: int
    reference_below: int
    censored: int
    p_value: float
    flagged: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "test": "fisher_exact_one_sided",
            "threshold": "reference_p90",
            "threshold_ns": self.threshold_ns,
            "table": [
                [self.window_above, self.window_below],
                [self.reference_above, self.reference_below],
            ],
            "censored": self.censored,
            "p_value": self.p_value,
            "flagged": self.flagged,
        }


@dataclass
class Window:
    """One analysis window of one case."""

    case_id: str
    start_ns: int
    end_ns: int
    requests: list[str]
    evaluated_at_ns: int = 0
    reference: list[str] = field(default_factory=list)
    tests: dict[str, FisherTest] = field(default_factory=dict)
    status: str | None = None  # why the window was not tested

    @property
    def flagged(self) -> bool:
        return any(test.flagged for test in self.tests.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "requests": len(self.requests),
            "evaluated_at_ns": self.evaluated_at_ns,
            "reference_requests": len(self.reference),
            "status": self.status,
            "flagged": self.flagged,
            "tests": [test.as_dict() for test in self.tests.values()],
        }


@dataclass
class Subject:
    """What one diagnosis explains: requests, and the reference they are
    compared with."""

    key: str
    kind: str  # window, requests or case
    case_id: str | None
    start_ns: int | None
    end_ns: int | None
    requests: list[str]
    reference: list[str]
    declared_by: str | None = None
    windows: list[Window] = field(default_factory=list)
    first_detectable_ns: int | None = None
    detection_unavailable: str | None = None
    # When an incident's degradation was first visible, and how finely the
    # selection placed it: its first flagged window's span.
    onset_ns: int | None = None
    resolution_ns: int | None = None
    # For a window without client requests (a server-only artifact): the
    # engine executions admitted in it, and those admitted before it.
    executions: list[EntityRef] = field(default_factory=list)
    reference_executions: list[EntityRef] = field(default_factory=list)

    @property
    def basis(self) -> str:
        """``client`` when the subject is the client's requests, ``engine``
        when it is only the engine's executions."""
        return "engine" if self.executions and not self.requests else "client"

    @property
    def incident(self) -> bool:
        """Declared subjects are incidents by declaration."""
        return self.declared_by is not None or bool(self.windows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "case_id": self.case_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "requests": len(self.requests),
            "reference_requests": len(self.reference),
            "basis": self.basis,
            "executions": len(self.executions),
            "reference_executions": len(self.reference_executions),
            "declared_by": self.declared_by,
            "first_detectable_ns": self.first_detectable_ns,
            "detection_unavailable": self.detection_unavailable,
            "onset_ns": self.onset_ns,
            "resolution_ns": self.resolution_ns,
            "windows": [window.as_dict() for window in self.windows],
        }


@dataclass(frozen=True)
class SelectionOptions:
    thresholds: dict[str, float] | None = None
    windows: tuple[tuple[int, int], ...] = ()
    request_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()


@dataclass
class Selection:
    subjects: list[Subject]
    windows: list[Window]
    tested: bool  # whether automatic selection ran

    def as_dict(self) -> dict[str, Any]:
        return {
            "automatic": self.tested,
            "windows": [window.as_dict() for window in self.windows],
            "subjects": [subject.as_dict() for subject in self.subjects],
        }


def select(view: RunView, options: SelectionOptions | None = None) -> Selection:
    """Declared subjects, else the incidents automatic selection finds."""
    options = options or SelectionOptions()
    windows = _all_windows(view, options)
    declared = _declared(view, options, windows)
    if declared:
        return Selection(declared, windows, tested=False)
    return Selection(_incidents(view, windows), windows, tested=True)


# ------------------------------------------------------------------ windows
def assignment_ns(request: ClientRequest) -> int | None:
    """When a request arrived: its intended time, else its send."""
    intended = request.intended_at_ns
    return intended if intended is not None else request.sent_at_ns


def _measured(view: RunView) -> dict[str, list[ClientRequest]]:
    by_case: dict[str, list[ClientRequest]] = {}
    for request in view.client.values():
        if request.phase != MEASURED or assignment_ns(request) is None:
            continue
        by_case.setdefault(request.case_id or "", []).append(request)
    for requests in by_case.values():
        requests.sort(key=lambda r: (assignment_ns(r) or 0, r.request_id))
    return by_case


def _all_windows(view: RunView, options: SelectionOptions) -> list[Window]:
    overrides = options.thresholds
    base = int(resolve_threshold(SELECTION_WINDOW_S, overrides)[0] * 1e9)
    cap = int(resolve_threshold(SELECTION_SPAN_CAP_S, overrides)[0] * 1e9)
    minimum = int(resolve_threshold(SELECTION_MIN_REQUESTS, overrides)[0])
    windows: list[Window] = []
    for case_id, requests in sorted(_measured(view).items()):
        case_windows = _coalesce(case_id, requests, base, cap, minimum)
        _evaluate_case(view, case_windows, overrides)
        windows.extend(case_windows)
    return windows


def _coalesce(
    case_id: str, requests: list[ClientRequest], base: int, cap: int, minimum: int
) -> list[Window]:
    """Consecutive base windows joined until they hold ``minimum`` requests,
    or span the cap; an empty stretch that reaches the cap is a window too,
    so a silence breaks a run of flagged windows."""
    if not requests:
        return []
    origin = assignment_ns(requests[0]) or 0
    by_base: dict[int, list[str]] = {}
    for request in requests:
        index = ((assignment_ns(request) or 0) - origin) // base
        by_base.setdefault(index, []).append(request.request_id)
    windows: list[Window] = []
    first, members = 0, []
    last = max(by_base)
    for index in range(last + 1):
        members += by_base.get(index, [])
        if len(members) >= minimum or (index + 1 - first) * base >= cap:
            windows.append(_window(case_id, origin, base, first, index, members))
            first, members = index + 1, []
    if members:
        windows.append(_window(case_id, origin, base, first, last, members))
    return windows


def _window(
    case_id: str, origin: int, base: int, first: int, last: int, members: list[str]
) -> Window:
    return Window(case_id, origin + first * base, origin + (last + 1) * base, members)


# --------------------------------------------------------------- the tests
def _evaluate_case(
    view: RunView, windows: list[Window], overrides: dict[str, float] | None
) -> None:
    """Judge each window at the end of the next, against the earlier
    unflagged windows of its case."""
    floor = int(resolve_threshold(SELECTION_REFERENCE_MIN, overrides)[0])
    minimum = int(resolve_threshold(SELECTION_MIN_REQUESTS, overrides)[0])
    censoring = _Censoring(view)
    for index, window in enumerate(windows):
        following = windows[index + 1] if index + 1 < len(windows) else window
        window.evaluated_at_ns = following.end_ns
        window.reference = [
            rid
            for earlier in windows[:index]
            if not earlier.flagged
            for rid in earlier.requests
        ]
        if len(window.requests) < minimum:
            window.status = TOO_FEW_REQUESTS
        elif len(window.reference) < floor:
            window.status = INSUFFICIENT_REFERENCE
        else:
            window.tests = {
                metric: _test(view, window, metric, censoring, overrides)
                for metric in METRICS
            }


# A failed request is beyond any latency threshold.
FAILED = 2**62


@dataclass(frozen=True)
class _Censoring:
    """What the run's records let a judgement at a past instant see: a
    request is known from its send only with dispatch records, and its TTFT
    before its end only with first-content records."""

    view: RunView

    def value(
        self, request: ClientRequest, metric: str, at: int
    ) -> tuple[int, bool] | None:
        """(value, known) at ``at``; (elapsed, False) for a request still
        running then; None when nothing about it could be seen."""
        sent = request.sent_at_ns
        if sent is None or sent > at:
            return None
        ended = request.ended_at_ns
        if ended is not None and ended <= at:
            return self._finished(request, metric, sent)
        # Before its end, only a first-content record says when it began.
        first = request.first_content_recorded_ns
        if metric == TTFT and first is not None and first <= at:
            return first - sent, True
        if not self._censorable(metric):
            return None
        return at - sent, False

    @staticmethod
    def _finished(
        request: ClientRequest, metric: str, sent: int
    ) -> tuple[int, bool] | None:
        """A request that had ended: a failure is beyond any threshold; a
        metric it did not measure (TTFT unstreamed) is unknown."""
        if request.status != "ok":
            return FAILED, True
        end = request.first_content_at_ns if metric == TTFT else request.ended_at_ns
        return None if end is None else (end - sent, True)

    def _censorable(self, metric: str) -> bool:
        if not self.view.has_dispatch_records():
            return False
        return metric == E2E or self.view.has_first_content_records()


def _test(
    view: RunView,
    window: Window,
    metric: str,
    censoring: _Censoring,
    overrides: dict[str, float] | None,
) -> FisherTest:
    at = window.evaluated_at_ns
    known = [
        found[0]
        for found in (
            censoring.value(view.client[r], metric, at) for r in window.reference
        )
        if found is not None and found[1]
    ]
    threshold = nearest_rank(known, 0.9)
    above, below, censored = _count(
        (censoring.value(view.client[r], metric, at) for r in window.requests),
        threshold,
    )
    reference_above = sum(1 for value in known if value > threshold)
    p_value = fisher_one_sided(
        above, below, reference_above, len(known) - reference_above
    )
    alpha = resolve_threshold(SELECTION_ALPHA, overrides)[0]
    minimum_above = resolve_threshold(SELECTION_MIN_ABOVE, overrides)[0]
    return FisherTest(
        metric=metric,
        threshold_ns=threshold,
        window_above=above,
        window_below=below,
        reference_above=reference_above,
        reference_below=len(known) - reference_above,
        censored=censored,
        p_value=p_value,
        flagged=above >= minimum_above and p_value < alpha,
    )


def _count(
    values: Iterable[tuple[int, bool] | None], threshold: int
) -> tuple[int, int, int]:
    """(above, below, censored): a request still running counts as above
    once its elapsed time passes the threshold, and is not counted before."""
    above = below = censored = 0
    for found in values:
        if found is None:
            continue
        value, complete = found
        censored += not complete
        if value > threshold:
            above += 1
        elif complete:
            below += 1
    return above, below, censored


def nearest_rank(values: Sequence[int], p: float) -> int:
    """The nearest-rank p-quantile: the smallest value with at least p of
    the values at or below it."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the table [[a, b], [c, d]] with fixed margins: how
    likely the window would hold at least ``a`` of the requests above the
    threshold if it were no different from its reference."""
    total, above, size = a + b + c + d, a + c, a + b
    if total == 0 or size == 0:
        return 1.0
    denominator = math.comb(total, size)
    tail = sum(
        math.comb(above, x) * math.comb(total - above, size - x)
        for x in range(a, min(above, size) + 1)
    )
    return float(Fraction(tail, denominator))


# -------------------------------------------------------------- incidents
def _incidents(view: RunView, windows: list[Window]) -> list[Subject]:
    """Runs of at least two consecutive flagged windows of one case."""
    subjects: list[Subject] = []
    run: list[Window] = []
    for window in windows:
        if run and (window.case_id != run[-1].case_id or not window.flagged):
            if len(run) >= 2:
                subjects.append(_incident(view, run))
            run = []
        if window.flagged:
            run.append(window)
    if len(run) >= 2:
        subjects.append(_incident(view, run))
    return subjects


def _incident(view: RunView, run: list[Window]) -> Subject:
    detected = run[1].evaluated_at_ns
    unavailable = None if view.has_dispatch_records() else LEGACY_NO_DISPATCH
    first = run[0]
    return Subject(
        key=f"window:{run[0].case_id}:{run[0].start_ns}",
        kind="window",
        case_id=run[0].case_id,
        start_ns=run[0].start_ns,
        end_ns=run[-1].end_ns,
        requests=[rid for window in run for rid in window.requests],
        reference=list(run[0].reference),
        windows=list(run),
        first_detectable_ns=None if unavailable else detected,
        detection_unavailable=unavailable,
        onset_ns=_onset(view, first),
        resolution_ns=first.end_ns - first.start_ns,
    )


def _onset(view: RunView, window: Window) -> int | None:
    """When the first flagged window's degradation was first visible: the
    earliest instant one of its requests over a flagged threshold had been
    running longer than that threshold, its send plus the threshold. A
    window joined forward from calm traffic can begin seconds before the
    burst that flagged it; this places the incident where it was seen."""
    censoring = _Censoring(view)
    late = []
    for test in window.tests.values():
        if not test.flagged:
            continue
        for request_id in window.requests:
            request = view.client[request_id]
            found = censoring.value(request, test.metric, window.evaluated_at_ns)
            sent = request.sent_at_ns
            if found is not None and found[0] > test.threshold_ns and sent is not None:
                late.append(sent + test.threshold_ns)
    return max(window.start_ns, min(late)) if late else None


# --------------------------------------------------------------- declared
def _declared(
    view: RunView, options: SelectionOptions, windows: list[Window]
) -> list[Subject]:
    flagged = {rid for window in windows if window.flagged for rid in window.requests}
    by_case = _measured(view)
    subjects = [
        _engine_side(view, _declared_window(by_case, flagged, start, end))
        for start, end in options.windows
    ]
    if options.request_ids:
        known = [rid for rid in options.request_ids if rid in view.client]
        subjects.append(
            Subject(
                key="requests:" + ",".join(known),
                kind="requests",
                case_id=None,
                start_ns=None,
                end_ns=None,
                requests=known,
                reference=[],
                declared_by=DECLARED_BY_CALLER,
            )
        )
    for case_id in options.case_ids:
        subjects.append(
            Subject(
                key=f"case:{case_id}",
                kind="case",
                case_id=case_id,
                start_ns=None,
                end_ns=None,
                requests=[r.request_id for r in by_case.get(case_id, [])],
                reference=[],
                declared_by=DECLARED_BY_CALLER,
            )
        )
    return subjects


def _engine_side(view: RunView, subject: Subject) -> Subject:
    """A window no client request arrived in, as in a server-only artifact,
    is the engine's executions admitted in it, against those admitted
    before it; the engine's wall clock must be the artifact's."""
    if subject.requests or subject.start_ns is None or subject.end_ns is None:
        return subject
    start, end = subject.start_ns, subject.end_ns
    for ref, execution in view.executions.items():
        admitted = execution.metadata.get("admitted_wall_ns")
        if not isinstance(admitted, int) or not _same_clock(view, execution):
            continue
        if start <= admitted < end:
            subject.executions.append(ref)
        elif admitted < start:
            subject.reference_executions.append(ref)
    return subject


def _same_clock(view: RunView, execution: Any) -> bool:
    domain = execution.event.context.clock_domain
    wall = domain.removesuffix("/monotonic_ns") + "/unix_epoch_ns"
    return view.clock_domain is not None and wall == view.clock_domain


def _declared_window(
    by_case: dict[str, list[ClientRequest]], flagged: set[str], start: int, end: int
) -> Subject:
    """A caller's window: the requests that arrived in it, against every
    earlier unflagged request of their cases."""
    inside: list[str] = []
    reference: list[str] = []
    for requests in by_case.values():
        arrived = [r.request_id for r in requests if start <= _at(r) < end]
        if arrived:
            inside.extend(arrived)
            reference.extend(
                r.request_id
                for r in requests
                if _at(r) < start and r.request_id not in flagged
            )
    return Subject(
        key=f"window:{start}:{end}",
        kind="window",
        case_id=None,
        start_ns=start,
        end_ns=end,
        requests=inside,
        reference=reference,
        declared_by=DECLARED_BY_CALLER,
    )


def _at(request: ClientRequest) -> int:
    return assignment_ns(request) or 0


__all__ = [
    "DECLARED_BY_CALLER",
    "E2E",
    "INSUFFICIENT_REFERENCE",
    "LEGACY_NO_DISPATCH",
    "METRICS",
    "TTFT",
    "FisherTest",
    "Selection",
    "SelectionOptions",
    "Subject",
    "Window",
    "assignment_ns",
    "fisher_one_sided",
    "nearest_rank",
    "select",
]
