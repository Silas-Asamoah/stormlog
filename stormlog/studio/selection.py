"""Filters, selections, rate scopes and pages of an index snapshot (§5.3, §3.3).

Every selection route parses its query with :func:`parse_filter` and picks
rows with :func:`select`, so ``requests``, ``summary``, ``distribution``,
``points`` and ``export`` agree on one selection for one query.

Filtering and sorting read :meth:`IndexSnapshot.column` arrays; no row view
is built for a selection, only for the rows a page returns (Fable's D-1,
F-7). The one exception is a case's cohort: ``populations_from_records``
reads row mappings, so :func:`case_facts` runs it once per snapshot, over
the measured rows, and keeps the result while the snapshot lives.

A selection has a rate only when its rows are a defined cohort of one case
(§3.3, with Fable's D-2 and D-3):

- **case**: the measured requests of one case, with no condition but a
  window that covers the case's rate interval. Whole case is defined by
  the interval the window clamps to, not by the rows it keeps (D-2).
- **segment**: the same, narrowed by an ``arrival`` window that clamps to a
  part of the rate interval; the interval is that part.

In both, the selected rows must be exactly the requests the interval
counts: every selected row arrives inside the clamped interval, so the
numerator and the interval cover the same rows (D-3). Otherwise the
selection has no rate and says why.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import weakref
from array import array
from bisect import bisect_left
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import quote

from stormlog.infer.events import REQUEST_STATUSES
from stormlog.infer.latency_report import CLIENT_METRICS
from stormlog.infer.populations import (
    CasePopulation,
    MeasuredInterval,
    populations_from_records,
)
from stormlog.infer.report_stats import is_number
from stormlog.infer.request_columns import SUMMARY_FIELDS
from stormlog.infer.tokens import TOKEN_SOURCES
from stormlog.studio.contracts import (
    ConflictError,
    IndexSnapshot,
    LatencyRange,
    NotFoundError,
    NotResident,
    Page,
    Phase,
    RateScope,
    RequestFilter,
    Selection,
    SortKey,
    TimeWindow,
    UsageError,
    columnar,
    selection_fingerprint,
)

__all__ = [
    "FILTER_PARAMETERS",
    "SORT_KEYS",
    "TEXT_FIELDS",
    "WARM_COLUMNS",
    "CaseFacts",
    "arrival_ns",
    "canonical_params",
    "case_facts",
    "in_selection",
    "page",
    "parse_filter",
    "request_row",
    "select",
    "warm",
    "without_latency",
]

PHASES: tuple[Phase, ...] = ("measured", "warmup", "all")
MEMBERSHIPS = ("arrival", "overlap")
# The parameters parse_filter reads; every selection route also takes
# artifact=, and its own (sort, cursor, metric, ...) through route_params.
FILTER_PARAMETERS = (
    "phase",
    "case",
    "status",
    "stream",
    "token_source",
    "q",
    "window",
    "window_ms",
    "window_membership",
    "completed_by_ns",
    "completed_by_ms",
    "latency",
)
ALWAYS_ALLOWED = ("artifact",)
REPEATABLE = frozenset({"case", "status", "token_source"})
TEXT_LIMIT = 256
# What q searches, in this order; error_message only while it is resident.
TEXT_FIELDS = ("request_id", "x_request_id", "case_id", "error_type", "error_message")
SORT_KEYS: dict[str, str] = {
    "artifact": "",
    "started_at": "started_at_ns",
    "client.e2e": "client.e2e",
    "client.ttft": "client.ttft",
    "first_chunk": "first_chunk_latency_ms",
    "output_tokens": "output_tokens",
    "status": "status",
    "case": "case_id",
    "request_id": "request_id",
}
_TEXT_SORTS = frozenset({"status", "case", "request_id"})
_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL_MS = re.compile(r"[+-]?[0-9]+(\.[0-9]{1,6})?")
_ANY_DECIMAL_MS = re.compile(r"[+-]?[0-9]+(\.[0-9]+)?")
_FLOAT = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
_NS_PER_MS = Decimal(1_000_000)
_CURSOR_VERSION = 1
_UNKNOWN = "unknown"


# --- Parsing ------------------------------------------------------------------


def parse_filter(
    query: Mapping[str, Sequence[str]],
    *,
    origin_ns: int | None,
    route_params: Collection[str] = (),
) -> RequestFilter:
    """The canonical filter of a query; UsageError for any bad parameter.

    ``query`` is ``parse_qs`` output, blanks kept. Every name must be a
    filter parameter, ``artifact`` or one of the route's own
    ``route_params``; anything else is ``unknown_parameter``, so a typo
    never widens a selection. A filter parameter given twice is
    ``bad_parameter`` unless it is repeatable (``case``, ``status``,
    ``token_source``). ``window_ms`` and ``completed_by_ms`` are ms from
    ``origin_ns``, converted exactly; without an origin they are refused.
    """
    allowed = set(FILTER_PARAMETERS) | set(ALWAYS_ALLOWED) | set(route_params)
    for name in query:
        if name not in allowed:
            raise UsageError(
                "unknown_parameter", f"{name}: not a parameter here", field=name
            )
    one = _single_values(query)
    return RequestFilter(
        phase=_phase(one.get("phase")),
        cases=_unique_sorted(query.get("case", ())),
        statuses=_choices(query.get("status", ()), "status", REQUEST_STATUSES),
        stream=_stream(one.get("stream")),
        token_sources=_choices(
            query.get("token_source", ()), "token_source", TOKEN_SOURCES
        ),
        text=_text(one.get("q")),
        window=_window(one, origin_ns),
        completed_by_ns=_completed_by(one, origin_ns),
        latency=_latency(one.get("latency")),
    )


def _single_values(query: Mapping[str, Sequence[str]]) -> dict[str, str]:
    one: dict[str, str] = {}
    for name in FILTER_PARAMETERS:
        values = list(query.get(name, ()))
        if not values or name in REPEATABLE:
            continue
        if len(values) > 1:
            raise UsageError(
                "bad_parameter", f"{name}: given more than once", field=name
            )
        one[name] = values[0]
    return one


def _phase(value: str | None) -> Phase:
    if value is None:
        return "measured"
    for phase in PHASES:
        if value == phase:
            return phase
    raise _bad("phase", f"unknown value {value!r}", allowed=list(PHASES))


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _choices(
    values: Iterable[str], name: str, allowed: Sequence[str]
) -> tuple[str, ...]:
    chosen = _unique_sorted(values)
    for value in chosen:
        if value not in allowed:
            raise _bad(name, f"unknown value {value!r}", allowed=list(allowed))
    return chosen


def _stream(value: str | None) -> bool | None:
    if value is None:
        return None
    if value in ("true", "false"):
        return value == "true"
    raise _bad("stream", f"unknown value {value!r}", allowed=["true", "false"])


def _text(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if len(value) > TEXT_LIMIT:
        raise UsageError(
            "limit_exceeded",
            f"q: at most {TEXT_LIMIT} characters",
            field="q",
            details={"limit": TEXT_LIMIT},
        )
    return value.casefold()


def _window(one: Mapping[str, str], origin_ns: int | None) -> TimeWindow | None:
    membership = one.get("window_membership")
    if membership is not None and membership not in MEMBERSHIPS:
        raise _bad(
            "window_membership",
            f"unknown value {membership!r}",
            allowed=list(MEMBERSHIPS),
        )
    if "window" in one and "window_ms" in one:
        raise _bad("window_ms", "give window or window_ms, not both")
    if "window" in one:
        start, end = _pair(one["window"], "window", _ns)
    elif "window_ms" in one:
        start, end = _pair(
            one["window_ms"], "window_ms", lambda t, f: _from_ms(t, f, origin_ns)
        )
    else:
        if membership is not None:
            raise _bad("window_membership", "needs window or window_ms")
        return None
    if end <= start:
        raise _bad("window", "the end must be after the start")
    return TimeWindow(start, end, "overlap" if membership == "overlap" else "arrival")


def _completed_by(one: Mapping[str, str], origin_ns: int | None) -> int | None:
    if "completed_by_ns" in one and "completed_by_ms" in one:
        raise _bad("completed_by_ms", "give completed_by_ns or _ms, not both")
    if "completed_by_ns" in one:
        return _ns(one["completed_by_ns"], "completed_by_ns")
    if "completed_by_ms" in one:
        return _from_ms(one["completed_by_ms"], "completed_by_ms", origin_ns)
    return None


def _latency(value: str | None) -> LatencyRange | None:
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        raise _bad("latency", "expected METRIC:MIN:MAX")
    metric, low, high = parts
    if metric not in CLIENT_METRICS:
        raise _bad(
            "latency", f"unknown metric {metric!r}", allowed=list(CLIENT_METRICS)
        )
    min_ms = _finite(low, "latency")
    max_ms = None if high == "" else _finite(high, "latency")
    if max_ms is not None and max_ms <= min_ms:
        raise _bad("latency", "MAX must be above MIN")
    return LatencyRange(metric, min_ms, max_ms)


def _pair(text: str, field: str, parse: Callable[[str, str], int]) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise _bad(field, "expected START,END")
    return parse(parts[0], field), parse(parts[1], field)


def _ns(text: str, field: str) -> int:
    if not _INTEGER.fullmatch(text):
        raise _bad(field, f"not an integer of nanoseconds: {text!r}")
    return int(text)


def _from_ms(text: str, field: str, origin_ns: int | None) -> int:
    """ms from ``origin_ns`` to ns, exactly: decimal times 10**6."""
    if origin_ns is None:
        raise _bad(field, "the artifact has no origin_ns yet; give nanoseconds")
    if not _DECIMAL_MS.fullmatch(text):
        reason = (
            "at most 6 decimals"
            if _ANY_DECIMAL_MS.fullmatch(text)
            else f"not a decimal number of ms: {text!r}"
        )
        raise _bad(field, reason)
    return origin_ns + int(Decimal(text) * _NS_PER_MS)


def _finite(text: str, field: str) -> float:
    if not _FLOAT.fullmatch(text):
        raise _bad(field, f"not a finite number: {text!r}")
    value = float(text)
    if not is_number(value):
        raise _bad(field, f"not a finite number: {text!r}")
    return value


def _bad(field: str, reason: str, **details: Any) -> UsageError:
    return UsageError(
        "bad_parameter", f"{field}: {reason}", field=field, details=details
    )


# --- Canonical form -------------------------------------------------------------


def canonical_params(f: RequestFilter) -> str:
    """The filter as a query string, in one form only.

    Names sorted, repeated values sorted, ``phase`` always present, the
    window in absolute ns (``window_membership`` only when ``overlap``),
    floats by ``repr``. ``parse_filter`` of it gives ``f`` back.
    """
    pairs: list[tuple[str, str]] = [("case", case) for case in f.cases]
    if f.completed_by_ns is not None:
        pairs.append(("completed_by_ns", str(f.completed_by_ns)))
    if f.latency is not None:
        pairs.append(("latency", _latency_text(f.latency)))
    pairs.append(("phase", f.phase))
    if f.text is not None:
        pairs.append(("q", f.text))
    pairs.extend(("status", status) for status in f.statuses)
    if f.stream is not None:
        pairs.append(("stream", "true" if f.stream else "false"))
    pairs.extend(("token_source", source) for source in f.token_sources)
    pairs.extend(_window_pairs(f.window))
    return "&".join(f"{name}={quote(value, safe=',:')}" for name, value in pairs)


def _latency_text(latency: LatencyRange) -> str:
    high = "" if latency.max_ms is None else repr(latency.max_ms)
    return f"{latency.metric}:{latency.min_ms!r}:{high}"


def _window_pairs(window: TimeWindow | None) -> list[tuple[str, str]]:
    if window is None:
        return []
    pairs = [("window", f"{window.start_ns},{window.end_ns}")]
    if window.membership == "overlap":
        pairs.append(("window_membership", "overlap"))
    return pairs


def without_latency(f: RequestFilter) -> RequestFilter:
    """The same filter with no latency condition: a distribution's denominator."""
    return replace(f, latency=None)


# --- Case facts (the cohorts, once per snapshot) -------------------------------


@dataclass(frozen=True)
class CaseFacts:
    """What one snapshot's measured cases are, computed once.

    ``populations`` is ``populations_from_records`` over the measured rows,
    the kept measured phase windows and the kept workload, as the analyzer
    runs it. ``missing_kept`` names the record types the analyzer reads
    that the artifact holds but the index did not keep (a record over the
    kept-record limit, Fable's D-4): then cohorts and rates are unavailable
    instead of silently computed without them.
    """

    measured: Sequence[int]
    by_case: Mapping[str, Sequence[int]]
    populations: Mapping[str, CasePopulation]
    planned: tuple[str, ...]
    missing_kept: tuple[str, ...]


# Record types populations_from_records reads, for the D-4 check.
COHORT_RECORDS = ("infer.workload", "infer.phase_window")
# The request fields populations_from_records reads (a test pins the set).
POPULATION_FIELDS = (
    "case_id",
    "request_id",
    "x_request_id",
    "request_index",
    "session_id",
    "status",
    "started_at_ns",
    "ended_at_ns",
    "intended_at_ns",
)


@dataclass(frozen=True)
class _Held:
    """One cached ``CaseFacts``: the table it was computed over (weakly,
    so the id is not reused while held), and the kept records and
    inventory it read, which a ``replace`` shares and a new snapshot does
    not."""

    table: weakref.ref[Any]
    kept: Mapping[str, Any]
    inventory: Mapping[str, Any]
    facts: CaseFacts

    def serves(self, snapshot: IndexSnapshot) -> bool:
        return (
            self.table() is snapshot.table
            and self.kept is snapshot.kept
            and self.inventory is snapshot.inventory
        )


_FACTS_LOCK = threading.Lock()
_FACTS: dict[int, _Held] = {}
_FACTS_KEPT = 8


def case_facts(snapshot: IndexSnapshot) -> CaseFacts:
    """The snapshot's case facts, computed on first use and kept with its
    table.

    The cost is linear in the measured rows (about 2 s at 500,000 on the
    cloud box, most of it in ``populations_from_records``), so the caller
    that publishes a snapshot warms it on the builder pool with
    :func:`warm` (§2.1); otherwise the first selection pays it. The key is
    ``id(snapshot.table)`` (Fable's F-7): a snapshot republished with
    ``dataclasses.replace`` for a touch or a ``growing`` flip shares its
    table, kept records and inventory, so it finds the facts; new rows or
    a new index make a new table. A table that cannot be referenced
    weakly is not cached.
    """
    table = snapshot.table
    key = id(table)
    with _FACTS_LOCK:
        held = _FACTS.get(key)
        if held is not None and held.serves(snapshot):
            return held.facts
    facts = _compute_facts(snapshot)
    try:
        reference = weakref.ref(table, _forget(key))
    except TypeError:
        return facts
    with _FACTS_LOCK:
        if key not in _FACTS and len(_FACTS) >= _FACTS_KEPT:
            _FACTS.pop(next(iter(_FACTS)))
        _FACTS[key] = _Held(reference, snapshot.kept, snapshot.inventory, facts)
    return facts


# What a first select (the default filter), page and summary read by
# column, besides case_facts' POPULATION_FIELDS; a test pins that the
# summary reads nothing else.
WARM_COLUMNS = (
    "phase",
    "timestamp_ns",
    *SUMMARY_FIELDS,
)


def warm(snapshot: IndexSnapshot) -> None:
    """Compute what the first select and summary of ``snapshot`` need.

    ``case_facts``, then the columns in :data:`WARM_COLUMNS` and, for a
    columnar table, its client arrays, gap counts and (when resident) gap
    pool; each is cached on the snapshot's table. C4 registers it as a
    publish listener (``contracts.publish_hook``), so it runs on the
    builder pool after each publish, never on a request (Fable's F-7).
    A snapshot without rows warms nothing.
    """
    if not len(snapshot.table):
        return
    case_facts(snapshot)
    for name in WARM_COLUMNS:
        snapshot.column(name)
    table = columnar(snapshot.table)
    if table is None:
        return
    for metric in CLIENT_METRICS:
        table.client_values(metric)
    table.gap_counts()
    if snapshot.gaps_resident:
        try:
            table.gap_pool()
        except NotResident:
            pass


def _forget(key: int) -> Callable[[weakref.ref[Any]], None]:
    def drop(reference: weakref.ref[Any]) -> None:
        with _FACTS_LOCK:
            held = _FACTS.get(key)
            if held is not None and held.table is reference:
                del _FACTS[key]

    return drop


def _compute_facts(snapshot: IndexSnapshot) -> CaseFacts:
    phases = snapshot.column("phase")
    measured = array("q", (i for i, phase in enumerate(phases) if phase == "measured"))
    cases = snapshot.column("case_id")
    by_case: dict[str, array[int]] = {}
    for i in measured:
        by_case.setdefault(_case_name(cases[i]), array("q")).append(i)
    rows = _cohort_records(snapshot, measured)
    windows = [
        record
        for record in snapshot.kept.get("infer.phase_window", ())
        if record.get("phase") == "measured"
    ]
    workloads = snapshot.kept.get("infer.workload", ())
    workload = workloads[0] if workloads else None
    return CaseFacts(
        measured=measured,
        by_case=dict(sorted(by_case.items())),
        populations=populations_from_records(rows, windows, workload),
        planned=_planned(workload),
        missing_kept=_missing_kept(snapshot),
    )


def _cohort_records(
    snapshot: IndexSnapshot, rows: Sequence[int]
) -> list[dict[str, Any]]:
    """The fields ``populations_from_records`` reads, per row, from columns.

    A plain dict per row instead of a row view: the populations code reads
    each field several times, and a view would pay for every read. A key
    whose column cell is None is left out, as the record left it out, so
    ``record.get(key, default)`` sees the same thing it would.
    """
    if not rows:
        return []
    columns = [_picked(snapshot.column(name), rows) for name in POPULATION_FIELDS]
    return [
        {
            name: value
            for name, value in zip(POPULATION_FIELDS, cells)
            if value is not None
        }
        for cells in zip(*columns)
    ]


def _picked(values: Sequence[Any], rows: Sequence[int]) -> Sequence[Any]:
    """``[values[i] for i in rows]``, without a copy when rows are all rows."""
    if len(rows) == len(values) and (not rows or rows[-1] == len(values) - 1):
        return values
    return [values[i] for i in rows]


def _planned(workload: Mapping[str, Any] | None) -> tuple[str, ...]:
    if workload is None:
        return ()
    cases = workload.get("cases")
    if not isinstance(cases, list):
        return ()
    return tuple(
        str(case["case_id"])
        for case in cases
        if isinstance(case, Mapping) and case.get("case_id") is not None
    )


def _missing_kept(snapshot: IndexSnapshot) -> tuple[str, ...]:
    missing = []
    for event_type in COHORT_RECORDS:
        written = sum(snapshot.inventory.get(event_type, {}).values())
        if written > len(snapshot.kept.get(event_type, ())):
            missing.append(event_type)
    return tuple(missing)


def _case_name(value: Any) -> str:
    """``str(record.get("case_id", "unknown"))`` from a column cell.

    A column gives None where the record lacks the key, so None is read as
    the missing key's ``"unknown"``.
    """
    return _UNKNOWN if value is None else str(value)


# --- Selecting ------------------------------------------------------------------


def select(snapshot: IndexSnapshot, f: RequestFilter) -> Selection:
    """The client requests of ``snapshot`` that match ``f``, in file order."""
    rows: Sequence[int] = range(len(snapshot.table))
    for keep in _conditions(snapshot, f):
        rows = [i for i in rows if keep(i)]
    selected = array("q", rows)
    case_column = snapshot.column("case_id")
    cases = tuple(sorted({_case_name(case_column[i]) for i in selected}))
    rate, whole_case = _rate_scope(snapshot, f, selected, cases)
    params = canonical_params(f)
    return Selection(
        filter=f,
        params=params,
        fingerprint=selection_fingerprint(
            snapshot.path, snapshot.index_id, snapshot.generation, params
        ),
        path=snapshot.path,
        index_id=snapshot.index_id,
        generation=snapshot.generation,
        rows=selected,
        cases=cases,
        whole_case=whole_case,
        rate=rate,
        searched_fields=_searched_fields(snapshot, f),
    )


Keep = Callable[[int], bool]


def _conditions(snapshot: IndexSnapshot, f: RequestFilter) -> list[Keep]:
    """One test per condition of ``f``, each reading one or a few columns."""
    column = snapshot.column
    keeps: list[Keep] = []
    if f.phase != "all":
        phases = column("phase")
        keeps.append(lambda i: phases[i] == f.phase)
    if f.cases:
        cases, wanted = column("case_id"), frozenset(f.cases)
        keeps.append(lambda i: _case_name(cases[i]) in wanted)
    if f.statuses:
        statuses, allowed = column("status"), frozenset(f.statuses)
        keeps.append(lambda i: str(statuses[i]) in allowed)
    if f.stream is not None:
        streams, stream = column("stream"), f.stream
        keeps.append(lambda i: bool(streams[i]) is stream)
    if f.token_sources:
        sources, chosen = column("output_token_source"), frozenset(f.token_sources)
        keeps.append(lambda i: _case_name(sources[i]) in chosen)
    keeps.extend(_time_conditions(snapshot, f))
    if f.latency is not None:
        keeps.append(_latency_condition(snapshot, f.latency))
    if f.text is not None:
        keeps.append(_text_condition(snapshot, f.text))
    return keeps


def _time_conditions(snapshot: IndexSnapshot, f: RequestFilter) -> list[Keep]:
    keeps: list[Keep] = []
    if f.window is not None:
        keeps.append(_window_condition(snapshot, f.window))
    if f.completed_by_ns is not None:
        ended, limit = snapshot.column("ended_at_ns"), f.completed_by_ns
        keeps.append(lambda i: is_number(ended[i]) and ended[i] <= limit)
    return keeps


def arrival_ns(intended: Any, started: Any) -> Any:
    """A request's arrival, as ``populations._belongs`` takes it.

    ``intended_at_ns`` when it is a number, else ``started_at_ns``; the
    caller checks the result with ``is_number``.
    """
    return intended if is_number(intended) else started


def _window_condition(snapshot: IndexSnapshot, window: TimeWindow) -> Keep:
    """``populations._belongs`` on columns: arrival or overlap membership."""
    begin, end = window.start_ns, window.end_ns
    started, ended = snapshot.column("started_at_ns"), snapshot.column("ended_at_ns")
    if window.membership == "arrival":
        intended = snapshot.column("intended_at_ns")

        def arrived(i: int) -> bool:
            at = arrival_ns(intended[i], started[i])
            return is_number(at) and begin <= at < end

        return arrived

    def overlaps(i: int) -> bool:
        a, b = started[i], ended[i]
        return is_number(a) and is_number(b) and a < end and b >= begin

    return overlaps


def _latency_condition(snapshot: IndexSnapshot, latency: LatencyRange) -> Keep:
    values = snapshot.column(latency.metric)  # a CLIENT_COLUMN_PREFIX name
    low, high = latency.min_ms, latency.max_ms

    def within(i: int) -> bool:
        value = values[i]
        if value is None or value < low:
            return False
        return high is None or value < high

    return within


def _text_condition(snapshot: IndexSnapshot, text: str) -> Keep:
    columns = [snapshot.column(name) for name in _searched_fields(snapshot, None)]

    def found(i: int) -> bool:
        return any(
            column[i] is not None and text in str(column[i]).casefold()
            for column in columns
        )

    return found


def _searched_fields(
    snapshot: IndexSnapshot, f: RequestFilter | None
) -> tuple[str, ...]:
    """The fields q searches: all but ``error_message`` when it is not resident."""
    if f is not None and f.text is None:
        return ()
    fields = list(TEXT_FIELDS)
    if not snapshot.error_messages_resident:
        fields.remove("error_message")
    else:
        try:
            snapshot.column("error_message")
        except NotResident:
            fields.remove("error_message")
    return tuple(fields)


# --- Rate scope (§3.3, D-2, D-3, D-4) --------------------------------------------

_OUTCOME_FIELDS = ("statuses", "token_sources")
_OUTCOME_VALUES = ("stream", "text", "completed_by_ns", "latency")


def _rate_scope(
    snapshot: IndexSnapshot,
    f: RequestFilter,
    rows: Sequence[int],
    cases: tuple[str, ...],
) -> tuple[RateScope, str | None]:
    """The selection's rate scope, and its case when it is the whole case."""
    reason = _filter_reason(f, cases)
    if reason is not None:
        return _no_rate(reason), None
    case = _the_case(f, cases)
    facts = case_facts(snapshot)
    population = None if case is None else facts.populations.get(case)
    if case is None or population is None:
        return _no_rate("empty_selection"), None
    if facts.missing_kept:
        return _no_rate("kept_record_over_limit"), None
    if population.intervals.rate is None:
        # The whole case is still the case's cohort; only its rate is
        # undefined, and a window cannot give it one (§3.3 rule 2).
        unrated = _no_rate(population.intervals.rate_reason or "no_rate_interval")
        return unrated, case if f.window is None else None
    return _windowed(snapshot, f, rows, facts.by_case[case], case, population)


def _the_case(f: RequestFilter, cases: tuple[str, ...]) -> str | None:
    """The one case the filter names, else the one case its rows are in."""
    if f.cases:
        return f.cases[0]
    return cases[0] if cases else None


def _filter_reason(f: RequestFilter, cases: tuple[str, ...]) -> str | None:
    if f.phase != "measured":
        return "not_measured_phase"
    if any(getattr(f, name) for name in _OUTCOME_FIELDS) or any(
        getattr(f, name) is not None for name in _OUTCOME_VALUES
    ):
        return "outcome_filter"
    if len(f.cases) > 1 or len(cases) > 1:
        return "multiple_cases"
    return None


def _windowed(
    snapshot: IndexSnapshot,
    f: RequestFilter,
    rows: Sequence[int],
    case_rows: Sequence[int],
    case: str,
    population: CasePopulation,
) -> tuple[RateScope, str | None]:
    interval = population.intervals.rate
    assert interval is not None
    case_scope = RateScope(True, "case", interval, population.intervals.rate_reason)
    if f.window is None:
        return case_scope, case
    if f.window.membership == "overlap":
        return _no_rate("overlap_membership"), None
    begin = max(f.window.start_ns, interval.started_at_ns)
    end = min(f.window.end_ns, interval.ended_at_ns)
    if begin >= end:
        return _no_rate("window_outside_case"), None
    if not _all_arrive_within(snapshot, rows, begin, end):
        return _no_rate("arrival_outside_interval"), None
    if (begin, end) == (interval.started_at_ns, interval.ended_at_ns):
        if len(rows) != len(case_rows):
            return _no_rate("arrival_outside_interval"), None
        return case_scope, case
    segment = MeasuredInterval("segment", begin, end, "arrival_cohort")
    return RateScope(True, "segment", segment, None), None


def _all_arrive_within(
    snapshot: IndexSnapshot, rows: Sequence[int], begin: int, end: int
) -> bool:
    """D-3: every selected row arrives inside the clamped interval."""
    intended = snapshot.column("intended_at_ns")
    started = snapshot.column("started_at_ns")
    for i in rows:
        at = arrival_ns(intended[i], started[i])
        if not (is_number(at) and begin <= at < end):
            return False
    return True


def _no_rate(reason: str) -> RateScope:
    return RateScope(eligible=False, rule=None, interval=None, reason=reason)


# --- Paging ---------------------------------------------------------------------

Order = Literal["asc", "desc"]


def page(
    snapshot: IndexSnapshot,
    sel: Selection,
    *,
    sort: SortKey,
    order: Order,
    cursor: str | None,
    limit: int,
) -> Page:
    """One keyset page of ``sel``; ConflictError when the cursor does not fit.

    Nulls sort last in both orders and ties break by file order. The
    cursor is opaque: base64url JSON of the index ID, generation, canonical
    params, sort, order and the last row's key. A cursor from another
    index ID, filter or sort is a 409 ``cursor_mismatch``; one from an older
    generation of the same index still pages, with ``generation_moved``.
    """
    if sort not in SORT_KEYS:
        raise _bad("sort", f"unknown key {sort!r}", allowed=list(SORT_KEYS))
    if order not in ("asc", "desc"):
        raise _bad("order", f"unknown value {order!r}", allowed=["asc", "desc"])
    _check_limit(limit)
    last = (
        None
        if cursor is None
        else _read_cursor(cursor, sel, sort, order, len(snapshot.table))
    )
    ordered, keys = _ordered(snapshot, sel.rows, sort, order)
    start = 0 if last is None else _after(keys, last["key"], order)
    chosen = tuple(ordered[start : start + limit])
    more = start + limit < len(ordered)
    return Page(
        rows=chosen,
        total=len(sel.rows),
        next_cursor=(
            _cursor(sel, sort, order, keys[start + limit - 1]) if more else None
        ),
        generation_moved=last is not None and last["generation"] < sel.generation,
    )


def _check_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise _bad("limit", "a positive integer")
    if limit > PAGE_LIMIT:
        raise UsageError(
            "limit_exceeded",
            f"limit: at most {PAGE_LIMIT}",
            field="limit",
            details={"limit": PAGE_LIMIT},
        )


PAGE_LIMIT = 1_000
# A row's place in a sort: (1 for null, else 0; the value or None; the row).
SortKeyValue = tuple[int, Any, int]


def _ordered(
    snapshot: IndexSnapshot, rows: Sequence[int], sort: str, order: str
) -> tuple[Sequence[int], list[SortKeyValue]]:
    """The rows in sort order, and each one's key, side by side."""
    if sort == "artifact":
        ordered: Sequence[int] = rows if order == "asc" else rows[::-1]
        return ordered, [(0, i, i) for i in ordered]
    values = snapshot.column(SORT_KEYS[sort])
    cell = _text_cell if sort in _TEXT_SORTS else _number_cell
    present: list[SortKeyValue] = []
    nulls: list[SortKeyValue] = []
    for i in rows:
        value = cell(values[i])
        (nulls if value is None else present).append(
            (1, None, i) if value is None else (0, value, i)
        )
    present.sort(key=_value_of, reverse=order == "desc")
    keys = present + nulls
    return [key[2] for key in keys], keys


def _value_of(key: SortKeyValue) -> Any:
    return key[1]


def _number_cell(value: Any) -> Any:
    return value if is_number(value) else None


def _text_cell(value: Any) -> Any:
    return None if value is None else str(value)


def _after(keys: Sequence[SortKeyValue], last: SortKeyValue, order: str) -> int:
    """The position of the first key that sorts after ``last``."""
    low, high = 0, len(keys)
    while low < high:
        middle = (low + high) // 2
        if _sorts_after(keys[middle], last, order):
            high = middle
        else:
            low = middle + 1
    return low


def _sorts_after(key: SortKeyValue, last: SortKeyValue, order: str) -> bool:
    """Values in ``order``, then nulls; ties and nulls by file order."""
    if key[0] != last[0]:
        return key[0] > last[0]
    if key[0] == 0 and key[1] != last[1]:
        return bool(key[1] > last[1] if order == "asc" else key[1] < last[1])
    return key[2] > last[2]


def _cursor(sel: Selection, sort: str, order: str, key: SortKeyValue) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "index_id": sel.index_id,
        "generation": sel.generation,
        "params": sel.params,
        "sort": sort,
        "order": order,
        "key": list(key),
    }
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _read_cursor(
    cursor: str, sel: Selection, sort: str, order: str, size: int
) -> dict[str, Any]:
    """The cursor's payload, its key a :data:`SortKeyValue` of ``sort``.

    The key is checked against the sort, as :func:`_ordered` makes keys,
    and its row against the snapshot's ``size`` rows, so a forged key is
    a 400 and never reaches a comparison it would break (Fable's F-6).
    """
    payload = _decode_cursor(cursor)
    if payload["index_id"] != sel.index_id:
        raise _mismatch("index_rebuilt", "the index was rebuilt; restart paging")
    if payload["params"] != sel.params:
        raise _mismatch("other_filter", "the cursor belongs to another filter")
    if payload["sort"] != sort or payload["order"] != order:
        raise _mismatch("other_sort", "the cursor belongs to another sort")
    key = payload["key"]
    if not _key_fits(sort, key, size):
        raise _bad("cursor", "not a cursor this server issued")
    payload["key"] = (key[0], key[1], key[2])
    return payload


def _key_fits(sort: str, key: Sequence[Any], size: int) -> bool:
    """Whether ``key`` is one :func:`_ordered` could have made for ``sort``
    over ``size`` rows: ``(1, None, row)`` for a null; ``(0, row, row)``
    in file order; else ``(0, value, row)`` with text for a text sort and
    a finite number for any other."""
    flag, value, row = key
    if not 0 <= row < size:
        return False
    if sort == "artifact":
        return flag == 0 and value == row and _is_int(value)
    if flag == 1:
        return value is None
    if sort in _TEXT_SORTS:
        return isinstance(value, str)
    return is_number(value)


def _decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        _check_cursor(payload)
    except (ValueError, UnicodeError, binascii.Error, TypeError) as exc:
        raise _bad("cursor", "not a cursor this server issued") from exc
    return dict(payload)


def _check_cursor(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        raise ValueError("cursor version")
    for name in ("index_id", "params", "sort", "order"):
        if not isinstance(payload.get(name), str):
            raise ValueError(name)
    if not _is_int(payload.get("generation")):
        raise ValueError("generation")
    key = payload.get("key")
    if not isinstance(key, list) or len(key) != 3:
        raise ValueError("key")
    if not _is_int(key[0]) or key[0] not in (0, 1):
        raise ValueError("key flag")
    if not _is_int(key[2]):
        raise ValueError("key row")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _mismatch(reason: str, message: str) -> ConflictError:
    return ConflictError(
        "cursor_mismatch", message, field="cursor", details={"reason": reason}
    )


# --- One request ----------------------------------------------------------------


def request_row(snapshot: IndexSnapshot, request_id: str, line: int | None) -> int:
    """The row of ``request_id``, picked by ``line`` when the ID repeats.

    NotFoundError ``request_not_found`` for no such ID or no such line;
    ConflictError ``request_ambiguous``, listing the lines, when several
    rows share the ID and no ``line`` picks one.
    """
    found = snapshot.by_request_id.get(request_id, ())
    if line is not None:
        found = tuple(i for i in found if snapshot.requests[i].ref.line == line)
    if not found:
        raise NotFoundError(
            "request_not_found", f"no request {request_id!r}", field="request_id"
        )
    if len(found) > 1:
        lines = [snapshot.requests[i].ref.line for i in found]
        raise ConflictError(
            "request_ambiguous",
            f"{len(found)} requests have this ID; pick one by line",
            field="request_id",
            details={"lines": lines},
        )
    return found[0]


def in_selection(sel: Selection, row: int) -> bool:
    """Whether ``row`` is one of ``sel``'s rows (they are in file order)."""
    at = bisect_left(sel.rows, row)
    return at < len(sel.rows) and sel.rows[at] == row
