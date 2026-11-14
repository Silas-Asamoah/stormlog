"""The documents of a selection: summary, cases, rows, one request, plots (§5.4).

Each function returns a document body without ``format``,
``schema_version`` or ``artifact``; C4 wraps it with
``contracts.envelope(name, body, snapshot)``.

Every number comes from Stormlog's analyzer functions, called here on the
selection's rows: ``latency_summary``, ``streaming_summary``,
``populations_from_records`` (through :func:`selection.case_facts`),
``count_population``, ``failure_share``, ``token_coverage``,
``successful_throughput``, ``spread``, ``elapsed_ms`` and the
``sample_report`` helpers. What this module adds is selection and
presentation: picking rows, converting ns to ms from ``origin_ns``,
counting values into bins and choosing which points to draw (§5.7).

Server evidence is ``not_indexed`` (decision 4(a)): the index keeps no
joined spans or execution records, so ``counts.server_admitted`` and
``server_evidence_coverage`` are null with ``server_evidence:
"not_indexed"``, and latency holds client metrics only. The analyzer
report's values are shown, labelled, in Sources (C4).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from typing import Any, cast

from stormlog.infer.latency_report import (
    CLIENT_METRICS,
    elapsed_ms,
    failure_share,
    latency_summary,
    spread,
    streaming_summary,
)
from stormlog.infer.observers import SCRAPE_GAP_FACTOR
from stormlog.infer.populations import (
    CasePopulation,
    MeasuredInterval,
    count_population,
)
from stormlog.infer.quantiles import quantile
from stormlog.infer.report_stats import is_number
from stormlog.infer.sample_report import (
    group_samples,
    request_window,
    sample_peaks,
    samples_in_window,
)
from stormlog.infer.slo import CriterionValue
from stormlog.infer.token_report import successful_throughput, token_coverage
from stormlog.studio.contracts import (
    IndexSnapshot,
    LineRef,
    NotResident,
    Page,
    RequestFilter,
    RequestRow,
    Selection,
    TimeWindow,
    UsageError,
    criterion_record,
    finite_copy,
)
from stormlog.studio.selection import (
    case_facts,
    in_selection,
    select,
    without_latency,
)

__all__ = [
    "DETAIL_GAPS",
    "MAX_BINS",
    "MAX_CHUNK_PAGE",
    "MAX_EDGES",
    "MAX_POINTS",
    "NON_OK_POINTS",
    "POINT_X",
    "POINT_Y",
    "SAMPLE_FIELDS",
    "cases",
    "chunks",
    "distribution",
    "points",
    "request_detail",
    "requests_page",
    "row_record",
    "samples",
    "selection_block",
    "summary",
]

DETAIL_GAPS = 128  # gaps request detail gives in full; the rest are on chunks
MAX_CHUNK_PAGE = 10_000
MAX_BINS = 200
MAX_EDGES = 201
MAX_POINTS = 5_000
NON_OK_POINTS = 500  # non-ok rows always drawn beyond the buckets (§5.7)
POINT_X = {"start": "started_at_ns", "end": "ended_at_ns", "arrival": "arrival"}
POINT_Y = (*CLIENT_METRICS, "first_chunk", "output_tokens", "chunk_count")
SAMPLE_FIELDS = (
    "process_rss_bytes",
    "device_used_bytes",
    "device_total_bytes",
    "gpu_utilization_percent",
)
STREAMING_UNAVAILABLE = {"state": "unavailable", "reason": "gaps_not_resident"}
SERVER_FIELDS = ("server_admitted", "server_evidence_coverage")


# --- The selection block ------------------------------------------------------------


def selection_block(sel: Selection, snapshot: IndexSnapshot) -> dict[str, Any]:
    """The §5.3 block every selection route returns under ``selection``."""
    return {
        "filter": _filter_record(sel.filter),
        "params": sel.params,
        "fingerprint": sel.fingerprint,
        "size": len(sel.rows),
        "pooled_cases": len(sel.cases),
        "whole_case": sel.whole_case,
        "cohort": _cohort(sel, snapshot),
        "rate": {
            "eligible": sel.rate.eligible,
            "rule": sel.rate.rule,
            "reason": sel.rate.reason,
            "interval": _interval(sel.rate.interval),
        },
        "searched_fields": list(sel.searched_fields),
    }


def _filter_record(f: RequestFilter) -> dict[str, Any]:
    return {
        "phase": f.phase,
        "cases": list(f.cases),
        "statuses": list(f.statuses),
        "stream": f.stream,
        "token_sources": list(f.token_sources),
        "text": f.text,
        "window": _window_record(f.window),
        "completed_by_ns": f.completed_by_ns,
        "latency": (
            None
            if f.latency is None
            else {
                "metric": f.latency.metric,
                "min_ms": f.latency.min_ms,
                "max_ms": f.latency.max_ms,
            }
        ),
    }


def _window_record(window: TimeWindow | None) -> dict[str, Any] | None:
    if window is None:
        return None
    return {
        "start_ns": window.start_ns,
        "end_ns": window.end_ns,
        "membership": window.membership,
    }


def _interval(interval: MeasuredInterval | None) -> dict[str, Any] | None:
    return None if interval is None else interval.to_record()


def _cohort(sel: Selection, snapshot: IndexSnapshot) -> dict[str, Any]:
    population = _whole_case(sel, snapshot)
    if population is None:
        reason = (
            sel.rate.reason
            if sel.rate.reason == "kept_record_over_limit"
            else "not_a_cohort"
        )
        return {"valid": None, "issues": [], "reason": reason}
    return {
        "valid": population.population.cohort_valid,
        "issues": list(population.population.issues),
        "reason": None,
    }


def _whole_case(sel: Selection, snapshot: IndexSnapshot) -> CasePopulation | None:
    if sel.whole_case is None:
        return None
    return case_facts(snapshot).populations.get(sel.whole_case)


# --- Summary ------------------------------------------------------------------------


def summary(snapshot: IndexSnapshot, sel: Selection) -> dict[str, Any]:
    """The numbers of one selection (§5.4), each from its analyzer function.

    A whole case (``sel.whole_case``) takes its counts, cohort and
    intervals from ``populations_from_records``, as the analyzer's case
    block does; any other selection is counted by ``count_population`` and
    is no cohort. ``throughput`` is null unless the selection is
    rate-eligible (§3.3). An all-failed case keeps the analyzer's 0.0
    requests per second over its interval; the UI labels it from
    ``counts.successful == 0`` (choice 21). An empty selection, such as a
    planned case with no observations, gives the same shape with every
    value empty (choice 16).
    """
    rows = _rows(snapshot, sel.rows)
    population = _whole_case(sel, snapshot)
    counts = _counts(rows, population)
    ok = [row for row in rows if row.get("status") == "ok"]
    return {
        "selection": selection_block(sel, snapshot),
        "counts": counts,
        "failure": failure_share(counts["offered"], counts["successful"]),
        "latency": latency_summary(rows, client_values_of=_client_values),
        "streaming": _streaming(snapshot, rows),
        "tokens": token_coverage(rows),
        "intervals": None if population is None else population.intervals.to_record(),
        "throughput": _throughput(sel, population, ok),
        "client_memory": _client_memory(snapshot, ok),
        "freshness": _freshness(snapshot),
    }


def _throughput(
    sel: Selection, population: CasePopulation | None, ok: Sequence[RequestRow]
) -> dict[str, Any] | None:
    """The rate block of an eligible selection, or of a whole case.

    A whole case whose rate interval is undefined (an interrupted phase)
    gets the analyzer's block, every rate null over no interval, so the
    case equals its analyzer block (§3.4); any other selection without a
    rate gets null (§5.4).
    """
    if sel.rate.eligible:
        return successful_throughput(ok, sel.rate.interval)
    if population is not None:
        return successful_throughput(ok, population.intervals.rate)
    return None


def _rows(snapshot: IndexSnapshot, indexes: Sequence[int]) -> list[RequestRow]:
    requests = snapshot.requests
    return [requests[i] for i in indexes]


def _client_values(record: Mapping[str, Any]) -> Mapping[str, CriterionValue]:
    """``slo.client_values`` of the row, as the index stored it."""
    return cast(RequestRow, record).client


def _counts(
    rows: Sequence[RequestRow], population: CasePopulation | None
) -> dict[str, Any]:
    if population is None:
        counts = count_population(rows).to_record()
        counts["cohort_valid"] = None
    else:
        counts = population.population.to_record()
    for field in SERVER_FIELDS:
        counts[field] = None
    counts["server_evidence"] = "not_indexed"
    return counts


def _streaming(snapshot: IndexSnapshot, rows: Sequence[RequestRow]) -> dict[str, Any]:
    if not snapshot.gaps_resident:
        return dict(STREAMING_UNAVAILABLE)
    try:
        return streaming_summary(rows)
    except NotResident:
        return dict(STREAMING_UNAVAILABLE)


def _client_memory(snapshot: IndexSnapshot, ok: Sequence[RequestRow]) -> dict[str, Any]:
    """Client samples during the successful requests, by device, as the
    analyzer windows them (first start to last end, both included)."""
    window = request_window(ok)
    memory: dict[str, Any] = {"observation_scope": "client_local"}
    if window is None:
        return {**memory, "window": None, "by_device": []}
    start, end = window
    in_window = samples_in_window(snapshot.samples, start, end)
    return {
        **memory,
        "window": {"started_at_ns": start, "ended_at_ns": end},
        "by_device": sample_peaks(in_window),
    }


def _freshness(snapshot: IndexSnapshot) -> dict[str, Any]:
    """How current the numbers are: the generation, the last client request's
    ``timestamp_ns`` in file order, and whether the run ended."""
    stamps = snapshot.column("timestamp_ns") if len(snapshot.table) else ()
    last = stamps[-1] if stamps else None
    return {
        "generation": snapshot.generation,
        "last_record_ns": last if _is_int(last) else None,
        "terminal": snapshot.session is not None and snapshot.session.terminal,
    }


# --- Cases ------------------------------------------------------------------------


def cases(snapshot: IndexSnapshot) -> dict[str, Any]:
    """One summary per measured case, planned cases included, no deltas.

    A case is listed when the workload record plans it or a measured client
    request names it. A planned case with no observations gets the
    empty-selection summary (choice 16); an all-failed case keeps its
    failures and its 0.0 rate (choice 21).
    """
    facts = case_facts(snapshot)
    planned = set(facts.planned)
    entries = []
    for case_id in sorted(planned | set(facts.by_case)):
        sel = select(snapshot, RequestFilter(cases=(case_id,)))
        entries.append(
            {
                "case_id": case_id,
                "planned": case_id in planned,
                "observed": case_id in facts.by_case,
                "summary": summary(snapshot, sel),
            }
        )
    return {"comparison_scope": "full_measured_case", "cases": entries}


# --- Rows ---------------------------------------------------------------------------


def requests_page(
    snapshot: IndexSnapshot, sel: Selection, page: Page
) -> dict[str, Any]:
    """One page of rows: the selection block, the rows, the cursor (§5.4)."""
    return {
        "selection": selection_block(sel, snapshot),
        "rows": [row_record(snapshot, i) for i in page.rows],
        "next_cursor": page.next_cursor,
        "total": page.total,
        "generation_moved": page.generation_moved,
    }


def row_record(snapshot: IndexSnapshot, index: int) -> dict[str, Any]:
    """One request row: no error message and no gaps (§5.4)."""
    row = snapshot.requests[index]
    started, ended = _ns(row.get("started_at_ns")), _ns(row.get("ended_at_ns"))
    return {
        "session_id": row.session_id,
        "request_id": row.request_id,
        "line": row.ref.line,
        "record_id": row.record_id,
        "line_sha256": row.ref.sha256,
        "case_id": row.case_id,
        "phase": row.phase,
        "status": str(row.get("status")),
        "http_status": _int_or_none(row.get("http_status")),
        "error_type": _text_or_none(row.get("error_type")),
        "started_at_ns": started,
        "ended_at_ns": ended,
        "start_ms": _ms_from_origin(started, snapshot.origin_ns),
        "end_ms": _ms_from_origin(ended, snapshot.origin_ns),
        "client": {key: criterion_record(value) for key, value in row.client.items()},
        "first_chunk_latency_ms": _number_or_none(row.get("first_chunk_latency_ms")),
        **_token_fields(row, "prompt"),
        **_token_fields(row, "output"),
        "stream": _bool_or_none(row.get("stream")),
        "gap_count": row.gap_count,
        "chunk_count": _chunk_count(
            row.get("stream"), row.get("ttft_ms"), row.numeric_gap_count
        ),
        "finish_reason": _text_or_none(row.get("finish_reason")),
    }


def _token_fields(row: RequestRow, kind: str) -> dict[str, Any]:
    return {
        f"{kind}_tokens": _int_or_none(row.get(f"{kind}_tokens")),
        f"{kind}_token_source": str(row.get(f"{kind}_token_source", "unknown")),
        f"{kind}_token_exact": row.get(f"{kind}_token_exact") is True,
    }


def _chunk_count(stream: Any, ttft_ms: Any, numeric_gaps: int) -> int | None:
    """``numeric_gaps + 1`` for a streamed response with a TTFT, as the
    analyzer counts content chunks (a gap that is not a finite number is
    no chunk); else None."""
    return numeric_gaps + 1 if stream and is_number(ttft_ms) else None


def _ms_from_origin(ns: int | None, origin_ns: int | None) -> float | None:
    """A client-clock time as ms from ``origin_ns``: the plots' time axis."""
    if ns is None or origin_ns is None:
        return None
    return (ns - origin_ns) / 1_000_000


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ns(value: Any) -> int | None:
    return value if _is_int(value) else None


def _int_or_none(value: Any) -> int | None:
    return value if _is_int(value) else None


def _number_or_none(value: Any) -> float | int | None:
    return value if is_number(value) else None


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


# --- One request --------------------------------------------------------------------


def request_detail(
    snapshot: IndexSnapshot, row: int, sel: Selection | None
) -> dict[str, Any]:
    """One request in full (§5.4): its record, gap spread, links, membership.

    The record is built from the row's keys plus ``load_error_message``,
    never by iterating a row whose pools are not resident, and passes
    through ``finite_copy``. Its gaps are left out of ``record``: the first
    128 are under ``gaps.first`` and every one is on the chunks route.
    ``sel`` is the filter the page holds; None means the default filter.
    """
    request = snapshot.requests[row]
    chosen = sel if sel is not None else select(snapshot, RequestFilter())
    record, non_finite = finite_copy(_record(request))
    gaps = request.load_gaps()
    numeric = [gap for gap in gaps if is_number(gap)]
    return {
        "selection": selection_block(chosen, snapshot),
        "in_selection": in_selection(chosen, row),
        "row": row_record(snapshot, row),
        "record": record,
        "non_finite_fields": non_finite,
        "record_omitted": ["chunk_interarrival_ms"],
        "elapsed_ms": None if request.get("status") == "ok" else elapsed_ms(request),
        "gaps": {
            **spread(numeric),
            "n": len(numeric),
            "gap_count": request.gap_count,
            "first": [_finite_or_none(gap) for gap in gaps[:DETAIL_GAPS]],
        },
        "links": _links(snapshot, request.request_id),
    }


_POOLED = frozenset({"chunk_interarrival_ms", "error_message"})


def _record(request: RequestRow) -> dict[str, Any]:
    # Iterating a row gives its keys without reading a value, so a pool that
    # is not resident is never touched here.
    keys = list(request)
    record = {key: request[key] for key in keys if key not in _POOLED}
    if "error_message" in keys:
        record["error_message"] = request.load_error_message()
    return record


def _finite_or_none(value: float) -> float | None:
    return value if is_number(value) else None


def _links(snapshot: IndexSnapshot, request_id: str) -> dict[str, Any]:
    dispatch, first_content = snapshot.request_links.get(request_id, (None, None))
    return {"dispatch": _link(dispatch), "first_content": _link(first_content)}


def _link(ref: LineRef | None) -> dict[str, Any] | None:
    return None if ref is None else {"line": ref.line, "sha256": ref.sha256}


def chunks(
    snapshot: IndexSnapshot, row: int, *, offset: int, limit: int
) -> dict[str, Any]:
    """A page of one request's chunk gaps, in written order; never truncated."""
    _check_count("offset", offset, minimum=0, maximum=None)
    _check_count("limit", limit, minimum=1, maximum=MAX_CHUNK_PAGE)
    request = snapshot.requests[row]
    gaps = request.load_gaps()
    return {
        "request_id": request.request_id,
        "line": request.ref.line,
        "offset": offset,
        "limit": limit,
        "total": len(gaps),
        "gaps_ms": [_finite_or_none(gap) for gap in gaps[offset : offset + limit]],
        "note": "chunk gaps in written order; null for a value that is not a number",
    }


def _check_count(name: str, value: Any, *, minimum: int, maximum: int | None) -> None:
    if not _is_int(value) or value < minimum:
        raise UsageError(
            "bad_parameter", f"{name}: an integer >= {minimum}", field=name
        )
    if maximum is not None and value > maximum:
        raise UsageError(
            "limit_exceeded",
            f"{name}: at most {maximum}",
            field=name,
            details={"limit": maximum},
        )


# --- Distribution -------------------------------------------------------------------


def distribution(
    snapshot: IndexSnapshot,
    sel: Selection,
    *,
    metric: str,
    edges: Sequence[float] | None,
    bins: int | None,
) -> dict[str, Any]:
    """Counts of one client metric in half-open bins (§5.4).

    The denominator is the selection without its latency condition, so a
    brush that narrows the latency keeps the same bars to compare with.
    With ``bins`` the edges run equal-width from the smallest value to just
    above the largest (``math.nextafter``), so the largest falls in the last
    bin; with ``edges`` the caller picks them, and values outside are
    counted in ``below`` and ``above``. Each bin carries the exact
    ``latency=`` parameter that selects it.
    """
    if metric not in CLIENT_METRICS:
        raise UsageError(
            "bad_parameter",
            f"metric: unknown {metric!r}",
            field="metric",
            details={"allowed": list(CLIENT_METRICS)},
        )
    base = select(snapshot, without_latency(sel.filter)) if sel.filter.latency else sel
    column = snapshot.column(metric)
    values = sorted(v for v in (column[i] for i in base.rows) if v is not None)
    chosen = _edges(values, edges, bins)
    return {
        "selection": selection_block(sel, snapshot),
        "metric": metric,
        "mode": "edges" if edges is not None else "bins",
        "bins": [
            {
                "lower_ms": low,
                "upper_ms": high,
                "count": bisect_left(values, high) - bisect_left(values, low),
                "latency": f"{metric}:{low!r}:{high!r}",
            }
            for low, high in zip(chosen, chosen[1:])
        ],
        "below": bisect_left(values, chosen[0]) if chosen else 0,
        "above": len(values) - bisect_left(values, chosen[-1]) if chosen else 0,
        "denominator": {
            "params": base.params,
            "size": len(base.rows),
            "with_value": len(values),
            "without_value": len(base.rows) - len(values),
        },
        "bounds": {"max_bins": MAX_BINS, "max_edges": MAX_EDGES},
    }


def _edges(
    values: Sequence[float], edges: Sequence[float] | None, bins: int | None
) -> list[float]:
    if (edges is None) == (bins is None):
        raise UsageError("bad_parameter", "give bins or edges, not both", field="bins")
    if edges is not None:
        return _given_edges(edges)
    _check_count("bins", bins, minimum=1, maximum=MAX_BINS)
    assert bins is not None
    if not values:
        return []
    low, high = float(values[0]), math.nextafter(float(values[-1]), math.inf)
    width = (high - low) / bins
    chosen = [low]
    for k in range(1, bins):
        edge = low + width * k
        if chosen[-1] < edge < high:
            chosen.append(edge)
    chosen.append(high)
    return chosen


def _given_edges(edges: Sequence[float]) -> list[float]:
    chosen = [float(edge) for edge in edges]
    if not 2 <= len(chosen) <= MAX_EDGES:
        raise UsageError(
            "limit_exceeded" if len(chosen) > MAX_EDGES else "bad_parameter",
            f"edges: 2 to {MAX_EDGES} values",
            field="edges",
            details={"limit": MAX_EDGES},
        )
    if not all(math.isfinite(edge) for edge in chosen) or any(
        a >= b for a, b in zip(chosen, chosen[1:])
    ):
        raise UsageError("bad_parameter", "edges: finite and increasing", field="edges")
    return chosen


# --- Points -------------------------------------------------------------------------


def points(
    snapshot: IndexSnapshot,
    sel: Selection,
    *,
    x: str,
    y: str,
    max_points: int,
    keep: Sequence[str],
) -> dict[str, Any]:
    """The selection's requests as (x, y) points, bounded (§5.4, §5.7).

    ``x`` is ``start``, ``end`` or ``arrival`` (``intended_at_ns``, else
    ``started_at_ns``), as ms from ``origin_ns`` with the exact ns beside
    it. Up to ``max_points`` rows are drawn as they are; beyond that,
    ``max_points // 2`` equal x-buckets each keep their lowest and highest
    y, so outliers survive; then every ``keep`` request ID and up to 500
    rows that did not succeed are added. Rows without x or y are counted
    in ``without_value``.
    """
    _check_point_args(x, y, max_points)
    xs = _x_values(snapshot, x)
    ys = _y_values(snapshot, sel.rows, y)
    valued = [i for i in sel.rows if _is_int(xs[i]) and is_number(ys[i])]
    drawn, method = _drawn(snapshot, valued, xs, ys, max_points, keep)
    requests = snapshot.requests
    statuses = snapshot.column("status")
    return {
        "selection": selection_block(sel, snapshot),
        "x": x,
        "y": y,
        "points": [
            {
                "line": requests[i].ref.line,
                "request_id": requests[i].request_id,
                "status": str(statuses[i]),
                "x_ms": _ms_from_origin(xs[i], snapshot.origin_ns),
                "x_ns": xs[i],
                "y": ys[i],
            }
            for i in drawn
        ],
        "drawn": len(drawn),
        "total": len(valued),
        "without_value": len(sel.rows) - len(valued),
        "method": method,
        "bounds": {
            "max_points": max_points,
            "buckets": max_points // 2,
            "non_ok": NON_OK_POINTS,
        },
    }


def _check_point_args(x: str, y: str, max_points: int) -> None:
    if x not in POINT_X:
        raise UsageError(
            "bad_parameter",
            f"x: unknown {x!r}",
            field="x",
            details={"allowed": list(POINT_X)},
        )
    if y not in POINT_Y:
        raise UsageError(
            "bad_parameter",
            f"y: unknown {y!r}",
            field="y",
            details={"allowed": list(POINT_Y)},
        )
    _check_count("max_points", max_points, minimum=2, maximum=MAX_POINTS)


def _x_values(snapshot: IndexSnapshot, x: str) -> Sequence[Any]:
    if x != "arrival":
        return snapshot.column(POINT_X[x])
    intended = snapshot.column("intended_at_ns")
    started = snapshot.column("started_at_ns")
    return _Arrivals(intended, started)


class _Arrivals(Sequence[Any]):
    """``intended_at_ns`` where it is a number, else ``started_at_ns``."""

    def __init__(self, intended: Sequence[Any], started: Sequence[Any]) -> None:
        self._intended = intended
        self._started = started

    def __len__(self) -> int:
        return len(self._started)

    def __getitem__(self, i: Any) -> Any:
        intended = self._intended[i]
        return intended if is_number(intended) else self._started[i]


_Y_COLUMNS = {"first_chunk": "first_chunk_latency_ms", "output_tokens": "output_tokens"}


def _y_values(snapshot: IndexSnapshot, rows: Sequence[int], y: str) -> Sequence[Any]:
    if y in CLIENT_METRICS:
        return snapshot.column(y)
    if y in _Y_COLUMNS:
        return snapshot.column(_Y_COLUMNS[y])
    streams = snapshot.column("stream")
    ttfts = snapshot.column("ttft_ms")
    counts: dict[int, int | None] = {}
    for i in rows:
        counts[i] = _chunk_count(
            streams[i], ttfts[i], snapshot.requests[i].numeric_gap_count
        )
    return _Sparse(counts)


class _Sparse(Sequence[Any]):
    """Values for some rows; None elsewhere."""

    def __init__(self, values: Mapping[int, Any]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, i: Any) -> Any:
        return self._values.get(i)


def _drawn(
    snapshot: IndexSnapshot,
    valued: Sequence[int],
    xs: Sequence[Any],
    ys: Sequence[Any],
    max_points: int,
    keep: Sequence[str],
) -> tuple[list[int], str]:
    if len(valued) <= max_points:
        return list(valued), "all"
    chosen = set(_bucket_extremes(valued, xs, ys, max_points // 2))
    wanted = set(keep)
    ids = snapshot.column("request_id")
    statuses = snapshot.column("status")
    chosen.update(i for i in valued if str(ids[i]) in wanted)
    failed = [i for i in valued if statuses[i] != "ok" and i not in chosen]
    chosen.update(failed[:NON_OK_POINTS])
    return sorted(chosen, key=lambda i: (xs[i], i)), "bucket_extremes"


def _bucket_extremes(
    valued: Sequence[int], xs: Sequence[Any], ys: Sequence[Any], buckets: int
) -> list[int]:
    """Each x-bucket's lowest and highest y; ties go to the earlier row."""
    low_x = min(xs[i] for i in valued)
    span = max(xs[i] for i in valued) - low_x + 1
    lowest: dict[int, int] = {}
    highest: dict[int, int] = {}
    for i in valued:
        bucket = (xs[i] - low_x) * buckets // span
        if bucket not in lowest or ys[i] < ys[lowest[bucket]]:
            lowest[bucket] = i
        if bucket not in highest or ys[i] > ys[highest[bucket]]:
            highest[bucket] = i
    return [*lowest.values(), *highest.values()]


# --- Samples ------------------------------------------------------------------------


def samples(
    snapshot: IndexSnapshot, *, window: TimeWindow | None, max_points: int
) -> dict[str, Any]:
    """Client system samples per ``(sampler, device_id)``, bounded (§5.7).

    Series are the groups ``sample_report.group_samples`` forms, each with
    its ``sample_peaks`` entry. Per field, a series of at most
    ``max_points`` samples is given whole; a longer one is cut into
    ``max_points // 3`` equal time buckets, each with its minimum, maximum
    and last value and their times. A gap is marked where consecutive
    samples are more than ``observers.SCRAPE_GAP_FACTOR`` times the series'
    median interval apart. ``window`` is half-open on the client clock.
    """
    _check_count("max_points", max_points, minimum=3, maximum=MAX_POINTS)
    timed = [s for s in snapshot.samples if _is_int(s.get("timestamp_ns"))]
    if window is not None:
        timed = samples_in_window(timed, window.start_ns, window.end_ns - 1)
    series = [
        _series(snapshot, sampler, device_id, members, max_points)
        for sampler, device_id, members in group_samples(timed)
    ]
    return {
        "observation_scope": "client_local",
        "window": _window_record(window),
        "total": len(timed),
        "series": series,
        "bounds": {"max_points": max_points, "buckets": max_points // 3},
        "gap_factor": SCRAPE_GAP_FACTOR,
    }


def _series(
    snapshot: IndexSnapshot,
    sampler: Any,
    device_id: Any,
    members: Sequence[Mapping[str, Any]],
    max_points: int,
) -> dict[str, Any]:
    ordered = sorted(members, key=lambda s: int(s["timestamp_ns"]))
    times = [int(s["timestamp_ns"]) for s in ordered]
    return {
        "sampler": sampler,
        "device_id": device_id,
        "samples": len(ordered),
        "peaks": sample_peaks(ordered)[0],
        "gaps": _sample_gaps(times),
        "fields": {
            field: _field_series(snapshot, ordered, field, max_points)
            for field in SAMPLE_FIELDS
            if any(is_number(s.get(field)) for s in ordered)
        },
    }


def _sample_gaps(times: Sequence[int]) -> list[dict[str, int]]:
    intervals = [b - a for a, b in zip(times, times[1:])]
    median = quantile(intervals, 0.5)
    if not median:
        return []
    return [
        {"after_ns": a, "before_ns": b}
        for a, b in zip(times, times[1:])
        if b - a > SCRAPE_GAP_FACTOR * median
    ]


def _field_series(
    snapshot: IndexSnapshot,
    ordered: Sequence[Mapping[str, Any]],
    field: str,
    max_points: int,
) -> dict[str, Any]:
    valued = [
        (int(s["timestamp_ns"]), s[field]) for s in ordered if is_number(s.get(field))
    ]
    origin = snapshot.origin_ns
    if len(valued) <= max_points:
        return {
            "method": "all",
            "points": [
                {"t_ns": t, "t_ms": _ms_from_origin(t, origin), "value": v}
                for t, v in valued
            ],
        }
    return {
        "method": "bucket_min_max_last",
        "buckets": _buckets(valued, max_points // 3, origin),
    }


def _buckets(
    valued: Sequence[tuple[int, Any]], count: int, origin: int | None
) -> list[dict[str, Any]]:
    first, span = valued[0][0], valued[-1][0] - valued[0][0] + 1
    grouped: dict[int, list[tuple[int, Any]]] = {}
    for t, v in valued:
        grouped.setdefault((t - first) * count // span, []).append((t, v))
    return [_bucket(members, origin) for _, members in sorted(grouped.items())]


def _bucket(members: Sequence[tuple[int, Any]], origin: int | None) -> dict[str, Any]:
    low = min(members, key=_second)
    high = max(members, key=_second)
    last = members[-1]
    return {
        "start_ns": members[0][0],
        "end_ns": last[0],
        "start_ms": _ms_from_origin(members[0][0], origin),
        "count": len(members),
        "min": low[1],
        "min_ns": low[0],
        "max": high[1],
        "max_ns": high[0],
        "last": last[1],
        "last_ns": last[0],
    }


def _second(pair: tuple[int, Any]) -> Any:
    return pair[1]
