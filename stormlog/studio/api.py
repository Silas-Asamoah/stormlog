"""Studio's resource routes: every slice-1 route but C3's four (design §5.4).

Each handler reads its query, takes the artifact's current snapshot from
C1's registry, picks rows with C2's ``selection``, builds the document
with C2's ``summaries`` and ``metric_catalog`` or C4's ``workspace``,
``export`` and ``analysis_cache``, and answers it through
``contracts.envelope`` and ``encode_json``. No number is computed here.

Rules every handler follows:

- **Parameters.** A name the route does not take is 400
  ``unknown_parameter``; a single-valued one given twice is 400
  ``bad_parameter``. Filters, cursors and sorts go through ``selection``
  only, so every selection route reads one query the same way (§5.3).
- **Index states.** ``GET /api/v1/artifact``, its capabilities, sources
  and analysis, and the raw-line and evidence routes answer for any
  index state, a failed one included, so the frontend can show why. Every
  other artifact route refuses a failed index with the §5.2 code: 422
  ``unreadable``, ``not_an_inference_artifact``, or
  ``unsupported_artifact`` (``details.reason`` ``audio`` or
  ``multiple_sessions``); an index that failed on an internal error is
  500 ``internal_error``. A building index is served: its documents are
  of the zero-row snapshot, with ``artifact.status: "building"``.
- **Caches.** Documents are cached in ``ctx.cache`` under the path, index
  ID, generation and canonical parameters, plus the snapshot's status,
  growth and file identity, since a republish of the same generation
  changes the artifact block. The analysis document and the panels are
  never cached: the analysis changes state with no new generation (C4a's
  gap 6).
- **Analysis.** While the analysis is ``pending`` or ``running`` its
  route raises :class:`NotReady` with the whole enveloped analysis
  document, artifact block included, which httpd answers with 202 and
  ``Retry-After``. This is Studio's only 202 (§5.2, F-2).
- **Errors.** C1, C2 and C4 raise the §5.2 errors themselves. What else a
  read can raise is mapped here: a file that vanishes or fails mid-read
  (``OSError``) is 422 ``unreadable``, and a field the index does not
  hold (``NotResident``) is 422 ``unreadable`` with ``details.reason``
  ``not_resident``.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from itertools import chain
from typing import Any, Literal, cast
from urllib.parse import quote, urlencode

from stormlog.infer.diagnosis_inputs import record_id as raw_record_id
from stormlog.infer.events import REQUEST_STATUSES
from stormlog.studio import export, metric_catalog, selection, summaries
from stormlog.studio.analysis_cache import AnalyzerRunner
from stormlog.studio.cache import cache_key
from stormlog.studio.contracts import (
    SCHEMA_VERSION,
    AnalysisState,
    Auth,
    ConflictError,
    HttpRequest,
    HttpResponse,
    IndexSnapshot,
    InvalidInputError,
    LineProblem,
    LineRef,
    NotFoundError,
    NotResident,
    Route,
    Selection,
    SessionInfo,
    SortKey,
    StudioContext,
    StudioError,
    UsageError,
    columnar,
    encode_json,
    envelope,
)
from stormlog.studio.index import refuse_oversized
from stormlog.studio.workspace import read_report

__all__ = [
    "API_VERSION",
    "DEFAULT_PAGE",
    "JSON_TYPE",
    "LINE_TYPE",
    "artifact_body",
    "resource_routes",
]

API_VERSION = SCHEMA_VERSION
JSON_TYPE = "application/json; charset=utf-8"
LINE_TYPE = "text/plain; charset=utf-8"
# Defaults for parameters the client may leave out.
DEFAULT_PAGE = 100
DEFAULT_CHUNKS = 1_000
DEFAULT_POINTS = 2_000
DEFAULT_SAMPLES = 1_000
DEFAULT_BINS = 20
_REQUEST_PARAMS = ("sort", "order", "cursor", "limit")
_DISTRIBUTION_PARAMS = ("metric", "bins", "edges")
_POINT_PARAMS = ("x", "y", "max_points", "keep")
_INTEGER = re.compile(r"-?[0-9]{1,18}")
_DETAIL_PARAMS = ("request_id", "line")
_WINDOW_PARAMS = ("window", "window_ms", "window_membership")
_UNSUPPORTED = {
    "unreadable": ("unreadable", None),
    "not_an_inference_artifact": ("not_an_inference_artifact", None),
    "audio": ("unsupported_artifact", "audio"),
    "multiple_sessions": ("unsupported_artifact", "multiple_sessions"),
}
_MESSAGES = {
    "unreadable": "the artifact could not be read",
    "not_an_inference_artifact": "no infer.session or infer.request record",
    "unsupported_artifact": "Studio does not support this artifact in slice 1",
}
_JOBS = {"state": "unavailable", "reason": "slice_2"}

Handler = Callable[[HttpRequest], HttpResponse]
Query = Mapping[str, Sequence[str]]
Method = Literal["GET", "POST"]
Order = Literal["asc", "desc"]


def resource_routes(ctx: StudioContext) -> list[Route]:
    """Every non-auth route of §5.4, bound to ``ctx``."""
    a = _Api(ctx)
    long_s = ctx.options.limits.long_deadline_s
    table: list[tuple[Method, str, Auth, Handler, float | None]] = [
        ("GET", "/api/v1/server", "bearer", a.server, None),
        ("GET", "/api/v1/capabilities", "bearer", a.capabilities, None),
        ("GET", "/api/v1/catalog/metrics", "bearer", a.catalog, None),
        ("GET", "/api/v1/workspace", "bearer", a.workspace, None),
        ("POST", "/api/v1/workspace/scan", "bearer", a.scan, None),
        ("GET", "/api/v1/workspace/items", "bearer", a.items, None),
        ("GET", "/api/v1/artifact", "bearer", a.artifact, None),
        ("GET", "/api/v1/artifact/capabilities", "bearer", a.panels, None),
        ("GET", "/api/v1/artifact/sources", "bearer", a.sources, None),
        ("GET", "/api/v1/artifact/cases", "bearer", a.cases, None),
        ("GET", "/api/v1/artifact/requests", "bearer", a.requests, None),
        ("GET", "/api/v1/artifact/summary", "bearer", a.summary, None),
        ("GET", "/api/v1/artifact/distribution", "bearer", a.distribution, None),
        ("GET", "/api/v1/artifact/points", "bearer", a.points, None),
        ("GET", "/api/v1/artifact/samples", "bearer", a.samples, None),
        ("GET", "/api/v1/artifact/request", "bearer", a.request, None),
        ("GET", "/api/v1/artifact/request/chunks", "bearer", a.chunks, None),
        ("GET", "/api/v1/artifact/line", "bearer_or_ticket", a.line, long_s),
        ("GET", "/api/v1/artifact/evidence", "bearer", a.evidence, None),
        ("GET", "/api/v1/artifact/export/preview", "bearer", a.preview, None),
        ("GET", "/api/v1/artifact/export", "bearer_or_ticket", a.export, long_s),
        ("GET", "/api/v1/artifact/analysis", "bearer", a.analysis, None),
        ("GET", "/api/v1/report", "bearer", a.report, None),
    ]
    deadline = ctx.options.limits.deadline_s
    return [
        Route(
            method,
            path,
            auth,
            _guarded(handler),
            deadline if seconds is None else seconds,
        )
        for method, path, auth, handler, seconds in table
    ]


def _guarded(handler: Handler) -> Handler:
    """``handler`` with read failures mapped to §5.2 (module docstring)."""

    def handle(request: HttpRequest) -> HttpResponse:
        try:
            return handler(request)
        except NotResident:
            raise InvalidInputError(
                "unreadable",
                "a value this route needs is not held in memory",
                details={"reason": "not_resident"},
            ) from None
        except OSError:
            raise InvalidInputError(
                "unreadable",
                "the artifact could not be read",
                details={"reason": "os_error"},
            ) from None

    return handle


# --- Responses and parameters ---------------------------------------------------


def _json(document: Mapping[str, Any]) -> HttpResponse:
    return HttpResponse(200, JSON_TYPE, encode_json(document))


def _check_names(query: Query, allowed: Iterable[str]) -> None:
    names = set(allowed)
    for name in query:
        if name not in names:
            raise UsageError(
                "unknown_parameter",
                f"{name[:64]}: not a parameter here",
                field=name[:64],
                details={"allowed": sorted(names)},
            )


def _one(query: Query, name: str) -> str | None:
    values = list(query.get(name, ()))
    if len(values) > 1:
        raise UsageError("bad_parameter", f"{name}: given more than once", field=name)
    return values[0] if values else None


def _required(query: Query, name: str) -> str:
    value = _one(query, name)
    if value is None or value == "":
        raise UsageError("bad_parameter", f"{name}: required", field=name)
    return value


def _integer(query: Query, name: str, default: int | None) -> int | None:
    """An integer parameter; range checks are the callee's (C2, C4).

    At most one ``-`` and 18 ASCII digits, so ``--5`` or ``+5`` is a 400
    and never reaches ``int`` (Fable's P3-1: ``--5`` was a 500).
    """
    text = _one(query, name)
    if text is None:
        return default
    if _INTEGER.fullmatch(text) is None:
        raise UsageError("bad_parameter", f"{name}: an integer", field=name)
    return int(text)


def _need_integer(query: Query, name: str, default: int) -> int:
    value = _integer(query, name, default)
    return default if value is None else value


def _floats(text: str, name: str) -> list[float]:
    try:
        return [float(part) for part in text.split(",")]
    except ValueError:
        raise UsageError(
            "bad_parameter", f"{name}: comma-separated numbers", field=name
        ) from None


def _route_params(query: Query, names: Iterable[str]) -> str:
    """The route's own parameters, sorted and encoded, for a cache key."""
    pairs = sorted((n, v) for n in names for v in query.get(n, ()))
    return urlencode(pairs, quote_via=quote)


def _stamp(snapshot: IndexSnapshot) -> str:
    """What a same-generation republish can change in a document."""
    identity = snapshot.identity
    file = "-" if identity is None else f"{identity.size}:{identity.mtime_ns}"
    return "/".join(
        (
            snapshot.status,
            str(snapshot.failure),
            str(snapshot.growing),
            str(snapshot.bytes_read),
            str(snapshot.pending_tail_bytes),
            file,
            str(snapshot.problem_count),
        )
    )


def refuse_failed(snapshot: IndexSnapshot) -> None:
    """The §5.2 error for a failed index; nothing for any other state."""
    if snapshot.status != "failed":
        return
    failure = snapshot.failure or "internal"
    if failure not in _UNSUPPORTED:
        raise StudioError("internal_error", "indexing failed on an internal error")
    code, reason = _UNSUPPORTED[failure]
    raise InvalidInputError(
        code,
        _MESSAGES[code],
        field="artifact",
        details={} if reason is None else {"reason": reason},
    )


# --- The routes -----------------------------------------------------------------


class _Api:
    """The handlers, each a bound method over one context."""

    def __init__(self, ctx: StudioContext) -> None:
        self.ctx = ctx
        self.limits = ctx.options.limits

    # --- shared steps -----------------------------------------------------

    def _snapshot(self, query: Query) -> IndexSnapshot:
        return self.ctx.indexes.snapshot(_required(query, "artifact"))

    def _served(self, query: Query) -> IndexSnapshot:
        snapshot = self._snapshot(query)
        refuse_failed(snapshot)
        return snapshot

    def _selection(
        self, snapshot: IndexSnapshot, query: Query, route_params: Sequence[str]
    ) -> Selection:
        f = selection.parse_filter(
            query, origin_ns=snapshot.origin_ns, route_params=route_params
        )
        params = selection.canonical_params(f)
        key = cache_key(
            "selection", snapshot.path, snapshot.index_id, snapshot.generation, params
        )
        return self.ctx.cache.selection(key, lambda: selection.select(snapshot, f))

    def _cached(
        self,
        name: str,
        snapshot: IndexSnapshot,
        params: str,
        make: Callable[[], Mapping[str, Any]],
    ) -> HttpResponse:
        key = cache_key(
            name,
            snapshot.path,
            snapshot.index_id,
            snapshot.generation,
            f"{_stamp(snapshot)}|{params}",
        )
        document = self.ctx.cache.document(
            key, lambda: envelope(name, make(), snapshot)
        )
        return _json(document)

    def _analysis_state(self, snapshot: IndexSnapshot) -> AnalysisState:
        return self.ctx.analysis.get(snapshot.path, snapshot)

    # --- server and workspace ---------------------------------------------

    def server(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ())
        body = {
            "version": self.ctx.version,
            "api_version": API_VERSION,
            "instance_id": self.ctx.instance_id,
            "root": str(self.ctx.root.path),
            "limits": asdict(self.limits),
        }
        return _json(envelope("server", body))

    def capabilities(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ())
        body = {
            "jobs": dict(_JOBS),
            "analyzer": {
                "enabled": True,
                "max_bytes": self.limits.analysis_bytes,
                "timeout_s": self.limits.analysis_seconds,
            },
            "export_profiles": [export.PROFILE],
            "remote": False,
        }
        return _json(envelope("capabilities", body))

    def catalog(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ())
        return _json(envelope("metric_catalog", metric_catalog.metric_catalog()))

    def workspace(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ())
        return _json(envelope("workspace", self.ctx.scanner.state()))

    def scan(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ())
        _empty_body(request)
        self.ctx.scanner.scan()
        return _json(envelope("workspace", self.ctx.scanner.state()))

    def items(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        _check_names(query, ("kind", "q", "cursor", "limit"))
        body = self.ctx.scanner.items(
            kind=_one(query, "kind"),
            q=_one(query, "q"),
            cursor=_one(query, "cursor"),
            limit=_need_integer(query, "limit", DEFAULT_PAGE),
        )
        return _json(envelope("items", body))

    def report(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ("path",))
        rel = _required(request.query, "path")
        body = read_report(self.ctx.root, rel, max_bytes=self.limits.report_bytes)
        return _json(envelope("report", body))

    # --- one artifact -----------------------------------------------------

    def artifact(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ("artifact",))
        snapshot = self._snapshot(request.query)
        return self._cached("artifact", snapshot, "", lambda: artifact_body(snapshot))

    def panels(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ("artifact",))
        snapshot = self._snapshot(request.query)
        body = metric_catalog.panels(snapshot, self._analysis_state(snapshot))
        return _json(envelope("artifact_capabilities", body, snapshot))

    def sources(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ("artifact",))
        snapshot = self._snapshot(request.query)
        body = sources_body(snapshot, self._analysis_state(snapshot))
        return _json(envelope("sources", body, snapshot))

    def analysis(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ("artifact",))
        snapshot = self._snapshot(request.query)
        runner = self.ctx.analysis
        if isinstance(runner, AnalyzerRunner):
            # Raises NotReady with the whole document while pending or running.
            return _json(runner.respond(snapshot.path, snapshot))
        return _json(_plain_analysis(self._analysis_state(snapshot), snapshot))

    def cases(self, request: HttpRequest) -> HttpResponse:
        _check_names(request.query, ("artifact",))
        snapshot = self._served(request.query)
        count = len(summaries.case_ids(snapshot))
        if count > self.limits.max_cases:
            # A summary costs ~15 ms a case, so a case-per-request artifact
            # would outrun the route deadline (Fable's P3-2); the summary
            # route still answers any one case.
            raise InvalidInputError(
                "limit_exceeded",
                f"{count} cases, over the {self.limits.max_cases} this route "
                "summarises; ask the summary route for one case at a time",
                field="artifact",
                details={"count": count, "limit": self.limits.max_cases},
            )
        return self._cached("cases", snapshot, "", lambda: summaries.cases(snapshot))

    # --- selection routes ---------------------------------------------------

    def summary(self, request: HttpRequest) -> HttpResponse:
        snapshot = self._served(request.query)
        sel = self._selection(snapshot, request.query, ())
        return self._cached(
            "summary", snapshot, sel.params, lambda: summaries.summary(snapshot, sel)
        )

    def requests(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        snapshot = self._served(query)
        sel = self._selection(snapshot, query, _REQUEST_PARAMS)

        def make() -> dict[str, Any]:
            chosen = selection.page(
                snapshot,
                sel,
                sort=cast(SortKey, _one(query, "sort") or "artifact"),
                order=cast(Order, _one(query, "order") or "asc"),
                cursor=_one(query, "cursor"),
                limit=_need_integer(query, "limit", DEFAULT_PAGE),
            )
            return summaries.requests_page(snapshot, sel, chosen)

        params = f"{sel.params}|{_route_params(query, _REQUEST_PARAMS)}"
        return self._cached("requests", snapshot, params, make)

    def distribution(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        snapshot = self._served(query)
        sel = self._selection(snapshot, query, _DISTRIBUTION_PARAMS)
        metric = _required(query, "metric")
        edges_text = _one(query, "edges")
        bins = _integer(query, "bins", None)
        if edges_text is not None and bins is not None:
            raise UsageError("bad_parameter", "bins: not with edges", field="bins")
        edges = None if edges_text is None else _floats(edges_text, "edges")
        if edges is None and bins is None:
            bins = DEFAULT_BINS

        def make() -> dict[str, Any]:
            return summaries.distribution(
                snapshot, sel, metric=metric, edges=edges, bins=bins
            )

        params = f"{sel.params}|{_route_params(query, _DISTRIBUTION_PARAMS)}"
        return self._cached("distribution", snapshot, params, make)

    def points(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        snapshot = self._served(query)
        sel = self._selection(snapshot, query, _POINT_PARAMS)
        x = _one(query, "x") or "start"
        y = _one(query, "y") or "client.e2e"
        max_points = _need_integer(query, "max_points", DEFAULT_POINTS)
        keep = tuple(query.get("keep", ()))

        def make() -> dict[str, Any]:
            return summaries.points(
                snapshot, sel, x=x, y=y, max_points=max_points, keep=keep
            )

        params = f"{sel.params}|{_route_params(query, _POINT_PARAMS)}"
        return self._cached("points", snapshot, params, make)

    def samples(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        _check_names(query, ("artifact", "max_points", *_WINDOW_PARAMS))
        snapshot = self._served(query)
        window_query = {name: query[name] for name in _WINDOW_PARAMS if name in query}
        window = selection.parse_filter(
            window_query, origin_ns=snapshot.origin_ns
        ).window
        max_points = _need_integer(query, "max_points", DEFAULT_SAMPLES)

        def make() -> dict[str, Any]:
            return summaries.samples(snapshot, window=window, max_points=max_points)

        params = _route_params(query, ("max_points", *_WINDOW_PARAMS))
        return self._cached("samples", snapshot, params, make)

    def request(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        snapshot = self._served(query)
        sel = self._selection(snapshot, query, _DETAIL_PARAMS)
        row = selection.request_row(
            snapshot, _required(query, "request_id"), _integer(query, "line", None)
        )
        body = summaries.request_detail(snapshot, row, sel)
        return _json(envelope("request", body, snapshot))

    def chunks(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        _check_names(query, ("artifact", "request_id", "line", "offset", "limit"))
        snapshot = self._served(query)
        row = selection.request_row(
            snapshot, _required(query, "request_id"), _integer(query, "line", None)
        )
        body = summaries.chunks(
            snapshot,
            row,
            offset=_need_integer(query, "offset", 0),
            limit=_need_integer(query, "limit", DEFAULT_CHUNKS),
        )
        return _json(envelope("chunks", body, snapshot))

    # --- export ---------------------------------------------------------------

    def preview(self, request: HttpRequest) -> HttpResponse:
        snapshot = self._served(request.query)
        sel = self._selection(snapshot, request.query, ())

        def make() -> dict[str, Any]:
            block = summaries.selection_block(sel, snapshot)
            return export.preview(snapshot, sel, selection_block=block)

        return self._cached("export_preview", snapshot, sel.params, make)

    def export(self, request: HttpRequest) -> HttpResponse:
        snapshot = self._served(request.query)
        sel = self._selection(snapshot, request.query, ())
        headers = export.download_headers(snapshot, sel)
        chunks = export.stream(self.ctx.indexes, snapshot, sel)
        # The first chunk is read before the headers go out, so a file
        # already gone or changed is an error document, not a cut stream.
        first = next(chunks, b"")
        return HttpResponse(200, export.CONTENT_TYPE, _primed(first, chunks), headers)

    # --- raw lines --------------------------------------------------------------

    def line(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        _check_names(query, ("artifact", "line", "sha256"))
        snapshot = self._snapshot(query)
        ref = _known_ref(snapshot, _line_number(query), self.limits.line_bytes)
        expected = _one(query, "sha256")
        if expected is not None and expected != ref.sha256:
            raise ConflictError(
                "stale_line",
                "the line's SHA-256 is not the one given",
                field="sha256",
                details={"line": ref.line, "reason": "sha256"},
            )
        data = self.ctx.indexes.read_line(snapshot.path, ref)
        headers = (
            ("X-Stormlog-Line", str(ref.line)),
            ("X-Stormlog-Line-Sha256", ref.sha256),
        )
        return HttpResponse(200, LINE_TYPE, data, headers)

    def evidence(self, request: HttpRequest) -> HttpResponse:
        query = request.query
        _check_names(query, ("artifact", "line", "record_id", "sha256"))
        snapshot = self._snapshot(query)
        line = _line_number(query)
        wanted = {
            "line": line,
            "record_id": _one(query, "record_id"),
            "sha256": _one(query, "sha256"),
        }
        found = _resolve(self.ctx, snapshot, line, wanted)
        return _json(envelope("evidence", {"query": wanted, **found}, snapshot))


def _empty_body(request: HttpRequest) -> None:
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise UsageError("bad_parameter", "the body is not JSON") from None
    if body != {}:
        raise UsageError("unknown_parameter", "the body must be {}")


def _primed(first: bytes, rest: Iterator[bytes]) -> Iterator[bytes]:
    try:
        if first:
            yield first
        yield from rest
    finally:
        close = getattr(rest, "close", None)
        if callable(close):
            close()


def _plain_analysis(state: AnalysisState, snapshot: IndexSnapshot) -> dict[str, Any]:
    """The analysis document from a bare :class:`AnalysisCache` (a stand-in)."""
    body = {
        "status": state.status,
        "reason": state.reason,
        "exit_code": state.exit_code,
        "analysed_bytes": state.analysed_bytes,
        "report": None if state.report is None else dict(state.report),
    }
    return envelope("analysis", body, snapshot)


# --- The artifact document (C4's, §5.4) -------------------------------------------


def artifact_body(snapshot: IndexSnapshot) -> dict[str, Any]:
    """The ``stormlog.studio.artifact`` body for one snapshot.

    ``problems.by_code`` is the index's ``problem_counts``, so the banner
    can say how many client request lines another schema version kept out
    (``unsupported_version``, Fable's F-8); the artifact block's ``cut``
    names the first line a budget left unindexed. ``warnings`` says both
    in words.
    """
    return {
        "session": _session(snapshot.session),
        "index": _index_block(snapshot),
        "inventory": _inventory(snapshot),
        "problems": {
            "count": snapshot.problem_count,
            "by_code": dict(sorted(snapshot.problem_counts.items())),
            "first": [_problem(p) for p in snapshot.problems],
        },
        "cases": _case_rows(snapshot),
        "warnings": _warnings(snapshot),
    }


def _session(session: SessionInfo | None) -> dict[str, Any] | None:
    if session is None:
        return None
    return {
        "session_id": session.session_id,
        "status": session.status,
        "started_at_ns": session.started_at_ns,
        "ended_at_ns": session.ended_at_ns,
        "terminal": session.terminal,
        "config": dict(session.config),
    }


def _index_block(snapshot: IndexSnapshot) -> dict[str, Any]:
    size = None if snapshot.identity is None else snapshot.identity.size
    progress = None if not size else min(1.0, snapshot.bytes_read / size)
    return {
        "status": snapshot.status,
        "failure": snapshot.failure,
        "bytes_read": snapshot.bytes_read,
        "size": size,
        "progress": progress,
        "pending_tail_bytes": snapshot.pending_tail_bytes,
        "prefix_sha256": snapshot.prefix_sha256,
        "requests": len(snapshot.requests),
        "samples": len(snapshot.samples),
    }


def _inventory(snapshot: IndexSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "event_type": event_type,
            "total": sum(versions.values()),
            "by_version": {str(v): n for v, n in sorted(versions.items())},
        }
        for event_type, versions in snapshot.inventory.items()
    ]


def _problem(problem: LineProblem) -> dict[str, Any]:
    ref = problem.ref
    return {
        "line": ref.line,
        "offset": ref.offset,
        "length": ref.length,
        "sha256": ref.sha256,
        "code": problem.code,
        "detail": problem.detail,
    }


def _case_rows(snapshot: IndexSnapshot) -> list[dict[str, Any]]:
    """Planned and observed measured cases, with counts by status."""
    if snapshot.status == "failed":
        return []
    facts = selection.case_facts(snapshot)
    statuses = snapshot.column("status") if len(snapshot.table) else ()
    planned = set(facts.planned)
    rows = []
    for case_id in sorted(planned | set(facts.by_case)):
        members = facts.by_case.get(case_id, ())
        counted = Counter(statuses[i] for i in members)
        rows.append(
            {
                "case_id": case_id,
                "planned": case_id in planned,
                "observed": case_id in facts.by_case,
                "measured_requests": len(members),
                "by_status": {
                    s: counted[s] for s in REQUEST_STATUSES if counted.get(s)
                },
            }
        )
    return rows


def _warnings(snapshot: IndexSnapshot) -> list[dict[str, str]]:
    """The banners: what the numbers leave out, said once each."""
    found: list[dict[str, str]] = []
    if snapshot.cut is not None:
        found.append(_cut_warning(snapshot.cut))
    excluded = snapshot.problem_counts.get("unsupported_version", 0)
    if excluded:
        found.append(
            {
                "code": "unsupported_version",
                "message": f"{excluded} client request line(s) at another schema "
                "version are not rows; the numbers leave them out",
            }
        )
    others = snapshot.problem_count - excluded
    if others > 0:
        found.append(
            {"code": "problem_lines", "message": f"{others} line(s) could not be read"}
        )
    found.extend(_state_warnings(snapshot))
    return found


def _cut_warning(cut: LineRef) -> dict[str, str]:
    return {
        "code": "index_cut",
        "message": f"the index stopped at line {cut.line} (byte {cut.offset}) at "
        "its budget; the numbers are about the lines before it",
    }


def _state_warnings(snapshot: IndexSnapshot) -> list[dict[str, str]]:
    found = []
    if snapshot.growing:
        found.append({"code": "growing", "message": "the file is still being written"})
    elif snapshot.session is not None and not snapshot.session.terminal:
        found.append(
            {"code": "no_terminal_record", "message": "the run has no terminal record"}
        )
    if not snapshot.gaps_resident:
        found.append(
            {"code": "gaps_not_resident", "message": "chunk gaps are read per request"}
        )
    if not snapshot.error_messages_resident:
        found.append(
            {
                "code": "error_messages_not_resident",
                "message": "error messages are read per request and not searched",
            }
        )
    return found


# --- Sources (C4's, §5.4) -----------------------------------------------------------


def sources_body(snapshot: IndexSnapshot, analysis: AnalysisState) -> dict[str, Any]:
    """The ``stormlog.studio.sources`` body: what the artifact holds."""
    capabilities = metric_catalog.panels(snapshot, analysis)
    return {
        "inventory": _inventory(snapshot),
        "kept": [
            {
                "event_type": event_type,
                "written": sum(snapshot.inventory.get(event_type, {}).values()),
                "kept": len(records),
                "records": [dict(record) for record in records],
            }
            for event_type, records in sorted(snapshot.kept.items())
        ],
        "panels": capabilities["panels"],
        "analysis": {"status": analysis.status, "reason": analysis.reason},
        "observers": capabilities["observers"],
        "observers_source": capabilities["observers_source"],
    }


# --- Raw lines and evidence -------------------------------------------------------


def _line_number(query: Query) -> int:
    line = _integer(query, "line", None)
    if line is None:
        raise UsageError("bad_parameter", "line: required", field="line")
    if line < 0:
        raise UsageError("bad_parameter", "line: an integer >= 0", field="line")
    return line


def find_ref(snapshot: IndexSnapshot, line: int, *, line_bytes: int) -> LineRef | None:
    """The reference the index holds for ``line``, or None.

    A client request row's line is always found (by bisection over the
    rows' references, which are in file order); another line only when the
    index kept its reference: the first 1,000 of each event type, a
    problem line, a request's dispatch or first-content line, or the cut.

    A line longer than ``line_bytes`` is an ``oversized`` problem the
    reader never parsed; its reference is refused with 422 ``unreadable``
    (``details.reason`` ``oversized``) rather than read whole (Fable's
    P2-1).
    """
    ref = _held_ref(snapshot, line)
    if ref is not None:
        refuse_oversized(ref, line_bytes)
    return ref


def _held_ref(snapshot: IndexSnapshot, line: int) -> LineRef | None:
    table = columnar(snapshot.table)
    refs: Sequence[LineRef] = (
        table.line_refs() if table is not None else [r.ref for r in snapshot.requests]
    )
    at = bisect_left(refs, line, key=_line_of)
    if at < len(refs) and refs[at].line == line:
        return refs[at]
    for ref in _other_refs(snapshot):
        if ref is not None and ref.line == line:
            return ref
    return None


def _line_of(ref: LineRef) -> int:
    return ref.line


def _other_refs(snapshot: IndexSnapshot) -> Iterable[LineRef | None]:
    return chain(
        chain.from_iterable(snapshot.refs.values()),
        (problem.ref for problem in snapshot.problems),
        chain.from_iterable(snapshot.request_links.values()),
        (snapshot.cut,),
    )


def _known_ref(snapshot: IndexSnapshot, line: int, line_bytes: int) -> LineRef:
    ref = find_ref(snapshot, line, line_bytes=line_bytes)
    if ref is not None:
        return ref
    count = snapshot.line_count
    past = count is not None and line >= count
    raise NotFoundError(
        "line_out_of_range",
        f"line {line} is past the end" if past else f"line {line} is not indexed",
        field="line",
        details={"line": line, "reason": "past_end" if past else "not_indexed"},
    )


def _resolve(
    ctx: StudioContext, snapshot: IndexSnapshot, line: int, wanted: Mapping[str, Any]
) -> dict[str, Any]:
    """``resolved`` with the line found, or ``unresolved`` with a reason."""
    ref = find_ref(snapshot, line, line_bytes=ctx.options.limits.line_bytes)
    if ref is None:
        return _unresolved("line_out_of_range", None)
    if wanted["sha256"] is not None and wanted["sha256"] != ref.sha256:
        return _unresolved("hash_mismatch", ref)
    try:
        data = ctx.indexes.read_line(snapshot.path, ref)
    except ConflictError:
        return _unresolved("hash_mismatch", ref)
    record_id = _record_id(snapshot, ref, data)
    if wanted["record_id"] is not None and wanted["record_id"] != record_id:
        return _unresolved("record_id_mismatch", ref, record_id)
    return {"state": "resolved", "reason": None, "found": _found(ref, record_id)}


def _unresolved(
    reason: str, ref: LineRef | None, record_id: str | None = None
) -> dict[str, Any]:
    found = None if ref is None else _found(ref, record_id)
    return {"state": "unresolved", "reason": reason, "found": found}


def _found(ref: LineRef, record_id: str | None) -> dict[str, Any]:
    return {
        "line": ref.line,
        "offset": ref.offset,
        "length": ref.length,
        "sha256": ref.sha256,
        "record_id": record_id,
    }


def _record_id(snapshot: IndexSnapshot, ref: LineRef, data: bytes) -> str | None:
    """A request row's decoded record ID, else the raw line's (§5.8)."""
    table = columnar(snapshot.table)
    refs: Sequence[LineRef] = (
        table.line_refs() if table is not None else [r.ref for r in snapshot.requests]
    )
    at = bisect_left(refs, ref.line, key=_line_of)
    if at < len(refs) and refs[at].line == ref.line:
        return snapshot.requests[at].record_id
    try:
        raw = json.loads(data)
    except (ValueError, RecursionError):
        return None
    return raw_record_id(raw) if isinstance(raw, dict) else None
