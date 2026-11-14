"""Every slice-1 route over a real StudioServer and the real composition.

The server here is ``app.compose`` and ``app.serve``: C3's httpd, security
and static files, C1's index, C2's selections and summaries, and C4's
workspace, export, analyzer and caches, over C5's standard artifacts. Each
JSON response is checked against its schema and each error against the
error schema (design §5.2, §9.2). The end-to-end smoke tests start
``stormlog studio <fixture>`` as a subprocess and drive it over HTTP the
way the frontend does: the launch URL's bootstrap, a session, then the
artifact, summary and requests documents.
"""

from __future__ import annotations

import base64
import dataclasses
import http.client
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer.cli import main as infer_main
from stormlog.infer.diagnosis_inputs import MAX_NESTING
from stormlog.report import build_report
from stormlog.studio import api, app, selection, summaries
from stormlog.studio.contracts import (
    IndexSnapshot,
    LaunchOptions,
    Limits,
    RequestFilter,
    StudioContext,
    publish_hook,
)
from stormlog.studio.httpd import StudioServer
from tests.studio.artifacts import GrowingCopy
from tests.studio.conftest import StandardArtifacts
from tests.studio.test_httpd import Reply
from tests.studio.test_selection import (
    ORIGIN,
    closed_case,
    request,
    snapshot_from_records,
)
from tests.test_infer_request_columns import HUGE_FIELDS

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"
REPO = Path(__file__).resolve().parents[2]
# The artifacts the shared server serves, by root-relative path.
SERVED = (
    "mixed",
    "samples",
    "truncated_mid_line",
    "unsupported_version",
    "audio",
    "not_an_inference_artifact",
    "multi_session",
)
REPORT = "reports/diagnosis.json"
MIXED = "mixed.jsonl"
# Telemetry JSONL: no workspace kind, so the scan does not list it.
NOT_LISTED = "not_an_inference_artifact.jsonl"
WAIT_S = 60.0
# 11,000 "[": json.loads raises RecursionError on 3.10 to 3.13 (about 10,000
# levels on 3.12), yet the request stays under the 16 KiB head limit.
CURSOR_BOMB = base64.urlsafe_b64encode(b"[" * 11_000).decode("ascii").rstrip("=")

# Every route Studio serves, with its auth (§5.4): C3's four, then C4's.
ROUTE_TABLE: dict[tuple[str, str], str] = {
    ("POST", "/api/v1/session"): "public",
    ("GET", "/api/v1/health"): "public",
    ("POST", "/api/v1/tickets"): "bearer",
    ("POST", "/api/v1/admin/bootstrap"): "admin",
    ("GET", "/api/v1/server"): "bearer",
    ("GET", "/api/v1/capabilities"): "bearer",
    ("GET", "/api/v1/catalog/metrics"): "bearer",
    ("GET", "/api/v1/workspace"): "bearer",
    ("POST", "/api/v1/workspace/scan"): "bearer",
    ("GET", "/api/v1/workspace/items"): "bearer",
    ("GET", "/api/v1/artifact"): "bearer",
    ("GET", "/api/v1/artifact/capabilities"): "bearer",
    ("GET", "/api/v1/artifact/sources"): "bearer",
    ("GET", "/api/v1/artifact/cases"): "bearer",
    ("GET", "/api/v1/artifact/requests"): "bearer",
    ("GET", "/api/v1/artifact/summary"): "bearer",
    ("GET", "/api/v1/artifact/distribution"): "bearer",
    ("GET", "/api/v1/artifact/points"): "bearer",
    ("GET", "/api/v1/artifact/samples"): "bearer",
    ("GET", "/api/v1/artifact/request"): "bearer",
    ("GET", "/api/v1/artifact/request/chunks"): "bearer",
    ("GET", "/api/v1/artifact/line"): "bearer_or_ticket",
    ("GET", "/api/v1/artifact/evidence"): "bearer",
    ("GET", "/api/v1/artifact/export/preview"): "bearer",
    ("GET", "/api/v1/artifact/export"): "bearer_or_ticket",
    ("GET", "/api/v1/artifact/analysis"): "bearer",
    ("GET", "/api/v1/report"): "bearer",
}


# --- Schemas ---------------------------------------------------------------------


def _registry() -> Registry:
    registry: Registry = Registry()
    for path in SCHEMA_DIR.glob("studio_*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(schema["$id"], resource)
        registry = registry.with_resource(path.name, resource)
    return registry


REGISTRY = _registry()


def schema_errors(document: Any, name: str) -> list[str]:
    path = SCHEMA_DIR / f"studio_{name}_v1.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, registry=REGISTRY)
    return [
        f"{list(error.absolute_path)}: {error.message[:300]}"
        for error in validator.iter_errors(document)
    ]


def assert_document(reply: Reply, name: str, status: int = 200) -> dict[str, Any]:
    assert reply.status == status, reply.body[:500]
    assert reply.headers["content-type"] == api.JSON_TYPE
    assert reply.headers["cache-control"] == "no-store"
    document: dict[str, Any] = reply.json()
    assert document["format"] == f"stormlog.studio.{name}"
    assert schema_errors(document, name) == []
    return document


def assert_error(reply: Reply, status: int, code: str) -> dict[str, Any]:
    assert reply.status == status, reply.body[:500]
    document: dict[str, Any] = reply.json()
    assert schema_errors(document, "error") == []
    error: dict[str, Any] = document["error"]
    assert (error["http_status"], error["code"]) == (status, code), error
    return error


# --- A client ----------------------------------------------------------------------


def fetch(
    port: int,
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
) -> Reply:
    """One request to 127.0.0.1:``port``, never through a proxy."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    sent: dict[str, str] = {"Host": f"127.0.0.1:{port}"}
    data = None
    if token is not None:
        sent["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        sent["Content-Type"] = "application/json"
        sent["Origin"] = f"http://127.0.0.1:{port}"
    sent.update(headers or {})
    try:
        connection.request(method, path, body=data, headers=sent)
        response = connection.getresponse()
        payload = response.read()
        lowered = {k.lower(): v for k, v in response.getheaders()}
        return Reply(response.status, lowered, payload)
    finally:
        connection.close()


def path_of(route: str, **params: Any) -> str:
    pairs: list[tuple[str, str]] = []
    for name, value in params.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        pairs += [(name, str(v)) for v in values]
    return route + ("?" + urlencode(pairs, quote_via=quote) if pairs else "")


@dataclass
class Served:
    """A real Studio: the composed context, its server and a session."""

    ctx: StudioContext
    server: StudioServer
    token: str
    root: Path

    @property
    def port(self) -> int:
        return self.server.bound_port

    def get(self, route: str, *, token: Any = "", **params: Any) -> Reply:
        chosen = self.token if token == "" else token
        return fetch(self.port, "GET", path_of(route, **params), token=chosen)

    def post(self, route: str, body: Any, *, token: str | None = "") -> Reply:
        chosen = self.token if token == "" else token
        return fetch(self.port, "POST", route, token=chosen, body=body)

    def ready(self, rel: str) -> IndexSnapshot:
        snapshot = self.ctx.indexes.wait(rel, WAIT_S)
        assert snapshot.status != "building", rel
        return snapshot


def start(root: Path, state: Path, limits: Limits | None = None) -> Served:
    options = LaunchOptions(
        root=root,
        open_path=None,
        host="127.0.0.1",
        port=0,
        open_browser=False,
        state_dir=state,
        log_requests=False,
        debug=False,
        limits=limits or Limits(),
    )
    ctx, routes = app.compose(options)
    server = app.serve(ctx, routes)
    bootstrap = ctx.tokens.issue_bootstrap()
    reply = fetch(
        server.bound_port, "POST", "/api/v1/session", body={"bootstrap": bootstrap}
    )
    assert reply.status == 200, reply.body
    return Served(ctx, server, reply.json()["session_token"], root)


def stop(served: Served) -> None:
    served.server.stop()
    app.shutdown(served.ctx)


def write_report(path: Path) -> Path:
    report = build_report(
        report_kind="inference_diagnosis",
        tool_name="stormlog",
        command="infer diagnose",
        exit_code=0,
        summary="ok",
        generated_at_utc="2026-11-13T00:00:00Z",
        tool_version="0.0.0",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def served(
    studio_artifacts: StandardArtifacts, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Served]:
    work = tmp_path_factory.mktemp("api-routes")
    root = work / "root"
    root.mkdir()
    for name in SERVED:
        shutil.copy2(studio_artifacts[name], root / f"{name}.jsonl")
    write_report(root / REPORT)
    state = work / "state"
    state.mkdir(mode=0o700)
    running = start(root, state)
    try:
        yield running
    finally:
        stop(running)


@pytest.fixture
def private(tmp_path: Path) -> Iterator[Callable[..., Served]]:
    """A Studio of a test's own, for tests that change files or limits."""
    started: list[Served] = []

    def make(limits: Limits | None = None) -> Served:
        root = tmp_path / "root"
        root.mkdir(exist_ok=True)
        state = tmp_path / "state"
        state.mkdir(mode=0o700, exist_ok=True)
        started.append(start(root, state, limits))
        return started[-1]

    yield make
    for running in started:
        stop(running)


def first_row(served: Served, rel: str = MIXED, **params: Any) -> dict[str, Any]:
    served.ready(rel)
    reply = served.get("/api/v1/artifact/requests", artifact=rel, limit=1, **params)
    rows: list[dict[str, Any]] = reply.json()["rows"]
    return rows[0]


def poll(check: Callable[[], bool], timeout_s: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.2)


# --- The composition ------------------------------------------------------------


def test_the_routes_are_section_5_4_with_their_auth(served: Served) -> None:
    routes = app.routes(served.ctx)
    assert {(r.method, r.path): r.auth for r in routes} == ROUTE_TABLE
    deadlines = {r.path: r.deadline_s for r in routes}
    assert deadlines["/api/v1/artifact/export"] == Limits().long_deadline_s
    assert deadlines["/api/v1/artifact/line"] == Limits().long_deadline_s
    assert deadlines["/api/v1/artifact/summary"] == Limits().deadline_s


def test_the_context_is_built_from_the_real_components(served: Served) -> None:
    ctx = served.ctx
    names = {
        "root": "RootDirectory",
        "tokens": "SessionTokens",
        "indexes": "ArtifactIndexes",
        "scanner": "ArtifactScanner",
        "analysis": "AnalyzerRunner",
        "cache": "Documents",
    }
    for field, name in names.items():
        assert type(getattr(ctx, field)).__name__ == name
    assert re.fullmatch(r"[0-9a-f]{32}", ctx.instance_id)
    # The analyzer works in this instance's own directory (fix a8329d4).
    analysis_dir = ctx.options.state_dir / "analysis" / ctx.instance_id
    assert getattr(ctx.analysis, "_dir") == analysis_dir
    assert publish_hook(ctx.indexes) is not None


def test_warm_uses_selection_warm_when_c2_provides_it(
    monkeypatch: pytest.MonkeyPatch, served: Served
) -> None:
    snapshot = served.ready(MIXED)
    seen: list[str] = []
    monkeypatch.setattr(selection, "case_facts", lambda s: seen.append("facts"))
    monkeypatch.delattr(selection, "warm", raising=False)
    app.warm(snapshot)
    assert seen == ["facts"]
    monkeypatch.setattr(selection, "warm", lambda s: seen.append("warm"), raising=False)
    app.warm(snapshot)
    assert seen == ["facts", "warm"]
    app.warm(dataclasses.replace(snapshot, status="failed", failure="unreadable"))
    assert seen == ["facts", "warm"]


def test_a_publish_warms_the_case_facts_on_the_builder_pool(
    monkeypatch: pytest.MonkeyPatch, private: Callable[..., Served]
) -> None:
    warmed: list[tuple[str, str]] = []
    real = app.warm

    def spy(snapshot: IndexSnapshot) -> None:
        warmed.append((snapshot.path, threading.current_thread().name))
        real(snapshot)

    monkeypatch.setattr(app, "warm", spy)
    studio = private()
    (studio.root / "run.jsonl").write_bytes(b'{"event_type": "infer.session"}\n')
    assert app.add_warm_listener(studio.ctx.indexes)
    studio.ready("run.jsonl")
    poll(lambda: bool(warmed))
    assert warmed[0][0] == "run.jsonl"
    assert warmed[0][1].startswith("stormlog-studio-builder")


def test_shutdown_closes_every_component(private: Callable[..., Served]) -> None:
    studio = private()
    (studio.root / "run.jsonl").write_bytes(b'{"event_type": "infer.session"}\n')
    studio.ready("run.jsonl")
    studio.server.stop()
    app.shutdown(studio.ctx)
    with pytest.raises(RuntimeError):
        studio.ctx.pool.submit(print)
    analysis_dir = studio.ctx.options.state_dir / "analysis" / studio.ctx.instance_id
    assert not analysis_dir.exists()


# --- Auth on every route ----------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "route"),
    [key for key, auth in ROUTE_TABLE.items() if auth != "public"],
)
def test_every_route_refuses_a_missing_or_wrong_token(
    served: Served, method: str, route: str
) -> None:
    admin = ROUTE_TABLE[(method, route)] == "admin"
    # An admin request never carries an Origin (a browser's is refused).
    plain = {"Content-Type": "application/json"}
    tokens: list[str | None] = [None, "not-a-token"]
    if admin:
        tokens.append(served.token)  # a session token is not the admin token
    for token in tokens:
        if admin:
            reply = fetch(served.port, method, route, token=token, headers=plain)
        else:
            body: dict[str, Any] | None = {} if method == "POST" else None
            target = path_of(route, artifact=MIXED)
            reply = fetch(served.port, method, target, token=token, body=body)
        assert_error(reply, 401, "unauthorized")


def test_a_ticket_is_refused_on_a_bearer_route(served: Served) -> None:
    query = f"artifact={MIXED}"
    reply = served.post(
        "/api/v1/tickets", {"route": "/api/v1/artifact/summary", "query": query}
    )
    ticket = assert_document(reply, "ticket")["ticket"]
    reply = served.get(
        "/api/v1/artifact/summary", token=None, artifact=MIXED, ticket=ticket
    )
    assert_error(reply, 401, "unauthorized")


# --- A valid document from every JSON route ---------------------------------------


def _cases(row: Mapping[str, Any]) -> dict[str, tuple[str, dict[str, Any], str]]:
    rid, line = row["request_id"], row["line"]
    a = {"artifact": MIXED}
    return {
        "server": ("/api/v1/server", {}, "server"),
        "capabilities": ("/api/v1/capabilities", {}, "capabilities"),
        "catalog": ("/api/v1/catalog/metrics", {}, "metric_catalog"),
        "workspace": ("/api/v1/workspace", {}, "workspace"),
        "items": ("/api/v1/workspace/items", {"limit": 2}, "items"),
        "items_kind": ("/api/v1/workspace/items", {"kind": "report"}, "items"),
        "artifact": ("/api/v1/artifact", a, "artifact"),
        "artifact_capabilities": (
            "/api/v1/artifact/capabilities",
            a,
            "artifact_capabilities",
        ),
        "sources": ("/api/v1/artifact/sources", a, "sources"),
        "cases": ("/api/v1/artifact/cases", a, "cases"),
        "requests": ("/api/v1/artifact/requests", {**a, "limit": 5}, "requests"),
        "requests_sorted": (
            "/api/v1/artifact/requests",
            {**a, "sort": "client.e2e", "order": "desc", "status": "ok"},
            "requests",
        ),
        "summary": ("/api/v1/artifact/summary", a, "summary"),
        "summary_case": (
            "/api/v1/artifact/summary",
            {**a, "case": row["case_id"]},
            "summary",
        ),
        "distribution": (
            "/api/v1/artifact/distribution",
            {**a, "metric": "client.e2e", "bins": 10},
            "distribution",
        ),
        "distribution_edges": (
            "/api/v1/artifact/distribution",
            {**a, "metric": "client.e2e", "edges": "0,10,100,10000"},
            "distribution",
        ),
        "points": ("/api/v1/artifact/points", {**a, "y": "client.ttft"}, "points"),
        "samples": (
            "/api/v1/artifact/samples",
            {"artifact": "samples.jsonl"},
            "samples",
        ),
        "samples_window": (
            "/api/v1/artifact/samples",
            {"artifact": "samples.jsonl", "window_ms": "0,500", "max_points": 30},
            "samples",
        ),
        "request": ("/api/v1/artifact/request", {**a, "request_id": rid}, "request"),
        "request_line": (
            "/api/v1/artifact/request",
            {**a, "request_id": rid, "line": line, "status": "error"},
            "request",
        ),
        "chunks": (
            "/api/v1/artifact/request/chunks",
            {**a, "request_id": rid, "offset": 0, "limit": 3},
            "chunks",
        ),
        "evidence": ("/api/v1/artifact/evidence", {**a, "line": line}, "evidence"),
        "export_preview": ("/api/v1/artifact/export/preview", a, "export_preview"),
        "report": ("/api/v1/report", {"path": REPORT}, "report"),
        "artifact_audio": ("/api/v1/artifact", {"artifact": "audio.jsonl"}, "artifact"),
        "capabilities_audio": (
            "/api/v1/artifact/capabilities",
            {"artifact": "audio.jsonl"},
            "artifact_capabilities",
        ),
        "artifact_truncated": (
            "/api/v1/artifact",
            {"artifact": "truncated_mid_line.jsonl"},
            "artifact",
        ),
    }


DOCUMENT_CASES = sorted(_cases({"request_id": "", "line": 0, "case_id": ""}))


@pytest.mark.parametrize("case", DOCUMENT_CASES)
def test_every_json_route_answers_a_valid_document(served: Served, case: str) -> None:
    for rel in ("audio.jsonl", "truncated_mid_line.jsonl", "samples.jsonl"):
        served.ready(rel)
    served.ctx.scanner.wait(WAIT_S)
    route, params, name = _cases(first_row(served))[case]
    document = assert_document(served.get(route, **params), name)
    if "artifact" in params and name not in ("report",):
        assert document["artifact"]["path"] == params["artifact"]


def test_the_scan_route_answers_200_with_the_workspace(served: Served) -> None:
    assert served.ctx.scanner.wait(WAIT_S)
    document = assert_document(served.post("/api/v1/workspace/scan", {}), "workspace")
    assert document["scan"]["running"] in (True, False)
    assert served.ctx.scanner.wait(WAIT_S)
    items = assert_document(served.get("/api/v1/workspace/items", limit=100), "items")
    paths = {item["path"] for item in items["items"]}
    assert {f"{name}.jsonl" for name in SERVED} - {NOT_LISTED} <= paths
    assert REPORT in paths


def test_the_selection_routes_agree_on_one_query(served: Served) -> None:
    served.ready(MIXED)
    query = {"artifact": MIXED, "status": ["ok", "error"], "phase": "measured"}
    summary = served.get("/api/v1/artifact/summary", **query).json()
    requests = served.get("/api/v1/artifact/requests", **query).json()
    preview = served.get("/api/v1/artifact/export/preview", **query).json()
    distribution = served.get(
        "/api/v1/artifact/distribution", metric="client.e2e", **query
    ).json()
    blocks = [doc["selection"] for doc in (summary, requests, preview, distribution)]
    assert all(block == blocks[0] for block in blocks)
    size = blocks[0]["size"]
    assert requests["total"] == preview["count"] == size
    export = served.get("/api/v1/artifact/export", **query)
    assert export.headers["x-stormlog-export-count"] == str(size)
    assert export.body.count(b"\n") == size


def test_the_artifact_document_counts_cases_and_problems(served: Served) -> None:
    served.ready(MIXED)
    document = served.get("/api/v1/artifact", artifact=MIXED).json()
    summary = served.get("/api/v1/artifact/summary", artifact=MIXED).json()
    measured = sum(case["measured_requests"] for case in document["cases"])
    assert measured == summary["selection"]["size"]
    assert document["problems"] == {"count": 0, "by_code": {}, "first": []}
    assert document["index"]["progress"] == 1.0
    assert document["artifact"]["cut"] is None


def test_the_cases_route_refuses_past_max_cases(
    served: Served,
    private: Callable[..., Served],
    studio_artifacts: StandardArtifacts,
) -> None:
    """Fable's P3-2: a case per request made /artifact/cases outrun its
    deadline; past ``max_cases`` it answers 422 with the count at once."""
    studio = private(dataclasses.replace(Limits(), max_cases=3))
    shutil.copy2(studio_artifacts["mixed"], studio.root / MIXED)
    studio.ready(MIXED)
    count = len(_artifact(studio, MIXED)["cases"])
    assert count == 4
    error = assert_error(
        studio.get("/api/v1/artifact/cases", artifact=MIXED), 422, "limit_exceeded"
    )
    assert error["details"] == {"count": 4, "limit": 3}
    assert error["field"] == "artifact"
    server = assert_document(studio.get("/api/v1/server"), "server")
    assert server["limits"]["max_cases"] == 3
    served.ready(MIXED)
    reply = served.get("/api/v1/artifact/cases", artifact=MIXED)  # 200 by default
    assert len(assert_document(reply, "cases")["cases"]) == 4


def test_unsupported_version_lines_are_counted_for_the_banner(served: Served) -> None:
    """F-8: the line kept out of every number is named, by code and in words."""
    served.ready("unsupported_version.jsonl")
    reply = served.get("/api/v1/artifact", artifact="unsupported_version.jsonl")
    document = assert_document(reply, "artifact")
    assert document["problems"]["by_code"] == {"unsupported_version": 1}
    codes = [warning["code"] for warning in document["warnings"]]
    assert "unsupported_version" in codes
    assert document["artifact"]["cut"] is None


# --- Errors --------------------------------------------------------------------------


ERROR_CASES: dict[str, tuple[str, dict[str, Any], int, str]] = {
    "unknown_parameter": (
        "/api/v1/artifact/summary",
        {"statuss": "ok"},
        400,
        "unknown_parameter",
    ),
    "unknown_on_plain_route": (
        "/api/v1/artifact",
        {"case": "x"},
        400,
        "unknown_parameter",
    ),
    "bad_filter_value": (
        "/api/v1/artifact/summary",
        {"status": "eror"},
        400,
        "bad_parameter",
    ),
    "window_ms_too_precise": (
        "/api/v1/artifact/requests",
        {"window_ms": "0.1234567,1"},
        400,
        "bad_parameter",
    ),
    "limit_exceeded": (
        "/api/v1/artifact/requests",
        {"limit": 1001},
        400,
        "limit_exceeded",
    ),
    "bad_integer": (
        "/api/v1/artifact/requests",
        {"limit": "ten"},
        400,
        "bad_parameter",
    ),
    "bad_sort": ("/api/v1/artifact/requests", {"sort": "speed"}, 400, "bad_parameter"),
    "bad_cursor": ("/api/v1/artifact/requests", {"cursor": "!!"}, 400, "bad_parameter"),
    "requests_cursor_bomb": (
        "/api/v1/artifact/requests",
        {"cursor": CURSOR_BOMB},
        400,
        "bad_parameter",
    ),
    "items_cursor_bomb": (
        "/api/v1/workspace/items",
        {"cursor": CURSOR_BOMB},
        400,
        "bad_parameter",
    ),
    "bins_and_edges": (
        "/api/v1/artifact/distribution",
        {"metric": "client.e2e", "bins": 3, "edges": "0,1"},
        400,
        "bad_parameter",
    ),
    "unknown_metric": (
        "/api/v1/artifact/distribution",
        {"metric": "e2e"},
        400,
        "bad_parameter",
    ),
    "request_not_found": (
        "/api/v1/artifact/request",
        {"request_id": "nope"},
        404,
        "request_not_found",
    ),
    "line_out_of_range": (
        "/api/v1/artifact/line",
        {"line": 10**9},
        404,
        "line_out_of_range",
    ),
    "stale_line_sha": (
        "/api/v1/artifact/line",
        {"line": 0, "sha256": "0" * 64},
        409,
        "stale_line",
    ),
    "missing_artifact": (
        "/api/v1/artifact/summary",
        {"artifact": None},
        400,
        "bad_parameter",
    ),
    "artifact_not_found": (
        "/api/v1/artifact",
        {"artifact": "nope.jsonl"},
        404,
        "artifact_not_found",
    ),
    "traversal": (
        "/api/v1/artifact",
        {"artifact": "../etc/passwd"},
        403,
        "path_not_allowed",
    ),
    "audio": (
        "/api/v1/artifact/summary",
        {"artifact": "audio.jsonl"},
        422,
        "unsupported_artifact",
    ),
    "not_inference": (
        "/api/v1/artifact/requests",
        {"artifact": "not_an_inference_artifact.jsonl"},
        422,
        "not_an_inference_artifact",
    ),
    "multi_session": (
        "/api/v1/artifact/cases",
        {"artifact": "multi_session.jsonl"},
        422,
        "unsupported_artifact",
    ),
    "report_not_a_report": ("/api/v1/report", {"path": MIXED}, 422, "unreadable"),
    "health_unknown_parameter": (
        "/api/v1/health",
        {"x": 1},
        400,
        "unknown_parameter",
    ),
    "items_bad_kind": (
        "/api/v1/workspace/items",
        {"kind": "zip"},
        400,
        "bad_parameter",
    ),
}


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_errors_are_the_section_5_2_documents(served: Served, case: str) -> None:
    route, params, status, code = ERROR_CASES[case]
    for rel in (
        "audio.jsonl",
        "not_an_inference_artifact.jsonl",
        "multi_session.jsonl",
    ):
        served.ready(rel)
    served.ready(MIXED)
    query = dict(params)
    if route.startswith("/api/v1/artifact"):
        query.setdefault("artifact", MIXED)
    query = {name: value for name, value in query.items() if value is not None}
    error = assert_error(served.get(route, **query), status, code)
    if case in ("audio", "multi_session"):
        expected = "audio" if case == "audio" else "multiple_sessions"
        assert error["details"]["reason"] == expected
    if status in (400, 404, 409, 422):
        assert error["verdict_status"] in ("usage", "invalid_input")


def test_a_cursor_from_another_filter_is_a_409(served: Served) -> None:
    served.ready(MIXED)
    first = served.get("/api/v1/artifact/requests", artifact=MIXED, limit=2).json()
    reply = served.get(
        "/api/v1/artifact/requests",
        artifact=MIXED,
        status="ok",
        limit=2,
        cursor=first["next_cursor"],
    )
    error = assert_error(reply, 409, "cursor_mismatch")
    assert error["details"]["reason"] == "other_filter"


def test_a_repeated_single_parameter_is_refused(served: Served) -> None:
    reply = served.get("/api/v1/artifact", artifact=[MIXED, MIXED])
    assert_error(reply, 400, "bad_parameter")


# Every integer parameter api.py reads, with what else its route needs.
INTEGER_PARAMETERS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("/api/v1/workspace/items", "limit", {}),
    ("/api/v1/artifact/requests", "limit", {"artifact": MIXED}),
    ("/api/v1/artifact/distribution", "bins", {"metric": "client.e2e"}),
    ("/api/v1/artifact/points", "max_points", {}),
    ("/api/v1/artifact/samples", "max_points", {}),
    ("/api/v1/artifact/request", "line", {"request_id": True}),
    ("/api/v1/artifact/request/chunks", "line", {"request_id": True}),
    ("/api/v1/artifact/request/chunks", "offset", {"request_id": True}),
    ("/api/v1/artifact/request/chunks", "limit", {"request_id": True}),
    ("/api/v1/artifact/line", "line", {}),
    ("/api/v1/artifact/evidence", "line", {}),
)
NOT_INTEGERS = ("--5", "-", "+5", "5-", "-0-", " 5", "1e3", "0x10", "１", "9" * 19)


@pytest.mark.parametrize(("route", "name", "extra"), INTEGER_PARAMETERS)
def test_every_integer_parameter_refuses_a_non_integer(
    served: Served, route: str, name: str, extra: dict[str, Any]
) -> None:
    """Fable's P3-1: ``limit=--5`` was a 500; any such text is a 400."""
    row = first_row(served)
    query: dict[str, Any] = {"artifact": MIXED} if "/artifact/" in route else {}
    query.update(extra)
    if "request_id" in extra:
        query["request_id"] = row["request_id"]
    for text in NOT_INTEGERS:
        error = assert_error(
            served.get(route, **{**query, name: text}), 400, "bad_parameter"
        )
        assert error["field"] == name, text
    valid = row["line"] if name == "line" else 10
    reply = served.get(route, **{**query, name: valid})
    assert reply.status == 200, reply.body[:300]


# --- Analysis: 202 then 200 ------------------------------------------------------------


def test_the_analysis_is_202_while_it_runs_then_200(
    private: Callable[..., Served], studio_artifacts: StandardArtifacts
) -> None:
    studio = private()
    shutil.copy2(studio_artifacts["mixed"], studio.root / MIXED)
    first = studio.get("/api/v1/artifact/analysis", artifact=MIXED)
    pending = assert_document(first, "analysis", status=202)
    assert pending["status"] in ("pending", "running")
    assert int(first.headers["retry-after"]) >= 1
    assert pending["artifact"]["path"] == MIXED
    replies: list[Reply] = []

    def done() -> bool:
        replies.append(studio.get("/api/v1/artifact/analysis", artifact=MIXED))
        return replies[-1].status == 200

    poll(done, timeout_s=120.0)
    ready = assert_document(replies[-1], "analysis")
    assert ready["status"] == "ready" and ready["report"] is not None
    assert all(reply.status in (200, 202) for reply in replies)
    panels = studio.get("/api/v1/artifact/capabilities", artifact=MIXED).json()
    assert panels["panels"]["analysis"]["state"] == "available"


# --- Export and tickets -----------------------------------------------------------------


def _source_lines(path: Path, lines: list[int]) -> bytes:
    data = path.read_bytes().split(b"\n")
    return b"".join(data[n] + b"\n" for n in lines)


def _all_rows(served: Served, **query: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params = dict(query, limit=7)
        if cursor is not None:
            params["cursor"] = cursor
        page = served.get("/api/v1/artifact/requests", **params).json()
        rows += page["rows"]
        cursor = page["next_cursor"]
        if cursor is None:
            return rows


def test_export_bytes_equal_the_source_lines(served: Served) -> None:
    snapshot = served.ready(MIXED)
    case = first_row(served)["case_id"]
    query = {"artifact": MIXED, "case": case, "phase": "all"}
    lines = [row["line"] for row in _all_rows(served, **query)]
    assert lines == sorted(lines) and lines
    reply = served.get("/api/v1/artifact/export", **query)
    assert reply.status == 200
    assert reply.headers["content-type"] == "application/x-ndjson"
    assert reply.headers["transfer-encoding"] == "chunked"
    assert reply.headers["content-disposition"].startswith("attachment;")
    assert reply.headers["x-stormlog-source-sha256"] == snapshot.sha256
    assert reply.body == _source_lines(served.root / MIXED, lines)
    preview = served.get("/api/v1/artifact/export/preview", **query).json()
    assert preview["bytes"] == len(reply.body) and preview["count"] == len(lines)


def test_a_ticket_downloads_the_export_once(served: Served) -> None:
    served.ready(MIXED)
    query = f"artifact={MIXED}&status=ok"
    reply = served.post(
        "/api/v1/tickets", {"route": "/api/v1/artifact/export", "query": query}
    )
    ticket = assert_document(reply, "ticket")
    with_bearer = served.get("/api/v1/artifact/export", artifact=MIXED, status="ok")
    download = fetch(served.port, "GET", ticket["url"])
    assert download.status == 200
    assert download.body == with_bearer.body and download.body
    again = fetch(served.port, "GET", ticket["url"])
    assert again.status == 401
    assert again.headers["content-type"].startswith("text/plain")


def test_a_raw_line_by_bearer_and_by_ticket(served: Served) -> None:
    row = first_row(served)
    line, sha = row["line"], row["line_sha256"]
    expected = (served.root / MIXED).read_bytes().split(b"\n")[line]
    reply = served.get("/api/v1/artifact/line", artifact=MIXED, line=line, sha256=sha)
    assert reply.status == 200 and reply.body == expected
    assert reply.headers["content-type"] == api.LINE_TYPE
    assert reply.headers["x-stormlog-line"] == str(line)
    assert reply.headers["x-stormlog-line-sha256"] == sha
    query = f"artifact={MIXED}&line={line}"
    ticket = served.post(
        "/api/v1/tickets", {"route": "/api/v1/artifact/line", "query": query}
    ).json()
    assert fetch(served.port, "GET", ticket["url"]).body == expected


def test_evidence_resolves_and_names_each_mismatch(served: Served) -> None:
    row = first_row(served)
    line, sha, record_id = row["line"], row["line_sha256"], row["record_id"]
    route = "/api/v1/artifact/evidence"

    def state(**params: Any) -> tuple[str, str | None]:
        document = assert_document(
            served.get(route, artifact=MIXED, line=line, **params), "evidence"
        )
        return document["state"], document["reason"]

    assert state(sha256=sha, record_id=record_id) == ("resolved", None)
    assert state(sha256="0" * 64) == ("unresolved", "hash_mismatch")
    assert state(record_id="infer.request/other") == (
        "unresolved",
        "record_id_mismatch",
    )
    far = served.get(route, artifact=MIXED, line=10**9).json()
    assert (far["state"], far["reason"], far["found"]) == (
        "unresolved",
        "line_out_of_range",
        None,
    )


def test_an_oversized_line_is_refused_unread(
    private: Callable[..., Served], studio_artifacts: StandardArtifacts
) -> None:
    """A line over ``line_bytes`` is a problem line with a reference, never
    served or parsed whole (Fable's P2-1, probe_api_local)."""
    studio = private(dataclasses.replace(Limits(), line_bytes=65536))
    data = studio_artifacts["mixed"].read_bytes()
    pad = b"x" * 300_000
    huge = b'{"schema_version": 1, "event_type": "infer.note", "pad": "' + pad + b'"}'
    (studio.root / "big.jsonl").write_bytes(data + huge + b"\n")
    line = data.count(b"\n")
    snapshot = studio.ready("big.jsonl")
    assert [(p.ref.line, p.code) for p in snapshot.problems] == [(line, "oversized")]
    for route in ("/api/v1/artifact/line", "/api/v1/artifact/evidence"):
        reply = studio.get(route, artifact="big.jsonl", line=line)
        details = assert_error(reply, 422, "unreadable")["details"]
        assert details == {
            "line": line,
            "reason": "oversized",
            "length": len(huge),
            "limit": 65536,
        }
        assert len(reply.body) < 1000


@pytest.mark.parametrize("depth", [MAX_NESTING, MAX_NESTING + 1, 50_000])
def test_evidence_reads_a_line_nested_past_the_limit_as_not_json(
    private: Callable[..., Served], studio_artifacts: StandardArtifacts, depth: int
) -> None:
    """Fable's P3: the route parsed a problem line with json.loads, so a line
    nested past MAX_NESTING had a record_id where json.loads took it (always
    at 129 levels, at 50,000 only from 3.14) though the index calls it
    not_json. Past the limit it now has none, on every Python."""
    studio = private()
    data = studio_artifacts["mixed"].read_bytes()
    inner = "[" * (depth - 1) + "]" * (depth - 1)
    deep = (
        '{"schema_version": 1, "event_type": "infer.system_sample", '
        f'"timestamp_ns": 7, "deep": {inner}}}'
    )
    (studio.root / "deep.jsonl").write_bytes(data + deep.encode() + b"\n")
    line = data.count(b"\n")
    snapshot = studio.ready("deep.jsonl")
    problems = [(p.ref.line, p.code) for p in snapshot.problems]
    reply = studio.get("/api/v1/artifact/evidence", artifact="deep.jsonl", line=line)
    found = assert_document(reply, "evidence")["found"]
    if depth <= MAX_NESTING:
        assert (problems, found["record_id"]) == ([], "infer.system_sample@7")
    else:
        assert (problems, found["record_id"]) == ([(line, "not_json")], None)


# --- Growth, refresh and the cut --------------------------------------------------------


def _artifact(studio: Served, rel: str) -> dict[str, Any]:
    document: dict[str, Any] = studio.get("/api/v1/artifact", artifact=rel).json()
    return document


def test_a_growing_file_moves_its_generation(
    private: Callable[..., Served], studio_artifacts: StandardArtifacts
) -> None:
    studio = private()
    growing = GrowingCopy(
        studio_artifacts["mixed"], studio.root / "run.jsonl", lines=30
    )
    studio.ready("run.jsonl")
    before = _artifact(studio, "run.jsonl")
    total = studio.get("/api/v1/artifact/requests", artifact="run.jsonl", phase="all")
    growing.finish()
    seen: list[dict[str, Any]] = []

    def moved() -> bool:
        seen.append(_artifact(studio, "run.jsonl"))
        block = seen[-1]["artifact"]
        return bool(block["generation"] > before["artifact"]["generation"])

    poll(moved)
    after = seen[-1]
    assert after["artifact"]["index_id"] == before["artifact"]["index_id"]
    poll(lambda: _artifact(studio, "run.jsonl")["artifact"]["terminal"] is True)
    now = studio.get("/api/v1/artifact/requests", artifact="run.jsonl", phase="all")
    assert now.json()["total"] > total.json()["total"]
    assert schema_errors(after, "artifact") == []


def test_a_cursor_pages_across_a_refresh_and_not_across_a_rebuild(
    private: Callable[..., Served], studio_artifacts: StandardArtifacts
) -> None:
    studio = private()
    source = studio_artifacts["mixed"]
    growing = GrowingCopy(source, studio.root / "run.jsonl", lines=40)
    studio.ready("run.jsonl")
    query = {"artifact": "run.jsonl", "phase": "all", "limit": 2}
    first = studio.get("/api/v1/artifact/requests", **query).json()
    generation = first["artifact"]["generation"]
    growing.finish()
    poll(lambda: _artifact(studio, "run.jsonl")["artifact"]["generation"] > generation)
    page = studio.get("/api/v1/artifact/requests", cursor=first["next_cursor"], **query)
    second = assert_document(page, "requests")
    assert second["generation_moved"] is True
    seen = [row["line"] for row in first["rows"] + second["rows"]]
    assert seen == sorted(set(seen))
    # A rerun to the same path rewrites the same inode: a new index ID.
    index_id = second["artifact"]["index_id"]
    with (studio.root / "run.jsonl").open("r+b") as handle:
        handle.truncate(0)
        handle.write(studio_artifacts["tokens_exact"].read_bytes())
    poll(lambda: _artifact(studio, "run.jsonl")["artifact"]["index_id"] != index_id)
    studio.ready("run.jsonl")
    reply = studio.get(
        "/api/v1/artifact/requests", cursor=first["next_cursor"], **query
    )
    assert assert_error(reply, 409, "cursor_mismatch")["details"]["reason"] == (
        "index_rebuilt"
    )


def test_a_partial_index_shows_its_cut(
    private: Callable[..., Served], studio_artifacts: StandardArtifacts
) -> None:
    studio = private(dataclasses.replace(Limits(), row_bytes=128 * 1024))
    shutil.copy2(studio_artifacts["synthetic"], studio.root / "big.jsonl")
    snapshot = studio.ready("big.jsonl")
    assert snapshot.status == "partial" and snapshot.cut is not None
    document = assert_document(
        studio.get("/api/v1/artifact", artifact="big.jsonl"), "artifact"
    )
    cut = document["artifact"]["cut"]
    assert cut is not None and cut["line"] == snapshot.cut.line
    assert "rows" in document["artifact"]["limits_hit"]
    warnings = {w["code"]: w["message"] for w in document["warnings"]}
    assert f"line {cut['line']}" in warnings["index_cut"]
    rows = document["index"]["requests"]
    assert 0 < rows < 2_000
    summary = studio.get("/api/v1/artifact/summary", artifact="big.jsonl", phase="all")
    assert assert_document(summary, "summary")["selection"]["size"] == rows
    panels = studio.get("/api/v1/artifact/capabilities", artifact="big.jsonl").json()
    assert panels["panels"]["requests"]["state"] == "partial"
    assert panels["panels"]["requests"]["reason"] == "index_budget"


@pytest.mark.parametrize("field", HUGE_FIELDS)
@pytest.mark.parametrize("huge", [10**400, -(10**400)], ids=["e400", "-e400"])
def test_an_int_past_a_floats_range_is_analyzed_indexed_and_served(
    private: Callable[..., Served], tmp_path: Path, field: str, huge: int
) -> None:
    """Fable's P2-2: such an int in a token count or a latency field made
    ``infer analyze`` exit 1 and the summary a 500, or failed the index
    (``slo._number`` called ``float()`` on it). The summary served from
    the index's columns is the one built from the records."""
    timed: dict[str, Any] = {"intended_at_ns": ORIGIN, "dispatch_lag_ms": 1.0}
    records = closed_case(4, **timed)
    records[3] = request(1, **{**timed, field: huge})
    studio = private()
    path = studio.root / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = tmp_path / "report.json"
    assert infer_main(["analyze", str(path), "--output", str(report)]) == 0
    assert studio.ready("run.jsonl").status == "ready"
    served = assert_document(
        studio.get("/api/v1/artifact/summary", artifact="run.jsonl"), "summary"
    )
    rows = snapshot_from_records(records)
    want = summaries.summary(rows, selection.select(rows, RequestFilter()))
    for key in ("counts", "latency", "streaming", "tokens", "throughput"):
        assert served[key] == json.loads(json.dumps(want[key])), key


# --- End to end: `stormlog studio <fixture>` as a subprocess -----------------------------


_LAUNCH = re.compile(r"Open (http://127\.0\.0\.1:([0-9]+))/#/auth\?t=([A-Za-z0-9_-]+)")


def _lines(stream: Any, sink: queue.Queue[str]) -> None:
    for line in stream:
        sink.put(line)


def _launch_url(process: subprocess.Popen[str], timeout_s: float) -> re.Match[str]:
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_lines, args=(process.stdout, lines), daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            assert process.poll() is None, "stormlog studio exited"
            continue
        found = _LAUNCH.search(line)
        if found is not None:
            return found
    raise AssertionError("no launch URL printed")


def _timed(port: int, route: str, token: str, timings: dict[str, float]) -> Reply:
    began = time.monotonic()
    reply = fetch(port, "GET", route, token=token, timeout=120.0)
    timings[route.split("?")[0]] = time.monotonic() - began
    return reply


def smoke(
    artifact: Path, state: Path, *, build_s: float, routes: list[str]
) -> dict[str, float]:
    """Start ``stormlog studio <artifact>``, sign in as the page does, fetch
    ``routes`` (each with ``?artifact=``), stop it with SIGINT; the timings."""
    env = dict(os.environ, XDG_STATE_HOME=str(state), PYTHONPATH=str(REPO))
    command = [sys.executable, "-m", "stormlog", "studio", str(artifact)]
    process = subprocess.Popen(
        [*command, "--port", "0", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    timings: dict[str, float] = {}
    try:
        found = _launch_url(process, 60.0)
        port = int(found.group(2))
        exchanged = fetch(
            port, "POST", "/api/v1/session", body={"bootstrap": found.group(3)}
        )
        token = assert_document(exchanged, "session")["session_token"]
        query = "?artifact=" + quote(artifact.name, safe="")
        began = time.monotonic()
        artifact_doc: list[dict[str, Any]] = []

        def built() -> bool:
            reply = fetch(port, "GET", "/api/v1/artifact" + query, token=token)
            artifact_doc.append(assert_document(reply, "artifact"))
            return bool(artifact_doc[-1]["artifact"]["status"] != "building")

        poll(built, timeout_s=build_s)
        timings["build"] = time.monotonic() - began
        for route in routes:
            reply = _timed(port, route + query, token, timings)
            assert_document(reply, SMOKE_DOCUMENTS[route])
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=60) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert not list((state / "stormlog" / "studio").glob("*.json"))
    return timings


SMOKE_ROUTES = [
    "/api/v1/artifact",
    "/api/v1/artifact/summary",
    "/api/v1/artifact/requests",
]
SMOKE_DOCUMENTS = {
    "/api/v1/artifact": "artifact",
    "/api/v1/artifact/summary": "summary",
    "/api/v1/artifact/requests": "requests",
    "/api/v1/artifact/capabilities": "artifact_capabilities",
    "/api/v1/artifact/export/preview": "export_preview",
    "/api/v1/artifact/points": "points",
}


def _print_timings(name: str, timings: Mapping[str, float]) -> None:
    for route, seconds in timings.items():
        print(f"studio smoke {name}: {route} {seconds * 1000:.0f} ms")


def test_stormlog_studio_serves_a_fixture_end_to_end(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    artifact = Path(shutil.copy2(studio_artifacts["mixed"], root / MIXED))
    timings = smoke(artifact, tmp_path / "state", build_s=WAIT_S, routes=SMOKE_ROUTES)
    _print_timings("mixed", timings)
    assert all(seconds < Limits().deadline_s for seconds in timings.values())


@pytest.mark.studio_large
def test_stormlog_studio_serves_the_500k_fixture_end_to_end(
    large_studio_artifact: Path, tmp_path: Path
) -> None:
    routes = [
        *SMOKE_ROUTES,
        "/api/v1/artifact/capabilities",
        "/api/v1/artifact/export/preview",
        "/api/v1/artifact/points",
    ]
    timings = smoke(
        large_studio_artifact, tmp_path / "state", build_s=300.0, routes=routes
    )
    _print_timings("500k", timings)
    report = os.environ.get("STORMLOG_STUDIO_TIMINGS")
    if report:
        Path(report).write_text(json.dumps(timings, indent=2), encoding="utf-8")
    deadline = Limits().deadline_s
    assert all(s < deadline for route, s in timings.items() if route != "build")
