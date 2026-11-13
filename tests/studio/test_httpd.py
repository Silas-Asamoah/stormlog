"""StudioServer end to end: a real server on a free port, with stub routes.

This file also holds the stub composer and the raw HTTP helper that
``test_cli.py`` reuses: ``auth_routes`` plus a few stub routes that
exercise every path through httpd (errors, 202, streams, deadlines,
tickets), over the stub context of ``test_security.py``.
"""

from __future__ import annotations

import http.client
import io
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from stormlog.studio.contracts import (
    HttpRequest,
    HttpResponse,
    LaunchOptions,
    Limits,
    NotReady,
    Route,
    StudioContext,
    UsageError,
    envelope,
)
from stormlog.studio.httpd import MESSAGE_MAX_BYTES, StudioServer
from stormlog.studio.security import JSON_TYPE, SessionTokens, auth_routes
from stormlog.studio.static_files import CONTENT_SECURITY_POLICY, StaticFiles
from tests.studio.test_security import (
    INSTANCE_ID,
    launch_options,
    stub_context,
    validate,
)

ARTIFACT_TEXT = b'{"event_type": "infer.session"}\n{"event_type": "infer.request"}\n'


def _json(name: str, body: Mapping[str, Any], status: int = 200) -> HttpResponse:
    from stormlog.studio.contracts import encode_json

    return HttpResponse(status, JSON_TYPE, encode_json(envelope(name, body)))


def stub_routes(ctx: StudioContext) -> list[Route]:
    """Routes standing in for C4's, one per path through httpd."""

    def server(request: HttpRequest) -> HttpResponse:
        return _json("stub", {"query": sorted(request.query)})

    def line(request: HttpRequest) -> HttpResponse:
        rel = request.query.get("artifact", ("",))[0]
        with ctx.root.open(rel) as handle:
            first = handle.readline()
        headers = (("X-Params", ",".join(sorted(request.query))),)
        return HttpResponse(200, "text/plain; charset=utf-8", first, headers)

    def analysis(request: HttpRequest) -> HttpResponse:
        state = envelope("analysis", {"status": "running", "reason": None})
        raise NotReady(state, retry_after_s=3)

    def error(request: HttpRequest) -> HttpResponse:
        message = request.query.get("message", ("bad",))[0]
        raise UsageError("bad_parameter", message, field="status")

    def crash(request: HttpRequest) -> HttpResponse:
        raise RuntimeError("secret detail /home/someone/.ssh/id_rsa")

    def stream(request: HttpRequest) -> HttpResponse:
        fail = "fail" in request.query

        def chunks() -> Iterator[bytes]:
            yield b'{"line": 0}\n'
            if fail:
                raise RuntimeError("hash mismatch")
            yield b'{"line": 1}\n'

        return HttpResponse(200, "application/x-ndjson", chunks())

    def sleep(request: HttpRequest) -> HttpResponse:
        time.sleep(float(request.query.get("s", ("0",))[0]))
        return _json("stub", {"slept": True})

    def echo(request: HttpRequest) -> HttpResponse:
        return _json("stub", {"bytes": len(request.body)})

    return [
        Route("GET", "/api/v1/server", "bearer", server),
        Route("GET", "/api/v1/artifact/line", "bearer_or_ticket", line, 600.0),
        Route("GET", "/api/v1/artifact/analysis", "bearer", analysis),
        Route("GET", "/api/v1/test/error", "bearer", error),
        Route("GET", "/api/v1/test/crash", "bearer", crash),
        Route("GET", "/api/v1/test/stream", "bearer", stream),
        Route("GET", "/api/v1/test/sleep", "bearer", sleep, 3.0),
        Route("GET", "/api/v1/test/quick", "bearer", sleep, 0.5),
        Route("POST", "/api/v1/test/echo", "bearer", echo),
        Route("POST", "/api/v1/test/echo-quick", "bearer", echo, 0.5),
    ]


def stub_compose(options: LaunchOptions) -> tuple[StudioContext, list[Route]]:
    ctx = stub_context(options)
    return ctx, auth_routes(ctx) + stub_routes(ctx)


# --- A raw HTTP client: every header as given --------------------------------


@dataclass
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def call(
    port: int,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str | None] | None = None,
    body: bytes | None = None,
) -> Reply:
    """One request with exactly ``headers`` (a None value drops a default)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        merged: dict[str, str | None] = {"Host": f"127.0.0.1:{port}"}
        merged.update(headers or {})
        for name, value in merged.items():
            if value is not None:
                connection.putheader(name, value)
        if body is not None:
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        data = response.read()
        lowered = {k.lower(): v for k, v in response.getheaders()}
        return Reply(response.status, lowered, data)
    finally:
        connection.close()


@dataclass
class Studio:
    server: StudioServer
    ctx: StudioContext
    tokens: SessionTokens
    stderr: io.StringIO

    @property
    def port(self) -> int:
        return self.server.bound_port

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def session(self) -> str:
        bootstrap = self.tokens.issue_bootstrap()
        reply = self.post("/api/v1/session", {"bootstrap": bootstrap})
        assert reply.status == 200, reply.body
        token: str = reply.json()["session_token"]
        return token

    def post(
        self,
        path: str,
        payload: Any,
        *,
        token: str | None = None,
        headers: Mapping[str, str | None] | None = None,
    ) -> Reply:
        merged: dict[str, str | None] = {
            "Origin": self.origin,
            "Content-Type": "application/json",
        }
        if token is not None:
            merged["Authorization"] = f"Bearer {token}"
        merged.update(headers or {})
        return call(
            self.port,
            "POST",
            path,
            headers=merged,
            body=json.dumps(payload).encode(),
        )

    def get(
        self,
        path: str,
        *,
        token: str | None = None,
        headers: Mapping[str, str | None] | None = None,
    ) -> Reply:
        merged: dict[str, str | None] = {}
        if token is not None:
            merged["Authorization"] = f"Bearer {token}"
        merged.update(headers or {})
        return call(self.port, "GET", path, headers=merged)


StudioFactory = Callable[..., Studio]


@pytest.fixture
def make_studio(tmp_path: Path) -> Iterator[StudioFactory]:
    started: list[Studio] = []
    root = tmp_path / "root"
    root.mkdir()
    (root / "run.jsonl").write_bytes(ARTIFACT_TEXT)

    def make(*, limits: Limits | None = None, log_requests: bool = False) -> Studio:
        options = launch_options(root, tmp_path / "state")
        tokens = SessionTokens()
        ctx = stub_context(options, tokens=tokens)
        stderr = io.StringIO()
        server = StudioServer(
            auth_routes(ctx) + stub_routes(ctx),
            tokens,
            StaticFiles(),
            host="127.0.0.1",
            port=0,
            limits=limits or Limits(),
            log_requests=log_requests,
            stderr=stderr,
        )
        server.start()
        studio = Studio(server, ctx, tokens, stderr)
        started.append(studio)
        return studio

    yield make
    for studio in started:
        studio.server.stop()
        studio.ctx.pool.shutdown(wait=False)


@pytest.fixture
def studio(make_studio: StudioFactory) -> Studio:
    return make_studio()


def assert_error(reply: Reply, status: int, code: str) -> dict[str, Any]:
    assert reply.status == status, reply.body
    assert reply.headers["content-type"] == JSON_TYPE
    document: dict[str, Any] = reply.json()
    validate(document, "error")
    assert document["error"]["code"] == code
    assert document["error"]["http_status"] == status
    return document


# --- Start, stop and the happy path ------------------------------------------


def test_start_serves_health_and_stop_closes_the_port(
    make_studio: StudioFactory,
) -> None:
    studio = make_studio()
    assert studio.server.base_url == f"http://127.0.0.1:{studio.port}"
    reply = studio.get("/api/v1/health")
    assert reply.status == 200
    validate(reply.json(), "health")
    assert reply.json()["instance_id"] == INSTANCE_ID
    port = studio.port
    studio.server.stop()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_every_response_carries_the_section_7_5_headers(studio: Studio) -> None:
    api = studio.get("/api/v1/health")
    assert api.headers["x-content-type-options"] == "nosniff"
    assert api.headers["referrer-policy"] == "no-referrer"
    assert api.headers["cache-control"] == "no-store"
    assert api.headers["connection"] == "close"
    assert "etag" not in api.headers
    assert not any(name.startswith("access-control-") for name in api.headers)
    page = studio.get("/")
    assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["cache-control"] == "no-cache"


def test_bootstrap_session_bearer_ticket_flow(studio: Studio) -> None:
    token = studio.session()
    assert studio.get("/api/v1/server", token=token).status == 200
    issued = studio.post(
        "/api/v1/tickets",
        {"route": "/api/v1/artifact/line", "query": "artifact=run.jsonl"},
        token=token,
    )
    assert issued.status == 200
    validate(issued.json(), "ticket")
    url = issued.json()["url"]
    first = studio.get(url)
    assert first.status == 200
    assert first.body == ARTIFACT_TEXT.split(b"\n")[0] + b"\n"
    # The handler never sees the ticket parameter.
    assert first.headers["x-params"] == "artifact"
    again = studio.get(url)
    assert again.status == 401
    assert again.headers["content-type"].startswith("text/plain")
    assert b"Open it again from Studio" in again.body


def test_a_bootstrap_is_single_use_over_http(studio: Studio) -> None:
    bootstrap = studio.tokens.issue_bootstrap()
    assert studio.post("/api/v1/session", {"bootstrap": bootstrap}).status == 200
    reply = studio.post("/api/v1/session", {"bootstrap": bootstrap})
    assert_error(reply, 401, "bootstrap_invalid")


# --- Security checks over the wire (§9.1) ------------------------------------


@pytest.mark.parametrize("host", ["evil.example:8790", "127.0.0.1.nip.io", None])
def test_dns_rebinding_host_is_refused_even_for_static(
    studio: Studio, host: str | None
) -> None:
    for path in ("/", "/api/v1/health"):
        reply = studio.get(path, headers={"Host": host})
        assert_error(reply, 403, "host_not_allowed")


def test_any_loopback_port_is_accepted_for_ssh_tunnels(studio: Studio) -> None:
    for host in ("127.0.0.1:9000", "localhost:9000", "[::1]:9000", "localhost"):
        assert studio.get("/api/v1/health", headers={"Host": host}).status == 200


def test_cross_site_fetch_metadata_is_refused_on_api_only(studio: Studio) -> None:
    token = studio.session()
    cross = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}
    assert studio.get("/", headers=cross).status == 200
    reply = studio.get("/api/v1/server", token=token, headers=cross)
    assert_error(reply, 403, "fetch_site_not_allowed")


def test_an_origin_other_than_the_host_is_refused(studio: Studio) -> None:
    token = studio.session()
    other = {"Origin": "http://127.0.0.1:1"}
    assert_error(
        studio.get("/api/v1/server", token=token, headers=other),
        403,
        "origin_not_allowed",
    )
    assert_error(studio.get("/", headers=other), 403, "origin_not_allowed")


def test_missing_and_wrong_tokens_are_refused(studio: Studio) -> None:
    reply = studio.get("/api/v1/server")
    assert_error(reply, 401, "unauthorized")
    assert reply.headers["www-authenticate"].startswith("Bearer")
    assert_error(studio.get("/api/v1/server", token="x" * 43), 401, "unauthorized")


def test_traversal_through_an_api_parameter_is_refused(studio: Studio) -> None:
    token = studio.session()
    for target in ("..%2Fsecret", "%2e%2e/secret", "%2Fetc%2Fpasswd", "a%5Cb"):
        reply = studio.get(f"/api/v1/artifact/line?artifact={target}", token=token)
        assert_error(reply, 403, "path_not_allowed")


def test_post_rules_415_and_origin(studio: Studio) -> None:
    token = studio.session()
    reply = studio.post(
        "/api/v1/test/echo", {}, token=token, headers={"Content-Type": "text/plain"}
    )
    assert_error(reply, 415, "unsupported_media_type")
    reply = studio.post("/api/v1/test/echo", {}, token=token, headers={"Origin": None})
    assert_error(reply, 403, "origin_not_allowed")


def test_admin_route_refuses_browsers_and_session_tokens(studio: Studio) -> None:
    admin = studio.tokens.issue_admin()
    browser = studio.post("/api/v1/admin/bootstrap", {}, token=admin)
    assert_error(browser, 403, "origin_not_allowed")
    session = studio.post(
        "/api/v1/admin/bootstrap", {}, token=studio.session(), headers={"Origin": None}
    )
    assert_error(session, 401, "unauthorized")
    good = studio.post(
        "/api/v1/admin/bootstrap",
        {"route": "/cases?artifact=run.jsonl"},
        token=admin,
        headers={"Origin": None},
    )
    assert good.status == 200
    validate(good.json(), "bootstrap")
    assert good.json()["url"].startswith(studio.origin + "/#/auth?t=")


# --- Bounds: heads, bodies, deadlines, connections ---------------------------


def raw_exchange(port: int, data: bytes, *, wait_s: float = 5.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=wait_s) as sock:
        sock.sendall(data)
        chunks = []
        while True:
            try:
                chunk = sock.recv(65536)
            except (ConnectionResetError, BrokenPipeError):
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def parse_raw(data: bytes) -> Reply:
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return Reply(status, headers, body)


def test_oversized_request_line_gets_431(studio: Studio) -> None:
    target = "/api/v1/health?" + "a" * (17 * 1024)
    data = f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode()
    assert_error(parse_raw(raw_exchange(studio.port, data)), 431, "head_too_large")


def test_oversized_headers_get_431(studio: Studio) -> None:
    headers = "".join(f"X-Pad-{i}: {'b' * 200}\r\n" for i in range(90))
    data = f"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}\r\n".encode()
    assert_error(parse_raw(raw_exchange(studio.port, data)), 431, "head_too_large")


def test_oversized_body_gets_413_without_reading_it(studio: Studio) -> None:
    token = studio.session()
    reply = studio.post("/api/v1/test/echo", {"pad": "x" * (65 * 1024)}, token=token)
    assert_error(reply, 413, "body_too_large")
    small = studio.post("/api/v1/test/echo", {"pad": "x"}, token=token)
    assert small.status == 200


def test_slow_loris_head_is_cut_at_the_deadline(make_studio: StudioFactory) -> None:
    studio = make_studio(limits=Limits(deadline_s=0.5))
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", studio.port), timeout=10) as sock:
        sock.sendall(b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n")
        try:
            data = sock.recv(1024)
        except ConnectionResetError:
            data = b""
    assert data == b""
    assert time.monotonic() - started < 5.0


def test_slow_body_is_cut_at_the_route_deadline(make_studio: StudioFactory) -> None:
    studio = make_studio(limits=Limits(deadline_s=10.0))
    token = studio.session()
    head = (
        "POST /api/v1/test/echo-quick HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{studio.port}\r\nOrigin: {studio.origin}\r\n"
        f"Authorization: Bearer {token}\r\nContent-Type: application/json\r\n"
        "Content-Length: 100\r\n\r\n{"
    )
    started = time.monotonic()
    data = raw_exchange(studio.port, head.encode(), wait_s=20.0)
    # The body never completes: the route's 0.5 s deadline cuts it, long
    # before the 10 s head deadline or the socket's own timeout.
    assert time.monotonic() - started < 5.0
    assert data == b""


def test_route_deadline_is_re_armed_after_the_head(make_studio: StudioFactory) -> None:
    studio = make_studio(limits=Limits(deadline_s=0.5))
    token = studio.session()
    # Longer than the 0.5 s head deadline, inside the route's 3 s.
    assert studio.get("/api/v1/test/sleep?s=1.0", token=token).status == 200
    with pytest.raises((http.client.HTTPException, OSError)):
        studio.get("/api/v1/test/quick?s=2.0", token=token)


def test_connections_over_the_cap_get_a_busy_document(
    make_studio: StudioFactory,
) -> None:
    studio = make_studio(limits=Limits(max_connections=2))
    held = [socket.create_connection(("127.0.0.1", studio.port)) for _ in range(2)]
    try:
        time.sleep(0.2)
        reply = parse_raw(raw_exchange(studio.port, b""))
        assert_error(reply, 503, "busy")
    finally:
        for sock in held:
            sock.close()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if studio.get("/api/v1/health").status == 200:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the slots were never given back")


# --- Routing and rendering ---------------------------------------------------


def test_unknown_route_is_404_route_not_found(studio: Studio) -> None:
    document = assert_error(studio.get("/api/v1/nope"), 404, "route_not_found")
    assert document["error"]["verdict_status"] == "invalid_input"


def test_wrong_method_and_options_are_405(studio: Studio) -> None:
    reply = studio.get("/api/v1/session")
    assert_error(reply, 405, "method_not_allowed")
    assert reply.headers["allow"] == "POST"
    options = call(studio.port, "OPTIONS", "/api/v1/health")
    assert_error(options, 405, "method_not_allowed")
    assert not any(k.startswith("access-control-") for k in options.headers)
    assert_error(call(studio.port, "DELETE", "/"), 405, "method_not_allowed")


def test_studio_errors_render_scrubbed_and_bounded(studio: Studio) -> None:
    token = studio.session()
    secret = "sk-" + "A1b2C3d4" * 6
    message = f"status: bad {secret} " + "x" * 2000
    reply = studio.get(f"/api/v1/test/error?message={quote(message)}", token=token)
    document = assert_error(reply, 400, "bad_parameter")
    text = document["error"]["message"]
    assert secret not in text
    assert len(text.encode("utf-8")) <= MESSAGE_MAX_BYTES
    assert document["error"]["field"] == "status"


def test_unexpected_exceptions_are_a_generic_500(studio: Studio) -> None:
    reply = studio.get("/api/v1/test/crash", token=studio.session())
    document = assert_error(reply, 500, "internal_error")
    assert ".ssh" not in reply.body.decode()
    assert document["error"]["verdict_status"] == "error"


def test_not_ready_is_202_with_its_state_verbatim(studio: Studio) -> None:
    reply = studio.get("/api/v1/artifact/analysis", token=studio.session())
    assert reply.status == 202
    assert reply.headers["retry-after"] == "3"
    assert reply.json() == {
        "format": "stormlog.studio.analysis",
        "schema_version": 1,
        "status": "running",
        "reason": None,
    }


def test_streamed_bodies_are_chunked(studio: Studio) -> None:
    reply = studio.get("/api/v1/test/stream", token=studio.session())
    assert reply.status == 200
    assert reply.headers["transfer-encoding"] == "chunked"
    assert reply.body == b'{"line": 0}\n{"line": 1}\n'


def test_a_failing_stream_ends_without_its_last_chunk(studio: Studio) -> None:
    with pytest.raises(http.client.IncompleteRead):
        studio.get("/api/v1/test/stream?fail=1", token=studio.session())


def test_bad_query_strings_are_400(studio: Studio) -> None:
    token = studio.session()
    reply = studio.get("/api/v1/server?%ff=1", token=token)
    assert_error(reply, 400, "malformed_request")


def test_static_index_etag_and_revalidation(studio: Studio) -> None:
    page = studio.get("/")
    assert page.status == 200
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    assert b"Stormlog Studio" in page.body
    again = studio.get("/", headers={"If-None-Match": page.headers["etag"]})
    assert again.status == 304
    assert again.body == b""
    assert_error(studio.get("/static/manifest.json"), 404, "route_not_found")
    assert_error(studio.get("/static/../security.py"), 404, "route_not_found")


def test_request_log_names_the_route_never_the_query(
    make_studio: StudioFactory,
) -> None:
    studio = make_studio(log_requests=True)
    studio.get("/api/v1/health?secret=hunter2")
    logged = studio.stderr.getvalue()
    assert "GET /api/v1/health 200" in logged
    assert "hunter2" not in logged


def test_no_request_log_by_default(studio: Studio) -> None:
    studio.get("/api/v1/health")
    assert studio.stderr.getvalue() == ""


def test_concurrent_requests_are_served(studio: Studio) -> None:
    token = studio.session()
    results: list[int] = []

    def worker() -> None:
        results.append(studio.get("/api/v1/server", token=token).status)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == [200] * 8
