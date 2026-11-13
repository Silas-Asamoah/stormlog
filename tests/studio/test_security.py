"""security.py: §7.3's checks in order, tokens, tickets, the root (§9.1).

This file also holds the stub context ``test_httpd.py`` and
``test_cli.py`` reuse: ``RootDirectory`` and ``SessionTokens`` with stubs
for the components C3 does not own (C1's indexes, C4's scanner, analyzer
and caches).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest

from stormlog.studio import security
from stormlog.studio.contracts import (
    ForbiddenError,
    HttpRequest,
    HttpResponse,
    IndexSnapshot,
    LaunchOptions,
    LineRef,
    NotFoundError,
    Route,
    Selection,
    StudioContext,
    StudioError,
    UnauthorizedError,
    UsageError,
)
from stormlog.studio.security import (
    BodyTooLarge,
    HeadTooLarge,
    MalformedRequest,
    MethodNotAllowed,
    RootDirectory,
    SessionTokens,
    UnsupportedMediaType,
    auth_routes,
    canonical_query,
    check_request,
    host_name,
    launch_url,
    refuse_broad_root,
    without_ticket,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
INSTANCE_ID = "0123456789abcdef0123456789abcdef"


def schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / f"studio_{name}_v1.schema.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def validate(document: Any, name: str) -> None:
    jsonschema.Draft202012Validator(schema(name)).validate(document)


# --- Stubs for the components C3 does not own --------------------------------


class StubIndexes:
    def snapshot(self, rel: str) -> IndexSnapshot:
        raise NotImplementedError

    def wait(self, rel: str, timeout_s: float) -> IndexSnapshot:
        raise NotImplementedError

    def read_line(self, rel: str, ref: LineRef) -> bytes:
        raise NotImplementedError

    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


class StubScanner:
    def scan(self) -> None:
        return None

    def wait(self, timeout_s: float) -> bool:
        return True

    def state(self) -> dict[str, Any]:
        return {}

    def items(
        self, *, kind: str | None, q: str | None, cursor: str | None, limit: int
    ) -> dict[str, Any]:
        return {}


class StubAnalysis:
    def get(self, rel: str, snapshot: IndexSnapshot) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


class StubCache:
    def selection(
        self, key: tuple[str, ...], make: Callable[[], Selection]
    ) -> Selection:
        return make()

    def document(
        self, key: tuple[str, ...], make: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        return make()


def stub_context(
    options: LaunchOptions,
    *,
    tokens: SessionTokens | None = None,
    pool: Executor | None = None,
) -> StudioContext:
    return StudioContext(
        instance_id=INSTANCE_ID,
        version="0.0.0",
        options=options,
        root=RootDirectory(options.root),
        tokens=tokens or SessionTokens(),
        indexes=StubIndexes(),
        scanner=StubScanner(),
        analysis=StubAnalysis(),
        cache=StubCache(),
        pool=pool or ThreadPoolExecutor(max_workers=1),
    )


def launch_options(root: Path, state: Path, **changes: Any) -> LaunchOptions:
    options = LaunchOptions(
        root=root,
        open_path=None,
        host="127.0.0.1",
        port=0,
        open_browser=False,
        state_dir=state,
        log_requests=False,
        debug=False,
    )
    return replace(options, **changes)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _ok(request: HttpRequest) -> HttpResponse:
    return HttpResponse(200, "text/plain", b"ok")


def route(auth: str = "bearer", method: str = "GET", path: str = "/api/v1/x") -> Route:
    return Route(method, path, auth, _ok)  # type: ignore[arg-type]


def request(
    path: str = "/api/v1/x",
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    query: Mapping[str, tuple[str, ...]] | None = None,
    body: bytes = b"",
    host: str | None = "127.0.0.1:8790",
) -> HttpRequest:
    merged = {} if host is None else {"host": host}
    merged.update({k.lower(): v for k, v in (headers or {}).items()})
    return HttpRequest(method, path, dict(query or {}), merged, body)


def code_of(call: Callable[[], Any]) -> tuple[int, str]:
    with pytest.raises(StudioError) as caught:
        call()
    return caught.value.http_status, caught.value.code


# --- Host (§7.3 step 1) -------------------------------------------------------


@pytest.mark.parametrize(
    "host, name",
    [
        ("127.0.0.1:8790", "127.0.0.1"),
        ("LOCALHOST:9000", "localhost"),
        ("[::1]:8791", "[::1]"),
        ("[::1]", "[::1]"),
        ("localhost", "localhost"),
        ("localhost:99999", None),
        ("localhost:", None),
        ("localhost:80x", None),
        ("[::1]x", None),
    ],
)
def test_host_name_parses_name_and_port(host: str, name: str | None) -> None:
    assert host_name(host) == name


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8790", "127.0.0.1:9000", "localhost:1", "[::1]:22"]
)
def test_loopback_names_pass_with_any_port(host: str) -> None:
    check_request(request(host=host), None, SessionTokens())


@pytest.mark.parametrize(
    "host",
    [
        "attacker.example:8790",
        "127.0.0.1.nip.io:8790",
        "localhost.:8790",
        "0x7f.0.0.1:8790",
        "127.1:8790",
        "",
        None,
    ],
)
def test_dns_rebinding_hosts_are_refused_first(host: str | None) -> None:
    # Even with a cross-site Origin and no token, Host is reported first.
    bad = request(host=host, headers={"Origin": "http://evil.example"})
    assert code_of(lambda: check_request(bad, route(), SessionTokens())) == (
        403,
        "host_not_allowed",
    )


# --- Origin (§7.3 step 2) -----------------------------------------------------


def test_origin_must_equal_http_and_the_host() -> None:
    tokens = SessionTokens()
    same = request(headers={"Origin": "http://127.0.0.1:8790"})
    check_request(same, None, tokens)
    for origin in (
        "http://127.0.0.1:8791",
        "https://127.0.0.1:8790",
        "http://localhost:8790",
        "null",
        "http://127.0.0.1:8790.evil.example",
        "http://ex\u00e4mple",
    ):
        bad = request(headers={"Origin": origin})
        assert code_of(lambda: check_request(bad, None, tokens)) == (
            403,
            "origin_not_allowed",
        )


# --- Fetch Metadata, /api/ only (§7.3 step 3) ---------------------------------


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_cross_site_fetch_metadata_is_refused_on_api(site: str) -> None:
    headers = {"Sec-Fetch-Site": site, "Sec-Fetch-Mode": "navigate"}
    bad = request("/api/v1/server", headers=headers)
    assert code_of(lambda: check_request(bad, None, SessionTokens())) == (
        403,
        "fetch_site_not_allowed",
    )
    for path in ("/", "/static/app.js"):
        check_request(request(path, headers=headers), None, SessionTokens())


@pytest.mark.parametrize("site", ["same-origin", "none"])
def test_same_origin_and_user_initiated_pass(site: str) -> None:
    check_request(request(headers={"Sec-Fetch-Site": site}), None, SessionTokens())


# --- Auth (§7.3 step 4) -------------------------------------------------------


def session_for(tokens: SessionTokens) -> str:
    return tokens.exchange(tokens.issue_bootstrap())


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_bearer_routes_need_a_live_session_token() -> None:
    tokens = SessionTokens()
    token = session_for(tokens)
    check_request(request(headers=bearer(token)), route(), tokens)
    for headers in ({}, bearer(""), bearer("x" * 43), {"Authorization": token}):
        bad = request(headers=headers)
        assert code_of(lambda: check_request(bad, route(), tokens)) == (
            401,
            "unauthorized",
        )


def test_public_routes_need_no_token() -> None:
    check_request(request(), route("public"), SessionTokens())


def test_session_tokens_expire_after_12_hours_idle() -> None:
    clock = Clock()
    tokens = SessionTokens(clock=clock)
    token = session_for(tokens)
    clock.now += 11 * 3600
    assert tokens.check(token, "session")
    # Use renews it: 11 more hours is still inside the window.
    clock.now += 11 * 3600
    assert tokens.check(token, "session")
    clock.now += 12 * 3600
    assert not tokens.check(token, "session")
    assert not tokens.check(token, "session")
    expired = request(headers=bearer(token))
    assert code_of(lambda: check_request(expired, route(), tokens))[0] == 401


def test_tokens_die_with_their_instance() -> None:
    first, second = SessionTokens(), SessionTokens()
    token = session_for(first)
    assert first.check(token, "session")
    assert not second.check(token, "session")


def test_admin_and_session_tokens_do_not_cross() -> None:
    tokens = SessionTokens()
    admin = tokens.issue_admin()
    session = session_for(tokens)
    assert tokens.check(admin, "admin")
    assert not tokens.check(admin, "session")
    assert not tokens.check(session, "admin")
    with pytest.raises(RuntimeError):
        tokens.issue_admin()


def test_admin_routes_refuse_any_browser_header() -> None:
    tokens = SessionTokens()
    admin = tokens.issue_admin()
    post = {"Content-Type": "application/json", **bearer(admin)}
    admin_route = route("admin", "POST")
    check_request(request(method="POST", headers=post), admin_route, tokens)
    origin = request(method="POST", headers={**post, "Origin": "http://127.0.0.1:8790"})
    assert code_of(lambda: check_request(origin, admin_route, tokens)) == (
        403,
        "origin_not_allowed",
    )
    mode = request(method="POST", headers={**post, "Sec-Fetch-Mode": "cors"})
    assert code_of(lambda: check_request(mode, admin_route, tokens)) == (
        403,
        "fetch_site_not_allowed",
    )
    wrong = request(method="POST", headers={**post, **bearer(session_for(tokens))})
    assert code_of(lambda: check_request(wrong, admin_route, tokens)) == (
        401,
        "unauthorized",
    )


def test_bootstrap_is_single_use_and_expires() -> None:
    clock = Clock()
    tokens = SessionTokens(clock=clock)
    bootstrap = tokens.issue_bootstrap()
    assert tokens.bootstrap_pending(bootstrap)
    tokens.exchange(bootstrap)
    assert not tokens.bootstrap_pending(bootstrap)
    with pytest.raises(UnauthorizedError) as reused:
        tokens.exchange(bootstrap)
    assert reused.value.code == "bootstrap_invalid"
    late = tokens.issue_bootstrap(ttl_s=600.0)
    clock.now += 600.0
    assert not tokens.bootstrap_pending(late)
    with pytest.raises(UnauthorizedError):
        tokens.exchange(late)


def test_tokens_are_43_url_safe_characters_and_distinct() -> None:
    tokens = SessionTokens()
    issued = {tokens.issue_bootstrap() for _ in range(50)}
    assert len(issued) == 50
    for token in issued:
        assert len(token) == 43
        assert set(token) <= set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )


# --- Tickets (§7.2) -----------------------------------------------------------

LINE = "/api/v1/artifact/line"


def ticket_request(ticket: str, **query: str) -> HttpRequest:
    pairs = {name: (value,) for name, value in query.items()}
    pairs["ticket"] = (ticket,)
    return request(LINE, query=pairs)


def ticket_route() -> Route:
    return route("bearer_or_ticket", path=LINE)


def test_a_ticket_works_once_for_its_route_and_query() -> None:
    tokens = SessionTokens()
    query = canonical_query([("line", "4"), ("artifact", "runs/a.jsonl")])
    ticket = tokens.issue_ticket(route=LINE, query=query)
    good = ticket_request(ticket, line="4", artifact="runs/a.jsonl")
    check_request(good, ticket_route(), tokens)
    assert code_of(lambda: check_request(good, ticket_route(), tokens)) == (
        401,
        "ticket_invalid",
    )


def test_a_ticket_expires_after_30_s() -> None:
    clock = Clock()
    tokens = SessionTokens(clock=clock)
    ticket = tokens.issue_ticket(route=LINE, query="line=1")
    clock.now += 30.0
    late = ticket_request(ticket, line="1")
    assert code_of(lambda: check_request(late, ticket_route(), tokens))[1] == (
        "ticket_invalid"
    )


@pytest.mark.parametrize(
    "path, query",
    [
        ("/api/v1/artifact/export", {"line": "1"}),
        (LINE, {"line": "2"}),
        (LINE, {"line": "1", "extra": ""}),
    ],
)
def test_a_ticket_is_bound_to_its_route_and_query(
    path: str, query: dict[str, str]
) -> None:
    tokens = SessionTokens()
    ticket = tokens.issue_ticket(route=LINE, query="line=1")
    pairs = {name: (value,) for name, value in query.items()}
    pairs["ticket"] = (ticket,)
    other = request(path, query=pairs)
    other_route = route("bearer_or_ticket", path=path)
    assert code_of(lambda: check_request(other, other_route, tokens))[1] == (
        "ticket_invalid"
    )
    # A mismatched use spends it too.
    good = ticket_request(ticket, line="1")
    assert code_of(lambda: check_request(good, ticket_route(), tokens))[0] == 401


def test_a_ticket_from_another_instance_is_refused() -> None:
    ticket = SessionTokens().issue_ticket(route=LINE, query="line=1")
    other = ticket_request(ticket, line="1")
    assert code_of(lambda: check_request(other, ticket_route(), SessionTokens()))[
        1
    ] == ("ticket_invalid")


def test_tickets_are_refused_on_bearer_routes() -> None:
    tokens = SessionTokens()
    ticket = tokens.issue_ticket(route=LINE, query="")
    only_bearer = route("bearer", path=LINE)
    assert code_of(
        lambda: check_request(ticket_request(ticket), only_bearer, tokens)
    ) == (401, "unauthorized")


def test_repeated_ticket_parameters_are_refused() -> None:
    tokens = SessionTokens()
    ticket = tokens.issue_ticket(route=LINE, query="")
    twice = request(LINE, query={"ticket": (ticket, ticket)})
    assert code_of(lambda: check_request(twice, ticket_route(), tokens))[1] == (
        "ticket_invalid"
    )


def test_canonical_query_sorts_encodes_and_drops_the_ticket() -> None:
    pairs = [("b", "2"), ("a", "x/y z"), ("ticket", "t"), ("a", "1")]
    assert canonical_query(pairs) == "a=1&a=x%2Fy%20z&b=2"
    assert canonical_query([]) == ""


def test_without_ticket_hides_the_ticket_from_the_handler() -> None:
    original = request(query={"ticket": ("t",), "line": ("1",)})
    assert dict(without_ticket(original).query) == {"line": ("1",)}
    plain = request(query={"line": ("1",)})
    assert without_ticket(plain) is plain


# --- POST rules (§7.3 step 5) -------------------------------------------------


def test_post_needs_json_and_an_origin() -> None:
    tokens = SessionTokens()
    token = session_for(tokens)
    post_route = route("bearer", "POST")
    origin = {"Origin": "http://127.0.0.1:8790"}
    ok = {**bearer(token), **origin, "Content-Type": "application/json; charset=utf-8"}
    check_request(request(method="POST", headers=ok), post_route, tokens)
    for content_type in ("text/plain", "application/x-www-form-urlencoded", ""):
        bad = request(method="POST", headers={**ok, "Content-Type": content_type})
        assert code_of(lambda: check_request(bad, post_route, tokens)) == (
            415,
            "unsupported_media_type",
        )
    no_origin = {k: v for k, v in ok.items() if k != "Origin"}
    bare = request(method="POST", headers=no_origin)
    assert code_of(lambda: check_request(bare, post_route, tokens)) == (
        403,
        "origin_not_allowed",
    )


def test_local_error_classes_carry_the_schema_pairs() -> None:
    pairs = {
        MethodNotAllowed: (405, None),
        BodyTooLarge: (413, None),
        UnsupportedMediaType: (415, None),
        HeadTooLarge: (431, None),
        MalformedRequest: (400, "usage"),
    }
    for cls, (status, verdict) in pairs.items():
        document = cls("some_code", "message").document()
        validate(document, "error")
        assert (cls.http_status, cls.verdict_status) == (status, verdict)


# --- The workspace root (§7.4) ------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> RootDirectory:
    root = tmp_path / "root"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "a.jsonl").write_text("{}\n")
    (tmp_path / "secret.jsonl").write_text("secret\n")
    return RootDirectory(root)


@pytest.mark.parametrize(
    "rel",
    [
        "../secret.jsonl",
        "runs/../../secret.jsonl",
        "runs/../runs/a.jsonl",
        "/etc/passwd",
        "runs\\a.jsonl",
        "runs/a.jsonl\x00",
        "C:/runs/a.jsonl",
        "",
        "./runs/a.jsonl",
        "runs//a.jsonl",
        "runs/",
    ],
)
def test_traversal_is_refused(workspace: RootDirectory, rel: str) -> None:
    with pytest.raises(ForbiddenError) as caught:
        workspace.resolve(rel)
    assert caught.value.code == "path_not_allowed"


def test_percent_encoding_is_not_decoded_twice(workspace: RootDirectory) -> None:
    # httpd decodes once; a literal "%2e%2e" is a file name, not "..".
    with pytest.raises(NotFoundError):
        workspace.resolve("%2e%2e/secret.jsonl")


def test_a_symlink_out_of_the_root_is_refused(
    workspace: RootDirectory, tmp_path: Path
) -> None:
    (workspace.path / "out.jsonl").symlink_to(tmp_path / "secret.jsonl")
    (workspace.path / "outdir").symlink_to(tmp_path)
    for rel in ("out.jsonl", "outdir/secret.jsonl"):
        with pytest.raises(ForbiddenError):
            workspace.resolve(rel)
        with pytest.raises(ForbiddenError):
            workspace.open(rel)


def test_a_symlink_inside_the_root_resolves(workspace: RootDirectory) -> None:
    (workspace.path / "link.jsonl").symlink_to(workspace.path / "runs" / "a.jsonl")
    assert workspace.resolve("link.jsonl") == workspace.path / "runs" / "a.jsonl"
    with workspace.open("link.jsonl") as handle:
        assert handle.read() == b"{}\n"


def test_missing_files_directories_and_fifos_are_not_found(
    workspace: RootDirectory,
) -> None:
    os.mkfifo(workspace.path / "pipe.jsonl")
    for rel in ("nope.jsonl", "runs", "pipe.jsonl"):
        with pytest.raises(NotFoundError) as caught:
            workspace.resolve(rel)
        assert caught.value.code == "artifact_not_found"


def test_open_reads_a_regular_file(workspace: RootDirectory) -> None:
    with workspace.open("runs/a.jsonl") as handle:
        assert handle.read() == b"{}\n"


@pytest.mark.parametrize("swap", ["symlink", "other_file"])
def test_a_file_swapped_after_the_check_is_refused(
    workspace: RootDirectory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    target = workspace.path / "runs" / "a.jsonl"
    real_open = os.open

    def swapping_open(path: Any, flags: int, *args: Any) -> int:
        monkeypatch.setattr(os, "open", real_open)
        # Made before the old file goes, so its inode number differs.
        replacement = tmp_path / "replacement"
        if swap == "symlink":
            replacement.symlink_to(tmp_path / "secret.jsonl")
        else:
            replacement.write_text("swapped\n")
        os.replace(replacement, target)
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(ForbiddenError):
        workspace.open("runs/a.jsonl")


def test_relative_gives_root_relative_posix_paths(
    workspace: RootDirectory, tmp_path: Path
) -> None:
    assert workspace.relative(workspace.path / "runs" / "a.jsonl") == "runs/a.jsonl"
    with pytest.raises(ForbiddenError):
        workspace.relative(tmp_path / "secret.jsonl")
    with pytest.raises(ForbiddenError):
        workspace.relative(workspace.path)


def test_the_root_must_be_a_directory(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("")
    with pytest.raises(NotADirectoryError):
        RootDirectory(tmp_path / "file")


def test_root_and_home_are_refused_unless_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert refuse_broad_root(Path("/"), named=False) == "the filesystem root"
    assert refuse_broad_root(tmp_path, named=False) == "your home directory"
    assert refuse_broad_root(Path("/"), named=True) is None
    assert refuse_broad_root(tmp_path, named=True) is None
    (tmp_path / "work").mkdir()
    assert refuse_broad_root(tmp_path / "work", named=False) is None


# --- The auth routes and their documents ---------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[StudioContext]:
    (tmp_path / "root").mkdir()
    context = stub_context(launch_options(tmp_path / "root", tmp_path / "state"))
    yield context
    context.pool.shutdown(wait=False)


def handlers(ctx: StudioContext) -> dict[str, Route]:
    return {r.path: r for r in auth_routes(ctx)}


def body_json(response: HttpResponse) -> Any:
    assert isinstance(response.body, bytes)
    return json.loads(response.body)


def json_request(path: str, body: bytes) -> HttpRequest:
    return request(path, method="POST", body=body)


def test_auth_routes_are_the_four_c3_routes(ctx: StudioContext) -> None:
    table = {(r.method, r.path): r.auth for r in auth_routes(ctx)}
    assert table == {
        ("POST", "/api/v1/session"): "public",
        ("GET", "/api/v1/health"): "public",
        ("POST", "/api/v1/tickets"): "bearer",
        ("POST", "/api/v1/admin/bootstrap"): "admin",
    }


def test_documents_validate_against_their_schemas(ctx: StudioContext) -> None:
    table = handlers(ctx)
    health = table["/api/v1/health"].handler(request("/api/v1/health"))
    validate(body_json(health), "health")
    bootstrap = ctx.tokens.issue_bootstrap()
    session = table["/api/v1/session"].handler(
        json_request("/api/v1/session", json.dumps({"bootstrap": bootstrap}).encode())
    )
    session_doc = body_json(session)
    validate(session_doc, "session")
    assert session_doc["instance_id"] == INSTANCE_ID
    ticket = table["/api/v1/tickets"].handler(
        json_request(
            "/api/v1/tickets",
            b'{"route": "/api/v1/artifact/export", "query": "phase=measured&artifact=r"}',
        )
    )
    ticket_doc = body_json(ticket)
    validate(ticket_doc, "ticket")
    assert ticket_doc["url"].startswith(
        "/api/v1/artifact/export?artifact=r&phase=measured&ticket="
    )
    assert ticket_doc["expires_in_s"] == 30.0
    for route_value in (None, "/requests?artifact=runs%2F042.jsonl&case=c8"):
        body = json.dumps({"route": route_value}).encode()
        made = table["/api/v1/admin/bootstrap"].handler(
            json_request("/api/v1/admin/bootstrap", body)
        )
        validate(body_json(made), "bootstrap")


def test_ticket_issue_refuses_bad_routes_and_queries(ctx: StudioContext) -> None:
    tickets = handlers(ctx)["/api/v1/tickets"].handler
    for body in (
        b'{"route": "/elsewhere"}',
        b'{"route": "/api/v1/artifact/line", "query": "ticket=x"}',
        b'{"route": "/api/v1/artifact/line", "query": "a"}',
        b'{"route": "/api/v1/artifact/line", "query": 3}',
        b'{"route": "/api/v1/artifact/line", "other": 1}',
        b"[]",
        b"not json",
    ):
        with pytest.raises(UsageError):
            tickets(json_request("/api/v1/tickets", body))


def test_bootstrap_route_refuses_bad_routes(ctx: StudioContext) -> None:
    make = handlers(ctx)["/api/v1/admin/bootstrap"].handler
    for value in ("requests", "/a#b", "/" + "a" * 3000, "/a\nb", 5):
        body = ("{" + f'"route": {json_value(value)}' + "}").encode()
        with pytest.raises(UsageError):
            make(json_request("/api/v1/admin/bootstrap", body))


def json_value(value: Any) -> str:
    return json.dumps(value)


def test_launch_url_carries_the_token_in_the_fragment() -> None:
    url = launch_url("http://127.0.0.1:8790", "T" * 43, "/requests?artifact=a/b")
    assert url == (
        "http://127.0.0.1:8790/#/auth?t="
        + "T" * 43
        + "&route=%2Frequests%3Fartifact%3Da%2Fb"
    )
    assert launch_url("http://[::1]:1", "T", None) == "http://[::1]:1/#/auth?t=T"


def test_security_module_is_exported() -> None:
    for name in security.__all__:
        assert hasattr(security, name), name
