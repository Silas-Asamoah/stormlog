"""Studio's request checks, tokens, workspace root and auth routes (§7).

Every request passes :func:`check_request` before its handler runs, in the
order of design §7.3: the Host name, then Origin, then Fetch Metadata (on
``/api/`` only), then the route's auth, then the POST rules. A failed check
raises a :class:`~stormlog.studio.contracts.StudioError`, which httpd
renders as a ``stormlog.studio.error`` document.

Credentials (§7.2): a bootstrap token from the launcher, traded once for a
session token; an admin token for ``stormlog studio url``; and single-use
tickets for the two routes a browser reaches by navigation. Every token is
32 random bytes, base64url, and is kept only as its SHA-256 digest, so a
lookup compares digests, never the secret, and the final comparison is
constant time.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal
from urllib.parse import parse_qsl, quote, urlencode

from stormlog.studio.contracts import (
    ForbiddenError,
    HttpRequest,
    HttpResponse,
    NotFoundError,
    Route,
    StudioContext,
    StudioError,
    TokenStore,
    UnauthorizedError,
    UsageError,
    encode_json,
    envelope,
)

__all__ = [
    "LOOPBACK_NAMES",
    "SESSION_IDLE_S",
    "BOOTSTRAP_TTL_S",
    "TICKET_TTL_S",
    "TICKET_PARAM",
    "JSON_TYPE",
    "MethodNotAllowed",
    "BodyTooLarge",
    "UnsupportedMediaType",
    "HeadTooLarge",
    "MalformedRequest",
    "host_name",
    "check_request",
    "canonical_query",
    "query_pairs",
    "without_ticket",
    "json_body",
    "auth_routes",
    "launch_url",
    "RootDirectory",
    "refuse_broad_root",
    "SessionTokens",
]

# The Host names Studio answers to, with any port (§7.3 step 1).
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "[::1]"})
# Lead's default: a session token dies after 12 hours without a request.
SESSION_IDLE_S = 12 * 3600.0
BOOTSTRAP_TTL_S = 600.0
TICKET_TTL_S = 30.0
TICKET_PARAM = "ticket"
JSON_TYPE = "application/json; charset=utf-8"

_TOKEN_BYTES = 32
_PORT = re.compile(r"[0-9]{1,5}")
_DRIVE = re.compile(r"[A-Za-z]:")
_TICKET_ROUTE = re.compile(r"/api/v1/[a-z0-9_/]+")
_ROUTE_MAX_CHARS = 2048
_API_PREFIX = "/api/"


# --- Errors httpd raises that the contract has no class for (§5.2) -----------


class MethodNotAllowed(StudioError):
    """405: the path exists for another method, or the method is never served."""

    http_status = 405
    verdict_status = None


class BodyTooLarge(StudioError):
    """413: a body past ``Limits.body_bytes``."""

    http_status = 413
    verdict_status = None


class UnsupportedMediaType(StudioError):
    """415: a POST whose Content-Type is not application/json (§7.3 step 5)."""

    http_status = 415
    verdict_status = None


class HeadTooLarge(StudioError):
    """431: a request line and headers past ``Limits.head_bytes``."""

    http_status = 431
    verdict_status = None


class MalformedRequest(UsageError):
    """400: a request line, header or query string httpd cannot parse."""


# --- Request checks (§7.3) ---------------------------------------------------


def host_name(host: str) -> str | None:
    """The lower-cased name of a Host value with a valid port, else None.

    ``[::1]:8790`` gives ``[::1]``; ``localhost`` gives ``localhost``. A
    port must be 1 to 5 digits and at most 65535.
    """
    host = host.strip().lower()
    if host.startswith("["):
        name, bracket, rest = host.partition("]")
        name += bracket
        port = rest[1:] if rest.startswith(":") else None
        if rest and port is None:
            return None
    else:
        name, colon, port_text = host.partition(":")
        port = port_text if colon else None
    if port is not None and not (_PORT.fullmatch(port) and int(port) <= 65535):
        return None
    return name


def check_request(
    request: HttpRequest, route: Route | None, tokens: TokenStore
) -> None:
    """Run §7.3's checks in order; raise a StudioError on the first failure.

    With ``route`` None (a static file, or a path no route serves) only the
    checks that apply to every request run: Host, Origin and, under
    ``/api/``, Fetch Metadata. For a ``bearer_or_ticket`` route that names
    a ``ticket`` parameter, the ticket is redeemed here, so it is spent
    whatever the handler does next.
    """
    _check_host(request)
    _check_origin(request)
    _check_fetch_site(request)
    if route is None:
        return
    _check_auth(request, route, tokens)
    _check_post(request, route)


def _check_host(request: HttpRequest) -> None:
    host = request.headers.get("host")
    if host is None or host_name(host) not in LOOPBACK_NAMES:
        raise ForbiddenError(
            "host_not_allowed",
            "Studio answers only to 127.0.0.1, localhost or [::1]",
        )


def _check_origin(request: HttpRequest) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    expected = "http://" + request.headers.get("host", "").strip().lower()
    if origin.strip().lower() != expected:
        raise ForbiddenError(
            "origin_not_allowed", "the request's Origin is not this Studio"
        )


def _check_fetch_site(request: HttpRequest) -> None:
    if not request.path.startswith(_API_PREFIX):
        return
    site = request.headers.get("sec-fetch-site")
    if site is not None and site.strip().lower() not in ("same-origin", "none"):
        raise ForbiddenError(
            "fetch_site_not_allowed",
            "the API answers only Studio's own pages",
            details={"sec_fetch_site": site[:32]},
        )


def _check_auth(request: HttpRequest, route: Route, tokens: TokenStore) -> None:
    if route.auth == "public":
        return
    if route.auth == "admin":
        _check_admin(request, tokens)
    elif route.auth == "bearer_or_ticket" and TICKET_PARAM in request.query:
        _check_ticket(request, route, tokens)
    elif not tokens.check(_bearer(request), "session"):
        raise UnauthorizedError(
            "unauthorized",
            "a live session token is required; run `stormlog studio url`",
        )


def _check_admin(request: HttpRequest, tokens: TokenStore) -> None:
    if "origin" in request.headers:
        raise ForbiddenError(
            "origin_not_allowed", "admin routes refuse browser requests"
        )
    if any(name.startswith("sec-fetch-") for name in request.headers):
        raise ForbiddenError(
            "fetch_site_not_allowed", "admin routes refuse browser requests"
        )
    if not tokens.check(_bearer(request), "admin"):
        raise UnauthorizedError("unauthorized", "the admin token is required")


def _check_ticket(request: HttpRequest, route: Route, tokens: TokenStore) -> None:
    values = request.query[TICKET_PARAM]
    query = canonical_query(query_pairs(request.query))
    if len(values) != 1 or not tokens.redeem_ticket(
        values[0], route=route.path, query=query
    ):
        raise UnauthorizedError(
            "ticket_invalid",
            "this link was for one use within 30 s; open it again from Studio",
        )


def _check_post(request: HttpRequest, route: Route) -> None:
    if route.method != "POST":
        return
    content_type = request.headers.get("content-type", "")
    media = content_type.split(";", 1)[0].strip().lower()
    if media != "application/json":
        raise UnsupportedMediaType(
            "unsupported_media_type",
            "POST bodies must be application/json",
            details={"content_type": media[:64], "allowed": ["application/json"]},
        )
    if route.auth != "admin" and "origin" not in request.headers:
        raise ForbiddenError("origin_not_allowed", "a POST must carry an Origin")


def _bearer(request: HttpRequest) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


# --- Queries and bodies ------------------------------------------------------


def query_pairs(query: Mapping[str, Iterable[str]]) -> list[tuple[str, str]]:
    """``HttpRequest.query`` as name and value pairs."""
    return [(name, value) for name, values in query.items() for value in values]


def canonical_query(pairs: Iterable[tuple[str, str]]) -> str:
    """Pairs sorted by name and value and re-encoded, ``ticket`` removed.

    The ticket check needs no filter semantics: the same canonical form at
    issue and at use is enough (§7.2). Every reserved character, ``/``
    included, is percent-encoded.
    """
    kept = sorted((name, value) for name, value in pairs if name != TICKET_PARAM)
    return urlencode(kept, quote_via=quote)


def without_ticket(request: HttpRequest) -> HttpRequest:
    """The request with its ``ticket`` parameter removed, for the handler."""
    if TICKET_PARAM not in request.query:
        return request
    query = {k: v for k, v in request.query.items() if k != TICKET_PARAM}
    return HttpRequest(
        method=request.method,
        path=request.path,
        query=query,
        headers=request.headers,
        body=request.body,
    )


def json_body(request: HttpRequest, allowed: Iterable[str]) -> dict[str, Any]:
    """The body as a JSON object with only ``allowed`` keys; UsageError else."""
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, ValueError):
        raise UsageError("bad_parameter", "the body is not JSON") from None
    if not isinstance(body, dict):
        raise UsageError("bad_parameter", "the body is not a JSON object")
    unknown = sorted(set(body) - set(allowed))
    if unknown:
        raise UsageError(
            "unknown_parameter",
            f"unknown body key: {unknown[0][:64]}",
            field=unknown[0][:64],
            details={"allowed": sorted(allowed)},
        )
    return body


def _string(body: Mapping[str, Any], key: str, *, required: bool) -> str | None:
    value = body.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise UsageError("bad_parameter", f"{key}: a string is required", field=key)
    return value


# --- Auth routes (§5.4) ------------------------------------------------------


def auth_routes(ctx: StudioContext) -> list[Route]:
    """The four routes C3 owns: session, health, tickets and admin bootstrap."""
    return [
        Route("POST", "/api/v1/session", "public", _session_handler(ctx)),
        Route("GET", "/api/v1/health", "public", _health_handler(ctx)),
        Route("POST", "/api/v1/tickets", "bearer", _tickets_handler(ctx)),
        Route("POST", "/api/v1/admin/bootstrap", "admin", _bootstrap_handler(ctx)),
    ]


def _document(name: str, body: Mapping[str, Any]) -> HttpResponse:
    return HttpResponse(200, JSON_TYPE, encode_json(envelope(name, body)))


Handler = Callable[[HttpRequest], HttpResponse]


def _session_handler(ctx: StudioContext) -> Handler:
    def handle(request: HttpRequest) -> HttpResponse:
        body = json_body(request, ("bootstrap",))
        bootstrap = _string(body, "bootstrap", required=True) or ""
        token = ctx.tokens.exchange(bootstrap)
        return _document(
            "session", {"session_token": token, "instance_id": ctx.instance_id}
        )

    return handle


def _health_handler(ctx: StudioContext) -> Handler:
    def handle(request: HttpRequest) -> HttpResponse:
        return _document("health", {"ok": True, "instance_id": ctx.instance_id})

    return handle


def _tickets_handler(ctx: StudioContext) -> Handler:
    def handle(request: HttpRequest) -> HttpResponse:
        body = json_body(request, ("route", "query"))
        route = _string(body, "route", required=True) or ""
        if not _TICKET_ROUTE.fullmatch(route):
            raise UsageError(
                "bad_parameter", "route: an /api/v1/ path is required", field="route"
            )
        query = canonical_query(_parse_query(body.get("query") or ""))
        ticket = ctx.tokens.issue_ticket(route=route, query=query, ttl_s=TICKET_TTL_S)
        joined = f"{query}&" if query else ""
        return _document(
            "ticket",
            {
                "ticket": ticket,
                "url": f"{route}?{joined}{TICKET_PARAM}={ticket}",
                "expires_in_s": TICKET_TTL_S,
            },
        )

    return handle


def _parse_query(text: Any) -> list[tuple[str, str]]:
    if not isinstance(text, str):
        raise UsageError("bad_parameter", "query: a string is required", field="query")
    try:
        pairs = parse_qsl(text, keep_blank_values=True, strict_parsing=bool(text))
    except ValueError:
        raise UsageError(
            "bad_parameter", "query: not a query string", field="query"
        ) from None
    if any(name == TICKET_PARAM for name, _ in pairs):
        raise UsageError(
            "bad_parameter", "query: must not carry a ticket", field="query"
        )
    return pairs


def _bootstrap_handler(ctx: StudioContext) -> Handler:
    def handle(request: HttpRequest) -> HttpResponse:
        body = json_body(request, ("route",))
        route = _string(body, "route", required=False)
        if route is not None and not _valid_route(route):
            raise UsageError(
                "bad_parameter",
                "route: a Studio page such as /requests?artifact=... is required",
                field="route",
            )
        token = ctx.tokens.issue_bootstrap(ttl_s=BOOTSTRAP_TTL_S)
        return _document("bootstrap", {"url": launch_url(_base(request), token, route)})

    return handle


def _valid_route(route: str) -> bool:
    return (
        route.startswith("/")
        and len(route) <= _ROUTE_MAX_CHARS
        and "#" not in route
        and route.isprintable()
    )


def _base(request: HttpRequest) -> str:
    # check_request has already held Host to a loopback name and port.
    host = request.headers["host"].strip().lower()
    if not host.endswith("]") and ":" in host.rpartition("]")[2]:
        return f"http://{host}"
    return f"http://{host}:80"


def launch_url(base_url: str, bootstrap: str, route: str | None) -> str:
    """``<base>/#/auth?t=<bootstrap>``, then ``&route=<route>`` when given.

    The token rides in the fragment, which no request line, log or Referer
    carries; the page sends it once to ``POST /api/v1/session``.
    """
    url = f"{base_url}/#/auth?t={bootstrap}"
    if route:
        url += "&route=" + quote(route, safe="")
    return url


# --- The workspace root (§7.4) -----------------------------------------------


def refuse_broad_root(path: Path, *, named: bool) -> str | None:
    """Why ``path`` may not be the root, or None when it may (decision 11).

    ``/`` and ``$HOME`` make every JSONL under them readable through the
    API, so Studio takes them only when ``--root`` names them (``named``).
    """
    if named:
        return None
    real = os.path.realpath(path)
    if real == os.path.realpath(os.sep):
        return "the filesystem root"
    home = os.environ.get("HOME") or os.path.expanduser("~")
    if home and real == os.path.realpath(home):
        return "your home directory"
    return None


class RootDirectory:
    """The one directory Studio reads; implements ``WorkspaceRoot`` (§7.4)."""

    def __init__(self, path: Path) -> None:
        real = Path(os.path.realpath(path))
        if not real.is_dir():
            raise NotADirectoryError(str(path))
        self.path = real

    def resolve(self, rel: str) -> Path:
        """The real path of a regular file under the root.

        ForbiddenError ``path_not_allowed`` for an absolute path, a drive
        letter, a backslash, a NUL, an empty, ``.`` or ``..`` segment, or a
        path whose real path leaves the root (a symlink out);
        NotFoundError ``artifact_not_found`` when nothing regular is there,
        including a path that changes while ``realpath`` reads it (before
        Python 3.13 it can raise when a symlink on the way goes).
        """
        _check_relative(rel)
        try:
            real = Path(os.path.realpath(self.path / rel))
        except OSError:
            raise _not_found(rel) from None
        if not _is_under(real, self.path):
            raise _not_allowed(rel)
        try:
            mode = os.stat(real).st_mode
        except OSError:
            raise _not_found(rel) from None
        if not stat.S_ISREG(mode):
            raise _not_found(rel)
        return real

    def open(self, rel: str) -> BinaryIO:
        """Resolve, open read-only, and check the open file is the one resolved.

        ``O_NOFOLLOW`` refuses a final component swapped for a symlink and
        ``O_NONBLOCK`` keeps a swapped-in FIFO from blocking the open; the
        ``fstat`` then requires a regular file with the resolved device and
        inode, so a swap between check and open is refused.
        """
        real = self.resolve(rel)
        try:
            expected = os.stat(real)
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            fd = os.open(real, flags)
        except OSError:
            raise _not_allowed(rel) from None
        actual = os.fstat(fd)
        same = (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
        if not (same and stat.S_ISREG(actual.st_mode)):
            os.close(fd)
            raise _not_allowed(rel)
        os.set_blocking(fd, True)
        return os.fdopen(fd, "rb")

    def relative(self, path: Path) -> str:
        """The root-relative POSIX path of ``path``; ForbiddenError outside."""
        real = Path(os.path.realpath(path))
        if not _is_under(real, self.path) or real == self.path:
            raise _not_allowed(str(path))
        return real.relative_to(self.path).as_posix()


def _check_relative(rel: str) -> None:
    bad = (
        not rel
        or rel.startswith("/")
        or "\\" in rel
        or "\x00" in rel
        or bool(_DRIVE.match(rel))
    )
    if bad or any(part in ("", ".", "..") for part in rel.split("/")):
        raise _not_allowed(rel)


def _is_under(path: Path, root: Path) -> bool:
    return os.path.commonpath([str(path), str(root)]) == str(root)


def _not_allowed(rel: str) -> ForbiddenError:
    return ForbiddenError(
        "path_not_allowed",
        "the path is not a file under the workspace root",
        details={"path": rel[:256]},
    )


def _not_found(rel: str) -> NotFoundError:
    return NotFoundError(
        "artifact_not_found",
        "no regular file at this path under the workspace root",
        details={"path": rel[:256]},
    )


# --- Tokens (§7.2) -----------------------------------------------------------


@dataclass
class _Ticket:
    route: str
    query: str
    expires_at: float


class SessionTokens:
    """Bootstrap, session, admin and ticket tokens; implements ``TokenStore``.

    Everything is held in memory, so every token dies with its instance. A
    session token also dies after ``idle_s`` (12 hours) without a check. A
    bootstrap is single-use and lives ``ttl_s`` (10 minutes); a ticket is
    single-use, lives 30 s, and is bound to one route and canonical query.
    Tokens are stored as SHA-256 digests.
    """

    def __init__(
        self,
        *,
        idle_s: float = SESSION_IDLE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle_s = idle_s
        self._clock = clock
        self._lock = threading.Lock()
        self._admin: str | None = None
        self._bootstraps: dict[str, float] = {}
        self._sessions: dict[str, float] = {}
        self._tickets: dict[str, _Ticket] = {}

    def issue_admin(self) -> str:
        """The admin token, once; a second call raises RuntimeError."""
        with self._lock:
            if self._admin is not None:
                raise RuntimeError("the admin token was already issued")
            token = _new_token()
            self._admin = _digest(token)
            return token

    def issue_bootstrap(self, *, ttl_s: float = BOOTSTRAP_TTL_S) -> str:
        token = _new_token()
        with self._lock:
            now = self._clock()
            _drop_expired(self._bootstraps, now)
            self._bootstraps[_digest(token)] = now + ttl_s
        return token

    def bootstrap_pending(self, bootstrap: str) -> bool:
        """Whether ``bootstrap`` is live and not yet exchanged."""
        with self._lock:
            expires = self._bootstraps.get(_digest(bootstrap))
            return expires is not None and self._clock() < expires

    def exchange(self, bootstrap: str) -> str:
        """A session token for a live bootstrap, which is spent."""
        with self._lock:
            now = self._clock()
            expires = self._bootstraps.pop(_digest(bootstrap), None)
            if expires is None or now >= expires:
                raise UnauthorizedError(
                    "bootstrap_invalid",
                    "this launch link was used or expired; "
                    "run `stormlog studio url` for a new one",
                )
            _drop_idle(self._sessions, now - self.idle_s)
            token = _new_token()
            self._sessions[_digest(token)] = now
            return token

    def check(self, token: str, kind: Literal["session", "admin"]) -> bool:
        """Whether ``token`` is live; a session check renews its idle time."""
        if not token:
            return False
        digest = _digest(token)
        with self._lock:
            if kind == "admin":
                return self._admin is not None and hmac.compare_digest(
                    self._admin, digest
                )
            return self._renew(digest)

    def _renew(self, digest: str) -> bool:
        now = self._clock()
        last = self._sessions.get(digest)
        if last is None:
            return False
        if now - last >= self.idle_s:
            del self._sessions[digest]
            return False
        self._sessions[digest] = now
        return True

    def issue_ticket(
        self, *, route: str, query: str, ttl_s: float = TICKET_TTL_S
    ) -> str:
        token = _new_token()
        with self._lock:
            now = self._clock()
            expired = [k for k, t in self._tickets.items() if now >= t.expires_at]
            for key in expired:
                del self._tickets[key]
            self._tickets[_digest(token)] = _Ticket(route, query, now + ttl_s)
        return token

    def redeem_ticket(self, ticket: str, *, route: str, query: str) -> bool:
        """True once for a live ticket of this route and query; spent either way."""
        with self._lock:
            held = self._tickets.pop(_digest(ticket), None)
            if held is None or self._clock() >= held.expires_at:
                return False
        # Route and query are not secrets; the ticket was found by digest.
        return held.route == route and held.query == query


def _new_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8", "surrogatepass")).hexdigest()


def _drop_expired(expiries: dict[str, float], now: float) -> None:
    for key in [k for k, expires in expiries.items() if now >= expires]:
        del expiries[key]


def _drop_idle(last_used: dict[str, float], cutoff: float) -> None:
    for key in [k for k, last in last_used.items() if last <= cutoff]:
        del last_used[key]
