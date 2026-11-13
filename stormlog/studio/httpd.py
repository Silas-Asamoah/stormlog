"""Studio's HTTP server: stdlib ``http.server``, hardened (design §2.2, §8.1).

It follows ``stormlog._export``'s ``MetricsServer``. Admission happens on
the accept thread: past ``Limits.max_connections`` a connection gets a
canned 503 ``busy`` document and is closed. Each admitted connection serves
one request (HTTP/1.1, ``Connection: close``) under a watchdog deadline of
``Limits.deadline_s`` from accept; once the head is parsed and the route is
known, the deadline is re-armed with the route's own (``Route.deadline_s``),
so an export can stream for 600 s while a client that dribbles its head is
still cut at 30 s. A head past ``Limits.head_bytes`` gets 431 and a body
past ``Limits.body_bytes`` gets 413, before either is held in memory.

Every request then passes :func:`stormlog.studio.security.check_request`.
A handler's :class:`StudioError` becomes a ``stormlog.studio.error``
document with a scrubbed message; :class:`NotReady` becomes 202 with its
``state`` verbatim and ``Retry-After``; anything else becomes a generic 500
(the traceback goes to stderr only with ``debug``). Request logging is off
unless ``log_requests``, and then names the route template, never a query,
header or body (§7.6).
"""

from __future__ import annotations

import http.client
import socket
import sys
import threading
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, TextIO, cast
from urllib.parse import parse_qs, unquote, urlsplit

from stormlog._export.watchdog import Watchdog, WatchdogStopped
from stormlog.scrub import scrub_text
from stormlog.studio.contracts import (
    BusyError,
    HttpRequest,
    HttpResponse,
    Limits,
    NotFoundError,
    NotReady,
    Route,
    StudioError,
    TokenStore,
    encode_json,
)
from stormlog.studio.security import (
    JSON_TYPE,
    BodyTooLarge,
    HeadTooLarge,
    MalformedRequest,
    MethodNotAllowed,
    check_request,
    without_ticket,
)
from stormlog.studio.static_files import StaticFiles

__all__ = ["StudioServer", "MESSAGE_MAX_BYTES", "render_error", "SECURITY_HEADERS"]

# §7.6: error messages are scrubbed and cut to this many bytes.
MESSAGE_MAX_BYTES = 512
# §7.5: on every response.
SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)
# API answers are data, never pages: nothing may run or frame them.
_API_CSP = "default-src 'none'; frame-ancestors 'none'; sandbox"
_SINGLE_HEADERS = frozenset(
    {
        "host",
        "origin",
        "authorization",
        "content-type",
        "content-length",
        "sec-fetch-site",
        "transfer-encoding",
    }
)
_TICKET_NOTE = (
    b"This Studio link was for one use within 30 seconds and has been used "
    b"or has expired.\nOpen it again from Studio.\n"
)


class StudioServer:
    """Serve Studio's routes and static files on one loopback address."""

    def __init__(
        self,
        routes: Sequence[Route],
        tokens: TokenStore,
        static: StaticFiles,
        *,
        host: str,
        port: int,
        limits: Limits,
        log_requests: bool = False,
        debug: bool = False,
        stderr: TextIO | None = None,
    ) -> None:
        self.routes: dict[tuple[str, str], Route] = {}
        for route in routes:
            key = (route.method, route.path)
            if key in self.routes:
                raise ValueError(f"two routes for {route.method} {route.path}")
            self.routes[key] = route
        self.paths = frozenset(path for _, path in self.routes)
        self.tokens = tokens
        self.static = static
        self.host = host
        self.port = port
        self.limits = limits
        self.log_requests = log_requests
        self.debug = debug
        self.stderr = stderr
        self.watchdog = Watchdog(name="stormlog-studio-watchdog")
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """``http://<host>:<bound port>``; IPv6 hosts are bracketed."""
        server = self._server
        port = server.server_address[1] if server is not None else self.port
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{port}"

    @property
    def bound_port(self) -> int:
        server = self._server
        return int(server.server_address[1]) if server is not None else self.port

    def start(self) -> None:
        """Bind and serve; an ``OSError`` from binding reaches the caller."""
        self._server = _Server((self.host, self.port), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="stormlog-studio-httpd",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting, cut open connections and stop the watchdog."""
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        server.close_connections()
        if self._thread is not None:
            self._thread.join(timeout)
        self.watchdog.stop()
        self._server = None

    def log(self, text: str) -> None:
        stream = self.stderr or sys.stderr
        print(f"stormlog studio: {text}", file=stream, flush=True)


def render_error(error: StudioError) -> HttpResponse:
    """The ``stormlog.studio.error`` response, its message scrubbed (§7.6)."""
    document = error.document()
    body = document["error"]
    body["message"] = scrub_text(str(body["message"]), max_bytes=MESSAGE_MAX_BYTES)
    headers: tuple[tuple[str, str], ...] = ()
    if error.http_status == 401:
        headers = (("WWW-Authenticate", 'Bearer realm="stormlog-studio"'),)
    if error.code == "method_not_allowed":
        allowed = ", ".join(error.details.get("allowed", ()))
        headers = (("Allow", allowed),)
    return HttpResponse(error.http_status, JSON_TYPE, encode_json(document), headers)


def _busy_bytes() -> bytes:
    body = encode_json(BusyError("busy", "too many connections").document())
    head = (
        "HTTP/1.1 503 Service Unavailable\r\n"
        f"Content-Type: {JSON_TYPE}\r\nContent-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
    )
    return head.encode("ascii") + body


_BUSY = _busy_bytes()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], owner: StudioServer) -> None:
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        self.owner = owner
        self._slots = threading.BoundedSemaphore(owner.limits.max_connections)
        self._tokens: dict[int, int] = {}
        self._connections: set[socket.socket] = set()
        self._lock = threading.Lock()
        super().__init__(address, _Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            _refuse(request)
            self.shutdown_request(request)
            return
        try:
            self.arm(request, self.owner.limits.deadline_s)
            with self._lock:
                self._connections.add(request)
            super().process_request(request, client_address)
        except BaseException:
            self._finished(request)
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._finished(request)

    def arm(self, request: socket.socket, deadline_s: float | None) -> bool:
        """(Re-)arm ``request``'s deadline from now; False once it was cut."""
        with self._lock:
            old = self._tokens.pop(id(request), None)
        if old is not None and not self.owner.watchdog.disarm(old):
            return False
        if deadline_s is None:
            return True
        try:
            token = self.owner.watchdog.arm(request, time.monotonic() + deadline_s)
        except WatchdogStopped:
            return False
        with self._lock:
            self._tokens[id(request)] = token
        return True

    def _finished(self, request: Any) -> None:
        with self._lock:
            token = self._tokens.pop(id(request), None)
            self._connections.discard(request)
        if token is not None:
            self.owner.watchdog.disarm(token)
        self._slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A connection cut at its deadline fails in whatever it was doing.
        if self.owner.debug:
            traceback.print_exc(file=self.owner.stderr or sys.stderr)

    def close_connections(self) -> None:
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _refuse(request: socket.socket) -> None:
    try:
        request.setblocking(False)
        request.send(_BUSY)
    except OSError:
        pass


class _HeadTooLarge(http.client.HTTPException):
    """The request line and headers went past the head limit."""


class _LineSource(Protocol):
    def readline(self, size: int = ..., /) -> bytes: ...

    def read(self, size: int = ..., /) -> bytes: ...

    def close(self) -> None: ...


class _BoundedReader:
    """The connection's reader: a head of at most ``limit`` bytes, then a body.

    Each head line is read with at most one byte more than is left, so a
    head is refused once ``limit + 1`` bytes of it were read, wherever they
    fall. ``read`` serves the body, whose length httpd checks first.
    """

    def __init__(self, raw: _LineSource, limit: int) -> None:
        self._raw = raw
        self._left = limit

    def readline(self, size: int = -1) -> bytes:
        cap = self._left + 1 if size < 0 else min(size, self._left + 1)
        line = self._raw.readline(cap)
        self._left -= len(line)
        if self._left < 0:
            raise _HeadTooLarge("request head too large")
        return line

    def read(self, size: int) -> bytes:
        return self._raw.read(size)

    def close(self) -> None:
        self._raw.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "stormlog-studio"
    sys_version = ""
    server: _Server

    def setup(self) -> None:
        # The watchdog bounds the connection; the per-operation timeout is
        # a fallback set past it, so the watchdog always acts first.
        super().setup()
        limits = self.server.owner.limits
        self.connection.settimeout(limits.deadline_s + 1.0)
        self.rfile = cast(Any, _BoundedReader(self.rfile, limits.head_bytes))
        self._started = time.monotonic()
        self._template = "-"
        self._path = ""

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except _HeadTooLarge:
            self.requestline = self.request_version = self.command = ""
            self.close_connection = True
            self._send(render_error(_head_too_large()), include_body=True)

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # Only the stock parser sends errors, for a head it refused.
        self.close_connection = True
        self._send(render_error(_parser_error(code)), include_body=True)

    def _dispatch(self) -> None:
        self.close_connection = True
        try:
            response = self._respond()
        except StudioError as error:
            response = _error_response(error)
        self._send(response, include_body=self.command != "HEAD")
        self._log(response.status)

    # Every method dispatches, so one no route serves gets 405, not 501.
    do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _dispatch

    def _respond(self) -> HttpResponse:
        owner = self.server.owner
        request = self._request()
        self._path = request.path
        route = owner.routes.get((request.method, request.path))
        self._template = route.path if route is not None else _template(request.path)
        check_request(request, route, owner.tokens)
        if route is None:
            return self._unrouted(request)
        if not self.server.arm(self.connection, route.deadline_s):
            raise StudioError("internal_error", "the connection was cut")
        if route.deadline_s is None:
            self.connection.settimeout(owner.limits.deadline_s)
        else:
            self.connection.settimeout(route.deadline_s + 1.0)
        request = self._with_body(request)
        if route.auth == "bearer_or_ticket":
            request = without_ticket(request)
        return _call(route, request, owner)

    def _unrouted(self, request: HttpRequest) -> HttpResponse:
        owner = self.server.owner
        static = owner.static.lookup(request.path)
        if static is not None and request.method == "GET":
            return owner.static.respond(static, request.headers.get("if-none-match"))
        allowed = sorted(m for m, p in owner.routes if p == request.path)
        if static is not None:
            allowed = ["GET"]
        if allowed:
            raise MethodNotAllowed(
                "method_not_allowed",
                f"{request.method} is not served here",
                details={"method": request.method[:16], "allowed": allowed},
            )
        raise NotFoundError(
            "route_not_found",
            "no route for this method and path",
            details={"method": request.method[:16], "path": request.path[:256]},
        )

    def _request(self) -> HttpRequest:
        target = urlsplit(self.path)
        # No Studio path starts with "//". http.server collapses leading
        # slashes in self.path ("//evil/x" became "/evil/x", a misleading
        # 404, and "///api/v1/health" was served), so read the target as
        # the request line sent it.
        sent = self.requestline.split(" ")
        if (
            not self.path.startswith("/")
            or (len(sent) > 1 and sent[1].startswith("//"))
            or target.fragment
        ):
            raise MalformedRequest("malformed_request", "the request target is bad")
        try:
            path = unquote(target.path, errors="strict")
            query = _parse_query(target.query)
        except (UnicodeDecodeError, ValueError):
            raise MalformedRequest(
                "malformed_request", "the path or query string is not valid"
            ) from None
        return HttpRequest(
            method=self.command,
            path=path,
            query=query,
            headers=self._headers(),
            body=b"",
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            key = name.lower()
            if key in headers and key in _SINGLE_HEADERS:
                raise MalformedRequest(
                    "malformed_request", f"the {key} header is repeated"
                )
            headers[key] = f"{headers[key]}, {value}" if key in headers else value
        return headers

    def _with_body(self, request: HttpRequest) -> HttpRequest:
        if "transfer-encoding" in request.headers:
            raise MalformedRequest(
                "malformed_request", "chunked request bodies are not accepted"
            )
        text = request.headers.get("content-length", "0").strip() or "0"
        if not text.isdigit():
            raise MalformedRequest("malformed_request", "bad Content-Length")
        length = int(text)
        limit = self.server.owner.limits.body_bytes
        if length > limit:
            raise BodyTooLarge(
                "body_too_large",
                f"the request body is over {limit} bytes",
                details={"limit_bytes": limit},
            )
        body = self.rfile.read(length) if length else b""
        if len(body) != length:
            raise MalformedRequest("malformed_request", "the body was cut short")
        return HttpRequest(
            method=request.method,
            path=request.path,
            query=request.query,
            headers=request.headers,
            body=body,
        )

    def _send(self, response: HttpResponse, *, include_body: bool) -> None:
        body = response.body
        streamed = not isinstance(body, (bytes, bytearray))
        self.send_response(response.status)
        for name, value in _headers_for(self._path, response, streamed):
            self.send_header(name, value)
        self.end_headers()
        if not include_body or response.status == 304:
            _close(body)
            return
        if streamed:
            self._stream(cast(Iterable[bytes], body))
        else:
            self.wfile.write(cast(bytes, body))

    def _stream(self, chunks: Iterable[bytes]) -> None:
        """Chunked transfer; a failure ends the body without its last chunk."""
        try:
            for chunk in chunks:
                if chunk:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")
        except Exception:
            # The client sees a cut stream, never a mixed file (§5.6).
            if self.server.owner.debug:
                traceback.print_exc(file=self.server.owner.stderr or sys.stderr)
        finally:
            _close(chunks)

    def _log(self, status: int) -> None:
        owner = self.server.owner
        if owner.log_requests:
            elapsed_ms = (time.monotonic() - self._started) * 1000.0
            owner.log(f"{self.command} {self._template} {status} {elapsed_ms:.1f} ms")

    def log_message(self, format: str, *args: Any) -> None:
        return None


def _call(route: Route, request: HttpRequest, owner: StudioServer) -> HttpResponse:
    try:
        return route.handler(request)
    except StudioError:
        raise
    except NotReady as pending:
        return HttpResponse(
            202,
            JSON_TYPE,
            encode_json(pending.state),
            (("Retry-After", str(max(1, int(pending.retry_after_s)))),),
        )
    except Exception:
        if owner.debug:
            traceback.print_exc(file=owner.stderr or sys.stderr)
        raise StudioError("internal_error", "internal error") from None


def _error_response(error: StudioError) -> HttpResponse:
    if error.code == "ticket_invalid":
        # A ticket URL is a navigation: the tab shows text, not JSON (§7.2).
        return HttpResponse(401, "text/plain; charset=utf-8", _TICKET_NOTE)
    return render_error(error)


def _headers_for(
    path: str, response: HttpResponse, streamed: bool
) -> list[tuple[str, str]]:
    headers = [("Content-Type", response.content_type), *SECURITY_HEADERS]
    if path.startswith("/api/") or response.status >= 400:
        headers += [
            ("Cache-Control", "no-store"),
            ("Content-Security-Policy", _API_CSP),
        ]
    headers += list(response.headers)
    if streamed:
        headers.append(("Transfer-Encoding", "chunked"))
    else:
        headers.append(("Content-Length", str(len(cast(bytes, response.body)))))
    headers.append(("Connection", "close"))
    return headers


def _parse_query(text: str) -> Mapping[str, tuple[str, ...]]:
    if not text:
        return {}
    parsed = parse_qs(
        text, keep_blank_values=True, strict_parsing=True, errors="strict"
    )
    return {name: tuple(values) for name, values in parsed.items()}


def _template(path: str) -> str:
    if path == "/" or path.startswith("/static/"):
        return "static"
    return "unrouted"


def _close(body: object) -> None:
    close = getattr(body, "close", None)
    if callable(close) and not isinstance(body, (bytes, bytearray)):
        close()


def _head_too_large() -> HeadTooLarge:
    return HeadTooLarge("head_too_large", "the request line and headers are too large")


def _parser_error(code: int) -> StudioError:
    if code in (
        HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
        HTTPStatus.REQUEST_URI_TOO_LONG,
    ):
        return _head_too_large()
    if code == HTTPStatus.NOT_IMPLEMENTED:
        return MethodNotAllowed(
            "method_not_allowed",
            "this method is never served",
            details={"allowed": ["GET", "POST"]},
        )
    return MalformedRequest("malformed_request", "the request could not be parsed")
