"""A ``/metrics`` endpoint that cannot be held open or overrun by its clients.

Admission happens on the accept thread, before a handler thread exists or a
header is read: past ``max_connections`` a connection gets a canned 503 if
its send buffer takes it, and is closed. Each admitted connection serves one
request and has a total deadline, enforced by the watchdog, so a client that
sends no headers, dribbles them, or never reads its answer is cut off at the
deadline and frees its slot. A request line and headers past ``HEAD_LIMIT``
bytes are answered 431 and the connection closed, so no client makes the
server hold more of a request than that.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, BinaryIO, cast

from .renders import RenderCache
from .watchdog import Watchdog

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
METRICS_PATH = "/metrics"
# The request line and headers together; a scraper sends a few hundred bytes.
HEAD_LIMIT = 16 * 1024
_BUSY = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
)


@dataclass
class ServerStats:
    ok: int = 0
    rejected_busy: int = 0
    # Connections cut at their deadline, whatever they were doing.
    timeout: int = 0
    not_found: int = 0
    # Requests refused while being read: a head past HEAD_LIMIT, or one the
    # request parser could not take.
    bad_request: int = 0
    errors: int = 0
    active: int = 0


def parse_listen(listen: str) -> tuple[str, int]:
    """``HOST:PORT`` or ``[IPV6]:PORT`` as a host and a port."""
    host, sep, port = listen.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 <= int(port) <= 65535:
        raise ValueError(f"expected HOST:PORT, got {listen!r}")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host, int(port)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class MetricsServer:
    """Serve the shared render at ``/metrics`` while a run lasts."""

    def __init__(
        self,
        listen: str,
        renders: RenderCache,
        *,
        max_connections: int = 4,
        deadline: float = 10.0,
        watchdog: Watchdog | None = None,
    ) -> None:
        self.host, self.port = parse_listen(listen)
        self.renders = renders
        self.max_connections = max_connections
        self.deadline = deadline
        self._owns_watchdog = watchdog is None
        self.watchdog = watchdog or Watchdog(name="stormlog-metrics-watchdog")
        self.stats = ServerStats()
        self._lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def loopback(self) -> bool:
        return is_loopback(self.host)

    @property
    def address(self) -> str:
        server = self._server
        port = server.server_address[1] if server is not None else self.port
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{port}"

    def start(self) -> None:
        """Bind and serve; an ``OSError`` from binding reaches the caller."""
        self._server = _Server((self.host, self.port), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="stormlog-metrics",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        server.close_connections()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._owns_watchdog:
            self.watchdog.stop()

    def count(self, name: str, delta: int = 1) -> None:
        with self._lock:
            setattr(self.stats, name, getattr(self.stats, name) + delta)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], owner: MetricsServer) -> None:
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        self.owner = owner
        self._slots = threading.BoundedSemaphore(owner.max_connections)
        self._tokens: dict[int, int] = {}
        self._connections: set[socket.socket] = set()
        self._lock = threading.Lock()
        super().__init__(address, _Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self.owner.count("rejected_busy")
            _refuse(request)
            self.shutdown_request(request)
            return
        token = self.owner.watchdog.arm(request, time.monotonic() + self.owner.deadline)
        with self._lock:
            self._tokens[id(request)] = token
            self._connections.add(request)
        self.owner.count("active")
        super().process_request(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._lock:
                token = self._tokens.pop(id(request), None)
                self._connections.discard(request)
            if token is not None and not self.owner.watchdog.disarm(token):
                self.owner.count("timeout")
            self.owner.count("active", -1)
            self._slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        self.owner.count("errors")

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
    """The request line and headers went past ``HEAD_LIMIT`` bytes."""


class _HeadReader:
    """The connection's reader, refusing a request head past ``limit`` bytes.

    Each line is read with at most one byte more than is left, so a head is
    refused once ``limit + 1`` bytes of it were read, wherever they fall.
    """

    def __init__(self, raw: BinaryIO, limit: int) -> None:
        self._raw = raw
        self._left = limit

    def readline(self, size: int = -1) -> bytes:
        cap = self._left + 1 if size < 0 else min(size, self._left + 1)
        line = self._raw.readline(cap)
        self._left -= len(line)
        if self._left < 0:
            raise _HeadTooLarge(f"request head over {HEAD_LIMIT} bytes")
        return line

    def close(self) -> None:
        self._raw.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def setup(self) -> None:
        # The watchdog bounds the connection as a whole. The per-operation
        # timeout is only a fallback, set past the deadline so the watchdog
        # always acts first and the cut is counted as one.
        super().setup()
        self.connection.settimeout(self.server.owner.deadline + 1.0)
        # Only GET and HEAD are served, so the head is all that is read.
        self.rfile = cast(BinaryIO, _HeadReader(self.rfile, HEAD_LIMIT))

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except _HeadTooLarge:
            # The request line alone was too long; header lines past the
            # limit are refused the same way by the request parser.
            self.requestline = self.request_version = self.command = ""
            self.send_error(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE)

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # Only the request parser sends errors: a request refused unread.
        self.server.owner.count("bad_request")
        super().send_error(code, message, explain)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(include_body=False)

    def _serve(self, *, include_body: bool) -> None:
        self.close_connection = True
        owner = self.server.owner
        if self.path.split("?", 1)[0] != METRICS_PATH:
            owner.count("not_found")
            self._respond(404, b"")
            return
        generation = owner.renders.acquire()
        try:
            self._respond(200, generation.body, include_body=include_body)
            owner.count("ok")
        finally:
            owner.renders.release(generation)

    def _respond(self, status: int, body: bytes, *, include_body: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None
