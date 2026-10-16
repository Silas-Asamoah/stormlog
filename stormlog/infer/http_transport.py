"""Shared bounded HTTP transport for text, audio and read-only server evidence."""

from __future__ import annotations

import http.client
import math
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

from stormlog.scrub import KnownSecrets, scrub_text


class EndpointHTTPError(RuntimeError):
    """The endpoint answered with an HTTP error status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class ConnectError(OSError):
    """``connect()`` failed, so no byte of the request was sent.

    That covers a refused or timed-out connection, a TLS handshake that did
    not finish, and a socket the peer reset before ``connect()`` returned.
    urllib wraps this in ``URLError``, as it does every error from sending,
    and the reason tells the two apart: a request that failed here never
    reached the server.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause


class NoResponseError(ConnectionError):
    """The connection failed after the request was sent, before any response.

    A small request fits in the socket buffer, so it counts as sent before
    the server has read a byte of it. A reset or close while waiting for the
    status line leaves delivery as unknown as a failure while sending.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause


class CutResponseError(ConnectionError):
    """The server answered with a status line, then the connection failed.

    The server took the request: delivery is known, and the status is the
    answer it gave, cut short.
    """

    def __init__(self, status: int, cause: BaseException) -> None:
        super().__init__(f"HTTP {status}, then {type(cause).__name__}: {cause}")
        self.status = status
        self.cause = cause


class _StatusLine:
    """The status a connection's response line gave, once it arrived."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.timer: threading.Timer | None = None


def _response_class(seen: _StatusLine) -> type[http.client.HTTPResponse]:
    class _Response(http.client.HTTPResponse):
        def close(self) -> None:
            if seen.timer is not None:
                seen.timer.cancel()
            super().close()

        def _read_status(self) -> tuple[str, int, str]:
            version, status, reason = super()._read_status()  # type: ignore[misc]
            seen.status = int(status)
            return str(version), int(status), str(reason)

    return _Response


class _Answered:
    """Tells a failure before the status line from one after it."""

    def _track(self) -> None:
        self._status_line = _StatusLine()
        self._status_line.timer = getattr(self, "_deadline_timer", None)
        self.response_class = _response_class(self._status_line)

    def _connect_deadline(self, call: Any) -> None:
        timeout = getattr(self, "timeout", None)
        deadline = (
            time.perf_counter() + timeout if isinstance(timeout, (int, float)) else None
        )
        self._create_connection = _bounded_connection
        call()
        self._request_deadline = deadline
        if deadline is not None:
            self._deadline_timer = threading.Timer(
                max(0, deadline - time.perf_counter()),
                _shutdown,
                args=(getattr(self, "sock"),),
            )
            self._deadline_timer.daemon = True
            self._deadline_timer.start()

    def close(self) -> None:
        timer = getattr(self, "_deadline_timer", None)
        if timer is not None:
            timer.cancel()
        getattr(super(), "close")()

    def _expired(self) -> bool:
        deadline = getattr(self, "_request_deadline", None)
        return deadline is not None and time.perf_counter() >= deadline

    def _failed(self, exc: ConnectionError) -> ConnectionError:
        status = self._status_line.status
        return NoResponseError(exc) if status is None else CutResponseError(status, exc)


class _TrackedHTTPConnection(_Answered, http.client.HTTPConnection):
    def connect(self) -> None:
        try:
            self._connect_deadline(super().connect)
        except OSError as exc:
            raise ConnectError(exc) from exc

    def getresponse(self) -> http.client.HTTPResponse:
        self._track()
        try:
            return super().getresponse()
        except Exception as exc:
            if self._expired():
                raise RequestDeadlineExceeded(self._status_line.status) from exc
            if isinstance(exc, ConnectionError):
                raise self._failed(exc) from exc
            raise


class _TrackedHTTPSConnection(_Answered, http.client.HTTPSConnection):
    def connect(self) -> None:
        try:
            self._connect_deadline(super().connect)
        except OSError as exc:
            raise ConnectError(exc) from exc

    def getresponse(self) -> http.client.HTTPResponse:
        self._track()
        try:
            return super().getresponse()
        except Exception as exc:
            if self._expired():
                raise RequestDeadlineExceeded(self._status_line.status) from exc
            if isinstance(exc, ConnectionError):
                raise self._failed(exc) from exc
            raise


class _TrackedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_TrackedHTTPConnection, req)


class _TrackedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        options: dict[str, Any] = {"context": getattr(self, "_context", None)}
        check_hostname = getattr(self, "_check_hostname", None)
        if check_hostname is not None:
            options["check_hostname"] = check_hostname
        return self.do_open(_TrackedHTTPSConnection, req, **options)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a redirect as the HTTP error it is instead of following it.

    urllib would re-send a redirected POST as a GET to another address, after
    the first server had already received the request.
    """

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def inference_opener() -> urllib.request.OpenerDirector:
    """An opener that marks where a request failed and goes straight there.

    It follows no redirects and ignores proxies from the environment. Through
    a proxy, ``connect()`` reaches the proxy, so a server that cannot be
    reached reads as the proxy's HTTP 502 rather than as ``unreachable``.
    """
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
        _TrackedHTTPHandler(),
        _TrackedHTTPSHandler(),
    )


def ignored_proxies() -> dict[str, Any]:
    """What the environment asked of proxies, which the opener ignores.

    Only the schemes: a proxy's URL can carry its credentials.
    """
    schemes = {
        name.lower()[: -len("_proxy")]
        for name, value in os.environ.items()
        if value and name.lower().endswith("_proxy")
    }
    return {"ignored": True, "schemes": sorted(schemes)}


class RequestDeadlineExceeded(TimeoutError):
    """The request's total deadline expired, including a trickling response."""

    def __init__(self, status: int | None = None) -> None:
        super().__init__("HTTP request total deadline exceeded")
        self.status = status


def _shutdown(sock: Any) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def validate_http_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must use http:// or https:// with a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "Use an explicit API key instead of credentials in the endpoint URL"
        )
    if parsed.fragment:
        raise ValueError("endpoint must not include a URL fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("endpoint has an invalid port") from exc


def safe_error(value: str, *, secrets: tuple[str | None, ...] = ()) -> str:
    value = scrub_text(value, max_bytes=4096, secrets=KnownSecrets(secrets))
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


class BoundedResponse:
    """Bound cumulative reads and stop the underlying socket at a total deadline."""

    def __init__(self, response: Any, *, deadline: float, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("response byte limit must be a positive integer")
        if not math.isfinite(deadline):
            raise ValueError("request deadline must be finite")
        self.response = response
        self.deadline = deadline
        self.max_bytes = max_bytes
        self.bytes_read = 0
        self.headers = getattr(response, "headers", {})
        self.status = getattr(response, "status", None)
        self._timer: threading.Timer | None = None

    def __enter__(self) -> "BoundedResponse":
        fp = getattr(self.response, "fp", None)
        sock = getattr(getattr(fp, "raw", None), "_sock", None)
        if sock is not None:
            self._timer = threading.Timer(
                max(0, self.deadline - time.perf_counter()), _shutdown, args=(sock,)
            )
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _checked(self, data: bytes) -> bytes:
        if time.perf_counter() >= self.deadline:
            raise RequestDeadlineExceeded(self.status)
        self.bytes_read += len(data)
        if self.bytes_read > self.max_bytes:
            raise ValueError("HTTP response exceeded the configured byte limit")
        return data

    def read_chunk(self, size: int = 65536) -> bytes:
        remaining = min(size, self.max_bytes - self.bytes_read + 1)
        read = getattr(self.response, "read1", self.response.read)
        try:
            data = self._checked(read(remaining))
        except OSError as exc:
            if time.perf_counter() >= self.deadline:
                raise RequestDeadlineExceeded(self.status) from exc
            raise
        if not data:
            self._validate_length()
        return data

    def _validate_length(self) -> None:
        expected = self.headers.get("Content-Length")
        if expected is not None and int(expected) != self.bytes_read:
            raise CutResponseError(
                self.status or 200, EOFError("Truncated HTTP response body")
            )

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        count = 0
        while size < 0 or count < size:
            data = self.read_chunk(65536 if size < 0 else min(65536, size - count))
            if not data:
                break
            chunks.append(data)
            count += len(data)
        return b"".join(chunks)

    def __iter__(self) -> Iterator[bytes]:
        pending = bytearray()
        while True:
            data = self.read_chunk()
            if not data:
                if pending:
                    yield bytes(pending)
                return
            pending.extend(data)
            while b"\n" in pending:
                line, _, rest = pending.partition(b"\n")
                pending = bytearray(rest)
                yield bytes(line) + b"\n"


def json_headers(
    api_key: str | None, *, accept: str = "application/json"
) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": accept}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


_DNS_SLOTS = threading.BoundedSemaphore(64)


def _addresses(address: tuple[str, int], deadline: float) -> list[Any]:
    """Bound DNS waiting and the number of unresolved daemon lookups."""
    if not _DNS_SLOTS.acquire(timeout=max(0, deadline - time.perf_counter())):
        raise TimeoutError("DNS resolution capacity deadline exceeded")
    replies: queue.Queue[Any] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            replies.put(socket.getaddrinfo(*address, 0, socket.SOCK_STREAM))
        except Exception as exc:
            replies.put(exc)
        finally:
            _DNS_SLOTS.release()

    thread = threading.Thread(target=resolve, daemon=True, name="stormlog-dns")
    try:
        thread.start()
    except BaseException:
        _DNS_SLOTS.release()
        raise
    try:
        result = replies.get(timeout=max(0, deadline - time.perf_counter()))
    except queue.Empty as exc:
        raise TimeoutError("DNS resolution total deadline exceeded") from exc
    if isinstance(result, Exception):
        raise result
    return list(result)


def _bounded_connection(
    address: tuple[str, int],
    timeout: float,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    deadline = time.perf_counter() + timeout
    last: OSError | None = None
    for family, kind, protocol, _canonical, destination in _addresses(
        address, deadline
    ):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Connection total deadline exceeded")
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(remaining)
            if source_address is not None:
                sock.bind(source_address)
            sock.connect(destination)
            sock.settimeout(max(1e-9, deadline - time.perf_counter()))
            return sock
        except OSError as exc:
            last = exc
            sock.close()
    raise last if last is not None else OSError("No endpoint addresses")
