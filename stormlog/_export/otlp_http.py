"""Send one OTLP/HTTP export, within a deadline, and say what is known to have happened.

Each attempt opens its own connection: an idle connection the collector has
closed would otherwise turn a request it never read into one whose fate is
unknown. The attempt walks the resolver's addresses under one deadline,
and the watchdog shuts its socket down when the deadline passes, so a
silent TLS handshake or a response dribbled a byte at a time ends on time.

The outcome separates what the collector definitely did from what it may
have done. A body that was never fully sent cannot have been stored. Once
it has been sent, only a readable 200 confirms what was kept; a timeout, a
reset or an unreadable answer leaves it unknown.
"""

from __future__ import annotations

import gzip
import http.client
import socket
import ssl
import threading
import time
import urllib.parse
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from .http_server import is_loopback
from .inflate import gunzip_capped
from .otlp_encoding import (
    JSON_MEDIA,
    PROTOBUF_MEDIA,
    ExportResult,
    NonconformantResponse,
    decode_response,
    status_message,
)
from .resolver import Candidate, Resolver
from .watchdog import Watchdog

TRACES_PATH = "/v1/traces"
# The most a response may hold, after decompression.
MAX_RESPONSE_BYTES = 64 * 1024
DEFAULT_ATTEMPT_SECONDS = 5.0
# Attempts in a row that reached none of the addresses before the name is
# resolved again: a collector whose address changed, such as a recreated
# service, is found within a few attempts.
RESOLVE_AFTER_FAILURES = 3

# What a transmission was.
CONFIRMED = "confirmed"
REFUSED = "refused"
AMBIGUOUS = "ambiguous"
NOT_SENT = "not_sent"

# Why, as a closed set.
CONNECT_REFUSED = "connect_refused"
CONNECT_TIMEOUT = "connect_timeout"
DNS = "dns"
TLS = "tls"
SEND_FAILED = "send_failed"
TIMEOUT_AFTER_SEND = "timeout_after_send"
RESET_AFTER_SEND = "reset_after_send"
HTTP_4XX = "http_4xx"
HTTP_5XX = "http_5xx"
THROTTLED = "throttled"
REDIRECT = "redirect"
UNREADABLE_RESPONSE = "unreadable_response"
NONCONFORMANT_RESPONSE = "nonconformant_response"
CATEGORIES = (
    CONNECT_REFUSED,
    CONNECT_TIMEOUT,
    DNS,
    TLS,
    SEND_FAILED,
    TIMEOUT_AFTER_SEND,
    RESET_AFTER_SEND,
    HTTP_4XX,
    HTTP_5XX,
    THROTTLED,
    REDIRECT,
    UNREADABLE_RESPONSE,
    NONCONFORMANT_RESPONSE,
)
_RETRY_AFTER_SEND = {502, 504}
_THROTTLE = {429, 503}


@dataclass(frozen=True)
class Destination:
    """Where exports go: an OTLP/HTTP traces URL."""

    scheme: str
    host: str
    port: int
    target: str

    @classmethod
    def parse(cls, url: str) -> Destination:
        """``url`` as given, or with ``/v1/traces`` added to a bare origin.

        Raises ``ValueError`` for anything but an http or https URL with a
        host, and for credentials in the URL: those belong in a header.
        """
        parts = urllib.parse.urlsplit(url)
        _check_url(parts)
        try:
            port = parts.port
        except ValueError as exc:
            raise ValueError(f"--otlp-endpoint has an invalid port: {exc}") from exc
        assert parts.hostname is not None
        path = parts.path if parts.path not in ("", "/") else TRACES_PATH
        target = path + (f"?{parts.query}" if parts.query else "")
        default = 443 if parts.scheme == "https" else 80
        return cls(parts.scheme, parts.hostname, port or default, target)

    @property
    def loopback(self) -> bool:
        """A loopback address, or the name localhost; never a name that only
        starts like one, such as 127.attacker.example."""
        return is_loopback(self.host)


def _check_url(parts: urllib.parse.SplitResult) -> None:
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("--otlp-endpoint must be an http:// or https:// URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("--otlp-endpoint must not hold credentials; use --otlp-header")
    if parts.fragment:
        raise ValueError("--otlp-endpoint must not have a fragment")


@dataclass(frozen=True)
class Transmission:
    """One attempt's outcome.

    ``kind`` is ``confirmed``, ``refused``, ``ambiguous`` or ``not_sent``;
    ``category`` says why, from ``CATEGORIES``, and is None for a clean
    confirmation. ``message`` is the collector's own text, a partial
    success's warning or an error's ``Status.message``: it may echo what
    was sent, so it is only for callers with consent to keep it.
    """

    kind: str
    category: str | None = None
    status: int | None = None
    result: ExportResult | None = None
    retryable: bool = False
    retry_after: float | None = None
    message: str | None = None
    sent_bytes: int = 0


class OtlpHttpTransport:
    """Send gzip-compressed export requests to one destination."""

    def __init__(
        self,
        destination: Destination,
        *,
        media_type: str,
        headers: Mapping[str, str] | None = None,
        resolver: Resolver | None = None,
        watchdog: Watchdog | None = None,
        attempt_seconds: float = DEFAULT_ATTEMPT_SECONDS,
        ssl_context: ssl.SSLContext | None = None,
        user_agent: str = "stormlog",
    ) -> None:
        self.destination = destination
        self.media_type = media_type
        self.attempt_seconds = attempt_seconds
        self.resolver = resolver or Resolver(destination.host, destination.port)
        self.watchdog = watchdog or Watchdog(name="stormlog-otlp-watchdog")
        self._ssl = ssl_context
        if destination.scheme == "https" and ssl_context is None:
            self._ssl = ssl.create_default_context()
        self._headers = {
            **dict(headers or {}),
            "Content-Type": media_type,
            "Content-Encoding": "gzip",
            "User-Agent": user_agent,
        }
        self._lock = threading.Lock()
        self._current: socket.socket | None = None
        self._aborted = False
        # Only the sending thread reads or changes it.
        self._connect_failures = 0
        # Whether the latest attempt began to send its body; kept until the
        # next attempt starts, so a late reader still sees it.
        self._body_started = False

    def start(self, wait: float = 2.0) -> bool:
        """Resolve the destination, waiting at most ``wait`` seconds."""
        return self.resolver.resolve(wait)

    def send(self, body: bytes, *, spans: int) -> Transmission:
        """Export ``body``, holding ``spans`` spans, in one attempt."""
        with self._lock:
            self._body_started = False
            if self._aborted:
                return Transmission(NOT_SENT, SEND_FAILED)
        deadline = time.monotonic() + self.attempt_seconds
        compressed = gzip.compress(body, compresslevel=6, mtime=0)
        sock, token, failure = self._connect(deadline)
        if sock is None or token is None:
            assert failure is not None
            return failure
        try:
            return self._exchange(sock, token, compressed, spans)
        finally:
            self.watchdog.disarm(token)
            with self._lock:
                self._current = None
            _close(sock)

    def abort(self) -> bool:
        """Stop for good: end any attempt in progress and refuse new ones.

        Returns whether the latest attempt had begun to send its body, which
        the collector may then have stored.
        """
        with self._lock:
            self._aborted = True
            sock = self._current
            started = self._body_started
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return started

    def _connect(
        self, deadline: float
    ) -> tuple[socket.socket | None, int | None, Transmission | None]:
        candidates = self._candidates(deadline)
        if not candidates:
            return None, None, Transmission(NOT_SENT, DNS, retryable=True)
        category = CONNECT_TIMEOUT
        for candidate in candidates:
            if _left(deadline) <= 0:
                break
            sock, token, category = self._try(candidate, deadline)
            if sock is not None and token is not None:
                self._connect_failures = 0
                self.resolver.mark_good(candidate)
                return self._secure(sock, token, deadline)
        self._connect_failures += 1
        return None, None, Transmission(NOT_SENT, category, retryable=True)

    def _candidates(self, deadline: float) -> list[Candidate]:
        """The addresses to try, resolved again when none is known, or when
        ``RESOLVE_AFTER_FAILURES`` attempts in a row reached none of them."""
        candidates = self.resolver.candidates()
        stale = self._connect_failures >= RESOLVE_AFTER_FAILURES
        if stale:
            self._connect_failures = 0
        if (not candidates or stale) and self.resolver.resolve(
            min(2.0, _left(deadline))
        ):
            candidates = self.resolver.candidates()
        return candidates

    def _try(
        self, candidate: Candidate, deadline: float
    ) -> tuple[socket.socket | None, int | None, str]:
        try:
            sock = socket.socket(candidate.family, candidate.type, candidate.proto)
        except OSError:
            return None, None, CONNECT_REFUSED
        token = self._register(sock, deadline)
        try:
            # Never 0, which would make the socket non-blocking.
            sock.settimeout(max(_left(deadline), 0.01))
            sock.connect(candidate.address)
            return sock, token, ""
        except OSError as exc:
            fired = not self.watchdog.disarm(token)
            _close(sock)
            if fired or isinstance(exc, socket.timeout):
                return None, None, CONNECT_TIMEOUT
            return None, None, CONNECT_REFUSED

    def _secure(
        self, sock: socket.socket, token: int, deadline: float
    ) -> tuple[socket.socket | None, int | None, Transmission | None]:
        if self._ssl is None:
            return sock, token, None
        # Registered before the handshake, so the watchdog can end a silent one.
        self.watchdog.disarm(token)
        try:
            wrapped = self._ssl.wrap_socket(
                sock,
                server_hostname=self.destination.host,
                do_handshake_on_connect=False,
            )
        except (OSError, ValueError):
            _close(sock)
            return None, None, Transmission(NOT_SENT, TLS, retryable=True)
        token = self._register(wrapped, deadline)
        try:
            wrapped.do_handshake()
        except OSError:  # ssl.SSLError is one
            self.watchdog.disarm(token)
            _close(wrapped)
            return None, None, Transmission(NOT_SENT, TLS, retryable=True)
        return wrapped, token, None

    def _register(self, sock: socket.socket, deadline: float) -> int:
        with self._lock:
            self._current = sock
        return self.watchdog.arm(sock, deadline)

    def _exchange(
        self, sock: socket.socket, token: int, body: bytes, spans: int
    ) -> Transmission:
        connection = _Connection(self.destination.host, self.destination.port, sock)
        with self._lock:
            if self._aborted:
                return Transmission(NOT_SENT, SEND_FAILED)
            self._body_started = True
        try:
            connection.request(
                "POST", self.destination.target, body=body, headers=self._headers
            )
        except (OSError, http.client.HTTPException):
            # The body did not all leave, so the collector cannot have it.
            return Transmission(NOT_SENT, SEND_FAILED, retryable=True)
        sent = len(body)
        timed_out = False
        try:
            response = connection.getresponse()
            raw = _read_capped(response)
        except (OSError, http.client.HTTPException) as exc:
            raw, timed_out = None, isinstance(exc, socket.timeout)
        if timed_out or self.watchdog.fired(token):
            # Cut at the deadline: whatever was read may be only part of it.
            return Transmission(
                AMBIGUOUS, TIMEOUT_AFTER_SEND, retryable=True, sent_bytes=sent
            )
        if raw is None:
            return Transmission(
                AMBIGUOUS, RESET_AFTER_SEND, retryable=True, sent_bytes=sent
            )
        return _classify(
            response.status, raw, response.headers, self.media_type, spans, sent
        )


class _Connection(http.client.HTTPConnection):
    """An HTTP connection over a socket that is already connected."""

    def __init__(self, host: str, port: int, sock: socket.socket) -> None:
        super().__init__(host, port)
        self.sock = sock

    def connect(self) -> None:  # The socket is never reconnected.
        raise OSError("the attempt's connection is closed")


def _read_capped(response: http.client.HTTPResponse) -> bytes | None:
    """Up to one byte over the cap; None when the body ended early."""
    chunks: list[bytes] = []
    total = 0
    while total <= MAX_RESPONSE_BYTES:
        chunk = response.read(MAX_RESPONSE_BYTES + 1 - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total <= MAX_RESPONSE_BYTES and response.length:
        # Fewer bytes than its Content-Length: the connection ended early.
        return None
    return b"".join(chunks)


def _classify(
    status: int,
    raw: bytes,
    headers: http.client.HTTPMessage,
    request_media: str,
    spans: int,
    sent: int,
) -> Transmission:
    body = _body(raw, headers)
    if status == 200:
        return _confirmed(body, headers, request_media, spans, sent)
    media = _media(headers, request_media)
    message = status_message(body, media) if body else None
    if 300 <= status < 400:
        return Transmission(REFUSED, REDIRECT, status, message=message, sent_bytes=sent)
    if status in _THROTTLE:
        return Transmission(
            REFUSED,
            THROTTLED,
            status,
            retryable=True,
            retry_after=retry_after(headers.get("Retry-After")),
            message=message,
            sent_bytes=sent,
        )
    if 400 <= status < 500:
        return Transmission(REFUSED, HTTP_4XX, status, message=message, sent_bytes=sent)
    if 200 <= status < 300:
        return Transmission(AMBIGUOUS, NONCONFORMANT_RESPONSE, status, sent_bytes=sent)
    return Transmission(
        AMBIGUOUS,
        HTTP_5XX,
        status,
        retryable=status in _RETRY_AFTER_SEND,
        message=message,
        sent_bytes=sent,
    )


def _confirmed(
    body: bytes | None,
    headers: http.client.HTTPMessage,
    request_media: str,
    spans: int,
    sent: int,
) -> Transmission:
    if body is None:
        return Transmission(AMBIGUOUS, UNREADABLE_RESPONSE, 200, sent_bytes=sent)
    try:
        result = decode_response(body, _media(headers, request_media), sent=spans)
    except NonconformantResponse:
        return Transmission(AMBIGUOUS, NONCONFORMANT_RESPONSE, 200, sent_bytes=sent)
    return Transmission(
        CONFIRMED, None, 200, result=result, message=result.message, sent_bytes=sent
    )


def _body(raw: bytes, headers: http.client.HTTPMessage) -> bytes | None:
    """The response body, inflated; None when over the cap or unreadable."""
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    if headers.get("Content-Encoding", "").strip().lower() != "gzip":
        return raw
    try:
        return gunzip_capped(raw, MAX_RESPONSE_BYTES)
    except (ValueError, zlib.error):
        return None


def _media(headers: http.client.HTTPMessage, request_media: str) -> str:
    """The response's media type, or the request's when it names neither."""
    media = headers.get_content_type()
    return media if media in (JSON_MEDIA, PROTOBUF_MEDIA) else request_media


def retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header; None when absent or unreadable."""
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        return None
    current = time.time() if now is None else now
    return max(0.0, when.timestamp() - current)


def _left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _close(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass
