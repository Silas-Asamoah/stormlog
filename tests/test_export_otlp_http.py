"""One OTLP/HTTP attempt: what is known to have happened, within its deadline."""

import contextlib
import gzip
import json
import os
import re
import shutil
import socket
import ssl
import struct
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stormlog._export import otlp_http
from stormlog._export.otlp_encoding import JsonEncoding, ProtobufEncoding
from stormlog._export.otlp_http import (
    AMBIGUOUS,
    CONFIRMED,
    CONNECT_REFUSED,
    DNS,
    HTTP_4XX,
    HTTP_5XX,
    NONCONFORMANT_RESPONSE,
    NOT_SENT,
    REDIRECT,
    REFUSED,
    RESET_AFTER_SEND,
    RESOLVE_AFTER_FAILURES,
    SEND_FAILED,
    THROTTLED,
    TIMEOUT_AFTER_SEND,
    TLS,
    UNREADABLE_RESPONSE,
    Destination,
    OtlpHttpTransport,
    retry_after,
)
from stormlog._export.resolver import Resolver
from stormlog._export.spans import KIND_CLIENT, Scope, Span
from stormlog._export.watchdog import Watchdog
from tests.fake_otlp_collector import (
    DRIBBLE,
    RESET,
    SHORT,
    SILENT,
    Reply,
    running,
)

pb2 = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")

SCOPE = Scope("stormlog.infer", "0")


@pytest.fixture(autouse=True)
def _stop_watchdogs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Each transport starts its own watchdog thread. In a run HttpSink.close
    # stops it; these tests use transports alone, so their end does.
    made: list[Watchdog] = []

    def watchdog(**kw: Any) -> Watchdog:
        made.append(Watchdog(**kw))
        return made[-1]

    monkeypatch.setattr(otlp_http, "Watchdog", watchdog)
    yield
    for dog in made:
        dog.stop()


def _body(count: int = 3, encoding: Any = None) -> bytes:
    encoding = encoding or ProtobufEncoding()
    spans = [
        Span(
            name="stormlog.infer.request",
            trace_id=f"{index + 1:032x}",
            span_id=f"{index + 1:016x}",
            kind=KIND_CLIENT,
            start_ns=1,
            end_ns=2,
        )
        for index in range(count)
    ]
    units = [encoding.unit(span)[0] for span in spans]
    return bytes(encoding.request((("service.name", "t"),), SCOPE, units))


def _transport(url: str, **kw: Any) -> OtlpHttpTransport:
    kw.setdefault("media_type", "application/x-protobuf")
    return OtlpHttpTransport(Destination.parse(url), **kw)


def _partial(rejected: int, message: str = "") -> bytes:
    response = pb2.ExportTraceServiceResponse()
    response.partial_success.rejected_spans = rejected
    response.partial_success.error_message = message
    return bytes(response.SerializeToString())


def test_a_200_confirms_and_the_collector_gets_the_spans() -> None:
    with running() as collector:
        transport = _transport(
            collector.url, headers={"X-Tenant": "a"}, user_agent="stormlog/9"
        )
        out = transport.send(_body(3), spans=3)
    assert (out.kind, out.category, out.status) == (CONFIRMED, None, 200)
    assert out.result is not None and out.result.rejected == 0
    (received,) = collector.received
    assert len(received.spans) == 3
    assert received.headers["content-encoding"] == "gzip"
    assert received.headers["content-type"] == "application/x-protobuf"
    assert received.headers["x-tenant"] == "a"
    assert received.headers["user-agent"] == "stormlog/9"
    assert out.sent_bytes > 0


def test_json_requests_and_partial_success() -> None:
    reply = Reply(
        body=b'{"partialSuccess":{"rejectedSpans":"2","errorMessage":"too old"}}',
        content_type="application/json",
    )
    with running([reply]) as collector:
        transport = _transport(collector.url, media_type="application/json")
        out = transport.send(_body(3, JsonEncoding()), spans=3)
    assert out.kind == CONFIRMED and out.result is not None
    assert (out.result.rejected, out.message) == (2, "too old")
    assert collector.received[0].headers["content-type"] == "application/json"


def test_a_protobuf_partial_success_and_a_gzip_response() -> None:
    reply = Reply(
        body=gzip.compress(_partial(1)), headers=(("Content-Encoding", "gzip"),)
    )
    with running([reply]) as collector:
        out = _transport(collector.url).send(_body(3), spans=3)
    assert out.kind == CONFIRMED and out.result is not None
    assert out.result.rejected == 1


@pytest.mark.parametrize(
    ("reply", "kind", "category", "retryable"),
    [
        (Reply(400, b"\x12\x03bad"), REFUSED, HTTP_4XX, False),
        (Reply(413), REFUSED, HTTP_4XX, False),
        (Reply(429, headers=(("Retry-After", "3"),)), REFUSED, THROTTLED, True),
        (Reply(503), REFUSED, THROTTLED, True),
        (Reply(302, headers=(("Location", "/x"),)), REFUSED, REDIRECT, False),
        (Reply(500), AMBIGUOUS, HTTP_5XX, False),
        (Reply(501), AMBIGUOUS, HTTP_5XX, False),
        (Reply(502), AMBIGUOUS, HTTP_5XX, True),
        (Reply(504), AMBIGUOUS, HTTP_5XX, True),
        (Reply(202), AMBIGUOUS, NONCONFORMANT_RESPONSE, False),
        (Reply(body=_partial(4)), AMBIGUOUS, NONCONFORMANT_RESPONSE, False),
        (Reply(body=b"\xff\xff"), AMBIGUOUS, NONCONFORMANT_RESPONSE, False),
        (
            Reply(body=b"\x1f\x8bnot gzip", headers=(("Content-Encoding", "gzip"),)),
            AMBIGUOUS,
            UNREADABLE_RESPONSE,
            False,
        ),
        (Reply(body=b"x" * (64 * 1024 + 1)), AMBIGUOUS, UNREADABLE_RESPONSE, False),
        (Reply(action=RESET), AMBIGUOUS, RESET_AFTER_SEND, True),
        (Reply(body=b"", action=SHORT), AMBIGUOUS, RESET_AFTER_SEND, True),
    ],
)
def test_answers_are_classified_by_what_they_say_was_kept(
    reply: Reply, kind: str, category: str, retryable: bool
) -> None:
    with running([reply]) as collector:
        out = _transport(collector.url).send(_body(3), spans=3)
    assert (out.kind, out.category, out.retryable) == (kind, category, retryable)
    assert out.sent_bytes > 0


def test_error_details_are_kept_apart_for_callers_with_consent() -> None:
    replies = [
        Reply(400, b"\x08\x03\x12\x0bcanary-text"),
        Reply(429, headers=(("Retry-After", "7"),)),
    ]
    with running(replies) as collector:
        transport = _transport(collector.url)
        refused = transport.send(_body(), spans=3)
        throttled = transport.send(_body(), spans=3)
    assert (refused.status, refused.message) == (400, "canary-text")
    assert throttled.retry_after == 7.0


@pytest.mark.parametrize("action", [SILENT, DRIBBLE])
def test_a_stuck_collector_is_cut_at_the_deadline(action: str) -> None:
    with running([Reply(action=action)]) as collector:
        transport = _transport(collector.url, attempt_seconds=0.6)
        started = time.monotonic()
        out = transport.send(_body(), spans=3)
        elapsed = time.monotonic() - started
    assert (out.kind, out.category, out.retryable) == (
        AMBIGUOUS,
        TIMEOUT_AFTER_SEND,
        True,
    )
    assert 0.5 <= elapsed < 1.5
    if action == DRIBBLE:
        # A byte every 0.2 s beats any per-read timeout; only the watchdog ends it.
        assert transport.watchdog.stats.fired == 1


def test_abort_ends_an_attempt_in_progress() -> None:
    with running([Reply(action=SILENT)]) as collector:
        transport = _transport(collector.url, attempt_seconds=10.0)
        results: list[Any] = []
        worker = threading.Thread(
            target=lambda: results.append(transport.send(_body(), spans=3))
        )
        worker.start()
        assert _wait(lambda: bool(collector.received))
        # The body had gone out, so the collector may have stored it.
        assert transport.abort() is True
        worker.join(2)
        assert not worker.is_alive()
        assert results[0].kind == AMBIGUOUS
        # Aborting is final: nothing more is sent.
        assert transport.send(_body(), spans=3).kind == NOT_SENT
    assert len(collector.received) == 1


def test_a_body_cut_off_part_way_was_not_sent() -> None:
    # The collector resets the connection after a little of the request, so
    # the body never all left: the collector cannot have stored it.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def reset_early() -> None:
        connection, _ = server.accept()
        connection.recv(1024)
        connection.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        connection.close()

    resetter = threading.Thread(target=reset_early, daemon=True)
    resetter.start()
    try:
        transport = _transport(f"http://127.0.0.1:{server.getsockname()[1]}")
        incompressible = os.urandom(4 * 1024 * 1024)
        out = transport.send(incompressible, spans=3)
    finally:
        resetter.join(5)
        server.close()
    assert (out.kind, out.category, out.retryable) == (NOT_SENT, SEND_FAILED, True)


def test_an_abort_during_the_connect_sends_no_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shutting down a socket that is still connecting does nothing, so only
    # the abort check before the body keeps it from leaving.
    with running() as collector:
        transport = _transport(collector.url)
        real_register = OtlpHttpTransport._register

        def register_then_abort(
            self: OtlpHttpTransport, sock: socket.socket, deadline: float
        ) -> int:
            token = real_register(self, sock, deadline)
            assert transport.abort() is False  # nothing had begun to send
            return token

        monkeypatch.setattr(OtlpHttpTransport, "_register", register_then_abort)
        out = transport.send(_body(), spans=3)
    assert out.kind == NOT_SENT and collector.received == []


def test_abort_before_anything_was_sent_says_so() -> None:
    transport = _transport(f"http://127.0.0.1:{_closed_port()}")
    assert transport.send(_body(), spans=3).kind == NOT_SENT
    assert transport.abort() is False


def _wait(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return bool(predicate())


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_refused_connection_was_never_sent() -> None:
    out = _transport(f"http://127.0.0.1:{_closed_port()}").send(_body(), spans=3)
    assert (out.kind, out.category, out.retryable) == (NOT_SENT, CONNECT_REFUSED, True)
    assert out.sent_bytes == 0


def test_a_name_that_does_not_resolve_was_never_sent() -> None:
    def fail(*_args: Any, **_kw: Any) -> list[Any]:
        raise socket.gaierror(socket.EAI_NONAME, "no such host")

    destination = Destination.parse("http://collector.invalid:4318")
    resolver = Resolver(destination.host, destination.port, getaddrinfo=fail)
    transport = OtlpHttpTransport(
        destination, media_type="application/x-protobuf", resolver=resolver
    )
    assert not transport.start(wait=1.0)
    out = transport.send(_body(), spans=3)
    assert (out.kind, out.category) == (NOT_SENT, DNS)


def test_a_refused_address_falls_through_to_the_next_one() -> None:
    # As localhost listing ::1 before 127.0.0.1 with the collector on only
    # one: the first address refuses, the second answers, in one attempt.
    with running() as collector:
        port = Destination.parse(collector.url).port
        closed = _closed_port()
        found = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", closed)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]
        resolver = Resolver(
            "localhost", port, getaddrinfo=lambda *_a, **_k: list(found)
        )
        transport = OtlpHttpTransport(
            Destination.parse(collector.url),
            media_type="application/x-protobuf",
            resolver=resolver,
        )
        assert transport.start()
        assert transport.send(_body(), spans=3).kind == CONFIRMED
        # The address that worked is tried first from now on.
        assert resolver.candidates()[0].address == ("127.0.0.1", port)
        assert transport.send(_body(), spans=3).kind == CONFIRMED
    assert len(collector.received) == 2


def _resolving_to(ports: list[int], calls: list[int]) -> Any:
    """A getaddrinfo that answers with the last port in ``ports``, and counts."""

    def getaddrinfo(*_args: Any, **_kw: Any) -> list[Any]:
        calls.append(ports[-1])
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", ports[-1]))]

    return getaddrinfo


def test_a_collector_that_moved_is_found_after_three_failed_attempts() -> None:
    # As a recreated service: the name now resolves to another address. The
    # name is resolved again once RESOLVE_AFTER_FAILURES attempts in a row
    # have reached none of the addresses known, not on every failure.
    with running() as collector:
        ports = [_closed_port()]
        calls: list[int] = []
        destination = Destination.parse(collector.url)
        resolver = Resolver(
            "collector.test", destination.port, getaddrinfo=_resolving_to(ports, calls)
        )
        transport = OtlpHttpTransport(
            destination, media_type="application/x-protobuf", resolver=resolver
        )
        assert transport.start()
        for _ in range(RESOLVE_AFTER_FAILURES):
            assert transport.send(_body(), spans=3).category == CONNECT_REFUSED
        assert len(calls) == 1
        ports.append(destination.port)
        assert transport.send(_body(), spans=3).kind == CONFIRMED
        assert len(calls) == 2
    assert len(collector.received) == 1


def test_a_destination_that_stays_down_is_resolved_again_once_per_three_attempts() -> (
    None
):
    calls: list[int] = []
    port = _closed_port()
    resolver = Resolver(
        "collector.test", port, getaddrinfo=_resolving_to([port], calls)
    )
    transport = OtlpHttpTransport(
        Destination.parse(f"http://127.0.0.1:{port}"),
        media_type="application/x-protobuf",
        resolver=resolver,
    )
    assert transport.start()
    for _ in range(2 * RESOLVE_AFTER_FAILURES + 1):
        assert transport.send(_body(), spans=3).category == CONNECT_REFUSED
    # At the start, and before the 4th and the 7th attempts.
    assert len(calls) == 3


@contextlib.contextmanager
def _silent_tcp() -> Iterator[int]:
    """Accepts connections and never says a word: a TLS handshake waits forever."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        server.settimeout(0.1)
        while not stop.is_set():
            with contextlib.suppress(OSError):
                held.append(server.accept()[0])

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield int(server.getsockname()[1])
    finally:
        stop.set()
        thread.join(2)
        for sock in held:
            sock.close()
        server.close()


def test_a_silent_tls_handshake_ends_at_the_deadline() -> None:
    with _silent_tcp() as port:
        transport = _transport(f"https://127.0.0.1:{port}", attempt_seconds=0.5)
        started = time.monotonic()
        out = transport.send(_body(), spans=3)
        elapsed = time.monotonic() - started
    assert (out.kind, out.category) == (NOT_SENT, TLS)
    assert elapsed < 1.5


@pytest.fixture(scope="module")
def tls_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A throwaway certificate and key for 127.0.0.1, made with ``openssl``."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("needs the openssl command to make a certificate")
    directory = tmp_path_factory.mktemp("tls")
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return cert, key


def _read_request(sock: socket.socket) -> None:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            return
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    match = re.search(rb"(?i)content-length:\s*(\d+)", head)
    length = int(match.group(1)) if match else 0
    while len(body) < length:
        chunk = sock.recv(65536)
        if not chunk:
            return
        body += chunk


@contextlib.contextmanager
def _tls_dribbler(cert: Path, key: Path) -> Iterator[int]:
    """Completes the handshake and reads the request, then sends the status
    line one byte every 0.2 s over TLS: no per-read timeout can end that."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(5.0)
    stop = threading.Event()

    def serve() -> None:
        with contextlib.suppress(OSError):
            conn, _ = server.accept()
            with context.wrap_socket(conn, server_side=True) as tls:
                _read_request(tls)
                for byte in b"HTTP/1.1 200 OK\r\n":
                    if stop.wait(0.2):
                        return
                    tls.sendall(bytes([byte]))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield int(server.getsockname()[1])
    finally:
        stop.set()
        thread.join(2)
        server.close()


def test_a_tls_collector_dribbling_its_answer_is_cut_at_the_deadline(
    tls_files: tuple[Path, Path],
) -> None:
    # The watchdog shuts the registered SSLSocket down in the middle of a read.
    cert, key = tls_files
    with _tls_dribbler(cert, key) as port:
        transport = _transport(
            f"https://127.0.0.1:{port}",
            attempt_seconds=0.6,
            ssl_context=ssl.create_default_context(cafile=str(cert)),
        )
        started = time.monotonic()
        out = transport.send(_body(), spans=3)
        elapsed = time.monotonic() - started
    assert (out.kind, out.category, out.retryable) == (
        AMBIGUOUS,
        TIMEOUT_AFTER_SEND,
        True,
    )
    assert out.sent_bytes > 0
    assert 0.5 <= elapsed < 1.5
    assert transport.watchdog.stats.fired == 1


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:4318", ("http", "127.0.0.1", 4318, "/v1/traces")),
        ("http://collector/", ("http", "collector", 80, "/v1/traces")),
        (
            "https://otel.example/custom/path",
            ("https", "otel.example", 443, "/custom/path"),
        ),
        ("http://[::1]:4318/v1/traces?x=1", ("http", "::1", 4318, "/v1/traces?x=1")),
    ],
)
def test_destinations(url: str, expected: tuple[Any, ...]) -> None:
    destination = Destination.parse(url)
    assert (
        destination.scheme,
        destination.host,
        destination.port,
        destination.target,
    ) == expected


@pytest.mark.parametrize(
    "url",
    [
        "grpc://127.0.0.1:4317",
        "127.0.0.1:4318",
        "http://user:pass@collector:4318",
        "http://collector:99999",
        "http://collector:4318/#frag",
    ],
)
def test_unusable_destinations_are_refused(url: str) -> None:
    with pytest.raises(ValueError):
        Destination.parse(url)


def test_retry_after_reads_seconds_and_dates() -> None:
    now = 1_000_000_000.0
    assert retry_after("120", now=now) == 120.0
    assert retry_after(" 0 ", now=now) == 0.0
    date = "Sun, 09 Sep 2001 01:46:50 GMT"  # now + 10 s
    assert retry_after(date, now=now) == pytest.approx(10.0)
    assert retry_after("Sun, 09 Sep 2001 01:46:30 GMT", now=now) == 0.0
    for unreadable in (None, "", "-5", "soon", "1.5"):
        assert retry_after(unreadable, now=now) is None
    assert json.dumps(retry_after("3")) == "3.0"


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:4318", True),
        ("http://127.3.4.5:4318", True),
        ("http://[::1]:4318", True),
        ("http://localhost:4318", True),
        ("http://LOCALHOST:4318", True),
        # Names that only start like a loopback address are other hosts:
        # they get no credentials in clear text.
        ("http://127.attacker.example:4318", False),
        ("http://127.0.0.1.nip.io:4318", False),
        ("http://collector.example:4318", False),
        ("http://10.0.0.1:4318", False),
    ],
)
def test_only_a_loopback_address_or_localhost_is_this_host(
    url: str, loopback: bool
) -> None:
    assert Destination.parse(url).loopback is loopback
