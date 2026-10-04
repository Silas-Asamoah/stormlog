"""The /metrics server: admission before threads, and a deadline per connection."""

import http.client
import socket
import threading
import time
from collections.abc import Iterator

import pytest

from stormlog._export.http_server import (
    CONTENT_TYPE,
    MetricsServer,
    is_loopback,
    parse_listen,
)
from stormlog._export.renders import RenderCache

BODY = b"# TYPE stormlog_up gauge\nstormlog_up 1\n"


@pytest.fixture
def server() -> Iterator[MetricsServer]:
    metrics = MetricsServer(
        "127.0.0.1:0", RenderCache(lambda: BODY), max_connections=2, deadline=0.5
    )
    metrics.start()
    yield metrics
    metrics.stop()


def _get(server: MetricsServer, path: str = "/metrics") -> http.client.HTTPResponse:
    host, port = parse_listen(server.address)
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request("GET", path)
    return connection.getresponse()


def _idle(server: MetricsServer) -> socket.socket:
    host, port = parse_listen(server.address)
    return socket.create_connection((host, port), timeout=5)


def _wait_for(condition, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_metrics_are_served_once_per_connection(server: MetricsServer) -> None:
    response = _get(server)
    assert response.status == 200
    assert response.getheader("Content-Type") == CONTENT_TYPE
    assert response.getheader("Connection") == "close"
    assert response.read() == BODY
    # Counted once the body is written, so the client can finish first.
    assert _wait_for(lambda: server.stats.ok == 1)


def test_other_paths_are_not_found(server: MetricsServer) -> None:
    assert _get(server, "/other").status == 404
    assert _get(server, "/metrics?x=1").status == 200
    assert server.stats.not_found == 1


def test_connections_past_the_limit_are_refused_before_a_thread_starts(
    server: MetricsServer,
) -> None:
    idle = [_idle(server), _idle(server)]  # send nothing, hold both slots
    try:
        assert _wait_for(lambda: server.stats.active == 2)
        threads_before = threading.active_count()
        refused = _idle(server)
        assert refused.recv(1024).startswith(b"HTTP/1.1 503")
        refused.close()
        assert server.stats.rejected_busy == 1
        assert threading.active_count() == threads_before
    finally:
        for sock in idle:
            sock.close()


def test_idle_connections_are_cut_at_the_deadline_and_free_their_slots(
    server: MetricsServer,
) -> None:
    idle = [_idle(server), _idle(server)]
    try:
        assert _wait_for(lambda: server.stats.timeout == 2 and server.stats.active == 0)
        assert _get(server).status == 200
    finally:
        for sock in idle:
            sock.close()


def test_dribbled_headers_do_not_extend_the_deadline(server: MetricsServer) -> None:
    sock = _idle(server)
    started = time.monotonic()
    try:
        for char in b"GET /metrics HTTP/1.1\r\nHost: x\r\n":
            try:
                sock.send(bytes([char]))
            except OSError:
                break
            time.sleep(0.05)  # well inside any per-read timeout
        assert _wait_for(lambda: server.stats.timeout >= 1)
        assert time.monotonic() - started < 3
    finally:
        sock.close()


def test_a_client_that_never_reads_is_cut_at_the_deadline() -> None:
    big = b"x" * (32 * 1024 * 1024)
    metrics = MetricsServer(
        "127.0.0.1:0", RenderCache(lambda: big), max_connections=1, deadline=0.5
    )
    metrics.start()
    try:
        sock = _idle(metrics)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        sock.sendall(b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n")
        assert _wait_for(
            lambda: metrics.stats.timeout == 1 and metrics.stats.active == 0
        )
        sock.close()
    finally:
        metrics.stop()


def test_stop_closes_open_connections() -> None:
    metrics = MetricsServer("127.0.0.1:0", RenderCache(lambda: BODY), deadline=30)
    metrics.start()
    sock = _idle(metrics)
    assert _wait_for(lambda: metrics.stats.active == 1)
    metrics.stop()
    assert _wait_for(lambda: metrics.stats.active == 0)
    sock.close()


def test_a_busy_port_raises_for_the_caller_to_record() -> None:
    first = MetricsServer("127.0.0.1:0", RenderCache(lambda: BODY))
    first.start()
    try:
        with pytest.raises(OSError):
            MetricsServer(first.address, RenderCache(lambda: BODY)).start()
    finally:
        first.stop()


@pytest.mark.parametrize(
    ("listen", "parsed"),
    [("127.0.0.1:9", ("127.0.0.1", 9)), ("[::1]:80", ("::1", 80))],
)
def test_parse_listen(listen: str, parsed: tuple[str, int]) -> None:
    assert parse_listen(listen) == parsed


@pytest.mark.parametrize("listen", ["9100", "host:", ":80", "host:99999", "h:x"])
def test_parse_listen_refuses_bad_addresses(listen: str) -> None:
    with pytest.raises(ValueError):
        parse_listen(listen)


@pytest.mark.parametrize(
    ("host", "loopback"),
    [("127.0.0.1", True), ("::1", True), ("localhost", True), ("0.0.0.0", False)],
)
def test_is_loopback(host: str, loopback: bool) -> None:
    assert is_loopback(host) is loopback
