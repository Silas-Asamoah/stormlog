"""The /metrics server: admission before threads, and a deadline per connection."""

import http.client
import socket
import threading
import time
import tracemalloc
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


def test_head_answers_the_length_without_the_body(server: MetricsServer) -> None:
    sock = _idle(server)
    try:
        sock.sendall(b"HEAD /metrics HTTP/1.1\r\nHost: x\r\n\r\n")
        head, _, body = _read_all(sock).partition(b"\r\n\r\n")
    finally:
        sock.close()
    assert head.startswith(b"HTTP/1.1 200")
    assert f"Content-Length: {len(BODY)}".encode() in head
    assert body == b""


def test_other_paths_are_not_found(server: MetricsServer) -> None:
    assert _get(server, "/other").status == 404
    assert _get(server, "/metrics?x=1").status == 200
    assert server.stats.not_found == 1


def test_connections_past_the_limit_are_refused_before_a_thread_starts() -> None:
    # A long deadline, so the idle pair still holds both slots however late
    # the third connection comes on a loaded machine.
    metrics = MetricsServer(
        "127.0.0.1:0", RenderCache(lambda: BODY), max_connections=2, deadline=30
    )
    metrics.start()
    idle = [_idle(metrics), _idle(metrics)]  # send nothing, hold both slots
    try:
        assert _wait_for(lambda: metrics.stats.active == 2)
        threads_before = threading.active_count()
        refused = _idle(metrics)
        assert refused.recv(1024).startswith(b"HTTP/1.1 503")
        refused.close()
        assert metrics.stats.rejected_busy == 1
        assert threading.active_count() == threads_before
    finally:
        for sock in idle:
            sock.close()
        metrics.stop()


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
        # The cut ends the send with an error: counted once, as the timeout.
        assert metrics.stats.errors == 0
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


def _read_all(sock: socket.socket) -> bytes:
    chunks = []
    while chunk := sock.recv(65536):
        chunks.append(chunk)
    return b"".join(chunks)


def test_a_head_of_a_few_kilobytes_is_served(server: MetricsServer) -> None:
    sock = _idle(server)
    try:
        pad = b"X-Pad: " + b"a" * 8000 + b"\r\n"
        sock.sendall(b"GET /metrics HTTP/1.1\r\nHost: x\r\n" + pad + b"\r\n")
        assert _read_all(sock).startswith(b"HTTP/1.1 200")
    finally:
        sock.close()


def test_a_head_past_sixteen_kibibytes_is_answered_431_and_closed(
    server: MetricsServer,
) -> None:
    sock = _idle(server)
    try:
        # Never ends: a server without a head limit waits for the blank line.
        pad = b"X-Pad: " + b"a" * (16 * 1024 + 100) + b"\r\n"
        sock.sendall(b"GET /metrics HTTP/1.1\r\nHost: x\r\n" + pad)
        assert _read_all(sock).startswith(b"HTTP/1.1 431")
        assert server.stats.bad_request == 1
        assert server.stats.timeout == 0
    finally:
        sock.close()


def test_a_request_line_past_sixteen_kibibytes_is_answered_431(
    server: MetricsServer,
) -> None:
    sock = _idle(server)
    try:
        sock.sendall(b"GET /metrics?" + b"a" * (16 * 1024 + 100))
        assert _read_all(sock).startswith(b"HTTP/1.1 431")
        assert server.stats.bad_request == 1
    finally:
        sock.close()


def test_header_floods_from_every_slot_hold_little_memory() -> None:
    # The review's flood: four connections, each sending 98 header lines of
    # 64 KiB inside the deadline. The stock parser holds them all (25 MiB),
    # then builds several copies of the head when it ends.
    metrics = MetricsServer(
        "127.0.0.1:0", RenderCache(lambda: BODY), max_connections=4, deadline=5
    )
    metrics.start()
    line = b"X-Pad: " + b"a" * (65536 - 10) + b"\r\n"
    sockets = [_idle(metrics) for _ in range(4)]

    def flood(sock: socket.socket) -> None:
        try:
            sock.sendall(b"GET /metrics HTTP/1.1\r\nHost: x\r\n")
            for _ in range(98):
                sock.sendall(line)
            sock.sendall(b"\r\n")
        except OSError:
            pass  # cut off by the server

    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        floods = [threading.Thread(target=flood, args=(sock,)) for sock in sockets]
        for thread in floods:
            thread.start()
        for thread in floods:
            thread.join(10)
        time.sleep(0.5)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        for sock in sockets:
            sock.close()
        metrics.stop()
    assert peak - baseline < 2 * 1024 * 1024
    assert metrics.stats.bad_request == 4


def test_a_handler_thread_that_cannot_start_gives_its_slot_back(
    server: MetricsServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_start = threading.Thread.start
    failures = [1, 1]

    def failing_start(thread: threading.Thread) -> None:
        if failures and "process_request_thread" in repr(getattr(thread, "_target")):
            failures.pop()
            raise RuntimeError("can't start new thread")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    for _ in range(2):  # as many as the server has slots
        sock = _idle(server)
        sock.sendall(b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n")
        try:
            _read_all(sock)  # closed unanswered
        except OSError:
            pass
        sock.close()
    assert _wait_for(lambda: server.stats.errors == 2)
    assert _get(server).status == 200
    assert _wait_for(lambda: server.stats.active == 0)
    assert server.stats.rejected_busy == 0


def test_a_stopped_watchdog_costs_no_slot(server: MetricsServer) -> None:
    # A watchdog stopped under the server (shared, or the server restarted)
    # refuses to arm; each connection is then an error, never a lost slot.
    assert _get(server).status == 200
    server.watchdog.stop()
    for _ in range(4):  # twice the slots
        sock = _idle(server)
        try:
            _read_all(sock)
        except OSError:
            pass
        sock.close()
    # The first request's handler ends on its own thread, when it sees the
    # client close, so its slot comes back in its own time.
    assert _wait_for(lambda: server.stats.errors == 4 and server.stats.active == 0)
    assert server.stats.rejected_busy == 0


class _NeverFires:
    """A watchdog that never acts, as one whose thread has died."""

    def arm(self, sock: socket.socket, deadline: float) -> int:
        return 1

    def disarm(self, token: int) -> bool:
        return True

    def fired(self, token: int) -> bool:
        return False

    def stop(self) -> None:
        pass


def test_without_its_watchdog_an_idle_client_is_cut_a_second_late() -> None:
    metrics = MetricsServer(
        "127.0.0.1:0",
        RenderCache(lambda: BODY),
        deadline=0.3,
        watchdog=_NeverFires(),  # type: ignore[arg-type]
    )
    metrics.start()
    try:
        sock = _idle(metrics)
        started = time.monotonic()
        assert sock.recv(10) == b""  # the per-read fallback timeout closed it
        assert 1.0 <= time.monotonic() - started < 4
        sock.close()
    finally:
        metrics.stop()


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
