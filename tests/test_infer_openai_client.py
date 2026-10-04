"""Where a request failed: before the connection, while sending, or after."""

from __future__ import annotations

import contextlib
import socket
import threading
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stormlog.infer.openai_client import (
    ConnectError,
    EndpointHTTPError,
    OpenAIChatCompletionsClient,
)
from stormlog.infer.profile import classify_failure


class _RedirectHandler(BaseHTTPRequestHandler):
    posts: list[str] = []
    gets: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).posts.append(self.path)
        self.send_response(302)
        self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        type(self).gets.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _redirecting_server() -> Iterator[str]:
    _RedirectHandler.posts = []
    _RedirectHandler.gets = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _client(endpoint: str, timeout: float = 5.0) -> OpenAIChatCompletionsClient:
    return OpenAIChatCompletionsClient(
        endpoint=endpoint, model="m", timeout_seconds=timeout
    )


def _complete(client: OpenAIChatCompletionsClient) -> None:
    client.complete(
        prompt="hello", output_tokens=4, stream=False, stream_include_usage=False
    )


def test_a_refused_connection_is_unreachable() -> None:
    client = _client(f"http://127.0.0.1:{_closed_port()}/v1/chat/completions")

    with pytest.raises(urllib.error.URLError) as raised:
        _complete(client)

    assert isinstance(raised.value.reason, ConnectError)
    assert isinstance(raised.value.reason.cause, ConnectionRefusedError)
    assert classify_failure(raised.value) == ("unreachable", None)


def test_a_redirect_is_an_http_error_and_is_not_followed() -> None:
    # urllib would re-send the POST as a GET elsewhere, after the first server
    # had already received it.
    with _redirecting_server() as endpoint:
        with pytest.raises(EndpointHTTPError) as raised:
            _complete(_client(endpoint))

    assert raised.value.status == 302
    assert classify_failure(raised.value) == ("error", 302)
    assert _RedirectHandler.posts == ["/v1/chat/completions"]
    assert _RedirectHandler.gets == []


def test_a_failure_while_sending_is_delivery_unknown() -> None:
    # The server reads the start of the request, then resets the connection,
    # so a large body fails while it is being sent, after connect() returned.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def accept_and_close() -> None:
        connection, _address = listener.accept()
        connection.recv(65536)
        connection.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00"
        )
        connection.close()

    thread = threading.Thread(target=accept_and_close, daemon=True)
    thread.start()
    client = _client(f"http://127.0.0.1:{port}/v1/chat/completions")
    try:
        with pytest.raises(urllib.error.URLError) as raised:
            client.complete(
                prompt="x" * (32 * 1024 * 1024),
                output_tokens=4,
                stream=False,
                stream_include_usage=False,
            )
    finally:
        thread.join(timeout=5)
        listener.close()

    assert not isinstance(raised.value.reason, ConnectError)
    assert classify_failure(raised.value) == ("delivery_unknown", None)


@pytest.mark.parametrize("read_first", [False, True], ids=["unread", "read"])
def test_a_close_before_any_response_byte_is_delivery_unknown(
    read_first: bool,
) -> None:
    # A small body fits in the socket buffer, so the request is "sent" before
    # the server has read a byte of it; the failure comes while waiting for
    # the response. The server may or may not have taken the request.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def accept_and_close() -> None:
        connection, _address = listener.accept()
        if read_first:
            connection.recv(65536)
        else:
            # Wait until the request is in the server's buffer, unread; a
            # reset before connect() returns would be unreachable instead.
            connection.recv(1, socket.MSG_PEEK)
            connection.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                b"\x01\x00\x00\x00\x00\x00\x00\x00",
            )
        connection.close()

    thread = threading.Thread(target=accept_and_close, daemon=True)
    thread.start()
    client = _client(f"http://127.0.0.1:{port}/v1/chat/completions")
    try:
        with pytest.raises(OSError) as raised:
            _complete(client)
    finally:
        thread.join(timeout=5)
        listener.close()

    assert classify_failure(raised.value) == ("delivery_unknown", None)


def test_a_tls_handshake_that_fails_is_unreachable() -> None:
    # The HTTPS connection is tracked too: the handshake is part of connect().
    with _redirecting_server() as endpoint:
        client = _client(endpoint.replace("http://", "https://"))
        with pytest.raises(urllib.error.URLError) as raised:
            _complete(client)

    assert isinstance(raised.value.reason, ConnectError)
    assert classify_failure(raised.value) == ("unreachable", None)
    assert _RedirectHandler.posts == []


def test_a_name_that_does_not_resolve_is_unreachable() -> None:
    client = _client("http://stormlog-test.invalid/v1/chat/completions")
    with pytest.raises(urllib.error.URLError) as raised:
        _complete(client)
    assert isinstance(raised.value.reason, ConnectError)
    assert isinstance(raised.value.reason.cause, socket.gaierror)
    assert classify_failure(raised.value) == ("unreachable", None)


def test_environment_proxies_are_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    # Through a proxy, connect() reaches the proxy, and a dead server reads as
    # the proxy's HTTP 502: an error that counts as accepted.
    with _redirecting_server() as proxy:
        proxy_url = proxy.rsplit("/v1/", 1)[0]
        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
            monkeypatch.setenv(name, proxy_url)
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
        client = _client(f"http://127.0.0.1:{_closed_port()}/v1/chat/completions")
        with pytest.raises(urllib.error.URLError) as raised:
            _complete(client)

    assert classify_failure(raised.value) == ("unreachable", None)
    assert _RedirectHandler.posts == [] and _RedirectHandler.gets == []
