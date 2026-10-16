"""Total request deadlines include DNS, headers, body and error-body reads."""

import contextlib
import socket
import threading
import time
import urllib.error

import pytest

from stormlog.infer.completions_client import OpenAICompletionsClient
from stormlog.infer.http_transport import (
    ConnectError,
    EndpointHTTPError,
    RequestDeadlineExceeded,
)
from stormlog.infer.openai_client import OpenAIChatCompletionsClient
from stormlog.infer.profile import classify_failure


def client(url):
    return OpenAIChatCompletionsClient(
        endpoint=url, model="opaque", timeout_seconds=0.12
    )


def complete(value):
    return value.complete(
        prompt="input", output_tokens=1, stream=False, stream_include_usage=False
    )


@pytest.mark.parametrize(
    "client_class", [OpenAIChatCompletionsClient, OpenAICompletionsClient]
)
@pytest.mark.parametrize(
    "url",
    [
        "http://user:password@localhost/v1/chat/completions",
        "http:///v1/chat/completions",
        "http://localhost:bad/v1/chat/completions",
        "http://localhost/v1/chat/completions#fragment",
        "file:///tmp/model",
    ],
)
def test_endpoint_validation_precedes_transport_creation(
    monkeypatch, client_class, url
):
    def unexpected_transport():
        pytest.fail(
            "An invalid endpoint must be rejected before creating its transport"
        )

    monkeypatch.setattr(
        "stormlog.infer.openai_client.inference_opener", unexpected_transport
    )
    with pytest.raises(ValueError):
        client_class(endpoint=url, model="opaque", timeout_seconds=1)


def test_dns_deadline_is_bounded_without_sending_a_request(monkeypatch):
    def stalled(*args):
        time.sleep(0.3)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", stalled)
    started = time.perf_counter()
    with pytest.raises(urllib.error.URLError) as raised:
        complete(client("http://invalid.local/v1/chat/completions"))
    assert time.perf_counter() - started < 0.25
    assert isinstance(raised.value.reason, ConnectError)
    assert classify_failure(raised.value) == ("unreachable", None)


@contextlib.contextmanager
def trickling_socket(wire, *, trickle_headers):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve():
        connection, _ = listener.accept()
        with connection:
            connection.recv(65536)
            try:
                if not trickle_headers:
                    header, _, body = wire.partition(b"\r\n\r\n")
                    connection.sendall(header + b"\r\n\r\n")
                else:
                    body = wire
                for byte in body:
                    connection.sendall(bytes([byte]))
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        listener.close()
        thread.join(2)


def test_trickling_headers_cannot_extend_total_deadline():
    with trickling_socket(
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}", trickle_headers=True
    ) as url:
        started = time.perf_counter()
        with pytest.raises(RequestDeadlineExceeded):
            complete(client(url))
        assert time.perf_counter() - started < 0.3


def test_error_body_deadline_preserves_rejected_http_status():
    with trickling_socket(
        b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 100\r\n\r\n" + b"x" * 100,
        trickle_headers=False,
    ) as url:
        with pytest.raises(EndpointHTTPError) as raised:
            complete(client(url))
    assert raised.value.status == 429
    assert classify_failure(raised.value) == ("rejected", 429)
