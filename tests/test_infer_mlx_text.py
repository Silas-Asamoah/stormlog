"""Portable MLX text adapter protocol qualification."""

import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stormlog.infer.completions_client import OpenAICompletionsClient
from stormlog.infer.config import resolve_endpoint
from stormlog.infer.engine_adapters import get_adapter
from stormlog.infer.http_transport import EndpointHTTPError, RequestDeadlineExceeded


@contextlib.contextmanager
def endpoint(body, content_type="application/json", *, trickle=False):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(
                (
                    self.path,
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                    self.headers.get("X-Request-Id"),
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if trickle:
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/completions", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def complete(url, *, stream=False, **kwargs):
    return OpenAICompletionsClient(
        endpoint=url,
        model="/opaque/model alias",
        timeout_seconds=kwargs.pop("timeout_seconds", 2),
        **kwargs,
    ).complete(
        prompt="actual input",
        output_tokens=8,
        stream=stream,
        stream_include_usage=True,
        request_id="client-id",
    )


@pytest.mark.parametrize("engine", ["omlx", "mlx-lm"])
@pytest.mark.parametrize("stream", [False, True])
def test_completion_protocol_and_separate_ids(engine, stream):
    get_adapter(engine, "completion")
    payload = {
        "id": "server-generated",
        "system_fingerprint": "build",
        "choices": [{"text": "answer", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
    }
    body = json.dumps(payload).encode()
    if stream:
        body = b": heartbeat\n\n" + b"data: " + body + b"\n\ndata: [DONE]\n\n"
    with endpoint(body) as (url, calls):
        result = complete(url, stream=stream)
    assert result.text == "answer"
    assert result.response_id == "server-generated"
    assert result.system_fingerprint == "build"
    assert result.usage == {"completion_tokens": 1}
    assert len(calls) == 1
    path, sent, request_id = calls[0]
    assert path == "/v1/completions" and request_id == "client-id"
    assert sent["prompt"] == "actual input" and "messages" not in sent
    assert sent["model"] == "/opaque/model alias"


def test_total_deadline_stops_continuous_body():
    with endpoint(b" " * 200, trickle=True) as (url, calls):
        started = time.perf_counter()
        with pytest.raises(RequestDeadlineExceeded):
            complete(url, timeout_seconds=0.12)
        assert time.perf_counter() - started < 0.5
    assert len(calls) == 1


def test_bound_and_embedded_error_are_terminal_and_secret_safe():
    with endpoint(b"x" * 100) as (url, calls):
        with pytest.raises(ValueError, match="byte limit"):
            complete(url, max_response_bytes=20)
    assert len(calls) == 1
    secret = "my-very-private-api-key"
    with endpoint(json.dumps({"error": secret}).encode()) as (url, calls):
        with pytest.raises(EndpointHTTPError) as raised:
            complete(url, api_key=secret)
    assert secret not in str(raised.value) and raised.value.status == 200
    assert len(calls) == 1


def test_endpoint_resolution_and_capability_errors():
    assert (
        resolve_endpoint(
            endpoint=None, base_url="http://localhost/v1/", task="completion"
        )
        == "http://localhost/v1/completions"
    )
    assert (
        resolve_endpoint(endpoint="http://localhost/custom", base_url=None, task="tts")
        == "http://localhost/custom"
    )
    with pytest.raises(ValueError):
        get_adapter("mlx-lm", "tts")
