"""An OTLP/HTTP traces endpoint that answers as scripted and keeps what it stored.

Each export is answered by the next ``Reply`` in the script, or by the
script function given the export's index. A reply can store the spans and
then reset the connection, never answer, or dribble its status line, so a
test can build exactly the histories the delivery accounting must settle.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from stormlog.infer.vllm_spans import RawSpan, decode_otlp_json, decode_otlp_protobuf

ANSWER = "answer"
RESET = "reset"
SILENT = "silent"
DRIBBLE = "dribble"
SHORT = "short"


@dataclass(frozen=True)
class Reply:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/x-protobuf"
    headers: tuple[tuple[str, str], ...] = ()
    action: str = ANSWER
    # Whether the spans count as stored, whatever the answer says.
    store: bool = True


@dataclass
class Received:
    headers: dict[str, str]
    body: bytes
    spans: list[RawSpan]
    stored: bool
    at: float = field(default_factory=time.monotonic)


@dataclass
class FakeCollector:
    script: Sequence[Reply] | Callable[[int], Reply] = ()
    received: list[Received] = field(default_factory=list)
    url: str = ""
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reply_for(self, index: int) -> Reply:
        if callable(self.script):
            return self.script(index)
        return self.script[index] if index < len(self.script) else Reply()

    def stored_spans(self) -> list[RawSpan]:
        with self._lock:
            return [span for r in self.received if r.stored for span in r.spans]

    def unique_stored(self) -> set[tuple[str | None, str | None]]:
        return {(s.trace_id, s.span_id) for s in self.stored_spans()}

    def record(self, received: Received) -> int:
        with self._lock:
            self.received.append(received)
            return len(self.received) - 1


def _handler(collector: FakeCollector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            media = self.headers.get_content_type()
            spans = (
                decode_otlp_protobuf(body)
                if media == "application/x-protobuf"
                else decode_otlp_json(json.loads(body))
            )
            headers = {k.lower(): v for k, v in self.headers.items()}
            index = len(collector.received)
            reply = collector.reply_for(index)
            collector.record(Received(headers, body, spans, reply.store))
            self.close_connection = True
            getattr(self, f"_{reply.action}")(reply)

        def _answer(self, reply: Reply) -> None:
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.content_type)
            for name, value in reply.headers:
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

        def _short(self, reply: Reply) -> None:
            # Promise more than is sent, then close.
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.content_type)
            self.send_header("Content-Length", str(len(reply.body) + 100))
            self.end_headers()
            self.wfile.write(reply.body)
            self.wfile.flush()

        def _reset(self, _reply: Reply) -> None:
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.connection.close()

        def _silent(self, _reply: Reply) -> None:
            collector._stop.wait(30)

        def _dribble(self, _reply: Reply) -> None:
            for byte in b"HTTP/1.1 200 OK\r\n":
                if collector._stop.wait(0.2):
                    return
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except OSError:
                    return

        def handle_one_request(self) -> None:
            with contextlib.suppress(OSError):
                super().handle_one_request()

        def finish(self) -> None:
            with contextlib.suppress(OSError):
                super().finish()

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

    return Handler


@contextlib.contextmanager
def running(
    script: Sequence[Reply] | Callable[[int], Reply] = (), host: str = "127.0.0.1"
) -> Iterator[FakeCollector]:
    collector = FakeCollector(script)
    server = ThreadingHTTPServer((host, 0), _handler(collector))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    collector.url = f"http://{host}:{server.server_port}/v1/traces"
    try:
        yield collector
    finally:
        collector._stop.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
