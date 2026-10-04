"""An OTLP/HTTP traces endpoint for outage tests: it stores first, then answers as told.

Each export is decoded and its spans appended to ``--store``, one JSON line
per span, flushed and fsynced before any answer is sent. A span counted
here was therefore really stored, even when its answer never reaches the
exporter: that is what the collector-side bounds compare against.

Then it waits ``--delay-seconds`` and answers ``--status``. ``GET /counts``
returns the requests, raw spans and unique spans so far, and ``--count``
reads them back from a store file after the collector has gone.

X3 of the #220 qualification, a collector slower than Stormlog's 5 s attempt
deadline, which Stormlog must count as ``unknown{timeout_after_send}``::

    python -m examples.observability.fake_collector --listen 127.0.0.1:4318 \\
        --store artifacts/x3/spans.jsonl --delay-seconds 8

    stormlog infer profile ... --otlp-endpoint http://127.0.0.1:4318

    python -m examples.observability.fake_collector --count artifacts/x3/spans.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from stormlog._export.inflate import gunzip_capped
from stormlog.infer.vllm_spans import (
    ProtobufDecodeError,
    decode_otlp_json,
    decode_otlp_protobuf,
    parse_listen_address,
)

MAX_BODY_BYTES = 64 * 1024 * 1024


class FakeCollector:
    """Store each export's spans durably, then answer after a delay."""

    def __init__(
        self,
        listen: str,
        store: Path,
        *,
        delay_seconds: float = 0.0,
        status: int = 200,
        refuse_first: int = 0,
    ) -> None:
        self.store = Path(store)
        self.delay_seconds = delay_seconds
        self.status = status
        self.refuse_first = refuse_first
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._requests = 0
        self._raw = 0
        self._unique: set[tuple[str, str]] = set()
        host, port = parse_listen_address(listen)
        self._server = ThreadingHTTPServer((host, port), _handler(self))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        host = str(self._server.server_address[0])
        return f"{host}:{self._server.server_port}"

    def start(self) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-collector", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(5)
        self._server.server_close()

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests": self._requests,
                "raw_spans": self._raw,
                "unique_spans": len(self._unique),
            }

    def receive(self, body: bytes, media: str) -> int:
        """Store an export's spans, fsynced; the status to answer with.

        One of the first ``refuse_first`` exports is refused with 503 and
        not stored, as a conformant collector that is unavailable does.
        """
        spans = (
            decode_otlp_json(json.loads(body))
            if media == "application/json"
            else decode_otlp_protobuf(body)
        )
        lines = [
            json.dumps(
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "name": span.name,
                    "received_at_ns": time.time_ns(),
                }
            )
            + "\n"
            for span in spans
        ]
        with self._lock:
            self._requests += 1
            if self._requests <= self.refuse_first:
                return 503
            with self.store.open("a", encoding="utf-8") as handle:
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            self._raw += len(spans)
            self._unique.update((s.trace_id or "", s.span_id or "") for s in spans)
        self._stop.wait(self.delay_seconds)
        return self.status


def _handler(collector: FakeCollector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/counts":
                self._answer(404)
                return
            self._answer(200, json.dumps(collector.counts()).encode(), "json")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                self._answer(413)
                return
            body = self.rfile.read(length)
            try:
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    inflated = gunzip_capped(body, MAX_BODY_BYTES)
                    if inflated is None:
                        self._answer(413)
                        return
                    body = inflated
                status = collector.receive(body, self.headers.get_content_type())
            except (ValueError, zlib.error, ProtobufDecodeError):
                self._answer(400)
                return
            self._answer(status)

        def _answer(self, status: int, body: bytes = b"", kind: str = "") -> None:
            self.send_response(status)
            media = "application/json" if kind else "application/x-protobuf"
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass  # the exporter gave up waiting; the spans are stored

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

    return Handler


def count_store(path: Path) -> dict[str, int]:
    """Raw and unique spans in a store file; a partial last line is skipped."""
    raw = 0
    unique: set[tuple[str, str]] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            span = json.loads(line)
        except ValueError:
            continue
        raw += 1
        unique.add((span.get("trace_id") or "", span.get("span_id") or ""))
    return {"raw_spans": raw, "unique_spans": len(unique)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen", default="127.0.0.1:4318", metavar="HOST:PORT")
    parser.add_argument("--store", type=Path, metavar="PATH")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--status", type=int, default=200)
    parser.add_argument(
        "--refuse-first",
        type=int,
        default=0,
        metavar="N",
        help="Answer the first N exports with 503, without storing them.",
    )
    parser.add_argument(
        "--count", type=Path, metavar="PATH", help="Print a store file's counts."
    )
    args = parser.parse_args(argv)
    if args.count is not None:
        print(json.dumps(count_store(args.count)))
        return 0
    if args.store is None:
        parser.error("--store is required unless --count is given")
    collector = FakeCollector(
        args.listen,
        args.store,
        delay_seconds=args.delay_seconds,
        status=args.status,
        refuse_first=args.refuse_first,
    )
    collector.start()
    print(f"fake collector on http://{collector.address}/v1/traces", flush=True)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    try:
        stopped.wait()
    except KeyboardInterrupt:
        pass
    collector.stop()
    print(json.dumps(collector.counts()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
