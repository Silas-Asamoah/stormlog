"""An OTLP/HTTP traces endpoint for outage tests: it stores first, then answers as told.

Each export is decoded and its spans appended to ``--store``, one JSON line
per span, flushed and fsynced before any answer is sent. A span counted
here was therefore really stored, even when its answer never reaches the
exporter: that is what the collector-side bounds compare against.

Then it waits ``--delay-seconds`` and answers ``--status``, in the
request's encoding, as an OTLP/HTTP collector does: with
``--partial-rejected N`` it keeps all but the last N spans of each export
and says so in a partial success, and ``--retry-after S`` adds that header
to its refusals. Like a collector, it answers 404 off ``/v1/traces`` and
415 for a body that is neither protobuf nor JSON. ``GET /counts`` returns
the requests, raw spans and unique spans so far, and ``--count`` reads them
back from a store file after the collector has gone.

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
TRACES_PATH = "/v1/traces"
PROTOBUF = "application/x-protobuf"
JSON = "application/json"


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
        partial_rejected: int = 0,
        retry_after: float | None = None,
    ) -> None:
        self.store = Path(store)
        self.delay_seconds = delay_seconds
        self.status = status
        self.refuse_first = refuse_first
        self.partial_rejected = partial_rejected
        self.retry_after = retry_after
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

    def receive(self, body: bytes, media: str) -> tuple[int, int]:
        """Store an export's spans, fsynced; the status, and the spans rejected.

        One of the first ``refuse_first`` exports is refused with 503 and
        not stored, as a conformant collector that is unavailable does. The
        last ``partial_rejected`` spans of an export are not stored.
        """
        spans = (
            decode_otlp_json(json.loads(body))
            if media == JSON
            else decode_otlp_protobuf(body)
        )
        rejected = min(self.partial_rejected, len(spans))
        spans = spans[: len(spans) - rejected]
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
                return 503, 0
            with self.store.open("a", encoding="utf-8") as handle:
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            self._raw += len(spans)
            self._unique.update((s.trace_id or "", s.span_id or "") for s in spans)
        self._stop.wait(self.delay_seconds)
        return self.status, rejected


def _handler(collector: FakeCollector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/counts":
                self._answer(404)
                return
            self._answer(200, json.dumps(collector.counts()).encode(), True)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                self._answer(413)
                return
            body = self.rfile.read(length)
            media = self.headers.get_content_type()
            if self.path.split("?", 1)[0] != TRACES_PATH:
                self._answer(404)
                return
            if media not in (PROTOBUF, JSON):
                self._answer(415)
                return
            try:
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    inflated = gunzip_capped(body, MAX_BODY_BYTES)
                    if inflated is None:
                        self._answer(413)
                        return
                    body = inflated
                status, rejected = collector.receive(body, media)
            except (ValueError, zlib.error, ProtobufDecodeError):
                self._answer(400)
                return
            if status == 200:
                self._answer(200, _export_response(media, rejected), media == JSON)
                return
            self._answer(status, retry_after=collector.retry_after)

        def _answer(
            self,
            status: int,
            body: bytes = b"",
            json_body: bool = False,
            retry_after: float | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", JSON if json_body else PROTOBUF)
            self.send_header("Content-Length", str(len(body)))
            if retry_after is not None and status in (429, 503):
                self.send_header("Retry-After", f"{retry_after:g}")
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass  # the exporter gave up waiting; the spans are stored

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

    return Handler


def _export_response(media: str, rejected: int) -> bytes:
    """An ExportTraceServiceResponse in the request's encoding."""
    message = f"{rejected} spans rejected by the fake collector" if rejected else ""
    if media == JSON:
        partial = (
            {"rejectedSpans": str(rejected), "errorMessage": message}
            if rejected
            else {}
        )
        return json.dumps({"partialSuccess": partial} if partial else {}).encode()
    if not rejected:
        return b""
    # partial_success (1) { rejected_spans (1): int64, error_message (2) }
    text = message.encode()
    inner = b"\x08" + _varint(rejected) + b"\x12" + _varint(len(text)) + text
    return b"\x0a" + _varint(len(inner)) + inner


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        low, value = value & 0x7F, value >> 7
        out.append(low | (0x80 if value else 0))
        if not value:
            return bytes(out)


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
        "--partial-rejected",
        type=int,
        default=0,
        metavar="N",
        help="Keep all but the last N spans of each export, as a partial success.",
    )
    parser.add_argument(
        "--retry-after",
        type=float,
        metavar="SECONDS",
        help="Add Retry-After to each 429 or 503.",
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
        partial_rejected=args.partial_rejected,
        retry_after=args.retry_after,
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
