"""A fake vLLM ``/metrics`` endpoint and ledger readers for watcher tests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from stormlog.telemetry_sink import resolve_telemetry_sink_segment_paths
from tests.vllm_scrape_helpers import START, exposition

CONFIG_FORMAT = "stormlog.infer.watch_config"


class FakeMetrics:
    """Serves vLLM-shaped metrics whose waiting queue a test sets.

    Each scrape advances the token counters, so the engine looks alive,
    unless ``advance`` is off; ``start`` is the exporter's process start.
    """

    def __init__(self) -> None:
        self.waiting = 0.0
        self.running = 4.0
        self.tokens = 0.0
        self.advance = True
        self.start = START
        self.status = 200
        self.delay = 0.0
        self.scrapes = 0
        self._lock = threading.Lock()

    def body(self) -> bytes:
        with self._lock:
            self.scrapes += 1
            self.tokens += 5 if self.advance else 0
            return exposition(
                start=self.start,
                gauges={
                    "vllm:num_requests_waiting": self.waiting,
                    "vllm:num_requests_running": self.running,
                },
                counters={
                    "vllm:generation_tokens_total": self.tokens,
                    "vllm:prompt_tokens_total": self.tokens,
                },
            ).encode()


@contextmanager
def serve_metrics(metrics: FakeMetrics) -> Iterator[str]:
    """Serve ``metrics`` on a free local port; yields the base URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            time.sleep(metrics.delay)
            if metrics.status != 200:
                self.send_response(metrics.status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = metrics.body()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def watch_config(base_url: str, **overrides: Any) -> dict[str, Any]:
    """A fast config: 0.1 s ticks and one gauge trigger on the waiting queue."""
    payload: dict[str, Any] = {
        "format": CONFIG_FORMAT,
        "version": 1,
        "server": {"base_url": base_url},
        "tick_seconds": 0.1,
        "history": {"seconds": 30},
        "incident": {"pre_seconds": 5, "post_seconds": 0.5},
        "triggers": [
            {
                "id": "queue",
                "kind": "metric",
                "window_seconds": 0.3,
                "hold_seconds": 0.4,
                "gauge": {"family": "vllm:num_requests_waiting", "at_least": 8},
            }
        ],
    }
    payload.update(overrides)
    return payload


def read_ledger(root: Path) -> list[dict[str, Any]]:
    """Every record the watcher's ledger holds, in order."""
    records: list[dict[str, Any]] = []
    for segment in resolve_telemetry_sink_segment_paths(root / "ledger"):
        for line in segment.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def of_type(records: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event_type") == event_type]
