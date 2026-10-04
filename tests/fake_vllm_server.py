"""A stand-in for `vllm serve` that the experiment runner can launch.

Serves /health, /version, /v1/models and non-streamed chat completions.
`--die-after N` exits after N completions, as a crashing server would.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Handler(BaseHTTPRequestHandler):
    served = 0
    die_after: int | None = None
    latency = 0.0

    def do_GET(self) -> None:  # noqa: N802
        answers: dict[str, Any] = {
            "/health": {},
            "/version": {"version": "0.30.0"},
            "/v1/models": {"data": [{"id": "m", "max_model_len": 4096}]},
        }
        if self.path not in answers:
            self.send_error(404)
            return
        self._json(answers[self.path])

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        time.sleep(type(self).latency)
        self._json(
            {
                "id": "chatcmpl-x",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            }
        )
        type(self).served += 1
        limit = type(self).die_after
        if limit is not None and type(self).served >= limit:
            os._exit(1)

    def _json(self, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--die-after", type=int, default=None)
    parser.add_argument("--latency", type=float, default=0.0)
    args = parser.parse_args()
    Handler.die_after = args.die_after
    Handler.latency = args.latency
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fake vLLM on {args.port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
