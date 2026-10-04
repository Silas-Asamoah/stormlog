"""The fake engine's HTTP front end, shaped like vLLM 0.30's API server."""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .config import MAX_MODEL_LEN, VLLM_VERSION, Controls, FakeEngineConfig
from .engine import Engine, FakeRequest, prompt_tokens
from .hold import Hold
from .hook_log import HookLog
from .identities import Identities
from .metrics import render_metrics
from .profiler import FakeProfiler
from .spans import SpanExporter

TOKEN_TEXT = " tok"


class FakeEngine:
    """A running fake engine: the step loop plus its HTTP server.

    ``with FakeEngine(FakeEngineConfig()) as engine:`` starts both on a free
    loopback port; ``engine.base_url`` is the server root.
    """

    def __init__(self, config: FakeEngineConfig | None = None) -> None:
        self.config = config or FakeEngineConfig()
        self.controls = Controls()
        self.ids = Identities(self.config.seed)
        self.engine = Engine(self.config)
        self._frontend = Hold()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self.hook: HookLog | None = None
        self.profiler: FakeProfiler | None = None
        self.spans: SpanExporter | None = None
        # Each exception a request handler raised, with its traceback.
        self.server_errors: list[str] = []

    # ------------------------------------------------------------ lifecycle

    def start(self) -> FakeEngine:
        """Start every part; if one fails, such as the bind, stop the parts
        already started before raising.

        Raises:
            RuntimeError: when it was already started.
        """
        if self._server is not None:
            raise RuntimeError("the fake engine is already started")
        try:
            self._start_parts()
        except BaseException:
            self.stop()
            raise
        return self

    def _start_parts(self) -> None:
        if self.config.hook_dir is not None:
            self.hook = HookLog(self.config.hook_dir, self.config)
            self.engine.observers.append(self.hook)
        producer = (
            self.hook.producer
            if self.hook is not None
            else f"vllm:fake:{os.getpid()}:{self.engine.start_ns}"
        )
        self.profiler = FakeProfiler(self.engine, self.config, self.controls, producer)
        self.engine.observers.append(self.profiler)
        if self.config.spans_endpoint is not None:
            self.spans = SpanExporter(
                self.config.spans_endpoint,
                self.controls,
                interval=self.config.span_export_seconds,
                encoding=self.config.span_encoding,
                timeout=self.config.span_export_timeout_seconds,
                ids=self.ids,
            )
            self.engine.observers.append(self.spans)
            self.spans.start()
        self._server = _Server((self.config.host, self.config.port), _Handler)
        self._server.fake = self
        self.engine.start()
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-engine-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop every part that started; safe to call again."""
        self._frontend.resume()
        self.engine.stop()
        if self._server is not None:
            # shutdown() waits for serve_forever to end, so only once it ran.
            if self._thread is not None:
                self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.profiler is not None:
            self.profiler.shutdown()
        if self.spans is not None:
            self.spans.close()
        if self.hook is not None:
            self.hook.close()

    def __enter__(self) -> FakeEngine:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("the fake engine is not started")
        host, port = self._server.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else str(host)
        return f"http://{name}:{port}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def metrics_url(self) -> str:
        return f"{self.base_url}/metrics"

    # ------------------------------------------------------------ faults

    def pause_frontend(self, seconds: float | None = None) -> None:
        """Hold every API response (as if the API server were stopped); the
        engine keeps stepping. ``/_fault/`` routes still answer. Pauses stack,
        as the engine's do."""
        self._frontend.pause(seconds)

    def resume_frontend(self) -> None:
        self._frontend.resume()

    @property
    def frontend_paused(self) -> bool:
        return self._frontend.held

    def pause_engine(self, seconds: float | None = None) -> None:
        self.engine.pause(seconds)

    def resume_engine(self) -> None:
        self.engine.resume()

    def wait_frontend(self) -> None:
        self._frontend.wait()

    # ------------------------------------------------------------ requests

    def new_request(
        self, body: dict[str, Any], request_id: str | None, traceparent: str | None
    ) -> FakeRequest:
        # vLLM's random_uuid() when no X-Request-Id names the request.
        external = f"chatcmpl-{request_id or self.ids.hex(16)}"
        internal = (
            f"{external}-{self.ids.hex(4)}"
            if self.config.request_id_randomization
            else external
        )
        text = " ".join(
            str(message.get("content") or "")
            for message in body.get("messages") or []
            if isinstance(message, dict)
        )
        limit = body.get("max_tokens") or body.get("max_completion_tokens") or 16
        request = FakeRequest(
            internal_id=internal,
            external_id=external,
            prompt=prompt_tokens(text),
            max_tokens=max(1, int(limit)),
            arrival_ns=time.time_ns(),
            traceparent=traceparent,
            top_p=_sampling(body, "top_p"),
            temperature=_sampling(body, "temperature"),
        )
        return self.engine.submit(request)

    def server_info(self) -> dict[str, Any]:
        config = self.config
        return {
            "vllm_config": {
                "model_config": {"model": config.model, "max_model_len": MAX_MODEL_LEN},
                "cache_config": {
                    "block_size": config.block_size,
                    "num_gpu_blocks": config.num_gpu_blocks,
                    "enable_prefix_caching": config.enable_prefix_caching,
                },
                "scheduler_config": {
                    "max_num_seqs": config.max_num_seqs,
                    "max_num_batched_tokens": config.max_num_batched_tokens,
                    "async_scheduling": False,
                },
                "parallel_config": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "distributed_executor_backend": "uni",
                },
                "compilation_config": {"cudagraph_mode": "NONE"},
                "profiler_config": {
                    "profiler": "torch" if config.trace_dir else None,
                    "torch_profiler_dir": (
                        str(config.trace_dir) if config.trace_dir else None
                    ),
                    "torch_profiler_use_gzip": True,
                    "torch_profiler_dump_cuda_time_total": (
                        config.torch_profiler_dump_cuda_time_total
                    ),
                },
            },
            "vllm_env": {"VLLM_SERVER_DEV_MODE": True},
            "system_env": {"fake_engine": True},
        }


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # uvicorn's backlog, which vLLM's API server keeps; the stdlib's 5 resets
    # connections in any burst of concurrent clients.
    request_queue_size = 2048
    fake: FakeEngine

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Print the handler's exception, as the stdlib does, and keep it."""
        self.fake.server_errors.append(traceback.format_exc())
        super().handle_error(request, client_address)


Route = Callable[["_Handler"], None]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def setup(self) -> None:
        super().setup()
        # Streamed chunks are small; without this, Nagle's algorithm and the
        # peer's delayed ACK hold each one for tens of milliseconds.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    @property
    def fake(self) -> FakeEngine:
        return self.server.fake

    @property
    def route(self) -> str:
        return urlparse(self.path).path

    def query(self, name: str) -> str | None:
        values = parse_qs(urlparse(self.path).query).get(name)
        return values[0] if values else None

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(GET_ROUTES)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch(POST_ROUTES)

    def _dispatch(self, routes: dict[str, Route]) -> None:
        handler = routes.get(self.route)
        if handler is None:
            self.send_json(404, {"error": f"no route {self.route}"})
            return
        if not self.route.startswith("/_fault/"):
            self.fake.wait_frontend()
        handler(self)

    def read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def read_json(self) -> dict[str, Any]:
        raw = self.read_body()
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {}

    def send_json(self, status: int, payload: Any) -> None:
        self.send_bytes(status, json.dumps(payload).encode(), "application/json")

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


def _target_pause(handler: _Handler, *, pause: bool) -> None:
    target = handler.query("target")
    seconds_text = handler.query("seconds")
    seconds = float(seconds_text) if seconds_text else None
    fake = handler.fake
    actions = {
        ("engine", True): lambda: fake.pause_engine(seconds),
        ("engine", False): fake.resume_engine,
        ("frontend", True): lambda: fake.pause_frontend(seconds),
        ("frontend", False): fake.resume_frontend,
    }
    action = actions.get((target or "", pause))
    if action is None:
        handler.send_json(400, {"error": "target must be engine or frontend"})
        return
    action()
    handler.send_json(200, {"target": target, "paused": pause})


def _fault_pause(handler: _Handler) -> None:
    handler.read_body()
    _target_pause(handler, pause=True)


def _fault_resume(handler: _Handler) -> None:
    handler.read_body()
    _target_pause(handler, pause=False)


def _fault_controls(handler: _Handler) -> None:
    changes = handler.read_json()
    controls = handler.fake.controls
    names = {field.name for field in dataclasses.fields(controls)}
    unknown = sorted(set(changes) - names)
    if unknown:
        handler.send_json(400, {"error": f"unknown controls: {', '.join(unknown)}"})
        return
    for name, value in changes.items():
        setattr(controls, name, value)
    handler.send_json(200, dataclasses.asdict(controls))


def _fault_state(handler: _Handler) -> None:
    engine = handler.fake.engine
    handler.send_json(
        200,
        {
            "steps": len(engine.steps),
            "waiting": len(engine.waiting),
            "running": len(engine.running),
            "finished": len(engine.finished),
            "preemptions": engine.stats.preemptions,
            "engine_paused": engine.paused,
            "frontend_paused": handler.fake.frontend_paused,
            "pid": os.getpid(),
        },
    )


def _fault_foreign_trace(handler: _Handler) -> None:
    handler.read_body()
    profiler = handler.fake.profiler
    if profiler is None or handler.fake.config.trace_dir is None:
        handler.send_json(409, {"error": "no torch_profiler_dir"})
        return
    handler.send_json(200, {"path": str(profiler.drop_foreign_trace())})


def _fault_span_body(handler: _Handler) -> None:
    handler.read_body()
    spans = handler.fake.spans
    kind = handler.query("kind") or ""
    if spans is None or kind not in ("oversized", "gzip_bomb"):
        handler.send_json(400, {"error": "needs spans_endpoint and a kind"})
        return
    handler.send_json(200, {"status": spans.send_abusive(kind)})


def _fault_kill(handler: _Handler) -> None:
    handler.read_body()
    if not handler.fake.config.allow_kill:
        handler.send_json(403, {"error": "the kill switch is for a subprocess"})
        return
    handler.send_json(200, {"killed": os.getpid()})
    handler.wfile.flush()
    # Like SIGKILL: no flush, no goodbye, no atexit.
    os._exit(137)


def _sampling(body: dict[str, Any], key: str) -> float:
    """A sampling parameter from the body; unset or null is vLLM's 1.0."""
    value = body.get(key)
    return 1.0 if value is None else float(value)


def _health(handler: _Handler) -> None:
    handler.send_bytes(200, b"", "text/plain")


def _models(handler: _Handler) -> None:
    model = handler.fake.config.model
    handler.send_json(
        200,
        {
            "object": "list",
            "data": [{"id": model, "object": "model", "max_model_len": MAX_MODEL_LEN}],
        },
    )


def _version(handler: _Handler) -> None:
    handler.send_json(200, {"version": VLLM_VERSION})


def _server_info(handler: _Handler) -> None:
    info = handler.fake.server_info()
    if handler.query("config_format") != "json":
        info = {**info, "vllm_config": str(info["vllm_config"])}
    handler.send_json(200, info)


def _metrics(handler: _Handler) -> None:
    fake = handler.fake
    controls = fake.controls
    if controls.metrics_mode == "fail":
        handler.send_bytes(500, b"metrics unavailable", "text/plain")
        return
    if controls.metrics_mode == "slow":
        time.sleep(controls.metrics_delay_seconds)
    text = render_metrics(
        fake.engine.metrics_snapshot(), fake.config, fake.engine.start_ns / 1e9
    )
    handler.send_bytes(200, text.encode(), "text/plain; version=0.0.4")


def _start_profile(handler: _Handler) -> None:
    handler.read_body()
    profiler = handler.fake.profiler
    assert profiler is not None
    status = profiler.start()
    if status == 200 and handler.fake.controls.drop_start_response:
        # (a) The profiler is running, but the caller never hears so.
        handler.close_connection = True
        handler.connection.shutdown(socket.SHUT_RDWR)
        return
    handler.send_json(status, {} if status == 200 else {"error": "not configured"})


def _stop_profile(handler: _Handler) -> None:
    handler.read_body()
    profiler = handler.fake.profiler
    assert profiler is not None
    status = profiler.stop()
    handler.send_json(status, {} if status == 200 else {"error": "not configured"})


def _reset_prefix_cache(handler: _Handler) -> None:
    """vLLM 0.30's dev route: 200 with ``success: false`` while blocks are
    held, unless ``reset_running_requests`` preempts every running request."""
    handler.read_body()
    running = (handler.query("reset_running_requests") or "").lower() == "true"
    success = handler.fake.engine.reset_prefix_cache(running)
    handler.send_json(200, {"success": success})


def _chat(handler: _Handler) -> None:
    body = handler.read_json()
    request = handler.fake.new_request(
        body,
        handler.headers.get("X-Request-Id"),
        handler.headers.get("traceparent"),
    )
    stream = bool(body.get("stream"))
    usage = bool((body.get("stream_options") or {}).get("include_usage"))
    try:
        if stream:
            _stream_completion(handler, request, include_usage=usage)
        else:
            _whole_completion(handler, request)
    except (BrokenPipeError, ConnectionResetError):
        handler.fake.engine.abort(request)
        handler.close_connection = True


def _usage(request: FakeRequest) -> dict[str, int]:
    return {
        "prompt_tokens": request.prompt_len,
        "completion_tokens": request.output_tokens,
        "total_tokens": request.prompt_len + request.output_tokens,
    }


def _whole_completion(handler: _Handler, request: FakeRequest) -> None:
    reason = _drain(request)
    handler.fake.wait_frontend()
    handler.send_json(
        200,
        {
            "id": request.external_id,
            "object": "chat.completion",
            "model": handler.fake.config.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": TOKEN_TEXT * request.output_tokens,
                    },
                    "finish_reason": reason,
                }
            ],
            "usage": _usage(request),
        },
    )


def _drain(request: FakeRequest) -> str:
    while True:
        kind, value = request.events.get()
        if kind == "finish":
            return str(value)


def _stream_completion(
    handler: _Handler, request: FakeRequest, *, include_usage: bool
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    model = handler.fake.config.model
    started = False
    while True:
        kind, value = request.events.get()
        handler.fake.wait_frontend()
        if kind == "finish":
            break
        if not started:
            _send_chunk(handler, _delta(request, model, {"role": "assistant"}))
            started = True
        _send_chunk(handler, _delta(request, model, {"content": TOKEN_TEXT}))
    _send_chunk(handler, _delta(request, model, {}, finish_reason=str(value)))
    if include_usage:
        _send_chunk(
            handler,
            {
                "id": request.external_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
                "usage": _usage(request),
            },
        )
    _write_chunk(handler, b"data: [DONE]\n\n")
    _write_chunk(handler, b"")


def _delta(
    request: FakeRequest,
    model: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request.external_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _send_chunk(handler: _Handler, payload: dict[str, Any]) -> None:
    _write_chunk(handler, f"data: {json.dumps(payload)}\n\n".encode())


def _write_chunk(handler: _Handler, data: bytes) -> None:
    handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
    handler.wfile.flush()


GET_ROUTES: dict[str, Route] = {
    "/health": _health,
    "/v1/models": _models,
    "/version": _version,
    "/server_info": _server_info,
    "/metrics": _metrics,
    "/_fault/state": _fault_state,
}
POST_ROUTES: dict[str, Route] = {
    "/v1/chat/completions": _chat,
    "/reset_prefix_cache": _reset_prefix_cache,
    "/start_profile": _start_profile,
    "/stop_profile": _stop_profile,
    "/_fault/pause": _fault_pause,
    "/_fault/resume": _fault_resume,
    "/_fault/controls": _fault_controls,
    "/_fault/foreign_trace": _fault_foreign_trace,
    "/_fault/span_body": _fault_span_body,
    "/_fault/kill": _fault_kill,
}

__all__ = ["GET_ROUTES", "POST_ROUTES", "FakeEngine"]
