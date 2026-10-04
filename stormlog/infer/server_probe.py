"""What a vLLM server reports about itself over HTTP, before and after a run.

``basic`` asks ``/version`` and ``/v1/models``. ``auto`` also asks
``/server_info?config_format=json``, vLLM 0.30.0's full configuration and
environment, which vLLM serves only with ``VLLM_SERVER_DEV_MODE=1``. Dev
routes are not covered by vLLM's API key check, so ``auto`` asks it only of
a loopback or private-network host unless ``allow_remote`` says otherwise.

``/server_info`` runs vLLM's ``collect_env`` (``pip``, ``nvidia-smi`` and
more) the first time and caches the answer. So it gets one deadline of 120
seconds and is never retried: a retry would start a second collector, and
a cached answer would say nothing about the first. A request that runs out
of time leaves the probe ``incomplete``, and a run must not measure next to
a collector that may still be running.

Every answer is capped at 4 MiB, redirects are never followed, and the API
key goes only to the endpoint's own origin. ``/server_info`` is kept
redacted (see ``server_privacy``).
"""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .openai_client import ConnectError, inference_opener
from .server_privacy import redact_vllm_config, redact_vllm_env, system_env_summary

AUTO = "auto"
BASIC = "basic"
NONE = "none"
PROBE_MODES = (AUTO, BASIC, NONE)
BEFORE = "before"
AFTER = "after"

VERSION = "/version"
MODELS = "/v1/models"
SERVER_INFO = "/server_info?config_format=json"
BASIC_DEADLINE_SECONDS = 60.0
SERVER_INFO_DEADLINE_SECONDS = 120.0
RESPONSE_CAP_BYTES = 4 * 1024 * 1024
_CHUNK = 64 * 1024
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)

OK = "ok"
HTTP_ERROR = "http_error"
UNREACHABLE = "unreachable"
FAILED = "failed"
TIMEOUT = "timeout"
TOO_LARGE = "too_large"
INVALID_JSON = "invalid_json"
SKIPPED = "skipped"

Opener = Callable[[urllib.request.Request, float], Any]


@dataclass(frozen=True)
class ProbeAnswer:
    """One route's answer, or why there is none."""

    route: str
    status: str
    http_status: int | None = None
    elapsed_ms: float | None = None
    size_bytes: int | None = None
    body: Any = None
    detail: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "http_status": self.http_status,
            "elapsed_ms": self.elapsed_ms,
            "size_bytes": self.size_bytes,
            "body": self.body,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ServerProbe:
    """Every route a probe asked, at one point of the run."""

    phase: str
    mode: str
    origin: str
    started_at_ns: int
    answers: Mapping[str, ProbeAnswer] = field(default_factory=dict)

    @property
    def incomplete(self) -> bool:
        """Whether ``/server_info`` ran out of time, so its collector may run on."""
        answer = self.answers.get(SERVER_INFO)
        return answer is not None and answer.status == TIMEOUT

    def to_record(self, *, session_id: str) -> dict[str, Any]:
        return {
            "event_type": "infer.server_probe",
            "session_id": session_id,
            "phase": self.phase,
            "mode": self.mode,
            "origin": self.origin,
            "started_at_ns": self.started_at_ns,
            "incomplete": self.incomplete,
            "answers": {
                route: answer.to_record() for route, answer in self.answers.items()
            },
        }


def probe_server(
    endpoint: str,
    *,
    mode: str = AUTO,
    phase: str = BEFORE,
    api_key: str | None = None,
    allow_remote: bool = False,
    opener: Opener | None = None,
) -> ServerProbe:
    """Ask the server about itself; never raises for what the server does."""
    if mode not in PROBE_MODES:
        raise ValueError(f"server probe mode must be one of {PROBE_MODES}")
    origin = endpoint_origin(endpoint)
    probe = ServerProbe(phase, mode, origin, time.time_ns())
    if mode == NONE:
        return probe
    send = opener or _default_opener()
    routes = [VERSION, MODELS] + ([SERVER_INFO] if mode == AUTO else [])
    answers: dict[str, ProbeAnswer] = {}
    down = False
    for route in routes:
        if down:
            # No connection: every other route would wait out its own deadline.
            answers[route] = ProbeAnswer(route, SKIPPED, detail="unreachable")
            continue
        if route == SERVER_INFO:
            answers[route] = _server_info(send, origin, api_key, allow_remote, phase)
        else:
            answer = _ask(send, origin, route, api_key, BASIC_DEADLINE_SECONDS)
            # A model's root can be a URL with credentials in it.
            answers[route] = replace(answer, body=_stripped(answer.body))
        down = answers[route].status == UNREACHABLE
    return ServerProbe(phase, mode, origin, probe.started_at_ns, answers)


def endpoint_origin(endpoint: str) -> str:
    """``scheme://host:port`` of the endpoint; every route hangs off it."""
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"endpoint {endpoint!r} is not an http(s) URL")
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port is not None else ""
    return f"{parts.scheme}://{host}{port}"


def is_private_host(host: str) -> bool:
    """Loopback, RFC 1918, link-local or unique-local; ``localhost`` too.

    Python's ``is_private`` also counts documentation and other reserved
    ranges, which are not a private network, so the ranges are listed.
    """
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return any(address in network for network in _PRIVATE_NETWORKS)


def _server_info(
    send: Opener, origin: str, api_key: str | None, allow_remote: bool, phase: str
) -> ProbeAnswer:
    host = urllib.parse.urlsplit(origin).hostname or ""
    if not allow_remote and not is_private_host(host):
        return ProbeAnswer(SERVER_INFO, SKIPPED, detail="non_private_host")
    answer = _ask(send, origin, SERVER_INFO, api_key, SERVER_INFO_DEADLINE_SECONDS)
    if answer.status != OK:
        if answer.http_status == 404:
            detail = "vLLM serves /server_info only with VLLM_SERVER_DEV_MODE=1"
            return replace(answer, detail=detail)
        return answer
    return replace(answer, body=_redacted_server_info(answer.body, phase))


def _stripped(body: Any) -> Any:
    """An answer whose URLs, wherever they sit, lose credentials and query."""
    return redact_vllm_config(body) if isinstance(body, dict) else body


def _redacted_server_info(body: Any, phase: str) -> Any:
    """The configuration and environment, redacted; collect_env summarized."""
    if not isinstance(body, dict):
        return body
    config = body.get("vllm_config")
    environment = body.get("vllm_env")
    system = body.get("system_env")
    return {
        "vllm_config": redact_vllm_config(config) if isinstance(config, dict) else None,
        "vllm_env": (
            redact_vllm_env(environment) if isinstance(environment, dict) else None
        ),
        "system_env": system_env_summary(system) if isinstance(system, dict) else None,
        # vLLM caches collect_env, so after the run it says nothing new.
        "system_env_freshness": "cached" if phase == AFTER else "first_or_cached",
    }


def _ask(
    send: Opener, origin: str, route: str, api_key: str | None, deadline: float
) -> ProbeAnswer:
    url = origin + route
    headers = {"Accept": "application/json"}
    if api_key and _same_origin(url, origin):
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    started = time.monotonic()
    try:
        response = send(request, deadline)
        with response:
            raw = _read_capped(response, started + deadline)
            status = int(getattr(response, "status", 200))
    except (OSError, _TooLarge) as exc:
        return _failure(route, exc, started)
    try:
        body = json.loads(raw)
    except ValueError:
        return _answer(route, INVALID_JSON, started, http_status=status, size=len(raw))
    return _answer(route, OK, started, http_status=status, size=len(raw), body=body)


def _failure(route: str, exc: Exception, started: float) -> ProbeAnswer:
    """Where a request failed: HTTP status, connection, deadline or size."""
    if isinstance(exc, urllib.error.HTTPError):
        return _answer(route, HTTP_ERROR, started, http_status=exc.code)
    if isinstance(exc, _TooLarge):
        return _answer(route, TOO_LARGE, started, detail=f"over {RESPONSE_CAP_BYTES}")
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            return _answer(route, TIMEOUT, started, detail=str(reason))
        kind = UNREACHABLE if isinstance(reason, ConnectError) else FAILED
        return _answer(route, kind, started, detail=str(reason))
    if isinstance(exc, TimeoutError):
        return _answer(route, TIMEOUT, started, detail=str(exc) or "deadline")
    return _answer(route, FAILED, started, detail=f"{type(exc).__name__}: {exc}")


class _TooLarge(Exception):
    pass


def _read_capped(response: Any, deadline: float) -> bytes:
    """The body, up to the cap and the deadline, whichever comes first."""
    chunks: list[bytes] = []
    size = 0
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("deadline reached while reading the answer")
        chunk = response.read(_CHUNK)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > RESPONSE_CAP_BYTES:
            raise _TooLarge()
        chunks.append(chunk)


def _answer(
    route: str,
    status: str,
    started: float,
    *,
    http_status: int | None = None,
    size: int | None = None,
    body: Any = None,
    detail: str | None = None,
) -> ProbeAnswer:
    return ProbeAnswer(
        route=route,
        status=status,
        http_status=http_status,
        elapsed_ms=(time.monotonic() - started) * 1000.0,
        size_bytes=size,
        body=body,
        detail=detail,
    )


def _same_origin(url: str, origin: str) -> bool:
    return endpoint_origin(url) == origin


def _default_opener() -> Opener:
    opener = inference_opener()

    def send(request: urllib.request.Request, timeout: float) -> Any:
        return opener.open(request, timeout=timeout)

    return send


__all__ = [
    "AFTER",
    "AUTO",
    "BASIC",
    "BEFORE",
    "MODELS",
    "NONE",
    "PROBE_MODES",
    "RESPONSE_CAP_BYTES",
    "SERVER_INFO",
    "SERVER_INFO_DEADLINE_SECONDS",
    "VERSION",
    "ProbeAnswer",
    "ServerProbe",
    "endpoint_origin",
    "is_private_host",
    "probe_server",
]
