"""Requested and verified prefix-cache state for each workload case.

A benchmark can ask for a cold cache and can call a reset endpoint before
each case, such as vLLM's ``/reset_prefix_cache`` (available when the server
runs in development mode) or SGLang's ``/flush_cache``. Asking is not
proof: until an engine adapter can read the cache's contents, every case
records its cache state as unverified and says why.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

UNSPECIFIED = "unspecified"
COLD = "cold"
CACHE_STATES = (UNSPECIFIED, COLD)
UNVERIFIED = "unverified"


@dataclass(frozen=True)
class CacheReset:
    """The outcome of one call to a cache reset endpoint."""

    url: str
    at_ns: int
    status: int | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    def to_record(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "at_ns": self.at_ns,
            "status": self.status,
            "error": self.error,
        }


def reset_cache(
    url: str, *, timeout_seconds: float, api_key: str | None = None
) -> CacheReset:
    """POST to a reset endpoint and record what happened; never raises.

    The API key, when there is one, goes along as it does with every
    request, since the reset route usually sits behind the same server.
    """
    at_ns = time.time_ns()
    recorded = _redact(url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, data=b"", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return CacheReset(recorded, at_ns, status=int(response.status))
    except urllib.error.HTTPError as exc:
        return CacheReset(recorded, at_ns, status=exc.code, error=f"HTTP {exc.code}")
    except OSError as exc:
        return CacheReset(recorded, at_ns, error=f"{type(exc).__name__}: {exc}")


def redact_url(url: str | None) -> str | None:
    """A URL as it may be recorded: no credentials and no query string.

    Either can carry a token, so an artifact keeps only the scheme, host,
    port and path, and marks a removed query.
    """
    return None if url is None else _redact(url)


def _redact(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    query = "?<redacted>" if parts.query else ""
    return f"{parts.scheme}://{host}{parts.path}{query}"


def run_kind(
    requested: str, warmup_requests: int, reset: CacheReset | None = None
) -> str:
    """Whether a case was designed as a cold start or a warmed-up steady state.

    This names the run's design, not evidence about the cache. A cold start
    whose reset failed is unspecified: the failure shows the cache was not
    cleared.
    """
    if warmup_requests > 0:
        return "steady_state"
    if requested == COLD and (reset is None or reset.succeeded):
        return "cold_start"
    return UNSPECIFIED


def cache_state_record(
    *,
    session_id: str,
    case_id: str,
    requested: str,
    reset: CacheReset | None,
    warmup_requests: int,
) -> dict[str, Any]:
    """The ``infer.cache_state`` record written before a case runs."""
    return {
        "schema_version": 1,
        "event_type": "infer.cache_state",
        "session_id": session_id,
        "case_id": case_id,
        "requested": requested,
        "reset": reset.to_record() if reset is not None else None,
        "verified": UNVERIFIED,
        "reason": _reason(requested, reset),
        "run_kind": run_kind(requested, warmup_requests, reset),
    }


def _reason(requested: str, reset: CacheReset | None) -> str:
    if reset is not None and not reset.succeeded:
        return f"the cache reset failed ({reset.error or reset.status})"
    if reset is not None:
        return "the cache reset succeeded, but no engine adapter can confirm it"
    if requested == COLD:
        return "nothing reset the cache, and no engine adapter can read it"
    return (
        "no cache state was requested; earlier traffic, including an earlier "
        "run with the same seed, decides what the cache holds"
    )


def cache_summary(record: dict[str, Any] | None) -> dict[str, Any]:
    """The cache block of a case report; older artifacts did not record one."""
    if record is None:
        return {
            "requested": UNSPECIFIED,
            "reset": None,
            "verified": UNVERIFIED,
            "reason": "the artifact does not record a cache state",
            "run_kind": None,
        }
    return {
        "requested": record.get("requested"),
        "reset": record.get("reset"),
        "verified": record.get("verified"),
        "reason": record.get("reason"),
        "run_kind": record.get("run_kind"),
    }


def cache_lines(cache: Any) -> list[str]:
    """A text-report line when a cache state was requested or reset."""
    if not isinstance(cache, dict):
        return []
    if cache.get("requested") != COLD and cache.get("reset") is None:
        return []
    return [
        f"  cache: {cache.get('requested')} requested, {cache.get('verified')} "
        f"({cache.get('reason')}); run kind {cache.get('run_kind')}"
    ]
