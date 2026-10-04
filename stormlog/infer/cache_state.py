"""Requested and verified prefix-cache state for each workload case.

A benchmark can ask for a cold cache and can call a reset endpoint before
each case, such as vLLM's ``/reset_prefix_cache`` (available when the server
runs in development mode) or SGLang's ``/flush_cache``. Asking is not
proof: until an engine adapter can read the cache's contents, every case
records its cache state as unverified and says why.

A 2xx answer is not proof of a reset either. vLLM answers HTTP 200 with
``{"success": false}`` while blocks are still held, so the body is read: a
reset is acknowledged only when the server says ``success: true``, refused
when it keeps saying anything else, and accepted but unconfirmed when a 2xx
body has no ``success`` field.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

# Re-exported: callers imported redact_url from here before it moved.
from ..scrub import redact_url

UNSPECIFIED = "unspecified"
COLD = "cold"
CACHE_STATES = (UNSPECIFIED, COLD)
UNVERIFIED = "unverified"

ACKNOWLEDGED = "acknowledged"
REFUSED = "refused"
ACCEPTED_UNVERIFIED = "accepted_unverified"
# vLLM refuses a reset while blocks are held and says callers may retry.
RESET_RETRY_SECONDS = 10.0
RESET_RETRY_INTERVAL_SECONDS = 0.5
_RESET_BODY_LIMIT = 64 * 1024


@dataclass(frozen=True)
class CacheReset:
    """The outcome of a cache reset, over every attempt it took.

    ``success`` is True when the body said ``"success": true``, False when it
    had a ``success`` field with any other value, and None without one.
    ``at_ns`` is when the first attempt was sent; ``answered_at_ns`` is when
    the answer recorded here, the last attempt's, came back.
    """

    url: str
    at_ns: int
    status: int | None = None
    error: str | None = None
    success: bool | None = None
    attempts: int = 1
    answered_at_ns: int | None = None

    @property
    def answer(self) -> str | None:
        """``acknowledged``, ``refused`` or ``accepted_unverified``; None if no 2xx."""
        if self.status is None or not 200 <= self.status < 300:
            return None
        if self.success is None:
            return ACCEPTED_UNVERIFIED
        return ACKNOWLEDGED if self.success else REFUSED

    @property
    def succeeded(self) -> bool:
        """The server accepted the reset, whether or not it confirmed it."""
        return self.answer in (ACKNOWLEDGED, ACCEPTED_UNVERIFIED)

    @property
    def acknowledged(self) -> bool:
        """The server said the reset happened."""
        return self.answer == ACKNOWLEDGED

    def to_record(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "at_ns": self.at_ns,
            "status": self.status,
            "error": self.error,
            "success": self.success,
            "answer": self.answer,
            "attempts": self.attempts,
            "answered_at_ns": self.answered_at_ns,
        }


def reset_cache(
    url: str,
    *,
    timeout_seconds: float,
    api_key: str | None = None,
    retry_seconds: float = RESET_RETRY_SECONDS,
) -> CacheReset:
    """POST to a reset endpoint and record what happened; never raises.

    A reset the server refuses (``success: false``) is tried again every
    half second for up to ``retry_seconds``. No attempt starts after that,
    though the last one may take up to ``timeout_seconds`` to answer. The API
    key, when there is one, goes along as it does with every request, since
    the reset route usually sits behind the same server.
    """
    at_ns = time.time_ns()
    deadline = time.monotonic() + retry_seconds
    attempts = 1
    reset = _post_reset(url, at_ns, timeout_seconds=timeout_seconds, api_key=api_key)
    while reset.answer == REFUSED and (
        deadline - time.monotonic() >= RESET_RETRY_INTERVAL_SECONDS
    ):
        time.sleep(RESET_RETRY_INTERVAL_SECONDS)
        attempts += 1
        reset = _post_reset(
            url, at_ns, timeout_seconds=timeout_seconds, api_key=api_key
        )
    if reset.answer == REFUSED:
        error = f"refused: success false on {attempts} attempts"
        return _with(reset, error=error, attempts=attempts)
    return _with(reset, error=reset.error, attempts=attempts)


def _post_reset(
    url: str, at_ns: int, *, timeout_seconds: float, api_key: str | None
) -> CacheReset:
    recorded = redact_url(url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, data=b"", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            success = _success_field(response.read(_RESET_BODY_LIMIT))
            return CacheReset(
                recorded,
                at_ns,
                status=int(response.status),
                success=success,
                answered_at_ns=time.time_ns(),
            )
    except urllib.error.HTTPError as exc:
        return CacheReset(
            recorded,
            at_ns,
            status=exc.code,
            error=f"HTTP {exc.code}",
            answered_at_ns=time.time_ns(),
        )
    except OSError as exc:
        return CacheReset(
            recorded,
            at_ns,
            error=f"{type(exc).__name__}: {exc}",
            answered_at_ns=time.time_ns(),
        )


def _success_field(body: bytes) -> bool | None:
    """vLLM's ``{"success": bool}``: only ``true`` is a yes; None without one."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or "success" not in payload:
        return None
    return payload["success"] is True


def _with(reset: CacheReset, *, error: str | None, attempts: int) -> CacheReset:
    return CacheReset(
        reset.url,
        reset.at_ns,
        status=reset.status,
        error=error,
        success=reset.success,
        attempts=attempts,
        answered_at_ns=reset.answered_at_ns,
    )


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
        "attempted": reset is not None,
        "acknowledged": reset is not None and reset.acknowledged,
        "verified": UNVERIFIED,
        "reason": _reason(requested, reset),
        "run_kind": run_kind(requested, warmup_requests, reset),
    }


def _reason(requested: str, reset: CacheReset | None) -> str:
    if reset is not None and not reset.succeeded:
        return f"the cache reset failed ({reset.error or reset.status})"
    if reset is not None and reset.acknowledged:
        return (
            "the server acknowledged the cache reset, but no engine adapter "
            "can confirm it"
        )
    if reset is not None:
        return (
            f"the cache reset returned HTTP {reset.status} without saying "
            "whether it succeeded, and no engine adapter can confirm it"
        )
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
            "attempted": None,
            "acknowledged": None,
            "verified": UNVERIFIED,
            "reason": "the artifact does not record a cache state",
            "run_kind": None,
        }
    return {
        "requested": record.get("requested"),
        "reset": record.get("reset"),
        # Artifacts written before resets were parsed do not say.
        "attempted": record.get("attempted"),
        "acknowledged": record.get("acknowledged"),
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
