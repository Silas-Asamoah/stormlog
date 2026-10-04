"""W3C trace context for the requests Stormlog sends, and sampling by trace ID.

Each request Stormlog sends can carry a ``traceparent`` header, so a server
that traces (vLLM with ``--otlp-traces-endpoint``) records its span as a
child of Stormlog's client span, in the same trace. The header is never
sent unless asked for; matched runs with and without it would otherwise
differ in what the server receives.

The sampled flag decides more than it seems. vLLM's tracer has no sampler
of its own, so the OpenTelemetry SDK's default applies: parent-based, so a
parent marked not sampled stops vLLM recording the request's span at all.
``preserve-engine`` therefore always sends ``01``: vLLM records what it
would have without the header, and Stormlog decides only which of its own
spans to export. ``follow-sampling`` sends Stormlog's head decision, which
saves engine spans and loses them for every request not sampled, including
the failed and slow ones Stormlog keeps after the fact.

Stormlog's own decision is the ProbabilitySampler's predicate on the trace
ID's lowest 56 bits, so it is reproducible from the recorded ID. Stormlog
does not send the ``th`` threshold or the randomness flag, so it claims no
agreement with any particular downstream sampler.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

OFF = "off"
PRESERVE_ENGINE = "preserve-engine"
FOLLOW_SAMPLING = "follow-sampling"
POLICIES = (OFF, PRESERVE_ENGINE, FOLLOW_SAMPLING)
TRACEPARENT = "traceparent"
_RANDOMNESS_BITS = 56
_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})\Z")


@dataclass(frozen=True)
class TraceIds:
    """A span's trace and span IDs, as lowercase hex, and the head decision."""

    trace_id: str
    span_id: str
    sampled: bool = True


def keeps(trace_id: str, ratio: float) -> bool:
    """Whether a trace is kept at ``ratio``: its lowest 56 bits against a threshold.

    A trace kept at one ratio is kept at every higher one.
    """
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    randomness = int(trace_id[-_RANDOMNESS_BITS // 4 :], 16)
    threshold = round((1.0 - ratio) * (1 << _RANDOMNESS_BITS))
    return randomness >= threshold


def new_trace_ids(
    ratio: float = 1.0, *, random_bytes: Callable[[int], bytes] = os.urandom
) -> TraceIds:
    """Random IDs for a request's span, with Stormlog's head decision."""
    trace_id = _nonzero_hex(random_bytes, 16)
    return TraceIds(
        trace_id=trace_id,
        span_id=_nonzero_hex(random_bytes, 8),
        sampled=keeps(trace_id, ratio),
    )


def derived_ids(*parts: str) -> TraceIds:
    """Stable IDs for a span Stormlog builds from a record, such as a phase.

    The same parts give the same IDs, so mapping one artifact twice gives
    the same spans; these IDs are never sent to a server.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    trace = digest[:16] if any(digest[:16]) else b"\x01" * 16
    span = digest[16:24] if any(digest[16:24]) else b"\x01" * 8
    return TraceIds(trace_id=trace.hex(), span_id=span.hex())


def traceparent(ids: TraceIds, policy: str) -> str:
    """The header value for ``policy``; ``preserve-engine`` always says sampled."""
    if policy not in (PRESERVE_ENGINE, FOLLOW_SAMPLING):
        raise ValueError(f"no traceparent is sent under trace-context policy {policy}")
    sampled = policy == PRESERVE_ENGINE or ids.sampled
    return f"00-{ids.trace_id}-{ids.span_id}-{'01' if sampled else '00'}"


def parse_traceparent(value: str) -> TraceIds | None:
    """A version-00 ``traceparent``, or None when it is not one."""
    match = _TRACEPARENT.match(value)
    if match is None:
        return None
    trace_id, span_id, flags = match.groups()
    if set(trace_id) == {"0"} or set(span_id) == {"0"}:
        return None
    return TraceIds(trace_id, span_id, sampled=bool(int(flags, 16) & 1))


def _nonzero_hex(random_bytes: Callable[[int], bytes], size: int) -> str:
    # An all-zero ID is invalid in W3C trace context; draw again.
    while True:
        value = random_bytes(size)
        if any(value):
            return value.hex()
