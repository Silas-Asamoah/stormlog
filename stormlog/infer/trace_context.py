"""W3C trace context for the requests Stormlog sends, and sampling by trace ID.

Each request Stormlog sends can carry a ``traceparent`` header, so a server
that traces (vLLM with ``--otlp-traces-endpoint``) records its span as a
child of Stormlog's client span, in the same trace. The header is never
sent unless asked for; matched runs with and without it would otherwise
differ in what the server receives.

The sampled flag decides more than it seems. vLLM's tracer takes its
sampler from ``OTEL_TRACES_SAMPLER``, by default ``parentbased_always_on``:
under a parent-based sampler a remote parent's flag decides, so a parent
marked not sampled stops vLLM recording the request's span at all, and one
marked sampled makes it record the span whatever its own ratio says.
``preserve-engine`` always sends ``01``: under the default sampler vLLM
records what it would have without the header, but under a parent-based
ratio sampler it records every Stormlog request, more than its ratio; the
profiler warns when the declared server sampler says so.
``follow-sampling`` sends Stormlog's head decision at the server's declared
ratio, which keeps the engine's volume, and loses the engine spans of every
request not sampled, including the failed and slow ones Stormlog keeps.

Every request span that carried a ``traceparent`` is exported, whatever
Stormlog's own ratio: the server may have recorded a child of it, which
would otherwise point at a parent no backend ever receives.

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


# The OpenTelemetry SDK's OTEL_TRACES_SAMPLER names. The ratio samplers
# take a ratio from 0 to 1 (1 when none is given); always_on and always_off,
# parent-based or not, take no argument; the remote samplers take their own.
RATIO_SAMPLERS = ("traceidratio", "parentbased_traceidratio")
FIXED_SAMPLERS = {
    "always_on": 1.0,
    "always_off": 0.0,
    "parentbased_always_on": 1.0,
    "parentbased_always_off": 0.0,
}
REMOTE_SAMPLERS = ("jaeger_remote", "parentbased_jaeger_remote", "xray")


@dataclass(frozen=True)
class ServerSampler:
    """A server's declared sampler: its name, and the share of new traces it keeps."""

    name: str
    # None when the sampler decides remotely, so the share is unknown.
    ratio: float | None

    @property
    def parent_based(self) -> bool:
        return self.name.startswith("parentbased_")


def parse_server_sampler(value: str) -> ServerSampler:
    """``NAME[:ARG]`` as ``OTEL_TRACES_SAMPLER`` and its argument; ValueError if unknown."""
    name, separator, argument = value.partition(":")
    if name in RATIO_SAMPLERS:
        return ServerSampler(name, _ratio(value, argument) if separator else 1.0)
    if name in FIXED_SAMPLERS and not separator:
        return ServerSampler(name, FIXED_SAMPLERS[name])
    if name in REMOTE_SAMPLERS:
        return ServerSampler(name, None)
    known = ", ".join(RATIO_SAMPLERS + tuple(FIXED_SAMPLERS) + REMOTE_SAMPLERS)
    raise ValueError(
        f"--server-trace-sampler {value!r} is not NAME[:ARG] with NAME one of "
        f"{known}; only the ratio samplers take a ratio"
    )


def _ratio(value: str, argument: str) -> float:
    try:
        ratio = float(argument)
    except ValueError:
        ratio = float("nan")
    if not 0.0 <= ratio <= 1.0:  # nan fails too
        raise ValueError(f"--server-trace-sampler {value!r} needs a ratio from 0 to 1")
    return ratio


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
