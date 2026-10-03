"""Per-case latency quantiles with sufficiency, and chunk-level streaming.

Each latency metric is a criterion key of ``stormlog.infer.slo``, so the
report uses the same boundaries as SLO policies: ``client.*`` values come
from Stormlog's client, ``server.*`` values only from a request's own joined
vLLM span. Every quantile is given twice, over the successful requests and
with every request that did not succeed ranked worst, and each says whether
the case has enough requests to trust it. What the requests that did not
succeed were observed to take is kept apart, by status.

Streamed chunks are reported as chunks. A chunk can carry several tokens,
so chunk gaps are never called inter-token latency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .quantiles import (
    OrderStatisticInterval,
    QuantileEstimate,
    SufficiencyRule,
    penalized_quantile,
    quantile,
    successful_quantile,
)
from .report_stats import is_number
from .slo import CriterionValue, client_values, server_values

if TYPE_CHECKING:
    from .vllm_analysis import JoinedSpans

ValuesOf = Callable[[Mapping[str, Any]], Mapping[str, CriterionValue]]

LEVELS = (0.5, 0.9, 0.95, 0.99)
CLIENT_METRICS = (
    "client.ttft",
    "client.ttft_from_intended",
    "client.e2e",
    "client.e2e_from_intended",
    "client.tpot",
)
SERVER_METRICS = ("server.ttft", "server.e2e", "server.queue")
CHUNK_NOTE = (
    "chunk-level timing: a streamed chunk can carry several tokens, so these "
    "are not inter-token latencies"
)


def latency_summary(
    requests: Sequence[Mapping[str, Any]],
    *,
    spans: JoinedSpans | None = None,
    rule: SufficiencyRule = SufficiencyRule(),
    levels: Sequence[float] = LEVELS,
) -> dict[str, Any]:
    """Quantiles of each latency metric over a case's measured requests."""
    metrics = {
        key: _metric(requests, key, client_values, rule, levels)
        for key in CLIENT_METRICS
    }
    if spans is not None and spans.by_request:
        joined = spans

        def from_span(record: Mapping[str, Any]) -> Mapping[str, CriterionValue]:
            return _server_values(record, joined)

        for key in SERVER_METRICS:
            metrics[key] = _metric(requests, key, from_span, rule, levels)
    return {
        "rule": rule.to_record(),
        "metrics": metrics,
        "unsuccessful": unsuccessful_summary(requests),
    }


def unsuccessful_summary(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """How long each request that did not succeed ran, by status.

    This is the time until Stormlog saw the request end, not a latency the
    request would have had: only for a timeout is it a lower bound on one.
    A dropped request was never sent and has no time.
    """
    by_status: dict[str, list[Mapping[str, Any]]] = {}
    for record in requests:
        if record.get("status") != "ok":
            by_status.setdefault(str(record.get("status")), []).append(record)
    return {
        status: _elapsed_record(records)
        for status, records in sorted(by_status.items())
    }


def streaming_summary(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Chunk counts and gaps of the successful streamed responses.

    Tokens per chunk is a mean over the responses whose output token count
    the server reported; chunk sizes themselves are not recorded.
    """
    streamed = [r for r in requests if _streamed(r)]
    gaps = [gap for r in streamed for gap in _gaps(r)]
    chunks = [float(len(_gaps(r)) + 1) for r in streamed]
    return {
        "granularity": "chunk",
        "responses": len(streamed),
        "content_chunks_per_response": _spread(chunks),
        "chunk_interarrival_ms": {**_spread(gaps), "n": len(gaps)},
        "mean_tokens_per_chunk": _tokens_per_chunk(streamed),
        "note": CHUNK_NOTE,
    }


def _streamed(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("status") == "ok"
        and record.get("stream")
        and is_number(record.get("ttft_ms"))
    )


def _gaps(record: Mapping[str, Any]) -> list[float]:
    return [
        float(gap)
        for gap in record.get("chunk_interarrival_ms") or []
        if is_number(gap)
    ]


def _metric(
    requests: Sequence[Mapping[str, Any]],
    key: str,
    values_of: ValuesOf,
    rule: SufficiencyRule,
    levels: Sequence[float],
) -> dict[str, Any]:
    successful: list[float] = []
    missing = 0
    for record in requests:
        if record.get("status") != "ok":
            continue
        value = values_of(record)[key].value_ms
        if value is None:
            missing += 1
        else:
            successful.append(value)
    failures = [r for r in requests if r.get("status") != "ok"]
    timeouts = _timeout_elapsed(failures) if key == "client.e2e" else None
    return {
        "boundary": key.split(".", 1)[0],
        "successful_missing": missing,
        "successful": {
            _level_key(p): _estimate_record(successful_quantile(successful, p, rule))
            for p in levels
        },
        "failure_penalized": {
            _level_key(p): _estimate_record(
                penalized_quantile(
                    successful, len(failures), p, rule, timeout_elapsed_ms=timeouts
                )
            )
            for p in levels
        },
    }


def _server_values(
    record: Mapping[str, Any], spans: JoinedSpans
) -> Mapping[str, CriterionValue]:
    request_id = str(record.get("x_request_id"))
    return server_values(
        spans.by_request.get(request_id),
        missing_span_reason=spans.quarantined.get(request_id, "no_joined_span"),
    )


def _timeout_elapsed(failures: Sequence[Mapping[str, Any]]) -> list[float] | None:
    """Each failure's elapsed time, when every failure was a real timeout."""
    if not failures or any(r.get("status") != "timeout" for r in failures):
        return None
    elapsed = [r.get("e2e_latency_ms") for r in failures]
    if not all(is_number(value) for value in elapsed):
        return None
    return [float(value) for value in elapsed if is_number(value)]


def _elapsed_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed = [
        float(value)
        for value in (r.get("e2e_latency_ms") for r in records)
        if is_number(value)
    ]
    return {
        "count": len(records),
        "elapsed_ms": {
            "min": min(elapsed, default=None),
            "p50": quantile(elapsed, 0.5),
            "max": max(elapsed, default=None),
        },
        "elapsed_missing": len(records) - len(elapsed),
    }


def _estimate_record(estimate: QuantileEstimate) -> dict[str, Any]:
    return {
        "value_ms": estimate.value_ms,
        "n": estimate.n,
        "sufficient": estimate.sufficient,
        "n_min": estimate.n_min,
        "n_min_exists": estimate.n_min_exists,
        "penalized": estimate.penalized,
        "observed_lower_bound_ms": estimate.observed_lower_bound_ms,
        "interval": _interval_record(estimate.interval),
    }


def _interval_record(interval: OrderStatisticInterval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {
        "lower_rank": interval.lower_rank,
        "upper_rank": interval.upper_rank,
        "lower_ms": interval.lower_ms,
        "upper_ms": interval.upper_ms,
        "width_ms": interval.width_ms,
        "coverage": interval.coverage,
    }


def _tokens_per_chunk(streamed: Sequence[Mapping[str, Any]]) -> float | None:
    counted = [r for r in streamed if r.get("output_token_source") == "server_usage"]
    tokens = sum(int(r.get("output_tokens") or 0) for r in counted)
    chunks = sum(len(r.get("chunk_interarrival_ms") or []) + 1 for r in counted)
    return tokens / chunks if chunks else None


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "max": max(values, default=None),
    }


def _level_key(p: float) -> str:
    return f"p{p * 100:g}".replace(".", "_")


__all__ = [
    "CHUNK_NOTE",
    "CLIENT_METRICS",
    "LEVELS",
    "SERVER_METRICS",
    "latency_summary",
    "streaming_summary",
    "unsuccessful_summary",
]
