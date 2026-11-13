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

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .quantiles import (
    OrderStatisticInterval,
    QuantileEstimate,
    SufficiencyRule,
    penalized_quantile,
    quantile,
    quantile_sorted,
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
    client_values_of: ValuesOf = client_values,
) -> dict[str, Any]:
    """Quantiles of each latency metric over a case's measured requests.

    ``client_values_of`` gives a record's client values. It defaults to
    ``slo.client_values``; a caller that stored them per row, as Studio's
    index does, passes a lookup instead. It is called once per successful
    request, for all the client metrics together.
    """
    failures = [r for r in requests if r.get("status") != "ok"]
    metrics = _metrics(
        requests, failures, CLIENT_METRICS, client_values_of, rule, levels
    )
    if spans is not None and spans.by_request:
        joined = spans

        def from_span(record: Mapping[str, Any]) -> Mapping[str, CriterionValue]:
            return _server_values(record, joined)

        metrics.update(
            _metrics(requests, failures, SERVER_METRICS, from_span, rule, levels)
        )
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
    A cancelled request's is its send to its end; a dropped request was
    never sent and has no time.
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
    chunks: list[float] = []
    gaps, largest = _pooled_gaps(streamed, chunks)
    return {
        "granularity": "chunk",
        "responses": len(streamed),
        "content_chunks_per_response": spread(chunks),
        "chunk_interarrival_ms": {**_sorted_spread(gaps, largest), "n": len(gaps)},
        "mean_tokens_per_chunk": _tokens_per_chunk(streamed),
        "note": CHUNK_NOTE,
    }


def _streamed(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("status") == "ok"
        and record.get("stream")
        and is_number(record.get("ttft_ms"))
    )


def _pooled_gaps(
    streamed: Sequence[Mapping[str, Any]], chunks: list[float]
) -> tuple[npt.NDArray[np.float64], float | None]:
    """Every response's gaps in one sorted float64 array, and the largest gap.

    Appends each response's chunk count to ``chunks``. The array costs 8
    bytes a gap where a pooled list costs a Python float each. The sort is
    stable, as ``sorted`` is, and the largest gap is ``max``'s first
    maximum, so equal values of either sign give the same floats as before.
    """
    largest: list[float] = []

    def each() -> Iterator[float]:
        for record in streamed:
            gaps = _gaps(record)
            chunks.append(float(len(gaps) + 1))
            if gaps:
                largest.append(max(gaps))
            yield from gaps

    pooled = np.fromiter(each(), dtype=np.float64)
    pooled.sort(kind="stable")
    return pooled, max(largest, default=None)


def _gaps(record: Mapping[str, Any]) -> list[float]:
    return [
        float(gap)
        for gap in record.get("chunk_interarrival_ms") or []
        if is_number(gap)
    ]


def _metrics(
    requests: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    values_of: ValuesOf,
    rule: SufficiencyRule,
    levels: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Each metric in ``keys``, reading each successful record's values once."""
    values: dict[str, list[float]] = {key: [] for key in keys}
    missing = dict.fromkeys(keys, 0)
    for record in requests:
        if record.get("status") != "ok":
            continue
        criteria = values_of(record)
        for key in keys:
            value = criteria[key].value_ms
            if value is None:
                missing[key] += 1
            else:
                values[key].append(value)
    return {
        key: _metric(key, values[key], missing[key], failures, rule, levels)
        for key in keys
    }


def _metric(
    key: str,
    values: list[float],
    missing: int,
    failures: Sequence[Mapping[str, Any]],
    rule: SufficiencyRule,
    levels: Sequence[float],
) -> dict[str, Any]:
    # Sorted once here, so the sorts inside quantiles each find one run and
    # take a linear pass; only _observed_bound's, with timeouts appended,
    # sorts more.
    successful = sorted(values)
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
                    successful,
                    len(failures),
                    p,
                    rule,
                    timeout_elapsed_ms=timeouts,
                    successful_missing=missing,
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
    elapsed = [value for value in map(elapsed_ms, records) if value is not None]
    return {
        "count": len(records),
        "elapsed_ms": {
            "min": min(elapsed, default=None),
            "p50": quantile(elapsed, 0.5),
            "max": max(elapsed, default=None),
        },
        "elapsed_missing": len(records) - len(elapsed),
    }


def elapsed_ms(record: Mapping[str, Any]) -> float | None:
    """The recorded e2e time, or, for a request sent but abandoned, send to end.

    A request cancelled at the drain deadline has no e2e time of its own,
    but its send and its end are recorded. A dropped one was never sent.
    """
    value = record.get("e2e_latency_ms")
    if is_number(value):
        return float(value)
    began, ended = record.get("started_at_ns"), record.get("ended_at_ns")
    if record.get("status") == "dropped" or not is_number(began):
        return None
    if not is_number(ended) or ended < began:
        return None
    return (ended - began) / 1_000_000.0


def _estimate_record(estimate: QuantileEstimate) -> dict[str, Any]:
    return {
        "value_ms": estimate.value_ms,
        "n": estimate.n,
        "sufficient": estimate.sufficient,
        "n_min": estimate.n_min,
        "n_min_exists": estimate.n_min_exists,
        "penalized": estimate.penalized,
        "observed_lower_bound_ms": estimate.observed_lower_bound_ms,
        "reason": estimate.reason,
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


def spread(values: Sequence[float]) -> dict[str, float | None]:
    """The median, 95th percentile and maximum of ``values``; None when empty.

    The report's shape for a pool of chunk gaps or chunk counts, and Studio's
    for one request's gaps.
    """
    return _sorted_spread(sorted(values), max(values, default=None))


def _sorted_spread(
    ordered: Sequence[float] | npt.NDArray[np.float64], largest: float | None
) -> dict[str, float | None]:
    return {
        "p50": quantile_sorted(ordered, 0.5),
        "p95": quantile_sorted(ordered, 0.95),
        "max": largest,
    }


def _level_key(p: float) -> str:
    return f"p{p * 100:g}".replace(".", "_")


__all__ = [
    "CHUNK_NOTE",
    "CLIENT_METRICS",
    "LEVELS",
    "SERVER_METRICS",
    "elapsed_ms",
    "latency_summary",
    "spread",
    "streaming_summary",
    "unsuccessful_summary",
]
