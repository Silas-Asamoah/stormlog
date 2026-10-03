"""Cheap signals that decide whether a window of scrapes looks like an incident.

``evaluate_signal`` is a pure function over one window of ``/metrics`` scrapes
the caller chose. It returns the kind's value, whether the window had enough
data to decide, and the verdict against the shared threshold table. It never
diagnoses: a value over its threshold says a mechanism is *suspected* in an
engine's aggregate metrics, which cover every client's traffic. Confirming
it, and saying whose requests it hurt, is the diagnoser's job.

Only kinds that ``/metrics`` alone can decide are evaluated; every other kind
answers ``requires_hook``, ``requires_trace`` or ``requires_client``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from . import diagnosis_vocabulary as kinds
from .diagnosis_thresholds import (
    DEFAULT_THRESHOLDS,
    KV_PREEMPTIONS,
    PREFIX_HIT_RATIO_DROP,
    QUEUE_MEDIAN_WAITING,
    THRESHOLDS_VERSION,
    resolve_threshold,
)
from .scrape_window import (
    REASON_NO_OBSERVATIONS,
    WindowCheck,
    check_window,
    counter_window,
    gauge_median,
    gauge_window,
    histogram_quantile_bounds,
)
from .vllm_telemetry import VllmScrapeRecord

WAITING = "vllm:num_requests_waiting"
WAITING_BY_REASON = "vllm:num_requests_waiting_by_reason"
QUEUE_TIME = "vllm:request_queue_time_seconds"
PREEMPTIONS = "vllm:num_preemptions_total"
KV_USAGE = "vllm:kv_cache_usage_perc"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"

REASON_REQUIRES_HOOK = "requires_hook"
REASON_REQUIRES_TRACE = "requires_trace"
REASON_REQUIRES_CLIENT = "requires_client"
REASON_REQUIRES_REFERENCE = "requires_reference"
# Metrics describe the engine's whole traffic; a value is never one client's.
SCOPE = "engine_global"

_REQUIREMENTS: Mapping[str, str] = MappingProxyType(
    {
        kinds.MIXED_PREFILL_INTERFERENCE: REASON_REQUIRES_HOOK,
        kinds.HOST_STALL: REASON_REQUIRES_HOOK,
        kinds.CAPTURE_PAUSE: REASON_REQUIRES_HOOK,
        kinds.RANK_DELAY: REASON_REQUIRES_TRACE,
        kinds.TRANSFER_DEGRADATION: REASON_REQUIRES_TRACE,
        kinds.CLIENT_ADMISSION: REASON_REQUIRES_CLIENT,
        **{kind: REASON_REQUIRES_CLIENT for kind in kinds.WORKLOAD_KINDS},
    }
)


@dataclass(frozen=True)
class SignalConfig:
    """How to evaluate one window.

    ``thresholds`` overrides entries of the shared table by key; a result
    says when it did. A key the table lacks, or a value that is not a finite
    number, is refused: a NaN would never be exceeded and a misspelt key
    never read, both silently. ``reference`` is the prefix-cache hit ratio a
    window is compared with, which only the caller can know.
    """

    engine: str | None = None
    min_scrapes: int = 2
    thresholds: Mapping[str, float] = field(default_factory=dict)
    reference: float | None = None

    def __post_init__(self) -> None:
        if self.reference is not None and not 0.0 <= self.reference <= 1.0:
            raise ValueError("reference must be a hit ratio between 0 and 1")
        unknown = sorted(set(self.thresholds) - set(DEFAULT_THRESHOLDS))
        if unknown:
            raise ValueError(f"unknown threshold keys: {', '.join(unknown)}")
        if not all(math.isfinite(value) for value in self.thresholds.values()):
            raise ValueError("threshold overrides must be finite numbers")


@dataclass(frozen=True)
class SignalValue:
    """One kind's value over a window, and the verdict against its threshold.

    ``exceeds`` is None whenever ``sufficient`` is False; ``reason`` is then
    the first of the reasons listed in ``detail["reasons"]``.
    """

    value: float | None
    sufficient: bool
    reason: str | None
    exceeds: bool | None
    threshold: float | None
    thresholds_version: str
    threshold_overridden: bool
    detail: Mapping[str, Any]


def evaluate_signal(
    kind: str,
    scrapes: Sequence[VllmScrapeRecord],
    config: SignalConfig | None = None,
) -> SignalValue:
    """Evaluate ``kind`` over a window of consecutive scrapes, in time order.

    Raises:
        ValueError: for a kind outside the closed vocabulary.
    """
    kinds.check_kind(kind)
    config = config or SignalConfig()
    evaluate = _EVALUATORS.get(kind)
    if evaluate is None:
        return _insufficient(None, None, False, (_REQUIREMENTS[kind],), {})
    check = check_window(scrapes, engine=config.engine, min_scrapes=config.min_scrapes)
    return evaluate(scrapes, config, check)


def _queue(
    scrapes: Sequence[VllmScrapeRecord], config: SignalConfig, check: WindowCheck
) -> SignalValue:
    threshold, overridden = resolve_threshold(QUEUE_MEDIAN_WAITING, config.thresholds)
    gauge = gauge_window(scrapes, WAITING, engine=config.engine)
    value = gauge_median(gauge)
    reasons = (*check.reasons, *gauge.reasons)
    detail = {
        **_window_detail(check),
        "max": gauge.max,
        "samples": gauge.n,
        "waiting_by_reason": _waiting_by_reason(scrapes, config.engine),
        "queue_time_p90_s": _queue_time_p90(scrapes, config.engine),
    }
    if reasons or value is None:
        return _insufficient(value, threshold, overridden, reasons, detail)
    return _decided(value, threshold, overridden, detail)


def _waiting_by_reason(
    scrapes: Sequence[VllmScrapeRecord], engine: str | None
) -> dict[str, float | None]:
    """The median of each waiting reason; a reason the server lacks is None."""
    return {
        reason: gauge_median(
            gauge_window(
                scrapes, WAITING_BY_REASON, labels={"reason": reason}, engine=engine
            )
        )
        for reason in ("capacity", "deferred")
    }


def _queue_time_p90(
    scrapes: Sequence[VllmScrapeRecord], engine: str | None
) -> list[float | None] | None:
    bounds = histogram_quantile_bounds(scrapes, QUEUE_TIME, 0.9, engine=engine)
    if bounds.count_delta is None or REASON_NO_OBSERVATIONS in bounds.reasons:
        return None
    return [bounds.lo, bounds.hi]


def _kv(
    scrapes: Sequence[VllmScrapeRecord], config: SignalConfig, check: WindowCheck
) -> SignalValue:
    threshold, overridden = resolve_threshold(KV_PREEMPTIONS, config.thresholds)
    counter = counter_window(scrapes, PREEMPTIONS, engine=config.engine)
    usage = gauge_window(scrapes, KV_USAGE, engine=config.engine)
    detail = {
        **_window_detail(check),
        "rate_per_s": counter.rate_per_s,
        "kv_cache_usage_max": usage.max,
    }
    reasons = (*check.reasons, *counter.reasons)
    if reasons or counter.delta is None:
        return _insufficient(counter.delta, threshold, overridden, reasons, detail)
    return _decided(counter.delta, threshold, overridden, detail)


def _prefix(
    scrapes: Sequence[VllmScrapeRecord], config: SignalConfig, check: WindowCheck
) -> SignalValue:
    threshold, overridden = resolve_threshold(PREFIX_HIT_RATIO_DROP, config.thresholds)
    hits = counter_window(scrapes, PREFIX_HITS, engine=config.engine)
    queries = counter_window(scrapes, PREFIX_QUERIES, engine=config.engine)
    reasons = [*check.reasons, *hits.reasons, *queries.reasons]
    ratio = _hit_ratio(hits.delta, queries.delta) if not reasons else None
    if not reasons and ratio is None:
        reasons.append(REASON_NO_OBSERVATIONS)
    detail = {
        **_window_detail(check),
        "hits": hits.delta,
        "queries": queries.delta,
        "reference": config.reference,
    }
    reference = config.reference
    if reference is None or reasons or ratio is None:
        if not reasons:
            reasons.append(REASON_REQUIRES_REFERENCE)
        return _insufficient(ratio, threshold, overridden, reasons, detail)
    # The value is the ratio; the verdict is on its fall below the reference.
    detail["drop"] = reference - ratio
    return replace(
        _decided(reference - ratio, threshold, overridden, detail), value=ratio
    )


def _hit_ratio(hits: float | None, queries: float | None) -> float | None:
    """Hits per queried token, or None when nothing was queried."""
    if hits is None or not queries:
        return None
    return hits / queries


def _window_detail(check: WindowCheck) -> dict[str, Any]:
    return {
        "scope": SCOPE,
        "scrapes": check.scrapes,
        # Failed scrapes inside the window: a caller need not count again.
        "failed_scrapes": check.failed,
        "window_seconds": check.seconds,
        "placement": check.placement,
    }


def _decided(
    value: float, threshold: float, overridden: bool, detail: dict[str, Any]
) -> SignalValue:
    detail["reasons"] = []
    return SignalValue(
        value,
        True,
        None,
        value >= threshold,
        threshold,
        THRESHOLDS_VERSION,
        overridden,
        MappingProxyType(detail),
    )


def _insufficient(
    value: float | None,
    threshold: float | None,
    overridden: bool,
    reasons: Sequence[str],
    detail: dict[str, Any],
) -> SignalValue:
    unique = list(dict.fromkeys(reasons))
    detail["reasons"] = unique
    detail.setdefault("scope", SCOPE)
    return SignalValue(
        value,
        False,
        unique[0] if unique else None,
        None,
        threshold,
        THRESHOLDS_VERSION,
        overridden,
        MappingProxyType(detail),
    )


_EVALUATORS = MappingProxyType(
    {
        kinds.QUEUE_SATURATION: _queue,
        kinds.KV_PREEMPTION_PRESSURE: _kv,
        kinds.PREFIX_CACHE_LOSS: _prefix,
    }
)


__all__ = [
    "REASON_REQUIRES_CLIENT",
    "REASON_REQUIRES_HOOK",
    "REASON_REQUIRES_REFERENCE",
    "REASON_REQUIRES_TRACE",
    "SignalConfig",
    "SignalValue",
    "evaluate_signal",
]
