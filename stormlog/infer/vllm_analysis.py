"""Explain a run's queueing, cache and token signals from vLLM's own telemetry.

Everything here is aggregate evidence: a scrape describes every request the
engine served, from every client, and nothing in it says which request used
which GPU time. Deltas are taken between the scrape just before a phase's
first send and the scrape after its drain, and only when both succeeded on
one engine epoch. A counter that went backwards, an engine that restarted,
or a series one scrape lacks leaves that field unresolved with the reason,
never a zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report_stats import percentile
from .vllm_metrics import (
    CATALOG,
    ENGINE_LABELS,
    CompactScrape,
    Discovery,
    HistogramValue,
    created_family_for,
    family_group,
    resolve_name,
)
from .vllm_spans import read_span_file, span_clock_domain, span_record
from .vllm_telemetry import (
    MARKER_PHASE_END,
    MARKER_PHASE_START,
    SCRAPE_OK,
    VllmScrapeRecord,
    VllmSpanRecord,
    load_vllm_records,
)

NOTE = (
    "engine-aggregate evidence over every request the engine served in the "
    "window, other clients' traffic included; nothing here attributes GPU "
    "time to a request"
)
RESIDENCY_NOTE = "wall-clock residency in a scheduler phase, not GPU time"

STATE_RESOLVED = "resolved"
STATE_UNRESOLVED = "unresolved"
REASON_SCRAPE_MISSING = "scrape_missing"
REASON_SCRAPE_FAILED = "scrape_failed"
REASON_ENGINE_RESTART = "engine_restart"
REASON_ENGINES_CHANGED = "engine_set_changed"
REASON_COUNTER_RESET = "counter_reset"
REASON_COUNTER_RECREATED = "counter_recreated"
REASON_SERIES_MISSING = "series_missing"
REASON_BOUNDARIES_CHANGED = "bucket_boundaries_changed"
REASON_NOT_ENABLED = "not_enabled"

SPAN_LATENCY_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("time_in_queue", "gen_ai.latency.time_in_queue"),
    ("time_to_first_token", "gen_ai.latency.time_to_first_token"),
    ("time_in_model_prefill", "gen_ai.latency.time_in_model_prefill"),
    ("time_in_model_decode", "gen_ai.latency.time_in_model_decode"),
    ("time_in_model_inference", "gen_ai.latency.time_in_model_inference"),
    ("e2e", "gen_ai.latency.e2e"),
)

SeriesKey = tuple[str, tuple[tuple[str, str], ...]]


# ----------------------------------------------------------------- entry points
def load_external_spans(
    records: list[dict[str, Any]], paths: Iterable[str | Path]
) -> list[VllmSpanRecord]:
    """Spans someone else collected, stamped with this artifact's identity."""
    session_id, run_id = _artifact_identity(records)
    spans: list[VllmSpanRecord] = []
    for path in paths:
        source, raw_spans = read_span_file(path)
        host = Path(path).stem
        spans.extend(
            span_record(
                raw,
                session_id=session_id,
                run_id=run_id,
                source=source,
                clock_domain=span_clock_domain(raw.resource, host),
            )
            for raw in raw_spans
        )
    return spans


def vllm_report(
    records: list[dict[str, Any]], external_spans: Iterable[VllmSpanRecord] = ()
) -> dict[str, Any]:
    """The ``telemetry.vllm`` block of an analysis report."""
    scrapes, spans = load_vllm_records(records)
    spans = [*spans, *external_spans]
    capabilities = _capabilities(records)
    if not scrapes and not spans and not capabilities:
        return {"status": "absent"}
    requests = _measured(records, "infer.request")
    case_ids = sorted(
        {
            str(window.get("case_id"))
            for window in _measured(records, "infer.phase_window")
        }
    )
    join = _join_spans(spans, requests, _warmup_request_ids(records))
    cases = {
        case_id: _case_block(case_id, scrapes, requests, join.by_request)
        for case_id in case_ids
    }
    return {
        "status": "collected",
        "observation_scope": "engine_aggregate",
        "note": NOTE,
        "capabilities": capabilities,
        "engine": _engine_summary(scrapes),
        "cases": cases,
        "spans": join.summary(),
    }


def _measured(records: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("event_type") == event_type and record.get("phase") == "measured"
    ]


def _artifact_identity(records: list[dict[str, Any]]) -> tuple[str, str]:
    for record in records:
        if record.get("event_type") == "infer.artifact":
            context = record.get("context") or {}
            return str(context.get("session_id") or "unknown"), str(
                context.get("run_id") or "unknown"
            )
    for record in records:
        if record.get("event_type") == "infer.session":
            return str(record.get("session_id") or "unknown"), "unknown"
    return "unknown", "unknown"


def _capabilities(records: list[dict[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for record in records:
        if record.get("event_type") != "infer.capabilities":
            continue
        component = str(record.get("component", ""))
        if component.startswith("vllm."):
            found[component] = {
                key: record.get(key)
                for key in (
                    "available",
                    "supported",
                    "enabled",
                    "collected",
                    "metadata",
                )
            }
    return found


# ----------------------------------------------------------------- engine summary
def _engine_summary(scrapes: list[VllmScrapeRecord]) -> dict[str, Any]:
    ok = [item for item in scrapes if item.status == SCRAPE_OK]
    discoveries = [item.discovery for item in ok if item.discovery is not None]
    return {
        "source_url": scrapes[0].source_url if scrapes else None,
        "scrapes": {"ok": len(ok), "failed": len(scrapes) - len(ok)},
        "epochs": _epochs(ok),
        **_discovery_union(discoveries),
    }


def _epochs(ok: list[VllmScrapeRecord]) -> list[dict[str, Any]]:
    """Runs of scrapes that saw one exporter process start time."""
    epochs: list[dict[str, Any]] = []
    for item in ok:
        start_ns = item.discovery.process_start_ns if item.discovery else None
        if epochs and epochs[-1]["process_start_ns"] == start_ns:
            epochs[-1]["last_observed_at_ns"] = item.observed_at_ns
            epochs[-1]["scrapes"] += 1
            continue
        epochs.append(
            {
                "process_start_ns": start_ns,
                "first_observed_at_ns": item.observed_at_ns,
                "last_observed_at_ns": item.observed_at_ns,
                "scrapes": 1,
            }
        )
    return epochs


def _discovery_union(discoveries: list[Discovery]) -> dict[str, Any]:
    def union(field_name: str) -> list[str]:
        return sorted({n for d in discoveries for n in getattr(d, field_name)})

    optional_absent: set[str] = set()
    if discoveries:
        optional_absent = set.intersection(
            *(set(d.optional_absent) for d in discoveries)
        )
    return {
        "engines": union("engines"),
        "model_names": union("model_names"),
        "unknown_series": union("unknown"),
        "deprecated_series": union("deprecated_present"),
        "removed_series": union("removed_present"),
        "optional_absent": sorted(optional_absent),
        "optional_present": union("optional_present"),
    }


# ----------------------------------------------------------------- per case
@dataclass(frozen=True)
class _Boundary:
    start: VllmScrapeRecord | None
    end: VllmScrapeRecord | None
    reasons: tuple[str, ...]


def _boundary(case_id: str, scrapes: list[VllmScrapeRecord]) -> _Boundary:
    mine = [s for s in scrapes if s.case_id == case_id and s.phase == "measured"]
    start = _pick(mine, MARKER_PHASE_START, first=True)
    end = _pick(mine, MARKER_PHASE_END, first=False)
    reasons = []
    for marker, found in ((MARKER_PHASE_START, start), (MARKER_PHASE_END, end)):
        if found is None:
            failed = any(s.marker == marker for s in mine)
            reasons.append(
                f"{REASON_SCRAPE_FAILED}:{marker}"
                if failed
                else f"{REASON_SCRAPE_MISSING}:{marker}"
            )
    return _Boundary(start, end, tuple(reasons))


def _pick(
    scrapes: list[VllmScrapeRecord], marker: str, *, first: bool
) -> VllmScrapeRecord | None:
    matching = [s for s in scrapes if s.marker == marker and s.status == SCRAPE_OK]
    if not matching:
        return None
    return matching[0] if first else matching[-1]


def _case_block(
    case_id: str,
    scrapes: list[VllmScrapeRecord],
    requests: list[dict[str, Any]],
    spans_by_request: dict[str, list[VllmSpanRecord]],
) -> dict[str, Any]:
    boundary = _boundary(case_id, scrapes)
    case_requests = [r for r in requests if str(r.get("case_id")) == case_id]
    block: dict[str, Any] = {
        "state": STATE_UNRESOLVED if boundary.reasons else STATE_RESOLVED,
        "reasons": list(boundary.reasons),
        "window": None,
        "engines": {},
        "spans": _case_spans(case_requests, spans_by_request),
    }
    start, end = boundary.start, boundary.end
    if start is None or end is None or start.scrape is None or end.scrape is None:
        return block
    reasons = _epoch_reasons(start, end)
    block["reasons"].extend(reasons)
    if reasons:
        block["state"] = STATE_UNRESOLVED
    seconds = (end.observed_at_ns - start.observed_at_ns) / 1e9
    block["window"] = {
        "start_observed_at_ns": start.observed_at_ns,
        "end_observed_at_ns": end.observed_at_ns,
        "seconds": seconds,
        "includes_drain": True,
    }
    inside = _inside(scrapes, case_id, start.observed_at_ns, end.observed_at_ns)
    engines = set(start.scrape.label_values("engine"))
    engines |= set(end.scrape.label_values("engine"))
    for engine in sorted(engines):
        block["engines"][engine] = _engine_block(
            engine, start.scrape, end.scrape, inside, seconds, bool(reasons)
        )
    return block


def _inside(
    scrapes: list[VllmScrapeRecord], case_id: str, start_ns: int, end_ns: int
) -> list[VllmScrapeRecord]:
    """The successful scrapes of one case between its boundary scrapes."""
    return [
        s
        for s in scrapes
        if s.case_id == case_id
        and s.phase == "measured"
        and s.status == SCRAPE_OK
        and start_ns <= s.observed_at_ns <= end_ns
    ]


def _epoch_reasons(start: VllmScrapeRecord, end: VllmScrapeRecord) -> list[str]:
    reasons = []
    first, last = start.discovery, end.discovery
    if first is not None and last is not None:
        if first.process_start_ns != last.process_start_ns:
            reasons.append(REASON_ENGINE_RESTART)
        if set(first.engines) != set(last.engines):
            reasons.append(REASON_ENGINES_CHANGED)
    return reasons


# ----------------------------------------------------------------- per engine
def _index(
    scrape: CompactScrape, engine: str
) -> dict[SeriesKey, float | HistogramValue]:
    """Series of one engine keyed by (family, extra labels beyond the engine's)."""
    indexed: dict[SeriesKey, float | HistogramValue] = {}
    for name, by_set in scrape.values.items():
        for set_id, value in by_set.items():
            labels = scrape.labels(set_id)
            if labels.get("engine") != engine:
                continue
            extra = tuple(
                sorted((k, v) for k, v in labels.items() if k not in ENGINE_LABELS)
            )
            indexed[(name, extra)] = value
    return indexed


def _engine_block(
    engine: str,
    start: CompactScrape,
    end: CompactScrape,
    inside: list[VllmScrapeRecord],
    seconds: float,
    epoch_broken: bool,
) -> dict[str, Any]:
    before, after = _index(start, engine), _index(end, engine)
    kinds = {**start.families, **end.families}
    blocks: dict[str, dict[str, Any]] = {"counter": {}, "histogram": {}, "gauge": {}}
    for name in sorted({key[0] for key in before} | {key[0] for key in after}):
        if name.endswith("_created") or not name.startswith("vllm:"):
            continue
        kind = kinds.get(name, "untyped")
        if kind not in blocks:
            continue
        canonical, alias = resolve_name(name)
        entry = CATALOG.get(canonical)
        field_name = entry.field if entry is not None else name
        if kind == "gauge":
            blocks[kind][field_name] = _gauge_field(name, alias, engine, inside)
        else:
            build = _counter_field if kind == "counter" else _histogram_field
            blocks[kind][field_name] = build(name, alias, before, after, epoch_broken)
    counters, histograms, gauges = (
        blocks["counter"],
        blocks["histogram"],
        blocks["gauge"],
    )
    return {
        "counters": counters,
        "histograms": histograms,
        "gauges": gauges,
        "derived": _derived(counters, histograms, gauges, seconds),
    }


def _by_extra(
    name: str, indexed: dict[SeriesKey, float | HistogramValue]
) -> dict[tuple[tuple[str, str], ...], float | HistogramValue]:
    return {
        extra: value for (family, extra), value in indexed.items() if family == name
    }


def _label_key(extra: tuple[tuple[str, str], ...]) -> str:
    return ",".join(f"{k}={v}" for k, v in extra) or "_"


def _counter_field(
    name: str,
    alias: str | None,
    before: dict[SeriesKey, float | HistogramValue],
    after: dict[SeriesKey, float | HistogramValue],
    epoch_broken: bool,
) -> dict[str, Any]:
    entry = CATALOG.get(resolve_name(name)[0])
    result: dict[str, Any] = {
        "native": name,
        "unit": entry.unit if entry else None,
        "provenance": entry.provenance if entry else "observed",
        "deprecated_alias_of": alias,
        "by_label": {},
    }
    first, last = _by_extra(name, before), _by_extra(name, after)
    created_name = created_family_for(name) or ""
    recreated = _by_extra(created_name, before) != _by_extra(created_name, after)
    for extra in sorted(set(first) | set(last)):
        a, b = first.get(extra), last.get(extra)
        result["by_label"][_label_key(extra)] = _delta(a, b, epoch_broken, recreated)
    values = result["by_label"].values()
    states = {item["state"] for item in values}
    result["state"] = STATE_RESOLVED if states == {STATE_RESOLVED} else STATE_UNRESOLVED
    result["delta"] = (
        sum(item["delta"] for item in values)
        if result["state"] == STATE_RESOLVED
        else None
    )
    return result


def _delta(
    a: float | HistogramValue | None,
    b: float | HistogramValue | None,
    epoch_broken: bool,
    recreated: bool,
) -> dict[str, Any]:
    if not isinstance(a, float) or not isinstance(b, float):
        return {"state": REASON_SERIES_MISSING, "delta": None}
    if epoch_broken:
        return {"state": REASON_ENGINE_RESTART, "delta": None, "start": a, "end": b}
    if recreated:
        return {"state": REASON_COUNTER_RECREATED, "delta": None, "start": a, "end": b}
    if b < a:
        return {"state": REASON_COUNTER_RESET, "delta": None, "start": a, "end": b}
    return {"state": STATE_RESOLVED, "delta": b - a, "start": a, "end": b}


def _histogram_field(
    name: str,
    alias: str | None,
    before: dict[SeriesKey, float | HistogramValue],
    after: dict[SeriesKey, float | HistogramValue],
    epoch_broken: bool,
) -> dict[str, Any]:
    entry = CATALOG.get(resolve_name(name)[0])
    first, last = _by_extra(name, before), _by_extra(name, after)
    a = next((v for v in first.values() if isinstance(v, HistogramValue)), None)
    b = next((v for v in last.values() if isinstance(v, HistogramValue)), None)
    result: dict[str, Any] = {
        "native": name,
        "unit": entry.unit if entry else None,
        "deprecated_alias_of": alias,
        "meaning": entry.meaning if entry else None,
    }
    result.update(_histogram_delta(a, b, epoch_broken))
    return result


def _histogram_delta(
    a: HistogramValue | None, b: HistogramValue | None, epoch_broken: bool
) -> dict[str, Any]:
    if a is None or b is None:
        return {"state": REASON_SERIES_MISSING}
    if epoch_broken:
        return {"state": REASON_ENGINE_RESTART}
    if [le for le, _ in a.buckets] != [le for le, _ in b.buckets]:
        return {"state": REASON_BOUNDARIES_CHANGED}
    if b.count < a.count:
        return {"state": REASON_COUNTER_RESET}
    count = b.count - a.count
    total = b.sum - a.sum
    return {
        "state": STATE_RESOLVED,
        "count": count,
        "sum": total,
        "mean": total / count if count else None,
        "buckets": [
            [le, after - before]
            for (le, before), (_le, after) in zip(a.buckets, b.buckets)
        ],
    }


def _gauge_field(
    name: str, alias: str | None, engine: str, inside: list[VllmScrapeRecord]
) -> dict[str, Any]:
    entry = CATALOG.get(resolve_name(name)[0])
    result: dict[str, Any] = {
        "native": name,
        "unit": entry.unit if entry else None,
        "deprecated_alias_of": alias,
    }
    series: dict[str, list[float]] = {}
    last_labels: dict[str, dict[str, str]] = {}
    for item in inside:
        if item.scrape is None:
            continue
        for extra, value in _by_extra(name, _index(item.scrape, engine)).items():
            if isinstance(value, float):
                series.setdefault(_label_key(extra), []).append(value)
                last_labels[_label_key(extra)] = dict(extra)
    if not series:
        result["state"] = REASON_SERIES_MISSING
        return result
    result["state"] = STATE_RESOLVED
    result["stats"] = {key: _stats(values) for key, values in series.items()}
    result["labels"] = last_labels
    return result


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "last": values[-1],
        "samples": len(values),
    }


def _derived(
    counters: dict[str, Any],
    histograms: dict[str, Any],
    gauges: dict[str, Any],
    seconds: float,
) -> dict[str, Any]:
    prompt = _resolved_delta(counters, "prompt_tokens")
    generated = _resolved_delta(counters, "generation_tokens")
    finished = _resolved_delta(counters, "request_success")
    queries = _resolved_delta(counters, "prefix_cache_queries")
    hits = _resolved_delta(counters, "prefix_cache_hits")
    derived: dict[str, Any] = {
        "rates": {
            "prompt_tokens_per_second": _rate(prompt, seconds),
            "generation_tokens_per_second": _rate(generated, seconds),
            "finished_requests_per_second": _rate(finished, seconds),
            "window_seconds": seconds,
        },
        "prefix_cache": {
            "queries": queries,
            "hits": hits,
            "hit_ratio": (hits / queries) if queries and hits is not None else None,
        },
        "kv_cache": _kv_cache(gauges),
        "mfu": _mfu(counters, generated),
    }
    optional = {
        field_name: item
        for field_name, item in counters.items()
        if family_group(item["native"]) in {"spec_decode", "kv_transfer"}
    }
    if optional:
        derived["optional_counters"] = optional
    return derived


def _resolved_delta(counters: dict[str, Any], field_name: str) -> float | None:
    item = counters.get(field_name)
    if item is None or item.get("state") != STATE_RESOLVED:
        return None
    delta = item.get("delta")
    return float(delta) if isinstance(delta, (int, float)) else None


def _rate(delta: float | None, seconds: float) -> float | None:
    return delta / seconds if delta is not None and seconds > 0 else None


def _kv_cache(gauges: dict[str, Any]) -> dict[str, Any]:
    usage = gauges.get("kv_cache_usage", {})
    stats = (usage.get("stats") or {}).get("_")
    config = _cache_config_labels(gauges)
    block_size = _int_label(config, "block_size")
    size_tokens = _int_label(config, "kv_cache_size_tokens")
    blocks = _block_count(config, block_size, size_tokens)
    max_usage = stats["max"] if stats else None
    return {
        "state": usage.get("state", REASON_SERIES_MISSING),
        "max_usage_fraction": max_usage,
        "mean_usage_fraction": stats["mean"] if stats else None,
        "num_gpu_blocks": blocks,
        "block_size": block_size,
        "kv_cache_size_tokens": size_tokens,
        "max_blocks_in_use": (
            round(max_usage * blocks) if max_usage is not None and blocks else None
        ),
        "meaning": "logical KV block occupancy; not device memory",
    }


def _cache_config_labels(gauges: dict[str, Any]) -> dict[str, str]:
    """``cache_config_info`` carries the configuration as labels of its one series."""
    labelled = gauges.get("cache_config", {}).get("labels") or {}
    if len(labelled) != 1:
        return {}
    labels: dict[str, str] = next(iter(labelled.values()))
    return labels


def _block_count(
    config: dict[str, str], block_size: int | None, size_tokens: int | None
) -> int | None:
    """KV blocks from the labels; 0.30.0 gives the cache size in tokens."""
    blocks = _int_label(config, "num_gpu_blocks")
    if blocks is None and size_tokens and block_size:
        blocks = size_tokens // block_size
    return blocks


def _int_label(labels: dict[str, str], key: str) -> int | None:
    value = labels.get(key)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _mfu(counters: dict[str, Any], generated: float | None) -> dict[str, Any]:
    fields = (
        "estimated_flops_per_gpu",
        "estimated_read_bytes_per_gpu",
        "estimated_write_bytes_per_gpu",
    )
    deltas = {name: _resolved_delta(counters, name) for name in fields}
    if all(value is None for value in deltas.values()):
        return {"state": REASON_SERIES_MISSING}
    if generated and all(not value for value in deltas.values()):
        return {
            "state": REASON_NOT_ENABLED,
            "detail": "the counters did not advance while tokens were generated; "
            "vLLM needs --enable-mfu-metrics",
        }
    return {
        "state": STATE_RESOLVED,
        "provenance": "estimated",
        "per_gpu": True,
        **deltas,
    }


# ----------------------------------------------------------------- spans
@dataclass
class _SpanJoin:
    by_request: dict[str, list[VllmSpanRecord]]
    total: int
    joined: int
    unjoined: dict[str, int]
    sources: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "joined": self.joined,
            "unjoined_by_reason": dict(sorted(self.unjoined.items())),
            "sources": dict(sorted(self.sources.items())),
            "note": RESIDENCY_NOTE,
        }


def _warmup_request_ids(records: list[dict[str, Any]]) -> set[str]:
    """The sent ids of requests outside the measured phase, such as warmup."""
    return {
        str(r["x_request_id"])
        for r in records
        if r.get("event_type") == "infer.request"
        and r.get("phase") != "measured"
        and isinstance(r.get("x_request_id"), str)
    }


def _join_spans(
    spans: list[VllmSpanRecord],
    requests: list[dict[str, Any]],
    warmup_ids: set[str] | None = None,
) -> _SpanJoin:
    known = {
        str(r["x_request_id"])
        for r in requests
        if isinstance(r.get("x_request_id"), str)
    }
    by_request: dict[str, list[VllmSpanRecord]] = {}
    unjoined: dict[str, int] = {}
    sources: dict[str, int] = {}
    for span in spans:
        sources[span.source] = sources.get(span.source, 0) + 1
        reason = _unjoined_reason(span, known, warmup_ids or set())
        if reason is None and span.request_id is not None:
            by_request.setdefault(span.request_id, []).append(span)
        else:
            unjoined[reason or "unknown"] = unjoined.get(reason or "unknown", 0) + 1
    joined = sum(len(items) for items in by_request.values())
    return _SpanJoin(by_request, len(spans), joined, unjoined, sources)


def _unjoined_reason(
    span: VllmSpanRecord, known: set[str], warmup_ids: set[str]
) -> str | None:
    if span.name != "llm_request":
        return "not_a_request_span"
    if span.request_id is None:
        return "no_request_id"
    if span.request_id in warmup_ids:
        return "warmup_request"
    if span.request_id not in known:
        return "request_not_in_run"
    return None


def _case_spans(
    requests: list[dict[str, Any]], by_request: dict[str, list[VllmSpanRecord]]
) -> dict[str, Any]:
    joined = [
        span
        for r in requests
        for span in by_request.get(str(r.get("x_request_id")), [])
    ]
    latency: dict[str, Any] = {}
    for short, attribute in SPAN_LATENCY_ATTRIBUTES:
        values = [
            float(span.attributes[attribute]) * 1000.0
            for span in joined
            if isinstance(span.attributes.get(attribute), (int, float))
            and not isinstance(span.attributes.get(attribute), bool)
        ]
        if values:
            latency[short] = {
                "p50_ms": percentile(values, 50),
                "p95_ms": percentile(values, 95),
                "mean_ms": sum(values) / len(values),
                "n": len(values),
            }
    return {
        "requests": len(requests),
        "requests_with_span": sum(
            1 for r in requests if by_request.get(str(r.get("x_request_id")))
        ),
        "spans": len(joined),
        "latency": latency,
        "note": RESIDENCY_NOTE,
    }


# ----------------------------------------------------------------- text
def vllm_lines(block: Any) -> list[str]:
    """Report-level lines for the text report."""
    if not isinstance(block, dict) or block.get("status") != "collected":
        return []
    engine = block.get("engine", {})
    scrapes = engine.get("scrapes", {})
    spans = block.get("spans", {})
    engines = ", ".join(engine.get("engines", [])) or "none seen"
    lines = [
        f"vLLM telemetry: {scrapes.get('ok', 0)} scrapes ok, "
        f"{scrapes.get('failed', 0)} failed; engine label(s) {engines}; "
        f"spans {spans.get('total', 0)} ({spans.get('joined', 0)} joined); "
        "engine-aggregate, no per-request attribution"
    ]
    if len(engine.get("epochs", [])) > 1:
        lines.append(
            f"vLLM telemetry: the server restarted {len(engine['epochs']) - 1} time(s) during the run"
        )
    for label, names in (
        ("unknown series", engine.get("unknown_series")),
        ("retired series", engine.get("deprecated_series")),
        ("removed series", engine.get("removed_series")),
    ):
        if names:
            lines.append(f"vLLM telemetry: {label}: {', '.join(names)}")
    return lines


def vllm_case_lines(block: Any) -> list[str]:
    """Per-case lines for the text report."""
    if not isinstance(block, dict):
        return []
    lines: list[str] = []
    if block.get("state") == STATE_UNRESOLVED:
        lines.append(f"  vllm: unresolved ({', '.join(block.get('reasons', []))})")
    for engine, item in block.get("engines", {}).items():
        lines.append(f"  vllm engine {engine}: {_engine_line(item)}")
    spans = block.get("spans", {})
    if spans.get("spans"):
        inference = spans.get("latency", {}).get("time_in_model_inference", {})
        lines.append(
            f"  vllm spans: {spans.get('requests_with_span')} of {spans.get('requests')} "
            f"requests; inference time p50 {_fmt(inference.get('p50_ms'))} ms "
            "(residency, not GPU time)"
        )
    return lines


def _engine_line(item: dict[str, Any]) -> str:
    gauges = item.get("gauges", {})
    counters = item.get("counters", {})
    derived = item.get("derived", {})
    waiting = _gauge_stat(gauges, "queue_depth")
    running = _gauge_stat(gauges, "running_requests")
    kv = derived.get("kv_cache", {})
    rates = derived.get("rates", {})
    prefix = derived.get("prefix_cache", {})
    queue = item.get("histograms", {}).get("queue_time", {})
    parts = [
        f"waiting max {_fmt(waiting.get('max'))} mean {_fmt(waiting.get('mean'))}",
        f"running max {_fmt(running.get('max'))}",
        f"preemptions {_fmt(_resolved_delta(counters, 'preemptions'))}",
        f"kv usage max {_pct(kv.get('max_usage_fraction'))}"
        + (
            f" ({kv['max_blocks_in_use']} of {kv['num_gpu_blocks']} blocks)"
            if kv.get("num_gpu_blocks")
            else ""
        ),
        f"prefix hits {_fmt(prefix.get('hits'))} of {_fmt(prefix.get('queries'))} tokens",
        f"queue time mean {_fmt(_ms(queue.get('mean')))} ms over {_fmt(queue.get('count'))} requests",
        f"{_fmt(rates.get('prompt_tokens_per_second'))} prompt and "
        f"{_fmt(rates.get('generation_tokens_per_second'))} generated tok/s "
        f"over {_fmt(rates.get('window_seconds'))} s incl. drain",
    ]
    return "; ".join(parts)


def _gauge_stat(gauges: dict[str, Any], field_name: str) -> dict[str, Any]:
    stats = gauges.get(field_name, {}).get("stats") or {}
    return dict(stats.get("_") or {})


def _ms(seconds: Any) -> float | None:
    return seconds * 1000.0 if isinstance(seconds, (int, float)) else None


def _pct(fraction: Any) -> str:
    return f"{fraction * 100:.1f}%" if isinstance(fraction, (int, float)) else "n/a"


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    return f"{value:,}" if isinstance(value, int) else str(value)
