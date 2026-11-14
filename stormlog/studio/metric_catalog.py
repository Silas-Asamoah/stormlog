"""What each displayed value means, and which panels an artifact can show (§5.5).

``metric_catalog`` is one entry per value Studio displays: its unit, scope,
the record fields it comes from, the analyzer function that computes it,
the requests it uses, the interval it aggregates over, and whether Studio
computes it (``indexed``) or only the analyzer report has it
(``not_indexed``). Latency entries are read from ``slo.CRITERIA``, so their
definitions cannot drift from the code that computes them.

``panels`` says, for one snapshot, which panels can be shown and why not:
the states and reasons of §5.5. Server evidence (joined vLLM spans, the
execution import's records) is ``not_indexed`` when the artifact holds it
(decision 4(a)): the index does not read those bodies, and the analyzer
report's values are shown, labelled, in Sources. The analyzer's
``observers`` block is passed through, labelled with its source, once the
out-of-process analysis is ready.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from stormlog.infer.latency_report import CHUNK_NOTE, CLIENT_METRICS, SERVER_METRICS
from stormlog.infer.observers import SCRAPE_GAP_FACTOR
from stormlog.infer.report_stats import is_number
from stormlog.infer.slo import CRITERIA
from stormlog.studio.contracts import AnalysisState, IndexSnapshot
from stormlog.studio.selection import case_facts

__all__ = [
    "PANELS",
    "PANEL_STATES",
    "SERVER_EVIDENCE_TYPES",
    "metric_catalog",
    "panels",
]

PANEL_STATES = (
    "available",
    "empty",
    "pending",
    "partial",
    "unavailable",
    "not_indexed",
    "in_artifact_panel_later",
    "unsupported",
)
PANELS = (
    "requests",
    "latency",
    "failures",
    "streaming",
    "tokens",
    "rates",
    "cases",
    "client_memory",
    "server_evidence",
    "vllm_metrics",
    "trace",
    "execution",
    "server_probe",
    "manifest",
    "analysis",
)
# Record types that carry server evidence the index does not read.
SERVER_EVIDENCE_TYPES = (
    "infer.vllm_span",
    "infer.iteration",
    "infer.membership",
    "infer.stage",
    "infer.clock_alignment",
)
_LATER = {
    "vllm_metrics": ("infer.vllm_scrape",),
    "trace": ("infer.activity_ref", "infer.trace_window"),
    "execution": ("infer.iteration", "infer.membership", "infer.stage"),
    "server_probe": ("infer.server_probe",),
    "manifest": ("infer.manifest",),
}
_COLLECTORS = {
    "client_memory": "stormlog infer profile --system-sampler psutil (or nvml)",
    "server_evidence": (
        "stormlog infer profile --vllm-spans-listen, or "
        "stormlog infer import-execution"
    ),
    "vllm_metrics": "stormlog infer profile --vllm-metrics",
    "trace": "stormlog infer import-trace",
    "execution": "stormlog infer import-execution",
    "server_probe": "stormlog infer profile (probes the server before and after)",
    "manifest": "stormlog infer attach-manifest",
    "tokens": "a server that reports usage, or --tokenizer",
}
SELECTION = "the selection"
RATE_INTERVAL = (
    "the case's rate interval (populations.CaseIntervals.rate), or the part "
    "of it a window clamps to; none for a selection that is not rate-eligible"
)
SAMPLE_WINDOW = (
    "first start to last end of the selection's successful requests, both "
    "ends included (sample_report.request_window)"
)
QUANTILE_RULE = (
    "rank p * (n - 1), linear interpolation between order statistics; "
    "sufficiency by the order-statistic interval rule (quantiles.py)"
)


# --- The catalog --------------------------------------------------------------------


def metric_catalog() -> dict[str, Any]:
    """The ``metric_catalog`` body: one entry per displayed value."""
    entries = [
        *(_latency_entry(key) for key in CLIENT_METRICS),
        *(_server_entry(key) for key in CRITERIA if key.startswith("server.")),
        *_count_entries(),
        *_rate_entries(),
        *_streaming_entries(),
        *_token_entries(),
        *_memory_entries(),
        *_request_entries(),
    ]
    return {"metrics": entries, "states": list(PANEL_STATES)}


def _entry(key: str, label: str, unit: str | None, **fields: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": key,
        "label": label,
        "unit": unit,
        "scope": "client",
        "sources": [],
        "computed_by": None,
        "formula": None,
        "missing_rule": None,
        "applies_to": SELECTION,
        "aggregation_window": SELECTION,
        "notes": [],
        "status": "indexed",
        "reason": None,
        "collector": "stormlog infer profile",
    }
    entry.update(fields)
    return entry


def _latency_entry(key: str) -> dict[str, Any]:
    definition = CRITERIA[key]
    return _entry(
        key,
        definition.definition,
        "ms",
        sources=[definition.per_request] if definition.per_request else [],
        computed_by="latency_report.latency_summary",
        formula=QUANTILE_RULE,
        missing_rule=(
            "a missing, negative or non-finite value is no value "
            "(slo.client_values); counted in successful_missing"
        ),
        applies_to=(
            "successful requests (successful); every offered request with the "
            "ones that did not succeed ranked worst (failure_penalized)"
        ),
    )


def _server_entry(key: str) -> dict[str, Any]:
    definition = CRITERIA[key]
    sources = [s for s in (definition.per_request, definition.aggregate) if s]
    return _entry(
        key,
        definition.definition,
        "ms",
        scope="server",
        sources=sources,
        computed_by=(
            "latency_report.latency_summary with joined spans"
            if key in SERVER_METRICS
            else "the vLLM metrics block of infer analyze"
        ),
        status="not_indexed",
        reason="server_evidence",
        notes=[
            "the index does not keep joined spans or execution records; the "
            "analyzer report's value is shown, labelled, in Sources"
        ],
        collector=_COLLECTORS["server_evidence"],
    )


_COUNTS = (
    ("offered", "requests offered: every measured request of the selection"),
    ("dropped", "never sent: the client's queue was full"),
    ("sent", "offered and not dropped"),
    ("unreachable", "never reached the server"),
    ("delivery_unknown", "may or may not have reached the server"),
    ("rejected", "refused by the server"),
    ("accepted", "sent and not refused, on the client's view"),
    ("successful", "status ok"),
    ("failed", "status error"),
    ("timed_out", "status timeout"),
    ("cancelled", "status cancelled"),
    ("censored", "timed out or cancelled: latency known only to exceed the time seen"),
    ("scheduled", "arrivals the phase window or the seeded workload scheduled"),
)


def _count_entries() -> list[dict[str, Any]]:
    entries = [
        _entry(
            f"counts.{name}",
            label,
            "requests",
            sources=["infer.request status"],
            computed_by=(
                "populations.populations_from_records for a whole case, else "
                "populations.count_population"
            ),
        )
        for name, label in _COUNTS
    ]
    entries.append(
        _entry(
            "failure.share",
            "share of offered requests that did not succeed",
            "ratio",
            sources=["infer.request status"],
            computed_by="latency_report.failure_share",
            formula="(offered - successful) / offered",
            missing_rule="null with empty_selection when nothing was offered",
        )
    )
    for name, label in (
        ("server_admitted", "requests the server confirmed it saw"),
        ("server_evidence_coverage", "share of accepted requests the server saw"),
    ):
        entries.append(
            _entry(
                f"counts.{name}",
                label,
                "requests" if name == "server_admitted" else "ratio",
                scope="server",
                sources=["joined vLLM spans", "infer.request (execution import)"],
                computed_by="populations.count_population with server IDs",
                status="not_indexed",
                reason="server_evidence",
                collector=_COLLECTORS["server_evidence"],
            )
        )
    return entries


def _rate_entries() -> list[dict[str, Any]]:
    rates = (
        ("requests_per_second", "successful requests per second", "requests/s"),
        (
            "output_tokens_per_second",
            "output tokens of successful requests per second",
            "tokens/s",
        ),
        (
            "total_tokens_per_second",
            "total tokens of successful requests per second",
            "tokens/s",
        ),
    )
    return [
        _entry(
            f"throughput.{name}",
            label,
            unit,
            sources=[
                "infer.request status, output_tokens, total_tokens",
                "infer.phase_window",
                "infer.workload",
            ],
            computed_by="token_report.successful_throughput",
            formula="count / interval seconds; a missing token count adds 0",
            applies_to="successful requests",
            aggregation_window=RATE_INTERVAL,
            notes=["0.0 for a case where nothing succeeded, over a defined interval"],
        )
        for name, label, unit in rates
    ]


def _streaming_entries() -> list[dict[str, Any]]:
    common = {
        "sources": ["infer.request chunk_interarrival_ms, stream, ttft_ms"],
        "computed_by": "latency_report.streaming_summary",
        "applies_to": "successful streamed responses with a TTFT",
        "notes": [CHUNK_NOTE],
        "missing_rule": "a gap that is not a finite number is left out",
    }
    return [
        _entry(
            "streaming.chunk_interarrival_ms",
            "gap between content chunks: p50, p95, max",
            "ms",
            formula=QUANTILE_RULE,
            **common,
        ),
        _entry(
            "streaming.content_chunks_per_response",
            "content chunks per response: p50, p95, max",
            "chunks",
            **common,
        ),
        _entry(
            "streaming.mean_tokens_per_chunk",
            "server-reported output tokens per chunk",
            "tokens/chunk",
            formula="sum of tokens / sum of chunks, server_usage counts only",
            **common,
        ),
    ]


def _token_entries() -> list[dict[str, Any]]:
    entries = []
    for kind in ("output", "prompt"):
        for name, label in (
            ("eligible", "successful requests"),
            ("known", "successful requests with a numeric count"),
            ("exact", "known counts flagged exact"),
            ("known_sum", "sum of the known counts"),
        ):
            entries.append(
                _entry(
                    f"tokens.{kind}.{name}",
                    f"{kind} tokens: {label}",
                    "tokens" if name == "known_sum" else "requests",
                    sources=[
                        f"infer.request {kind}_tokens, {kind}_token_source, {kind}_token_exact"
                    ],
                    computed_by="token_report.token_coverage",
                    applies_to="successful requests",
                    missing_rule="a count that is not a number is not known; it is never zero",
                    collector=_COLLECTORS["tokens"],
                )
            )
    return entries


def _memory_entries() -> list[dict[str, Any]]:
    return [
        _entry(
            f"client_memory.peak_{field}",
            f"peak {field.replace('_', ' ')} per sampler and device",
            "bytes",
            scope="client_local_host",
            sources=[f"infer.system_sample {field}"],
            computed_by="sample_report.sample_peaks",
            applies_to="system samples of the client's host, by (sampler, device_id)",
            aggregation_window=SAMPLE_WINDOW,
            notes=["the machine running Stormlog's client, not the server"],
            collector=_COLLECTORS["client_memory"],
        )
        for field in ("process_rss_bytes", "device_used_bytes")
    ] + [
        _entry(
            "samples.gaps",
            "where one series of system samples breaks",
            "ns",
            scope="client_local_host",
            sources=["infer.system_sample timestamp_ns"],
            computed_by="sample_report.sample_gaps",
            formula=(
                "consecutive samples more than SCRAPE_GAP_FACTOR "
                f"({SCRAPE_GAP_FACTOR:g}) times the series' median interval apart"
            ),
            missing_rule="none when the median interval is 0 or under two samples",
            applies_to="one series of system samples, by (sampler, device_id)",
            aggregation_window="the samples route's window",
            notes=[
                "the series' own median interval stands in for a configured "
                "sampling interval, which the artifact does not record"
            ],
            collector=_COLLECTORS["client_memory"],
        )
    ]


def _request_entries() -> list[dict[str, Any]]:
    return [
        _entry(
            "request.elapsed_ms",
            "how long a request that did not succeed ran",
            "ms",
            sources=["infer.request e2e_latency_ms, started_at_ns, ended_at_ns"],
            computed_by="latency_report.elapsed_ms",
            applies_to="one request whose status is not ok",
            aggregation_window="one request",
            notes=["only a timeout's is a lower bound on a latency"],
        ),
        _entry(
            "request.gaps",
            "one response's chunk gaps: p50, p95, max",
            "ms",
            sources=["infer.request chunk_interarrival_ms"],
            computed_by="latency_report.spread",
            applies_to="one request",
            aggregation_window="one request",
            notes=[CHUNK_NOTE],
        ),
    ]


# --- Panels -------------------------------------------------------------------------


def panels(snapshot: IndexSnapshot, analysis: AnalysisState) -> dict[str, Any]:
    """The ``artifact_capabilities`` body: each panel's state and reason."""
    gate = _gate(snapshot)
    states = {
        name: gate if gate is not None else _panel(snapshot, name)
        for name in PANELS
        if name != "analysis"
    }
    states["analysis"] = _analysis_panel(analysis)
    return {
        "panels": states,
        "observers": _observers(analysis),
        "observers_source": "infer analyze (out of process)",
    }


def _state(state: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "state": state,
        "reason": reason,
        "collector": extra.get("collector"),
        "detail": extra.get("detail"),
    }


def _gate(snapshot: IndexSnapshot) -> dict[str, Any] | None:
    """A state every index-backed panel shares, before looking at the data."""
    if snapshot.status == "building":
        return _state("pending", "index_building")
    if snapshot.failure == "audio":
        return _state("unsupported", "audio_artifact")
    if snapshot.status == "failed":
        return _state("unavailable", snapshot.failure or "index_failed")
    return None


def _panel(snapshot: IndexSnapshot, name: str) -> dict[str, Any]:
    if name in _LATER:
        return _later(snapshot, name)
    judged = _JUDGES[name](snapshot)
    if judged["state"] == "available" and _partial_reason(snapshot) is not None:
        return _state("partial", _partial_reason(snapshot), detail=judged["detail"])
    return judged


def _partial_reason(snapshot: IndexSnapshot) -> str | None:
    """Why the numbers cover only a prefix: the index budget, or a growing file."""
    if snapshot.status == "partial" or "rows" in snapshot.limits_hit:
        return "index_budget"
    if snapshot.growing or (
        snapshot.session is not None and not snapshot.session.terminal
    ):
        return "growing"
    return None


def _statuses(snapshot: IndexSnapshot) -> Sequence[Any]:
    return snapshot.column("status") if len(snapshot.table) else ()


def _requests_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    if not len(snapshot.table):
        return _state("empty", "no_client_requests")
    return _state("available", detail={"rows": len(snapshot.table)})


def _latency_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    if not any(status == "ok" for status in _statuses(snapshot)):
        return _state("empty", "no_successful_requests")
    return _state("available")


def _failures_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    if not any(status != "ok" for status in _statuses(snapshot)):
        return _state("empty", "no_failed_requests")
    return _state("available")


def _streaming_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    if not snapshot.gaps_resident:
        return _state("unavailable", "gaps_not_resident")
    if not len(snapshot.table):
        return _state("empty", "no_streamed_responses")
    streams, ttfts = snapshot.column("stream"), snapshot.column("ttft_ms")
    statuses = _statuses(snapshot)
    streamed = any(
        statuses[i] == "ok" and streams[i] and is_number(ttfts[i])
        for i in range(len(snapshot.table))
    )
    if not streamed:
        return _state("empty", "no_streamed_responses")
    return _state("available")


def _tokens_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    if not any(status == "ok" for status in _statuses(snapshot)):
        return _state(
            "empty", "no_successful_requests", collector=_COLLECTORS["tokens"]
        )
    return _state("available")


def _rates_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    facts = case_facts(snapshot)
    if facts.missing_kept:
        return _state(
            "unavailable",
            "kept_record_over_limit",
            detail={"record_types": list(facts.missing_kept)},
        )
    if not facts.populations:
        return _state("empty", "no_measured_cases")
    reasons = {
        case: population.intervals.rate_reason
        for case, population in facts.populations.items()
        if population.intervals.rate is None
    }
    if len(reasons) == len(facts.populations):
        first = next(iter(reasons.values()))
        return _state(
            "unavailable", first or "no_rate_interval", detail={"cases": reasons}
        )
    return _state(
        "available", detail={"cases_without_rate": reasons} if reasons else None
    )


def _cases_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    facts = case_facts(snapshot)
    if not facts.by_case and not facts.planned:
        return _state("empty", "no_measured_cases")
    return _state(
        "available",
        detail={"observed": len(facts.by_case), "planned": len(facts.planned)},
    )


def _memory_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    if not snapshot.samples:
        return _state(
            "unavailable", "no_system_samples", collector=_COLLECTORS["client_memory"]
        )
    if "samples" in snapshot.limits_hit:
        return _state(
            "partial", "index_budget", detail={"samples": len(snapshot.samples)}
        )
    return _state("available", detail={"samples": len(snapshot.samples)})


def _server_panel(snapshot: IndexSnapshot) -> dict[str, Any]:
    held = _held(snapshot, SERVER_EVIDENCE_TYPES)
    server_view = snapshot.inventory.get("infer.request", {})
    other_versions = sum(
        count for version, count in server_view.items() if version != 1
    )
    # Client requests at another version are no server evidence: C1 counts
    # them as unsupported_version problems, apart (§4.4, Fable's F-9).
    unsupported = snapshot.problem_counts.get("unsupported_version", 0)
    server_requests = max(0, other_versions - unsupported)
    if server_requests:
        held["infer.request (server view)"] = server_requests
    if not held:
        return _state(
            "unavailable",
            "no_server_evidence",
            collector=_COLLECTORS["server_evidence"],
        )
    return _state(
        "not_indexed",
        "server_evidence",
        detail={"records": held, "see": "the analyzer report in Sources"},
    )


def _later(snapshot: IndexSnapshot, name: str) -> dict[str, Any]:
    held = _held(snapshot, _LATER[name])
    if not held:
        return _state("unavailable", f"no_{name}", collector=_COLLECTORS[name])
    return _state("in_artifact_panel_later", "panel_later", detail={"records": held})


def _held(snapshot: IndexSnapshot, event_types: Sequence[str]) -> dict[str, int]:
    held = {}
    for event_type in event_types:
        count = sum(snapshot.inventory.get(event_type, {}).values())
        if count:
            held[event_type] = count
    return held


_JUDGES = {
    "requests": _requests_panel,
    "latency": _latency_panel,
    "failures": _failures_panel,
    "streaming": _streaming_panel,
    "tokens": _tokens_panel,
    "rates": _rates_panel,
    "cases": _cases_panel,
    "client_memory": _memory_panel,
    "server_evidence": _server_panel,
}


def _analysis_panel(analysis: AnalysisState) -> dict[str, Any]:
    if analysis.status == "ready":
        return _state("available", detail={"analysed_bytes": analysis.analysed_bytes})
    if analysis.status in ("pending", "running"):
        return _state("pending", f"analysis_{analysis.status}")
    return _state(
        "unavailable",
        analysis.reason or analysis.status,
        detail={"status": analysis.status, "exit_code": analysis.exit_code},
    )


def _observers(analysis: AnalysisState) -> Mapping[str, Any] | None:
    if analysis.status != "ready" or analysis.report is None:
        return None
    observers = analysis.report.get("observers")
    return observers if isinstance(observers, Mapping) else None
