"""Prometheus families for an inference profile, and how its records map onto them.

Every family here is something Stormlog measured or decided. Nothing scraped
from vLLM's ``/metrics`` or received from its span exporter is re-exposed,
and no engine quantity is re-aggregated: Prometheus can scrape vLLM itself,
and a copy would count the same events twice. Where a Stormlog metric
overlaps an engine one in meaning (Stormlog's client-side latency against
vLLM's engine-side histograms), it has its own name, and the docs say never
to add the two.

The work is split in two. ``envelope`` runs on the producer's thread and
copies a fixed list of fields into an envelope; ``apply`` runs on the
metric worker, inside the registry's lock, and updates the families.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._export.envelope import Envelope, EnvelopeLimits, Value, make_envelope
from .._export.registry import Family, FamilySpec, Registry
from .events import REQUEST_PHASES, REQUEST_STATUSES
from .tokens import TOKEN_SOURCES

# Upper bounds, in seconds. Fixed, and listed in docs/inference_export.md.
# Request latency: fine from 5 s to 2 min, where LLM requests live, and up
# to 10 min, so a long generation still lands below +Inf.
LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0,
    120.0, 180.0, 300.0, 600.0,
)  # fmt: skip
FIRST_TOKEN_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)  # fmt: skip
CHUNK_BUCKETS = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.5, 5.0,
    10.0, 30.0,
)  # fmt: skip
DISPATCH_LAG_BUCKETS = (
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
)  # fmt: skip
SCRAPE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

ALL_CASES = "all"
CLOSED_LOOP = "closed"
DIRECTIONS = ("prompt", "output")
# A token count a counter can add exactly; the server's usage is checked
# against it, since a negative count would make a counter go down.
MAX_TOKEN_COUNT = 2**53
SCRAPE_OUTCOMES = ("ok", "error")
# How a trace window ended; trace_capture.py writes the first five.
TRACE_STOP_REASONS = (
    "phase_end",
    "time_bound",
    "cancelled",
    "stopped_by_server",
    "start_unknown",
    "not_started",
)
BOOLEANS = ("true", "false")
REQUEST_LIMITS = EnvelopeLimits(max_fields=24, max_string=128)


@dataclass(frozen=True)
class ProfileLabels:
    """The configuration-bounded label values of one profile run."""

    model: str
    server: str
    # Each case's ID and arrival mode, in configuration order.
    cases: tuple[tuple[str, str], ...]
    run_id: str
    session_id: str
    version: str
    command: str = "infer profile"
    # The redacted origin of the metrics endpoint scraped, when scraping.
    metrics_server: str | None = None
    traces: bool = False
    # With the case label off, every series has case="all".
    case_label: bool = True

    def case(self, case_id: str) -> str:
        return case_id if self.case_label else ALL_CASES

    def case_values(self) -> list[str]:
        return list(dict.fromkeys(self.case(case_id) for case_id, _ in self.cases))

    def case_modes(self) -> list[tuple[str, str]]:
        return list(dict.fromkeys((self.case(case), mode) for case, mode in self.cases))


def summarize_chunk_gaps(gaps_ms: Sequence[float]) -> tuple[tuple[int, ...], float]:
    """Counts per ``CHUNK_BUCKETS`` bucket (the last is +Inf), and the sum in s.

    Run on the request's own thread, where the gaps were measured, so the
    producer copies a fixed-size summary instead of walking the list.
    """
    counts = [0] * (len(CHUNK_BUCKETS) + 1)
    total = 0.0
    for gap_ms in gaps_ms:
        seconds = gap_ms / 1000.0
        if not math.isfinite(seconds):
            continue
        counts[bisect.bisect_left(CHUNK_BUCKETS, seconds)] += 1
        total += seconds
    return tuple(counts), total


@dataclass
class _Families:
    requests: Family
    duration: Family
    first_token: Family
    first_chunk: Family
    chunk_gaps: Family
    from_intended: Family
    dispatch_lag: Family
    tokens: Family
    held: Family
    phases: Family
    abandoned: Family
    run_info: Family
    run_start: Family
    scrapes: Family | None = None
    scrape_duration: Family | None = None
    last_scrape: Family | None = None
    source_changes: Family | None = None
    trace_windows: Family | None = None


@dataclass
class ProfileMetrics:
    """Declares a profile's families and maps its records onto them."""

    registry: Registry
    labels: ProfileLabels
    _families: _Families = field(init=False)
    # The metrics exporter's start time per server, from the last good scrape.
    _process_start: dict[str, float] = field(init=False, default_factory=dict)
    # Token counts no counter can take (negative, or too large to add
    # exactly): the token totals then differ from the artifact's.
    tokens_rejected: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._families = _declare(self.registry, self.labels)
        self._builders: dict[str, Callable[[dict[str, Any], Mapping | None], Envelope]]
        self._builders = {
            "infer.request": _request_envelope,
            "infer.phase_window": _phase_envelope,
            "infer.vllm_scrape": _scrape_envelope,
            "infer.trace_window": _trace_window_envelope,
        }

    def set_run_info(self, started_at: float) -> None:
        labels = self.labels
        families = self._families
        families.run_info.set(
            (labels.run_id, labels.session_id, labels.version, labels.command), 1.0
        )
        families.run_start.set((), started_at)

    def envelope(
        self, record: dict[str, Any], extras: Mapping[str, Any] | None
    ) -> Envelope | None:
        """The envelope for a record this catalog maps, or None. Producer side."""
        builder = self._builders.get(str(record.get("event_type")))
        return builder(record, extras) if builder is not None else None

    def apply(self, envelope: Envelope) -> None:
        """Update the families from one envelope. Inside ``Registry.apply``."""
        fields = envelope.as_dict()
        if envelope.kind == "request":
            self._apply_request(fields)
        elif envelope.kind == "phase":
            self._apply_phase(fields)
        elif envelope.kind == "scrape":
            self._apply_scrape(fields)
        elif envelope.kind == "trace_window":
            self._apply_trace_window(fields)

    # ------------------------------------------------------------- appliers
    def _apply_request(self, fields: Mapping[str, Value]) -> None:
        families = self._families
        labels = self.labels
        case = labels.case(str(fields.get("case")))
        base = (labels.model, labels.server, case, str(fields.get("phase")))
        status = str(fields.get("status"))
        families.requests.inc(base + (status,))
        if status == "ok":
            self._apply_completed(base, fields)
        lag = fields.get("lag_ms")
        if isinstance(lag, (int, float)):
            families.dispatch_lag.observe((case, str(fields.get("arrival"))), lag / 1e3)
        if fields.get("held") is True:
            families.held.inc((case,))

    def _apply_completed(
        self, base: tuple[str, ...], fields: Mapping[str, Value]
    ) -> None:
        families = self._families
        for family, key in (
            (families.duration, "e2e_ms"),
            (families.first_token, "ttft_ms"),
            (families.first_chunk, "first_chunk_ms"),
        ):
            value = fields.get(key)
            if isinstance(value, (int, float)):
                family.observe(base, value / 1e3)
        self._apply_from_intended(base, fields)
        self._apply_tokens(base, fields)
        counts, total = fields.get("chunk_counts"), fields.get("chunk_sum")
        if isinstance(counts, tuple) and isinstance(total, (int, float)):
            families.chunk_gaps.observe_counts(
                base, [int(c or 0) for c in counts], total
            )

    def _apply_tokens(self, base: tuple[str, ...], fields: Mapping[str, Value]) -> None:
        tokens = self._families.tokens
        for direction in DIRECTIONS:
            count = fields.get(f"{direction}_tokens")
            if count is None:
                continue
            if (
                isinstance(count, int)
                and not isinstance(count, bool)
                and 0 <= count < MAX_TOKEN_COUNT
            ):
                source = str(fields.get(f"{direction}_source"))
                tokens.inc(base + (direction, source), count)
            else:
                self.tokens_rejected += 1

    def _apply_from_intended(
        self, base: tuple[str, ...], fields: Mapping[str, Value]
    ) -> None:
        # Open-loop only: a closed loop's arrival is its send.
        intended, ended = fields.get("intended_ns"), fields.get("ended_ns")
        if fields.get("arrival") == CLOSED_LOOP:
            return
        if isinstance(intended, int) and isinstance(ended, int):
            self._families.from_intended.observe(base, (ended - intended) / 1e9)

    def _apply_phase(self, fields: Mapping[str, Value]) -> None:
        families = self._families
        key = (self.labels.case(str(fields.get("case"))), str(fields.get("phase")))
        families.phases.inc(key)
        abandoned = fields.get("abandoned")
        if isinstance(abandoned, int) and abandoned > 0:
            families.abandoned.inc(key, abandoned)

    def _apply_scrape(self, fields: Mapping[str, Value]) -> None:
        families = self._families
        server = self.labels.metrics_server
        if server is None or families.scrapes is None:
            return
        assert families.scrape_duration and families.last_scrape
        assert families.source_changes
        outcome = str(fields.get("status"))
        families.scrapes.inc((server, outcome))
        duration = fields.get("duration_ms")
        if isinstance(duration, (int, float)):
            families.scrape_duration.observe((server,), duration / 1e3)
        observed = fields.get("observed_ns")
        if isinstance(observed, int):
            families.last_scrape.set((server, outcome), observed / 1e9)
        start = fields.get("process_start")
        if isinstance(start, float):
            previous = self._process_start.get(server)
            if previous is not None and previous != start:
                families.source_changes.inc((server,))
            self._process_start[server] = start

    def _apply_trace_window(self, fields: Mapping[str, Value]) -> None:
        if self._families.trace_windows is None:
            return
        started = "true" if fields.get("started") else "false"
        self._families.trace_windows.inc((str(fields.get("stop_reason")), started))


# ------------------------------------------------------------------ envelopes
def _request_envelope(
    record: dict[str, Any], extras: Mapping[str, Any] | None
) -> Envelope:
    summary = (extras or {}).get("chunk_summary")
    counts, total = summary if isinstance(summary, tuple) else ((), None)
    return make_envelope(
        "request",
        [
            ("case", record.get("case_id")),
            ("phase", record.get("phase")),
            ("status", record.get("status")),
            ("e2e_ms", record.get("e2e_latency_ms")),
            ("ttft_ms", record.get("ttft_ms")),
            ("first_chunk_ms", record.get("first_chunk_latency_ms")),
            ("intended_ns", record.get("intended_at_ns")),
            ("ended_ns", record.get("ended_at_ns")),
            ("arrival", record.get("arrival_mode")),
            ("lag_ms", record.get("dispatch_lag_ms")),
            ("held", record.get("held_for_slot")),
            ("prompt_tokens", record.get("prompt_tokens")),
            ("prompt_source", record.get("prompt_token_source")),
            ("output_tokens", record.get("output_tokens")),
            ("output_source", record.get("output_token_source")),
            ("chunk_counts", tuple(counts) if counts else None),
            ("chunk_sum", total),
        ],
        REQUEST_LIMITS,
    )


def _phase_envelope(record: dict[str, Any], _extras: Mapping | None) -> Envelope:
    abandoned = record.get("abandoned_requests")
    running = abandoned.get("running_at_start") if isinstance(abandoned, dict) else None
    return make_envelope(
        "phase",
        [
            ("case", record.get("case_id")),
            ("phase", record.get("phase")),
            ("abandoned", running),
        ],
        REQUEST_LIMITS,
    )


def _scrape_envelope(record: dict[str, Any], _extras: Mapping | None) -> Envelope:
    return make_envelope(
        "scrape",
        [
            ("status", record.get("status")),
            ("duration_ms", record.get("duration_ms")),
            ("observed_ns", record.get("observed_at_ns")),
            ("process_start", _process_start(record)),
        ],
        REQUEST_LIMITS,
    )


def _process_start(record: dict[str, Any]) -> float | None:
    """The exporter's start time from a scrape record, by two dict lookups."""
    scrape = record.get("scrape")
    values = scrape.get("values") if isinstance(scrape, dict) else None
    series = (
        values.get("process_start_time_seconds") if isinstance(values, dict) else None
    )
    if not isinstance(series, dict) or len(series) != 1:
        return None
    value = next(iter(series.values()))
    return float(value) if isinstance(value, (int, float)) else None


def _trace_window_envelope(record: dict[str, Any], _extras: Mapping | None) -> Envelope:
    return make_envelope(
        "trace_window",
        [
            ("stop_reason", record.get("stop_reason") or "not_started"),
            ("started", bool(record.get("started"))),
        ],
        REQUEST_LIMITS,
    )


# ------------------------------------------------------------------ catalog
def _client_spec(name: str, help_text: str, buckets: tuple[float, ...]) -> FamilySpec:
    return FamilySpec(
        name,
        "histogram",
        help_text,
        unit="seconds",
        labels=("model", "server", "case", "phase"),
        enums={"phase": REQUEST_PHASES},
        buckets=buckets,
    )


def _client_known(labels: ProfileLabels) -> list[dict[str, str]]:
    return [
        {"model": labels.model, "server": labels.server, "case": case}
        for case in labels.case_values()
    ]


def _declare(registry: Registry, labels: ProfileLabels) -> _Families:
    known = _client_known(labels)
    cases = [{"case": case} for case in labels.case_values()]
    families = _Families(
        requests=registry.add(
            FamilySpec(
                "stormlog_infer_requests_total",
                "counter",
                "Requests by outcome, as the client saw them. Never add to "
                "vllm:request_success_total, which counts every client.",
                labels=("model", "server", "case", "phase", "status"),
                enums={"phase": REQUEST_PHASES, "status": REQUEST_STATUSES},
            ),
            known,
        ),
        duration=registry.add(
            _client_spec(
                "stormlog_infer_request_duration_seconds",
                "Client-observed end-to-end latency of completed requests. Never "
                "add to vllm:e2e_request_latency_seconds, the engine's residency.",
                LATENCY_BUCKETS,
            ),
            known,
        ),
        first_token=registry.add(
            _client_spec(
                "stormlog_infer_time_to_first_token_seconds",
                "Client-observed time to the first non-empty content delta. Never "
                "add to vllm:time_to_first_token_seconds.",
                FIRST_TOKEN_BUCKETS,
            ),
            known,
        ),
        first_chunk=registry.add(
            _client_spec(
                "stormlog_infer_time_to_first_chunk_seconds",
                "Client-observed time to the first streamed chunk.",
                FIRST_TOKEN_BUCKETS,
            ),
            known,
        ),
        chunk_gaps=registry.add(
            _client_spec(
                "stormlog_infer_chunk_interarrival_seconds",
                "Gaps between streamed chunks; chunk timing, not token ITL. Never "
                "add to vllm:inter_token_latency_seconds.",
                CHUNK_BUCKETS,
            ),
            known,
        ),
        from_intended=registry.add(
            _client_spec(
                "stormlog_infer_e2e_from_intended_seconds",
                "Open-loop latency of completed requests from their intended "
                "arrival, so a held arrival's wait is included.",
                LATENCY_BUCKETS,
            ),
            known,
        ),
        dispatch_lag=registry.add(
            FamilySpec(
                "stormlog_infer_dispatch_lag_seconds",
                help="How late each request was sent against its intended arrival.",
                unit="seconds",
                kind="histogram",
                labels=("case", "arrival_mode"),
                buckets=DISPATCH_LAG_BUCKETS,
            ),
            [
                {"case": case, "arrival_mode": mode}
                for case, mode in labels.case_modes()
            ],
        ),
        tokens=registry.add(
            FamilySpec(
                "stormlog_infer_tokens_total",
                help="Prompt and output tokens of completed requests, by the source "
                "of each count. Never add to vllm:prompt_tokens_total or "
                "vllm:generation_tokens_total, which count every client.",
                labels=("model", "server", "case", "phase", "direction", "source"),
                enums={
                    "phase": REQUEST_PHASES,
                    "direction": DIRECTIONS,
                    "source": TOKEN_SOURCES,
                },
                kind="counter",
            ),
            known,
        ),
        held=registry.add(
            FamilySpec(
                "stormlog_infer_requests_held_for_slot_total",
                help="Open-loop arrivals that found every in-flight slot taken: "
                "they waited for one, or with --overflow drop were dropped.",
                kind="counter",
                labels=("case",),
            ),
            cases,
        ),
        phases=registry.add(
            FamilySpec(
                "stormlog_infer_phases_total",
                help="Completed phase windows.",
                kind="counter",
                labels=("case", "phase"),
                enums={"phase": REQUEST_PHASES},
            ),
            cases,
        ),
        abandoned=registry.add(
            FamilySpec(
                "stormlog_infer_abandoned_requests_total",
                help="Requests from an earlier drain still running when a phase was "
                "ready to start.",
                kind="counter",
                labels=("case", "phase"),
                enums={"phase": REQUEST_PHASES},
            ),
            cases,
        ),
        run_info=registry.add(
            FamilySpec(
                "stormlog_run_info",
                help="1 for the run this producer is exporting.",
                kind="gauge",
                labels=("run_id", "session_id", "version", "command"),
            ),
            [
                {
                    "run_id": labels.run_id,
                    "session_id": labels.session_id,
                    "version": labels.version,
                    "command": labels.command,
                }
            ],
        ),
        run_start=registry.add(
            FamilySpec(
                "stormlog_run_start_time_seconds",
                help="When the run started, in Unix seconds.",
                kind="gauge",
                unit="seconds",
            )
        ),
    )
    _declare_engine(registry, labels, families)
    if labels.traces:
        families.trace_windows = registry.add(
            FamilySpec(
                "stormlog_trace_windows_total",
                help="Profiler trace windows by how they ended.",
                kind="counter",
                labels=("stop_reason", "started"),
                enums={"stop_reason": TRACE_STOP_REASONS, "started": BOOLEANS},
            )
        )
    return families


def _declare_engine(
    registry: Registry, labels: ProfileLabels, families: _Families
) -> None:
    server = labels.metrics_server
    if server is None:
        return
    known = [{"server": server}]
    families.scrapes = registry.add(
        FamilySpec(
            "stormlog_engine_scrapes_total",
            help="Scrapes of the engine's metrics endpoint by outcome.",
            kind="counter",
            labels=("server", "outcome"),
            enums={"outcome": SCRAPE_OUTCOMES},
        ),
        known,
    )
    families.scrape_duration = registry.add(
        FamilySpec(
            "stormlog_engine_scrape_duration_seconds",
            help="How long each scrape of the engine's metrics took.",
            kind="histogram",
            unit="seconds",
            labels=("server",),
            buckets=SCRAPE_BUCKETS,
        ),
        known,
    )
    families.last_scrape = registry.add(
        FamilySpec(
            "stormlog_engine_last_scrape_timestamp_seconds",
            help="When the engine was last scraped, by outcome, in Unix seconds.",
            kind="gauge",
            unit="seconds",
            labels=("server", "outcome"),
            enums={"outcome": SCRAPE_OUTCOMES},
        ),
        known,
        # Absent until a scrape has that outcome: a 0 would read as 1970.
        precreate=False,
    )
    families.source_changes = registry.add(
        FamilySpec(
            "stormlog_engine_metrics_source_changes_total",
            help="Times the process answering the metrics endpoint changed: an "
            "engine restart, or a load balancer reaching another process.",
            kind="counter",
            labels=("server",),
        ),
        known,
    )
