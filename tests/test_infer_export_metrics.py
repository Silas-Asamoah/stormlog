"""How an inference profile's records become Prometheus metrics."""

from collections.abc import Mapping
from typing import Any

import pytest

from stormlog._export.registry import DEFAULT_MAX_SAMPLES, Registry, render
from stormlog.infer.events import REQUEST_PHASES, REQUEST_STATUSES
from stormlog.infer.export_metrics import (
    ALL_CASES,
    CHUNK_BUCKETS,
    REQUEST_LIMITS,
    ProfileLabels,
    ProfileMetrics,
    summarize_chunk_gaps,
)
from tests.export_conformance import Exposition, check_exposition

SERVER = "http://127.0.0.1:8000"
CASES = (("c4_in8_out8", "closed"), ("poisson2_in8_out8", "poisson"))


def _metrics(**overrides: Any) -> tuple[Registry, ProfileMetrics]:
    settings: dict[str, Any] = {
        "model": "m",
        "server": SERVER,
        "cases": CASES,
        "run_id": "run-1",
        "session_id": "session-1",
        "version": "0.0.0",
    }
    settings.update(overrides)
    registry = Registry(headroom=0, const_labels={"stormlog_producer": "p"})
    return registry, ProfileMetrics(registry, ProfileLabels(**settings))


def _feed(
    registry: Registry,
    metrics: ProfileMetrics,
    record: dict[str, Any],
    extras: Mapping[str, Any] | None = None,
) -> None:
    envelope = metrics.envelope(record, extras)
    assert envelope is not None
    assert registry.apply(lambda: metrics.apply(envelope))


def _exposition(registry: Registry) -> Exposition:
    return check_exposition(render(registry.snapshot()).decode())


def _request(**fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_type": "infer.request",
        "case_id": "poisson2_in8_out8",
        "phase": "measured",
        "status": "ok",
        "e2e_latency_ms": 250.0,
        "ttft_ms": 40.0,
        "first_chunk_latency_ms": 35.0,
        "intended_at_ns": 1_000_000_000,
        "ended_at_ns": 1_400_000_000,
        "arrival_mode": "poisson",
        "dispatch_lag_ms": 2.0,
        "held_for_slot": True,
        "prompt_tokens": 8,
        "prompt_token_source": "server_usage",
        "output_tokens": 5,
        "output_token_source": "estimated",
        "chunk_interarrival_ms": [10.0, 20.0],
    }
    record.update(fields)
    return record


def test_every_configured_series_exists_before_any_request() -> None:
    registry, _ = _metrics()
    exposition = _exposition(registry)
    requests = exposition.matching("stormlog_infer_requests_total")
    assert len(requests) == len(CASES) * len(REQUEST_PHASES) * len(REQUEST_STATUSES)
    assert set(requests) == {0.0}


def test_a_completed_request_fills_every_client_family() -> None:
    registry, metrics = _metrics()
    _feed(
        registry,
        metrics,
        _request(),
        {"chunk_summary": summarize_chunk_gaps([10.0, 20.0])},
    )
    exposition = _exposition(registry)
    case = {"case": "poisson2_in8_out8", "phase": "measured"}
    assert exposition.value("stormlog_infer_requests_total", status="ok", **case) == 1
    assert (
        exposition.value("stormlog_infer_request_duration_seconds_sum", **case) == 0.25
    )
    assert (
        exposition.value("stormlog_infer_time_to_first_token_seconds_sum", **case)
        == 0.04
    )
    assert (
        exposition.value("stormlog_infer_time_to_first_chunk_seconds_sum", **case)
        == 0.035
    )
    assert (
        exposition.value("stormlog_infer_e2e_from_intended_seconds_sum", **case) == 0.4
    )
    assert (
        exposition.value("stormlog_infer_chunk_interarrival_seconds_count", **case) == 2
    )
    assert exposition.value(
        "stormlog_infer_chunk_interarrival_seconds_sum", **case
    ) == pytest.approx(0.03)
    tokens = "stormlog_infer_tokens_total"
    assert (
        exposition.value(tokens, direction="prompt", source="server_usage", **case) == 8
    )
    assert exposition.value(tokens, direction="output", source="estimated", **case) == 5
    assert (
        exposition.value(
            "stormlog_infer_dispatch_lag_seconds_sum",
            case="poisson2_in8_out8",
            arrival_mode="poisson",
        )
        == 0.002
    )
    assert (
        exposition.value(
            "stormlog_infer_requests_held_for_slot_total", case="poisson2_in8_out8"
        )
        == 1
    )


def test_a_closed_loop_request_has_no_latency_from_intended() -> None:
    registry, metrics = _metrics()
    _feed(registry, metrics, _request(case_id="c4_in8_out8", arrival_mode="closed"))
    exposition = _exposition(registry)
    assert (
        exposition.value(
            "stormlog_infer_e2e_from_intended_seconds_count",
            case="c4_in8_out8",
            phase="measured",
        )
        == 0
    )


def test_a_failed_request_counts_its_status_and_nothing_measured() -> None:
    registry, metrics = _metrics()
    _feed(registry, metrics, _request(status="timeout", ttft_ms=None))
    exposition = _exposition(registry)
    case = {"case": "poisson2_in8_out8", "phase": "measured"}
    assert (
        exposition.value("stormlog_infer_requests_total", status="timeout", **case) == 1
    )
    assert (
        exposition.value("stormlog_infer_request_duration_seconds_count", **case) == 0
    )
    assert sum(exposition.matching("stormlog_infer_tokens_total")) == 0


def test_an_unlisted_status_is_rejected_and_counted_not_invented() -> None:
    registry, metrics = _metrics()
    _feed(registry, metrics, _request(status="teleported"))
    assert sum(_exposition(registry).matching("stormlog_infer_requests_total")) == 0
    family = next(
        f for f in registry.families if f.spec.name == "stormlog_infer_requests_total"
    )
    assert family.stats.rejected == 1


@pytest.mark.parametrize("count", [-1_000_000, 2**53, 10**400, True])
def test_a_token_count_no_counter_can_take_is_rejected(count: object) -> None:
    # Usage counts come from the server: a negative one would make a counter
    # go down, and a huge one would not add exactly.
    registry, metrics = _metrics()
    _feed(registry, metrics, _request(prompt_tokens=count, output_tokens=7))
    exposition = _exposition(registry)
    case = {"case": "poisson2_in8_out8", "phase": "measured"}
    tokens = "stormlog_infer_tokens_total"
    assert (
        exposition.value(tokens, direction="prompt", source="server_usage", **case) == 0
    )
    assert exposition.value(tokens, direction="output", source="estimated", **case) == 7
    assert exposition.value("stormlog_infer_requests_total", status="ok", **case) == 1
    family = next(f for f in registry.families if f.spec.name == tokens)
    assert family.stats.rejected == 1


def test_with_the_case_label_off_every_case_is_all() -> None:
    registry, metrics = _metrics(case_label=False)
    _feed(registry, metrics, _request())
    _feed(registry, metrics, _request(case_id="c4_in8_out8", arrival_mode="closed"))
    exposition = _exposition(registry)
    cases = {
        dict(labels)["case"]
        for (name, labels) in exposition.samples
        if name == "stormlog_infer_requests_total"
    }
    assert cases == {ALL_CASES}
    assert (
        exposition.value("stormlog_infer_requests_total", status="ok", phase="measured")
        == 2
    )


def test_phase_windows_count_phases_and_abandoned_requests() -> None:
    registry, metrics = _metrics()
    _feed(
        registry,
        metrics,
        {
            "event_type": "infer.phase_window",
            "case_id": "c4_in8_out8",
            "phase": "measured",
            "abandoned_requests": {"running_at_start": 3},
        },
    )
    exposition = _exposition(registry)
    labels = {"case": "c4_in8_out8", "phase": "measured"}
    assert exposition.value("stormlog_infer_phases_total", **labels) == 1
    assert exposition.value("stormlog_infer_abandoned_requests_total", **labels) == 3


def _scrape(status: str, start: float | None, observed: int) -> dict[str, Any]:
    values = {"process_start_time_seconds": {"0": start}} if start is not None else {}
    return {
        "event_type": "infer.vllm_scrape",
        "status": status,
        "duration_ms": 12.0,
        "observed_at_ns": observed,
        "scrape": {"values": values} if status == "ok" else None,
    }


def test_scrapes_report_health_and_count_metrics_source_changes() -> None:
    registry, metrics = _metrics(metrics_server=SERVER)
    for record in (
        _scrape("ok", 100.0, 1_000_000_000),
        _scrape("ok", 100.0, 2_000_000_000),
        _scrape("error", None, 3_000_000_000),
        _scrape("ok", 200.0, 4_000_000_000),
    ):
        _feed(registry, metrics, record)
    exposition = _exposition(registry)
    assert exposition.value("stormlog_engine_scrapes_total", outcome="ok") == 3
    assert exposition.value("stormlog_engine_scrapes_total", outcome="error") == 1
    assert exposition.value("stormlog_engine_metrics_source_changes_total") == 1
    assert (
        exposition.value("stormlog_engine_last_scrape_timestamp_seconds", outcome="ok")
        == 4.0
    )
    assert exposition.value("stormlog_engine_scrape_duration_seconds_count") == 4


def test_trace_windows_count_how_they_ended() -> None:
    registry, metrics = _metrics(traces=True)
    _feed(
        registry,
        metrics,
        {
            "event_type": "infer.trace_window",
            "stop_reason": "time_bound",
            "started": True,
        },
    )
    _feed(
        registry,
        metrics,
        {"event_type": "infer.trace_window", "stop_reason": None, "started": False},
    )
    exposition = _exposition(registry)
    windows = "stormlog_trace_windows_total"
    assert exposition.value(windows, stop_reason="time_bound", started="true") == 1
    assert exposition.value(windows, stop_reason="not_started", started="false") == 1


def test_engine_and_trace_families_exist_only_when_collected() -> None:
    registry, _ = _metrics()
    names = {family.spec.name for family in registry.families}
    assert "stormlog_engine_scrapes_total" not in names
    assert "stormlog_trace_windows_total" not in names


@pytest.mark.parametrize(
    "event_type",
    ["infer.vllm_span", "infer.activity_ref", "infer.system_sample", "infer.session"],
)
def test_ingested_and_unexported_records_map_to_nothing(event_type: str) -> None:
    _, metrics = _metrics(metrics_server=SERVER)
    assert metrics.envelope({"event_type": event_type}, None) is None


def test_every_family_is_stormlogs_own() -> None:
    registry, _ = _metrics(metrics_server=SERVER, traces=True)
    assert all(f.spec.name.startswith("stormlog_") for f in registry.families)


def test_the_envelope_does_not_grow_with_the_record() -> None:
    _, metrics = _metrics()
    huge = _request(
        error_message="x" * 1_000_000, chunk_interarrival_ms=[1.0] * 100_000
    )
    envelope = metrics.envelope(huge, {"chunk_summary": ((1,) * 15, 1.0)})
    # Its size is the memory it holds, within the envelope's own bound.
    assert envelope is not None and envelope.size <= REQUEST_LIMITS.max_bytes


def test_run_info_names_the_run_once() -> None:
    registry, metrics = _metrics()
    metrics.set_run_info(1_700_000_000.0)
    exposition = _exposition(registry)
    assert (
        exposition.value(
            "stormlog_run_info",
            run_id="run-1",
            session_id="session-1",
            command="infer profile",
        )
        == 1
    )
    assert exposition.value("stormlog_run_start_time_seconds") == 1_700_000_000


def test_chunk_gaps_are_summarized_into_the_histogram_buckets() -> None:
    counts, total = summarize_chunk_gaps([0.5, 1.0, 1.5, 40_000.0, float("nan")])
    assert len(counts) == len(CHUNK_BUCKETS) + 1
    assert counts[0] == 2  # 0.0005 s and 0.001 s are within the first bound
    assert counts[1] == 1 and counts[-1] == 1  # 40 s lands in +Inf
    assert total == pytest.approx(0.0005 + 0.001 + 0.0015 + 40.0)


@pytest.mark.parametrize(("cases", "fits"), [(200, True), (300, False)])
def test_the_default_budget_fits_200_cases_but_not_300(cases: int, fits: bool) -> None:
    registry, _ = _metrics(
        cases=tuple((f"c{index}_in512_out128", "closed") for index in range(cases)),
        metrics_server=SERVER,
        traces=True,
    )
    assert (registry.budget().samples <= DEFAULT_MAX_SAMPLES) is fits
