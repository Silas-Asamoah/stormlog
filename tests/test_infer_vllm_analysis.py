"""vLLM telemetry in ``infer analyze``: deltas, gauges, resets and spans."""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.analysis import analyze_inference_events, format_analysis_text
from stormlog.infer.cli import main as infer_main
from stormlog.infer.vllm_analysis import (
    REASON_BOUNDARIES_CHANGED,
    REASON_COUNTER_RECREATED,
    REASON_COUNTER_RESET,
    REASON_ENGINE_RESTART,
    REASON_ENGINES_CHANGED,
    REASON_NOT_ENABLED,
    REASON_SCRAPE_MISSING,
    REASON_SERIES_MISSING,
    STATE_RESOLVED,
    STATE_UNRESOLVED,
    joined_span_attributes,
)
from stormlog.infer.vllm_metrics import compact_scrape, discover, parse_prometheus_text
from stormlog.infer.vllm_telemetry import (
    MARKER_INTERVAL,
    MARKER_PHASE_END,
    MARKER_PHASE_START,
    SCRAPE_ERROR,
    SCRAPE_OK,
    VllmScrapeRecord,
    VllmSpanRecord,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "vllm"
PRE = (FIXTURES / "q05_c08_metrics_pre.txt").read_text(encoding="utf-8")
POST = (FIXTURES / "q05_c08_metrics_post.txt").read_text(encoding="utf-8")
CLOCK = "client/boot/unix_epoch_ns"
CASE = "c8_in512_out128"
T0 = 1_790_000_000_000_000_000
SECOND = 1_000_000_000


def _value(text: str, name: str, **labels: str) -> float:
    """One scalar series value straight from the text, for expectations."""
    compact = compact_scrape(parse_prometheus_text(text))
    for set_id, value in compact.series(name).items():
        found = compact.labels(set_id)
        if all(found.get(k) == v for k, v in labels.items()) and isinstance(
            value, float
        ):
            return value
    raise KeyError(name)


def _scrape(
    text: str | None, marker: str, observed_at_ns: int, *, case_id: str = CASE
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "session_id": "session",
        "run_id": "run-1",
        "observed_at_ns": observed_at_ns,
        "source_url": "http://127.0.0.1:8000/metrics",
        "marker": marker,
        "interval_ms": 1000,
        "clock_domain": CLOCK,
        "case_id": case_id,
        "phase": "measured",
        "duration_ms": 2.0,
    }
    if text is None:
        record = VllmScrapeRecord(
            status=SCRAPE_ERROR, error="HTTP 503", http_status=503, **common
        )
    else:
        compact = compact_scrape(parse_prometheus_text(text))
        record = VllmScrapeRecord(
            status=SCRAPE_OK,
            http_status=200,
            content_digest="ab" * 32,
            content_bytes=len(text),
            scrape=compact,
            discovery=discover(compact),
            **common,
        )
    return record.to_record()


def _request(
    index: int,
    *,
    case_id: str = CASE,
    x_request_id: str | None = "auto",
    phase: str = "measured",
) -> dict[str, Any]:
    request_id = f"{case_id}_{phase}_0_{index}"
    return {
        "schema_version": 1,
        "event_type": "infer.request",
        "timestamp_ns": T0 + index * SECOND,
        "session_id": "session",
        "request_id": request_id,
        "x_request_id": (
            f"stormlog-run-1-{request_id}" if x_request_id == "auto" else x_request_id
        ),
        "case_id": case_id,
        "phase": phase,
        "started_at_ns": T0 + index * SECOND,
        "ended_at_ns": T0 + index * SECOND + SECOND // 2,
        "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "m",
        "concurrency": 8,
        "target_input_tokens": 512,
        "target_output_tokens": 128,
        "stream": False,
        "status": "ok",
        "e2e_latency_ms": 500.0,
        "ttft_ms": None,
        "first_chunk_latency_ms": None,
        "prompt_tokens": 512,
        "output_tokens": 128,
        "total_tokens": 640,
    }


def _span(
    request_id: str | None,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    **attributes: Any,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "gen_ai.request.id": (
            f"chatcmpl-{request_id}" if request_id else "chatcmpl-random"
        ),
        "gen_ai.latency.time_in_queue": 0.001,
        "gen_ai.latency.time_in_model_prefill": 0.03,
        "gen_ai.latency.time_in_model_decode": 0.4,
        "gen_ai.latency.time_in_model_inference": 0.43,
        "gen_ai.latency.e2e": 0.5,
        "gen_ai.usage.prompt_tokens": 512,
        "gen_ai.usage.completion_tokens": 128,
    }
    values.update(attributes)
    return VllmSpanRecord(
        session_id="session",
        run_id="run-1",
        source="otlp_http_receiver",
        name="llm_request",
        clock_domain="gpu-box/unix_epoch_ns",
        trace_id=trace_id,
        span_id=span_id,
        start_unix_ns=T0,
        end_unix_ns=T0 + SECOND // 2,
        attributes=values,
        request_id=request_id,
    ).to_record()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _artifact(
    tmp_path: Path,
    scrapes: list[dict[str, Any]],
    requests: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
) -> Path:
    requests = [_request(i) for i in range(2)] if requests is None else requests
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "event_type": "infer.session",
            "session_id": "session",
            "timestamp_ns": T0,
            "status": "running",
            "config": {"endpoint": "http://127.0.0.1:8000/v1/chat/completions"},
        },
        {
            "schema_version": 2,
            "event_type": "infer.artifact",
            "event_id": "artifact",
            "artifact_kind": "inference_jsonl",
            "created_at_ns": T0,
            "metadata": {},
            "context": {
                "run_id": "run-1",
                "session_id": "session",
                "producer_id": "stormlog.infer.profile",
                "source": "stormlog.infer.profile",
                "clock_domain": CLOCK,
                "clock_kind": "wall",
                "collection_mode": "active",
                "provenance": "observed",
            },
        },
        {
            "schema_version": 1,
            "event_type": "infer.phase_window",
            "session_id": "session",
            "case_id": CASE,
            "phase": "measured",
            "arrival_mode": "closed",
            "started_at_ns": T0,
            "window_ended_at_ns": T0 + 10 * SECOND,
            "drained_at_ns": T0 + 11 * SECOND,
            "drain_timeout_seconds": 60.0,
            "scheduled_arrivals": None,
            "prompts_digest": "x",
            "abandoned_requests": {},
        },
        *requests,
        *scrapes,
        *(spans or []),
    ]
    path = tmp_path / "infer.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _standard_scrapes(start: str = PRE, end: str = POST) -> list[dict[str, Any]]:
    return [
        _scrape(start, MARKER_PHASE_START, T0 - SECOND),
        _scrape(start, MARKER_INTERVAL, T0 + 2 * SECOND),
        _scrape(end, MARKER_INTERVAL, T0 + 5 * SECOND),
        _scrape(end, MARKER_PHASE_END, T0 + 12 * SECOND),
    ]


def _vllm_case(path: Path, **kwargs: Any) -> dict[str, Any]:
    report = analyze_inference_events(path, **kwargs)
    block = report["telemetry"]["vllm"]
    assert block["status"] == "collected"
    case = block["cases"][CASE]
    assert isinstance(case, dict)
    return case


class TestDeltas:
    def test_real_pre_post_pair_resolves_every_counter(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, _standard_scrapes())
        report = analyze_inference_events(path)
        block = report["telemetry"]["vllm"]
        assert block["observation_scope"] == "engine_aggregate"
        assert "no" in block["note"] and "GPU" in block["note"]
        engine = block["engine"]
        assert engine["scrapes"] == {"ok": 4, "failed": 0}
        assert engine["engines"] == ["0"]
        assert len(engine["epochs"]) == 1
        case = block["cases"][CASE]
        assert case["state"] == STATE_RESOLVED and case["reasons"] == []
        assert case["window"]["seconds"] == 13.0
        counters = case["engines"]["0"]["counters"]
        expected = _value(POST, "vllm:prompt_tokens_total") - _value(
            PRE, "vllm:prompt_tokens_total"
        )
        assert counters["prompt_tokens"]["delta"] == expected
        assert counters["prompt_tokens"]["state"] == STATE_RESOLVED
        assert counters["prompt_tokens"]["native"] == "vllm:prompt_tokens_total"
        success = counters["request_success"]
        assert success["delta"] == 32.0
        assert set(success["by_label"]) >= {
            "finished_reason=length",
            "finished_reason=stop",
        }
        assert all(item["state"] == STATE_RESOLVED for item in counters.values())
        derived = case["engines"]["0"]["derived"]
        assert derived["rates"]["prompt_tokens_per_second"] == pytest.approx(
            expected / 13.0
        )
        assert derived["prefix_cache"]["queries"] == expected
        assert derived["prefix_cache"]["hit_ratio"] == 0.0
        assert derived["mfu"]["state"] == REASON_NOT_ENABLED
        assert derived["kv_cache"]["meaning"].startswith("logical")

    def test_histograms_keep_native_boundaries_and_deltas(self, tmp_path: Path) -> None:
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes()))
        queue = case["engines"]["0"]["histograms"]["queue_time"]
        assert queue["state"] == STATE_RESOLVED
        assert queue["native"] == "vllm:request_queue_time_seconds"
        assert queue["count"] == 32.0
        assert queue["unit"] == "seconds"
        assert "residency" in queue["meaning"]
        assert queue["buckets"][0][0] == "0.3" and queue["buckets"][-1][0] == "+Inf"
        assert queue["buckets"][-1][1] == 32.0
        assert queue["mean"] == pytest.approx(queue["sum"] / 32.0)
        inference = case["engines"]["0"]["histograms"]["inference_time"]
        assert "time_in_model_inference" in inference["meaning"]
        assert "not GPU time" in inference["meaning"]

    def test_gauges_are_summarised_over_the_window(self, tmp_path: Path) -> None:
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes()))
        gauges = case["engines"]["0"]["gauges"]
        waiting = gauges["queue_depth"]["stats"]["_"]
        assert waiting["samples"] == 4
        assert waiting["min"] <= waiting["mean"] <= waiting["max"]
        by_reason = gauges["queue_depth_by_reason"]["stats"]
        assert set(by_reason) == {"reason=capacity", "reason=deferred"}
        usage = gauges["kv_cache_usage"]
        assert usage["unit"] == "fraction"
        kv = case["engines"]["0"]["derived"]["kv_cache"]
        assert kv["state"] == STATE_RESOLVED
        # 0.30.0 labels the cache in tokens; blocks follow from the block size.
        assert kv["block_size"] == 16
        assert kv["kv_cache_size_tokens"] == 1715728
        assert kv["num_gpu_blocks"] == 1715728 // 16
        assert kv["max_blocks_in_use"] == round(
            kv["max_usage_fraction"] * (1715728 // 16)
        )


class TestUnresolved:
    def test_counter_reset_is_unresolved_not_negative(self, tmp_path: Path) -> None:
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(start=POST, end=PRE)))
        assert case["state"] == STATE_RESOLVED  # the epoch itself is fine
        prompt = case["engines"]["0"]["counters"]["prompt_tokens"]
        assert prompt["state"] == STATE_UNRESOLVED and prompt["delta"] is None
        assert prompt["by_label"]["_"]["state"] == REASON_COUNTER_RESET
        queue = case["engines"]["0"]["histograms"]["queue_time"]
        assert queue["state"] == REASON_COUNTER_RESET
        assert (
            case["engines"]["0"]["derived"]["rates"]["prompt_tokens_per_second"] is None
        )

    def test_histogram_sum_or_bucket_going_backwards_is_a_reset(
        self, tmp_path: Path
    ) -> None:
        # Count holds, but the sum shrinks: not a window to difference.
        sum_line = re.compile(
            r"^(vllm:request_queue_time_seconds_sum\{[^}]*\}) (\S+)$", re.MULTILINE
        )
        shrunk = sum_line.sub(r"\1 0.0", POST)
        assert shrunk != POST
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=shrunk)))
        queue = case["engines"]["0"]["histograms"]["queue_time"]
        assert queue["state"] == REASON_COUNTER_RESET
        # One bucket lower than before while count and sum grew.
        bucket_line = re.compile(
            r'^(vllm:request_queue_time_seconds_bucket\{engine="0",le="\+Inf"[^}]*\}) (\S+)$',
            re.MULTILINE,
        )
        lowered = bucket_line.sub(r"\1 1.0", POST)
        assert lowered != POST
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=lowered)))
        queue = case["engines"]["0"]["histograms"]["queue_time"]
        assert queue["state"] == REASON_COUNTER_RESET

    def test_recreated_families_are_unresolved(self, tmp_path: Path) -> None:
        created = re.compile(
            r"^(vllm:(?:request_queue_time_seconds|prompt_tokens)_created\{[^}]*\}) (\S+)$",
            re.MULTILINE,
        )
        recreated = created.sub(r"\1 1.791e+09", POST)
        assert recreated != POST
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=recreated)))
        engine = case["engines"]["0"]
        queue = engine["histograms"]["queue_time"]
        assert queue["state"] == REASON_COUNTER_RECREATED
        assert "count" not in queue
        prompt = engine["counters"]["prompt_tokens"]
        assert prompt["state"] == STATE_UNRESOLVED and prompt["delta"] is None
        assert prompt["by_label"]["_"]["state"] == REASON_COUNTER_RECREATED
        # The sibling families keep their own, untouched, stamps.
        assert engine["counters"]["generation_tokens"]["state"] == STATE_RESOLVED
        assert engine["histograms"]["prefill_time"]["state"] == STATE_RESOLVED

    def test_a_recreated_native_total_histogram_is_unresolved(
        self, tmp_path: Path
    ) -> None:
        # vllm:iteration_tokens_total is a histogram despite its suffix, so
        # its stamp is vllm:iteration_tokens_total_created; the counter rule
        # of dropping _total would look for a gauge that does not exist and
        # never notice the recreation.
        recreated = re.sub(
            r"^(vllm:iteration_tokens_total_created\{[^}]*\}) (\S+)$",
            r"\1 1.791e+09",
            POST,
            flags=re.MULTILINE,
        )
        assert recreated != POST
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=recreated)))
        engine = case["engines"]["0"]
        iteration = engine["histograms"]["iteration_tokens"]
        assert iteration["native"] == "vllm:iteration_tokens_total"
        assert iteration["state"] == REASON_COUNTER_RECREATED
        assert engine["histograms"]["queue_time"]["state"] == STATE_RESOLVED

    def test_changed_bucket_boundaries_are_unresolved(self, tmp_path: Path) -> None:
        reshaped = POST.replace(
            'vllm:request_queue_time_seconds_bucket{engine="0",le="0.3"',
            'vllm:request_queue_time_seconds_bucket{engine="0",le="0.35"',
        )
        assert reshaped != POST
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=reshaped)))
        engine = case["engines"]["0"]
        assert engine["histograms"]["queue_time"]["state"] == REASON_BOUNDARIES_CHANGED
        assert engine["histograms"]["prefill_time"]["state"] == STATE_RESOLVED

    def test_engine_set_change_leaves_the_window_unresolved(
        self, tmp_path: Path
    ) -> None:
        grown = POST + '\nvllm:num_requests_running{engine="1",model_name="m"} 0.0\n'
        path = _artifact(tmp_path, _standard_scrapes(end=grown))
        report = analyze_inference_events(path)
        block = report["telemetry"]["vllm"]
        assert len(block["engine"]["epochs"]) == 1
        case = block["cases"][CASE]
        assert case["state"] == STATE_UNRESOLVED
        assert case["reasons"] == [REASON_ENGINES_CHANGED]
        assert "vllm: unresolved (engine_set_changed)" in format_analysis_text(report)

    def test_zero_mfu_counters_need_a_resolved_token_delta(
        self, tmp_path: Path
    ) -> None:
        # Without the generated token delta, zero counters are not a measurement.
        without = "\n".join(
            line
            for line in POST.splitlines()
            if not line.startswith("vllm:generation_tokens_total{")
        )
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=without)))
        mfu = case["engines"]["0"]["derived"]["mfu"]
        assert mfu["state"] == STATE_UNRESOLVED
        assert (
            "estimated_flops_per_gpu" in mfu and mfu["estimated_flops_per_gpu"] == 0.0
        )
        # One of the MFU counters itself unresolved: unresolved, with the states.
        reset = re.sub(
            r"^(vllm:estimated_flops_per_gpu_total\{[^}]*\}) (\S+)$",
            r"\1 5.0",
            PRE,
            flags=re.MULTILINE,
        )
        assert reset != PRE
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(start=reset)))
        mfu = case["engines"]["0"]["derived"]["mfu"]
        assert mfu["state"] == STATE_UNRESOLVED
        assert mfu["counters"]["estimated_flops_per_gpu"] == STATE_UNRESOLVED
        assert mfu["counters"]["estimated_read_bytes_per_gpu"] == STATE_RESOLVED
        # Zero generated tokens and zero counters is a consistent, resolved zero.
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(start=POST, end=POST)))
        mfu = case["engines"]["0"]["derived"]["mfu"]
        assert mfu["state"] == STATE_RESOLVED and mfu["estimated_flops_per_gpu"] == 0.0
        assert "no tokens" in mfu["detail"]

    def test_engine_restart_leaves_the_window_unresolved(self, tmp_path: Path) -> None:
        restarted = re.sub(
            r"^(process_start_time_seconds) (\S+)$",
            r"\1 1.79e+09",
            POST,
            flags=re.MULTILINE,
        )
        assert restarted != POST
        path = _artifact(tmp_path, _standard_scrapes(end=restarted))
        report = analyze_inference_events(path)
        block = report["telemetry"]["vllm"]
        assert len(block["engine"]["epochs"]) == 2
        case = block["cases"][CASE]
        assert case["state"] == STATE_UNRESOLVED
        assert REASON_ENGINE_RESTART in case["reasons"]
        prompt = case["engines"]["0"]["counters"]["prompt_tokens"]
        assert prompt["by_label"]["_"]["state"] == REASON_ENGINE_RESTART
        text = format_analysis_text(report)
        assert "restarted 1 time" in text
        assert "vllm: unresolved (engine_restart)" in text

    def test_a_histogram_missing_its_sum_is_not_differenced(
        self, tmp_path: Path
    ) -> None:
        # The start scrape has the queue-time buckets and count but no _sum;
        # treating the missing sum as 0.0 would resolve a delta against a
        # number that was never observed.
        without_sum = "\n".join(
            line
            for line in PRE.splitlines()
            if not line.startswith("vllm:request_queue_time_seconds_sum{")
        )
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(start=without_sum)))
        engine = case["engines"]["0"]
        queue = engine["histograms"]["queue_time"]
        assert queue["state"] == REASON_SERIES_MISSING
        assert queue["missing"] == ["start_sum"]
        assert "sum" not in queue and "count" not in queue
        assert engine["histograms"]["prefill_time"]["state"] == STATE_RESOLVED

    def test_missing_series_is_reported_not_zero(self, tmp_path: Path) -> None:
        without = "\n".join(
            line
            for line in POST.splitlines()
            if not line.startswith("vllm:num_requests_waiting{")
            and not line.startswith("vllm:prompt_tokens_total{")
        )
        case = _vllm_case(_artifact(tmp_path, _standard_scrapes(end=without)))
        engine = case["engines"]["0"]
        assert engine["counters"]["prompt_tokens"]["by_label"]["_"]["state"] == (
            REASON_SERIES_MISSING
        )
        assert engine["counters"]["prompt_tokens"]["delta"] is None
        # The gauge was present in the start scrape and one interval scrape.
        assert engine["gauges"]["queue_depth"]["stats"]["_"]["samples"] == 2

    def test_failed_or_missing_boundary_scrapes(self, tmp_path: Path) -> None:
        scrapes = [
            _scrape(None, MARKER_PHASE_START, T0 - SECOND),
            _scrape(POST, MARKER_INTERVAL, T0 + 2 * SECOND),
        ]
        case = _vllm_case(_artifact(tmp_path, scrapes))
        assert case["state"] == STATE_UNRESOLVED
        assert case["reasons"] == [
            "scrape_failed:phase_start",
            f"{REASON_SCRAPE_MISSING}:phase_end",
        ]
        assert case["window"] is None and case["engines"] == {}


def _ranked_queue_histogram(ranks: list[tuple[int, float, float]]) -> str:
    """One queue-time histogram with a rank label, in the order given."""
    lines = [
        "# TYPE process_start_time_seconds gauge",
        "process_start_time_seconds 100",
        "# TYPE vllm:request_queue_time_seconds histogram",
    ]
    for rank, count, total in ranks:
        labels = f'engine="0",model_name="m",rank="{rank}"'
        lines += [
            f'vllm:request_queue_time_seconds_bucket{{{labels},le="+Inf"}} {count}',
            f"vllm:request_queue_time_seconds_sum{{{labels}}} {total}",
            f"vllm:request_queue_time_seconds_count{{{labels}}} {count}",
        ]
    return "\n".join(lines) + "\n"


class TestHistogramLabelSets:
    def test_histograms_are_differenced_per_label_set(self, tmp_path: Path) -> None:
        # The end scrape lists the two ranks in the other order. Each rank is
        # differenced against itself and the total is their sum, never one
        # rank's end minus the other's start.
        start = _ranked_queue_histogram([(0, 10, 1.0), (1, 100, 20.0)])
        end = _ranked_queue_histogram([(1, 102, 22.0), (0, 11, 1.1)])
        scrapes = [
            _scrape(start, MARKER_PHASE_START, T0 - SECOND),
            _scrape(end, MARKER_PHASE_END, T0 + 12 * SECOND),
        ]
        case = _vllm_case(_artifact(tmp_path, scrapes))
        queue = case["engines"]["0"]["histograms"]["queue_time"]
        assert queue["state"] == STATE_RESOLVED
        assert (queue["count"], queue["sum"]) == (3.0, pytest.approx(2.1))
        assert queue["buckets"] == [["+Inf", 3.0]]
        assert queue["by_label"]["rank=0"]["count"] == 1.0
        assert queue["by_label"]["rank=0"]["sum"] == pytest.approx(0.1)
        assert queue["by_label"]["rank=1"]["count"] == 2.0
        assert queue["by_label"]["rank=1"]["sum"] == 2.0

    def test_a_label_set_missing_from_one_scrape_is_unresolved(
        self, tmp_path: Path
    ) -> None:
        start = _ranked_queue_histogram([(0, 10, 1.0), (1, 100, 20.0)])
        end = _ranked_queue_histogram([(0, 11, 1.1)])
        scrapes = [
            _scrape(start, MARKER_PHASE_START, T0 - SECOND),
            _scrape(end, MARKER_PHASE_END, T0 + 12 * SECOND),
        ]
        case = _vllm_case(_artifact(tmp_path, scrapes))
        queue = case["engines"]["0"]["histograms"]["queue_time"]
        assert queue["state"] == STATE_UNRESOLVED
        assert queue["reasons"] == [REASON_SERIES_MISSING]
        assert "count" not in queue and "sum" not in queue
        assert queue["by_label"]["rank=0"]["state"] == STATE_RESOLVED
        assert queue["by_label"]["rank=1"]["state"] == REASON_SERIES_MISSING


def _exporter_scrape(process_start: int | None, tokens: float, waiting: float) -> str:
    """A tiny scrape from one exporter process, or from none identifiable."""
    lines = ["# TYPE process_start_time_seconds gauge"]
    if process_start is not None:
        lines.append(f"process_start_time_seconds {process_start}")
    lines += [
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {tokens}',
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}',
    ]
    return "\n".join(lines) + "\n"


class TestExporterIdentity:
    def test_another_exporter_inside_the_window_is_caught(self, tmp_path: Path) -> None:
        # A load balancer answered the interval scrape from exporter B. The
        # two boundaries agree, but the window is not A's alone, and B's
        # gauge must not land in A's summary.
        scrapes = [
            _scrape(_exporter_scrape(100, 10, 0), MARKER_PHASE_START, T0 - SECOND),
            _scrape(_exporter_scrape(200, 1000, 999), MARKER_INTERVAL, T0 + 5 * SECOND),
            _scrape(_exporter_scrape(100, 20, 0), MARKER_PHASE_END, T0 + 12 * SECOND),
        ]
        report = analyze_inference_events(_artifact(tmp_path, scrapes))
        block = report["telemetry"]["vllm"]
        assert len(block["engine"]["epochs"]) == 3
        case = block["cases"][CASE]
        assert case["state"] == STATE_UNRESOLVED
        assert case["reasons"] == [REASON_ENGINE_RESTART]
        assert case["window"]["foreign_scrapes"] == 1
        engine = case["engines"]["0"]
        prompt = engine["counters"]["prompt_tokens"]["by_label"]["_"]
        assert prompt["state"] == REASON_ENGINE_RESTART
        waiting = engine["gauges"]["queue_depth"]["stats"]["_"]
        assert (waiting["max"], waiting["samples"]) == (0.0, 2)

    def test_scrapes_with_no_exporter_identity_are_unresolved(
        self, tmp_path: Path
    ) -> None:
        # Without process_start_time_seconds (prometheus multiprocess mode
        # drops it) a restart cannot be told apart from a quiet window.
        scrapes = [
            _scrape(_exporter_scrape(None, 10, 0), MARKER_PHASE_START, T0 - SECOND),
            _scrape(_exporter_scrape(None, 20, 0), MARKER_PHASE_END, T0 + 12 * SECOND),
        ]
        case = _vllm_case(_artifact(tmp_path, scrapes))
        assert case["state"] == STATE_UNRESOLVED
        assert case["reasons"] == ["exporter_identity_unknown"]
        prompt = case["engines"]["0"]["counters"]["prompt_tokens"]["by_label"]["_"]
        assert prompt["state"] == "exporter_identity_unknown"


def _with_sample(text: str, family: str, sample: str) -> str:
    """The text with every sample of one scalar family replaced."""
    changed = re.sub(
        rf"(?m)^({re.escape(family)}\{{[^\n]+?\}}) [^\n]+$", rf"\1 {sample}", text
    )
    assert changed != text
    return changed


class TestNonFinite:
    @pytest.mark.parametrize("sample", ["NaN", "+Inf"])
    def test_a_non_finite_gauge_leaves_only_its_fields_unresolved(
        self, tmp_path: Path, sample: str
    ) -> None:
        # The record keeps the value as the strict-JSON string; the report
        # must not die on it, and must not carry it into a block count.
        start = _with_sample(PRE, "vllm:kv_cache_usage_perc", sample)
        end = _with_sample(POST, "vllm:kv_cache_usage_perc", sample)
        report = analyze_inference_events(
            _artifact(tmp_path, _standard_scrapes(start, end))
        )
        json.dumps(report, allow_nan=False)
        engine = report["telemetry"]["vllm"]["cases"][CASE]["engines"]["0"]
        usage = engine["gauges"]["kv_cache_usage"]
        assert usage["state"] == "non_finite_sample"
        assert usage["stats"]["_"]["non_finite"] == 4
        assert usage["stats"]["_"]["max"] is None
        kv = engine["derived"]["kv_cache"]
        assert kv["state"] == "non_finite_sample"
        assert kv["max_usage_fraction"] is None and kv["max_blocks_in_use"] is None
        assert kv["num_gpu_blocks"] == 1715728 // 16
        assert engine["counters"]["prompt_tokens"]["state"] == STATE_RESOLVED

    def test_a_non_finite_counter_is_unresolved_not_a_nan_delta(
        self, tmp_path: Path
    ) -> None:
        end = _with_sample(POST, "vllm:prompt_tokens_total", "NaN")
        report = analyze_inference_events(
            _artifact(tmp_path, _standard_scrapes(end=end))
        )
        json.dumps(report, allow_nan=False)
        counters = report["telemetry"]["vllm"]["cases"][CASE]["engines"]["0"][
            "counters"
        ]
        prompt = counters["prompt_tokens"]
        assert prompt["by_label"]["_"]["state"] == "non_finite_sample"
        assert prompt["by_label"]["_"]["end"] == "NaN"
        assert prompt["state"] == STATE_UNRESOLVED and prompt["delta"] is None
        assert counters["generation_tokens"]["state"] == STATE_RESOLVED


class TestDuplicateSpans:
    def _spans(self) -> list[dict[str, Any]]:
        """Two requests whose spans say 10 ms and 1,000 ms of inference."""
        spans = []
        for index, seconds in enumerate([0.01, 1.0]):
            span = _span(
                _request(index)["x_request_id"],
                trace_id=str(index) * 32,
                span_id=str(index) * 16,
            )
            span["attributes"]["gen_ai.latency.time_in_model_inference"] = seconds
            spans.append(span)
        return spans

    def test_a_span_delivered_twice_weighs_once(self, tmp_path: Path) -> None:
        # An OTLP retry after a lost response, or two overlapping files,
        # deliver the same span again. It must count once in the statistics,
        # and the extra delivery must be visible as what it is.
        spans = self._spans()
        once = analyze_inference_events(
            _artifact(tmp_path, _standard_scrapes(), spans=spans)
        )["telemetry"]["vllm"]
        twice = analyze_inference_events(
            _artifact(
                tmp_path, _standard_scrapes(), spans=[spans[0], spans[0], spans[1]]
            )
        )["telemetry"]["vllm"]
        key = "time_in_model_inference"
        assert once["cases"][CASE]["spans"]["latency"][key]["p50_ms"] == 505.0
        assert twice["cases"][CASE]["spans"]["latency"][key]["p50_ms"] == 505.0
        assert twice["cases"][CASE]["spans"]["latency"][key]["mean_ms"] == 505.0
        assert twice["cases"][CASE]["spans"]["spans"] == 2
        assert (twice["spans"]["total"], twice["spans"]["joined"]) == (2, 2)
        assert twice["spans"]["deliveries"] == 3
        assert twice["spans"]["duplicates"] == 1
        assert twice["spans"]["conflicting_duplicates"] == 0

    def test_a_conflicting_duplicate_is_diagnosed(self, tmp_path: Path) -> None:
        spans = self._spans()
        conflicting = dict(spans[0])
        conflicting["attributes"] = {
            **spans[0]["attributes"],
            "gen_ai.latency.time_in_model_inference": 5.0,
        }
        block = analyze_inference_events(
            _artifact(tmp_path, _standard_scrapes(), spans=[*spans, conflicting])
        )["telemetry"]["vllm"]
        key = "time_in_model_inference"
        # Neither delivery can be trusted, so the request is quarantined:
        # only the other request's 1,000 ms remains, and nothing is averaged.
        case_spans = block["cases"][CASE]["spans"]
        assert case_spans["latency"][key]["mean_ms"] == 1000.0
        assert case_spans["latency"][key]["n"] == 1
        assert (
            case_spans["requests_with_span"],
            case_spans["quarantined_requests"],
        ) == (
            1,
            1,
        )
        assert block["spans"]["duplicates"] == 1
        assert block["spans"]["conflicting_duplicates"] == 1
        assert block["spans"]["quarantined_requests"] == {"conflicting_spans": 1}

    def test_a_request_with_two_different_spans_is_quarantined(
        self, tmp_path: Path
    ) -> None:
        spans = self._spans()
        second = dict(spans[0])
        second["span_id"] = "f" * 16
        records = _records(
            _artifact(tmp_path, _standard_scrapes(), spans=[*spans, second])
        )
        joined = joined_span_attributes(records)
        first_id = _request(0)["x_request_id"]
        assert joined.quarantined == {first_id: "multiple_spans"}
        assert set(joined.by_request) == {_request(1)["x_request_id"]}

    def test_joined_span_attributes_give_each_trusted_request_its_span(
        self, tmp_path: Path
    ) -> None:
        records = _records(
            _artifact(tmp_path, _standard_scrapes(), spans=self._spans())
        )
        joined = joined_span_attributes(records)
        attributes = joined.by_request[_request(1)["x_request_id"]]
        assert attributes["gen_ai.latency.time_in_model_inference"] == 1.0
        assert joined.quarantined == {}


class TestNamesAndEngines:
    def test_retired_name_is_normalised_and_flagged(self, tmp_path: Path) -> None:
        old = PRE.replace("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
        new = POST.replace("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
        path = _artifact(tmp_path, _standard_scrapes(start=old, end=new))
        report = analyze_inference_events(path)
        block = report["telemetry"]["vllm"]
        assert block["engine"]["deprecated_series"] == ["vllm:gpu_cache_usage_perc"]
        usage = block["cases"][CASE]["engines"]["0"]["gauges"]["kv_cache_usage"]
        assert usage["native"] == "vllm:gpu_cache_usage_perc"
        assert usage["deprecated_alias_of"] == "vllm:gpu_cache_usage_perc"
        assert "retired series: vllm:gpu_cache_usage_perc" in format_analysis_text(
            report
        )

    def test_two_engines_are_reported_separately(self, tmp_path: Path) -> None:
        def doubled(text: str) -> str:
            lines = []
            for line in text.splitlines():
                lines.append(line)
                if line.startswith("vllm:") and 'engine="0"' in line:
                    lines.append(line.replace('engine="0"', 'engine="1"'))
            return "\n".join(lines)

        case = _vllm_case(
            _artifact(tmp_path, _standard_scrapes(doubled(PRE), doubled(POST)))
        )
        assert set(case["engines"]) == {"0", "1"}
        zero = case["engines"]["0"]["counters"]["prompt_tokens"]["delta"]
        one = case["engines"]["1"]["counters"]["prompt_tokens"]["delta"]
        assert zero == one and zero is not None
        assert "engines" not in case["engines"]  # never a combined entry

    def test_unknown_series_are_kept_and_listed(self, tmp_path: Path) -> None:
        extra = (
            POST
            + '\n# TYPE vllm:brand_new_total counter\nvllm:brand_new_total{engine="0",model_name="m"} 7.0\n'
            + '# TYPE vllm:kv_offload_load_bytes_total counter\nvllm:kv_offload_load_bytes_total{engine="0",model_name="m"} 3.0\n'
        )
        path = _artifact(tmp_path, _standard_scrapes(end=extra))
        report = analyze_inference_events(path)
        engine = report["telemetry"]["vllm"]["engine"]
        assert engine["unknown_series"] == ["vllm:brand_new_total"]
        # An uncatalogued family of a known optional subsystem is listed on
        # its own, not as unknown, and still kept raw under its native name.
        assert engine["optional_present"] == ["vllm:kv_offload_load_bytes_total"]
        counters = report["telemetry"]["vllm"]["cases"][CASE]["engines"]["0"][
            "counters"
        ]
        assert (
            counters["vllm:brand_new_total"]["by_label"]["_"]["state"]
            == REASON_SERIES_MISSING
        )


class TestSpans:
    def test_spans_join_by_recorded_header_and_summarise_residency(
        self, tmp_path: Path
    ) -> None:
        requests = [
            _request(0),
            _request(1),
            _request(2, x_request_id=None),
            _request(0, phase="warmup"),
        ]
        slower = _span("stormlog-run-1-c8_in512_out128_measured_0_1")
        slower["attributes"]["gen_ai.latency.time_in_model_inference"] = 0.63
        spans = [
            _span("stormlog-run-1-c8_in512_out128_measured_0_0"),
            slower,
            _span("stormlog-run-9-other"),
            _span(None),
            _span("stormlog-run-1-c8_in512_out128_warmup_0_0"),
        ]
        path = _artifact(tmp_path, _standard_scrapes(), requests, spans)
        report = analyze_inference_events(path)
        block = report["telemetry"]["vllm"]
        assert block["spans"]["total"] == 5
        assert block["spans"]["joined"] == 2
        assert block["spans"]["unjoined_by_reason"] == {
            "no_request_id": 1,
            "request_not_in_run": 1,
            "warmup_request": 1,
        }
        case = block["cases"][CASE]["spans"]
        assert case["requests"] == 3 and case["requests_with_span"] == 2
        inference = case["latency"]["time_in_model_inference"]
        assert inference["n"] == 2
        assert inference["p50_ms"] == pytest.approx(530.0)
        assert "not GPU time" in case["note"]
        text = format_analysis_text(report)
        assert "vllm spans: 2 of 3 requests" in text
        assert "residency, not GPU time" in text

    def test_external_span_file_via_cli(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, _standard_scrapes())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = infer_main(
                [
                    "analyze",
                    str(path),
                    "--format",
                    "json",
                    "--vllm-spans",
                    str(FIXTURES / "q05_spans_sample.jsonl"),
                ]
            )
        assert code == int(ExitCode.OK)
        report = json.loads(stdout.getvalue())
        spans = report["telemetry"]["vllm"]["spans"]
        assert spans["sources"] == {"jsonl_file": 16}
        assert spans["unjoined_by_reason"] == {
            "not_a_request_span": 4,
            "request_not_in_run": 12,
        }

    def test_unreadable_span_file_is_an_input_error(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, _standard_scrapes())
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            missing = infer_main(
                ["analyze", str(path), "--vllm-spans", str(tmp_path / "nope.json")]
            )
            bad = tmp_path / "bad.jsonl"
            bad.write_text("{not json\n")
            broken = infer_main(["analyze", str(path), "--vllm-spans", str(bad)])
        assert missing == int(ExitCode.INVALID_INPUT)
        assert broken == int(ExitCode.INVALID_INPUT)


class TestReportShape:
    def test_artifact_without_vllm_records_says_absent(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, [])
        report = analyze_inference_events(path)
        assert report["telemetry"]["vllm"] == {"status": "absent"}
        assert "vLLM" not in format_analysis_text(report)

    def test_text_report_lines(self, tmp_path: Path) -> None:
        report = analyze_inference_events(_artifact(tmp_path, _standard_scrapes()))
        text = format_analysis_text(report)
        assert "vLLM telemetry: 4 scrapes ok, 0 failed; engine label(s) 0" in text
        assert "engine-aggregate, no per-request attribution" in text
        assert "vllm engine 0: waiting max" in text
        assert "incl. drain" in text

    def test_capabilities_are_carried(self, tmp_path: Path) -> None:
        capability = {
            "schema_version": 2,
            "event_type": "infer.capabilities",
            "event_id": "capability:vllm.metrics",
            "component": "vllm.metrics",
            "available": True,
            "supported": ["queue_depth"],
            "enabled": ["queue_depth"],
            "collected": ["queue_depth"],
            "metadata": {"url": "x"},
            "context": {},
        }
        path = _artifact(tmp_path, [*_standard_scrapes(), capability])
        report = analyze_inference_events(path)
        assert report["telemetry"]["vllm"]["capabilities"]["vllm.metrics"][
            "collected"
        ] == ["queue_depth"]
