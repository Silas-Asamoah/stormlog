"""Latency quantiles with sufficiency, and chunk-level streaming, in reports."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.analysis import analyze_inference_events
from stormlog.infer.latency_report import latency_summary, streaming_summary
from stormlog.infer.vllm_analysis import JoinedSpans

SECOND = 1_000_000_000


def _streamed(index: int, **overrides: Any) -> dict[str, Any]:
    """A 128-token response delivered in 8 chunks of 16 tokens."""
    record: dict[str, Any] = {
        "event_type": "infer.request",
        "phase": "measured",
        "session_id": "s1",
        "case_id": "c1",
        "request_id": f"r{index}",
        "x_request_id": f"x{index}",
        "request_index": index,
        "status": "ok",
        "stream": True,
        "started_at_ns": index * SECOND,
        "ended_at_ns": index * SECOND + SECOND // 2,
        "intended_at_ns": index * SECOND,
        "dispatch_lag_ms": 1.0,
        "ttft_ms": 100.0,
        "e2e_latency_ms": 100.0 + 127 * 3.0,
        "chunk_interarrival_ms": [54.0] * 7,
        "output_tokens": 128,
        "output_token_source": "server_usage",
    }
    record.update(overrides)
    return record


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_chunk_timing_stays_chunk_level_and_tpot_comes_from_tokens() -> None:
    requests = [_streamed(i) for i in range(4)]
    streaming = streaming_summary(requests)
    latency = latency_summary(requests)

    assert streaming["granularity"] == "chunk"
    assert streaming["content_chunks_per_response"]["p50"] == 8.0
    assert streaming["mean_tokens_per_chunk"] == 16.0
    assert streaming["chunk_interarrival_ms"]["p50"] == 54.0
    assert "not inter-token latencies" in streaming["note"]
    # TPOT is (e2e - ttft) / (output tokens - 1), not a chunk gap.
    tpot = latency["metrics"]["client.tpot"]["successful"]["p50"]["value_ms"]
    assert tpot == pytest.approx(3.0)


def test_no_inter_token_latency_appears_outside_vllms_own_block(tmp_path: Path) -> None:
    path = tmp_path / "infer.jsonl"
    records = [{"event_type": "infer.session", "session_id": "s1"}] + [
        _streamed(i) for i in range(4)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    report = analyze_inference_events(path)
    vllm_free = {key: value for key, value in report.items() if key != "telemetry"}

    names = list(_keys(vllm_free))
    assert not [name for name in names if re.search(r"(^|[._])itl($|[._])", name)]
    assert not [name for name in names if "inter_token" in name.lower()]
    case = report["cases"]["c1"]
    assert case["streaming"]["granularity"] == "chunk"
    assert case["latency"]["metrics"]["client.ttft"]["boundary"] == "client"


def test_each_quantile_says_whether_the_case_has_enough_requests() -> None:
    requests = [_streamed(i, e2e_latency_ms=float(100 + i)) for i in range(230)]
    e2e = latency_summary(requests)["metrics"]["client.e2e"]["successful"]

    assert e2e["p95"]["sufficient"] is True
    assert (e2e["p95"]["n"], e2e["p95"]["n_min"], e2e["p95"]["n_min_exists"]) == (
        230,
        230,
        59,
    )
    assert e2e["p95"]["interval"]["coverage"] >= 0.95
    assert e2e["p99"]["sufficient"] is False
    assert e2e["p99"]["interval"] is None


def test_the_penalized_estimand_ranks_failures_worst() -> None:
    requests = [_streamed(i, e2e_latency_ms=float(100 + i)) for i in range(95)] + [
        _streamed(95 + i, status="error", e2e_latency_ms=5.0) for i in range(5)
    ]
    e2e = latency_summary(requests)["metrics"]["client.e2e"]

    assert e2e["successful"]["p95"]["n"] == 95
    penalized = e2e["failure_penalized"]
    assert penalized["p95"]["n"] == 100
    assert penalized["p95"]["value_ms"] == 194.0  # the largest success
    assert penalized["p99"]["penalized"] is True
    assert penalized["p99"]["value_ms"] is None
    # Errors are not timeouts: no observed lower bound is claimed.
    assert penalized["p99"]["observed_lower_bound_ms"] is None


def test_only_timeouts_give_the_penalized_e2e_an_observed_bound() -> None:
    requests = [_streamed(i) for i in range(90)] + [
        _streamed(90 + i, status="timeout", e2e_latency_ms=60_000.0) for i in range(10)
    ]
    metrics = latency_summary(requests)["metrics"]
    e2e = metrics["client.e2e"]["failure_penalized"]["p99"]
    assert e2e["penalized"] is True
    assert e2e["observed_lower_bound_ms"] == 60_000.0
    # A timed-out stream may have had its first token long before.
    ttft = metrics["client.ttft"]["failure_penalized"]["p99"]
    assert ttft["observed_lower_bound_ms"] is None


def test_non_streamed_responses_have_no_client_ttft() -> None:
    requests = [
        _streamed(i, stream=False, ttft_ms=None, chunk_interarrival_ms=[])
        for i in range(3)
    ]
    latency = latency_summary(requests)
    ttft = latency["metrics"]["client.ttft"]
    assert ttft["successful_missing"] == 3
    assert ttft["successful"]["p50"]["value_ms"] is None
    assert streaming_summary(requests)["responses"] == 0


def test_a_case_without_failures_is_never_penalized_for_missing_values() -> None:
    requests = [
        _streamed(i, stream=False, ttft_ms=None, chunk_interarrival_ms=[])
        for i in range(30)
    ]
    ttft = latency_summary(requests)["metrics"]["client.ttft"]
    p50 = ttft["failure_penalized"]["p50"]
    assert p50["penalized"] is False
    assert p50["value_ms"] is None
    assert p50["n"] == 30
    assert p50["reason"] == "successful_values_missing"


def test_missing_values_do_not_change_the_share_that_failed() -> None:
    # 95 successes and 5 errors; 47 successes have no span. The client e2e
    # p95 is finite over 100 offered; the server one must not read as lying
    # in the failure mass with n = 53.
    requests = [_streamed(i, e2e_latency_ms=float(100 + i)) for i in range(95)] + [
        _streamed(95 + i, status="error") for i in range(5)
    ]
    spans = JoinedSpans(
        by_request={f"x{i}": {"gen_ai.latency.e2e": 0.1} for i in range(48)},
        quarantined={},
    )
    metrics = latency_summary(requests, spans=spans)["metrics"]
    assert metrics["client.e2e"]["failure_penalized"]["p95"]["value_ms"] == 194.0
    server = metrics["server.e2e"]["failure_penalized"]["p95"]
    assert server["penalized"] is False
    assert server["value_ms"] is None
    assert server["n"] == 100
    assert server["reason"] == "successful_values_missing"


def test_server_latency_comes_only_from_each_requests_own_span() -> None:
    requests = [_streamed(i) for i in range(3)]
    spans = JoinedSpans(
        by_request={
            "x0": {"gen_ai.latency.time_to_first_token": 0.4},
            "x1": {"gen_ai.latency.time_to_first_token": 0.6},
        },
        quarantined={"x2": "conflicting_spans"},
    )
    metrics = latency_summary(requests, spans=spans)["metrics"]

    server = metrics["server.ttft"]
    assert server["boundary"] == "server"
    assert server["successful_missing"] == 1
    assert server["successful"]["p50"]["value_ms"] == pytest.approx(500.0)
    # The client's own TTFT is unchanged by the spans.
    assert metrics["client.ttft"]["successful"]["p50"]["value_ms"] == 100.0


def test_without_spans_there_are_no_server_metrics() -> None:
    metrics = latency_summary([_streamed(0)])["metrics"]
    assert not [key for key in metrics if key.startswith("server.")]


def test_requests_that_did_not_succeed_keep_their_observed_time_by_status() -> None:
    requests = [
        _streamed(0),
        _streamed(1, status="timeout", e2e_latency_ms=60_000.0),
        _streamed(2, status="timeout", e2e_latency_ms=59_990.0),
        _streamed(3, status="error", e2e_latency_ms=12.0),
        _streamed(4, status="dropped", e2e_latency_ms=None),
    ]
    unsuccessful = latency_summary(requests)["unsuccessful"]

    assert set(unsuccessful) == {"timeout", "error", "dropped"}
    timeouts = unsuccessful["timeout"]
    assert timeouts["count"] == 2
    assert timeouts["elapsed_ms"] == {"min": 59_990.0, "p50": 59_995.0, "max": 60_000.0}
    assert timeouts["elapsed_missing"] == 0
    assert unsuccessful["error"]["elapsed_ms"]["max"] == 12.0
    # A dropped request was never sent, so it has no elapsed time at all.
    assert unsuccessful["dropped"] == {
        "count": 1,
        "elapsed_ms": {"min": None, "p50": None, "max": None},
        "elapsed_missing": 1,
    }


def test_a_case_where_everything_succeeded_has_no_unsuccessful_entries() -> None:
    assert latency_summary([_streamed(0), _streamed(1)])["unsuccessful"] == {}
