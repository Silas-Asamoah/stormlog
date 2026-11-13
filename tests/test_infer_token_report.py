"""Token coverage and the rate block, in ``stormlog.infer.token_report``."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.populations import MeasuredInterval
from stormlog.infer.token_report import throughput, token_coverage

SECOND = 1_000_000_000


def _request(status: str = "ok", **fields: Any) -> dict[str, Any]:
    return {"event_type": "infer.request", "status": status, **fields}


def test_coverage_counts_known_exact_and_sources_over_successful_requests() -> None:
    requests = [
        _request(
            output_tokens=100,
            output_token_source="server_usage",
            output_token_exact=True,
        ),
        _request(
            output_tokens=40, output_token_source="estimated", output_token_exact=False
        ),
        _request(output_tokens=None, output_token_source="unknown"),
        _request(),
        _request("error", output_tokens=7, output_token_source="server_usage"),
    ]
    output = token_coverage(requests)["output"]
    assert output == {
        "eligible": 4,
        "known": 2,
        "exact": 1,
        "by_source": {"estimated": 1, "server_usage": 1, "unknown": 2},
        "known_sum": 140,
        "partial": True,
    }


def test_every_count_known_is_not_partial() -> None:
    requests = [
        _request(
            prompt_tokens=n, prompt_token_source="tiktoken", prompt_token_exact=True
        )
        for n in (10, 20, 30)
    ]
    prompt = token_coverage(requests)["prompt"]
    assert (prompt["eligible"], prompt["known"], prompt["exact"]) == (3, 3, 3)
    assert prompt["known_sum"] == 60
    assert prompt["partial"] is False
    assert prompt["by_source"] == {"tiktoken": 3}


@pytest.mark.parametrize("value", [None, "12", True, False, [], float("nan")])
def test_a_count_that_is_not_a_number_is_not_known(value: Any) -> None:
    coverage = token_coverage([_request(output_tokens=value)])["output"]
    assert (coverage["known"], coverage["known_sum"], coverage["partial"]) == (
        0,
        0,
        True,
    )


def test_only_a_true_exact_flag_counts_as_exact() -> None:
    requests = [
        _request(output_tokens=5, output_token_exact=flag)
        for flag in (True, "true", 1, None, False)
    ]
    assert token_coverage(requests)["output"]["exact"] == 1


def test_known_sum_is_the_sum_the_analyzer_reports() -> None:
    # A float count is truncated as report_stats.int_value truncates it.
    requests = [_request(output_tokens=2.9), _request(output_tokens=3)]
    assert token_coverage(requests)["output"]["known_sum"] == 5


def test_no_successful_requests_have_nothing_eligible() -> None:
    coverage = token_coverage([_request("timeout", output_tokens=9)])
    for kind in ("output", "prompt"):
        assert coverage[kind] == {
            "eligible": 0,
            "known": 0,
            "exact": 0,
            "by_source": {},
            "known_sum": 0,
            "partial": False,
        }


def test_throughput_divides_by_the_rate_interval() -> None:
    interval = MeasuredInterval("measured_span", 0, 4 * SECOND, "all_measured")
    assert throughput(8, 400, 1_000, interval) == {
        "interval_seconds": 4.0,
        "interval_kind": "measured_span",
        "numerator_cohort": "all_measured",
        "requests_per_second": 2.0,
        "output_tokens_per_second": 100.0,
        "total_tokens_per_second": 250.0,
    }


def test_throughput_without_an_interval_length_has_no_rates() -> None:
    empty = MeasuredInterval("request_span", SECOND, SECOND, "all_measured")
    for interval in (None, empty):
        block = throughput(3, 30, 60, interval)
        assert block["requests_per_second"] is None
        assert block["output_tokens_per_second"] is None
        assert block["total_tokens_per_second"] is None
