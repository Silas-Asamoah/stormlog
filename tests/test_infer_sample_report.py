"""Client system samples by request window and by device."""

from __future__ import annotations

import random
from typing import Any

from stormlog.infer import analysis
from stormlog.infer.sample_report import (
    request_window,
    sample_peaks,
    samples_in_window,
)

SECOND = 1_000_000_000


def _sample(timestamp_ns: Any, **fields: Any) -> dict[str, Any]:
    return {"event_type": "infer.system_sample", "timestamp_ns": timestamp_ns, **fields}


def test_the_window_runs_from_the_first_start_to_the_last_end() -> None:
    requests: list[dict[str, Any]] = [
        {"started_at_ns": 5 * SECOND, "ended_at_ns": 6 * SECOND},
        {"started_at_ns": 2 * SECOND, "ended_at_ns": 3 * SECOND},
        {"started_at_ns": 1 * SECOND, "ended_at_ns": None},
        {"started_at_ns": "0", "ended_at_ns": 9 * SECOND},
    ]
    assert request_window(requests) == (2 * SECOND, 6 * SECOND)


def test_requests_without_both_times_have_no_window() -> None:
    assert request_window([]) is None
    assert request_window([{"started_at_ns": 1}, {"ended_at_ns": 2}]) is None


def test_samples_in_the_window_include_both_ends() -> None:
    samples = [_sample(t) for t in (0, 10, 15, 20, 21)] + [_sample(None), _sample("12")]
    kept = samples_in_window(samples, 10, 20)
    assert [s["timestamp_ns"] for s in kept] == [10, 15, 20]


def test_peaks_are_kept_apart_by_sampler_and_device() -> None:
    samples = [
        _sample(
            1, sampler="nvml", device_id=0, device_used_bytes=5, process_rss_bytes=9
        ),
        _sample(2, sampler="nvml", device_id=0, device_used_bytes=7),
        _sample(3, sampler="nvml", device_id=1, device_used_bytes=3),
        _sample(4, sampler="psutil", process_rss_bytes=11),
        _sample(5, sampler="nvml", device_id="1", device_used_bytes="big"),
    ]
    assert sample_peaks(samples) == [
        {
            "sampler": "nvml",
            "device_id": 0,
            "samples": 2,
            "peak_device_used_bytes": 7,
            "peak_process_rss_bytes": 9,
        },
        {
            "sampler": "nvml",
            "device_id": 1,
            "samples": 1,
            "peak_device_used_bytes": 3,
            "peak_process_rss_bytes": None,
        },
        {
            "sampler": "nvml",
            "device_id": "1",
            "samples": 1,
            "peak_device_used_bytes": None,
            "peak_process_rss_bytes": None,
        },
        {
            "sampler": "psutil",
            "device_id": None,
            "samples": 1,
            "peak_device_used_bytes": None,
            "peak_process_rss_bytes": 11,
        },
    ]


def test_a_malformed_device_id_gets_its_own_group() -> None:
    samples = [
        _sample(1, sampler="nvml", device_id=[0], device_used_bytes=1),
        _sample(2, sampler="nvml", device_id=True, device_used_bytes=2),
        _sample(3, sampler="nvml", device_id=1, device_used_bytes=3),
        _sample(4, sampler="nvml", device_id=float("nan"), device_used_bytes=4),
    ]
    peaks = sample_peaks(samples)
    assert len(peaks) == 4
    assert sorted(p["peak_device_used_bytes"] for p in peaks) == [1, 2, 3, 4]


def test_no_samples_have_no_peaks() -> None:
    assert sample_peaks([]) == []


def test_the_helpers_follow_the_analyzers_rules() -> None:
    rng = random.Random(227)
    for _ in range(200):
        requests: list[dict[str, Any]] = [
            {
                "started_at_ns": rng.choice((rng.randint(0, 50), None, 1.5, True)),
                "ended_at_ns": rng.choice((rng.randint(0, 99), None, "9")),
            }
            for _ in range(rng.randint(0, 6))
        ]
        samples = [
            _sample(
                rng.choice((rng.randint(-5, 105), None, 7.9)),
                sampler="nvml",
                device_id=0,
                device_used_bytes=rng.choice((rng.randint(0, 10**12), None, 2.5e9)),
                process_rss_bytes=rng.choice((rng.randint(0, 10**9), "x")),
            )
            for _ in range(rng.randint(0, 8))
        ]
        window = request_window(requests)
        assert window == analysis._request_time_window(requests)
        kept = [] if window is None else samples_in_window(samples, *window)
        assert kept == analysis._samples_for_request_window(samples, requests)
        for field in ("device_used_bytes", "process_rss_bytes"):
            pooled = analysis._peak_sample_value([dict(s) for s in kept], field)
            peaks = [p[f"peak_{field}"] for p in sample_peaks(kept)]
            assert peaks == ([] if not kept else [pooled])
