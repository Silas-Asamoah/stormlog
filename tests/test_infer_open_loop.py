"""Open-loop arrivals: schedule-driven dispatch and in-flight accounting."""

import asyncio
import contextlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.analysis import analyze_inference_events, format_analysis_text
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.open_loop import Arrival, InFlightLimiter, dispatch_schedule
from stormlog.infer.openai_client import ChatCompletionResult, EndpointHTTPError
from stormlog.infer.profile import InferenceProfiler
from tests.infer_workload_helpers import SleepingClient, run_profile_with_fake_client


def test_fixed_rate_keeps_sending_while_earlier_requests_run(tmp_path: Path) -> None:
    requests, report, client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.2,
        arrival_mode="fixed-rate",
        rates=(50.0,),
        request_count=5,
    )
    # A closed loop with one worker would never overlap requests.
    assert client.max_active >= 4
    intended = sorted(r["intended_at_ns"] for r in requests)
    gaps = [later - earlier for earlier, later in zip(intended, intended[1:])]
    assert gaps == [20_000_000] * 4
    assert {r["arrival_mode"] for r in requests} == {"fixed-rate"}
    assert sorted(r["request_index"] for r in requests) == [0, 1, 2, 3, 4]
    assert all(r["dispatch_lag_ms"] >= 0 for r in requests)
    case = report["cases"]["fixed50_in8_out4"]
    arrivals = case["arrivals"]
    assert (arrivals["offered"], arrivals["sent"], arrivals["completed"]) == (5, 5, 5)
    assert arrivals["peak_in_flight"] >= 4
    assert arrivals["offered_rate_per_second"] == pytest.approx(50.0)
    assert case["latency_ms"]["e2e_from_intended_p50"] >= 200.0


def test_duration_keeps_arrivals_before_the_window_closes(tmp_path: Path) -> None:
    requests, _report, _client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.01,
        arrival_mode="fixed-rate",
        rates=(20.0,),
        request_count=None,
        duration_seconds=0.24,
    )
    assert len(requests) == 5


def test_drop_overflow_records_unsent_requests(tmp_path: Path) -> None:
    requests, report, client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.3,
        arrival_mode="fixed-rate",
        rates=(100.0,),
        request_count=3,
        max_in_flight=1,
        overflow="drop",
    )
    assert client.max_active == 1
    statuses = sorted(r["status"] for r in requests)
    assert statuses == ["dropped", "dropped", "ok"]
    dropped = [r for r in requests if r["status"] == "dropped"]
    assert all(r["e2e_latency_ms"] is None for r in dropped)
    assert all("in-flight limit of 1" in r["error_message"] for r in dropped)
    arrivals = report["cases"]["fixed100_in8_out4"]["arrivals"]
    assert (arrivals["offered"], arrivals["sent"], arrivals["dropped"]) == (3, 1, 2)
    # A dropped arrival was never held: it never got a slot.
    assert arrivals["held_for_slot"] == 0
    assert report["summary"]["failures_by_status"] == {"dropped": 2}
    text = format_analysis_text(report)
    assert "arrivals: fixed-rate, offered 3, sent 1, dropped 2" in text


def test_wait_overflow_delays_arrivals_and_records_the_wait(tmp_path: Path) -> None:
    requests, report, client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.1,
        arrival_mode="burst",
        burst_size=3,
        burst_interval_seconds=10.0,
        request_count=3,
        max_in_flight=1,
    )
    assert client.max_active == 1
    ordered = sorted(requests, key=lambda r: r["request_index"])
    assert [r["held_for_slot"] for r in ordered] == [False, True, True]
    # All three were due at 0; each waited for the one before it to finish.
    lags = [r["dispatch_lag_ms"] for r in ordered]
    assert lags[0] < 50 and lags[1] >= 80 and lags[2] >= 180
    arrivals = report["cases"]["burst3x10s_in8_out4"]["arrivals"]
    assert arrivals["held_for_slot"] == 2
    assert arrivals["dispatch_lag_ms"]["max"] >= 180
    assert arrivals["peak_in_flight"] == 1
    text = format_analysis_text(report)
    assert "held for a slot 2" in text
    # The text report shows the wait the send-time latency leaves out.
    assert "E2E from due time p95" in text


def test_closed_loop_records_arrivals_without_an_offered_rate(tmp_path: Path) -> None:
    requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.01, concurrency=(2,), request_count=4
    )
    assert {r["arrival_mode"] for r in requests} == {"closed"}
    assert all(1 <= r["in_flight_at_dispatch"] <= 2 for r in requests)
    assert all(r["dispatch_lag_ms"] >= 0 for r in requests)
    arrivals = report["cases"]["c2_in8_out4"]["arrivals"]
    assert arrivals["offered_rate_per_second"] is None
    assert arrivals["mode"] == "closed"
    # Nothing unusual happened, so the text report adds no arrivals line.
    assert "arrivals:" not in format_analysis_text(report)


def test_open_loop_warmup_is_recorded_but_not_counted(tmp_path: Path) -> None:
    requests, report, _client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.01,
        arrival_mode="poisson",
        rates=(200.0,),
        request_count=4,
        warmup_requests=2,
    )
    phases = sorted(r["phase"] for r in requests)
    assert phases == ["measured"] * 4 + ["warmup"] * 2
    assert report["cases"]["poisson200_in8_out4"]["arrivals"]["offered"] == 4


def test_replay_from_the_cli_sends_every_recorded_arrival(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(f'{{"offset_ms": {ms}}}\n' for ms in (0, 10, 25)))
    output = tmp_path / "infer.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "stormlog.infer.profile.OpenAIChatCompletionsClient",
                lambda **_kwargs: SleepingClient(0.01),
            )
            code = infer_main(
                [
                    "profile",
                    "--endpoint",
                    "http://127.0.0.1:1/v1/chat/completions",
                    "--model",
                    "fake-model",
                    "--input-tokens",
                    "8",
                    "--output-tokens",
                    "4",
                    "--arrival",
                    "replay",
                    "--arrival-trace",
                    str(trace),
                    "--system-sampler",
                    "none",
                    "--tokenizer",
                    "none",
                    "--output",
                    str(output),
                ]
            )
    assert code == 0
    report = analyze_inference_events(output)
    arrivals = report["cases"]["replay_in8_out4"]["arrivals"]
    assert (arrivals["mode"], arrivals["offered"]) == ("replay", 3)
    sessions = [
        json.loads(line)
        for line in output.read_text().splitlines()
        if '"infer.session"' in line
    ]
    assert sessions[0]["config"]["arrivals"][0]["trace"]["arrivals"] == 3


def _windows(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "infer.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    return [r for r in records if r.get("event_type") == "infer.case_window"]


def test_requests_still_running_at_the_drain_deadline_are_cancelled(
    tmp_path: Path,
) -> None:
    requests, report, _client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.5,
        arrival_mode="fixed-rate",
        rates=(100.0,),
        request_count=2,
        drain_timeout_seconds=0.1,
    )
    assert [r["status"] for r in requests] == ["cancelled", "cancelled"]
    assert all(r["e2e_latency_ms"] is None for r in requests)
    arrivals = report["cases"]["fixed100_in8_out4"]["arrivals"]
    assert arrivals["sent"] == 2 and arrivals["failed"] == {"cancelled": 2}
    assert 0.08 <= arrivals["drain_seconds"] < 0.4
    assert report["summary"]["failures_by_status"] == {"cancelled": 2}
    (window,) = _windows(tmp_path)
    assert (window["phase"], window["scheduled_arrivals"]) == ("measured", 2)
    assert window["drain_timeout_seconds"] == 0.1


def test_a_closed_loop_duration_drains_then_cancels(tmp_path: Path) -> None:
    requests, report, _client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.4,
        request_count=None,
        duration_seconds=0.05,
        drain_timeout_seconds=0.05,
    )
    assert [r["status"] for r in requests] == ["cancelled"]
    arrivals = report["cases"]["c1_in8_out4"]["arrivals"]
    assert arrivals["window_seconds"] == pytest.approx(0.05)
    assert arrivals["failed"] == {"cancelled": 1}
    # A cancellation is worth showing even in a closed loop.
    assert "cancelled 1" in format_analysis_text(report)


def test_a_phase_waits_for_requests_an_earlier_drain_gave_up_on(
    tmp_path: Path,
) -> None:
    requests, _report, client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.01,
        first_latencies=(0.5,),
        arrival_mode="fixed-rate",
        rates=(10.0,),
        max_in_flight=1,
        request_count=1,
        warmup_requests=1,
        drain_timeout_seconds=0.05,
    )
    # The warmup request was given up on; the measured one still ran.
    statuses = {r["phase"]: r["status"] for r in requests}
    assert statuses == {"warmup": "cancelled", "measured": "ok"}
    assert client.calls == 2
    measured = [w for w in _windows(tmp_path) if w["phase"] == "measured"]
    abandoned = measured[0]["abandoned_requests"]
    assert abandoned["running_at_start"] == 1 and abandoned["still_running"] == 0
    assert abandoned["waited_seconds"] >= 0.3


def test_the_next_case_waits_after_a_closed_loop_duration(tmp_path: Path) -> None:
    requests, _report, _client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.01,
        first_latencies=(0.5,),
        input_tokens=(8, 16),
        request_count=None,
        duration_seconds=0.05,
        drain_timeout_seconds=0.05,
    )
    by_case: dict[str, list[str]] = {}
    for record in requests:
        by_case.setdefault(record["case_id"], []).append(record["status"])
    assert by_case["c1_in8_out4"] == ["cancelled"]
    assert "ok" in by_case["c1_in16_out4"]
    second = [w for w in _windows(tmp_path) if w["case_id"] == "c1_in16_out4"]
    assert second[0]["abandoned_requests"]["running_at_start"] == 1


def test_every_phase_records_its_window(tmp_path: Path) -> None:
    _requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.02, request_count=3, warmup_requests=1
    )
    windows = _windows(tmp_path)
    assert [w["phase"] for w in windows] == ["warmup", "measured"]
    for window in windows:
        assert (
            window["started_at_ns"]
            <= window["window_ended_at_ns"]
            <= window["drained_at_ns"]
        )
        # The default drain timeout is the request timeout.
        assert window["drain_timeout_seconds"] == 60.0
    arrivals = report["cases"]["c1_in8_out4"]["arrivals"]
    # The window closes when the last request is sent, then the drain runs.
    assert arrivals["drain_seconds"] >= 0.015
    assert arrivals["failed"] == {}


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--arrival", "poisson"], "--arrival poisson needs --rate"),
        (["--rate", "2"], "--rate does not apply to --arrival closed"),
        (["--arrival", "burst", "--burst-size", "2"], "needs --burst-interval"),
        (["--arrival", "replay"], "needs --arrival-trace"),
        (["--arrival-trace-case", "a"], "only applies to --arrival replay"),
        (
            ["--arrival", "fixed-rate", "--rate", "2", "--concurrency", "4"],
            "use --max-in-flight",
        ),
        (
            ["--arrival", "fixed-rate", "--rate", "2", "--max-in-flight", "0"],
            "--max-in-flight must be >= 1",
        ),
        (["--arrival", "fixed-rate", "--rate", "0"], "rate values must be > 0"),
        (["--duration", "1", "--requests", "2"], "either --duration or --requests"),
        (["--drain-timeout", "0"], "--drain-timeout must be > 0"),
    ],
)
def test_arrival_flags_are_checked_before_any_request(
    tmp_path: Path, flags: list[str], message: str
) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "fake-model",
                "--output",
                str(tmp_path / "never.jsonl"),
                *flags,
            ]
        )
    assert code == 1
    assert message in stderr.getvalue()
    assert not (tmp_path / "never.jsonl").exists()


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_dispatcher_drops_only_while_every_slot_is_busy() -> None:
    sent: list[Arrival] = []
    dropped: list[Arrival] = []

    async def scenario() -> None:
        limiter = InFlightLimiter(1)

        async def send(arrival: Arrival) -> None:
            sent.append(arrival)
            await asyncio.sleep(0.1)

        dispatch = await dispatch_schedule(
            [0.0, 0.02, 0.2],
            mode="fixed-rate",
            limiter=limiter,
            overflow="drop",
            send=send,
            drop=dropped.append,
        )
        await asyncio.gather(*dispatch.tasks)
        assert limiter.active == 0 and limiter.peak == 1

    _run(scenario())
    assert [a.index for a in sent] == [0, 2]
    assert [a.index for a in dropped] == [1]
    assert dropped[0].held_for_slot and dropped[0].in_flight_at_dispatch is None
    assert not sent[1].held_for_slot
    assert sent[1].intended_at_ns - sent[0].intended_at_ns == 200_000_000


def test_limiter_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        InFlightLimiter(0)


class _SlowThenRejectingClient:
    """The first call succeeds slowly; later calls are rejected at once."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        stream: bool,
        stream_include_usage: bool,
    ) -> ChatCompletionResult:
        self.calls += 1
        if self.calls > 1:
            raise EndpointHTTPError(429, "slow down")
        started_at_ns = time.time_ns()
        time.sleep(0.2)
        return ChatCompletionResult(
            text="ok",
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
            e2e_latency_ms=200.0,
            ttft_ms=None,
            first_chunk_latency_ms=None,
        )


def test_failures_are_timed_from_when_their_thread_starts(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    profiler = InferenceProfiler(
        ProfileConfig(
            endpoint="http://127.0.0.1:1/v1/chat/completions",
            model="fake-model",
            concurrency=(2,),
            input_tokens=(8,),
            output_tokens=(4,),
            request_count=2,
            output_path=str(output),
            stream=False,
            system_sampler="none",
            tokenizer="none",
        )
    )
    # One thread for two workers: the second request waits for the first.
    profiler.request_executor = ThreadPoolExecutor(max_workers=1)
    profiler.client = _SlowThenRejectingClient()  # type: ignore[assignment]
    profiler.run()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    rejected = [r for r in records if r.get("status") == "rejected"]
    assert len(rejected) == 1
    # The wait for the thread counts as dispatch lag, as it does for successes.
    assert rejected[0]["dispatch_lag_ms"] >= 150
    assert rejected[0]["e2e_latency_ms"] < 100


def _profile_cli(tmp_path: Path, *flags: str) -> tuple[int, str]:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "fake-model",
                "--timeout",
                "0.5",
                "--system-sampler",
                "none",
                "--tokenizer",
                "none",
                "--output",
                str(tmp_path / "infer.jsonl"),
                *flags,
            ]
        )
    return code, stderr.getvalue()


def test_earlier_command_lines_keep_their_meaning(tmp_path: Path) -> None:
    # --requests 1 was the default, so it could always be given with --duration.
    _code, stderr = _profile_cli(tmp_path, "--duration", "0.05", "--requests", "1")
    assert "either --duration or --requests" not in stderr
    assert (tmp_path / "infer.jsonl").exists()
    code, stderr = _profile_cli(tmp_path / "empty", "--concurrency", "")
    assert code == 1
    assert "concurrency must contain at least one value" in stderr
