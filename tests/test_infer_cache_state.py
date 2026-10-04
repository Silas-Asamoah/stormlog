"""Requested and verified prefix-cache state, and the reset hook."""

import contextlib
import io
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.analysis import format_analysis_text
from stormlog.infer.cache_state import (
    CacheReset,
    cache_state_record,
    cache_summary,
    reset_cache,
    run_kind,
)
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler
from tests.infer_workload_helpers import SleepingClient, run_profile_with_fake_client


class _ResetHandler(BaseHTTPRequestHandler):
    calls: list[str] = []
    authorizations: list[str | None] = []
    called_at: list[float] = []

    def do_POST(self) -> None:  # noqa: N802
        type(self).calls.append(self.path)
        type(self).called_at.append(time.time())
        type(self).authorizations.append(self.headers.get("Authorization"))
        status = 200 if self.path == "/reset_prefix_cache" else 404
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _reset_server() -> Iterator[str]:
    _ResetHandler.calls = []
    _ResetHandler.authorizations = []
    _ResetHandler.called_at = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _cache_records(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "infer.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    return [r for r in records if r.get("event_type") == "infer.cache_state"]


def test_reset_records_success_http_errors_and_unreachable_servers() -> None:
    with _reset_server() as base:
        ok = reset_cache(f"{base}/reset_prefix_cache", timeout_seconds=5)
        missing = reset_cache(f"{base}/flush", timeout_seconds=5)
    assert (ok.status, ok.error, ok.succeeded) == (200, None, True)
    assert (missing.status, missing.error, missing.succeeded) == (
        404,
        "HTTP 404",
        False,
    )
    refused = reset_cache("http://127.0.0.1:1/reset", timeout_seconds=1)
    assert refused.status is None and not refused.succeeded
    assert refused.error is not None and "URLError" in refused.error


@pytest.mark.parametrize(
    ("requested", "warmup", "kind"),
    [
        ("cold", 0, "cold_start"),
        ("cold", 2, "steady_state"),
        ("unspecified", 2, "steady_state"),
        ("unspecified", 0, "unspecified"),
    ],
)
def test_run_kind_separates_cold_starts_from_steady_states(
    requested: str, warmup: int, kind: str
) -> None:
    assert run_kind(requested, warmup) == kind


def test_a_failed_reset_is_not_labelled_a_cold_start() -> None:
    failed = CacheReset("http://host/reset", at_ns=1, status=404, error="HTTP 404")
    succeeded = CacheReset("http://host/reset", at_ns=1, status=200)
    assert run_kind("cold", 0, failed) == "unspecified"
    assert run_kind("cold", 0, succeeded) == "cold_start"


def test_each_case_resets_the_cache_and_records_its_state(tmp_path: Path) -> None:
    with _reset_server() as base:
        _requests, report, _client = run_profile_with_fake_client(
            tmp_path,
            latency_seconds=0.0,
            request_count=2,
            input_tokens=(8, 16),
            cache_state="cold",
            cache_reset_url=f"{base}/reset_prefix_cache",
        )
        assert _ResetHandler.calls == ["/reset_prefix_cache"] * 2
    records = _cache_records(tmp_path)
    assert [r["case_id"] for r in records] == ["c1_in8_out4", "c1_in16_out4"]
    for record in records:
        assert record["reset"]["status"] == 200
        assert (record["requested"], record["verified"]) == ("cold", "unverified")
        assert record["run_kind"] == "cold_start"
        assert "no engine adapter can confirm" in record["reason"]
    cache = report["cases"]["c1_in8_out4"]["cache"]
    assert (cache["requested"], cache["run_kind"]) == ("cold", "cold_start")
    assert cache["reset"]["status"] == 200
    text = format_analysis_text(report)
    assert "cache: cold requested, unverified" in text
    assert "run kind cold_start" in text


def test_a_failed_reset_is_recorded_not_hidden(tmp_path: Path) -> None:
    with _reset_server() as base:
        _requests, report, _client = run_profile_with_fake_client(
            tmp_path,
            latency_seconds=0.0,
            request_count=1,
            warmup_requests=1,
            cache_state="cold",
            cache_reset_url=f"{base}/flush",
        )
    (record,) = _cache_records(tmp_path)
    assert record["reset"]["status"] == 404
    assert record["reason"] == "the cache reset failed (HTTP 404)"
    # Warmup ran after the reset, so the case is a steady state.
    assert record["run_kind"] == "steady_state"
    assert "the cache reset failed" in format_analysis_text(report)


def test_cases_without_a_requested_state_say_so_quietly(tmp_path: Path) -> None:
    _requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.0, request_count=1
    )
    (record,) = _cache_records(tmp_path)
    assert (record["requested"], record["reset"]) == ("unspecified", None)
    assert record["reason"].startswith("no cache state was requested; earlier")
    assert "cache:" not in format_analysis_text(report)


def test_older_artifacts_report_an_unrecorded_cache_state() -> None:
    summary = cache_summary(None)
    assert (summary["requested"], summary["verified"]) == ("unspecified", "unverified")
    assert summary["reason"] == "the artifact does not record a cache state"


def _cli(tmp_path: Path, *flags: str) -> tuple[int, str]:
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


def test_cli_warns_when_a_cold_cache_has_no_reset(tmp_path: Path) -> None:
    _code, stderr = _cli(tmp_path, "--cache-state", "cold")
    assert "--cache-state cold without --cache-reset-url" in stderr


def test_cli_rejects_a_reset_url_that_is_not_http(tmp_path: Path) -> None:
    code, stderr = _cli(tmp_path, "--cache-reset-url", "file:///tmp/reset")
    assert code == ExitCode.USAGE
    assert "--cache-reset-url must use http:// or https://" in stderr
    assert not (tmp_path / "infer.jsonl").exists()


def test_resets_carry_the_api_key(tmp_path: Path) -> None:
    with _reset_server() as base:
        run_profile_with_fake_client(
            tmp_path,
            latency_seconds=0.0,
            request_count=1,
            api_key="sk-test",
            cache_state="cold",
            cache_reset_url=f"{base}/reset_prefix_cache",
        )
        assert _ResetHandler.authorizations == ["Bearer sk-test"]
        reset_cache(f"{base}/reset_prefix_cache", timeout_seconds=5)
        assert _ResetHandler.authorizations[-1] is None


def test_a_failed_reset_is_reported_as_a_warning(tmp_path: Path) -> None:
    warnings: list[str] = []
    with _reset_server() as base:
        profiler = InferenceProfiler(
            ProfileConfig(
                endpoint="http://127.0.0.1:1/v1/chat/completions",
                model="fake-model",
                concurrency=(1,),
                input_tokens=(8,),
                output_tokens=(4,),
                request_count=1,
                output_path=str(tmp_path / "infer.jsonl"),
                stream=False,
                system_sampler="none",
                tokenizer="none",
                cache_state="cold",
                cache_reset_url=f"{base}/flush",
            ),
            on_warning=warnings.append,
        )
        profiler.client = SleepingClient(0.0)  # type: ignore[assignment]
        profiler.run()
    assert warnings == [
        "cache reset failed before c1_in8_out4 (HTTP 404); the case is not a "
        "cold start"
    ]


@pytest.mark.parametrize(
    ("flags", "warned"),
    [
        (["--prompt-mode", "unique"], True),
        (
            [
                "--prompt-mode",
                "unique",
                "--cache-reset-url",
                "http://127.0.0.1:1/reset_prefix_cache",
            ],
            False,
        ),
        ([], False),
    ],
)
def test_cli_warns_that_repeated_seeds_send_cached_prompts(
    tmp_path: Path, flags: list[str], warned: bool
) -> None:
    _code, stderr = _cli(tmp_path, *flags)
    assert ("prompts are the same for every run with --seed 0" in stderr) is warned


def test_a_reset_waits_for_calls_the_last_case_gave_up_on(tmp_path: Path) -> None:
    with _reset_server() as base:
        _requests, _report, client = run_profile_with_fake_client(
            tmp_path,
            latency_seconds=0.01,
            first_latencies=(0.5,),
            input_tokens=(8, 16),
            request_count=None,
            duration_seconds=0.05,
            drain_timeout_seconds=0.05,
            cache_state="cold",
            cache_reset_url=f"{base}/reset_prefix_cache",
        )
        resets = list(_ResetHandler.called_at)
    # The first case's only call was abandoned; the second reset follows it.
    assert len(resets) == 2
    assert resets[1] >= client.finished_at[0]


class _AnsweringResetHandler(BaseHTTPRequestHandler):
    """Answers each reset with the next body in ``answers``, then the last."""

    answers: list[bytes] = []
    calls = 0
    started: list[float] = []

    def do_POST(self) -> None:  # noqa: N802
        cls = type(self)
        cls.started.append(time.monotonic())
        body = cls.answers[min(cls.calls, len(cls.answers) - 1)]
        cls.calls += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _answering_server(*answers: bytes) -> Iterator[str]:
    _AnsweringResetHandler.answers = list(answers)
    _AnsweringResetHandler.calls = 0
    _AnsweringResetHandler.started = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnsweringResetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/reset_prefix_cache"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_a_reset_vllm_acknowledges_is_acknowledged() -> None:
    with _answering_server(b'{"success": true}') as url:
        reset = reset_cache(url, timeout_seconds=5)
    assert (reset.status, reset.success, reset.answer) == (200, True, "acknowledged")
    assert reset.succeeded and reset.acknowledged
    assert reset.attempts == 1


def test_a_held_reset_is_retried_until_vllm_acknowledges_it() -> None:
    held = b'{"success": false}'
    with _answering_server(held, held, b'{"success": true}') as url:
        reset = reset_cache(url, timeout_seconds=5, retry_seconds=5)
    assert reset.acknowledged
    assert reset.attempts == 3
    assert reset.error is None


def test_a_reset_vllm_keeps_refusing_is_a_failed_reset() -> None:
    # vLLM answers HTTP 200 with success false while blocks are held; a 2xx
    # alone is not a reset.
    with _answering_server(b'{"success": false}') as url:
        reset = reset_cache(url, timeout_seconds=5, retry_seconds=0.6)
    assert reset.status == 200
    assert reset.answer == "refused"
    assert not reset.succeeded and not reset.acknowledged
    assert reset.attempts >= 2
    assert reset.error == f"refused: success false on {reset.attempts} attempts"
    assert run_kind("cold", 0, reset) == "unspecified"
    record = cache_state_record(
        session_id="s", case_id="c1", requested="cold", reset=reset, warmup_requests=0
    )
    assert record["acknowledged"] is False
    assert record["reason"].startswith("the cache reset failed (refused")


def test_the_record_says_when_the_acknowledged_reset_ran() -> None:
    held = b'{"success": false}'
    with _answering_server(held, held, b'{"success": true}') as url:
        reset = reset_cache(url, timeout_seconds=5, retry_seconds=5)
    assert reset.acknowledged
    # Two refusals half a second apart came before the reset that ran.
    assert reset.answered_at_ns is not None
    assert reset.answered_at_ns - reset.at_ns >= 900_000_000
    assert reset.to_record()["answered_at_ns"] == reset.answered_at_ns


@pytest.mark.parametrize(("budget", "attempts"), [(0.2, 1), (1.0, 2)])
def test_no_retry_starts_after_the_reset_timeout(budget: float, attempts: int) -> None:
    with _answering_server(b'{"success": false}') as url:
        began = time.monotonic()
        reset = reset_cache(url, timeout_seconds=5, retry_seconds=budget)
    assert reset.attempts == attempts
    assert all(start - began <= budget for start in _AnsweringResetHandler.started)


@pytest.mark.parametrize(
    "body",
    [
        b'{"success": "false"}',
        b'{"success": 0}',
        b'{"success": null}',
        b'{"success": "true"}',
    ],
)
def test_a_success_field_that_is_not_true_is_a_refusal(body: bytes) -> None:
    # It answered the question, and not with true: no cold start.
    with _answering_server(body) as url:
        reset = reset_cache(url, timeout_seconds=5, retry_seconds=0)
    assert reset.answer == "refused"
    assert not reset.succeeded
    assert run_kind("cold", 0, reset) == "unspecified"


def test_the_report_says_whether_a_reset_was_attempted() -> None:
    reset = CacheReset("http://host/reset", at_ns=1, status=200, success=True)
    record = cache_state_record(
        session_id="s", case_id="c1", requested="cold", reset=reset, warmup_requests=0
    )
    assert cache_summary(record)["attempted"] is True
    assert cache_summary({"requested": "cold"})["attempted"] is None  # older
    assert cache_summary(None)["attempted"] is None


@pytest.mark.parametrize("body", [b"", b"Cache flushed.", b'{"ok": true}', b"[]"])
def test_a_2xx_without_a_success_field_is_accepted_but_unconfirmed(
    body: bytes,
) -> None:
    with _answering_server(body) as url:
        reset = reset_cache(url, timeout_seconds=5)
    assert reset.answer == "accepted_unverified"
    assert reset.succeeded and not reset.acknowledged
    record = cache_state_record(
        session_id="s", case_id="c1", requested="cold", reset=reset, warmup_requests=0
    )
    assert record["attempted"] is True and record["acknowledged"] is False
    assert "without saying whether it succeeded" in record["reason"]
    assert record["run_kind"] == "cold_start"


def test_an_acknowledged_reset_says_so_in_the_record() -> None:
    reset = CacheReset("http://host/reset", at_ns=1, status=200, success=True)
    record = cache_state_record(
        session_id="s", case_id="c1", requested="cold", reset=reset, warmup_requests=0
    )
    assert record["reset"]["answer"] == "acknowledged"
    assert record["acknowledged"] is True
    assert record["reason"] == (
        "the server acknowledged the cache reset, but no engine adapter can "
        "confirm it"
    )


def test_a_case_without_a_reset_was_not_attempted() -> None:
    record = cache_state_record(
        session_id="s", case_id="c1", requested="cold", reset=None, warmup_requests=0
    )
    assert (record["attempted"], record["acknowledged"]) == (False, False)


def test_the_profile_retries_a_held_reset_for_the_configured_time(
    tmp_path: Path,
) -> None:
    with _answering_server(b'{"success": false}') as url:
        run_profile_with_fake_client(
            tmp_path,
            latency_seconds=0.0,
            request_count=1,
            cache_state="cold",
            cache_reset_url=url,
            cache_reset_timeout_seconds=0.0,
        )
    (record,) = _cache_records(tmp_path)
    # No time to retry: the one refusal is the answer.
    assert record["reset"]["answer"] == "refused"
    assert record["reset"]["attempts"] == 1
    assert _AnsweringResetHandler.calls == 1
    lines = (tmp_path / "infer.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    session = next(r for r in records if r.get("event_type") == "infer.session")
    assert session["config"]["cache_reset_timeout_seconds"] == 0.0


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_cli_rejects_a_reset_timeout_it_cannot_wait(tmp_path: Path, value: str) -> None:
    code, stderr = _cli(tmp_path, "--cache-reset-timeout", value)
    assert code == ExitCode.USAGE
    assert "--cache-reset-timeout must be a finite number >= 0" in stderr
    assert not (tmp_path / "infer.jsonl").exists()
