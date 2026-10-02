"""``stormlog infer profile --vllm-metrics``: scrapes around every phase."""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import pytest
from jsonschema import Draft202012Validator

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import build_parser
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.vllm_scraper import (
    INTERRUPT_SCRAPE_TIMEOUT_SECONDS,
    VllmMetricsScraper,
    fetch_metrics,
    metrics_api_key,
    resolve_metrics_url,
)
from stormlog.infer.vllm_telemetry import (
    MARKER_INTERVAL,
    MARKER_PHASE_END,
    MARKER_PHASE_START,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "vllm"
VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "docs/schemas/inference_vllm_v1.schema.json").read_text())
)
METRICS_TEXT = (FIXTURES / "q05_c08_metrics_post.txt").read_text(encoding="utf-8")


class _FakeVllmHandler(BaseHTTPRequestHandler):
    """A chat endpoint plus a /metrics page; records the request-id headers."""

    protocol_version = "HTTP/1.1"
    seen_request_ids: list[str | None] = []
    seen_metrics_auth: list[str | None] = []
    metrics_status = 200
    metrics_body = METRICS_TEXT
    # GETs of /metrics past this many hang until the fixture releases them.
    metrics_hang_after: int | None = None
    metrics_release = threading.Event()
    metrics_gets = 0

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_error(404)
            return
        cls = type(self)
        cls.seen_metrics_auth.append(self.headers.get("Authorization"))
        cls.metrics_gets += 1
        if (
            cls.metrics_hang_after is not None
            and cls.metrics_gets > cls.metrics_hang_after
        ):
            cls.metrics_release.wait(timeout=30)
        body = cls.metrics_body.encode("utf-8")
        self.send_response(type(self).metrics_status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).seen_request_ids.append(self.headers.get("X-Request-Id"))
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _fake_vllm(
    *,
    metrics_status: int = 200,
    metrics_body: str = METRICS_TEXT,
    metrics_hang_after: int | None = None,
) -> Iterator[str]:
    _FakeVllmHandler.seen_request_ids = []
    _FakeVllmHandler.seen_metrics_auth = []
    _FakeVllmHandler.metrics_status = metrics_status
    _FakeVllmHandler.metrics_body = metrics_body
    _FakeVllmHandler.metrics_hang_after = metrics_hang_after
    _FakeVllmHandler.metrics_release = threading.Event()
    _FakeVllmHandler.metrics_gets = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVllmHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        _FakeVllmHandler.metrics_release.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(
    tmp_path: Path,
    origin: str,
    *,
    raises: type[BaseException] | None = None,
    **changes: Any,
) -> list[dict[str, Any]]:
    """Run a profile against the fake server and return its records plus warnings.

    ``raises`` is the exception the run is expected to end with, for a run
    that is interrupted.
    """
    output = tmp_path / "infer.jsonl"
    values: dict[str, Any] = {
        "endpoint": f"{origin}/v1/chat/completions",
        "model": "fake-model",
        "concurrency": (1,),
        "input_tokens": (8,),
        "output_tokens": (4,),
        "output_path": str(output),
        "stream": False,
        "request_count": 3,
        "warmup_requests": 1,
        "tokenizer": "none",
        "system_sampler": "none",
        "run_id": "run-1",
        "vllm_metrics_url": f"{origin}/metrics",
        "vllm_metrics_interval_seconds": 0.1,
    }
    values.update(changes)
    warnings: list[str] = []
    profiler = InferenceProfiler(ProfileConfig(**values), on_warning=warnings.append)
    if raises is None:
        profiler.run()
    else:
        with pytest.raises(raises):
            profiler.run()
    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.append({"event_type": "_warnings", "warnings": warnings})
    return records


def _of_type(records: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event_type") == event_type]


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hung_scrape_config(origin: str, output: Path) -> ProfileConfig:
    """A 5 s closed-loop phase scraped every 0.1 s with a 10 s request timeout."""
    return ProfileConfig(
        endpoint=f"{origin}/v1/chat/completions",
        model="fake-model",
        concurrency=(1,),
        input_tokens=(8,),
        output_tokens=(4,),
        output_path=str(output),
        stream=False,
        request_count=None,
        duration_seconds=5.0,
        warmup_requests=0,
        tokenizer="none",
        system_sampler="none",
        run_id="run-1",
        timeout_seconds=10.0,
        vllm_metrics_url=f"{origin}/metrics",
        vllm_metrics_interval_seconds=0.1,
    )


def _assert_interrupted_artifact(records: list[dict[str, Any]]) -> None:
    """The artifact of a run stopped while an interval scrape hung."""
    scrapes = _of_type(records, "infer.vllm_scrape")
    # The given-up interval scrape leaves no record at all; the end scrape
    # timed out on its own short clock.
    assert [s["marker"] for s in scrapes] == [MARKER_PHASE_START, MARKER_PHASE_END]
    assert scrapes[0]["status"] == "ok"
    assert scrapes[1]["status"] == "error" and "timed out" in scrapes[1]["error"]
    assert [r["event_type"] for r in records[-2:]] == [
        "infer.capabilities",
        "infer.session",
    ]
    assert records[-1]["status"] == "interrupted"


class TestProfileScrapes:
    def test_scrapes_bracket_every_phase_and_requests_carry_the_header(
        self, tmp_path: Path
    ) -> None:
        with _fake_vllm() as origin:
            records = _run(tmp_path, origin)
        scrapes = _of_type(records, "infer.vllm_scrape")
        for scrape in scrapes:
            VALIDATOR.validate(scrape)
        case_id = "c1_in8_out4"
        by_phase = {
            phase: [s["marker"] for s in scrapes if s["phase"] == phase]
            for phase in ("warmup", "measured")
        }
        assert by_phase["warmup"][0] == MARKER_PHASE_START
        assert by_phase["warmup"][-1] == MARKER_PHASE_END
        assert by_phase["measured"][0] == MARKER_PHASE_START
        assert by_phase["measured"][-1] == MARKER_PHASE_END
        assert all(s["case_id"] == case_id for s in scrapes)
        assert all(s["status"] == "ok" for s in scrapes)
        assert all(s["observation_scope"] == "engine_aggregate" for s in scrapes)
        assert all(s["interval_ms"] == 100 for s in scrapes)
        # Scrapes sit on the client's clock, like the phase windows.
        windows = _of_type(records, "infer.phase_window")
        measured = [s for s in scrapes if s["phase"] == "measured"]
        assert measured[0]["observed_at_ns"] <= windows[-1]["started_at_ns"]
        assert measured[-1]["observed_at_ns"] >= windows[-1]["window_ended_at_ns"]
        assert "vllm:kv_cache_usage_perc" in measured[0]["discovery"]["present"]
        assert "vllm:prompt_tokens_total" in measured[0]["scrape"]["values"]
        requests = _of_type(records, "infer.request")
        expected = {f"stormlog-run-1-{r['request_id']}" for r in requests}
        assert {r["x_request_id"] for r in requests} == expected
        assert set(_FakeVllmHandler.seen_request_ids) == expected
        session = _of_type(records, "infer.session")[0]
        assert session["config"]["vllm_metrics"] == {
            "url": f"{origin}/metrics",
            "interval_seconds": 0.1,
            "timeout_seconds": 60.0,
            "authorization": None,
        }
        capability = _of_type(records, "infer.capabilities")[0]
        assert capability["component"] == "vllm.metrics"
        assert capability["available"] is True
        assert "queue_depth" in capability["enabled"]
        assert "kv_cache_usage" in capability["collected"]
        assert set(capability["collected"]) <= set(capability["supported"])
        assert capability["metadata"]["per_request_attribution"] == "none"
        assert records[-1]["warnings"] == []

    def test_interval_scrapes_appear_in_a_longer_phase(self, tmp_path: Path) -> None:
        with _fake_vllm() as origin:
            records = _run(
                tmp_path,
                origin,
                request_count=None,
                duration_seconds=0.5,
                warmup_requests=0,
                drain_timeout_seconds=0.2,
            )
        markers = [s["marker"] for s in _of_type(records, "infer.vllm_scrape")]
        assert markers[0] == MARKER_PHASE_START
        assert markers[-1] == MARKER_PHASE_END
        assert markers.count(MARKER_INTERVAL) >= 2

    def test_failed_scrapes_are_recorded_and_the_run_still_completes(
        self, tmp_path: Path
    ) -> None:
        with _fake_vllm(metrics_status=503) as origin:
            records = _run(tmp_path, origin, warmup_requests=0)
        scrapes = _of_type(records, "infer.vllm_scrape")
        assert scrapes and all(s["status"] == "error" for s in scrapes)
        assert all(s["error"] == "HTTP 503" for s in scrapes)
        assert all(s["scrape"] is None for s in scrapes)
        for scrape in scrapes:
            VALIDATOR.validate(scrape)
        capability = _of_type(records, "infer.capabilities")[0]
        assert capability["available"] is False
        assert capability["enabled"] == []
        assert capability["metadata"]["failed_scrapes"] == len(scrapes)
        assert len(_of_type(records, "infer.request")) == 3
        warnings = records[-1]["warnings"]
        assert len(warnings) == 1 and "HTTP 503" in warnings[0]

    def test_unparseable_metrics_are_a_recorded_failure(self, tmp_path: Path) -> None:
        with _fake_vllm(metrics_body="<html>login</html>") as origin:
            records = _run(tmp_path, origin, warmup_requests=0)
        scrapes = _of_type(records, "infer.vllm_scrape")
        assert all(s["status"] == "error" for s in scrapes)
        assert all(s["error"].startswith("unparseable response") for s in scrapes)
        assert all(s["http_status"] == 200 for s in scrapes)

    def test_bearer_token_goes_only_to_the_endpoints_origin(
        self, tmp_path: Path
    ) -> None:
        with _fake_vllm() as origin:
            records = _run(tmp_path, origin, api_key="secret", warmup_requests=0)
            assert set(_FakeVllmHandler.seen_metrics_auth) == {"Bearer secret"}
        session = _of_type(records, "infer.session")[0]
        assert session["config"]["vllm_metrics"]["authorization"] == "bearer"
        assert records[-1]["warnings"] == []
        # A metrics page on another port is another origin: no token.
        with _fake_vllm() as origin, _fake_vllm() as other:
            records = _run(
                tmp_path,
                origin,
                api_key="secret",
                warmup_requests=0,
                vllm_metrics_url=f"{other}/metrics",
            )
            assert set(_FakeVllmHandler.seen_metrics_auth) == {None}
        session = _of_type(records, "infer.session")[0]
        assert session["config"]["vllm_metrics"]["authorization"] is None
        assert all(s["status"] == "ok" for s in _of_type(records, "infer.vllm_scrape"))
        warnings = records[-1]["warnings"]
        assert len(warnings) == 1 and "not sent" in warnings[0]

    def test_an_interrupted_run_still_scrapes_and_writes_capabilities(
        self, tmp_path: Path
    ) -> None:
        timeouts: list[float] = []

        def spy(url: str, **kwargs: Any) -> Any:
            timeouts.append(kwargs["timeout_seconds"])
            return fetch_metrics(url, **kwargs)

        with (
            _fake_vllm() as origin,
            mock.patch("stormlog.infer.vllm_scraper.fetch_metrics", spy),
            mock.patch.object(
                InferenceProfiler, "_run_arrivals", side_effect=asyncio.CancelledError
            ),
        ):
            records = _run(
                tmp_path, origin, warmup_requests=0, raises=asyncio.CancelledError
            )
        scrapes = _of_type(records, "infer.vllm_scrape")
        assert [s["marker"] for s in scrapes] == [MARKER_PHASE_START, MARKER_PHASE_END]
        assert all(s["status"] == "ok" for s in scrapes)
        # The end scrape of a cancelled phase runs on the short clock.
        assert timeouts == [60.0, INTERRUPT_SCRAPE_TIMEOUT_SECONDS]
        capabilities = _of_type(records, "infer.capabilities")
        assert [c["component"] for c in capabilities] == ["vllm.metrics"]
        assert capabilities[0]["available"] is True
        # Capabilities land before the session's last word, which says why.
        tail = [r["event_type"] for r in records[-3:-1]]
        assert tail == ["infer.capabilities", "infer.session"]
        assert records[-2]["status"] == "interrupted"

    def test_cancelling_during_a_hung_interval_scrape_unwinds_promptly(
        self, tmp_path: Path
    ) -> None:
        # The first GET of /metrics is answered (the phase-start scrape) and
        # every later one hangs, so the 0.1 s interval scrape is blocked when
        # the run's task is cancelled 0.6 s in. The phase waits 2 s for that
        # scrape, gives it up, and takes the end scrape on its own 2 s clock.
        output = tmp_path / "infer.jsonl"
        with _fake_vllm(metrics_hang_after=1) as origin:
            profiler = InferenceProfiler(_hung_scrape_config(origin, output))

            async def interrupt() -> float:
                run = asyncio.create_task(profiler._run_async())
                await asyncio.sleep(0.6)
                started = time.perf_counter()
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await run
                return time.perf_counter() - started

            try:
                unwound = asyncio.run(interrupt())
            finally:
                profiler.request_executor.shutdown(wait=True)
        # Two bounded waits, well short of the 10 s request timeout the hung
        # fetch would otherwise be waited for.
        assert unwound < 6.0
        _assert_interrupted_artifact(_records(output))

    def test_a_real_sigint_during_a_hung_scrape_still_ends_the_artifact(
        self, tmp_path: Path
    ) -> None:
        # Python 3.10's asyncio.run answers Ctrl+C by cancelling every task,
        # helpers included: a bare await of a helper inside a finally would
        # raise there and skip the end scrape, the span drain and the
        # capability records. The signal is a real one, from a timer thread.
        output = tmp_path / "infer.jsonl"
        with _fake_vllm(metrics_hang_after=1) as origin:
            profiler = InferenceProfiler(_hung_scrape_config(origin, output))
            timer = threading.Timer(0.6, signal.raise_signal, (signal.SIGINT,))
            started = time.perf_counter()
            timer.start()
            try:
                with pytest.raises(KeyboardInterrupt):
                    profiler.run()
            finally:
                timer.cancel()
            elapsed = time.perf_counter() - started
        # The end scrape's 2 s, plus the 0.6 s before the signal; nothing
        # waits for the hung fetch, not even interpreter shutdown.
        assert elapsed < 6.0
        _assert_interrupted_artifact(_records(output))

    def test_without_the_flag_nothing_is_scraped_or_sent_differently(
        self, tmp_path: Path
    ) -> None:
        with _fake_vllm() as origin:
            records = _run(tmp_path, origin, vllm_metrics_url=None)
        assert _of_type(records, "infer.vllm_scrape") == []
        assert _of_type(records, "infer.capabilities") == []
        assert _of_type(records, "infer.session")[0]["config"]["vllm_metrics"] is None
        # The header is sent whether or not metrics are scraped.
        assert all(h is not None for h in _FakeVllmHandler.seen_request_ids)


class TestScraperUnits:
    def test_fetch_reports_connection_failures_as_results(self) -> None:
        result = fetch_metrics(
            f"http://127.0.0.1:{_free_port()}/metrics", timeout_seconds=0.5
        )
        assert result.text is None
        assert result.http_status is None
        assert result.error is not None
        assert result.duration_ms >= 0

    def test_scraper_warns_once_and_counts(self) -> None:
        warnings: list[str] = []
        scraper = VllmMetricsScraper(
            url=f"http://127.0.0.1:{_free_port()}/metrics",
            interval_seconds=1.0,
            timeout_seconds=0.5,
            session_id="s",
            run_id="r",
            clock_domain="host/boot/unix_epoch_ns",
            on_warning=warnings.append,
        )
        first = scraper.scrape(marker=MARKER_PHASE_START)
        second = scraper.scrape(marker=MARKER_PHASE_END)
        assert first.status == second.status == "error"
        assert (scraper.ok_scrapes, scraper.failed_scrapes) == (0, 2)
        assert len(warnings) == 1
        VALIDATOR.validate(first.to_record())

    def test_metrics_api_key_stays_on_the_endpoints_origin(self) -> None:
        endpoint = "http://host:8000/v1/chat/completions"
        warnings: list[str] = []
        same = "http://host:8000/metrics"
        assert metrics_api_key(endpoint, same, "k", warnings.append) == "k"
        assert (
            metrics_api_key(endpoint, "HTTP://HOST:8000/m", "k", warnings.append) == "k"
        )
        assert metrics_api_key(endpoint, "http://other:8000/m", None) is None
        # A default port written out, or left out, is the same origin.
        for same_origin in (
            ("https://host/v1", "https://host:443/metrics"),
            ("https://host:443/v1", "https://host/metrics"),
            ("http://host/v1", "http://host:80/metrics"),
            ("http://host:80/v1", "HTTP://Host/metrics"),
        ):
            assert metrics_api_key(*same_origin, "k", warnings.append) == "k"
        assert warnings == []
        for other in (
            "http://host:9000/metrics",
            "https://host:8000/metrics",
            "http://other:8000/metrics",
        ):
            assert metrics_api_key(endpoint, other, "k", warnings.append) is None
        # Scheme and port each count: http on 80 is not https on 443.
        assert metrics_api_key("https://host/v1", "http://host/metrics", "k") is None
        assert metrics_api_key("https://host/v1", "https://host:80/m", "k") is None
        assert len(warnings) == 3 and all("not sent" in w for w in warnings)

    def test_resolve_metrics_url(self) -> None:
        endpoint = "http://host:8000/v1/chat/completions"
        assert resolve_metrics_url(endpoint, None) is None
        assert resolve_metrics_url(endpoint, "auto") == "http://host:8000/metrics"
        assert resolve_metrics_url(endpoint, "http://other/m") == "http://other/m"
        with pytest.raises(ValueError):
            resolve_metrics_url("not a url", "auto")


class TestCli:
    def test_flag_without_url_derives_the_metrics_url(self) -> None:
        args = build_parser().parse_args(
            [
                "profile",
                "--base-url",
                "http://h:8000/v1",
                "--model",
                "m",
                "--output",
                "out.jsonl",
                "--vllm-metrics",
            ]
        )
        assert args.vllm_metrics == "auto"
        assert args.vllm_metrics_interval is None
        args = build_parser().parse_args(
            [
                "profile",
                "--base-url",
                "http://h:8000/v1",
                "--model",
                "m",
                "--vllm-metrics",
                "http://h:9000/metrics",
                "--vllm-metrics-interval",
                "0.5",
                "--output",
                "out.jsonl",
            ]
        )
        assert args.vllm_metrics == "http://h:9000/metrics"
        assert args.vllm_metrics_interval == 0.5

    @pytest.mark.parametrize(
        "extra",
        [
            ["--vllm-metrics", "ftp://h/metrics"],
            ["--vllm-metrics", "--vllm-metrics-interval", "0.01"],
            ["--vllm-metrics", "--vllm-metrics-interval", "nan"],
            ["--vllm-metrics-interval", "2"],
        ],
    )
    def test_bad_flags_are_usage_errors(self, extra: list[str], tmp_path: Path) -> None:
        argv = [
            "profile",
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--model",
            "m",
            "--output",
            str(tmp_path / "out.jsonl"),
            "--tokenizer",
            "none",
            *extra,
        ]
        with mock.patch("sys.stderr"):
            assert infer_main(argv) == int(ExitCode.USAGE)
        assert not (tmp_path / "out.jsonl").exists()
