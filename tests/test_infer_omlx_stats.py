"""Read-only oMLX statistics without admin payload/credential retention."""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stormlog.infer.config import ProfileConfig
from stormlog.infer.omlx_stats import OmlxStatsSampler, normalize_stats, stats_identity
from stormlog.infer.profile import InferenceProfiler

PINNED_RESPONSE = {
    "total_requests": 2,
    "total_prompt_tokens": 10,
    "total_completion_tokens": 4,
    "total_tokens_served": 14,
    "total_cached_tokens": 5,
    "uptime_seconds": 100,
    "avg_generation_tps": 20,
    "avg_prefill_tps": 40,
    "cache_efficiency": 50,
    "api_key": "NEVER-PERSIST-THIS-ADMIN-KEY",
    "cli_prefix": "private command",
    "active_models": {
        "total_active_requests": 1,
        "total_waiting_requests": 2,
        "models": [{"input": "private prompt"}],
    },
}


@contextlib.contextmanager
def endpoint(*, failing=False, cookie_headers=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(("GET", self.path, self.headers.get("Authorization")))
            if self.path != "/admin/api/stats":
                self.send_response(404)
                self.end_headers()
                return
            if cookie_headers is not None:
                cookie_headers.append(self.headers.get("Cookie"))
            self.send_response(401 if failing else 200)
            self.end_headers()
            self.wfile.write(json.dumps(PINNED_RESPONSE).encode())

        def do_POST(self):
            calls.append(("POST", self.path, self.headers.get("Authorization")))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')

        def log_message(self, *_):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def sampler(url, **kwargs):
    return OmlxStatsSampler(
        url=url + "/admin/api/stats",
        api_key=kwargs.pop("api_key", None),
        interval=0.01,
        timeout=1,
        session_id="session",
        run_id="run",
        server_version=None,
        **kwargs,
    )


def test_units_scope_and_no_sensitive_fields():
    identity = stats_identity(
        "http://localhost/admin/api/stats?scope=alltime&model=opaque"
    )
    metrics = normalize_stats(PINNED_RESPONSE, identity)
    by_path = {metric["raw_path"]: metric for metric in metrics}
    assert by_path["total_completion_tokens"]["unit"] == "tokens"
    assert by_path["total_completion_tokens"]["scope"] == "model"
    assert by_path["total_completion_tokens"]["counter_scope"] == "alltime"
    assert by_path["active_models.total_waiting_requests"]["scope"] == "server"
    assert "private" not in json.dumps(metrics) and "api_key" not in json.dumps(metrics)
    for value in (float("nan"), -1, True, "3"):
        assert normalize_stats({"total_requests": value}, identity)[4]["value"] is None


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin/api/stats/clear",
        "http://localhost/admin/api/stats?api_key=secret",
        "http://user:secret@localhost/admin/api/stats",
        "http://localhost/admin/api/stats?scope=invalid",
    ],
)
def test_read_only_preflight(url):
    with pytest.raises(ValueError):
        stats_identity(url)


def test_stats_credentials_never_inherit_inference_key(tmp_path):
    with endpoint() as (url, calls):
        config = ProfileConfig(
            endpoint=url + "/v1/chat/completions",
            model="opaque",
            concurrency=(1,),
            input_tokens=(16,),
            output_tokens=(4,),
            request_count=1,
            output_path=str(tmp_path / "run.jsonl"),
            engine="omlx",
            stream=False,
            tokenizer="none",
            system_sampler="none",
            api_key="inference-key",
            omlx_stats_url=url + "/admin/api/stats",
            omlx_stats_api_key="admin-key",
        )
        report = InferenceProfiler(config).run()
    assert report["telemetry"]["mlx"]["omlx_stats"]["samples"] >= 2
    assert report["summary"]["successful_requests"] == 1
    assert [key for method, path, key in calls if path == "/admin/api/stats"] == [
        "Bearer admin-key",
        "Bearer admin-key",
    ]
    assert [key for method, _, key in calls if method == "POST"] == [
        "Bearer inference-key"
    ]
    assert all(
        key == "Bearer inference-key"
        for method, path, key in calls
        if method == "GET" and path != "/admin/api/stats"
    )
    artifact = Path(config.output_path).read_text()
    assert "admin-key" not in artifact and "inference-key" not in artifact
    assert (
        PINNED_RESPONSE["api_key"] not in artifact and "private prompt" not in artifact
    )


def test_stats_auth_failure_does_not_fail_inference(tmp_path):
    with endpoint(failing=True) as (url, calls):
        config = ProfileConfig(
            endpoint=url + "/v1/chat/completions",
            model="opaque",
            concurrency=(1,),
            input_tokens=(16,),
            output_tokens=(4,),
            output_path=str(tmp_path / "run.jsonl"),
            engine="omlx",
            stream=False,
            tokenizer="none",
            system_sampler="none",
            api_key="inference-key",
            omlx_stats_url=url + "/admin/api/stats",
        )
        report = InferenceProfiler(config).run()
    assert report["summary"]["successful_requests"] == 1
    assert all(key is None for _, path, key in calls if path == "/admin/api/stats")
    stats = [
        row
        for row in map(json.loads, Path(config.output_path).read_text().splitlines())
        if row.get("event_type") == "infer.mlx.stats"
    ]
    assert all(row["status"] == "error" and row["http_status"] == 401 for row in stats)


def test_reset_and_outage_are_distinct_from_negative_throughput():
    capture = sampler("http://localhost")
    identity = stats_identity(capture.url)
    first = {"metrics": normalize_stats(PINNED_RESPONSE, identity), "status": "ok"}
    capture._epoch(first)
    assert first["epoch_change"] is None
    outage = {"metrics": [], "status": "error"}
    capture._epoch(outage)
    reset = {
        "metrics": normalize_stats({**PINNED_RESPONSE, "total_requests": 0}, identity),
        "status": "ok",
    }
    capture._epoch(reset)
    assert reset["epoch_change"] == "counter_reset_or_restart"
    assert reset["preceded_by_outage"]
    restart = {
        "metrics": normalize_stats({**PINNED_RESPONSE, "uptime_seconds": 1}, identity),
        "status": "ok",
    }
    capture._epoch(restart)
    assert restart["epoch_change"] == "restart_or_uptime_reset"


def test_cancelled_scrape_is_recorded_without_late_append(monkeypatch):
    import asyncio
    import time

    capture = sampler("http://localhost")

    def delayed(*args):
        time.sleep(0.05)
        return {"should_never_append": True}

    monkeypatch.setattr(capture, "fetch", delayed)
    records = []

    async def run():
        task = asyncio.create_task(
            capture.sample(records.append, "case", "measured", "interval")
        )
        await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert len(records) == 1 and records[0]["status"] == "abandoned"
        await asyncio.sleep(0.07)
        assert len(records) == 1

    asyncio.run(run())


def test_admin_session_cookie_is_separate_and_never_retained(tmp_path):
    cookies = []
    token = "normally-issued-admin-session-token"
    with endpoint(cookie_headers=cookies) as (url, calls):
        config = ProfileConfig(
            endpoint=url + "/v1/chat/completions",
            model="model",
            engine="omlx",
            concurrency=(1,),
            input_tokens=(4,),
            output_tokens=(2,),
            output_path=str(tmp_path / "run.jsonl"),
            stream=False,
            request_count=1,
            tokenizer="none",
            system_sampler="none",
            api_key="inference-key",
            omlx_stats_url=url + "/admin/api/stats",
            omlx_stats_cookie=token,
        )
        InferenceProfiler(config).run()
    assert cookies and all(
        cookie == "omlx_admin_session=" + token for cookie in cookies
    )
    assert all(auth is None for _, path, auth in calls if path == "/admin/api/stats")
    assert all(
        auth == "Bearer inference-key" for method, _, auth in calls if method == "POST"
    )
    assert token not in (tmp_path / "run.jsonl").read_text()


@pytest.mark.parametrize(
    "cookie", ["bad\r\nheader", "cookie;other=value", "cookie=value", " ", "x" * 8193]
)
def test_invalid_admin_cookie_fails_before_traffic(cookie):
    with pytest.raises(ValueError):
        OmlxStatsSampler(
            url="http://127.0.0.1:1/admin/api/stats",
            api_key=None,
            cookie=cookie,
            interval=1,
            timeout=1,
            session_id="s",
            run_id="r",
            server_version=None,
        )


def test_interrupted_phase_does_not_start_a_final_scrape(monkeypatch):
    import asyncio
    import time

    capture = sampler("http://localhost")
    calls = []

    async def sample(append, case, phase, marker):
        calls.append(marker)
        if marker == "phase_end":
            await asyncio.sleep(0.3)

    monkeypatch.setattr(capture, "sample", sample)

    async def run():
        entered = asyncio.Event()

        async def phase():
            async with capture.phase(list().append, "case", "measured"):
                entered.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(phase())
        await entered.wait()
        started = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.perf_counter() - started < 0.15

    asyncio.run(run())
    assert calls == ["phase_start"]


def test_backwards_wall_clock_remains_analyzable(monkeypatch):
    import stormlog.infer.omlx_stats as module
    from stormlog.infer.omlx_stats import stats_report

    capture = sampler("http://localhost")
    monkeypatch.setattr(
        capture,
        "_fetch_metrics",
        lambda _: normalize_stats(PINNED_RESPONSE, capture.identity),
    )
    stamps = iter([2_000_000_000, 1_000_000_000])
    monkeypatch.setattr(module.time, "time_ns", lambda: next(stamps))
    record = capture.observe("case", "measured", "phase_start")
    assert record["scrape_started_at_ns"] > record["scrape_ended_at_ns"]
    assert record["scrape_elapsed_ns"] >= 0
    assert record["clock_uncertainty_ns"] >= 1_000_000_000
    assert stats_report([record])["failed_samples"] == 0


def test_observation_outside_either_wall_clock_endpoint_is_invalid(monkeypatch):
    from stormlog.infer.omlx_stats import validate_stats_record

    capture = sampler("http://localhost")
    monkeypatch.setattr(capture, "_fetch_metrics", lambda _: [])
    record = capture.observe("case", "measured", "phase_start")
    record["observed_at_ns"] = record["scrape_ended_at_ns"] + 1
    with pytest.raises(ValueError, match="time window"):
        validate_stats_record(record)


def test_cancelled_interval_does_not_hold_asyncio_runner_open(monkeypatch):
    import asyncio
    import time

    capture = sampler("http://localhost")
    entered, release = threading.Event(), threading.Event()
    records = []
    original = capture.fetch
    monkeypatch.setattr(capture, "_fetch_metrics", lambda _: [])

    def fetch(case, phase, marker):
        if marker == "interval":
            entered.set()
            release.wait(timeout=0.6)
        return original(case, phase, marker)

    monkeypatch.setattr(capture, "fetch", fetch)

    async def run():
        async def phase():
            async with capture.phase(records.append, "case", "measured"):
                await asyncio.sleep(60)

        task = asyncio.create_task(phase())
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.perf_counter()
    try:
        asyncio.run(run())
        assert time.perf_counter() - started < 0.3
    finally:
        release.set()
    assert [r["marker"] for r in records] == ["phase_start", "interval"]
    assert records[-1]["status"] == "abandoned"


def test_stats_header_trickle_cannot_outlive_the_scrape_deadline():
    import time

    class DribblingHeaders(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                for _ in range(30):
                    self.connection.sendall(b"x")
                    time.sleep(0.03)
                self.connection.sendall(b"\r\nContent-Length: 2\r\n\r\n{}")
            except OSError:
                pass

        def log_message(self, *_):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), DribblingHeaders)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        capture = OmlxStatsSampler(
            url=f"http://127.0.0.1:{http.server_port}/admin/api/stats",
            api_key=None,
            interval=1,
            timeout=0.1,
            session_id="s",
            run_id="r",
            server_version=None,
        )
        started = time.perf_counter()
        record = capture.observe("case", "measured", "phase_start")
        assert time.perf_counter() - started < 0.5
        assert record["status"] == "error"
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)
