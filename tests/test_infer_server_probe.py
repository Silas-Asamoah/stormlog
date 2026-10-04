"""Probing what a vLLM server reports about itself."""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer import server_probe
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.errors import InferInputError
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.server_probe import (
    AFTER,
    MODELS,
    SERVER_INFO,
    VERSION,
    ProbeAnswer,
    ServerProbe,
    endpoint_origin,
    is_private_host,
    probe_server,
)
from tests.infer_workload_helpers import run_profile_with_fake_client

SECRET = "hf_plantedSecret0123456789"
SERVER_INFO_BODY = {
    "vllm_config": {
        "model_config": {"model": "Qwen/Qwen2.5-0.5B", "hf_token": SECRET},
        "scheduler_config": {"max_num_batched_tokens": 8192},
    },
    "vllm_env": {"VLLM_PORT": 8000, "VLLM_HTTP_TOKEN": SECRET},
    "system_env": {"torch_version": "2.9.0", "env_vars": f"HF_TOKEN={SECRET}"},
}


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, Any] = {}
    seen: list[tuple[str, str | None]] = []
    server_info_calls = 0

    def do_GET(self) -> None:  # noqa: N802
        cls = type(self)
        cls.seen.append((self.path, self.headers.get("Authorization")))
        action = cls.routes.get(self.path)
        if self.path == SERVER_INFO:
            cls.server_info_calls += 1
        if action is None:
            self.send_error(404)
            return
        if callable(action):
            action(self)
            return
        body = action if isinstance(action, bytes) else json.dumps(action).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _server(routes: dict[str, Any]) -> Iterator[str]:
    _Handler.routes = routes
    _Handler.seen = []
    _Handler.server_info_calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _dev_routes() -> dict[str, Any]:
    return {
        VERSION: {"version": "0.30.0"},
        MODELS: {"data": [{"id": "Qwen/Qwen2.5-0.5B", "max_model_len": 32768}]},
        SERVER_INFO: SERVER_INFO_BODY,
    }


def test_a_dev_mode_server_answers_every_route_redacted() -> None:
    with _server(_dev_routes()) as endpoint:
        probe = probe_server(endpoint, api_key="sk-local")

    assert not probe.incomplete
    assert probe.answers[VERSION].body == {"version": "0.30.0"}
    assert probe.answers[MODELS].status == "ok"
    info = probe.answers[SERVER_INFO].body
    assert SECRET not in json.dumps(info)
    assert info["vllm_config"]["scheduler_config"]["max_num_batched_tokens"] == 8192
    assert info["system_env"]["torch_version"] == "2.9.0"
    assert info["system_env_freshness"] == "first_or_cached"
    # The key goes along: every route is on the endpoint's own origin.
    assert {auth for _path, auth in _Handler.seen} == {"Bearer sk-local"}
    # The server's own clock, which a manifest's timing checks anchor on.
    assert probe.answers[VERSION].to_record()["date"].endswith(" GMT")


def test_urls_in_the_basic_answers_lose_their_credentials() -> None:
    routes = _dev_routes()
    routes[MODELS] = {
        "data": [{"id": "m", "root": f"https://user:{SECRET}@models.example/m"}]
    }
    with _server(routes) as endpoint:
        probe = probe_server(endpoint, mode="basic")
    body = probe.answers[MODELS].body
    assert SECRET not in json.dumps(body)
    assert body["data"][0]["root"] == "https://models.example/m"


def test_a_server_without_dev_mode_says_why_server_info_is_missing() -> None:
    routes = _dev_routes()
    del routes[SERVER_INFO]
    with _server(routes) as endpoint:
        probe = probe_server(endpoint)
    answer = probe.answers[SERVER_INFO]
    assert (answer.status, answer.http_status) == ("http_error", 404)
    assert "VLLM_SERVER_DEV_MODE=1" in str(answer.detail)
    assert not probe.incomplete


def test_basic_never_asks_for_server_info_and_none_asks_nothing() -> None:
    with _server(_dev_routes()) as endpoint:
        basic = probe_server(endpoint, mode="basic")
        nothing = probe_server(endpoint, mode="none")
    assert set(basic.answers) == {VERSION, MODELS}
    assert nothing.answers == {}
    assert _Handler.server_info_calls == 0


def test_a_slow_server_info_leaves_the_probe_incomplete_without_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_probe, "SERVER_INFO_DEADLINE_SECONDS", 0.3)

    def collect_env_still_running(handler: BaseHTTPRequestHandler) -> None:
        time.sleep(1.5)

    routes = {**_dev_routes(), SERVER_INFO: collect_env_still_running}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint)
        calls = _Handler.server_info_calls
    assert probe.answers[SERVER_INFO].status == "timeout"
    assert probe.incomplete
    assert calls == 1


def _trickle(head: bool) -> Any:
    """Send an answer one byte every 50 ms: the headers, or only the body."""

    def answer(handler: BaseHTTPRequestHandler) -> None:
        body = json.dumps({"version": "0.30.0"}).encode()
        lines = (
            b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
        )
        if not head:
            handler.wfile.write(lines)
            lines = b""
        try:
            for byte in lines + body:
                handler.wfile.write(bytes([byte]))
                handler.wfile.flush()
                time.sleep(0.05)
        except OSError:
            return

    return answer


@pytest.mark.parametrize("head", [False, True], ids=["body", "headers"])
def test_the_deadline_bounds_the_whole_exchange(
    monkeypatch: pytest.MonkeyPatch, head: bool
) -> None:
    # A server that trickles its answer kept the probe for as long as each
    # read made progress: the deadline was checked only between reads.
    monkeypatch.setattr(server_probe, "BASIC_DEADLINE_SECONDS", 0.5)
    routes = {**_dev_routes(), VERSION: _trickle(head)}
    began = time.monotonic()
    with _server(routes) as endpoint:
        probe = probe_server(endpoint, mode="basic")
    assert probe.answers[VERSION].status == "timeout"
    assert time.monotonic() - began < 1.5


def test_an_answer_cut_off_at_its_deadline_stops_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The abandoned exchange kept its connection reading for as long as the
    # server trickled: a client connection the run never planned.
    monkeypatch.setattr(server_probe, "BASIC_DEADLINE_SECONDS", 0.5)
    routes = {**_dev_routes(), VERSION: _trickle(head=True)}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint, mode="basic")
        assert probe.answers[VERSION].status == "timeout"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and _probing():
            time.sleep(0.05)
        assert not _probing()


def _probing() -> bool:
    return any(
        t.name == "stormlog-probe" and t.is_alive() for t in threading.enumerate()
    )


def test_after_a_route_times_out_the_others_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_probe, "BASIC_DEADLINE_SECONDS", 0.3)

    def hung(handler: BaseHTTPRequestHandler) -> None:
        time.sleep(1.0)

    routes = {**_dev_routes(), VERSION: hung}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint)
        calls = _Handler.server_info_calls
    assert probe.answers[VERSION].status == "timeout"
    for route in (MODELS, SERVER_INFO):
        assert (probe.answers[route].status, probe.answers[route].detail) == (
            "skipped",
            "no answer to an earlier route",
        )
    # Nothing was asked of /server_info, so no collector can be running.
    assert calls == 0 and not probe.incomplete


def test_a_server_info_lost_after_it_was_sent_leaves_the_probe_incomplete() -> None:
    # The server took the request and closed without a byte of answer: its
    # collector may still be running.
    def close(handler: BaseHTTPRequestHandler) -> None:
        handler.close_connection = True
        handler.wfile.flush()
        handler.connection.shutdown(2)

    routes = {**_dev_routes(), SERVER_INFO: close}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint)
    assert probe.answers[SERVER_INFO].status == "delivery_unknown"
    assert probe.incomplete


def test_an_answer_nested_too_deeply_is_invalid_json() -> None:
    routes = {**_dev_routes(), VERSION: b"[" * 100_000 + b"]" * 100_000}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint, mode="basic")
    assert probe.answers[VERSION].status == "invalid_json"


def test_an_oversized_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_probe, "RESPONSE_CAP_BYTES", 1024)
    routes = {**_dev_routes(), MODELS: b"x" * 4096}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint, mode="basic")
    assert probe.answers[MODELS].status == "too_large"


def test_a_redirect_is_not_followed() -> None:
    def redirect(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(302)
        handler.send_header("Location", "http://198.51.100.7/elsewhere")
        handler.end_headers()

    routes = {**_dev_routes(), VERSION: redirect}
    with _server(routes) as endpoint:
        probe = probe_server(endpoint, mode="basic", api_key="sk-local")
    answer = probe.answers[VERSION]
    assert (answer.status, answer.http_status) == ("http_error", 302)
    assert [path for path, _auth in _Handler.seen].count(VERSION) == 1


def test_an_unreachable_server_is_asked_once() -> None:
    probe = probe_server("http://127.0.0.1:1/v1/chat/completions")
    assert probe.answers[VERSION].status == "unreachable"
    # The other routes would each wait out a deadline on a host that is down.
    for route in (MODELS, SERVER_INFO):
        assert (probe.answers[route].status, probe.answers[route].detail) == (
            "skipped",
            "unreachable",
        )


class _Recorder:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, request: Any, timeout: float) -> Any:
        self.urls.append(request.full_url)
        raise OSError("not sent in this test")


def test_server_info_is_skipped_on_a_public_host_unless_allowed() -> None:
    recorder = _Recorder()
    endpoint = "http://203.0.113.5:8000/v1/chat/completions"
    skipped = probe_server(endpoint, opener=recorder)
    assert skipped.answers[SERVER_INFO].status == "skipped"
    assert skipped.answers[SERVER_INFO].detail == "non_private_host"
    assert not any("server_info" in url for url in recorder.urls)

    allowed = probe_server(endpoint, opener=recorder, allow_remote=True)
    assert allowed.answers[SERVER_INFO].status == "failed"
    assert any("server_info" in url for url in recorder.urls)


def test_after_the_run_collect_env_is_labelled_cached() -> None:
    with _server(_dev_routes()) as endpoint:
        probe = probe_server(endpoint, phase=AFTER)
    assert probe.answers[SERVER_INFO].body["system_env_freshness"] == "cached"
    record = probe.to_record(session_id="s1")
    assert (record["event_type"], record["phase"]) == ("infer.server_probe", "after")


@pytest.mark.parametrize(
    ("host", "private"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("10.1.2.3", True),
        ("192.168.0.4", True),
        ("[::1]", True),
        ("fd00::1", True),
        ("169.254.1.1", True),
        ("203.0.113.5", False),
        ("8.8.8.8", False),
        ("example.com", False),
    ],
)
def test_private_hosts(host: str, private: bool) -> None:
    assert is_private_host(host) is private


def test_the_origin_keeps_scheme_host_and_port() -> None:
    assert endpoint_origin("https://u:p@h:8443/v1/chat/completions?x=1") == (
        "https://h:8443"
    )
    with pytest.raises(ValueError):
        endpoint_origin("file:///tmp/x")


# ------------------------------------------------------------ with a profile


def _records(path: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_profile_records_what_the_server_said_before_and_after(
    tmp_path: Any,
) -> None:
    with _server(_dev_routes()) as endpoint:
        run_profile_with_fake_client(
            tmp_path, latency_seconds=0.0, endpoint=endpoint, server_probe="auto"
        )
    probes = [
        r
        for r in _records(tmp_path / "infer.jsonl")
        if r.get("event_type") == "infer.server_probe"
    ]
    assert [probe["phase"] for probe in probes] == ["before", "after"]
    assert probes[0]["answers"][VERSION]["body"] == {"version": "0.30.0"}
    after = probes[1]["answers"][SERVER_INFO]["body"]
    assert after["system_env_freshness"] == "cached"


def test_a_profile_does_not_measure_next_to_a_running_collector(
    tmp_path: Any,
) -> None:
    def incomplete(phase: str) -> ServerProbe:
        timed_out = ProbeAnswer(SERVER_INFO, "timeout")
        return ServerProbe(phase, "auto", "http://x", 0, {SERVER_INFO: timed_out})

    config = ProfileConfig(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model="m",
        concurrency=(1,),
        input_tokens=(8,),
        output_tokens=(4,),
        output_path=str(tmp_path / "infer.jsonl"),
        system_sampler="none",
        tokenizer="none",
    )
    profiler = InferenceProfiler(config, prober=incomplete)
    with pytest.raises(InferInputError, match="restart the server"):
        profiler.run()
    assert not (tmp_path / "infer.jsonl").exists()


def test_probing_can_be_turned_off(tmp_path: Any) -> None:
    run_profile_with_fake_client(tmp_path, latency_seconds=0.0, server_probe="none")
    kinds = {r.get("event_type") for r in _records(tmp_path / "infer.jsonl")}
    assert "infer.server_probe" not in kinds


def test_the_cli_refuses_an_unknown_probe_mode(tmp_path: Any) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(SystemExit) as raised:
            infer_main(["profile", "--model", "m", "--server-probe", "full"])
    assert raised.value.code == ExitCode.USAGE
