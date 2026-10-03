"""The fake engine's fault controls: pauses, switches, signals and the kill switch."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import socket
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig, hook_log
from examples.qualification.fake_engine.__main__ import config_from_args
from examples.qualification.fake_engine.process import (
    ROOT,
    FakeEngineProcess,
    _environment,
)
from stormlog.infer.vllm_hook.writer import EpochWriter
from tests.qualification_fake_engine_helpers import (
    chats_in_background,
    get,
    join_all,
    post,
    wait_until,
    words,
)

FAST = FakeEngineConfig(step_seconds=0.001, decode_token_seconds=0.002)
POSIX_SIGNALS = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(signal, "SIGSTOP"),
    reason="needs SIGSTOP and SIGCONT",
)


def _state(base_url: str) -> dict[str, object]:
    status, body = get(f"{base_url}/_fault/state")
    assert status == 200
    state = json.loads(body)
    assert isinstance(state, dict)
    return state


def _steps(base_url: str) -> int:
    steps = _state(base_url)["steps"]
    assert isinstance(steps, int)
    return steps


def test_an_engine_pause_holds_steps_until_released() -> None:
    with FakeEngine(FAST) as engine:
        assert post(f"{engine.base_url}/_fault/pause?target=engine")[0] == 200
        threads = chats_in_background(engine, [words(4, "a")], max_tokens=5)
        assert wait_until(lambda: _state(engine.base_url)["waiting"] == 1)
        time.sleep(0.1)
        held = _steps(engine.base_url)
        assert post(f"{engine.base_url}/_fault/resume?target=engine")[0] == 200
        join_all(threads)
        finished = _state(engine.base_url)["finished"]
    assert held == 0
    assert finished == 1


def test_a_frontend_pause_holds_answers_while_the_engine_steps() -> None:
    with FakeEngine(FAST) as engine:
        threads = chats_in_background(engine, [words(4, "a")], max_tokens=60)
        assert wait_until(lambda: _steps(engine.base_url) > 0)
        post(f"{engine.base_url}/_fault/pause?target=frontend")
        before = _steps(engine.base_url)
        stepping = wait_until(lambda: _steps(engine.base_url) >= before + 5)
        with pytest.raises((TimeoutError, socket.timeout, urllib.error.URLError)):
            get(f"{engine.base_url}/health", timeout=0.3)
        post(f"{engine.base_url}/_fault/resume?target=frontend")
        join_all(threads)
    assert stepping


def test_a_timed_pause_releases_itself() -> None:
    with FakeEngine(FAST) as engine:
        post(f"{engine.base_url}/_fault/pause?target=engine&seconds=0.1")
        assert wait_until(lambda: not engine.engine.paused, timeout=5)


def test_the_controls_route_flips_switches_and_refuses_unknown_ones() -> None:
    with FakeEngine(FAST) as engine:
        url = f"{engine.base_url}/_fault/controls"
        changed = post(url, json.dumps({"metrics_mode": "fail"}).encode())
        failing = get(engine.metrics_url)[0]
        unknown = post(url, json.dumps({"no_such_switch": 1}).encode())[0]
        bad_target = post(f"{engine.base_url}/_fault/pause?target=gpu")[0]
    assert changed[0] == 200
    assert json.loads(changed[1])["metrics_mode"] == "fail"
    assert (failing, unknown, bad_target) == (500, 400, 400)


def test_trace_and_span_switches_are_reachable_over_http(tmp_path: Path) -> None:
    config = FakeEngineConfig(step_seconds=0.001, trace_dir=tmp_path / "traces")
    with FakeEngine(config) as engine:
        status, body = post(f"{engine.base_url}/_fault/foreign_trace")
        no_spans = post(f"{engine.base_url}/_fault/span_body?kind=oversized")[0]
    assert status == 200
    assert Path(json.loads(body)["path"]).exists()
    assert no_spans == 400


def test_the_kill_switch_is_refused_in_process() -> None:
    with FakeEngine(FAST) as engine:
        assert post(f"{engine.base_url}/_fault/kill")[0] == 403


@POSIX_SIGNALS
def test_a_stopped_process_answers_nothing_until_continued() -> None:
    with FakeEngineProcess(["--step-seconds", "0.001"]) as server:
        assert get(f"{server.base_url}/health")[0] == 200
        os.kill(server.pid, signal.SIGSTOP)
        try:
            with pytest.raises((TimeoutError, socket.timeout, urllib.error.URLError)):
                get(f"{server.base_url}/health", timeout=0.3)
        finally:
            os.kill(server.pid, signal.SIGCONT)
        assert get(f"{server.base_url}/health")[0] == 200


@POSIX_SIGNALS
def test_the_kill_switch_ends_the_process_without_goodbye(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    args = ["--step-seconds", "0.001", "--hook-dir", str(hook)]
    with FakeEngineProcess(args) as server:
        assert post(f"{server.base_url}/_fault/kill")[0] == 200
        process = server.process
        assert process is not None
        assert process.wait(timeout=10) == 137
    logs = [path.read_text() for path in hook.rglob("*.jsonl*")]
    assert logs
    assert not any('"kind":"goodbye"' in text for text in logs)


def test_the_subprocess_keeps_the_callers_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/opt/a", "/opt/b"]))
    assert _environment()["PYTHONPATH"].split(os.pathsep) == [
        str(ROOT),
        "/opt/a",
        "/opt/b",
    ]
    monkeypatch.delenv("PYTHONPATH")
    assert _environment()["PYTHONPATH"] == str(ROOT)


def test_the_command_lines_defaults_are_the_configs() -> None:
    # Only the kill switch differs: a separate process may be killed.
    assert config_from_args([]) == dataclasses.replace(
        FakeEngineConfig(), allow_kill=True
    )


@pytest.mark.parametrize("target", ["engine", "frontend"])
@pytest.mark.parametrize(
    "pauses",
    [(0.2, 0.7), (0.7, 0.2), (0.2, None), (None, 0.2)],
    ids=["shorter-first", "longer-first", "untimed-last", "untimed-first"],
)
def test_stacked_pauses_hold_until_the_last_one_ends(
    target: str, pauses: tuple[float | None, float | None]
) -> None:
    # The gate opens when the pause that ends last ends, whatever the order:
    # neither a shorter pause's timer nor the latest pause releases it.
    with FakeEngine(FAST) as engine:
        if target == "engine":
            pause, resume = engine.pause_engine, engine.resume_engine
        else:
            pause, resume = engine.pause_frontend, engine.resume_frontend

        def paused() -> bool:
            if target == "engine":
                return engine.engine.paused
            return engine.frontend_paused

        for seconds in pauses:
            pause(seconds)
        time.sleep(0.45)
        held = paused()
        if None in pauses:
            resume()
        released = wait_until(lambda: not paused(), timeout=5)
    assert (held, released) == (True, True)


def test_a_failed_bind_stops_everything_started_before_it(tmp_path: Path) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = int(taken.getsockname()[1])
        engine = FakeEngine(
            FakeEngineConfig(
                port=port,
                hook_dir=tmp_path / "hook",
                spans_endpoint="http://127.0.0.1:9/v1/traces",
            )
        )
        with pytest.raises(OSError):
            engine.start()
    assert engine.hook is not None and engine.spans is not None
    threads = [
        engine.hook.engine_writer._thread,
        engine.hook.worker_writer._thread,
        engine.spans._thread,
    ]
    assert [thread.name for thread in threads if thread.is_alive()] == []
    engine.stop()  # safe again, and for parts that never started


def test_a_hook_log_that_fails_to_open_stops_the_writer_it_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[EpochWriter] = []

    def writer(root: Path, role: str, **kwargs: Any) -> EpochWriter:
        if role == "worker":
            raise OSError("no room for the worker epoch")
        opened.append(EpochWriter(root, role, **kwargs))
        return opened[-1]

    monkeypatch.setattr(hook_log, "EpochWriter", writer)
    with pytest.raises(OSError):
        hook_log.HookLog(tmp_path / "hook", FakeEngineConfig())
    assert [w._thread.is_alive() for w in opened] == [False]
