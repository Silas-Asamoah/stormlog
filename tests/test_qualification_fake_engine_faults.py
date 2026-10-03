"""The fake engine's fault controls: pauses, switches, signals and the kill switch."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
import urllib.error
from pathlib import Path

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.fake_engine.process import (
    ROOT,
    FakeEngineProcess,
    _environment,
)
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


@pytest.mark.parametrize("target", ["engine", "frontend"])
def test_stacked_pauses_hold_until_the_last_one_ends(target: str) -> None:
    # An earlier, shorter pause's timer must not release a later one.
    with FakeEngine(FAST) as engine:
        pause = engine.pause_engine if target == "engine" else engine.pause_frontend

        def paused() -> bool:
            if target == "engine":
                return engine.engine.paused
            return engine.frontend_paused

        pause(0.2)
        pause(0.7)
        time.sleep(0.45)
        held_past_the_first = paused()
        assert wait_until(lambda: not paused(), timeout=5)
        pause(0.2)
        pause()
        time.sleep(0.45)
        held_without_a_deadline = paused()
        if target == "engine":
            engine.resume_engine()
        else:
            engine.resume_frontend()
        released = not paused()
    assert (held_past_the_first, held_without_a_deadline, released) == (
        True,
        True,
        True,
    )
