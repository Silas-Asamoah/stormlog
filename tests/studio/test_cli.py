"""cli.py: the launcher, the port choice, the state files and studio url."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from stormlog import entrypoint
from stormlog.studio import cli
from stormlog.studio.contracts import LaunchOptions, Route, StudioContext
from stormlog.studio.security import SessionTokens
from tests.studio.test_httpd import call, stub_compose
from tests.studio.test_security import REPO_ROOT, validate

URL = re.compile(r"Open (http://\S+)")


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "stormlog" / "studio"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    directory = tmp_path / "work"
    (directory / "runs").mkdir(parents=True)
    (directory / "runs" / "a.jsonl").write_text("{}\n")
    return directory


def run_main(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, compose=stub_compose, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@dataclass
class Running:
    """``cli.main`` serving in a thread until ``stop`` is set."""

    thread: threading.Thread
    stop: threading.Event
    out: io.StringIO
    err: io.StringIO
    opened: list[str]
    result: list[int] = field(default_factory=list)
    contexts: list[StudioContext] = field(default_factory=list)

    def url(self) -> str:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            match = URL.search(self.out.getvalue())
            if match:
                return match.group(1)
            if not self.thread.is_alive():
                break
            time.sleep(0.02)
        raise AssertionError(f"no URL printed: {self.err.getvalue()}")

    def port(self) -> int:
        found = re.search(r":(\d+)/", self.url())
        assert found is not None
        return int(found.group(1))

    def finish(self) -> int:
        self.stop.set()
        self.thread.join(15)
        assert not self.thread.is_alive()
        return self.result[0]


@pytest.fixture
def serve() -> Iterator[Callable[[list[str]], Running]]:
    started: list[Running] = []

    def start(argv: list[str]) -> Running:
        running = Running(
            thread=threading.Thread(target=lambda: None),
            stop=threading.Event(),
            out=io.StringIO(),
            err=io.StringIO(),
            opened=[],
        )

        def compose(options: LaunchOptions) -> tuple[StudioContext, list[Route]]:
            ctx, routes = stub_compose(options)
            running.contexts.append(ctx)
            return ctx, routes

        def target() -> None:
            running.result.append(
                cli.main(
                    argv,
                    compose=compose,
                    stop=running.stop,
                    open_browser=running.opened.append,
                    stdout=running.out,
                    stderr=running.err,
                )
            )

        running.thread = threading.Thread(target=target, daemon=True)
        running.thread.start()
        started.append(running)
        return running

    yield start
    for running in started:
        running.stop.set()
        running.thread.join(15)


def bootstrap_of(url: str) -> str:
    found = re.search(r"#/auth\?t=([A-Za-z0-9_-]+)", url)
    assert found is not None
    return found.group(1)


def exchange(port: int, bootstrap: str) -> int:
    reply = call(
        port,
        "POST",
        "/api/v1/session",
        headers={
            "Origin": f"http://127.0.0.1:{port}",
            "Content-Type": "application/json",
        },
        body=json.dumps({"bootstrap": bootstrap}).encode(),
    )
    return reply.status


# --- Refusals before serving --------------------------------------------------


def test_windows_exits_2_with_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    for argv in ([], ["url"]):
        code, _, err = run_main(argv)
        assert code == 2
        assert "macOS and Linux" in err


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2", "example.com", "::"])
def test_non_loopback_hosts_exit_2(root: Path, state: Path, host: str) -> None:
    code, _, err = run_main([str(root), "--host", host])
    assert code == 2
    assert "--host" in err


def test_bad_flags_exit_2(root: Path, state: Path) -> None:
    for argv in (["--port", "70000"], ["--port", "x"], ["--remote"]):
        code, _, _ = run_main([str(root), *argv])
        assert code == 2


def test_a_missing_path_exits_5(tmp_path: Path, state: Path) -> None:
    code, _, err = run_main([str(tmp_path / "missing.jsonl")])
    assert code == 5
    assert "cannot read" in err


def test_root_and_home_are_refused_unless_root_names_them(
    tmp_path: Path, state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for argv in (["/"], [str(tmp_path)]):
        code, _, err = run_main(argv)
        assert code == 2
        assert "--root" in err
    monkeypatch.chdir(tmp_path)
    code, _, _ = run_main([])
    assert code == 2
    options = cli.build_options(cli._parser().parse_args(["--root", str(tmp_path)]))
    assert options.root == Path(os.path.realpath(tmp_path))


def test_an_artifact_path_makes_its_directory_the_root(root: Path, state: Path) -> None:
    artifact = root / "runs" / "a.jsonl"
    options = cli.build_options(cli._parser().parse_args([str(artifact)]))
    assert options.root == Path(os.path.realpath(root / "runs"))
    assert options.open_path == "a.jsonl"
    nested = cli.build_options(
        cli._parser().parse_args([str(artifact), "--root", str(root)])
    )
    assert nested.root == Path(os.path.realpath(root))
    assert nested.open_path == "runs/a.jsonl"


def test_a_path_outside_root_exits_2(root: Path, tmp_path: Path, state: Path) -> None:
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n")
    code, _, err = run_main([str(other), "--root", str(root / "runs")])
    assert code == 2
    assert "not under the root" in err


def test_options_defaults(root: Path, state: Path) -> None:
    options = cli.build_options(cli._parser().parse_args([str(root)]))
    assert options.host == "127.0.0.1"
    assert options.port == 8790
    assert options.open_browser
    assert options.state_dir == state
    assert not options.log_requests and not options.debug
    capped = cli.build_options(
        cli._parser().parse_args([str(root), "--analysis-max-bytes", "1024"])
    )
    assert capped.limits.analysis_bytes == 1024


def test_state_directory_defaults_to_local_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.state_directory() == tmp_path / ".local/state/stormlog/studio"


# --- Serving ------------------------------------------------------------------


def test_serve_prints_a_one_time_url_and_writes_private_state(
    root: Path, state: Path, serve: Callable[[list[str]], Running]
) -> None:
    running = serve([str(root), "--port", "0"])
    url = running.url()
    port = running.port()
    assert url.startswith(f"http://127.0.0.1:{port}/#/auth?t=")
    assert f"serving {os.path.realpath(root)}" in running.out.getvalue()
    record_path = state / f"{port}.json"
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    record = json.loads(record_path.read_text())
    assert record["pid"] == os.getpid()
    assert record["port"] == port
    assert record["instance_id"] == running.contexts[0].instance_id
    assert running.contexts[0].tokens.check(record["admin_token"], "admin")
    assert exchange(port, bootstrap_of(url)) == 200
    assert exchange(port, bootstrap_of(url)) == 401
    assert running.finish() == 0
    assert not record_path.exists()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_an_artifact_path_opens_its_requests_page(
    root: Path, state: Path, serve: Callable[[list[str]], Running]
) -> None:
    running = serve([str(root / "runs" / "a.jsonl"), "--port", "0"])
    assert running.url().endswith("&route=%2Frequests%3Fartifact%3Da.jsonl")
    running.finish()


def test_the_browser_opens_a_private_redirect_file(
    root: Path, state: Path, serve: Callable[[list[str]], Running]
) -> None:
    running = serve([str(root), "--port", "0"])
    url = running.url()
    deadline = time.monotonic() + 5.0
    while not running.opened and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(running.opened) == 1
    opened = running.opened[0]
    assert opened.startswith("file://")
    assert bootstrap_of(url) not in opened
    launch = Path(opened[len("file://") :])
    assert stat.S_IMODE(launch.stat().st_mode) == 0o600
    assert stat.S_IMODE(launch.parent.stat().st_mode) == 0o700
    assert url.replace("&", "&amp;") in launch.read_text()
    # Spending the bootstrap removes the file.
    assert exchange(running.port(), bootstrap_of(url)) == 200
    deadline = time.monotonic() + 5.0
    while launch.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not launch.exists()
    running.finish()


def test_no_browser_opens_nothing(
    root: Path, state: Path, serve: Callable[[list[str]], Running]
) -> None:
    running = serve([str(root), "--port", "0", "--no-browser"])
    running.url()
    assert running.opened == []
    assert not (state / "launch").exists()
    running.finish()


def test_studio_url_mints_a_fresh_bootstrap(
    root: Path, state: Path, serve: Callable[[list[str]], Running]
) -> None:
    running = serve([str(root), "--port", "0", "--no-browser"])
    port = running.port()
    code, out, err = run_main(["url", "/cases?artifact=runs%2Fa.jsonl"])
    assert code == 0, err
    fresh = out.strip()
    validate(
        {"format": "stormlog.studio.bootstrap", "schema_version": 1, "url": fresh},
        "bootstrap",
    )
    assert fresh.startswith(f"http://127.0.0.1:{port}/#/auth?t=")
    assert fresh.endswith("&route=%2Fcases%3Fartifact%3Druns%252Fa.jsonl")
    assert bootstrap_of(fresh) != bootstrap_of(running.url())
    assert exchange(port, bootstrap_of(fresh)) == 200
    code, _, err = run_main(["url", "--port", str(port + 1)])
    assert code == 2
    running.finish()
    code, _, err = run_main(["url"])
    assert code == 2
    assert "no Studio instance" in err


def test_studio_url_ignores_stale_state_files(state: Path) -> None:
    state.mkdir(parents=True)
    dead = subprocess.run([sys.executable, "-c", "pass"], check=True)
    assert dead.returncode == 0
    stale = {"pid": 2**22 + 12345, "port": 8790, "base_url": "http://127.0.0.1:8790"}
    (state / "8790.json").write_text(json.dumps(stale))
    code, _, err = run_main(["url"])
    assert code == 2
    assert "no Studio instance" in err


def test_the_default_port_falls_back_to_the_next_nine(
    root: Path,
    state: Path,
    serve: Callable[[list[str]], Running],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        monkeypatch.setattr(cli, "DEFAULT_PORT", taken)
        running = serve([str(root), "--no-browser"])
        assert taken < running.port() <= taken + 9
        running.finish()


def test_an_explicit_busy_port_exits_2(root: Path, state: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        code, _, err = run_main([str(root), "--port", str(taken)])
    assert code == 2
    assert f"no free port in {taken}" in err


def test_shutdown_closes_the_stub_components(
    root: Path, state: Path, serve: Callable[[list[str]], Running]
) -> None:
    running = serve([str(root), "--port", "0", "--no-browser"])
    running.url()
    running.finish()
    ctx = running.contexts[0]
    assert getattr(ctx.indexes, "closed", False)
    assert getattr(ctx.analysis, "closed", False)


@pytest.mark.skipif(
    importlib.util.find_spec("stormlog.studio.app") is not None,
    reason="C4's app module exists",
)
def test_without_app_the_default_composer_says_so(root: Path, state: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(root), "--port", "0"], stdout=out, stderr=err)
    assert code == 1
    assert "stormlog.studio.app" in err.getvalue()


SIGNAL_SCRIPT = """
import sys
from stormlog.studio import cli
from tests.studio.test_httpd import stub_compose
sys.exit(cli.main(sys.argv[1:], compose=stub_compose))
"""


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_sigint_and_sigterm_stop_cleanly(
    root: Path, state: Path, tmp_path: Path, signum: int
) -> None:
    script = tmp_path / "run_studio.py"
    script.write_text(SIGNAL_SCRIPT)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    process = subprocess.Popen(
        [sys.executable, str(script), str(root), "--port", "0", "--no-browser"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        line = ""
        while "Open http" not in line:
            line = process.stdout.readline()
            assert line, process.stderr.read() if process.stderr else ""
        assert list(state.glob("*.json"))
        process.send_signal(signum)
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert not list(state.glob("*.json"))


# --- The stormlog entrypoint ---------------------------------------------------


def test_entrypoint_dispatches_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_main(argv: Any) -> int:
        seen.append(list(argv))
        return 7

    monkeypatch.setattr(cli, "main", fake_main)
    assert entrypoint.main(["studio", "url", "--port", "8790"]) == 7
    assert seen == [["url", "--port", "8790"]]
    choices = entrypoint._build_parser()._actions[1].choices
    assert choices is not None and "studio" in choices


def test_other_commands_never_import_studio() -> None:
    probe = (
        "import sys, stormlog, stormlog.entrypoint\n"
        "sys.exit(any(m.startswith('stormlog.studio') for m in sys.modules))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT)
    assert done.returncode == 0


def test_python_m_stormlog_studio_help_runs() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "stormlog.studio", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0
    assert "stormlog studio" in done.stdout
    for flag in ("--root", "--host", "--port", "--no-browser", "--log-requests"):
        assert flag in done.stdout


def test_session_tokens_are_the_launch_token_store() -> None:
    assert isinstance(SessionTokens(), SessionTokens)
