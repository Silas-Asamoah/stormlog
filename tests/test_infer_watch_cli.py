"""``stormlog infer watch`` on the command line: exit codes and signals."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from stormlog.infer.cli import main as infer_main
from tests.watch_test_helpers import FakeMetrics, serve_metrics, watch_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_signals() -> Iterator[None]:
    """A watch run as the console script, or the CLI's signal handling
    driven directly, leaves SIGINT and SIGTERM ignored, as the process it
    is written for is about to exit."""
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for signum, handler in previous.items():
        signal.signal(signum, handler)


class _SignalledWatch:
    """Stands in for a Watcher under the CLI's signal handling: it signals
    this process and records what the handlers did."""

    def __init__(self, *, ending: bool) -> None:
        self.ending = ending
        self.hurried = 0
        self.stopped = False
        # After the signal that hurried, whether a further one would still
        # reach the CLI's handler rather than the default interrupt.
        self.handled_after_hurry: bool | None = None

    def hurry(self) -> None:
        self.hurried += 1

    async def run(
        self, stop: asyncio.Event, *, before_report: Callable[[], None]
    ) -> str:
        loop = asyncio.get_running_loop()
        if not self.ending:
            loop.call_soon(os.kill, os.getpid(), signal.SIGINT)
            await asyncio.wait_for(stop.wait(), 5)
            self.stopped = True
            self.ending = True  # the shutdown has begun
        loop.call_soon(os.kill, os.getpid(), signal.SIGINT)
        deadline = loop.time() + 5
        while not self.hurried and loop.time() < deadline:
            await asyncio.sleep(0.01)
        handler = signal.getsignal(signal.SIGINT)
        self.handled_after_hurry = handler is not signal.default_int_handler
        if self.handled_after_hurry:  # the user's second: still not the default
            loop.call_soon(os.kill, os.getpid(), signal.SIGINT)
            await asyncio.sleep(0.2)
        return "outcome"


@pytest.mark.parametrize("ended_by_duration", [False, True])
def test_a_signal_during_a_shutdown_cuts_it_short(ended_by_duration: bool) -> None:
    """A second signal hurries the shutdown; nothing tested that wiring. A
    signal while a shutdown --duration began is under way hurries it too:
    it only set the stop, which nothing read any more. Either way, the
    user's third signal is the default interrupt: after a duration's end,
    the second was, and a quick double Ctrl+C lost the report."""
    from stormlog.infer.watch.cli import _run

    watch = _SignalledWatch(ending=ended_by_duration)
    outcome: object = asyncio.run(_run(watch))  # type: ignore[arg-type]
    assert outcome == "outcome"
    assert watch.hurried == (2 if ended_by_duration else 1)
    assert watch.stopped is not ended_by_duration
    assert watch.handled_after_hurry is ended_by_duration


def _watch(tmp_path: Path, *extra: str) -> int:
    return infer_main(["watch", "--root", str(tmp_path / "watch"), *extra])


def test_an_unreadable_config_exits_five(tmp_path: Path) -> None:
    assert _watch(tmp_path, "--config", str(tmp_path / "absent.json")) == 5
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{", encoding="utf-8")
    assert _watch(tmp_path, "--config", str(garbled)) == 5


def test_an_unusable_config_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    path.write_text(json.dumps(watch_config("http://x", surprise=1)), encoding="utf-8")
    assert _watch(tmp_path, "--config", str(path)) == 2


def test_a_second_watcher_on_a_root_exits_two(tmp_path: Path) -> None:
    from stormlog.infer.watch.config import resolve_watch_config
    from stormlog.infer.watch.watcher import Watcher

    config = resolve_watch_config(watch_config("http://127.0.0.1:9"))
    first = Watcher(config, tmp_path / "watch")
    try:
        assert _watch(tmp_path, "--base-url", "http://127.0.0.1:9") == 2
    finally:
        first.close()


@pytest.mark.parametrize(
    "extra",
    [
        (),  # no server
        ("--base-url", "http://127.0.0.1:9", "--test-trigger", "sometimes"),
        ("--base-url", "http://127.0.0.1:9", "--test-trigger", "every=0"),
        ("--base-url", "http://127.0.0.1:9", "--duration", "0"),
        ("--base-url", "http://127.0.0.1:9", "--interval", "-1"),
        ("--base-url", "http://127.0.0.1:9", "--api-key-env", "STORMLOG_UNSET_KEY"),
        # JSON and float() accept these; each crashed with exit 1, no report.
        ("--base-url", "http://127.0.0.1:9", "--duration", "inf"),
        ("--base-url", "http://127.0.0.1:9", "--duration", "nan"),
        ("--base-url", "http://127.0.0.1:9", "--interval", "nan"),
        ("--base-url", "http://127.0.0.1:9", "--interval", "1e-12"),
        ("--base-url", "http://127.0.0.1:9", "--test-trigger", "every=inf"),
        ("--base-url", "ftp://127.0.0.1:9"),
        ("--base-url", "not a url"),
    ],
)
def test_bad_arguments_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    from stormlog.infer.watch import cli as watch_cli

    def no_watch(*_args: object, **_kwargs: object) -> None:
        # A bad argument let through would start an endless watch (with
        # --duration inf, say) and hang the test: it fails here instead.
        raise AssertionError("the watch started")

    monkeypatch.setattr(watch_cli, "Watcher", no_watch)
    monkeypatch.delenv("STORMLOG_UNSET_KEY", raising=False)
    assert _watch(tmp_path, *extra) == 2


def test_a_watch_of_a_quiet_server_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "watch.json"
    with serve_metrics(FakeMetrics()) as base_url:
        path.write_text(json.dumps(watch_config(base_url)), encoding="utf-8")
        code = _watch(tmp_path, "--config", str(path), "--duration", "0.5")
    assert code == 0
    report = tmp_path / "watch" / "report.json"
    assert f"Watch report: {report}" in capsys.readouterr().out
    assert json.loads(report.read_text())["verdict"]["exit_code"] == 0


def test_a_watch_run_in_process_gives_the_signals_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main(["watch", ...]) left SIGINT and SIGTERM ignored for the rest of
    the process. They are ignored from the report's write, so no signal
    changes the code it holds, and given back when main returns; the
    console script, which exits at once, keeps them ignored."""
    from stormlog.entrypoint import main as stormlog_main
    from stormlog.infer.watch import watcher as watcher_module

    seen: list[object] = []
    real_write = watcher_module.write_report

    def write(path: Path, report: object) -> None:
        seen.append(signal.getsignal(signal.SIGINT))
        real_write(path, report)  # type: ignore[arg-type]

    monkeypatch.setattr(watcher_module, "write_report", write)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    path = tmp_path / "watch.json"
    with serve_metrics(FakeMetrics()) as base_url:
        path.write_text(json.dumps(watch_config(base_url)), encoding="utf-8")
        assert _watch(tmp_path, "--config", str(path), "--duration", "0.5") == 0
        assert seen == [signal.SIG_IGN]
        assert {s: signal.getsignal(s) for s in previous} == previous
        argv = ["stormlog", "infer", "watch", "--root", str(tmp_path / "again")]
        monkeypatch.setattr(sys, "argv", [*argv, "--config", str(path)])
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--duration", "0.5"])
        assert stormlog_main() == 0
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN


def test_sigterm_ends_the_watch_with_a_report(tmp_path: Path) -> None:
    root = tmp_path / "watch"
    ready = tmp_path / "ready"
    with serve_metrics(FakeMetrics()) as base_url:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stormlog.entrypoint",
                "infer",
                "watch",
                "--root",
                str(root),
                "--base-url",
                base_url,
                "--interval",
                "0.1",
                "--ready-file",
                str(ready),
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "the watcher never became ready"
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=60)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert process.returncode == 0, err
    assert "Watch report:" in out
    report = json.loads((root / "report.json").read_text())
    assert report["verdict"]["exit_code"] == 0
    assert report["metrics"]["scrapes_ok"] >= 1


def test_sigterm_ends_a_watch_whose_scrape_trickles(tmp_path: Path) -> None:
    """A /metrics sending a byte every 0.2 s kept the watch past SIGTERM
    until it was killed, losing its report."""
    root = tmp_path / "watch"
    metrics = FakeMetrics()
    metrics.dribble = 0.2
    with serve_metrics(metrics) as base_url:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stormlog.entrypoint",
                "infer",
                "watch",
                "--root",
                str(root),
                "--base-url",
                base_url,
                "--interval",
                "0.5",
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while metrics.scrapes == 0:
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "the watcher never scraped"
                time.sleep(0.05)
            time.sleep(1.0)  # inside a trickling scrape, or past one
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert process.returncode == 1, err  # no scrape ever finished
    report = json.loads((root / "report.json").read_text())
    assert report["payload"]["unsound"] == ["no_successful_scrape"]


def test_two_quick_signals_end_with_the_exit_code_the_report_holds(
    tmp_path: Path,
) -> None:
    """A second SIGINT right after the first escaped as KeyboardInterrupt
    once the watch had returned: exit -2 after a report saying 3."""
    root = tmp_path / "watch"
    ready = tmp_path / "ready"
    metrics = FakeMetrics()
    metrics.waiting = 20
    with serve_metrics(metrics) as base_url:
        config = tmp_path / "watch.json"
        config.write_text(json.dumps(watch_config(base_url)), encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stormlog.entrypoint",
                "infer",
                "watch",
                "--root",
                str(root),
                "--config",
                str(config),
                "--ready-file",
                str(ready),
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "the watcher never became ready"
                time.sleep(0.05)
            time.sleep(1.0)  # the queue trigger has fired: an incident is open
            process.send_signal(signal.SIGINT)
            report_path = root / "report.json"
            while not report_path.exists() and process.poll() is None:
                time.sleep(0.001)
            # The second lands as the watch returns, report written.
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            _out, err = process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    report = json.loads((root / "report.json").read_text())
    assert process.returncode == report["verdict"]["exit_code"], err
